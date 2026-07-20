"""
"Online" DINOv2 patch-grid training: uses ALL raw patch tokens (256 per frame, no
pooling to 4x4) instead of the cached, adaptive-pooled patch_grid4 embeddings —
infeasible to permanently cache to disk at this resolution for reuse across many
future experiments (200,000 frames x 256 patches x 768dim x fp16 is ~78.6GB, vs.
~4.9GB for the pooled 16-token version this repo already keeps).

Design: at neg_ratio=15 with a 200,000-frame training bound, the negative pool is
already saturated (every epoch draws the SAME full available negative set — see
results/vision/mice/frame/patchgrid/'s own training log) — so the set of frames
needed for training is identical every epoch, not resampled. That means there is no
benefit to redundantly re-decoding+re-encoding frames every batch or every epoch (the
first, naive version of this script did exactly that and was far too slow — still
mid-way through its first training epoch after 15+ minutes). Instead: compute every
needed frame's patch tokens through the frozen encoder exactly ONCE, via a properly
parallelized DataLoader (matching the ~450-500 frames/sec this repo's own extraction
scripts achieve), hold the result in CPU RAM only for the duration of this one run
(never written to disk), then train fast from that in-memory cache. This is still
"online" in the sense that matters (no permanent disk cache of the full-resolution
representation), just not recomputed on every single batch.

Reuses FrameBatchData's tested sample-selection/context-window/label logic (built
with a tiny placeholder load_embeddings_fn so its own memory footprint stays
negligible — n_patches=1, emb_dim=1 — then its `flat` placeholder is discarded
entirely); only `gi` (global frame index per sample), `offsets_grid`, `pad_mask`,
and `labels` are used to determine which raw frames actually need encoding.

Reuses the confirmed-best patch-grid hyperparameters (results/vision/mice/frame/
patchgrid/config.json) rather than re-searching — this is a representation-quality
test (pooled 16 tokens vs. all 256), not a hyperparameter test.

Capped at 30 epochs (patience 10), not the usual 100/15 — prior patch-grid runs have
consistently converged well before epoch 30 anyway, and this bounds the (much
cheaper, post-encode) training-loop cost.

Usage:
    python scripts/mice_behavior/train_patchgrid_online.py
    python scripts/mice_behavior/train_patchgrid_online.py --smoke   # tiny local sanity check
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.dataset.get_dataset import load_dataset

DATA_DIR = gsf.DATA_DIR
DATASET_DIR = gsf.DATASET_DIR
SEED = gsf.SEED
MODEL_ID = 'facebook/dinov2-base'
EMB_DIM = 768
N_PATCHES_FULL = 256  # 16x16, no pooling, no register tokens — confirmed empirically


def dummy_loader(n_patches, emb_dim):
    def _load(obs_boundary):
        return {obs_s: np.zeros((obs_e - obs_s, n_patches, emb_dim), dtype=np.float16) for obs_s, obs_e in obs_boundary.values()}
    return _load


class _ImageDataset(Dataset):
    """Indexes the HF frame dataset directly by a list of global row indices, applying the
    DINOv2 processor per-item (so workers do the CPU-bound decode+resize in parallel)."""

    def __init__(self, hf_dataset, global_indices, processor):
        self.ds, self.indices, self.proc = hf_dataset, global_indices, processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image = self.ds[int(self.indices[i])]['image']
        processed = self.proc(images=image, return_tensors='pt')
        return processed['pixel_values'].squeeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true', help='tiny local sanity check: 1 epoch, small budgets')
    p.add_argument('--max-train-frames', type=int, default=200_000)
    p.add_argument('--n-epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--encode-batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=8)
    args = p.parse_args()

    if args.smoke:
        args.max_train_frames, args.n_epochs, args.patience, args.batch_size = 30_000, 2, 2, 64

    RESULTS_DIR = gsf.FRAME_DIR / ('patchgrid256_dinov2_smoke' if args.smoke else 'patchgrid256_dinov2')
    best_cfg = json.load(open(gsf.FRAME_DIR / 'patchgrid4x4_dinov2' / 'config.json'))['cfg']
    print(f'Reusing confirmed-best patchgrid cfg: {best_cfg}', flush=True)
    print(f'max_train_frames={args.max_train_frames}  n_epochs={args.n_epochs}  patience={args.patience}', flush=True)

    pair_labels_path = gsf.build_pair_labels(DATA_DIR, DATASET_DIR, overwrite=False)
    annotations_csv = DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    obs_to_pool = gsf.load_obs_to_pool_map(DATA_DIR)
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    pools = sorted({obs_to_pool[o] for o in all_obs})
    rng_split = random.Random(SEED)
    shuffled = pools[:]
    rng_split.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.2))
    val_pool_set = set(shuffled[:n_val])
    train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]
    print(f'Split: {len(train_obs)} train obs / {len(val_obs)} val obs', flush=True)

    print('Building sample index (placeholder embeddings, cheap)...', flush=True)
    train_meta = FrameBatchData(
        str(annotations_csv), str(pair_labels_path), train_obs, best_cfg['context_k'], 1,
        dummy_loader(1, 1), n_patches=1, stride=best_cfg['stride'], max_frames=args.max_train_frames, seed=SEED,
    )
    val_meta = FrameBatchData(
        str(annotations_csv), str(pair_labels_path), val_obs, best_cfg['context_k'], 1,
        dummy_loader(1, 1), n_patches=1, stride=best_cfg['stride'],
    )
    del train_meta.flat, val_meta.flat  # never used — only gi/offsets_grid/pad_mask/labels matter

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {MODEL_ID} on {dev}...', flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    encoder = AutoModel.from_pretrained(MODEL_ID).to(dev)
    encoder.eval()
    encoder.requires_grad_(False)

    print('Loading full-frame HF dataset (raw JPEGs)...', flush=True)
    hf_dataset = load_dataset(subject='mice', version='v1', dataset_root=str(DATASET_DIR), frame_type='full')

    labels_all = train_meta.labels
    pos_idx = np.where(labels_all.sum(axis=1) > 0)[0]
    neg_idx = np.where(labels_all.sum(axis=1) == 0)[0]
    n_pos_frames = max(len(pos_idx), 1)
    n_neg_draw = min(len(neg_idx), best_cfg['neg_ratio'] * n_pos_frames)
    rng = np.random.default_rng(SEED)
    neg_sample = rng.choice(neg_idx, size=n_neg_draw, replace=False)
    epoch_idx_fixed = np.concatenate([pos_idx, neg_sample])
    saturated = n_neg_draw >= len(neg_idx)
    print(f'neg pool: requested {best_cfg["neg_ratio"] * n_pos_frames:,}, available {len(neg_idx):,} '
          f'-> {"SATURATED (identical every epoch, sampling once is exact)" if saturated else "sampled once per run (NOT resampled per epoch — a simplification for this test)"}',
          flush=True)

    v_labels = val_meta.labels
    v_pos = np.where(v_labels.sum(axis=1) > 0)[0]
    v_neg = np.where(v_labels.sum(axis=1) == 0)[0]
    v_rng = np.random.default_rng(SEED)
    n_v_neg = min(len(v_neg), 2 * max(len(v_pos), 1))
    v_neg_sample = v_rng.choice(v_neg, size=n_v_neg, replace=False)
    val_keep = np.sort(np.concatenate([v_pos, v_neg_sample]))
    print(f'val subsample: {len(val_meta):,} -> {len(val_keep):,} (all positives + 2x negatives)', flush=True)

    def needed_raw_indices(meta, sample_idx):
        gi = meta.gi[sample_idx]
        offsets = meta.offsets_grid
        abs_idx = gi[:, None] + offsets[None, :]
        mask = meta.pad_mask[sample_idx]
        return np.unique(abs_idx[~mask])

    print('Determining unique raw frames needed (train epoch pool + val subsample + full val)...', flush=True)
    need_train = needed_raw_indices(train_meta, epoch_idx_fixed)
    need_val_sub = needed_raw_indices(val_meta, val_keep)
    need_val_full = needed_raw_indices(val_meta, np.arange(len(val_meta)))
    all_needed = np.unique(np.concatenate([need_train, need_val_sub, need_val_full]))
    print(f'  {len(need_train):,} unique train frames, {len(need_val_full):,} unique val frames '
          f'(full val incl. subsample) -> {len(all_needed):,} total to encode once', flush=True)

    print(f'Encoding {len(all_needed):,} frames once via a parallel DataLoader '
          f'({args.num_workers} workers, batch {args.encode_batch_size})...', flush=True)
    t_encode0 = time.time()
    loader = DataLoader(
        _ImageDataset(hf_dataset, all_needed, processor), batch_size=args.encode_batch_size,
        num_workers=args.num_workers, pin_memory=(dev.type == 'cuda'), shuffle=False,
        prefetch_factor=4 if args.num_workers > 0 else None, persistent_workers=args.num_workers > 0,
    )
    cache = torch.empty((len(all_needed), N_PATCHES_FULL, EMB_DIM), dtype=torch.float16)
    idx_of_global = {int(g): i for i, g in enumerate(all_needed)}
    cursor = 0
    with torch.inference_mode():
        for pixel_values in loader:
            pixel_values = pixel_values.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
                out = encoder(pixel_values=pixel_values)
            tokens = out.last_hidden_state[:, 1:].half().cpu()  # dinov2-base: CLS only, no register tokens
            cache[cursor:cursor + tokens.shape[0]] = tokens
            cursor += tokens.shape[0]
            if cursor % (args.encode_batch_size * 20) == 0 or cursor == len(all_needed):
                elapsed = time.time() - t_encode0
                print(f'  encoded {cursor:,}/{len(all_needed):,} ({cursor/elapsed:.1f} frames/s)', flush=True)
    print(f'Encoding done in {(time.time()-t_encode0)/60:.1f} min '
          f'({len(all_needed)/(time.time()-t_encode0):.1f} frames/s average)', flush=True)
    del encoder, processor, loader
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

    def build_batch_tensor(meta, sample_idx):
        gi = meta.gi[sample_idx]
        offsets = meta.offsets_grid
        abs_idx = gi[:, None] + offsets[None, :]
        mask = meta.pad_mask[sample_idx]
        B, T = abs_idx.shape
        valid = ~mask
        flat_idx = abs_idx[valid]
        positions = np.array([idx_of_global[int(g)] for g in flat_idx], dtype=np.int64)
        gathered = cache[positions].to(dev, non_blocking=True)
        ctx = torch.zeros((B, T, N_PATCHES_FULL, EMB_DIM), dtype=torch.float16, device=dev)
        ctx[torch.from_numpy(valid)] = gathered
        offsets_t = torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()).to(dev)
        labels_t = torch.from_numpy(meta.labels[sample_idx]).to(dev)
        mask_t = torch.from_numpy(mask).to(dev)
        return ctx, offsets_t, labels_t, mask_t

    model = MouseFrameClassifier(
        emb_dim=EMB_DIM, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'],
        use_patch_grid=True, dropout=best_cfg['dropout'],
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_cfg['lr'], weight_decay=best_cfg['weight_decay'])

    n_neg_frames = max(min(best_cfg['neg_ratio'] * n_pos_frames, len(neg_idx)), 1)
    pos_counts = labels_all[pos_idx].sum(axis=0).clip(min=1)
    pos_weight = torch.tensor(n_neg_frames / pos_counts, dtype=torch.float32).to(dev)
    print(f'pos_weight: nt={pos_weight[0]:.2f} nn={pos_weight[1]:.2f}', flush=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_ap = -1.0
    epochs_since_best = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'macro_ap': []}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shuffle_rng = np.random.default_rng(SEED)
    # GradScaler is required under fp16 autocast, not optional -- without it, gradients can
    # underflow to exact zero (float16's limited dynamic range), which is exactly what
    # happened on the first run of this script: train_loss stuck flat at ~1.01-1.02 for 16
    # straight epochs and macro_ap was bit-for-bit identical epoch 10 through 16 -- the
    # unmistakable signature of weights that had silently stopped updating. train.py's own
    # train()/train_frame() always pair autocast with a GradScaler for exactly this reason.
    amp_enabled = dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_idx = epoch_idx_fixed.copy()
        shuffle_rng.shuffle(epoch_idx)

        total_loss, n_seen = 0.0, 0
        t0 = time.time()
        for b0 in range(0, len(epoch_idx), args.batch_size):
            batch_idx = epoch_idx[b0:b0 + args.batch_size]
            ctx, offs, lbl, mask = build_batch_tensor(train_meta, batch_idx)
            optimizer.zero_grad()
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(ctx.float(), offsets=offs, key_padding_mask=mask)
                loss = criterion(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(batch_idx)
            n_seen += len(batch_idx)
        train_loss = total_loss / n_seen

        model.eval()
        all_probs, all_labels, val_loss_sum, v_n = [], [], 0.0, 0
        with torch.no_grad():
            for b0 in range(0, len(val_keep), args.batch_size):
                batch_idx = val_keep[b0:b0 + args.batch_size]
                ctx, offs, lbl, mask = build_batch_tensor(val_meta, batch_idx)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
                    logits = model(ctx.float(), offsets=offs, key_padding_mask=mask)
                    batch_loss = criterion(logits, lbl)
                all_probs.append(torch.sigmoid(logits).float().cpu())
                all_labels.append(lbl.cpu())
                val_loss_sum += batch_loss.item() * len(batch_idx)
                v_n += len(batch_idx)
        probs = torch.cat(all_probs).numpy()
        labels_np = torch.cat(all_labels).numpy()
        aps = [average_precision_score(labels_np[:, i], probs[:, i]) for i in range(2)]
        macro_ap = float(np.mean(aps))
        val_loss = val_loss_sum / v_n

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['macro_ap'].append(macro_ap)
        print(f'epoch {epoch:3d}/{args.n_epochs}  loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
              f'macro_ap={macro_ap:.4f}  nt={aps[0]:.3f} nn={aps[1]:.3f}  ({time.time()-t0:.1f}s)', flush=True)

        if macro_ap > best_ap:
            best_ap = macro_ap
            epochs_since_best = 0
            torch.save(model.state_dict(), RESULTS_DIR / 'best_model.pt')
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f'early stopping: no improvement for {args.patience} epochs', flush=True)
                break

    model.load_state_dict(torch.load(RESULTS_DIR / 'best_model.pt', map_location=dev, weights_only=True))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for b0 in range(0, len(val_meta), args.batch_size):
            batch_idx = np.arange(b0, min(b0 + args.batch_size, len(val_meta)))
            ctx, offs, lbl, mask = build_batch_tensor(val_meta, batch_idx)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
                logits = model(ctx.float(), offsets=offs, key_padding_mask=mask)
            all_probs.append(torch.sigmoid(logits).float().cpu())
            all_labels.append(lbl.cpu())
    probs = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    per_label = {name: average_precision_score(labels_np[:, i], probs[:, i]) for i, name in enumerate(['nt', 'nn'])}
    final_score = float(np.mean(list(per_label.values())))
    print(f'FULL-VAL macro AP: {final_score:.4f}  {per_label}', flush=True)

    with open(RESULTS_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(RESULTS_DIR / 'config.json', 'w') as f:
        json.dump({'cfg': best_cfg, 'val_pools': sorted(val_pool_set), 'n_epochs_cap': args.n_epochs,
                   'n_patches': N_PATCHES_FULL, 'max_train_frames': args.max_train_frames,
                   'neg_pool_saturated': bool(saturated),
                   'best_ap': final_score, 'best_per_label': per_label}, f, indent=2)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
