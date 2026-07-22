"""
Hyperparameter search for the patchgrid256 (all 256 raw patch tokens, online-
computed, no 4x4 pooling) per-frame classifier, for either DINOv2 or DINOv3.

Key efficiency insight: encoding is independent of every hyperparameter searched
here (neg_ratio only changes which already-encoded frames get subsampled per
epoch, not which frames need encoding — see train_patchgrid_online.py's design
note; context_k/stride are fixed, see below, so the context window never varies
either). So the full candidate pool (all of train_meta + all of val_meta) is
encoded exactly ONCE via a frozen forward pass, cached in CPU RAM, and every
trial trains a fresh MouseFrameClassifier on top of that same cache — only the
~10-25 min/trial classifier training cost repeats, not the ~3-5 min encode.

context_k=2, stride=1 fixed (matching the current best patchgrid256_dinov2
result) rather than swept — sweeping stride would require encoding frames at
every possible offset up to MAX_OFFSET for every trial, adding real complexity
and cache size for a knob that mattered far less than hidden_dim/dropout/neg_ratio
in every prior search on this task. Search space (n_heads, hidden_dim, neg_ratio,
dropout, weight_decay, lr) matches grid_search_frame.py's own patch-grid range,
already tuned for this same classifier at the pooled-4x4 resolution.

Each trial trains to convergence via early stopping, so — unlike
grid_search_frame.py's search-then-separate-final-retrain — the winning trial's
own checkpoint is promoted directly.

Usage:
    python scripts/mice_behavior/search_patchgrid_online.py --encoder dinov2 --n-trials 15
    python scripts/mice_behavior/search_patchgrid_online.py --encoder dinov3 --n-trials 15
"""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import get_val_pools
from src.dataset.get_dataset import load_dataset
from train_patchgrid_online import dummy_loader, _ImageDataset, EMB_DIM

MODEL_IDS = {
    'dinov2': 'facebook/dinov2-base',
    'dinov3': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
}
# Patch count differs by patch size, not just image size: DINOv2 uses patch-size 14 at
# 224x224 -> 16x16=256 tokens; DINOv3 uses patch-size 16 at 224x224 -> 14x14=196 tokens.
# Confirmed empirically (a hardcoded 256 crashed the first DINOv3 search attempt with a
# tensor-shape mismatch at the very first encoded batch).
N_PATCHES = {'dinov2': 256, 'dinov3': 196}
CONTEXT_K, STRIDE = 2, 1


def sample_cfg(rng):
    return dict(
        n_heads=rng.choice([1, 2, 4, 8]),
        context_k=CONTEXT_K,
        stride=STRIDE,
        hidden_dim=rng.choice([128, 256, 384, 512]),
        neg_ratio=rng.choice([5, 10, 15, 20]),
        dropout=rng.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        weight_decay=rng.choice([0.0, 1e-5, 1e-4, 1e-3]),
        lr=rng.choice([3e-4, 1e-3, 3e-3]),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', required=True, choices=sorted(MODEL_IDS))
    p.add_argument('--n-trials', type=int, default=15)
    p.add_argument('--max-train-frames', type=int, default=200_000)
    p.add_argument('--search-epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--encode-batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=16)
    args = p.parse_args()
    MODEL_ID = MODEL_IDS[args.encoder]

    n_patches_full = N_PATCHES[args.encoder]
    OUT_DIR = gsf.FRAME_DIR / f'patchgrid256_{args.encoder}'
    LOG_PATH = gsf.SEARCH_DIR / f'log_patchgrid256_{args.encoder}.jsonl'
    gsf.SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    def log_result(record):
        record['ts'] = time.time()
        with open(LOG_PATH, 'a') as f:
            f.write(json.dumps(record) + '\n')
        print(f'[LOG] {record}', flush=True)

    pair_labels_path = gsf.build_pair_labels(gsf.DATA_DIR, gsf.DATASET_DIR, overwrite=False)
    annotations_csv = gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    obs_to_pool = gsf.load_obs_to_pool_map(gsf.DATA_DIR)
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    pools = sorted({obs_to_pool[o] for o in all_obs})
    val_pool_set = get_val_pools(pools, seed=gsf.SEED)
    train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]
    print(f'Split: {len(train_obs)} train obs / {len(val_obs)} val obs', flush=True)

    print('Building sample index (placeholder embeddings, cheap)...', flush=True)
    train_meta = FrameBatchData(
        str(annotations_csv), str(pair_labels_path), train_obs, CONTEXT_K, 1,
        dummy_loader(1, 1), n_patches=1, stride=STRIDE, max_frames=args.max_train_frames, seed=gsf.SEED,
    )
    val_meta = FrameBatchData(
        str(annotations_csv), str(pair_labels_path), val_obs, CONTEXT_K, 1,
        dummy_loader(1, 1), n_patches=1, stride=STRIDE,
    )
    del train_meta.flat, val_meta.flat  # only gi/offsets_grid/pad_mask/labels matter

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {MODEL_ID} on {dev}...', flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    encoder = AutoModel.from_pretrained(MODEL_ID).to(dev)
    encoder.eval()
    encoder.requires_grad_(False)
    n_register_tokens = getattr(encoder.config, 'num_register_tokens', 0)
    n_prefix_tokens = 1 + n_register_tokens
    print(f'{MODEL_ID}: {n_register_tokens} register tokens, dropping first {n_prefix_tokens} tokens per frame', flush=True)

    print('Loading full-frame HF dataset (raw JPEGs)...', flush=True)
    hf_dataset = load_dataset(subject='mice', version='v1', dataset_root=str(gsf.DATASET_DIR), frame_type='full')

    # Union across ALL of train_meta (not a neg_ratio-specific subsample — every trial's
    # neg_ratio just subsamples this same fixed pool) + all of val_meta. context_k/stride are
    # fixed across every trial, so this single offsets_grid covers every trial exactly.
    # Padding positions (context window running past an observation's start/end) MUST be
    # excluded here via pad_mask — without it, this previously computed out-of-range raw
    # indices (e.g. exactly len(hf_dataset), one past the last valid frame) and crashed the
    # DataLoader; train_patchgrid_online.py's equivalent computation already did this
    # correctly, this script's refactor dropped it by accident.
    def _needed_raw_indices(meta):
        abs_idx = meta.gi[:, None] + meta.offsets_grid[None, :]
        return np.unique(abs_idx[~meta.pad_mask])

    need_train = _needed_raw_indices(train_meta)
    need_val = _needed_raw_indices(val_meta)
    all_needed = np.unique(np.concatenate([need_train, need_val]))
    print(f'{len(need_train):,} unique train frames, {len(need_val):,} unique val frames '
          f'-> {len(all_needed):,} total to encode once', flush=True)

    t_encode0 = time.time()
    loader = DataLoader(
        _ImageDataset(hf_dataset, all_needed, processor), batch_size=args.encode_batch_size,
        num_workers=args.num_workers, pin_memory=(dev.type == 'cuda'), shuffle=False,
        prefetch_factor=4 if args.num_workers > 0 else None, persistent_workers=args.num_workers > 0,
    )
    cache = torch.empty((len(all_needed), n_patches_full, EMB_DIM), dtype=torch.float16)
    idx_of_global = {int(g): i for i, g in enumerate(all_needed)}
    cursor = 0
    with torch.inference_mode():
        for pixel_values in loader:
            pixel_values = pixel_values.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
                out = encoder(pixel_values=pixel_values)
            tokens = out.last_hidden_state[:, n_prefix_tokens:].half().cpu()
            cache[cursor:cursor + tokens.shape[0]] = tokens
            cursor += tokens.shape[0]
    print(f'Encoding done in {(time.time()-t_encode0)/60:.1f} min '
          f'({len(all_needed)/(time.time()-t_encode0):.1f} frames/s)', flush=True)
    del encoder, processor, loader
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

    offsets = train_meta.offsets_grid  # same array (context_k/stride fixed) for train and val

    def build_batch_tensor(meta, sample_idx):
        gi = meta.gi[sample_idx]
        abs_idx = gi[:, None] + offsets[None, :]
        mask = meta.pad_mask[sample_idx]
        B, T = abs_idx.shape
        valid = ~mask
        flat_idx = abs_idx[valid]
        positions = np.array([idx_of_global[int(g)] for g in flat_idx], dtype=np.int64)
        gathered = cache[positions].to(dev, non_blocking=True)
        ctx = torch.zeros((B, T, n_patches_full, EMB_DIM), dtype=torch.float16, device=dev)
        ctx[torch.from_numpy(valid)] = gathered
        offsets_t = torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()).to(dev)
        labels_t = torch.from_numpy(meta.labels[sample_idx]).to(dev)
        mask_t = torch.from_numpy(mask).to(dev)
        return ctx, offsets_t, labels_t, mask_t

    labels_all = train_meta.labels
    pos_idx = np.where(labels_all.sum(axis=1) > 0)[0]
    neg_idx = np.where(labels_all.sum(axis=1) == 0)[0]
    n_pos = max(len(pos_idx), 1)

    v_labels = val_meta.labels
    v_pos = np.where(v_labels.sum(axis=1) > 0)[0]
    v_neg = np.where(v_labels.sum(axis=1) == 0)[0]
    v_rng = np.random.default_rng(gsf.SEED)
    n_v_neg = min(len(v_neg), 2 * max(len(v_pos), 1))
    v_neg_sample = v_rng.choice(v_neg, size=n_v_neg, replace=False)
    val_keep = np.sort(np.concatenate([v_pos, v_neg_sample]))

    def run_trial(cfg):
        n_neg_draw = min(len(neg_idx), cfg['neg_ratio'] * n_pos)
        rng = np.random.default_rng(gsf.SEED)
        neg_sample = rng.choice(neg_idx, size=n_neg_draw, replace=False)
        epoch_idx_fixed = np.concatenate([pos_idx, neg_sample])

        pos_counts = labels_all[pos_idx].sum(axis=0).clip(min=1)
        pos_weight = torch.tensor(max(n_neg_draw, 1) / pos_counts, dtype=torch.float32).to(dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        lr = min(cfg['lr'], 3e-4)  # patch-grid PatchAttnPool collapse precedent (grid_search_frame.py)
        model = MouseFrameClassifier(
            emb_dim=EMB_DIM, n_heads=cfg['n_heads'], hidden_dim=cfg['hidden_dim'],
            use_patch_grid=True, dropout=cfg['dropout'],
        ).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
        amp_enabled = dev.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

        shuffle_rng = np.random.default_rng(gsf.SEED)
        best_ap, epochs_since_best, best_state = -1.0, 0, None
        for epoch in range(1, args.search_epochs + 1):
            model.train()
            epoch_idx = epoch_idx_fixed.copy()
            shuffle_rng.shuffle(epoch_idx)
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

            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for b0 in range(0, len(val_keep), args.batch_size):
                    batch_idx = val_keep[b0:b0 + args.batch_size]
                    ctx, offs, lbl, mask = build_batch_tensor(val_meta, batch_idx)
                    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp_enabled):
                        logits = model(ctx.float(), offsets=offs, key_padding_mask=mask)
                    all_probs.append(torch.sigmoid(logits).float().cpu())
                    all_labels.append(lbl.cpu())
            probs = torch.cat(all_probs).numpy()
            labels_np = torch.cat(all_labels).numpy()
            macro_ap = float(np.mean([average_precision_score(labels_np[:, i], probs[:, i]) for i in range(2)]))

            if macro_ap > best_ap:
                best_ap, epochs_since_best = macro_ap, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                epochs_since_best += 1
                if epochs_since_best >= args.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for b0 in range(0, len(val_meta), args.batch_size):
                batch_idx = np.arange(b0, min(b0 + args.batch_size, len(val_meta)))
                ctx, offs, lbl, mask = build_batch_tensor(val_meta, batch_idx)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp_enabled):
                    logits = model(ctx.float(), offsets=offs, key_padding_mask=mask)
                all_probs.append(torch.sigmoid(logits).float().cpu())
                all_labels.append(lbl.cpu())
        probs = torch.cat(all_probs).numpy()
        labels_np = torch.cat(all_labels).numpy()
        per_label = {name: average_precision_score(labels_np[:, i], probs[:, i]) for i, name in enumerate(['nt', 'nn'])}
        full_ap = float(np.mean(list(per_label.values())))
        return full_ap, per_label, best_state

    baseline_ap = -1.0
    if (OUT_DIR / 'config.json').exists():
        baseline_ap = json.load(open(OUT_DIR / 'config.json'))['best_ap']
    print(f'Current baseline for {OUT_DIR.name}: {baseline_ap:.4f}', flush=True)

    sample_rng = random.Random(gsf.SEED + 2)
    best_overall_ap, best_overall_cfg, best_overall_state = -1.0, None, None
    for trial_i in range(args.n_trials):
        cfg = sample_cfg(sample_rng)
        tag = f'patchgrid256_{args.encoder}_{trial_i}'
        t0 = time.time()
        try:
            full_ap, per_label, state = run_trial(cfg)
        except Exception as e:
            log_result({'tag': tag, 'cfg': cfg, 'error': str(e), 'traceback': traceback.format_exc()})
            continue
        dt = time.time() - t0
        log_result({'tag': tag, 'cfg': cfg, 'full_val_macro_ap': full_ap, 'full_val_per_label': per_label, 'seconds': dt})
        if full_ap > best_overall_ap:
            best_overall_ap, best_overall_cfg, best_overall_state = full_ap, cfg, state

    print(f'Search done: best trial full-val macro AP = {best_overall_ap:.4f}  cfg={best_overall_cfg}', flush=True)

    if best_overall_ap <= baseline_ap:
        print(f'Did NOT beat baseline ({baseline_ap:.4f}) — not promoting.', flush=True)
        return

    print(f'NEW BEST {best_overall_ap:.4f} > baseline {baseline_ap:.4f} — promoting.', flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_overall_state, OUT_DIR / 'best_model.pt')

    # Recompute per-label breakdown for the record (re-run the winning model once more).
    model = MouseFrameClassifier(
        emb_dim=EMB_DIM, n_heads=best_overall_cfg['n_heads'], hidden_dim=best_overall_cfg['hidden_dim'],
        use_patch_grid=True, dropout=best_overall_cfg['dropout'],
    ).to(dev)
    model.load_state_dict(best_overall_state)
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

    with open(OUT_DIR / 'config.json', 'w') as f:
        json.dump({'cfg': best_overall_cfg, 'val_pools': sorted(val_pool_set), 'n_patches': n_patches_full,
                   'max_train_frames': args.max_train_frames, 'search_epochs_cap': args.search_epochs,
                   'patience': args.patience, 'n_trials': args.n_trials,
                   'best_ap': best_overall_ap, 'best_per_label': per_label, 'encoder': args.encoder}, f, indent=2)
    print(f'Saved {OUT_DIR}/{{best_model.pt,config.json}}', flush=True)


if __name__ == '__main__':
    main()
