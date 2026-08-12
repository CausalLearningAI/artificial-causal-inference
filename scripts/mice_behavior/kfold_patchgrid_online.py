"""
Pool-level k-fold cross-validation for the patchgrid256 (all raw patch tokens,
online-computed) classifier — measures how much full-val macro AP varies across
different held-out pool groups, since with only ~24 total pools any single fixed
split carries real sampling variance (confirmed: nt prevalence swung from 1.97% to
1.14% between two different-but-equally-valid splits, enough to explain most of an
apparent "regression" that was actually just a harder validation subset).

Reuses the ALREADY-FOUND best hyperparameters from search_patchgrid_online.py
(results/vision/mice/frame/patchgrid256_{encoder}/config.json) rather than
re-searching per fold — this is a variance-of-the-estimate question, not a
hyperparameter question, so re-running the 15-trial search k times would be
wasteful. Loads the frozen encoder once and reuses it across all k folds; each
fold still needs its own encode pass (the needed raw frames differ per fold), but
training itself is a single run per fold (no early-stopping-based model selection
across many configs, just the one fixed config).

Usage:
    python scripts/mice_behavior/kfold_patchgrid_online.py --encoder dinov2 --k 5
    python scripts/mice_behavior/kfold_patchgrid_online.py --encoder dinov3 --k 5
"""
import argparse
import json
import sys
import time
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
from src.mice_behavior.pools import get_kfold_assignment
from src.dataset.get_dataset import load_dataset
from train_patchgrid_online import dummy_loader, _ImageDataset

MODEL_IDS = {
    'dinov2': 'facebook/dinov2-base',
    'dinov3': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
}
N_PATCHES = {'dinov2': 256, 'dinov3': 196}
EMB_DIM = 768


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', required=True, choices=sorted(MODEL_IDS))
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--max-train-frames', type=int, default=200_000)
    p.add_argument('--n-epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--encode-batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--only-fold', type=int, default=None,
                    help='Run just this one fold index (e.g. to finish a run that hit its time limit '
                         'partway through, without re-running already-completed folds).')
    args = p.parse_args()
    MODEL_ID = MODEL_IDS[args.encoder]
    n_patches_full = N_PATCHES[args.encoder]

    best_cfg = json.load(open(gsf.FRAME_DIR / f'patchgrid256_{args.encoder}' / 'config.json'))['cfg']
    print(f'Reusing already-found best cfg for {args.encoder}: {best_cfg}', flush=True)

    OUT_DIR = gsf.FRAME_DIR / f'patchgrid256_{args.encoder}_kfold'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUT_DIR / 'log.jsonl'

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
    fold_of_pool = get_kfold_assignment(pools, k=args.k, seed=gsf.SEED)
    print(f'{len(pools)} pools across {args.k} folds: '
          f'{ {f: sum(1 for v in fold_of_pool.values() if v == f) for f in range(args.k)} }', flush=True)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {MODEL_ID} on {dev} (once, reused across all folds)...', flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    encoder = AutoModel.from_pretrained(MODEL_ID).to(dev)
    encoder.eval()
    encoder.requires_grad_(False)
    n_register_tokens = getattr(encoder.config, 'num_register_tokens', 0)
    n_prefix_tokens = 1 + n_register_tokens

    print('Loading full-frame HF dataset (raw JPEGs)...', flush=True)
    hf_dataset = load_dataset(subject='mice', version='v1', dataset_root=str(gsf.DATASET_DIR), frame_type='full')

    fold_range = [args.only_fold] if args.only_fold is not None else list(range(args.k))
    fold_scores = []
    for fold in fold_range:
        t_fold0 = time.time()
        val_pool_set = {p for p, f in fold_of_pool.items() if f == fold}
        train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
        val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]
        print(f'\n=== Fold {fold}/{args.k - 1}: val_pools={sorted(val_pool_set)} '
              f'({len(train_obs)} train obs / {len(val_obs)} val obs) ===', flush=True)

        train_meta = FrameBatchData(
            str(annotations_csv), str(pair_labels_path), train_obs, best_cfg['context_k'], 1,
            dummy_loader(1, 1), n_patches=1, stride=best_cfg['stride'], max_frames=args.max_train_frames, seed=gsf.SEED,
        )
        val_meta = FrameBatchData(
            str(annotations_csv), str(pair_labels_path), val_obs, best_cfg['context_k'], 1,
            dummy_loader(1, 1), n_patches=1, stride=best_cfg['stride'],
        )
        del train_meta.flat, val_meta.flat

        def _needed_raw_indices(meta):
            abs_idx = meta.gi[:, None] + meta.offsets_grid[None, :]
            return np.unique(abs_idx[~meta.pad_mask])

        all_needed = np.unique(np.concatenate([_needed_raw_indices(train_meta), _needed_raw_indices(val_meta)]))
        print(f'  encoding {len(all_needed):,} unique frames for this fold...', flush=True)
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
        print(f'  encoding done in {(time.time()-t_encode0)/60:.1f} min', flush=True)
        del loader

        offsets = train_meta.offsets_grid

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
        n_neg_draw = min(len(neg_idx), best_cfg['neg_ratio'] * n_pos)
        rng = np.random.default_rng(gsf.SEED)
        neg_sample = rng.choice(neg_idx, size=n_neg_draw, replace=False)
        epoch_idx_fixed = np.concatenate([pos_idx, neg_sample])

        pos_counts = labels_all[pos_idx].sum(axis=0).clip(min=1)
        pos_weight = torch.tensor(max(n_neg_draw, 1) / pos_counts, dtype=torch.float32).to(dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        lr = min(best_cfg['lr'], 3e-4)
        model = MouseFrameClassifier(
            emb_dim=EMB_DIM, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'],
            use_patch_grid=True, dropout=best_cfg['dropout'],
        ).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=best_cfg['weight_decay'])
        amp_enabled = dev.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

        shuffle_rng = np.random.default_rng(gsf.SEED)
        best_ap, epochs_since_best, best_state = -1.0, 0, None
        for epoch in range(1, args.n_epochs + 1):
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
                for b0 in range(0, len(val_meta), args.batch_size):
                    batch_idx = np.arange(b0, min(b0 + args.batch_size, len(val_meta)))
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
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
                    logits = model(ctx.float(), offsets=offs, key_padding_mask=mask)
                all_probs.append(torch.sigmoid(logits).float().cpu())
                all_labels.append(lbl.cpu())
        probs = torch.cat(all_probs).numpy()
        labels_np = torch.cat(all_labels).numpy()
        per_label = {name: average_precision_score(labels_np[:, i], probs[:, i]) for i, name in enumerate(['nt', 'nn'])}
        fold_ap = float(np.mean(list(per_label.values())))
        fold_scores.append(fold_ap)
        log_result({'fold': fold, 'val_pools': sorted(val_pool_set), 'full_val_macro_ap': fold_ap,
                    'full_val_per_label': per_label, 'seconds': time.time() - t_fold0})

        del cache
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    if args.only_fold is not None:
        print(f'\n=== fold {args.only_fold} done (--only-fold run, not writing an aggregate summary) ===', flush=True)
    else:
        mean_ap, std_ap = float(np.mean(fold_scores)), float(np.std(fold_scores))
        print(f'\n=== {args.encoder} k={args.k}-fold CV: mean={mean_ap:.4f} std={std_ap:.4f} '
              f'folds={[round(s, 4) for s in fold_scores]} ===', flush=True)
        with open(OUT_DIR / 'summary.json', 'w') as f:
            json.dump({'encoder': args.encoder, 'cfg': best_cfg, 'k': args.k,
                       'fold_scores': fold_scores, 'mean_ap': mean_ap, 'std_ap': std_ap}, f, indent=2)
    print(f'Saved {OUT_DIR}/summary.json', flush=True)


if __name__ == '__main__':
    main()
