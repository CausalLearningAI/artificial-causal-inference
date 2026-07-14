"""
Quick diagnostic runs for the per-frame classifier, testing two hypotheses
raised after inspecting the promoted model's overfitting (train loss falls
smoothly while val loss rises/gets noisier past ~epoch 5):

  --mode reg:       same winning hyperparams (n_heads=8, context_k=2, stride=1,
                     neg_ratio=5, lr=1e-3) but a smaller/more-regularized head
                     (hidden_dim 512->128, dropout 0.2->0.4, weight_decay 1e-4->1e-3)
                     to see if capacity/regularization was the bottleneck.
  --mode patchgrid: same winning hyperparams, but patch-grid (16x4x4 DINOv2
                     tokens) instead of the pooled CLS embedding, to see if the
                     frozen CLS token was throwing away load-bearing spatial detail.

Not a full hyperparameter search — one run each, to see which lever (if either)
is worth investing a real search budget in. Writes to
results/vision/mice/frame/diag_{reg,patchgrid}/.

Usage:
    python scripts/mice_behavior/diag_frame_variants.py --mode reg
    python scripts/mice_behavior/diag_frame_variants.py --mode patchgrid
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.build_pair_labels import build_pair_labels
from src.mice_behavior.batch_data import FrameBatchData, load_cls_embeddings, load_patchgrid_embeddings
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import load_obs_to_pool_map
from src.mice_behavior.report import collect_frame_val_predictions, generate_frame_report
from src.mice_behavior.train import train_frame

DATA_DIR = Path('./data')
DATASET_DIR = Path('./dataset')
RESULTS_DIR = Path('./results/vision/mice')
SEED = 42
ENCODER, TOKEN = 'dinov2', 'class_l-2'
PATCH_GRID_DIR = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / 'dinov2' / 'patch_grid4'

# Shared across both variants — only what each hypothesis is actually testing differs.
BASE_CFG = dict(n_heads=8, context_k=2, stride=1, neg_ratio=5, lr=1e-3)
N_EPOCHS = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['reg', 'patchgrid'], required=True)
    args = parser.parse_args()

    pair_labels_path = build_pair_labels(DATA_DIR, DATASET_DIR, overwrite=False)
    annotations_csv = DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    cls_embeddings_path = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / ENCODER / TOKEN / 'embeddings.npy'
    n_frames = sum(1 for _ in open(annotations_csv)) - 1
    emb_dim = cls_embeddings_path.stat().st_size // (4 * n_frames)

    obs_to_pool = load_obs_to_pool_map(DATA_DIR)
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    pools = sorted({obs_to_pool[o] for o in all_obs})
    rng = random.Random(SEED)
    shuffled = pools[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.2))
    val_pool_set = set(shuffled[:n_val])
    train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]
    print(f'Split: {len(train_obs)} train obs / {len(val_obs)} val obs ({len(pools)-n_val}/{n_val} pools)')

    out_dir = RESULTS_DIR / 'frame' / f'diag_{args.mode}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'reg':
        cfg = dict(BASE_CFG, hidden_dim=128, dropout=0.4, weight_decay=1e-3)
        use_patch_grid = False
        batch_size = 4096
        extra = {}
    else:
        cfg = dict(BASE_CFG, hidden_dim=512, dropout=0.2, weight_decay=1e-4)
        use_patch_grid = True
        batch_size = 1024
        extra = dict(
            patch_embeddings_path=str(PATCH_GRID_DIR / 'embeddings.npy'),
            patch_global_idx_path=str(PATCH_GRID_DIR / 'global_idx.npy'),
            n_patches=16,
        )

    print(f'[{args.mode}] cfg={cfg} use_patch_grid={use_patch_grid}')
    result = train_frame(
        annotations_csv=str(annotations_csv),
        pair_labels_parquet=str(pair_labels_path),
        embeddings_path=str(cls_embeddings_path),
        output_dir=str(out_dir),
        train_obs_ids=train_obs,
        val_obs_ids=val_obs,
        context_k=cfg['context_k'],
        stride=cfg['stride'],
        emb_dim=emb_dim,
        n_heads=cfg['n_heads'],
        hidden_dim=cfg['hidden_dim'],
        n_epochs=N_EPOCHS,
        neg_ratio=cfg['neg_ratio'],
        lr=cfg['lr'],
        dropout=cfg['dropout'],
        weight_decay=cfg['weight_decay'],
        batch_size=batch_size,
        device='cuda',
        seed=SEED,
        verbose=True,
        use_patch_grid=use_patch_grid,
        eval_every=1,
        **extra,
    )
    history = result['history']
    print(f"[{args.mode}] internal (subsampled) best_ap: {result['best_ap']:.4f}  {result['best_per_label']}")

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({'cfg': cfg, 'use_patch_grid': use_patch_grid, 'val_pools': sorted(val_pool_set),
                   'n_epochs': N_EPOCHS, 'internal_best_ap': result['best_ap'],
                   'internal_best_per_label': result['best_per_label'], 'emb_dim': emb_dim}, f, indent=2)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MouseFrameClassifier(
        emb_dim=emb_dim, n_heads=cfg['n_heads'], hidden_dim=cfg['hidden_dim'],
        use_patch_grid=use_patch_grid, dropout=cfg['dropout'],
    ).to(dev)
    model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=dev, weights_only=True))
    model.eval()

    load_fn = (
        load_patchgrid_embeddings(str(PATCH_GRID_DIR / 'embeddings.npy'), str(PATCH_GRID_DIR / 'global_idx.npy'), 16, emb_dim)
        if use_patch_grid else load_cls_embeddings(str(cls_embeddings_path), emb_dim)
    )
    val_data = FrameBatchData(
        str(annotations_csv), str(pair_labels_path), val_obs, cfg['context_k'], emb_dim, load_fn,
        n_patches=16 if use_patch_grid else None, stride=cfg['stride'],
    )
    probs, labels = collect_frame_val_predictions(model, val_data, dev)
    import numpy as np
    from sklearn.metrics import average_precision_score
    full_per_label = {name: average_precision_score(labels[:, i], probs[:, i]) for i, name in enumerate(['nt', 'nn'])}
    full_macro = float(np.mean(list(full_per_label.values())))
    print(f'[{args.mode}] FULL-VAL macro AP: {full_macro:.4f}  {full_per_label}')
    with open(out_dir / 'config.json') as f:
        cfg_out = json.load(f)
    cfg_out['full_val_ap'] = full_macro
    cfg_out['full_val_per_label'] = full_per_label
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(cfg_out, f, indent=2)

    generate_frame_report(probs, labels, history, f'Per-frame diagnostic [{args.mode}]', cfg, out_dir)
    print(f'Saved {out_dir / "report.png"}')


if __name__ == '__main__':
    main()
