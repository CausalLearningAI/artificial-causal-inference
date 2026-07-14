"""
Visual error overview for the per-frame behavior classifier: for each behavior
(nt, nn), shows k=10 random example frames from each confusion-matrix bucket
(TP/TN/FP/FN, p>0.5 decision rule), with the bucket's share of the full
validation set annotated next to its row label.

Unlike error_overview.py (pairwise model), there's no a1/a2 mouse-identity to
annotate per example — this model only ever sees "did a behavior happen in
this frame", not who was involved.

Usage:
    python scripts/mice_behavior/error_overview_frame.py
"""
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.build_pair_labels import build_pair_labels
from src.mice_behavior.batch_data import FrameBatchData, load_cls_embeddings
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import load_obs_to_pool_map
from src.mice_behavior.report import collect_frame_val_predictions

DATA_DIR = Path('./data')
DATASET_DIR = Path('./dataset')
RESULTS_DIR = Path('./results/vision/mice')
SEED = 42
ENCODER, TOKEN = 'dinov2', 'class_l-2'
K = 10
BUCKETS = ['TP', 'TN', 'FP', 'FN']
BEHAVIORS = [(0, 'nt', 'Nose-Tail Sniffing'), (1, 'nn', 'Nose-Nose Sniffing')]


def fmt_pct(pct):
    return f'{pct:.0f}%' if pct >= 1 else f'{pct:.2f}%'


def main():
    pair_labels_path = build_pair_labels(DATA_DIR, DATASET_DIR, overwrite=False)
    annotations_csv = DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    cls_embeddings_path = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / ENCODER / TOKEN / 'embeddings.npy'
    n_frames = sum(1 for _ in open(annotations_csv)) - 1
    emb_dim = cls_embeddings_path.stat().st_size // (4 * n_frames)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    obs_to_pool = load_obs_to_pool_map(DATA_DIR)
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    pools = sorted({obs_to_pool[o] for o in all_obs})
    rng = random.Random(SEED)
    shuffled = pools[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.2))
    val_pool_set = set(shuffled[:n_val])
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]

    out_dir = RESULTS_DIR / 'frame'
    cfg = json.load(open(out_dir / 'config.json'))['cfg']
    model = MouseFrameClassifier(
        emb_dim=emb_dim, n_heads=cfg['n_heads'], hidden_dim=cfg['hidden_dim'], dropout=cfg.get('dropout', 0.1),
    ).to(dev)
    model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=dev, weights_only=True))
    model.eval()

    load_fn = load_cls_embeddings(str(cls_embeddings_path), emb_dim)
    val_data = FrameBatchData(
        str(annotations_csv), str(pair_labels_path), val_obs, cfg['context_k'], emb_dim, load_fn,
        stride=cfg.get('stride', 1),
    )
    probs, labels = collect_frame_val_predictions(model, val_data, dev)
    pred = probs > 0.5
    gi = val_data.gi
    n_total = len(labels)

    frame_paths = pd.read_csv(annotations_csv, usecols=['frame_path'])['frame_path'].values

    row_heights, row_plan = [], []
    for bi, (behavior_c, behavior_name, behavior_title) in enumerate(BEHAVIORS):
        row_heights.append(0.45); row_plan.append(('title', behavior_c, behavior_title))
        for bucket in BUCKETS:
            row_heights.append(1.0); row_plan.append(('bucket', behavior_c, bucket))
        if bi < len(BEHAVIORS) - 1:
            row_heights.append(0.35); row_plan.append(('spacer', None, None))

    fig = plt.figure(figsize=(2 * K, 2 * sum(row_heights)))
    gs = fig.add_gridspec(len(row_heights), K, height_ratios=row_heights, hspace=0.15, wspace=0.05)
    pick_rng = np.random.default_rng(SEED)

    for r, (kind, behavior_c, payload) in enumerate(row_plan):
        if kind == 'spacer':
            continue
        if kind == 'title':
            ax = fig.add_subplot(gs[r, :])
            ax.axis('off')
            ax.text(0.5, 0.5, f'{payload} (validation set)', ha='center', va='center', fontsize=13, fontweight='bold')
            continue

        bucket = payload
        is_true = labels[:, behavior_c].astype(bool)
        is_pred = pred[:, behavior_c]
        mask = {
            'TP': is_true & is_pred, 'TN': ~is_true & ~is_pred,
            'FP': ~is_true & is_pred, 'FN': is_true & ~is_pred,
        }[bucket]
        idx = np.where(mask)[0]
        pct = 100 * len(idx) / n_total
        row_label = f'{bucket}\n({fmt_pct(pct)})'
        chosen = pick_rng.choice(idx, size=min(K, len(idx)), replace=False) if len(idx) else np.array([], dtype=int)

        for col in range(K):
            ax = fig.add_subplot(gs[r, col])
            ax.axis('off')
            if col < len(chosen):
                i = chosen[col]
                img_path = DATASET_DIR / frame_paths[gi[i]]
                try:
                    ax.imshow(Image.open(img_path))
                except Exception:
                    ax.text(0.5, 0.5, '(image not found)', ha='center', va='center', fontsize=6, transform=ax.transAxes)
                ax.set_title(f'p={probs[i, behavior_c]:.2f}', fontsize=7)
            if col == 0:
                ax.text(-0.15, 0.5, row_label, transform=ax.transAxes, fontsize=9, fontweight='bold',
                        ha='right', va='center', rotation=0)

    fig.suptitle(f'frame: error overview (p>0.5 decision rule, k={K} random examples per bucket)', fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.02)
    out_path = out_dir / 'error_overview.png'
    fig.savefig(out_path, dpi=120)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
