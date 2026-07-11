"""
Full training run of the patch-grid (attention-pooled 4x4 DINOv2 tokens)
variant of the mouse behavior classifier, on the standard 80/20 pool split,
with per-epoch validation (eval_every=1) so a full loss curve is available.
Mirrors train_cls_final.py (same CFG, same seed/split) for a direct
comparison against the CLS-token baseline.

Writes results/mice_behavior/patchgrid/{best_model.pt, history.json,
config.json, report.png, roc_pr_data.npz}.

Usage:
    python scripts/mice_behavior/train_patchgrid_final.py
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.build_pair_labels import build_pair_labels
from src.mice_behavior.fast_data import FastBatchData, load_patchgrid_embeddings
from src.mice_behavior.model import MouseBehaviorClassifier
from src.mice_behavior.pools import load_obs_to_pool_map
from src.mice_behavior.report import collect_val_predictions_fast, generate_report
from src.mice_behavior.train import train_fast

DATA_DIR = Path('./data')
DATASET_DIR = Path('./dataset')
RESULTS_DIR = Path('./results/mice_behavior')
SEED = 42
ENCODER, TOKEN = 'dinov2', 'class_l-2'
PATCH_GRID_DIR = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / 'dinov2' / 'patch_grid4'

# Must match train_cls_final.py's CFG exactly for a fair comparison.
# n_heads raised 1->8 (more attention subspaces, same total width) after the first full runs
# showed severe overfitting (best macro PR-AUC hit within ~15 epochs, then val_loss climbed
# for the remaining 85). Dropout/weight_decay/early_stop_patience are train_fast() defaults now.
CFG = dict(n_heads=8, context_k=2, hidden_dim=256, neg_ratio=10, loss_type='ce')
N_EPOCHS = 100

# CLS is stable at train_fast()'s 1e-3 default; this variant's extra PatchAttnPool
# stage repeatedly collapsed into a dead (uniform-softmax) state at 1e-3 even with
# grad clipping down to max_norm=0.5.
LR = 3e-4

# The available GPU has limited VRAM. The default batch_size=4096
# (patch-grid: ~1GB/batch of fp32 context, plus the GPU-resident val array) OOM'd
# during backward(); 1024 leaves comfortable headroom.
BATCH_SIZE = 1024

# Patch-grid's full train split is ~15GB GPU-resident (16x CLS's per-frame footprint) —
# never fits alongside the val set on a GPU with limited VRAM, silently falling back to slow
# CPU-gather (~24-29s/epoch, confirmed present in every patch-grid run this session). Bounding
# to 200k frames (every positive-containing frame + a big random negative pool, ~2.6GB
# GPU-resident) cut that to ~8s/epoch — a real, verified ~3x speedup, not just fewer negatives
# used per epoch (positives are unaffected; all are always kept).
MAX_TRAIN_FRAMES = 200_000


def main():
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

    out_dir = RESULTS_DIR / 'patchgrid'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Training patch-grid model: {CFG}, {N_EPOCHS} epochs, eval every epoch...')
    result = train_fast(
        annotations_csv=str(annotations_csv),
        pair_labels_parquet=str(pair_labels_path),
        embeddings_path=str(cls_embeddings_path),
        output_dir=str(out_dir),
        train_obs_ids=train_obs,
        val_obs_ids=val_obs,
        context_k=CFG['context_k'],
        emb_dim=emb_dim,
        n_heads=CFG['n_heads'],
        hidden_dim=CFG['hidden_dim'],
        n_epochs=N_EPOCHS,
        neg_ratio=CFG['neg_ratio'],
        loss_type=CFG['loss_type'],
        lr=LR,
        device='cuda',
        seed=SEED,
        verbose=True,
        use_patch_grid=True,
        patch_embeddings_path=str(PATCH_GRID_DIR / 'embeddings.npy'),
        patch_global_idx_path=str(PATCH_GRID_DIR / 'global_idx.npy'),
        n_patches=16,
        eval_every=1,
        batch_size=BATCH_SIZE,
        max_train_frames=MAX_TRAIN_FRAMES,
    )
    history = result['history']
    print(f"Best macro PR-AUC: {result['best_pr_auc']:.4f}  {result['best_per_class']}")

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({'cfg': CFG, 'val_pools': sorted(val_pool_set), 'n_epochs': N_EPOCHS,
                   'best_pr_auc': result['best_pr_auc'], 'best_per_class': result['best_per_class'],
                   'emb_dim': emb_dim}, f, indent=2)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MouseBehaviorClassifier(
        emb_dim=emb_dim, n_heads=CFG['n_heads'], hidden_dim=CFG['hidden_dim'], use_patch_grid=True,
    ).to(dev)
    model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=dev, weights_only=True))
    model.eval()

    val_data = FastBatchData(
        str(annotations_csv), str(pair_labels_path), val_obs, CFG['context_k'], emb_dim,
        load_patchgrid_embeddings(str(PATCH_GRID_DIR / 'embeddings.npy'), str(PATCH_GRID_DIR / 'global_idx.npy'), 16, emb_dim),
        n_patches=16,
    )
    probs, labels = collect_val_predictions_fast(model, val_data, dev)
    generate_report(probs, labels, history, 'Patch-grid (attention-pooled) mouse behavior classifier', CFG, out_dir)
    print(f'Saved {out_dir / "report.png"}')


if __name__ == '__main__':
    main()
