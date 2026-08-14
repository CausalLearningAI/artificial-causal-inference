"""
Retrains the already-confirmed-best DINOv2 patch-grid config
(n_heads=8, hidden_dim=384, dropout=0.4, weight_decay=1e-4, lr=3e-4, neg_ratio=15)
with max_train_frames raised to 1,000,000 — effectively unbounded (the full annotated
pool is ~756,000 train-observation frames) — to test whether the negative-sample
budget itself (not the neg_ratio hyperparameter, which the 200,000-frame search bound
already saturated at ~1.9:1 delivered vs. 15:1 requested) was the bottleneck.

Writes to results/vision/mice/frame/patchgrid_fulldata/ (kept separate from the
currently-promoted patchgrid/ until this is confirmed to actually beat it).

Usage:
    python scripts/mice_behavior/retrain_patchgrid_fulldata.py
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.report import collect_frame_val_predictions, generate_frame_report
from src.mice_behavior.train import train_frame
from src.mice_behavior.head_cfg import get_head_cfg, LEGACY_4X4_BASELINE_AP

pair_labels_path = gsf.build_pair_labels(gsf.DATA_DIR, gsf.DATASET_DIR, overwrite=False)
annotations_csv = gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
cls_embeddings_path = gsf.DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / gsf.ENCODER / gsf.TOKEN / 'embeddings.npy'
n_frames = sum(1 for _ in open(annotations_csv)) - 1
emb_dim = cls_embeddings_path.stat().st_size // (4 * n_frames)

obs_to_pool = gsf.load_obs_to_pool_map(gsf.DATA_DIR)
all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
pools = sorted({obs_to_pool[o] for o in all_obs})
rng_split = random.Random(gsf.SEED)
shuffled = pools[:]
rng_split.shuffle(shuffled)
n_val = max(1, int(len(shuffled) * 0.2))
val_pool_set = set(shuffled[:n_val])
train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]

best_cfg = get_head_cfg()
print(f'Reusing confirmed-best patchgrid cfg: {best_cfg}', flush=True)

MAX_TRAIN_FRAMES = 1_000_000  # effectively unbounded — full annotated pool is ~756,000 train frames
BATCH_SIZE = 4096  # H100 headroom is large; single-encoder 768-dim fp16 patch-grid is small per-sample

load_fn = gsf.cached_loader(gsf.load_patchgrid_embeddings(
    str(gsf.PATCH_GRID_DIR / 'embeddings.npy'), str(gsf.PATCH_GRID_DIR / 'global_idx.npy'), 16, emb_dim,
))

print('Training on the full negative pool (no 200k cap)...', flush=True)
result = train_frame(
    annotations_csv=str(annotations_csv), pair_labels_parquet=str(pair_labels_path),
    embeddings_path='', output_dir=str(gsf.TMP_DIR / 'patchgrid4x4_dinov2_1Mframes'),
    train_obs_ids=train_obs, val_obs_ids=val_obs, context_k=best_cfg['context_k'], stride=best_cfg['stride'],
    emb_dim=emb_dim, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'], n_epochs=gsf.FINAL_EPOCHS,
    neg_ratio=best_cfg['neg_ratio'], lr=best_cfg['lr'], dropout=best_cfg['dropout'], weight_decay=best_cfg['weight_decay'],
    device='cuda', seed=gsf.SEED, verbose=True, use_patch_grid=True, eval_every=1,
    embeddings_loader=load_fn, n_patches=16, batch_size=BATCH_SIZE, max_train_frames=MAX_TRAIN_FRAMES,
)
model = result['model']
dev = next(model.parameters()).device
final_score, final_per_label = gsf.full_val_frame_macro_ap(
    model, dev, annotations_csv, pair_labels_path, val_obs,
    best_cfg['context_k'], best_cfg['stride'], emb_dim, load_fn, n_patches=16,
)
print(f'FINAL full-val macro AP: {final_score:.4f}  {final_per_label}', flush=True)

baseline_ap = LEGACY_4X4_BASELINE_AP
print(f'Currently-promoted patchgrid baseline: {baseline_ap:.4f}', flush=True)

out_dir = gsf.FRAME_DIR / 'patchgrid4x4_dinov2_1Mframes'
out_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out_dir / 'best_model.pt')
with open(out_dir / 'history.json', 'w') as f:
    json.dump(result['history'], f, indent=2)
with open(out_dir / 'config.json', 'w') as f:
    json.dump({'cfg': best_cfg, 'val_pools': sorted(val_pool_set), 'n_epochs': gsf.FINAL_EPOCHS,
               'best_ap': final_score, 'best_per_label': final_per_label,
               'emb_dim': emb_dim, 'max_train_frames': MAX_TRAIN_FRAMES,
               'baseline_patchgrid_200k_ap': baseline_ap}, f, indent=2)

val_data = FrameBatchData(
    str(annotations_csv), str(pair_labels_path), val_obs, best_cfg['context_k'], emb_dim, load_fn,
    n_patches=16, stride=best_cfg['stride'],
)
probs, labels = collect_frame_val_predictions(model, val_data, dev)
generate_frame_report(probs, labels, result['history'],
                       'Patch-grid DINOv2, full negative pool (max_train_frames=1M)', best_cfg, out_dir)
print(f'Saved {out_dir}/{{best_model.pt,config.json,history.json,report.png}}', flush=True)
print(f'{"BEAT baseline" if final_score > baseline_ap else "did NOT beat baseline"}: {final_score:.4f} vs {baseline_ap:.4f}', flush=True)
