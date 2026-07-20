"""
One-off: finish promoting patchgrid_concat after grid_search_frame.py's --variant
patchgrid_concat search found a winning config (full-val macro AP 0.1219, 85 trials)
but its own final 100-epoch retrain crashed with CUDA OOM (10.58GB card; train+val
data residency at the original 60k-frame bound already left ~10.8GB in use before
any batch compute). Reruns just the final retrain/promotion at a safer memory
footprint (max_train_frames 60k -> 40k) instead of redoing the whole 3.5h search.

Usage:
    python scripts/mice_behavior/finish_concat_promotion.py
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

# Winning config from the patchgrid_concat search (results/vision/mice/frame/search/log.jsonl,
# tag patchgrid_concat_26, full_val_frame_macro_ap=0.12193070169473455).
best_cfg = dict(n_heads=2, context_k=2, stride=1, hidden_dim=128, neg_ratio=10, dropout=0.2, weight_decay=0.0, lr=0.0003)
variant_emb_dim = 2 * emb_dim

load_fn = gsf.cached_loader(gsf.load_patchgrid_concat_embeddings(
    str(gsf.PATCH_GRID_DIR / 'embeddings.npy'), str(gsf.PATCH_GRID_DIR_DINOV3 / 'embeddings.npy'),
    str(gsf.PATCH_GRID_DIR / 'global_idx.npy'), 16, emb_dim, emb_dim,
))

MAX_TRAIN_FRAMES = 40_000
BATCH_SIZE = 128

print('Final retrain (concat, reduced-memory retry)...', flush=True)
result = train_frame(
    annotations_csv=str(annotations_csv), pair_labels_parquet=str(pair_labels_path),
    embeddings_path='', output_dir=str(gsf.TMP_DIR / 'final_patchgrid_concat_retry'),
    train_obs_ids=train_obs, val_obs_ids=val_obs, context_k=best_cfg['context_k'], stride=best_cfg['stride'],
    emb_dim=variant_emb_dim, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'], n_epochs=gsf.FINAL_EPOCHS,
    neg_ratio=best_cfg['neg_ratio'], lr=best_cfg['lr'], dropout=best_cfg['dropout'], weight_decay=best_cfg['weight_decay'],
    device='cuda', seed=gsf.SEED, verbose=True, use_patch_grid=True, eval_every=1,
    embeddings_loader=load_fn, n_patches=16, batch_size=BATCH_SIZE, max_train_frames=MAX_TRAIN_FRAMES,
)
model = result['model']
dev = next(model.parameters()).device
final_score, final_per_label = gsf.full_val_frame_macro_ap(
    model, dev, annotations_csv, pair_labels_path, val_obs,
    best_cfg['context_k'], best_cfg['stride'], variant_emb_dim, load_fn, n_patches=16,
)
print(f'FINAL full-val macro AP: {final_score:.4f}  {final_per_label}', flush=True)

out_dir = gsf.FRAME_DIR / 'patchgrid4x4_concat'
out_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out_dir / 'best_model.pt')
with open(out_dir / 'history.json', 'w') as f:
    json.dump(result['history'], f, indent=2)
with open(out_dir / 'config.json', 'w') as f:
    json.dump({'cfg': best_cfg, 'val_pools': sorted(val_pool_set), 'n_epochs': gsf.FINAL_EPOCHS,
               'best_ap': final_score, 'best_per_label': final_per_label,
               'emb_dim': variant_emb_dim, 'promoted_by_search': True,
               'max_train_frames_used_for_final_retrain': MAX_TRAIN_FRAMES}, f, indent=2)

val_data = FrameBatchData(
    str(annotations_csv), str(pair_labels_path), val_obs, best_cfg['context_k'], variant_emb_dim, load_fn,
    n_patches=16, stride=best_cfg['stride'],
)
probs, labels = collect_frame_val_predictions(model, val_data, dev)
generate_frame_report(probs, labels, result['history'],
                       'Patch-grid (attention-pooled) per-frame mouse behavior classifier — DINOv2+DINOv3 concat (L2-normalized)',
                       best_cfg, out_dir)
print(f'Saved {out_dir}/{{best_model.pt,config.json,history.json,report.png}}', flush=True)

with open(gsf.SEARCH_DIR / 'SUMMARY.md', 'a') as f:
    f.write('\n## patchgrid_concat final retrain (retried at reduced memory footprint after original OOM)\n')
    f.write(f'- **PROMOTED** to results/vision/mice/frame/patchgrid_concat/ (full-val score {final_score:.4f})\n\n')
print('Done.', flush=True)
