"""
One-off: compute full-val macro AP for a saved patchgrid256 checkpoint whose training
job died (TIMEOUT) before reaching the script's own final full-val evaluation pass --
e.g. the 504px native-resolution run (job 62686220), which got to epoch 12/30 with a
promising subsample AP (0.7178) but never got to run its own end-of-training full-val
pass before the 12h time limit hit. Only re-encodes the (much smaller) val set at the
same input resolution, not the full train budget -- cheap.

Usage:
    python scripts/mice_behavior/eval_val_ap_only.py --tag res504 --input-size 504
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import get_fixed_val_pools
from src.mice_behavior.head_cfg import get_head_cfg
from src.dataset.get_dataset import load_dataset
from train_patchgrid_online import dummy_loader, _ImageDataset

MODEL_ID = 'facebook/dinov2-base'
EMB_DIM = 768
PATCH_SIZE = 14

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
p.add_argument('--input-size', type=int, default=None)
p.add_argument('--blur-to', type=int, default=None)
p.add_argument('--cross-attn-dim', type=int, default=None)
p.add_argument('--patch-pool-dim', type=int, default=None)
p.add_argument('--batch-size', type=int, default=128)
args = p.parse_args()
n_patches_full = 256 if args.input_size is None else (args.input_size // PATCH_SIZE) ** 2

OUT_DIR = gsf.FRAME_DIR / f'patchgrid256_dinov2_{args.tag}'
best_cfg = get_head_cfg()
print(f'Evaluating {OUT_DIR}/best_model.pt at input_size={args.input_size} (n_patches={n_patches_full})', flush=True)

pair_labels_path = gsf.build_pair_labels(gsf.DATA_DIR, gsf.DATASET_DIR, overwrite=False)
annotations_csv = gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
obs_to_pool = gsf.load_obs_to_pool_map(gsf.DATA_DIR)
all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
pools = sorted({obs_to_pool[o] for o in all_obs})
val_pool_set = get_fixed_val_pools(pools)
val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]

val_meta = FrameBatchData(
    str(annotations_csv), str(pair_labels_path), val_obs, best_cfg['context_k'], 1,
    dummy_loader(1, 1), n_patches=1, stride=best_cfg['stride'],
)
del val_meta.flat
print(f'{len(val_meta):,} val samples', flush=True)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
encoder = AutoModel.from_pretrained(MODEL_ID).to(dev)
encoder.eval()
encoder.requires_grad_(False)

hf_dataset = load_dataset(subject='mice', version='v1', dataset_root=str(gsf.DATASET_DIR), frame_type='full')

need = np.unique((val_meta.gi[:, None] + val_meta.offsets_grid[None, :])[~val_meta.pad_mask])
print(f'Encoding {len(need):,} unique val frames...', flush=True)
t0 = time.time()
loader = DataLoader(_ImageDataset(hf_dataset, need, processor, input_size=args.input_size, blur_to=args.blur_to), batch_size=128,
                     num_workers=16, pin_memory=(dev.type == 'cuda'), shuffle=False, prefetch_factor=4, persistent_workers=True)
cache = torch.empty((len(need), n_patches_full, EMB_DIM), dtype=torch.float16)
idx_of_global = {int(g): i for i, g in enumerate(need)}
cursor = 0
with torch.inference_mode():
    for pixel_values in loader:
        pixel_values = pixel_values.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
            out = encoder(pixel_values=pixel_values)
        tokens = out.last_hidden_state[:, 1:].half().cpu()
        cache[cursor:cursor + tokens.shape[0]] = tokens
        cursor += tokens.shape[0]
print(f'Encoding done in {(time.time()-t0)/60:.1f} min', flush=True)
del encoder, processor, loader
if dev.type == 'cuda':
    torch.cuda.empty_cache()

model = MouseFrameClassifier(
    emb_dim=EMB_DIM, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'],
    use_patch_grid=True, dropout=best_cfg['dropout'],
    cross_attn_dim=args.cross_attn_dim, patch_pool_dim=args.patch_pool_dim,
).to(dev)
model.load_state_dict(torch.load(OUT_DIR / 'best_model.pt', map_location=dev, weights_only=True))
model.eval()

offsets = val_meta.offsets_grid
all_probs, all_labels = [], []
with torch.no_grad():
    for b0 in range(0, len(val_meta), args.batch_size):
        sample_idx = np.arange(b0, min(b0 + args.batch_size, len(val_meta)))
        gi = val_meta.gi[sample_idx]
        abs_idx = gi[:, None] + offsets[None, :]
        mask = val_meta.pad_mask[sample_idx]
        B, T = abs_idx.shape
        valid = ~mask
        positions = np.array([idx_of_global[int(g)] for g in abs_idx[valid]], dtype=np.int64)
        gathered = cache[positions].to(dev, non_blocking=True)
        ctx = torch.zeros((B, T, n_patches_full, EMB_DIM), dtype=torch.float16, device=dev)
        ctx[torch.from_numpy(valid)] = gathered
        offsets_t = torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()).to(dev)
        mask_t = torch.from_numpy(mask).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
            logits = model(ctx.float(), offsets=offsets_t, key_padding_mask=mask_t)
        all_probs.append(torch.sigmoid(logits).float().cpu())
        all_labels.append(torch.from_numpy(val_meta.labels[sample_idx]))

probs = torch.cat(all_probs).numpy()
labels = torch.cat(all_labels).numpy()
per_label = {name: average_precision_score(labels[:, i], probs[:, i]) for i, name in enumerate(['nt', 'nn'])}
macro_ap = float(np.mean(list(per_label.values())))
print(f'\nFULL-VAL macro AP (recovered from timed-out job): {macro_ap:.4f}  {per_label}', flush=True)

cfg_path = OUT_DIR / 'config.json'
existing = json.load(open(cfg_path)) if cfg_path.exists() else {}
existing.update({'best_ap': macro_ap, 'best_per_label': per_label, 'recovered_from_timeout': True,
                 'input_size': args.input_size, 'n_patches': n_patches_full})
with open(cfg_path, 'w') as f:
    json.dump(existing, f, indent=2)
print('Saved config.json', flush=True)
