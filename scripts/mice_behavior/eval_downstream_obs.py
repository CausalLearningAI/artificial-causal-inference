"""
Phase 0.3 -- measure the quantity the project ACTUALLY needs, not frame-level AP.

Frame-level macro AP is a means, not the end. The downstream PPCI estimand is a causal
effect of genotype (wt vs het) on behavior, and per configs/dataset/mice/v1.yaml genotype
is a POOL-LEVEL treatment (verified: constant within all 24 pools; 18 het / 6 wt). So the
model does not need to attribute behavior to individual mice -- it needs the per-observation
(and per-pool) AGGREGATE behavior rate to be accurate. Per-frame errors partially cancel
under aggregation, so a model with modest frame AP can still deliver an accurate rate.

This script answers: given a trained checkpoint, how well does the predicted per-observation
behavior rate track the true one? That is the number that decides whether ~0.35 frame AP is
already good enough downstream, or whether the classifier really is the bottleneck.

Reported per behavior (nt, nn):
  - Pearson / Spearman correlation of predicted vs true per-observation rate
  - calibration slope+intercept (OLS true ~ pred) and R^2
  - MAE / relative bias of the rate
  - the same at pool level (6 observations aggregate into one pool)
Two predictors are compared:
  - soft: mean predicted probability over the observation's frames
  - hard: fraction of frames with p > threshold (threshold swept, best reported)

Usage:
    python scripts/mice_behavior/eval_downstream_obs.py --tag res448 --input-size 448
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import get_fixed_val_pools
from src.mice_behavior.metrics import ap_report, format_ap_report
from src.mice_behavior.head_cfg import get_head_cfg
from src.dataset.get_dataset import load_dataset
from train_patchgrid_online import dummy_loader, _ImageDataset

MODEL_ID = 'facebook/dinov2-base'
EMB_DIM = 768
PATCH_SIZE = 14

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True, help='results/vision/mice/frame/patchgrid256_dinov2_<tag>')
p.add_argument('--input-size', type=int, default=None)
p.add_argument('--cross-attn-dim', type=int, default=None)
p.add_argument('--patch-pool-dim', type=int, default=None)
p.add_argument('--batch-size', type=int, default=256)
args = p.parse_args()
n_patches_full = 256 if args.input_size is None else (args.input_size // PATCH_SIZE) ** 2

OUT_DIR = gsf.FRAME_DIR / (f'patchgrid256_dinov2_{args.tag}' if args.tag != 'promoted' else 'patchgrid256_dinov2')
best_cfg = get_head_cfg()
print(f'Downstream eval of {OUT_DIR}/best_model.pt  (input_size={args.input_size}, n_patches={n_patches_full})', flush=True)

pair_labels_path = gsf.build_pair_labels(gsf.DATA_DIR, gsf.DATASET_DIR, overwrite=False)
annotations_csv = gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
obs_to_pool = gsf.load_obs_to_pool_map(gsf.DATA_DIR)
all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
pools = sorted({obs_to_pool[o] for o in all_obs})
val_pool_set = get_fixed_val_pools(pools)
val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]
print(f'{len(val_obs)} val observations from {len(val_pool_set)} pools: {sorted(val_pool_set)}', flush=True)

val_meta = FrameBatchData(
    str(annotations_csv), str(pair_labels_path), val_obs, best_cfg['context_k'], 1,
    dummy_loader(1, 1), n_patches=1, stride=best_cfg['stride'],
)
del val_meta.flat

# Map each val sample back to its observation via the same global-index boundaries
# FrameBatchData builds internally (annotations.csv row order defines global_idx).
ann = pd.read_csv(annotations_csv, usecols=['observation_id', 'frame_idx'])
ann = ann[ann['observation_id'].isin(set(val_obs))].reset_index()
bounds = {oid: (int(g['index'].values[0]), int(g['index'].values[-1]) + 1)
          for oid, g in ann.groupby('observation_id', sort=False)}
starts = np.array([v[0] for v in bounds.values()])
names = np.array(list(bounds.keys()))
order = np.argsort(starts)
starts, names = starts[order], names[order]
sample_obs = names[np.searchsorted(starts, val_meta.gi, side='right') - 1]
print(f'{len(val_meta):,} val samples mapped to {len(set(sample_obs))} observations', flush=True)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
encoder = AutoModel.from_pretrained(MODEL_ID).to(dev)
encoder.eval(); encoder.requires_grad_(False)
hf_dataset = load_dataset(subject='mice', version='v1', dataset_root=str(gsf.DATASET_DIR), frame_type='full')

need = np.unique((val_meta.gi[:, None] + val_meta.offsets_grid[None, :])[~val_meta.pad_mask])
print(f'Encoding {len(need):,} unique val frames '
      f'({len(need)*n_patches_full*EMB_DIM*2/1024**3:.0f} GiB cache)...', flush=True)
t0 = time.time()
loader = DataLoader(_ImageDataset(hf_dataset, need, processor, input_size=args.input_size), batch_size=128,
                    num_workers=16, pin_memory=(dev.type == 'cuda'), shuffle=False,
                    prefetch_factor=4, persistent_workers=True)
cache = torch.empty((len(need), n_patches_full, EMB_DIM), dtype=torch.float16)
cursor = 0
with torch.inference_mode():
    for pixel_values in loader:
        pixel_values = pixel_values.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
            out = encoder(pixel_values=pixel_values)
        tok = out.last_hidden_state[:, 1:].half().cpu()
        cache[cursor:cursor + tok.shape[0]] = tok
        cursor += tok.shape[0]
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
all_probs = []
with torch.no_grad():
    for b0 in range(0, len(val_meta), args.batch_size):
        si = np.arange(b0, min(b0 + args.batch_size, len(val_meta)))
        abs_idx = val_meta.gi[si][:, None] + offsets[None, :]
        mask = val_meta.pad_mask[si]
        B, T = abs_idx.shape
        valid = ~mask
        pos = np.searchsorted(need, abs_idx[valid])
        ctx = torch.zeros((B, T, n_patches_full, EMB_DIM), dtype=torch.float16, device=dev)
        ctx[torch.from_numpy(valid)] = cache[pos].to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
            logits = model(ctx.float(),
                           offsets=torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()).to(dev),
                           key_padding_mask=torch.from_numpy(mask).to(dev))
        all_probs.append(torch.sigmoid(logits).float().cpu())
probs = torch.cat(all_probs).numpy()
labels = val_meta.labels

frame_ap = {n: average_precision_score(labels[:, i], probs[:, i]) for i, n in enumerate(['nt', 'nn'])}
ap_rep = ap_report(probs, labels, sample_obs, tolerances=(0, 1, 2))
print(f'\n{"="*70}\nFRAME-LEVEL AP -- plain (tol 0) and tolerant (tol 1,2 frames)\n{"="*70}')
print(format_ap_report(ap_rep, tolerances=(0, 1, 2)))
print('\n  tol>0 dilates the LABEL by +-tol frames (never the prediction), so a detection within\n'
      '  tol frames of an annotated bout counts as correct -- justified because 22%/38% of nt/nn\n'
      '  bouts are a single 0.2s frame and boundary annotation is not reliable to that precision.\n'
      '  Dilation raises prevalence, so compare ENRICHMENT (AP/prevalence) across tolerances, not AP.', flush=True)

# ---- per-observation aggregation ----
df = pd.DataFrame({'obs': sample_obs,
                   'nt_true': labels[:, 0], 'nn_true': labels[:, 1],
                   'nt_p': probs[:, 0], 'nn_p': probs[:, 1]})
df['pool'] = [obs_to_pool[o] for o in df.obs]
df['genotype'] = [o.split('_')[0] for o in df.obs]


def report(level, key):
    print(f'\n{"="*70}\n{level.upper()}-LEVEL aggregate behavior rate  (n={df[key].nunique()})\n{"="*70}')
    out = {}
    for beh in ['nt', 'nn']:
        g = df.groupby(key).agg(true=(f'{beh}_true', 'mean'), soft=(f'{beh}_p', 'mean'))
        # hard predictor: sweep threshold, pick the one minimising MAE of the rate
        best = None
        for th in np.quantile(df[f'{beh}_p'], np.linspace(0.90, 0.999, 40)):
            hard = df.assign(h=(df[f'{beh}_p'] > th).astype(float)).groupby(key).h.mean()
            mae = np.abs(hard - g.true).mean()
            if best is None or mae < best[0]:
                best = (mae, th, hard)
        mae_h, th, hard = best
        g['hard'] = hard
        r_s = stats.pearsonr(g.true, g.soft)
        rho_s = stats.spearmanr(g.true, g.soft)
        r_h = stats.pearsonr(g.true, g.hard)
        sl, ic, rv, _, _ = stats.linregress(g.soft, g.true)
        print(f'\n  [{beh}]  true rate: mean {g.true.mean()*100:.2f}%  range {g.true.min()*100:.2f}-{g.true.max()*100:.2f}%')
        print(f'    soft (mean prob):  Pearson r={r_s[0]:+.3f} (p={r_s[1]:.3g})  Spearman rho={rho_s[0]:+.3f}  R2={rv**2:.3f}')
        print(f'                       calibration: true = {sl:.3f} x pred + {ic:+.4f}   MAE={np.abs(g.soft-g.true).mean()*100:.2f}pp')
        print(f'    hard (p>{th:.3f}):     Pearson r={r_h[0]:+.3f} (p={r_h[1]:.3g})  MAE={mae_h*100:.2f}pp')
        out[beh] = dict(pearson_soft=float(r_s[0]), p_soft=float(r_s[1]), spearman_soft=float(rho_s[0]),
                        r2_soft=float(rv**2), calib_slope=float(sl), calib_intercept=float(ic),
                        mae_soft_pp=float(np.abs(g.soft-g.true).mean()*100),
                        pearson_hard=float(r_h[0]), best_threshold=float(th), mae_hard_pp=float(mae_h*100),
                        true_rate_mean=float(g.true.mean()))
    return out


res = {'frame_ap': frame_ap, 'frame_macro_ap': float(np.mean(list(frame_ap.values()))), 'ap_report': ap_rep}
res['observation'] = report('observation', 'obs')
res['pool'] = report('pool', 'pool')

# genotype contrast -- the actual estimand (caveat: val split is 4 het / 1 wt, so this is
# reported for completeness only and is NOT a usable effect estimate)
print(f'\n{"="*70}\nGENOTYPE CONTRAST (val split composition: '
      f'{df.groupby("genotype").pool.nunique().to_dict()} pools)\n{"="*70}')
for beh in ['nt', 'nn']:
    pg = df.groupby(['pool', 'genotype']).agg(true=(f'{beh}_true', 'mean'), pred=(f'{beh}_p', 'mean')).reset_index()
    t = pg.groupby('genotype').true.mean()
    q = pg.groupby('genotype').pred.mean()
    if len(t) == 2:
        print(f'  [{beh}] true het-wt: {(t.get("het",np.nan)-t.get("wt",np.nan))*100:+.3f}pp | '
              f'predicted het-wt: {(q.get("het",np.nan)-q.get("wt",np.nan))*100:+.3f}pp')
    print(f'        (per-pool true rates by genotype: {t.round(4).to_dict()})')

with open(OUT_DIR / 'downstream_obs_eval.json', 'w') as f:
    json.dump(res, f, indent=2)
df.groupby(['obs', 'pool', 'genotype']).agg(
    nt_true=('nt_true', 'mean'), nt_pred=('nt_p', 'mean'),
    nn_true=('nn_true', 'mean'), nn_pred=('nn_p', 'mean')).to_csv(OUT_DIR / 'downstream_obs_rates.csv')
print(f'\nSaved {OUT_DIR}/downstream_obs_eval.json and downstream_obs_rates.csv', flush=True)
