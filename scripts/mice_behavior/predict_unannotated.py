#!/usr/bin/env python3
"""Run a trained frame classifier on observations that have NO labels: the 288 unannotated
v1 observations and all 216 of v2.

Why this exists
---------------
Every evaluation path in the repo enumerates observations from `pair_labels.parquet`, which
covers only the 144 annotated v1 observations. That means the model has never once been looked
at on the data the causal design actually needs it for:

  * v1 has 48 unannotated pools. PPI's whole value comes from rectifying predictions made
    THERE, so "does f behave sanely off the labelled set" is a prerequisite, not a nicety.
  * v2 has 36 pools and zero labels of any kind. It is the target domain, it is a different
    cohort recorded months later, and no number computed on v1 says whether the model
    transfers to it. The only evidence available without annotation is (a) the predicted-rate
    distribution being physically plausible rather than collapsed or saturated, and (b) the
    confident detections actually depicting the behaviour when a human looks at them.

So this script emits exactly those two things: per-observation predicted rates, and a grid of
the most confident detections with their frames, for a human to check.

What it CANNOT do: validate accuracy. With no labels there is no error to measure. Treat the
output as a sanity check and a qualitative exhibit, never as a performance claim.

Usage:
    python scripts/mice_behavior/predict_unannotated.py --tag res448_k2_frozen_d4photo_sslinit \
        --version v2 --n-obs 24 --frame-stride 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from regen_confusion_figs import load_jpeg_cache, _BytesReader          # noqa: E402
from src.mice_behavior.model import MouseFrameClassifier                # noqa: E402
from src.mice_behavior.head_cfg import get_head_cfg                     # noqa: E402
from transformers import AutoModel                                      # noqa: E402
from PIL import Image                                                   # noqa: E402
import io                                                               # noqa: E402

MODEL_ID = 'facebook/dinov2-base'
EMB_DIM, PATCH_SIZE = 768, 14
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
FRAME_DIR = ROOT / 'results' / 'vision' / 'mice' / 'frame'


def build_model(cfg, run_dir, dev, init_encoder=None):
    """Rebuild the EXACT encoder the head was trained against.

    Two independent ways the encoder can differ from stock DINOv2, and both must be honoured
    or the head is fed features it has never seen:

      unfreeze_blocks > 0  -> the run trained encoder weights itself; they are in its own
                              best_encoder.pt.
      init_encoder set     -> the run FROZE the encoder at someone else's weights (e.g. an
                              SSL-adapted checkpoint). Nothing trained, so the run dir has no
                              best_encoder.pt at all, and the weights live wherever
                              --init-encoder pointed.

    The second case is the dangerous one: it leaves no artefact in the run directory, so a
    naive reload silently substitutes stock DINOv2 and produces garbage that still looks like
    probabilities. Measured when this was wrong: predicted nt rate ~80% against a true ~1%.
    """
    head = get_head_cfg()
    encoder = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    encoder.requires_grad_(False)
    src = None
    if cfg.get('unfreeze_blocks', 0) > 0:
        src = run_dir / 'best_encoder.pt'
        if not src.exists():
            raise SystemExit(f'{run_dir.name} fine-tuned the encoder but has no best_encoder.pt')
    elif init_encoder or cfg.get('init_encoder'):
        src = Path(init_encoder or cfg['init_encoder'])
        if not src.is_absolute():
            src = ROOT / src
        if not src.exists():
            raise SystemExit(f'init encoder {src} not found -- refusing to score this head '
                             'against stock DINOv2, which would silently be a different model.')
    if src is not None:
        sd = torch.load(src, map_location=dev, weights_only=True)
        _, unexpected = encoder.load_state_dict(sd, strict=False)
        if len(sd) - len(unexpected) == 0:
            raise SystemExit(f'{src} shares no keys with the encoder')
        print(f'  encoder <- {src} ({len(sd) - len(unexpected)}/{len(sd)} tensors)', flush=True)
    else:
        print('  encoder <- stock DINOv2 (run recorded no init_encoder and trained none)',
              flush=True)
    model = MouseFrameClassifier(
        emb_dim=EMB_DIM, n_heads=head['n_heads'], hidden_dim=head['hidden_dim'],
        use_patch_grid=True, dropout=cfg.get('dropout', head['dropout']),
        use_motion=cfg.get('use_motion', False),
        cross_attn_dim=cfg.get('cross_attn_dim') or None,
        patch_pool_dim=cfg.get('patch_pool_dim') or None,
        patch_selfattn_dim=cfg.get('patch_selfattn_dim') or None,
        n_pool_queries=cfg.get('pool_queries', 1),
        pool_grid=cfg.get('pool_grid', 0) or 0).to(dev)
    model.load_state_dict(torch.load(run_dir / 'best_model.pt', map_location=dev,
                                     weights_only=True))
    model.eval()
    return encoder, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--version', default='v2', choices=['v1', 'v2'])
    p.add_argument('--n-obs', type=int, default=24)
    p.add_argument('--frame-stride', type=int, default=5,
                   help='anchor every Nth frame. Rates are means over hundreds of anchors, so '
                        'striding costs little precision and divides the cost outright.')
    p.add_argument('--n-examples', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--read-workers', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--init-encoder', default=None,
                   help='encoder checkpoint the head was frozen at. Needed for runs written '
                        'before config.json recorded init_encoder; without it a frozen-at-SSL '
                        'head is silently scored against stock DINOv2.')
    args = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = FRAME_DIR / args.tag
    cfg = json.load(open(run_dir / 'config.json'))
    S, K = cfg['input_size'], cfg['context_k']
    n_patches = (S // PATCH_SIZE) ** 2
    print(f'{args.tag}: input={S} k={K} on {args.version}', flush=True)

    ann = pd.read_csv(ROOT / 'dataset' / 'mice' / args.version / 'annotations.csv',
                      usecols=['observation_id', 'frame_idx', 'frame_path', 'Y_nt', 'Y_nn'],
                      low_memory=False).reset_index().rename(columns={'index': 'gi'})
    exp = pd.read_csv(ROOT / 'data' / 'mice' / args.version / 'experiment.csv')
    # UNANNOTATED only: for v1 that is the 288 observations with no annotator; v2 has none at all.
    unann = set(exp.loc[exp.annotator.isna(), 'observation_id'])
    if not unann:
        raise SystemExit(f'{args.version}: every observation has an annotator?')
    rng = np.random.default_rng(args.seed)
    obs = sorted(unann)
    obs = list(rng.permutation(obs)[:args.n_obs])
    sub = ann[ann.observation_id.isin(obs)].copy()
    assert sub[['Y_nt', 'Y_nn']].isna().all().all(), 'expected NO labels on these observations'
    print(f'  {len(obs)} observations, {len(sub):,} frames on disk', flush=True)

    # anchors + context windows, clamped inside each observation (no cross-video context)
    rows = []
    for oid, g in sub.groupby('observation_id', sort=False):
        g = g.sort_values('frame_idx')
        gi = g.gi.to_numpy()
        anchors = np.arange(0, len(gi), args.frame_stride)
        for a in anchors:
            idx = np.clip(np.arange(a - K, a + K + 1), 0, len(gi) - 1)
            rows.append((oid, gi[a], gi[idx]))
    obs_of = np.array([r[0] for r in rows])
    center = np.array([r[1] for r in rows])
    windows = np.stack([r[2] for r in rows])
    print(f'  {len(rows):,} anchors (stride {args.frame_stride}) -> '
          f'{len(np.unique(windows)):,} distinct frames to encode', flush=True)

    frame_paths = ann.frame_path.values
    need = np.unique(windows)
    cache = load_jpeg_cache(None, need, frame_paths, args.read_workers)

    @torch.no_grad()
    def run():
        out = []
        for i in range(0, len(rows), args.batch_size):
            w = windows[i:i + args.batch_size]
            B, T = w.shape
            x = torch.zeros((B, T, 3, S, S))
            for b in range(B):
                for t in range(T):
                    with Image.open(io.BytesIO(cache[int(w[b, t])].tobytes())) as im:
                        arr = np.asarray(im.convert('RGB').resize((S, S), Image.BILINEAR),
                                         dtype=np.uint8).copy()
                    v = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                    x[b, t] = (v - IMAGENET_MEAN) / IMAGENET_STD
            x = x.to(dev)
            offs = torch.arange(-K, K + 1, device=dev).expand(B, T).contiguous()
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                tok = encoder(pixel_values=x.view(B * T, 3, S, S)).last_hidden_state[:, 1:]
                lg = model(tok.view(B, T, n_patches, EMB_DIM), offsets=offs,
                           key_padding_mask=torch.zeros(B, T, dtype=torch.bool, device=dev))
            out.append(torch.sigmoid(lg).float().cpu().numpy())
            if i % (args.batch_size * 50) == 0:
                print(f'    {i:,}/{len(rows):,}', flush=True)
        return np.concatenate(out)

    encoder, model = build_model(cfg, run_dir, dev, args.init_encoder)
    t0 = time.time()
    probs = run()
    print(f'  inference {(time.time()-t0)/60:.1f} min', flush=True)

    # ---- per-observation predicted rates -------------------------------------------------
    df = pd.DataFrame({'observation_id': obs_of, 'p_nt': probs[:, 0], 'p_nn': probs[:, 1]})
    rate = df.groupby('observation_id').mean().reset_index().merge(
        exp[['observation_id', 'pool', 'phase', 'odor', 'line', 'sex', 'genotype']],
        on='observation_id')
    for lab in ('p_nt', 'p_nn'):
        m = rate[lab].mean() * 100
        if m > 25:
            print(f'\n!! {lab} mean predicted rate {m:.1f}% -- the labelled v1 prior is ~1-2%. '
                  'That is a load/preprocessing mismatch, not a finding. Refusing to write.',
                  flush=True)
            raise SystemExit(2)
    out_csv = run_dir / f'pred_unannotated_{args.version}.csv'
    rate.to_csv(out_csv, index=False)
    print(f'\npredicted rate (%) by phase on {args.version} [UNLABELLED -- sanity check only]:')
    print((rate.groupby(['odor', 'phase'])[['p_nt', 'p_nn']].mean() * 100).round(3).to_string())
    print(f'\nwrote {out_csv}')

    # ---- confident detections ------------------------------------------------------------
    for j, name in enumerate(('nt', 'nn')):
        order = np.argsort(-probs[:, j])
        picked, seen = [], set()
        for i in order:                      # one frame per observation -> independent cases
            if obs_of[i] in seen:
                continue
            seen.add(obs_of[i]); picked.append(i)
            if len(picked) >= args.n_examples:
                break
        fig, axes = plt.subplots(1, len(picked), figsize=(2.1 * len(picked), 2.7))
        for ax, i in zip(np.atleast_1d(axes), picked):
            with Image.open(io.BytesIO(cache[int(center[i])].tobytes())) as im:
                ax.imshow(im.convert('L'), cmap='gray')
            ax.set_title(f'p={probs[i, j]:.3f}', fontsize=9, color='#1a7f37')
            ax.set_xlabel(obs_of[i], fontsize=5)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'{args.tag} — most confident {name} detections on {args.version} '
                     f'(UNLABELLED: no ground truth exists for these frames)', fontsize=11)
        fig.tight_layout()
        f = run_dir / f'confident_{name}_{args.version}.png'
        fig.savefig(f, dpi=110, bbox_inches='tight'); plt.close(fig)
        print(f'  [viz] {f}')


if __name__ == '__main__':
    main()
