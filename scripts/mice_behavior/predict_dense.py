#!/usr/bin/env python3
"""DENSE per-frame inference on the unlabelled pools, so the causal outcome can be BOUT COUNTS.

Why this replaces `predict_unannotated.py` for anything causal
=============================================================
`predict_unannotated.py` anchors every `--frame-stride` frames and stores ONE number per
observation: the mean predicted probability. Two consequences, and the second one is a
correctness problem rather than a precision one.

1. It only yields OCCUPANCY. Section 02 of the status report spends six arguments establishing
   that the causal outcome should be BOUTS PER MINUTE, not time-in-behaviour -- and then every
   PPI number in the report is computed on occupancy, because a mean probability is all that
   was ever saved. A bout is a CONTIGUOUS run of frames, so counting bouts needs neighbouring
   frames, and the existing dumps were produced at stride 16 (one anchor every 3.2 s at 5 fps)
   against bouts whose median length is 2-3 frames and of which 22-49% are a SINGLE frame.
   Bout counts are not recoverable from that at any precision; they are not defined.

2. Striding is a false economy in the first place. The old loop re-encodes a fresh 2k+1 = 5
   frame window per anchor, so it spends 5 encoder passes per predicted frame and shares
   nothing between neighbouring anchors. Encoding each frame ONCE and sliding the temporal
   window over the cached tokens costs 1 pass per predicted frame. At the same encoder budget
   that is 5x more predicted frames, which is how full stride-1 coverage becomes affordable:
   the old v1 pass spent 540,384 encodes to predict 108,192 frames, and this predicts all
   1,728,000 for 1,728,000.

So: encode once, slide the head, keep every frame. Occupancy still falls out (it is the mean),
and bouts/min becomes available for the first time on pools nobody annotated.

What it emits, per (tag, version)
---------------------------------
`pred_dense_<version>.csv`  one row per observation: predicted occupancy (`po_*`, pp) AND
                            predicted bouts/min (`p_*`) at each requested threshold, plus the
                            design columns.
`pred_dense_<version>.npz`  the per-frame probabilities themselves (float16), so a threshold or
                            a post-processing rule can be changed later without another GPU pass.
                            v1 unlabelled is 1.73 M frames x 2 labels = 6.9 MB. Cheap insurance.

The threshold question, stated honestly
--------------------------------------
Bout counts need an operating point; occupancy does not. There are no labels here, so the
threshold CANNOT be chosen on this data -- and `event_eval.py` picks its threshold by maximising
F1 on the very split it then scores, which is an oracle and not available off the labelled set.
This script therefore sweeps a fixed grid and writes every threshold, and the analysis step is
required to state which one it used and where it came from. The defensible choice is a threshold
fitted on the LABELLED pools out-of-fold and then applied unchanged here; nothing about the
unlabelled predictions may inform it.

None of this makes the predictions valid on their own. PPI's rectifier is what buys validity on
v1; on v2 there is no rectifier and the numbers stay an extrapolation.

Usage
-----
    python scripts/mice_behavior/predict_dense.py --tag xfit_f1 --version v1
    python scripts/mice_behavior/predict_dense.py --tag xfit_f1 --version v1 --labelled-too
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from regen_confusion_figs import load_jpeg_cache                        # noqa: E402
from predict_unannotated import build_model                            # noqa: E402
from event_eval import runs, postprocess                               # noqa: E402
from PIL import Image                                                  # noqa: E402
import io                                                              # noqa: E402
from concurrent.futures import ThreadPoolExecutor                      # noqa: E402

EMB_DIM, PATCH_SIZE = 768, 14
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
FRAME_DIR = ROOT / 'results' / 'vision' / 'mice' / 'frame'
FPS = 5.0
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def decode_batch(cache, gis, S, pool):
    """JPEG bytes -> normalised float tensor (n, 3, S, S). Decoded ONCE per frame, in threads.

    The old loop decoded inside a python double loop over (batch, context position), so every
    frame was decoded 2k+1 = 5 times and the GPU waited on a single CPU core. Pillow releases
    the GIL in decode/resize, so a thread pool is enough -- no worker processes, no pickling of
    the memory-mapped cache.
    """
    def one(g):
        with Image.open(io.BytesIO(cache[int(g)].tobytes())) as im:
            return np.asarray(im.convert('RGB').resize((S, S), Image.BILINEAR), dtype=np.uint8)
    arrs = list(pool.map(one, gis))
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float().div_(255.0)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


@torch.no_grad()
def dense_probs(encoder, model, cache, gis, S, K, n_patches, dev, chunk, head_bs, pool):
    """Per-frame probabilities for ONE observation, encoding each frame exactly once.

    Frames are processed in chunks with a K-frame halo on each side, so a window centred on the
    first frame of a chunk can still see its left context without re-encoding it for the whole
    chunk. Window indices are CLAMPED inside the observation, matching train_online_aug's
    padding convention exactly -- context never crosses a video boundary.
    """
    n = len(gis)
    out = np.zeros((n, 2), dtype=np.float32)
    offs_row = torch.arange(-K, K + 1, device=dev)
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        lo, hi = max(0, c0 - K), min(n, c1 + K)
        toks = []
        for b0 in range(lo, hi, head_bs):
            b1 = min(hi, b0 + head_bs)
            x = decode_batch(cache, gis[b0:b1], S, pool).to(dev, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                toks.append(encoder(pixel_values=x).last_hidden_state[:, 1:].half())
        tok = torch.cat(toks)                                     # (hi-lo, n_patches, D)
        del toks
        centres = np.arange(c0, c1)
        widx = np.clip(centres[:, None] + np.arange(-K, K + 1)[None, :], 0, n - 1) - lo
        for s0 in range(0, len(centres), head_bs):
            s1 = min(len(centres), s0 + head_bs)
            w = torch.from_numpy(widx[s0:s1]).to(dev)
            B, T = w.shape
            win = tok[w.reshape(-1)].view(B, T, n_patches, EMB_DIM)
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                lg = model(win.float() if dev.type != 'cuda' else win,
                           offsets=offs_row.expand(B, T).contiguous(),
                           key_padding_mask=torch.zeros(B, T, dtype=torch.bool, device=dev))
            out[c0 + s0:c0 + s1] = torch.sigmoid(lg).float().cpu().numpy()
        del tok
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--version', default='v1', choices=['v1', 'v2'])
    p.add_argument('--labelled-too', action='store_true',
                   help='also score the ANNOTATED v1 observations. Only valid for a fold model '
                        'on the pools it held out; the script checks val_pools and refuses '
                        'otherwise, because an in-sample prediction breaks PPI\'s rectifier.')
    p.add_argument('--n-obs', type=int, default=0,
                   help='score only the first N target observations (0 = all). For smoke tests '
                        'and for the validation-against-val_probs check.')
    p.add_argument('--only-labelled', action='store_true',
                   help='score ONLY the held-out labelled observations, not the unlabelled ones. '
                        'Used to validate this path against the val_probs.npz the training code '
                        'wrote for the same model and the same frames.')
    p.add_argument('--out-suffix', default='',
                   help='appended to the output filenames, so a validation pass cannot overwrite '
                        'a real deployment dump.')
    p.add_argument('--chunk', type=int, default=512, help='frames of encoder cache held at once')
    p.add_argument('--head-batch', type=int, default=64)
    p.add_argument('--read-workers', type=int, default=32)
    p.add_argument('--decode-threads', type=int, default=16)
    p.add_argument('--jpeg-cache-file', default=None)
    p.add_argument('--init-encoder', default=None)
    p.add_argument('--merge-gap', type=int, default=1)
    p.add_argument('--min-len', type=int, default=1)
    args = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = FRAME_DIR / args.tag
    cfg = json.load(open(run_dir / 'config.json'))
    S, K = cfg['input_size'], cfg['context_k']
    n_patches = (S // PATCH_SIZE) ** 2
    print(f'{args.tag}: input={S} k={K} on {args.version}  (dense, stride 1)', flush=True)

    ann = pd.read_csv(ROOT / 'dataset' / 'mice' / args.version / 'annotations.csv',
                      usecols=['observation_id', 'frame_idx', 'frame_path', 'Y_nt', 'Y_nn'],
                      low_memory=False).reset_index().rename(columns={'index': 'gi'})
    exp = pd.read_csv(ROOT / 'data' / 'mice' / args.version / 'experiment.csv')
    unann = set(exp.loc[exp.annotator.isna(), 'observation_id'])
    targets = [] if args.only_labelled else sorted(unann)
    if args.labelled_too or args.only_labelled:
        val_pools = set(cfg.get('val_pools') or [])
        if not val_pools:
            raise SystemExit(f'{args.tag} records no val_pools; refusing to score labelled data '
                             'with a model that may have trained on it.')
        held = exp[exp.pool.isin(val_pools) & exp.annotator.notna()]
        print(f'  + {len(held)} labelled observations from held-out pools '
              f'{sorted(val_pools)}', flush=True)
        targets += sorted(held.observation_id)
    if not targets:
        raise SystemExit(f'{args.version}: nothing to score')
    if args.n_obs:
        targets = targets[:args.n_obs]
        print(f'  --n-obs {args.n_obs}: scoring {len(targets)} observations only', flush=True)

    sub = ann[ann.observation_id.isin(set(targets))]
    print(f'  {len(targets)} observations, {len(sub):,} frames -> {len(sub):,} encoder passes '
          f'(the strided path would have needed {5*len(sub)//16:,} to cover 1/16 of them)',
          flush=True)

    frame_paths = ann.frame_path.values
    encoder, model = build_model(cfg, run_dir, dev, args.init_encoder)
    pool = ThreadPoolExecutor(args.decode_threads)

    per_frame, rows = {}, []
    t0 = time.time()
    for i, (oid, g) in enumerate(sub.groupby('observation_id', sort=False)):
        gis = g.sort_values('frame_idx').gi.to_numpy()
        cache = load_jpeg_cache(args.jpeg_cache_file, gis, frame_paths, args.read_workers)
        pr = dense_probs(encoder, model, cache, gis, S, K, n_patches, dev,
                         args.chunk, args.head_batch, pool)
        del cache
        per_frame[oid] = pr.astype(np.float16)
        mins = len(pr) / FPS / 60
        rec = {'observation_id': oid, 'n_frames': len(pr)}
        for j, lab in enumerate(('nt', 'nn')):
            rec['po_' + lab] = float(pr[:, j].mean() * 100)          # occupancy, pp
            for th in THRESHOLDS:
                m = postprocess(pr[:, j] >= th, args.merge_gap, args.min_len)
                rec[f'p_{lab}_t{th:.2f}'] = len(runs(m)) / mins      # bouts/min
        rows.append(rec)
        if i % 10 == 0 or i == len(targets) - 1:
            el = time.time() - t0
            print(f'    {i+1}/{len(targets)} obs  {el/60:.1f} min  '
                  f'({(i+1)/max(el,1e-9)*60:.1f} obs/min)', flush=True)

    out = pd.DataFrame(rows).merge(
        exp[['observation_id', 'pool', 'phase', 'odor', 'line', 'sex', 'genotype', 'annotator']],
        on='observation_id')
    # Same guard as predict_unannotated: a load/preprocessing mismatch shows up as a predicted
    # rate an order of magnitude off the ~1-3% labelled prior, and that is a bug, not a finding.
    for lab in ('nt', 'nn'):
        m = out['po_' + lab].mean()
        if m > 25:
            raise SystemExit(f'!! po_{lab} mean {m:.1f}% -- labelled v1 prior is ~1-3%. '
                             'Load/preprocessing mismatch; refusing to write.')
    csv = run_dir / f'pred_dense_{args.version}{args.out_suffix}.csv'
    out.to_csv(csv, index=False)
    np.savez_compressed(run_dir / f'pred_dense_{args.version}{args.out_suffix}.npz',
                        **{k: v for k, v in per_frame.items()})
    print(f'\nwrote {csv}\nwrote {run_dir / f"pred_dense_{args.version}{args.out_suffix}.npz"}')
    print(f'total {(time.time()-t0)/60:.1f} min for {len(sub):,} frames')
    print('\npredicted occupancy (pp) by phase [UNLABELLED -- sanity check only]:')
    print(out.groupby(['odor', 'phase'])[['po_nt', 'po_nn']].mean().round(3).to_string())
    print('\npredicted bouts/min at threshold 0.50:')
    print(out.groupby(['odor', 'phase'])[['p_nt_t0.50', 'p_nn_t0.50']].mean().round(3).to_string())


if __name__ == '__main__':
    main()
