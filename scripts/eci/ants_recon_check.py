"""
Diagnostic: how well the published ants atlas SAEs (antsfg, antsfull; trained on v2 + v3 10-min tokens) reconstruct
the tokens of the frames they are now applied to (v3 minutes 10-30, v4, v4_infested), against their held-out
videos at 10 min. A clearly worse reconstruction on new frames means the SAE dictionary does not cover them well and
neurons there should be read with care. No retraining: neuron ids and interpretations stay valid either way.

Groups (frames of the domain layout, ANTS_ECI): 'reference' = the SAE's held-out videos (metrics.json val_pools)
present in the layout, minutes 0-10 (it should land near metrics.json val_threshold, which was measured on the 1 fps
token store of the same videos); then every experiment (v4_infested split from v4 by design) x 10-min window of the
video (0-10, 10-20, 20-30) that has frames. Per group at most --max-videos videos (seeded), --frames frames per video
evenly spread over the window, at most --patches tokens per frame (random, seeded; every foreground patch counts
before the cap). Tokens: DINOv2-base at 448 (fp32, rounded to fp16), the SAE's own mask rule and backgrounds;
metrics: src/eci/sae.py evaluate_sae (threshold inference): FVE (about the group's own mean), MSE per token (in the
SAE's normalised units: comparable across groups), L0 per token, dead fraction; and mse_ratio = MSE / the reference
group's MSE. Output: results/vision/eci_ants30/recon_check_<ANTS_ECI>.json (+ printed table).

Usage (GPU): ANTS_ECI=eci30 python scripts/eci/ants_recon_check.py --sae <antsfg> --sae <antsfull>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from src.eci.domain import get_domain  # noqa: E402
from src.eci.foreground import FrameDatasetFG, encode_batch, load_encoder_fg, obs_rows  # noqa: E402
from src.eci.sae import evaluate_sae  # noqa: E402
from fg_encode_extend import load_saes  # noqa: E402

WIN = 3000  # 10 min at 5 fps


def groups(dom, val_ids, max_videos, frames, seed):
    """{group name: rows of annotations.csv}."""
    e = pd.read_csv(dom.experiment_csv, dtype={'observation_id': str})
    ranges = obs_rows(dom.ann_path)
    rng = np.random.default_rng(seed)

    def rows_of(ids, k):
        out = []
        for o in ids:
            lo, hi = ranges[o]
            if lo + (k + 1) * WIN <= hi:
                out.append(np.round(np.linspace(lo + k * WIN, lo + (k + 1) * WIN - 1, frames)).astype(np.int64))
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    out = {'reference (held-out videos, min 0-10)': rows_of([o for o in val_ids if o in ranges], 0)}
    for x in sorted(e['experiment'].unique()):
        ids = e.loc[e['experiment'] == x, 'observation_id'].values
        ids = np.sort(rng.choice(ids, min(max_videos, len(ids)), replace=False))
        for k in range(3):
            r = rows_of(ids, k)
            if len(r):
                out[f'{x} min {10 * k}-{10 * (k + 1)}'] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sae', action='append', required=True)
    ap.add_argument('--max-videos', type=int, default=40)
    ap.add_argument('--frames', type=int, default=12, help='frames per video and window')
    ap.add_argument('--patches', type=int, default=256, help='tokens per frame at most')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num-workers', type=int, default=12)
    args = ap.parse_args()
    dom = get_domain('ants')
    dev = torch.device('cuda')
    saes = load_saes(dom, args.sae, dev)
    metr = {s: json.loads((dom.eci_dir / 'sae' / s / 'metrics.json').read_text()) for s in args.sae}
    val_ids = sorted(set().union(*(m['val_pools'] for m in metr.values())))
    G = groups(dom, val_ids, args.max_videos, args.frames, args.seed)
    fp = pd.read_csv(dom.ann_path, usecols=['frame_path'])['frame_path'].values
    _, proc, model = load_encoder_fg(device=dev)
    rng = np.random.default_rng(args.seed)
    res = {'layout': str(dom.eci_dir), 'settings': vars(args), 'saes': {}}
    for s in args.sae:
        p = metr[s]['val_threshold']['prefixes']
        res['saes'][s] = {'metrics_json_val_fve': {m: p[m]['fve'] for m in p},
                          'metrics_json_val_mse': {m: p[m]['mse_per_token'] for m in p}, 'groups': {}}
    t0 = time.time()
    for g, rows in G.items():
        toks = {s: [] for s in args.sae}
        dl = torch.utils.data.DataLoader(FrameDatasetFG([str(REPO / 'dataset' / f) for f in fp[rows]], proc, rows),
                                         batch_size=64, num_workers=args.num_workers)
        for pix, grey, r in dl:
            tok = encode_batch(model, pix, dev)
            grey, r = grey.to(dev), r.numpy()
            for (s, _, _, bgs) in saes:
                mask = torch.ones(tok.shape[:2], dtype=torch.bool, device=dev) if bgs is None else bgs.mask(tok, grey, r)[0]
                for i in range(len(r)):
                    idx = torch.nonzero(mask[i]).flatten().cpu().numpy()
                    if len(idx) > args.patches:
                        idx = np.sort(rng.choice(idx, args.patches, replace=False))
                    toks[s].append(tok[i, torch.from_numpy(idx).to(dev)].cpu())
        for (s, sae, norm, _) in saes:
            x = torch.cat(toks[s])
            ev = evaluate_sae(sae, norm, x, n_patches=1, mode='threshold')
            res['saes'][s]['groups'][g] = {'n_frames': int(len(rows)), 'n_tokens': int(ev['n_tokens']),
                                           'total_var_per_token': ev['total_var_per_token'],
                                           'prefixes': {m: {k: v[k] for k in ('fve', 'mse_per_token', 'l0_per_token',
                                                                              'dead_frac')}
                                                        for m, v in ev['prefixes'].items()}}
        print(f'{g}: {len(rows)} frames ({time.time() - t0:.0f}s)', flush=True)
    for s in args.sae:
        gr = res['saes'][s]['groups']
        ref = next(iter(gr.values()))['prefixes']
        print(f'\n{s}  (metrics.json held-out FVE @1024 {res["saes"][s]["metrics_json_val_fve"]["1024"]:.4f})')
        print(f'{"group":42s} {"frames":>6s} {"tokens":>8s}  FVE@128  FVE@1024  MSE@1024  ratio  L0@1024')
        for g, v in gr.items():
            for m in v['prefixes']:
                v['prefixes'][m]['mse_ratio'] = v['prefixes'][m]['mse_per_token'] / ref[m]['mse_per_token']
            q, Q = v['prefixes']['128'], v['prefixes']['1024']
            print(f'{g:42s} {v["n_frames"]:6d} {v["n_tokens"]:8d}  {q["fve"]:.4f}   {Q["fve"]:.4f}   '
                  f'{Q["mse_per_token"]:8.2f}  {Q["mse_ratio"]:.3f}  {Q["l0_per_token"]:.2f}')
    out = REPO / f'results/vision/eci_ants30/recon_check_{dom.ECI}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
