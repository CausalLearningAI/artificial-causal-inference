"""
Tokens per frame of every frozen-v2 neuron, for the atlas page (scripts/eci/build_atlas_v37.py reads the output). CPU.

Per series (configs/eci_v2/<domain>.json) and neuron j: the mean number of image tokens per analysed frame on which the
neuron is active (its per-token SAE code > 0, i.e. above the SAE's global threshold theta*), counted exactly from the
stored per-token codes of the full pass (<V2 BEE>/<domain>/videos/<series>/<obs>.npz: act_idx / act_count / n_kept,
the store scripts/eci/v2/tools.py TokenCodes reads). Raw frame tokens (before the 0.6 s readout smoothing). Analysed
frames = the NES primary window: mice habituation videos (obs ending '_H') their last 15 min (n_match 4500 frames),
every other video whole. Also the mean number of kept (animal-mask) tokens per frame.

Output: results/vision/eci_atlas_v37/token_share.json {domain: {series: {n_videos, n_frames, kept_per_frame,
tokens_per_frame: [W]}}}
Usage: srun -p defaultp -c 4 --mem=16G python scripts/eci/atlas_token_share.py
"""
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2')
OUT = REPO / 'results/vision/eci_atlas_v37/token_share.json'
N_MATCH = {'mice': 4500}


def main():
    out = {}
    for dom in ('mice', 'ants', 'frogs'):
        c = json.loads((REPO / f'configs/eci_v2/{dom}.json').read_text())
        for sc in c['series']:
            d = BEE / dom / 'videos' / sc['name']
            if not d.is_dir():
                continue
            t0 = time.time()
            P = json.loads((REPO / sc['plan']).read_text())
            cnt, nf, nk, nv, W = None, 0, 0, 0, None
            for v in P['videos']:
                f = d / f'{v["obs"]}.npz'
                if not f.exists():
                    continue
                z = np.load(f)
                W = int(z['W'])
                n_kept = z['n_kept'].astype(np.int64)
                n = len(n_kept)
                k0 = max(n - N_MATCH[dom], 0) if dom in N_MATCH and v['obs'].endswith('_H') else 0
                tok0 = np.r_[0, np.cumsum(n_kept)]
                act0 = np.r_[0, np.cumsum(z['act_count'].astype(np.int64))]
                a0, a1 = act0[tok0[k0]], act0[tok0[n]]
                b = np.bincount(z['act_idx'][a0:a1].astype(np.int64), minlength=W)
                cnt = b if cnt is None else cnt + b
                nf += n - k0
                nk += int(n_kept[k0:].sum())
                nv += 1
            if not nv:
                continue
            out.setdefault(dom, {})[sc['name']] = {
                'n_videos': nv, 'n_frames': nf, 'kept_per_frame': round(nk / nf, 3),
                'tokens_per_frame': [round(float(x), 4) for x in cnt / nf]}
            tp = cnt / nf
            print(f'{dom} {sc["name"]}: {nv} videos, {nf} frames, {nk / nf:.1f} kept tokens / frame; tokens / frame of '
                  f'an active neuron: median {np.median(tp):.3f}, max {tp.max():.2f} ({time.time() - t0:.0f} s)', flush=True)
    OUT.write_text(json.dumps(out, separators=(',', ':')))
    print('->', OUT)


if __name__ == '__main__':
    main()
