"""
MABe22 mouse ECI round 3, E2 pooled selection (GPU job; rules: eci_round3.sh E2). Reads the saved eval codes of the
per-lag dictionaries (eci_r3_train.py --save-codes) and scores the pooled neuron sets with the usual cross-fit
(eci_train.score: selection on one clip half, scored on the other, both directions, split seeds 0 and 1).
    union  run j: L15, L30, R (= lag 60), L120 dictionaries of seed 200+j, codes concatenated (4W neurons)
    mp     matched pool run j: R dictionaries of seeds 200+j, 210+j, 220+j, 230+j
Run names carry the width suffix ('' for 4096, '_w<W>' otherwise). Output OUT/r3/eval/{U,MP}<suf>_s<200+j>.json with
'pool' (score result), 'members' and, per behaviour, the member index of the AUROC- / AP-picked neuron per split and
direction (member = neuron // W; for the union the member order is lag 15, 30, 60, 120 frames).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/eci'))
from eci_common import BEHAVIOURS, OUT, PLAN, log  # noqa: E402
import eci_train as ET  # noqa: E402

RUN = OUT / 'r3'


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=4096)
    ap.add_argument('--runs', nargs='+', type=int, default=list(range(6)))
    a = ap.parse_args()
    suf = '' if a.width == 4096 else f'_w{a.width}'
    dev = torch.device('cuda')
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    Ht = {s: ET.halves(E, s) for s in (0, 1)}
    for j in a.runs:
        for kind, members in (('U', [(f'L15{suf}', 200 + j), (f'L30{suf}', 200 + j), (f'R{suf}', 200 + j),
                                     (f'L120{suf}', 200 + j)]),
                              ('MP', [(f'R{suf}', s) for s in (200 + j, 210 + j, 220 + j, 230 + j)])):
            C = []
            for r, s in members:
                c = np.load(RUN / 'codes' / f'{r}_s{s}.npy')
                assert c.shape == (len(E), a.width), (r, s, c.shape)
                C.append(c)
            codes = torch.from_numpy(np.concatenate(C, 1)).to(dev)
            res = ET.score(codes, E, Ht, dev)
            pick = {b: {sp: {'auc': [d['auc']['neuron'] // a.width for d in res[sp][b]['dirs']],
                             'ap': [d['ap']['neuron'] // a.width for d in res[sp][b]['dirs']]} for sp in res}
                    for b in BEHAVIOURS}
            out = {'run': f'{kind}{suf}', 'seed': 200 + j, 'members': members, 'pool': res, 'full': 'pool',
                   'member_of_pick': pick}
            (RUN / 'eval' / f'{kind}{suf}_s{200 + j}.json').write_text(json.dumps(out, indent=1, default=float))
            r0 = res['split0']
            log(f'{kind}{suf} run {j}: ' + ', '.join(f"{b[:10]} {r0[b]['cf_auroc']:.3f}" for b in BEHAVIOURS[:5]))
            del codes
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
