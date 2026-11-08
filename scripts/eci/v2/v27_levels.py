"""
ECI v2.7 level curve (pre-registration scripts/eci/v2/v27_prereg.md): for m = 1..5 (Matryoshka levels: latents < 128, 256,
512, 2048, 4096), the held-out AUROC of the best selection-set neuron restricted to levels 1..m (eligible: readout > 0 on
>= 0.2% of the scored frames). Mice: select on SG pools, measure on non-SG (edge-excluded ext frames), nose-nose and
nose-tail, readout = HMM P(on). Ants: select on half A, measure on B, grooming-to-focal (any). Mean over seeds.
usage: v27_levels.py --arm PREDc --seeds 2701,2702,2703 [--out NAME]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v27_eval as W7  # noqa: E402

EV = W7.EV
C, AN = EV.C, EV.AN
PREF = (128, 256, 512, 2048, 4096)


def curve(sel_auc, ev_auc, el):
    out = []
    for m in PREF:
        sc = np.where(el[:m], sel_auc[:m], np.nan)
        j = int(np.nanargmax(sc))
        out.append({'level_max': PREF.index(m) + 1, 'n_latents': m, 'pick': j, 'sel_auc': float(sel_auc[j]), 'heldout_auc': float(ev_auc[j]),
                    'oracle_heldout_auc': float(np.nanmax(np.where(el[:m], ev_auc[:m], np.nan)))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    F, _ = C.mice_table()
    sel, ev = np.flatnonzero(F.sg.values), np.flatnonzero(F.nonsg.values)
    A = AN.table()
    ya = A.y_groom_any.values
    res = {'arm': a.arm, 'seeds': {}}
    for s in a.seeds.split(','):
        key = f'{a.arm}_s{s}'
        X, R, _ = C.load(key)
        el = (X > 0).mean(0) >= C.ELIG
        st_s, st_e = C.neuron_stats_multi(X, F, [sel, ev], None)
        r = {b: curve(st_s[f'auc_{b}'], st_e[f'auc_{b}'], el) for b in C.BEH}
        del X, R
        Xa, _, _ = AN.readout(key, A)
        ela = (Xa > 0).mean(0) >= C.ELIG
        cat = np.stack([np.where(A.half.values == h, np.where(ya, 2, 1), 0) for h in 'AB'], 1).astype(np.int8)
        aa = C._auc_multi(np.ascontiguousarray(Xa.T, dtype=np.float32), cat)
        r['groom_any'] = curve(aa[0], aa[1], ela)
        res['seeds'][s] = r
        print(key, {b: [round(c['heldout_auc'], 3) for c in v] for b, v in r.items()}, flush=True)
    res['mean'] = {b: [float(np.mean([res['seeds'][s][b][i]['heldout_auc'] for s in res['seeds']])) for i in range(5)] for b in ('nose_nose', 'nt', 'groom_any')}
    od = W7.T7.OUT27 / 'levels'
    od.mkdir(parents=True, exist_ok=True)
    (od / f'{a.out or a.arm}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res['mean']))


if __name__ == '__main__':
    main()
