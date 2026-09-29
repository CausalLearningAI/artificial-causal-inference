"""
Gates for a new foreground SAE representation vs fg448 (ECI mice v1; behaviour labels are used
for EVALUATION ONLY, never for training, masks or NES).

Per representation (two seeds s0/s1, eval codes from scripts/eci/fg_eval_encode.py):
1. token level on the held-out pools (metrics.json): FVE / L0 / dead per prefix (+ per-block FVE
   of motion SAEs: static token half vs token-change half);
2. cross-seed decoder stability (src.eci.sae.decoder_stability);
3. behaviour check on every 5 fps frame of the 18 annotated held-out videos, events Y_nn / Y_np /
   Y_nt > 0: best single-latent AUROC of the per-frame codes_max, per prefix;
4. size-controlled AUROC: the same, computed WITHIN quintiles of a foreground-size covariate and
   pooled over quintiles (stratified AUROC = sum_q n1_q n0_q AUROC_q / sum_q n1_q n0_q, i.e. the
   probability that a positive frame outranks a negative frame of the SAME size quintile).
   Covariates: 'size_fg448' = the fg448 n_fg (common to all representations, how spread out /
   big the mouse blob is) and 'size_own' = the representation's own n_fg. The raw AUROC of the
   covariate itself is reported for reference.
Gate (pass): FVE(1024) >= 0.7, dead(1024) < 2%, and the best size-controlled (size_fg448) AUROC
beats fg448 for >= 2 of the 3 events at prefix 1024.

Output: --out JSON (default dataset/mice/v1/eci/fgv3/gates.json) and a printed table.

Usage:
    python scripts/eci/eval_sae_gates.py --rep fg448=matryoshka_btk_1024_k16_fg448:dataset/mice/v1/eci/fg448/eval_codes \
        --rep v3=matryoshka_btk_1024_k16_fgv3:dataset/mice/v1/eci/fgv3/eval_codes
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from eval_sae_fg import EVENTS, auroc_cols  # noqa: E402
from src.eci.sae import decoder_stability, load_sae  # noqa: E402

PREFIXES = (128, 256, 512, 1024)


def load_eval(d, sae):
    parts = [np.load(f) for f in sorted(Path(d).glob('task_*.npz')) if '.tmp' not in f.name]
    rows = np.concatenate([z['rows'] for z in parts])
    o = np.argsort(rows)
    return rows[o], np.concatenate([z['n_fg'] for z in parts])[o], np.concatenate([z[f'codes_max_{sae}'] for z in parts])[o]


def stratified_auroc(X, y, strata):
    num, den = np.zeros(X.shape[1]), 0.0
    for g in np.unique(strata):
        s = strata == g
        n1, n0 = int(y[s].sum()), int((~y[s]).sum())
        if n1 == 0 or n0 == 0:
            continue
        num += n1 * n0 * auroc_cols(X[s], y[s])
        den += n1 * n0
    return num / den


def quintiles(x):
    cuts = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(cuts, x, side='right')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rep', action='append', required=True, help='label=sae_prefix:eval_codes_dir (seeds _s0/_s1)')
    p.add_argument('--sae-dir', default=str(REPO / 'dataset/mice/v1/eci/sae'))
    p.add_argument('--out', default=str(REPO / 'dataset/mice/v1/eci/fgv3/gates.json'))
    args = p.parse_args()
    reps = {}
    for r in args.rep:
        lab, rest = r.split('=', 1)
        pre, d = rest.split(':', 1)
        reps[lab] = (pre, REPO / d if not Path(d).is_absolute() else Path(d))
    ann = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv', usecols=list(EVENTS))
    base_rows = base_nfg = None
    res = {}
    for lab, (pre, d) in reps.items():
        out = res[lab] = {'sae': pre}
        tl = {}
        for s in ('s0', 's1'):
            m = json.loads((Path(args.sae_dir) / f'{pre}_{s}' / 'metrics.json').read_text())
            tl[s] = {k: {kk: v[kk] for kk in ('fve', 'l0_per_token', 'dead_frac')} for k, v in m['val_threshold']['prefixes'].items()}
            tl[s]['n_val_tokens'] = m['n_val_tokens']
            tl[s]['blocks'] = m.get('val_blocks')
        out['token_level'] = tl
        W = [load_sae(Path(args.sae_dir) / f'{pre}_{s}' / 'sae.pt')[0].W_dec.detach() for s in ('s0', 's1')]
        out['stability'] = decoder_stability(W[0], W[1], PREFIXES)
        out['auroc'] = {}
        for s in ('s0', 's1'):
            rows, nfg, X = load_eval(d, f'{pre}_{s}')
            if base_rows is None:
                base_rows, base_nfg = rows, nfg.astype(np.float64)
            if not np.array_equal(rows, base_rows):
                raise ValueError(f'{lab} {s}: eval rows differ from the first representation')
            X = X.astype(np.float32)
            lab_df = ann.iloc[rows]
            q_common, q_own = quintiles(base_nfg), quintiles(nfg.astype(np.float64))
            a = {}
            for ev in EVENTS:
                y = lab_df[ev].values > 0
                raw = auroc_cols(X, y)
                sc = stratified_auroc(X, y, q_common)
                so = stratified_auroc(X, y, q_own)
                a[ev] = {'n_pos': int(y.sum()), 'n': int(len(y)),
                         'size_fg448_auroc': float(auroc_cols(base_nfg[:, None], y)[0]),
                         'size_own_auroc': float(auroc_cols(nfg[:, None].astype(np.float64), y)[0]),
                         'per_prefix': {str(m): {
                             'raw': {'latent': int(np.argmax(raw[:m])), 'auroc': float(raw[:m].max())},
                             'size_fg448': {'latent': int(np.argmax(sc[:m])), 'auroc': float(sc[:m].max()),
                                            'raw_auroc_of_that_latent': float(raw[int(np.argmax(sc[:m]))])},
                             'size_own': {'latent': int(np.argmax(so[:m])), 'auroc': float(so[:m].max())}}
                             for m in PREFIXES},
                         'top5_size_fg448': [{'latent': int(j), 'auroc_sc': float(sc[j]), 'auroc_raw': float(raw[j])}
                                             for j in np.argsort(-sc)[:5]]}
            out['auroc'][s] = a
    base = list(res)[0]
    for lab in list(res)[1:]:
        r, b = res[lab], res[base]
        t = r['token_level']['s0']['1024']
        wins = [ev for ev in EVENTS if r['auroc']['s0'][ev]['per_prefix']['1024']['size_fg448']['auroc']
                > b['auroc']['s0'][ev]['per_prefix']['1024']['size_fg448']['auroc']]
        r['gate'] = {'fve_ok': t['fve'] >= 0.7, 'dead_ok': t['dead_frac'] < 0.02, 'size_controlled_wins_vs_' + base: wins,
                     'passed': bool(t['fve'] >= 0.7 and t['dead_frac'] < 0.02 and len(wins) >= 2)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    for lab, r in res.items():
        print(f'== {lab} ({r["sae"]})')
        for s in ('s0', 's1'):
            print(' ', s, {m: (round(v['fve'], 3), round(v['l0_per_token'], 1), round(100 * v['dead_frac'], 1))
                           for m, v in r['token_level'][s].items() if m.isdigit()}, 'blocks', r['token_level'][s]['blocks'])
        print('  stability', {m: round(v['frac_gt_0.9_same_prefix'], 3) for m, v in r['stability'].items()})
        for s in ('s0', 's1'):
            for ev in EVENTS:
                a = r['auroc'][s][ev]
                print(f'  {s} {ev} size-only {a["size_fg448_auroc"]:.3f}/{a["size_own_auroc"]:.3f} ',
                      ' '.join(f'm{m}: raw {v["raw"]["auroc"]:.3f}(#{v["raw"]["latent"]}) sc {v["size_fg448"]["auroc"]:.3f}'
                               f'(#{v["size_fg448"]["latent"]}) so {v["size_own"]["auroc"]:.3f}'
                               for m, v in a['per_prefix'].items()))
        if 'gate' in r:
            print('  GATE', r['gate'])


if __name__ == '__main__':
    main()
