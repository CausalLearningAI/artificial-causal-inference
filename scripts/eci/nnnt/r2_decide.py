"""Round 2 decision under the pre-registered rules (r2.py docstring) -> results/vision/eci_nnnt/r2/{decision.json,table.md}.
CPU, reads only the json results: r2/sweep.json (diagnosis frames, seeds 0-5), r2/sweep_sub20.json (1024 SAEs on the
every-20th frames), r2_s2048/sweep.json (2048 px SAEs on the same frames), r2/diag.json (token / code ceilings)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parents[3] / 'results/vision/eci_nnnt'
LB = ('nose_nose', 'nose_tail')


def welch(a, b):
    """mean(a) - mean(b) with the Welch 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean()
    t = stats.t.ppf(0.975, df)
    return {'diff': float(d), 'lo': float(d - t * se), 'hi': float(d + t * se), 'df': float(df)}


def vals(sets, arm, seeds, q):
    out = []
    for s in seeds:
        r = sets.get(f'{arm}_s{s}')
        if r is None:
            continue
        out.append(r['primary'] if q == 'primary' else r['mean'][q[0]][q[1]])
    return out


def row(sets, arm, seeds):
    r = {'n_seeds': len(vals(sets, arm, seeds, 'primary'))}
    for q, key in (('primary', 'primary'), ('nn_R', ('nose_nose', 'R')), ('nn_A', ('nose_nose', 'A')),
                   ('nt_R', ('nose_tail', 'R')), ('nt_A', ('nose_tail', 'A'))):
        v = vals(sets, arm, seeds, key)
        r[q] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float('nan')) if v else None
    return r


def fmt(x):
    return '-' if x is None else f'{x[0]:.3f} ± {x[1]:.3f}'


def main():
    sw = json.loads((OUT / 'r2/sweep.json').read_text())['sets']
    arms = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    dec = {'table_seeds012': {a: row(sw, a, (0, 1, 2)) for a in arms}}
    cand = {a: dec['table_seeds012'][a]['primary'][0] for a in arms[1:] if dec['table_seeds012'][a]['primary']}
    win = max(cand, key=cand.get)
    dec['winner_seeds012'] = win
    dec['table_seeds0_5'] = {a: row(sw, a, range(6)) for a in arms}
    S6 = range(6)
    conf = {'n_winner': len(vals(sw, win, S6, 'primary')), 'n_S0': len(vals(sw, 'S0', S6, 'primary'))}
    if conf['n_winner'] >= 6 and conf['n_S0'] >= 6:
        conf['primary'] = welch(vals(sw, win, S6, 'primary'), vals(sw, 'S0', S6, 'primary'))
        for lb in LB:
            conf[lb] = welch(vals(sw, win, S6, (lb, 'R')), vals(sw, 'S0', S6, (lb, 'R')))
        conf['fresh_seeds345_primary'] = welch(vals(sw, win, (3, 4, 5), 'primary'), vals(sw, 'S0', (3, 4, 5), 'primary'))
        p = conf['primary']
        conf['KEEP'] = bool(p['lo'] > 0 and p['diff'] >= 0.02)
    dec['confirmation_vs_S0'] = conf
    # 2048 zoom
    z = {}
    f20, f48 = OUT / 'r2/sweep_sub20.json', OUT / 'r2_s2048/sweep.json'
    if f20.exists() and f48.exists():
        s20 = json.loads(f20.read_text()).get('sets_sub20', {})
        s48 = json.loads(f48.read_text())['sets']
        rec = sorted({'S0', 'S2', 'S3', win})
        z['recipes'] = rec
        z['at1024_sub20'] = {a: row(s20, a, (0, 1, 2)) for a in arms}
        z['at2048'] = {a: row(s48, a, (0, 1, 2)) for a in arms}
        ok = [a for a in rec if z['at2048'][a]['primary'] and z['at1024_sub20'][a]['primary']]
        if ok:
            b48 = max(ok, key=lambda a: z['at2048'][a]['primary'][0])
            b10 = max(ok, key=lambda a: z['at1024_sub20'][a]['primary'][0])
            z['best2048'], z['best1024'] = b48, b10
            z['primary'] = welch(vals(s48, b48, (0, 1, 2), 'primary'), vals(s20, b10, (0, 1, 2), 'primary'))
            for lb in LB:
                z[lb] = welch(vals(s48, b48, (0, 1, 2), (lb, 'R')), vals(s20, b10, (0, 1, 2), (lb, 'R')))
            z['KEEP'] = bool(z['primary']['lo'] > 0 and z['primary']['diff'] >= 0.02 and
                             min(z['nose_nose']['diff'], z['nose_tail']['diff']) >= -0.02)
            z['same_recipe_2048_minus_1024'] = {a: welch(vals(s48, a, (0, 1, 2), 'primary'),
                                                         vals(s20, a, (0, 1, 2), 'primary')) for a in ok}
    dec['zoom2048'] = z
    fd = OUT / 'r2/diag.json'
    if fd.exists():
        d = json.loads(fd.read_text())
        dec['ceilings'] = {'N-best': d['N-best'], **d['probes']}
    (OUT / 'r2/decision.json').write_text(json.dumps(dec, indent=1))
    L = ['# NN/NT round 2 — decision (pre-registered rule, r2.py)', '',
         'R~ = cross-fitted normalised R-precision (10 random pool splits); AUROC = test AUROC of the R~-selected neuron '
         '(A_at_R is in sweep.json; here cf AUROC); mean ± sd over SAE seeds. Diagnosis frames: 86,400.', '',
         '| arm | seeds | primary R~ | nn R~ | nn AUROC | nt R~ | nt AUROC |', '|---|---|---|---|---|---|---|']
    for tag, key in (('0-2', 'table_seeds012'), ('0-5', 'table_seeds0_5')):
        for a in arms:
            r = dec[key][a]
            if r['primary'] is None or (tag == '0-5' and r['n_seeds'] < 6):
                continue
            L.append(f'| {a} | {tag} | {fmt(r["primary"])} | {fmt(r["nn_R"])} | {fmt(r["nn_A"])} | {fmt(r["nt_R"])} | '
                     f'{fmt(r["nt_A"])} |')
    L += ['', f'Winner (seeds 0-2, S1-S7): **{win}**. Confirmation vs S0 (6 vs 6, Welch 95% CI): '
          + json.dumps(conf), '']
    if z:
        L += ['## 2048 px zoom (43,200 frames; 1024 SAEs re-read on the same frames)', '',
              '| arm | primary R~ 1024 | primary R~ 2048 | nn R~ 1024 | nn R~ 2048 | nt R~ 1024 | nt R~ 2048 |',
              '|---|---|---|---|---|---|---|']
        for a in arms:
            r1, r2_ = z['at1024_sub20'][a], z['at2048'][a]
            if r1['primary'] is None and r2_['primary'] is None:
                continue
            L.append(f'| {a} | {fmt(r1["primary"])} | {fmt(r2_["primary"])} | {fmt(r1["nn_R"])} | {fmt(r2_["nn_R"])} | '
                     f'{fmt(r1["nt_R"])} | {fmt(r2_["nt_R"])} |')
        L += ['', 'Zoom rule: ' + json.dumps({k: z.get(k) for k in ('best2048', 'best1024', 'primary', 'nose_nose',
                                                                    'nose_tail', 'KEEP')}), '']
    if 'ceilings' in dec:
        L += ['## Ceilings (supervised, labels used only here)', '', '| readout | nn R~ | nn AUROC | nt R~ | nt AUROC |',
              '|---|---|---|---|---|']
        for k, v in dec['ceilings'].items():
            L.append(f'| {k} | {v["nose_nose"]["R"]:.3f} | {v["nose_nose"]["A"]:.3f} | {v["nose_tail"]["R"]:.3f} | '
                     f'{v["nose_tail"]["A"]:.3f} |')
    (OUT / 'r2/table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
