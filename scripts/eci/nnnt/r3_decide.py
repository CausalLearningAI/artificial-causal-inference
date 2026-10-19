"""Round 3 decision under the pre-registered rule (r3.py docstring) -> results/vision/eci_nnnt/r3/{decision.json,table.md}.
CPU; reads r3/score*.json (all arms incl. refs S0 / S2 re-scored by r3), r2/decision.json (round-2 reference)."""
import json
from pathlib import Path

import numpy as np

from r2_decide import welch

OUT = Path(__file__).resolve().parents[3] / 'results/vision/eci_nnnt'
ONLINE = {'deployed': {'nose_nose': 0.070, 'nose_tail': 0.108}, 'C1': {'nose_nose': 0.059, 'nose_tail': 0.087}}
ARMS = ['S0', 'S2', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']


def get(sets, arm, seeds, f):
    return [f(sets[f'{arm}_s{s}']) for s in seeds if f'{arm}_s{s}' in sets]


Q = {'primary': lambda r: r['primary'], 'nn_R': lambda r: r['mean']['nose_nose']['R'],
     'nt_R': lambda r: r['mean']['nose_tail']['R'], 'nn_A': lambda r: r['mean']['nose_nose']['A'],
     'nt_A': lambda r: r['mean']['nose_tail']['A'], 'primary_c': lambda r: r['spec']['primary_c'],
     'nn_cR': lambda r: r['spec']['nose_nose']['cR'], 'nt_cR': lambda r: r['spec']['nose_tail']['cR'],
     'nn_top_label': lambda r: r['spec']['nose_nose']['top_label'],
     'nt_top_label': lambda r: r['spec']['nose_tail']['top_label'],
     'nn_top_generic': lambda r: r['spec']['nose_nose']['top_generic'],
     'nt_top_generic': lambda r: r['spec']['nose_tail']['top_generic'],
     'nn_on_nt': lambda r: r['mean']['nose_nose']['cross_R'][1], 'nt_on_nn': lambda r: r['mean']['nose_tail']['cross_R'][0]}


def ms(v):
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float('nan'), len(v)) if v else None


def main():
    sets = {}
    for f in sorted((OUT / 'r3').glob('score*.json')):
        sets.update(json.loads(f.read_text()))
    d2 = json.loads((OUT / 'r2/decision.json').read_text())
    ref = d2['winner_seeds012'] if d2.get('confirmation_vs_S0', {}).get('KEEP') else 'S0'
    tab = {a: {tag: {q: ms(get(sets, a, sd, f)) for q, f in Q.items()} for tag, sd in (('0-2', (0, 1, 2)),
                                                                                        ('0-5', range(6)))}
           for a in ARMS + ([ref] if ref not in ARMS else [])}
    cand = {a: tab[a]['0-2']['primary'][0] for a in ARMS if a.startswith('R') and tab[a]['0-2']['primary']}
    win = max(cand, key=cand.get)
    candc = {a: tab[a]['0-2']['primary_c'][0] for a in ARMS if a.startswith('R') and tab[a]['0-2']['primary_c']}
    winc = max(candc, key=candc.get)
    dec = {'reference': ref, 'winner_primary': win, 'winner_primary_c': winc, 'table': tab}
    for tag, w in (('confirm_primary', win), ('confirm_primary_c', winc)):
        c = {}
        for q in ('primary', 'nn_R', 'nt_R', 'primary_c', 'nn_cR', 'nt_cR'):
            a_, b_ = get(sets, w, range(6), Q[q]), get(sets, ref, range(6), Q[q])
            if len(a_) >= 6 and len(b_) >= 6:
                c[q] = welch(a_, b_)
        if 'primary' in c:
            c['KEEP'] = bool(c['primary']['lo'] > 0 and c['primary']['diff'] >= 0.02)
        dec[tag] = {'arm': w, **c}
    # goal criterion
    g = {}
    for lb, q in (('nose_nose', 'nn_R'), ('nose_tail', 'nt_R')):
        v = get(sets, win, range(6), Q[q])
        best_online = max(ONLINE['deployed'][lb], ONLINE['C1'][lb])
        if len(v) >= 2:
            from scipy import stats
            m, se = np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))
            lo = m - stats.t.ppf(0.975, len(v) - 1) * se
            g[lb] = {'mean': float(m), 'lo95': float(lo), 'target_2x_online': 2 * best_online,
                     'pass_R': bool(m >= 2 * best_online and lo > best_online)}
    dec['goal'] = g
    (OUT / 'r3/decision.json').write_text(json.dumps(dec, indent=1))
    f = lambda x: '-' if x is None else f'{x[0]:.3f} ± {x[1]:.3f}'  # noqa: E731
    L = ['# NN/NT round 3 — specificity (pre-registered rule, r3.py)', '',
         'R~ = cross-fitted normalised R-precision (10 random pool splits); cR~ = the same on CONTACT frames only (label-'
         'free mask proxy: largest blob >= 1.5 x one mouse); top generic = share of the top-N_pos held-out frames that are '
         'contact frames with no nn/np/nt label; mean ± sd over SAE seeds 0-5 (R6: 0-2). Online refs: deployed nn R~ '
         '0.070 / nt 0.108, C1 0.059 / 0.087.', '',
         '| arm | seeds | primary R~ | nn R~ | nt R~ | nn AUROC | nt AUROC | nn cR~ | nt cR~ | nn top-label | '
         'nn top-generic | nt top-label | nt top-generic |', '|' + '---|' * 13]
    for a in tab:
        t = tab[a]['0-5'] if tab[a]['0-5']['primary'] and tab[a]['0-5']['primary'][2] >= 6 else tab[a]['0-2']
        if t['primary'] is None:
            continue
        L.append(f'| {a} | {t["primary"][2]} | ' + ' | '.join(f(t[q]) for q in (
            'primary', 'nn_R', 'nt_R', 'nn_A', 'nt_A', 'nn_cR', 'nt_cR', 'nn_top_label', 'nn_top_generic',
            'nt_top_label', 'nt_top_generic')) + ' |')
    L += ['', f'Reference (round-2 decision): {ref}. Winner by primary R~ (seeds 0-2): **{win}**; by contact-frame '
          f'cR~: **{winc}**.', '', '```', json.dumps({k: dec[k] for k in ('confirm_primary', 'confirm_primary_c',
                                                                        'goal')}, indent=1), '```']
    (OUT / 'r3/table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
