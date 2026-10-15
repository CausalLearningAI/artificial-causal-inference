"""
MABe22 mouse ECI - one summary figure for the PI (login safe: reads JSONs, matplotlib only).

a  delta primary AUROC vs A0 per arm, grouped by the assumption tested, points +- 1.96 SE over SAE seeds.
   Primary here = the CONFIRMATION definition (mean cf AUROC over oral_oral, oral_genital, approaches, huddles) for
   every arm, so the graded grid (3 seeds, vs the graded A0's 3 seeds) and the confirmation run (6 fresh seeds, vs its
   own A0) share one scale. SE of a delta = sqrt(s_a^2 / n_a + s_0^2 / n_0) (sd with ddof 1). Hollow = post hoc.
b  oral_oral and oral_genital: AUROC and AP of the A0 best neuron, the best arm's best neuron, the pose reference
   (-min nose-nose / -min nose-tailbase distance) and the supervised ceilings (mean_lin, mil_ctx on A0 tokens);
   dashed = chance (AUROC 0.5, AP = weighted base rate).
Outputs: OUT/fig_grid.png, OUT/fig_grid.pdf
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import OUT, PLAN  # noqa: E402

P4 = ['oral_oral_contact', 'oral_genital_contact', 'approaches', 'huddles']
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
GROUPS = [('temporal context', [('A4', 'none'), ('D02', '0.2 s'), ('D05', '0.5 s'), ('D2', '2 s'), ('X_D4', '4 s*')]),
          ('mask', [('A0_labmask', 'lab mask'), ('A6', 'no mask')]),
          ('dictionary', [('A3', 'plain BTK'), ('A2k8', 'frame budget k8'), ('A2k16', 'frame budget k16'),
                          ('A1', 'k = 8')]),
          ('resolution', [('A5', 'native 224')]),
          ('orbit', [('A7', 'D4 orbit'), ('X_A7D2', 'orbit + 2 s*')])]
CONF = [('D2', '2 s'), ('X_D4', '4 s'), ('A7', 'D4 orbit'), ('X_A7D2', 'orbit + 2 s')]


def seeds(d, arm, lo=0, hi=99):
    out = {}
    for f in sorted(Path(d).glob(f'{arm}_s*.json')):
        m = re.fullmatch(rf'{re.escape(arm)}_s(\d+)\.json', f.name)
        if m and lo <= int(m.group(1)) <= hi:
            r = json.loads(f.read_text())['m4096']
            out[int(m.group(1))] = r
    return out


def prim(R):
    return np.array([np.mean([np.mean([r[sp][b]['cf_auroc'] for sp in r]) for b in P4]) for r in R.values()])


def delta(a, b):
    return a.mean() - b.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def lab(R, b, q):
    return float(np.mean([r[sp][b][q] for r in R.values() for sp in r]))


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6, 'axes.linewidth': 0.5,
                         'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2,
                         'ytick.major.size': 2, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.labelcolor': INK, 'pdf.fonttype': 42})
    ev = OUT / 'eval'
    A0 = prim(seeds(ev, 'A0'))
    rows = []  # (group, label, delta, se, colour, hollow)
    for g, arms in GROUPS:
        for arm, name in arms:
            d_ = OUT / 'labmask' / 'eval' if arm == 'A0_labmask' else ev
            R = seeds(d_, 'A0' if arm == 'A0_labmask' else arm, 0, 2)
            if not R:
                continue
            d, se = delta(prim(R), A0)
            rows.append((g, name, d, se, BLUE, arm.startswith('X_')))
    cd = OUT / 'confirm' / 'eval'
    C0 = seeds(cd, 'A0', 100, 199)
    if C0:
        c0 = prim(C0)
        for arm, name in CONF:
            R = seeds(cd, arm, 100, 199)
            if R:
                d, se = delta(prim(R), c0)
                rows.append(('confirmation (6 fresh seeds)', name, d, se, ORANGE, False))
    fig = plt.figure(figsize=(7.1, 3.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], hspace=0.55, wspace=0.45, left=0.15, right=0.99, top=0.92,
                          bottom=0.2)
    ax = fig.add_subplot(gs[:, 0])
    y, yt, yl, gy = 0, [], [], []
    last = None
    for g, name, d, se, col, hollow in rows:
        if g != last:
            y += 0.7 if last is not None else 0
            gy.append((y, g))
            last = g
        ax.errorbar(d, -y, xerr=1.96 * se, fmt='o', ms=3.5, color=col, mfc='white' if hollow else col, mew=0.8,
                    elinewidth=0.8, capsize=0)
        yt.append(-y)
        yl.append(name)
        y += 1
    for yy, g in gy:
        ax.text(-0.02, -yy + 0.55, g, transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=6,
                fontweight='bold', color=INK)
    ax.axvline(0, color=MUTED, lw=0.6, ls='--')
    ax.set_yticks(yt, yl)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-y + 0.3, 1.3)
    ax.set_xlabel('Δ primary AUROC vs A0 (1 s Δ, Matryoshka, mask, 448)\n± 1.96 SE over SAE seeds')
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%+.2f'))
    ax.grid(axis='x', color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.set_title('a', loc='left', fontweight='bold', fontsize=8, x=-0.55)
    ax.text(0.0, 1.0, '● graded grid (3 seeds)   ○ post hoc   ', transform=ax.transAxes, fontsize=5, color=BLUE,
            va='bottom')
    if C0:
        ax.text(0.62, 1.0, '● confirmation', transform=ax.transAxes, fontsize=5, color=ORANGE, va='bottom')
    # panel b
    E = json.loads((PLAN / 'eval_plan.json').read_text())['weighted_base_rate']
    S = json.loads((OUT / 'summary.json').read_text())
    pose = json.loads((OUT / 'pose' / 'pose.json').read_text())['scores']
    best = None
    cj = OUT / 'confirm' / 'confirm.json'
    if cj.exists():
        dec = json.loads(cj.read_text()).get('decision', '')
        best = {'D2': ('D2', cd), 'D4': ('X_D4', cd), 'A7': ('A7', cd), 'A7D2': ('X_A7D2', cd)}.get(dec)
    if best is None:  # no winner: the confirmation arm with the highest primary mean, else the graded D2
        cand = [(prim(seeds(cd, a, 100, 199)).mean(), a) for a, _ in CONF if seeds(cd, a, 100, 199)]
        best = (max(cand)[1], cd) if cand else ('D2', ev)
    Rbest = seeds(best[1], best[0], 100 if best[1] == cd else 0, 199 if best[1] == cd else 2)
    RA0 = C0 if (best[1] == cd and C0) else seeds(ev, 'A0', 0, 2)
    bname = {'X_D4': 'D4', 'X_A7D2': 'A7D2'}.get(best[0], best[0])
    names = ['A0 neuron', f'{bname} neuron', 'pose', 'linear probe', 'MIL probe']
    cols = [BLUE, BLUE, ORANGE, AQUA, AQUA]
    alphas = [0.55, 1, 1, 0.55, 1]
    for j, (b, pkey, title) in enumerate((('oral_oral_contact', 'neg_min_nose_nose', 'oral-oral'),
                                          ('oral_genital_contact', 'neg_min_nose_tailbase', 'oral-genital'))):
        ce = S['ceiling']
        for i, (q, pq, lab_) in enumerate((('cf_auroc', 'auroc', 'AUROC'), ('cf_ap', 'ap', 'AP'))):
            v = [lab(RA0, b, q), lab(Rbest, b, q), pose[pkey][b][pq],
                 ce[f'{b}/mean_lin'][f'{pq}_mean'], ce[f'{b}/mil_ctx'][f'{pq}_mean']]
            a = fig.add_subplot(gs[i, 1 + j])
            for k, (vv, c, al) in enumerate(zip(v, cols, alphas)):
                a.bar(k, vv, width=0.7, color=c, alpha=al, edgecolor='white', linewidth=0.6)
                a.text(k, vv, f'{vv:.2f}' if q == 'cf_auroc' else f'{vv:.3f}', ha='center', va='bottom', fontsize=5,
                       color=INK)
            chance = 0.5 if q == 'cf_auroc' else E[b]
            a.axhline(chance, color=MUTED, lw=0.6, ls='--')
            a.text(4.45, chance, 'chance' if q == 'cf_auroc' else 'base rate', fontsize=5, color=MUTED, ha='right',
                   va='bottom')
            a.set_ylim(0.4 if q == 'cf_auroc' else 0, 1.0 if q == 'cf_auroc' else max(v) * 1.25)
            a.set_xticks(range(5), names if i == 1 else [''] * 5, rotation=40, ha='right')
            a.set_ylabel(lab_)
            for s in ('top', 'right'):
                a.spines[s].set_visible(False)
            if i == 0:
                a.set_title(title, fontsize=6.5, color=INK)
                if j == 0:
                    a.set_title('b', loc='left', fontweight='bold', fontsize=8, x=-0.35)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig_grid.{ext}', dpi=300)
    print('saved', OUT / 'fig_grid.png', 'best arm', best[0], 'rows', len(rows))


if __name__ == '__main__':
    main()
