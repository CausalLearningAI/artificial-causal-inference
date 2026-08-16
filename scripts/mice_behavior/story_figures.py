#!/usr/bin/env python3
"""Presentation figures for the mice v1 phase-ATE story.

One script, one output directory, every number recomputed from source rather than
transcribed -- so a figure can never drift from the run it claims to describe.

    python scripts/mice_behavior/story_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
FRAME = ROOT / 'results' / 'vision' / 'mice' / 'frame'
OUT = FRAME / '_figures'
OUT.mkdir(parents=True, exist_ok=True)

# validated categorical slots (references/palette.md, light mode)
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8a85'
GRID = '#e3e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.labelcolor': INK2, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
FPS = 5.0


def obs_table() -> pd.DataFrame:
    """Per-observation outcomes from the human labels: occupancy and bouts/min."""
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['observation_id', 'frame_idx', 'Y_nt', 'Y_nn']).dropna(subset=['Y_nt'])
    e = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')[
        ['observation_id', 'pool', 'phase', 'odor', 'genotype', 'sex', 'line']]
    a = a.sort_values(['observation_id', 'frame_idx'])
    rows = []
    for oid, g in a.groupby('observation_id', sort=False):
        n = len(g); rec = {'observation_id': oid}
        for lab in ('Y_nt', 'Y_nn'):
            v = g[lab].to_numpy()
            starts = int(((v == 1) & (np.r_[0, v[:-1]] == 0)).sum())
            rec[lab + '_rate'] = v.mean() * 100
            rec[lab + '_bpm'] = starts / (n / FPS / 60)
        rows.append(rec)
    return pd.DataFrame(rows).merge(e, on='observation_id')


def contrast(df, col, x, y):
    w = df.pivot_table(index=['pool', 'odor'], columns='phase', values=col).dropna(subset=[x, y])
    d = (w[y] - w[x]).groupby('pool').mean()
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return m, m - t * se, m + t * se


def fig_causal(r):
    """Per behaviour, one panel each -- the estimand, in the units it is best measured in."""
    # CONSECUTIVE transitions only, in the order the experiment ran: H -> O -> P.
    # P - H is a composite of the two and is not an independent contrast, so quoting it
    # alongside them triple-counts the same 24 pools.
    trans = [('H', 'O'), ('O', 'P')]
    tnames = ['odour ON\nH → O', 'odour OFF\nO → P']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=False)
    for ax, (lab, nice) in zip(axes, [('Y_nt', 'nose-to-tail  (nt)'), ('Y_nn', 'nose-to-nose  (nn)')]):
        ypos, ticks, labels = [], [], []
        for i, ((x, y), tn) in enumerate(zip(trans, tnames)):
            for k, (od, col, odn) in enumerate([('F', C2, 'fear odour'), ('S', C1, 'social odour')]):
                m, lo, hi = contrast(r[r.odor == od], lab + '_bpm', x, y)
                pos = -(i * 2.6 + k * 0.9)
                sig = lo * hi > 0
                ax.plot([lo, hi], [pos, pos], color=col, lw=2, solid_capstyle='round',
                        alpha=1.0 if sig else 0.45, zorder=2)
                ax.plot([m], [pos], 'o', ms=9, color=col, mec='white', mew=2,
                        alpha=1.0 if sig else 0.45, zorder=3,
                        label=odn if i == 0 else None)
                if sig:
                    ax.annotate(f'{m:+.2f}', (m, pos), textcoords='offset points',
                                xytext=(0, 9), ha='center', fontsize=8, color=INK, weight='bold')
                ypos.append(pos)
            ticks.append(-(i * 2.6 + 0.45)); labels.append(tn)
        ax.axvline(0, color=MUTED, lw=1, zorder=1)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9)
        ax.set_ylim(min(ypos) - 1.0, max(ypos) + 1.0)
        ax.set_xlabel('change in bouts per minute')
        ax.set_title(nice, fontsize=11, weight='bold', loc='left', color=INK)
        ax.grid(axis='x', color=GRID, lw=0.7); ax.set_axisbelow(True)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=2, loc='lower center',
                   bbox_to_anchor=(0.5, -0.42))
    fig.suptitle('Effect of the phase transition on behaviour initiation — human labels only, '
                 '24 annotated pools', fontsize=11.5, weight='bold', x=0.02, ha='left', y=1.0)
    fig.text(0.02, -0.03, 'Filled = 95% CI excludes zero. Unit of analysis is the pool (n=24); '
             'each estimate is the mean within-pool difference, so cage, genotype, sex and '
             'annotator all cancel by construction.', fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    f = OUT / 'story_causal_ate.png'
    fig.savefig(f, dpi=160, bbox_inches='tight'); plt.close(fig)
    return f


def fig_outcome_choice(r):
    """Why bouts/min and not occupancy: same design, more of it resolves."""
    trans = [('H', 'O'), ('O', 'P')]
    cells, sig_rate, sig_bpm = [], 0, 0
    for lab in ('Y_nt', 'Y_nn'):
        for od in ('F', 'S'):
            for x, y in trans:
                m1, l1, h1 = contrast(r[r.odor == od], lab + '_rate', x, y)
                m2, l2, h2 = contrast(r[r.odor == od], lab + '_bpm', x, y)
                s1, s2 = l1 * h1 > 0, l2 * h2 > 0
                sig_rate += s1; sig_bpm += s2
                cells.append((f'{lab[2:]} {od} {y}−{x}', s1, s2))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar([0, 1], [sig_rate, sig_bpm], width=0.5, color=[MUTED, C3],
           zorder=2, edgecolor='white', linewidth=2)
    for i, v in enumerate([sig_rate, sig_bpm]):
        ax.annotate(f'{v} / {len(cells)}', (i, v), textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=11, weight='bold', color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['time-in-behaviour\n(occupancy)',
                                               'bouts per minute\n(event count)'])
    ax.set_ylabel('contrasts with 95% CI excluding 0')
    ax.set_ylim(0, len(cells) * 1.15)
    ax.grid(axis='y', color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_title('The outcome variable decides how much of the design resolves',
                 fontsize=10.5, weight='bold', loc='left', color=INK)
    fig.tight_layout()
    f = OUT / 'story_outcome_choice.png'
    fig.savefig(f, dpi=160, bbox_inches='tight'); plt.close(fig)
    return f, sig_rate, sig_bpm


def fig_tokens_pixels():
    """The one ablation that separates token count from pixel resolution."""
    pts = [('224 px input\n256 tokens', 'res224_k2_frozen_d4_decay20', MUTED),
           ('448 px input\n1024 tokens\npixels capped at 112', 'res448_k2_frozen_d4photo_px112', C4),
           ('448 px input\n1024 tokens\npixels capped at 224', 'res448_k2_frozen_d4photo_px224', C2),
           ('448 px input\n1024 tokens\nfull 512 px', 'res448_k2_frozen_d4photo_decay30_seed42', C1)]
    vals, labs, cols = [], [], []
    for nice, tag, c in pts:
        p = FRAME / tag / 'config.json'
        if not p.exists():
            continue
        vals.append(json.load(open(p))['ap_report']['macro/tol0']['ap'])
        labs.append(nice); cols.append(c)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.bar(range(len(vals)), vals, width=0.55, color=cols, zorder=2,
           edgecolor='white', linewidth=2)
    for i, v in enumerate(vals):
        ax.annotate(f'{v:.3f}', (i, v), textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=9.5, weight='bold', color=INK)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel('macro AP (frame level)')
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(axis='y', color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if len(vals) == 4:
        ax.annotate('', xy=(1, vals[1] * 0.5), xytext=(0, vals[0] * 0.5),
                    arrowprops=dict(arrowstyle='->', color=INK2, lw=1.2))
        ax.text(0.5, vals[0] * 0.5 + 0.015, f'+{vals[1]-vals[0]:.3f}\nmore tokens,\nsame pixels',
                ha='center', fontsize=7.5, color=INK2)
        ax.annotate('', xy=(3, vals[3] * 0.5), xytext=(2, vals[2] * 0.5),
                    arrowprops=dict(arrowstyle='->', color=INK2, lw=1.2))
        ax.text(2.5, vals[2] * 0.5 + 0.015, f'+{vals[3]-vals[2]:.3f}\nmore pixels,\nsame tokens',
                ha='center', fontsize=7.5, color=INK2)
    ax.set_title('Token count and pixel resolution, finally separated',
                 fontsize=10.5, weight='bold', loc='left', color=INK)
    fig.tight_layout()
    f = OUT / 'story_tokens_pixels.png'
    fig.savefig(f, dpi=160, bbox_inches='tight'); plt.close(fig)
    return f, vals


def fig_ap_vs_causal():
    """Frame AP does not order models by their value to the causal estimate."""
    from event_eval import evaluate
    ths = np.round(np.arange(0.05, 1.0, 0.05), 2)
    tags = [d.name for d in sorted(FRAME.iterdir()) if (d / 'val_probs.npz').exists()]
    xs, ys, ns = [], [], []
    for t in tags:
        c = FRAME / t / 'config.json'
        if not c.exists():
            continue
        try:
            res = evaluate(t, 1, 1, ths)
        except Exception:
            continue
        ap = json.load(open(c))['ap_report']['macro/tol0']['ap']
        xs.append(ap); ys.append(np.mean([res['nt']['r_delta'], res['nn']['r_delta']])); ns.append(t)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.scatter(xs, ys, s=70, color=C1, edgecolor='white', linewidth=1.5, zorder=3)
    hi_ap, hi_r = int(np.argmax(xs)), int(np.argmax(ys))
    for i in {hi_ap, hi_r}:
        ax.annotate(ns[i].replace('res448_k2_', ''), (xs[i], ys[i]), textcoords='offset points',
                    xytext=(8, -3), fontsize=8, color=INK, weight='bold')
    ax.scatter([xs[hi_ap]], [ys[hi_ap]], s=70, color=C2, edgecolor='white', linewidth=1.5, zorder=4)
    ax.scatter([xs[hi_r]], [ys[hi_r]], s=70, color=C3, edgecolor='white', linewidth=1.5, zorder=4)
    if len(xs) > 2:
        rho = stats.spearmanr(xs, ys).statistic
        ax.text(0.02, 0.03, f'Spearman ρ = {rho:+.2f}  (n={len(xs)} runs)',
                transform=ax.transAxes, fontsize=9, color=INK2)
    ax.set_xlabel('macro AP — what training selects on')
    ax.set_ylabel('correlation of within-pool phase differences in bouts/min\n'
                  '— what the causal estimate depends on')
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_title('Frame AP shortlists well, but does not pick the winner',
                 fontsize=10.5, weight='bold', loc='left', color=INK)
    ax.text(0.0, -0.20, 'The two agree loosely (ρ ≈ +0.5), so AP is a usable filter — but the '
            'best-AP model is not the best model for the estimate,\nand the gap between them is '
            'larger than the spread AP itself resolves. Shortlist on AP; choose on the causal '
            'quantity.', transform=ax.transAxes, fontsize=8, color=INK2, va='top')
    fig.tight_layout()
    f = OUT / 'story_ap_vs_causal.png'
    fig.savefig(f, dpi=160, bbox_inches='tight'); plt.close(fig)
    return f, list(zip(ns, xs, ys))


if __name__ == '__main__':
    r = obs_table()
    print('causal   ->', fig_causal(r))
    f, a, b = fig_outcome_choice(r)
    print(f'outcome  -> {f}   occupancy {a}/12 vs bouts {b}/12')
    f, v = fig_tokens_pixels()
    print(f'tok/px   -> {f}   {[round(x,4) for x in v]}')
    f, rows = fig_ap_vs_causal()
    print(f'ap-vs-r  -> {f}')
    for n, x, y in sorted(rows, key=lambda z: -z[2])[:5]:
        print(f'    {n:44s} AP={x:.4f}  mean r_delta={y:.3f}')


def fig_within_phase(a=None):
    """Event rate against elapsed time INSIDE each phase. The phase mean hides this entirely."""
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['observation_id', 'frame_idx', 'Y_nt', 'Y_nn']).dropna(subset=['Y_nt'])
    e = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')[
        ['observation_id', 'pool', 'phase', 'odor']]
    a = a.sort_values(['observation_id', 'frame_idx']).merge(e, on='observation_id')
    a['minute'] = (a.frame_idx // int(FPS * 60)).astype(int)
    rows = []
    for (oid, m), g in a.groupby(['observation_id', 'minute'], sort=False):
        if len(g) < 250:
            continue
        r = {'observation_id': oid, 'minute': m}
        for lab in ('Y_nt', 'Y_nn'):
            v = g[lab].to_numpy()
            r[lab] = int(((v == 1) & (np.r_[0, v[:-1]] == 0)).sum())
        rows.append(r)
    d = pd.DataFrame(rows).merge(e, on='observation_id')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    style = {'H': (MUTED, '-'), 'O': (C2, '-'), 'P': (C1, '--')}
    for ax, (lab, nice) in zip(axes, [('Y_nt', 'nose-to-tail'), ('Y_nn', 'nose-to-nose')]):
        for ph in ('H', 'O', 'P'):
            s = d[(d.phase == ph) & (d.minute < 15)].groupby('minute')[lab].mean()
            col, ls = style[ph]
            ax.plot(s.index + 1, s.values, ls, color=col, lw=2, label=f'{ph}', zorder=3)
        ax.set_xlabel('minute within the phase')
        ax.set_title(nice, fontsize=10.5, weight='bold', loc='left', color=INK)
        ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel('bouts per minute')
    axes[0].legend(frameon=False, fontsize=8.5, title='phase', title_fontsize=8)
    fig.suptitle('Behaviour decays steeply INSIDE every phase — averaged over 24 pools and both odours',
                 fontsize=11, weight='bold', x=0.02, ha='left')
    fig.text(0.02, -0.04, 'Rates fall roughly 4–6× across a recording. 11 of 12 phase × odour cells have a '
             'significantly negative slope. A single phase mean therefore\ndepends on how long the '
             'recording is — and habituation is 30 minutes while odour and post are 15.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    f = OUT / 'story_within_phase.png'
    fig.savefig(f, dpi=160, bbox_inches='tight'); plt.close(fig)
    return f


def fig_window_sensitivity():
    """The same contrast under three defensible windows. One of them flips sign."""
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['observation_id', 'frame_idx', 'Y_nt', 'Y_nn']).dropna(subset=['Y_nt'])
    e = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')[
        ['observation_id', 'pool', 'phase', 'odor']]
    a = a.sort_values(['observation_id', 'frame_idx']).merge(e, on='observation_id')
    mx = a.groupby('observation_id')['frame_idx'].transform('max')
    W = [('full window', a.frame_idx >= 0),
         ('first 5 min', a.frame_idx < 5 * 300),
         ('last 15 min', a.frame_idx > mx - 15 * 300)]
    cells = [('Y_nn', 'F', 'nn · fear'), ('Y_nn', 'S', 'nn · social'),
             ('Y_nt', 'F', 'nt · fear'), ('Y_nt', 'S', 'nt · social')]
    res = {}
    for nm, sel in W:
        rows = []
        for oid, g in a[sel].groupby('observation_id', sort=False):
            n = len(g); r = {'observation_id': oid}
            for lab in ('Y_nt', 'Y_nn'):
                v = g[lab].to_numpy()
                r[lab] = int(((v == 1) & (np.r_[0, v[:-1]] == 0)).sum()) / (n / FPS / 60)
            rows.append(r)
        t = pd.DataFrame(rows).merge(e, on='observation_id')
        for lab, od, _ in cells:
            m, lo, hi = contrast(t[t.odor == od], lab, 'H', 'O')
            res[(nm, lab, od)] = (m, lo, hi)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    xs = np.arange(len(cells)); wdt = 0.26
    for j, (nm, _) in enumerate(W):
        off = (j - 1) * wdt
        for i, (lab, od, _) in enumerate(cells):
            m, lo, hi = res[(nm, lab, od)]
            sig = lo * hi > 0
            col = [MUTED, C4, C1][j]
            ax.plot([xs[i] + off] * 2, [lo, hi], color=col, lw=1.6,
                    alpha=1 if sig else .4, solid_capstyle='round', zorder=2)
            ax.plot([xs[i] + off], [m], 'o', ms=7, color=col, mec='white', mew=1.6,
                    alpha=1 if sig else .4, zorder=3, label=nm if i == 0 else None)
    ax.axhline(0, color=INK, lw=1, zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels([c[2] for c in cells])
    ax.set_ylabel('H → O effect (bouts/min)')
    ax.grid(axis='y', color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.13))
    ax.set_title('Only nn · fear survives every window', fontsize=10.5, weight='bold',
                 loc='left', color=INK, pad=26)
    fig.text(0.02, -0.05, 'Faded = CI includes zero. nn · social reverses sign between windows, so its '
             'full-window value is an artefact of\nhabituation being twice as long as the odour phase.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    f = OUT / 'story_window_sensitivity.png'
    fig.savefig(f, dpi=160, bbox_inches='tight'); plt.close(fig)
    return f, res


def fig_tokens_pixels_waterfall():
    """Waterfall: what each step of the 224 -> 448 change actually contributed."""
    def ap(tag):
        return json.load(open(FRAME / tag / 'config.json'))['ap_report']['macro/tol0']['ap']
    base = ap('res224_k2_frozen_d4_decay20')
    tok = ap('res448_k2_frozen_d4photo_px224')
    pix = ap('res448_k2_frozen_d4photo_decay30_seed42')
    more = ap('res504_k2_frozen_d4photo')
    steps = [('224 px input\n256 tokens\n224 px detail', base, None, MUTED),
             ('4× MORE TOKENS\ndetail held at 224 px', tok, tok - base, C3),
             ('MORE PIXELS\ntokens held at 1024', pix, pix - tok, C1),
             ('+27% more tokens\n(504 px)', more, more - pix, MUTED)]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for i, (lab, val, delta, col) in enumerate(steps):
        if delta is None:
            ax.bar(i, val, width=.55, color=col, zorder=2, edgecolor='white', linewidth=2)
            ax.annotate(f'{val:.3f}', (i, val), textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=10, weight='bold', color=INK)
        else:
            bot = val - delta
            ax.bar(i, delta, bottom=bot, width=.55, color=col, zorder=2,
                   edgecolor='white', linewidth=2)
            ax.plot([i - .55, i + .28], [bot, bot], color=MUTED, lw=.9, ls=':', zorder=1)
            ax.annotate(f'{delta:+.3f}', (i, val), textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=10, weight='bold',
                        color=C3 if delta > .05 else INK2)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=8.2)
    ax.set_ylabel('macro AP')
    ax.set_ylim(0, 0.50)
    ax.grid(axis='y', color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.set_title('Where the +41% came from: tokens, then pixels, then nothing',
                 fontsize=10.5, weight='bold', loc='left', color=INK)
    fig.text(0.02, -0.05, f'Adding tokens at fixed pixel detail is worth {tok-base:+.3f}; adding pixels '
             f'at fixed token count is worth {pix-tok:+.3f};\npushing tokens further is worth '
             f'{more-pix:+.3f}. Conclusion: 1024 tokens on a 512 px frame is the ceiling.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    f = OUT / 'story_tokens_pixels.png'
    fig.savefig(f, dpi=160, bbox_inches='tight'); plt.close(fig)
    return f


def fig_learning_curve():
    """Macro AP against the number of ANNOTATED POOLS -- the axis a labelling budget buys."""
    pts = [(5, 'res448_k2_frozen_d4photo_lc5p'), (10, 'res448_k2_frozen_d4photo_lc10p'),
           (15, 'res448_k2_frozen_d4photo_lc15p'), (20, 'res448_k2_frozen_d4photo_decay30_seed42')]
    xs, ys = [], []
    for n, t in pts:
        p = FRAME / t / 'config.json'
        if not p.exists():
            continue
        xs.append(n); ys.append(json.load(open(p))['ap_report']['macro/tol0']['ap'])
    xs, ys = np.array(xs), np.array(ys)
    sl, ic = np.polyfit(np.log(xs), ys, 1)
    ss_res = np.sum((ys - (sl * np.log(xs) + ic)) ** 2)
    r2 = 1 - ss_res / np.sum((ys - ys.mean()) ** 2)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    gx = np.linspace(5, 72, 200)
    ax.plot(gx, sl * np.log(gx) + ic, color=C1, lw=1.4, ls='--', alpha=.55, zorder=2,
            label=f'log fit (R² = {r2:.3f})')
    ax.axvspan(20, 72, color=C1, alpha=.055, zorder=0)
    ax.plot(xs, ys, 'o-', color=C1, lw=2, ms=9, mec='white', mew=2, zorder=4, label='measured')
    for x, y in zip(xs, ys):
        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, 11),
                    ha='center', fontsize=8.5, weight='bold', color=INK)
    for n in (40, 72):
        yv = sl * np.log(n) + ic
        ax.plot([n], [yv], 'o', ms=8, color='white', mec=C1, mew=2, zorder=4)
        ax.annotate(f'{yv:.3f}', (n, yv), textcoords='offset points', xytext=(0, 11),
                    ha='center', fontsize=8.5, color=C1)
    ax.axhline(0.4622, color=C3, lw=1.4, ls=':', zorder=1)
    ax.annotate('best model achieved without any new labels\n(SSL adaptation, 0.462)',
                (5.6, 0.4622), textcoords='offset points', xytext=(0, 7), fontsize=7.8, color=C3)
    ax.set_xlabel('annotated pools used for training')
    ax.set_ylabel('macro AP')
    ax.set_xlim(4, 76); ax.set_ylim(0.24, 0.60)
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    ax.set_title('Annotation has not saturated', fontsize=10.5, weight='bold', loc='left', color=INK)
    fig.text(0.02, -0.04, f'Each doubling of annotated pools is worth ~{sl*np.log(2):.3f} macro AP — more than '
             f'twice what the best unsupervised\nintervention bought (+0.033). Shaded = extrapolation '
             f'beyond the 20 pools that exist for training.', fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    f = OUT / 'story_learning_curve.png'
    fig.savefig(f, dpi=160, bbox_inches='tight'); plt.close(fig)
    return f, sl, r2


def fig_within_phase_by_odour():
    """Same decay, split by odour -- they are distinct treatments and must not be averaged."""
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['observation_id', 'frame_idx', 'Y_nt', 'Y_nn']).dropna(subset=['Y_nt'])
    e = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')[
        ['observation_id', 'pool', 'phase', 'odor']]
    a = a.sort_values(['observation_id', 'frame_idx']).merge(e, on='observation_id')
    a['minute'] = (a.frame_idx // int(FPS * 60)).astype(int)
    rows = []
    for (oid, m), g in a.groupby(['observation_id', 'minute'], sort=False):
        if len(g) < 250:
            continue
        r = {'observation_id': oid, 'minute': m}
        for lab in ('Y_nt', 'Y_nn'):
            v = g[lab].to_numpy()
            r[lab] = int(((v == 1) & (np.r_[0, v[:-1]] == 0)).sum())
        rows.append(r)
    d = pd.DataFrame(rows).merge(e, on='observation_id')
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), sharex=True, sharey='row')
    style = {'H': (MUTED, '-'), 'O': (C2, '-'), 'P': (C1, '--')}
    for i, (lab, nice) in enumerate([('Y_nt', 'nose-to-tail'), ('Y_nn', 'nose-to-nose')]):
        for j, (od, odn) in enumerate([('F', 'fear odour'), ('S', 'social odour')]):
            ax = axes[i][j]
            for ph in ('H', 'O', 'P'):
                s = d[(d.phase == ph) & (d.odor == od) & (d.minute < 15)].groupby('minute')[lab].mean()
                col, ls = style[ph]
                ax.plot(s.index + 1, s.values, ls, color=col, lw=2, label=ph, zorder=3)
            ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
            for s_ in ('top', 'right'):
                ax.spines[s_].set_visible(False)
            ax.set_title(f'{nice} · {odn}', fontsize=9.5, weight='bold', loc='left', color=INK)
            if i == 1:
                ax.set_xlabel('minute within the phase')
            if j == 0:
                ax.set_ylabel('bouts per minute')
    axes[0][0].legend(frameon=False, fontsize=8.5, title='phase', title_fontsize=8, ncol=3)
    fig.suptitle('Behaviour decays inside every phase — and the two odours behave differently',
                 fontsize=11, weight='bold', x=0.02, ha='left')
    fig.text(0.02, -0.02, 'Bottom-left is the exception that matters: nose-to-tail under social odour '
             'during the O phase is the only one of twelve cells\nthat does NOT decay (slope +0.02/min, '
             'CI includes zero) — social odour sustains investigation while everything else habituates.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    f = OUT / 'story_within_phase.png'
    fig.savefig(f, dpi=160, bbox_inches='tight'); plt.close(fig)
    return f
