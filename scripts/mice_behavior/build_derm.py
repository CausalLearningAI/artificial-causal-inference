#!/usr/bin/env python3
"""Why DERM does not beat ERM here, measured on the axis that decides -- one JSON out.

WHAT QUESTION THIS ANSWERS
==========================
The status report's DERM section previously said "costs a little AP, buys nothing" and left the
mechanism as an inference: "whatever route DERM closes, this model was not using it enough for the
closing to show". That is an argument from a null on the WRONG metric. AP cannot see what DERM
targets, and neither can r-delta on a 4-pool split. This script measures the thing itself.

THE ONLY PART OF MODEL ERROR THAT REACHES THE ESTIMAND
======================================================
The estimand is a WITHIN-POOL difference between two phases. Write the model's expected output in
phase p as

    E[f | p] = a_p + b * E[Y | p]

Then the plug-in (PPCI) target is

    E[D_f] = b * E[D_Y] + (a_O - a_H)

  * `b` is a SCALE. One free parameter, absorbed by PPI++'s lambda and declined outright by
    uncalibrated PPCI, which reports sign and pattern only. Harmless.
  * `a_O - a_H` is a BIAS IN THE ESTIMAND. It is non-zero exactly when the model's error moves
    WITH the phase, and it is the only term that can flip a sign or manufacture an effect.

So the question "does DERM help" is the question "does DERM shrink a_O - a_H", and nothing else.

THE MEASUREMENT: A LEAK AUC, NOT A BIAS IN UNITS
================================================
Measuring a_p directly in bouts per minute confounds it with the global scale and with the
threshold each run happens to pick. DERM also raises the overall output level (it upweights
positives in low-prevalence environments), so any absolute comparison flatters ERM by construction.

The scale-free version is a rank statistic. Among frames with the SAME ground truth, ask how well
the model's own output separates one phase from another:

    leak(p, q | y) = AUC( f on {Y = y, phase q} vs f on {Y = y, phase p} )

0.5 means the output carries no phase information beyond the behaviour. Any monotone rescaling of
f -- including the level shift DERM introduces -- leaves it untouched. Bootstrapped over the four
validation POOLS, because frames within a recording are anything but independent.

WHAT IT FINDS, AND WHY IT IS NOT A NULL
=======================================
DERM does not shrink the leak. It ENLARGES it, in the direction its own weights predict.

DERM's weights are w(y=1,e) = (1-p_e)/P(e) and w(y=0,e) = p_e/P(e). The 1/P(e) cancels in the
ratio, so the only thing DERM does to environment e's operating point is shift it by the prior
odds:

    w(y=0,e) / w(y=1,e) = p_e / (1 - p_e)

A high-prevalence environment therefore has its NEGATIVES upweighted -- it is pushed toward
predicting negative -- and a low-prevalence one is pushed toward positive. That is the intended
deconfounding: divide out the prior. But here the environments ARE the phases, so the shift DERM
installs is itself a function of the treatment. This script tests the resulting prediction:

    the more DERM's weights push phase q toward negative relative to phase p,
    the LOWER q's held-out scores should sit relative to p's, at fixed truth.

i.e. ΔAUC (DERM minus ERM) should run OPPOSITE to log[ odds(p_q) / odds(p_p) ], the log odds
ratio computed from the TRAINING pools. Two environment definitions give two tests:

    env = the 3 phases                -> 4 predictions (behaviour x transition)
    env = the 6 phase x exposure cells -> 8 predictions (behaviour x exposure x transition)

The second is the sharper test: every cell gets its own odds ratio, so the prediction varies
across all eight points rather than four.

THE SECOND CHANNEL: NUISANCE-LINKED BIAS, WHICH IS WHAT PPI++ WOULD ACTUALLY NEED HELP WITH
==========================================================================================
PPI++ is algebraically unbiased for any predictor, so a treatment-linked bias costs it variance
rather than validity. The one thing that CAN break its validity on v1 is that the 24 labelled
pools are not a random sample -- annotation is 3:1 het-enriched -- so the rectifier measured there
has to transport to 48 wt-enriched pools. That needs the model's bias not to depend on genotype.

So the same decomposition is run on the DEPLOYED cross-fitted predictions over all 24 annotated
pools (144 observations, a real denominator rather than four pools), asking how much of the model's
bias is explained by each pool-level factor, at the LEVEL and in the WITHIN-POOL DIFFERENCE. This
is the analogue for MODEL bias of the annotator decomposition the report already runs on LABEL
noise, and it is what decides whether DERM on nuisance environments has a job.

WHICH ENVIRONMENTS ARE SAFE, AND WHY IT IS STRUCTURAL
====================================================
DERM's correction is a per-environment shift of the decision logit. Whether that shift is free or
fatal is decided by one property of the environment variable, which this script measures rather
than assumes (`pool_constant`):

  varies WITHIN a pool   (phase, exposure) -- the shift differs between the two sides of the
                         contrast, so it lands in the estimand. Guaranteed to bias.
  constant WITHIN a pool (line, sex, genotype, cage, and annotator for 22 of 24 pools) -- the
                         shift is identical on both sides of every within-pool difference and
                         cancels. Free.

WHAT IS *NOT* CLAIMED
=====================
The leak comparison rests on 4 validation pools. The correlations below carry n = 4 and n = 8 and
are reported with their p-values, which are not small. The claim is a DIRECTION with a mechanism
behind it, not an effect size. The experiment that would settle it is cross-fitted DERM over all
24 annotated pools -- which also happens to be the only way to compute a real PPI++ interval under
either objective.

    python scripts/mice_behavior/build_derm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from event_eval import runs, postprocess                                    # noqa: E402

FRAME = ROOT / 'results' / 'vision' / 'mice' / 'frame'
OUT = FRAME / '_figures'
FPS = 5.0
NBIN = 2048
TRANS = (('H', 'O'), ('O', 'P'))
LABELS = ('nt', 'nn')

# arm -> (tag, family, seed). The two ERM controls are the matched baseline: same head, same
# augmentation, same split, same schedule; only --derm and --env-key differ.
ARMS = [
    ('ERM',            'res448_k2_frozen_d4photo_ermH5M',      'erm',  42),
    ('ERM',            'res448_k2_frozen_d4photo_ermH5M_s1',    'erm',   1),
    ('DERM · phases',  'res448_k2_frozen_d4photo_dermPhase',   'derm', 42),
    ('DERM · phases',  'res448_k2_frozen_d4photo_dermPhase_s1', 'derm',  1),
    ('DERM · cells',   'res448_k2_frozen_d4photo_dermCond',    'cond', 42),
    # the BitFit arms, once they land -- the base model whose shortcut is worth closing
    ('BitFit ERM',      'res448_k2_bit6_d4',                    'bit_erm',  42),
    ('BitFit ERM',      'res448_k2_bit6_d4_seed1',              'bit_erm',   1),
    ('BitFit DERM',     'res448_k2_bit6_d4_dermPhase',          'bit_derm', 42),
    ('BitFit DERM',     'res448_k2_bit6_d4_dermPhase_s1',       'bit_derm',  1),
]


# ------------------------------------------------------------------ training-set prevalence
def prevalence(val_pools: set[str]) -> dict:
    """p_e on the TRAINING pools, for both environment definitions DERM was run with.

    Two versions, because the training loop resamples negatives at 1:1 every epoch and the table
    is rebuilt on that sample:

      raw       every annotated frame of the training pools
      sampled   all any-label positives kept, negatives thinned uniformly by rho = n_pos / n_neg,
                which is what `neg_ratio=1` does. Negatives are drawn GLOBALLY, not per
                environment, so rho is one number.

    Only the ORDERING of p_e across environments enters the prediction, and the two versions are
    reported side by side so a reader can see whether it depends on the sampling detail.
    """
    exp = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['observation_id', 'frame_idx', 'Y_nt', 'Y_nn'],
                    low_memory=False).dropna(subset=['Y_nt'])
    a = a.merge(exp[['observation_id', 'pool', 'phase', 'odor']], on='observation_id')
    tr = a[~a.pool.isin(val_pools)].copy()
    tr['anypos'] = (tr.Y_nt > 0.5) | (tr.Y_nn > 0.5)
    rho = min(1.0, tr.anypos.sum() / max((~tr.anypos).sum(), 1))

    def table(keys):
        out = {}
        for k, g in tr.groupby(keys, sort=True):
            pos = g.anypos.sum()
            den = pos + rho * (len(g) - pos)
            key = k if isinstance(k, str) else '·'.join(k)
            out[key] = {l: {'raw': float((g['Y_' + l] > 0.5).mean()),
                            'sampled': float((g['Y_' + l] > 0.5).sum() / den)} for l in LABELS}
        return out

    return {'rho': float(rho), 'n_train_pools': int(tr.pool.nunique()),
            'n_train_frames': int(len(tr)),
            'phase': table('phase'), 'cond': table(['phase', 'odor'])}


def log_or(p_hi: float, p_lo: float) -> float:
    """log of the odds ratio. This IS DERM's per-environment logit shift, up to sign."""
    o = lambda p: p / (1 - p)
    return float(np.log(o(p_hi) / o(p_lo)))


# ------------------------------------------------------------------ the leak AUC
def hist_auc(ha: np.ndarray, hb: np.ndarray) -> float:
    """AUC(a > b) with half credit for ties, from two histograms over shared bins."""
    na, nb = ha.sum(), hb.sum()
    if na == 0 or nb == 0:
        return float('nan')
    below = np.concatenate([[0.0], np.cumsum(hb)[:-1]])
    return float((ha * (below + 0.5 * hb)).sum() / (na * nb))


def leak_hists(tag: str, exp: pd.DataFrame):
    """probs binned per (pool, odour, phase, behaviour, truth) -- everything the AUCs need."""
    d = np.load(FRAME / tag / 'val_probs.npz', allow_pickle=True)
    df = pd.DataFrame({'obs': d['obs'], 'p_nt': d['probs'][:, 0], 'p_nn': d['probs'][:, 1],
                       'y_nt': d['labels'][:, 0], 'y_nn': d['labels'][:, 1]})
    df = df.merge(exp, left_on='obs', right_on='observation_id')
    H = {}
    for (pool, od, ph), g in df.groupby(['pool', 'odor', 'phase'], sort=False):
        for l in LABELS:
            b = np.clip((g['p_' + l].to_numpy() * NBIN).astype(int), 0, NBIN - 1)
            y = g['y_' + l].to_numpy() > 0.5
            for cls, m in ((0, ~y), (1, y)):
                H[(pool, od, ph, l, cls)] = np.bincount(b[m], minlength=NBIN).astype(np.float64)
    return H, sorted(df.pool.unique())


def auc_ci(H, pools, sel, x, y, reps=400, seed=0):
    """AUC(phase y vs phase x) and a 95% interval bootstrapped over POOLS, not frames."""
    def acc(ps, ph):
        t = np.zeros(NBIN)
        for p in ps:
            for od, l, cls in sel:
                h = H.get((p, od, ph, l, cls))
                if h is not None:
                    t += h
        return t
    point = hist_auc(acc(pools, y), acc(pools, x))
    rng = np.random.default_rng(seed)
    bs = [hist_auc(acc(s, y), acc(s, x)) for s in
          (list(np.asarray(pools)[rng.integers(0, len(pools), len(pools))]) for _ in range(reps))]
    bs = np.array([v for v in bs if np.isfinite(v)])
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > reps // 4 else (np.nan, np.nan))
    n = int(sum(H[(p, od, y, l, cls)].sum() for p in pools for od, l, cls in sel
                if (p, od, y, l, cls) in H))
    return point, float(lo), float(hi), n


# ------------------------------------------------------------------ the estimand-level bias
def match_threshold(tag: str, l: str, exp: pd.DataFrame):
    """The one threshold whose TOTAL predicted bout count matches the total true count.

    Fixing the global rate spends the single degree of freedom the estimand allows (the scale b),
    so whatever phase-dependence is left cannot be explained away as calibration. Reported next to
    each run's own best-F1 threshold rather than instead of it.
    """
    d = np.load(FRAME / tag / 'val_probs.npz', allow_pickle=True)
    j = LABELS.index(l)
    df = pd.DataFrame({'obs': d['obs'], 'p': d['probs'][:, j], 'y': d['labels'][:, j]})
    gs = [g for _, g in df.groupby('obs', sort=False)]
    T = sum(len(runs(g['y'].to_numpy() > 0.5)) for g in gs)
    best = (np.nan, np.inf)
    for th in np.round(np.arange(0.05, 1.0, 0.01), 2):
        P = sum(len(runs(postprocess(g['p'].to_numpy() >= th, 1, 1))) for g in gs)
        if abs(P - T) < best[1]:
            best = (float(th), abs(P - T))
    th = best[0]
    rows = []
    for oid, g in df.groupby('obs', sort=False):
        mins = len(g) / FPS / 60
        rows.append({'observation_id': oid,
                     'true': len(runs(g['y'].to_numpy() > 0.5)) / mins,
                     'pred': len(runs(postprocess(g['p'].to_numpy() >= th, 1, 1))) / mins})
    po = pd.DataFrame(rows).merge(exp, on='observation_id')
    ph = po.groupby('phase')[['true', 'pred']].mean()
    b = {p: float(ph.loc[p, 'pred'] - ph.loc[p, 'true']) for p in 'HOP'}
    return th, b


# ------------------------------------------------------------------ nuisance-linked model bias
def pool_constant(exp: pd.DataFrame, annotated: pd.DataFrame) -> dict:
    """For each candidate environment variable: is it constant within a pool?

    Measured on the 24 ANNOTATED pools, because those are the ones a DERM run trains on. The
    answer is what decides whether DERM's per-environment logit shift cancels in a within-pool
    contrast or lands in the estimand, so it is counted rather than assumed.
    """
    out = {}
    for c in ('line', 'sex', 'genotype', 'annotator', 'date', 'odor', 'phase'):
        n = annotated.groupby('pool')[c].nunique(dropna=False)
        out[c] = {'constant_pools': int((n <= 1).sum()), 'n_pools': int(len(n)),
                  'max_per_pool': int(n.max())}
    return out


def nuisance_bias(exp_full: pd.DataFrame) -> dict:
    """How much of the DEPLOYED model's bias is explained by each pool-level factor.

    Two quantities, and the contrast between them is the point:

      LEVEL    per-pool mean of (predicted - true) bouts/min. A factor that moves this moves the
               model's calibration, which matters for transporting a PPI++ rectifier from a
               non-randomly annotated subset.
      DELTA    per-pool (bias in O - bias in H). A factor that does NOT move this cannot bias the
               estimand, because the estimand is exactly that difference.

    One-way ANOVA with the factor as the grouping, eta^2 as the share of between-pool variance it
    explains. Out-of-fold predictions from the three deployment folds, so every pool is scored by
    a model that never saw it.
    """
    from build_estimates import labelled_truth, out_of_fold_predictions
    oof, _, _ = out_of_fold_predictions()
    d = (labelled_truth().merge(oof, on='observation_id')
         .merge(exp_full[['observation_id', 'pool', 'phase', 'odor', 'annotator',
                          'genotype', 'line']], on='observation_id'))
    out = {'n_obs': int(len(d)), 'n_pools': int(d.pool.nunique()), 'level': {}, 'delta': {}}

    def anova(frame, val, fac):
        g = [v[val].to_numpy() for _, v in frame.groupby(fac, dropna=False) if len(v) > 1]
        if len(g) < 2:
            return None
        gm = np.concatenate(g).mean()
        ss_b = sum(len(x) * (x.mean() - gm) ** 2 for x in g)
        ss_t = ((np.concatenate(g) - gm) ** 2).sum()
        return {'eta2': round(float(ss_b / ss_t), 4) if ss_t > 0 else None,
                'p': round(float(stats.f_oneway(*g).pvalue), 4), 'n_groups': len(g)}

    for l in LABELS:
        d['bias'] = d[f'f_events_{l}'] - d[f't_events_{l}']
        lvl = d.groupby(['pool', 'annotator', 'genotype', 'line'],
                        dropna=False, as_index=False).bias.mean()
        out['level'][l] = {f: anova(lvl, 'bias', f) for f in ('annotator', 'genotype', 'line')}
        rows = []
        for (pool, od), g in d.groupby(['pool', 'odor']):
            m = g.drop_duplicates('phase').set_index('phase')
            if not {'H', 'O'} <= set(m.index):
                continue
            rows.append({'db': m.loc['O', 'bias'] - m.loc['H', 'bias'],
                         'annotator': m.loc['H', 'annotator'],
                         'genotype': m.loc['H', 'genotype'], 'line': m.loc['H', 'line']})
        r = pd.DataFrame(rows)
        out['delta'][l] = {f: anova(r, 'db', f) for f in ('annotator', 'genotype', 'line')}
    return out


def main():
    exp = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')[
        ['observation_id', 'pool', 'phase', 'odor']]
    exp_full = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')
    cfg = json.load(open(FRAME / 'res448_k2_frozen_d4photo_dermPhase' / 'config.json'))
    val = set(cfg['val_pools'])
    prev = prevalence(val)
    print(f"training prevalence on {prev['n_train_pools']} pools "
          f"({prev['n_train_frames']:,} frames), rho = {prev['rho']:.4f}")
    for ph in 'HOP':
        r = prev['phase'][ph]
        print(f"  {ph}   nt {r['nt']['raw']*100:.3f}% raw / {r['nt']['sampled']*100:.1f}% sampled"
              f"   nn {r['nn']['raw']*100:.3f}% raw / {r['nn']['sampled']*100:.1f}% sampled")

    present = [(nice, tag, fam, sd) for nice, tag, fam, sd in ARMS
               if (FRAME / tag / 'val_probs.npz').exists()]
    absent = [tag for _, tag, _, _ in ARMS if (FRAME / tag / 'val_probs.npz').exists() is False]
    if absent:
        print('  not landed yet: ' + ', '.join(absent))

    # ---- leak AUCs, per arm, both groupings -------------------------------------------------
    leak = []
    for nice, tag, fam, sd in present:
        H, pools = leak_hists(tag, exp)
        for l in LABELS:
            for cls in (0, 1):
                for x, y in TRANS:                                    # pooled over exposure
                    a, lo, hi, n = auc_ci(H, pools, [(od, l, cls) for od in ('F', 'S')], x, y)
                    leak.append({'arm': nice, 'tag': tag, 'family': fam, 'seed': sd,
                                 'behav': l, 'odour': 'both', 'trans': f'{x}->{y}',
                                 'truth': cls, 'auc': round(a, 4),
                                 'lo': round(lo, 4), 'hi': round(hi, 4), 'n': n})
                for od in ('F', 'S'):                                 # per exposure cell
                    for x, y in TRANS:
                        a, lo, hi, n = auc_ci(H, pools, [(od, l, cls)], x, y)
                        leak.append({'arm': nice, 'tag': tag, 'family': fam, 'seed': sd,
                                     'behav': l, 'odour': od, 'trans': f'{x}->{y}',
                                     'truth': cls, 'auc': round(a, 4),
                                     'lo': round(lo, 4), 'hi': round(hi, 4), 'n': n})
        print(f'  leak done: {tag}')

    L = pd.DataFrame(leak)

    def mean_auc(fam, behav, odour, trans, cls=0):
        s = L[(L.family == fam) & (L.behav == behav) & (L.odour == odour)
              & (L.trans == trans) & (L.truth == cls)]
        return float(s.auc.mean()) if len(s) else float('nan')

    # ---- test 1: environments = the 3 phases ------------------------------------------------
    summ_phase = []
    for l in LABELS:
        for x, y in TRANS:
            lor = {k: log_or(prev['phase'][y][l][k], prev['phase'][x][l][k])
                   for k in ('raw', 'sampled')}
            e, d = mean_auc('erm', l, 'both', f'{x}->{y}'), mean_auc('derm', l, 'both', f'{x}->{y}')
            summ_phase.append({'behav': l, 'trans': f'{x}->{y}',
                               'log_or': round(lor['raw'], 4),
                               'log_or_sampled': round(lor['sampled'], 4),
                               'erm': round(e, 4), 'derm': round(d, 4),
                               'delta': round(d - e, 4),
                               'agree': bool((d - e) * lor['raw'] < 0)})

    # ---- test 2: environments = the 6 phase x exposure cells --------------------------------
    summ_cond = []
    for l in LABELS:
        for od in ('F', 'S'):
            for x, y in TRANS:
                kx, ky = f'{x}·{od}', f'{y}·{od}'
                lor = {k: log_or(prev['cond'][ky][l][k], prev['cond'][kx][l][k])
                       for k in ('raw', 'sampled')}
                e = mean_auc('erm', l, od, f'{x}->{y}')
                c = mean_auc('cond', l, od, f'{x}->{y}')
                summ_cond.append({'behav': l, 'odour': od, 'trans': f'{x}->{y}',
                                  'log_or': round(lor['raw'], 4),
                                  'log_or_sampled': round(lor['sampled'], 4),
                                  'erm': round(e, 4), 'derm': round(c, 4),
                                  'delta': round(c - e, 4),
                                  'agree': bool((c - e) * lor['raw'] < 0)})

    def corr(rows, key='log_or'):
        x = np.array([r[key] for r in rows]); y = np.array([r['delta'] for r in rows])
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return {'r': None, 'p': None, 'n': int(m.sum())}
        rr = stats.pearsonr(x[m], y[m])
        return {'r': round(float(rr.statistic), 3), 'p': round(float(rr.pvalue), 4),
                'n': int(m.sum()),
                'agree': int(sum(1 for a, b in zip(x[m], y[m]) if a * b < 0))}

    corrs = {'phase': corr(summ_phase), 'phase_sampled': corr(summ_phase, 'log_or_sampled'),
             'cond': corr(summ_cond), 'cond_sampled': corr(summ_cond, 'log_or_sampled')}

    print('\nenv = 3 phases   (prediction: delta runs OPPOSITE to log OR)')
    for r in summ_phase:
        print(f"  {r['behav']} {r['trans']:6s} logOR {r['log_or']:+.3f}  ERM {r['erm']:.3f} -> "
              f"DERM {r['derm']:.3f}   delta {r['delta']:+.3f}  {'OK' if r['agree'] else '--'}")
    print(f"  Pearson r = {corrs['phase']['r']} (p = {corrs['phase']['p']}, "
          f"n = {corrs['phase']['n']}), signs agree {corrs['phase']['agree']}/"
          f"{corrs['phase']['n']}")
    print('\nenv = 6 phase x exposure cells')
    for r in summ_cond:
        print(f"  {r['behav']} {r['odour']} {r['trans']:6s} logOR {r['log_or']:+.3f}  "
              f"ERM {r['erm']:.3f} -> DERM {r['derm']:.3f}   delta {r['delta']:+.3f}  "
              f"{'OK' if r['agree'] else '--'}")
    print(f"  Pearson r = {corrs['cond']['r']} (p = {corrs['cond']['p']}, "
          f"n = {corrs['cond']['n']}), signs agree {corrs['cond']['agree']}/{corrs['cond']['n']}")

    # ---- the estimand-level bias, in the unit the report estimates on -----------------------
    est = []
    for nice, tag, fam, sd in present:
        for l in LABELS:
            th, b = match_threshold(tag, l, exp)
            est.append({'arm': nice, 'tag': tag, 'family': fam, 'seed': sd, 'behav': l,
                        'thr': th, 'b_H': round(b['H'], 4), 'b_O': round(b['O'], 4),
                        'b_P': round(b['P'], 4),
                        'b_HO': round(b['O'] - b['H'], 4), 'b_OP': round(b['P'] - b['O'], 4)})
        print(f'  bias done: {tag}')
    print('\nestimand-level bias, bouts/min, at the rate-matched threshold '
          '(a_O - a_H and a_P - a_O)')
    for r in est:
        print(f"  {r['arm']:14s} s{r['seed']:<3d} {r['behav']}  thr {r['thr']:.2f}  "
              f"a_O-a_H {r['b_HO']:+.3f}   a_P-a_O {r['b_OP']:+.3f}")

    # ---- which environments are pool-level constants, and the nuisance-bias channel ---------
    pc = pool_constant(exp_full, exp_full[exp_full.annotation_file.notna()])
    print('\nconstant within a pool, on the 24 annotated pools:')
    for k, v in pc.items():
        print(f"  {k:11s} {v['constant_pools']}/{v['n_pools']} pools "
              f"(max {v['max_per_pool']} per pool)")
    nb = nuisance_bias(exp_full)
    print(f"\nmodel bias explained by a pool-level factor "
          f"({nb['n_obs']} obs, {nb['n_pools']} pools, out-of-fold):")
    for l in LABELS:
        for f in ('annotator', 'genotype', 'line'):
            a, b = nb['level'][l][f], nb['delta'][l][f]
            fmt = lambda x: 'n/a' if x is None else f"eta2 {x['eta2']:.1%} (p {x['p']:.3f})"
            print(f"  {l} ~ {f:10s}  LEVEL {fmt(a):24s}  DELTA(H->O) {fmt(b)}")

    # ---- the PPI++ bound, for the report's box ----------------------------------------------
    n, N = 24, 48
    grid = [{'r': round(r, 2), 'ratio': round(float(np.sqrt(1 - r ** 2 * N / (n + N))), 4)}
            for r in np.arange(0.0, 1.001, 0.05)]
    bound = {'n': n, 'N': N, 'shrink_factor': round(N / (n + N), 4),
             'floor': round(float(np.sqrt(n / (n + N))), 4), 'grid': grid}

    payload = {'meta': {'val_pools': sorted(val), 'n_val_pools': len(val),
                        'arms_present': [t for _, t, _, _ in present],
                        'arms_absent': [t for _, t, _, _ in ARMS
                                        if not (FRAME / t / 'val_probs.npz').exists()],
                        'nbin': NBIN, 'boot_reps': 400,
                        'leak': 'AUC of one phase against another from the model output alone, '
                                'at fixed ground truth. 0.5 = the output carries no phase '
                                'information. Rank-based, so DERM\'s higher output level cannot '
                                'move it. 95% interval bootstrapped over the 4 validation pools.'},
               'prevalence': prev, 'leak': leak, 'summary_phase': summ_phase,
               'summary_cond': summ_cond, 'corr': corrs, 'estimand': est, 'ppi_bound': bound,
               'pool_constant': pc, 'nuisance': nb}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(OUT / 'derm.json', 'w'), indent=1)
    print(f"\nwrote {OUT / 'derm.json'}  ({len(present)} arms, {len(leak)} leak AUCs)")


if __name__ == '__main__':
    main()
