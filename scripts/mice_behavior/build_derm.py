#!/usr/bin/env python3
"""Does the frame classifier read the treatment, and does DERM stop it? One JSON out.

THE MECHANISM AT ISSUE
======================
The phase is VISIBLE. The experimenter opens the cage and the odour port is present in O and gone
in P, so a frame carries the phase whether or not it carries behaviour. Prevalence also moves with
the phase. So a classifier can score a frame by which phase it LOOKS like instead of by what the
mice are doing, and ERM's objective contains nothing that penalises it: the ERM optimum is
P(Y=1|x), which INCLUDES the phase-conditional prior. DERM's optimum divides that prior out. The
prior on DERM helping here is therefore high, and this script tests it in three steps.

STEP 1 -- IS THE SHORTCUT AVAILABLE?  (`probe`)
===============================================
Measured, not assumed. A leave-one-pool-out linear probe on a 24x24 grey thumbnail of a QUIET frame
-- no scored behaviour, >=25 frames (5 s) from any bout, where the model has nothing legitimate to
go on. Result: phase at 0.75 balanced accuracy against 0.333 chance, with O recalled at 0.94.
EXPOSURE (fear vs social) is probed alongside as a negative control and sits at chance, which says
the probe reads the PROTOCOL -- port present, handling -- and not the cage, the animal or the time
of day. If 592 downsampled pixels carry the phase, DINOv2 at 1024 tokens certainly does.

STEP 2 -- WHAT THE SHORTCUT WOULD COST  (`estimand_bias`)
=========================================================
The estimand is a WITHIN-POOL difference. Write the model's expected output in phase p as

    E[f | p] = a_p + b * E[Y | p]      ->      E[D_f] = b * E[D_Y] + (a_O - a_H)

  * `b` is a SCALE: absorbed by PPI++'s lambda, and uncalibrated PPCI never quotes a magnitude.
  * `a_O - a_H` is a BIAS IN THE ESTIMAND, non-zero exactly when the model's error moves WITH the
    phase. It is the only term that can flip a sign or manufacture an effect.

So `a_O - a_H` is the whole question. Measured in bouts per minute at the RATE-MATCHED threshold,
which spends the one scale the estimand allows, per (pool x exposure) -- the estimand's own unit,
giving 8 of them. The ATE is a MEAN over pools, so only the MEAN of this biases the answer;
per-unit scatter costs variance instead and is reported apart. Seeds are averaged within a unit
before the interval, because seed noise is not sampling error.

STEP 3 -- THE OUTPUT-SIDE LEAK, AND WHY IT IS ONLY CORROBORATIVE  (`leak`, `summary_*`)
=======================================================================================
An AUC separating two phases from the model's output at fixed ground truth. It is reported because
it is scale-free -- no threshold, no calibration -- and because DERM's shift shows up in it cleanly.

    READ IT AS A DIRECTION, NEVER AS A TEST FOR ABSENCE. Two reasons, both fatal to that use:

    (a) WRONG PART OF THE DISTRIBUTION. The estimand is a count of threshold crossings at
        tau ~ 0.90-0.98. It lives entirely in the far upper tail. An AUC weights the whole
        distribution, so a tail shift big enough to move bout counts barely moves it.
    (b) NO BASELINE. "At fixed truth" frames are not exchangeable across phases -- the quiet
        stretches of O are not the quiet stretches of H. So 0.5 is NOT the no-shortcut baseline
        and a deviation from it cannot be read in either direction.

    An earlier version of this file read leak ~= 0.5 under ERM as "the shortcut is not open". That
    was wrong on both counts, and it also had the sign backwards: DERM deflates the
    high-prevalence environment BY CONSTRUCTION (see below), so DERM moving O downward relative to
    ERM is the correction operating as designed, not an artefact.

WHY THE ENVIRONMENT MUST BE THE PHASE, AND WHAT THAT COSTS  (`pool_constant`)
============================================================================
DERM's weights are w(y=1,e) = (1-p_e)/P(e) and w(y=0,e) = p_e/P(e). The 1/P(e) cancels in the
ratio, so its entire effect on environment e's operating point is a shift by that environment's
prior odds, w(0,e)/w(1,e) = p_e/(1-p_e) -- exactly the prior a prevalence shortcut would exploit.

That shift reaches a within-pool contrast only if the environment VARIES within a pool. Measured:
phase 0/24 pools constant, odor 0/24, line/sex/genotype 24/24, annotator 22/24, date 21/24. So

    --env-key phase        the ONLY version that can touch a phase shortcut -- and therefore also
                           the only one that can create one, if it overshoots. Validate it on the
                           estimand, never on AP.
    --env-key annotator    cancels in every within-pool difference. Free, and no substitute: it
                           cannot reach this shortcut at all.

THE OTHER CHANNEL: NUISANCE-LINKED BIAS  (`nuisance`)
=====================================================
PPI++ is unbiased for any predictor, so a treatment-linked bias costs it variance rather than
validity. What CAN break its validity on v1 is that the 24 labelled pools are not a random sample
-- annotation is 3:1 het-enriched -- so the rectifier has to transport to 48 wt-enriched pools.
That needs the model's bias not to depend on genotype. Decomposed on the DEPLOYED cross-fitted
predictions over all 24 annotated pools (144 observations, a real denominator), at the LEVEL and in
the WITHIN-POOL DIFFERENCE.

WHAT IS NOT CLAIMED
===================
Steps 2 and 3 rest on FOUR validation pools. On nt the mean a_O - a_H runs +0.163 under ERM against
+0.100 under DERM -- both positive, DERM nearer zero, the direction the mechanism predicts, and
2.6x against 1.6x the size of the pooled true effect. But the 95% intervals are +-0.29 and the
paired test gives p = 0.44, so NONE of it is resolved. This script does not show that DERM helps
and it does not show that it does not.

The experiment that settles it is cross-fitting the matched pair -- DERM-on-phases and its ERM
control -- over the three deployment folds. That measures a_O - a_H on 24 pools instead of 4,
shrinking the SE about 2.4x, and it is also the only way to get real PPI++ intervals and PPCI point
estimates under both objectives with CI as ground truth.

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
from build_estimates import WINDOW, labelled_truth                          # noqa: E402

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


# ------------------------------------------------------------------ rate matching, in RANK space
# THE PROBABILITY SCALE IS NOT COMPARABLE BETWEEN OBJECTIVES; THE RANK OF A FRAME IS.
#
# Every ERM-vs-DERM number here turns a score into a bout count at a threshold chosen so the total
# predicted rate matches the total true rate -- the one degree of freedom the estimand allows.
# Searching that threshold in PROBABILITY space is the mistake. DERM reweights environments, so its
# weight ratio is the prior odds and its scores pile up against 1: a grid in probability space has
# almost no resolution where all of DERM's mass sits and plenty where none of it does. Measured, the
# old fixed grids (0.01 / 0.005, both stopping below 1.0) missed the target rate by -22% to +18% on
# DERM arms -- twice pinned at the last grid point -- against +-6% for every ERM arm. That residual
# IS unmatched rate, i.e. calibration, and it lands straight in a_O - a_H, against the arm under
# test.
#
# So parameterise by WHAT FRACTION OF FRAMES IS CALLED POSITIVE and read the threshold off as a
# quantile of the score distribution. That fraction is uniformly resolvable for ANY score
# distribution, however compressed, and it is the same question for both objectives. Log-spaced,
# because the positive rate is ~1-3%.
FRAC_GRID = np.geomspace(1e-4, 0.5, 160)


def rate_match(count_at, T, scores):
    """(threshold, count) whose total bout count is closest to T, searched in rank space.

    `count_at(t)` -> total bouts at threshold t. `scores` is the pooled per-frame score vector the
    threshold will be applied to. Not assumed monotone: postprocess() with merge_gap=1 can split
    one bout into two as the threshold rises, so this searches rather than bisects.
    """
    q = np.asarray(scores, dtype=np.float64)
    seen = {}

    def at(frac):
        frac = float(min(max(frac, 1e-6), 0.9))
        if frac not in seen:
            th = float(np.quantile(q, 1.0 - frac))
            seen[frac] = (th, count_at(th))
        return seen[frac]

    for f in FRAC_GRID:
        at(f)
    best = min(seen, key=lambda f: (abs(seen[f][1] - T), f))
    lo, hi = best * 0.7, best * 1.45                     # local refine, still in rank space
    for f in np.linspace(lo, hi, 60):
        at(f)
    best = min(seen, key=lambda f: (abs(seen[f][1] - T), f))
    return seen[best]


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
    arrs = [g['p'].to_numpy() for g in gs]
    th, P = rate_match(lambda t: sum(len(runs(postprocess(a >= t, 1, 1))) for a in arrs), T,
                       np.concatenate(arrs))
    if T and abs(P - T) / T > 0.05:
        print(f'  !! {tag} {l}: rate match off by {100 * (P - T) / T:+.1f}% at thr {th:.4f} '
              f'-- the estimand bias below carries that as calibration', flush=True)
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



# ------------------------------------------------------------------ is the shortcut AVAILABLE?
# Everything else in this file measures whether the model's output moves with the phase. That is
# downstream of a prior question: CAN it? If the phase is invisible in a frame there is no shortcut
# to learn and no reason to expect DERM to do anything. So probe it directly, and probe it on the
# frames where a shortcut would do damage -- the QUIET ones, carrying no scored behaviour and far
# from any bout, where the model has nothing legitimate to go on.
#
# Descriptor: a 24x24 grayscale thumbnail plus a 16-bin intensity histogram. Deliberately crude --
# if a 592-dimensional linear probe on downsampled pixels can read the phase, DINOv2 at 1024 tokens
# certainly can, and the argument does not depend on what the encoder does with it.
#
# Held out by POOL, so the probe cannot win by memorising a cage.
#
# The EXPOSURE (fear vs social) is probed alongside as a negative control. Both exposures use the
# same port and the same handling, and the odour itself is not visible, so a probe that reads phase
# but not exposure is reading the protocol rather than the cage, the animal or the time of day.
PROBE_DIST = 25          # frames from the nearest scored bout -- 5 s at 5 fps
PROBE_PER_CELL = 120     # frames per pool x phase x exposure, so the design is balanced
PROBE_THUMB = 32         # fine enough that one corner is resolvable


def _probe_frames(exp: pd.DataFrame, tag: str):
    """Balanced sample of QUIET frames as small grey thumbnails, with their pool and phase."""
    import io
    from PIL import Image
    ix = np.load(ROOT / 'dataset' / 'mice' / 'v1' / 'jpegcache_k2.npz')
    keys, offs = ix['all_needed'], ix['offsets']
    blob = np.memmap(ROOT / 'dataset' / 'mice' / 'v1' / 'jpegcache_k2.bin', dtype=np.uint8, mode='r')
    pos = {int(k): i for i, k in enumerate(keys)}
    d = np.load(FRAME / tag / 'val_probs.npz', allow_pickle=True)
    df = pd.DataFrame({'obs': d['obs'], 'gi': d['gi'],
                       'y': (d['labels'][:, 0] > 0.5) | (d['labels'][:, 1] > 0.5)}
                      ).sort_values(['obs', 'gi'])
    df = df.merge(exp, left_on='obs', right_on='observation_id')
    dist = []
    for _, g in df.groupby('obs', sort=False):
        y = g.y.to_numpy(); q = np.arange(len(y))
        dist.append(pd.Series(np.abs(q[:, None] - np.flatnonzero(y)[None, :]).min(1)
                              if y.any() else np.full(len(y), 10 ** 6), index=g.index))
    df['dist'] = pd.concat(dist)
    quiet = df[(~df.y) & (df.dist >= PROBE_DIST)]
    rng = np.random.default_rng(0)
    samp = pd.concat([g.iloc[rng.choice(len(g), min(len(g), PROBE_PER_CELL), replace=False)]
                      for _, g in quiet.groupby(['pool', 'phase', 'odor'], sort=True)]
                     ).reset_index(drop=True)
    T, keep = [], []
    for k, gi in enumerate(samp.gi.to_numpy()):
        i = pos.get(int(gi))
        if i is None:
            continue
        im = Image.open(io.BytesIO(blob[offs[i]:offs[i + 1]].tobytes())).convert('L')
        T.append(np.asarray(im.resize((PROBE_THUMB, PROBE_THUMB), Image.BILINEAR),
                            dtype=np.float32) / 255.0)
        keep.append(k)
    return np.stack(T), samp.iloc[keep].reset_index(drop=True), int(len(quiet))


def phase_probe(exp: pd.DataFrame, tag: str = 'res448_k2_frozen_d4photo_ermH5M') -> dict:
    """Can the PHASE be read off a QUIET frame -- and if so, from where?

    Everything else in this file measures whether the model's output moves with the phase. That is
    downstream of a prior question: CAN it? A physical bag is placed in a corner of the cage during
    the O phase, so a frame carries the phase whether or not it carries behaviour. This measures how
    freely.

    Deliberately crude: a leave-one-POOL-out linear probe on a 32x32 grey thumbnail. If 1024
    downsampled pixels carry the phase, DINOv2 at 1024 tokens at 448 px certainly does, and the
    argument does not depend on what the encoder does with it.

    Four things are reported, and the last two are the ones with a fix attached:

      phase / exposure    3-way and 2-way. EXPOSURE is the NEGATIVE CONTROL: both exposures use the
                          same bag and the same handling, so a probe that reads phase but not
                          exposure is reading the PROTOCOL, not the cage, the animal or the hour.
      O vs not-O, H vs P  the bag is present in O only, so the first should be much the easier.
      region              the same probe restricted to one part of the frame. Localises the cue.
      mask               the probe with the bag's corner blanked. The residual is NOT failure: the
                          bag also changes WHERE THE ANIMALS ARE, which is real behaviour and must
                          not be removed. Masking removes the non-behavioural cue and leaves that.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from PIL import Image                                              # noqa: F401
    except ImportError as e:
        return {'note': f'probe skipped: {e}'}
    if not (ROOT / 'dataset' / 'mice' / 'v1' / 'jpegcache_k2.bin').exists():
        return {'note': 'probe skipped: no jpeg cache on disk'}

    T, samp, n_quiet = _probe_frames(exp, tag)
    S = PROBE_THUMB
    pool, ph = samp['pool'].to_numpy(), samp.phase.to_numpy()

    def fit(X, tgt, sub=None):
        """Leave-one-pool-out balanced accuracy, plus the pooled confusion matrix."""
        labs = sorted(set(tgt if sub is None else tgt[sub]))
        accs, cms = [], []
        for pl in sorted(set(pool)):
            tr, te = pool != pl, pool == pl
            if sub is not None:
                tr, te = tr & sub, te & sub
            if len(set(tgt[tr])) < 2 or not te.any():
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
            m.fit(X[tr], tgt[tr]); pr = m.predict(X[te])
            accs.append(balanced_accuracy_score(tgt[te], pr))
            cms.append(confusion_matrix(tgt[te], pr, labels=labs))
        cm = np.sum(cms, axis=0)
        return {'labels': [str(x) for x in labs], 'chance': round(1 / len(labs), 4),
                'bal_acc': round(float(np.mean(accs)), 4),
                'per_pool': [round(float(a), 4) for a in accs],
                'confusion': cm.tolist(),
                'recall': {str(l): round(float(cm[i, i] / max(cm[i].sum(), 1)), 4)
                           for i, l in enumerate(labs)}}

    flat = T.reshape(len(T), -1)
    yO = (ph == 'O').astype(int)
    out = {'n_frames': int(len(T)), 'thumb': S, 'dim': int(flat.shape[1]),
           'dist_frames': PROBE_DIST, 'quiet_available': n_quiet,
           'n_pools': int(len(set(pool))),
           'targets': {'phase': fit(flat, ph),
                       'exposure': fit(flat, samp.odor.to_numpy()),
                       'O_vs_rest': fit(flat, yO),
                       'H_vs_P': fit(flat, ph, sub=(ph != 'O'))}}

    # --- where is it? one region at a time, O vs not-O
    reg = {'top-left': (slice(0, S // 2), slice(0, S // 2)),
           'top-right': (slice(0, S // 2), slice(S // 2, S)),
           'bottom-left': (slice(S // 2, S), slice(0, S // 2)),
           'bottom-right': (slice(S // 2, S), slice(S // 2, S))}
    out['region'] = {k: fit(T[:, r, c].reshape(len(T), -1), yO)['bal_acc']
                     for k, (r, c) in reg.items()}
    out['region']['centre'] = fit(
        T[:, S // 4:3 * S // 4, S // 4:3 * S // 4].reshape(len(T), -1), yO)['bal_acc']
    bor = T.copy(); bor[:, S // 4:3 * S // 4, S // 4:3 * S // 4] = 0
    out['region']['border'] = fit(bor.reshape(len(T), -1), yO)['bal_acc']

    # --- the same corner in every pool? mean(O) - mean(not-O) says where without any fitting
    out['corner'] = {}
    for pl in sorted(set(pool)):
        m = pool == pl
        dm = np.abs(T[m & (ph == 'O')].mean(0) - T[m & (ph != 'O')].mean(0))
        r, c = np.unravel_index(dm.argmax(), dm.shape)
        r0, c0 = S // 4 * (r // (S // 4)), S // 4 * (c // (S // 4))
        out['corner'][str(pl)] = {
            'peak': round(float(dm.max()), 4), 'row': int(r), 'col': int(c),
            'quadrant': ('top' if r < S / 2 else 'bottom') + '-' + ('left' if c < S / 2 else 'right'),
            'share_in_cell': round(float(dm[r0:r0 + S // 4, c0:c0 + S // 4].sum() / dm.sum()), 4)}

    # --- does masking it work? no training needed to answer this
    out['mask'] = {'unmasked': out['targets']['O_vs_rest']['bal_acc']}
    for frac in (0.125, 0.1875, 0.25):
        k = int(round(S * frac))
        M = T.copy(); M[:, S - k:, :k] = 0
        out['mask'][f'bottom_left_{frac:g}'] = {
            'bal_acc': fit(M.reshape(len(M), -1), yO)['bal_acc'],
            'frame_share': round(k * k / S / S, 4)}
    k = int(round(S * 0.25))
    M = T.copy()
    for r in (slice(0, k), slice(S - k, S)):
        for c in (slice(0, k), slice(S - k, S)):
            M[:, r, c] = 0
    out['mask']['four_corners_0.25'] = {'bal_acc': fit(M.reshape(len(M), -1), yO)['bal_acc'],
                                       'frame_share': round(4 * k * k / S / S, 4)}
    return out


# ------------------------------------------------- the estimand-level bias, with an interval
# a_O - a_H in the unit the report estimates on, per (pool x exposure) -- the estimand's own unit
# of analysis, which gives 8 of them instead of 1. The ATE is a MEAN over pools, so it is the MEAN
# of this that biases the answer; per-unit scatter costs variance instead and is reported apart.
# Seeds are averaged WITHIN a unit before the interval, because seed noise is not sampling error.
def estimand_bias(exp: pd.DataFrame, families: dict) -> dict:
    def units(tag, l):
        j = LABELS.index(l)
        d = np.load(FRAME / tag / 'val_probs.npz', allow_pickle=True)
        df = pd.DataFrame({'obs': d['obs'], 'gi': d['gi'], 'p': d['probs'][:, j],
                           'y': d['labels'][:, j]}).sort_values(['obs', 'gi'])
        gs = [(o, g) for o, g in df.groupby('obs', sort=False)]
        T = sum(len(runs(g['y'].to_numpy() > 0.5)) for _, g in gs)
        # rate_match(), NOT a fixed grid. This path had its own 0.005 grid capped below 1.0, which
        # is finer than match_threshold()'s but fails the same way on a DERM arm whose probability
        # mass is compressed against 1: xfit_derm_f1 nt pinned at the ceiling and under-counted
        # bouts by 22% while every ERM arm matched inside 6%. The residual is calibration, and it
        # lands straight in a_O - a_H -- against DERM specifically. See rate_match()'s docstring.
        arrs = [g['p'].to_numpy() for _, g in gs]
        th, P = rate_match(lambda t: sum(len(runs(postprocess(a >= t, 1, 1))) for a in arrs), T,
                           np.concatenate(arrs))
        out['match_resid'].setdefault(l, {})[tag] = round((P - T) / T, 4) if T else None
        if T and abs(P - T) / T > 0.05:
            print(f'  !! {tag} {l}: rate match off by {100 * (P - T) / T:+.1f}% '
                  f'at thr {th:.4f}', flush=True)
        po = pd.DataFrame([
            {'observation_id': o,
             'true': len(runs(g['y'].to_numpy() > 0.5)) / (len(g) / FPS / 60),
             'pred': len(runs(postprocess(g['p'].to_numpy() >= th, 1, 1))) / (len(g) / FPS / 60)}
            for o, g in gs]).merge(exp, on='observation_id')
        rows = []
        for (pool, od), g in po.groupby(['pool', 'odor']):
            m = g.drop_duplicates('phase').set_index('phase')
            if not {'H', 'O'} <= set(m.index):
                continue
            rows.append({'pool': pool, 'odor': od,
                         'bias': (m.loc['O', 'pred'] - m.loc['O', 'true'])
                                 - (m.loc['H', 'pred'] - m.loc['H', 'true']),
                         'dY': m.loc['O', 'true'] - m.loc['H', 'true']})
        return pd.DataFrame(rows).set_index(['pool', 'odor']), th

    # A family is either SEEDS of one split -- average per unit, seed noise is not sampling error
    # -- or FOLDS of a cross-fit, which cover DISJOINT pools and must be concatenated. Averaging
    # folds gives NaN, because no (pool, exposure) unit appears in more than one of them.
    def combine(tags, l):
        frames = []
        for t in tags:
            u, th = units(t, l)
            out['thresholds'].setdefault(l, {})[t] = th
            frames.append(u.bias)
        if len(frames) == 1:
            return frames[0]
        idx = set(frames[0].index)
        disjoint = all(not (idx & set(f.index)) for f in frames[1:])
        if disjoint:
            return pd.concat(frames)                     # cross-fitted folds: tile the pools
        return sum(frames) / len(frames)                 # seeds of one split: average

    out = {'unit': 'bouts per minute, H->O', 'thresholds': {}, 'match_resid': {},
           'truth': {}, 'families': {}}
    for l in LABELS:
        ref, _ = units(families['ERM'][0], l)
        out['truth'][l] = {'pooled': round(float(ref.dY.mean()), 4),
                           'fear': round(float(ref.dY.xs('F', level='odor').mean()), 4),
                           'social': round(float(ref.dY.xs('S', level='odor').mean()), 4)}
        vals = {}
        for fam, tags in families.items():
            if not tags:
                continue
            v = combine(tags, l)
            vals[fam] = v
            a = v.to_numpy(); n = len(a)
            se = a.std(ddof=1) / np.sqrt(n)
            q = float(stats.t.ppf(0.975, n - 1))
            out['families'].setdefault(l, {})[fam] = {
                'n_units': n, 'n_seeds': len(tags),
                'mean': round(float(a.mean()), 4),
                'lo': round(float(a.mean() - q * se), 4),
                'hi': round(float(a.mean() + q * se), 4),
                'mean_abs': round(float(np.abs(a).mean()), 4),
                'positive': int((a > 0).sum()),
                'share_of_truth': (round(abs(float(a.mean() / out['truth'][l]['pooled'])), 2)
                                   if out['truth'][l]['pooled'] else None)}
        for key, (ke, kd) in {'paired_erm_minus_derm': ('ERM', 'DERM'),
                              'paired_xfit': ('ERM · 24 pools', 'DERM · 24 pools'),
                              'paired_xfit_ssl': ('ERM · 24 pools · SSL',
                                                  'DERM · 24 pools · SSL')}.items():
            if ke not in vals or kd not in vals:
                continue
            e, d = vals[ke].align(vals[kd], join='inner')
            e, d = e.to_numpy(), d.to_numpy()
            pr = stats.ttest_rel(e, d)
            out['families'][l][key] = {
                'diff': round(float(np.mean(e - d)), 4), 'p': round(float(pr.pvalue), 4),
                'shrunk_units': int((np.abs(d) < np.abs(e)).sum()), 'n_units': len(e)}
    return out

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



# ------------------------------------------------------ the exposure-split (odour) experiment
# `xfit_odour.sh` trains on one exposure session and leaves the other as the test session, holding
# the cage, the animals, the annotator and the lighting fixed. The test session is NOT in the run's
# val_probs.npz -- that holds the monitor set, which stays inside the TRAINING exposure so early
# stopping never touches the test -- so it comes from the dense pass instead.
#
# WHAT MAKES THIS A TEST RATHER THAN A MEASUREMENT. The two exposures carry opposite true effects on
# nose-to-tail (H->O is +0.18 bouts/min under fear, -0.31 under social). A model that has learnt its
# training session's phase prior imports that session's prevalence gap into the test session, so
#
#     bias on test T  ~  (gap in training session S) - (gap in T)
#
# which FLIPS SIGN when the direction flips. A plain generalisation gap -- the model simply being
# worse on an exposure it never saw -- cannot produce a sign flip tied to which session trained.
# So the reported quantity is the PAIR of biases, one per direction, and the prediction is
# ERM: opposite signs, large; DERM: both nearer zero.
#
# NOT a deployment estimate: the model has seen every test pool, so its bias there is a LOWER BOUND
# on what PPCI suffers on the 48 unannotated pools. And PPI++ cannot use it at all -- the rectifier
# would sit on pools the model trained on.
# Both SELECTION RULES, side by side. The bare tags keep the epoch with the highest UNWEIGHTED
# monitor AP, which for a DERM arm asks for the epoch that best exploits the phase prior the
# objective exists to remove -- so the selection rule pulls against the objective and the ERM
# comparison inherits that. The `_last` tags use a fixed epoch budget, identical for both arms,
# which takes selection out of the contrast entirely. Absent arms are simply skipped.
# variant -> (ERM tag suffix, DERM tag suffix). Three comparisons, each INTERNALLY matched:
#   ''       the original pair: selection on unweighted monitor AP, DERM weights from the
#            1:1-subsampled epoch and a frame-share P(E).
#   '_last'  fixed epoch budget for both arms, so the selection rule is out of the contrast.
#   '_popw'  same fixed budget, plus DERM's weights computed for the POPULATION (undoing the
#            subsampling attenuation) with a duration-neutral P(E). ERM is untouched by those
#            flags, so this variant REUSES the _last ERM leg rather than retraining it.
#   '_popw_s1'/'_popw_s2'  seed replicates of the decisive pair (fear direction): the headline
#            paired test otherwise rests on seed 42 alone, and this project has already seen a
#            metric move 0.18 -> 0.85 between two seeds of one config. Paired WITHIN seed.
ODOUR_VARIANTS = {'': ('', ''), '_last': ('_last', '_last'), '_popw': ('_last', '_last_popw'),
                  '_popw_s1': ('_last_s1', '_last_popw_s1'),
                  '_popw_s2': ('_last_s2', '_last_popw_s2')}
ODOUR_ARMS = {f'tr{d}_{o}{sfx}': (f'odour_tr{d}_{o}{sfx}', d)
              for d in ('F', 'S')
              for o, sfx in {('erm', es) for es, _ in ODOUR_VARIANTS.values()}
              | {('derm', ds) for _, ds in ODOUR_VARIANTS.values()}}


def odour_split(exp_full: pd.DataFrame) -> dict:
    """a_O - a_H on the HELD-OUT exposure, per pool, for each arm. Empty until the runs land."""
    out = {'note': 'mechanism split: the model has seen every test pool, so this is a LOWER bound '
                   'on the deployment bias, and PPI++ cannot use it',
           'arms': {}, 'landed': [], 'absent': []}
    for key, (tag, train_od) in ODOUR_ARMS.items():
        # `_heldout` ONLY. The plain pred_dense_v1.csv on these arms is the first attempt's dump of
        # the 48 UNANNOTATED pools -- no truth to compare against, zero overlap with the labelled
        # 24, and it is what left this block reading n_obs 0. Reading it again would silently
        # reintroduce that. See predict_dense.py --held-out-odour.
        csv = FRAME / tag / 'pred_dense_v1_heldout.csv'
        npz = FRAME / tag / 'pred_dense_v1_heldout.npz'
        if not (csv.exists() and npz.exists()):
            out['absent'].append(tag)
            continue
        out['landed'].append(tag)
        test_od = 'S' if train_od == 'F' else 'F'
        d = pd.read_csv(csv)
        rec = d[['observation_id', 'pool', 'phase', 'odor']].copy()
        m = rec.merge(labelled_truth(), on='observation_id', how='inner')
        m = m[m.odor == test_od].reset_index(drop=True)
        arm = {'tag': tag, 'train_odour': train_od, 'test_odour': test_od,
               'n_obs': int(len(m)), 'n_pools': int(m.pool.nunique()),
               'window_min': round(WINDOW / FPS / 60, 1), 'thr': {}, 'behav': {}}
        # TWO CONVENTIONS THIS BLOCK GOT WRONG ONCE, both of which corrupt a_O - a_H:
        #
        # (1) THE WINDOW. It read the CSV's `p_*_t*` columns, which aggregate over the WHOLE
        #     recording -- and H is 30 minutes while O and P are 15, so the H term was averaged
        #     over twice the span the truth uses. Behaviour decays inside a phase (that is what
        #     section 02b measures), so this is not a wash: it biases the H leg specifically.
        #     build_estimates.py says it outright -- the window has to be applied BEFORE
        #     aggregating and cannot be undone afterwards. So read the npz and slice.
        #
        # (2) THE THRESHOLD. It took the column nearest 0.90, fixed. These single-exposure models
        #     run 3-8x hot in occupancy, so a fixed threshold leaves the global bout RATE
        #     unmatched and what gets measured is mostly calibration. Worse, it is not neutral
        #     between the arms being compared: DERM's probabilities are inflated BY CONSTRUCTION
        #     (the weight ratio is the prior odds), so a fixed threshold manufactures a larger
        #     apparent bias for DERM than for ERM and would have been read as DERM failing.
        #     Rate-match instead, exactly as match_threshold() does for the deployment folds:
        #     spend the one degree of freedom the estimand allows -- the global scale -- and
        #     whatever phase-dependence survives cannot be explained away as calibration.
        z = np.load(npz, allow_pickle=True)
        seq = {o: z[o][:WINDOW].astype(np.float32) for o in m.observation_id if o in z.files}
        mins = WINDOW / FPS / 60
        for lab in LABELS:
            j = LABELS.index(lab)
            have = [o for o in m.observation_id if o in seq]
            if len(have) < 3:
                continue
            keep = m[m.observation_id.isin(have)]
            total_true = float(keep[f't_events_{lab}'].sum() * mins)
            th, n_at = rate_match(
                lambda t: sum(len(runs(postprocess(seq[o][:, j] >= t, 1, 1))) for o in have),
                total_true, np.concatenate([seq[o][:, j] for o in have]))
            arm['thr'][lab] = round(th, 5)
            arm.setdefault('match_resid', {})[lab] = round(
                (n_at - total_true) / total_true, 4) if total_true else None
            f = {o: len(runs(postprocess(seq[o][:, j] >= th, 1, 1))) / mins for o in have}
            # BOTH transitions. The protocol turns the odour ON at H->O and OFF again at O->P, and
            # a phase-prior shortcut has no reason to respect that asymmetry -- so an objective
            # that removes it should show up on both, and one that only appears on the ON leg is
            # not the shortcut. Riccardo asked for both legs for exactly this reason.
            for x, y in TRANS:
                rows = []
                for pool, g in keep.groupby('pool'):
                    q = g.drop_duplicates('phase').set_index('phase')
                    if not {x, y} <= set(q.index):
                        continue
                    fy, fx = f[q.loc[y, 'observation_id']], f[q.loc[x, 'observation_id']]
                    rows.append({
                        'pool': pool,
                        'bias': (fy - q.loc[y, f't_events_{lab}'])
                                - (fx - q.loc[x, f't_events_{lab}']),
                        'dY': q.loc[y, f't_events_{lab}'] - q.loc[x, f't_events_{lab}'],
                        'dF': fy - fx})
                if len(rows) < 3:
                    continue
                r = pd.DataFrame(rows)
                v = r.bias.to_numpy()
                se = v.std(ddof=1) / np.sqrt(len(v))
                qt = float(stats.t.ppf(0.975, len(v) - 1))
                trans = f'{x}->{y}'
                arm.setdefault('per_pool', {}).setdefault(lab, {})[trans] = {
                    str(k): round(float(xx), 4) for k, xx in zip(r['pool'], v)}
                cell = {
                    'n_pools': len(v), 'mean': round(float(v.mean()), 4),
                    'lo': round(float(v.mean() - qt * se), 4),
                    'hi': round(float(v.mean() + qt * se), 4),
                    'true_dY': round(float(r.dY.mean()), 4),
                    'pred_dF': round(float(r.dF.mean()), 4),
                    'r_delta': (round(float(np.corrcoef(r.dY, r.dF)[0, 1]), 4)
                                if r.dY.std() > 0 and r.dF.std() > 0 else None)}
                arm.setdefault('cells', {}).setdefault(lab, {})[trans] = cell
                if trans == 'H->O':
                    arm['behav'][lab] = cell            # the ON leg keeps its old key
        out['arms'][key] = arm
    # the sign test: ERM's bias should reverse between directions, DERM's should sit nearer zero
    seen_sign = set()
    for v, (es, ds) in ODOUR_VARIANTS.items():
        for obj, sfx in (('erm', es), ('derm', ds)):
            key = f'{obj}{sfx}'
            if key in seen_sign:
                continue
            seen_sign.add(key)
            a, b = out['arms'].get(f'trF_{obj}{sfx}'), out['arms'].get(f'trS_{obj}{sfx}')
            if not (a and b):
                continue
            obj = key                                   # label the row by the tag it came from
            for lab in LABELS:
                for x, y in TRANS:
                    tr = f'{x}->{y}'
                    ca = a.get('cells', {}).get(lab, {}).get(tr)
                    cb = b.get('cells', {}).get(lab, {}).get(tr)
                    if not (ca and cb):
                        continue
                    out.setdefault('sign_test', {}).setdefault(obj, {}).setdefault(lab, {})[tr] = {
                        'train_fear': ca['mean'], 'train_social': cb['mean'],
                        'reverses': bool(ca['mean'] * cb['mean'] < 0)}
    # PAIRED, because the design's whole advantage is that both objectives are scored on the SAME
    # 24 pools of the SAME held-out exposure. Comparing two independent CIs throws that away: the
    # between-pool variance is common to both arms and cancels in the difference.
    pairs = [(f'train_{n}{v}', k, es, ds)
             for v, (es, ds) in ODOUR_VARIANTS.items()
             for n, k in (('fear', 'trF'), ('social', 'trS'))]
    for direction, key, es, ds in pairs:
        a, b = out['arms'].get(f'{key}_erm{es}'), out['arms'].get(f'{key}_derm{ds}')
        if not (a and b):
            continue
        for lab in LABELS:
            for x, y in TRANS:
                tr = f'{x}->{y}'
                pa = a.get('per_pool', {}).get(lab, {}).get(tr)
                pb = b.get('per_pool', {}).get(lab, {}).get(tr)
                if not (pa and pb):
                    continue
                shared = sorted(set(pa) & set(pb))
                if len(shared) < 3:
                    continue
                dv = np.array([pa[k] - pb[k] for k in shared])
                _, pv = stats.ttest_1samp(dv, 0.0)
                out.setdefault('paired', {}).setdefault(direction, {}).setdefault(lab, {})[tr] = {
                    'n_pools': len(shared), 'erm_minus_derm': round(float(dv.mean()), 4),
                    'p': round(float(pv), 4),
                    'toward_zero': int(sum(abs(pb[k]) < abs(pa[k]) for k in shared))}
    # SEED-AVERAGED paired test over the replicate arms: per-pool bias averaged across seeds
    # WITHIN arm first (the report's standing "seeds averaged within a unit" convention), then
    # ERM-DERM paired over the same pools. This is the seed-robust headline; the per-seed rows in
    # `paired` show the spread it averages over -- ERM's imported bias varies with seed while
    # DERM's stays near zero, so single-seed paired p values range from <1e-4 to 0.39 on the same
    # comparison.
    SEED_SETS = {'train_fear_popw_seedavg':
                 ('trF', ['_last', '_last_s1', '_last_s2'],
                  ['_last_popw', '_last_popw_s1', '_last_popw_s2'])}
    for key, (kdir, es_l, ds_l) in SEED_SETS.items():
        E_ = [out['arms'].get(f'{kdir}_erm{s}') for s in es_l]
        D_ = [out['arms'].get(f'{kdir}_derm{s}') for s in ds_l]
        if not (all(E_) and all(D_)):
            continue
        for lab in LABELS:
            for x, y in TRANS:
                tr = f'{x}->{y}'
                pp = [a.get('per_pool', {}).get(lab, {}).get(tr) for a in E_ + D_]
                if not all(pp):
                    continue
                pools = sorted(set.intersection(*[set(p) for p in pp]))
                if len(pools) < 3:
                    continue
                ev = np.array([np.mean([p[q] for p in pp[:len(E_)]]) for q in pools])
                dm = np.array([np.mean([p[q] for p in pp[len(E_):]]) for q in pools])
                diff = ev - dm
                _, pv = stats.ttest_1samp(diff, 0.0)
                out.setdefault('seed_avg', {}).setdefault(key, {}).setdefault(lab, {})[tr] = {
                    'n_seeds': len(E_), 'n_pools': len(pools),
                    'erm_mean': round(float(ev.mean()), 4),
                    'derm_mean': round(float(dm.mean()), 4),
                    'erm_minus_derm': round(float(diff.mean()), 4), 'p': round(float(pv), 4)}
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

    # This is NOT a bug test. DERM shifts each environment's operating point by its prior odds, so
    # delta running opposite to the log odds ratio is the correction OPERATING, and the sign
    # agreement below is a check that the implementation does what the weights say. An earlier
    # version of this file read it as evidence that DERM had installed a bias of its own.
    print('\nenv = 3 phases   (DERM shifts each environment by its prior odds, so delta should run')
    print('                  OPPOSITE to the log OR -- this checks the mechanism, not a fault)')
    for r in summ_phase:
        print(f"  {r['behav']} {r['trans']:6s} logOR {r['log_or']:+.3f}  ERM {r['erm']:.3f} -> "
              f"DERM {r['derm']:.3f}   delta {r['delta']:+.3f}  {'OK' if r['agree'] else '--'}")
    print(f"  Pearson r = {corrs['phase']['r']} (p = {corrs['phase']['p']}, "
          f"n = {corrs['phase']['n']}), signs agree {corrs['phase']['agree']}/"
          f"{corrs['phase']['n']}")
    print('\nenv = 6 phase x exposure cells   (same check, one odds ratio per cell)')
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

    # ---- is the shortcut available at all? --------------------------------------------------
    probe = phase_probe(exp)
    if 'note' in probe:
        print('\n' + probe['note'])
    else:
        print(f"\nphase probe: {probe['n_frames']} quiet frames "
              f"(no scored behaviour, >={probe['dist_frames']} frames from any bout), "
              f"{probe['thumb']}x{probe['thumb']} grey + histogram, leave-one-pool-out")
        for k, v in probe['targets'].items():
            print(f"  {k:11s} balanced acc {v['bal_acc']:.3f}  (chance {v['chance']:.3f})  "
                  f"recall " + ', '.join(f'{a} {b:.2f}' for a, b in v['recall'].items()))
        print('  where the O cue is (O-vs-rest accuracy from one region): '
              + ', '.join(f'{k} {v:.3f}' for k, v in probe['region'].items()))
        print('  bag corner per pool: ' + ', '.join(
            f"{k} {v['quadrant']} (peak {v['peak']:.2f}, {v['share_in_cell']:.0%} of the diff)"
            for k, v in probe['corner'].items()))
        print('  masking it: ' + ', '.join(
            f"{k} -> {v['bal_acc']:.3f} ({v['frame_share']:.1%} of frame)"
            for k, v in probe['mask'].items() if isinstance(v, dict)))

    # ---- the estimand-level bias, with an interval ------------------------------------------
    # The 4-pool families are the standing split (plain 5.03 M head). The xfit_* families are the
    # SAME comparison over the three deployment folds, so a_O - a_H lands on 24 pools instead of 4
    # -- but on the 0.52 M cross-attention head, so the two are separate families on purpose and
    # must not be read as one series. They appear as soon as their runs land; nothing to edit.
    #
    # THE `· SSL` PAIR IS A SECOND, INDEPENDENT REPLICATION of that same 24-pool comparison, on a
    # different backbone. `xfit_f{1,2,3}` (the ORIGINAL deployment folds) are SSL-adapted encoder
    # + plain 5.04 M head under ERM; `xfit_derm_ssl_f{1,2,3}` are the identical recipe -- same
    # encoder checkpoint, same head, same encoder lr, same seed, same folds, same val_pools --
    # with the objective switched to DERM on phases. So this pair varies ONE thing, which the
    # stock-encoder pair also does but on the other backbone. Two backbones agreeing is a
    # replication; two backbones disagreeing says the correction is backbone-dependent, and
    # either answer is worth more than one pair alone. Four families, two paired tests, one code
    # path -- the rate-matched threshold in units() applies to all of them equally.
    FAMS = {'ERM': ['res448_k2_frozen_d4photo_ermH5M', 'res448_k2_frozen_d4photo_ermH5M_s1'],
            'DERM': ['res448_k2_frozen_d4photo_dermPhase',
                     'res448_k2_frozen_d4photo_dermPhase_s1'],
            'DERM-cells': ['res448_k2_frozen_d4photo_dermCond'],
            'ERM · 24 pools': [f'xfit_erm_f{k}' for k in (1, 2, 3)],
            'DERM · 24 pools': [f'xfit_derm_f{k}' for k in (1, 2, 3)],
            'ERM · 24 pools · SSL': [f'xfit_f{k}' for k in (1, 2, 3)],
            'DERM · 24 pools · SSL': [f'xfit_derm_ssl_f{k}' for k in (1, 2, 3)]}
    FAMS = {k: [t for t in v if (FRAME / t / 'val_probs.npz').exists()] for k, v in FAMS.items()}
    eb = estimand_bias(exp, {k: v for k, v in FAMS.items() if v})
    print('\nmean a_O - a_H over the 8 (pool x exposure) units -- the ATE-relevant component')
    for l in LABELS:
        print(f"  {l}: true D_Y pooled {eb['truth'][l]['pooled']:+.3f} "
              f"(fear {eb['truth'][l]['fear']:+.3f}, social {eb['truth'][l]['social']:+.3f})")
        for fam in FAMS:
            if fam not in eb['families'][l]:
                continue
            r = eb['families'][l][fam]
            print(f"    {fam:11s} {r['mean']:+.3f}  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
                  f"{'RESOLVED' if r['lo'] * r['hi'] > 0 else 'not resolved'}  "
                  f"|mean| = {r['share_of_truth']}x the pooled true effect")
        # Every paired test that has both of its arms on disk. The 4-pool pair is the original
        # one; the two 24-pool pairs are the cross-fitted replications, one per backbone, and
        # they are printed side by side because the interesting question is whether they agree.
        for key, nice in (('paired_erm_minus_derm', 'paired ERM-DERM  (4 pools)     '),
                          ('paired_xfit',           'paired ERM-DERM  (24p, stock)  '),
                          ('paired_xfit_ssl',       'paired ERM-DERM  (24p, SSL)    ')):
            pr = eb['families'][l].get(key)
            if pr is None:
                continue
            print(f"    {nice} {pr['diff']:+.3f}  p = {pr['p']:.3f}  "
                  f"shrunk in {pr['shrunk_units']}/{pr['n_units']} units")

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

    # ---- the exposure-split experiment, once its dense passes land --------------------------
    od = odour_split(exp_full)
    if od['absent']:
        print(f"\nexposure split: not landed yet -- {', '.join(od['absent'])}")
    for key, arm in od['arms'].items():
        print(f"\nexposure split {key}: trained on {arm['train_odour']}, tested on "
              f"{arm['test_odour']} ({arm['n_obs']} obs, {arm['n_pools']} pools)")
        for lab, byt in arm.get('cells', {}).items():
            for tr, r in byt.items():
                leg = 'ON ' if tr == 'H->O' else 'OFF'
                print(f"    {lab} {tr:5s} [{leg}]  bias {r['mean']:+.3f}  "
                      f"CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  n={r['n_pools']}   "
                      f"true D_Y {r['true_dY']:+.3f}  pred D_f {r['pred_dF']:+.3f}  "
                      f"rd {r['r_delta']}  resid {arm.get('match_resid', {}).get(lab)}")
    print('\n  SIGN TEST -- does the bias reverse with the training direction?')
    for obj, d in od.get('sign_test', {}).items():
        for lab, byt in d.items():
            for tr, r in byt.items():
                print(f"    {obj:4s} {lab} {tr:5s}: train-fear {r['train_fear']:+.3f} vs "
                      f"train-social {r['train_social']:+.3f}  -> "
                      f"{'REVERSES' if r['reverses'] else 'same sign'}")
    print('\n  PAIRED ERM-DERM on the same pools')
    for direction, d in od.get('paired', {}).items():
        for lab, byt in d.items():
            for tr, r in byt.items():
                print(f"    {direction:12s} {lab} {tr:5s}: {r['erm_minus_derm']:+.3f}  "
                      f"p={r['p']:.4f}  DERM nearer zero {r['toward_zero']}/{r['n_pools']}")

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
               'pool_constant': pc, 'nuisance': nb, 'probe': probe, 'estimand_bias': eb,
               'odour_split': od}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(OUT / 'derm.json', 'w'), indent=1)
    print(f"\nwrote {OUT / 'derm.json'}  ({len(present)} arms, {len(leak)} leak AUCs)")


if __name__ == '__main__':
    main()
