"""
Shared analysis for the v0 audit, round 2 (audit2.py = our re-implementation, orig_analyse.py = the original
repo's own s2/s3/s4 scripts run on our data). PRE-REGISTERED 2026-10-20, before any round-2 score was seen.

Discovery path (matches the paper; NO split anywhere): the SAE is trained on all 1200 frames of all 44 valid
videos; the treatment test uses all 44 videos (20 treated, 24 control).
  aggregation  'mean' (primary, the notebook's per-video mean of the code) and 'rate' (secondary: fraction of
               the video's frames with code > 0)
  test         single-pass Welch t-test per code, raw p < 0.05 (primary: the paper's published 394/550 came out
               of this) + the original recursive neural_effect_search (method 't-test'), reported next to it;
               Bonferroni count over the testable codes.
Alignment with grooming (label = yellow OR blue grooms the focal ant), over ALL alive codes of a run
(alive = code > 0 on at least one frame):
  AUROC (primary; threshold-free, signed: > 0.5 = more active during grooming), R~ = normalised R-precision
  (R = number of grooming frames; ties at the cut broken in expectation, 0 = chance, 1 = perfect), F1 of the
  rule code > 0 (the paper's prf_rule_sign0_multi).
  For each NES hit: percentile of its AUROC / R~ / F1 among the run's alive codes (ties count half).
  "Much more aligned than average" := AUROC percentile >= 95 within its run.
  Hit-set test: mean AUROC percentile of the run's hits vs 10,000 random alive subsets of the same size
  (one-sided p); pooled over runs by summing the per-run statistics (same null).
Cross-fitting is NOT part of discovery; it is only the separate alignment diagnostic kept from round 1.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_ind

N_FRAMES = 1200


def auroc_cols(codes, y, cols):
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    out = np.full(codes.shape[1], np.nan)
    for j in cols:
        r = rankdata(codes[:, j])
        out[j] = (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def rtilde_cols(codes, y, cols):
    R = int(y.sum())
    prev = y.mean()
    out = np.full(codes.shape[1], np.nan)
    for j in cols:
        c = codes[:, j]
        v = np.partition(c, len(c) - R)[len(c) - R]  # R-th largest value
        gt, eq = c > v, c == v
        need = R - int(gt.sum())
        exp_pos = y[gt].sum() + need * y[eq].mean()
        out[j] = (exp_pos / R - prev) / (1 - prev)
    return out


def f1_cols(act, y):
    pos = y == 1
    tp = act[pos].sum(0).astype(np.float64)
    fp = act[~pos].sum(0).astype(np.float64)
    fn = pos.sum() - tp
    with np.errstate(invalid='ignore', divide='ignore'):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0)
        return np.where(p + r > 0, 2 * p * r / (p + r), 0)


def pct(values, x):
    v = values[np.isfinite(values)]
    return float(100 * ((v < x).sum() + 0.5 * (v == x).sum()) / len(v))


def eta2(v, g):
    v = np.asarray(v, float)
    sst = ((v - v.mean()) ** 2).sum()
    if sst == 0:
        return float('nan')
    return float(sum((g == k).sum() * (v[g == k].mean() - v.mean()) ** 2 for k in np.unique(g)) / sst)


def perm_p(v, g, rng, n=5000):
    obs = eta2(v, g)
    null = np.array([eta2(v, rng.permutation(g)) for _ in range(n)])
    return obs, float((1 + (null >= obs - 1e-12).sum()) / (1 + n))


def welch(vc, T):
    with np.errstate(invalid='ignore', divide='ignore'):
        t, p = ttest_ind(vc[T == 1], vc[T == 0], axis=0, equal_var=False)
    return t, p


def analyse(codes, videos, y, pos1, nes_fn, rng):
    """codes: (44*1200, n_codes) frame codes in CSV video order. nes_fn = original neural_effect_search."""
    nv, nc = len(videos), codes.shape[1]
    T = (videos.treatment.to_numpy() == 2).astype(int)
    position = videos.position.to_numpy()
    act = codes > 0
    alive = np.where(act.any(0))[0]
    auc = auroc_cols(codes, y, alive)
    rt = rtilde_cols(codes, y, alive)
    f1 = f1_cols(act, y)
    f1[np.setdiff1d(np.arange(nc), alive)] = np.nan
    f1pos = f1_cols(act, pos1)
    aucpos = auroc_cols(codes, pos1, alive)
    per_code = dict(alive=alive, auroc=auc, r_tilde=rt, f1=f1, f1_pos1ad=f1pos, auroc_pos1ad=aucpos)
    alive_auc = auc[alive]
    res = dict(n_alive=int(len(alive)), prevalence=float(y.mean()), always_on_f1=float(2 * y.mean() / (1 + y.mean())),
               alive_auroc_quantiles={q: float(np.quantile(alive_auc, q)) for q in (0.05, 0.5, 0.95, 0.99)},
               best_auroc=float(np.nanmax(auc)), best_f1=float(np.nanmax(f1)), best_r_tilde=float(np.nanmax(rt)))
    for agg in ['mean', 'rate']:
        vc = codes.reshape(nv, N_FRAMES, nc)
        vc = vc.astype(np.float64).mean(1) if agg == 'mean' else (vc > 0).mean(1)
        t, p = welch(vc, T)
        valid = np.isfinite(p)
        m = int(valid.sum())
        hits = sorted((int(j) for j in np.where(valid & (p < 0.05))[0]), key=lambda j: p[j])
        df = pd.DataFrame(vc, columns=[f'N_{j + 1}' for j in range(nc)])
        df.insert(0, 'T', T)
        rec = nes_fn(df, method='t-test', alpha=0.05, k=2000, neuron_prefix='N_', fallback_start_neur_index=4)
        rec_idx = [int(s) - 1 for s in rec['neurons']]
        a = dict(n_testable=m, n_hits=len(hits), expected_fp=0.05 * m, n_bonf=int((valid & (p < 0.05 / max(m, 1))).sum()),
                 recursive_hits=[f'N_{j + 1}' for j in rec_idx], recursive_p=[float(x) for x in rec['p_vals']])
        rows = []
        for j in sorted(set(hits) | set(rec_idx), key=lambda j: p[j] if np.isfinite(p[j]) else 1):
            e, pp = perm_p(vc[:, j], position, rng)
            rows.append(dict(neuron=f'N_{j + 1}', single_pass_hit=j in hits, recursive_hit=j in rec_idx,
                             p=float(p[j]), t=float(t[j]), frac_active=float(act[:, j].mean()),
                             auroc=float(auc[j]), r_tilde=float(rt[j]), f1=float(f1[j]),
                             pct_auroc=pct(alive_auc, auc[j]), pct_r_tilde=pct(rt[alive], rt[j]), pct_f1=pct(f1[alive], f1[j]),
                             f1_pos1ad=float(f1pos[j]), auroc_pos1ad=float(aucpos[j]), eta2_position=e, eta2_position_perm_p=pp))
        a['hits'] = rows
        # hit-set test on AUROC percentile (single-pass hits)
        if hits:
            pcts_alive = np.array([pct(alive_auc, x) for x in alive_auc])
            hp = np.array([pct(alive_auc, auc[j]) for j in hits])
            null = np.array([pcts_alive[rng.choice(len(alive), len(hits), replace=False)].mean() for _ in range(10000)])
            a['hitset_mean_pct_auroc'] = float(hp.mean())
            a['hitset_p_more_aligned'] = float((1 + (null >= hp.mean()).sum()) / 10001)
        res[agg] = a
    return res, per_code
