"""
Video-level contrasts for Neural Effect Search (NES) on mice v1 SAE codes.

Builds the unit-level matrices that src/eci/nes.py consumes:
  * per-video summaries of each neuron's per-frame activation, streamed from the (2.59M, m)
    float16 memmap of per-frame SAE codes (rows in dataset/mice/v1/annotations.csv order):
      mean  = mean activation over the frames of a window,
      rate  = fraction of frames in the window with activation > 0 (firing rate);
  * paired stage transitions (unit = pool, same pool at stage a and b, one genotype);
  * genotype contrasts within one stage (unit = video, het = 1 vs wt = 0);
  * generic two-sample contrasts (two_sample: the rows of one analysis, src/eci/domain.py
    Analysis.select, T = 1 treatment / 0 control), e.g. the ants treatment contrasts
    (load_design_ants: one row per video of dataset/ants/eci/annotations.csv, unit = video).

Design (data/mice/DATA_STRUCTURE.md): 72 pools x 6 stages, stage 1..6 = (S,H) (S,O) (S,P)
(F,H) (F,O) (F,P); a pool is one genotype. At 5 fps habituation is 9000 frames (30 min),
odor and post are 4500 frames (15 min).

Windows (per video):
  full   all frames
  last   the last `n_match` frames (time-matched sensitivity for H -> O transitions: the half of
         habituation adjacent to odor onset, same frame count as the odor stage)
  trim   all frames except the first `n_trim` (default 150 = 30 s): drops the start-of-video
         handling artefacts (white card, experimenter hand) seen in some neurons
These per-video windows never change; which one each side of an analysis uses is the domain's choice
(src/eci/domain.py Domain.window_map / video_window). Mice since 2026-10-07: habituation = 'last' (its last
15 min) in the primary, odor / post = 'full'.
"""

from pathlib import Path

import numpy as np
import pandas as pd

STAGES = {('S', 'H'): 1, ('S', 'O'): 2, ('S', 'P'): 3, ('F', 'H'): 4, ('F', 'O'): 5, ('F', 'P'): 6}
TRANSITIONS = {'1to2': (1, 2), '2to3': (2, 3), '4to5': (4, 5), '5to6': (5, 6)}
WINDOWS = ('full', 'last', 'trim')


def observation_blocks(annotations_csv):
    """One row per observation in annotations.csv block order: observation_id, row_start, row_end
    (half-open rows into the codes memmap). Raises unless every observation is one contiguous block
    in increasing frame_idx order."""
    a = pd.read_csv(annotations_csv, usecols=['observation_id', 'frame_idx'])
    oid = a['observation_id'].values
    brk = np.flatnonzero(oid[1:] != oid[:-1]) + 1
    starts, ends = np.r_[0, brk], np.r_[brk, len(oid)]
    blocks = pd.DataFrame({'observation_id': oid[starts], 'row_start': starts, 'row_end': ends})
    if blocks['observation_id'].duplicated().any():
        raise ValueError('an observation is split over non-contiguous row blocks')
    fi = a['frame_idx'].values
    for s, e in zip(starts, ends):
        if not (np.diff(fi[s:e]) > 0).all():
            raise ValueError('frames are not in increasing frame_idx order within an observation')
    return blocks


def load_design(annotations_csv, experiment_csv):
    """One row per observation in annotations.csv block order: observation_id, pool, genotype,
    T (het=1), stage, row_start, row_end (half-open rows into the codes memmap), n_frames."""
    blocks = observation_blocks(annotations_csv)
    e = pd.read_csv(experiment_csv)
    e['stage'] = [STAGES[(o, p)] for o, p in zip(e['odor'], e['phase'])]
    e['T'] = (e['genotype'] == 'het').astype(int)
    d = blocks.merge(e[['observation_id', 'pool', 'genotype', 'T', 'stage', 'line', 'sex']], on='observation_id',
                     how='left', validate='1:1')
    if d['pool'].isna().any():
        raise ValueError('observations missing from experiment.csv')
    d['n_frames'] = d['row_end'] - d['row_start']
    d['obs_row'] = np.arange(len(d))  # row into the video_summaries arrays
    return d


def load_design_ants(annotations_csv, experiment_csv):
    """Ants (dataset/ants/eci/, scripts/eci/ants_prepare.py): one row per video in annotations.csv
    block order: observation_id, experiment (v2 / v3), T (raw treatment value), batch, position,
    annotator, recording_date, nestbox (v3 only), row_start, row_end, n_frames, obs_row."""
    blocks = observation_blocks(annotations_csv)
    e = pd.read_csv(experiment_csv, dtype={'batch': str, 'position': str, 'nestbox': str})
    d = blocks.merge(e, on='observation_id', how='left', validate='1:1')
    if d['T'].isna().any():
        raise ValueError('observations missing from experiment.csv')
    d['T'] = d['T'].astype(int)
    d['n_frames'] = d['row_end'] - d['row_start']
    d['obs_row'] = np.arange(len(d))
    return d


LINES = ('ash1l', 'kdm6b', 'kmt5b')
SEXES = ('f', 'm')


def subset_name(line='all', sex='all'):
    """'<line>_<sex>' ('all' = that dimension unrestricted), e.g. 'ash1l_all', 'all_f'."""
    return f'{line}_{sex}'


def subset_design(design, line='all', sex='all'):
    """The observations of the pools of one gene line and / or sex ('all' = no restriction).
    Rows keep their obs_row (row into the per-video summary arrays, which stay full-cohort), so
    per-video caches and n_fg must be computed on the full design and indexed with obs_row."""
    if line not in ('all',) + LINES or sex not in ('all',) + SEXES:
        raise ValueError(f'unknown subgroup line={line!r} sex={sex!r}')
    m = np.ones(len(design), bool)
    if line != 'all':
        m &= (design['line'] == line).values
    if sex != 'all':
        m &= (design['sex'] == sex).values
    return design[m].copy()


def video_summaries(codes_path, design, n_match, n_trim=150):
    """Stream the memmap once; returns dict {(window, stat): (n_obs, m) float64} for windows
    full/last/trim and stats mean/rate (activation > 0)."""
    Z = np.load(codes_path, mmap_mode='r')
    n_obs, m = len(design), Z.shape[1]
    out = {(w, s): np.zeros((n_obs, m)) for w in WINDOWS for s in ('mean', 'rate')}
    for i, (s, e) in enumerate(zip(design['row_start'], design['row_end'])):
        X = np.asarray(Z[s:e], dtype=np.float32)
        for w, sl in (('full', slice(None)), ('last', slice(-n_match, None)), ('trim', slice(n_trim, None))):
            Xw = X[sl]
            out[(w, 'mean')][i] = Xw.mean(0, dtype=np.float64)
            out[(w, 'rate')][i] = (Xw > 0).mean(0)
    return out


def per_video(values, wins):
    """values: {contrasts window: (n_obs, ...) array}; wins: (n_obs,) window per video (Domain.video_window)
    -> (n_obs, ...) array whose row i is values[wins[i]][i]."""
    wins = np.asarray(wins, dtype=object)
    out = np.array(values[wins[0]], copy=True)
    for w in set(wins):
        m = wins == w
        out[m] = values[w][m]
    return out


def cached_summaries(codes_path, design, cache_path, n_match, n_trim=150):
    cache_path = Path(cache_path)
    keys = [(w, s) for w in WINDOWS for s in ('mean', 'rate')]
    if cache_path.exists():
        f = np.load(cache_path, allow_pickle=False)
        if np.array_equal(f['observation_id'], design['observation_id'].values.astype(str)) and int(f['n_match']) == n_match \
                and 'n_trim' in f and int(f['n_trim']) == n_trim:
            return {k: f[f'{k[0]}__{k[1]}'] for k in keys}
    out = video_summaries(codes_path, design, n_match, n_trim)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, observation_id=design['observation_id'].values.astype(str), n_match=n_match, n_trim=n_trim,
             **{f'{k[0]}__{k[1]}': v for k, v in out.items()})
    return out


def paired(summ, design, genotype, a, b, stat='mean', window_a='full', window_b='full', prefix=None):
    """Za, Zb (n_pools, prefix) of one genotype, rows aligned by pool; returns (pools, Za, Zb)."""
    d = design[design['genotype'] == genotype]
    ia = d[d['stage'] == a].set_index('pool')
    ib = d[d['stage'] == b].set_index('pool')
    pools = sorted(set(ia.index) & set(ib.index))
    if len(pools) != len(ia) or len(pools) != len(ib):
        raise ValueError(f'unpaired pools for {genotype} {a}->{b}')
    ra, rb = ia.loc[pools, 'obs_row'].values, ib.loc[pools, 'obs_row'].values
    Za, Zb = summ[(window_a, stat)][ra], summ[(window_b, stat)][rb]
    if prefix is not None:
        Za, Zb = Za[:, :prefix], Zb[:, :prefix]
    return pools, Za, Zb


def genotype_contrast(summ, design, stage, stat='mean', window='full', prefix=None):
    """Z (n_videos, prefix), T (het=1) of one stage; returns (pools, Z, T)."""
    d = design[design['stage'] == stage].sort_values('pool')
    Z = summ[(window, stat)][d['obs_row'].values]
    if prefix is not None:
        Z = Z[:, :prefix]
    return d['pool'].values, Z, d['T'].values


def two_sample(summ, rows, stat='mean', window='full', prefix=None, unit='pool'):
    """Z (n_units, prefix), T of the design rows of one two-sample analysis (T = 1 treatment,
    0 control; src/eci/domain.py Analysis.select), sorted by unit; returns (units, Z, T).
    genotype_contrast(summ, design, s) == two_sample(summ, design rows of stage s)."""
    d = rows.sort_values(unit)
    Z = summ[(window, stat)][d['obs_row'].values]
    if prefix is not None:
        Z = Z[:, :prefix]
    return d[unit].values, Z, d['T'].values


def frame_matrix(codes_path, design, stage, prefix=None):
    """Raw per-frame codes (n_frames, prefix) float64 of all videos of a stage, with T per frame
    and the video index per frame (for the pseudo-replication illustration)."""
    return frame_matrix_rows(codes_path, design[design['stage'] == stage], prefix)


def frame_matrix_rows(codes_path, d, prefix=None):
    """frame_matrix of the videos of the design rows d (design order, column T)."""
    Z = np.load(codes_path, mmap_mode='r')
    m = Z.shape[1] if prefix is None else prefix
    parts, T, g = [], [], []
    for k, (s, e, t) in enumerate(zip(d['row_start'], d['row_end'], d['T'])):
        parts.append(np.asarray(Z[s:e, :m], dtype=np.float64))
        T.append(np.full(e - s, t)), g.append(np.full(e - s, k))
    return np.vstack(parts), np.concatenate(T), np.concatenate(g)


def to_jsonable(x):
    """NES result dicts (DataFrames, numpy scalars/arrays) -> plain JSON types."""
    if isinstance(x, pd.DataFrame):
        return {c: to_jsonable(x[c].values) for c in x.columns}
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ---------------------------------------------------------------------------------------------
# Bout outcomes (per-frame codes_max above a per-neuron, treatment-agnostic threshold)
#
# threshold  thr_j(q) = q-quantile of codes_max[:, j] pooled over a fixed random subsample of
#            frames of ALL videos (every stage, both genotypes): it never sees T or stage.
# above      a frame is "above" when codes_max > thr_j (strict), so a neuron whose q-quantile is
#            0 counts any activation > 0; a neuron never above its threshold has 0 bouts.
# bout       a maximal run of consecutive above frames inside one video window (min length
#            1 frame). A run cut by the window edge counts as a bout of its in-window length.
# merge gap  g >= 0: two bouts separated by <= g below frames are merged into one bout (the gap
#            frames count towards its duration). Primary g = 0 (no merging).
# hysteresis optional sensitivity, config (q, gap, q_exit, min_len) instead of (q, gap): a bout
#            starts at a frame > thr_j(q) and continues while codes_max > thr_j(q_exit) (q_exit <= q;
#            forward-only Schmitt trigger: the frames above thr(q_exit) BEFORE the entry frame are not
#            part of the bout); then bouts shorter than min_len frames are discarded. 'frames' is
#            then the number of frames inside the kept bouts. (q, gap) == (q, gap, q, 1).
# outcomes   per video x neuron: bout count, frames above, bout rate = count / window minutes,
#            mean bout duration = covered frames / count / fps seconds (NaN when count = 0),
#            median bout duration (s, NaN when count = 0).
# ---------------------------------------------------------------------------------------------
BOUT_STATS = ('count', 'frames', 'covered', 'median_len')


def pooled_thresholds(codes_path, qs, n_sample=2_000_000, seed=0, col_chunk=64):
    """(len(qs), m) per-neuron quantiles of the codes over a fixed random subsample of rows
    (sorted, without replacement; all rows when n_sample >= n). Treatment-agnostic by design."""
    Z = np.load(codes_path, mmap_mode='r')
    n, m = Z.shape
    rows = np.arange(n) if n_sample >= n else np.sort(np.random.default_rng(seed).choice(n, n_sample, replace=False))
    sub = np.empty((len(rows), m), dtype=Z.dtype)
    step = 200_000  # read in contiguous spans so the memmap is streamed, not randomly paged
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        a, b = np.searchsorted(rows, [lo, hi])
        if b > a:
            sub[a:b] = np.asarray(Z[lo:hi])[rows[a:b] - lo]
    out = np.empty((len(qs), m))
    for c in range(0, m, col_chunk):
        out[:, c:c + col_chunk] = np.quantile(sub[:, c:c + col_chunk].astype(np.float32), qs, axis=0)
    return out, len(rows)


def bout_stats(above, gap=0, enter=None, min_len=1):
    """above: (n_frames, m) bool. Returns dict of (m,) arrays: count (bouts), frames (above
    frames), covered (frames inside bouts, incl. merged gaps), median_len (frames, NaN if 0 bouts).
    enter: optional (n_frames, m) bool (hysteresis, `above` is then the exit mask, enter must imply
    above): a run of `above` holds a bout from its first `enter` frame to the run end, runs without
    an `enter` frame hold none. min_len: bouts shorter than min_len frames (after merging) are
    dropped. With enter=None and min_len=1 the result is the plain run count (unchanged)."""
    n, m = above.shape
    pad = np.zeros((m, n + 2), dtype=np.int8)
    pad[:, 1:-1] = above.T
    d = np.diff(pad, axis=1)
    sc, st = np.nonzero(d == 1)    # sorted by column, then frame
    ec, en = np.nonzero(d == -1)   # run ends (exclusive), aligned with the starts
    frames = above.sum(0).astype(np.float64)
    if enter is not None:
        if (enter & ~above).any():
            raise ValueError('hysteresis: enter frames must be above the exit threshold')
        fc, ff = np.nonzero(enter.T)  # sorted by column, then frame
        W = n + 2
        ek = fc.astype(np.int64) * W + ff
        i = np.searchsorted(ek, sc.astype(np.int64) * W + st)  # first enter frame at or after the run start
        ok = i < len(ek)
        first = np.where(ok, ek[np.minimum(i, len(ek) - 1)], -1)
        ok &= first < sc.astype(np.int64) * W + en
        sc, st, en = sc[ok], (first - sc.astype(np.int64) * W)[ok], en[ok]
    if gap > 0 and len(st) > 1:
        same = sc[1:] == sc[:-1]
        g = st[1:] - en[:-1]
        merge = same & (g <= gap)
        keep_start = np.r_[True, ~merge]  # a bout starts where the previous run is not merged into it
        keep_end = np.r_[~merge, True]
        sc, st, en = sc[keep_start], st[keep_start], en[keep_end]
    if min_len > 1:
        k = (en - st) >= min_len
        sc, st, en = sc[k], st[k], en[k]
    L = en - st
    if enter is not None or min_len > 1:
        frames = np.bincount(sc, weights=L, minlength=m).astype(np.float64) if gap == 0 else frames
    count = np.bincount(sc, minlength=m).astype(np.float64)
    covered = np.bincount(sc, weights=L, minlength=m)
    med = np.full(m, np.nan)
    if len(L):
        o = np.lexsort((L, sc))
        Ls, cs = L[o], sc[o]
        first = np.searchsorted(cs, np.arange(m), 'left')
        cnt = count.astype(np.int64)
        has = cnt > 0
        lo = first[has] + (cnt[has] - 1) // 2
        hi = first[has] + cnt[has] // 2
        med[has] = (Ls[lo] + Ls[hi]) / 2
    return {'count': count, 'frames': frames, 'covered': covered, 'median_len': med}


def _cfg4(c):
    """(q, gap) or (q, gap, q_exit, min_len) -> the 4-tuple."""
    return tuple(c) if len(c) == 4 else (c[0], c[1], c[0], 1)


def bout_summaries(codes_path, design, thresholds, configs, n_match, n_trim=150):
    """Stream the memmap once by observation (contiguous rows, checked in load_design).
    thresholds: dict q -> (m,) array; configs: list of (q, gap) or (q, gap, q_exit, min_len)
    (hysteresis, see bout_stats; q_exit must be a key of thresholds). Returns dict
    {(window, *config, stat): (n_obs, m) float64} for windows full/last/trim and BOUT_STATS,
    plus {(window, 'n_frames'): (n_obs,)}."""
    Z = np.load(codes_path, mmap_mode='r')
    n_obs, m = len(design), Z.shape[1]
    configs = [tuple(c) for c in configs]
    out = {(w, *c, s): np.zeros((n_obs, m)) for w in WINDOWS for c in configs for s in BOUT_STATS}
    for w in WINDOWS:
        out[(w, 'n_frames')] = np.zeros(n_obs)
    sls = (('full', slice(None)), ('last', slice(-n_match, None)), ('trim', slice(n_trim, None)))
    qs = sorted({q for c in configs for q in (_cfg4(c)[0], _cfg4(c)[2])})
    for i, (s, e) in enumerate(zip(design['row_start'], design['row_end'])):
        X = np.asarray(Z[s:e])
        for w, sl in sls:
            Xw = X[sl]
            out[(w, 'n_frames')][i] = len(Xw)
            above = {q: Xw > thresholds[q].astype(Xw.dtype) for q in qs}
            for c in configs:
                q, g, qx, ml = _cfg4(c)
                if len(c) == 2:
                    r = bout_stats(above[q], g)
                else:
                    r = bout_stats(above[qx], g, enter=None if qx == q else above[q] & above[qx], min_len=int(ml))
                for k, v in r.items():
                    out[(w, *c, k)][i] = v
    return out


def _bkey(k):
    return '__'.join(str(x) for x in k)


def cached_bout_summaries(codes_path, design, cache_path, thresholds, configs, n_match, n_trim=150):
    """bout_summaries with an npz cache keyed on observation order, thresholds, configs, windows.
    Plain (q, gap) config lists keep the original cache key; lists with hysteresis configs are
    keyed on the 4-tuples."""
    cache_path = Path(cache_path)
    configs = [tuple(c) for c in configs]
    keys = [(w, *c, s) for w in WINDOWS for c in configs for s in BOUT_STATS] + [(w, 'n_frames') for w in WINDOWS]
    plain = all(len(c) == 2 for c in configs)
    carr = np.array(configs if plain else [_cfg4(c) + (len(c),) for c in configs], dtype=np.float64)
    thr = np.stack([thresholds[q] for q in sorted({q for c in configs for q in (_cfg4(c)[0], _cfg4(c)[2])})]) \
        if not plain else np.stack([thresholds[c[0]] for c in configs])
    if cache_path.exists():
        f = np.load(cache_path, allow_pickle=False)
        if np.array_equal(f['observation_id'], design['observation_id'].values.astype(str)) \
                and int(f['n_match']) == n_match and int(f['n_trim']) == n_trim \
                and f['configs'].shape == carr.shape and np.array_equal(f['configs'], carr) \
                and f['thr'].shape == thr.shape and np.array_equal(f['thr'], thr):
            return {k: f[_bkey(k)] for k in keys}
    out = bout_summaries(codes_path, design, thresholds, configs, n_match, n_trim)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, observation_id=design['observation_id'].values.astype(str), n_match=n_match, n_trim=n_trim,
             configs=carr, thr=thr, **{_bkey(k): v for k, v in out.items()})
    return out


def bout_outcomes(bs, window, q, gap, fps=5.0, q_exit=None, min_len=None):
    """Video-level outcome matrices (n_obs, m) from bout_summaries for one window/q/gap (and, for
    a hysteresis config, q_exit / min_len): rate (bouts per minute), mean_dur (s, NaN at 0 bouts),
    median_dur (s, NaN at 0 bouts), frac (fraction of window frames above threshold / in bouts)."""
    c = (q, gap) if q_exit is None and min_len is None else (q, gap, q_exit, min_len)
    nf = bs[(window, 'n_frames')][:, None]
    cnt = bs[(window, *c, 'count')]
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'rate': cnt / (nf / fps / 60.0),
                'mean_dur': np.where(cnt > 0, bs[(window, *c, 'covered')] / cnt, np.nan) / fps,
                'median_dur': bs[(window, *c, 'median_len')] / fps,
                'frac': bs[(window, *c, 'frames')] / nf}


# ---------------------------------------------------------------------------------------------
# Size covariate (foreground SAEs): per-video mean number of foreground patches n_fg, a proxy for
# how spread out / huddled the mice are. size_adjusted_test re-tests one neuron with it:
#   paired      D_j = Zb_j - Za_j regressed on 1 + (n_b - n_a) across pools (OLS, df = n - 2);
#               tau = intercept = the change not explained by the change in foreground size.
#   two-sample  Z_j = a + tau T + b n_fg (OLS, HC2 standard errors, df = n - 3).
# ---------------------------------------------------------------------------------------------
def video_nfg(nfg_path, design, n_match, n_trim=150):
    """{window: (n_obs,) float64} per-video mean of the per-frame foreground patch count."""
    nf = np.load(nfg_path, mmap_mode='r')
    out = {w: np.zeros(len(design)) for w in WINDOWS}
    for i, (s, e) in enumerate(zip(design['row_start'], design['row_end'])):
        x = np.asarray(nf[s:e], dtype=np.float64)
        out['full'][i], out['last'][i], out['trim'][i] = x.mean(), x[-n_match:].mean(), x[n_trim:].mean()
    return out


def size_adjusted_test(y_or_ya, cov_or_ca, T_or_yb=None, cb=None, paired_design=False):
    """paired_design=True: (ya, ca, yb, cb) per pool -> dict tau, se, t, df, p, slope.
    Else: (y, cov, T) per video."""
    from scipy import stats
    if paired_design:
        ya, ca, yb = map(np.asarray, (y_or_ya, cov_or_ca, T_or_yb))
        y, X = yb - ya, np.column_stack([np.ones(len(ya)), np.asarray(cb) - ca])
    else:
        y, c, T = map(np.asarray, (y_or_ya, cov_or_ca, T_or_yb))
        X = np.column_stack([np.ones(len(y)), T, c])
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    idx = 0 if paired_design else 1
    if paired_design:
        V = XtXi * (e @ e) / (n - k)
    else:
        h = np.einsum('ij,jk,ik->i', X, XtXi, X)
        V = XtXi @ (X.T * (e ** 2 / np.clip(1 - h, 1e-12, None))) @ X @ XtXi
    se = float(np.sqrt(max(V[idx, idx], 0)))
    t = float(b[idx] / se) if se > 0 else 0.0
    return {'tau': float(b[idx]), 'se': se, 't': t, 'df': n - k, 'p': float(2 * stats.t.sf(abs(t), n - k)),
            'slope': float(b[-1])}


def size_adjusted_round1(r1, design, values, nfg, window_map, analyses=None):
    """Re-test round-1 neurons with the per-video mean foreground size as a covariate.

    r1: DataFrame of round-1 rows (analysis_id, prefix, window, neuron, tau, p, threshold, ...);
    values: {contrasts window ('full'/'last'/'trim'): (n_obs, m) outcome matrix};
    nfg: video_nfg output; window_map: runner window name -> (window a, window b), or a callable Analysis -> such a
    dict (src/eci/domain.py Domain.window_map; needs analyses);
    analyses: {analysis_id: src/eci/domain.py Analysis} (None: the mice ids A_<g>_<tr> / B_stage<s>).
    Returns r1 with tau_adj, se_adj, p_adj, slope_nfg, survives (p_adj < the round-1 threshold and
    the same sign as tau)."""
    summ = {(w, 'v'): v for w, v in values.items()}
    cov = {(w, 'v'): nfg[w][:, None] for w in nfg}
    out = []
    for r in r1.to_dict('records'):
        an = analyses[r['analysis_id']] if analyses is not None else None
        j, (wa, wb) = int(r['neuron']), (window_map(an) if callable(window_map) else window_map)[r['window']]
        if (an.family == 'A') if an is not None else r['analysis_id'].startswith('A'):
            if an is not None:
                g, (a, b) = an.genotype, an.stages
            else:
                _, g, tr = r['analysis_id'].split('_')
                a, b = TRANSITIONS[tr]
            _, Za, Zb = paired(summ, design, g, a, b, 'v', wa, wb)
            _, Ca, Cb = paired(cov, design, g, a, b, 'v', wa, wb)
            res = size_adjusted_test(Za[:, j], Ca[:, 0], Zb[:, j], Cb[:, 0], paired_design=True)
        elif an is not None:
            rows = an.select(design)
            _, Z, T = two_sample(summ, rows, 'v', wa, unit=an.unit)
            _, Cv, _ = two_sample(cov, rows, 'v', wa, unit=an.unit)
            res = size_adjusted_test(Z[:, j], Cv[:, 0], T)
        else:
            _, Z, T = genotype_contrast(summ, design, int(r['analysis_id'][len('B_stage'):]), 'v', wa)
            _, Cv, _ = genotype_contrast(cov, design, int(r['analysis_id'][len('B_stage'):]), 'v', wa)
            res = size_adjusted_test(Z[:, j], Cv[:, 0], T)
        out.append({**r, 'tau_adj': res['tau'], 'se_adj': res['se'], 'p_adj': res['p'], 'df_adj': res['df'],
                    'slope_nfg': res['slope'],
                    'survives': bool(res['p'] < r['threshold'] and np.sign(res['tau']) == np.sign(r['tau']))})
    return pd.DataFrame(out)
