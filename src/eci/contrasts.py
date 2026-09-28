"""
Video-level contrasts for Neural Effect Search (NES) on mice v1 SAE codes.

Builds the unit-level matrices that src/eci/nes.py consumes:
  * per-video summaries of each neuron's per-frame activation, streamed from the (2.59M, m)
    float16 memmap of per-frame SAE codes (rows in dataset/mice/v1/annotations.csv order):
      mean  = mean activation over the frames of a window,
      rate  = fraction of frames in the window with activation > 0 (firing rate);
  * paired stage transitions (unit = pool, same pool at stage a and b, one genotype);
  * genotype contrasts within one stage (unit = video, het = 1 vs wt = 0).

Design (data/mice/DATA_STRUCTURE.md): 72 pools x 6 stages, stage 1..6 = (S,H) (S,O) (S,P)
(F,H) (F,O) (F,P); a pool is one genotype. At 5 fps habituation is 9000 frames (30 min),
odor and post are 4500 frames (15 min).

Windows (per video):
  full   all frames
  last   the last `n_match` frames (time-matched sensitivity for H -> O transitions: the half of
         habituation adjacent to odor onset, same frame count as the odor stage)
  trim   all frames except the first `n_trim` (default 150 = 30 s): drops the start-of-video
         handling artefacts (white card, experimenter hand) seen in some neurons
"""

from pathlib import Path

import numpy as np
import pandas as pd

STAGES = {('S', 'H'): 1, ('S', 'O'): 2, ('S', 'P'): 3, ('F', 'H'): 4, ('F', 'O'): 5, ('F', 'P'): 6}
TRANSITIONS = {'1to2': (1, 2), '2to3': (2, 3), '4to5': (4, 5), '5to6': (5, 6)}
WINDOWS = ('full', 'last', 'trim')


def load_design(annotations_csv, experiment_csv):
    """One row per observation in annotations.csv block order: observation_id, pool, genotype,
    T (het=1), stage, row_start, row_end (half-open rows into the codes memmap), n_frames."""
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


def frame_matrix(codes_path, design, stage, prefix=None):
    """Raw per-frame codes (n_frames, prefix) float64 of all videos of a stage, with T per frame
    and the video index per frame (for the pseudo-replication illustration)."""
    Z = np.load(codes_path, mmap_mode='r')
    d = design[design['stage'] == stage]
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


def bout_stats(above, gap=0):
    """above: (n_frames, m) bool. Returns dict of (m,) arrays: count (bouts), frames (above
    frames), covered (frames inside bouts, incl. merged gaps), median_len (frames, NaN if 0 bouts)."""
    n, m = above.shape
    pad = np.zeros((m, n + 2), dtype=np.int8)
    pad[:, 1:-1] = above.T
    d = np.diff(pad, axis=1)
    sc, st = np.nonzero(d == 1)    # sorted by column, then frame
    ec, en = np.nonzero(d == -1)   # run ends (exclusive), aligned with the starts
    frames = above.sum(0).astype(np.float64)
    if gap > 0 and len(st) > 1:
        same = sc[1:] == sc[:-1]
        g = st[1:] - en[:-1]
        merge = same & (g <= gap)
        keep_start = np.r_[True, ~merge]  # a bout starts where the previous run is not merged into it
        keep_end = np.r_[~merge, True]
        sc, st, en = sc[keep_start], st[keep_start], en[keep_end]
    L = en - st
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


def bout_summaries(codes_path, design, thresholds, configs, n_match, n_trim=150):
    """Stream the memmap once by observation (contiguous rows, checked in load_design).
    thresholds: dict q -> (m,) array; configs: list of (q, gap). Returns dict
    {(window, q, gap, stat): (n_obs, m) float64} for windows full/last/trim and BOUT_STATS,
    plus {(window, 'n_frames'): (n_obs,)}."""
    Z = np.load(codes_path, mmap_mode='r')
    n_obs, m = len(design), Z.shape[1]
    out = {(w, q, g, s): np.zeros((n_obs, m)) for w in WINDOWS for q, g in configs for s in BOUT_STATS}
    for w in WINDOWS:
        out[(w, 'n_frames')] = np.zeros(n_obs)
    sls = (('full', slice(None)), ('last', slice(-n_match, None)), ('trim', slice(n_trim, None)))
    for i, (s, e) in enumerate(zip(design['row_start'], design['row_end'])):
        X = np.asarray(Z[s:e])
        for w, sl in sls:
            Xw = X[sl]
            out[(w, 'n_frames')][i] = len(Xw)
            for q in sorted({q for q, _ in configs}):
                above = Xw > thresholds[q].astype(Xw.dtype)
                for g in [g for qq, g in configs if qq == q]:
                    for k, v in bout_stats(above, g).items():
                        out[(w, q, g, k)][i] = v
    return out


def _bkey(k):
    return '__'.join(str(x) for x in k)


def cached_bout_summaries(codes_path, design, cache_path, thresholds, configs, n_match, n_trim=150):
    """bout_summaries with an npz cache keyed on observation order, thresholds, configs, windows."""
    cache_path = Path(cache_path)
    keys = [(w, q, g, s) for w in WINDOWS for q, g in configs for s in BOUT_STATS] + [(w, 'n_frames') for w in WINDOWS]
    thr = np.stack([thresholds[q] for q, _ in configs])
    if cache_path.exists():
        f = np.load(cache_path, allow_pickle=False)
        if np.array_equal(f['observation_id'], design['observation_id'].values.astype(str)) \
                and int(f['n_match']) == n_match and int(f['n_trim']) == n_trim \
                and np.array_equal(f['configs'], np.array(configs, dtype=np.float64)) and np.array_equal(f['thr'], thr):
            return {k: f[_bkey(k)] for k in keys}
    out = bout_summaries(codes_path, design, thresholds, configs, n_match, n_trim)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, observation_id=design['observation_id'].values.astype(str), n_match=n_match, n_trim=n_trim,
             configs=np.array(configs, dtype=np.float64), thr=thr, **{_bkey(k): v for k, v in out.items()})
    return out


def bout_outcomes(bs, window, q, gap, fps=5.0):
    """Video-level outcome matrices (n_obs, m) from bout_summaries for one window/q/gap:
    rate (bouts per minute), mean_dur (s, NaN at 0 bouts), median_dur (s, NaN at 0 bouts),
    frac (fraction of window frames above threshold)."""
    nf = bs[(window, 'n_frames')][:, None]
    c = bs[(window, q, gap, 'count')]
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'rate': c / (nf / fps / 60.0),
                'mean_dur': np.where(c > 0, bs[(window, q, gap, 'covered')] / c, np.nan) / fps,
                'median_dur': bs[(window, q, gap, 'median_len')] / fps,
                'frac': bs[(window, q, gap, 'frames')] / nf}
