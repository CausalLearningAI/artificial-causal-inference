"""
Atlas v37 methods: DINES (src/eci/dines.py) on the per-video outcomes of an existing NES result set, and (mice only)
the genotype x phase-transition contrast family C (difference in differences) with both NES and DINES. Writes tidy
files in the format of the NES runners (scripts/eci/run_nes.py, run_nes_bouts.py, run_nes_latency.py), so the atlas
builder (scripts/eci/build_explorer.py) can read them like NES result sets.

Input: one NES result set <res> = results/vision/<domain>/<eci layout>/nes/<sae> (its _cache/ per-video caches, its
sanity.json settings and its result.json files; nothing in it is written). The NES reference searches are those the
page reads (build_explorer.rdir_of): <res>/pairs/ when that analysis-set run exists (ants; its first analyses repeat the
core ones with identical rows), else <res>. Domain from the path: mice (mice/eci),
ants (ants/<ANTS_ECI>: eci = the 10-min layout, eci_r4v2, eci_r4v3, eci_r4v4), frogs (frogs/eci).

Outcomes (the three temporal aggregations of the per-frame codes_max, read from <res>/_cache, exactly the NES runners'):
  mean      per-video mean of codes_max (video_summaries_max.npz, run_nes.py)
  bouts     bout rate, bouts / min, threshold q 0.95 (pooled treatment-agnostic quantile), merge gap 0
            (bout_summaries_max.npz, run_nes_bouts.py)
  latency   seconds from the window start to the first frame above thr(q 0.95), right-censored at the analysis's
            common window W (= the shortest window of the analysis's videos; censored videos get W) (latency.npz,
            run_nes_latency.py)

Units, Z and arms: exactly what the NES runner used for that analysis (same videos, the same per-side windows
Domain.window_map, the same prefix truncation: first k latents; the prefilter is dines' own min_active = 0.01 on
the prefix). The analysis units follow the NES run's own settings from its sanity.json: when the NES run used --day,
the family B rows keep only the recording days that hold both arms (Analysis.select under Domain.day_nuisance).

Mice habituation window (--hab-window): 'last15' = the current primary (habituation = its last 15 min, since
2026-10-07); 'whole' = the pre-2026-10-07 primary = the runners' 'hwhole' window (habituation on its whole 30 min;
for the analyses without a habituation side 'hwhole' does not exist and 'full' is the same thing). The tidy rows
keep window = 'full' (latency 'common') so that the builder's primary filter reads them; the definition used is in
the extra column hab_window and in code.json.

DINES settings (decided 2026-10-16): alpha 0.05, delta 0, gate bonferroni, learner ridge, seed 0, keep_first True,
min_active 0.01, other dines() defaults.
  two-sample (mice B_stage*, ants, frogs, mice C): T binary (1 = treatment), W = the per-unit mean foreground
      patch count n_fg in the outcome's window (C.video_nfg; always, whatever the NES run conditioned on) plus, when
      the NES run used --day (ants r4), the recording-day indicators of C.day_indicators (reference = the most common
      day of the analysis).
  paired mice family A (A_<geno>_<a>to<b>, unit = pool): DINES on the 2 videos per pool after WITHIN-POOL
      CENTRING (Z~ = Z - pool mean, the same for n_fg), T = 1 for stage b, clusters = pool id (cluster-robust SE with
      a t_{G-1} reference, cluster-level folds). This is the pool fixed-effects analogue of the paired design: after
      centring the two-sample contrast of the stage-b vs stage-a rows is the mean within-pool change, and pool levels
      cancel. (The prefilter then counts a pool's two centred values as active when the pool changes.)
  latency: Z = latency in seconds capped at the analysis's W (censored videos = W), as the page shows it.

Family C (mice only): genotype x phase transition, ids C_1to2, C_2to3, C_4to5, C_5to6 (social H->O, social O->P,
fear H->O, fear O->P). Unit = pool (72 pools, each one genotype: 36 het, 36 wt); outcome per pool and latent
D = Z(pool, stage b) - Z(pool, stage a) of the per-video outcome (each of the three aggregations; stage windows as in
family A); treatment = genotype (het = 1, wt = 0); tau > 0 = the transition change is larger in het than in wt.
  NES: nes.neural_effect_search two-sample, Bonferroni alpha 0.05, select 'p', nuisance = the per-pool difference of
       the per-video mean n_fg (stage b - stage a), conditioned on from round 0 (as run_nes.py --nuisance nfg).
  DINES: dines(D, T, W = that n_fg difference), settings as above.
  latency: W = the shortest window over all videos of both stages of both genotypes (the rule of run_nes_latency.py).

Tidy rows (one per (analysis, prefix, outcome setting) per REPORTED atom = DINES 'selected' (kept + undetermined) in
forward order, round = 1, 2, ...; one row with round 0 and neuron NaN when nothing is reported). Columns = the NES
runner's summary.csv columns for that outcome, with for DINES:
  test 't' (so the builder's default primary selects them; the method is the result-set directory), correction
  'bonferroni'; tau = the plain difference in arm means of the atom (paired A: mean within-pool change b - a; C: mean
  het change - mean wt change); se / t / df = the Welch (paired A: one-sample) statistics of that plain contrast,
  descriptive only; p = the atom's terminal max p over the non-skipped conditioning sets (status kept) or its p given
  F \\ {j} (status undetermined), column p_kind = 'max_nonskipped' / 'given_rest'; threshold = dines level
  (alpha / m); n_tested = n_tested_total = m (atoms after the prefilter); n_dropped = prefiltered atoms;
  extra columns: status (kept / undetermined), rho (partial correlation of T and the atom given F \\ {j}),
  set_aside (bool), terminal_complete (bool), method 'dines' (mice also hab_window).
Family C NES rows: the NES runner's tidy rows (round-r tau / se / t / df / p of NES), family 'C', genotype
'het_vs_wt', stage '', transition, stages ('1-2', ...), method 'nes'.

result.json [setting key, the NES runners' keys]:
  DINES: {'first_round': {'neuron', 'tau' (arm mean difference), 'p' (first-round p given W only), 'rho'},
          'selected', 'kept', 'undetermined', 'set_aside', 'n_units', 'units', 'm', 'level', ...; latency also
          'W_frames', 'W_s', 'window_frames', 'window_s' (copied from the NES latency run of that analysis when it
          exists; recomputed with the same rule otherwise, checked equal when both exist)}
  family C NES: the NES runners' result.json content (strip(res) + n_units, units, n_het, n_wt; latency W_*).
dines.json [setting]: the full dines() output (forward path, terminal records, sets) without first_round arrays.

Built-in identity check (every analysis x prefix x outcome where the NES run has that setting): NES's own first round
is recomputed on the Z built here (with the NES run's own nuisance / day settings) and its tau compared with the NES
result.json first_round tau; the largest absolute difference is in checks.csv and the run fails loud above 1e-6.

Output: results/vision/eci_atlas_v37/methods/<domain_key>/<sae>/
  dines/{summary.csv, <aid>/result.json, <aid>/dines.json, maxpool_bouts/..., maxpool_latency/...}
  familyC_nes/{C_summary.csv, <aid>/result.json, maxpool_bouts/..., maxpool_latency/...}     (mice)
  familyC_dines/{C_summary.csv, <aid>/result.json, <aid>/dines.json, ...}                    (mice)
  code.json (git HEAD, settings), checks.csv, timing.csv, tests.json (--tests)
domain_key = mice | ants_<ANTS_ECI> (ants_eci = the 10-min layout, ants_eci_r4v2, ...) | frogs.

Usage (Slurm wrapper scripts/eci/atlas_v37_methods.sh runs it from a git archive of HEAD):
    python scripts/eci/atlas_v37_methods.py --res results/vision/ants/eci_r4v2/nes/matryoshka_btk_512_k16_r4d2_s2_fnorm
    python scripts/eci/atlas_v37_methods.py --res results/vision/mice/eci/nes/matryoshka_btk_1024_k16_fg448al_s0 \
        --hab-window whole
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

OUT_ROOT = ROOT / 'results/vision/eci_atlas_v37/methods'
Q, GAP = 0.95, 0
OUTCOMES = ('mean', 'bouts', 'latency')
SUBDIR = {'mean': '', 'bouts': 'maxpool_bouts', 'latency': 'maxpool_latency'}
DINES_KW = dict(alpha=0.05, delta=0.0, gate='bonferroni', learner='ridge', seed=0, keep_first=True, min_active=0.01)
CHECK_TOL = 1e-6
C_IDS = {'1to2': (1, 2), '2to3': (2, 3), '4to5': (4, 5), '5to6': (5, 6)}


def parse_res(res):
    """-> (domain name, ANTS_ECI or None, domain_key) from the result-set path."""
    p = Path(res).resolve().relative_to((ROOT / 'results/vision').resolve()).parts
    if p[0] == 'mice' and p[1] == 'eci' and p[2] == 'nes':
        return 'mice', None, 'mice'
    if p[0] == 'frogs' and p[1] == 'eci' and p[2] == 'nes':
        return 'frogs', None, 'frogs'
    if p[0] == 'ants' and p[2] == 'nes':
        return 'ants', p[1], f'ants_{p[1]}'
    raise SystemExit(f'{res}: not a mice / ants / frogs NES result set')


# ----------------------------------------------------------------------------------------------------------------
# globals of one run (set in main, read by the forked workers)
# ----------------------------------------------------------------------------------------------------------------
G = {}


def key_of(outcome, prefix, window):
    """The NES runners' setting keys."""
    if outcome == 'mean':
        return f'p{prefix}_max_mean_t_bonferroni_{window}'
    if outcome == 'bouts':
        return f'p{prefix}_bout_rate_q{Q:.2f}_g{GAP}_t_bonferroni_{window}'
    return f'p{prefix}_latency_q{Q:.2f}_t_bonferroni_{window}'


def setting_cols(outcome):
    if outcome == 'mean':
        return dict(pooling='max', outcome_type='mean', test='t', correction='bonferroni')
    if outcome == 'bouts':
        return dict(pooling='max', outcome_type='bout_rate', threshold_q=Q, merge_gap=GAP, bout_rule=0, test='t',
                    correction='bonferroni')
    return dict(pooling='max', outcome_type='latency', threshold_q=Q, transform='none', test='t', correction='bonferroni')


def runner_window(an):
    """The runner window of analysis an used here ('full' or, mice --hab-window whole, 'hwhole' where it exists)."""
    D = G['D']
    if G['hab'] == 'whole' and 'hwhole' in D.window_map(an):
        return 'hwhole'
    return 'full'


def page_window(outcome):
    return 'common' if outcome == 'latency' else 'full'


def nes_window(outcome, rw):
    """The window name of the NES run's setting key."""
    return ('common' if rw == 'full' else rw) if outcome == 'latency' else rw


def load_caches(res, dfull, fps):
    """{outcome: {(contrasts window, 'v'): (n_obs, m)}} for mean and bouts; latency raw first-above index and window
    lengths; checked against the design's observation order."""
    from eci import contrasts as C
    oid = dfull['observation_id'].values.astype(str)
    cache = Path(res) / '_cache'
    f = np.load(cache / 'video_summaries_max.npz')
    if not np.array_equal(f['observation_id'], oid):
        raise SystemExit(f'{cache}/video_summaries_max.npz: observation order differs from the design')
    mean = {(w, 'v'): f[f'{w}__mean'] for w in C.WINDOWS}
    f = np.load(cache / 'bout_summaries_max.npz')
    if not np.array_equal(f['observation_id'], oid):
        raise SystemExit(f'{cache}/bout_summaries_max.npz: observation order differs from the design')
    bouts = {(w, 'v'): f[f'{w}__{Q:g}__{GAP}__count'] / (f[f'{w}__n_frames'][:, None] / fps / 60.0) for w in C.WINDOWS}
    f = np.load(cache / 'latency.npz')
    if not np.array_equal(f['observation_id'], oid):
        raise SystemExit(f'{cache}/latency.npz: observation order differs from the design')
    lat = {w: f[f'{w}__{Q:.2f}'] for w in C.WINDOWS}
    nf = {w: f[f'nf__{w}'] for w in C.WINDOWS}
    return mean, bouts, lat, nf


def window_len(rows_a, rows_b, wa, wb):
    nf = G['nf']
    return int(min(nf[wa][rows_a].min(), nf[wb][rows_b].min()))


def lat_summ(L):
    fps = G['fps']
    return {(w, 'v'): np.minimum(G['lat'][w], L) / fps for w in G['lat']}


def welch(z1, z0):
    n1, n0 = len(z1), len(z0)
    v1, v0 = z1.var(0, ddof=1) / n1, z0.var(0, ddof=1) / n0
    se = np.sqrt(v1 + v0)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(se > 0, (z1.mean(0) - z0.mean(0)) / se, 0.0)
        df = np.where(se > 0, (v1 + v0) ** 2 / (v1 ** 2 / (n1 - 1) + v0 ** 2 / (n0 - 1)), np.nan)
    return se, t, df


def one_sample(d):
    n = len(d)
    se = d.std(0, ddof=1) / np.sqrt(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(se > 0, d.mean(0) / se, 0.0)
    return se, t, np.full(d.shape[1], n - 1.0)


def jsonable(x):
    from eci import contrasts as C
    return C.to_jsonable(x)


def dines_rows(meta, out, tau, se, t, df, directions):
    """Tidy rows of one DINES search (module docstring)."""
    base = {**meta, 'n_tested_total': out['m'], 'n_dropped': out['n_dropped']}
    term = {r['atom']: r for r in out['terminal']}
    if not out['selected']:
        return [{**base, 'round': 0, 'neuron': np.nan, 'terminal_complete': bool(out['terminal_complete'])}]
    rows = []
    for i, j in enumerate(out['selected']):
        r = term[j]
        if r['status'] == 'kept':
            p, pk = r['max_p'], 'max_nonskipped'
        else:
            p, pk = (r['p_rest'], 'given_rest') if r['p_rest'] is not None else (r['max_p'], 'max_nonskipped')
        rows.append({**base, 'round': i + 1, 'neuron': int(j), 'tau': float(tau[j]), 'se': float(se[j]),
                     't': float(t[j]), 'df': float(df[j]), 'p': float(p), 'threshold': float(out['level']),
                     'n_tested': out['m'], 'direction': directions[0] if tau[j] > 0 else directions[1],
                     'status': r['status'], 'p_kind': pk, 'rho': r['rho_rest'], 'set_aside': j in out['set_aside'],
                     'terminal_complete': bool(out['terminal_complete'])})
    return rows


def nes_rows(meta, res, directions, latency):
    base = {**meta, 'n_tested_total': res['n_tested'], 'n_dropped': res['n_dropped']}
    if latency:
        base['stopped'] = res.get('stopped', '')
    if len(res['rounds']) == 0:
        return [{**base, 'round': 0}]
    return [{**base, **{k: r[k] for k in ('round', 'neuron', 'tau', 'se', 't', 'df', 'p', 'threshold', 'n_tested')},
             'direction': directions[0] if r['tau'] > 0 else directions[1]} for r in res['rounds'].to_dict('records')]


def dines_result(out, tau, units, extra):
    fr = out['first_round']
    first = {'neuron': [int(a) for a in fr['atom']], 'tau': [float(tau[a]) for a in fr['atom']],
             'p': [float(np.exp(v)) for v in fr['logp']], 'rho': [float(v) for v in fr['rho']]}
    keep = ('selected', 'kept', 'undetermined', 'set_aside', 'forward_set', 'terminal_complete', 'm', 'level',
            'n_dropped', 'delta', 'forward_capped', 'elapsed_s')
    return jsonable({'first_round': first, **{k: out[k] for k in keep}, 'n_units': len(units), 'units': list(units),
                     **extra})


def dines_audit(out):
    return jsonable({k: v for k, v in out.items() if k != 'first_round'})


# ----------------------------------------------------------------------------------------------------------------
# one task = (method family, analysis id, outcome, prefix)
# ----------------------------------------------------------------------------------------------------------------

def nes_check(Z, T, nuis, ref_first, paired=None):
    """Recompute NES round 1 on the Z built here with the NES run's own nuisance; max |tau - NES tau| over neurons."""
    from eci.nes import neural_effect_search, paired_effect_search
    if ref_first is None:
        return None
    if paired is not None:
        res = paired_effect_search(paired[0], paired[1], nuisance=nuis, max_rounds=1)
    else:
        res = neural_effect_search(Z, T, nuisance=nuis, max_rounds=1)
    fr = res['first_round']
    a = pd.Series(fr['tau'].values, index=fr['neuron'].values)
    b = pd.Series(np.asarray(ref_first['tau'], dtype=float), index=np.asarray(ref_first['neuron'], dtype=int))
    if set(a.index) != set(b.index):
        return {'max_abs_diff': float('inf'), 'n': int(len(b)), 'note': 'tested neuron sets differ'}
    d = (a - b.reindex(a.index)).abs()
    return {'max_abs_diff': float(d.max()), 'n': int(len(d))}


def nes_ref(outcome, aid, prefix, rw):
    """The NES run's result.json record of this setting (None when the run has no such setting)."""
    f = Path(G['rdir']) / SUBDIR[outcome] / aid / 'result.json'
    if not f.exists():
        return None
    k = ('nes_ref', str(f))
    if k not in G:
        G[k] = json.loads(f.read_text())
    return G[k].get(key_of(outcome, prefix, nes_window(outcome, rw)))


def task(spec):
    from eci import contrasts as C
    from eci.dines import dines
    from eci.nes import neural_effect_search
    fam, aid, outcome, prefix = spec
    t0 = time.time()
    D, design = G['D'], G['design']
    cov = G['cov_nfg']
    if fam == 'C':
        tr = aid[2:]
        a, b = C_IDS[tr]
        an_h, an_w = D.analysis(f'A_het_{tr}'), D.analysis(f'A_wt_{tr}')
        rw = runner_window(an_h)
        wa, wb = D.window_map(an_h)[rw]
        extra = {}
        if outcome == 'latency':
            ra = design.loc[design['stage'] == a, 'obs_row'].values
            rb = design.loc[design['stage'] == b, 'obs_row'].values
            L = window_len(ra, rb, wa, wb)
            sm = lat_summ(L)
            extra = {'W_frames': L, 'W_s': L / G['fps'], 'window_frames': L, 'window_s': L / G['fps']}
        else:
            sm = G[outcome]
        parts = []
        for g in ('het', 'wt'):
            pools, Za, Zb = C.paired(sm, design, g, a, b, 'v', wa, wb, prefix)
            _, Na, Nb = C.paired(cov, design, g, a, b, 'v', wa, wb)
            parts.append((pools, Zb - Za, (Nb - Na)[:, 0], np.full(len(pools), int(g == 'het'))))
        units = np.concatenate([p[0] for p in parts])
        o = np.argsort(units, kind='stable')
        units = units[o]
        Dz = np.vstack([p[1] for p in parts])[o]
        dN = np.concatenate([p[2] for p in parts])[o]
        T = np.concatenate([p[3] for p in parts])[o]
        tau = Dz[T == 1].mean(0) - Dz[T == 0].mean(0)
        se, t, df = welch(Dz[T == 1], Dz[T == 0])
        meta = dict(analysis_id=aid, family='C', genotype='het_vs_wt', stage='', transition=tr, stages=f'{a}-{b}',
                    prefix=prefix, **setting_cols(outcome), window=page_window(outcome), n_units=len(units),
                    setting=key_of(outcome, prefix, page_window(outcome)), hab_window=G['hab_label'])
        directions = ('het>wt', 'het<wt')
        res = neural_effect_search(Dz, T, nuisance=dN, select='p')
        nes_out = {**jsonable({k: v for k, v in res.items() if k != 'tables'}), 'n_units': len(units),
                   'units': list(units), 'n_het': int(T.sum()), 'n_wt': int((1 - T).sum()), **extra}
        nrows = nes_rows({**meta, 'method': 'nes'}, res, directions, outcome == 'latency')
        out = dines(Dz, T, W=dN, **DINES_KW)
        drows = dines_rows({**meta, 'method': 'dines'}, out, tau, se, t, df, directions)
        return {'spec': spec, 'nes_rows': nrows, 'nes_result': nes_out, 'dines_rows': drows,
                'dines_result': dines_result(out, tau, units, {'n_het': int(T.sum()), 'n_wt': int((1 - T).sum()),
                                                               **extra}),
                'dines_audit': dines_audit(out), 'check': None, 'elapsed_s': time.time() - t0,
                'n_selected_nes': len(res['selected']), 'n_selected_dines': len(out['selected'])}

    an = D.analysis(aid)
    rw = runner_window(an)
    wa, wb = D.window_map(an)[rw]
    extra = {}
    if outcome == 'latency':
        if an.family == 'A':
            d = an.select(design)
            ra = d.loc[d['stage'] == an.stages[0], 'obs_row'].values
            rb = d.loc[d['stage'] == an.stages[1], 'obs_row'].values
        else:
            ra = rb = an.select(design)['obs_row'].values
        L = window_len(ra, rb, wa, wb)
        ref = nes_ref('latency', aid, prefix, rw)
        if ref is not None and (int(ref['W_frames']) != L or int(ref['window_frames']) != L):
            raise SystemExit(f'{aid} p{prefix}: latency window {L} != NES W_frames {ref["W_frames"]} / '
                             f'window_frames {ref["window_frames"]}')
        sm = lat_summ(L)
        extra = {'W_frames': L, 'W_s': L / G['fps'], 'window_frames': L, 'window_s': L / G['fps'],
                 'W_source': 'copied from the NES latency run (equal)' if ref is not None else 'recomputed'}
    else:
        sm = G[outcome]
    ref = nes_ref(outcome, aid, prefix, rw)
    meta = dict(analysis_id=aid, family=an.family, **an.meta, prefix=prefix, **setting_cols(outcome),
                window=page_window(outcome), n_units=None, setting=key_of(outcome, prefix, page_window(outcome)))
    if G['dom'] == 'mice':
        meta['hab_window'] = G['hab_label']
    if outcome == 'latency':
        meta['stopped'] = ''
    if an.family == 'A':
        g, (a, b) = an.genotype, an.stages
        pools, Za, Zb = C.paired(sm, design, g, a, b, 'v', wa, wb, prefix)
        _, Na, Nb = C.paired(cov, design, g, a, b, 'v', wa, wb)
        n = len(pools)
        M, Mn = (Za + Zb) / 2, (Na + Nb) / 2
        Zs = np.vstack([Za - M, Zb - M])
        Ws = np.vstack([Na - Mn, Nb - Mn])
        T = np.r_[np.zeros(n, int), np.ones(n, int)]
        cl = np.r_[np.arange(n), np.arange(n)]
        tau = (Zb - Za).mean(0)
        se, t, df = one_sample(Zb - Za)
        nuis_nes = (Na, Nb) if G['nes_nuisance'] == 'nfg' else None
        chk = nes_check(None, None, nuis_nes, None if ref is None else ref['first_round'], paired=(Za, Zb))
        out = dines(Zs, T, W=Ws, clusters=cl, **DINES_KW)
        units = list(pools)
        extra_r = {}
    else:
        rows = an.select(design)
        units, Z, T = C.two_sample(sm, rows, 'v', wa, prefix, an.unit)
        Wn = C.two_sample(cov, rows, 'v', wa, unit=an.unit)[1]
        dX = C.day_indicators(rows, an.day_col, an.unit) if an.day_col else None
        W = Wn if dX is None else np.column_stack([Wn, dX])
        nuis_nes = [Wn] if G['nes_nuisance'] == 'nfg' else []
        nuis_nes += [] if dX is None else [dX]
        nuis_nes = np.column_stack(nuis_nes) if nuis_nes else None
        tau = Z[T == 1].mean(0) - Z[T == 0].mean(0)
        se, t, df = welch(Z[T == 1], Z[T == 0])
        chk = nes_check(Z, T, nuis_nes, None if ref is None else ref['first_round'])
        out = dines(Z, T, W=W, **DINES_KW)
        units = list(units)
        extra_r = {'n_het': int(T.sum()), 'n_wt': int((1 - T).sum()), 'n_day_indicators': 0 if dX is None else dX.shape[1]}
    meta['n_units'] = len(units)
    rows_ = dines_rows({**meta, 'method': 'dines'}, out, tau, se, t, df, an.directions)
    return {'spec': spec, 'dines_rows': rows_, 'dines_result': dines_result(out, tau, units, {**extra_r, **extra}),
            'dines_audit': dines_audit(out), 'check': chk, 'elapsed_s': time.time() - t0,
            'n_selected_dines': len(out['selected']), 'runner_window': rw}


def null_test(aid, n_shuffles=5):
    """DINES under T shuffled across units (mean outcome, prefix 128, same W): number of atoms reported per shuffle."""
    from eci import contrasts as C
    from eci.dines import dines
    D, design = G['D'], G['design']
    an = D.analysis(aid)
    wa = D.window_map(an)[runner_window(an)][0]
    rows = an.select(design)
    _, Z, T = C.two_sample(G['mean'], rows, 'v', wa, 128, an.unit)
    Wn = C.two_sample(G['cov_nfg'], rows, 'v', wa, unit=an.unit)[1]
    dX = C.day_indicators(rows, an.day_col, an.unit) if an.day_col else None
    W = Wn if dX is None else np.column_stack([Wn, dX])
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n_shuffles):
        r = dines(Z, rng.permutation(T), W=W, **DINES_KW)
        out.append({'n_reported': len(r['selected']), 'selected': r['selected'], 'forward_set': r['forward_set']})
    return {'analysis_id': aid, 'prefix': 128, 'outcome': 'mean', 'shuffles': out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', required=True, help='NES result set results/vision/<domain>/<layout>/nes/<sae>')
    ap.add_argument('--hab-window', default='last15', choices=('last15', 'whole'),
                    help="mice: habituation window (whole = the runners' 'hwhole', the pre-2026-10-07 primary)")
    ap.add_argument('--prefixes', default=None, help='default 128,256 and the full width (512 / 1024)')
    ap.add_argument('--outcomes', default=','.join(OUTCOMES))
    ap.add_argument('--analyses', default=None, help='comma list (default: every core analysis, + C_* for mice)')
    ap.add_argument('--no-family-c', action='store_true')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--tests', action='store_true', help='also run test 3 (family C vs A taus) and 4 (DINES null)')
    args = ap.parse_args()
    res = Path(args.res).resolve()
    dom, ants_eci, dkey = parse_res(res)
    if ants_eci is not None:
        os.environ['ANTS_ECI'] = ants_eci
    sys.path.insert(0, str(ROOT / 'src'))
    from eci import contrasts as C
    import eci.domain as Dm
    if dom == 'ants' and ants_eci.startswith('eci_r4v3'):  # as scripts/eci/rollout_r4_nes.py: the pooled v3 2 vs 6+8
        A = Dm.AntsDomain
        A.LAYOUT_CONTRASTS = {**A.LAYOUT_CONTRASTS, ants_eci: A.LAYOUT_CONTRASTS['eci30']}
    D = Dm.get_domain(dom, 'core')
    if dom == 'ants' and D.ECI != ants_eci:
        raise SystemExit(f'AntsDomain.ECI {D.ECI} != {ants_eci}')
    if dom == 'mice' and not str(res).startswith(str(D.nes_root.resolve())):
        raise SystemExit(f'{res} not under {D.nes_root}')
    if args.hab_window == 'whole' and dom != 'mice':
        raise SystemExit('--hab-window whole: mice only')
    # the NES searches the page reads (build_explorer.rdir_of): <res>/pairs/ when that analysis-set run exists (ants;
    # its first analyses repeat the core ones), else <res>
    rdir = res / 'pairs' if (res / 'pairs' / 'summary.csv').exists() else res
    sanity = json.loads((rdir / 'sanity.json').read_text())
    nes_nuis = sanity.get('nuisance') or 'none'
    nes_day = sanity.get('day')
    for sub in ('maxpool_bouts', 'maxpool_latency'):
        s2 = json.loads((rdir / sub / 'sanity.json').read_text())
        if (s2.get('nuisance') or 'none') != nes_nuis or s2.get('day') != nes_day:
            raise SystemExit(f'{rdir}/{sub}: nuisance / day {s2.get("nuisance")} / {s2.get("day")} differ from the root '
                             f'run {nes_nuis} / {nes_day}')
    if nes_day:
        D.day_nuisance = True
    sae = res.name
    codes_dir = D.codes_root / sae
    if not (codes_dir / 'n_fg.npy').exists():
        raise SystemExit(f'{codes_dir}/n_fg.npy missing: no n_fg covariate')
    dfull = D.load_design()
    t0 = time.time()
    mean, bouts, lat, nf = load_caches(res, dfull, D.fps)
    nfg_v = C.video_nfg(codes_dir / 'n_fg.npy', dfull, D.n_match)
    print(f'caches + n_fg loaded in {time.time() - t0:.0f}s; {len(dfull)} videos, {mean[("full", "v")].shape[1]} '
          f'latents; NES run nuisance {nes_nuis}, day {nes_day}', flush=True)
    width = mean[('full', 'v')].shape[1]
    prefixes = tuple(int(x) for x in args.prefixes.split(',')) if args.prefixes else \
        tuple(sorted({p for p in (128, 256) if p < width} | {width}))
    G.update(D=D, design=dfull, mean=mean, bouts=bouts, lat=lat, nf=nf, fps=D.fps, res=str(res), rdir=str(rdir), dom=dom,
             cov_nfg={(w, 'v'): nfg_v[w][:, None] for w in C.WINDOWS}, nes_nuisance=nes_nuis,
             hab=args.hab_window if dom == 'mice' else 'last15',
             hab_label=('whole30' if args.hab_window == 'whole' else 'last15') if dom == 'mice' else '')
    outcomes = tuple(args.outcomes.split(','))
    aids = args.analyses.split(',') if args.analyses else D.analysis_ids()
    specs = [('D', a, o, p) for a in aids if not a.startswith('C_') for o in outcomes for p in prefixes]
    if dom == 'mice' and not args.no_family_c:
        cids = [a for a in aids if a.startswith('C_')] if args.analyses else [f'C_{t}' for t in C_IDS]
        specs += [('C', a, o, p) for a in cids for o in outcomes for p in prefixes]
    base = Path(args.out_root) / dkey / sae
    base.mkdir(parents=True, exist_ok=True)
    try:  # the wrapper passes ATLAS_V37_HEAD (the archive has no .git); some nodes have no git
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        head = ''
    code = {'head': os.environ.get('ATLAS_V37_HEAD') or head, 'code_root': str(ROOT), 'res': str(res),
            'nes_reference_dir': str(rdir),
            'domain': dom, 'domain_key': dkey, 'ants_eci': ants_eci, 'sae': sae, 'prefixes': list(prefixes),
            'outcomes': list(outcomes), 'hab_window': G['hab_label'] or None,
            'hab_window_text': {'whole30': "habituation on its whole 30 min (runner window 'hwhole'; 'full' for the "
                                           "analyses without a habituation side): the pre-2026-10-07 primary",
                                'last15': 'habituation on its last 15 min (current primary)'}.get(G['hab_label']),
            'nes_run_nuisance': nes_nuis, 'nes_run_day': nes_day,
            'dines_W': 'per-unit mean n_fg of the outcome window' + (' + recording-day indicators' if nes_day else ''),
            'dines_settings': DINES_KW, 'family_c_nes': 'neural_effect_search two-sample, bonferroni 0.05, select p, '
                                                        'nuisance = per-pool n_fg difference (b - a)',
            'n_tasks': len(specs), 'start': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'ants_layout_patch': 'LAYOUT_CONTRASTS[eci_r4v3*] = LAYOUT_CONTRASTS[eci30]' if dom == 'ants' and
            ants_eci.startswith('eci_r4v3') else None}
    (base / 'code.json').write_text(json.dumps(code, indent=1))
    print(f'{len(specs)} tasks, {args.workers} workers -> {base}', flush=True)

    results = []
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(task, specs):
                results.append(r)
                print(f'{r["spec"]}: DINES {r["n_selected_dines"]}'
                      + (f', NES {r["n_selected_nes"]}' if 'n_selected_nes' in r else '')
                      + f' ({r["elapsed_s"]:.1f}s) check {r["check"]}', flush=True)
    else:
        for s in specs:
            r = task(s)
            results.append(r)
            print(f'{r["spec"]}: DINES {r["n_selected_dines"]}' + (f', NES {r["n_selected_nes"]}'
                  if 'n_selected_nes' in r else '') + f' ({r["elapsed_s"]:.1f}s) check {r["check"]}', flush=True)
    print(f'all tasks in {time.time() - t0:.0f}s', flush=True)

    # ---- write
    checks, timing = [], []
    groups = {}  # (dir, csv name) -> rows ; (dir, aid) -> {setting: result} ; audits
    jres, jaud = {}, {}
    for r in results:
        fam, aid, outcome, prefix = r['spec']
        sub = SUBDIR[outcome]
        setting = key_of(outcome, prefix, page_window(outcome))
        if fam == 'C':
            dn = base / 'familyC_nes' / sub
            dd = base / 'familyC_dines' / sub
            groups.setdefault((dn, 'C_summary.csv'), []).extend(r['nes_rows'])
            groups.setdefault((dd, 'C_summary.csv'), []).extend(r['dines_rows'])
            jres.setdefault((dn, aid), {})[setting] = r['nes_result']
        else:
            dd = base / 'dines' / sub
            groups.setdefault((dd, 'summary.csv'), []).extend(r['dines_rows'])
            c = r['check']
            checks.append({'analysis_id': aid, 'outcome': outcome, 'prefix': prefix, 'runner_window': r['runner_window'],
                           'nes_setting': key_of(outcome, prefix, nes_window(outcome, r['runner_window'])),
                           'max_abs_tau_diff': None if c is None else c['max_abs_diff'],
                           'n_neurons': None if c is None else c['n'], 'note': '' if c else 'no such NES setting'})
        jres.setdefault((dd, aid), {})[setting] = r['dines_result']
        jaud.setdefault((dd, aid), {})[setting] = r['dines_audit']
        timing.append({'family': fam, 'analysis_id': aid, 'outcome': outcome, 'prefix': prefix,
                       'elapsed_s': r['elapsed_s']})
    for (d, name), rows in groups.items():
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(d / name, index=False)
    for (d, aid), v in jres.items():
        (d / aid).mkdir(parents=True, exist_ok=True)
        (d / aid / 'result.json').write_text(json.dumps(v))
    for (d, aid), v in jaud.items():
        (d / aid / 'dines.json').write_text(json.dumps(v))
    ck = pd.DataFrame(checks)
    ck.to_csv(base / 'checks.csv', index=False)
    pd.DataFrame(timing).to_csv(base / 'timing.csv', index=False)
    bad = ck[ck['max_abs_tau_diff'].astype(float) > CHECK_TOL] if len(ck) else ck
    code.update(end=time.strftime('%Y-%m-%dT%H:%M:%S'), n_checked=int(ck['max_abs_tau_diff'].notna().sum()) if len(ck) else 0,
                max_check_diff=float(ck['max_abs_tau_diff'].astype(float).max()) if len(ck) and ck['max_abs_tau_diff'].notna().any() else None,
                n_check_failed=int(len(bad)))
    (base / 'code.json').write_text(json.dumps(code, indent=1))
    print(f'identity checks: {code["n_checked"]} settings, max |tau diff| {code["max_check_diff"]}, failed {len(bad)}',
          flush=True)

    if args.tests:
        tests = {}
        if dom == 'mice':  # test 3: tau_C = mean het change - mean wt change, from family A first rounds (no nuisance)
            from eci.nes import paired_effect_search
            t3 = []
            for tr, (a, b) in C_IDS.items():
                if (base / 'familyC_dines', f'C_{tr}') not in jres or 'mean' not in outcomes:
                    continue
                an = D.analysis(f'A_het_{tr}')
                wa, wb = D.window_map(an)[runner_window(an)]
                for prefix in prefixes[:1]:
                    fa = {}
                    for g in ('het', 'wt'):
                        _, Za, Zb = C.paired(mean, dfull, g, a, b, 'v', wa, wb, prefix)
                        fr = paired_effect_search(Za, Zb, max_rounds=1)['first_round']
                        fa[g] = pd.Series(fr['tau'].values, index=fr['neuron'].values)
                    rj = jres[(base / 'familyC_dines', f'C_{tr}')][key_of('mean', prefix, 'full')]['first_round']
                    tc = pd.Series(rj['tau'], index=rj['neuron'])
                    common = tc.index.intersection(fa['het'].index).intersection(fa['wt'].index)
                    diff = (tc[common] - (fa['het'][common] - fa['wt'][common])).abs()
                    t3.append({'transition': tr, 'prefix': prefix, 'n_neurons': int(len(common)),
                               'max_abs_diff': float(diff.max()), 'example_neuron': int(common[0]),
                               'tau_C': float(tc[common[0]]), 'tau_A_het': float(fa['het'][common[0]]),
                               'tau_A_wt': float(fa['wt'][common[0]])})
            tests['test3_familyC_vs_A'] = t3
            print('test 3:', t3, flush=True)
        two = [a for a in D.analysis_ids() if D.analysis(a).family == 'B']
        tests['test4_null'] = null_test(two[0])
        print('test 4:', tests['test4_null'], flush=True)
        (base / 'tests.json').write_text(json.dumps(jsonable(tests), indent=1))
    if len(bad):
        print(bad.to_string(), flush=True)
        raise SystemExit(f'{len(bad)} identity checks failed (|tau diff| > {CHECK_TOL}): Z differs from the NES run')
    print(f'done -> {base}', flush=True)


if __name__ == '__main__':
    main()
