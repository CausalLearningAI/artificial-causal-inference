"""
Neural Effect Search (NES) on mice v1 SAE codes with BOUT outcomes from max-pooled codes.

Per-frame value = codes_max (max over patches). Per neuron j, threshold thr_j(q) = q-quantile of
codes_max[:, j] over a fixed random subsample of frames pooled over all videos (treatment- and
stage-agnostic). A bout = maximal run of consecutive frames of one video (window) with
codes_max > thr_j; min length 1 frame; merge gap g = bouts separated by <= g frames are merged.
Outcome per video = bout rate (bouts / minute of the window); mean / median bout duration (s)
are descriptive only (undefined at 0 bouts, so never tested).

Analyses (as scripts/eci/run_nes.py): family A = paired stage transition within genotype
(het/wt x 1->2, 2->3, 4->5, 5->6; unit = pool), family B = het vs wt within stage 1..6.
Primary: bout rate, q = 0.95, gap 0, t-test, Bonferroni, windows full and trim30, prefixes 128 and 1024.
Sensitivity (one change from the full-window primary): signflip (A), BH, q 0.90, q 0.99,
merge gap 2, time-matched window (A 1->2, 4->5), outcome = per-video mean of codes_max, and
min2hyst (bout_rule 1 in summary.csv): hysteresis bouts that enter above thr(0.95), exit when
codes_max drops to <= thr(0.90), and last >= 2 frames (src/eci/contrasts.py bout_stats).
Size check (foreground SAEs, when <codes>/n_fg.npy exists): every round-1 neuron of the primary
(full and trim30) re-tested with the per-video mean foreground patch count as a covariate
(contrasts.size_adjusted_round1) -> size_adjusted.csv.
Sanity: 20x genotype shuffle across pools (B stage 2) and 20x within-pool stage-label swap
(A het 1->2), primary setting.

Nuisance conditioning (--nuisance nfg): every search and null conditions on the per-video mean
foreground patch count (same window as the outcome) from round 0 (src/eci/nes.py `nuisance`).

Subgroups (--line ash1l|kdm6b|kmt5b, --sex f|m, default all): units restricted to the pools of that gene
line and / or sex before the search (as scripts/eci/run_nes.py); thresholds and per-video summaries stay
full-cohort (label-free). Output defaults to <out-root>/<sae>/subsets/<line>_<sex>/maxpool_bouts/.
Analyses with fewer than --min-units units per arm are skipped (sanity.json 'skipped').
--primary-only: only the primary setting (full window, both prefixes), no sensitivity columns.
--n-shuffles 0 skips the sanity nulls.

Domains (--domain, default mice; src/eci/domain.py): analyses, design, paths, fps and the default
--nuisance come from the domain (ants: family B only, no stage-swap null; see scripts/eci/run_nes.py).

Usage: python scripts/eci/run_nes_bouts.py --sae matryoshka_btk_1024_k16_ep20_s0
       python scripts/eci/run_nes_bouts.py --domain ants --sae <ants sae> --compare-pooling max
Writes results/vision/<domain>/eci/nes/<sae>/[<subdir>/]maxpool_bouts/; caches under .../nes/<sae>/_cache/.

--analysis-set (default core): the domain's analysis set (see scripts/eci/run_nes.py); a set other than core
writes to <out-root>/<sae>/<set>/<P>_bouts/ and compares with <sae>/<set>/summary.csv.

--frame-pooling P (default max): the per-frame values are <codes>/codes_P.npy instead of codes_max.npy (e.g. somp
for the SOMP codes of scripts/eci/somp_encode_all.py, mean for mean-pooled codes); output subdir P_bouts/
(meanpool_bouts/ for mean), 'pooling' column = P, caches with a _P suffix. Default outputs are unchanged.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from eci import contrasts as C  # noqa: E402
from eci.domain import DOMAINS, get_domain  # noqa: E402
from eci.nes import active_neurons, neural_effect_search, paired_effect_search  # noqa: E402

FPS = 5.0  # mice; main() sets the domain's frame rate
FP = 'max'  # per-frame pooling of the codes (--frame-pooling); main() sets it
PREFIXES = (128, 1024)
QS = (0.90, 0.95, 0.99)
CONFIGS = [(0.90, 0), (0.95, 0), (0.99, 0), (0.95, 2), (0.95, 0, 0.90, 2)]  # (quantile, merge gap[, exit q, min len])
HYST = (0.95, 0, 0.90, 2)  # bout_rule 1
WINDOW_MAP = {'full': ('full', 'full'), 'matched': ('last', 'full'), 'trim30': ('trim', 'trim')}
# neurons flagged as recording artefacts: only valid for the ep20 SAE (neuron ids are SAE-specific)
ARTEFACTS_EP20 = {64: 'white card / experimenter hand at video start', 50: 'white card / experimenter hand at video start',
                  113: 'grey arena rim in one camera setup'}
ARTEFACTS = {}
PRIMARY = dict(outcome_type='bout_rate', threshold_q=0.95, merge_gap=0, bout_rule=0, test='t', correction='bonferroni',
               window='full')
# name -> overrides of PRIMARY; applicability filter by analysis id
SENS = {'trim30': dict(window='trim30'), 'signflip': dict(test='signflip'), 'BH': dict(correction='bh'),
        'q0.90': dict(threshold_q=0.90), 'q0.99': dict(threshold_q=0.99), 'gap2': dict(merge_gap=2),
        'matched': dict(window='matched'), 'min2hyst': dict(bout_rule=1),
        'mean-outcome': dict(outcome_type='mean', threshold_q=np.nan, merge_gap=np.nan, bout_rule=np.nan)}


def applicable(name, an):
    """an: src/eci/domain.py Analysis."""
    if name == 'signflip':
        return an.family == 'A'
    if name == 'matched':
        return an.family == 'A' and an.matched
    return True


def settings_for(an):
    out = [dict(PRIMARY)]
    for name, ov in SENS.items():
        if applicable(name, an):
            out.append({**PRIMARY, **ov})
    return out


def skey(prefix, s):
    thr = 'na' if s['outcome_type'] == 'mean' else f"q{s['threshold_q']:.2f}_g{int(s['merge_gap'])}"
    if s['outcome_type'] != 'mean' and s.get('bout_rule', 0) == 1:
        thr += f'_x{HYST[2]:.2f}_min{HYST[3]}'
    return f"p{prefix}_{s['outcome_type']}_{thr}_{s['test']}_{s['correction']}_{s['window']}"


def strip(res):
    return C.to_jsonable({k: v for k, v in res.items() if k != 'tables'})


def tidy_rows(meta, res, directions):
    """directions: (label of tau > 0, label of tau < 0) of the analysis."""
    base = {**meta, 'n_tested_total': res['n_tested'], 'n_dropped': res['n_dropped']}
    if len(res['rounds']) == 0:
        return [{**base, 'round': 0}]
    rows = []
    for r in res['rounds'].to_dict('records'):
        d = directions[0] if r['tau'] > 0 else directions[1]
        rows.append({**base, **{k: r[k] for k in ('round', 'neuron', 'tau', 'se', 't', 'df', 'p', 'threshold', 'n_tested')},
                     'direction': d})
    return rows


def select(tidy, aid, prefix, s, round1=False):
    t = tidy[(tidy['analysis_id'] == aid) & (tidy['prefix'] == prefix) & (tidy['round'] > 0)]
    for k, v in s.items():
        t = t[t[k].isna()] if isinstance(v, float) and np.isnan(v) else t[t[k] == v]
    if round1:
        t = t[t['round'] == 1]
    return dict(zip(t['neuron'].astype(int), t['direction']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='mice', choices=DOMAINS)
    ap.add_argument('--analysis-set', default='core',
                    help="the domain's analysis set (src/eci/domain.py; ants 'pairs' = every within-experiment "
                         "treatment pair); a set other than core writes to <sae>/<set>/ unless --subdir")
    ap.add_argument('--sae', default='matryoshka_btk_1024_k16_ep20_s0')
    ap.add_argument('--codes-root', default=None, help='default <domain eci dir>/codes')
    ap.add_argument('--out-root', default=None, help='default the domain NES root (results/vision/<domain>/eci/nes)')
    ap.add_argument('--n-match', type=int, default=None, help='default the domain value (mice 4500)')
    ap.add_argument('--n-trim', type=int, default=150)
    ap.add_argument('--n-thr-sample', type=int, default=2_000_000)
    ap.add_argument('--n-shuffles', type=int, default=20)
    ap.add_argument('--skip-signflip', action='store_true')
    ap.add_argument('--compare-pooling', default='mean', help='pooling of the mean-activation run to compare with')
    ap.add_argument('--nuisance', default=None, choices=('none', 'nfg'), help='default: the domain primary (none)')
    ap.add_argument('--subdir', default='', help='write to <out-root>/<sae>/<subdir>/maxpool_bouts/')
    ap.add_argument('--line', default='all', choices=('all',) + C.LINES)
    ap.add_argument('--sex', default='all', choices=('all',) + C.SEXES)
    ap.add_argument('--min-units', type=int, default=5, help='skip an analysis with fewer units per arm')
    ap.add_argument('--primary-only', action='store_true')
    ap.add_argument('--frame-pooling', default='max', help='per-frame values = <codes>/codes_<this>.npy')
    ap.add_argument('--prefixes', default='128,1024', help="as scripts/eci/run_nes.py ('all' = every column, one prefix)")
    args = ap.parse_args()
    global FPS, FP
    FP = args.frame_pooling
    sfx = '' if FP == 'max' else f'_{FP}'
    D = get_domain(args.domain, args.analysis_set)
    global PREFIXES
    from run_nes import resolve_prefixes
    PREFIXES = resolve_prefixes(args.prefixes, Path(args.codes_root or D.codes_root) / args.sae / f'codes_{FP}.npy')
    FPS = D.fps
    args.codes_root = args.codes_root or str(D.codes_root)
    args.out_root = args.out_root or str(D.nes_root)
    args.n_match = D.n_match if args.n_match is None else args.n_match
    args.nuisance = args.nuisance or D.nuisance
    if (args.line, args.sex) != ('all', 'all') and not args.subdir:
        args.subdir = f'subsets/{C.subset_name(args.line, args.sex)}'
    if args.analysis_set != 'core' and not args.subdir:
        args.subdir = args.analysis_set
    t_start = time.time()
    global ARTEFACTS
    ARTEFACTS = ARTEFACTS_EP20 if args.sae in [f'matryoshka_btk_1024_k16_ep20_s0{x}' for x in ('', '_somp', '_mean')] else {}
    codes_dir = Path(args.codes_root) / args.sae
    if not (codes_dir / 'DONE').exists():
        raise SystemExit(f'{codes_dir}/DONE missing: codes not finished')
    base = Path(args.out_root) / args.sae / args.subdir
    out = base / {'max': 'maxpool_bouts', 'mean': 'meanpool_bouts'}.get(FP, f'{FP}_bouts')
    out.mkdir(parents=True, exist_ok=True)
    if args.analysis_set != 'core' and hasattr(D, 'analysis_table'):  # per-analysis n, day counts, confound flag
        (out / 'analyses.json').write_text(json.dumps(C.to_jsonable(D.analysis_table()), indent=1))
    cache = Path(args.out_root) / args.sae / '_cache'
    cache.mkdir(parents=True, exist_ok=True)

    dfull = D.load_design()
    design = D.subset(dfull, args.line, args.sex)  # obs_row still indexes the full-cohort summaries
    print(f'subgroup line={args.line} sex={args.sex}: {D.describe(design)}', flush=True)
    print(f'{len(design)} observations, contiguous row blocks verified; frames per {D.duration_col}:',
          design.groupby(D.duration_col)['n_frames'].agg(['min', 'max']).to_dict('index'), flush=True)

    # ---- thresholds (treatment-agnostic pooled quantiles of codes_max)
    t0 = time.time()
    thr_path = cache / f'bout_thresholds_n{args.n_thr_sample}{sfx}.npz'
    if thr_path.exists():
        f = np.load(thr_path)
        thr_arr, n_used = f['thr'], int(f['n_used'])
    else:
        thr_arr, n_used = C.pooled_thresholds(codes_dir / f'codes_{FP}.npy', QS, args.n_thr_sample, seed=0)
        np.savez(thr_path, thr=thr_arr, qs=np.array(QS), n_used=n_used)
    thresholds = {q: thr_arr[i] for i, q in enumerate(QS)}
    pd.DataFrame({'neuron': np.arange(thr_arr.shape[1]), **{f'thr_q{q:.2f}': thresholds[q] for q in QS}}).to_csv(
        out / 'thresholds.csv', index=False)
    thr_info = {'n_frames_sampled': n_used, 'seed': 0,
                'n_zero_threshold': {f'{q:.2f}': int((thresholds[q] == 0).sum()) for q in QS}}
    print(f'thresholds from {n_used} frames ({time.time() - t0:.0f}s):', thr_info, flush=True)

    # ---- per-video bout summaries (streamed once, cached)
    t0 = time.time()
    bs = C.cached_bout_summaries(codes_dir / f'codes_{FP}.npy', dfull, cache / f'bout_summaries_max{sfx}.npz', thresholds,
                                 CONFIGS, args.n_match, args.n_trim)
    print(f'bout summaries: {time.time() - t0:.0f}s', flush=True)
    t0 = time.time()
    msum = C.cached_summaries(codes_dir / f'codes_{FP}.npy', dfull, cache / f'video_summaries_max{sfx}.npz', args.n_match,
                              args.n_trim)
    print(f'mean summaries codes_{FP}: {time.time() - t0:.0f}s', flush=True)

    # outcome matrices: (outcome_type, q, gap) -> {(window, 'v'): (n_obs, m)}
    def summ_for(s):
        if s['outcome_type'] == 'mean':
            return {(w, 'v'): msum[(w, 'mean')] for w in C.WINDOWS}
        if s.get('bout_rule', 0) == 1:
            return {(w, 'v'): C.bout_outcomes(bs, w, *HYST[:2], FPS, *HYST[2:])['rate'] for w in C.WINDOWS}
        return {(w, 'v'): C.bout_outcomes(bs, w, s['threshold_q'], int(s['merge_gap']), FPS)['rate'] for w in C.WINDOWS}

    cov = None
    if args.nuisance == 'nfg':
        nfg_v = C.video_nfg(codes_dir / 'n_fg.npy', dfull, args.n_match, args.n_trim)
        cov = {(w, 'v'): nfg_v[w][:, None] for w in C.WINDOWS}

    def nuis_paired(geno, a, b, wa, wb):
        return None if cov is None else C.paired(cov, design, geno, a, b, 'v', wa, wb)[1:]

    def nuis_two(an, w):
        return None if cov is None else C.two_sample(cov, an.select(design), 'v', w, unit=an.unit)[1]

    all_rows, results, skipped = [], {}, []
    analyses = []  # the domain's Analysis objects with enough units, in domain order
    for an in D.analyses:
        aid = an.id
        if an.family == 'A':
            n = len(C.paired(msum, design, an.genotype, *an.stages, 'mean')[0])
            sk = {'analysis_id': aid, 'n_units': n} if n < args.min_units else None
        else:
            T0 = C.two_sample(msum, an.select(design), unit=an.unit)[2]
            sk = ({'analysis_id': aid, 'n_het': int(T0.sum()), 'n_wt': int((1 - T0).sum())}
                  if min(T0.sum(), (1 - T0).sum()) < args.min_units else None)
        if sk:
            skipped.append(sk)
            print(f'{aid}: SKIPPED, too few units {sk}', flush=True)
        else:
            analyses.append(an)
    for an in analyses:
        aid, fam = an.id, an.family
        results[aid] = {}
        for s in ([dict(PRIMARY)] if args.primary_only else settings_for(an)):
            if s['test'] == 'signflip' and args.skip_signflip:
                continue
            sm = summ_for(s)
            wa, wb = WINDOW_MAP[s['window']]
            for prefix in PREFIXES:
                t0 = time.time()
                if fam == 'A':
                    geno, (a, b) = an.genotype, an.stages
                    units, Za, Zb = C.paired(sm, design, geno, a, b, 'v', wa, wb, prefix)
                    res = paired_effect_search(Za, Zb, correction=s['correction'], test=s['test'],
                                               nuisance=nuis_paired(geno, a, b, wa, wb))
                    extra = {}
                else:
                    units, Z, T = C.two_sample(sm, an.select(design), 'v', wa, prefix, an.unit)
                    res = neural_effect_search(Z, T, correction=s['correction'], nuisance=nuis_two(an, wa))
                    extra = {'n_het': int(T.sum()), 'n_wt': int((1 - T).sum())}
                k = skey(prefix, s)
                results[aid][k] = {**strip(res), 'n_units': len(units), 'units': list(units), **extra}
                meta = dict(analysis_id=aid, family=fam, **an.meta, prefix=prefix, pooling=FP, **s,
                            n_units=len(units), setting=k)
                all_rows += tidy_rows(meta, res, an.directions)
                print(f'{aid} {k}: selected={res["selected"]} dropped={res["n_dropped"]} {time.time() - t0:.1f}s',
                      flush=True)

    for aid, r in results.items():
        (out / aid).mkdir(exist_ok=True)
        with open(out / aid / 'result.json', 'w') as f:
            json.dump(r, f)
    tidy = pd.DataFrame(all_rows)
    tidy.to_csv(out / 'summary.csv', index=False)

    # ---- sanity nulls on the primary setting
    rng = np.random.default_rng(0)
    sm = summ_for(PRIMARY)
    sanity = {'thresholds': thr_info, 'nulls': {}, 'nuisance': args.nuisance, 'primary_only': args.primary_only,
              'subgroup': D.subgroup_info(design, args.line, args.sex),
              'skipped': skipped,
              'n_units': {an.id: int(results[an.id][skey(PREFIXES[0], PRIMARY)]['n_units']) for an in analyses}}
    an2 = D.analysis(D.null_two)  # label shuffle across units (mice: genotype across pools, B stage 2)
    anp = D.analysis(D.null_paired) if D.null_paired else None  # within-unit swap (mice: A het 1->2)
    for prefix in PREFIXES if args.n_shuffles > 0 else ():
        _, Z, T = C.two_sample(sm, an2.select(design), 'v', 'full', prefix, an2.unit)
        cnt = [len(neural_effect_search(Z, rng.permutation(T), nuisance=nuis_two(an2, 'full'))['selected'])
               for _ in range(args.n_shuffles)]
        sanity['nulls'][f'p{prefix}'] = {f'{an2.id}_{D.shuffle_word}_shuffle_n_selected': cnt}
        cnt_a = []
        if anp is not None:
            _, Za, Zb = C.paired(sm, design, anp.genotype, *anp.stages, 'v', 'full', 'full', prefix)
            Nab = nuis_paired(anp.genotype, *anp.stages, 'full', 'full')
            for _ in range(args.n_shuffles):
                sw = rng.random(len(Za)) < 0.5
                Ya, Yb = np.where(sw[:, None], Zb, Za), np.where(sw[:, None], Za, Zb)
                nu = None if Nab is None else (np.where(sw[:, None], Nab[1], Nab[0]), np.where(sw[:, None], Nab[0], Nab[1]))
                cnt_a.append(len(paired_effect_search(Ya, Yb, nuisance=nu)['selected']))
            sanity['nulls'][f'p{prefix}'][f'{anp.id}_stage_swap_n_selected'] = cnt_a
        print(f'null prefix {prefix}: {D.shuffle_word} shuffle {cnt}; stage swap {cnt_a}', flush=True)

    # ---- dropped (near-constant) neuron counts for the primary setting
    sanity['n_dropped_primary'] = {
        f'{r.analysis_id}_p{r.prefix}': int(r.n_dropped)
        for r in tidy[(tidy['setting'].str.contains('bout_rate_q0.95_g0_t_bonferroni_full'))].drop_duplicates(
            ['analysis_id', 'prefix']).itertuples()}

    # ---- size check: round-1 neurons re-tested with the per-video mean foreground size
    sadj = size_check(tidy, design, codes_dir, sm, args, out, dfull=dfull, analyses={a.id: a for a in D.analyses})
    if sadj is not None:
        sanity['size_check'] = {'n_round1': int(len(sadj)), 'n_survive': int(sadj['survives'].sum()),
                                'n_frames_without_fg_frac': float((np.load(codes_dir / 'n_fg.npy', mmap_mode='r') == 0).mean())}

    # ---- descriptives for round-1 neurons, comparison with the mean-pool run
    desc, round1 = descriptives(tidy, design, bs, analyses, D.desc_by)
    desc.to_csv(out / 'round1_descriptives.csv', index=False)
    prev = compare_meanpool(tidy, base / 'summary.csv', analyses, args.compare_pooling)
    sanity['runtime_s'] = time.time() - t_start
    with open(out / 'sanity.json', 'w') as f:
        json.dump(C.to_jsonable(sanity), f, indent=1)
    write_reports(out, tidy, analyses, bs, design, desc, round1, prev, sanity, args.sae, sadj, args.primary_only, D)
    print(f'done in {time.time() - t_start:.0f}s -> {out}', flush=True)


def size_check(tidy, design, codes_dir, values, args, out, prim=None, window_map=None, dfull=None, analyses=None):
    """Round-1 neurons of the primary setting (windows full and trim30) re-tested with the per-video
    mean n_fg as a covariate; None when the codes have no n_fg.npy (patch-level SAEs)."""
    if not (Path(codes_dir) / 'n_fg.npy').exists():
        return None
    prim = prim or PRIMARY
    nfg = C.video_nfg(Path(codes_dir) / 'n_fg.npy', design if dfull is None else dfull, args.n_match, args.n_trim)
    t = tidy[tidy['round'] == 1]
    for k, v in prim.items():
        if k != 'window':
            t = t[t[k].isna()] if isinstance(v, float) and np.isnan(v) else t[t[k] == v]
    t = t[t['window'].isin(['full', 'trim30'])]
    vals = {w: values[(w, 'v')] for w in C.WINDOWS}
    r = C.size_adjusted_round1(t, design, vals, nfg, window_map or WINDOW_MAP, analyses)
    cols = ['analysis_id', 'prefix', 'window', 'neuron', 'direction', 'tau', 'p', 'threshold', 'tau_adj', 'se_adj',
            'p_adj', 'df_adj', 'slope_nfg', 'survives']
    r = r[cols] if len(r) else pd.DataFrame(columns=cols)
    r.to_csv(Path(out) / 'size_adjusted.csv', index=False)
    print(f'size check: {int(r["survives"].sum()) if len(r) else 0}/{len(r)} round-1 neurons survive', flush=True)
    return r


def size_mark(sadj, aid, prefix, window, j):
    if sadj is None:
        return '-'
    r = sadj[(sadj['analysis_id'] == aid) & (sadj['prefix'] == prefix) & (sadj['window'] == window) & (sadj['neuron'] == j)]
    return '-' if not len(r) else ('Y' if bool(r['survives'].iloc[0]) else 'N')


def unit_rows(design, an):
    """obs_rows of the videos entering analysis an (A: both stages; B: both arms)."""
    return an.select(design)['obs_row'].values


def descriptives(tidy, design, bs, analyses, by=('stage', 'genotype')):
    """Per round-1 neuron of the primary full-window setting (either prefix): median over videos of
    bouts/min, per-video median bout duration (s) and per-video mean bout duration (s), per group of
    the design columns `by` (mice: stage x genotype; primary q 0.95, gap 0, full window). 'robust' =
    also selected under trim30 at that prefix."""
    o = C.bout_outcomes(bs, 'full', 0.95, 0, FPS)
    round1 = {}
    for aid in (an.id for an in analyses):
        for prefix in PREFIXES:
            r1 = select(tidy, aid, prefix, PRIMARY, round1=True)
            tr = select(tidy, aid, prefix, {**PRIMARY, 'window': 'trim30'})
            for j, d in r1.items():
                round1.setdefault(j, []).append({'analysis_id': aid, 'prefix': prefix, 'direction': d,
                                                 'robust_trim30': j in tr})
    rows = []
    for j in sorted(round1):
        for (st, g), dd in design.groupby(list(by)):
            r = dd['obs_row'].values
            rows.append({'neuron': j, by[0]: st, by[1]: g, 'n_videos': len(r),
                         'median_bouts_per_min': float(np.median(o['rate'][r, j])),
                         'median_bout_dur_s': float(np.nanmedian(o['median_dur'][r, j])) if np.isfinite(o['median_dur'][r, j]).any() else np.nan,
                         'median_mean_bout_dur_s': float(np.nanmedian(o['mean_dur'][r, j])) if np.isfinite(o['mean_dur'][r, j]).any() else np.nan,
                         'frac_videos_0_bouts': float((o['rate'][r, j] == 0).mean()),
                         'median_frac_frames_above': float(np.median(o['frac'][r, j]))})
    return pd.DataFrame(rows), round1


def compare_meanpool(tidy, prev_csv, analyses, pooling='mean'):
    """Primary selections of the previous mean-pool run (codes_mean, per-video mean, t, Bonferroni,
    full) vs this primary; {aid: {prefix: {'meanpool': [...], 'bouts': [...], 'overlap': [...]}}}."""
    if not Path(prev_csv).exists():
        return {}
    p = pd.read_csv(prev_csv)
    p = p[(p['pooling'] == pooling) & (p['outcome_type'] == 'mean') & (p['test'] == 't') &
          (p['correction'] == 'bonferroni') & (p['window'] == 'full') & (p['round'] > 0)]
    out = {}
    for aid in (an.id for an in analyses):
        out[aid] = {}
        for prefix in PREFIXES:
            mp = p[(p['analysis_id'] == aid) & (p['prefix'] == prefix)].sort_values('round')['neuron'].astype(int).tolist()
            b = list(select(tidy, aid, prefix, PRIMARY))
            out[aid][prefix] = {'meanpool': mp, 'bouts': b, 'overlap': [j for j in b if j in mp]}
    return out


def write_reports(out, tidy, analyses, bs, design, desc, round1, prev, sanity, sae, sadj=None, primary_only=False,
                  D=None):
    D = D or get_domain('mice')
    o = C.bout_outcomes(bs, 'full', 0.95, 0, FPS)
    names = [] if primary_only else list(SENS)
    sg = sanity.get('subgroup', {})
    L = [f'# NES summary ({FP}-pool, bout outcomes): {sae}', '',
         *([f'**Subgroup: line {sg["line"]}, sex {sg["sex"]}** ({sg["n_pools"]} pools, {sg["n_het_pools"]} het; '
            f'{sg["n_videos"]} videos).' + (' Primary setting only.' if primary_only else ''),
            f'Skipped (too few units): {sanity["skipped"]}' if sanity.get('skipped') else '', '']
           if sg and (sg.get('line', 'all'), sg.get('sex', 'all')) != ('all', 'all') else []),
         (f'Per-frame value = codes_{FP} (max over patches). ' if FP == 'max' else f'Per-frame value = codes_{FP}. ')
         + f'Threshold per neuron = q-quantile of codes_{FP} pooled over '
         f'{sanity["thresholds"]["n_frames_sampled"]} randomly sampled frames of all videos (treatment-agnostic; seed 0). '
         f'Above = codes_{FP} > threshold (strict: a neuron whose quantile is 0 counts any activation > 0; zero thresholds '
         f'per q: {sanity["thresholds"]["n_zero_threshold"]}). Bout = maximal run of consecutive above frames within one '
         'video window, min 1 frame; merge gap g merges bouts separated by <= g frames (primary g = 0). Outcome = bout '
         'rate (bouts per minute of the window). Mean/median bout duration is undefined at 0 bouts and is reported '
         'descriptively only (never tested).', '',
         f'Primary: bout rate, q 0.95, gap 0, t-test, Bonferroni alpha 0.05, full window. {D.text["families_bouts"]} '
         'Near-constant neurons (active in < 1% of units or zero '
         'variance) are dropped before testing, as in nes.py.'
         + (' Nuisance conditioning: every search conditions on the per-video mean foreground patch count (n_fg, how '
            f'spread out / huddled the {D.subjects} are; same window as the outcome) from round 0, as an already-selected neuron.'
            if sanity.get('nuisance') == 'nfg' else ' No nuisance conditioning.'), '',
         'Robustness columns: Y = also selected (any round) under that single change from the primary (same prefix); '
         '"-" = not applicable. "mean-pool" = selected by the mean-activation primary (per-video mean). '
         'min2hyst = bouts enter above thr(0.95), exit at <= thr(0.90), min 2 frames. size-adj = round-1 neuron still '
         'significant (same sign, p below the round-1 Bonferroni threshold) with the per-video mean foreground patch '
         'count as a covariate ("-" = not a round-1 neuron / no foreground counts). '
         'med dur = median over the analysis videos of each video\'s median bout duration (s, videos with bouts).', '']
    for an in analyses:
        aid = an.id
        L.append(f'## {aid}')
        for prefix in PREFIXES:
            sub = tidy[(tidy['analysis_id'] == aid) & (tidy['prefix'] == prefix) & (tidy['setting'] == skey(prefix, PRIMARY))]
            nd = int(sub['n_dropped'].iloc[0])
            ps = sub[sub['round'] > 0].sort_values('round')
            ntrim = 'n/a (not run)' if primary_only else len(select(tidy, aid, prefix, {**PRIMARY, 'window': 'trim30'}))
            if len(ps) == 0:
                L.append(f'- prefix {prefix}: nothing selected ({nd} dropped); trim30 selects {ntrim}')
                continue
            L += [f'- prefix {prefix}: {len(ps)} selected ({nd} dropped); trim30 selects {ntrim}', '',
                  '| round | neuron | direction | tau (bouts/min) | p | med dur (s) | ' + ' | '.join(names) +
                  ' | other prefix | mean-pool | size-adj | artefact flag |', '|' + '---|' * (10 + len(names))]
            others = [p for p in PREFIXES if p != prefix]
            other = select(tidy, aid, others[0], PRIMARY) if others else {}
            mp = prev.get(aid, {}).get(prefix, {}).get('meanpool', [])
            rows = unit_rows(design, an)
            for _, r in ps.iterrows():
                j = int(r['neuron'])
                marks = ['Y' if j in select(tidy, aid, prefix, {**PRIMARY, **SENS[n]}) else 'N'
                         if applicable(n, an) else '-' for n in names]
                op = '-' if (prefix == 1024 and j >= 128) or len(PREFIXES) == 1 else ('Y' if j in other else 'N')
                md = np.nanmedian(o['median_dur'][rows, j]) if np.isfinite(o['median_dur'][rows, j]).any() else np.nan
                L.append(f'| {int(r["round"])} | {j} | {r["direction"]} | {r["tau"]:.3g} | {r["p"]:.2e} | {md:.2f} | '
                         + ' | '.join(marks) + f' | {op} | {"Y" if j in mp else "N"} | '
                         f'{size_mark(sadj, aid, prefix, "full", j) if int(r["round"]) == 1 else "-"} | {ARTEFACTS.get(j, "")} |')
            L.append('')
        L.append('')
    rc, rvals, cc, cvals = D.desc_table  # table rows (mice: genotype), table columns (mice: stage)
    L += [f'## Round-1 neurons: descriptives per {cc} x {rc} (primary q 0.95, gap 0, full window)', '',
          'Median over videos. dur = per-video median bout duration (s); 0-bout = fraction of videos with no bout.', '']
    for j in sorted(round1):
        hits = '; '.join(f'{h["analysis_id"]} p{h["prefix"]} {h["direction"]}'
                         f'{"" if primary_only else " (trim30 Y)" if h["robust_trim30"] else " (trim30 N)"}'
                         for h in round1[j])
        L += [f'### neuron {j} {ARTEFACTS.get(j, "")}', f'round 1 in: {hits}', '',
              f'| {rc} | ' + ' | '.join(f'{cc} {s}' for s in cvals) + ' |', '|---|' + '---|' * len(cvals)]
        d = desc[desc['neuron'] == j]
        for g in rvals:
            cells = [d[(d[rc] == g) & (d[cc] == s)] for s in cvals]  # a missing group (ants: T not in that experiment) = '-'
            L.append(f'| {g} | ' + ' | '.join(f'{c.median_bouts_per_min:.2f}/min, dur {c.median_bout_dur_s:.2f}s, '
                                              f'0-bout {c.frac_videos_0_bouts:.2f}' if c is not None else '-'
                                              for c in (x.iloc[0] if len(x) else None for x in cells)) + ' |')
        L.append('')
    L += ['## Overlap with the previous mean-pool primary (codes_mean, per-video mean)', '',
          '| analysis | prefix | mean-pool selected | bout selected | overlap |', '|---|---|---|---|---|']
    for aid, v in prev.items():
        for prefix, d in v.items():
            L.append(f'| {aid} | {prefix} | {d["meanpool"]} | {d["bouts"]} | {d["overlap"]} |')
    L += ['', '## Sanity checks (primary setting)', '']
    for p, v in sanity['nulls'].items():
        v2 = v[f'{D.null_two}_{D.shuffle_word}_shuffle_n_selected']
        L.append(f'- {p}: {D.text["null_two_bouts"]}, {len(v2)} shuffles: NES selected {v2}')
        if D.null_paired:
            va = v[f'{D.null_paired}_stage_swap_n_selected']
            L.append(f'- {p}: {D.text["null_paired_bouts"]}, {len(va)} swaps: NES selected {va}')
    if 'size_check' in sanity:
        L.append(f'- size check (round-1 neurons, full + trim30): {sanity["size_check"]["n_survive"]}/'
                 f'{sanity["size_check"]["n_round1"]} survive the per-video mean n_fg covariate (size_adjusted.csv)')
    L.append(f'- runtime: {sanity["runtime_s"]:.0f} s')
    (out / 'SUMMARY.md').write_text('\n'.join(L) + '\n')
    union = {}
    for aid in (an.id for an in analyses):
        for prefix in PREFIXES:
            for _, r in tidy[(tidy['analysis_id'] == aid) & (tidy['prefix'] == prefix) &
                             (tidy['setting'] == skey(prefix, PRIMARY)) & (tidy['round'] > 0)].iterrows():
                union.setdefault(int(r['neuron']), {'artefact_flag': ARTEFACTS.get(int(r['neuron']), ''), 'hits': []})[
                    'hits'].append({'analysis_id': aid, 'prefix': prefix, 'round': int(r['round']),
                                    'direction': r['direction'], 'tau_bouts_per_min': float(r['tau']), 'p': float(r['p'])})
    with open(out / 'selected_neurons.json', 'w') as f:
        json.dump({'sae': sae, 'settings': f'primary (codes_{FP}, bout rate q0.95 gap0, t, bonferroni, full window)',
                   'neurons': {str(j): v for j, v in sorted(union.items())}}, f, indent=1)


if __name__ == '__main__':
    main()
