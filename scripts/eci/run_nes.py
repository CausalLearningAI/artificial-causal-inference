"""
Neural Effect Search (NES) analyses on the mice v1 SAE codes, at video level.

  A  paired stage transitions within one genotype (unit = pool, 36 pairs):
     1->2 social H->O, 2->3 social O->P, 4->5 fear H->O, 5->6 fear O->P; het and wt separately.
     paired_effect_search, test 't' (primary) and 'signflip' (sensitivity).
     Time-matched sensitivity for the H->O transitions (habituation 9000 frames vs odor 4500):
     last 4500 frames of habituation vs the whole odor stage.
  B  genotype het vs wt within each stage 1..6 (unit = video, 36 vs 36): neural_effect_search.
  C  pseudo-replication illustration: genotype at stage 2, frame level vs video level.
  Sanity: genotype labels shuffled across pools (stage 2), 20 times.

Grid per analysis: prefix {128, 1024} x pooling {codes_mean, codes_max} x outcome {mean, rate}
x correction {bonferroni, bh} x window {full, trim30 (first 30 s of every video dropped: handling
artefacts)} (+ window matched for A 1->2, 4->5) (+ test signflip for A, primary setting only). Primary: codes_mean, outcome mean, test t, bonferroni, full window, both prefixes.
--primary-pooling max makes codes_max the primary pooling (reports, sign-flip, shuffles, frame-level C)
and codes_mean its sensitivity. Size check (foreground SAEs, when <codes>/n_fg.npy exists): round-1
neurons of the primary (full, trim30) re-tested with the per-video mean foreground patch count as a
covariate -> size_adjusted.csv (contrasts.size_adjusted_round1).

Nuisance conditioning (--nuisance nfg, foreground SAEs): every A / B search (and the shuffle nulls)
conditions on the per-video mean foreground patch count n_fg (same window as the outcome) from
round 0, as an already-selected neuron (src/eci/nes.py `nuisance`). --subdir writes the run to
<out-root>/<sae>/<subdir>/ (e.g. the unconditioned sensitivity: --nuisance none --subdir unconditioned);
the per-video summary cache stays in <out-root>/<sae>/_cache/.

Subgroups (--line ash1l|kdm6b|kmt5b, --sex f|m, default all): the units are restricted to the pools of
that gene line and / or sex BEFORE the search (A: that genotype's pools of the subgroup; B: het and wt
videos of the subgroup); per-video summaries, n_fg and bout thresholds stay full-cohort (they do not use
the labels). Output defaults to <out-root>/<sae>/subsets/<line>_<sex>/. An analysis with fewer than
--min-units units per arm (A: pools, B: het or wt videos) is skipped (sanity.json 'skipped').
--primary-only: only the primary setting (codes_<primary pooling>, per-video mean, t, Bonferroni, full
window, both prefixes); no frame-level C, no sign-flip, no sensitivity columns in SUMMARY.md.

Domains (--domain, default mice; src/eci/domain.py): the analyses, design, paths and default
--nuisance come from the domain. ants: family B only (v2_1_vs_2, v3_2_vs_6, v3_2_vs_8; unit = video,
tau > 0 = higher in the treated videos), sanity null = treatment shuffled across the v2 videos,
frame-level C = v2_1_vs_2; v3_2_vs_8 is confounded with the recording day (summary.csv 'confound').

Usage: python scripts/eci/run_nes.py --sae matryoshka_btk_1024_k16_ep20_s0
       python scripts/eci/run_nes.py --sae <fg sae> --primary-pooling max --nuisance nfg
       python scripts/eci/run_nes.py --domain ants --sae <ants sae> --primary-pooling max
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
from eci.nes import neural_effect_search, paired_effect_search  # noqa: E402

PRIMARY = dict(pooling='mean', stat='mean', test='t', correction='bonferroni', window='full')
PREFIXES = (128, 1024)
# window name -> (window of stage a, window of stage b) in contrasts.video_summaries
WINDOW_MAP = {'full': ('full', 'full'), 'matched': ('last', 'full'), 'trim30': ('trim', 'trim')}
# neurons the gallery worker identified as recording artefacts (ep20 SAE, prefix 128)
ARTEFACTS_EP20 = {64: 'white card / experimenter hand at video start', 50: 'white card / experimenter hand at video start',
             113: 'grey arena rim in one camera setup (het ~0.7 vs wt ~0.33 flat over stages)'}
ARTEFACTS = {}  # set in main: the ep20 flags only apply to the ep20 SAE (neuron ids are SAE-specific)


def key(prefix, pooling, stat, test, correction, window):
    return f'p{prefix}_{pooling}_{stat}_{test}_{correction}_{window}'


def strip(res):
    r = {k: v for k, v in res.items() if k != 'tables'}
    return C.to_jsonable(r)


def tidy_rows(meta, res, directions):
    """directions: (label of tau > 0, label of tau < 0) of the analysis."""
    base = {**meta, 'n_tested_total': res['n_tested'], 'n_dropped': res['n_dropped']}
    rounds = res['rounds']
    if len(rounds) == 0:
        return [{**base, 'round': 0}]
    rows = []
    for r in rounds.to_dict('records'):
        direction = directions[0] if r['tau'] > 0 else directions[1]
        rows.append({**base, **{k: r[k] for k in ('round', 'neuron', 'tau', 'se', 't', 'df', 'p', 'threshold', 'n_tested')},
                     'direction': direction})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='mice', choices=DOMAINS)
    ap.add_argument('--sae', default='matryoshka_btk_1024_k16_ep20_s0')
    ap.add_argument('--codes-root', default=None, help='default <domain eci dir>/codes')
    ap.add_argument('--out-root', default=None, help='default the domain NES root (results/vision/<domain>/eci/nes)')
    ap.add_argument('--n-match', type=int, default=None, help='default the domain value (mice 4500)')
    ap.add_argument('--n-shuffles', type=int, default=20)
    ap.add_argument('--frame-max-rounds', type=int, default=8)  # strata double per round at frame level
    ap.add_argument('--skip-frame', action='store_true')
    ap.add_argument('--primary-pooling', default='mean', choices=('mean', 'max'))
    ap.add_argument('--nuisance', default=None, choices=('none', 'nfg'),
                    help='nfg: condition every search on the per-video mean foreground size from round 0 '
                         '(default: the domain primary, none for mice and ants)')
    ap.add_argument('--subdir', default='', help='write to <out-root>/<sae>/<subdir>/')
    ap.add_argument('--line', default='all', choices=('all',) + C.LINES)
    ap.add_argument('--sex', default='all', choices=('all',) + C.SEXES)
    ap.add_argument('--min-units', type=int, default=5, help='skip an analysis with fewer units per arm')
    ap.add_argument('--primary-only', action='store_true')
    args = ap.parse_args()
    D = get_domain(args.domain)
    args.codes_root = args.codes_root or str(D.codes_root)
    args.out_root = args.out_root or str(D.nes_root)
    args.n_match = D.n_match if args.n_match is None else args.n_match
    args.nuisance = args.nuisance or D.nuisance
    subset = (args.line, args.sex) != ('all', 'all')
    if subset and not args.subdir:
        args.subdir = f'subsets/{C.subset_name(args.line, args.sex)}'
    if args.primary_only:
        args.skip_frame = True
    t_start = time.time()
    global ARTEFACTS
    ARTEFACTS = ARTEFACTS_EP20 if args.sae == 'matryoshka_btk_1024_k16_ep20_s0' else {}
    pp = args.primary_pooling
    codes_dir = Path(args.codes_root) / args.sae
    if not (codes_dir / 'DONE').exists():
        raise SystemExit(f'{codes_dir}/DONE missing: codes not finished')
    out = Path(args.out_root) / args.sae / args.subdir
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.out_root) / args.sae / '_cache'

    dfull = D.load_design()
    design = D.subset(dfull, args.line, args.sex)  # obs_row still indexes the full-cohort summaries
    print(f'subgroup line={args.line} sex={args.sex}: {D.describe(design)}', flush=True)
    dur = design.groupby(D.duration_col)['n_frames'].agg(['count', 'min', 'max', 'mean']).reset_index()
    dur[f'minutes_at_{D.fps:g}fps'] = dur['mean'] / D.fps / 60
    dur.to_csv(out / 'stage_durations.csv', index=False)
    print(f'{D.duration_col} durations (frames per observation):\n', dur.to_string(index=False), flush=True)

    summ = {}
    for pooling in ('mean', 'max'):
        t0 = time.time()
        summ[pooling] = C.cached_summaries(codes_dir / f'codes_{pooling}.npy', dfull,
                                           cache_dir / f'video_summaries_{pooling}.npz', args.n_match)
        print(f'summaries codes_{pooling}: {time.time() - t0:.0f}s', flush=True)

    cov = None
    if args.nuisance == 'nfg':
        nfg_v = C.video_nfg(codes_dir / 'n_fg.npy', dfull, args.n_match)
        cov = {(w, 'v'): nfg_v[w][:, None] for w in C.WINDOWS}

    def nuis_paired(geno, a, b, wa, wb):
        return None if cov is None else C.paired(cov, design, geno, a, b, 'v', wa, wb)[1:]

    def nuis_two(an, w):
        return None if cov is None else C.two_sample(cov, an.select(design), 'v', w, unit=an.unit)[1]

    all_rows, results, sanity = [], {}, {D.balance_key: D.balance(design), 'nuisance': args.nuisance,
                                         'subgroup': D.subgroup_info(design, args.line, args.sex),
                                         'primary_only': args.primary_only, 'skipped': []}
    poolings = (pp,) if args.primary_only else ('mean', 'max')
    stats_a = ('mean',) if args.primary_only else ('mean', 'rate')
    tests = ('t',) if args.primary_only else ('t', 'signflip')
    corrs = ('bonferroni',) if args.primary_only else ('bonferroni', 'bh')
    b_settings = (('mean', 'full'),) if args.primary_only else (('mean', 'full'), ('rate', 'full'), ('mean', 'trim30'),
                                                               ('rate', 'trim30'))
    print(f'{D.balance_key}:', json.dumps(sanity[D.balance_key])[:2000], flush=True)

    for an in D.analyses:
        aid = an.id
        # ---- A: paired stage transitions
        if an.family == 'A':
            geno, (a, b) = an.genotype, an.stages
            n_a = len(C.paired(summ[pp], design, geno, a, b, 'mean')[0])
            if n_a < args.min_units:
                sanity['skipped'].append({'analysis_id': aid, 'n_units': n_a})
                print(f'{aid}: SKIPPED, {n_a} paired pools < {args.min_units}', flush=True)
                continue
            results[aid] = {}
            windows = ['full'] if args.primary_only else ['full', 'trim30'] + (['matched'] if an.matched else [])
            for prefix in PREFIXES:
                for pooling in poolings:
                    for stat in stats_a:
                        for window in windows:
                            wa, wb = WINDOW_MAP[window]
                            pools, Za, Zb = C.paired(summ[pooling], design, geno, a, b, stat, wa, wb, prefix)
                            for test in tests:
                                for corr in corrs:
                                    # sign-flip is a one-change sensitivity of the primary setting only
                                    # (~3 min per search at m = 1024 on one core)
                                    if test == 'signflip' and (pooling, stat, corr, window) != (pp, 'mean', 'bonferroni', 'full'):
                                        continue
                                    t0 = time.time()
                                    res = paired_effect_search(Za, Zb, correction=corr, test=test,
                                                               nuisance=nuis_paired(geno, a, b, wa, wb))
                                    k = key(prefix, pooling, stat, test, corr, window)
                                    results[aid][k] = {**strip(res), 'n_units': len(pools), 'units': list(pools)}
                                    meta = dict(analysis_id=aid, family='A', **an.meta, prefix=prefix,
                                                pooling=pooling, outcome_type=stat, test=test, correction=corr,
                                                window=window, n_units=len(pools), setting=k)
                                    all_rows += tidy_rows(meta, res, an.directions)
                                    print(f'{aid} {k}: n={len(pools)} selected={res["selected"]} '
                                          f'dropped={res["n_dropped"]} {time.time() - t0:.1f}s', flush=True)

            continue
        # ---- B: two-sample (mice: genotype within stage; the 'n_het' / 'n_wt' keys = treatment / control)
        rows = an.select(design)
        T0 = C.two_sample(summ[pp], rows, unit=an.unit)[2]
        if min(T0.sum(), (1 - T0).sum()) < args.min_units:
            sanity['skipped'].append({'analysis_id': aid, 'n_het': int(T0.sum()), 'n_wt': int((1 - T0).sum())})
            print(f'{aid}: SKIPPED, {int(T0.sum())} het / {int((1 - T0).sum())} wt videos (< {args.min_units})', flush=True)
            continue
        results[aid] = {}
        for prefix in PREFIXES:
            for pooling in poolings:
                for stat, window in b_settings:
                    pools, Z, T = C.two_sample(summ[pooling], rows, stat, WINDOW_MAP[window][0], prefix, an.unit)
                    for corr in corrs:
                        res = neural_effect_search(Z, T, correction=corr, nuisance=nuis_two(an, WINDOW_MAP[window][0]))
                        k = key(prefix, pooling, stat, 't', corr, window)
                        results[aid][k] = {**strip(res), 'n_units': len(pools), 'n_het': int(T.sum()),
                                           'n_wt': int((1 - T).sum()), 'units': list(pools)}
                        meta = dict(analysis_id=aid, family='B', **an.meta, prefix=prefix, pooling=pooling,
                                    outcome_type=stat, test='t', correction=corr, window=window, n_units=len(pools),
                                    setting=k)
                        all_rows += tidy_rows(meta, res, an.directions)
                        print(f'{aid} {k}: n={int(T.sum())}+{int((1 - T).sum())} selected={res["selected"]} '
                              f'dropped={res["n_dropped"]}', flush=True)

    for aid, r in results.items():
        (out / aid).mkdir(exist_ok=True)
        with open(out / aid / 'result.json', 'w') as f:
            json.dump(r, f)
    tidy = pd.DataFrame(all_rows)
    tidy.to_csv(out / 'summary.csv', index=False)

    # ---- sanity: permutation of the two-sample labels across units (mice: genotype across pools, stage 2),
    # primary settings
    rng = np.random.default_rng(0)
    perm = {}
    an0 = D.analysis(D.null_two)
    for prefix in PREFIXES if args.n_shuffles > 0 else ():
        pools, Z, T = C.two_sample(summ[pp], an0.select(design), 'mean', 'full', prefix, an0.unit)
        counts, naive = [], []
        for _ in range(args.n_shuffles):
            res = neural_effect_search(Z, rng.permutation(T), nuisance=nuis_two(an0, 'full'))
            counts.append(len(res['selected'])), naive.append(int(res['first_round']['significant'].sum()))
        perm[prefix] = {'n_selected': counts, 'n_naive_significant': naive}
        print(f'permutation prefix {prefix}: selected {counts} naive {naive}', flush=True)
    sanity[f'permutation_{D.null_two}'] = perm

    # ---- C: pseudo-replication (mice: genotype at stage 2)
    if not args.skip_frame:
        frame = {}
        for prefix in PREFIXES:
            t0 = time.time()
            Zf, Tf, gf = C.frame_matrix_rows(codes_dir / f'codes_{pp}.npy', D.analysis(D.frame_analysis).select(design),
                                             prefix)
            fres = neural_effect_search(Zf, Tf, max_rounds=args.frame_max_rounds)
            vres = neural_effect_search(Zf, Tf, groups=gf)
            frame[prefix] = {
                'n_frames': int(len(Zf)), 'n_videos': int(gf.max() + 1),
                'frame_naive_significant': int(fres['first_round']['significant'].sum()),
                'frame_nes_selected': len(fres['selected']), 'frame_nes_capped': len(fres['selected']) >= args.frame_max_rounds,
                'video_naive_significant': int(vres['first_round']['significant'].sum()),
                'video_nes_selected': len(vres['selected']),
                'n_tested_frame': fres['n_tested'], 'n_tested_video': vres['n_tested'],
                'frame_selected': fres['selected'], 'video_selected': vres['selected'],
                'elapsed_s': time.time() - t0}
            (out / f'C_frame_{D.frame_key}').mkdir(exist_ok=True)
            with open(out / f'C_frame_{D.frame_key}' / f'result_p{prefix}.json', 'w') as f:
                json.dump({'frame': strip(fres), 'video': strip(vres)}, f)
            print(f'frame-level prefix {prefix}:', {k: v for k, v in frame[prefix].items() if 'selected' not in k or 'nes' in k}, flush=True)
            del Zf
        sanity[f'pseudo_replication_{D.frame_key}'] = frame

    # ---- size check: round-1 neurons of the primary re-tested with the per-video mean n_fg
    sadj = None
    if (codes_dir / 'n_fg.npy').exists():
        nfg = C.video_nfg(codes_dir / 'n_fg.npy', dfull, args.n_match)
        t = tidy[(tidy['round'] == 1) & (tidy['pooling'] == pp) & (tidy['outcome_type'] == 'mean') & (tidy['test'] == 't')
                 & (tidy['correction'] == 'bonferroni') & tidy['window'].isin(['full', 'trim30'])]
        sadj = C.size_adjusted_round1(t, design, {w: summ[pp][(w, 'mean')] for w in C.WINDOWS}, nfg, WINDOW_MAP,
                                      {a.id: a for a in D.analyses})
        cols = ['analysis_id', 'prefix', 'window', 'neuron', 'direction', 'tau', 'p', 'threshold', 'tau_adj', 'se_adj',
                'p_adj', 'df_adj', 'slope_nfg', 'survives']
        sadj = sadj[cols] if len(sadj) else pd.DataFrame(columns=cols)
        sadj.to_csv(out / 'size_adjusted.csv', index=False)
        sanity['size_check'] = {'n_round1': int(len(sadj)), 'n_survive': int(sadj['survives'].sum())}
        print('size check:', sanity['size_check'], flush=True)

    sanity['runtime_s'] = time.time() - t_start
    with open(out / 'sanity.json', 'w') as f:
        json.dump(C.to_jsonable(sanity), f, indent=1)
    write_reports(out, tidy, results, dur, sanity, args.sae, pp, sadj, args.nuisance, args.primary_only, D)
    print(f'done in {time.time() - t_start:.0f}s -> {out}', flush=True)


def selected_set(tidy, aid, **kw):
    s = tidy[(tidy['analysis_id'] == aid) & (tidy['round'] > 0)]
    for k, v in kw.items():
        s = s[s[k] == v]
    return dict(zip(s['neuron'].astype(int), s['direction']))


def write_reports(out, tidy, results, dur, sanity, sae, pp='mean', sadj=None, nuisance='none', primary_only=False,
                  D=None):
    D = D or get_domain('mice')
    dc = D.duration_col
    alt = 'max' if pp == 'mean' else 'mean'
    sens = {} if primary_only else {  # name -> overrides of the primary setting (same prefix)
        'signflip': dict(test='signflip'), f'{alt}-pool': dict(pooling=alt), 'rate': dict(outcome_type='rate'),
        'BH': dict(correction='bh'), 'matched': dict(window='matched'), 'trim30': dict(window='trim30')}
    sg = sanity.get('subgroup', {})
    prim = dict(pooling=pp, outcome_type='mean', test='t', correction='bonferroni', window='full')
    union = {}
    lines = [f'# NES summary: {sae}', '',
             *([f'**Subgroup: line {sg["line"]}, sex {sg["sex"]}** ({sg["n_pools"]} pools, {sg["n_het_pools"]} het; '
                f'{sg["n_videos"]} videos). Primary setting only.' if primary_only else f'**Subgroup: line {sg["line"]}, sex '
                f'{sg["sex"]}** ({sg["n_pools"]} pools, {sg["n_het_pools"]} het; {sg["n_videos"]} videos).', '']
               if sg and (sg.get('line', 'all'), sg.get('sex', 'all')) != ('all', 'all') else []),
             f'Video-level Neural Effect Search on {D.title}. Primary: codes_{pp} pooling, outcome = per-video mean '
             f'activation, t-test, Bonferroni alpha 0.05, full {D.text["window"]} window. {D.text["families_nes"]} '
             'Prefix 128 = first 128 Matryoshka latents; neuron ids are shared with prefix 1024.'
             + (' Nuisance conditioning: every search conditions on the per-video mean foreground patch count '
                f'(n_fg, how spread out / huddled the {D.subjects} are) from round 0, as an already-selected neuron.'
                if nuisance == 'nfg' else ' No nuisance conditioning.'), '',
             'Robustness columns: Y = the neuron is also selected (any round) under that single change from primary '
             '(same prefix); N = not; "-" = not applicable. "other prefix" = selected in the same analysis at the '
             'other prefix (only neurons < 128 can appear at prefix 128). size-adj = round-1 neuron still significant '
             '(same sign, p below its round-1 threshold) with the per-video mean foreground patch count as a covariate.', '',
             f'{dc.capitalize()} durations (frames at {D.fps:g} fps): '
             + ', '.join(f'{dc} {int(r[dc]) if dc == "stage" else r[dc]}: {int(r["mean"])}' for _, r in dur.iterrows()), '']
    for aid in results:
        an = D.analysis(aid)
        fam = an.family
        lines.append(f'## {aid}')
        any_sel = False
        for prefix in PREFIXES:
            ps = selected_set(tidy, aid, prefix=prefix, **prim)
            sub = tidy[(tidy['analysis_id'] == aid) & (tidy['prefix'] == prefix)]
            for k, v in prim.items():
                sub = sub[sub[k] == v]
            nd = int(sub['n_dropped'].iloc[0]) if len(sub) else -1
            if not ps:
                lines.append(f'- prefix {prefix}: nothing selected ({nd} near-constant neurons dropped)')
                continue
            any_sel = True
            other = selected_set(tidy, aid, prefix=[p for p in PREFIXES if p != prefix][0], **prim)
            lines.append(f'- prefix {prefix}: {len(ps)} selected ({nd} dropped)')
            lines.append('')
            lines.append('| round | neuron | direction | tau | p | ' + ' | '.join(sens) + ' | other prefix | size-adj | artefact flag |')
            lines.append('|' + '---|' * (8 + len(sens)))
            rows = sub[sub['round'] > 0].sort_values('round')
            for _, r in rows.iterrows():
                j = int(r['neuron'])
                marks = []
                for name, ov in sens.items():
                    if name == 'signflip' and fam != 'A' or name == 'matched' and not an.matched:
                        marks.append('-')
                        continue
                    marks.append('Y' if j in selected_set(tidy, aid, prefix=prefix, **{**prim, **ov}) else 'N')
                op = '-' if (prefix == 1024 and j >= 128) else ('Y' if j in other else 'N')
                sa = '-'
                if sadj is not None and int(r['round']) == 1:
                    q = sadj[(sadj['analysis_id'] == aid) & (sadj['prefix'] == prefix) & (sadj['window'] == 'full') & (sadj['neuron'] == j)]
                    sa = ('Y' if bool(q['survives'].iloc[0]) else 'N') if len(q) else '-'
                lines.append(f'| {int(r["round"])} | {j} | {r["direction"]} | {r["tau"]:.4g} | {r["p"]:.2e} | '
                             + ' | '.join(marks) + f' | {op} | {sa} | {ARTEFACTS.get(j, "")} |')
                union.setdefault(j, {'artefact_flag': ARTEFACTS.get(j, ''), 'hits': []})['hits'].append({'analysis_id': aid, 'prefix': prefix, 'round': int(r['round']),
                                                'direction': r['direction'], 'tau': float(r['tau']), 'p': float(r['p'])})
            lines.append('')
        if lines[-1] != '':
            lines.append('')
    perm = sanity.get(f'permutation_{D.null_two}', {})
    for sk in sanity.get('skipped', []):
        lines += [f'## {sk["analysis_id"]}', f'- SKIPPED: too few units ({sk})', '']
    lines += D.balance_report(sanity)
    lines += ['', '## Sanity checks', '']
    for p, v in perm.items():
        lines.append(f'- {D.text["null_two"]}, primary, prefix {p}), {len(v["n_selected"])} '
                     f'shuffles: NES selected {v["n_selected"]}; naive first-round Bonferroni {v["n_naive_significant"]}')
    for p, v in sanity.get(f'pseudo_replication_{D.frame_key}', {}).items():
        lines.append(f'- Pseudo-replication, {D.text["frame"]}, prefix {p}: frame level ({v["n_frames"]} frames) naive '
                     f'{v["frame_naive_significant"]}/{v["n_tested_frame"]} significant, NES {v["frame_nes_selected"]}'
                     f'{" (capped)" if v["frame_nes_capped"] else ""}; video level ({v["n_videos"]} videos) naive '
                     f'{v["video_naive_significant"]}/{v["n_tested_video"]}, NES {v["video_nes_selected"]}')
    if 'size_check' in sanity:
        lines.append(f'- size check (round-1 neurons, full + trim30): {sanity["size_check"]["n_survive"]}/'
                     f'{sanity["size_check"]["n_round1"]} survive the per-video mean n_fg covariate (size_adjusted.csv)')
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    with open(out / 'selected_neurons.json', 'w') as f:
        json.dump({'sae': sae, 'settings': f'primary (codes_{pp}, per-video mean, t, bonferroni, full window)',
                   'neurons': {str(j): v for j, v in sorted(union.items())}}, f, indent=1)


if __name__ == '__main__':
    main()
