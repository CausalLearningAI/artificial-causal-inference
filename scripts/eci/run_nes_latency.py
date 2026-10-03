"""
Neural Effect Search (NES) with a LATENCY outcome: per video and latent, the time (s) from the start of the video
window to its first bout.

Bout definition = the event-rate outcome of scripts/eci/run_nes_bouts.py: per latent, threshold thr_j(q) = q-quantile
of the per-frame values pooled over a fixed random subsample of frames of all videos (treatment-agnostic, seed 0,
the same cached thresholds as the bout runs), a frame is above when its value > thr_j (strict), a bout = a maximal
run of above frames, min 1 frame. The first bout starts at the first above frame, so merge gaps do not change the
latency. latency = (index of the first above frame in the window) / fps.

Common window: every analysis measures latency in the SAME window for all its videos: the first W frames, W = the
shortest video among the analysis's videos (Analysis.select rows). Bouts are looked for only inside the window and a
video with no bout there is right-censored at W. Without this, videos of different lengths are censored at different
values and the latency difference is the length difference (mice: habituation videos, stages 1 and 4, last 1800 s,
the others 900 s, so H->O comparisons would compare censoring points; mice W = 900 s in every analysis with an O or
P stage, 1800 s in B_stage1 / B_stage4; ants: all videos 600 s, W = 600 s). W and the window length are in
censoring.csv, result.json and SUMMARY.md.

Censoring: a video with no bout in the window gets latency = the window length (s), i.e. it is treated as if the
first bout came at the very end (right-censored at the window length). censoring.csv gives, per analysis and prefix,
the fraction of censored (video, latent) cells among the tested latents and how many tested latents are censored in
more than half of the analysis videos; SUMMARY.md gives the censored fraction per arm for every selected latent.
Because heavy censoring piles values at one point, a rank version is run as a sensitivity: each latent's latencies
are replaced by their ranks (average ranks for ties, e.g. all censored videos tie) across the units of the analysis
(family A: across the 2n stacked pool-stage values) before the same NES (a t-test on ranks ~ Mann-Whitney /
paired rank test).

Analyses, families and units as scripts/eci/run_nes.py (src/eci/domain.py). Primary: latency, q 0.95, t-test,
Bonferroni alpha 0.05, common window (window 'common' in summary.csv), prefixes 128, 256 and 1024 (or --prefixes;
the nulls of 128 / 1024 keep the shared seed-0 draws, other prefixes their own, run_nes.null_rng).
Sensitivities (one change each): q 0.90, q 0.99, trim30 (window = frames [30 s, W): starts after the first 30 s,
latency counted from there, censored at W - 30 s), BH, rank, signflip (family A), matched (family A 1->2 and 4->5:
the last n_match frames of habituation vs the first frames of the later stage, both capped at min(n_match, W)).
tau > 0 = LATER first bout in the treated
arm (family B) / at the later stage (family A). Nulls (primary setting, common window of the null analysis): the domain's two-sample label shuffle
(mice: genotype across pools, B stage 2; ants: v2_1_vs_2) and, mice, the within-pool stage swap (A het 1->2),
20 each.

--frame-pooling P: per-frame values = <codes>/<sae>/codes_P.npy (max for <sae>, mean for <sae>_mean, somp for
<sae>_somp). Output: <nes root>/<sae>/[<set>/]<P>_latency/ ('maxpool_latency' / 'meanpool_latency' for max / mean);
cache <nes root>/<sae>/_cache/latency_<P>.npz (first-above frame index in the whole full / last / trim window, NOT
capped at W: the common-window latency of a video is min(cached index, window length), censored when >= it).

Usage: python scripts/eci/run_nes_latency.py --domain mice --sae matryoshka_btk_1024_k16_fg448_s0 --frame-pooling max
       python scripts/eci/run_nes_latency.py --domain ants --sae matryoshka_btk_1024_k16_antsfg_s0_somp --frame-pooling somp --analysis-set pairs
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from eci import contrasts as C  # noqa: E402
from eci.domain import DOMAINS, get_domain  # noqa: E402
from eci.nes import neural_effect_search, paired_effect_search  # noqa: E402
from run_nes import null_rng, resolve_prefixes  # noqa: E402

QS = (0.90, 0.95, 0.99)  # the bout runs' quantiles (same cached thresholds)
# runner window -> (contrasts window of stage a / control, of stage b / treated); every window is capped at the
# analysis's common length (window_frames)
WINDOW_MAP = {'common': ('full', 'full'), 'matched': ('last', 'full'), 'trim30': ('trim', 'trim')}
PRIMARY = dict(outcome_type='latency', threshold_q=0.95, transform='none', test='t', correction='bonferroni',
               window='common')
SENS = {'q0.90': dict(threshold_q=0.90), 'q0.99': dict(threshold_q=0.99), 'trim30': dict(window='trim30'),
        'BH': dict(correction='bh'), 'rank': dict(transform='rank'), 'signflip': dict(test='signflip'),
        'matched': dict(window='matched')}
PREFIXES = (128, 256, 1024)


def applicable(name, an):
    if name == 'signflip':
        return an.family == 'A'
    if name == 'matched':
        return an.family == 'A' and an.matched
    return True


def common_frames(an, design):
    """W (frames) of an analysis: the shortest video among its videos."""
    return int(an.select(design)['n_frames'].min())


def window_frames(window, W, n_match, n_trim):
    """Length (frames) of the runner window inside the common window of W frames: the latency is capped (censored)
    at this value in both arms."""
    return {'common': W, 'matched': min(n_match, W), 'trim30': W - n_trim}[window]


def skey(prefix, s):
    return (f"p{prefix}_latency_q{s['threshold_q']:.2f}{'_rank' if s['transform'] == 'rank' else ''}_{s['test']}_"
            f"{s['correction']}_{s['window']}")


def latencies(codes_path, design, thresholds, n_match, n_trim):
    """{(window, q): (n_obs, m) first-above frame index within the window (= window length if none)},
    {window: (n_obs,) window length in frames}."""
    Z = np.load(codes_path, mmap_mode='r')
    n_obs, m = len(design), Z.shape[1]
    out = {(w, q): np.zeros((n_obs, m), np.int32) for w in C.WINDOWS for q in QS}
    nf = {w: np.zeros(n_obs, np.int32) for w in C.WINDOWS}
    sls = (('full', slice(None)), ('last', slice(-n_match, None)), ('trim', slice(n_trim, None)))
    for i, (s, e) in enumerate(zip(design['row_start'], design['row_end'])):
        X = np.asarray(Z[s:e])
        for w, sl in sls:
            Xw = X[sl]
            nf[w][i] = len(Xw)
            for q in QS:
                ab = Xw > thresholds[q].astype(Xw.dtype)
                any_ = ab.any(0)
                out[(w, q)][i] = np.where(any_, ab.argmax(0), len(Xw))
    return out, nf


def cached_latencies(codes_path, design, cache_path, thresholds, n_match, n_trim):
    thr = np.stack([thresholds[q] for q in QS])
    keys = [(w, q) for w in C.WINDOWS for q in QS]
    if Path(cache_path).exists():
        f = np.load(cache_path)
        if np.array_equal(f['observation_id'], design['observation_id'].values.astype(str)) and \
                int(f['n_match']) == n_match and int(f['n_trim']) == n_trim and np.array_equal(f['thr'], thr):
            return {k: f[f'{k[0]}__{k[1]:.2f}'] for k in keys}, {w: f[f'nf__{w}'] for w in C.WINDOWS}
    lat, nf = latencies(codes_path, design, thresholds, n_match, n_trim)
    np.savez(cache_path, observation_id=design['observation_id'].values.astype(str), n_match=n_match, n_trim=n_trim,
             thr=thr, **{f'{k[0]}__{k[1]:.2f}': v for k, v in lat.items()}, **{f'nf__{w}': v for w, v in nf.items()})
    return lat, nf


def strip(res):
    return C.to_jsonable({k: v for k, v in res.items() if k != 'tables'})


def tidy_rows(meta, res, directions):
    base = {**meta, 'n_tested_total': res['n_tested'], 'n_dropped': res['n_dropped'], 'stopped': res.get('stopped', '')}
    if len(res['rounds']) == 0:
        return [{**base, 'round': 0}]
    return [{**base, **{k: r[k] for k in ('round', 'neuron', 'tau', 'se', 't', 'df', 'p', 'threshold', 'n_tested')},
             'direction': directions[0] if r['tau'] > 0 else directions[1]} for r in res['rounds'].to_dict('records')]


def select(tidy, aid, prefix, s):
    t = tidy[(tidy['analysis_id'] == aid) & (tidy['prefix'] == prefix) & (tidy['round'] > 0)]
    for k, v in s.items():
        t = t[t[k] == v]
    if 'neuron' not in t:  # no search of the run selected anything: summary.csv has no neuron column
        return {}
    return dict(zip(t['neuron'].astype(int), t['direction']))


def rank_cols(Z):
    return np.apply_along_axis(rankdata, 0, Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='mice', choices=DOMAINS)
    ap.add_argument('--analysis-set', default='core')
    ap.add_argument('--sae', required=True)
    ap.add_argument('--frame-pooling', default='max')
    ap.add_argument('--codes-root', default=None)
    ap.add_argument('--out-root', default=None)
    ap.add_argument('--n-match', type=int, default=None)
    ap.add_argument('--n-trim', type=int, default=150)
    ap.add_argument('--n-thr-sample', type=int, default=2_000_000)
    ap.add_argument('--n-shuffles', type=int, default=20)
    ap.add_argument('--prefixes', default='128,256,1024')
    ap.add_argument('--subdir', default='')
    args = ap.parse_args()
    global PREFIXES
    FP = args.frame_pooling
    sfx = '' if FP == 'max' else f'_{FP}'
    D = get_domain(args.domain, args.analysis_set)
    if D.nuisance != 'none':
        raise SystemExit('latency runner supports domains without nuisance conditioning only')
    fps = D.fps
    codes_root = Path(args.codes_root or D.codes_root)
    out_root = Path(args.out_root or D.nes_root)
    n_match = D.n_match if args.n_match is None else args.n_match
    subdir = args.subdir or ('' if args.analysis_set == 'core' else args.analysis_set)
    codes_dir = codes_root / args.sae
    if not (codes_dir / 'DONE').exists():
        raise SystemExit(f'{codes_dir}/DONE missing')
    codes_path = codes_dir / f'codes_{FP}.npy'
    PREFIXES = resolve_prefixes(args.prefixes, codes_path)
    out = out_root / args.sae / subdir / {'max': 'maxpool_latency', 'mean': 'meanpool_latency'}.get(FP, f'{FP}_latency')
    out.mkdir(parents=True, exist_ok=True)
    cache = out_root / args.sae / '_cache'
    cache.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    if args.analysis_set != 'core' and hasattr(D, 'analysis_table'):
        (out / 'analyses.json').write_text(json.dumps(C.to_jsonable(D.analysis_table()), indent=1))

    design = D.load_design()
    thr_path = cache / f'bout_thresholds_n{args.n_thr_sample}{sfx}.npz'  # the bout runs' file name: same thresholds
    if thr_path.exists():
        f = np.load(thr_path)
        thr_arr, n_used = f['thr'], int(f['n_used'])
        if not np.allclose(f['qs'], QS):
            raise SystemExit(f'{thr_path}: quantiles {f["qs"]} != {QS}')
    else:
        thr_arr, n_used = C.pooled_thresholds(codes_path, QS, args.n_thr_sample, seed=0)
        np.savez(thr_path, thr=thr_arr, qs=np.array(QS), n_used=n_used)
    thresholds = {q: thr_arr[i] for i, q in enumerate(QS)}
    t0 = time.time()
    lat, nf = cached_latencies(codes_path, design, cache / f'latency{sfx}.npz', thresholds, n_match, args.n_trim)
    print(f'latencies: {time.time() - t0:.0f}s (thresholds from {n_used} frames)', flush=True)

    def summ(q, L):  # seconds, inside a window of L frames from the window start; censored = L
        return {(w, 'v'): np.minimum(lat[(w, q)], L) / fps for w in C.WINDOWS}

    def censored(q, L):
        return {(w, 'v'): (lat[(w, q)] >= L).astype(float) for w in C.WINDOWS}

    def check_len(an, wa, wb, L):  # the capped window must fit inside every video's own window
        d = an.select(design)
        if an.family == 'A':
            ra, rb = (d.loc[d['stage'] == x, 'obs_row'].values for x in an.stages)
        else:
            ra = rb = d['obs_row'].values
        if L <= 0 or (nf[wa][ra] < L).any() or (nf[wb][rb] < L).any():
            raise SystemExit(f'{an.id}: window of {L} frames does not fit every video ({wa}/{wb})')

    W_an = {an.id: common_frames(an, design) for an in D.analyses}
    print('common window W (s):', {k: v / fps for k, v in W_an.items()}, flush=True)

    def settings(an):
        return [dict(PRIMARY)] + [{**PRIMARY, **ov} for n, ov in SENS.items() if applicable(n, an)]

    rows, results, cens_rows = [], {}, []
    for an in D.analyses:
        aid = an.id
        results[aid] = {}
        W = W_an[aid]
        for s in settings(an):
            wa, wb = WINDOW_MAP[s['window']]
            L = window_frames(s['window'], W, n_match, args.n_trim)
            check_len(an, wa, wb, L)
            sm = summ(s['threshold_q'], L)
            for prefix in PREFIXES:
                if an.family == 'A':
                    units, Za, Zb = C.paired(sm, design, an.genotype, *an.stages, 'v', wa, wb, prefix)
                    if s['transform'] == 'rank':
                        R = rank_cols(np.vstack([Za, Zb]))
                        Za, Zb = R[:len(Za)], R[len(Za):]
                    res = paired_effect_search(Za, Zb, correction=s['correction'], test=s['test'])
                    extra = {}
                else:
                    units, Z, T = C.two_sample(sm, an.select(design), 'v', wa, prefix, an.unit)
                    if s['transform'] == 'rank':
                        Z = rank_cols(Z)
                    res = neural_effect_search(Z, T, correction=s['correction'])
                    extra = {'n_het': int(T.sum()), 'n_wt': int((1 - T).sum())}
                k = skey(prefix, s)
                results[aid][k] = {**strip(res), 'n_units': len(units), 'units': list(units), **extra,
                                   'W_frames': W, 'W_s': W / fps, 'window_frames': L, 'window_s': L / fps}
                meta = dict(analysis_id=aid, family=an.family, **an.meta, prefix=prefix, pooling=FP, **s,
                            n_units=len(units), setting=k)
                rows += tidy_rows(meta, res, an.directions)
                print(f'{aid} {k}: selected={res["selected"]} dropped={res["n_dropped"]}', flush=True)
                if s == PRIMARY or (s['window'] == 'trim30' and s['threshold_q'] == 0.95 and s['transform'] == 'none'
                                    and s['test'] == 't' and s['correction'] == 'bonferroni'):
                    cz = censored(s['threshold_q'], L)
                    if an.family == 'A':
                        _, Ca, Cb = C.paired(cz, design, an.genotype, *an.stages, 'v', wa, wb, prefix)
                        Cc = np.vstack([Ca, Cb])
                    else:
                        _, Cc, _ = C.two_sample(cz, an.select(design), 'v', wa, prefix, an.unit)
                    tested = np.setdiff1d(np.arange(Cc.shape[1]), res['dropped'])
                    ct = Cc[:, tested]
                    cens_rows.append({'analysis_id': aid, 'prefix': prefix, 'window': s['window'],
                                      'W_s': W / fps, 'window_s': L / fps, 'n_tested': len(tested), 'censored_cell_frac': float(ct.mean()),
                                      'median_latent_censored_frac': float(np.median(ct.mean(0))),
                                      'n_latents_censored_over_half': int((ct.mean(0) > 0.5).sum()),
                                      'n_latents_never_censored': int((ct.mean(0) == 0).sum())})
    for aid, r in results.items():
        (out / aid).mkdir(exist_ok=True)
        (out / aid / 'result.json').write_text(json.dumps(r))
    tidy = pd.DataFrame(rows)
    tidy.to_csv(out / 'summary.csv', index=False)
    cens = pd.DataFrame(cens_rows)
    cens.to_csv(out / 'censoring.csv', index=False)

    rng = np.random.default_rng(0)
    sanity = {'thresholds_n_frames': n_used, 'nulls': {}, 'fps': fps,
              'common_window_s': {k: v / fps for k, v in W_an.items()}, 'n_trim': args.n_trim, 'n_match': n_match,
              'censoring_primary_overall': float(cens[cens['window'] == 'common']['censored_cell_frac'].mean())}
    an2 = D.analysis(D.null_two)
    anp = D.analysis(D.null_paired) if D.null_paired else None
    rng0 = rng
    for prefix in PREFIXES if args.n_shuffles > 0 else ():
        rng = null_rng(rng0, prefix)
        sm = summ(0.95, W_an[an2.id])
        _, Z, T = C.two_sample(sm, an2.select(design), 'v', 'full', prefix, an2.unit)
        cnt = [len(neural_effect_search(Z, rng.permutation(T))['selected']) for _ in range(args.n_shuffles)]
        sanity['nulls'][f'p{prefix}'] = {f'{an2.id}_{D.shuffle_word}_shuffle_n_selected': cnt}
        if anp is not None:
            sm = summ(0.95, W_an[anp.id])
            _, Za, Zb = C.paired(sm, design, anp.genotype, *anp.stages, 'v', 'full', 'full', prefix)
            ca = []
            for _ in range(args.n_shuffles):
                sw = rng.random(len(Za)) < 0.5
                ca.append(len(paired_effect_search(np.where(sw[:, None], Zb, Za), np.where(sw[:, None], Za, Zb))['selected']))
            sanity['nulls'][f'p{prefix}'][f'{anp.id}_stage_swap_n_selected'] = ca
        print(f'nulls p{prefix}:', sanity['nulls'][f'p{prefix}'], flush=True)
    sanity['runtime_s'] = time.time() - t_start
    (out / 'sanity.json').write_text(json.dumps(C.to_jsonable(sanity), indent=1))
    write_reports(out, tidy, cens, design, D, {a: censored(0.95, w) for a, w in W_an.items()}, sanity, args.sae, FP)
    print(f'done in {time.time() - t_start:.0f}s -> {out}', flush=True)


def write_reports(out, tidy, cens, design, D, cz_an, sanity, sae, FP):
    """cz_an: {analysis_id: censored indicators of the primary (common) window, q 0.95}."""
    names = list(SENS)
    L = [f'# NES summary ({FP} frame values, LATENCY outcome): {sae}', '',
         f'Outcome = time (s) from the start of the window to the first bout of each latent. Bout as the event-rate '
         f'runs: frame value codes_{FP} > the latent\'s pooled q-quantile threshold (treatment-agnostic, '
         f'{sanity["thresholds_n_frames"]} sampled frames, seed 0), min 1 frame. Common window: bouts are looked for '
         'only in the first W s of every video, W = the shortest video of the analysis (W per analysis below), and '
         'videos with no bout there get W (right-censored at W), so videos of different lengths are compared on the '
         'same window. tau > 0 = later first bout in the treated arm (family B) / at the later stage '
         '(family A). Primary: q 0.95, t-test, Bonferroni alpha 0.05, common window. trim30 = window [30 s, W). '
         'matched = last min(n_match, W) frames of habituation vs the first min(n_match, W) frames of the later '
         'stage. rank = the same NES on per-latent '
         'ranks (ties averaged: all censored videos tie). No nuisance conditioning.', '',
         f'Common window W (s) per analysis: '
         + ', '.join(f'{k} {v:g}' for k, v in sanity['common_window_s'].items()) + '.', '',
         f'Censoring (primary, common window, tested latents): mean over analyses / prefixes of the censored cell '
         f'fraction = {sanity["censoring_primary_overall"]:.3f}; per analysis in censoring.csv.', '',
         'Robustness columns: Y = also selected (any round) under that single change; "-" = not applicable. '
         'cens = censored fraction of the selected latent in each arm (control / treated; family A: stage a / b).', '']
    for an in D.analyses:
        aid = an.id
        L.append(f'## {aid}' + (f' ({an.meta.get("confound")})' if an.meta.get('confound') else '')
                 + f' - W = {sanity["common_window_s"][aid]:g} s')
        cz = cz_an[aid]
        for prefix in PREFIXES:
            sub = tidy[(tidy['analysis_id'] == aid) & (tidy['prefix'] == prefix) & (tidy['setting'] == skey(prefix, PRIMARY))]
            nd = int(sub['n_dropped'].iloc[0])
            c = cens[(cens['analysis_id'] == aid) & (cens['prefix'] == prefix) & (cens['window'] == 'common')].iloc[0]
            ps = sub[sub['round'] > 0].sort_values('round')
            head = (f'- prefix {prefix}: {len(ps)} selected ({nd} dropped; censored cells {c.censored_cell_frac:.2f}, '
                    f'{int(c.n_latents_censored_over_half)}/{int(c.n_tested)} latents censored in > half the videos)')
            if not len(ps):
                L.append(head.replace(f'{len(ps)} selected', 'nothing selected'))
                continue
            L += [head, '', '| round | neuron | direction | tau (s) | p | cens ctl/trt | ' + ' | '.join(names) + ' |',
                  '|' + '---|' * (6 + len(names))]
            for _, r in ps.iterrows():
                j = int(r['neuron'])
                marks = ['Y' if j in select(tidy, aid, prefix, {**PRIMARY, **SENS[n]}) else 'N' if applicable(n, an) else '-'
                         for n in names]
                if an.family == 'A':
                    _, Ca, Cb = C.paired(cz, design, an.genotype, *an.stages, 'v', 'full', 'full')
                    c0, c1 = Ca[:, j].mean(), Cb[:, j].mean()
                else:
                    _, Cc, T = C.two_sample(cz, an.select(design), 'v', 'full', None, an.unit)
                    c0, c1 = Cc[T == 0, j].mean(), Cc[T == 1, j].mean()
                L.append(f'| {int(r["round"])} | {j} | {r["direction"]} | {r["tau"]:.3g} | {r["p"]:.2e} | {c0:.2f}/{c1:.2f} | '
                         + ' | '.join(marks) + ' |')
            L.append('')
        L.append('')
    L += ['## Sanity checks (primary setting)', '']
    for p, v in sanity['nulls'].items():
        for k, cnt in v.items():
            L.append(f'- {p} {k}: {cnt}')
    L.append(f'- runtime: {sanity["runtime_s"]:.0f} s')
    (out / 'SUMMARY.md').write_text('\n'.join(L) + '\n')
    union = {}
    for _, r in tidy[(tidy['round'] > 0)].iterrows():
        if r['setting'] != skey(int(r['prefix']), PRIMARY):
            continue
        union.setdefault(int(r['neuron']), {'hits': []})['hits'].append(
            {'analysis_id': r['analysis_id'], 'prefix': int(r['prefix']), 'round': int(r['round']),
             'direction': r['direction'], 'tau_s': float(r['tau']), 'p': float(r['p'])})
    (out / 'selected_neurons.json').write_text(json.dumps(
        {'sae': sae, 'settings': f'primary (codes_{FP}, latency to first bout q0.95, t, bonferroni, common window: '
                                 'first W s of every video, W = shortest video of the analysis)',
         'common_window_s': sanity['common_window_s'],
         'neurons': {str(j): v for j, v in sorted(union.items())}}, indent=1))


if __name__ == '__main__':
    main()
