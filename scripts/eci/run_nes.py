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

Usage: python scripts/eci/run_nes.py --sae matryoshka_btk_1024_k16_ep20_s0
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


def tidy_rows(meta, res):
    base = {**meta, 'n_tested_total': res['n_tested'], 'n_dropped': res['n_dropped']}
    rounds = res['rounds']
    if len(rounds) == 0:
        return [{**base, 'round': 0}]
    rows = []
    for r in rounds.to_dict('records'):
        if meta['family'] == 'A':
            direction = 'up' if r['tau'] > 0 else 'down'
        else:
            direction = 'het>wt' if r['tau'] > 0 else 'het<wt'
        rows.append({**base, **{k: r[k] for k in ('round', 'neuron', 'tau', 'se', 't', 'df', 'p', 'threshold', 'n_tested')},
                     'direction': direction})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sae', default='matryoshka_btk_1024_k16_ep20_s0')
    ap.add_argument('--codes-root', default=str(ROOT / 'dataset/mice/v1/eci/codes'))
    ap.add_argument('--out-root', default=str(ROOT / 'results/vision/mice/eci/nes'))
    ap.add_argument('--n-match', type=int, default=4500)
    ap.add_argument('--n-shuffles', type=int, default=20)
    ap.add_argument('--frame-max-rounds', type=int, default=8)  # strata double per round at frame level
    ap.add_argument('--skip-frame', action='store_true')
    ap.add_argument('--primary-pooling', default='mean', choices=('mean', 'max'))
    args = ap.parse_args()
    t_start = time.time()
    global ARTEFACTS
    ARTEFACTS = ARTEFACTS_EP20 if args.sae == 'matryoshka_btk_1024_k16_ep20_s0' else {}
    pp = args.primary_pooling
    codes_dir = Path(args.codes_root) / args.sae
    if not (codes_dir / 'DONE').exists():
        raise SystemExit(f'{codes_dir}/DONE missing: codes not finished')
    out = Path(args.out_root) / args.sae
    out.mkdir(parents=True, exist_ok=True)

    design = C.load_design(ROOT / 'dataset/mice/v1/annotations.csv', ROOT / 'data/mice/v1/experiment.csv')
    dur = design.groupby('stage')['n_frames'].agg(['count', 'min', 'max', 'mean']).reset_index()
    dur['minutes_at_5fps'] = dur['mean'] / 5 / 60
    dur.to_csv(out / 'stage_durations.csv', index=False)
    print('stage durations (frames per observation):\n', dur.to_string(index=False), flush=True)
    print('pools:', design['pool'].nunique(), 'het pools:', design[design['T'] == 1]['pool'].nunique(),
          'rd64 rows:', design[design['pool'] == 'rd64'][['observation_id', 'genotype', 'stage']].values.tolist(), flush=True)

    summ = {}
    for pooling in ('mean', 'max'):
        t0 = time.time()
        summ[pooling] = C.cached_summaries(codes_dir / f'codes_{pooling}.npy', design,
                                           out / '_cache' / f'video_summaries_{pooling}.npz', args.n_match)
        print(f'summaries codes_{pooling}: {time.time() - t0:.0f}s', flush=True)

    all_rows, results, sanity = [], {}, {'genotype_balance': genotype_balance(design)}
    print('genotype balance across pool-level fields:', json.dumps(sanity['genotype_balance'])[:2000], flush=True)

    # ---- A: paired stage transitions
    for geno in ('het', 'wt'):
        for tr, (a, b) in C.TRANSITIONS.items():
            aid = f'A_{geno}_{tr}'
            results[aid] = {}
            windows = ['full', 'trim30'] + (['matched'] if a in (1, 4) else [])
            for prefix in PREFIXES:
                for pooling in ('mean', 'max'):
                    for stat in ('mean', 'rate'):
                        for window in windows:
                            wa, wb = WINDOW_MAP[window]
                            pools, Za, Zb = C.paired(summ[pooling], design, geno, a, b, stat, wa, wb, prefix)
                            for test in ('t', 'signflip'):
                                for corr in ('bonferroni', 'bh'):
                                    # sign-flip is a one-change sensitivity of the primary setting only
                                    # (~3 min per search at m = 1024 on one core)
                                    if test == 'signflip' and (pooling, stat, corr, window) != (pp, 'mean', 'bonferroni', 'full'):
                                        continue
                                    t0 = time.time()
                                    res = paired_effect_search(Za, Zb, correction=corr, test=test)
                                    k = key(prefix, pooling, stat, test, corr, window)
                                    results[aid][k] = {**strip(res), 'n_units': len(pools), 'units': list(pools)}
                                    meta = dict(analysis_id=aid, family='A', genotype=geno, stage='', transition=tr,
                                                prefix=prefix, pooling=pooling, outcome_type=stat, test=test,
                                                correction=corr, window=window, n_units=len(pools), setting=k)
                                    all_rows += tidy_rows(meta, res)
                                    print(f'{aid} {k}: n={len(pools)} selected={res["selected"]} '
                                          f'dropped={res["n_dropped"]} {time.time() - t0:.1f}s', flush=True)

    # ---- B: genotype within stage
    for stage in range(1, 7):
        aid = f'B_stage{stage}'
        results[aid] = {}
        for prefix in PREFIXES:
            for pooling in ('mean', 'max'):
                for stat, window in (('mean', 'full'), ('rate', 'full'), ('mean', 'trim30'), ('rate', 'trim30')):
                    pools, Z, T = C.genotype_contrast(summ[pooling], design, stage, stat, WINDOW_MAP[window][0], prefix)
                    for corr in ('bonferroni', 'bh'):
                        res = neural_effect_search(Z, T, correction=corr)
                        k = key(prefix, pooling, stat, 't', corr, window)
                        results[aid][k] = {**strip(res), 'n_units': len(pools), 'n_het': int(T.sum()),
                                           'n_wt': int((1 - T).sum()), 'units': list(pools)}
                        meta = dict(analysis_id=aid, family='B', genotype='het_vs_wt', stage=stage, transition='',
                                    prefix=prefix, pooling=pooling, outcome_type=stat, test='t', correction=corr,
                                    window=window, n_units=len(pools), setting=k)
                        all_rows += tidy_rows(meta, res)
                        print(f'{aid} {k}: n={int(T.sum())}+{int((1 - T).sum())} selected={res["selected"]} '
                              f'dropped={res["n_dropped"]}', flush=True)

    for aid, r in results.items():
        (out / aid).mkdir(exist_ok=True)
        with open(out / aid / 'result.json', 'w') as f:
            json.dump(r, f)
    tidy = pd.DataFrame(all_rows)
    tidy.to_csv(out / 'summary.csv', index=False)

    # ---- sanity: permutation of genotype across pools, stage 2, primary settings
    rng = np.random.default_rng(0)
    perm = {}
    for prefix in PREFIXES:
        pools, Z, T = C.genotype_contrast(summ[pp], design, 2, 'mean', 'full', prefix)
        counts, naive = [], []
        for _ in range(args.n_shuffles):
            res = neural_effect_search(Z, rng.permutation(T))
            counts.append(len(res['selected'])), naive.append(int(res['first_round']['significant'].sum()))
        perm[prefix] = {'n_selected': counts, 'n_naive_significant': naive}
        print(f'permutation prefix {prefix}: selected {counts} naive {naive}', flush=True)
    sanity['permutation_B_stage2'] = perm

    # ---- C: pseudo-replication, genotype at stage 2
    if not args.skip_frame:
        frame = {}
        for prefix in PREFIXES:
            t0 = time.time()
            Zf, Tf, gf = C.frame_matrix(codes_dir / f'codes_{pp}.npy', design, 2, prefix)
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
            (out / 'C_frame_stage2').mkdir(exist_ok=True)
            with open(out / 'C_frame_stage2' / f'result_p{prefix}.json', 'w') as f:
                json.dump({'frame': strip(fres), 'video': strip(vres)}, f)
            print(f'frame-level prefix {prefix}:', {k: v for k, v in frame[prefix].items() if 'selected' not in k or 'nes' in k}, flush=True)
            del Zf
        sanity['pseudo_replication_stage2'] = frame

    # ---- size check: round-1 neurons of the primary re-tested with the per-video mean n_fg
    sadj = None
    if (codes_dir / 'n_fg.npy').exists():
        nfg = C.video_nfg(codes_dir / 'n_fg.npy', design, args.n_match)
        t = tidy[(tidy['round'] == 1) & (tidy['pooling'] == pp) & (tidy['outcome_type'] == 'mean') & (tidy['test'] == 't')
                 & (tidy['correction'] == 'bonferroni') & tidy['window'].isin(['full', 'trim30'])]
        sadj = C.size_adjusted_round1(t, design, {w: summ[pp][(w, 'mean')] for w in C.WINDOWS}, nfg, WINDOW_MAP)
        cols = ['analysis_id', 'prefix', 'window', 'neuron', 'direction', 'tau', 'p', 'threshold', 'tau_adj', 'se_adj',
                'p_adj', 'df_adj', 'slope_nfg', 'survives']
        sadj = sadj[cols] if len(sadj) else pd.DataFrame(columns=cols)
        sadj.to_csv(out / 'size_adjusted.csv', index=False)
        sanity['size_check'] = {'n_round1': int(len(sadj)), 'n_survive': int(sadj['survives'].sum())}
        print('size check:', sanity['size_check'], flush=True)

    sanity['runtime_s'] = time.time() - t_start
    with open(out / 'sanity.json', 'w') as f:
        json.dump(C.to_jsonable(sanity), f, indent=1)
    write_reports(out, tidy, results, dur, sanity, args.sae, pp, sadj)
    print(f'done in {time.time() - t_start:.0f}s -> {out}', flush=True)


def genotype_balance(design):
    """Pool-level het/wt counts per recording field, with a chi-square test of independence.
    'cage_pos' = the _1/_2/_3 suffix of the pool id (cage position on the recording day; 'none' if
    the pool id has no suffix), 'month' = recording month."""
    from scipy.stats import chi2_contingency
    e = pd.read_csv(ROOT / 'data/mice/v1/experiment.csv')
    p = e.drop_duplicates('pool').copy()
    p['cage_pos'] = p['pool'].str.extract(r'_(\d)$')[0].fillna('none')
    p['month'] = p['date'].str[:7]
    p['annotator'] = p['annotator'].fillna('none')
    p['hour'] = p['time'].str[:2]
    out = {}
    for c in ('line', 'sex', 'seed', 'cage_pos', 'month', 'hour', 'annotator'):
        tab = pd.crosstab(p[c], p['genotype'])
        pval = float(chi2_contingency(tab.values)[1]) if tab.shape[0] > 1 else 1.0
        out[c] = {'counts': {str(k): {g: int(v) for g, v in r.items()} for k, r in tab.iterrows()}, 'chi2_p': pval}
    return out


def selected_set(tidy, aid, **kw):
    s = tidy[(tidy['analysis_id'] == aid) & (tidy['round'] > 0)]
    for k, v in kw.items():
        s = s[s[k] == v]
    return dict(zip(s['neuron'].astype(int), s['direction']))


def write_reports(out, tidy, results, dur, sanity, sae, pp='mean', sadj=None):
    alt = 'max' if pp == 'mean' else 'mean'
    sens = {  # name -> overrides of the primary setting (same prefix)
        'signflip': dict(test='signflip'), f'{alt}-pool': dict(pooling=alt), 'rate': dict(outcome_type='rate'),
        'BH': dict(correction='bh'), 'matched': dict(window='matched'), 'trim30': dict(window='trim30')}
    prim = dict(pooling=pp, outcome_type='mean', test='t', correction='bonferroni', window='full')
    union = {}
    lines = [f'# NES summary: {sae}', '',
             f'Video-level Neural Effect Search on mice v1. Primary: codes_{pp} pooling, outcome = per-video mean '
             'activation, t-test, Bonferroni alpha 0.05, full stage window. Family A = paired stage transition within '
             'genotype (unit = pool, tau > 0 = increase from stage a to b); family B = het vs wt within stage '
             '(unit = video, tau > 0 = higher in het). Prefix 128 = first 128 Matryoshka latents; neuron ids are '
             'shared with prefix 1024.', '',
             'Robustness columns: Y = the neuron is also selected (any round) under that single change from primary '
             '(same prefix); N = not; "-" = not applicable. "other prefix" = selected in the same analysis at the '
             'other prefix (only neurons < 128 can appear at prefix 128). size-adj = round-1 neuron still significant '
             '(same sign, p below its round-1 threshold) with the per-video mean foreground patch count as a covariate.', '',
             'Stage durations (frames at 5 fps): ' + ', '.join(f'stage {int(r.stage)}: {int(r["mean"])}' for _, r in dur.iterrows()), '']
    for aid in results:
        fam = aid[0]
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
                    if name == 'signflip' and fam != 'A' or name == 'matched' and not aid.endswith(('1to2', '4to5')):
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
    perm = sanity.get('permutation_B_stage2', {})
    lines += ['## Genotype balance across recording fields (pool level, 36 het / 36 wt)', '']
    for c, v in sanity.get('genotype_balance', {}).items():
        cnt = ', '.join(f'{k}: {d.get("het", 0)}/{d.get("wt", 0)}' for k, d in v['counts'].items())
        lines.append(f'- {c} (het/wt): {cnt}; chi-square p = {v["chi2_p"]:.3g}')
    lines += ['', '## Sanity checks', '']
    for p, v in perm.items():
        lines.append(f'- Genotype labels shuffled across pools (stage 2, primary, prefix {p}), {len(v["n_selected"])} '
                     f'shuffles: NES selected {v["n_selected"]}; naive first-round Bonferroni {v["n_naive_significant"]}')
    for p, v in sanity.get('pseudo_replication_stage2', {}).items():
        lines.append(f'- Pseudo-replication, genotype at stage 2, prefix {p}: frame level ({v["n_frames"]} frames) naive '
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
