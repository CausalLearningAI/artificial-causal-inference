"""
Hand-crafted blob measures of the foreground mask, tested per analysis with plain per-video tests (no dictionary,
no NES recursion). Frame measures from <codes>/<sae>_blobs/blobs.npy (src/eci/spatial_encode.py: connected
components of the foreground mask on the 32 x 32 patch grid, 8-connectivity, blob = component of >= 2 patches):
    n_blobs              number of blobs in the frame
    mean_blob_distance   mean pairwise distance between blob centroids (patch units; 0 when < 2 blobs)
Per-video outcomes (full window), as the NES runners:
    mean   average over the video's frames ("average time")
    rate   bouts per minute, bout = maximal run of frames with the measure > its pooled 0.95-quantile threshold over
           all frames of all videos (treatment-agnostic; scripts/eci/run_nes_bouts.py definition, gap 0, min 1 frame)
Tests: family B Welch two-sample t (nes.neural_effect_test with nothing conditioned), family A paired t on the
per-pool change (nes.paired_effect_test). Bonferroni over the 4 tests of an analysis (2 measures x 2 outcomes),
alpha 0.05. Calibration: the domain's two-sample null analysis with labels shuffled 1000 times -> fraction of
shuffles with any Bonferroni rejection (should be <= 0.05).
Mice extra (descriptive, outside the Bonferroni family): near_odor_frac = per frame, the fraction of foreground
patches in the near-odor zone (<sae>_zones/zone_nfg.npy column 0 / n_fg; frames without foreground skipped from the
mean), per-video mean, same tests.

Output: <nes root>/<sae>_blobs/[<set>/]blob_tests.csv, SUMMARY.md, video_outcomes.csv.
Usage: python scripts/eci/blob_tests.py --domain mice --sae matryoshka_btk_1024_k16_fg448_s0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from eci import contrasts as C  # noqa: E402
from eci.domain import DOMAINS, get_domain  # noqa: E402
from eci.nes import neural_effect_test, paired_effect_test  # noqa: E402

MEASURES = ('n_blobs', 'mean_blob_distance')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='mice', choices=DOMAINS)
    ap.add_argument('--analysis-set', default='core')
    ap.add_argument('--sae', required=True)
    ap.add_argument('--n-shuffles', type=int, default=1000)
    args = ap.parse_args()
    D = get_domain(args.domain, args.analysis_set)
    cdir = D.codes_root / f'{args.sae}_blobs'
    if not (cdir / 'DONE').exists():
        raise SystemExit(f'{cdir}/DONE missing')
    out = D.nes_root / f'{args.sae}_blobs' / ('' if args.analysis_set == 'core' else args.analysis_set)
    out.mkdir(parents=True, exist_ok=True)
    cache = D.nes_root / f'{args.sae}_blobs' / '_cache'
    cache.mkdir(parents=True, exist_ok=True)
    design = D.load_design()
    B = np.load(cdir / 'blobs.npy', mmap_mode='r')
    X = np.asarray(B[:, :2], dtype=np.float32)
    extra = []
    if args.domain == 'mice' and (D.codes_root / f'{args.sae}_zones' / 'zone_nfg.npy').exists():
        zn = np.load(D.codes_root / f'{args.sae}_zones' / 'zone_nfg.npy')
        nfg = np.load(cdir / 'n_fg.npy').astype(np.float32)
        near = np.where(nfg > 0, zn[:, 0] / np.maximum(nfg, 1), np.nan).astype(np.float32)
        extra = ['near_odor_frac']
    fpath = cache / 'blob_frames.npy'
    np.save(fpath, X)
    thr, n_used = C.pooled_thresholds(fpath, (0.95,), 2_000_000, seed=0)
    msum = C.video_summaries(fpath, design, D.n_match)
    bs = C.bout_summaries(fpath, design, {0.95: thr[0]}, [(0.95, 0)], D.n_match)
    rate = C.bout_outcomes(bs, 'full', 0.95, 0, D.fps)['rate']
    vals = {('mean', i): msum[('full', 'mean')][:, i] for i in range(2)}
    vals.update({('rate', i): rate[:, i] for i in range(2)})
    names = {('mean', i): (m, 'mean') for i, m in enumerate(MEASURES)}
    names.update({('rate', i): (m, 'rate') for i, m in enumerate(MEASURES)})
    if extra:
        nv = np.array([np.nanmean(near[s:e]) for s, e in zip(design['row_start'], design['row_end'])])
        vals[('mean', 2)] = nv
        names[('mean', 2)] = ('near_odor_frac', 'mean')
    keys = list(vals)
    V = np.column_stack([vals[k] for k in keys])
    vo = design[[c for c in design.columns if c not in ('row_start', 'row_end')]].copy()
    for k, col in zip(keys, V.T):
        vo['_'.join(names[k])] = col
    vo.to_csv(out / 'video_outcomes.csv', index=False)
    summ = {('full', 'v'): V}
    fam = [i for i, k in enumerate(keys) if names[k][0] in MEASURES]  # the Bonferroni family (4 tests)
    rows = []
    for an in D.analyses:
        if an.family == 'A':
            _, Za, Zb = C.paired(summ, design, an.genotype, *an.stages, 'v')
            tab, _ = paired_effect_test(Za, Zb)
            ma, mb = Za.mean(0), Zb.mean(0)
            n = len(Za)
        else:
            _, Z, T = C.two_sample(summ, an.select(design), 'v', unit=an.unit)
            tab, _ = neural_effect_test(Z, T)
            ma, mb = Z[T == 0].mean(0), Z[T == 1].mean(0)
            n = len(Z)
        for i, k in enumerate(keys):
            r = tab.iloc[i]
            infam = i in fam
            rows.append({'analysis_id': an.id, 'family': an.family, **{m: an.meta.get(m, '') for m in an.meta},
                         'measure': names[k][0], 'outcome': names[k][1], 'n_units': n,
                         'mean_control_or_a': ma[i], 'mean_treated_or_b': mb[i], 'tau': r['tau'], 't': r['t'],
                         'df': r['df'], 'p': r['p'], 'in_bonferroni_family': infam,
                         'p_bonferroni': min(1.0, r['p'] * len(fam)) if infam else np.nan,
                         'significant': bool(infam and r['p'] < 0.05 / len(fam))})
    res = pd.DataFrame(rows)
    res.to_csv(out / 'blob_tests.csv', index=False)
    # calibration: label shuffle on the two-sample null analysis
    an0 = D.analysis(D.null_two)
    _, Z, T = C.two_sample(summ, an0.select(design), 'v', unit=an0.unit)
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(args.n_shuffles):
        tab, _ = neural_effect_test(Z[:, fam], rng.permutation(T))
        hits += int((tab['p'].values < 0.05 / len(fam)).any())
    calib = {'null_analysis': an0.id, 'n_shuffles': args.n_shuffles, 'frac_any_rejection': hits / args.n_shuffles,
             'thresholds_q0.95': {m: float(thr[0][i]) for i, m in enumerate(MEASURES)}, 'n_frames_threshold': n_used}
    (out / 'calibration.json').write_text(json.dumps(calib, indent=1))
    L = [f'# Blob measures: {args.sae} ({D.title}, analysis set {args.analysis_set})', '',
         'Connected components of the foreground mask (32 x 32 patches, 8-connectivity, blob >= 2 patches). '
         'mean = per-video average; rate = bouts/min above the pooled q0.95 threshold '
         f'({", ".join(f"{m} > {thr[0][i]:.3g}" for i, m in enumerate(MEASURES))}). Bonferroni over the 4 tests of each '
         f'analysis. Label-shuffle calibration ({an0.id}, {args.n_shuffles} shuffles): any rejection in '
         f'{calib["frac_any_rejection"]:.3f} of shuffles.' + (' near_odor_frac is descriptive (outside the family).'
                                                              if extra else ''), '',
         '| analysis | confound | measure | outcome | control/a | treated/b | tau | p | p Bonf | sig |', '|' + '---|' * 10]
    for r in res.itertuples():
        L.append(f'| {r.analysis_id} | {getattr(r, "confound", "") or ""} | {r.measure} | {r.outcome} | '
                 f'{r.mean_control_or_a:.3g} | {r.mean_treated_or_b:.3g} | {r.tau:.3g} | {r.p:.2e} | '
                 f'{"" if np.isnan(r.p_bonferroni) else f"{r.p_bonferroni:.3g}"} | {"*" if r.significant else ""} |')
    (out / 'SUMMARY.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
