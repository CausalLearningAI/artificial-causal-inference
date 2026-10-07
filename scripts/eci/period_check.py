"""
Camera-period checks of the mice v1 family-B (het vs wt) NES results of one SAE (src/eci/period.py):

  1. period-dependence flag of every latent (folded within-genotype period AUC on mouse-level outcomes, threshold =
     95th percentile of the max over latents under a within-genotype label shuffle), per outcome;
  2. day-adjusted re-test of the primary family-B picks (all 72 videos per stage, recording day as covariate).

Read-only on the existing NES outputs: uses the per-video caches (<nes>/<sae>/_cache/) and the primary picks of
<nes>/<sae>/summary.csv (outcome 'mean' = per-video mean activation) and <nes>/<sae>/<P>pool_bouts/summary.csv
(outcome 'bout_rate', q 0.95, gap 0). The primary analysis is unchanged (unadjusted, all videos). Per-video values in
the primary window of every video (src/eci/domain.py Domain.video_window; mice: habituation = its last 15 min).

Writes <nes>/<sae>/period_check/:
  period_flags.json  per outcome: threshold, null max quantiles, per-latent score / pair / auc / p_perm / br_z /
                     flagged (lists of length m, index = latent id), and the primary picks with their flag
  day_adjusted.csv   one row per primary family-B pick: unadjusted tau / p / threshold and tau_day / p_day / survives
  SUMMARY.md

Usage: python scripts/eci/period_check.py --sae matryoshka_btk_1024_k16_fg448al_s0 --pooling max
       python scripts/eci/period_check.py --sae matryoshka_btk_1024_k16_fg448al_s0_mean --pooling mean
Mice only: ants have no within-arm period contrast (v3 day C holds only t = 8 / 9, days A / B only t = 2 / 4 / 6 / 7).
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
from eci import period as P  # noqa: E402
from eci.domain import get_domain  # noqa: E402

FPS = 5.0
BOUT = (0.95, 0)  # primary bout rule: q, merge gap


def load_cache(path, design):
    f = np.load(path, allow_pickle=False)
    if not np.array_equal(f['observation_id'], design['observation_id'].values.astype(str)):
        raise SystemExit(f'{path}: observation order differs from the design')
    return f


def primary_picks(tidy, pooling, bouts):
    t = tidy[(tidy['round'] > 0) & (tidy['pooling'] == pooling) & (tidy['test'] == 't')
             & (tidy['correction'] == 'bonferroni') & (tidy['window'] == 'full')]
    if bouts:
        t = t[(t['outcome_type'] == 'bout_rate') & np.isclose(t['threshold_q'], BOUT[0]) & (t['merge_gap'] == BOUT[1])
              & (t['bout_rule'] == 0)]
    else:
        t = t[t['outcome_type'] == 'mean']
    return t.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sae', required=True)
    ap.add_argument('--pooling', required=True, choices=('max', 'mean'), help='primary per-frame pooling of the SAE')
    ap.add_argument('--n-shuffles', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-root', default=None)
    args = ap.parse_args()
    t_start = time.time()
    D = get_domain('mice')
    nes = Path(args.out_root or D.nes_root) / args.sae
    out = nes / 'period_check'
    out.mkdir(parents=True, exist_ok=True)
    design = D.load_design()
    if not np.array_equal(design['obs_row'].values, np.arange(len(design))):
        raise SystemExit('design obs_row is not 0..n-1')
    per = P.pool_periods(design, D.eci_dir / 'odor_corner.csv')
    day = P.recording_days(design, D.experiment_csv)
    sfx = '' if args.pooling == 'max' else f'_{args.pooling}'
    vs = load_cache(nes / '_cache' / f'video_summaries_{args.pooling}.npz', design)
    bs = load_cache(nes / '_cache' / f'bout_summaries_max{sfx}.npz', design)
    q, g = BOUT
    wins = D.video_window(design)  # the primary window of every video (mice: habituation = its last 15 min)
    values = {'mean': C.per_video({w: vs[f'{w}__mean'].astype(np.float64) for w in C.WINDOWS}, wins),
              'bout_rate': C.per_video({w: bs[f'{w}__{q}__{g}__count'] / (bs[f'{w}__n_frames'][:, None] / FPS / 60.0)
                                        for w in C.WINDOWS}, wins)}
    tidies = {'mean': pd.read_csv(nes / 'summary.csv'),
              'bout_rate': pd.read_csv(nes / f'{args.pooling}pool_bouts' / 'summary.csv')}
    analyses = {a.id: a for a in D.analyses}

    flags = {'sae': args.sae, 'pooling': args.pooling, 'definition': P.__doc__.split('2. Day-adjusted')[0].strip(),
             'unit': 'mouse (pool): per-video outcome (primary window: habituation = its last 15 min) averaged over '
                     'the 6 stage videos',
             'pairs': [f'{gn}: {a} vs {b}' for gn, a, b in P.PAIRS], 'n_shuffles': args.n_shuffles,
             'seed': args.seed, 'bout_rule': {'q': q, 'merge_gap': g}, 'outcomes': {}}
    day_rows, lines = [], [f'# Camera-period checks: {args.sae} (codes_{args.pooling})', '']
    for oc, Y in values.items():
        pools, Yp = P.pool_table(design, Y, per)
        s = P.period_scores(pools, Yp, n_shuffles=args.n_shuffles, seed=args.seed)
        flagged = s['score'] > s['threshold']
        picks = primary_picks(tidies[oc], args.pooling, oc == 'bout_rate')
        disc = []
        for r in picks.sort_values(['family', 'analysis_id', 'prefix', 'round']).to_dict('records'):
            j = int(r['neuron'])
            disc.append({'analysis_id': r['analysis_id'], 'family': r['family'], 'prefix': int(r['prefix']),
                         'round': int(r['round']), 'neuron': j, 'score': round(float(s['score'][j]), 4),
                         'pair': flags['pairs'][int(s['pair'][j])], 'auc': round(float(s['auc'][j]), 4),
                         'p_perm': round(float(s['p_perm'][j]), 4), 'flagged': bool(flagged[j])})
        nq = np.quantile(s['null_max'], [0.5, 0.9, 0.95, 0.99])
        flags['outcomes'][oc] = {
            'threshold': round(s['threshold'], 4), 'pointwise_threshold': round(s['pointwise_threshold'], 4),
            'null_max_quantiles': {'0.5': nq[0], '0.9': nq[1], '0.95': nq[2],
                                                                          '0.99': nq[3]},
            'n_mice': s['n_mice'], 'n_br_mice': s['n_br'], 'n_latents': int(len(s['score'])),
            'n_flagged': int(flagged.sum()),
            'latents': {'score': np.round(s['score'], 4).tolist(), 'pair': s['pair'].astype(int).tolist(),
                        'auc': np.round(s['auc'], 4).tolist(), 'p_perm': np.round(s['p_perm'], 4).tolist(),
                        'br_z': np.round(np.nan_to_num(s['br_z']), 3).tolist(), 'flagged': flagged.tolist()},
            'discovered': disc}
        db = [d for d in disc if d['family'] == 'B']
        lines += [f'## Outcome {oc}', '',
                  f'Threshold (95th pct of the max folded AUC over {len(s["score"])} latents, {args.n_shuffles} '
                  f'within-genotype shuffles): {s["threshold"]:.3f} (null max median {nq[0]:.3f}, 99th pct {nq[3]:.3f}; one '
                  f'latent alone, no multiplicity control: {s["pointwise_threshold"]:.3f}). '
                  f'Mice per pair (period a, b): {s["n_mice"]}. Flagged latents: {int(flagged.sum())}/{len(s["score"])}.',
                  f'Family-B primary picks flagged: {sum(d["flagged"] for d in db)}/{len(db)}; family A (not '
                  f'confounded by the period: paired within mouse) {sum(d["flagged"] for d in disc if d["family"] == "A")}'
                  f'/{len(disc) - len(db)}.', '',
                  '| analysis | prefix | round | neuron | score | pair | AUC | p_perm | flagged |', '|---|---|---|---|---|---|---|---|---|']
        lines += [f'| {d["analysis_id"]} | {d["prefix"]} | {d["round"]} | {d["neuron"]} | {d["score"]:.3f} | {d["pair"]} | '
                  f'{d["auc"]:.3f} | {d["p_perm"]:.3g} | {"FLAG" if d["flagged"] else ""} |' for d in db]
        lines.append('')
        pb = picks[picks['family'] == 'B']
        if len(pb):
            da = P.day_adjusted_picks(pb, design, Y, day, analyses)
            da.insert(0, 'outcome', oc)
            day_rows.append(da)
        print(f'{oc}: threshold {s["threshold"]:.4f}, flagged {int(flagged.sum())}, B picks flagged '
              f'{sum(d["flagged"] for d in db)}/{len(db)}', flush=True)

    with open(out / 'period_flags.json', 'w') as f:
        json.dump(flags, f, default=float)
    cols = ['outcome', 'analysis_id', 'prefix', 'round', 'neuron', 'direction', 'tau', 'p', 'threshold', 'tau_day',
            'se_day', 'p_day', 'df_day', 'n_cells', 'n_units', 'survives', 'not_estimable']
    da = pd.concat(day_rows)[cols] if day_rows else pd.DataFrame(columns=cols)
    da.to_csv(out / 'day_adjusted.csv', index=False)
    lines += ['## Day-adjusted re-test (family B primary picks, all 72 videos per stage)', '',
              'Covariate: days since the first recording (file-name timestamp), conditioned on as in NES (`S = [day] + '
              'earlier picks`). survives = p_day < the round threshold, same sign.', '',
              '| outcome | analysis | prefix | round | neuron | tau | p | threshold | tau_day | p_day | survives |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    lines += [f'| {r.outcome} | {r.analysis_id} | {int(r.prefix)} | {int(r.round)} | {int(r.neuron)} | {r.tau:.4g} | {r.p:.2e} | '
              f'{r.threshold:.2e} | {r.tau_day:.4g} | {r.p_day:.2e} | {"Y" if r.survives else "N"} |'
              for r in da.itertuples()]
    lines += ['', f'Runtime {time.time() - t_start:.0f}s.']
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print(f'day-adjusted: {int(da["survives"].sum())}/{len(da)} survive; done in {time.time() - t_start:.0f}s -> {out}',
          flush=True)


if __name__ == '__main__':
    main()
