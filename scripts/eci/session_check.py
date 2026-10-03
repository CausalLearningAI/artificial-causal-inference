"""
Recording-session checks of the frogs family-B (mutant vs WT) NES results of one SAE (src/eci/session.py):

  1. session-dependence flag of every latent (eta^2 of the within-group ranks on session, max over the groups with
     >= 2 sessions; threshold = 95th percentile of the max over latents under a within-group session shuffle), per
     outcome;
  2. leave-one-session-out re-test of the primary picks.
Group and session are fully confounded in frogs v1 (no session holds two groups): these checks say which picks also
vary between sessions of one group, or rest on one session; they cannot separate group from session.

Read-only on the NES outputs (as scripts/eci/period_check.py): the per-video caches (<nes>/<sae>/_cache/) and the
primary picks of <nes>/<sae>/summary.csv (outcome 'mean') and <nes>/<sae>/<P>pool_bouts/summary.csv (outcome
'bout_rate', q 0.95, gap 0).

Writes <nes>/<sae>/session_check/:
  session_flags.json  per outcome: threshold, null max quantiles, per-latent score / group / p_perm / flagged (lists
                      of length m, index = latent id), and the primary picks with their flag
  loso.csv            one row per primary pick: unadjusted tau / p / threshold and the leave-one-session-out p_max,
                      worst session, same_sign
  SUMMARY.md

Usage: python scripts/eci/session_check.py --sae matryoshka_btk_1024_k16_frogsfg_s0 --pooling max
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
sys.path.insert(0, str(ROOT / 'scripts/eci'))

from eci import session as SC  # noqa: E402
from eci.domain import get_domain  # noqa: E402
from period_check import BOUT, FPS, load_cache, primary_picks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='frogs')
    ap.add_argument('--sae', required=True)
    ap.add_argument('--pooling', required=True, choices=('max', 'mean'), help='primary per-frame pooling of the SAE')
    ap.add_argument('--n-shuffles', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-root', default=None)
    args = ap.parse_args()
    t_start = time.time()
    D = get_domain(args.domain)
    nes = Path(args.out_root or D.nes_root) / args.sae
    out = nes / 'session_check'
    out.mkdir(parents=True, exist_ok=True)
    design = D.load_design()
    if not np.array_equal(design['obs_row'].values, np.arange(len(design))):
        raise SystemExit('design obs_row is not 0..n-1')
    sfx = '' if args.pooling == 'max' else f'_{args.pooling}'
    vs = load_cache(nes / '_cache' / f'video_summaries_{args.pooling}.npz', design)
    bs = load_cache(nes / '_cache' / f'bout_summaries_max{sfx}.npz', design)
    q, g = BOUT
    values = {'mean': vs['full__mean'].astype(np.float64),
              'bout_rate': bs[f'full__{q}__{g}__count'] / (bs['full__n_frames'][:, None] / FPS / 60.0)}
    tidies = {'mean': pd.read_csv(nes / 'summary.csv'),
              'bout_rate': pd.read_csv(nes / f'{args.pooling}pool_bouts' / 'summary.csv')}
    analyses = {a.id: a for a in D.analyses}

    flags = {'sae': args.sae, 'pooling': args.pooling, 'definition': SC.__doc__.split('2. Leave-one')[0].strip(),
             'unit': 'frog (one video): per-video outcome, full window', 'n_shuffles': args.n_shuffles,
             'seed': args.seed, 'bout_rule': {'q': q, 'merge_gap': g}, 'outcomes': {}}
    loso_rows, lines = [], [f'# Recording-session checks: {args.sae} (codes_{args.pooling})', '',
                            'Group and session are fully confounded (no session holds two groups). The flag measures '
                            'session dependence WITHIN a group; it cannot separate group from session.', '']
    for oc, Y in values.items():
        s = SC.session_scores(design, Y, n_shuffles=args.n_shuffles, seed=args.seed)
        flagged = s['score'] > s['threshold']
        picks = primary_picks(tidies[oc], args.pooling, oc == 'bout_rate')
        disc = []
        for r in picks.sort_values(['analysis_id', 'prefix', 'round']).to_dict('records'):
            j = int(r['neuron'])
            disc.append({'analysis_id': r['analysis_id'], 'prefix': int(r['prefix']), 'round': int(r['round']),
                         'neuron': j, 'direction': r['direction'], 'tau': float(r['tau']), 'p': float(r['p']),
                         'score': round(float(s['score'][j]), 4), 'group': s['groups'][int(s['group'][j])],
                         'p_perm': round(float(s['p_perm'][j]), 4), 'flagged': bool(flagged[j])})
        nq = np.quantile(s['null_max'], [0.5, 0.9, 0.95, 0.99])
        flags['outcomes'][oc] = {
            'threshold': round(s['threshold'], 4), 'pointwise_threshold': round(s['pointwise_threshold'], 4),
            'null_max_quantiles': {'0.5': nq[0], '0.9': nq[1], '0.95': nq[2], '0.99': nq[3]},
            'groups_scored': s['groups'], 'frogs_per_session': s['n_frogs'], 'n_latents': int(len(s['score'])),
            'n_flagged': int(flagged.sum()),
            'latents': {'score': np.round(s['score'], 4).tolist(), 'group': s['group'].astype(int).tolist(),
                        'p_perm': np.round(s['p_perm'], 4).tolist(), 'flagged': flagged.tolist()},
            'discovered': disc}
        lines += [f'## Outcome {oc}', '',
                  f'Threshold (95th pct of the max within-group session eta^2 over {len(s["score"])} latents, '
                  f'{args.n_shuffles} within-group session shuffles): {s["threshold"]:.3f} (null max median {nq[0]:.3f}, '
                  f'99th pct {nq[3]:.3f}; one latent alone, no multiplicity control: {s["pointwise_threshold"]:.3f}). '
                  f'Groups scored: {s["groups"]}; frogs per session: {s["n_frogs"]}. Flagged latents: '
                  f'{int(flagged.sum())}/{len(s["score"])}. Primary picks flagged: {sum(d["flagged"] for d in disc)}/{len(disc)}.',
                  '', '| analysis | prefix | round | neuron | direction | score | group | p_perm | flagged |',
                  '|---|---|---|---|---|---|---|---|---|']
        lines += [f'| {d["analysis_id"]} | {d["prefix"]} | {d["round"]} | {d["neuron"]} | {d["direction"]} | '
                  f'{d["score"]:.3f} | {d["group"]} | {d["p_perm"]:.3g} | {"FLAG" if d["flagged"] else ""} |' for d in disc]
        lines.append('')
        if len(picks):
            lo = SC.leave_one_session_out(picks, design, Y, analyses)
            lo.insert(0, 'outcome', oc)
            loso_rows.append(lo)
        print(f'{oc}: threshold {s["threshold"]:.4f}, flagged {int(flagged.sum())}, picks flagged '
              f'{sum(d["flagged"] for d in disc)}/{len(disc)}', flush=True)

    with open(out / 'session_flags.json', 'w') as f:
        json.dump(flags, f, default=float)
    cols = ['outcome', 'analysis_id', 'prefix', 'round', 'neuron', 'direction', 'tau', 'p', 'threshold', 'n_sessions',
            'n_not_estimable', 'worst_session', 'p_max', 'tau_min_abs', 'same_sign', 'p_max_below']
    lo = pd.concat(loso_rows)[cols] if loso_rows else pd.DataFrame(columns=cols)
    lo.to_csv(out / 'loso.csv', index=False)
    lines += ['## Leave-one-session-out re-test (primary picks)', '',
              'Each pick re-tested once per session of the analysis with that session dropped (S = earlier picks). '
              'p_max = the largest p over the drops; same sign = tau keeps its sign in every drop.', '',
              '| outcome | analysis | prefix | round | neuron | tau | p | threshold | worst session | p_max | same sign |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    lines += [f'| {r.outcome} | {r.analysis_id} | {int(r.prefix)} | {int(r.round)} | {int(r.neuron)} | {r.tau:.4g} | '
              f'{r.p:.2e} | {r.threshold:.2e} | {r.worst_session} | {r.p_max:.2e} | {"Y" if r.same_sign else "N"} |'
              for r in lo.itertuples()]
    lines += ['', f'Runtime {time.time() - t_start:.0f}s.']
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print(f'leave-one-session-out: {int(lo["same_sign"].sum())}/{len(lo)} keep their sign, '
          f'{int(lo["p_max_below"].sum())} stay below threshold; done in {time.time() - t_start:.0f}s -> {out}', flush=True)


if __name__ == '__main__':
    main()
