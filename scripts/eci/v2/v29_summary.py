"""
ECI v2.9 summary: applies the pre-registered decisions (scripts/eci/v2/v29_prereg.md) literally to the per-dictionary
results of v29_sgap.py -> results/vision/eci_v2_9/{summary.json, tables.md}. Login-node safe (small JSON files only).
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/eci_v2_9'
O7 = REPO / 'results/vision/eci_v2_7'
K16 = [f'PREDk16_s{s}' for s in range(2901, 2907)]
K8 = [f'PREDk8_s{s}' for s in (2901, 2902, 2903)]
K32 = [f'PREDk32_s{s}' for s in (2901, 2902, 2903)]
OLD = [f'PREDk_s{s}' for s in (2701, 2702, 2703)]
REF = [f'V_s{s}' for s in range(1001, 1007)] + ['v1_s0']
NEW = K16 + K8 + K32
BN = {'nose_nose': 'nose-nose', 'nt': 'nose-tail'}


def f3(x):
    return 'n/a' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.3f}'


def load(dom, keys):
    D = {}
    for k in keys:
        p = OUT / dom / 'dicts' / f'{k}.json'
        if p.exists():
            D[k] = json.loads(p.read_text())
    return D


def neuron(d, rule, b):
    pk = d['picks'].get(rule)
    return None if pk is None else d['neurons'][f"{pk[b]}|{b}"]


def criteria(d):
    pk = d['picks'].get('sgap')
    if pk is None:
        return {'has_pair': False, 'accepted': False}
    nn, nt, pr = neuron(d, 'sgap', 'nose_nose'), neuron(d, 'sgap', 'nt'), d['pairs']['sgap']
    c = {'has_pair': True,
         'nn_fire<0.10': [nn['fire_raw_all'], nn['fire_raw_all'] < 0.10],
         'nn_AP_lowerCI>=0.10': [nn['ho_ci']['ap'][0], nn['ho_ci']['ap'][0] >= 0.10],
         'nn_top3>=0.35': [nn['ho']['top3'], nn['ho']['top3'] >= 0.35],
         'nn_AUROC>=0.75': [nn['ho']['auc'], nn['ho']['auc'] >= 0.75],
         'nt_distinct': [pk['nose_nose'] != pk['nt'], pk['nose_nose'] != pk['nt']],
         'nt_fire<0.10': [nt['fire_raw_all'], nt['fire_raw_all'] < 0.10],
         'nt_AP_lowerCI>=0.10': [nt['ho_ci']['ap'][0], nt['ho_ci']['ap'][0] >= 0.10],
         'nt_top3>=0.30': [nt['ho']['top3'], nt['ho']['top3'] >= 0.30],
         'top3_overlap<0.10': [pr['top3_overlap'], pr['top3_overlap'] < 0.10]}
    c['accepted'] = all(v[1] for kk, v in c.items() if kk != 'has_pair')
    c['n_pass'] = int(sum(v[1] for kk, v in c.items() if kk not in ('has_pair', 'accepted')))
    return c


def sg_sum(d):
    pr = d['pairs'].get('sgap')
    return pr['sg_ap_sum'] if pr else 0.0


def ants_ok(a):
    if a is None:
        return None
    return all(a['dirs'][dn]['groom_any|auc'] >= 0.85 and a['dirs'][dn]['groom_any|top3'] >= 0.75 for dn in ('A2B', 'B2A'))


def train_health():
    rows = []
    for dom in ('mice', 'ants'):
        for k in NEW + OLD:
            p = (OUT / 'train' / f'{dom}_{k}.json') if k in NEW else (O7 / 'train' / f'{dom}_{k}.json')
            if not p.exists():
                rows.append({'dom': dom, 'key': k, 'missing': True})
                continue
            r = json.loads(p.read_text())
            rows.append({'dom': dom, 'key': k, 'k': r['cfg']['k'], 'eval_fve': r['eval_fve'], 'eval_l0': r['eval_l0'], 'theta_star': r['theta_star'],
                         'dead_train': r['dead_at_end_train'], 'never_fire_eval_frac': r['never_fire_on_eval_frac'], 'steps': r['steps_run'],
                         'passes': r['passes'], 'train_time_s': r['train_time_s'], 'node': r['node'], 'gpu_util_median': r['gpu_util_median']})
    return rows


def main():
    M = load('mice', NEW + OLD + REF)
    Aa = load('ants', NEW + OLD + [f'V_s{s}' for s in range(1001, 1007)])
    crit = {k: criteria(d) for k, d in M.items()}
    res = {'mice_scored': sorted(M), 'ants_scored': sorted(Aa), 'criteria': crit, 'sg_ap_sum': {k: sg_sum(d) for k, d in M.items()},
           'ants_accepted': {k: ants_ok(a) for k, a in Aa.items()}, 'train': train_health()}
    acc = lambda ks: [k for k in ks if k in crit and crit[k]['accepted']]  # noqa: E731
    # Q1
    res['Q1'] = {'fresh_K16_accepted': acc(K16), 'fresh_K16_scored': [k for k in K16 if k in M], 'old_PREDk_accepted': acc(OLD),
                 'old_PREDk_scored': [k for k in OLD if k in M]}
    # Q2
    arms = {'K8': K8, 'K16': K16[:3], 'K32': K32}
    q2 = {a: {'accepted_frac': len(acc(ks)) / len(ks), 'accepted': acc(ks), 'median_sg_ap_sum': float(np.median([sg_sum(M[k]) for k in ks if k in M])) if any(k in M for k in ks) else None,
              'complete': all(k in M for k in ks)} for a, ks in arms.items()}
    qual = [a for a in ('K8', 'K32') if q2[a]['complete'] and q2['K16']['complete'] and q2[a]['accepted_frac'] > q2['K16']['accepted_frac']
            and q2[a]['median_sg_ap_sum'] > q2['K16']['median_sg_ap_sum']]
    win = 'K16' if not qual else sorted(qual, key=lambda a: (-q2[a]['accepted_frac'], -q2[a]['median_sg_ap_sum']))[0]
    res['Q2'] = {'per_arm_seeds_2901_2903': q2, 'qualifying': qual, 'winning_k': {'K8': 8, 'K16': 16, 'K32': 32}[win], 'winning_arm': win}
    # Q3
    pool = (K16 + OLD) if win == 'K16' else arms[win]
    pool = [k for k in pool if k in M]
    ok = [k for k in pool if crit[k]['accepted']]
    best = max(ok, key=lambda k: sg_sum(M[k])) if ok else None
    res['Q3'] = {'pool': pool, 'accepted_in_pool': ok, 'deliverable_mice': best,
                 'best_sg_ap_sum_in_pool': max(pool, key=lambda k: sg_sum(M[k])) if pool else None}
    if best:
        res['Q3']['picks'] = M[best]['picks']['sgap']
    # ants deliverable
    if best and best in Aa and ants_ok(Aa[best]):
        ad, why = best, 'same arm and seed meets the ants criteria'
    else:
        apool = [k for k in pool if k in Aa]
        ad = max(apool, key=lambda k: Aa[k]['dirs']['A2B']['sel_auc_any']) if apool else None
        why = 'fallback: best selection-half (A) grooming-to-focal (any) AUROC of the A -> B pick among the winning-k dictionaries'
    res['Q3']['deliverable_ants'] = ad
    res['Q3']['deliverable_ants_reason'] = why
    res['Q3']['deliverable_ants_accepted'] = ants_ok(Aa[ad]) if ad else None
    (OUT / 'summary.json').write_text(json.dumps(res, indent=1, default=float))
    tables(res, M, Aa)


def tables(res, M, Aa):
    md = ['# ECI v2.9 tables (pre-registration scripts/eci/v2/v29_prereg.md; code v29_train.py, v29_eval.py, v29_sgap.py, v29_summary.py)', '',
          'Mice: 863,424 edge-excluded frames scored; selection (SG) = 11 annotator-SG pools, 431,712 frames; held-out (HO) = 12 non-SG pools, '
          '72 videos, 431,712 frames. AUROC / AP on the P(on) readout (AP raw-code tie-break); [CI] = 95% video bootstrap (1,000); '
          'precision@top-k/video = raw-code ranking over videos with >= 1 positive; chance = mean per-video prevalence over those videos; '
          'firing = fraction of all 863,424 frames with raw code > 0; level 1-5 = latents 1-128 / 129-256 / 257-512 / 513-2048 / 2049-4096. '
          'K8 / K16 / K32 = PRED-contact with BatchTopK k = 8 / 16 / 32; PREDk = the v2.7 PRED-contact (k = 16). V and v1 s0 = references (existing files).', '']
    if M:
        g = next(iter(M.values()))['ground_truth']
        md += ['Held-out ground truth: ' + '; '.join(f"{BN[b]} {v['positive_frames']:,} positive frames of {v['heldout_frames']:,} (prevalence {v['prevalence']:.4f}), "
                                                    f"{v['videos_with_positive']} of {v['heldout_videos']} videos with >= 1 positive" for b, v in g.items()), '']
    md += ['## 1. Training health', '', '| domain | dictionary | k | eval FVE | L0 per token (eval) | dead latents (train) | never fire on eval tokens | theta* |',
           '|---|---|---|---|---|---|---|---|']
    for r in res['train']:
        if r.get('missing'):
            md.append(f"| {r['dom']} | {r['key']} | missing | | | | | |")
            continue
        md.append(f"| {r['dom']} | {r['key']} | {r['k']} | {r['eval_fve']:.4f} | {r['eval_l0']:.2f} | {r['dead_train']} / 4096 | {r['never_fire_eval_frac']:.4f} | {r['theta_star']:.4f} |")
    for rule in ('sgap', 'dist', 'auc'):
        md += ['', f'## 2{"abc"[("sgap", "dist", "auc").index(rule)]}. Mice per dictionary, rule {rule}' + (' (pre-registered)' if rule == 'sgap' else ' (official v2.7 rule, continuity)'), '',
               '| dictionary | behaviour | neuron (level) | firing | SG AUROC / AP / precision@top3/video | HO AUROC [CI] | HO AP [CI] | HO precision@top3/video [CI] (chance) | HO precision@top10/video [CI] | per-video time bias / MAE, s (mean +- SE) |',
               '|---|---|---|---|---|---|---|---|---|---|']
        for k in NEW + OLD + REF:
            if k not in M:
                continue
            d = M[k]
            if d['picks'].get(rule) is None:
                md.append(f'| {k} | - | no {rule} pair | | | | | | | |')
                continue
            for b in ('nose_nose', 'nt'):
                n = neuron(d, rule, b)
                h, c = n['ho'], n['ho_ci']
                bias = f"{n['video_bias_s'][0]:+.1f} +- {n['video_bias_s'][1]:.1f} / {n['video_mae_s'][0]:.1f} +- {n['video_mae_s'][1]:.1f}" if 'video_bias_s' in n else 'n/a'
                md.append(f"| {k} | {BN[b]} | #{n['neuron']} (L{n['level'] if n['level'] else '-'}) | {n['fire_raw_all']:.4f} | {f3(n['sg|auc'])} / {f3(n['sg|ap'])} / {f3(n['sg|top3'])} | "
                          f"{f3(h['auc'])} [{f3(c['auc'][0])}, {f3(c['auc'][1])}] | {f3(h['ap'])} [{f3(c['ap'][0])}, {f3(c['ap'][1])}] | "
                          f"{f3(h['top3'])} [{f3(c['top3'][0])}, {f3(c['top3'][1])}] ({n['ho_chance_topk']:.4f}) | {f3(h['top10'])} [{f3(c['top10'][0])}, {f3(c['top10'][1])}] | {bias} |")
    md += ['', '## 3. Acceptance criteria (sgap pair; measured value, PASS / fail)', '',
           '| dictionary | SG AP sum | nose-nose firing < 0.10 | nose-nose AP lower CI >= 0.10 | nose-nose precision@top3/video >= 0.35 | nose-nose AUROC >= 0.75 | '
           'nose-tail distinct | nose-tail firing < 0.10 | nose-tail AP lower CI >= 0.10 | nose-tail precision@top3/video >= 0.30 | top-3 overlap < 0.10 | ACCEPTED |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    keys = ['nn_fire<0.10', 'nn_AP_lowerCI>=0.10', 'nn_top3>=0.35', 'nn_AUROC>=0.75', 'nt_distinct', 'nt_fire<0.10', 'nt_AP_lowerCI>=0.10', 'nt_top3>=0.30', 'top3_overlap<0.10']
    for k in NEW + OLD + REF:
        if k not in M:
            continue
        c = res['criteria'][k]
        if not c['has_pair']:
            md.append(f'| {k} | 0 | no sgap pair | | | | | | | | | no |')
            continue
        cells = []
        for kk in keys:
            v, ok = c[kk]
            cells.append(f"{v if isinstance(v, bool) else f'{v:.3f}'} {'PASS' if ok else 'fail'}")
        md.append(f"| {k} | {res['sg_ap_sum'][k]:.3f} | " + ' | '.join(cells) + f" | {'YES' if c['accepted'] else 'no'} |")
    q1, q2, q3 = res['Q1'], res['Q2'], res['Q3']
    md += ['', '## 4. Decisions (pre-registered, applied literally)', '',
           f"- Q1 confirmation: {len(q1['fresh_K16_accepted'])} of {len(q1['fresh_K16_scored'])} fresh K16 dictionaries accepted {q1['fresh_K16_accepted']}; "
           f"old v2.7 PREDk (same v2.9 code): {len(q1['old_PREDk_accepted'])} of {len(q1['old_PREDk_scored'])} {q1['old_PREDk_accepted']}.",
           '- Q2 k (seeds 2901-2903): ' + '; '.join(f"{a}: accepted {v['accepted_frac']:.2f} ({len(v['accepted'])}/3), median SG AP sum {f3(v['median_sg_ap_sum'])}"
                                                    for a, v in q2['per_arm_seeds_2901_2903'].items()) +
           f". Qualifying over K16: {q2['qualifying'] or 'none'} -> k = {q2['winning_k']}.",
           f"- Q3 mice deliverable candidate: {q3['deliverable_mice'] or 'NONE (no dictionary of the winning k meets the numeric criteria)'}"
           + (f" (sgap nose-nose #{q3['picks']['nose_nose']}, nose-tail #{q3['picks']['nt']})" if q3.get('picks') else '')
           + f"; pool {q3['pool']}; accepted in pool {q3['accepted_in_pool']}; max SG AP sum in pool {q3['best_sg_ap_sum_in_pool']}.",
           f"- Ants deliverable: {q3['deliverable_ants']} ({q3['deliverable_ants_reason']}); meets the ants criteria: {q3['deliverable_ants_accepted']}."]
    md += ['', '## 5. Ants grooming-to-focal per dictionary (pick = best selection-half AUROC of any; v3 + v4, 1 fps)', '',
           '| dictionary | direction | neuron (level) | sel AUROC any | HO AUROC any [CI] | HO AUROC single | HO AUROC double | HO precision@top3/video any [CI] | single | double | HO AP any | ants criteria |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k in NEW + OLD + [f'V_s{s}' for s in range(1001, 1007)]:
        if k not in Aa:
            continue
        for dn, r in Aa[k]['dirs'].items():
            md.append(f"| {k} | {dn} | #{r['pick']} (L{r['level']}) | {r['sel_auc_any']:.3f} | {r['groom_any|auc']:.3f} [{r['groom_any|auc|ci'][0]:.3f}, {r['groom_any|auc|ci'][1]:.3f}] | "
                      f"{r['single|auc']:.3f} | {r['double|auc']:.3f} | {r['groom_any|top3']:.3f} [{r['groom_any|top3|ci'][0]:.3f}, {r['groom_any|top3|ci'][1]:.3f}] | "
                      f"{r['single|top3']:.3f} | {r['double|top3']:.3f} | {r['groom_any|ap']:.3f} | {'PASS' if res['ants_accepted'][k] else 'fail'} |")
    (OUT / 'tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
