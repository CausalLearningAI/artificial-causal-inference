"""
v2.2 FULL WINDOW summary (AMENDMENT 2): focus table (full window vs the old 2-min windows side by side), all-arm tables,
C1-C3 verdicts on both frame sets, per-video time stats (mean over held-out videos +- SE) and, for the selected
nose-nose neuron, AUROC against nose-nose-mutual (nn) and nose-nose-passive (np) separately.
Inputs: results/vision/eci_v2_2/full/{report.json, tags/} (V22_FRAMES=full) and full/fullsub/{report.json, tags/}
(= the old 2-min numbers, identical to results/vision/eci_v2_2/report.json; identity_pipeline.json).
CPU job (mmap reads of the cached readouts). -> results/vision/eci_v2_2/full/summary_full.md, summary_full.json
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OF = REPO / 'results/vision/eci_v2_2/full'
DT = 0.2
NAME = {'nose_nose': 'nose-nose (nose_nose)', 'nt': 'nose-tail (nt)', 'nn': 'nose-nose-mutual (nn)', 'np': 'nose-nose-passive (np)'}
FOCUS = ['v1|auc', 'v1s0|auc', 'V|auc', 'V|bal', 'V|hyb', 'V@top10|hyb']
SETS = {'full': OF, 'old': OF / 'fullsub'}


def video_stats(tags, rule, dset, b):
    """per-video time stats (held-out videos; sgcf: averaged over the directions holding the video out), mean over
    videos +- SE per seed, then mean over seeds."""
    out = {}
    for k in ('true_s', 'bias_raw_s', 'mae_raw_s', 'bias_cal_s', 'mae_cal_s', 'bias_raw_frac', 'mae_raw_frac',
              'bias_cal_frac', 'mae_cal_frac', 'mae_const_s', 'mae_const_frac'):
        out[k] = []
    for tg in tags:
        acc = {}
        for d in tg['sets'][f'{dset}|{rule}']:
            h = d[b]
            V = h['video']
            S, T_, n = np.asarray(V['S']), np.asarray(V['T']), np.asarray(V['n'])
            c = h['c']
            row = {'true_s': T_ * DT, 'bias_raw_s': (S - T_) * DT, 'bias_cal_s': (c * S - T_) * DT,
                   'mae_const_s': np.abs(h['const_rate'] * n - T_) * DT}
            row['mae_raw_s'] = np.abs(row['bias_raw_s'])
            row['mae_cal_s'] = np.abs(row['bias_cal_s'])
            for kk in ('bias_raw', 'mae_raw', 'bias_cal', 'mae_cal', 'mae_const'):
                row[f'{kk}_frac'] = row[f'{kk}_s'] / (n * DT)
            for i, v in enumerate(V['vid']):
                for kk, arr in row.items():
                    acc.setdefault(v, {}).setdefault(kk, []).append(arr[i])
        for kk in out:
            vals = np.array([np.mean(acc[v][kk]) for v in acc])
            out[kk].append((vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals)), len(vals)))
    return {kk: {'mean': float(np.mean([x[0] for x in v])), 'se': float(np.mean([x[1] for x in v])), 'videos': v[0][2]}
            for kk, v in out.items()}


def nnnp_all(fs, R):
    """AUROC of the design (a) nose-nose pick vs nn and vs np labels separately (held-out non-SG frames), focus arms.
    One sequential read of each model's cached readout."""
    os.environ['V22_FRAMES'] = 'full' if fs == 'full' else 'fullsub'
    sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
    import importlib
    import v22_core as C
    C = importlib.reload(C)
    F, _ = C.mice_table()
    ev = np.flatnonzero(F.nonsg.values)
    need = {}
    for a in FOCUS:
        if a not in R[fs]['sets']['sg2non']['levels']:
            continue
        spec = R[fs]['arms'][a]
        for k in spec['models']:
            tg = pickle.load(open(SETS[fs] / 'tags' / f'{k}.pkl', 'rb'))
            need.setdefault(k, set()).add(int(tg['sets'][f'sg2non|{spec["rule"]}'][0]['picks']['nose_nose']))
    cols = {}
    for k, js in need.items():
        pr, pp = C.cache_paths(k)
        X = np.load(pr if k.startswith('v1_') else pp, mmap_mode='r')
        js = sorted(js)
        out = np.empty((len(X), len(js)), np.float32)
        for r0 in range(0, len(X), 20000):
            out[r0:r0 + 20000] = np.asarray(X[r0:r0 + 20000])[:, js]
        for i, j in enumerate(js):
            cols[(k, j)] = out[:, i]
    res = {}
    for a in FOCUS:
        if a not in R[fs]['sets']['sg2non']['levels']:
            continue
        spec = R[fs]['arms'][a]
        v = {'nn': [], 'np': []}
        for k in spec['models']:
            tg = pickle.load(open(SETS[fs] / 'tags' / f'{k}.pkl', 'rb'))
            j = int(tg['sets'][f'sg2non|{spec["rule"]}'][0]['picks']['nose_nose'])
            for lb in ('nn', 'np'):
                v[lb].append(float(C.auroc_cols(cols[(k, j)][:, None], ev, F[f'y_{lb}'].values[ev])[0]))
        res[a] = {lb: float(np.mean(x)) for lb, x in v.items()}
    return res


def main():
    R = {fs: json.loads((d / 'report.json').read_text()) for fs, d in SETS.items()}
    out = {'verdict': {fs: R[fs].get('verdict_C1_C3') for fs in SETS}, 'video': {}, 'nnnp': {}, 'gt': {fs: R[fs]['ground_truth'] for fs in SETS}}
    arms = [a for a in R['full']['arms'] if all(m for m in R['full']['arms'][a]['models'])]
    have = {fs: set(R[fs]['sets']['sg2non']['levels']) for fs in SETS}
    for fs in SETS:
        for dset in ('sg2non', 'sgcf'):
            for a in have[fs]:
                spec = R[fs]['arms'][a]
                tags = [pickle.load(open(SETS[fs] / 'tags' / f'{k}.pkl', 'rb')) for k in spec['models']]
                for b in ('nose_nose', 'nt'):
                    out['video'][f'{fs}|{dset}|{a}|{b}'] = video_stats(tags, spec['rule'], dset, b)
        for a, v in nnnp_all(fs, R).items():
            out['nnnp'][f'{fs}|{a}'] = v
    (OF / 'summary_full.json').write_text(json.dumps(out, indent=1))

    def lv(fs, dset, a, key, d=3):
        L = R[fs]['sets'][dset].get('level_ci', {}).get(a, {}).get(key)
        if L is None:
            return '-'
        return f"{L['mean']:.{d}f} [{L['ci'][0]:.{d}f}, {L['ci'][1]:.{d}f}]"

    def both(dset, a, key, d=3):
        o = R['old']['sets'][dset].get('level_ci', {}).get(a, {}).get(key)
        return f"{lv('full', dset, a, key, d)} (2-min: {o['mean']:.{d}f})" if o else lv('full', dset, a, key, d)

    def vs(fs, dset, a, b, k, d=2):
        v = out['video'].get(f'{fs}|{dset}|{a}|{b}', {}).get(k)
        return f"{v['mean']:.{d}f} ± {v['se']:.{d}f}" if v else '-'

    md = ['# v2.2 mice re-scored on the FULL 15-min atlas window (AMENDMENT 2)', '',
          'Numbers: mean over seeds [95% CI, bootstrap over held-out pools and seeds, 1000 replicates]; "(2-min: x)" = the same '
          'quantity on the old 2-min windows (identical to results/vision/eci_v2_2/report.json). Per-video time: mean over held-out '
          'videos ± SE (mean over seeds). v1|auc = seeds s0-s10; v1s0|auc = seed 0 only (full window: the stored deployed codes).', '']
    for dset, title in (('sg2non', 'design (a): select on the 11 SG pools, evaluate on the 12 non-SG pools'),
                        ('sgcf', 'design (b): cross-fitted within the 11 SG pools (12 directions)')):
        part = 'nonsg' if dset == 'sg2non' else 'sg'
        md += [f'## {title}', '']
        for b in ('nose_nose', 'nt'):
            g, go = R['full']['ground_truth'][part][b], R['old']['ground_truth'][part][b]
            md += [f"### {NAME[b]}", '',
                   f"Held-out prevalence {g['rate']:.4f} of frames, {g['frames_on']:,} frames, {g['bouts']:,} bouts, "
                   f"{g['videos_with_any']} videos with >= 1 positive frame; per-video rate {g['per_video_mean']:.4f} ± {g['per_video_se']:.4f} "
                   f"(2-min: {go['rate']:.4f}, {go['frames_on']} frames, {go['bouts']} bouts; per video {go['per_video_mean']:.4f} ± {go['per_video_se']:.4f}). "
                   f"True time per video: {vs('full', dset, 'V|auc', b, 'true_s', 1)} s of 900 s (2-min: {vs('old', dset, 'V|auc', b, 'true_s', 2)} s of 120 s). "
                   f"Constant-rate reference, per-video MAE: {vs('full', dset, 'V|auc', b, 'mae_const_s', 1)} s = "
                   f"{vs('full', dset, 'V|auc', b, 'mae_const_frac', 4)} of the window (2-min: {vs('old', dset, 'V|auc', b, 'mae_const_frac', 4)}).", '',
                   '| arm | distinct picks | AUROC | precision@100 | F1 (P(on) > 0.5) | discrimination AUROC nt vs nose_nose | '
                   'raw time bias, s | raw time MAE, s (frac) | calibrated time bias, s | calibrated time MAE, s (frac) | Pearson r per-video time |',
                   '|---|---|---|---|---|---|---|---|---|---|---|']
            for a in R['full']['sets'][dset]['levels']:
                pk = R['full']['sets'][dset]['picks'][a]
                dist = sum(all(p['nose_nose'] != p['nt'] for p in v) for v in pk.values())
                md.append(f"| {a} | {dist}/{len(pk)} | {both(dset, a, b + '|auc')} | {both(dset, a, b + '|p100')} | {both(dset, a, b + '|F1')} | "
                          f"{both(dset, a, b + '|disc', 2)} | {vs('full', dset, a, b, 'bias_raw_s', 1)} | {vs('full', dset, a, b, 'mae_raw_s', 1)} "
                          f"({vs('full', dset, a, b, 'mae_raw_frac', 3)}) | {vs('full', dset, a, b, 'bias_cal_s', 1)} | {vs('full', dset, a, b, 'mae_cal_s', 1)} "
                          f"({vs('full', dset, a, b, 'mae_cal_frac', 4)}) | {both(dset, a, b + '|r_soft', 2)} |")
            md.append('')
        md += [f'### Contrasts, {title}', '', '| contrast | nose-nose AUROC | nose-tail AUROC | nose-nose precision@100 | nose-tail precision@100 |',
               '|---|---|---|---|---|']
        for cn, r in R['full']['sets'][dset]['contrasts'].items():
            o = R['old']['sets'][dset]['contrasts'].get(cn)
            cell = lambda k: f"{r[k]['diff']:+.3f} [{r[k]['ci'][0]:+.3f}, {r[k]['ci'][1]:+.3f}]" + (f" (2-min {o[k]['diff']:+.3f} [{o[k]['ci'][0]:+.3f}, {o[k]['ci'][1]:+.3f}])" if o else '')  # noqa: E731
            md.append(f"| {cn} | {cell('nose_nose|auc')} | {cell('nt|auc')} | {cell('nose_nose|p100')} | {cell('nt|p100')} |")
        md.append('')
    md += ['## Verdict C1-C3 (design (a)); C4 (ants) unchanged: ants already scored every labelled frame, not re-run', '',
           '| arm | frames | C1 distinct | C2 nt disc / nose_nose disc (median) | C3 lower CI nose-nose / nose-tail AUROC vs reference | C1 | C2 | C3 |',
           '|---|---|---|---|---|---|---|---|']
    for a in out['verdict']['full']:
        for fs in ('full', 'old'):
            v = (out['verdict'][fs] or {}).get(a)
            if not v:
                continue
            c3 = v['C3 lower CI (AUROC vs V|auc)']
            md.append(f"| {a} | {'full' if fs == 'full' else '2-min'} | {v['C1 distinct seeds']} | {v['C2 nt disc median']:.2f} / {v['C2 nose_nose disc median']:.2f} | "
                      f"{c3.get('nose_nose', float('nan')):+.3f} / {c3.get('nt', float('nan')):+.3f} (vs {v['C3 reference']}) | {'yes' if v['C1'] else 'NO'} | "
                      f"{'yes' if v['C2'] else 'NO'} | {'yes' if v['C3'] else 'NO'} |")
    md += ['', '## Selected nose-nose neuron vs nose-nose-mutual (nn) and nose-nose-passive (np) separately (design (a), held-out non-SG frames, AUROC, mean over seeds)', '',
           '| arm | AUROC vs nose-nose-mutual (nn) | AUROC vs nose-nose-passive (np) |', '|---|---|---|']
    for a in FOCUS:
        f_, o_ = out['nnnp'].get(f'full|{a}'), out['nnnp'].get(f'old|{a}')
        if f_:
            md.append(f"| {a} | {f_['nn']:.3f}" + (f" (2-min {o_['nn']:.3f})" if o_ else '') + f" | {f_['np']:.3f}" + (f" (2-min {o_['np']:.3f})" if o_ else '') + ' |')
    md += ['', '## Ground truth (full window / 2-min)', '', '| set | behaviour | frames | bouts | videos with >= 1 positive | rate | per-video rate mean ± SE |', '|---|---|---|---|---|---|---|']
    for part in ('sg', 'nonsg', 'all'):
        for b in ('nose_nose', 'nn', 'np', 'nt'):
            for fs in ('full', 'old'):
                g = R[fs]['ground_truth'][part][b]
                md.append(f"| {part} {'full' if fs == 'full' else '2-min'} | {NAME[b]} | {g['frames_on']:,} | {g['bouts']:,} | {g['videos_with_any']} | "
                          f"{g['rate']:.4f} | {g['per_video_mean']:.4f} ± {g['per_video_se']:.4f} |")
    (OF / 'summary_full.md').write_text('\n'.join(md) + '\n')
    print('summary written')


if __name__ == '__main__':
    main()
