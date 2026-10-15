"""
CRIM13 alignment test - scoring of the PRE-REGISTERED rules of eci_align.sh (login safe: reads the eval / probe /
encode JSONs, the token-store frame tables and the GPU logs; small files only).
    pair primary AUROC per SAE seed = mean over the 7 pair behaviours of (test AUROC of the AUROC-picked neuron, averaged
    over folds A and B); Welch 95% CI on 6 vs 6 seeds; WIN delta >= +0.010 and CI excludes 0, LOSES delta <= -0.010 and
    CI excludes 0, else EQUIVALENT. Contrasts C2 vs C1, C3 vs C1 (C4 vs C1 exploratory, no verdict).
Outputs: OUT/res/res.md, OUT/res/res.json
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_common import OUT, PAIR, PLAN, RESIDENTS, SCORED, TOK  # noqa: E402

ARMS = {'C1': 'C1 D3L-640 L20, Delta 2 s (reference)', 'C2': 'C2 D3L-640 L20, no Delta',
        'C3': 'C3 D2L-560 L20, Delta 2 s', 'C4': 'C4 D3L, Delta, NO mask (EXPLORATORY)'}
RD = OUT / 'res'


def load(arm):
    R = {}
    for f in sorted((OUT / 'eval').glob(f'{arm}_?_s*.json')):
        m = re.fullmatch(rf'{arm}_([AB])_s(\d+)\.json', f.name)
        if m:
            R.setdefault(int(m.group(2)), {})[m.group(1)] = json.loads(f.read_text())
    return {s: v for s, v in R.items() if len(v) == 2}


def fm(R, f):
    """mean over seeds x folds of f(fold result) (nan-aware)."""
    return float(np.nanmean([f(r) for v in R.values() for r in v.values()]))


def per_seed(R, labs, f=lambda r, b: r['main'][b]['auc']['test_auc']):
    return {s: float(np.nanmean([np.mean([f(v[fo], b) for fo in 'AB']) for b in labs])) for s, v in sorted(R.items())}


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean()
    h = stats.t.ppf(0.975, df) * se
    return float(d), float(d - h), float(d + h), float(df)


def verdict(d, lo, hi):
    if d >= 0.010 and lo > 0:
        return 'WIN'
    if d <= -0.010 and hi < 0:
        return 'LOSES'
    return 'EQUIVALENT'


def mask_recall():
    Fs = [pd.read_parquet(j) for j in sorted(TOK.glob('*of*_frames.parquet')) if 'smoke' not in j.name]
    if not Fs:
        return None, []
    F = pd.concat(Fs).merge(pd.read_parquet(PLAN / 'videos.parquet')[['video', 'strain', 'resident']], on='video')
    out = {'frames': len(F), 'kept_cells_mean': float(F.n40.mean()), 'kept_frac_of_video_cells': float(F.n40.mean() / 1200),
           'frames_0_kept': int((F.n40 == 0).sum())}
    rows = []
    for nm, sub in (('resident', F[F.has_res]), ('intruder (all)', F[F.has_int]),
                    ('intruder BC (pale)', F[F.has_int & (F.strain == 'BC')]),
                    ('intruder C57 (dark)', F[F.has_int & (F.strain == 'C57')]),
                    ('resident, BC-intruder videos', F[F.has_res & (F.strain == 'BC')]),
                    ('resident, C57-intruder videos', F[F.has_res & (F.strain == 'C57')])):
        k = 'res' if nm.startswith('resident') else 'int'
        pv = sub.groupby('video')[f'rec_{k}'].mean()
        r = {'animal': nm, 'frames': len(sub), 'recall_own_cell': float(sub[f'rec_{k}'].mean()),
             'recall_3x3': float(sub[f'rec3_{k}'].mean()), 'videos': int(pv.size),
             'video_recall_p10': float(pv.quantile(0.1)) if len(pv) else float('nan')}
        out[nm] = r
        rows.append(r)
    return out, rows


def gpu_util():
    rows = []
    for f in sorted((OUT / 'logs').glob('gpu_*.csv')):
        m = re.fullmatch(r'gpu_(.+)_(\d+)\.csv', f.name)
        if not m:
            continue
        u = []
        with open(f) as fh:
            for r in csv.reader(fh):
                try:
                    u.append(float(r[1].strip().split()[0]))
                except (ValueError, IndexError):
                    pass
        el = subprocess.run(['sacct', '-j', m.group(2), '-X', '-n', '-o', 'ElapsedRaw,State,NodeList'],
                            capture_output=True, text=True).stdout.split()
        rows.append({'job': m.group(1), 'id': int(m.group(2)), 'samples': len(u),
                     'util_mean': float(np.mean(u)) if u else float('nan'),
                     'gpu_h': int(el[0]) / 3600 if el else 0.0, 'state': el[1] if len(el) > 1 else '',
                     'node': el[2] if len(el) > 2 else ''})
    return rows


def main():
    D = {a: load(a) for a in ARMS}
    D = {a: v for a, v in D.items() if v}
    out = {'arms': {}, 'contrasts': {}}
    L = ['# CRIM13: single-neuron alignment with pair behaviours (pre-registered, eci_align.sh)', '',
         'pair primary AUROC = mean over ' + ', '.join(PAIR) + ' of the test AUROC of the AUROC-picked neuron, folds A '
         '(SAE + pick on the released train half, scored on test) and B (swapped) averaged; one value per SAE seed '
         '500-505. all-behaviour mean AUROC = the same over the 11 scored behaviours.', '',
         '| arm | seeds | per-seed pair primary AUROC | pair primary AUROC mean ± SE | all-behaviour mean AUROC |',
         '|---|---|---|---|---|']
    S = {}
    for a, R in D.items():
        S[a] = per_seed(R, PAIR)
        allm = per_seed(R, SCORED)
        v = np.array(list(S[a].values()))
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')
        out['arms'][a] = {'seeds': list(S[a]), 'pair_primary_auroc': list(S[a].values()), 'mean': float(v.mean()),
                          'se': float(se), 'all_behaviour_auroc': list(allm.values()),
                          'all_behaviour_mean': float(np.mean(list(allm.values())))}
        L.append(f"| {ARMS[a]} | {len(v)} | {', '.join(f'{x:.3f}' for x in v)} | {v.mean():.4f} ± {se:.4f} | "
                 f"{np.mean(list(allm.values())):.4f} |")
    L += ['', '| contrast | delta pair primary AUROC [Welch 95% CI] | df | verdict |', '|---|---|---|---|']
    for k, (a, b) in {'C2 vs C1 (no Delta vs Delta)': ('C2', 'C1'), 'C3 vs C1 (DINOv2-L vs DINOv3-L)': ('C3', 'C1'),
                      'C4 vs C1 (no mask vs mask; EXPLORATORY, no verdict)': ('C4', 'C1')}.items():
        if a in S and b in S and len(S[a]) > 1 and len(S[b]) > 1:
            d, lo, hi, df = welch(list(S[a].values()), list(S[b].values()))
            vd = verdict(d, lo, hi) if a != 'C4' else 'descriptive'
            out['contrasts'][k] = {'delta': d, 'ci': [lo, hi], 'df': df, 'verdict': vd}
            L.append(f'| {k} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {df:.1f} | {vd} |')
        else:
            L.append(f'| {k} | missing arm | | |')
    # per behaviour
    L += ['', '## per behaviour (mean over 6 seeds x 2 folds; scoring half)', '',
          'AUROC (AUROC pick) / AP (AP pick) / AP lift = AP / base rate / honest top-1% (top-1% pick among neurons firing '
          'on >= 1%) / top-1% lift / firing rate of the AUROC pick / weighted base rate', '']
    for a, R in D.items():
        L += [f'### {ARMS[a]}', '', '| behaviour | AUROC | AP | AP lift | honest top-1% | top-1% lift | firing rate (AUROC pick) | base rate |',
              '|---|---|---|---|---|---|---|---|']
        pb = {}
        for b in SCORED:
            g = lambda f: fm(R, lambda r: f(r['main'][b]))  # noqa: E731
            base = g(lambda e: e['base_test'])
            ap_ = g(lambda e: e['ap']['test_ap'])
            t1 = g(lambda e: e['top1']['test_top1'])
            q = {'auroc': g(lambda e: e['auc']['test_auc']), 'ap': ap_,
                 'ap_lift': g(lambda e: e['ap']['test_ap'] / e['base_test']), 'top1': t1,
                 'top1_lift': g(lambda e: e['top1']['test_top1'] / e['base_test']),
                 'rate_auc_pick': g(lambda e: e['auc']['test_rate']), 'base': base,
                 'rprec_at_auc': g(lambda e: e['auc']['test_rprec']), 'rprec_picked': g(lambda e: e['rprec']['test_rprec']),
                 'f1_at_auc': g(lambda e: e['auc']['test_f1']), 'f1_picked': g(lambda e: e['f1']['test_f1'])}
            pb[b] = q
            L.append(f"| {b} | {q['auroc']:.3f} | {q['ap']:.3f} | {q['ap_lift']:.2f} | {q['top1']:.3f} | "
                     f"{q['top1_lift']:.2f} | {q['rate_auc_pick']:.3f} | {q['base']:.4f} |")
        out['arms'][a]['per_behaviour'] = pb
        L += ['', f'secondary (addendum 2), {a}: R-precision and F1-max, (a) of the AUROC-picked neuron, (b) of the neuron '
              'picked by the same metric on the selection half; base rate', '',
              '| behaviour | R-precision (a) AUROC pick | R-precision (b) R-prec pick | F1-max (a) AUROC pick | F1-max (b) F1 pick | base rate |',
              '|---|---|---|---|---|---|']
        for b in SCORED:
            q = pb[b]
            L.append(f"| {b} | {q['rprec_at_auc']:.3f} | {q['rprec_picked']:.3f} | {q['f1_at_auc']:.3f} | "
                     f"{q['f1_picked']:.3f} | {q['base']:.4f} |")
        L.append('')
    if 'C1' in D:
        q = out['arms']['C1']['per_behaviour']['sniff']
        ok = q['top1'] >= 0.25
        out['sniff_bar'] = {'top1': q['top1'], 'passes': ok, 'ap_lift': q['ap_lift'], 'top1_lift': q['top1_lift']}
        L += ['## sniffing bar (C1)', '', f"sniff honest top-1% = {q['top1']:.3f} (bar 0.25) -> **{'PASSES' if ok else 'FAILS'}**; "
              f"sniff AP lift = {q['ap_lift']:.2f}, honest top-1% lift = {q['top1_lift']:.2f}, weighted base rate "
              f"{q['base']:.4f}", '']
    # strain split
    L += ['## pair results by intruder type (scoring half restricted to BC- or C57-intruder videos; pick unchanged)', '',
          '| arm | pair primary AUROC, BC (pale) | pair primary AUROC, C57 (dark) | sniff AUROC BC | sniff AUROC C57 | sniff R-precision (AUROC pick) BC | C57 |',
          '|---|---|---|---|---|---|---|']
    for a, R in D.items():
        sp = {st: per_seed(R, PAIR, lambda r, b, st=st: r['strain'][st][b]['test_auc']) for st in ('BC', 'C57')}
        sn = {st: fm(R, lambda r, st=st: r['strain'][st]['sniff']['test_auc']) for st in ('BC', 'C57')}
        sr = {st: fm(R, lambda r, st=st: r['strain'][st]['sniff']['test_rprec_at_auc']) for st in ('BC', 'C57')}
        out['arms'][a]['strain'] = {st: {'pair_primary_mean': float(np.mean(list(sp[st].values()))),
                                         'per_seed': list(sp[st].values()), 'sniff_auroc': sn[st],
                                         'sniff_rprec_at_auc': sr[st]} for st in sp}
        L.append(f"| {a} | {np.mean(list(sp['BC'].values())):.4f} | {np.mean(list(sp['C57'].values())):.4f} | "
                 f"{sn['BC']:.3f} | {sn['C57']:.3f} | {sr['BC']:.3f} | {sr['C57']:.3f} |")
    # LORO
    L += ['', '## leave one resident out (DESCRIPTIVE): pair primary AUROC per held-out resident (mean over seeds x folds; '
          'behaviours without a positive in the held-out subset skipped)', '',
          '| arm | ' + ' | '.join(RESIDENTS) + ' |', '|---|' + '---|' * len(RESIDENTS)]
    for a, R in D.items():
        cells, lo = [], {}
        for r_ in RESIDENTS:
            vals, skip = [], set()
            for v in R.values():
                for fo in 'AB':
                    x = v[fo]['loro'][r_]
                    bb = [b for b in PAIR if x[b]['n_pos_test'] > 0 and np.isfinite(x[b]['test_auc'])]
                    skip |= set(PAIR) - set(bb)
                    if bb:
                        vals.append(np.mean([x[b]['test_auc'] for b in bb]))
            lo[r_] = {'pair_primary': float(np.mean(vals)) if vals else float('nan'), 'skipped_in_some_fold': sorted(skip)}
            cells.append(f"{lo[r_]['pair_primary']:.3f}" + (f" (skip {','.join(sorted(skip))})" if skip else ''))
        out['arms'][a]['loro'] = lo
        L.append(f'| {a} | ' + ' | '.join(cells) + ' |')
    # mask
    mr, rows = mask_recall()
    out['mask'] = mr
    if mr:
        L += ['', '## mask check (released track points; all plan frames of both halves)', '',
              f"kept cells per frame {mr['kept_cells_mean']:.1f} of 1200 video cells ({mr['kept_frac_of_video_cells']:.3f}); "
              f"frames with 0 kept cells {mr['frames_0_kept']} of {mr['frames']}", '',
              '| animal | frames tracked | recall (own cell kept) | recall (3 x 3 neighbourhood) | videos | 10th-percentile video recall |',
              '|---|---|---|---|---|---|']
        for r in rows:
            L.append(f"| {r['animal']} | {r['frames']:,} | {r['recall_own_cell']:.3f} | {r['recall_3x3']:.3f} | "
                     f"{r['videos']} | {r['video_recall_p10']:.3f} |")
    # ceiling
    f = OUT / 'probe' / 'probe_d3l_640_L20.json'
    if f.exists():
        pj = json.loads(f.read_text())
        L += ['', '## supervised ceiling (DIAGNOSTIC; mil_ctx on C1 z_t tokens, folds = released halves, 3 seeds)', '',
              '| probe | behaviour | AUROC | AP |', '|---|---|---|---|']
        for s in pj['summary']:
            L.append(f"| {s['probe']} | {s['behaviour']} | {s['auroc_mean']:.3f} | {s['ap_mean']:.3f} |")
        out['ceiling'] = pj['summary']
    sh = OUT / 'sheets'
    for a in ('C1', 'C4'):
        if (sh / f'sniff_{a}.json').exists():
            j = json.loads((sh / f'sniff_{a}.json').read_text())
            out.setdefault('sheets', {})[a] = j
            for k, v in j.items():
                L.append(f"\nsheet {a} {k}-pick: {v['sheet']} (fold {v['fold']} seed {v['seed']} neuron {v['neuron']}; "
                         f"{v['top16_true_sniff']}/16 top frames are sniff)")
    # GPU
    G = gpu_util()
    tot = sum(r['gpu_h'] for r in G)
    out['gpu'] = {'jobs': G, 'gpu_hours': tot}
    L += ['', '## GPU jobs (nvidia-smi every 60 s)', '', '| job | id | GPU-hours | mean util % | state | node |',
          '|---|---|---|---|---|---|']
    for r in G:
        L.append(f"| {r['job']} | {r['id']} | {r['gpu_h']:.2f} | {r['util_mean']:.1f} | {r['state']} | {r['node']} |")
    L.append(f'\nTotal GPU-hours: {tot:.2f}')
    for j in sorted((OUT / 'encode_json').glob('*of*.json')):
        if 'smoke' in j.name:
            continue
        e = json.loads(j.read_text())
        L.append(f"- encode {j.stem}: {e['frames']} frames, {e['frames_per_wall_s']:.2f} frames / wall-s on {e['gpu']}; "
                 + ', '.join(f"{c} {v['inputs_per_gpu_s']:.1f} images / GPU-s" for c, v in e['per_config'].items()))
    RD.mkdir(parents=True, exist_ok=True)
    (RD / 'res.md').write_text('\n'.join(L) + '\n')
    (RD / 'res.json').write_text(json.dumps(out, indent=1, default=float))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
