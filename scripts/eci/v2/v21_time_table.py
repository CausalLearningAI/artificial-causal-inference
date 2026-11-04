"""Prevalence and per-video time (bias, MAE) section for the v21 extended report.
Reads results/vision/eci_v2_1/extended/report.json (+ recomputes calibrated/raw MAE from the per-seed pickles with the
existing v21_extended.cell_metrics as an identity check), and writes the section into report.md and tables.md in place."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v21_extended as E

OUT = E.OUT
HEAD = '## Prevalence and per-video time (bias, MAE)'
ORD = {'mice': ['nose_nose', 'nt', 'nn', 'np'], 'ants': ['groom_any', 'Y2F', 'B2F', 'single', 'double']}


def med(R, g, k):
    m = R['models'][g]
    return m['median'][k], m['min'][k], m['max'][k]


def cell(R, g, k, nd, sc=1.0):
    """median over seeds [range]  ± median per-seed SE across videos"""
    v, lo, hi = med(R, g, k)
    se = np.median([R['models'][g]['per_seed'][t][k + '_se'] for t in R['models'][g]['per_seed']])
    rng = '' if g == 'v1' and lo == hi else f' [{lo*sc:+.{nd}f}, {hi*sc:+.{nd}f}]'
    return f'{v*sc:+.{nd}f} ± {se*sc:.{nd}f}{rng}'


def build(REP):
    md = [HEAD, '',
          'Per-video numbers are averaged over the held-out videos of the cell; "± X" after a bias or MAE is the standard error (SE) '
          'over videos (for v2 the median of the per-seed SE). v2 = median over the 6 seeds with [min, max] seed range; v1 = median over '
          'its 11 seeds (HMM P(on) fitted to the raw code, as in the rest of this report). Bias = mean over videos of (predicted time - true time); '
          'MAE = mean over videos of |predicted time - true time|; predicted time = sum over frames of P(on). '
          '"raw" = unscaled sum P(on); "calibrated" = sum P(on) times one scale factor fitted on the selection half. '
          'Video length: mice 120 s (600 frames at 5 fps), ants 600 s (600 frames at 1 fps). Frac = fraction of video length.', '']
    for dom, D in REP.items():
        md += [f'### {dom}', '', '**Ground truth (identical for v1 and v2).** Overall = pooled fraction of frames with the label; '
               'per-video = mean ± SE over videos of the per-video fraction; time = true seconds of the behaviour per video, mean ± SE.', '',
               '| behaviour | videos | overall frame fraction | per-video frame fraction, mean ± SE | true time per video, s, mean ± SE |', '|---|---|---|---|---|']
        keys = [f'{"v1" if dom == "mice" else "pooled"}|{b}' for b in ORD[dom]]
        for c in keys:
            R = D['cells'][c]
            md.append(f'| {R["name"]} | {R["n_videos"]} | {R["base_rate"]:.4f} | {R["rate_per_video_mean"]:.4f} ± {R["rate_per_video_se"]:.4f} | '
                      f'{R["true_time_per_video_s_mean"]:.1f} ± {R["true_time_per_video_s_se"]:.1f} |')
        md += ['', '**Per-video total-time bias and MAE** (held-out videos, mean over videos ± SE).', '',
               '| behaviour | model | raw bias, s | raw bias, frac | raw MAE, s | raw MAE, frac | calibrated bias, s | calibrated bias, frac | calibrated MAE, s | calibrated MAE, frac |',
               '|---|---|---|---|---|---|---|---|---|---|']
        for c in keys:
            R = D['cells'][c]
            for g in ('v1', 'v2'):
                md.append(f'| {R["name"]} | {g} | ' + ' | '.join([
                    cell(R, g, 'bias_raw_soft_s', 1), cell(R, g, 'bias_raw_soft_frac', 3),
                    cell(R, g, 'mae_raw_soft_s', 1), cell(R, g, 'mae_raw_soft_frac', 3),
                    cell(R, g, 'bias_cal_soft_s', 2), cell(R, g, 'bias_cal_soft_frac', 4),
                    cell(R, g, 'mae_cal_soft_s', 1), cell(R, g, 'mae_cal_soft_frac', 3)]) + ' |')
        md.append('')
    return '\n'.join(md)


def identity(REP):
    """recompute raw/calibrated soft MAE and bias (frac) per seed from the pickles; compare with report.json per_seed"""
    mx, n = 0.0, 0
    prev = 0.0
    for dom, D in REP.items():
        for g, tags in E.TAGS.items():
            for t in tags:
                J = pickle.load(open(OUT / 'tags' / dom / f'{t}.pkl', 'rb'))
                for c, R in D['cells'].items():
                    C = J['cells'][c]
                    m = E.cell_metrics(C)
                    prev = max(prev, abs(C['T'].sum() / C['n'].sum() - R['base_rate']))
                    ps = R['models'][g]['per_seed'][t]
                    for k in ('mae_cal_soft_frac', 'mae_raw_soft_frac', 'bias_cal_soft_frac', 'bias_raw_soft_frac'):
                        mx = max(mx, abs(float(m[k][0]) - ps[k])); n += 1
                    for k in ('mae_cal_soft_s', 'mae_raw_soft_s'):
                        mx = max(mx, abs(float(m[k][0]) * E.DT[dom] - ps[k])); n += 1
    return mx, n, prev


def put(path, md, before=None):
    s = path.read_text()
    if HEAD in s:  # replace old section
        i = s.index(HEAD)
        j = s.find('\n## ', i + 5) if before else len(s)
        s = s[:i] + (s[j + 1:] if j >= 0 and before else '')
    if before and before in s:
        i = s.index(before)
        s = s[:i] + md + '\n\n' + s[i:]
    else:
        s = s.rstrip('\n') + '\n\n' + md + '\n'
    path.write_text(s)


if __name__ == '__main__':
    REP = json.loads((OUT / 'report.json').read_text())
    mx, n, prev = identity(REP)
    print(f'IDENTITY recomputed-from-pickles vs report.json: max abs diff {mx:.3e} over {n} values; base-rate diff {prev:.3e}')
    for dom in REP:
        for c, R in REP[dom]['cells'].items():
            print(dom, R['name'], 'base_rate', round(R['base_rate'], 4))
    md = build(REP)
    put(OUT / 'report.md', md, before='## How the numbers were made')
    put(OUT / 'tables.md', md)
    print('written')
