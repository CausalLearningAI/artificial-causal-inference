"""
ECI v2.8 "alignment card": one figure per behaviour showing how well ONE sparse-autoencoder neuron of each model
tracks a human-labelled behaviour on held-out videos. One block per model (e.g. old representation v1 vs new
representation v2), same axes for every block. POST HOC illustration of already-scored neurons; no selection here.

Per model block:
  (a) activity over time in 5 held-out videos, the neuron's P(on) (3-state HMM on its frame code; the reports' time
      readout; for v1 the same HMM fitted to its raw code) as a filled trace, annotated bouts as a bar underneath.
      Videos chosen by a fixed, label-only rule (identical for every model): among held-out videos with >= 1 labelled
      frame of the behaviour, the ones at the 25th / 50th / 75th / 95th percentile of annotated time (nearest rank),
      plus the alphabetically first held-out video with no labelled frame (if any).
  (b) precision-recall curve over all held-out frames (ranking = the reports' AUROC readout: P(on) with the raw code as
      tie-break for v2, the raw code for v1), chance line = prevalence; AUROC and average precision printed.
  (c) precision among each video's top 3 / 10 / 30 frames (raw-code ranking, earliest frame first, videos with >= 1
      positive; = the audit's precision@top-k/video), chance = prevalence in those videos.
  (d) per video: predicted vs annotated time in the behaviour (s). Prediction = c x sum of P(on) x frame duration,
      c = (labelled frames) / (sum of P(on)) fitted on the SELECTION frames only (mice: annotator-SG pools; ants: half A).
  (e) the 6 top held-out frames by raw code (1 per video) with the neuron's per-token heat map (cyan box = max token).
Frames: mice = held-out non-SG pools (edge-excluded ext frames, v2.7 table), ants = half B of v3 + v4 (1 fps, 10 min).
Numbers on the card are recomputed here with the audit's code (v28_audit_neurons.boot_metrics) and compared with the
audit tables / reports in <out>.json.

usage:
  v28_card.py data --dom mice --beh nose_nose --rows 'v1=v1_s0:611,v2=PREDk_s2703:3593' --out mice_nose_nose
  v28_card.py plot --out mice_nose_nose
  (row label keys: v1 -> 'old representation (v1)', v2 -> 'new representation (v2)', anything else is printed as given)
Outputs: results/vision/eci_v2_8/figs/<out>.png / .pdf / .json; card data (not committed) figs/data/<out>.pkl
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[2]
OUT = REPO / 'results/vision/eci_v2_8/figs'
NAME = {'nose_nose': 'nose-nose', 'nt': 'nose-tail', 'groom_any': 'grooming-to-focal (any)'}
TITLE = {'nose_nose': 'Mice: nose-nose sniffing', 'nt': 'Mice: nose-tail sniffing',
         'groom_any': 'Ants: grooming-to-focal (any)'}
LABEL = {'v1': 'old representation (v1)', 'v2': 'new representation (v2)'}
COLOR = {'v1': '#8c8c8c', 'v2': '#1f6fb4'}
NFR = 6
CROP = {'mice': 0.42, 'ants': 0.42}
TILE = 220


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------------------------------------- data
def load_cols(dom, key, js, F):
    """-> score readout X (n, len(js)) used for AUROC, raw code R, P(on) P; aligned to F."""
    import v28_audit_neurons as A
    import v22_full as VF
    import v24_top3 as T3
    if dom == 'mice':
        pr, pp = A.mice_paths(key)
        R = T3.cols_chunked(pr, js)
        st = np.r_[np.flatnonzero(np.r_[True, F.obs.values[1:] != F.obs.values[:-1]]), len(F)].astype(np.int64)
        if key.startswith('v1_'):  # v1: the reports' HMM on its raw code (v22_eval.cmd_tag, one sequence per video)
            return R, R, VF.pon_fast(R, st)[0]
        P = T3.cols_chunked(pp, js)
        return P, R, P
    import v23_ants as AN
    st = np.r_[np.flatnonzero(np.r_[True, F.obs.values[1:] != F.obs.values[:-1]]), len(F)].astype(np.int64)
    if key.startswith('v1_'):
        Rall = AN.readout(key, F)[1]
        R = np.ascontiguousarray(Rall[:, js], np.float32)
        del Rall
        return R, R, VF.pon_fast(R, st)[0]
    pr, pp = A.ants_paths(key)
    R = np.asarray(np.load(pr, mmap_mode='r')[:, js], np.float32)
    P = np.asarray(np.load(pp, mmap_mode='r')[:, js], np.float32)
    return P, R, P


def pick_videos(vids, tt):
    """fixed label-only rule: percentiles 25/50/75/95 of annotated time among positive videos + first zero video."""
    vids = np.asarray(vids)
    pos = np.flatnonzero(tt > 0)
    o = pos[np.lexsort((vids[pos], tt[pos]))]
    got = []
    for q in (25, 50, 75, 95):
        i = o[min(len(o) - 1, max(0, int(np.ceil(q / 100 * len(o))) - 1))]
        if i not in got:
            got.append(int(i))
    zero = sorted(np.flatnonzero(tt == 0), key=lambda i: vids[i])
    if zero:
        got.append(int(zero[0]))
    return got


def pr_curve(score, y, npts=600):
    o = np.argsort(-score, kind='stable')
    yo = y[o].astype(np.float64)
    tp = np.cumsum(yo)
    n = np.arange(1, len(yo) + 1)
    prec, rec = tp / n, tp / max(yo.sum(), 1)
    idx = np.unique(np.r_[np.geomspace(1, len(yo), npts).astype(int) - 1, np.flatnonzero(np.diff(rec) > 0)[::max(1, int(yo.sum() // npts))]])
    return rec[idx].astype(np.float32), prec[idx].astype(np.float32)


def crop_tile(img, m):
    """crop CROP x side around the max token; map upsampled (nearest) to the image grid; both resized to TILE px."""
    from PIL import Image
    H, Wd = img.shape[:2]
    G = m.shape[0]
    q = int(np.argmax(m))
    pr, pc = q // G, q % G
    ps = H / G
    hw = CROP_D * H / 2
    cy, cx = (pr + 0.5) * ps, (pc + 0.5) * ps
    y0, x0 = int(np.clip(cy - hw, 0, H - 2 * hw)), int(np.clip(cx - hw, 0, Wd - 2 * hw))
    s = int(2 * hw)
    ic = img[y0:y0 + s, x0:x0 + s]
    ys = ((np.arange(y0, y0 + s) + 0.5) / ps).astype(int).clip(0, G - 1)
    xs = ((np.arange(x0, x0 + s) + 0.5) / ps).astype(int).clip(0, G - 1)
    mc = m[np.ix_(ys, xs)]
    ic = np.asarray(Image.fromarray(ic).resize((TILE, TILE), Image.BILINEAR))
    mc = np.asarray(Image.fromarray(mc.astype(np.float32)).resize((TILE, TILE), Image.NEAREST))
    box = ((pc * ps - x0) * TILE / s, (pr * ps - y0) * TILE / s, ps * TILE / s)
    return ic, mc.astype(np.float16), box


CROP_D = 0.42


def tiles_for(dom, key, j, F, rows_top, rot, plans):
    """heat-map tiles of neuron j of model key on the given F rows."""
    import torch
    from PIL import Image
    SH = sys.modules['v28_sheets']
    S, XS = SH.G['S'], SH.G['XS']
    out = []
    if key.startswith('v1_'):
        need = {int(r): {j} for r in rows_top}
        got = S.v1_maps(dom, F, need, torch.device('cpu'))
        if dom == 'mice':
            maps, fp, arow, rr = got
            jpg = {int(r): REPO / 'dataset' / fp[ar] for r, ar in zip(rr, arow)}
        else:
            maps = got
            jpg = {int(r): REPO / 'dataset' / F.frame_path.values[r] for r in need}
        for r in rows_top:
            img = np.asarray(Image.open(jpg[int(r)]).convert('RGB'))
            img = np.ascontiguousarray(np.rot90(img, rot.get(str(F.obs.values[r]), 0)))
            out.append((img, maps[(int(r), j)]))
    else:
        enc = SH.Enc(dom, key)
        for r in rows_top:
            m, Ss = enc.map(F, int(r), j)
            img = XS.decode_v2(dom, S.store_series(dom, F, int(r)), str(F.obs.values[r]), int(F.k.values[r]), Ss, plans, rot)
            out.append((img, m))
    return out


def cmd_data(a):
    import torch
    t0 = time.time()
    import v28_sheets as SH
    SH.init('27')
    import v28_audit_neurons as A
    S = SH.G['S']
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    dom, b = a.dom, a.beh
    global CROP_D
    CROP_D = CROP[dom]
    F, ev = S.table(dom)
    sel = F.sg.values if dom == 'mice' else (F.half.values == 'A')
    dt = 0.2 if dom == 'mice' else 1.0
    rr = np.flatnonzero(ev)
    obs = F.obs.values[rr]
    vcode, vids = pd.factorize(obs)
    nv = len(vids)
    y_all = F[f'y_{b}'].values.astype(bool)
    v_all = F[f'v_{b}'].values.astype(bool) if f'v_{b}' in F.columns else np.ones(len(F), bool)
    y, v = y_all[rr], v_all[rr]
    tt = np.bincount(vcode, weights=y) * dt
    nfr = np.bincount(vcode).astype(float)
    show = pick_videos(vids, tt)
    log(f'{dom} {b}: held-out rows {len(rr):,}, videos {nv}, positives {int(y.sum()):,} ({100 * y.mean():.2f}%), '
        f'valid {int(v.sum()):,}; raster videos {[vids[i] for i in show]} ({time.time() - t0:.0f}s)')
    rot, plans = {}, SH.G['XS'].plans_of(dom)
    if dom == 'mice':
        from src.eci.foreground import ODOR_ROT90
        oc = pd.read_csv(REPO / 'dataset/mice/v1/eci/odor_corner.csv').set_index('observation_id')['odor_corner']
        rot = {o: int(ODOR_ROT90[oc[o]]) for o in F.obs.unique()}
    kk = F.k.values[rr]
    st = np.r_[np.flatnonzero(np.r_[True, vcode[1:] != vcode[:-1]]), len(vcode)]
    D = {'dom': dom, 'beh': b, 'dt': dt, 'n_rows': int(len(rr)), 'n_videos': int(nv), 'n_pos': int(y.sum()),
         'prevalence': float(y.mean()), 'n_units': int(len(set(F.unit.values[rr]))),
         'show': [{'vid': str(vids[i]), 'annot_s': float(tt[i]), 'len_s': float(nfr[i] * dt)} for i in show], 'rows': []}
    for spec in a.rows.split(','):
        lab, ks = spec.split('=')
        key, j = ks.split(':')
        j = int(j)
        X, R, P = load_cols(dom, key, [j], F)
        x, r, p = X[:, 0], R[:, 0], P[:, 0]
        sc = x.astype(np.float64) + 1e-6 * r / max(float(r.max()), 1e-12)
        log(f'{lab} {key} #{j}: loaded ({time.time() - t0:.0f}s)')
        m = A.boot_metrics(sc[rr], r[rr], y, v, obs, vcode, np.ones((1, nv)), A.KS)
        met = {k_: float(val[0]) for k_, val in m.items()}
        # top-k chance: prevalence in videos with >= 1 positive (valid frames)
        posv = np.bincount(vcode, weights=y & v) > 0
        ev_ = posv[vcode] & v
        met['topk_chance'] = float(y[ev_].mean())
        rec, prec = pr_curve(sc[rr], y)
        c = float(y_all[sel].sum() / p[sel].sum())
        Sp = np.bincount(vcode, weights=p[rr]) * dt
        pred = c * Sp
        const = float(y_all[sel].mean()) * nfr * dt
        tim = {'c': c, 'pred_s': pred.tolist(), 'annot_s': tt.tolist(), 'mae_s': float(np.abs(pred - tt).mean()),
               'bias_s': float((pred - tt).mean()), 'pearson_r': float(np.corrcoef(pred, tt)[0, 1]),
               'const_mae_s': float(np.abs(const - tt).mean()), 'sumP_s': Sp.tolist()}
        tr = []
        for i in show:
            a0, b0 = st[i], st[i + 1]
            tr.append({'t_s': (kk[a0:b0] * 0.2).astype(np.float32),
                       'p': p[rr][a0:b0].astype(np.float16), 'raw': r[rr][a0:b0].astype(np.float16), 'y': y[a0:b0]})
        rq = float(np.percentile(r[rr][r[rr] > 0], 99.5)) if (r[rr] > 0).any() else 1.0
        # top frames: 1 per video by raw code (v25_sheet.top_frames rule), valid frames of the behaviour
        cand = np.flatnonzero(ev & v_all)
        o = np.lexsort((np.arange(len(cand)), -r[cand].astype(np.float64)))
        top, seen = [], set()
        for i in o:
            vv = F.obs.values[cand[i]]
            if vv in seen:
                continue
            seen.add(vv)
            top.append(int(cand[i]))
            if len(top) == NFR:
                break
        tiles = []
        if not a.no_frames:
            for (img, mp), row in zip(tiles_for(dom, key, j, F, top, rot, plans), top):
                ic, mc, box = crop_tile(img, mp)
                tiles.append({'img': ic, 'map': mc, 'box': box, 'map_max': float(mp.max()), 'code': float(r[row]),
                              'y': bool(y_all[row]), 'obs': str(F.obs.values[row]), 'k': int(F.k.values[row])})
            log(f'{lab}: {len(tiles)} tiles, identity max |map max - stored code| '
                f'{max(abs(np.float16(t["map_max"]) - np.float16(t["code"])) for t in tiles):.4f} ({time.time() - t0:.0f}s)')
        D['rows'].append({'label': lab, 'key': key, 'neuron': j, 'metrics': met, 'pr': (rec, prec), 'time': tim,
                          'traces': tr, 'raw_q995': rq, 'tiles': tiles, 'fire_rate_raw': float((r[rr] > 0).mean()),
                          'top_labelled': int(sum(bool(y_all[t]) for t in top)), 'top_rows': top})
        log(f'{lab} {key} #{j}: AUROC {met["auc"]:.4f} AP {met["ap"]:.4f} top3 {met["top3"]:.4f} top10 {met["top10"]:.4f} '
            f'top30 {met["top30"]:.4f}; time r {tim["pearson_r"]:.3f} MAE {tim["mae_s"]:.1f} s (const {tim["const_mae_s"]:.1f} s)')
    (OUT / 'data').mkdir(parents=True, exist_ok=True)
    pickle.dump(D, open(OUT / 'data' / f'{a.out}.pkl', 'wb'))
    log(f'data -> {OUT / "data" / (a.out + ".pkl")} ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- figure
def cmd_plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Rectangle
    D = pickle.load(open(OUT / 'data' / f'{a.out}.pkl', 'rb'))
    dom, b, dt = D['dom'], D['beh'], D['dt']
    bn = NAME[b]
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans',
                         'image.composite_image': False})  # PDF: uint8 frame + float heat map cannot be composited
    rows = D['rows']
    nr = len(rows)
    H = 1.5 + 8.0 * nr + 1.3
    fig = plt.figure(figsize=(22, H))
    gs = GridSpec(nr, 1, figure=fig, left=0.075, right=0.975, top=1 - 1.75 / H, bottom=1.45 / H, hspace=0.22)
    tmax = max(float(tr['t_s'][-1] - tr['t_s'][0]) + dt for tr in rows[0]['traces'])
    unit_t = 60.0
    bin_s = 3.0 if dom == 'mice' else 1.0
    allp = [x for r in rows for x in r['time']['pred_s']] + rows[0]['time']['annot_s']
    lim = max(allp) * 1.05
    for ri, R in enumerate(rows):
        col = COLOR.get(R['label'], '#555555')
        lab = LABEL.get(R['label'], R['label'])
        g = GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[ri], height_ratios=[1.75, 1.0], width_ratios=[2.6, 1.05, 0.95, 1.05],
                                    hspace=0.48, wspace=0.34)
        # (a) raster
        ax = fig.add_subplot(g[0, 0])
        nshow = len(R['traces'])
        for i, (tr, sv) in enumerate(zip(R['traces'], D['show'])):
            y0 = (nshow - 1 - i) * 1.6
            t = (tr['t_s'].astype(np.float64) - float(tr['t_s'][0])) / unit_t
            # display only: max of P(on) over BIN_S-second bins (a 1-frame spike stays visible at this width)
            nb_ = max(1, int(round(bin_s / dt)))
            bi = np.floor((tr['t_s'] - tr['t_s'][0]) / (nb_ * dt)).astype(int)
            ub, inv = np.unique(bi, return_inverse=True)
            pb = np.zeros(len(ub))
            np.maximum.at(pb, inv, tr['p'].astype(np.float64))
            tb = ub * nb_ * dt / unit_t
            seg = np.r_[0, np.flatnonzero(np.diff(ub) > 1) + 1, len(ub)]
            for s0, s1 in zip(seg[:-1], seg[1:]):  # scored stretches; gaps (no scored frames) stay blank
                xs = np.r_[tb[s0:s1], tb[s1 - 1] + nb_ * dt / unit_t]
                ax.stairs(y0 + pb[s0:s1], xs, baseline=y0, fill=True, color=col, lw=0, alpha=0.95)
                ax.plot([xs[0], xs[-1]], [y0 - 0.06, y0 - 0.06], color='#bbbbbb', lw=0.8)
            yy = tr['y']
            if yy.any():
                e = np.diff(np.r_[0, yy.astype(int), 0])
                for s_, e_ in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
                    ax.add_patch(Rectangle((t[s_] - dt / 2 / unit_t, y0 - 0.42), max((t[e_ - 1] - t[s_]) + dt / unit_t, 0.12), 0.3,
                                           color='#d62728', lw=0))
            ax.text(-0.01, y0 + 0.35, f'{sv["vid"]}\nannotated {sv["annot_s"]:.0f} s', transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=10)
        ax.set_ylim(-0.6, (nshow - 1) * 1.6 + 1.1)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlim(0, tmax / unit_t)
        ax.set_xlabel('time in scored window (min)' if dom == 'mice' else 'time (min)')
        ax.set_title(f'{lab}  |  neuron #{R["neuron"]} ({R["key"].replace("_s", " seed ")})', loc='left', fontsize=15,
                     weight='bold', color='black', pad=24)
        ax.text(0, 1.02, f'filled = P(on), 0 to 1' + (f' (max per {bin_s:.0f} s)' if bin_s > dt else '') + ';  red = annotated bouts;  grey = scored frames',
                transform=ax.transAxes, fontsize=10.5, color='#333333')
        # (b) PR curve
        M = R['metrics']
        ax = fig.add_subplot(g[0, 1])
        for R2 in rows:
            if R2 is not R:
                ax.plot(*R2['pr'], color='#dddddd', lw=1.5, zorder=1)
        ax.plot(*R['pr'], color=col, lw=2.4, zorder=3)
        ax.axhline(D['prevalence'], color='k', ls=':', lw=1.2)
        ax.text(0.98, D['prevalence'] + 0.015, f'chance {100 * D["prevalence"]:.1f}%', ha='right', va='bottom', fontsize=9.5,
                transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f'recall (share of {bn} frames found)')
        ax.set_ylabel('precision (share of hits that are correct)')
        ax.set_title(f'AUROC {M["auc"]:.3f}\naverage precision {M["ap"]:.3f}', fontsize=12.5, loc='left')
        # (c) precision@top-k per video
        ax = fig.add_subplot(g[0, 2])
        ks = (3, 10, 30)
        vals = [M[f'top{k}'] for k in ks]
        ax.bar(range(3), vals, color=col, width=0.6)
        for i_, v_ in enumerate(vals):
            ax.text(i_, v_ + 0.02, f'{v_:.2f}', ha='center', fontsize=12)
        ax.axhline(M['topk_chance'], color='k', ls=':', lw=1.2)
        ax.text(2.45, M['topk_chance'] + 0.015, f'chance {100 * M["topk_chance"]:.1f}%', ha='right', va='bottom', fontsize=9.5)
        ax.set_xticks(range(3), [f'top {k}' for k in ks])
        ax.set_ylim(0, 1)
        ax.set_ylabel('precision (share labelled)')
        ax.set_title('precision of each video\'s\ntop frames (per video)', fontsize=12.5, loc='left')
        # (d) per-video time
        ax = fig.add_subplot(g[0, 3])
        T_ = R['time']
        ax.plot([0, lim], [0, lim], color='k', lw=1, ls='--')
        ax.scatter(T_['annot_s'], T_['pred_s'], s=22, color=col, alpha=0.8, edgecolor='white', lw=0.4)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('annotated time per video (s)')
        ax.set_ylabel('predicted time per video (s)')
        ax.set_title(f'time per video: Pearson r {T_["pearson_r"]:.2f}\nmean abs. error {T_["mae_s"]:.0f} s\n(constant-rate guess {T_["const_mae_s"]:.0f} s)',
                     fontsize=11.5, loc='left')
        # (e) frames
        gf = GridSpecFromSubplotSpec(1, NFR, subplot_spec=g[1, :], wspace=0.06)
        for i_, tl in enumerate(R['tiles']):
            ax = fig.add_subplot(gf[0, i_])
            ax.imshow(tl['img'])
            mp = np.asarray(tl['map'], np.float32)
            ax.imshow(np.ma.masked_less_equal(mp, 0), cmap='inferno', alpha=0.45, vmin=0, vmax=max(float(mp.max()), 1e-6),
                      interpolation='nearest')
            bx, by, bs = tl['box']
            ax.add_patch(Rectangle((bx, by), bs, bs, fill=False, ec='cyan', lw=1.6))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color('#2ca02c' if tl['y'] else '#d62728')
                s.set_linewidth(4)
            tsec = tl['k'] * 0.2
            ax.set_title(f'{"labelled " + bn if tl["y"] else "NOT labelled"}\n{tl["obs"]} {int(tsec // 60):02d}:{tsec % 60:04.1f}',
                         fontsize=9.5, color='#1a7f1a' if tl['y'] else '#c00000')
            if i_ == 0:
                ax.text(0, 1.25, f'6 strongest held-out frames (1 per video): {R["top_labelled"]} of {len(R["tiles"])} labelled {bn}'
                        '   (heat = neuron activity per image patch, cyan box = strongest patch)',
                        transform=ax.transAxes, fontsize=11.5)
    held = ('held-out mice: 12 pools not annotated by annotator SG, '
            f'{D["n_videos"]} videos, {D["n_rows"]:,} frames at 5 fps; {bn} in {100 * D["prevalence"]:.2f}% of frames'
            if dom == 'mice' else
            f'held-out ants: half B of series v3 + v4, {D["n_videos"]} videos x first 10 min at 1 fps = {D["n_rows"]:,} frames; '
            f'{bn} in {100 * D["prevalence"]:.1f}% of frames')
    fig.suptitle(f'{TITLE[b]}: how well does one neuron track it?   ' + '  vs  '.join(LABEL.get(r['label'], r['label']) for r in rows),
                 x=0.01, ha='left', y=1 - 0.25 / H, fontsize=19, weight='bold')
    fig.text(0.01, 1 - 0.8 / H, held, fontsize=13)
    sel = 'the annotator-SG pools' if dom == 'mice' else 'half A'
    zero = ('plus the first video without it' if any(s['annot_s'] == 0 for s in D['show'])
            else f'no held-out video is without {bn}')
    foot = (f'Neurons were chosen on the selection frames ({sel}), never on these frames. Videos in the time plots: fixed rule on the '
            f'labels only (25th / 50th / 75th / 95th percentile of annotated time among videos with the behaviour; {zero}), '
            'same videos for every model.\n'
            'P(on) = probability the neuron is "on" from a 3-state hidden Markov model of its activity (the reports\' readout; v1: same model '
            'fitted to its raw code). Precision-recall and AUROC rank frames as the reports do (v2: P(on), ties by raw activity; v1: raw '
            'activity).\nTop-frame precision: raw activity, videos with >= 1 labelled frame. Predicted time = c x summed P(on), c fitted '
            f'on {sel} only (no held-out frame). Constant guess = selection-set rate x video length.')
    cc = crosscheck(dom, b, rows)
    for d_ in cc:  # AUROC under the reports' tie rule when it differs from the card's (P(on) saturates: ties)
        ref = d_.get('audit_auc', d_.get('report_auc'))
        if ref is not None and abs(ref - d_['card_auc']) > 0.001:
            foot += (f"\nAUROC of {LABEL.get(next(r['label'] for r in rows if r['key'] == d_['key']), d_['key'])} #{d_['neuron']} with "
                     f"tied P(on) values averaged (the reports' and audit's rule): {ref:.3f}; the card breaks ties by raw activity "
                     f"({d_['card_auc']:.3f}) as the audit's final table does.")
    import textwrap
    foot = '\n'.join(textwrap.fill(' '.join(par.split()), 235) for par in foot.split('\n'))
    fig.text(0.01, 0.2 / H, foot, fontsize=10.5, va='bottom', color='#333333')
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{a.out}.png', dpi=a.dpi)
    fig.savefig(OUT / f'{a.out}.pdf')
    plt.close(fig)
    js = {'dom': dom, 'beh': b, 'frames': {k_: D[k_] for k_ in ('n_rows', 'n_videos', 'n_pos', 'prevalence', 'n_units')},
          'raster_videos': D['show'],
          'rows': [{k_: R[k_] for k_ in ('label', 'key', 'neuron', 'metrics', 'fire_rate_raw', 'top_labelled')} |
                   {'time': {k_: v_ for k_, v_ in R['time'].items() if not isinstance(v_, list)},
                    'tiles': [{k_: t[k_] for k_ in ('obs', 'k', 'code', 'map_max', 'y')} for t in R['tiles']]} for R in rows]}
    js['crosscheck'] = cc
    (OUT / f'{a.out}.json').write_text(json.dumps(js, indent=1, default=float))
    log(f'card -> {OUT / (a.out + ".png")} ({(OUT / (a.out + ".png")).stat().st_size / 1e6:.2f} MB); crosscheck {js["crosscheck"]}')


def crosscheck(dom, b, rows):
    """card numbers vs the v2.8 audit tables (per-dictionary pickles) / v2.7 ants report, same neuron and frame set."""
    out = []
    for R in rows:
        M, key, j = R['metrics'], R['key'], R['neuron']
        d = {'key': key, 'neuron': j}
        wp = REPO / 'results/vision/eci_v2_8/audit/work' / f'{dom}_{key}.pkl'
        if wp.exists():
            W = pickle.load(open(wp, 'rb'))
            S = W['stats']
            h = 'non' if dom == 'mice' else 'B'
            d['audit_auc'] = float(S[f'auc|{h}|{b}'][j])
            d['audit_ap'] = float(S[f'ap|{h}|{b}'][j])
            if dom == 'mice':
                d['audit_top3'] = W['topk'].get(('non', b, j, 3))
            else:
                d['audit_top3'] = W['dirs']['A2B']['top3'].get(('held', b, j))
        else:
            d['audit'] = 'no audit pickle for this dictionary'
        if dom == 'ants':
            rep = json.loads((REPO / 'results/vision/eci_v2_7/ants/report.json').read_text())['dirs']['A2B']
            for arm, ks in rep['picks'].items():
                if key in ks and int(ks[key]) == j:
                    idx = list(ks).index(key)
                    d['report_auc'] = rep['levels'][arm][f'{b}|auc']['seeds'][idx]
                    d['report_top3'] = rep['levels'][arm][f'{b}|top3']['seeds'][idx]
                    break
        d['card_auc'], d['card_ap'], d['card_top3'] = M['auc'], M['ap'], M['top3']
        out.append(d)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('data', 'plot', 'all'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--beh')
    p.add_argument('--rows', default='')
    p.add_argument('--out', required=True)
    p.add_argument('--no-frames', action='store_true')
    p.add_argument('--dpi', type=int, default=80)
    a = p.parse_args()
    if a.step in ('data', 'all'):
        cmd_data(a)
    if a.step in ('plot', 'all'):
        cmd_plot(a)
