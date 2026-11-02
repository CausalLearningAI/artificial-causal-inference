"""
ECI v2.1 freeze: one clean token mask + one frame readout for arm V, confirmed on fresh seeds.
Pre-registration: scripts/eci/v2/v21_freeze_prereg.md (written after the b-token characterisation, before any SAE of
this file was trained and before any score).

Token categories inside the stored U_t (flag bit0 = M_t mask v5, bit1 = N2): a = M and N2, b = M only, c = N2 only.

Steps
  btok --dom D     (CPU)  where are the b tokens: 4-neighbour count inside M_t, thin parts (removed by a one-patch
                          opening), probe value; share of a / b kept by the candidate single rules -> OUT/freeze/btok_<dom>.json
  sheet --dom D    (GPU)  overlay sheet of a / b / c tokens + the eroded mask on 6 eval frames -> OUT/freeze/sheet_<dom>.png
  prep --dom D --mask E   (CPU)  S / V training samples on mask E (the S window plan, rng 0; 16M / 1.5M / 1.5M, rng 1)
  train --dom D --arm SE --seed s   (GPU)  v21_train.cmd_train
  codes --dom D --arms S,V --mask M --seeds ...  (GPU)  frame codes under every readout (max, mean, top10, maxmean)
  v1codes --dom D  (GPU)  v1 s0-10 frame codes under every readout (v1's own tokens, same frames as eval.py)
  score --dom D --model KEY --agg R   (CPU)  HMM P(on) once + gate (AMENDMENT 5) + ladder columns
  boot / report
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
import v21_train as T  # noqa: E402

FZ = T.OUT / 'freeze'
LBF = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_1/ladder_freeze')
AGGS = ('max', 'mean', 'top10', 'maxmean')
log = T.log


# ---------------------------------------------------------------------------------------------- token geometry
def nbr_counts(pos, fi, inM, G):
    """for every row: number of 4-neighbours (n4) and 8-neighbours (n8) that are M_t tokens of the same frame."""
    key = fi.astype(np.int64) * (G * G) + pos
    mk = np.sort(key[inM])
    r, c = pos // G, pos % G

    def has(dr, dc):
        rr, cc = r + dr, c + dc
        ok = (rr >= 0) & (rr < G) & (cc >= 0) & (cc < G)
        q = fi.astype(np.int64) * (G * G) + np.clip(rr, 0, G - 1) * G + np.clip(cc, 0, G - 1)
        j = np.searchsorted(mk, q)
        hit = (j < len(mk)) & (mk[np.minimum(j, len(mk) - 1)] == q)
        return ok & hit
    d4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    d8 = d4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    h = {d: has(*d) for d in d8}
    return sum(h[d].astype(np.int8) for d in d4), sum(h[d].astype(np.int8) for d in d8)


def mask_rows(mode, flag, pos, fi, prob, G):
    """rows of a stored window that belong to the token set `mode`."""
    inM = (flag & 1) > 0
    if mode == 'M':
        return inM
    if mode == 'E':  # mask v5 eroded by one patch: M_t tokens whose 4 neighbours are all M_t tokens
        n4, _ = nbr_counts(pos, fi, inM, G)
        return inM & (n4 == 4)
    if mode.startswith('P'):  # stricter probe threshold inside M_t: P<uint8>
        return inM & (prob >= int(mode[1:]))
    raise ValueError(mode)


def opening_thin(pos_m, G):
    """M tokens of ONE frame (positions) that a one-patch opening (erode then dilate, 4-connected cross) removes."""
    from scipy import ndimage as nd
    m = np.zeros((G, G), bool)
    m.ravel()[pos_m] = True
    cr = nd.generate_binary_structure(2, 1)
    op = nd.binary_dilation(nd.binary_erosion(m, cr), cr)
    return ~op.ravel()[pos_m]


# ---------------------------------------------------------------------------------------------- btok
def cmd_btok(a):
    import eval as EV
    dom = a.dom
    F = EV.frame_table(dom)
    E = T.frame_sets(dom)['eval']
    labs = [c[2:] for c in F.columns if c.startswith('y_')]
    sub = np.arange(0, len(E), a.every)
    rules = ['E', 'O'] + [f'P{t}' for t in (242, 248, 250, 251, 252, 253, 254, 255)]
    H = {cat: {'n4': np.zeros(5, np.int64), 'n8': np.zeros(9, np.int64), 'thin': 0, 'n': 0,
               'prob_hist': np.zeros(256, np.int64)} for cat in 'ab'}
    keep = {r: {'a': 0, 'b': 0} for r in rules}
    per_frame = {r: [] for r in rules + ['M', 'a', 'b']}
    rows_used = []
    t0 = time.time()
    Es = E.iloc[sub]
    for (ser, obs), g in Es.groupby(['series', 'obs'], sort=False):
        z = np.load(T.STORE / dom / T.VER / 'videos' / ser / obs / 'index.npz')
        G = int(z['G'])
        kpos = {int(k): i for i, k in enumerate(z['k'])}
        ptr, idx, flag, prob = z['ptr'], z['idx'].astype(np.int64), z['flag'], z['prob']
        for row, k in zip(g.index.values, g.k.values):
            i = kpos[int(k)]
            r0, r1 = int(ptr[i]), int(ptr[i + 1])
            f, p, pr = flag[r0:r1] & 3, idx[r0:r1], prob[r0:r1]
            inM = (f & 1) > 0
            fi = np.zeros(len(p), np.int64)
            n4, n8 = nbr_counts(p, fi, inM, G)
            thin = np.zeros(len(p), bool)
            if inM.any():
                thin[inM] = opening_thin(p[inM], G)
            ca, cb = f == 3, f == 1
            for cat, m in (('a', ca), ('b', cb)):
                H[cat]['n4'] += np.bincount(n4[m], minlength=5)
                H[cat]['n8'] += np.bincount(n8[m], minlength=9)
                H[cat]['thin'] += int(thin[m].sum())
                H[cat]['n'] += int(m.sum())
                H[cat]['prob_hist'] += np.bincount(pr[m], minlength=256)
            kr = {'E': inM & (n4 == 4), 'O': inM & ~thin}
            kr.update({r: inM & (pr >= int(r[1:])) for r in rules if r.startswith('P')})
            for r in rules:
                keep[r]['a'] += int((kr[r] & ca).sum())
                keep[r]['b'] += int((kr[r] & cb).sum())
                per_frame[r].append(int(kr[r].sum()))
            per_frame['M'].append(int(inM.sum()))
            per_frame['a'].append(int(ca.sum()))
            per_frame['b'].append(int(cb.sum()))
            rows_used.append(row)
    log(f'{dom} btok loop {len(rows_used)} frames ({time.time() - t0:.0f}s)')
    ru = np.array(rows_used)
    pf = {k: np.array(v) for k, v in per_frame.items()}
    out = {'frames': int(len(ru)), 'every': a.every, 'G': G}
    for cat in 'ab':
        h = H[cat]
        ph = h['prob_hist']
        cdf = np.cumsum(ph) / max(ph.sum(), 1)
        out[cat] = {'tokens': h['n'], 'per_frame_mean': float(pf[cat].mean()),
                    'n4_share_0_1_2_3_4': (h['n4'] / max(h['n'], 1)).round(4).tolist(),
                    'n8_share_0..8': (h['n8'] / max(h['n'], 1)).round(4).tolist(),
                    'boundary_share_n4_lt_4': float(1 - h['n4'][4] / max(h['n'], 1)),
                    'thin_share_removed_by_opening': float(h['thin'] / max(h['n'], 1)),
                    'probe_uint8_q10_25_50_75_90': [int(np.searchsorted(cdf, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)]}
    out['rules'] = {}
    for r in rules:
        out['rules'][r] = {'a_kept_share': keep[r]['a'] / max(H['a']['n'], 1), 'b_kept_share': keep[r]['b'] / max(H['b']['n'], 1),
                           'tokens_per_frame_mean': float(pf[r].mean()), 'tokens_per_frame_median': float(np.median(pf[r])),
                           'share_frames_empty': float((pf[r] == 0).mean()),
                           'share_of_M_tokens_kept': float(pf[r].sum() / max(pf['M'].sum(), 1))}
    out['M_tokens_per_frame_mean'] = float(pf['M'].mean())
    # tokens per frame on positive vs negative frames of every label (does a rule drop interaction frames?)
    out['by_label'] = {}
    for b in labs:
        v, y = F[f'v_{b}'].values[ru], F[f'y_{b}'].values[ru].astype(bool)
        if (v & y).sum() == 0:
            continue
        out['by_label'][b] = {f'{r}_{s}': float(pf[r][v & (y if s == 'pos' else ~y)].mean())
                              for r in ['M', 'E', 'O', 'P250', 'a', 'b'] for s in ('pos', 'neg')}
        out['by_label'][b]['E_empty_pos'] = float((pf['E'][v & y] == 0).mean())
    FZ.mkdir(parents=True, exist_ok=True)
    (FZ / f'btok_{dom}.json').write_text(json.dumps(out, indent=1))
    log(json.dumps({k: out[k] for k in ('a', 'b', 'rules')}))


# ---------------------------------------------------------------------------------------------- sheet
def cmd_sheet(a):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import eval as EV
    import v20 as V
    import v21_store as ST
    dev = torch.device('cuda')
    dom = a.dom
    F = EV.frame_table(dom)
    E = T.frame_sets(dom)['eval']
    rng = np.random.default_rng(0)
    lab = 'nn' if dom == 'mice' else 'groom_any'
    ypos = np.flatnonzero(F[f'y_{lab}'].fillna(0).values.astype(bool) & F[f'v_{lab}'].values)
    pick = list(rng.choice(ypos, 3, replace=False)) + list(rng.choice(len(E), 3, replace=False))
    vids = {(s, v['obs']): v for s, v in ST.lab_videos(dom)}
    fig, axs = plt.subplots(2, 3, figsize=(18, 12.5))
    local = Path(os.environ['LOCAL_DIR'])
    NS = V.make_series_class(dom)
    SE = {}
    for ax, row in zip(axs.ravel(), pick):
        ser, obs, k = E.series[row], E.obs[row], int(E.k[row])
        v, srcf = ST.stage_video(vids[(ser, obs)], local)
        if ser not in SE:
            SE[ser] = NS(V.R.series_of(V.dcfg(dom), ser), dev, encoder=None, load_encoder=False)
        Se = SE[ser]
        jb = Se.job(v, k, 1, threads=2)
        fr = Se.run_job(**jb)
        with torch.no_grad():
            x, _ = Se.frames(v, fr[:1], np.array([k]))
        img = x[0].permute(1, 2, 0).float().cpu().numpy()
        img = (img - img.min()) / max(img.max() - img.min(), 1e-6)
        z = np.load(T.STORE / dom / T.VER / 'videos' / ser / obs / 'index.npz')
        G, S = int(z['G']), int(z['S'])
        i = int(np.flatnonzero(z['k'] == k)[0])
        r0, r1 = int(z['ptr'][i]), int(z['ptr'][i + 1])
        f, p = z['flag'][r0:r1] & 3, z['idx'][r0:r1].astype(np.int64)
        er = mask_rows('E', f, p, np.zeros(len(p), np.int64), z['prob'][r0:r1], G)
        ax.imshow(img)
        ps = S / G
        col = {3: (0.1, 0.8, 0.2), 1: (0.95, 0.1, 0.1), 2: (0.2, 0.4, 1.0)}
        for q, ff, e in zip(p, f, er):
            if ff == 0:
                continue
            yy, xx = (q // G) * ps, (q % G) * ps
            ax.add_patch(plt.Rectangle((xx, yy), ps, ps, fill=True, alpha=0.28, color=col[int(ff)], lw=0))
            if e:
                ax.plot(xx + ps / 2, yy + ps / 2, '.', ms=2.5, color='white')
        na, nb, nc = int((f == 3).sum()), int((f == 1).sum()), int((f == 2).sum())
        ttl = f'{ser} {obs} k{k} {"+" + lab if row in ypos else "random"}\na={na} b={nb} c={nc}, eroded kept {int(er.sum())}'
        ax.set_title(ttl, fontsize=9)
        ax.axis('off')
        srcf.unlink(missing_ok=True)
    fig.suptitle(f'{dom}: green a = mask v5 AND N2, red b = mask v5 only, blue c = N2 only; white dot = kept by the '
                 f'one-patch erosion of mask v5 (all 4 neighbours in mask v5)', fontsize=11)
    fig.tight_layout()
    FZ.mkdir(parents=True, exist_ok=True)
    fig.savefig(FZ / f'sheet_{dom}.png', dpi=110)
    log(f'sheet -> {FZ / f"sheet_{dom}.png"}')


# ---------------------------------------------------------------------------------------------- prep
def mode_of(arm):
    """arm key -> (feature 'S' | 'V', token-set mode). S / V = mask v5 (M); S<X> / V<X> = mask rule X (E, P250, ...)."""
    return arm[0], (arm[1:] or 'M')


def video_sel(z, mode):
    """bool over all rows of a stored video: rows in the token set `mode` at their own frame."""
    fi = np.repeat(np.arange(len(z['k'])), np.diff(z['ptr']))
    return mask_rows(mode, z['flag'], z['idx'].astype(np.int64), fi, z['prob'], int(z['G'])), fi


def plan_mode(dom, mode):
    """the window plan of v21_train.cmd_prep (rng 0) with the token rate of `mode` (mode M reproduces it exactly)."""
    rng = np.random.default_rng(0)
    vids = T.videos_of(dom)
    rates = []
    for v in vids[::max(1, len(vids) // 6)][:6]:
        z = np.load(T.STORE / dom / T.VER / 'videos' / v[0] / v[1] / 'index.npz')
        rates.append(video_sel(z, mode)[0].sum() / len(z['k']))
    mpf = float(np.mean(rates))
    plan = []
    for r in T.CAP:
        vs = [v for v in vids if v[2] == r]
        n_win = max(1, int(np.ceil(1.1 * T.CAP[r] / mpf / (T.WIN * max(1, len(vs))))))
        for ser, obs, role in vs:
            nf = len(T.Video(dom, ser, obs).k)
            starts = np.sort(rng.choice(max(1, nf - T.WIN), min(n_win, max(1, nf - T.WIN)), replace=False))
            plan += [(ser, obs, role, int(s_)) for s_ in starts]
    return plan, mpf


def feats_np(tok, fi, pos, sel_c):
    """numpy x and c_t = (x_{t+1} - x_{t-1}) / 2 (v21_train edge rule) for the centre rows sel_c of a window."""
    key = fi.astype(np.int64) * 65536 + pos
    s = pd.Series(np.arange(len(fi)), index=key)
    c = np.flatnonzero(sel_c)
    nb = {d: s.reindex(key[c] + d * 65536).fillna(-1).values.astype(np.int64) for d in (-1, 1)}
    x = tok[c].astype(np.float32)
    xp, xm = tok[np.maximum(nb[1], 0)].astype(np.float32), tok[np.maximum(nb[-1], 0)].astype(np.float32)
    mp, mm = (nb[1] >= 0)[:, None], (nb[-1] >= 0)[:, None]
    cd = np.where(mp & mm, (xp - xm) / 2, np.where(mp, xp - x, np.where(mm, x - xm, 0.0)))
    return x, cd


def _prep_video(args):
    dom, ser, obs, wins, mode, keep_off, paths = args
    d = T.STORE / dom / T.VER / 'videos' / ser / obs
    z = dict(np.load(d / 'index.npz'))
    tok = np.memmap(d / 'tok.f16', np.float16, 'r', shape=(int(z['ptr'][-1]), 1024))
    sel_all, fi_all = video_sel(z, mode)
    outs = {k: np.load(p, mmap_mode='r+') for k, p in paths.items()}
    n = len(z['k'])
    for wi, (role, s_) in wins:
        a0, a1 = max(0, s_ - 2), min(n, s_ + T.WIN + 2)
        r0, r1 = int(z['ptr'][a0]), int(z['ptr'][a1])
        fi = fi_all[r0:r1]
        selc = sel_all[r0:r1] & (fi >= s_) & (fi < min(n, s_ + T.WIN))
        kk, off = keep_off[wi]
        if not len(kk):
            continue
        x, cd = feats_np(np.asarray(tok[r0:r1]), fi, z['idx'][r0:r1].astype(np.int64), selc)
        outs[('S', role)][off:off + len(kk)] = x[kk].astype(np.float16)
        outs[('V', role)][off:off + len(kk)] = (x[kk] + cd[kk]).astype(np.float16)
    for o in outs.values():
        o.flush()
    return ser, obs


def cmd_prep(a):
    from multiprocessing import get_context
    dom, mode = a.dom, a.mask
    t0 = time.time()
    plan, mpf = plan_mode(dom, mode)
    cnt = np.zeros(len(plan), np.int64)
    byv = {}
    for wi, (ser, obs, role, s_) in enumerate(plan):
        byv.setdefault((ser, obs), []).append((wi, (role, s_)))
    for (ser, obs), L in byv.items():
        z = np.load(T.STORE / dom / T.VER / 'videos' / ser / obs / 'index.npz')
        sel, fi = video_sel(z, mode)
        per_f = np.bincount(fi[sel], minlength=len(z['k']))
        cs = np.r_[0, np.cumsum(per_f)]
        for wi, (role, s_) in L:
            cnt[wi] = cs[min(len(z['k']), s_ + T.WIN)] - cs[s_]
    roles = np.array([p[2] for p in plan])
    info = {'mode': mode, 'windows': len(plan), 'tokens_per_frame_est': mpf}
    rng2 = np.random.default_rng(1)
    keep_off, paths = {}, {}
    od = T.BEE / dom / 'data'
    for r in T.CAP:
        w = np.flatnonzero(roles == r)
        c = cnt[w]
        tot = int(c.sum())
        keep = np.sort(rng2.choice(tot, min(tot, T.CAP[r]), replace=False))
        info[r] = {'available': tot, 'used': len(keep)}
        start = np.r_[0, np.cumsum(c)]
        lo, hi = np.searchsorted(keep, start[:-1]), np.searchsorted(keep, start[1:])
        for j, wi in enumerate(w):
            keep_off[int(wi)] = (keep[lo[j]:hi[j]] - start[j], int(lo[j]))
        for f in ('S', 'V'):
            p = od / f'{r}_{f}{mode}.npy'
            np.lib.format.open_memmap(p, 'w+', np.float16, (len(keep), 1024)).flush()
            paths[(f, r)] = p
    log(f'{dom} {mode} keep plan ({time.time() - t0:.0f}s): {json.dumps(info)}')
    jobs = [(dom, ser, obs, L, mode, {wi: keep_off[wi] for wi, _ in L}, paths) for (ser, obs), L in byv.items()]
    with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        for j, _ in enumerate(pool.imap_unordered(_prep_video, jobs)):
            if j % 25 == 0:
                log(f'{dom} prep {j}/{len(jobs)} videos ({time.time() - t0:.0f}s)')
    FZ.mkdir(parents=True, exist_ok=True)
    (FZ / f'{dom}_prep_{mode}.json').write_text(json.dumps(info, indent=1))
    log(f'{dom} prep {mode} done ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- train
def cmd_train(a):
    if a.arm not in ('S', 'V'):
        T.ARMS2[a.arm] = 1024
    import torch
    T.cmd_train(a)
    f = T.OUT / f'{a.dom}_train_{a.arm}_s{a.seed}.json'
    rec = json.loads(f.read_text())
    rec['gpu'] = torch.cuda.get_device_name(0)
    rec['tf32_matmul'] = bool(torch.backends.cuda.matmul.allow_tf32)
    f.write_text(json.dumps(rec, indent=1, default=float))
    log(f'gpu {rec["gpu"]} recorded in {f.name}')


# ---------------------------------------------------------------------------------------------- codes
CODES = lambda dom: T.BEE / dom / 'codes_fz'  # noqa: E731


def agg_codes(cc, lf, nfr, dev):
    """token codes cc (n, W) with local frame index lf -> {agg: (nfr, W)} for max / mean / top10 / maxmean."""
    import torch
    W = cc.shape[1]
    out = {}
    out['max'] = torch.zeros((nfr, W), device=dev).index_reduce_(0, lf, cc, 'amax', include_self=True)
    cnt = torch.bincount(lf, minlength=nfr)
    cf = cnt.clamp(min=1).float()[:, None]
    out['mean'] = torch.zeros((nfr, W), device=dev).index_add_(0, lf, cc) / cf
    o_ = torch.argsort(lf, stable=True)
    lfs = lf[o_]
    st_ = torch.cumsum(cnt, 0) - cnt
    rk = torch.arange(len(lfs), device=dev) - st_[lfs]
    D = torch.zeros((nfr, max(1, int(cnt.max())), W), device=dev)
    D[lfs, rk] = cc[o_]
    out['top10'] = D.topk(min(10, D.shape[1]), dim=1).values.sum(1) / cf.clamp(max=10)
    del D
    out['maxmean'] = 0.5 * (out['max'] + out['mean'])
    return out


def cmd_codes(a):
    """frame codes of the eval spans (rows = the stored eval_rows.parquet) for arms x seeds under every readout;
    identity check (fail loud) vs the stored v21_train codes for the mask-v5 arms S / V."""
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    dom = a.dom
    arms = a.arms.split(',')
    modes = {mode_of(x)[1] for x in arms}
    assert len(modes) == 1
    mode = modes.pop()
    seeds = [int(s) for s in a.seeds.split(',')]
    saes = {f'{arm}_s{s}': (SA.load(T.BEE / dom / 'sae' / f'{arm}_s{s}' / 'sae_theta.pt', dev), mode_of(arm)[0])
            for arm in arms for s in seeds}
    Fs = T.frame_sets(dom)['eval']
    spans = [(ser, obs, int(g.k.min()), int(g.k.max()) + 1) for (ser, obs), g in Fs.groupby(['series', 'obs'], sort=False)]
    R = pd.DataFrame([(ser, obs, k) for ser, obs, k0, k1 in spans for k in range(k0, k1)], columns=['series', 'obs', 'k'])
    assert R.equals(pd.read_parquet(T.BEE / dom / 'codes' / 'eval_rows.parquet')), 'eval rows differ'
    Xc = {(k, g): np.zeros((len(R), 2048), np.float16) for k in saes for g in AGGS}
    ntok = 0
    t0 = time.time()
    base = 0
    for ser, obs, k0, k1 in spans:
        d = T.STORE / dom / T.VER / 'videos' / ser / obs
        z = dict(np.load(d / 'index.npz'))
        tok = np.memmap(d / 'tok.f16', np.float16, 'r', shape=(int(z['ptr'][-1]), 1024))
        sel_all, fi_all = video_sel(z, mode)
        kpos = {int(k): i for i, k in enumerate(z['k'])}
        i0, i1 = kpos[k0], kpos[k1 - 1] + 1
        n = len(z['k'])
        for b0 in range(i0, i1, 600):
            b1 = min(i1, b0 + 600)
            a0, a1 = max(0, b0 - 2), min(n, b1 + 2)
            r0, r1 = int(z['ptr'][a0]), int(z['ptr'][a1])
            fi = fi_all[r0:r1]
            selc = sel_all[r0:r1] & (fi >= b0) & (fi < b1)
            if not selc.any():
                continue
            ntok += int(selc.sum())
            x, cd = feats_np(np.asarray(tok[r0:r1]), fi, z['idx'][r0:r1].astype(np.int64), selc)
            xt = {'S': torch.from_numpy(x).to(dev)}
            xt['V'] = xt['S'] + torch.from_numpy(cd.astype(np.float32)).to(dev)
            lf = torch.from_numpy(fi[selc] - b0).to(dev)
            o = base + b0 - i0
            for k, ((sae, norm), f) in saes.items():
                with torch.no_grad():
                    cc = sae.encode(norm(xt[f]), mode='threshold').float()
                    ag = agg_codes(cc, lf, b1 - b0, dev)
                for g, v in ag.items():
                    Xc[(k, g)][o:o + b1 - b0] = v.half().cpu().numpy()
        base += i1 - i0
    od = CODES(dom)
    od.mkdir(parents=True, exist_ok=True)
    chk = {}
    for (k, g), x in Xc.items():
        np.save(od / f'{k}_{g}.npy', x)
        if mode == 'M' and g != 'maxmean':
            ref = T.BEE / dom / 'codes' / (f'{k}_eval.npy' if g == 'max' else f'{k}_eval_{g}.npy')
            if ref.exists():
                r_ = np.load(ref).astype(np.float32)
                dd = float(np.abs(r_ - x.astype(np.float32)).max())
                # tolerance: 1e-2 absolute or 2 fp16 ulps (2^-9 relative), whichever is larger (fp32 summation order)
                ex = float((np.abs(r_ - x.astype(np.float32)) - np.maximum(1e-2, np.abs(r_) * 2 ** -9)).max())
                chk[f'{k}_{g}'] = dd
                if ex > 0:
                    raise SystemExit(f'IDENTITY CHECK FAILED {k}_{g}: max abs diff {dd}')
    info = {'mode': mode, 'tokens_per_eval_frame': ntok / len(R), 'identity_max_abs_diff': chk, 'wall_s': time.time() - t0}
    FZ.mkdir(parents=True, exist_ok=True)
    (FZ / f'{dom}_codes_{"_".join(arms)}_{"_".join(map(str, seeds))}.json').write_text(json.dumps(info, indent=1))
    log(f'{dom} codes {arms} {seeds}: {json.dumps(info)}')


def v1_path(dom, key, agg):
    return T.BEE.parent / 'v2_1' / dom / 'codes_fz' / f'{key}_{agg}.npy'


def cmd_v1codes(a):
    """v1 s0-10 frame codes under every readout on v1's OWN tokens of the eval frames (mice: the v2_prob probe store,
    probe order; ants: v1's exact token pipeline of eval.py codes_ants, frames.parquet order); identity check of the
    max readout vs the stored codes eval.py scores (fail loud)."""
    import torch
    from src.eci.sae import load_sae
    import eval as EV
    dev = torch.device('cuda')
    dom = a.dom
    saes = {'v1_s0': load_sae(REPO / EV.V1SAE[dom], dev)[:2]}
    for s in range(1, 11):
        saes[f'v1_s{s}'] = load_sae(EV.EC / f'v2_diag/v1retrain/{dom}/V1R_full_s{s}/sae.pt', dev)[:2]
    t0 = time.time()
    if dom == 'mice':
        sys.path.insert(0, str(REPO / 'scripts/eci/v2_prob'))
        import study as SY
        Pr = pd.read_parquet(SY.OUT / dom / 'probe.parquet')
        key = lambda F_: pd.MultiIndex.from_arrays([F_.vid.values.astype(int), F_.k.values.astype(int)])  # noqa: E731
        pidx = pd.Series(np.arange(len(Pr)), index=key(Pr))
        St = SY.BH.Store(dom, 'v1', 'dense', 768)
        fidx = pidx.reindex(key(St.F)).values
        assert not np.isnan(fidx).any() and len(np.unique(fidx)) == len(Pr)
        fidx = fidx.astype(np.int64)
        kp = St.keep('v1')
        X = {(n_, g): np.zeros((len(Pr), 1024), np.float16) for n_ in saes for g in AGGS}
        for f0 in range(0, St.nf, 800):
            f1 = min(St.nf, f0 + 800)
            a_, b_ = int(St.starts[f0]), int(St.starts[f1])
            if a_ == b_:
                continue
            zt = torch.from_numpy(St.tok(a_, b_)).to(dev)
            sel = torch.from_numpy(kp[a_:b_]).to(dev)
            lf = torch.from_numpy(St.tf[a_:b_] - f0).to(dev)[sel]
            rows = fidx[f0:f1]
            for n_, (sae, norm) in saes.items():
                with torch.no_grad():
                    c = sae.encode(norm(zt[sel].float()), mode='threshold').float()
                    ag = agg_codes(c, lf, f1 - f0, dev)
                for g, v in ag.items():
                    X[(n_, g)][rows] = v.half().cpu().numpy()
        ref_of = lambda n_: np.load(EV.EC / 'v2_prob/mice/codes' / f'{n_}.npy')  # noqa: E731
    else:
        sys.path.insert(0, str(REPO / 'scripts/eci/v2_diag'))
        from src.eci.foreground import RULES, FgBackgrounds, FgEncoder, encode_batch, obs_rows
        F = pd.read_parquet(EV.BEE / 'ants' / 'frames.parquet')
        enc = FgEncoder('dinov2_base', dev)
        X = {(n_, g): np.zeros((len(F), 1024), np.float16) for n_ in saes for g in AGGS}
        for src, g_ in F.groupby('src'):
            ann = REPO / EV.ANN[src]
            fp = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
            fb = FgBackgrounds(REPO / 'dataset/ants/eci30/fg448/background', obs_rows(ann), RULES['ants'], dev,
                               align='none', cache=8)
            rows = g_.row.values.astype(np.int64)
            ds = enc.dataset([str(REPO / 'dataset' / p) for p in fp[rows]], rows, None)
            dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, pin_memory=True, prefetch_factor=4,
                                             num_workers=max(2, int(os.environ.get('SLURM_CPUS_PER_TASK', 8)) - 2))
            gi, o = g_.index.values, 0
            for pix, grey, rr in dl:
                with torch.no_grad():
                    tok = encode_batch(enc.model, pix, dev)
                    mk, _ = fb.mask(tok, grey.to(dev), rr.numpy())
                    fi, pi = torch.nonzero(mk, as_tuple=True)
                    x = tok[fi, pi]
                    for n_, (sae, norm) in saes.items():
                        zc = sae.encode(norm(x), mode='threshold').float()
                        ag = agg_codes(zc, fi, len(tok), dev)
                        for g, v in ag.items():
                            X[(n_, g)][gi[o:o + len(tok)]] = v.half().cpu().numpy()
                o += len(tok)
            log(f'v1 ants {src} done ({time.time() - t0:.0f}s)')
        ref_of = lambda n_: np.load(EV.BEE / 'ants/codes' / f'{n_}.npy')  # noqa: E731
    chk = {}
    for (n_, g), x in X.items():
        p = v1_path(dom, n_, g)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, x)
        if g == 'max':
            r = ref_of(n_).astype(np.float32)
            chk[n_] = {'max_abs_diff': float(np.abs(r - x.astype(np.float32)).max()),
                       'fire_agree': float(((r > 0) == (x > 0)).mean())}
    log(f'v1 {dom} identity (max readout vs stored codes): {json.dumps(chk)}')
    FZ.mkdir(parents=True, exist_ok=True)
    (FZ / f'{dom}_v1codes.json').write_text(json.dumps({'identity_max': chk, 'wall_s': time.time() - t0}, indent=1))
    bad = {k: v for k, v in chk.items() if v['fire_agree'] < 0.999}
    if bad:
        raise SystemExit(f'v1 IDENTITY CHECK FAILED: {bad}')


# ---------------------------------------------------------------------------------------------- score
def ladder_mod():
    import v21_ladder as L
    L.LO = FZ / 'ladder'
    L.LB = LBF
    return L


def readout_fz(dom, key, agg, F):
    if key.startswith('v1_'):
        X = np.load(v1_path(dom, key, agg))
        X = X[F.code_row.values] if dom == 'mice' else X
        assert len(X) == len(F)
        return np.asarray(X, np.float32)
    from multiprocessing import get_context
    from src.eci.v2 import hmm as HM
    import v20 as V20
    od = T.BEE / dom / 'codes'
    R = pd.read_parquet(od / 'eval_rows.parquet')
    E = T.frame_sets(dom)['eval']
    rix = pd.Series(np.arange(len(R)), index=pd.MultiIndex.from_frame(R)).reindex(pd.MultiIndex.from_frame(E)).values
    assert not np.isnan(rix).any()
    rix = rix.astype(np.int64)
    st = np.r_[0, np.flatnonzero((R.obs.values[1:] != R.obs.values[:-1]) | (R.series.values[1:] != R.series.values[:-1])) + 1, len(R)]
    raw = np.load(CODES(dom) / f'{key}_{agg}.npy').astype(np.float32)
    Xt = np.ascontiguousarray(raw.T)
    V20._G.update(Xc=Xt, sc=st.astype(np.int64))
    with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        prm = [None] * Xt.shape[0]
        for j, p in pool.imap_unordered(V20._fit_one, range(Xt.shape[0]), chunksize=16):
            prm[j] = p
    pon = np.zeros_like(raw)
    for j in range(Xt.shape[0]):
        pon[:, j] = HM.apply(Xt[j], st, prm[j])[0]
    return pon[rix]


def cmd_score(a):
    """readout (P(on) HMM fitted once on the scoring spans for v2; raw for v1) -> v21_ladder.score2 on the gate cells
    (+ ants focal) -> ladder json / npz; identity check vs the existing ladder json where one exists (fail loud)."""
    if ',' in a.agg:
        import copy
        for g in a.agg.split(','):
            b = copy.copy(a)
            b.agg = g
            cmd_score(b)
        return
    import v20_gate as G
    L = ladder_mod()
    dom, key, agg = a.dom, a.model, a.agg
    tag = f'{key}_{agg}'
    if (FZ / 'ladder' / dom / f'{tag}.json').exists():
        log(f'{tag} exists, skipped')
        return
    t0 = time.time()
    F, SP = L.frames(dom)
    X = readout_fz(dom, key, agg, F)
    use = np.ones(len(F), bool)
    sc = {'gate': L.score2(F, X, use, SP)}
    if dom == 'ants':
        sc['focal'] = L.score2(L.focal_frame(F), X, use, SP)
    chk = {'n': 0, 'max_abs_diff': 0.0}
    old = {'max': T.OUT / 'ladder' / dom / f'{key}.json', 'top10': T.OUT / 'ladder' / dom / f'{key}_top10.json'}.get(agg)
    if old is not None and old.exists() and '_s' in key and (key.startswith('v1_') or mode_of(key.split('_s')[0])[1] == 'M'):
        J = json.loads(old.read_text())
        for part in sc:
            for c, r in sc[part].items():
                if c not in J.get(part, {}) or c.split('|')[1] in G.ONLID:
                    continue
                for M in ('AUROC', 'nAP', 'top1_prec'):
                    x, y = r.get(M), J[part][c].get(M)
                    if x is None or y is None:
                        continue
                    chk['n'] += 1
                    chk['max_abs_diff'] = max(chk['max_abs_diff'], abs(x - y))
        # ants v1: the v1 tokens are re-encoded (DINOv2 + fg mask, GPU nondeterminism: firing agreement 0.9997 vs the
        # stored codes, freeze/ants_v1codes.json), so a small metric drift is tolerated and recorded; v2: exact
        tol = 1e-2 if (dom == 'ants' and key.startswith('v1_')) else 1e-6
        chk['tolerance'] = tol
        if chk['max_abs_diff'] > tol:
            raise SystemExit(f'IDENTITY CHECK FAILED {tag}: {chk}')
    else:
        chk['note'] = 'no earlier ladder json'
    js = sorted({p[2] for part in sc.values() for r in part.values() for p in r['AUROC_neurons'] + r['nAP_neurons']})
    (LBF / dom).mkdir(parents=True, exist_ok=True)
    np.savez(LBF / dom / f'{tag}.npz', cols=X[:, js].astype(np.float32), neurons=np.array(js, np.int64))
    keep = ('n_frames', 'base_rate', 'n_dirs', 'AUROC', 'nAP', 'F1max', 'Rt', 'AUROC_neurons', 'nAP_neurons', 'top1_prec',
            'top1_lift', 'top1_base', 'best_neuron', 'AUROC_dirs')
    out = {'key': key, 'agg': agg, 'identity': chk,
           **{part: {c: {k: v for k, v in r.items() if k in keep} for c, r in d.items()} for part, d in sc.items()}}
    (FZ / 'ladder' / dom).mkdir(parents=True, exist_ok=True)
    (FZ / 'ladder' / dom / f'{tag}.json').write_text(json.dumps(out, default=float))
    log(f'{dom} {tag}: identity {chk}; {len(js)} picked neurons ({time.time() - t0:.0f}s)')


def cmd_boot(a):
    L = ladder_mod()
    L.cmd_boot(a)


# ---------------------------------------------------------------------------------------------- report
MASKS = ('M', 'E', 'P253')
ARMK = lambda f, m: f if m == 'M' else f'{f}{m}'  # noqa: E731
S1, S2 = (1004, 1005, 1006), (1007, 1008, 1009)
DOMS = ('mice', 'ants')


def _groups():
    g = {}
    for agg in AGGS:
        g[f'v1|{agg}'] = [f'v1_s{i}_{agg}' for i in range(11)]
        for m in MASKS:
            for f in ('S', 'V'):
                g[f'{f}@{m}|{agg}'] = [f'{ARMK(f, m)}_s{s}_{agg}' for s in S1]
            g[f'V@{m}|{agg}|s2'] = [f'V{"" if m == "M" else m}_s{s}_{agg}' for s in S2]
    return g


def cmd_report(a):
    import v20_gate as G
    L = ladder_mod()
    NB = L.NB
    out, md = {'domains': {}}, ['# v2.1 freeze: mask x readout (P(on) for v2, raw pooled code for v1; 95% CI = '
                                'pools/videos + seeds resampled, 2000 replicates)', '']
    rng = np.random.default_rng(L.SEED + 11)
    D = {}
    for dom in DOMS:
        bp = LBF / f'boot_{dom}.npz'
        if not bp.exists():
            continue
        z = np.load(bp)
        S0 = json.loads((G.OUT / f'{dom}_scores.json').read_text())
        gcells = [c for c, r in S0['eligibility'].items() if r['eligible'] and c.split('|')[1] not in G.ONLID]
        fcells = {}
        if dom == 'ants':
            fel = L.focal_elig()
            fcells = {'single': [c for c, e in fel.items() if e and c.endswith('|single')],
                      'double': [c for c, e in fel.items() if e and c.endswith('|double')],
                      'or': [c for c in gcells if c.endswith('|groom_any')]}
        dm, lv = {}, {}
        for gname, tags in _groups().items():
            tags = [t for t in tags if all(f'{t}|gate|{c}' in z for c in gcells)]
            if not tags:
                continue
            dm[gname] = np.stack([np.mean([z[f'{t}|gate|{c}'] for c in gcells], 0) for t in tags])
            R = {'n': len(tags)}
            for i, M in enumerate(L.MN):
                R[M] = dm[gname][:, i, 0].tolist()
            for f, cs in fcells.items():
                if cs:
                    part = 'gate' if f == 'or' else 'focal'
                    R[f'focal_{f}_AUROC'] = [float(np.mean([z[f'{t}|{part}|{c}'][0, 0] for c in cs])) for t in tags]
            if not gname.startswith('v1'):
                agg = gname.split('|')[1]
                v1f = {}
                for i in range(11):
                    jf = FZ / 'ladder' / dom / f'v1_s{i}_{agg}.json'
                    if jf.exists():
                        v1f[str(i)] = json.loads(jf.read_text())['gate']
                pa, pf = [], []
                for t in tags:
                    J = json.loads((FZ / 'ladder' / dom / f'{t}.json').read_text())['gate']
                    pa.append(sum(r['PASS_final'] for c, r in G.gate_rows(dom, S0['v1'], J, S0['eligibility']).items() if c in gcells))
                    if v1f:
                        pf.append(sum(r['PASS_final'] for c, r in G.gate_rows(dom, v1f, J, S0['eligibility']).items() if c in gcells))
                R['gate_pass_am5'] = pa
                R['gate_pass_v1_same_readout'] = pf
            lv[gname] = R
        D[dom] = dm
        out['domains'][dom] = {'gate_cells': gcells, 'focal_cells': fcells, 'levels': lv, 'contrasts': {}}

    def con(dom, x, y):
        dm = D.get(dom, {})
        if x not in dm or y not in dm:
            return None
        r = {}
        for mi, M in enumerate(L.MN):
            ax, ay = dm[x][:, mi, 1:], dm[y][:, mi, 1:]
            ix = rng.integers(0, len(ax), (NB, len(ax)))
            iy = rng.integers(0, len(ay), (NB, len(ay)))
            rr = np.arange(NB)[:, None]
            d = np.nanmean(ax[ix, rr], 1) - np.nanmean(ay[iy, rr], 1)
            r[M] = {'diff': float(dm[x][:, mi, 0].mean() - dm[y][:, mi, 0].mean()),
                    'ci': np.nanpercentile(d, [2.5, 97.5]).tolist()}
        out['domains'][dom]['contrasts'][f'{x} - {y}'] = r
        return r
    eff = lambda r: r is not None and r['AUROC']['ci'][0] > 0  # noqa: E731
    noninf = lambda r: r is not None and r['AUROC']['ci'][0] >= -0.01  # noqa: E731
    med = lambda dom, g: float(np.median(out['domains'][dom]['levels'][g]['AUROC']))  # noqa: E731
    dec = {}
    # 1 mask
    qual = []
    for m in ('E', 'P253'):
        cm, ca = con('mice', f'V@{m}|max', 'V@M|max'), con('ants', f'V@{m}|max', 'V@M|max')
        if eff(cm) and noninf(ca):
            qual.append(m)
    for m in ('E', 'P253'):
        for dom in DOMS:
            con(dom, f'S@{m}|max', 'S@M|max')
    mask = max(qual, key=lambda m: np.mean([med(d, f'V@{m}|max') for d in DOMS])) if qual else 'M'
    dec['mask'] = {'qualifying': qual, 'chosen': mask}
    # 2 readout
    qual, why = [], {}
    for r_ in ('mean', 'top10', 'maxmean'):
        cs = {dom: con(dom, f'V@{mask}|{r_}', f'V@{mask}|max') for dom in DOMS}
        e = [d for d in DOMS if eff(cs[d])]
        if e and all(eff(cs[d]) or noninf(cs[d]) for d in DOMS):
            qual.append(r_)
            why[r_] = e
    ro = max(qual, key=lambda r_: np.mean([med(d, f'V@{mask}|{r_}') for d in DOMS])) if qual else 'max'
    dec['readout'] = {'qualifying': qual, 'chosen': ro}
    rec = f'V@{mask}|{ro}'
    dec['recipe'] = rec
    eff_doms = set(why.get(ro, [])) | ({'mice'} if mask != 'M' else set())
    dec['stage1_effect_domains'] = sorted(eff_doms)
    # contrasts reported for every readout and arm
    for dom in DOMS:
        for agg in AGGS:
            for m in MASKS:
                for f in ('S', 'V'):
                    con(dom, f'{f}@{m}|{agg}', f'v1|{agg}')
            for m in ('E', 'P253'):
                con(dom, f'V@{m}|{agg}', f'V@M|{agg}')
        for m in ('E', 'P253'):
            con(dom, f'S@{m}|max', 'S@M|max')
        for agg in ('mean', 'top10', 'maxmean'):
            for m in MASKS:
                con(dom, f'V@{m}|{agg}', f'V@{m}|max')
    # confirmation
    if rec != 'V@M|max':
        c2 = {dom: con(dom, f'{rec}|s2', 'V@M|max|s2') for dom in DOMS}
        if any(v is None for v in c2.values()):
            dec['confirmation'] = 'PENDING (stage-2 scores missing)'
        else:
            ok = all(eff(c2[d]) if d in eff_doms else noninf(c2[d]) for d in DOMS)
            dec['confirmation'] = 'CONFIRMED' if ok else 'NOT CONFIRMED -> freeze V@M|max'
            if not ok:
                dec['frozen'] = 'V@M|max'
    else:
        dec['confirmation'] = 'not needed (recipe = V@M|max)'
    dec.setdefault('frozen', rec)
    fr = dec['frozen']
    agg = fr.split('|')[1]
    for dom in DOMS:
        con(dom, fr, f'v1|{agg}')
        con(dom, f'{fr}|s2', f'v1|{agg}')
    vs = {}
    for tag, g in (('s1', fr), ('s2', f'{fr}|s2')):
        cm = out['domains'].get('mice', {}).get('contrasts', {}).get(f'{g} - v1|{agg}')
        ca = out['domains'].get('ants', {}).get('contrasts', {}).get(f'{g} - v1|{agg}')
        if cm and ca:
            vs[tag] = {'C3_mice_effect': eff(cm), 'C4_ants_noninferior': noninf(ca)}
    dec['vs_v1'] = vs
    out['decision'] = dec
    # markdown
    f3 = lambda v: '-' if not v else (f'{np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]' if len(v) > 1 else f'{v[0]:.3f}')  # noqa
    for dom in DOMS:
        if dom not in out['domains']:
            continue
        O = out['domains'][dom]
        md += [f'## {dom} ({len(O["gate_cells"])} gate cells)', '',
               '| group (arm@mask, readout) | n | mean AUROC | top-1% prec | nAP | gate passes AM5 (v1 max bar) | gate passes (v1 same readout)'
               + (' | focal single / double / or AUROC |' if dom == 'ants' else ' |'), '|' + '---|' * (8 if dom == 'ants' else 7)]
        for gname, R in O['levels'].items():
            tail = (f' | {f3(R.get("focal_single_AUROC"))} / {f3(R.get("focal_double_AUROC"))} / {f3(R.get("focal_or_AUROC"))} |'
                    if dom == 'ants' else ' |')
            md.append(f'| {gname} | {R["n"]} | {f3(R["AUROC"])} | {f3(R["top1_prec"])} | {f3(R["nAP"])} | '
                      f'{R.get("gate_pass_am5", "-")} | {R.get("gate_pass_v1_same_readout", "-")}' + tail)
        md += ['', '| contrast | d mean AUROC [95% CI] | d top-1% prec | d nAP |', '|---|---|---|---|']
        for k, r in O['contrasts'].items():
            md.append(f'| {k} | ' + ' | '.join(f'{r[M]["diff"]:+.3f} [{r[M]["ci"][0]:+.3f}, {r[M]["ci"][1]:+.3f}]'
                                               for M in ('AUROC', 'top1_prec', 'nAP')) + ' |')
        md.append('')
    md += ['## Decision (pre-registered rules, scripts/eci/v2/v21_freeze_prereg.md)', '', '```', json.dumps(dec, indent=1), '```']
    (FZ / 'report.json').write_text(json.dumps(out, indent=1, default=float))
    (FZ / 'report.md').write_text('\n'.join(md))
    print('\n'.join(md))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step')
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--every', type=int, default=5)
    p.add_argument('--mask', default='E')
    p.add_argument('--arm', default='')
    p.add_argument('--arms', default='S,V')
    p.add_argument('--seed', type=int, default=1004)
    p.add_argument('--seeds', default='1004,1005,1006')
    p.add_argument('--model', default='')
    p.add_argument('--agg', default='max')
    a = p.parse_args()
    globals()[f'cmd_{a.step}'](a)


if __name__ == '__main__':
    main()
