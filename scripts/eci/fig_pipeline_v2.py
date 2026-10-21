"""
Pipeline figure of the FROZEN ECI v2 representation (configs/eci_v2/representation.json), one row per domain (ants,
mice, frogs), the same steps on real data of that domain. Derived from scripts/eci/fig_pipeline.py v16 (same sizes,
fonts, grey rules, labels under the panels, dims labels); the v1 script is left untouched.

    a  video              temporal volume: 9 canonical frames evenly over the video (the encoder input: native side
                          rounded to a multiple of 16, never upsampled; mice rot90 odour corner top right; frogs hflip
                          mutant side right), the example frame t outlined; line above: source vs used size, 5 fps
    b  frame repr.        (1) the frame split into its G x G tokens (16 px patches) + the tight-mask outline;
                          (2) zoom, tokens outside the tight mask blanked (kept iff >= 25% animal pixels), the chosen
                          token (the neuron's peak) outlined; (3) its DINOv3-L/16 block-20 token z_t (1024-d glyph);
                          (4) the domain SAE's dense ReLU pre-activations (2048, Matryoshka prefixes 128 / 512 marked)
                          with theta* -> sparse code, the shown neuron in pink
    c  neural activation  the neuron on every kept token -> max over kept tokens = frame value c_j(t) -> 10 s of real
                          frame values: raw c_j (grey) and the smoothed s_j(t) = mean(c_j(t-1), c_j(t), c_j(t+1))
                          (pink, 0.6 s at 5 fps), the three frames of s_j(t) highlighted; fnorm firing threshold
                          theta_n sigma_j dashed; below: firing frames (s_j / sigma_j > theta_n) over the whole video
    d  video repr.        time spent / bout rate / latency of ALL 2048 neurons in that video (summaries_<series>.npz,
                          src/eci/v2/readout.py), the shown neuron in pink

Example per domain = the peak frame (heat asset peak_rows[0]: the top video's peak of the smoothed code) of
    mice  #161 of v2_mice_c0_w2048_s7   (aligned to nose-nose, nn_mutual, np_directional; results/vision/eci_v2/mice/align.json)
    ants  #99  of v2_ants_c0_w2048_s6   (aligned to groom_any / groom_yellow, ants v2)
    frogs #52  of v2_frogs_c0_w2048_s6  (NES primary FoxP1 > WT, all prefixes; juveniles)

Stages
    compute  (GPU, scripts/eci/fig_pipeline_v2.sh) per domain: decode + canonical + tight mask with the v2 library
             (src/eci/v2/pipeline.py Series), VERIFY the kept tokens against the full-pass token store, DINOv3-L on
             frame t, SAE on the kept tokens of t VERIFIED against the stored sparse token codes; the stored
             token codes give the activation map (exact); whole-video raw frame codes -> smooth -> fnorm firing ->
             summaries VERIFIED against summaries_<series>.npz and the stored smoothed codes -> OUT/cache/<d>.npz/.json
    plot     (light, reads only the cache) -> OUT/fig_pipeline_v2.png (full res) + fig_pipeline_v2_preview.png (1600 px)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
V2 = REPO / 'results/vision/eci_v2'
OUT = V2 / 'fig'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2')
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731
FPS = 5.0
N_THUMB = 9
WIN = 50  # frames of the smoothing panel (10 s at 5 fps)

DOM = {
    'ants': dict(series='v2', key=99, name='Ants', species=r'$\it{Lasius\ neglectus}$', short='grooming',
                 why='aligned to grooming (groom_any / groom_yellow, ants v2 align.json)', canon=''),
    'mice': dict(series='v1', key=161, name='Mice', species=r'$\it{Mus\ musculus}$', short='nose-nose contact',
                 why='aligned to nose_nose / nn_mutual / np_directional (mice v1 align.json)',
                 canon='odour corner top right'),
    'frogs': dict(series='juv', key=52, name='Frogs', species=r'$\it{Xenopus}$ froglets', short='FoxP1 > WT (NES)',
                  why='NES primary discovery FoxP1 > WT (prefixes 128 / 512 / 2048, juveniles)',
                  canon='mutant side right'),
}


# ------------------------------------------------------------------------------------------------ compute
class TokenCodes:
    """Sparse per-token codes of one video (full-pass npz, scripts/eci/v2/tools.py TokenCodes)."""

    def __init__(self, f):
        z = np.load(f)
        self.G = int(z['G'])
        self.nk = z['n_kept'].astype(np.int64)
        self.tok0 = np.r_[0, np.cumsum(self.nk)]
        self.pos = z['tok_pos'].astype(np.int64)
        self.act0 = np.r_[0, np.cumsum(z['act_count'].astype(np.int64))]
        self.idx, self.val = z['act_idx'].astype(np.int64), z['act_val'].astype(np.float32)
        self.cm = z['codes_max']

    def kept(self, k):
        return self.pos[self.tok0[k]:self.tok0[k + 1]]

    def dense(self, k, W):
        """(n_kept, W) stored token codes of frame k (rows in kept-token order)."""
        t0, t1 = self.tok0[k], self.tok0[k + 1]
        C = np.zeros((t1 - t0, W), np.float32)
        for i, t in enumerate(range(t0, t1)):
            a0, a1 = self.act0[t], self.act0[t + 1]
            C[i, self.idx[a0:a1]] = self.val[a0:a1]
        return C

    def heat(self, k, j):
        h = np.zeros(self.G * self.G, np.float32)
        h[self.kept(k)] = self.dense(k, max(j + 1, int(self.idx.max()) + 1))[:, j]
        return h


def compute_domain(name, dev):
    import torch
    from PIL import Image
    from src.eci.v2 import readout as RO
    from src.eci.v2.mask import THR
    from src.eci.v2.pipeline import Series
    from src.eci.v2.sae import load as load_sae
    t0 = time.time()
    cfg = DOM[name]
    man = json.loads((V2 / 'manifest.json').read_text())['domains'][name]
    dcfg = json.loads((REPO / f'configs/eci_v2/{name}.json').read_text())
    sc = next(s for s in dcfg['series'] if s['name'] == cfg['series'])
    msr = man['series'][cfg['series']]
    j = cfg['key']
    hz = np.load(Path(msr['heat_dir']) / f'n{j:04d}.npz')
    row = int(hz['peak_rows'][0])
    Se = Series(sc, dev)
    lay = np.array([v['lay_lo'] for v in Se.videos])
    v = Se.videos[int(np.searchsorted(lay, row, side='right') - 1)]
    obs, n, k = v['obs'], int(v['n']), row - v['lay_lo']
    assert 1 <= k < n - 1, (obs, k, n)
    log(f'[{name}] neuron {j}: example {obs} frame {k} / {n} (layout row {row}, heat peak_rows[0])')
    # ---------------------------------------------------------------- frames t-1, t, t+1: canonical + tight mask
    ks = np.arange(k - 1, k + 2)
    fr = Se.run_job(**Se.job(v, k - 1, 3))
    x, fd = Se.frames(v, fr, ks)
    kept = (fd >= THR).cpu().numpy()
    px = Se.mask.pixels(x, v['src_lo'] + ks).cpu().numpy()  # animal pixels (display)
    T = TokenCodes(BEE / name / 'videos' / sc['name'] / f'{obs}.npz')
    G = T.G
    assert G == Se.G, (G, Se.G)
    for i, kk in enumerate(ks):
        a, b = np.sort(T.kept(kk)), np.flatnonzero(kept[i])
        if not np.array_equal(a, b):
            raise RuntimeError(f'[{name}] kept tokens of frame {kk} differ from the token store: {len(a)} vs {len(b)}, '
                               f'sym diff {len(np.setxor1d(a, b))}')
    # ---------------------------------------------------------------- encoder + SAE on frame t, verified
    z = Se.enc(x[1:2])  # (1, G*G, 1024) fp16: frame t only
    sae, norm = load_sae(man['sae_checkpoint'], dev)
    W = int(sae.n_latents)
    pos_t = T.kept(k)
    with torch.no_grad():
        zc = z[0][torch.from_numpy(pos_t).to(z.device)].float()
        C = sae.encode(norm(zc), mode='threshold').float().cpu().numpy()
    Cs = T.dense(k, W)
    diff = np.abs(C - Cs)
    flip = (C > 0) != (Cs > 0)
    flips = int(flip.sum())
    theta0 = float(sae.threshold)
    # an entry whose pre-activation sits near theta* can switch on / off between GPUs (bf16 kernels differ; the full
    # pass ran on L40S / A100): such flips are counted and must stay near theta*, never on the shown neuron's peak
    flip_far = int((flip & (np.maximum(C, Cs) > 1.5 * theta0)).sum())
    cm_re = C.max(0)
    verify = dict(n_kept=int(len(pos_t)), token_code_max_abs_diff=float(diff.max()),
                  token_code_active_flips=flips, flips_above_1p5_theta=flip_far,
                  max_abs_diff_non_flip=float(diff[~flip].max()), n_token_entries=int((Cs > 0).sum()),
                  frame_code_max_abs_diff=float(np.abs(cm_re - T.cm[k].astype(np.float32)).max()),
                  neuron_frame_value_recomputed=float(cm_re[j]), neuron_frame_value_stored=float(T.cm[k, j]))
    log(f'[{name}] verify encoder + SAE vs token store: {verify}')
    if abs(verify['neuron_frame_value_recomputed'] - verify['neuron_frame_value_stored']) > 0.05 * max(1.0, float(T.cm[k, j])) \
            or flip_far > 0 or verify['max_abs_diff_non_flip'] > 0.05 * max(1.0, float(Cs.max())):
        raise RuntimeError(f'[{name}] recomputed codes do not match the deployed token store: {verify}')
    heat = T.heat(k, j)  # stored (exact) activation map of the neuron
    assert abs(heat.max() - float(T.cm[k, j])) < 1e-3
    heat_pm = [T.heat(kk, j) for kk in ks]
    sel = int(np.argmax(heat))
    with torch.no_grad():
        pre = torch.relu(sae.pre_acts(norm(z[0][sel:sel + 1].float())))[0].float().cpu().numpy()
    theta = float(sae.threshold)
    tok_code = pre * (pre > theta)
    i_sel = int(np.flatnonzero(pos_t == sel)[0])
    verify['sel_token_code_max_abs_diff_vs_store'] = float(np.abs(tok_code - Cs[i_sel]).max())
    tok_z = z[0][sel].float().cpu().numpy()
    # ---------------------------------------------------------------- whole video: raw -> smooth -> fnorm -> summaries
    sm = np.load(Path(msr['summaries']))
    vi = list(sm['obs']).index(obs)
    assert int(sm['n_frames'][vi]) == n, (sm['n_frames'][vi], n)
    sig, th_n = sm['sigma'].astype(np.float32), float(sm['theta_n'])
    raw = T.cm.astype(np.float32)
    S = RO.smooth(raw, 1)
    F = RO.fires(S, sig, th_n)
    su = RO.video_summaries(F, FPS)
    chk = dict(time_spent=float(np.abs(su['time_spent'] - sm['time_spent'][vi]).max()),
               n_bouts=float(np.abs(su['n_bouts'] - sm['n_bouts'][vi]).max()),
               latency_s=float(np.abs(su['latency_s'] - sm['latency_s'][vi]).max()),
               mean_smoothed=float(np.abs(S.mean(0) - sm['mean_smoothed'][vi]).max()))
    st = np.load(Path(msr['sets']['primary']['codes']) / 'codes_max.npy', mmap_mode='r')
    stored_s = np.asarray(st[row - k:row - k + n, j], dtype=np.float32)
    chk['smoothed_vs_primary_codes'] = float(np.abs(stored_s - S[:, j]).max())
    log(f'[{name}] verify summaries (max abs diff over 2048 neurons) + primary smoothed codes: {chk}')
    if chk['time_spent'] > 1e-6 or chk['n_bouts'] > 0 or chk['latency_s'] > 1e-4 or chk['smoothed_vs_primary_codes'] > 0.02:
        raise RuntimeError(f'[{name}] whole-video summaries do not match the stored v2 outputs: {chk}')
    # ---------------------------------------------------------------- thumbnails over the video (canonical frames)
    tpos = [int((i + 0.5) * n / N_THUMB) for i in range(N_THUMB)]
    j_ex = int(np.argmin([abs(q - k) for q in tpos]))
    tpos[j_ex] = k
    thumbs = []
    for q in tpos:
        xq, _ = Se.frames(v, Se.run_job(**Se.job(v, q, 1)), np.array([q]))
        im = Image.fromarray(xq[0].permute(1, 2, 0).cpu().numpy())
        thumbs.append(np.asarray(im.resize((384, 384), Image.LANCZOS)))
    lo = int(np.clip(k - WIN // 2, 0, n - WIN))
    meta = dict(domain=name, sae=man['sae']['name'], sae_seed=man['sae']['seed'], key=j, why=cfg['why'],
                obs=obs, frame=int(k), row=row, video_frames=n, video_min=n / FPS / 60, path=v['path'],
                probe=v['probe'], hflip=int(v['hflip']), rot90=int(Se.rot.get(obs, 0)), S=int(Se.S), G=int(G),
                step=sc['step'], fps_src=sc['fps'], theta_star=theta, theta_n=th_n, sigma_j=float(sig[j]),
                fire_thr=float(th_n * sig[j]), n_kept=int(len(pos_t)), sel_token=[sel // G, sel % G],
                frame_value=float(T.cm[k, j]), smoothed_value=float(S[k, j]),
                raw_3=[float(raw[kk, j]) for kk in ks], n_active_frame_code=int((T.cm[k] > 0).sum()),
                n_active_token_code=int((tok_code > 0).sum()), n_tokens_neuron_active=int((heat > 0).sum()),
                win=[lo, lo + WIN], prefixes=list(sae.prefixes) if hasattr(sae, 'prefixes') else None,
                summary=dict(time_spent=float(su['time_spent'][j]), n_bouts=int(su['n_bouts'][j]),
                             bout_rate_per_min=float(su['bout_rate_per_min'][j]), latency_s=float(su['latency_s'][j]),
                             censored=bool(su['censored'][j])),
                verify=verify, summary_check=chk, thumb_pos=tpos, thumb_ex=j_ex,
                name=cfg['name'], species=cfg['species'], short=cfg['short'], canon=cfg['canon'])
    OUT.joinpath('cache').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'cache' / f'{name}.npz', frames=x.permute(0, 2, 3, 1).cpu().numpy(), kept=kept,
                        px=px[1], heat=heat, heat_pm=np.stack(heat_pm), tok_z=tok_z, pre=pre, tok_code=tok_code,
                        raw=raw[:, j], smooth=S[:, j], fire=F[:, j], thumbs=np.stack(thumbs),
                        V=np.stack([sm['time_spent'][vi], sm['bout_rate_per_min'][vi], sm['latency_s'][vi],
                                    sm['censored'][vi]]).astype(np.float32))
    (OUT / 'cache' / f'{name}.json').write_text(json.dumps(meta, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    log(f'[{name}] done in {time.time() - t0:.0f}s: {meta["summary"]}')
    return meta


# ------------------------------------------------------------------------------------------------ plotting
MM = 1 / 25.4
C_KEPT = '#E69F00'   # kept tokens / example frame (Okabe-Ito orange)
C_N = '#CC79A7'      # the shown neuron (Okabe-Ito reddish purple)
C_TOK = '#0072B2'    # the chosen token (Okabe-Ito blue)
C_GREY, C_DARK = '#8c8c8c', '#4d4d4d'
FS, FS_S = 6.0, 5.2


def zoom_box(mk, margin=1, min_side=5):
    """Square token-grid box (x0, y0, side) around the kept tokens (stray 1-2 token pieces ignored)."""
    from scipy import ndimage
    G = mk.shape[0]
    lab, n = ndimage.label(mk, structure=np.ones((3, 3)))
    sizes = ndimage.sum(mk, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.flatnonzero(sizes >= min(3, sizes.max())))
    ii, jj = np.nonzero(keep)
    cy, cx = (ii.min() + ii.max() + 1) / 2, (jj.min() + jj.max() + 1) / 2
    side = int(min(G, max(min_side, max(ii.max() - ii.min(), jj.max() - jj.min()) + 1 + 2 * margin)))
    x0 = int(np.clip(round(cx - side / 2), 0, G - side))
    y0 = int(np.clip(round(cy - side / 2), 0, G - side))
    return x0, y0, side


def edge(ax, mk, color, lw):
    """Outline of the union of kept tokens (grid units)."""
    from matplotlib.collections import LineCollection
    G = mk.shape[0]
    segs = []
    for i in range(G):
        for j in range(G):
            if not mk[i, j]:
                continue
            if i == 0 or not mk[i - 1, j]:
                segs.append([(j, i), (j + 1, i)])
            if i == G - 1 or not mk[i + 1, j]:
                segs.append([(j, i + 1), (j + 1, i + 1)])
            if j == 0 or not mk[i, j - 1]:
                segs.append([(j, i), (j, i + 1)])
            if j == G - 1 or not mk[i, j + 1]:
                segs.append([(j + 1, i), (j + 1, i + 1)])
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, snap=False))
    ax.set_xlim(0, G); ax.set_ylim(G, 0)


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import to_rgb
    from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle
    fams = [f for f in ('Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans')
            if any(f == x.name for x in font_manager.fontManager.ttflist)]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': fams, 'font.size': FS, 'axes.linewidth': 0.4,
                         'xtick.major.width': 0.4, 'ytick.major.width': 0.4, 'xtick.major.size': 1.2,
                         'ytick.major.size': 1.2, 'xtick.labelsize': FS_S, 'ytick.labelsize': FS_S,
                         'xtick.major.pad': 0.8, 'ytick.major.pad': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'mathtext.default': 'regular', 'mathtext.fontset': 'custom', 'mathtext.it': f'{fams[0]}:italic',
                         'mathtext.rm': fams[0], 'axes.labelpad': 0.8})
    doms = args.domains.split(',')
    data = {d: (dict(np.load(OUT / 'cache' / f'{d}.npz')), json.loads((OUT / 'cache' / f'{d}.json').read_text()))
            for d in doms}
    sp = REPO / 'results/vision/eci_pipeline_fig/source_sizes.json'  # ffprobe of every source video (v1 figure)
    sizes = json.loads(sp.read_text()) if sp.exists() else {}
    spec = json.loads((REPO / 'configs/eci_v2/representation.json').read_text())
    K, WL = spec['sae']['k'], spec['sae']['W']
    # ------------------------------------------------ layout (mm)
    W = 183.0
    left, top_h, row_h, sz = 7.0, 9.5, 25.0, 15.0
    Y_TITLE = 4.1
    ag = 2.8
    fw, dx, dy = 10.0, 1.9, 0.5
    gw, bw = 1.6, 7.0
    VPAD = 6.0
    xa = left
    wa = fw + (N_THUMB - 1) * dx
    xb = xa + wa + 5.0
    xs, x = {}, xb
    for k, w in (('grid', sz), ('zoom', sz), ('vec', gw + 1.4 + VPAD), ('pre', bw + 2.0 + 3.5), ('spatial', sz)):
        xs[k] = x
        x += w + ag
    x += 9.5 - ag
    xs['trace'] = x
    vw, vg = 7.6, 1.3
    xs['vid'] = W - 0.5 - (3 * vw + 2 * vg)
    wt = xs['vid'] - 5.5 - xs['trace']
    H = top_h + row_h * (len(doms) - 1) + 18.2
    fig = plt.figure(figsize=(W * MM, H * MM))

    def ax_mm(x, y, w, h):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    def txt(x, y, s, **kw):
        kw.setdefault('fontsize', FS_S)
        kw.setdefault('va', 'top')
        kw.setdefault('linespacing', 1.05)
        return fig.text(x / W, 1 - y / H, s, **kw)

    def img(x, y, w, h=None, ec=C_DARK, lw=0.35):
        a = ax_mm(x, y, w, h or w)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(lw); s.set_color(ec)
        return a

    def arrow(x0, x1, yc, label=None, color='#9e9e9e', label_color=C_DARK):
        fig.patches.append(FancyArrowPatch((x0 / W, 1 - yc / H), (x1 / W, 1 - yc / H), transform=fig.transFigure,
                                           arrowstyle='-|>', mutation_scale=4.5, lw=0.5, color=color,
                                           shrinkA=0, shrinkB=0))
        if label:
            txt((x0 + x1) / 2, yc - 0.6, label, ha='center', va='bottom', fontsize=FS_S - 0.6, color=label_color)

    def grey(frame):
        return np.asarray(frame, np.float32).mean(2) / 255

    def overlay(ax, g, val, mk, color, vmax, G, amax=0.85):
        ax.imshow(g, cmap='gray', vmin=0, vmax=1, extent=(0, G, G, 0), interpolation='antialiased')
        rgba = np.zeros((G, G, 4))
        rgba[..., :3] = to_rgb(color)
        rgba[..., 3] = np.where(mk, amax * np.clip(val / max(vmax, 1e-9), 0, 1), 0)
        ax.imshow(rgba, extent=(0, G, G, 0), interpolation='nearest')

    def gridlines(ax, G, lw, alpha):
        for i in range(1, G):
            ax.axhline(i, color='w', lw=lw, alpha=alpha)
            ax.axvline(i, color='w', lw=lw, alpha=alpha)

    def bars(ax, v, n, hi=None, thr=None, sparse=False):
        """vertical bar glyph of an n-d vector (latent 0 at the top)"""
        idx = np.flatnonzero(v > 0)
        if sparse:
            ax.hlines(idx, 0, v[idx], color='#5a5a5a', lw=0.5)
        else:
            ax.hlines(idx, 0, v[idx], color='#b0b0b0', lw=0.12)
            if thr is not None:
                kk = idx[v[idx] > thr]
                ax.hlines(kk, 0, v[kk], color='#5a5a5a', lw=0.4)
                ax.axvline(thr, color='k', lw=0.35, ls=(0, (1.5, 1.2)))
        if hi is not None:
            ax.hlines([hi], 0, v[hi], color=C_N, lw=0.9)
            v_hi = float(v[hi]) if np.isfinite(v[hi]) else 0.0
            if v_hi > 0:
                ax.plot([v_hi], [hi], 'o', ms=1.3, color=C_N, mec='none')
            else:
                ax.plot([0], [hi], marker='>', ms=2.2, color=C_N, mec='none', clip_on=False)
        ax.set_ylim(n, 0); ax.set_xlim(0, float(np.nanmax(v)) * 1.1 or 1.0)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    for xx, lab, title in ((xa, 'a', 'Video'), (xb, 'b', 'Frame representation'),
                           (xs['spatial'], 'c', 'Neural activation'), (xs['vid'], 'd', 'Video representation')):
        txt(xx, 0.6, lab, fontsize=8, weight='bold')
        txt(xx + 2.6, 0.85, title, fontsize=FS)

    for ri, d in enumerate(doms):
        A, m = data[d]
        G, key = m['G'], m['key']
        y0 = top_h + ri * row_h
        yc = y0 + sz / 2
        kept = A['kept'].reshape(3, G, G)
        mk = kept[1]
        zx, zy, zs = zoom_box(mk)
        sy, sx_ = m['sel_token']
        g = grey(A['frames'][1])  # the canonical encoder input of frame t (S x S)
        iy = (np.arange(g.shape[0]) * G) // g.shape[0]
        upK = mk[iy][:, iy]
        if ri > 0:
            y_rule = (y0 - row_h + 17.4 + y0 - (top_h - Y_TITLE)) / 2
            fig.lines.append(matplotlib.lines.Line2D([0.5 / W, 1 - 0.5 / W], [1 - y_rule / H] * 2,
                                                     transform=fig.transFigure, lw=0.4, color='#d0d0d0'))
        txt(0.4, yc, m['name'], fontsize=6.5, weight='bold', rotation=90, va='center', ha='left')
        txt(3.4, yc, m['species'], fontsize=FS_S - 0.6, rotation=90, va='center', ha='left')
        # ---- a: temporal volume over the whole video; t outlined, arrow from t into b
        tpos, j_ex = m['thumb_pos'], int(m['thumb_ex'])
        top0 = y0 + 0.4
        for i in range(len(tpos)):
            a = img(xa + i * dx, top0 + i * dy, fw, ec=C_KEPT if i == j_ex else '#ffffff', lw=0.9 if i == j_ex else 0.4)
            a.imshow(grey(A['thumbs'][i]), cmap='gray', vmin=0, vmax=1, interpolation='antialiased')
        ya = top0 + (len(tpos) - 1) * dy + fw + 0.9
        vmin_ = m['video_min']
        ta = ax_mm(xa + fw / 2 - dx / 2, ya, len(tpos) * dx, 0.01)
        ta.set_xlim(0, vmin_); ta.set_yticks([])
        for s in ('top', 'right', 'left'):
            ta.spines[s].set_visible(False)
        ta.set_xticks([0, vmin_]); ta.set_xticklabels(['0', f'{vmin_:.0f} min'])
        ta.tick_params(length=1.0, labelsize=FS_S - 0.6)
        ta.plot([tpos[j_ex] / FPS / 60], [0], marker='|', ms=3.0, mew=0.8, color=C_KEPT, clip_on=False)
        sz_d = sizes.get(d, {})
        pr = m['probe']
        wh = next(iter(sz_d.get('sizes', {})), f'{pr["w"]}x{pr["h"]}').replace('x', '×')
        fps0 = float(next(iter(sz_d.get('fps', {})), m['fps_src']))
        used = f'used: {m["S"]}×{m["S"]} px, {FPS:.0f} fps'
        txt(xa, y0 - (top_h - Y_TITLE), f'original: {wh} px, {fps0:.0f} fps\n{used}', fontsize=5.0, color=C_GREY)
        x0 = xa + j_ex * dx + fw * 0.55
        fig.patches.append(FancyArrowPatch((x0 / W, 1 - (top0 + j_ex * dy) / H), ((xs['grid'] - 0.4) / W, 1 - (y0 + 1.5) / H),
                                           transform=fig.transFigure, arrowstyle='-|>', mutation_scale=4.5, lw=0.6,
                                           color=C_KEPT, connectionstyle='arc3,rad=-0.12', shrinkA=0, shrinkB=0))
        # ---- b1: token grid + tight-mask outline
        a = img(xs['grid'], y0, sz)
        a.imshow(g, cmap='gray', vmin=0, vmax=1, extent=(0, G, G, 0), interpolation='antialiased')
        gridlines(a, G, 0.1 * 32 / G, 0.55)
        edge(a, mk, C_KEPT, 0.5)
        arrow(xs['grid'] + sz + 0.3, xs['zoom'] - 0.3, yc)
        # ---- b2: zoom, tokens outside the tight mask blanked, the chosen token outlined
        a = img(xs['zoom'], y0, sz)
        a.imshow(np.where(upK, g, 0.93), cmap='gray', vmin=0, vmax=1, extent=(0, G, G, 0), interpolation='antialiased')
        gridlines(a, G, 0.15, 0.6)
        edge(a, mk, C_KEPT, 0.4)
        a.add_patch(Rectangle((sx_, sy), 1, 1, fill=False, ec=C_TOK, lw=0.9))
        a.set_xlim(zx, zx + zs); a.set_ylim(zy + zs, zy)
        zoom_ax = a
        # ---- b3: the chosen token z_t (DINOv3-L/16 block 20, 1024-d)
        vy, vh = y0 + 0.6, sz - 3.2
        vh_d = 0.6 * vh
        vy_d = vy + (vh - vh_d) / 2
        lim = float(np.percentile(np.abs(A['tok_z']), 98))
        xg = xs['vec'] + 0.7 + VPAD / 2
        g_ = ax_mm(xg, vy_d, gw, vh_d)
        g_.imshow(A['tok_z'][:, None], aspect='auto', cmap='gray_r', vmin=-lim, vmax=lim, interpolation='nearest')
        g_.set_xticks([]); g_.set_yticks([])
        for s in g_.spines.values():
            s.set_linewidth(0.5); s.set_color(C_DARK)
        txt(xg + gw / 2, vy_d + vh_d + 0.3, 'z$_t$', ha='center', fontsize=FS_S - 0.6)
        txt(xg + gw / 2, vy + vh + 0.3, f'd = {len(A["tok_z"])}', ha='center', fontsize=5.0, color='#a0a0a0')
        txt(xs['pre'], vy + vh + 0.3, f'd = {len(A["pre"])}, k = {K}', fontsize=5.0, color='#a0a0a0')
        txt((xs['vec'] + xs['pre'] + bw + 2.0) / 2, vy + vh + 2.2, 'token representation', ha='center', color=C_TOK)
        for k_, lab_ in (('grid', f'{G} × {G} tokens'), ('zoom', 'animal mask'), ('spatial', 'neural activation')):
            txt(xs[k_], y0 + sz + 0.5, lab_, color=C_DARK)
        fig.add_artist(ConnectionPatch(xyA=(sx_ + 1, sy + 0.5), coordsA=zoom_ax.transData, xyB=(0, 0.5),
                                       coordsB=g_.transAxes, color=C_TOK, lw=0.5, arrowstyle='-|>', mutation_scale=4))
        txt(xg + gw / 2, vy - 0.3, 'DINOv3-L', va='bottom', ha='center', color=C_DARK)
        txt(xs['pre'], vy - 0.3, 'SAE (Matryoshka)', va='bottom', ha='left', color=C_DARK)
        # ---- b4: dense pre-activations -> sparse code (theta*), prefixes marked, the neuron highlighted
        pa = ax_mm(xs['pre'], vy, bw + 2.0, vh)
        bars(pa, A['pre'], len(A['pre']), hi=key, thr=m['theta_star'])
        for p_ in (m.get('prefixes') or [])[:-1]:
            pa.plot([0], [p_], marker='_', ms=2.0, mew=0.4, color=C_DARK, clip_on=False)
            pa.text(-0.06, p_, str(p_), transform=pa.get_yaxis_transform(), ha='right', va='center',
                    fontsize=3.6, color='#a0a0a0')
        arrow(xg + gw + 0.3, xs['pre'] - 0.9, yc)
        # ---- b5 (c): the neuron on every kept token
        a = img(xs['spatial'], y0, sz, ec='k', lw=0.5)
        hm = A['heat'].reshape(G, G)
        overlay(a, np.where(upK, g, 0.93), hm, mk, C_N, max(float(hm.max()), 1e-6), G, amax=0.9)
        gridlines(a, G, 0.15, 0.6)
        edge(a, mk, C_KEPT, 0.4)
        a.set_xlim(zx, zx + zs); a.set_ylim(zy + zs, zy)
        fig.add_artist(ConnectionPatch(xyA=(float(A['pre'][key]), float(key)), coordsA=pa.transData,
                                       xyB=(sx_ + 0.5, sy + 0.5), coordsB=a.transData, color=C_N, lw=0.6,
                                       arrowstyle='-|>', mutation_scale=5, shrinkA=2.0, shrinkB=2.5,
                                       connectionstyle='arc3,rad=-0.15'))
        arrow(xs['spatial'] + sz + 0.5, xs['trace'] - 3.4, yc, 'max\npooling', color=C_TOK, label_color=C_TOK)
        # ---- c: 10 s of frame values, raw vs smoothed (0.6 s centred mean), three frames of s(t) highlighted
        raw, smo, fire = A['raw'], A['smooth'], A['fire'].astype(bool)
        lo, hi = m['win']
        t = m['frame']
        tt = (np.arange(lo, hi) - lo) / FPS
        t1 = txt(xs['trace'], y0 - 0.2, f'Neuron {key}: ', color=C_N)
        bb = t1.get_window_extent(renderer=fig.canvas.get_renderer())
        txt(xs['trace'] + bb.width / fig.dpi / MM, y0 - 0.2, m['short'], color=C_N, style='italic')
        a = ax_mm(xs['trace'], y0 + 2.3, wt, sz - 6.4)
        ymax = float(max(raw[lo:hi].max(), smo[lo:hi].max(), m['fire_thr'])) * 1.3  # headroom for the band label
        a.axvspan((t - 1.5 - lo) / FPS, (t + 1.5 - lo) / FPS, color=C_KEPT, alpha=0.18, lw=0)
        a.plot(tt, raw[lo:hi], color='#9e9e9e', lw=0.4)
        a.plot(tt, smo[lo:hi], color=C_N, lw=0.7)
        a.plot((np.arange(t - 1, t + 2) - lo) / FPS, raw[t - 1:t + 2], 'o', ms=1.4, color=C_DARK, mec='none')
        a.plot([(t - lo) / FPS], [smo[t]], 'o', ms=2.0, mfc='none', mec=C_N, mew=0.6)
        a.axhline(m['fire_thr'], color='k', lw=0.35, ls=(0, (2, 1.5)))
        a.set_xlim(0, WIN / FPS); a.set_ylim(0, ymax)
        a.set_xticks([0, WIN / FPS]); a.set_xticklabels(['0', f'{WIN / FPS:.0f} s'])
        a.tick_params(axis='x', length=1.0, labelsize=FS_S - 0.6)
        a.set_yticks([0, max(1, int(round(ymax / 1.3 / 5) * 5))])
        a.text((t - lo) / FPS, ymax * 0.985, 't−1, t, t+1', ha='center', va='top', fontsize=FS_S - 1.2, color='#b07800')
        txt(xs['trace'], y0 - 2.7, 'temporal context: 0.6 s smoothing', color=C_DARK)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        # whole-video firing raster under the trace (s / sigma > theta_n), the 10 s window marked
        n = len(smo)
        vmin_ = n / FPS / 60
        r = ax_mm(xs['trace'], y0 + sz - 1.1, wt, 1.1)
        r.vlines(np.flatnonzero(fire) / FPS / 60, 0, 1, color='#6b6b6b', lw=0.1)
        r.axvspan(lo / FPS / 60, hi / FPS / 60, color=C_KEPT, alpha=0.5, lw=0)
        r.set_xlim(0, vmin_); r.set_ylim(0, 1); r.set_yticks([])
        for s in ('top', 'right', 'left'):
            r.spines[s].set_visible(False)
        r.set_xticks([0, vmin_]); r.set_xticklabels(['0', f'{vmin_:.0f} min'])
        r.tick_params(length=1.0, labelsize=FS_S - 0.6)
        arrow(xs['trace'] + wt + 0.6, xs['vid'] - 0.7, yc, color=C_KEPT)
        # ---- d: the three summaries of ALL neurons in this video
        V = A['V']
        s_ = m['summary']
        assert abs(V[0, key] - s_['time_spent']) < 1e-6 and abs(V[1, key] - s_['bout_rate_per_min']) < 1e-4 \
            and abs(V[2, key] - s_['latency_s']) < 1e-4, (d, V[:3, key], s_)
        lat = f'{s_["latency_s"]:.1f} s' if not s_['censored'] else f'> {s_["latency_s"]:.0f} s'
        vals = (f'{100 * V[0, key]:.0f}%', f'{V[1, key]:.1f}/min', lat)
        for i in range(3):
            va_ = ax_mm(xs['vid'] + i * (vw + vg), vy, vw, vh)
            v_ = V[i].copy()
            if i == 2:  # latency of neurons that never fire in the video (censored) left blank
                v_[V[3] > 0] = np.nan
            bars(va_, v_, V.shape[1], hi=key)
            va_.annotate(vals[i], (0, key), xytext=(0.6, 0.7), textcoords='offset points', ha='left', va='bottom',
                         fontsize=4.2, color=C_N, annotation_clip=False,
                         bbox=dict(fc='white', ec='none', pad=0.25, alpha=0.85))
            txt(xs['vid'] + i * (vw + vg), vy + vh + 0.3, ('time', 'bouts', 'latency')[i], fontsize=FS_S - 0.6,
                color=C_DARK)
        txt(xs['vid'], vy + vh + 2.2, f'd = 3 × {V.shape[1]}', fontsize=5.0, color='#a0a0a0')
        assert V.shape[1] == WL
    out = OUT / 'fig_pipeline_v2.png'
    fig.savefig(out, dpi=args.dpi)
    from PIL import Image
    im = Image.open(out)
    im.convert('RGB').resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS).save(
        OUT / 'fig_pipeline_v2_preview.png')
    log(f'saved {out} ({im.width} x {im.height} px, {W:.0f} x {H:.0f} mm) + preview')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('compute', 'plot', 'all'))
    ap.add_argument('--domains', default='ants,mice,frogs')
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ('compute', 'all'):
        if not os.environ.get('SLURM_JOB_ID'):
            raise SystemExit('compute runs through Slurm (scripts/eci/fig_pipeline_v2.sh): GPU encoder + large reads')
        import torch
        dev = torch.device('cuda')
        facts = {d: compute_domain(d, dev) for d in args.domains.split(',')}
        (OUT / 'facts.json').write_text(json.dumps(facts, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    if args.stage in ('plot', 'all'):
        plot(args)


if __name__ == '__main__':
    main()
