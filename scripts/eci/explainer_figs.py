"""
Explainer figures: how the ECI pipeline turns ONE video frame into a frame representation, at the current
resolution (512 px stored frame -> DINOv2 at 448 -> 32 x 32 patches) and at the ideal (native) resolution
(native frame -> DINOv2 at 896 -> 64 x 64 patches). Visualisation only: no training, no new science.

Per domain (ants v3 with SAE antsfg / rule 'ants'; mice v1 with SAE fg448al / rule 'fg448', odor-aligned):
  1. pick an annotated frame (ants: Y_Y2F = 1 in v3; mice: Y_nn = 1): of 40 random annotated candidates the one whose
     key neuron (ants 90, mice 611) has the highest frame code (SELECTION, stated on the figure)
  2. current pipeline on that frame: DINOv2-base at 448 -> foreground parts (src/eci/foreground.py) -> SAE per kept
     patch (threshold mode, as src/eci/fg_encode.py fg_sae_pool) -> frame code = max (and mean) over kept patches
  3. the same video's frame codes on 40 evenly spaced frames (video value = mean over frames, as contrasts.py)
  4. the native frame: ffmpeg decodes source frames start_frame + 6 k + o, o in -3..3, at native size; o is the offset
     with the lowest grey MAD to the stored JPEG (as scripts/eci/split_test_native.py); DINOv2 at 896 is NOT run
     through the SAE (trained at 448): only the 64 x 64 grid and the zoom are drawn
Plus one shared schematic (pipeline boxes). Output results/vision/eci_explainer/{ants,mice}.png, schematic.png,
facts.json. CPU job: sbatch scripts/eci/explainer_figs.sh
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.fg_encode import fg_sae_pool  # noqa: E402,F401
from src.eci.extract import load_encoder  # noqa: E402
from src.eci.foreground import (GRID, PATCH_PX, RULES, FgBackgrounds, FgEncoder, align_rot90, cosine_distance,  # noqa: E402
                                dark_fraction, encode_batch, foreground_parts, obs_rows, rotate_image, segment_of)
from src.eci.sae import load_sae  # noqa: E402

OUT = REPO / 'results/vision/eci_explainer'
FPS_SRC, STEP = 30.0, 6
log = lambda s: print(s, flush=True)  # noqa: E731

DOM = {
    'ants': dict(
        ann=REPO / 'dataset/ants/eci/annotations.csv', ann_cols=['observation_id', 'frame_idx', 'frame_path', 'experiment', 'Y_Y2F'],
        cand=lambda a: a[(a.experiment == 'v3') & (a.Y_Y2F == 1) & (a.frame_idx >= 10)],
        sae=REPO / 'dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0/sae.pt',
        bg=REPO / 'dataset/ants/eci/fg448/background', align='none', rule='ants', key=90,
        nes=REPO / 'results/vision/ants/eci/nes/matryoshka_btk_1024_k16_antsfg_s0',
        exp=REPO / 'data/ants/v3/experiment.csv', src=REPO / 'data/ants/v3/observations/source',
        title='Ants (v3), SAE antsfg, foreground rule "ants" (= v3 rule)', label='grooming contact (Y2F = 1)',
        native_px=824),
    'mice': dict(
        ann=REPO / 'dataset/mice/v1/annotations.csv', ann_cols=['observation_id', 'frame_idx', 'frame_path', 'Y_nn'],
        cand=lambda a: a[(a.Y_nn == 1) & (a.frame_idx >= 10)],
        sae=REPO / 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448al_s0/sae.pt',
        bg=REPO / 'dataset/mice/v1/eci/fg448al/background', align='odor', rule='fg448', key=611,
        nes=REPO / 'results/vision/mice/eci/nes/matryoshka_btk_1024_k16_fg448al_s0',
        exp=REPO / 'data/mice/v1/experiment.csv', src=REPO / 'data/mice/source',
        title='Mice (v1), SAE fg448al (odor-aligned), foreground rule "fg448"', label='nose-nose contact (Y_nn = 1)',
        native_px=2064),
}


# ---------------------------------------------------------------------------------------------------------------
def fg_parts_for(bgs, tok, grey, rows):
    """As FgBackgrounds.mask, but returns (feature core, dark core, final mask, dist, dark fraction, thr)."""
    rows = np.asarray(rows)
    k = bgs.obs_index(rows)
    b = bgs.get(bgs.ids[k[0]])
    assert (k == k[0]).all()
    seg = segment_of(rows, b['rows'], b['seg_bounds'])
    bg = b['bg'][torch.from_numpy(seg)]
    d = 1 - F.cosine_similarity(tok.float(), bg, dim=-1)
    dk = dark_fraction(grey, b['pix_bg'], bgs.rule['dark_abs'], bgs.rule['dark_rel'])
    feat, dkc, final = foreground_parts(d, dk, b['thr'], bgs.rule)
    return feat, dkc, final, d, dk, b['thr']


@torch.no_grad()
def encode_frames(enc, paths, rots):
    ds = enc.dataset(paths, rot=rots)
    pix = torch.stack([ds[i][0] for i in range(len(ds))])
    grey = torch.stack([ds[i][1] for i in range(len(ds))])
    toks = torch.cat([encode_batch(enc.model, pix[a:a + 8], enc.device) for a in range(0, len(ds), 8)])
    return toks, grey


@torch.no_grad()
def frame_codes(sae, norm, tok, mask):
    """tok (1024, d) fp16, mask (1024,) bool -> per-patch codes (1024, m) (zeros off the mask), max (m,), mean (m,)."""
    z = torch.zeros(tok.shape[0], sae.n_latents)
    if mask.any():
        z[mask] = sae.encode(norm(tok[mask]), mode='threshold')
    mx = z.max(0).values
    mean = z.sum(0) / mask.sum().clamp_min(1)
    return z, mx, mean


def decode_native(src, first, tmp, n=7):
    """n consecutive source frames from `first`, PNG at native size -> list of uint8 RGB arrays."""
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{first / FPS_SRC:.4f}', '-i', str(src), '-vsync', '0', '-frames:v', str(n),
           f'{tmp}/%02d.png']
    subprocess.run(cmd, check=True)
    return [np.asarray(Image.open(f'{tmp}/{i + 1:02d}.png').convert('RGB')) for i in range(n)]


def grid(ax, n, size, color, lw, alpha=1.0, x0=0, y0=0):
    for i in range(n + 1):
        ax.plot([x0, x0 + size], [y0 + i * size / n] * 2, color=color, lw=lw, alpha=alpha)
        ax.plot([x0 + i * size / n] * 2, [y0, y0 + size], color=color, lw=lw, alpha=alpha)


def run_domain(name, cfg, tmp, n_cand=40, n_video=40):
    t0 = pd.Timestamp.now()
    ann = pd.read_csv(cfg['ann'], usecols=cfg['ann_cols'])
    cand = cfg['cand'](ann).sample(n_cand, random_state=0)
    ranges = obs_rows(cfg['ann'])
    rule = RULES[cfg['rule']]
    rot = align_rot90(cfg['align'], cfg['ann'])
    sae, norm, ck = load_sae(cfg['sae'])
    assert ck.get('fg_rule', 'fg448') == cfg['rule'] and ck.get('align', 'none') == cfg['align'], (name, ck.get('fg_rule'))
    enc = FgEncoder('dinov2_base', 'cpu')
    bgs = FgBackgrounds(cfg['bg'], ranges, rule, 'cpu', align=cfg['align'])
    interp = json.load(open(cfg['nes'] / 'interp/interpretations.json'))
    key = cfg['key']

    def codes_of(rows):
        paths = [str(REPO / 'dataset' / ann.frame_path.iloc[r]) for r in rows]
        tok, grey = encode_frames(enc, paths, None if rot is None else rot[np.asarray(rows)])
        out = []
        for i, r in enumerate(rows):
            _, _, final, *_ = fg_parts_for(bgs, tok[i:i + 1], grey[i:i + 1], [r])
            out.append((tok[i], grey[i], final[0]))
        return out

    full = ann  # row index == annotations.csv row (cand keeps the original index)
    # 1. selection
    rows = list(cand.index)
    best, best_val, table = None, -1, []
    for r, (tok, grey, m) in zip(rows, codes_of(rows)):
        _, mx, _ = frame_codes(sae, norm, tok, m)
        table.append((int(r), float(mx[key]), int(m.sum())))
        if mx[key] > best_val:
            best, best_val = r, float(mx[key])
    log(f'[{name}] selected row {best} key-neuron code {best_val:.2f} (candidate median {np.median([t[1] for t in table]):.2f})')
    row = int(best)
    obs, fidx = full.observation_id.iloc[row], int(full.frame_idx.iloc[row])
    lo, hi = ranges[obs]
    # 2. current pipeline on the frame
    path = REPO / 'dataset' / full.frame_path.iloc[row]
    k_rot = 0 if rot is None else int(rot[row])
    tok, grey = encode_frames(enc, [str(path)], None if rot is None else rot[[row]])
    feat, dkc, final, dist, dk, thr = fg_parts_for(bgs, tok, grey, [row])
    mask = final[0]
    z, mx, mean = frame_codes(sae, norm, tok[0], mask)
    others = [int(i) for i in torch.argsort(mx, descending=True) if int(i) != key][:2]
    neurons = [key] + others
    log(f'[{name}] obs {obs} frame {fidx} rot90 {k_rot} kept {int(mask.sum())}/1024 (feature core {int(feat.sum())}, '
        f'dark core {int(dkc.sum())}, thr {thr:.4f}) neurons {neurons}')
    # 3. video trace
    vrows = np.unique(np.r_[np.linspace(lo, hi - 1, n_video).astype(int), row])
    vals = [(frame_codes(sae, norm, t, m)[1:]) for t, _, m in codes_of(list(vrows))]
    vmax = np.stack([v[0].numpy() for v in vals])  # (n, m)
    vmean = np.stack([v[1].numpy() for v in vals])
    # 4. native frame
    exp = pd.read_csv(cfg['exp']).set_index('observation_id')
    srcf = cfg['src'] / exp.loc[obs, 'observation_file']
    start = int(exp.loc[obs, 'start_frame'])
    first = start + STEP * fidx - 3
    with tempfile.TemporaryDirectory(dir=tmp) as d:
        nat = decode_native(srcf, first, d)
    ref = np.asarray(Image.open(path).convert('L')).astype(np.float32)  # stored JPEG (unrotated)
    mad = [float(np.abs(np.asarray(Image.fromarray(x).convert('L').resize((512, 512), Image.BICUBIC)).astype(np.float32) - ref).mean())
           for x in nat]
    off = int(np.argmin(mad)) - 3
    native = Image.fromarray(nat[off + 3])
    native_w = native.size[0]
    log(f'[{name}] native size {native.size}, MAD offsets -3..3: {np.round(mad, 2).tolist()} -> offset {off}')
    native_r = rotate_image(native, k_rot)
    stored = rotate_image(Image.open(path).convert('RGB'), k_rot)
    # DINOv2 input views
    in448 = stored.resize((448, 448), Image.BICUBIC)
    in896 = native_r.resize((896, 896), Image.BICUBIC)
    # zoom region: 6 x 6 patches centred on the key neuron's peak patch
    zk = z[:, key].view(GRID, GRID)
    py, px = divmod(int(zk.argmax()), GRID)
    cy, cx = int(np.clip(py - 2, 0, GRID - 6)), int(np.clip(px - 2, 0, GRID - 6))
    facts = dict(domain=name, obs=obs, frame_idx=fidx, row=row, frame_path=str(full.frame_path.iloc[row]), rot90_ccw=k_rot,
                 selection=f'highest key-neuron ({key}) frame code among {n_cand} random annotated candidates (seed 0): '
                           f'{best_val:.2f}; candidate median {np.median([t[1] for t in table]):.2f}, '
                           f'range {min(t[1] for t in table):.2f}..{max(t[1] for t in table):.2f}',
                 n_kept=int(mask.sum()), n_feature_core=int(feat.sum()), n_dark_core=int(dkc.sum()),
                 n_dark_only_or_feature=int((feat | dkc).sum()), video_thr=float(thr), rule=rule,
                 native_size=list(native.size), native_frame=int(first + 3 + off), offset=off,
                 mad_offsets_m3_p3=mad, neurons=neurons,
                 neuron_frame_max={int(n): float(mx[n]) for n in neurons}, neuron_frame_mean={int(n): float(mean[n]) for n in neurons},
                 neuron_n_active_patches={int(n): int((z[:, n] > 0).sum()) for n in neurons},
                 neuron_video_value_sample={int(n): float(vmax[:, n].mean()) for n in neurons}, n_video_frames=len(vrows),
                 zoom_patch_origin_yx=[cy, cx], peak_patch_yx=[py, px])
    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(22, 24))
    gs = fig.add_gridspec(4, 4, hspace=0.22, wspace=0.08, left=0.02, right=0.99, top=0.945, bottom=0.015)
    fig.suptitle(f'{cfg["title"]}\nOne frame: {obs}, frame {fidx} ({cfg["label"]}); '
                 f'model sees the stored 512 px frame', fontsize=17, y=0.985)
    S = native_w
    ppn = PATCH_PX * S / 512  # native px per current patch

    def show(ax, im, title, sz):
        ax.imshow(im, extent=(0, sz, sz, 0))
        ax.set_title(title, fontsize=12, loc='left')
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 0]); show(ax, native_r, f'(a) Native video frame: {S} x {S} px', S)
    ax.text(0.02, 0.98, f'{S} x {S} px', transform=ax.transAxes, va='top', color='yellow', fontsize=16, weight='bold',
            bbox=dict(fc='k', alpha=.5, lw=0))
    ax = fig.add_subplot(gs[0, 1]); show(ax, stored, '(b1) Stored frame: 512 x 512 px\n(whole frame scaled, no crop)', 512)
    grid(ax, GRID, 512, 'cyan', 0.35, 0.8)
    ax.text(0.02, 0.98, f'1 patch = 16 px of the 512 frame\n= {ppn:.1f} px of the native video', transform=ax.transAxes, va='top',
            color='yellow', fontsize=12, weight='bold', bbox=dict(fc='k', alpha=.6, lw=0))
    ax = fig.add_subplot(gs[0, 2]); show(ax, in448, '(b2) DINOv2 input: 448 x 448 px\n(patch 14 px -> 32 x 32 = 1024 patches)', 448)
    grid(ax, GRID, 448, 'cyan', 0.35, 0.8)
    ax.text(0.02, 0.98, '1 patch = 14 px of the 448 input\n= 16 px of the 512 frame', transform=ax.transAxes, va='top',
            color='yellow', fontsize=12, weight='bold', bbox=dict(fc='k', alpha=.6, lw=0))
    # mask
    ax = fig.add_subplot(gs[0, 3])
    ov = np.asarray(stored).astype(np.float32) / 255
    mk = np.kron(mask.view(GRID, GRID).numpy(), np.ones((PATCH_PX, PATCH_PX)))[..., None]
    dkm = np.kron(dkc[0].view(GRID, GRID).numpy(), np.ones((PATCH_PX, PATCH_PX)))[..., None]
    gm = np.asarray(stored.convert('L')).astype(np.float32)[..., None] / 255 * 0.35
    show(ax, np.where(mk > 0, ov, np.repeat(gm, 3, 2)), f'(c) Animal mask: {int(mask.sum())} of 1024 patches kept', 512)
    grid(ax, GRID, 512, 'white', 0.25, 0.35)
    # outline of kept patches
    mm = mask.view(GRID, GRID).numpy()
    for i in range(GRID):
        for j in range(GRID):
            if mm[i, j]:
                ax.add_patch(Rectangle((j * 16, i * 16), 16, 16, fill=False, ec='lime', lw=0.9))
    ax.text(0.02, 0.98, f'kept {int(mask.sum())}/1024 ({100 * mask.float().mean():.0f}%)\ngrey = dropped', transform=ax.transAxes,
            va='top', color='lime', fontsize=13, weight='bold', bbox=dict(fc='k', alpha=.6, lw=0))
    # (d) heatmaps
    for c, n in enumerate(neurons):
        ax = fig.add_subplot(gs[1, c])
        hm = z[:, n].view(GRID, GRID).numpy()
        ax.imshow(stored, extent=(0, 512, 512, 0), alpha=0.35)
        hmm = np.ma.masked_where(~mm, hm)
        vmaxv = max(hm.max(), 1e-6)
        im = ax.imshow(hmm, extent=(0, 512, 512, 0), cmap='magma', vmin=0, vmax=vmaxv, interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        txt = interp.get(str(n), {}).get('text') if isinstance(interp.get(str(n)), dict) else None
        tag = 'known grooming neuron' if (name == 'ants' and n == 90) else ('deployed nose-nose candidate' if n == 611 else 'strong in this frame')
        ax.set_title(f'(d) Neuron {n}: {tag}\n' + textwrap.fill(f'atlas: {txt}' if txt else 'no atlas text for this neuron', 58),
                     fontsize=10.5, loc='left')
        ax.text(0.02, 0.02, f'frame code: max {mx[n]:.2f}, mean {mean[n]:.2f}\nactive on {int((z[:, n] > 0).sum())} of {int(mask.sum())} kept patches',
                transform=ax.transAxes, color='w', fontsize=11, bbox=dict(fc='k', alpha=.6, lw=0))
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 3]); ax.axis('off')
    ax.text(0, 1, 'Mask rule (src/eci/foreground.py)\n\n'
            f'rule "{cfg["rule"]}" = FG_RULE + dilate_requires_dark\n'
            '1. token cue: cosine distance of the patch token to the video\'s\n   background token (same position, nearest of 4 time blocks)\n'
            f'   > video threshold {thr:.3f}; isolated patches dropped\n   -> {int(feat.sum())} patches\n'
            '2. dark cue: fraction of 16x16 px that are dark (grey < 60 and\n   > 40 below the video\'s pixel background) > 0.15\n'
            f'   -> {int(dkc.sum())} patches\n'
            '3. core = 1 OR 2, then dilate by 1 patch; new\n   patches must contain >= 1 dark pixel\n'
            f'   -> {int(mask.sum())} kept\n\n'
            + ('Mice: frame rotated by k x 90 deg so the odor corner is\ntop right (here k = %d), mask and SAE use the aligned frame.' % k_rot
               if cfg['align'] == 'odor' else 'No frame alignment for ants.'),
            va='top', fontsize=9, family='monospace')
    # (e) aggregation
    ax = fig.add_subplot(gs[2, 0])
    xs = np.arange(len(neurons)); w = 0.38
    ax.bar(xs - w / 2, [float(mx[n]) for n in neurons], w, color='#c0392b', label='MAX over kept patches (deployed codes_max)')
    ax.bar(xs + w / 2, [float(mean[n]) for n in neurons], w, color='#7f8c8d', label='mean over kept patches (codes_mean)')
    ax.set_xticks(xs); ax.set_xticklabels([f'neuron {n}' for n in neurons])
    ax.set_title('(e1) Frame code of each neuron\n= pool the patch codes over the kept patches', fontsize=12, loc='left')
    ax.set_ylim(0, 1.35 * float(mx[neurons].max())); ax.legend(fontsize=9, loc='upper right'); ax.set_ylabel('SAE activation')
    ax = fig.add_subplot(gs[2, 1])
    order = np.argsort(vrows)
    for n, col in zip(neurons, ['#c0392b', '#2471a3', '#229954']):
        ax.plot(vrows[order] - lo, vmax[order, n], '.-', color=col, lw=0.8, ms=4, label=f'neuron {n}')
        ax.axhline(vmax[:, n].mean(), color=col, ls='--', lw=1)
    ax.axvline(row - lo, color='k', lw=1, alpha=.5)
    ax.set_title(f'(e2) Frame codes (max) over {len(vrows)} frames of this video;\ndashed = video value (mean over frames)', fontsize=12, loc='left')
    ax.set_xlabel('frame index in the video (5 fps); black line = example frame'); ax.legend(fontsize=9)
    ax = fig.add_subplot(gs[2, 2:]); ax.axis('off')
    ax.text(0, 1, 'From patches to NES\n\n'
            '  patch codes (1024 per frame, sparse; ~16 of 1024 latents active per patch)\n'
            '        |  MAX over the kept patches  (codes_max; codes_mean = alternative)\n'
            '        v\n'
            '  frame code (1024 numbers per frame)\n'
            '        |  MEAN over the frames of the video window   (src/eci/contrasts.py video_summaries;\n'
            '        |   rate = fraction of frames > 0 and bout rate are the other video statistics)\n'
            '        v\n'
            '  video value per neuron (one number per video)\n'
            '        |  compare between conditions (treatment vs control videos; paired stages for mice)\n'
            '        v\n'
            '  NES: effect size + p-value per neuron\n\n'
            'Note: e2 uses a 40-frame sample of the video; the real pipeline averages ALL frames of the window.',
            va='top', fontsize=12, family='monospace')
    # (f) ideal resolution
    ax = fig.add_subplot(gs[3, 0]); show(ax, native_r, f'(f) IDEAL: native frame {S} px -> DINOv2 input 896 px\n64 x 64 grid, 1 patch = {14 * S / 896:.1f} native px', S)
    grid(ax, 64, S, 'cyan', 0.25, 0.6)
    grid(ax, 32, S, 'yellow', 0.6, 0.8)
    zx0, zy0 = cx * ppn, cy * ppn
    ax.add_patch(Rectangle((zx0, zy0), 6 * ppn, 6 * ppn, fill=False, ec='red', lw=2.5))
    ax.text(0.02, 0.98, 'yellow = current 32 x 32 grid\ncyan = ideal 64 x 64 grid\nred box = zoom region\n(SAE not run at 896: trained at 448)',
            transform=ax.transAxes, va='top', color='w', fontsize=10.5, weight='bold', bbox=dict(fc='k', alpha=.65, lw=0))
    # zoom panels
    st = np.asarray(stored); na = np.asarray(native_r)
    c512 = st[cy * 16:(cy + 6) * 16, cx * 16:(cx + 6) * 16]
    n0, n1 = int(round(zy0)), int(round(zy0 + 6 * ppn))
    m0, m1 = int(round(zx0)), int(round(zx0 + 6 * ppn))
    cnat = na[n0:n1, m0:m1]
    zs = [(c512, 96, 6, 'current: 512 px crop (96 x 96 px),\n6 x 6 patches of 16 px', 'cyan'),
          (cnat, 96, 6, f'native crop ({cnat.shape[1]} x {cnat.shape[0]} px), same 6 x 6\ncurrent patches ({ppn:.0f} native px each)', 'yellow'),
          (cnat, 96, 12, f'native crop, ideal 12 x 12 patches\n({14 * S / 896:.1f} native px each, 896 input)', 'cyan')]
    for c, (im, sz, ng, ttl, col) in enumerate(zs):
        ax = fig.add_subplot(gs[3, 1 + c])
        ax.imshow(im, extent=(0, sz, sz, 0), interpolation='lanczos' if im is cnat else 'nearest')
        grid(ax, ng, sz, col, 1.2, 0.95)
        ax.set_title(f'(f{c + 1}) ' + ttl, fontsize=11.5, loc='left')
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(OUT / f'{name}.png', dpi=130)
    plt.close(fig)
    json.dump(facts, open(OUT / f'facts_{name}.json', 'w'), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    log(f'[{name}] done in {(pd.Timestamp.now() - t0).seconds}s')
    return facts


def schematic():
    fig, ax = plt.subplots(figsize=(22, 7.5))
    ax.set_xlim(0, 22); ax.set_ylim(0, 7.5); ax.axis('off')

    def box(x, y, w, h, txt, fc='#eaf2f8', ec='#1f618d', fs=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc=fc, ec=ec, lw=1.8))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs)

    def arr(x0, y0, x1, y1, ls='-', col='k'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=2, ls=ls, color=col))
    y, h = 4.2, 2.0
    xs = [0.2, 3.4, 6.6, 9.8, 13.0, 16.2, 19.4]
    ws = 2.4
    txt = ['Video frame\n512 x 512 px stored\n(5 fps; native 824 ants /\n2064 mice, 30 fps)',
           'DINOv2-base\nframe resized to 448\npatch 14 px: 32 x 32\n= 1024 patch tokens\n(768-d)',
           'Animal mask\ntoken distance to video\nbackground + dark pixels,\ndilated by 1 patch\n(~tens of 1024 kept)',
           'Matryoshka SAE\n(per kept patch)\n1024 latents, k = 16,\nprefixes 128/256/512/1024\n-> sparse patch code',
           'MAX over kept patches\n(deployed codes_max;\nmean = alternative)\n-> frame code\n(1024 numbers)',
           'MEAN over frames\nof the video window\n-> video code\n(1024 numbers)',
           'NES\ncompare video codes\nbetween conditions\n(effect + p-value\nper neuron)']
    cols = ['#eaf2f8', '#eaf2f8', '#fdebd0', '#e8f8f5', '#fadbd8', '#fadbd8', '#e8daef']
    for x, t, c in zip(xs, txt, cols):
        box(x, y, ws, h + 0.6, t, fc=c, fs=10.5)
    for a, b in zip(xs[:-1], xs[1:]):
        arr(a + ws + 0.05, y + 1.3, b - 0.05, y + 1.3)
    # motion option
    box(6.0, 0.6, 5.2, 2.2, 'MOTION OPTION (SAEs "mot": motion_delta D = 5 frames = 1 s at 5 fps)\nSAE input = [token at t , token at t  minus  token at t-D]\n'
        '(same patch position, 1536-d). Not used by the deployed\nantsfg / fg448al SAEs (static 768-d token).\nNo optical flow anywhere.', fc='#fcf3cf', ec='#b7950b', fs=10)
    arr(8.0, 4.2, 8.0, 2.85, ls='--', col='#b7950b')
    arr(8.6, 2.85, 10.9, 4.2, ls='--', col='#b7950b')
    ax.text(8.1, 3.5, 'token_t', fontsize=9, color='#b7950b', ha='right')
    ax.text(0.2, 2.6, 'Mask and tokens use DINOv2 at 448; the mask cue (a) needs per-video\nbackground tokens (median over 4 time blocks of frames without a dark animal)\n(b) dark pixels. Mice only: frame rotated so the odor corner is top right.',
            fontsize=10, va='top')
    ax.text(13, 2.6, 'Ideal version (not built): native frame -> DINOv2 at 896 -> 64 x 64 patches\n(each patch = 8 px of the 512 frame); needs a new mask, background and SAE.',
            fontsize=10, va='top', color='#7b241c')
    ax.set_title('ECI: one frame -> frame code -> video code -> NES (current pipeline)', fontsize=16, loc='left')
    fig.savefig(OUT / 'schematic.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domains', default='ants,mice')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = os.environ.get('STAGE', '/tmp')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    schematic()
    for d in args.domains.split(','):
        run_domain(d, DOM[d], tmp)


if __name__ == '__main__':
    main()
