"""
Nature-style multi-panel figure of the ECI (Exploratory Causal Inference) representation pipeline "C1": one row per
domain (ants, mice, frogs), the same eight steps on a real example frame of that domain.

    a  raw video          native frame (ffprobe size / fps / duration of the example's source file)
    b  standardised       5 fps, 512 x 512 (whole frame scaled, no crop): the example frame and the frames 1 s and 2 s
                          earlier; timeline = the video, the analysis window shaded (mice habituation: its last 15 min)
    c  frozen encoder     the 448 px encoder input with the patch grid (32 x 32 for DINOv2-base at 448)
    d  animal mask        kept patches in colour, the rest grey (src/eci/foreground.py rule of the SAE checkpoint)
    e  motion             the SAE input per kept patch is [x_t, x_t - x_(t-1 s)]: ||x_t - x_(t-1 s)|| per kept patch
    f  sparse dictionary  Matryoshka BatchTopK SAE (4096 latents, k = 16): two neurons' activations per kept patch
    g  frame read-out     max over kept patches -> one 4096-d sparse frame code (the deployed codes_max)
    h  video -> NES       the neuron's trace over the video (10 s running mean of codes_max) and its per-video value
                          (mean over the analysis window, the NES outcome) by condition, with the NES statistics

Stages (Slurm only, scripts/eci/fig_pipeline.sh):
    compute  CPU: per domain pick the example frame from the deployed codes (of 200 random eligible frames, seed 0,
             the one where BOTH shown neurons rank highest: max of the smaller within-candidate percentile rank),
             copy its JPEGs to $STAGE, run the encoder + mask + SAE on it (fg_encode.fg_sae_pool semantics, per
             patch), VERIFY the pooled max against the stored codes_max row and n_fg (latents flipping at the SAE
             inference threshold are counted, at most 3 and never a shown neuron), decode the mapped native frame
             (start + step k + expect) with ffmpeg (grey MAD to the stored JPEG over offsets -3..3 recorded as a check),
             read the video trace from codes_max.npy and the per-video values from the NES cache
             -> OUT/cache/<domain>.npz + OUT/facts.json
    plot     reads only the cache -> OUT/fig_pipeline.{pdf,png} (PNG at 300 dpi)

Encoder switch: the encoder label, the patch-grid size and the input size are written to the cache by compute and can
be overridden at plot time (--encoder-label, --grid, --input-px). compute supports SAEs on DINOv2-family tokens
(src/eci/foreground.py FgEncoder, the deployed C1 path); a V-JEPA SAE needs its own token function (enc_tokens()).
Change the SAEs with --sae ants=<name>,mice=<name>,frogs=<name>.

Usage: STAGE_ARG=all sbatch scripts/eci/fig_pipeline.sh  (or compute / plot)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / 'results/vision/eci_pipeline_fig'
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731

SAE = {'ants': 'matryoshka_btk_4096_k16_antsfgm5_s0', 'mice': 'matryoshka_btk_4096_k16_fg448m5_s0',
       'frogs': 'matryoshka_btk_4096_k16_frogsfgm5_s0'}
NES_KEY = 'p512_max_mean_t_bonferroni_full'  # primary NES settings (codes_max, per-video mean, t, Bonferroni)

DOM = {
    'ants': dict(
        eci=REPO / 'dataset/ants/eci', nes=REPO / 'results/vision/ants/eci/nes', name='Ants', species=r'$\it{Lasius\ neglectus}$',
        raw_exp=REPO / 'data/ants/v2/experiment.csv', src=REPO / 'data/ants/v2/observations/source',
        step=6, expect=2, hflip=False, version='v2 (ISTAnt)',
        cand=lambda a: a[(a.experiment == 'v2') & ((a.Y_Y2F == 1) | (a.Y_B2F == 1)) & (a.frame_idx >= 10)],
        ann_cols=['observation_id', 'frame_idx', 'frame_path', 'experiment', 'Y_Y2F', 'Y_B2F'],
        neurons=[186, 418], key=186,
        desc={186: 'grooming\nAUROC 0.86', 418: 'grooming\nAUROC 0.73'},
        analysis='v2_1_vs_2', groups=('T = 1', 'T = 2'),
        cand_text='annotated grooming frames (Y2F or B2F) of ants v2'),
    'mice': dict(
        eci=REPO / 'dataset/mice/v1/eci', ann=REPO / 'dataset/mice/v1/annotations.csv', nes=REPO / 'results/vision/mice/eci/nes', name='Mice', species=r'$\it{Mus\ musculus}$',
        raw_exp=REPO / 'data/mice/v1/experiment.csv', src=REPO / 'data/mice/source',
        step=6, expect=2, hflip=False, version='v1',
        ann_cols=['observation_id', 'frame_idx', 'frame_path', 'Y_nn'],
        neurons=[240, 915], key=915,
        desc={240: 'NES hit\nhet H → O', 915: 'contact\nAUROC 0.72'},
        analysis='A_het_1to2', groups=('H', 'O'),
        cand_text='annotated nose-nose frames (Y_nn = 1) in the last 15 min of het social-odour habituation videos'),
    'frogs': dict(
        eci=REPO / 'dataset/frogs/eci', nes=REPO / 'results/vision/frogs/eci/nes', name='Frogs',
        species=r'$\it{Xenopus}$ froglets', raw_exp=REPO / 'data/frogs/v1/experiment.csv',
        src=REPO / 'data/frogs/v1/observations/source', step=12, expect=5, hflip=True, version='v1 (juvenile)',
        ann_cols=['observation_id', 'frame_idx', 'frame_path', 'group'],
        cand=lambda a: a[(a.group == 'FoxP1') & (a.frame_idx >= 10)],
        neurons=[500, 147], key=500,
        desc={500: 'NES hit\nFoxP1 > WT', 147: 'NES hit (bouts)\nFoxP1 > WT'},
        analysis='FoxP1_vs_WT', groups=('WT', 'FoxP1'),
        cand_text='frames of FoxP1 (half-embryo crispant) froglets'),
}


# ------------------------------------------------------------------------------------------------ compute helpers
def ffprobe(path):
    o = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                   'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name:format=duration',
                                   '-of', 'json', str(path)], check=True, capture_output=True, text=True).stdout)
    s = o['streams'][0]
    num, den = map(float, s['avg_frame_rate'].split('/'))
    return dict(width=int(s['width']), height=int(s['height']), fps=num / den, codec=s['codec_name'],
                nb_frames=s.get('nb_frames'), duration_s=float(o['format']['duration']))


def decode_native(src, first, fps, tmp, n=7):
    """n consecutive source frames from index `first` at native size -> list of uint8 RGB arrays."""
    from PIL import Image
    d = Path(tmp) / 'native'
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{first / fps:.4f}', '-i', str(src), '-vsync', '0', '-frames:v',
                    str(n), f'{d}/%02d.png'], check=True)
    return [np.asarray(Image.open(f'{d}/{i + 1:02d}.png').convert('RGB')) for i in range(n)]


def per_video(name, cfg, sae_name):
    """Per-video NES outcome (mean of codes_max over the analysis window) of cfg['neurons'] + NES statistics."""
    nes = cfg['nes'] / sae_name
    vs = np.load(nes / '_cache/video_summaries_max.npz')
    ids = list(vs['observation_id'])
    res = json.load(open(nes / cfg['analysis'] / 'result.json'))[NES_KEY]
    fr = res['first_round']
    stats = {}
    for n in cfg['neurons']:
        if n in fr['neuron']:
            i = fr['neuron'].index(n)
            stats[n] = {k: fr[k][i] for k in ('tau', 'se', 't', 'df', 'p', 'significant')}
    stats['selected'] = res['selected']
    stats['n_units'] = res['n_units']
    stats['threshold'] = (res['rounds'].get('threshold') or [0.05 / res['n_tested']])[0]
    stats['n_tested'] = res['n_tested']
    if name == 'mice':  # paired: het pools, stage 1 (S, H; last 15 min) vs stage 2 (S, O; whole)
        e = pd.read_csv(cfg['raw_exp'])
        e = e[(e.genotype == 'het') & (e.odor == 'S') & e.phase.isin(['H', 'O'])]
        pv = e.pivot(index='pool', columns='phase', values='observation_id').dropna()
        a = np.stack([vs['last__mean'][[ids.index(o) for o in pv['H']]][:, cfg['neurons']],
                      vs['full__mean'][[ids.index(o) for o in pv['O']]][:, cfg['neurons']]])  # (2, pools, neurons)
        return dict(values=a, unit_ids=np.array(pv.index), paired=True), stats
    e = pd.read_csv(cfg['eci'] / 'experiment.csv')
    if name == 'ants':
        e = e[e.experiment == 'v2']
        g = {1: 0, 2: 1}
        lab = e['T'].map(g)
    else:
        g = {'WT': 0, 'FoxP1': 1}
        e = e[e.group.isin(g)]
        lab = e['group'].map(g)
    val = vs['full__mean'][[ids.index(o) for o in e.observation_id]][:, cfg['neurons']]
    return dict(values=val, group=lab.values.astype(int), unit_ids=e.observation_id.values, paired=False), stats


def compute_domain(name, cfg, sae_name, tmp, n_cand=200):
    import torch
    from PIL import Image
    from src.eci.foreground import GRID, RULES, FgBackgrounds, FgEncoder, encode_batch, obs_rows
    from src.eci.sae import load_sae
    t0 = time.time()
    eci = cfg['eci']
    ann_path = cfg.get('ann', eci / 'annotations.csv')
    cdir = eci / 'codes' / sae_name
    ccfg = json.load(open(cdir / 'config.json'))
    sae, norm, ck = load_sae(eci / 'sae' / sae_name / 'sae.pt')
    D = int(ck.get('motion_delta', 0) or 0)
    rule_name = ck.get('fg_rule', 'fg448')
    assert ck.get('align', 'none') == 'none' and not ck.get('bg_sub', False), (name, ck.get('align'), ck.get('bg_sub'))
    assert rule_name == ccfg['foreground_rule_name'], (rule_name, ccfg['foreground_rule_name'])
    encoder = ck.get('encoder', 'dinov2_base')
    ann = pd.read_csv(ann_path, usecols=cfg['ann_cols'])
    ranges = obs_rows(ann_path)
    codes = np.load(cdir / 'codes_max.npy', mmap_mode='r')
    nfg_all = np.load(cdir / 'n_fg.npy', mmap_mode='r')
    assert codes.shape[0] == len(ann), (codes.shape, len(ann))
    # ---------------------------------------------------------------- 1. example frame
    if name == 'mice':
        e = pd.read_csv(cfg['raw_exp'])
        hab = set(e[(e.genotype == 'het') & (e.odor == 'S') & (e.phase == 'H')].observation_id)
        cand = ann[ann.observation_id.isin(hab) & (ann.Y_nn == 1) & (ann.frame_idx >= 4500 + 10)]
    else:
        cand = cfg['cand'](ann)
    cand = cand.sample(min(n_cand, len(cand)), random_state=0)
    rows = np.sort(cand.index.values)
    cv = np.asarray(codes[rows][:, cfg['neurons']], dtype=np.float32)  # (n_cand, 2)
    kv = cv[:, cfg['neurons'].index(cfg['key'])]
    # both shown neurons active: maximise the smaller of the two within-candidate percentile ranks (ties: key value)
    pr = np.stack([pd.Series(c).rank(pct=True).values for c in cv.T], 1)
    score = np.where((cv > 0).all(1), pr.min(1), -1) + 1e-6 * kv / max(kv.max(), 1e-6)
    best = int(np.argmax(score))
    row = int(rows[best])
    obs, fidx = ann.observation_id.iloc[row], int(ann.frame_idx.iloc[row])
    lo, hi = ranges[obs]
    log(f'[{name}] {len(cand)} candidates, key {cfg["key"]} codes_max median {np.median(kv):.2f} max {kv.max():.2f} '
        f'-> row {row} ({obs} frame {fidx}), codes {dict(zip(cfg["neurons"], np.round(cv[best], 2)))}, '
        f'min percentile {pr[best].min():.3f}')
    # ---------------------------------------------------------------- 2. encoder + mask + SAE on the frame
    want = [row, row - D, row - 2 * D]  # t, t - 1 s, t - 2 s (D = 5 frames at 5 fps)
    assert min(want) >= lo
    loc = []
    for r in want:
        p = Path(tmp) / 'jpg' / Path(ann.frame_path.iloc[r]).name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / 'dataset' / ann.frame_path.iloc[r], p)
        loc.append(str(p))
    enc = FgEncoder(encoder, 'cpu')
    ds = enc.dataset(loc)
    pix = torch.stack([ds[i][0] for i in range(len(ds))])
    grey = torch.stack([ds[i][1] for i in range(len(ds))])
    tok = encode_batch(enc.model, pix, 'cpu')  # (3, 1024, d) fp16, the mask tokens
    assert tok.shape[1] == GRID * GRID
    bgs = FgBackgrounds(ccfg['backgrounds'], ranges, RULES[rule_name], 'cpu')
    mask = bgs.mask(tok[:1], grey[:1], np.array([row]))[0][0]  # (1024,) bool
    stok = enc.sae_tokens(tok)  # SAE input tokens (= tok for DINOv2-base)
    x = stok[0][mask]
    if D > 0:
        x = torch.cat([x, (x.float() - stok[1][mask].float()).half()], 1)
    with torch.no_grad():
        z = sae.encode(norm(x), mode='threshold').float()
    Z = torch.zeros(GRID * GRID, z.shape[1])
    Z[mask] = z
    mx = Z.max(0).values.numpy()
    stored = np.asarray(codes[row], dtype=np.float32)
    mx16 = mx.astype(np.float16).astype(np.float32)
    diff_all = np.abs(mx16 - stored)
    thr = float(sae.threshold)
    # JumpReLU-style inference (z = pre * (pre > theta)): a CPU vs GPU rounding difference can flip a latent whose
    # pre-activation sits at theta, giving a jump of ~theta. Those flips are counted separately, not hidden.
    flip = ((mx16 == 0) | (stored == 0)) & (np.maximum(mx16, stored) < 1.15 * thr)
    diff = float(diff_all[~flip].max()) if (~flip).any() else 0.0
    n_kept = int(mask.sum())
    verify = dict(max_abs_diff_vs_codes_max=diff, n_threshold_flips=int((flip & (diff_all > 0)).sum()),
                  sae_threshold=thr, max_abs_diff_incl_flips=float(diff_all.max()),
                  n_kept=n_kept, n_fg_stored=int(nfg_all[row]),
                  argmax_match=bool(int(np.argmax(mx)) == int(np.argmax(stored))),
                  n_active_recomputed=int((mx > 0).sum()), n_active_stored=int((stored > 0).sum()))
    log(f'[{name}] verify {verify}')
    if (n_kept != int(nfg_all[row]) or diff > 0.05 * max(1.0, float(stored.max())) or verify['n_threshold_flips'] > 3
            or any(flip[n] and diff_all[n] > 0 for n in cfg['neurons'])):
        raise RuntimeError(f'[{name}] recomputed frame code does not match the deployed codes: {verify}')
    motion = (stok[0].float() - stok[1].float()).norm(dim=1).numpy()  # ||x_t - x_(t-D)|| per patch
    heat = Z[:, cfg['neurons']].numpy().T.reshape(len(cfg['neurons']), GRID, GRID)
    # ---------------------------------------------------------------- 3. native frame
    e = pd.read_csv(cfg['raw_exp']).set_index('observation_id')
    srcf = cfg['src'] / e.loc[obs, 'observation_file']
    probe = ffprobe(srcf)
    start = int(e.loc[obs, 'start_frame'])
    hflip = bool(cfg['hflip'] and int(e.loc[obs, 'hflip']) == 1)
    centre = start + cfg['step'] * fidx + cfg['expect']
    nat = decode_native(srcf, centre - 3, probe['fps'], tmp)
    if hflip:
        nat = [np.ascontiguousarray(a[:, ::-1]) for a in nat]
    ref = np.asarray(Image.open(loc[0]).convert('L')).astype(np.float32)
    mad = [float(np.abs(np.asarray(Image.fromarray(a).convert('L').resize((512, 512), Image.BICUBIC)).astype(np.float32)
                        - ref).mean()) for a in nat]
    off = int(np.argmin(mad)) - 3
    native = nat[3]  # the mapped frame (start + step k + expect); the MAD argmin is recorded as a check
    log(f'[{name}] native {native.shape[1]}x{native.shape[0]} {probe}; MAD -3..3 {np.round(mad, 2).tolist()} '
        f'-> source frame {centre + off} (expected {centre})')
    nat_small = np.asarray(Image.fromarray(native).resize((768, 768), Image.LANCZOS)) if native.shape[0] > 768 else native
    frames = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in loc])  # t, t-1s, t-2s (512)
    in_px = int(pix.shape[-1])
    enc_in = np.asarray(Image.fromarray(frames[0]).resize((in_px, in_px), Image.BICUBIC))
    # ---------------------------------------------------------------- 4. traces + per-video values
    if name == 'mice':
        ee = pd.read_csv(cfg['raw_exp'])
        pool = ee.set_index('observation_id').loc[obs, 'pool']
        o2 = ee[(ee.pool == pool) & (ee.odor == 'S') & (ee.phase == 'O')].observation_id.iloc[0]
        segs = [(obs, *ranges[obs]), (o2, *ranges[o2])]
    else:
        segs = [(obs, lo, hi)]
    traces = [np.asarray(codes[a:b], dtype=np.float32)[:, cfg['neurons']] for _, a, b in segs]
    pv, stats = per_video(name, cfg, sae_name)
    # window of the example video (frames at 5 fps)
    n_frames = hi - lo
    win = (n_frames - 4500, n_frames) if name == 'mice' else (0, n_frames)
    win_mean = traces[0][win[0]:win[1]].mean(0)
    if pv['paired']:
        pool_ids = list(pv['unit_ids'])
        i_ex = pool_ids.index(pool)
        chk = float(np.abs(pv['values'][0, i_ex] - win_mean).max())
    else:
        i_ex = list(pv['unit_ids']).index(obs)
        chk = float(np.abs(pv['values'][i_ex] - win_mean).max())
    log(f'[{name}] per-video value of the example vs its window mean of codes_max: max abs diff {chk:.4f}')
    meta = dict(domain=name, sae=sae_name, encoder=encoder, encoder_label='DINOv2-base' if encoder == 'dinov2_base' else encoder,
                grid=GRID, input_px=in_px, frame_px=512, motion_delta=D, fg_rule=rule_name, row=row, obs=obs,
                frame_idx=fidx, n_kept=n_kept, n_patches=GRID * GRID, neurons=cfg['neurons'], key=cfg['key'],
                selection=f'among {len(cand)} random {cfg["cand_text"]} (seed 0), the frame where both shown neurons '
                          f'{cfg["neurons"]} rank highest (max of the smaller percentile rank, {pr[best].min():.3f}); '
                          f'codes_max there {dict(zip(cfg["neurons"], np.round(cv[best], 2).tolist()))}; key neuron '
                          f'{cfg["key"]} candidate median {np.median(kv):.2f}, max {kv.max():.2f}',
                native=dict(probe, frame=int(centre), mad_argmin_frame=int(centre + off), mad_argmin_offset=off, mad=mad, hflip=hflip,
                            size=[int(native.shape[1]), int(native.shape[0])], file=str(srcf.name)),
                video_frames_5fps=n_frames, window_5fps=list(win), segs=[[s, int(a), int(b)] for s, a, b in segs],
                verify=verify, frame_code_values={int(n): float(mx[n]) for n in cfg['neurons']},
                n_active_frame_code=int((mx > 0).sum()), n_active_patches={int(n): int((Z[:, n] > 0).sum()) for n in cfg['neurons']},
                motion_kept_median=float(np.median(motion[mask.numpy()])), motion_kept_max=float(motion[mask.numpy()].max()),
                nes=stats, analysis=cfg['analysis'], groups=list(cfg['groups']), desc={str(k): v for k, v in cfg['desc'].items()},
                example_unit_index=int(i_ex), example_value_check=chk, version=cfg['version'],
                name=cfg['name'], species=cfg['species'])
    OUT.joinpath('cache').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'cache' / f'{name}.npz', native=nat_small, frames=frames, enc_in=enc_in,
                        mask=mask.numpy().reshape(GRID, GRID), motion=motion.reshape(GRID, GRID), heat=heat, code=mx,
                        pv_values=pv['values'], pv_group=pv.get('group', np.zeros(0, int)),
                        **{f'trace{i}': t for i, t in enumerate(traces)})
    json.dump(meta, open(OUT / 'cache' / f'{name}.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    log(f'[{name}] done in {time.time() - t0:.0f}s')
    return meta


# ------------------------------------------------------------------------------------------------ plotting
MM = 1 / 25.4
C_KEPT = '#E69F00'      # accent: kept / active patches (Okabe-Ito orange)
C_N = ['#0072B2', '#CC79A7']  # neuron 1 / neuron 2 (Okabe-Ito blue, reddish purple)
C_GREY, C_LIGHT, C_DARK = '#8c8c8c', '#d9d9d9', '#4d4d4d'
FS, FS_S = 6.0, 5.2


def fmt_p(p):
    if p >= 0.01:
        return f'{p:.2f}'
    m, e = f'{p:.1e}'.split('e')
    return f'{float(m):.1f}×10$^{{{int(e)}}}$'


def zoom_box(mk, margin=2, min_side=10):
    """Square patch-grid box (x0, y0, side) around the kept patches, for the zoomed panels e and f."""
    from scipy import ndimage
    G = mk.shape[0]
    lab, n = ndimage.label(mk, structure=np.ones((3, 3)))
    sizes = ndimage.sum(mk, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.flatnonzero(sizes >= min(3, sizes.max())))  # stray 1-2 patch pieces do not set the zoom
    ii, jj = np.nonzero(keep)
    cy, cx = (ii.min() + ii.max() + 1) / 2, (jj.min() + jj.max() + 1) / 2
    side = int(min(G, max(min_side, max(ii.max() - ii.min(), jj.max() - jj.min()) + 1 + 2 * margin)))
    x0 = int(np.clip(round(cx - side / 2), 0, G - side))
    y0 = int(np.clip(round(cy - side / 2), 0, G - side))
    return x0, y0, side


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Polygon, Rectangle
    fams = [f for f in ('Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans')
            if any(f == x.name for x in font_manager.fontManager.ttflist)]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': fams, 'font.size': FS, 'axes.linewidth': 0.5,
                         'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 1.5,
                         'ytick.major.size': 1.5, 'xtick.labelsize': FS_S, 'ytick.labelsize': FS_S,
                         'xtick.major.pad': 1, 'ytick.major.pad': 1, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'mathtext.default': 'regular', 'mathtext.fontset': 'custom', 'mathtext.it': f'{fams[0]}:italic',
                         'mathtext.rm': fams[0], 'axes.labelpad': 1})
    log(f'font: {fams[0]}')
    doms = args.domains.split(',')
    data = {d: (dict(np.load(OUT / 'cache' / f'{d}.npz')), json.load(open(OUT / 'cache' / f'{d}.json'))) for d in doms}
    m0 = data[doms[0]][1]
    enc_label = args.encoder_label or m0['encoder_label']
    grid_n = args.grid or m0['grid']
    in_px = args.input_px or m0['input_px']
    # ------------------------------------------------ layout (mm)
    W = 183.0
    left, top_h, row_h, im = 7.0, 10.5, 33.0, 18.0
    widths = dict(a=18.0, b=18.5, c=18.0, d=18.0, e=18.0, f=36.0, g=10.5, h=23.0)
    gap = (W - left - 0.5 - sum(widths.values())) / (len(widths) - 1)
    xs, x = {}, left
    for k, w in widths.items():
        xs[k] = x
        x += w + gap
    H = top_h + row_h * len(doms) - 3.0
    fig = plt.figure(figsize=(W * MM, H * MM))

    def ax_mm(x, y, w, h, **kw):  # y measured from the top
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)

    def txt(x, y, s, **kw):
        kw.setdefault('fontsize', FS)
        kw.setdefault('va', 'top')
        return fig.text(x / W, 1 - y / H, s, **kw)

    def img_ax(x, y, w, h=None):
        a = ax_mm(x, y, w, h or w)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.4); s.set_color(C_DARK)
        return a

    heads = dict(a='Raw video', b='Standardized\nframes', c=f'Frozen encoder\n({enc_label})',
                 d='Animal mask', e='Motion\n[x$_t$, x$_t$ − x$_{t-1\\,s}$]',
                 f='Sparse dictionary\n(Matryoshka SAE)', g='Frame\nread-out',
                 h='Video summary\n→ NES')
    for k, s in heads.items():
        txt(xs[k], 1.0, k, fontsize=8, weight='bold')
        txt(xs[k] + 2.7, 1.25, s, fontsize=FS, linespacing=1.05)

    def chevron(xc, yc):
        w, h = 1.1, 2.0
        fig.patches.append(Polygon([((xc - w / 2) / W, 1 - (yc - h / 2) / H), ((xc + w / 2) / W, 1 - yc / H),
                                    ((xc - w / 2) / W, 1 - (yc + h / 2) / H)], closed=True, transform=fig.transFigure,
                                   fc=C_LIGHT, ec='none'))

    def cbar(x, y, w, cmap, vmax):
        a = ax_mm(x, y, w, 0.9)
        a.imshow(np.linspace(0, 1, 128)[None], aspect='auto', cmap=cmap, extent=(0, 1, 0, 1))
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.3)
        txt(x, y + 1.1, '0', fontsize=FS_S, ha='left')
        txt(x + w, y + 1.1, f'{vmax:.1f}', fontsize=FS_S, ha='right')

    def under(k, y, s, **kw):
        kw.setdefault('linespacing', 1.1)
        return txt(xs[k], y, s, **kw)

    vm_motion = max(float(np.ma.masked_where(~A_['mask'].astype(bool), A_['motion']).max()) for A_, _ in data.values())
    for ri, d in enumerate(doms):
        A, m = data[d]
        y0 = top_h + ri * row_h
        G = grid_n
        ut = y0 + im + 0.9  # first text line under the images
        txt(0.6, y0 + im / 2, m['name'], fontsize=7, weight='bold', rotation=90, va='center', ha='left')
        txt(3.6, y0 + im / 2, m['species'], fontsize=FS, rotation=90, va='center', ha='left')
        for k in list(widths)[:-1]:
            chevron(xs[k] + widths[k] + gap / 2, y0 + im / 2)
        nat = m['native']
        fps_src = nat['fps']
        dur = nat['duration_s'] / 60
        # a raw video
        a = img_ax(xs['a'], y0, im)
        a.imshow(A['native'], interpolation='lanczos')
        under('a', ut, f'{nat["size"][0]} × {nat["size"][1]} px\n{fps_src:.0f} fps, {dur:.0f} min')
        # b standardized: stack of frames t - 2 s, t - 1 s, t (front)
        fw, dx, dy = 12.5, 3.0, 2.75
        for j, (fi, lab) in enumerate(zip((2, 1, 0), ('−2 s', '−1 s', 't'))):
            a = img_ax(xs['b'] + j * dx, y0 + j * dy, fw)
            a.imshow(A['frames'][fi], interpolation='lanczos')
            a.text(0.05, 0.95, lab, transform=a.transAxes, color='w', fontsize=FS_S, va='top', ha='left')
        nfr = m['video_frames_5fps']
        w0, w1 = m['window_5fps']
        tl = ax_mm(xs['b'], ut + 0.2, widths['b'], 1.2)
        tl.add_patch(Rectangle((0, 0), nfr, 1, fc=C_LIGHT, ec='none'))
        tl.add_patch(Rectangle((w0, 0), w1 - w0, 1, fc=C_GREY, ec='none'))
        tl.plot([m['frame_idx']] * 2, [-0.4, 1.4], color=C_KEPT, lw=0.9, solid_capstyle='butt')
        tl.set_xlim(0, nfr); tl.set_ylim(-0.4, 1.4); tl.axis('off')
        step = int(round(fps_src / 5))
        vmin, wmin = nfr / 300, (w1 - w0) / 300
        if wmin < vmin - 0.5:
            wt = f'last {wmin:.0f} of {vmin:.0f} min used'
        elif vmin < dur - 0.5:
            wt = f'first {vmin:.0f} of {dur:.0f} min used'
        else:
            wt = f'all {vmin:.0f} min used'
        under('b', ut + 2.0, f'512 × 512 px, 5 fps\n{wt}', fontsize=FS_S)
        # c encoder input + grid
        a = img_ax(xs['c'], y0, im)
        a.imshow(A['enc_in'], extent=(0, in_px, in_px, 0), interpolation='lanczos')
        for i in range(1, G):
            v = i * in_px / G
            a.axhline(v, color='w', lw=0.12, alpha=0.75)
            a.axvline(v, color='w', lw=0.12, alpha=0.75)
        a.set_xlim(0, in_px); a.set_ylim(in_px, 0)
        under('c', ut, f'{in_px} px input\n{G} × {G} patches')
        # d mask (full frame) + zoom box
        mk = A['mask'].astype(bool)
        zx, zy, zs = zoom_box(mk)
        fr = A['frames'][0].astype(np.float32) / 255
        up = np.kron(mk, np.ones((512 // G, 512 // G), bool))[..., None]
        grey = fr.mean(2, keepdims=True) * 0.45 + 0.35
        a = img_ax(xs['d'], y0, im)
        a.imshow(np.where(up, fr, np.repeat(grey, 3, 2)), extent=(0, G, G, 0), interpolation='nearest')
        edge(a, mk, C_KEPT, 0.45)
        a.add_patch(Rectangle((zx, zy), zs, zs, fill=False, ec='w', lw=0.5, ls=(0, (2, 1.5))))
        under('d', ut, f'kept {m["n_kept"]} / {G * G}')
        under('d', ut + 2.6, 'box = zoom (e, f)', fontsize=FS_S)
        # e motion (zoom)
        a = img_ax(xs['e'], y0, im)
        a.imshow(fr.mean(2), cmap='gray', vmin=-0.1, vmax=1.5, extent=(0, G, G, 0), interpolation='bilinear')
        mo = np.ma.masked_where(~mk, A['motion'])
        vm = vm_motion
        a.imshow(mo, cmap='viridis', vmin=0, vmax=vm, extent=(0, G, G, 0), interpolation='nearest')
        a.set_xlim(zx, zx + zs); a.set_ylim(zy + zs, zy)
        cbar(xs['e'], ut - 0.2, im, 'viridis', vm)
        under('e', ut + 2.9, '|x$_t$ − x$_{t-1\\,s}$| per patch\n(one scale, all rows)', fontsize=FS_S)
        # f SAE heatmaps (zoom)
        hw = (widths['f'] - 1.5) / 2
        for j, n in enumerate(m['neurons']):
            xx = xs['f'] + j * (hw + 1.5)
            a = img_ax(xx, y0, im, im)
            a.set_position([xx / W, 1 - (y0 + im) / H, hw / W, im / H])
            a.imshow(fr.mean(2), cmap='gray', vmin=-0.2, vmax=1.6, extent=(0, G, G, 0), interpolation='bilinear')
            hm = A['heat'][j]
            vmh = max(float(hm.max()), 1e-6)
            a.imshow(np.ma.masked_where(~mk | (hm <= 0), hm), cmap='magma', vmin=0, vmax=vmh, extent=(0, G, G, 0),
                     interpolation='nearest')
            a.set_aspect('auto')
            cx0 = zx + zs / 2
            half = zs * hw / im / 2
            a.set_xlim(cx0 - half, cx0 + half); a.set_ylim(zy + zs, zy)
            for s in a.spines.values():
                s.set_color(C_N[j]); s.set_linewidth(0.9)
            cbar(xx, ut - 0.2, hw, 'magma', vmh)
            txt(xx, ut + 2.9, f'neuron {n}\n{DOM[d]["desc"][n]}', fontsize=FS_S, color=C_N[j], linespacing=1.1)
        # g frame read-out: sparse code, neuron index on y
        a = ax_mm(xs['g'] + 3.6, y0, widths['g'] - 3.6, im)
        code = A['code']
        nz = np.flatnonzero(code > 0)
        a.hlines(nz, 0, code[nz], color=C_GREY, lw=0.3)
        for j, n in enumerate(m['neurons']):
            a.hlines([n], 0, code[n], color=C_N[j], lw=1.0)
            a.plot([code[n]], [n], 'o', ms=1.5, color=C_N[j], mec='none')
        a.set_ylim(len(code), 0)
        a.set_xlim(0, code.max() * 1.08)
        a.set_yticks([0, len(code)]); a.set_yticklabels([])
        a.set_xticks([])
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        under('g', ut, f'{len(code)}-d\ncode: max\nover kept\n{m["n_active_frame_code"]} active', fontsize=FS_S)
        # h trace + per-video values
        hx = xs['h'] + 3.2
        hwid = widths['h'] - 3.2
        n0 = m['neurons'][0]
        a = ax_mm(hx, y0, hwid, 7.5)
        smooth = lambda v: pd.Series(v).rolling(50, center=True, min_periods=1).mean().values  # noqa: E731
        t_off, mids = 0.0, []
        for si, (sid, lo_, hi_) in enumerate(m['segs']):
            tr = A[f'trace{si}'][:, 0]
            tt = np.arange(len(tr)) / 300 + t_off
            if si == 0:
                a.axvspan(t_off + m['window_5fps'][0] / 300, t_off + m['window_5fps'][1] / 300, color='#ececec', lw=0)
            else:
                a.axvspan(tt[0], tt[-1], color='#ececec', lw=0)
            a.plot(tt, smooth(tr), color=C_N[0], lw=0.45)
            mids.append((tt[0] + tt[-1]) / 2)
            t_off = tt[-1] + 1.5
        a.axvline(m['frame_idx'] / 300, color=C_KEPT, lw=0.6)
        if len(mids) > 1:
            for mid, lab in zip(mids, ('H', 'O')):
                a.text(mid, 1.02, lab, transform=a.get_xaxis_transform(), fontsize=FS_S, ha='center', va='bottom')
        a.set_xlim(0, t_off - 1.5)
        a.set_ylim(bottom=0)
        a.locator_params(axis='y', nbins=2)
        a.locator_params(axis='x', nbins=3)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        a.set_ylabel(f'#{n0}', fontsize=FS_S, color=C_N[0], labelpad=0.5)
        a.set_xlabel('time (min)', fontsize=FS_S, labelpad=0.5)
        # per-video values
        b = ax_mm(hx, y0 + 17.5, hwid, 10.0)
        v = A['pv_values']
        rng = np.random.default_rng(0)
        ex = m['example_unit_index']
        if v.ndim == 3:  # paired (2, units, neurons)
            for u in range(v.shape[1]):
                b.plot([0, 1], v[:, u, 0], color=C_LIGHT, lw=0.3, zorder=1)
            for g_ in (0, 1):
                b.scatter(np.full(v.shape[1], g_), v[g_, :, 0], s=1.8, c=C_GREY if g_ == 0 else C_N[0], lw=0, zorder=2)
                b.hlines(np.median(v[g_, :, 0]), g_ - 0.25, g_ + 0.25, color='k', lw=0.7, zorder=3)
            b.scatter([0], [v[0, ex, 0]], s=6, facecolors='none', edgecolors=C_KEPT, lw=0.6, zorder=4)
        else:
            gr = A['pv_group']
            jit = rng.uniform(-0.18, 0.18, len(gr))
            for g_ in (0, 1):
                sel = gr == g_
                b.scatter(g_ + jit[sel], v[sel, 0], s=1.8, c=C_GREY if g_ == 0 else C_N[0], lw=0, zorder=2)
                b.hlines(np.median(v[sel, 0]), g_ - 0.28, g_ + 0.28, color='k', lw=0.7, zorder=3)
            b.scatter([gr[ex] + jit[ex]], [v[ex, 0]], s=6, facecolors='none', edgecolors=C_KEPT, lw=0.6, zorder=4)
        b.set_xlim(-0.55, 1.55)
        b.set_xticks([0, 1]); b.set_xticklabels(DOM[d]['groups'], fontsize=FS_S)
        b.set_ylim(bottom=0)
        b.locator_params(axis='y', nbins=3)
        for s in ('top', 'right'):
            b.spines[s].set_visible(False)
        b.set_ylabel('video mean', fontsize=FS_S, labelpad=0.5)
        st = m['nes'][str(n0)]
        sig = 'NES hit' if n0 in m['nes']['selected'] else 'not significant'
        b.text(0.5, 1.0, f'{sig}\nτ = {st["tau"]:.2f}, p = {fmt_p(st["p"])}', transform=b.transAxes,
               fontsize=FS_S, ha='center', va='bottom', linespacing=1.05)
    for fmt in ('pdf', 'png'):
        fig.savefig(OUT / f'fig_pipeline.{fmt}', dpi=args.dpi if fmt == 'png' else 600)
    log(f'saved {OUT}/fig_pipeline.pdf/.png ({W:.0f} x {H:.0f} mm)')


def edge(ax, mk, color, lw):
    """Outline of the union of kept patches (grid units)."""
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
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw))
    ax.set_xlim(0, G); ax.set_ylim(G, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('compute', 'plot', 'all'))
    ap.add_argument('--domains', default='ants,mice,frogs')
    ap.add_argument('--sae', default='', help='domain=sae_name,... overrides of the C1 SAEs')
    ap.add_argument('--encoder-label', default='', help='plot label of the encoder (default: from the cache)')
    ap.add_argument('--grid', type=int, default=0, help='patch-grid side drawn in c (default: from the cache)')
    ap.add_argument('--input-px', type=int, default=0, help='encoder input size drawn in c (default: from the cache)')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('run through Slurm (scripts/eci/fig_pipeline.sh): no compute or rendering on the login node')
    OUT.mkdir(parents=True, exist_ok=True)
    sae = dict(SAE, **dict(kv.split('=') for kv in args.sae.split(',') if kv))
    if args.stage in ('compute', 'all'):
        import torch
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
        tmp = os.environ.get('STAGE') or f'/localhome/{os.environ["USER"]}/{os.environ["SLURM_JOB_ID"]}'
        facts = {d: compute_domain(d, DOM[d], sae[d], tmp) for d in args.domains.split(',')}
        json.dump(facts, open(OUT / 'facts.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    if args.stage in ('plot', 'all'):
        plot(args)


if __name__ == '__main__':
    main()
