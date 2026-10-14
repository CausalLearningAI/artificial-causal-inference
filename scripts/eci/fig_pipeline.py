"""
Nature-style multi-panel figure of the ECI (Exploratory Causal Inference) REPRESENTATION "C1": one row per domain
(ants, mice, frogs), the same steps on a real example frame of that domain, from the raw video to the video-level
summaries of one behaviour-like neuron. Discovery (NES) is not shown.

    a  video              temporal volume: 9 frames evenly over the analysed clip (earliest at the back), thin time
                          axis (frame centre = its time), the example frame t (orange) replacing the nearest one
    b  one frame, steps   greyscale + single-hue overlays: (1) the 32 x 32 token grid of the whole frame + the mask
                          outline; (2) zoom, non-mask tokens blanked, the chosen token (the neuron's peak) outlined;
                          (3) its DINOv2 vector x_t and x_t - x_(t-1 s) as two slim 768-d glyphs; (4) the SAE's dense
                          ReLU pre-activations (4096) with the inference threshold -> the sparse code, neuron in blue;
                          (5) the neuron on every kept token (blue opacity). Orange arrow from the example frame in a.
    c  aggregation        arrow 'spatial (max)' -> the neuron's RAW value per frame over a 2-min window (--window-s)
                          + firing raster -> arrow 'temporal' -> time / bouts per min / latency over the whole clip from
                          the firing rule --event (default nonzero = frame value > 0; q95; hmm)
    All explanatory text lives in results/vision/eci_pipeline_fig/caption.md.
    sheets stage          top-frame sheets of candidate frog neurons (used to choose / describe the frog neuron)

Stages (Slurm only, scripts/eci/fig_pipeline.sh):
    compute  CPU: per domain the example = among 200 random eligible frames (seed 0), walked from the strongest
             codes_max of the neuron down, the first whose mask has no isolated piece (8-connected components >= 3
             tokens; else the strongest, flagged); its JPEGs (t, t - 1 s) are copied to $STAGE; encoder + mask + SAE per token (fg_encode.fg_sae_pool
             semantics); VERIFY the pooled max against the stored codes_max row and n_fg (latents flipping at the SAE
             inference threshold are counted: at most 3, never the shown neuron); the mapped native frame (start +
             step k + expect) decoded with ffmpeg (grey MAD to the stored JPEG over offsets -3..3 recorded as a check);
             ffprobe of every ants v2 / v3 source file (file length vs analysed clip); the clip trace from codes_max.npy;
             the three summaries recomputed and VERIFIED against the NES caches (bout counts, frames above, latency)
             -> OUT/cache/<domain>.npz/.json + OUT/facts.json
    plot     reads only the cache -> OUT/fig_pipeline.{pdf,png} (PNG at 300 dpi)

Encoder switch: the encoder label, the token-grid size and the input size are written to the cache by compute and can be
overridden at plot time (--encoder-label, --grid, --input-px). compute supports SAEs on DINOv2-family tokens
(src/eci/foreground.py FgEncoder, the deployed C1 path); a V-JEPA SAE needs its own token function.
Change the SAEs with --sae ants=<name>,mice=<name>,frogs=<name>.

Usage: sbatch scripts/eci/fig_pipeline.sh  (STAGE_ARG=compute | plot | all)
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
HMM = REPO / 'results/vision/eci_event_hmm'  # per-frame HMM P(ON) (scripts/eci/event_hmm.py), when present
MIN_PIECE, MAX_TRY = 3, 48
N_THUMB = 9  # temporal-volume frames over the analysed clip
# event_hmm per-video tables covering the same clip (mice: the table holds the whole 30-min video, not comparable)
HMM_REF = {'ants': ('bouts_186_groom_any.csv', 'hmm_mean_p', 'hmm'),
           'frogs': ('per_video.csv', 'mean_p_on', 'hmm_bouts_per_min')}  # clean-mask example search
Q_EVENT = 0.95  # firing threshold quantile (primary of run_nes_bouts.py / run_nes_latency.py)
FPS = 5.0

DOM = {
    'ants': dict(
        eci=REPO / 'dataset/ants/eci', nes=REPO / 'results/vision/ants/eci/nes', name='Ants',
        species=r'$\it{Lasius\ neglectus}$', raw_exp=REPO / 'data/ants/v2/experiment.csv',
        src=REPO / 'data/ants/v2/observations/source', step=6, expect=2, hflip=False, version='v2 (ISTAnt)',
        cand=lambda a: a[(a.experiment == 'v2') & ((a.Y_Y2F == 1) | (a.Y_B2F == 1)) & (a.frame_idx >= 10)],
        ann_cols=['observation_id', 'frame_idx', 'frame_path', 'experiment', 'Y_Y2F', 'Y_B2F'],
        short='grooming', key=186, desc='grooming\n(AUROC 0.86)', window='full',
        cand_text='annotated grooming frames (Y2F or B2F) of ants v2'),
    'mice': dict(
        eci=REPO / 'dataset/mice/v1/eci', ann=REPO / 'dataset/mice/v1/annotations.csv',
        nes=REPO / 'results/vision/mice/eci/nes', name='Mice', species=r'$\it{Mus\ musculus}$',
        raw_exp=REPO / 'data/mice/v1/experiment.csv', src=REPO / 'data/mice/source', step=6, expect=2, hflip=False,
        version='v1', ann_cols=['observation_id', 'frame_idx', 'frame_path', 'Y_nn'],
        short='two mice in contact', key=915, desc='two mice in contact\n(nose-nose AUROC 0.72)', window='last',
        cand_text='annotated nose-nose frames (Y_nn = 1) in the analysed last 15 min of social-odour habituation videos'),
    'frogs': dict(
        eci=REPO / 'dataset/frogs/eci', nes=REPO / 'results/vision/frogs/eci/nes', name='Frogs',
        species=r'$\it{Xenopus}$ froglets', raw_exp=REPO / 'data/frogs/v1/experiment.csv',
        src=REPO / 'data/frogs/v1/observations/source', step=12, expect=5, hflip=True, version='v1 (juvenile)',
        ann_cols=['observation_id', 'frame_idx', 'frame_path', 'group'],
        cand=lambda a: a[(a.group == 'FoxP1') & (a.frame_idx >= 10)],
        short='extended hind limb', key=500, desc='extended hind limb', window='full',
        cand_text='frames of FoxP1 (half-embryo crispant) froglets'),
}


# ------------------------------------------------------------------------------------------------ compute helpers
def ffprobe(path):
    o = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                   'stream=width,height,avg_frame_rate,nb_frames,codec_name:format=duration',
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


def ants_lengths():
    """File length (ffprobe) vs analysed clip (experiment.csv end - start) of every valid ants v2 / v3 video."""
    out = {}
    for v in ('v2', 'v3'):
        e = pd.read_csv(REPO / f'data/ants/{v}/experiment.csv')
        e = e[e['valid'] == 1]
        durs, fpss, sizes = [], [], set()
        for f in e.observation_file:
            p = ffprobe(REPO / f'data/ants/{v}/observations/source' / f)
            durs.append(p['duration_s'] / 60)
            fpss.append(p['fps'])
            sizes.add((p['width'], p['height']))
        clip = (e.end_frame - e.start_frame) / np.asarray(fpss) / 60
        out[v] = dict(n=len(e), file_min=[float(np.min(durs)), float(np.median(durs)), float(np.max(durs))],
                      fps=sorted({round(f, 3) for f in fpss}), sizes=sorted(sizes),
                      clip_min=[float(clip.min()), float(clip.max())], start_frames=sorted(e.start_frame.unique().tolist()))
        log(f'[ants {v}] {out[v]}')
    return out


def pieces(mk):
    """Sizes of the 8-connected components of a token mask."""
    from scipy import ndimage
    lab, n = ndimage.label(mk, structure=np.ones((3, 3)))
    return [int(v) for v in ndimage.sum(mk, lab, range(1, n + 1))] or [0]


def events(tr, thr):
    """Firing = tr > thr (strict). -> above mask, bout (start, end) runs, time budget, bouts / min, latency s."""
    above = tr > thr
    d = np.diff(np.r_[0, above.astype(np.int8), 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    minutes = len(tr) / FPS / 60
    lat = float(st[0] / FPS) if len(st) else float(len(tr) / FPS)
    return above, np.stack([st, en], 1), dict(time_budget=float(above.mean()), n_bouts=int(len(st)),
                                              bout_rate_per_min=float(len(st) / minutes), latency_s=lat,
                                              censored=not len(st), frac_nonzero=float((tr > 0).mean()),
                                              clip_min=float(minutes), threshold=float(thr))


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
    key = cfg['key']
    ann = pd.read_csv(ann_path, usecols=cfg['ann_cols'])
    ranges = obs_rows(ann_path)
    codes = np.load(cdir / 'codes_max.npy', mmap_mode='r')
    nfg_all = np.load(cdir / 'n_fg.npy', mmap_mode='r')
    assert codes.shape[0] == len(ann), (codes.shape, len(ann))
    # ---------------------------------------------------------------- 1. example frame
    if name == 'mice':
        e = pd.read_csv(cfg['raw_exp'])
        hab = set(e[(e.odor == 'S') & (e.phase == 'H')].observation_id)
        cand = ann[ann.observation_id.isin(hab) & (ann.Y_nn == 1) & (ann.frame_idx >= 4500 + D)]
    else:
        cand = cfg['cand'](ann)
    cand = cand.sample(min(n_cand, len(cand)), random_state=0)
    rows = np.sort(cand.index.values)
    kv = np.asarray(codes[rows][:, key], dtype=np.float32)
    order = rows[np.argsort(-kv)]
    enc = FgEncoder(encoder, 'cpu')
    bgs = FgBackgrounds(ccfg['backgrounds'], ranges, RULES[rule_name], 'cpu')

    def encode_rows(rr, tag):
        loc = []
        for r in rr:
            p = Path(tmp) / tag / ann.frame_path.iloc[r].replace('/', '_')
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / 'dataset' / ann.frame_path.iloc[r], p)
            loc.append(str(p))
        ds = enc.dataset(loc)
        pix = torch.stack([ds[i][0] for i in range(len(ds))])
        grey = torch.stack([ds[i][1] for i in range(len(ds))])
        tok = torch.cat([encode_batch(enc.model, pix[a:a + 8], 'cpu') for a in range(0, len(ds), 8)])
        return loc, pix, grey, tok

    # walk the candidates from the strongest neuron value down; take the first whose mask has no isolated piece
    # (every 8-connected component >= MIN_PIECE tokens), up to MAX_TRY candidates; else the strongest (flagged)
    tried, row, pool = [], None, []  # pool: kept tokens of the encoded candidates (PCA fit)
    for a in range(0, min(MAX_TRY, len(order)), 8):
        rr = [int(r) for r in order[a:a + 8]]
        _, _, grey_b, tok_b = encode_rows(rr, 'try')
        for k, r in enumerate(rr):
            mk = bgs.mask(tok_b[k:k + 1], grey_b[k:k + 1], np.array([r]))[0][0].numpy().reshape(-1)
            sizes = pieces(mk.reshape(int(np.sqrt(mk.size)), -1))
            pool.append(tok_b[k][torch.from_numpy(mk)].float())
            tried.append(dict(row=r, value=float(codes[r, key]), pieces=sizes))
            if row is None and min(sizes) >= MIN_PIECE:
                row = r
        if row is not None:
            break
    clean = row is not None
    if not clean:
        row = int(order[0])
    rank = [t['row'] for t in tried].index(row) + 1 if clean else 1
    obs, fidx = ann.observation_id.iloc[row], int(ann.frame_idx.iloc[row])
    lo, hi = ranges[obs]
    log(f'[{name}] {len(cand)} candidates, neuron {key} codes_max median {np.median(kv):.2f} max {kv.max():.2f}; '
        f'clean mask {clean} at strength rank {rank} -> row {row} ({obs} frame {fidx}, value {float(codes[row, key]):.2f}); '
        f'tried {[(t["value"], t["pieces"]) for t in tried[:rank + 2]]}')
    # ---------------------------------------------------------------- 2. encoder + mask + SAE on the frame
    want = [row, row - D]  # t, t - 1 s (D = 5 frames at 5 fps)
    assert min(want) >= lo
    loc, pix, grey, tok = encode_rows(want, 'jpg')  # (2, 1024, d) fp16: the WHOLE frame is encoded, then masked
    assert tok.shape[1] == GRID * GRID
    mask = bgs.mask(tok[:1], grey[:1], np.array([row]))[0][0]  # (1024,) bool
    mask_p = bgs.mask(tok[1:2], grey[1:2], np.array([row - D]))[0][0]  # the mask of the t - 1 s frame (display only)
    # PCA-RGB of the DINOv2 tokens: PCA on the kept tokens of the encoded candidate frames + the two shown frames,
    # first 3 components -> RGB, each scaled to [0, 1] by its 1st / 99th percentile over the fit tokens
    fit = torch.cat(pool + [tok[0][mask].float(), tok[1][mask_p].float()])
    mu = fit.mean(0)
    _, _, V = torch.linalg.svd(fit - mu, full_matrices=False)
    P = V[:3].T
    pf = ((fit - mu) @ P).numpy()
    lo_q, hi_q = np.percentile(pf, 1, axis=0), np.percentile(pf, 99, axis=0)
    pca_rgb = [np.clip((((tok[i].float() - mu) @ P).numpy() - lo_q) / (hi_q - lo_q), 0, 1).reshape(GRID, GRID, 3)
               for i in (0, 1)]
    pca_n = int(fit.shape[0])
    stok = enc.sae_tokens(tok)
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
    thr_sae = float(sae.threshold)
    # threshold inference (z = pre * (pre > theta)): a CPU vs GPU rounding difference can flip a latent whose
    # pre-activation sits at theta, a jump of ~theta. Those flips are counted separately, not hidden.
    flip = ((mx16 == 0) | (stored == 0)) & (np.maximum(mx16, stored) < 1.15 * thr_sae)
    diff = float(diff_all[~flip].max()) if (~flip).any() else 0.0
    n_kept = int(mask.sum())
    verify = dict(max_abs_diff_vs_codes_max=diff, n_threshold_flips=int((flip & (diff_all > 0)).sum()),
                  sae_threshold=thr_sae, max_abs_diff_incl_flips=float(diff_all.max()),
                  n_kept=n_kept, n_fg_stored=int(nfg_all[row]),
                  n_active_recomputed=int((mx > 0).sum()), n_active_stored=int((stored > 0).sum()))
    log(f'[{name}] verify frame code {verify}')
    if (n_kept != int(nfg_all[row]) or diff > 0.05 * max(1.0, float(stored.max())) or verify['n_threshold_flips'] > 3
            or (flip[key] and diff_all[key] > 0)):
        raise RuntimeError(f'[{name}] recomputed frame code does not match the deployed codes: {verify}')
    motion = (stok[0].float() - stok[1].float()).norm(dim=1).numpy()  # |x_t - x_(t-D)| per token
    heat = Z[:, key].numpy().reshape(GRID, GRID)
    sel = int(torch.argmax(Z[:, key]))  # the kept token where the neuron peaks
    tok_code = Z[sel].numpy()
    # the chosen token's SAE input [x_t, x_t - x_(t-1 s)] and its dense pre-activations (ReLU), for the walkthrough
    xin = torch.cat([stok[0][sel:sel + 1], (stok[0][sel:sel + 1].float() - stok[1][sel:sel + 1].float()).half()], 1)
    with torch.no_grad():
        pre = torch.relu(sae.pre_acts(norm(xin)))[0].float()
    z_chk = (pre * (pre > sae.threshold)).numpy()
    assert np.abs(z_chk - tok_code).max() < 1e-4, np.abs(z_chk - tok_code).max()
    tok_x, tok_d = stok[0][sel].float().numpy(), (stok[0][sel].float() - stok[1][sel].float()).numpy()
    pre = pre.numpy()
    n_topk = int(sae.k)
    # ---------------------------------------------------------------- 3. native frame + lengths
    e = pd.read_csv(cfg['raw_exp']).set_index('observation_id')
    srcf = cfg['src'] / e.loc[obs, 'observation_file']
    probe = ffprobe(srcf)
    start, end = int(e.loc[obs, 'start_frame']), int(e.loc[obs, 'end_frame'])
    hflip = bool(cfg['hflip'] and int(e.loc[obs, 'hflip']) == 1)
    centre = start + cfg['step'] * fidx + cfg['expect']
    nat = decode_native(srcf, centre - 3, probe['fps'], tmp)
    if hflip:
        nat = [np.ascontiguousarray(a[:, ::-1]) for a in nat]
    ref = np.asarray(Image.open(loc[0]).convert('L')).astype(np.float32)
    mad = [float(np.abs(np.asarray(Image.fromarray(a).convert('L').resize((512, 512), Image.BICUBIC)).astype(np.float32)
                        - ref).mean()) for a in nat]
    off = int(np.argmin(mad)) - 3
    native = nat[3]  # the mapped frame; the MAD argmin is recorded as a check
    log(f'[{name}] native {native.shape[1]}x{native.shape[0]} {probe}; MAD -3..3 {np.round(mad, 2).tolist()} '
        f'-> argmin source frame {centre + off} (shown: mapped {centre})')
    nat_small = np.asarray(Image.fromarray(native).resize((768, 768), Image.LANCZOS)) if native.shape[0] > 768 else native
    frames = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in loc])  # t, t - 1 s (512)
    in_px = int(pix.shape[-1])
    enc_in = np.asarray(Image.fromarray(frames[0]).resize((in_px, in_px), Image.BICUBIC))
    lengths = dict(file_min=probe['duration_s'] / 60, standardised_min=(end - start) / probe['fps'] / 60)
    if name == 'ants':
        lengths['per_version'] = ants_lengths()
    # ---------------------------------------------------------------- 4. clip trace + temporal summaries
    n_frames = hi - lo
    w0, w1 = (n_frames - 4500, n_frames) if cfg['window'] == 'last' else (0, n_frames)
    tr = np.asarray(codes[lo + w0:lo + w1, key], dtype=np.float32)
    nes = cfg['nes'] / sae_name / '_cache'
    th = np.load(nes / f'bout_thresholds_n2000000.npz')
    qi = int(np.argmin(np.abs(th['qs'] - Q_EVENT)))
    thr = float(th['thr'][qi, key])
    above, bouts, summ = events(tr, thr)
    bs = np.load(nes / 'bout_summaries_max.npz')
    vi = list(bs['observation_id']).index(obs)
    w = cfg['window']
    cnt = int(bs[f'{w}__{Q_EVENT}__0__count'][vi, key])
    frm = int(bs[f'{w}__{Q_EVENT}__0__frames'][vi, key])
    la = np.load(nes / 'latency.npz')
    lat = float(la[f'{w}__{Q_EVENT:.2f}'][list(la['observation_id']).index(obs), key]) / FPS  # cache: frames
    vs = np.load(nes / 'video_summaries_max.npz')
    rate0 = float(vs[f'{w}__rate'][list(vs['observation_id']).index(obs), key])
    lengths['clip_min'] = summ['clip_min']
    ncl = w1 - w0
    t_rel = fidx - w0
    tpos = [int((i + 0.5) * ncl / N_THUMB) for i in range(N_THUMB)]
    j_ex = int(np.argmin([abs(q - t_rel) for q in tpos]))
    tpos[j_ex] = t_rel
    thumbs = []
    for q in tpos:
        pth = Path(tmp) / 'thumb' / ann.frame_path.iloc[lo + w0 + q].replace('/', '_')
        pth.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / 'dataset' / ann.frame_path.iloc[lo + w0 + q], pth)
        thumbs.append(np.asarray(Image.open(pth).convert('RGB').resize((192, 192), Image.LANCZOS)))
    thumbs = np.stack(thumbs)
    hmm = None
    hp = HMM / name / 'posteriors.npz'
    if hp.exists():
        hz = np.load(hp)
        if key in list(hz['neurons']):
            hrows = hz['rows']
            pos = np.searchsorted(hrows, np.arange(lo + w0, lo + w1))
            ok = (pos < len(hrows)) & (hrows[np.minimum(pos, len(hrows) - 1)] == np.arange(lo + w0, lo + w1))
            if ok.all():
                pon = np.asarray(hz['hmm'][pos, list(hz['neurons']).index(key)], dtype=np.float32)
                _, hb, hs = events(pon, 0.5)
                hs['time_budget'] = float(pon.mean())  # mean P(ON); bouts / latency at P(ON) > 0.5
                hmm = dict(p_on=pon, bouts=hb, summary=hs, file=str(hp.relative_to(REPO)))
                log(f'[{name}] HMM P(ON) clip: {hs}')
                ref = HMM_REF.get(name)
                if ref is not None and (HMM / name / ref[0]).exists():
                    t = pd.read_csv(HMM / name / ref[0])
                    t = t[(t.obs == obs) & ((t.neuron == key) if 'neuron' in t else True)]
                    if len(t):
                        rm, rb = float(t[ref[1]].iloc[0]), float(t[ref[2]].iloc[0])
                        mine_b = hs['n_bouts'] if ref[2] == 'hmm' else hs['bout_rate_per_min']
                        hmm['check'] = dict(ref_mean_p=rm, mean_p=hs['time_budget'], ref_bouts=rb, bouts=mine_b)
                        log(f'[{name}] HMM check vs {ref[0]}: {hmm["check"]}')
                        if abs(rm - hs['time_budget']) > 1e-3 or abs(rb - mine_b) > 1e-3:
                            raise RuntimeError(f'[{name}] HMM summaries do not match {ref[0]}: {hmm["check"]}')
            else:
                log(f'[{name}] HMM posteriors do not cover the clip ({int(ok.sum())} / {len(ok)} frames)')
        else:
            log(f'[{name}] HMM posteriors have no neuron {key} ({list(hz["neurons"])})')
    else:
        log(f'[{name}] no HMM posteriors at {hp}')
    chk = dict(cache_bouts=cnt, cache_frames_above=frm, cache_latency_s=lat, cache_frac_nonzero=rate0,
               bouts=summ['n_bouts'], frames_above=int(above.sum()), latency_s=summ['latency_s'],
               frac_nonzero=summ['frac_nonzero'])
    log(f'[{name}] clip {summ["clip_min"]:.1f} min, thr {thr:.3f}: {summ}; cache check {chk}')
    if (cnt != summ['n_bouts'] or frm != int(above.sum()) or abs(lat - summ['latency_s']) > 1e-6
            or abs(rate0 - summ['frac_nonzero']) > 1e-6):
        raise RuntimeError(f'[{name}] temporal summaries do not match the NES caches: {chk}')
    meta = dict(domain=name, sae=sae_name, encoder=encoder,
                encoder_label='DINOv2-base' if encoder == 'dinov2_base' else encoder,
                grid=GRID, input_px=in_px, frame_px=512, motion_delta=D, fg_rule=rule_name, row=row, obs=obs,
                frame_idx=fidx, n_kept=n_kept, n_patches=GRID * GRID, key=key, desc=cfg['desc'],
                selection=f'highest codes_max of neuron {key} among {len(cand)} random {cfg["cand_text"]} (seed 0): '
                          f'{kv.max():.2f} (candidate median {np.median(kv):.2f})',
                native=dict(probe, frame=int(centre), mad_argmin_frame=int(centre + off), mad_argmin_offset=off, mad=mad,
                            hflip=hflip, size=[int(native.shape[1]), int(native.shape[0])], file=str(srcf.name)),
                lengths=lengths, window=[int(w0), int(w1)], window_name=w, video_frames_5fps=int(n_frames),
                verify=verify, summary_check=chk, frame_code_value=float(mx[key]),
                n_active_frame_code=int((mx > 0).sum()), n_active_tokens_neuron=int((Z[:, key] > 0).sum()),
                sel_token=[sel // GRID, sel % GRID], n_active_token_code=int((tok_code > 0).sum()),
                token_code_value=float(tok_code[key]),
                motion_kept_median=float(np.median(motion[mask.numpy()])), motion_kept_max=float(motion[mask.numpy()].max()),
                summary=summ, hmm_summary=None if hmm is None else hmm['summary'],
                hmm_file=None if hmm is None else hmm['file'], hmm_check=None if hmm is None else hmm.get('check'), clean_mask=clean, clean_rank=rank,
                tried=tried[:rank + 2], sel_motion=float(motion[sel]), sae_k=n_topk, sae_theta=thr_sae,
                n_pre_positive=int((pre > 0).sum()), pca_fit_tokens=pca_n,
                pca_fit_frames=len(pool) + 2, n_kept_prev=int(mask_p.sum()),
                version=cfg['version'], name=cfg['name'], species=cfg['species'])
    OUT.joinpath('cache').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'cache' / f'{name}.npz', native=nat_small, frames=frames, enc_in=enc_in,
                        mask=mask.numpy().reshape(GRID, GRID), motion=motion.reshape(GRID, GRID), heat=heat, code=mx,
                        tok_code=tok_code, trace=tr, above=above, bouts=bouts, pca_t=pca_rgb[0], pca_p=pca_rgb[1],
                        thumbs=thumbs, thumb_pos=np.array(tpos), thumb_ex=j_ex, tok_x=tok_x, tok_d=tok_d, pre=pre,
                        mask_p=mask_p.numpy().reshape(GRID, GRID),
                        **({} if hmm is None else dict(p_on=hmm['p_on'], hmm_bouts=hmm['bouts'])))
    json.dump(meta, open(OUT / 'cache' / f'{name}.json', 'w'), indent=1,
              default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    log(f'[{name}] done in {time.time() - t0:.0f}s')
    return meta


def source_sizes():
    """ffprobe of every valid source video of the shown experiments (ants v2, mice v1, frogs v1) -> size / fps
    counts per domain -> OUT/source_sizes.json (read by the plot for the line under the video stack)."""
    from collections import Counter
    out = {}
    for d in ('ants', 'mice', 'frogs'):
        cfg = DOM[d]
        e = pd.read_csv(cfg['raw_exp'])
        if 'valid' in e:
            e = e[e['valid'] == 1]
        sizes, fpss = Counter(), Counter()
        for f in e.observation_file:
            q = ffprobe(cfg['src'] / f)
            sizes[f"{q['width']}x{q['height']}"] += 1
            fpss[f"{q['fps']:.2f}"] += 1
        out[d] = dict(n=int(len(e)), sizes=dict(sizes.most_common()), fps=dict(fpss.most_common()))
        log(f'[sizes] {d}: {out[d]}')
    json.dump(out, open(OUT / 'source_sizes.json', 'w'), indent=1)


def frog_sheets(tmp, n_top=12, stride=25):
    """Top-frame sheets of candidate frog neurons (C1 frogs SAE) to choose a neuron with a clear visual meaning:
    500 and 147 + the 6 most frequent neurons of prefix 1024 + the 4 most frequent of those whose decoder relies on the
    1-s change half (change share > 0.5). Top frames = highest codes_max on every `stride`-th frame, one per video;
    each tile = 128 px crop of the 512 frame around the neuron's peak token, active tokens outlined (recomputed
    with the encoder + mask + SAE). -> OUT/frog_sheets/n<neuron>.jpg + index.json"""
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    from src.eci.foreground import GRID, RULES, FgBackgrounds, FgEncoder, encode_batch, obs_rows
    from src.eci.sae import load_sae
    cfg, name = DOM['frogs'], SAE['frogs']
    eci = cfg['eci']
    ann_path = eci / 'annotations.csv'
    ann = pd.read_csv(ann_path, usecols=['observation_id', 'frame_idx', 'frame_path'])
    ranges = obs_rows(ann_path)
    sae, norm, ck = load_sae(eci / 'sae' / name / 'sae.pt')
    D = int(ck['motion_delta'])
    W = sae.W_dec.detach().float()
    share = ((W[:, 768:] ** 2).sum(1) / (W ** 2).sum(1)).numpy()
    vs = np.load(cfg['nes'] / name / '_cache/video_summaries_max.npz')
    rate = vs['full__rate'].mean(0)
    o = [int(i) for i in np.argsort(-rate[:1024])]
    cands = [500, 147] + [i for i in o if i not in (500, 147)][:6]
    cands += [i for i in o if share[i] > 0.5 and i not in cands][:4]
    log(f'[sheets] candidates {[(c, round(float(rate[c]), 2), round(float(share[c]), 2)) for c in cands]}')
    codes = np.load(eci / 'codes' / name / 'codes_max.npy', mmap_mode='r')
    rows = np.arange(0, len(ann), stride)
    sub = np.asarray(codes[rows][:, cands], dtype=np.float32)
    obs = ann.observation_id.values[rows]
    enc = FgEncoder('dinov2_base', 'cpu')
    ccfg = json.load(open(eci / 'codes' / name / 'config.json'))
    bgs = FgBackgrounds(ccfg['backgrounds'], ranges, RULES[ck['fg_rule']], 'cpu')
    out = OUT / 'frog_sheets'
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for j, n in enumerate(cands):
        order = np.argsort(-sub[:, j])
        pick, seen = [], set()
        for i in order:
            if obs[i] in seen or ann.frame_idx.values[rows[i]] < D:
                continue
            seen.add(obs[i]); pick.append(int(rows[i]))
            if len(pick) == n_top:
                break
        loc = []
        for r in pick + [r - D for r in pick]:
            p = Path(tmp) / 'sheet' / ann.frame_path.iloc[r].replace('/', '_')
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / 'dataset' / ann.frame_path.iloc[r], p)
            loc.append(str(p))
        ds = enc.dataset(loc)
        pix = torch.stack([ds[i][0] for i in range(len(ds))])
        grey = torch.stack([ds[i][1] for i in range(len(ds))])
        tok = torch.cat([encode_batch(enc.model, pix[a:a + 8], 'cpu') for a in range(0, len(ds), 8)])
        fig, axs = plt.subplots(2, n_top // 2, figsize=(12, 4.4))
        for k, r in enumerate(pick):
            m = bgs.mask(tok[k:k + 1], grey[k:k + 1], np.array([r]))[0][0]
            x = torch.cat([tok[k][m], (tok[k][m].float() - tok[n_top + k][m].float()).half()], 1)
            with torch.no_grad():
                zz = sae.encode(norm(x), mode='threshold')[:, n].float()
            act = torch.zeros(GRID * GRID); act[m] = zz
            pk = int(act.argmax()); py, px = divmod(pk, GRID)
            cy, cx = int(np.clip(py * 16 + 8 - 64, 0, 384)), int(np.clip(px * 16 + 8 - 64, 0, 384))
            im = np.asarray(Image.open(loc[k]).convert('RGB'))[cy:cy + 128, cx:cx + 128]
            ax = axs.flat[k]
            ax.imshow(im)
            for q in np.flatnonzero(act.numpy() > 0):
                qy, qx = divmod(int(q), GRID)
                ax.add_patch(Rectangle((qx * 16 - cx, qy * 16 - cy), 16, 16, fill=False, lw=1.2,
                                       ec='yellow' if q == pk else 'cyan'))
            ax.set_title(f'{ann.observation_id.iloc[r]} f{ann.frame_idx.iloc[r]} {float(act.max()):.1f}', fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'frogs C1 neuron {n}: firing rate {rate[n]:.2f}, change share {share[n]:.2f} '
                     f'(cyan = active kept tokens, yellow = peak)', fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f'n{n:04d}.jpg', dpi=110)
        plt.close(fig)
        index[n] = dict(rate=float(rate[n]), change_share=float(share[n]), rows=pick)
        log(f'[sheets] neuron {n} done')
    json.dump(index, open(out / 'index.json', 'w'), indent=1)


# ------------------------------------------------------------------------------------------------ plotting
MM = 1 / 25.4
C_KEPT = '#E69F00'   # accent: kept tokens / selected token (Okabe-Ito orange)
C_N = '#CC79A7'      # the shown neuron, everywhere (Okabe-Ito reddish purple / pink)
C_TOK = '#0072B2'    # the chosen token (Okabe-Ito blue)
C_GREY, C_LIGHT, C_DARK = '#8c8c8c', '#d9d9d9', '#4d4d4d'
FS, FS_S = 6.0, 5.2


def zoom_box(mk, margin=1, min_side=5):
    """Square token-grid box (x0, y0, side) around the kept tokens (stray 1-2 token pieces ignored), for e and f."""
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
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, snap=False))  # no pixel snapping: on the token grid
    ax.set_xlim(0, G); ax.set_ylim(G, 0)


EVENT_RULES = ('nonzero', 'q95', 'hmm')  # add e.g. 'topk' here and in firing()


def firing(A, m, rule):
    """Firing frames of the shown neuron over the whole analysed clip + the three video-level summaries.
    nonzero: frame value (max over kept tokens) > 0, i.e. active on >= 1 kept token; q95: frame value > the neuron's
    q 0.95 threshold; hmm: P(ON) > 0.5 (time budget = mean P(ON)). Bout = run of consecutive firing frames."""
    tr = A['trace']
    if rule == 'nonzero':
        fire = tr > 0
        tb = float(fire.mean())
        assert abs(tb - m['summary']['frac_nonzero']) < 1e-9, (tb, m['summary']['frac_nonzero'])  # = NES 'rate'
    elif rule == 'q95':
        fire = tr > m['summary']['threshold']
        tb = float(fire.mean())
    elif rule == 'hmm':
        fire = A['p_on'] > 0.5
        tb = float(A['p_on'].mean())
    else:
        raise ValueError(rule)
    d = np.diff(np.r_[0, fire.astype(np.int8), 0])
    st = np.flatnonzero(d == 1)
    minutes = len(tr) / FPS / 60
    lat = float(st[0] / FPS) if len(st) else float(len(tr) / FPS)
    return fire, dict(time_budget=tb, n_bouts=int(len(st)), bout_rate_per_min=float(len(st) / minutes), latency_s=lat,
                      censored=not len(st), clip_min=float(minutes))


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import to_rgb
    from matplotlib.patches import ConnectionPatch, FancyArrowPatch
    fams = [f for f in ('Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans')
            if any(f == x.name for x in font_manager.fontManager.ttflist)]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': fams, 'font.size': FS, 'axes.linewidth': 0.4,
                         'xtick.major.width': 0.4, 'ytick.major.width': 0.4, 'xtick.major.size': 1.2,
                         'ytick.major.size': 1.2, 'xtick.labelsize': FS_S, 'ytick.labelsize': FS_S,
                         'xtick.major.pad': 0.8, 'ytick.major.pad': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'mathtext.default': 'regular', 'mathtext.fontset': 'custom', 'mathtext.it': f'{fams[0]}:italic',
                         'mathtext.rm': fams[0], 'axes.labelpad': 0.8})
    log(f'font: {fams[0]}')
    doms = args.domains.split(',')
    data = {d: (dict(np.load(OUT / 'cache' / f'{d}.npz')), json.load(open(OUT / 'cache' / f'{d}.json'))) for d in doms}
    m0 = data[doms[0]][1]
    enc_label = args.encoder_label or m0['encoder_label'].split('-')[0]
    G = args.grid or m0['grid']
    C_DX = '#009E73'  # token-difference hue (Okabe-Ito bluish green)
    # ------------------------------------------------ layout (mm)
    W = 183.0
    left, top_h, row_h, sz = 7.0, 11.0, 23.5, 15.0
    ag = 2.8                    # arrow gap between steps
    fw, nvol, dx, dy = 10.0, 9, 1.9, 0.5  # temporal volume
    gw, gg, bw = 2.0, 0.5, 7.0  # vector glyph strip width, gap between the two strips, sparse-bar panel width
    VPAD = 5.0                  # extra width of the glyph column (room for its title)
    xa = left
    wa = fw + (nvol - 1) * dx
    xb = xa + wa + 5.0
    xs, x = {}, xb
    for k, w in (('grid', sz), ('zoom', sz), ('vec', 2 * gw + gg + 1.4 + VPAD), ('pre', bw + 2.0 + 3.5), ('spatial', sz)):
        xs[k] = x
        x += w + ag
    x += 9.5 - ag
    xs['trace'] = x
    xs['num'] = W - 0.5 - 11.0
    wt = xs['num'] - 4.8 - xs['trace']
    H = top_h + row_h * len(doms) + 0.5
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

    def arrow(x0, x1, yc, label=None, color='#9e9e9e'):
        fig.patches.append(FancyArrowPatch((x0 / W, 1 - yc / H), (x1 / W, 1 - yc / H), transform=fig.transFigure,
                                           arrowstyle='-|>', mutation_scale=4.5, lw=0.5, color=color,
                                           shrinkA=0, shrinkB=0))
        if label:
            txt((x0 + x1) / 2, yc - 0.6, label, ha='center', va='bottom', fontsize=FS_S - 0.6, color=C_DARK)

    def grey(frame):
        return np.asarray(frame, np.float32).mean(2) / 255

    def overlay(ax, g, val, mk, color, vmax, amax=0.85, interp='bilinear'):
        ax.imshow(g, cmap='gray', vmin=0, vmax=1, extent=(0, G, G, 0), interpolation=interp)
        rgba = np.zeros((G, G, 4))
        rgba[..., :3] = to_rgb(color)
        rgba[..., 3] = np.where(mk, amax * np.clip(val / max(vmax, 1e-9), 0, 1), 0)
        ax.imshow(rgba, extent=(0, G, G, 0), interpolation='nearest')

    def hue_bar(x, y, w, color, vmax):
        a = ax_mm(x, y, w, 0.7)
        rgba = np.zeros((1, 128, 4)); rgba[..., :3] = to_rgb(color); rgba[..., 3] = np.linspace(0, 0.85, 128)
        a.imshow(np.ones((1, 128, 3)), aspect='auto', extent=(0, 1, 0, 1))
        a.imshow(rgba, aspect='auto', extent=(0, 1, 0, 1))
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.25)
        txt(x + w + 0.4, y - 0.15, f'{vmax:.0f}', fontsize=FS_S - 0.6, ha='left')

    def bars(ax, v, n, hi=None, thr=None, sparse=False):
        """vertical sparse / dense bar glyph of a 4096-d vector (latent 1 at the top)"""
        idx = np.flatnonzero(v > 0)
        if sparse:
            ax.hlines(idx, 0, v[idx], color='#5a5a5a', lw=0.5)
        else:
            ax.hlines(idx, 0, v[idx], color='#b0b0b0', lw=0.12)
            if thr is not None:
                k = idx[v[idx] > thr]
                ax.hlines(k, 0, v[k], color='#5a5a5a', lw=0.4)
                ax.axvline(thr, color='k', lw=0.35, ls=(0, (1.5, 1.2)))
        if hi is not None and v[hi] > 0:
            ax.hlines([hi], 0, v[hi], color=C_N, lw=0.9)
            ax.plot([v[hi]], [hi], 'o', ms=1.3, color=C_N, mec='none')
        ax.set_ylim(n, 0); ax.set_xlim(0, float(v.max()) * 1.1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    # headers
    for xx, lab, title in ((xa, 'a', 'Video'), (xb, 'b', 'Frame representation'),
                           (xs['spatial'], 'c', 'Neural activation')):
        txt(xx, 0.6, lab, fontsize=8, weight='bold')
        txt(xx + 2.6, 0.85, title, fontsize=FS)
    for k, title in (('grid', f'{G} × {G} tokens'), ('zoom', 'animals mask'),
                     ('vec', f'{enc_label} token\nrepresentation'), ('pre', 'SAE (top-k)\nrepresentation'),
                     ('spatial', 'frame neural\nactivation'), ('num', 'per video')):
        txt(xs[k], 4.1, title, color=C_DARK)
    summ_out = {}
    sp = OUT / 'source_sizes.json'  # scripts/eci/fig_pipeline.py sizes (ffprobe of every source video)
    sizes = json.load(open(sp)) if sp.exists() else {}
    for ri, d in enumerate(doms):
        A, m = data[d]
        y0 = top_h + ri * row_h
        yc = y0 + sz / 2
        key = m['key']
        mk = A['mask'].astype(bool)
        zx, zy, zs = zoom_box(mk)
        sy, sx_ = m['sel_token']
        gt = grey(A['frames'][0])
        txt(0.4, yc, m['name'], fontsize=6.5, weight='bold', rotation=90, va='center', ha='left')
        txt(3.4, yc, m['species'], fontsize=FS_S - 0.6, rotation=90, va='center', ha='left')
        # ---- a: temporal volume over the clip; t outlined, arrow from t into b
        clip_fr = m['window'][1] - m['window'][0]
        clip_min = clip_fr / FPS / 60
        tpos, j_ex = A['thumb_pos'], int(A['thumb_ex'])
        top0 = y0 + 0.4
        for i in range(len(tpos)):
            a = img(xa + i * dx, top0 + i * dy, fw, ec=C_KEPT if i == j_ex else '#ffffff', lw=0.9 if i == j_ex else 0.4)
            if i == j_ex:
                ex_ax = a
            a.imshow(grey(A['thumbs'][i]), cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
        ya = top0 + (len(tpos) - 1) * dy + fw + 0.9
        ta = ax_mm(xa + fw / 2 - dx / 2, ya, len(tpos) * dx, 0.01)  # frame i centre = its time
        ta.set_xlim(0, clip_min); ta.set_yticks([])
        for s in ('top', 'right', 'left'):
            ta.spines[s].set_visible(False)
        ta.set_xticks([0, clip_min]); ta.set_xticklabels(['0', f'{clip_min:.0f} min'])
        ta.tick_params(length=1.0, labelsize=FS_S - 0.6)
        ta.plot([tpos[j_ex] / FPS / 60], [0], marker='|', ms=3.0, mew=0.8, color=C_KEPT, clip_on=False)
        sz_d = sizes.get(d, {})
        wh = next(iter(sz_d.get('sizes', {})), f'{m["native"]["size"][0]}x{m["native"]["size"][1]}').replace('x', '×')
        fps0 = float(next(iter(sz_d.get('fps', {})), m['native']['fps']))
        txt(xa, y0 - 5.0, f'original: {wh} px, {fps0:.0f} fps\nused: {m["input_px"]}×{m["input_px"]} px, {FPS:.0f} fps',
            fontsize=FS_S - 0.2, color=C_DARK)
        x0 = xa + j_ex * dx + fw * 0.55
        y_top = top0 + j_ex * dy
        fig.patches.append(FancyArrowPatch((x0 / W, 1 - y_top / H), ((xs['grid'] - 0.4) / W, 1 - (y0 + 1.5) / H),
                                           transform=fig.transFigure, arrowstyle='-|>', mutation_scale=4.5, lw=0.6,
                                           color=C_KEPT, connectionstyle='arc3,rad=-0.12', shrinkA=0, shrinkB=0))
        # ---- b1: the frame split into its token grid + the animal mask (outline)
        a = img(xs['grid'], y0, sz)
        a.imshow(gt, cmap='gray', vmin=0, vmax=1, extent=(0, G, G, 0), interpolation='bilinear')
        for i in range(1, G):
            a.axhline(i, color='w', lw=0.1, alpha=0.55)
            a.axvline(i, color='w', lw=0.1, alpha=0.55)
        edge(a, mk, C_KEPT, 0.5)
        arrow(xs['grid'] + sz + 0.3, xs['zoom'] - 0.3, yc)
        # ---- b2: zoom on the animals, non-mask tokens blanked, the chosen token outlined
        a = img(xs['zoom'], y0, sz)
        up = np.kron(mk, np.ones((512 // G, 512 // G), bool))
        a.imshow(np.where(up, gt, 0.93), cmap='gray', vmin=0, vmax=1, extent=(0, G, G, 0), interpolation='nearest')
        for i in range(1, G):
            a.axhline(i, color='w', lw=0.15, alpha=0.6)
            a.axvline(i, color='w', lw=0.15, alpha=0.6)
        edge(a, mk, C_KEPT, 0.4)
        a.add_patch(matplotlib.patches.Rectangle((sx_, sy), 1, 1, fill=False, ec=C_TOK, lw=0.9))
        a.set_xlim(zx, zx + zs); a.set_ylim(zy + zs, zy)
        zoom_ax = a
        # ---- b3: the chosen token -> [x_t, x_t - x_(t-1 s)] as two slim 768-d glyphs
        vy, vh = y0 + 0.6, sz - 3.2
        lim = float(np.percentile(np.abs(np.r_[A['tok_x'], A['tok_d']]), 98))
        gl = []
        for j, (v, ec) in enumerate(((A['tok_x'], C_DARK), (A['tok_d'], C_DARK))):
            g_ = ax_mm(xs['vec'] + 0.7 + VPAD / 2 + j * (gw + gg), vy, gw, vh)
            g_.imshow(v[:, None], aspect='auto', cmap='gray_r', vmin=-lim, vmax=lim, interpolation='nearest')
            g_.set_xticks([]); g_.set_yticks([])
            for s in g_.spines.values():
                s.set_linewidth(0.5); s.set_color(ec)
            gl.append(g_)
        txt(xs['vec'] + 0.7 + VPAD / 2 + gw / 2, vy + vh + 0.3, 'x$_t$', ha='center', fontsize=FS_S - 0.6)
        txt(xs['vec'] + 0.7 + VPAD / 2 + 1.5 * gw + gg, vy + vh + 0.3, 'Δx$_t$', ha='center', fontsize=FS_S - 0.6)
        fig.add_artist(ConnectionPatch(xyA=(sx_ + 1, sy + 0.5), coordsA=zoom_ax.transData, xyB=(0, 0.5),
                                       coordsB=gl[0].transAxes, color=C_TOK, lw=0.5, arrowstyle='-|>',
                                       mutation_scale=4))
        txt(xs['vec'] + 0.7 + VPAD / 2 + gw + gg / 2, vy - 0.25, 'per token', ha='center', va='bottom',
            fontsize=FS_S - 0.6, color=C_TOK)
        # ---- b4: SAE encoder: dense pre-activations -> sparse code (threshold / top-k), the neuron highlighted
        pa = ax_mm(xs['pre'], vy, bw + 2.0, vh)
        bars(pa, A['pre'], len(A['pre']), hi=key, thr=m['sae_theta'])
        arrow(xs['vec'] + 2 * gw + gg + 1.4 + VPAD / 2 + 0.3, xs['pre'] - 0.3, yc)
        # ---- b5: the neuron on every kept token (one-hue overlay), -> max -> c
        a = img(xs['spatial'], y0, sz, ec='k', lw=0.5)
        hm = A['heat']
        vmh = max(float(hm.max()), 1e-6)
        overlay(a, np.where(up, gt, 0.93), hm, mk, C_N, vmh, amax=0.9, interp='nearest')  # non-mask tokens blanked
        for i in range(1, G):
            a.axhline(i, color='w', lw=0.15, alpha=0.6)
            a.axvline(i, color='w', lw=0.15, alpha=0.6)
        edge(a, mk, C_KEPT, 0.4)
        a.set_xlim(zx, zx + zs); a.set_ylim(zy + zs, zy)
        # pink arrow: the neuron's bar in the SAE plot -> the token where it peaks
        fig.add_artist(ConnectionPatch(xyA=(float(A['pre'][key]), float(key)), coordsA=pa.transData,
                                       xyB=(sx_ + 0.5, sy + 0.5), coordsB=a.transData, color=C_N, lw=0.6,
                                       arrowstyle='-|>', mutation_scale=5, shrinkA=2.0, shrinkB=2.5,
                                       connectionstyle='arc3,rad=-0.15'))
        arrow(xs['spatial'] + sz + 0.5, xs['trace'] - 3.4, yc, 'max\npooling')  # clear of the y tick labels
        # ---- c: one value per frame (raw), firing raster -> temporal -> per video
        fire, s_ = firing(A, m, args.event)
        summ_out[d] = s_
        tr = A['trace']
        n = len(tr)
        nw = int(round(args.window_s * FPS))
        c = m['frame_idx'] - m['window'][0]
        w0 = int(np.clip(c - nw // 2, 0, max(n - nw, 0)))
        seg, fseg = tr[w0:w0 + nw], fire[w0:w0 + nw]
        tt = np.arange(len(seg)) / FPS
        beh = DOM[d]['short'].replace(' ', '\\ ')
        txt(xs['trace'], y0 - 0.2, f'Neuron {key}: ' + '$\\it{' + beh + '}$', color=C_N)
        a = ax_mm(xs['trace'], y0 + 2.3, wt, sz - 6.4)
        # the clip continues beyond the window: fade the trace / raster towards each open side, mark it with '...'
        more_l, more_r = w0 > 0, w0 + nw < n
        nf = max(1, int(0.12 * len(seg)))
        alpha = np.ones(len(seg))
        if more_l:
            alpha[:nf] = np.linspace(0.12, 1, nf)
        if more_r:
            alpha[-nf:] = np.linspace(1, 0.12, nf)
        rgb = np.array(to_rgb(C_N))
        pts = np.c_[tt, seg]
        a.add_collection(matplotlib.collections.LineCollection(
            np.stack([pts[:-1], pts[1:]], 1), colors=np.c_[np.tile(rgb, (len(seg) - 1, 1)), alpha[:-1]], linewidths=0.35))
        a.plot([(c - w0) / FPS], [float(tr[c])], 'o', ms=2.0, mfc='none', mec=C_KEPT, mew=0.6)
        if args.event == 'q95':
            a.axhline(m['summary']['threshold'], color='k', lw=0.35, ls=(0, (2, 1.5)))
        a.set_xlim(0, args.window_s); a.set_ylim(0, float(tr.max()) * 1.05)
        a.set_xticks([]); a.set_yticks([0, round(float(tr.max()) / 5) * 5 or 1])
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        for side, on in (('left', more_l), ('right', more_r)):
            if on:
                a.text(1.0 if side == 'right' else 0.0, 0.45, '…', transform=a.transAxes, color=C_GREY,
                       fontsize=FS, ha='left' if side == 'right' else 'right', va='center')
        r = ax_mm(xs['trace'], y0 + sz - 3.6, wt, 1.1)
        on_ = np.flatnonzero(fseg)
        r.vlines(on_ / FPS, 0, 1, colors=[(0.42, 0.42, 0.42, float(al)) for al in alpha[on_]], lw=0.15)
        r.set_xlim(0, args.window_s); r.set_ylim(0, 1); r.set_yticks([])
        for s in ('top', 'right', 'left'):
            r.spines[s].set_visible(False)
        t0m, t1m = w0 / FPS / 60, (w0 + nw) / FPS / 60  # the window's place in the clip (min)
        r.set_xticks([0, args.window_s])
        fm = lambda v: f'{v:.1f}'.rstrip('0').rstrip('.')  # noqa: E731
        r.set_xticklabels([fm(t0m), f'{fm(t1m)} min'])
        r.tick_params(length=1.0, labelsize=FS_S - 0.6)
        lat = f'{s_["latency_s"]:.1f} s' if not s_['censored'] else f'> {s_["latency_s"]:.0f} s'
        for i, (lab, val) in enumerate((('time', f'{100 * s_["time_budget"]:.0f}%'),
                                        ('bouts', f'{s_["bout_rate_per_min"]:.1f}/min'), ('latency', lat))):
            yy = y0 + 0.4 + i * 5.0
            txt(xs['num'], yy, lab, fontsize=FS_S - 0.6, color=C_DARK)
            txt(xs['num'], yy + 1.5, val, fontsize=6.2, weight='bold')
    for fmt in ('pdf', 'png'):
        fig.savefig(OUT / f'fig_pipeline.{fmt}', dpi=args.dpi if fmt == 'png' else 600)
    json.dump({'event': args.event, 'window_s': args.window_s, 'summaries': summ_out},
              open(OUT / 'temporal_summaries.json', 'w'), indent=1)
    log(f'saved {OUT}/fig_pipeline.pdf/.png ({W:.0f} x {H:.0f} mm); event {args.event}: {summ_out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('compute', 'plot', 'all', 'sheets', 'sizes'))
    ap.add_argument('--domains', default='ants,mice,frogs')
    ap.add_argument('--sae', default='', help='domain=sae_name,... overrides of the C1 SAEs')
    ap.add_argument('--encoder-label', default='', help='plot label of the encoder (default: from the cache)')
    ap.add_argument('--grid', type=int, default=0, help='token-grid side drawn in c (default: from the cache)')
    ap.add_argument('--input-px', type=int, default=0, help='encoder input size drawn in c (default: from the cache)')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--event', choices=EVENT_RULES, default='nonzero',
                    help='firing rule of the temporal block: nonzero = frame value > 0 (the neuron is active on at least '
                         'one kept token); q95 = above the neuron\'s q 0.95 threshold (NES bout / latency event); '
                         'hmm = HMM P(ON) > 0.5, time budget = mean P(ON)')
    ap.add_argument('--window-s', type=float, default=120.0, help='temporal block: shown window length (s)')
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('run through Slurm (scripts/eci/fig_pipeline.sh): no compute or rendering on the login node')
    OUT.mkdir(parents=True, exist_ok=True)
    sae = dict(SAE, **dict(kv.split('=') for kv in args.sae.split(',') if kv))
    if args.stage == 'sizes':
        source_sizes()
        return
    if args.stage == 'sheets':
        import torch
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
        frog_sheets(os.environ.get('STAGE') or f'/localhome/{os.environ["USER"]}/{os.environ["SLURM_JOB_ID"]}')
        return
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
