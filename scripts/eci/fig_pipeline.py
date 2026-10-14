"""
Nature-style multi-panel figure of the ECI (Exploratory Causal Inference) REPRESENTATION "C1": one row per domain
(ants, mice, frogs), the same steps on a real example frame of that domain, from the raw video to the video-level
summaries of one behaviour-like neuron. Discovery (NES) is not shown.

    a  raw video          native frame; size / fps (ffprobe); the ANALYSED clip length (experiment.csv start / end frames
                          + the analysis window); note "analysed at 5 fps, 512 px"
    b  encoder            the WHOLE frame is encoded first: 448 px input, 32 x 32 grid of 768-d tokens (DINOv2-base)
    c  animal mask        the real mask (whole frame); each kept token still carries whole-image context
    d  temporal context   the same token at t - 1 s and t; SAE input [x_t, x_t - x_(t-1 s)] (DINOv2 token difference,
                          not optical flow)
    e  one neuron         its activation on every kept token (display crop around the kept tokens)
    f  per-frame value    max over kept tokens over the analysed clip, as the HMM event confidence P(ON) in [0, 1]
                          (results/vision/eci_event_hmm, when present; --trace q95 = the raw value with its q 0.95
                          firing threshold, the NES bout / latency event)
    g  per-video          time budget (mean P(ON)), bout rate (runs of P(ON) > 0.5 per min), latency (s to the first
                          P(ON) > 0.5); q95: fraction above the threshold, runs above it, first frame above it
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
        key=186, desc='grooming\n(AUROC 0.86)', window='full',
        cand_text='annotated grooming frames (Y2F or B2F) of ants v2'),
    'mice': dict(
        eci=REPO / 'dataset/mice/v1/eci', ann=REPO / 'dataset/mice/v1/annotations.csv',
        nes=REPO / 'results/vision/mice/eci/nes', name='Mice', species=r'$\it{Mus\ musculus}$',
        raw_exp=REPO / 'data/mice/v1/experiment.csv', src=REPO / 'data/mice/source', step=6, expect=2, hflip=False,
        version='v1', ann_cols=['observation_id', 'frame_idx', 'frame_path', 'Y_nn'],
        key=915, desc='two mice in contact\n(nose-nose AUROC 0.72)', window='last',
        cand_text='annotated nose-nose frames (Y_nn = 1) in the analysed last 15 min of social-odour habituation videos'),
    'frogs': dict(
        eci=REPO / 'dataset/frogs/eci', nes=REPO / 'results/vision/frogs/eci/nes', name='Frogs',
        species=r'$\it{Xenopus}$ froglets', raw_exp=REPO / 'data/frogs/v1/experiment.csv',
        src=REPO / 'data/frogs/v1/observations/source', step=12, expect=5, hflip=True, version='v1 (juvenile)',
        ann_cols=['observation_id', 'frame_idx', 'frame_path', 'group'],
        cand=lambda a: a[(a.group == 'FoxP1') & (a.frame_idx >= 10)],
        key=500, desc='extended hind limb', window='full',
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
    tried, row = [], None
    for a in range(0, min(MAX_TRY, len(order)), 8):
        rr = [int(r) for r in order[a:a + 8]]
        _, _, grey_b, tok_b = encode_rows(rr, 'try')
        for k, r in enumerate(rr):
            mk = bgs.mask(tok_b[k:k + 1], grey_b[k:k + 1], np.array([r]))[0][0].numpy().reshape(-1)
            sizes = pieces(mk.reshape(int(np.sqrt(mk.size)), -1))
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
                tried=tried[:rank + 2], sel_motion=float(motion[sel]),
                version=cfg['version'], name=cfg['name'], species=cfg['species'])
    OUT.joinpath('cache').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'cache' / f'{name}.npz', native=nat_small, frames=frames, enc_in=enc_in,
                        mask=mask.numpy().reshape(GRID, GRID), motion=motion.reshape(GRID, GRID), heat=heat, code=mx,
                        tok_code=tok_code, trace=tr, above=above, bouts=bouts,
                        **({} if hmm is None else dict(p_on=hmm['p_on'], hmm_bouts=hmm['bouts'])))
    json.dump(meta, open(OUT / 'cache' / f'{name}.json', 'w'), indent=1,
              default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    log(f'[{name}] done in {time.time() - t0:.0f}s')
    return meta


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
C_N = '#0072B2'      # the shown neuron (Okabe-Ito blue)
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
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw))
    ax.set_xlim(0, G); ax.set_ylim(G, 0)


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
    left, top_h, row_h, im = 7.0, 10.5, 32.0, 20.0
    widths = dict(a=20.0, b=20.0, c=20.0, d=27.0, e=20.0, f=36.0, g=19.0)
    gap = (W - left - 0.5 - sum(widths.values())) / (len(widths) - 1)
    xs, x = {}, left
    for k, w in widths.items():
        xs[k] = x
        x += w + gap
    H = top_h + row_h * len(doms) - 2.0
    fig = plt.figure(figsize=(W * MM, H * MM))

    def ax_mm(x, y, w, h, **kw):  # y measured from the top
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)

    def txt(x, y, s, **kw):
        kw.setdefault('fontsize', FS)
        kw.setdefault('va', 'top')
        kw.setdefault('linespacing', 1.1)
        return fig.text(x / W, 1 - y / H, s, **kw)

    def img_ax(x, y, w, h=None, ec=C_DARK, lw=0.4):
        a = ax_mm(x, y, w, h or w)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(lw); s.set_color(ec)
        return a

    all_hmm = args.trace == 'hmm' and all('p_on' in A_ for A_, _ in data.values())
    heads = dict(a='Raw video', b='Encoder on the\nwhole frame', c='Mask keeps\nanimal tokens',
                 d='Local temporal context\n[x$_t$, x$_t$ − x$_{t-1\\,s}$]', e='One neuron\non every kept token',
                 f='Per-frame value over the clip\n' + ('(max over tokens → HMM P(ON))' if all_hmm else '(max over kept tokens)'), g='Per-video\nsummaries')
    for k, s in heads.items():
        txt(xs[k], 1.0, k, fontsize=8, weight='bold')
        txt(xs[k] + 2.7, 1.25, s, linespacing=1.05)

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
        return txt(xs[k], y, s, **kw)

    for ri, d in enumerate(doms):
        A, m = data[d]
        y0 = top_h + ri * row_h
        G = grid_n
        ut = y0 + im + 0.9  # first text line under the images
        txt(0.6, y0 + im / 2, m['name'], fontsize=7, weight='bold', rotation=90, va='center', ha='left')
        txt(3.6, y0 + im / 2, m['species'], rotation=90, va='center', ha='left')
        for k in ('a', 'b', 'c', 'd'):  # image steps only (the plots in f, g carry their own axes)
            chevron(xs[k] + widths[k] + gap / 2, y0 + im / 2)
        nat = m['native']
        key = m['key']
        mk = A['mask'].astype(bool)
        zx, zy, zs = zoom_box(mk)
        fr = A['frames'][0].astype(np.float32) / 255
        sy, sx_ = m['sel_token']
        # a raw video
        a = img_ax(xs['a'], y0, im)
        a.imshow(A['native'], interpolation='lanczos')
        clip = m['lengths']['clip_min']
        under('a', ut, f'{nat["size"][0]} × {nat["size"][1]} px\n{nat["fps"]:.0f} fps, {clip:.0f}-min clip')
        under('a', ut + 4.6, 'analysed at 5 fps, 512 px', fontsize=FS_S)
        # b encoder on the whole frame + token grid
        a = img_ax(xs['b'], y0, im)
        a.imshow(A['enc_in'], extent=(0, in_px, in_px, 0), interpolation='lanczos')
        for i in range(1, G):
            v = i * in_px / G
            a.axhline(v, color='w', lw=0.12, alpha=0.75)
            a.axvline(v, color='w', lw=0.12, alpha=0.75)
        a.set_xlim(0, in_px); a.set_ylim(in_px, 0)
        under('b', ut, f'{enc_label}, {in_px} px', fontsize=FS_S)
        under('b', ut + 2.3, f'→ {G} × {G} tokens (768-d)', fontsize=FS_S)
        # c mask keeps animal tokens (the real mask, whole frame)
        up = np.kron(mk, np.ones((512 // G, 512 // G), bool))[..., None]
        grey = fr.mean(2, keepdims=True) * 0.45 + 0.35
        a = img_ax(xs['c'], y0, im)
        a.imshow(np.where(up, fr, np.repeat(grey, 3, 2)), extent=(0, G, G, 0), interpolation='nearest')
        edge(a, mk, C_KEPT, 0.45)
        under('c', ut, f'kept {m["n_kept"]} / {G * G}')
        under('c', ut + 2.6, 'each kept token keeps\nwhole-frame context', fontsize=FS_S)
        # d local temporal context: the same token at t - 1 s and t
        cw = (widths['d'] - 1.0) / 2
        for j, (fi, lab) in enumerate(zip((1, 0), ('t − 1 s', 't'))):
            a = img_ax(xs['d'] + j * (cw + 1.0), y0, cw)
            a.imshow(A['frames'][fi], extent=(0, G, G, 0), interpolation='lanczos')
            a.add_patch(Rectangle((sx_, sy), 1, 1, fill=False, ec=C_KEPT, lw=0.8))
            a.set_xlim(zx, zx + zs); a.set_ylim(zy + zs, zy)
            a.text(0.05, 0.95, lab, transform=a.transAxes, color='w', fontsize=FS_S, va='top', ha='left')
        yb = y0 + cw + 1.2
        sb = ax_mm(xs['d'], yb, widths['d'], 2.4)
        sb.add_patch(Rectangle((0, 0), 0.5, 1, fc='#f0f0f0', ec=C_DARK, lw=0.4))
        sb.add_patch(Rectangle((0.5, 0), 0.5, 1, fc='#d8eadf', ec=C_DARK, lw=0.4))
        sb.text(0.25, 0.5, 'x$_t$ (768)', ha='center', va='center', fontsize=FS_S)
        sb.text(0.75, 0.5, 'x$_t$ − x$_{t-1\\,s}$ (768)', ha='center', va='center', fontsize=FS_S)
        sb.set_xlim(0, 1); sb.set_ylim(0, 1); sb.axis('off')
        txt(xs['d'], yb + 2.9, 'DINOv2 token difference,\nnot optical flow; same as\nalso giving x$_{t-1\\,s}$',
            fontsize=FS_S)
        # e one neuron on every kept token (zoom, display crop only)
        a = img_ax(xs['e'], y0, im, ec=C_N, lw=0.9)
        a.imshow(fr.mean(2), cmap='gray', vmin=-0.2, vmax=1.6, extent=(0, G, G, 0), interpolation='bilinear')
        hm = A['heat']
        vmh = max(float(hm.max()), 1e-6)
        a.imshow(np.ma.masked_where(~mk | (hm <= 0), hm), cmap='magma', vmin=0, vmax=vmh, extent=(0, G, G, 0),
                 interpolation='nearest')
        edge(a, mk, '#ffffff', 0.3)
        a.add_patch(Rectangle((sx_, sy), 1, 1, fill=False, ec=C_KEPT, lw=0.8))
        a.set_xlim(zx, zx + zs); a.set_ylim(zy + zs, zy)
        cbar(xs['e'], ut - 0.2, im, 'magma', vmh)
        under('e', ut + 2.9, f'neuron {key}\n{DOM[d]["desc"]}', fontsize=FS_S, color=C_N)
        # f per-frame value over the clip: HMM P(ON) in [0, 1] (or the raw value with its q 0.95 threshold)
        use_hmm = args.trace == 'hmm' and 'p_on' in A
        if args.trace == 'hmm' and not use_hmm:
            log(f'[{d}] no HMM P(ON) in the cache: raw frame value + q 0.95 firing')
        tr = A['p_on'] if use_hmm else A['trace']
        bt = A['hmm_bouts'] if use_hmm else A['bouts']
        s_ = m['hmm_summary'] if use_hmm else m['summary']
        thr = 0.5 if use_hmm else s_['threshold']
        tt = np.arange(len(tr)) / FPS / 60
        hx, hw = xs['f'] + 3.6, widths['f'] - 3.6
        a = ax_mm(hx, y0, hw, 14.0)
        if use_hmm:
            a.fill_between(tt, 0, tr, color=C_N, alpha=0.25, lw=0)
        a.plot(tt, tr, color=C_N, lw=0.35)
        a.axhline(thr, color='k', lw=0.5, ls=(0, (2, 1.5)))
        a.set_xlim(0, tt[-1])
        if use_hmm:
            a.set_ylim(0, 1.02); a.set_yticks([0, 0.5, 1]); a.set_yticklabels(['0', '0.5', '1'])
        else:
            a.set_ylim(0, max(float(tr.max()), thr) * 1.05); a.locator_params(axis='y', nbins=2)
        a.locator_params(axis='x', nbins=3)
        a.set_xticklabels([])
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        a.set_ylabel('P(ON)' if use_hmm else f'#{key}', fontsize=FS_S, color=C_N, labelpad=0.5)
        r = ax_mm(hx, y0 + 14.8, hw, 1.6)
        r.vlines(bt[:, 0] / FPS / 60, 0, 1, color='k', lw=0.25)
        if not s_['censored']:
            r.plot([s_['latency_s'] / 60], [1.35], marker='v', ms=2.2, color=C_KEPT, mec='none', clip_on=False)
        r.set_xlim(0, tt[-1]); r.set_ylim(0, 1)
        r.set_yticks([])
        for s in ('top', 'right', 'left'):
            r.spines[s].set_visible(False)
        r.locator_params(axis='x', nbins=3)
        r.set_xlabel('clip time (min)', fontsize=FS_S, labelpad=0.5)
        under('f', y0 + 22.6, ('ticks: bout starts (P(ON) > 0.5); ▼ first bout' if use_hmm else
                               'ticks: bout starts (above q 0.95); ▼ first bout'), fontsize=FS_S)
        # g per-video summaries
        lat = f'{s_["latency_s"]:.1f} s' if not s_['censored'] else f'> {s_["latency_s"]:.0f} s'
        rows_ = [('time budget', f'{100 * s_["time_budget"]:.0f}%'),
                 ('bout rate', f'{s_["bout_rate_per_min"]:.1f} / min'), ('latency', lat)]
        for i, (lab, val) in enumerate(rows_):
            yy = y0 + 1.0 + i * 5.2
            txt(xs['g'], yy, lab, fontsize=FS_S, color=C_DARK)
            txt(xs['g'], yy + 2.0, val, fontsize=7, weight='bold')
        under('g', y0 + 17.0, ('mean P(ON); bouts and\nlatency at P(ON) > 0.5' if use_hmm else
                               'frames above q 0.95;\nbouts = runs of them'), fontsize=FS_S - 0.4)
    for fmt in ('pdf', 'png'):
        fig.savefig(OUT / f'fig_pipeline.{fmt}', dpi=args.dpi if fmt == 'png' else 600)
    log(f'saved {OUT}/fig_pipeline.pdf/.png ({W:.0f} x {H:.0f} mm)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('compute', 'plot', 'all', 'sheets'))
    ap.add_argument('--domains', default='ants,mice,frogs')
    ap.add_argument('--sae', default='', help='domain=sae_name,... overrides of the C1 SAEs')
    ap.add_argument('--encoder-label', default='', help='plot label of the encoder (default: from the cache)')
    ap.add_argument('--grid', type=int, default=0, help='token-grid side drawn in c (default: from the cache)')
    ap.add_argument('--input-px', type=int, default=0, help='encoder input size drawn in c (default: from the cache)')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--trace', choices=('hmm', 'q95'), default='hmm',
                    help='f/g: HMM P(ON) (when cached) or the raw frame value with its q 0.95 firing threshold')
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('run through Slurm (scripts/eci/fig_pipeline.sh): no compute or rendering on the login node')
    OUT.mkdir(parents=True, exist_ok=True)
    sae = dict(SAE, **dict(kv.split('=') for kv in args.sae.split(',') if kv))
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
