"""
NN/NT round 1 (2026-10-16): which input / mask / readout choices give ONE clean nose-nose and ONE clean nose-tail neuron
on mice v1, with an interpretable spatial heat map. Labels are used ONLY to score / select among recipes, never in
training (no probes, no label-guided SAE). Development on ALL mice v1 labels (user's permission, 2026-10-16).

Fixed parts (as r4, scripts/eci/rollout_r4.py): DINOv3-L/16 block 20 (bf16 + SDPA), mice input 1024 px (64 x 64 tokens),
5 fps grid of annotations.csv (source frame 6 k + 2), change channel x = [z_t, z_t - z_{t-10}] (2 s), SAE = Matryoshka
BatchTopK 512 latents, prefixes 128/256/512, k = 16, 6000 steps x 4096 tokens, theta* calibrated (mean L0 = k on the
held-out CALIBRATION videos). Same train / calib / eval video roles and the same prepass bursts as r4 (all 432 mice v1
videos, all phases, full length: 10,080 train frames from 360 videos, 1,008 calib, 1,008 eval). A0 trains on r4's own
prepass tokens (train-video tokens subsampled p = 0.45, as r4); the new arms keep EVERY on-mask token of the same frames
(p = 1; at 0.45 the pooled-tight arm would have ~0.2M train tokens, > 100 passes over 6000 x 4096).

ARMS (all DINOv3-L block 20 at 1024 px)
    A0  r4 reference: no rotation, LOOSE mask (lab rule fg448 on the 512 px JPG, 32 grid with its 1-cell dilation,
        projected to 64 x 64), 64 grid, d2. Its SAEs train on the r4 prepass tokens themselves.
    A1  A0 + odour-corner rotation (frames rotated k x 90 deg CCW so the odour corner is top right, as the online
        fg448al model; loose mask from the fg448al backgrounds on the rotated JPG).
    A2  A1 with the TIGHT mask: a 64-grid token is kept iff >= 25% of its 16 x 16 px patch of the 1024 px DINOv3 input
        is animal pixels by the lab mask's own pixel cue (grey < 60 and pix_bg - grey > 40, pix_bg = the video's
        per-pixel background of fg448al, bilinearly upsampled 512 -> 1024). No dilation, no DINOv2 cue.
    A3  A2 without the change channel (d0: x = z_t).
    A4  A2 with 2 x 2 average pooling of the 64-grid tokens to 32 x 32 before the SAE (all 4 children averaged; the
        pooled cell is kept iff >= 25% of its 32 x 32 px block is animal pixels = the same rule at the coarse cell;
        change channel on pooled tokens).
    A5  A4 without the change channel.
    W1024 variants of A2-A5 (1024 latents, prefixes 128/256/512/1024), seeds 0-2 only.
    A0ref = r4's own deployed d2 SAE (matryoshka_btk_512_k16_r4d2_s0) applied in the A0 stream: a pipeline check (its
        codes must equal the stored r4 codes).
SEEDS 0-5 for every 512-wide arm, all in ONE encode pass (round 1 = seeds 0-2, fresh confirmation = seeds 3-5).

READOUTS (per frame and neuron; fixed BEFORE any label was looked at)
    max   max over the frame's kept tokens (r4).
    top   mean of the m largest kept-token activations (sum / m; zeros fill when fewer than m tokens are kept),
          m = 8 at the 64 grid, 2 at the 32 grid: the same image area (8 x 16^2 px of the 1024 px frame), about two
          mouse heads (mouse body 9.4 patches long at the 64 grid -> head ~2.4 x 2 patches ~ 5 tokens; two heads ~ 8-10).

FRAMES SCORED
    rotated arms A1-A5: ALL annotated frames, 144 videos (24 fully annotated pools x 6), 864,000 frames at 5 fps.
    A0 / A0ref: the r4 align subset only (every 5th frame, offset per video = fg_extract_train.fps1_rows(all 432
        videos, 5, seed 0), 172,800 frames), to save the second (unrotated) DINOv3 pass on 4/5 of the frames; every
        A0-vs-A1 comparison is made on this same subset ('fps1').
LABELS (annotations.csv row space, per frame per observation; reference_mice_label_vocabulary)
    nose_nose = Y_nn | Y_np (mutual or directional nose-to-nose = the directed-pair union; r4's 'nose_nose'),
    nose_tail = Y_nt; secondary: nn_mutual = Y_nn, np_directional = Y_np.

METRICS (scripts/eci/nnnt/nnnt.py score; continuous codes, rank based, no firing threshold)
    R~ = (P - pi) / (1 - pi), P = precision in the top N_pos frames (ties at the cut pro rata), pi = base rate.
    Cross-fitted: the neuron with the highest R~ on one half of the 24 POOLS is scored on the other half, both
    directions averaged; 10 random 12/12 pool splits (rng 1000 + s, identical for every arm), plus r4's split (sorted
    pools alternating) for the reproduction. Secondary: test AUROC of the R~-selected neuron, cf AUROC (AUROC-selected,
    as r4 cf_auroc), AP-lift, frame fire rate under theta*.
PRE-REGISTERED DECISION RULE (written before any score was computed)
    primary per (arm, readout, seed) = mean(R~ nose_nose, R~ nose_tail), mean over the 10 random splits.
    Round 1: the best (arm in A1-A5 at width 512, readout) by the mean over seeds 0-2. Confirmation: seeds 0-5 and the
    fresh seeds 3-5, Welch 95% CI vs A1-max (and vs A0-max on fps1). Component effects (6 seeds, Welch 95% CI; readout
    effect paired by seed): rotation A1 - A0 (fps1), tight mask A2 - A1, change channel A3 - A2 / A5 - A4, pooling
    A4 - A2 / A5 - A3, top vs max per arm, width 1024 - 512 (3 seeds).

Steps (scripts/eci/nnnt/nnnt.sh runs each in Slurm; heavy inputs / outputs staged on $LOCAL_DIR)
    plan      (CPU) videos, rotations, fps1 offsets, encode shards -> OUT/plan.json
    prepass   (GPU) the r4 bursts on ROTATED frames: loose / tight 64-grid tokens and pooled-tight 32-grid tokens with
              their t-10 partners -> BEE/prepass/, mask stats + sheet -> OUT/prepass.json, OUT/mask_sheet.jpg
    train     (GPU) every arm x width x seed, theta*, held-out FVE -> OUT/sae/<name>/
    encode    (GPU, one task per shard) 144 annotated videos, rotated pass on every frame + unrotated pass on the fps1
              frames -> BEE/videos/<obs>.npz (per SAE: max and top readouts (n, W) fp16; kept counts)
    score     (GPU) all metrics -> OUT/score.json, OUT/table.md
    locality  (GPU) per arm: best nn / nt neurons, top held-out frames re-encoded, per-token heat maps, locality
              numbers -> OUT/locality.json, OUT/sheets/*.jpg
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import rollout_r4 as R4  # noqa: E402  (decode, Prefetch, stage_jpgs, constants; read-only use)

OUT = Path(os.environ.get('NNNT_OUT') or REPO / 'results/vision/eci_nnnt')
BEE = Path(os.environ.get('NNNT_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/nnnt')
R4OUT = REPO / 'results/vision/eci_rollout_r4/mice'
R4BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/r4/mice')
C = R4.DOMS['mice']
S, G64, D, LAG = 1024, 64, 1024, R4.LAG
K, P_STORE, THR_TIGHT = 16, 1.0, 0.25
BURST = R4.BURST
SEEDS6, SEEDS3 = (0, 1, 2, 3, 4, 5), (0, 1, 2)
TOPM = {64: 8, 32: 2}
# token kinds: (rotated?, mask, grid)
KINDS = {'u64L': (False, 'loose', 64), 'r64L': (True, 'loose', 64), 'r64T': (True, 'tight', 64),
         'r32T': (True, 'tight', 32)}
ARMS = {'A0': ('u64L', 'd2'), 'A1': ('r64L', 'd2'), 'A2': ('r64T', 'd2'), 'A3': ('r64T', 'd0'),
        'A4': ('r32T', 'd2'), 'A5': ('r32T', 'd0')}
LABELS = ('nose_nose', 'nose_tail', 'nn_mutual', 'np_directional')
N_SPLITS = 10
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def sae_list():
    """[(name, arm, width, seed)] of every SAE trained here."""
    out = [(f'{a}_w512_s{s}', a, 512, s) for a in ARMS for s in SEEDS6]
    out += [(f'{a}_w1024_s{s}', a, 1024, s) for a in ('A2', 'A3', 'A4', 'A5') for s in SEEDS3]
    return out


def prefixes_of(w):
    return (128, 256, 512) if w == 512 else (128, 256, 512, 1024)


def local_dir():
    return R4.local_dir()


# ---------------------------------------------------------------------------------------------- plan
def cmd_plan(args):
    from fg_extract_train import fps1_rows
    from src.eci.foreground import ODOR_ROT90
    P4 = R4.load_plan('mice')
    corner = pd.read_csv(REPO / 'dataset/mice/v1/eci/odor_corner.csv').set_index('observation_id')['odor_corner']
    ranges = {v['obs']: (v['lay_lo'], v['lay_lo'] + v['n']) for v in P4['videos']}
    f1 = fps1_rows(ranges, 5, 0)
    a = pd.read_csv(REPO / C['ann'], usecols=['observation_id', 'Y_nn', 'Y_np', 'Y_nt'])
    vids = []
    for i, v in enumerate(P4['videos']):
        assert v['src_lo'] == v['lay_lo']
        y = a.iloc[v['src_lo']:v['src_lo'] + v['n']]
        assert (y.observation_id == v['obs']).all()
        nna = int(y[['Y_nn', 'Y_np', 'Y_nt']].notna().all(1).sum())
        ann = nna == v['n']
        assert nna in (0, v['n']), (v['obs'], nna)
        vids.append({**{k_: v[k_] for k_ in ('obs', 'src_lo', 'n', 'path', 'hflip', 'role', 'bursts')},
                     'offset': R4.voff(C, v),
                     'r4_index': i, 'rot': int(ODOR_ROT90[corner[v['obs']]]), 'corner': corner[v['obs']],
                     'annotated': ann, 'fps1_off': int(f1[v['obs']][0] - v['lay_lo'])})
    annv = [i for i, v in enumerate(vids) if v['annotated']]
    n = np.array([vids[i]['n'] for i in annv])
    shards, cur, nf = [], [], 0
    for i in annv:  # consecutive annotated videos, ~args.shard_frames each
        if cur and nf + vids[i]['n'] > args.shard_frames:
            shards.append(cur)
            cur, nf = [], 0
        cur.append(i)
        nf += vids[i]['n']
    shards.append(cur)
    P = {'n_videos': len(vids), 'n_annotated': len(annv), 'annotated_frames': int(n.sum()),
         'rot_annotated': pd.Series([vids[i]['corner'] for i in annv]).value_counts().to_dict(),
         'shards': shards, 'shard_frames': [int(sum(vids[i]['n'] for i in s)) for s in shards], 'videos': vids}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'plan.json').write_text(json.dumps(P, indent=1, default=str))
    log(f'plan: {len(vids)} videos, {len(annv)} annotated ({n.sum():,} frames), corners {P["rot_annotated"]}, '
        f'{len(shards)} shards {P["shard_frames"]}')


def load_plan():
    return json.loads((OUT / 'plan.json').read_text())


# ---------------------------------------------------------------------------------------------- pipeline
class Pipe:
    """One orientation of a batch of frames of ONE video: DINOv2-base mask tokens (loose lab rule, 32 grid, projected
    to 64), the pixel dark cue at the 1024 px DINOv3 input (tight masks), DINOv3-L block-20 tokens."""

    def __init__(self, dev, ranges):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from src.eci.foreground import RULES, FgBackgrounds
        self.torch, self.dev = torch, dev
        self.proc2 = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=True,
                                                        size={'height': 448, 'width': 448}, do_center_crop=False)
        self.m2 = AutoModel.from_pretrained('facebook/dinov2-base', attn_implementation='sdpa').to(dev).eval()
        m3 = AutoModel.from_pretrained(R4.D3L, attn_implementation='sdpa').to(dev).eval()
        assert len(m3.layer) == 24
        m3.layer = torch.nn.ModuleList(list(m3.layer)[:R4.LAYER])
        self.m3 = m3.requires_grad_(False)
        self.m2.requires_grad_(False)
        self.mean = torch.tensor(R4.MEAN, device=dev)[None, :, None, None]
        self.std = torch.tensor(R4.STD, device=dev)[None, :, None, None]
        self.rule = RULES['fg448']
        self.bgs = {'none': FgBackgrounds(REPO / 'dataset/mice/v1/eci/fg448/background', ranges, self.rule, dev,
                                          align='none'),
                    'odor': FgBackgrounds(REPO / 'dataset/mice/v1/eci/fg448al/background', ranges, self.rule, dev,
                                          align='odor')}
        g = np.floor((np.arange(G64) + 0.5) * 32 / G64).astype(np.int64)
        self.proj = torch.from_numpy((g[:, None] * 32 + g[None, :]).ravel()).to(dev)
        self._pbg = {}
        self.gpu_s = 0.0

    def pix_bg1024(self, align, obs):
        key = (align, obs)
        if key not in self._pbg:
            if len(self._pbg) > 8:
                self._pbg.clear()
            pb = self.bgs[align].get(obs)['pix_bg'].float()
            self._pbg[key] = self.torch.nn.functional.interpolate(pb[None, None], size=(S, S), mode='bilinear',
                                                                  align_corners=False)[0, 0]
        return self._pbg[key]

    def run(self, fr, jp, rows, obs, align, rot, encode=True):
        """fr (B, S, S, 3) uint8 cuda (decoded, UNrotated), jp (B, 512, 512, 3) uint8 cuda (dataset JPG, unrotated),
        rows (B,) annotations rows, align 'none' | 'odor', rot = 90-degree CCW turns (0 for 'none').
        -> dict z (B, 4096, 1024) fp16 | None, loose (B, 4096) bool, dark64 (B, 4096) float, dark64_jpg (B, 4096),
        n32 (B,), x (B, 3, S, S) uint8 (the frame the model saw)."""
        torch = self.torch
        F = torch.nn.functional
        t0 = time.time()
        with torch.no_grad():
            x = fr.permute(0, 3, 1, 2)
            x512 = jp.permute(0, 3, 1, 2)
            if rot:
                x = torch.rot90(x, rot, dims=(2, 3))
                x512 = torch.rot90(x512, rot, dims=(2, 3))
            x, x512 = x.contiguous(), x512.contiguous()
            grey512 = R4.Pipe.grey(x512)
            pix = self.proc2(images=list(x512), return_tensors='pt', device=self.dev)['pixel_values'].to(self.dev)
            with torch.autocast('cuda', dtype=torch.float16):
                h2 = self.m2(pixel_values=pix).last_hidden_state
            tok2 = h2[:, 1:].float().half()
            m32, _ = self.bgs[align].mask(tok2, grey512, np.asarray(rows))
            g = R4.Pipe.grey(x).float()
            pb = self.pix_bg1024(align, obs)
            dark = (g < self.rule['dark_abs']) & ((pb[None] - g) > self.rule['dark_rel'])
            dark64 = F.avg_pool2d(dark.float()[:, None], S // G64).flatten(1)
            pb512 = self.bgs[align].get(obs)['pix_bg'].float()
            g5 = grey512.float()
            dj = (g5 < self.rule['dark_abs']) & ((pb512[None] - g5) > self.rule['dark_rel'])
            dark64j = F.avg_pool2d(dj.float()[:, None], 512 // G64).flatten(1)
            z = None
            if encode:
                xin = (x.float() / 255 - self.mean) / self.std
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    h3 = self.m3(pixel_values=xin).last_hidden_state
                z = h3[:, R4.N_PRE:].float()
                assert z.shape[1] == G64 * G64 and torch.isfinite(z).all(), z.shape
                z = z.half()
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        return {'z': z, 'loose': m32[:, self.proj], 'dark64': dark64, 'dark64_jpg': dark64j, 'n32': m32.sum(1),
                'x': x}


def pool2(z):
    """(B, 4096, d) fp16 -> (B, 1024, d) fp16: 2 x 2 mean of the 64-grid tokens."""
    B, _, d = z.shape
    return z.view(B, 32, 2, 32, 2, d).float().mean((2, 4)).reshape(B, 1024, d).half()


def dark32_of(dark64):
    return dark64.view(-1, 32, 2, 32, 2).mean((2, 4)).reshape(-1, 1024)


def kind_tokens(kind, out, zp):
    """-> (z, zp, mask) of a token kind for one orientation's pipe output and the t-10 grid zp (B, 4096, 1024)."""
    _, mk, grid = KINDS[kind]
    if grid == 64:
        m = out['loose'] if mk == 'loose' else out['dark64'] >= THR_TIGHT
        return out['z'], zp, m
    return pool2(out['z']), pool2(zp), dark32_of(out['dark64']) >= THR_TIGHT


class Ctx:
    """Per video: full 64-grid token grids by 5 fps frame index, for the t-10 partner (clipped to frame 0)."""

    def __init__(self):
        self.d = {}

    def put(self, ks, z):
        for i, k in enumerate(ks):
            self.d[int(k)] = z[i]

    def prev(self, ks):
        import torch
        return torch.stack([self.d[max(int(k) - LAG, 0)] for k in ks])

    def prune(self, k_last):
        for k in [k for k in self.d if k != 0 and k < k_last - LAG + 1]:
            del self.d[k]


def sae_input(var, zt, zp_t):
    import torch
    return torch.cat([zt, (zt.float() - zp_t.float()).half()], 1) if var == 'd2' else zt


def readouts(torch, zz, bi, B, m):
    """zz (n, W) token codes, bi (n,) frame of each token (non-decreasing) -> max (B, W), top-m mean (B, W) fp16."""
    W = zz.shape[1]
    cnt = torch.bincount(bi, minlength=B)
    if len(bi) == 0:
        z0 = torch.zeros((B, W), dtype=torch.float16, device=zz.device)
        return z0, z0.clone()
    off = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(len(bi), device=bi.device) - off[bi]
    T = int(cnt.max())
    dense = torch.zeros((B, max(T, m), W), dtype=torch.float32, device=zz.device)
    dense[bi, pos] = zz
    mx = dense.max(1).values
    top = dense.topk(m, dim=1).values.sum(1) / m
    return mx.half(), top.half()


def apply_kind(torch, saes, z, zp, mask, m):
    """saes [(name, sae, norm, var)] -> {name: (max, top)}; tokens gathered once per kind."""
    B = z.shape[0]
    bi, pi = mask.nonzero(as_tuple=True)
    zt, zpt = z[bi, pi], zp[bi, pi]
    res, cache = {}, {}
    for name, sae, norm, var in saes:
        if var not in cache:
            cache[var] = sae_input(var, zt, zpt)
        x = cache[var]
        zz = sae.encode(norm(x.float()), mode='threshold') if len(x) else torch.zeros((0, sae.n_latents),
                                                                                       device=z.device)
        res[name] = readouts(torch, zz, bi, B, m)
    return res, mask.sum(1)


# ---------------------------------------------------------------------------------------------- prepass
def cmd_prepass(args):
    """r4's bursts (same videos / frames / roles) on ROTATED frames. Per target frame (burst offsets 10..25, partner
    offset - 10): 64-grid tokens of loose | tight (flag bit 0 loose, bit 1 tight) and 32-grid pooled tokens of the
    pooled-tight cells, each with its t-10 partner; train-video tokens kept with p = P_STORE (seeded per frame)."""
    import torch
    dev = torch.device('cuda')
    P = load_plan()
    _, Rr = R4.obs_ranges_of(REPO / C['ann'])
    fp = pd.read_csv(REPO / C['ann'], usecols=['frame_path'])['frame_path'].values
    jdir = local_dir() / 'jpg'
    loc = local_dir() / 'prepass'
    loc.mkdir(parents=True, exist_ok=True)
    jobs = []
    for vi, v in enumerate(P['videos']):
        for a in v['bursts']:
            rr = [v['src_lo'] + a + o for o in BURST]
            jobs.append(((vi, a), dict(path=v['path'], s0=v['offset'] + C['step'] * a, n=len(BURST),
                                       step=C['step'] * 5, S=S, hflip=v['hflip'], fps=C['fps'], threads=2,
                                       jpgs=[jdir / fp[r] for r in rr])))
    if args.limit:
        jobs = jobs[::max(1, len(jobs) // args.limit)][:args.limit]
    st = ThreadPoolExecutor(1).submit(R4.stage_jpgs, sorted({str(f.relative_to(jdir)) for _, kw in jobs
                                                             for f in kw['jpgs']}), jdir)
    pipe = Pipe(dev, Rr)
    st.result()
    f64 = {k_: open(loc / f'{k_}64.f16', 'wb') for k_ in ('tok', 'prev')}
    f32 = {k_: open(loc / f'{k_}32.f16', 'wb') for k_ in ('tok', 'prev')}
    meta64 = {'frame': [], 'pos': [], 'flag': []}
    meta32 = {'frame': [], 'pos': []}
    frames, sheet = [], []
    t0 = time.time()
    pf = R4.Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) // 2))
    for bi_, ((vi, a), (fr, jp)) in enumerate(pf):
        v = P['videos'][vi]
        rows = np.array([v['src_lo'] + a + o for o in BURST])
        with torch.no_grad():
            o = pipe.run(torch.from_numpy(fr).to(dev), torch.from_numpy(jp).to(dev), rows, v['obs'], 'odor', v['rot'])
            loose, d64, d64j = o['loose'], o['dark64'], o['dark64_jpg']
            tight = d64 >= THR_TIGHT
            t32 = dark32_of(d64) >= THR_TIGHT
            z32 = pool2(o['z'])
        for j in range(2, len(BURST)):
            i, ip = j, j - 2
            fid = len(frames)
            g = np.random.default_rng([3, vi, a, j])
            keep = (loose[i] | tight[i])
            u = torch.from_numpy(g.random(G64 * G64)).to(dev)
            if v['role'] == 'train':
                keep = keep & (u < P_STORE)
            pos = torch.nonzero(keep).flatten()
            f64['tok'].write(o['z'][i, pos].cpu().numpy().tobytes())
            f64['prev'].write(o['z'][ip, pos].cpu().numpy().tobytes())
            meta64['frame'].append(np.full(len(pos), fid, np.int32))
            meta64['pos'].append(pos.cpu().numpy().astype(np.int16))
            meta64['flag'].append((loose[i, pos].to(torch.uint8) + 2 * tight[i, pos].to(torch.uint8)).cpu().numpy())
            u2 = torch.from_numpy(np.random.default_rng([4, vi, a, j]).random(1024)).to(dev)
            k32 = t32[i] & ((u2 < P_STORE) if v['role'] == 'train' else torch.ones_like(t32[i]))
            p32 = torch.nonzero(k32).flatten()
            f32['tok'].write(z32[i, p32].cpu().numpy().tobytes())
            f32['prev'].write(z32[ip, p32].cpu().numpy().tobytes())
            meta32['frame'].append(np.full(len(p32), fid, np.int32))
            meta32['pos'].append(p32.cpu().numpy().astype(np.int16))
            # mask statistics (all kept tokens, before the p subsample)
            dk = d64[i]
            tj = d64j[i] >= THR_TIGHT
            dark_px = float(dk.sum())
            frames.append({'video': vi, 'obs': v['obs'], 'k': a + BURST[j], 'row': v['src_lo'] + a + BURST[j],
                           'role': v['role'], 'rot': v['rot'],
                           'n_loose': int(loose[i].sum()), 'n_tight': int(tight[i].sum()),
                           'n_loose_on_animal': int((loose[i] & tight[i]).sum()),
                           'n_loose_no_dark_px': int((loose[i] & (dk == 0)).sum()),
                           'n_tight_not_loose': int((tight[i] & ~loose[i]).sum()),
                           'n_tight32': int(t32[i].sum()), 'n_loose32': int(o['n32'][i]),
                           'dark_px_recall_tight': float(dk[tight[i]].sum()) / dark_px if dark_px > 0 else np.nan,
                           'dark_px_recall_loose': float(dk[loose[i]].sum()) / dark_px if dark_px > 0 else np.nan,
                           'n_tight_jpg': int(tj.sum()),
                           'iou_tight_jpg': float((tj & tight[i]).sum()) / max(1, int((tj | tight[i]).sum())),
                           'n_tight_isolated': int((tight[i] & (R4_isolated(torch, tight[i]))).sum()),
                           'n_stored64': int(len(pos)), 'n_stored32': int(len(p32))})
            if v['role'] == 'eval' and j == 2 and len(sheet) < 8 and (not sheet or sheet[-1][3] != v['obs']):
                sheet.append((o['x'][i].cpu().numpy(), loose[i].cpu().numpy(), tight[i].cpu().numpy(), v['obs'],
                              a + BURST[j]))
        if bi_ % 500 == 0:
            log(f'prepass {bi_}/{len(jobs)} bursts, {time.time() - t0:.0f}s (GPU {pipe.gpu_s:.0f}s)')
    for f in (*f64.values(), *f32.values()):
        f.close()
    for k_, x in meta64.items():
        np.save(loc / f'tok64_{k_}.npy', np.concatenate(x))
    for k_, x in meta32.items():
        np.save(loc / f'tok32_{k_}.npy', np.concatenate(x))
    Fm = pd.DataFrame(frames)
    Fm.to_parquet(loc / 'frames.parquet')
    q = lambda s: {'mean': float(s.mean()), 'median': float(s.median()), 'p10': float(s.quantile(.1)),  # noqa: E731
                   'p90': float(s.quantile(.9))}
    info = {'n_bursts': len(jobs), 'n_frames': len(Fm), 'n_stored64': int(Fm.n_stored64.sum()),
            'n_stored32': int(Fm.n_stored32.sum()),
            'per_frame': {c: q(Fm[c]) for c in ('n_loose', 'n_tight', 'n_loose_on_animal', 'n_loose_no_dark_px',
                                                'n_tight_not_loose', 'n_tight32', 'n_loose32', 'n_tight_jpg',
                                                'n_tight_isolated', 'dark_px_recall_tight', 'dark_px_recall_loose',
                                                'iou_tight_jpg')},
            'loose_on_animal_share': float(Fm.n_loose_on_animal.sum() / Fm.n_loose.sum()),
            'loose_no_dark_px_share': float(Fm.n_loose_no_dark_px.sum() / Fm.n_loose.sum()),
            'frames_0_tight': int((Fm.n_tight == 0).sum()), 'frames_0_tight32': int((Fm.n_tight32 == 0).sum()),
            'tokens_by_role64': Fm.groupby('role').n_stored64.sum().to_dict(),
            'tokens_by_role32': Fm.groupby('role').n_stored32.sum().to_dict(),
            'wall_s': time.time() - t0, 'gpu_s': pipe.gpu_s, 'gpu': torch.cuda.get_device_name(0)}
    log(json.dumps(info))
    OUT.mkdir(parents=True, exist_ok=True)
    if sheet:
        mask_sheet(sheet, OUT / ('mask_sheet_selftest.jpg' if args.limit else 'mask_sheet.jpg'))
    (OUT / ('prepass_selftest.json' if args.limit else 'prepass.json')).write_text(json.dumps(info, indent=1))
    if not args.limit:
        R4.atomic_copy(loc, BEE / 'prepass')
        log(f'-> {BEE / "prepass"}')


def R4_isolated(torch, m):
    """(4096,) bool -> tokens of m with no 8-neighbour in m (64 grid)."""
    F = torch.nn.functional
    x = m.view(1, 1, G64, G64).float()
    nb = F.conv2d(x, torch.ones(1, 1, 3, 3, device=x.device), padding=1) - x
    return (nb.view(-1) == 0)


def mask_sheet(items, path):
    from PIL import Image, ImageDraw
    tiles = []
    for x, lo, ti, obs, k in items:
        im = np.asarray(Image.fromarray(x.transpose(1, 2, 0)).resize((512, 512), Image.BILINEAR)).astype(np.float32)
        up = lambda m: np.array(Image.fromarray(m.reshape(G64, G64).astype(np.uint8) * 255).resize(  # noqa: E731
            (512, 512), Image.NEAREST)) > 0
        L_, T_ = up(lo), up(ti)
        a = im.copy()
        a[L_ & ~T_] = 0.5 * a[L_ & ~T_] + 0.5 * np.array([255, 0, 0])
        a[T_] = 0.6 * a[T_] + 0.4 * np.array([0, 255, 0])
        t = Image.fromarray(a.clip(0, 255).astype(np.uint8))
        ImageDraw.Draw(t).text((6, 6), f'{obs} k{k} red=loose only, green=tight', fill=(255, 255, 0))
        tiles.append(t)
    sh = Image.new('RGB', (4 * 512, 512 * int(np.ceil(len(tiles) / 4))))
    for i, t in enumerate(tiles):
        sh.paste(t, ((i % 4) * 512, (i // 4) * 512))
    sh.save(path, quality=80)


# ---------------------------------------------------------------------------------------------- train
def train_one(cfg, seed, X, steps, out, dev, width, prefixes, n_init=500_000, bs=4096, lr=5e-4, warmup=500):
    """rollout_r4.train_one_r4 (= the benchmark train_one, token mode) with the width / prefixes as arguments."""
    import torch
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    N, din = X.shape
    ii = torch.from_numpy(np.sort(g.choice(N, min(n_init, N), replace=False))).to(dev)
    init = X[ii].float()
    norm = TokenNorm(din).to(dev).fit_blocks(init, [D] * (din // D), seed=seed)
    sae = MatryoshkaBatchTopKSAE(d_in=din, n_latents=width, prefixes=prefixes, k=K, seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    perm, p = torch.randperm(N, generator=gen, device=dev), 0
    history, t0 = [], time.time()
    for step in range(steps):
        if p + bs > N:
            perm, p = torch.randperm(N, generator=gen, device=dev), 0
        b = perm[p:p + bs]
        p += bs
        logs, gn = _train_step(sae, opt, norm(X[b]), step, steps, lr, warmup, 1.0)
        if step % 2000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            history.append(logs)
    sae.eval()
    extra = {'config': 'nnnt_round1', 'cfg': cfg, 'seed': seed}
    save_checkpoint(out / 'sae.pt', sae, norm, extra=extra)
    info = {**extra, 'n_pool_tokens': N, 'steps': steps, 'train_time_s': time.time() - t0,
            'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()), 'history': history}
    (out / 'train.json').write_text(json.dumps(info, indent=1, default=float))
    return info


def load_store(kind, root):
    """-> (tok memmap, prev memmap, role per token, frames DataFrame, token mask) of a kind's training store."""
    if kind == 'u64L':
        F = pd.read_parquet(root / 'r4' / 'frames.parquet')
        tf = np.load(root / 'r4' / 'tok_frame.npy')
        N = len(tf)
        tok = np.memmap(root / 'r4' / 'tok.f16', np.float16, 'r', shape=(N, D))
        prv = np.memmap(root / 'r4' / 'prev.f16', np.float16, 'r', shape=(N, D))
        return tok, prv, F.role.values[tf], np.ones(N, bool)
    pre = root / 'nnnt'
    F = pd.read_parquet(pre / 'frames.parquet')
    g = '64' if KINDS[kind][2] == 64 else '32'
    tf = np.load(pre / f'tok{g}_frame.npy')
    N = len(tf)
    tok = np.memmap(pre / f'tok{g}.f16', np.float16, 'r', shape=(N, D))
    prv = np.memmap(pre / f'prev{g}.f16', np.float16, 'r', shape=(N, D))
    if g == '64':
        fl = np.load(pre / 'tok64_flag.npy')
        sel = (fl & (1 if KINDS[kind][1] == 'loose' else 2)) > 0
    else:
        sel = np.ones(N, bool)
    return tok, prv, F.role.values[tf], sel


def cmd_train(args):
    import torch
    from src.eci.sae import load_sae
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    dev = torch.device('cuda')
    root = local_dir() / 'stores'
    t0 = time.time()
    R4.atomic_copy(R4BEE / 'prepass', root / 'r4')
    R4.atomic_copy(BEE / 'prepass' if not args.selftest else local_dir() / 'prepass', root / 'nnnt')
    log(f'staged stores in {time.time() - t0:.0f}s')
    todo = [x for x in sae_list() if not args.arms or x[1] in args.arms.split(',')]
    res = json.loads((OUT / 'train.json').read_text()) if (OUT / 'train.json').exists() and not args.selftest else {}
    groups = {}
    for name, arm, w, s in todo:
        groups.setdefault(ARMS[arm], []).append((name, arm, w, s))
    for (kind, var), items in groups.items():
        tok, prv, role, sel = load_store(kind, root)

        def X_of(r):
            ii = np.flatnonzero((role == r) & sel)
            X = torch.empty((len(ii), D * (2 if var == 'd2' else 1)), dtype=torch.float16, device=dev)
            for a in range(0, len(ii), 200_000):
                s_ = ii[a:a + 200_000]
                t = torch.from_numpy(np.ascontiguousarray(tok[s_])).to(dev)
                X[a:a + len(s_), :D] = t
                if var == 'd2':
                    X[a:a + len(s_), D:] = (t.float() - torch.from_numpy(np.ascontiguousarray(prv[s_])).to(dev).float()
                                            ).half()
            return X
        Xtr = X_of('train')
        log(f'{kind} {var}: {len(Xtr):,} train tokens')
        for name, arm, w, s in items:
            od = OUT / ('sae_selftest' if args.selftest else 'sae') / name
            if (od / 'sae_theta.pt').exists() and not args.force:
                log(f'[SKIP] {name}')
                continue
            cfg = {'arm': arm, 'kind': kind, 'variant': var, 'width': w, 'prefixes': list(prefixes_of(w)),
                   'k': K, 'encoder': R4.D3L, 'layer': R4.LAYER, 'S': S, 'mask': KINDS[kind][1],
                   'grid': KINDS[kind][2], 'rotated': KINDS[kind][0], 'topm': TOPM[KINDS[kind][2]]}
            ti = train_one(cfg, s, Xtr, args.steps, od, dev, w, prefixes_of(w))
            res[name] = {'cfg': cfg, 'n_train_tokens': int(len(Xtr)), 'train_time_s': ti['train_time_s'],
                         'dead_at_end_train': ti['dead_at_end_train']}
            log(f'{name}: trained {ti["train_time_s"]:.0f}s')
        del Xtr
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = False
        Xca, Xev = X_of('calib'), X_of('eval')
        for name, arm, w, s in items:
            od = OUT / ('sae_selftest' if args.selftest else 'sae') / name
            if (od / 'sae_theta.pt').exists() and not args.force and 'eval_fve' in res.get(name, {}):
                continue
            sae, norm, ck = load_sae(od / 'sae.pt', dev)
            th, info = calibrate_threshold(sae, Xca, norm=norm)
            set_threshold(sae, th)
            stt = sparsity_stats(sae, Xev, th, norm=norm)
            fire = torch.zeros(w, dtype=torch.float64, device=dev)
            with torch.no_grad():
                for a in range(0, len(Xev), 65536):
                    fire += (sae.encode(norm(Xev[a:a + 65536].float()), mode='threshold') > 0).sum(0).double()
            fr = (fire / len(Xev)).cpu().numpy()
            ck['state_dict']['threshold'] = torch.tensor(float(th))
            ck['theta_star'] = float(th)
            torch.save(ck, od / 'sae_theta.pt')
            res.setdefault(name, {}).update({'theta_star': th, 'eval_fve': stt['fve'], 'eval_l0': stt['l0'],
                                             'eval_dead_frac': float((fr == 0).mean()), 'n_calib_tokens': len(Xca),
                                             'n_eval_tokens': len(Xev)})
            log(f'{name}: theta* {th:.4f} eval FVE {stt["fve"]:.4f} L0 {stt["l0"]:.2f} dead {(fr == 0).mean():.3f}')
        del Xca, Xev
        torch.cuda.empty_cache()
    (OUT / ('train_selftest.json' if args.selftest else 'train.json')).write_text(json.dumps(res, indent=1,
                                                                                             default=float))


def load_all_saes(dev, which, selftest=False):
    """which 'u' (A0 + A0ref) | 'r' (A1-A5) -> {kind: [(name, sae, norm, var)]}."""
    from src.eci.sae import load_sae
    out = {}
    root = OUT / ('sae_selftest' if selftest else 'sae')
    for name, arm, w, s in sae_list():
        kind, var = ARMS[arm]
        if (kind == 'u64L') != (which == 'u'):
            continue
        f = root / name / 'sae_theta.pt'
        if not f.exists():
            if selftest:
                continue
            raise SystemExit(f'missing {f}')
        sae, norm, ck = load_sae(f, dev)
        assert abs(float(sae.threshold) - ck['theta_star']) < 1e-6
        out.setdefault(kind, []).append((name, sae, norm, var))
    if which == 'u':
        best = json.loads((R4OUT / 'sae' / 'best.json').read_text())['d2']
        sae, norm, ck = load_sae(R4OUT / 'sae' / best['name'] / 'sae.pt', dev)
        assert abs(float(sae.threshold) - best['theta_star']) < 1e-6
        out.setdefault('u64L', []).append(('A0ref', sae, norm, 'd2'))
    return out


# ---------------------------------------------------------------------------------------------- encode
def cmd_encode(args):
    import torch
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    P = load_plan()
    _, Rr = R4.obs_ranges_of(REPO / C['ann'])
    outdir = Path(args.out_dir) if args.out_dir else BEE / 'videos'
    outdir.mkdir(parents=True, exist_ok=True)
    vids = P['shards'][args.shard] if args.videos is None else [int(x) for x in args.videos.split(',')]
    todo = [vi for vi in vids if not (outdir / f'{P["videos"][vi]["obs"]}.npz').exists()]
    log(f'shard {args.shard}: {len(vids)} videos, {len(todo)} to do -> {outdir}')
    if not todo:
        return
    fp = pd.read_csv(REPO / C['ann'], usecols=['frame_path'])['frame_path'].values
    jdir = local_dir() / 'jpg'
    loc = local_dir() / 'enc'
    loc.mkdir(parents=True, exist_ok=True)
    jobs, rels = [], []
    for vi in todo:
        v = P['videos'][vi]
        n = v['n'] if not args.max_frames else min(v['n'], args.max_frames)
        rels += list(fp[v['src_lo']:v['src_lo'] + n])
        for k0 in range(0, n, C['chunk']):
            m_ = min(C['chunk'], n - k0)
            jobs.append(((vi, k0, n), dict(path=v['path'], s0=v['offset'] + C['step'] * k0, n=m_, step=C['step'], S=S,
                                           hflip=v['hflip'], fps=C['fps'], threads=C['threads'],
                                           jpgs=[jdir / f for f in fp[v['src_lo'] + k0:v['src_lo'] + k0 + m_]])))
    st = ThreadPoolExecutor(1).submit(R4.stage_jpgs, rels, jdir)
    pipe = Pipe(dev, Rr)
    SR, SU = load_all_saes(dev, 'r', args.selftest), load_all_saes(dev, 'u', args.selftest)
    st.result()
    pf = R4.Prefetch(jobs, C['par'])
    T0 = time.time()
    state = {}
    tot = 0

    def finish(vi, s_):
        v = P['videos'][vi]
        arr = {k_: np.concatenate(x) for k_, x in s_['acc'].items()}
        f = loc / f'{v["obs"]}.npz'
        np.savez(f, **arr, obs=v['obs'], n=len(arr['k']), rot=v['rot'], fps1_off=v['fps1_off'])
        R4.atomic_copy(f, outdir / f.name)
        f.unlink()
        dt = time.time() - s_['t0']
        log(f'{v["obs"]} rot {v["rot"]}: {len(arr["k"])} frames in {dt:.0f}s ({len(arr["k"]) / dt:.1f} fps) | kept/frame '
            + json.dumps({k_[3:]: round(float(arr[k_].mean()), 1) for k_ in arr if k_.startswith('nk_')}) +
            f' | fps1 frames {len(arr["ku"])}')

    cur = None
    for (vi, k0, n), (fr, jp) in pf:
        v = P['videos'][vi]
        if vi != cur:
            if cur is not None:
                finish(cur, state)
            cur = vi
            state = {'t0': time.time(), 'cr': Ctx(), 'cu': Ctx(), 'acc': {}}
        acc = state['acc']
        add = lambda k_, x: acc.setdefault(k_, []).append(x)  # noqa: E731
        sub = lambda k: (k - v['fps1_off']) % 5 == 0 and k >= v['fps1_off']  # noqa: E731
        for b in range(0, len(fr), C['bs']):
            kk = k0 + b + np.arange(len(fr[b:b + C['bs']]))
            rows = v['src_lo'] + kk
            ft = torch.from_numpy(fr[b:b + C['bs']]).to(dev)
            jt = torch.from_numpy(jp[b:b + C['bs']]).to(dev)
            with torch.no_grad():
                o = pipe.run(ft, jt, rows, v['obs'], 'odor', v['rot'])
                state['cr'].put(kk, o['z'])
                zp = state['cr'].prev(kk)
                add('k', kk.astype(np.int32))
                add('n32_loose', o['n32'].to(torch.int16).cpu().numpy())
                add('nk_r64_onanimal', (o['loose'] & (o['dark64'] >= THR_TIGHT)).sum(1).to(torch.int16).cpu().numpy())
                for kind, saes in SR.items():
                    z_, zp_, m_ = kind_tokens(kind, o, zp)
                    r, nk = apply_kind(torch, saes, z_, zp_, m_, TOPM[KINDS[kind][2]])
                    add(f'nk_{kind}', nk.to(torch.int16).cpu().numpy())
                    for name, (mx, tp) in r.items():
                        add(f'{name}|max', mx.cpu().numpy())
                        add(f'{name}|top', tp.cpu().numpy())
                state['cr'].prune(int(kk[-1]))
                # unrotated pass on the fps1 frames (+ frame 0 as the clipped partner)
                su = np.array([sub(k) or k == 0 for k in kk])
                if su.any():
                    ii = torch.from_numpy(np.flatnonzero(su)).to(dev)
                    ou = pipe.run(ft[ii], jt[ii], rows[su], v['obs'], 'none', 0)
                    state['cu'].put(kk[su], ou['z'])
                    keep = np.array([sub(k) for k in kk[su]])
                    if keep.any():
                        ik = torch.from_numpy(np.flatnonzero(keep)).to(dev)
                        zpu = state['cu'].prev(kk[su][keep])
                        ou_k = {'z': ou['z'][ik], 'loose': ou['loose'][ik], 'dark64': ou['dark64'][ik]}
                        add('ku', kk[su][keep].astype(np.int32))
                        add('n32u_loose', ou['n32'][ik].to(torch.int16).cpu().numpy())
                        for kind, saes in SU.items():
                            z_, zp_, m_ = kind_tokens(kind, ou_k, zpu)
                            r, nk = apply_kind(torch, saes, z_, zp_, m_, TOPM[64])
                            add(f'nk_{kind}', nk.to(torch.int16).cpu().numpy())
                            for name, (mx, tp) in r.items():
                                add(f'{name}|max', mx.cpu().numpy())
                                add(f'{name}|top', tp.cpu().numpy())
                    state['cu'].prune(int(kk[-1]))
        tot += len(fr)
    if cur is not None:
        finish(cur, state)
    wall = time.time() - T0
    log(f'shard {args.shard} DONE: {tot:,} frames in {wall:.0f}s ({tot / wall:.1f} fps wall), GPU busy (pipe) '
        f'{pipe.gpu_s:.0f}s, decode thread-s {pf.dec_s:.0f}, {torch.cuda.get_device_name(0)}')


# ---------------------------------------------------------------------------------------------- score
NB = 0x7C01


def half_stats(torch, X, Y, rows, block=512):
    """X (n, W) fp16 >= 0 cuda, Y (n, L) bool cuda, rows LongTensor -> per neuron x label: R~ (normalised
    R-precision, ties at the cut pro rata), AUROC (Mann-Whitney, ties averaged), AP (src/eci/levers.align_gpu
    definitions); rate (W,) = share of frames > 0; base (L,)."""
    Yr = Y[rows]
    n, L = Yr.shape
    W = X.shape[1]
    Pn = Yr.sum(0).double()
    Nn = n - Pn
    Rt = torch.full((W, L), float('nan'), dtype=torch.float64, device=X.device)
    Au, Ap = Rt.clone(), Rt.clone()
    rate = torch.zeros(W, dtype=torch.float64, device=X.device)
    Xr = X.index_select(0, rows)
    for j0 in range(0, W, block):
        xb = Xr[:, j0:j0 + block]
        b = xb.shape[1]
        bits = (xb.contiguous().view(torch.int16).long() & 0x7FFF) + NB * torch.arange(b, device=X.device)[None]
        tot = torch.bincount(bits.flatten(), minlength=b * NB).view(b, NB).double()
        rate[j0:j0 + b] = 1 - tot[:, 0] / n
        tot_d = tot.flip(1)
        ccnt = tot_d.cumsum(1)
        for li in range(L):
            if Pn[li] == 0 or Nn[li] == 0:
                continue
            pos = torch.bincount(bits[Yr[:, li]].flatten(), minlength=b * NB).view(b, NB).double().flip(1)
            neg = tot_d - pos
            ctp, cfp = pos.cumsum(1), neg.cumsum(1)
            Au[j0:j0 + b, li] = (pos * (Nn[li] - cfp) + 0.5 * pos * neg).sum(1) / (Pn[li] * Nn[li])
            Ap[j0:j0 + b, li] = (pos / Pn[li] * ctp / (ctp + cfp).clamp_min(1)).sum(1)
            k = Pn[li]
            ib = torch.searchsorted(ccnt.contiguous(), k.expand(b, 1).contiguous()).clamp_max(NB - 1)
            cb, tb, pb = ccnt.gather(1, ib)[:, 0], tot_d.gather(1, ib)[:, 0], pos.gather(1, ib)[:, 0]
            tp = ctp.gather(1, ib)[:, 0] - pb + pb * (k - (cb - tb)) / tb.clamp_min(1)
            pi = k / n
            Rt[j0:j0 + b, li] = (tp / k - pi) / (1 - pi)
        del bits
    return {'R': Rt.cpu().numpy(), 'A': Au.cpu().numpy(), 'AP': Ap.cpu().numpy(), 'rate': rate.cpu().numpy(),
            'base': (Pn / n).cpu().numpy(), 'n': n}


def splits_of(units):
    """{split name: half (n_units,) 0/1}: 'r4' (sorted units alternating, rollout_r4.cmd_align) + 10 random 12/12."""
    u = sorted(units)
    out = {'r4': np.array([i % 2 for i in range(len(u))])}
    for s in range(N_SPLITS):
        h = np.zeros(len(u), int)
        h[np.random.default_rng(1000 + s).permutation(len(u))[:len(u) // 2]] = 1
        out[f'rnd{s}'] = h
    return u, out


def score_set(torch, X, Y, half_of, dev):
    """All splits for one code matrix X (n, W) -> summary dict (per label, per split) + in-sample arrays."""
    L = Y.shape[1]
    out = {'splits': {}}
    for sn, half in half_of.items():
        H = {}
        for h in (0, 1):
            H[h] = half_stats(torch, X, Y, torch.from_numpy(np.flatnonzero(half == h)).to(dev))
        rec = {}
        for li, lb in enumerate(LABELS[:L]):
            dirs = []
            for s_, t_ in ((0, 1), (1, 0)):
                sc = np.where(np.isfinite(H[s_]['R'][:, li]), H[s_]['R'][:, li], -np.inf)
                j = int(np.argmax(sc))
                sa = np.where(np.isfinite(H[s_]['A'][:, li]), H[s_]['A'][:, li], -np.inf)
                ja = int(np.argmax(sa))
                dirs.append({'neuron': j, 'R': float(H[t_]['R'][j, li]), 'A_at_R': float(H[t_]['A'][j, li]),
                             'AP_lift_at_R': float(H[t_]['AP'][j, li] / H[t_]['base'][li]),
                             'rate_at_R': float(H[t_]['rate'][j]),
                             'cross_R': [float(H[t_]['R'][j, l2]) for l2 in range(L)],
                             'neuron_A': ja, 'A': float(H[t_]['A'][ja, li])})
            rec[lb] = {k_: float(np.mean([d[k_] for d in dirs])) for k_ in ('R', 'A_at_R', 'AP_lift_at_R',
                                                                             'rate_at_R', 'A')}
            rec[lb]['cross_R'] = np.mean([d['cross_R'] for d in dirs], 0).tolist()
            rec[lb]['neurons'] = [d['neuron'] for d in dirs]
        out['splits'][sn] = rec
    rnd = [k_ for k_ in half_of if k_.startswith('rnd')]
    out['mean'] = {lb: {k_: float(np.mean([out['splits'][s][lb][k_] for s in rnd]))
                        for k_ in ('R', 'A_at_R', 'AP_lift_at_R', 'rate_at_R', 'A')} for lb in LABELS[:L]}
    for lb in LABELS[:L]:
        out['mean'][lb]['cross_R'] = np.mean([out['splits'][s][lb]['cross_R'] for s in rnd], 0).tolist()
        out['mean'][lb]['R_sd_splits'] = float(np.std([out['splits'][s][lb]['R'] for s in rnd]))
        sel = [j for s in rnd for j in out['splits'][s][lb]['neurons']]
        vals, cnt = np.unique(sel, return_counts=True)
        out['mean'][lb]['modal_neuron'] = int(vals[np.argmax(cnt)])
        out['mean'][lb]['modal_share'] = float(cnt.max() / len(sel))
    out['primary'] = float(np.mean([out['mean']['nose_nose']['R'], out['mean']['nose_tail']['R']]))
    # in-sample, all frames: specificity of the modal nn / nt neurons
    allr = torch.arange(X.shape[0], device=dev)
    A = half_stats(torch, X, Y, allr)
    jn, jt = out['mean']['nose_nose']['modal_neuron'], out['mean']['nose_tail']['modal_neuron']
    ntop = max(1, int(round(0.01 * X.shape[0])))
    tn = set(torch.topk(X[:, jn].float(), ntop).indices.cpu().numpy().tolist())
    tt = set(torch.topk(X[:, jt].float(), ntop).indices.cpu().numpy().tolist())
    out['specificity'] = {'nn_neuron': jn, 'nt_neuron': jt, 'same_neuron': bool(jn == jt),
                          'top1pct_jaccard': len(tn & tt) / len(tn | tt),
                          'insample_R_nn_neuron': {lb: float(A['R'][jn, li]) for li, lb in enumerate(LABELS[:L])},
                          'insample_R_nt_neuron': {lb: float(A['R'][jt, li]) for li, lb in enumerate(LABELS[:L])},
                          'rate_nn_neuron': float(A['rate'][jn]), 'rate_nt_neuron': float(A['rate'][jt]),
                          'neurons_ever_firing': int((A['rate'] > 0).sum())}
    return out, A


def cmd_score(args):
    import torch
    dev = torch.device('cuda')
    P = load_plan()
    T = {} if args.refs_only else json.loads((OUT / 'train.json').read_text())
    vids = [v for v in P['videos'] if v['annotated']]
    a = pd.read_csv(REPO / C['ann'], usecols=['observation_id', 'Y_nn', 'Y_np', 'Y_nt'])
    rows = np.concatenate([np.arange(v['src_lo'], v['src_lo'] + v['n']) for v in vids])
    y = a.iloc[rows]
    assert y[['Y_nn', 'Y_np', 'Y_nt']].notna().all().all()
    Yall = np.stack([(y.Y_nn.values > 0) | (y.Y_np.values > 0), y.Y_nt.values > 0, y.Y_nn.values > 0,
                     y.Y_np.values > 0], 1)
    obs = y.observation_id.values
    unit = pd.Series(obs).str.rsplit('_', n=2).str[0].values
    units, SPL = splits_of(set(unit))
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    # fps1 subset (rows of the 'all' order)
    f1 = np.concatenate([np.flatnonzero(np.isin(np.arange(v['n']), np.arange(v['fps1_off'], v['n'], 5))) +
                         sum(w['n'] for w in vids[:i]) for i, v in enumerate(vids)])
    sets = {'all': np.arange(len(rows)), 'fps1': f1}
    log(f'{len(rows):,} annotated frames, {len(vids)} videos, {len(units)} pools; fps1 {len(f1):,}; base rates ' +
        json.dumps({lb: float(Yall[:, i].mean()) for i, lb in enumerate(LABELS)}))
    res = {'n_frames': {k_: int(len(v_)) for k_, v_ in sets.items()}, 'units': units,
           'splits': {k_: v_.tolist() for k_, v_ in SPL.items()},
           'base_rate': {fs: {lb: float(Yall[ix, i].mean()) for i, lb in enumerate(LABELS)} for fs, ix in sets.items()},
           'sets': {}}
    prev = {}
    if (OUT / 'score.json').exists() and not args.force:
        prev = json.loads((OUT / 'score.json').read_text()).get('sets', {})
    vdir = BEE / 'videos'
    Z = {} if args.refs_only else {v['obs']: np.load(vdir / f'{v["obs"]}.npz') for v in vids}
    for v in ([] if args.refs_only else vids):  # integrity
        z = Z[v['obs']]
        assert int(z['n']) == v['n'] and (z['k'] == np.arange(v['n'])).all(), v['obs']
        assert (z['ku'] == np.arange(v['fps1_off'], v['n'], 5)).all(), v['obs']
    insample = {}

    def run(name, X, fs):
        key = f'{name}|{fs}'
        if key in prev:
            res['sets'][key] = prev[key]
            return
        t0 = time.time()
        ix = sets[fs]
        Xg = torch.from_numpy(np.ascontiguousarray(X)).to(dev)
        assert Xg.shape[0] == len(ix)
        Yg = torch.from_numpy(Yall[ix]).to(dev)
        half_of = {sn: h[uidx[ix]] for sn, h in SPL.items()}
        out, A = score_set(torch, Xg, Yg, half_of, dev)
        out['width'] = int(X.shape[1])
        res['sets'][key] = out
        insample[key] = A['R'].astype(np.float32)
        m = out['mean']
        log(f'{key}: nn R~ {m["nose_nose"]["R"]:.4f} AUROC {m["nose_nose"]["A"]:.4f} | nt R~ {m["nose_tail"]["R"]:.4f} '
            f'AUROC {m["nose_tail"]["A"]:.4f} | r4 split nn {out["splits"]["r4"]["nose_nose"]["R"]:.4f} '
            f'({time.time() - t0:.0f}s)')
        del Xg, Yg
        torch.cuda.empty_cache()
        if len(res['sets']) % 10 == 0:  # partial save; a re-run resumes from it
            (OUT / 'score.json').write_text(json.dumps(res, indent=1, default=float))

    # references: stored r4 d2 codes, C1, deployed (annotations.csv rows)
    refs = {'r4_d2_stored': R4BEE / 'codes' / 'matryoshka_btk_512_k16_r4d2_s0',
            'C1': REPO / C['c1'], 'deployed': REPO / C['deployed']}
    for nm, cd in refs.items():
        cm = np.load(cd / 'codes_max.npy', mmap_mode='r')
        X = np.empty((len(rows), cm.shape[1]), np.float16)
        for v_, lo in zip(vids, np.cumsum([0] + [w['n'] for w in vids])[:-1]):
            X[lo:lo + v_['n']] = cm[v_['src_lo']:v_['src_lo'] + v_['n']]
        run(nm, X, 'all')
        run(nm, X[sets['fps1']], 'fps1')
        if nm == 'r4_d2_stored':
            r4X = X[sets['fps1']].copy()
        del X
    if args.refs_only:
        (OUT / 'score_refs.json').write_text(json.dumps(res, indent=1, default=float))
        return
    # A0ref vs stored r4 codes on the fps1 frames (pipeline check)
    Xa = np.concatenate([Z[v['obs']]['A0ref|max'] for v in vids])
    d = np.abs(Xa.astype(np.float32) - r4X.astype(np.float32))
    res['A0ref_vs_r4_stored'] = {'max_abs_diff': float(d.max()), 'mean_abs_diff': float(d.mean()),
                                 'fire_agreement': float(((Xa > 0) == (r4X > 0)).mean()),
                                 'rel_l1': float(d.sum() / max(np.abs(r4X.astype(np.float32)).sum(), 1e-9))}
    log(f'A0ref vs stored r4 codes: {json.dumps(res["A0ref_vs_r4_stored"])}')
    del Xa, r4X, d
    for name, arm, w, s in [('A0ref', 'A0', 512, -1)] + sae_list():
        for ro in ('max', 'top'):
            if args.refs_only:
                continue
            X = np.concatenate([Z[v['obs']][f'{name}|{ro}'] for v in vids])
            if arm == 'A0':
                run(f'{name}|{ro}', X, 'fps1')
            else:
                run(f'{name}|{ro}', X, 'all')
                run(f'{name}|{ro}', X[sets['fps1']], 'fps1')
            del X
    res['train'] = T
    res['kept'] = {k_: float(np.concatenate([Z[v['obs']][f'nk_{k_}'] for v in vids]).mean())
                   for k_ in (*KINDS, 'r64_onanimal')}
    (OUT / 'score.json').write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(OUT / 'insample_R.npz', **{k_.replace('|', '__'): v_ for k_, v_ in insample.items()})
    report(res)


def welch(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else 1
    h = stats.t.ppf(0.975, df) * se
    return float(d), float(d - h), float(d + h)


def paired(a, b):
    from scipy import stats
    x = np.asarray(a, float) - np.asarray(b, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def report(res=None):
    """table.md + decisions.json from score.json (pre-registered rule, module docstring)."""
    res = res or json.loads((OUT / 'score.json').read_text())
    Sx = res['sets']

    def vals(arm, w, ro, seeds, fs, lb, q='R'):
        return [Sx[f'{arm}_w{w}_s{s}|{ro}|{fs}']['mean'][lb][q] if lb != 'primary' else
                Sx[f'{arm}_w{w}_s{s}|{ro}|{fs}']['primary'] for s in seeds if f'{arm}_w{w}_s{s}|{ro}|{fs}' in Sx]
    ms = lambda x: f'{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}' if len(x) > 1 else (f'{x[0]:.3f}' if x else '-')  # noqa
    lines = ['# NN/NT round 1 — results', '',
             'R~ = cross-fitted normalised R-precision (neuron selected on one pool half, scored on the other; mean over '
             '10 random pool splits); AUROC = cf AUROC (AUROC-selected); mean ± sd over SAE seeds.', '',
             f'Frames: all = {res["n_frames"]["all"]:,} annotated frames (5 fps); fps1 = {res["n_frames"]["fps1"]:,} '
             '(r4 subset). Base rates (all): ' + ', '.join(f'{k_} {v_:.4f}' for k_, v_ in res['base_rate']['all'].items()),
             '']
    for fs in ('all', 'fps1'):
        lines += [f'## Frames: {fs}', '', '| arm | W | readout | seeds | nn R~ | nn AUROC | nt R~ | nt AUROC | '
                  'nn_mutual R~ | np R~ | primary |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for arm in ARMS:
            for w in (512, 1024):
                for ro in ('max', 'top'):
                    for sd_name, sds in (('0-2', SEEDS3), ('0-5', SEEDS6)):
                        if w == 1024 and sd_name == '0-5':
                            continue
                        r = [vals(arm, w, ro, sds, fs, lb, q) for lb, q in
                             (('nose_nose', 'R'), ('nose_nose', 'A'), ('nose_tail', 'R'), ('nose_tail', 'A'),
                              ('nn_mutual', 'R'), ('np_directional', 'R'), ('primary', 'R'))]
                        if not r[0]:
                            continue
                        lines.append(f'| {arm} | {w} | {ro} | {sd_name} | ' + ' | '.join(ms(x) for x in r) + ' |')
        for nm in ('r4_d2_stored', 'C1', 'deployed', 'A0ref|max', 'A0ref|top'):
            k_ = f'{nm}|{fs}'
            if k_ in Sx:
                m = Sx[k_]['mean']
                lines.append(f'| {nm} (ref) | {Sx[k_]["width"]} | max | 1 | {m["nose_nose"]["R"]:.3f} | '
                             f'{m["nose_nose"]["A"]:.3f} | {m["nose_tail"]["R"]:.3f} | {m["nose_tail"]["A"]:.3f} | '
                             f'{m["nn_mutual"]["R"]:.3f} | {m["np_directional"]["R"]:.3f} | {Sx[k_]["primary"]:.3f} |')
        lines.append('')
    # r4-split reproduction
    lines += ['## Reproduction (r4 split, fps1 frames)', '']
    for nm in ('r4_d2_stored', 'C1', 'deployed', 'A0ref|max'):
        k_ = f'{nm}|fps1'
        if k_ in Sx:
            sp = Sx[k_]['splits']['r4']
            lines.append(f'- {nm}: nn R~ {sp["nose_nose"]["R"]:.4f} cf AUROC {sp["nose_nose"]["A"]:.4f}; nt R~ '
                         f'{sp["nose_tail"]["R"]:.4f} cf AUROC {sp["nose_tail"]["A"]:.4f}')
    if 'A0ref_vs_r4_stored' in res:
        lines.append(f'- A0ref codes vs stored r4 codes: {json.dumps(res["A0ref_vs_r4_stored"])}')
    lines.append('')
    # decision rule
    cand = [(arm, ro) for arm in ('A1', 'A2', 'A3', 'A4', 'A5') for ro in ('max', 'top')]
    r1 = {f'{a}|{r}': float(np.mean(vals(a, 512, r, SEEDS3, 'all', 'primary'))) for a, r in cand
          if vals(a, 512, r, SEEDS3, 'all', 'primary')}
    dec = {'round1_primary_seeds0-2': r1}
    if r1:
        best = max(r1, key=r1.get)
        ba, br = best.split('|')
        dec['best'] = best
        dec['confirm'] = {}
        for ref in ('A1|max', 'A2|max'):
            ra, rr = ref.split('|')
            if ref == best:
                continue
            for sds, tag in ((SEEDS6, 'seeds0-5'), ((3, 4, 5), 'seeds3-5')):
                for lb in ('primary', 'nose_nose', 'nose_tail'):
                    dec['confirm'][f'{best} - {ref} {tag} {lb} R~'] = welch(vals(ba, 512, br, sds, 'all', lb),
                                                                           vals(ra, 512, rr, sds, 'all', lb))
        for lb in ('primary', 'nose_nose', 'nose_tail'):
            dec['confirm'][f'{best} - A0|max fps1 seeds0-5 {lb} R~'] = welch(vals(ba, 512, br, SEEDS6, 'fps1', lb),
                                                                            vals('A0', 512, 'max', SEEDS6, 'fps1', lb))
    eff = {}
    comps = [('rotation A1-A0 (fps1)', 'A1', 'A0', 'fps1'), ('tight mask A2-A1', 'A2', 'A1', 'all'),
             ('no change channel A3-A2 (64 grid)', 'A3', 'A2', 'all'), ('no change channel A5-A4 (32 grid)', 'A5', 'A4', 'all'),
             ('2x2 pooling A4-A2 (d2)', 'A4', 'A2', 'all'), ('2x2 pooling A5-A3 (d0)', 'A5', 'A3', 'all')]
    for nm, a1, a2, fs in comps:
        for ro in ('max', 'top'):
            for lb, q in (('primary', 'R'), ('nose_nose', 'R'), ('nose_tail', 'R'), ('nose_nose', 'A'),
                          ('nose_tail', 'A')):
                x, y_ = vals(a1, 512, ro, SEEDS6, fs, lb, q), vals(a2, 512, ro, SEEDS6, fs, lb, q)
                if len(x) > 1 and len(y_) > 1:
                    eff[f'{nm} | {ro} | {lb} {"R~" if q == "R" else "AUROC"}'] = welch(x, y_)
    for arm in ARMS:
        fs = 'fps1' if arm == 'A0' else 'all'
        for lb in ('primary', 'nose_nose', 'nose_tail'):
            x, y_ = vals(arm, 512, 'top', SEEDS6, fs, lb), vals(arm, 512, 'max', SEEDS6, fs, lb)
            if len(x) > 1 and len(x) == len(y_):
                eff[f'top - max readout {arm} (paired) | {lb} R~'] = paired(x, y_)
    for arm in ('A2', 'A3', 'A4', 'A5'):
        for ro in ('max', 'top'):
            x, y_ = vals(arm, 1024, ro, SEEDS3, 'all', 'primary'), vals(arm, 512, ro, SEEDS3, 'all', 'primary')
            if len(x) > 1 and len(y_) > 1:
                eff[f'width 1024-512 {arm} | {ro} | primary R~ (seeds 0-2)'] = welch(x, y_)
    dec['effects'] = eff
    lines += ['## Decision (pre-registered rule)', '', '```', json.dumps(dec, indent=1), '```', '',
              '## Specificity (modal cross-fit nn / nt neurons, in-sample on the scored frames)', '',
              '| set | nn neuron | nt neuron | same | top-1% Jaccard | R~ nn-neuron on nt | R~ nt-neuron on nn | '
              'cf cross R~ (nn sel -> nt) | cf cross R~ (nt sel -> nn) | modal share nn / nt | ever firing |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for k_, s_ in Sx.items():
        if '_s0|' not in k_ and not k_.startswith(('C1', 'deployed', 'r4_', 'A0ref')):
            continue
        sp, m = s_['specificity'], s_['mean']
        lines.append(f'| {k_} | {sp["nn_neuron"]} | {sp["nt_neuron"]} | {sp["same_neuron"]} | '
                     f'{sp["top1pct_jaccard"]:.3f} | {sp["insample_R_nn_neuron"]["nose_tail"]:.3f} | '
                     f'{sp["insample_R_nt_neuron"]["nose_nose"]:.3f} | {m["nose_nose"]["cross_R"][1]:.3f} | '
                     f'{m["nose_tail"]["cross_R"][0]:.3f} | {m["nose_nose"]["modal_share"]:.2f} / '
                     f'{m["nose_tail"]["modal_share"]:.2f} | {sp["neurons_ever_firing"]} |')
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'decisions.json').write_text(json.dumps(dec, indent=1))
    log('\n'.join(lines[:60]))


# ---------------------------------------------------------------------------------------------- locality
def cmd_locality(args):
    """Per arm (width 512, the seed with the highest held-out FVE: label-free) and readout: the nn / nt neuron selected
    by R~ on half 0 of split rnd0; its 24 top frames of half 1 (held out), at most one per video, re-encoded (frame and
    its t-10 partner) -> per-token activations. Locality on those frames: (a) share of activation mass on animal pixels
    = sum_t a_t f_t / sum_t a_t (f_t = animal-pixel fraction of token t's patch, pixel dark cue), vs the uniform-kept
    baseline mean_t f_t; mass share on body tokens (f_t >= 0.25); (b) spread = tokens with a_t >= 0.5 max_t a_t,
    absolute and / animal token area (tokens with f_t >= 0.25 at that grid). Contact sheets -> OUT/sheets/."""
    import torch
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    P = load_plan()
    T = json.loads((OUT / 'train.json').read_text())
    SC = json.loads((OUT / 'score.json').read_text())
    _, Rr = R4.obs_ranges_of(REPO / C['ann'])
    vids = [v for v in P['videos'] if v['annotated']]
    a = pd.read_csv(REPO / C['ann'], usecols=['observation_id', 'frame_path', 'Y_nn', 'Y_np', 'Y_nt'])
    units = SC['units']
    half = dict(zip(units, SC['splits']['rnd0']))
    pipe = Pipe(dev, Rr)
    vdir = BEE / 'videos'
    out, todo = {}, []
    arms = args.arms.split(',') if args.arms else list(ARMS)
    for arm in arms:
        seeds = [s for s in SEEDS6 if f'{arm}_w512_s{s}' in T]
        s = max(seeds, key=lambda s_: T[f'{arm}_w512_s{s_}']['eval_fve'])
        name = f'{arm}_w512_s{s}'
        fs = 'fps1' if arm == 'A0' else 'all'
        for ro in ('max', 'top'):
            key = f'{name}|{ro}|{fs}'
            for lb in ('nose_nose', 'nose_tail'):
                j = SC['sets'][key]['splits']['rnd0'][lb]['neurons'][0]  # selected on half 0
                cand = []
                for v in vids:
                    if half[v['obs'].rsplit('_', 2)[0]] != 1:
                        continue
                    z = np.load(vdir / f'{v["obs"]}.npz')
                    x = z[f'{name}|{ro}'][:, j].astype(np.float32)
                    kk = z['ku'] if arm == 'A0' else z['k']
                    i = int(np.argmax(x))
                    cand.append((float(x[i]), v, int(kk[i])))
                cand.sort(key=lambda t: -t[0])
                todo.append({'arm': arm, 'name': name, 'readout': ro, 'label': lb, 'neuron': j,
                             'frames': [(c[1], c[2], c[0]) for c in cand[:24]]})
    saes = {}
    for t in todo:
        if t['name'] not in saes:
            saes[t['name']] = load_sae(OUT / 'sae' / t['name'] / 'sae_theta.pt', dev)[:2]
    # decode every needed frame once (k and max(k - 10, 0))
    need = sorted({(v['obs'], kk) for t in todo for v, k, _ in t['frames'] for kk in (k, max(k - LAG, 0))})
    V = {v['obs']: v for v in vids}

    def dec(item):
        o, k = item
        v = V[o]
        fr = R4.decode(v['path'], v['offset'] + C['step'] * k, 1, C['step'], S, v['hflip'], C['fps'], threads=2)[0]
        from PIL import Image
        jp = np.asarray(Image.open(REPO / 'dataset' / a.frame_path.values[v['src_lo'] + k]).convert('RGB'))
        return item, fr, jp
    with ThreadPoolExecutor(16) as ex:
        FR = {it: (fr, jp) for it, fr, jp in ex.map(dec, need)}
    log(f'decoded {len(FR)} frames')
    cache = {}

    def enc(v, k, align):
        key = (v['obs'], k, align)
        if key not in cache:
            fr, jp = FR[(v['obs'], k)]
            o = pipe.run(torch.from_numpy(fr[None]).to(dev), torch.from_numpy(jp[None]).to(dev),
                         np.array([v['src_lo'] + k]), v['obs'], align, v['rot'] if align == 'odor' else 0)
            cache[key] = {k_: (x.clone() if hasattr(x, 'clone') else x) for k_, x in o.items()}
        return cache[key]
    sheets = OUT / 'sheets'
    sheets.mkdir(parents=True, exist_ok=True)
    for t in todo:
        kind, var = ARMS[t['arm']]
        rot_, mk, grid = KINDS[kind]
        sae, norm = saes[t['name']]
        recs, tiles = [], []
        for v, k, stored in t['frames']:
            align = 'odor' if rot_ else 'none'
            o, op = enc(v, k, align), enc(v, max(k - LAG, 0), align)
            with torch.no_grad():
                z_, zp_, m_ = kind_tokens(kind, o, op['z'])
                pos = torch.nonzero(m_[0]).flatten()
                x = sae_input(var, z_[0, pos], zp_[0, pos])
                act = torch.zeros(grid * grid, device=dev)
                act[pos] = sae.encode(norm(x.float()), mode='threshold')[:, t['neuron']]
            act = act.cpu().numpy()
            fdark = (o['dark64'] if grid == 64 else dark32_of(o['dark64']))[0].cpu().numpy()
            kept = m_[0].cpu().numpy()
            mm = TOPM[grid]
            re_ = act.max() if t['readout'] == 'max' else np.sort(act[kept])[::-1][:mm].sum() / mm
            tot = act.sum()
            animal = fdark >= THR_TIGHT
            hi = act >= 0.5 * act.max() if act.max() > 0 else np.zeros_like(kept)
            y = a.iloc[v['src_lo'] + k]
            recs.append({'obs': v['obs'], 'k': k, 'stored': stored, 'recomputed': float(re_),
                         'label': ''.join(c for c, col in (('nn', 'Y_nn'), ('np', 'Y_np'), ('nt', 'Y_nt'))
                                          if y[col] > 0) or '-',
                         'mass_animal_px': float((act * fdark).sum() / tot) if tot > 0 else np.nan,
                         'baseline_kept_animal_px': float(fdark[kept].mean()) if kept.any() else np.nan,
                         'mass_on_body_tokens': float(act[animal].sum() / tot) if tot > 0 else np.nan,
                         'mass_off_animal_tokens': float(act[fdark == 0].sum() / tot) if tot > 0 else np.nan,
                         'n_half_max': int(hi.sum()), 'animal_tokens': int(animal.sum()),
                         'spread_rel': float(hi.sum() / max(1, animal.sum())), 'n_active': int((act > 0).sum()),
                         'n_kept': int(kept.sum())})
            tiles.append((o['x'][0].cpu().numpy(), act.reshape(grid, grid), recs[-1]))
        R = pd.DataFrame(recs)
        summ = {c: float(R[c].mean()) for c in ('mass_animal_px', 'baseline_kept_animal_px', 'mass_on_body_tokens',
                                                 'mass_off_animal_tokens', 'n_half_max', 'animal_tokens', 'spread_rel',
                                                 'n_active', 'n_kept')}
        summ['single_token_frames'] = int((R.n_half_max == 1).sum())
        summ['label_counts'] = R.label.value_counts().to_dict()
        summ['recompute_max_rel_err'] = float((np.abs(R.recomputed - R.stored) / np.maximum(R.stored, 1e-6)).max())
        fn = sheets / f'{t["arm"]}_{t["readout"]}_{t["label"]}_n{t["neuron"]}.jpg'
        contact_sheet(tiles, fn, f'{t["name"]} {t["readout"]} neuron {t["neuron"]} ({t["label"]}-selected on pools '
                                 f'half 0; top held-out frames, 1 per video)', grid)
        key = f'{t["name"]}|{t["readout"]}|{t["label"]}'
        out[key] = {**{k_: t[k_] for k_ in ('arm', 'name', 'readout', 'label', 'neuron')}, 'summary': summ,
                    'frames': recs, 'sheet': str(fn)}
        log(f'{key}: ' + json.dumps(summ))
    (OUT / 'locality.json').write_text(json.dumps(out, indent=1, default=float))


def contact_sheet(tiles, path, title, grid, tw=256):
    from matplotlib import colormaps
    from PIL import Image, ImageDraw
    from src.eci.viz import _overlay_rgb
    pos_vals = np.concatenate([h[h > 0].ravel() for _, h, _ in tiles] + [np.zeros(1)])
    vmax = float(np.quantile(pos_vals[pos_vals > 0], 0.99)) if (pos_vals > 0).any() else 1.0
    cmap = colormaps['inferno']
    cols = 6
    cell = (2 * tw, tw + 14)
    sh = Image.new('RGB', (cols * cell[0], 20 + int(np.ceil(len(tiles) / cols)) * cell[1]), (255, 255, 255))
    ImageDraw.Draw(sh).text((4, 4), title + f'  | colour max = p99 {vmax:.2f}', fill=(0, 0, 0))
    for i, (x, h, r) in enumerate(tiles):
        img = x.transpose(1, 2, 0)
        full = _overlay_rgb(img, h, (0, 0, img.shape[0]), vmax, cmap)
        cy, cx = np.unravel_index(np.argmax(h), h.shape)
        cpx = img.shape[0] / grid
        c = 384
        y0 = int(np.clip((cy + 0.5) * cpx - c / 2, 0, img.shape[0] - c))
        x0 = int(np.clip((cx + 0.5) * cpx - c / 2, 0, img.shape[1] - c))
        zoom = full[y0:y0 + c, x0:x0 + c]
        t = Image.new('RGB', cell, (255, 255, 255))
        t.paste(Image.fromarray(full).resize((tw, tw), Image.BILINEAR), (0, 0))
        t.paste(Image.fromarray(zoom).resize((tw, tw), Image.BILINEAR), (tw, 0))
        dr = ImageDraw.Draw(t)
        dr.rectangle([x0 * tw / img.shape[1], y0 * tw / img.shape[0], (x0 + c) * tw / img.shape[1],
                      (y0 + c) * tw / img.shape[0]], outline=(0, 255, 255))
        dr.text((2, tw), f'{r["obs"]} k{r["k"]} [{r["label"]}] a={r["stored"]:.2f} hm={r["n_half_max"]}',
                fill=(0, 0, 0))
        sh.paste(t, ((i % cols) * cell[0], 20 + (i // cols) * cell[1]))
    sh.save(path, quality=80)


# ---------------------------------------------------------------------------------------------- cost
def cmd_cost(args):
    """GPU hours / utilisation per job log of this experiment."""
    out = {}
    for f in sorted((OUT / 'logs').glob('*.out')):
        ut, ts, name = [], [], None
        for line in f.read_text(errors='replace').splitlines():
            if line.startswith('[gpu] '):
                p_ = [x.strip() for x in line[6:].split(',')]
                try:
                    ts.append(pd.Timestamp(p_[0].replace('/', '-')))
                    name = p_[1]
                    ut.append(float(p_[2].split()[0]))
                except Exception:
                    pass
        if ut:
            out[f.name] = {'gpu': name, 'gpu_h': (ts[-1] - ts[0]).total_seconds() / 3600 + 1 / 60,
                           'util_mean': float(np.mean(ut)), 'samples': len(ut)}
            for line in f.read_text(errors='replace').splitlines():
                if ' DONE: ' in line and 'fps wall' in line:  # encode shard throughput
                    w = line.split()
                    out[f.name]['shard_frames'] = int(w[w.index('DONE:') + 1].replace(',', ''))
                    out[f.name]['shard_fps_wall'] = float(line.split('(')[1].split()[0])
    tot = {}
    for k_, v_ in out.items():
        st = k_.split('_')[0]
        tot.setdefault(st, [0.0, []])
        tot[st][0] += v_['gpu_h']
        tot[st][1].append(v_['util_mean'])
    summ = {k_: {'gpu_h': round(v_[0], 2), 'util_mean': round(float(np.mean(v_[1])), 1), 'jobs': len(v_[1])}
            for k_, v_ in tot.items()}
    (OUT / 'cost.json').write_text(json.dumps({'jobs': out, 'by_step': summ}, indent=1))
    print(json.dumps(summ, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('plan', 'prepass', 'train', 'encode', 'score', 'report', 'locality', 'cost'))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--videos', default=None)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--out-dir', default='')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--arms', default='')
    p.add_argument('--shard-frames', type=int, default=36_000)
    p.add_argument('--force', action='store_true')
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--refs-only', action='store_true', help='score: the reference codes only (scorer check)')
    args = p.parse_args()
    {'plan': cmd_plan, 'prepass': cmd_prepass, 'train': cmd_train, 'encode': cmd_encode, 'score': cmd_score,
     'report': lambda a: report(), 'locality': cmd_locality, 'cost': cmd_cost}[args.step](args)


if __name__ == '__main__':
    main()
