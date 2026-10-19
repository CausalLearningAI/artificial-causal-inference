"""
TEMPORAL CONTEXT for the v2 representation (2026-10-19). Goal: a principled, label-free way to bring temporal context
into the round-1 winner A3 so that e.g. sniffing (sustained head contact) is separated from passing-by contact, tested
on mice v1. Labels are used ONLY to score; every representation / dictionary here is trained label-free.

BASE = round-1 A3 (scripts/eci/nnnt/nnnt.py): DINOv3-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m, cached revision
ea8dc2863c51be0a264bab82070e3e8836b02d51) block 20 at 1024 px (64 x 64 tokens), odour-corner rotation, TIGHT mask (token
kept iff >= 25 % of its 16 x 16 px patch is animal pixels by the lab pixel cue), no change channel, Matryoshka BatchTopK
512 latents (prefixes 128 / 256 / 512), k = 16, 6000 steps x 4096 tokens, lr 5e-4, theta* calibrated on the CALIB videos
(mean L0 = k), frame readout = max over kept tokens. The round-1 roles of the videos are kept (SAE on 'train' videos,
theta* on 'calib', FVE on 'eval').

FRAMES SCORED
    D    the round-2 DIAGNOSIS set: every 10th 5 fps frame of the 144 annotated videos (k = fps1_off + 10 j), 86,400
         frames; its A3 centre tokens are stored (BEE/r2/diag). T0, T1, T2 are scored here.
    S20  the every-20th subset of D (43,200 frames, as round 2's 2048 px escalation): T3 (video model) and T0 on it.
    all  every annotated 5 fps frame (864,000): T0 and T1 only (secondary).
    *_cons  the same frames restricted to the videos of the CONSISTENT annotators SG / SD / MR / CP (nose_tail labels
         are annotator-inconsistent; nnnt_analysis Q1a). Pools are still the cross-fitting unit.

ARMS (all 512 latents, k 16, Matryoshka 128 / 256 / 512, 6000 x 4096, lr 5e-4 = the A3 recipe; seeds 0-5)
    T0     A3 itself: the stored round-1 A3 frame codes (seed s = A3_w512_s<s>|max).
    T1     LEVEL-2 temporal dictionary on the level-1 A3 FRAME codes c_t (512, max readout; level-1 seed = level-2 seed),
           per frame, centred windows on the 5 fps grid (edges replicate / mean over the available frames):
             T1id  x = c_t (no context: the 'second dictionary' control)
             T1m   multi-scale centred means x = [c_t, mean c over |dt| <= 2, 5, 10, 20 frames] (= +-0.4 / 1 / 2 / 4 s;
                   +-0.4 s is the 5 fps grid's nearest to +-0.5 s), 5 x 512 = 2560-d
             T1d   dilated sequence x = c at dt in (-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20), 11 x 512 = 5632-d
           TokenNorm per 512-block (each time block gets average norm sqrt(512)). Train = every non-D frame of the
           'train'-role annotated videos (123), theta* = 'calib' (9), FVE = 'eval' (12). The per-token-location variant
           was NOT run (a fixed token location does not follow a moving mouse, and token codes at neighbouring frames
           would need the T2 re-encode); this is the per-frame-after-max variant the brief allows.
    T2     ONE symmetric short context channel at token level, x = [z_t, (z_{t-k} + z_{t+k}) / 2] at the SAME token
           position (fixed camera), kept iff the CENTRE token is on the tight mask; TokenNorm per 1024-block (as the d2
           change channel). T2a k = 5 (+-1 s), T2b k = 2 (+-0.4 s); T2id x = z_t (same pool, same recipe = the clean
           context control). TRAIN POOL (step 'pool'): 28 random centre frames per video (rng 11; k in [5, n - 6],
           never on the D grid), all 432 videos, = round 1's 10,080 train / 1,008 calib / 1,008 eval frame counts.
    T3     VIDEO FOUNDATION MODEL: V-JEPA 2 ViT-g/16 (facebook/vjepa2-vitg-fpc64-384, revision
           12ca91694b230e0d4b5b0078af6f4ae1d51e933d, the largest public V-JEPA 2 checkpoint, 1B, 40 blocks, 1408-d,
           pretrained at 384 px) at S px (S chosen by the label-free FEASIBILITY rule below), RoPE positions
           INTERPOLATED into the 24 x 24 pretraining grid (h, w scaled by 24 / G, the MABe22 encoder-study winner
           convention), config.crop_size = S. Clip = 16 source frames at stride 8 (3.75 fps, nearest to V-JEPA 2's
           4 fps pretraining), centred: source frames s_t + 8 (j - 7), j = 0..15, clamped to the video; the frame token =
           tubelet 3 = clip positions (6, 7) = source frames (s_t - 8, s_t) (lab convention, scripts/eci/vjepa.py).
           Odour rotation as A3. Mask = the SAME tight rule at its own grid (>= 25 % animal pixels in the 16 x 16 px
           patch of the S px input, pixel cue on the target frame, background bilinearly resized 512 -> S).
             T3g33  block 33 of 40 (5 / 6 depth = DINOv3's 20 / 24) + the model's final LayerNorm (PRIMARY)
             T3g40  block 40 = last_hidden_state (secondary)
           Train pool = the T2 pool frames (same videos / roles). Scored on S20 against T0 on S20.
           FEASIBILITY RULE (label-free, written before any T3 encode): bf16 autocast iff its token cosine vs fp32 on the
           smoke clips has mean >= 0.98 at block 33; S = 1024 (A3's 64 grid) iff the projected T3 encode (pool + S20
           = 55,296 clips) costs <= 16 GPU-h at the smoke throughput, else 768 (48 grid), else 512.

READOUT / METRICS (labels: nose_nose = Y_nn | Y_np, nose_tail = Y_nt; nn_mutual = Y_nn, np_directional = Y_np)
    Frame value = max over kept tokens (T0 / T2 / T3), the level-2 code itself (T1). Round-1 scorer
    (nnnt.score_set): R~ = cross-fitted normalised R-precision (neuron chosen on one half of the 24 pools, scored on
    the other, both directions, mean of 10 random 12 / 12 pool splits), test AUROC of the AUROC-selected neuron.
    primary per (arm, seed) = mean(R~ nose_nose, R~ nose_tail).
    SECONDARY (for the R~-selected neurons, on the held-out half, mean over splits x directions):
      long / short bout AUROC  positives restricted to frames of label bouts >= 5 frames (>= 1 s) / < 5 frames, vs all
                               negatives (bouts = runs of consecutive labelled 5 fps frames, per video)
      generic share   share of the top-N_pos held-out frames that are label-free CONTACT frames with no nn / np / nt
                      label (CONTACT = largest 8-connected tight-mask component >= 1.5 x A1(video), round 3's rule;
                      A1 = median largest component over the video's 4-component frames)
      contact duration AUROC  label-free: frames in contact runs >= 10 frames (>= 2 s, sustained) vs <= 3 frames
                      (<= 0.6 s, brief pass-by)
      kinematics      Spearman rho of the neuron with mask-blob SPEED (size-weighted mean distance of each >= 5-token
                      component's centroid to the nearest component centroid one 5 fps frame later, token units / s)
                      and |ACCELERATION| (|speed_t - speed_{t-1}| x 5); per arm also max |rho| over all 512 neurons
      locality (step 'sheets')  24 held-out frames (<= 1 per video) of the modal-seed nn / nt neuron (seed = highest
                      eval FVE, neuron = chosen on pools half 0 of split rnd0): activation mass on animal pixels, tokens
                      above half max; contact sheets with real per-token heat maps on the rotated frame (T1: the
                      level-2 neuron's centre-frame attribution sum_i max(W_enc[i, j], 0) * scale_i * a_i(token) over
                      the level-1 A3 token codes; labelled as such)
      cost            GPU-h per step (nvidia-smi samples), peak memory
PRE-REGISTERED DECISION RULE (2026-10-19, before any score of this study was computed)
    Per family F in {T1: T1m, T1d; T2: T2a, T2b; T3: T3g33} the candidate = highest mean primary over seeds 0-2.
    KEEP a candidate iff, on 6 vs 6 seeds, its primary gain over T0 (same frames: D for T1 / T2, S20 for T3) is
    >= +0.02 with the Welch 95% CI excluding 0. For T2 also vs T2id (same pool and recipe; the clean context effect)
    and for T1 also vs T1id (same second-dictionary step): both reported, the KEEP needs the T0 test only.
    nose_nose and nose_tail are always reported separately (a gain carried by one label is said so); nose_tail also on
    the consistent annotators. 'Separates sustained from pass-by' is claimed for an arm only if its selected neuron's
    long-bout AUROC rises vs T0 (mean over seeds, paired by seed for T1 / T2) while the generic share does not rise.
    RECOMMENDATION = the kept arm with the highest primary (ties within 0.01 -> cheaper arm); none kept -> stay with A3.

Steps (temporal.sh runs each in Slurm; staging in $LOCAL_DIR; big arrays on BEE)
    plan    (CPU) D / S20 frames, pool frames, annotators, shards -> OUT/plan.json, OUT/pool_frames.parquet
    vjsmoke (GPU) V-JEPA 2-g: RoPE override check, bf16 vs fp32 cosine, throughput at 512 / 768 / 1024 -> OUT/vj_smoke.json
    pool    (GPU, array) T2 pool: centre tokens + context means k = 2, 5 -> BEE/pool/part_XX/
    ctx     (GPU, array over video shards) D-frame context means k = 2, 5 at the stored diag token positions (+ checks
            vs the stored diag mask / tokens), tight masks of EVERY annotated frame -> BEE/ctx/shard_XX/, BEE/masks/
    vj      (GPU, array) V-JEPA tokens (blocks 33, 40) of pool and S20 frames -> BEE/vj/part_XX/
    geom    (CPU) components / contact / run lengths / speed / acceleration of every annotated frame -> BEE/geom.npz
    train   (GPU) every arm x seed -> OUT/sae/<name>/
    codes   (GPU) frame codes of every SAE on D (S20 for T3; all frames for T0 / T1) -> BEE/codes/
    score   (GPU) metrics + rule -> OUT/score.json, OUT/table.md, OUT/decisions.json
    sheets  (GPU) contact sheets + locality -> OUT/sheets/, OUT/locality.json
    cost    (any) GPU-h per step -> OUT/cost.json
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'nnnt'))
import nnnt as N  # noqa: E402  (scorer, plan, SAE helpers; read-only use)

R4, REPO, C = N.R4, N.REPO, N.C
OUT = Path(os.environ.get('TMP_OUT') or REPO / 'results/vision/eci_temporal')
BEE = Path(os.environ.get('TMP_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/temporal')
DIAG = N.BEE / 'r2' / 'diag'
NPZ = N.BEE / 'videos'  # round-1 per-video A3 frame codes
D3_REV = 'ea8dc2863c51be0a264bab82070e3e8836b02d51'
VJ_MODEL, VJ_REV = 'facebook/vjepa2-vitg-fpc64-384', '12ca91694b230e0d4b5b0078af6f4ae1d51e933d'
VJ_CACHE = '/mnt/beegfs/locatgrp/shared/rcadei/hf_cache'
VJ_LAYERS, VJ_NBLK, VJ_G0, VJ_D = (33, 40), 40, 24, 1408
CLIP, TPOS, VSTRIDE = 16, 7, 8
TUB = TPOS // 2
G, S, D = 64, 1024, 1024
K, W, PREFIXES = 16, 512, (128, 256, 512)
SEEDS = (0, 1, 2, 3, 4, 5)
CONS = ('SG', 'SD', 'MR', 'CP')
MS_WIN = (2, 5, 10, 20)
DIL = (-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20)
CTX_K = (2, 5)
POOL_PER_VIDEO = 28
ARMS = {'T0': dict(kind='t0'),
        'T1id': dict(kind='t1', win='id'), 'T1m': dict(kind='t1', win='ms'), 'T1d': dict(kind='t1', win='dil'),
        'T2id': dict(kind='t2', ctx=0), 'T2a': dict(kind='t2', ctx=5), 'T2b': dict(kind='t2', ctx=2),
        'T3g33': dict(kind='t3', layer=33), 'T3g40': dict(kind='t3', layer=40)}
LABELS = N.LABELS
log = N.log


def names(arms=None, seeds=SEEDS):
    return [(f'{a}_s{s}', a, s) for a in (arms or ARMS) for s in seeds]


def local_dir():
    return R4.local_dir()


def load_plan():
    return json.loads((OUT / 'plan.json').read_text())


# ---------------------------------------------------------------------------------------------- plan
def cmd_plan(args):
    P = N.load_plan()
    vids = P['videos']
    vi_of = {v['obs']: i for i, v in enumerate(vids)}
    F = pd.read_parquet(DIAG / 'frames.parquet')
    F['video'] = F.obs.map(vi_of).astype(int)
    off = F.video.map(lambda i: vids[i]['fps1_off'])
    assert ((F.k - off) % 10 == 0).all()
    F['sub20'] = ((F.k - off) % 20 == 0)
    assert (np.diff(F.video.values) >= 0).all() and F.groupby('video').k.apply(lambda x: (np.diff(x) > 0).all()).all()
    a = pd.read_csv(REPO / C['ann'], usecols=['observation_id', 'W_annotator'])
    ann = a.groupby('observation_id').W_annotator.first().fillna('?')
    annv = [i for i, v in enumerate(vids) if v['annotated']]
    assert sorted(F.video.unique()) == annv
    # T2 / T3 pool: 28 centre frames per video (all 432), k in [5, n - 6], never on the D grid
    g = np.random.default_rng(11)
    pool = []
    for vi, v in enumerate(vids):
        ks = np.arange(5, v['n'] - 5)
        ks = ks[(ks - v['fps1_off']) % 10 != 0]
        for k in np.sort(g.choice(ks, POOL_PER_VIDEO, replace=False)):
            pool.append({'video': vi, 'obs': v['obs'], 'k': int(k), 'role': v['role'], 'annotated': v['annotated']})
    Pool = pd.DataFrame(pool)
    # ctx shards: consecutive annotated videos, ~shard_frames each
    shards, cur, nf = [], [], 0
    for i in annv:
        if cur and nf + vids[i]['n'] > args.shard_frames:
            shards.append(cur)
            cur, nf = [], 0
        cur.append(i)
        nf += vids[i]['n']
    shards.append(cur)
    # vj parts: pool frames then S20 frames, ~vj_part clips each, whole videos per part
    items = [('pool', int(r.video), int(r.k)) for r in Pool.itertuples()] + \
        [('s20', int(r.video), int(r.k)) for r in F[F.sub20].itertuples()]
    vparts, cur = [], []
    for it in items:
        if len(cur) >= args.vj_part and (it[0], it[1]) != (cur[-1][0], cur[-1][1]):
            vparts.append(cur)
            cur = []
        cur.append(it)
    vparts.append(cur)
    pparts = np.array_split(np.arange(len(Pool)), args.pool_parts)
    out = {'n_D': len(F), 'n_S20': int(F.sub20.sum()), 'n_pool': len(Pool),
           'pool_roles': Pool.role.value_counts().to_dict(),
           'annotator': {vids[i]['obs']: ann[vids[i]['obs']] for i in annv},
           'cons_videos': int(sum(ann[vids[i]['obs']] in CONS for i in annv)),
           'ctx_shards': shards, 'ctx_shard_frames': [int(sum(vids[i]['n'] for i in s)) for s in shards],
           'pool_parts': [[int(p[0]), int(p[-1]) + 1] for p in pparts],
           'vj_parts': [[list(x) for x in p] for p in vparts]}
    OUT.mkdir(parents=True, exist_ok=True)
    Pool.to_parquet(OUT / 'pool_frames.parquet')
    (OUT / 'plan.json').write_text(json.dumps(out))
    log(f'plan: D {len(F):,} S20 {int(F.sub20.sum()):,} pool {len(Pool):,} {out["pool_roles"]}; cons videos '
        f'{out["cons_videos"]}; {len(shards)} ctx shards; {len(pparts)} pool parts; {len(vparts)} vj parts '
        f'({[len(p) for p in vparts]})')


# ---------------------------------------------------------------------------------------------- DINOv3 pipe
class D3:
    """nnnt.Pipe without the DINOv2 / JPG loose mask: rotated frame -> DINOv3-L block-20 tokens and the tight pixel cue
    (identical operations to nnnt.Pipe.run)."""

    def __init__(self, dev, ranges, encode=True):
        import torch
        from transformers import AutoModel
        from src.eci.foreground import RULES, FgBackgrounds
        self.torch, self.dev = torch, dev
        if encode:
            m3 = AutoModel.from_pretrained(R4.D3L, attn_implementation='sdpa').to(dev).eval()
            assert len(m3.layer) == 24
            m3.layer = torch.nn.ModuleList(list(m3.layer)[:R4.LAYER])
            self.m3 = m3.requires_grad_(False)
        self.mean = torch.tensor(R4.MEAN, device=dev)[None, :, None, None]
        self.std = torch.tensor(R4.STD, device=dev)[None, :, None, None]
        self.rule = RULES['fg448']
        self.bg = FgBackgrounds(REPO / 'dataset/mice/v1/eci/fg448al/background', ranges, self.rule, dev, align='odor')
        self._pbg = {}
        self.gpu_s = 0.0
        self.n_enc = 0

    def pix_bg(self, obs, side):
        key = (obs, side)
        if key not in self._pbg:
            if len(self._pbg) > 8:
                self._pbg.clear()
            pb = self.bg.get(obs)['pix_bg'].float()
            self._pbg[key] = self.torch.nn.functional.interpolate(pb[None, None], size=(side, side), mode='bilinear',
                                                                  align_corners=False)[0, 0]
        return self._pbg[key]

    def rot(self, fr, rot):
        """(B, s, s, 3) uint8 cuda -> (B, 3, s, s) rotated contiguous."""
        x = fr.permute(0, 3, 1, 2)
        if rot:
            x = self.torch.rot90(x, rot, dims=(2, 3))
        return x.contiguous()

    def dark(self, x, obs, cell=16):
        """x (B, 3, s, s) uint8 rotated -> (B, (s / cell)^2) animal-pixel fraction per cell."""
        F = self.torch.nn.functional
        g = R4.Pipe.grey(x).float()
        pb = self.pix_bg(obs, x.shape[-1])
        dk = (g < self.rule['dark_abs']) & ((pb[None] - g) > self.rule['dark_rel'])
        return F.avg_pool2d(dk.float()[:, None], cell).flatten(1)

    def tokens(self, x):
        torch = self.torch
        t0 = time.time()
        with torch.no_grad():
            xin = (x.float() / 255 - self.mean) / self.std
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h3 = self.m3(pixel_values=xin).last_hidden_state
            z = h3[:, R4.N_PRE:].float()
            assert z.shape[1] == G * G and torch.isfinite(z).all(), z.shape
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        self.n_enc += len(x)
        return z.half()


def dec_job(v, k0, n, step=C['step'], side=S, threads=2):
    return dict(path=v['path'], s0=v['offset'] + C['step'] * k0, n=n, step=step, S=side, hflip=v['hflip'],
                fps=C['fps'], threads=threads, jpgs=None)


def ncpu():
    return int(os.environ.get('SLURM_CPUS_PER_TASK', 16))


# ---------------------------------------------------------------------------------------------- pool (T2)
def cmd_pool(args):
    """T2 pool part: per centre frame k decode 5 fps frames k-5..k+5, encode k-5, k-2, k, k+2, k+5; store the centre's
    tight-mask tokens and the k = 2 / 5 symmetric context means at the same positions."""
    import torch
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    P, PL = N.load_plan(), load_plan()
    Pool = pd.read_parquet(OUT / 'pool_frames.parquet')
    a, b = PL['pool_parts'][args.shard]
    Q = Pool.iloc[a:b].reset_index(drop=True)
    if args.limit:
        Q = Q.iloc[:args.limit]
    _, Rr = R4.obs_ranges_of(REPO / C['ann'])
    pipe = D3(dev, Rr)
    loc = local_dir() / 'pool'
    loc.mkdir(parents=True, exist_ok=True)
    fo = {n_: open(loc / f'{n_}.f16', 'wb') for n_ in ('tok', 'ctx2', 'ctx5')}
    meta = {'tok_frame': [], 'tok_pos': [], 'tok_fdark': []}
    jobs = [((i, r.video, r.k), dec_job(P['videos'][r.video], r.k - 5, 11)) for i, r in enumerate(Q.itertuples())]
    pf = R4.Prefetch(jobs, max(4, ncpu() - 2))
    sel = [0, 3, 5, 7, 10]  # k-5, k-2, k, k+2, k+5
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    buf = []

    def flush():
        fr = torch.from_numpy(np.concatenate([b_[1][sel] for b_ in buf])).to(dev)
        rots = [P['videos'][key[1]]['rot'] for key, _ in buf]
        xs = torch.cat([pipe.rot(fr[5 * i_:5 * i_ + 5], rots[i_]) for i_ in range(len(buf))])
        z = pipe.tokens(xs).view(len(buf), 5, G * G, D)
        for i_, (key, _) in enumerate(buf):
            obs = P['videos'][key[1]]['obs']
            fd = pipe.dark(xs[5 * i_ + 2:5 * i_ + 3], obs)[0]
            pos = torch.nonzero(fd >= N.THR_TIGHT).flatten()
            zz = z[i_]
            fo['tok'].write(zz[2, pos].cpu().numpy().tobytes())
            fo['ctx2'].write(((zz[1, pos].float() + zz[3, pos].float()) / 2).half().cpu().numpy().tobytes())
            fo['ctx5'].write(((zz[0, pos].float() + zz[4, pos].float()) / 2).half().cpu().numpy().tobytes())
            meta['tok_frame'].append(np.full(len(pos), key[0], np.int32))
            meta['tok_pos'].append(pos.cpu().numpy().astype(np.int16))
            meta['tok_fdark'].append(fd[pos].cpu().numpy().astype(np.float16))
        buf.clear()

    for j, (key, (fr, _)) in enumerate(pf):
        buf.append((key, fr))
        if len(buf) == 3:
            flush()
        if j % 500 == 0:
            el = time.time() - t0
            log(f'pool part {args.shard}: {j}/{len(jobs)} frames, {el:.0f}s ({pipe.n_enc / max(el, 1):.1f} encodes/s, '
                f'GPU busy {pipe.gpu_s:.0f}s, decode thread-s {pf.dec_s:.0f})')
    if buf:
        flush()
    for f in fo.values():
        f.close()
    for k_, x in meta.items():
        np.save(loc / f'{k_}.npy', np.concatenate(x))
    Q.to_parquet(loc / 'frames.parquet')
    el = time.time() - t0
    info = {'part': args.shard, 'n_frames': len(Q), 'n_tokens': int(sum(len(x) for x in meta['tok_frame'])),
            'wall_s': el, 'encodes': pipe.n_enc, 'encodes_per_s_wall': pipe.n_enc / el, 'gpu_busy_s': pipe.gpu_s,
            'decode_thread_s': pf.dec_s, 'peak_mem_gb': torch.cuda.max_memory_allocated() / 2**30,
            'gpu': torch.cuda.get_device_name(0)}
    log(json.dumps(info))
    dst = (OUT / 'selftest' if args.limit else BEE / 'pool') / f'part_{args.shard:02d}'
    R4.atomic_copy(loc, dst)
    (OUT / 'logs' / 'json').mkdir(parents=True, exist_ok=True)
    (OUT / 'logs' / 'json' / f'pool_{args.shard:02d}{"_selftest" if args.limit else ""}.json').write_text(
        json.dumps(info, indent=1))


# ---------------------------------------------------------------------------------------------- ctx (T2 on D) + masks
def diag_index():
    F = pd.read_parquet(DIAG / 'frames.parquet')
    tf = np.load(DIAG / 'tok_frame.npy')
    st = np.r_[0, np.cumsum(np.bincount(tf, minlength=len(F)))]
    return F, tf, st


def cmd_ctx(args):
    import torch
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    P, PL = N.load_plan(), load_plan()
    vids = PL['ctx_shards'][args.shard]
    if args.videos:
        vids = [int(x) for x in args.videos.split(',')]
    F, tf, st = diag_index()
    pos_all = np.load(DIAG / 'tok_pos.npy')
    dtok = np.memmap(DIAG / 'tok.f16', np.float16, 'r', shape=(len(tf), D))
    vi_of = {v['obs']: i for i, v in enumerate(P['videos'])}
    F['video'] = F.obs.map(vi_of)
    fsel = np.flatnonzero(F.video.isin(vids).values)
    assert (np.diff(fsel) == 1).all()
    ta, tb = int(st[fsel[0]]), int(st[fsel[-1] + 1])
    _, Rr = R4.obs_ranges_of(REPO / C['ann'])
    pipe = D3(dev, Rr)
    loc = local_dir() / 'ctx'
    (loc / 'masks').mkdir(parents=True, exist_ok=True)
    fo = {n_: open(loc / f'{n_}.f16', 'wb') for n_ in ('ctx2', 'ctx5')}
    chk = {'mask_exact': [], 'mask_iou': [], 'tok_max_abs': [], 'tok_mean_abs': []}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    n_written = 0
    for vi in vids:
        v = P['videos'][vi]
        n = v['n'] if not args.max_frames else min(v['n'], args.max_frames)
        fr_ids = fsel[F.video.values[fsel] == vi]
        Dk = F.k.values[fr_ids]
        if args.max_frames:  # selftest: only D frames whose +-5 neighbours are decoded
            ok = Dk + 5 < n
            fr_ids, Dk = fr_ids[ok], Dk[ok]
        need = {}
        for t in Dk:
            for o in (-5, -2, 2, 5):
                need.setdefault(int(np.clip(t + o, 0, v['n'] - 1)), True)
        check_t = set(int(t) for t in Dk[::20])
        jobs = [((vi, k0), dec_job(v, k0, min(300, n - k0), threads=4)) for k0 in range(0, n, 300)]
        pf = R4.Prefetch(jobs, max(4, ncpu() // 4))
        tight = np.zeros((n, G * G), bool)
        zbuf, zc = {}, {}
        di = 0  # next D frame to emit
        tv = time.time()
        for (_, k0), (fr, _) in pf:
            for b in range(0, len(fr), 16):
                kk = k0 + b + np.arange(len(fr[b:b + 16]))
                x = pipe.rot(torch.from_numpy(fr[b:b + 16]).to(dev), v['rot'])
                fd = pipe.dark(x, v['obs'])
                tight[kk] = (fd >= N.THR_TIGHT).cpu().numpy()
                want = [i for i, k in enumerate(kk) if int(k) in need or int(k) in check_t]
                if want:
                    z = pipe.tokens(x[want])
                    for j, i in enumerate(want):
                        zbuf[int(kk[i])] = z[j]
            last = int(k0 + len(fr) - 1)
            while di < len(Dk) and min(int(Dk[di]) + 5, v['n'] - 1) <= last:
                t = int(Dk[di])
                fid = fr_ids[di]
                p = pos_all[st[fid]:st[fid + 1]].astype(np.int64)
                pt = torch.from_numpy(p).to(dev)
                for kc in CTX_K:
                    zm = zbuf[int(np.clip(t - kc, 0, v['n'] - 1))][pt].float()
                    zp = zbuf[int(np.clip(t + kc, 0, v['n'] - 1))][pt].float()
                    fo[f'ctx{kc}'].write(((zm + zp) / 2).half().cpu().numpy().tobytes())
                n_written += len(p)
                mine = np.flatnonzero(tight[t])
                chk['mask_exact'].append(float(np.array_equal(mine, np.sort(p))))
                chk['mask_iou'].append(len(np.intersect1d(mine, p)) / max(1, len(np.union1d(mine, p))))
                if t in check_t:
                    zt = zbuf[t][pt].float().cpu().numpy()
                    ref = np.asarray(dtok[st[fid]:st[fid + 1]]).astype(np.float32)
                    dd = np.abs(zt - ref)
                    chk['tok_max_abs'].append(float(dd.max()))
                    chk['tok_mean_abs'].append(float(dd.mean()))
                di += 1
                for kd in [k_ for k_ in zbuf if k_ < t - 5]:
                    del zbuf[kd]
        assert di == len(Dk), (v['obs'], di, len(Dk))
        np.savez(loc / 'masks' / f'{v["obs"]}.npz', bits=np.packbits(tight, axis=1), n=n)
        dt = time.time() - tv
        log(f'{v["obs"]}: {n} frames in {dt:.0f}s ({n / dt:.1f} fps decoded), D frames {len(Dk)}, encodes so far '
            f'{pipe.n_enc}, GPU busy {pipe.gpu_s:.0f}s, decode thread-s {pf.dec_s:.0f}; mask exact '
            f'{np.mean(chk["mask_exact"][-len(Dk):]):.4f}')
    for f in fo.values():
        f.close()
    el = time.time() - t0
    info = {'shard': args.shard, 'videos': vids, 'token_range': [ta, tb], 'n_written': n_written,
            'wall_s': el, 'encodes': pipe.n_enc, 'gpu_busy_s': pipe.gpu_s,
            'peak_mem_gb': torch.cuda.max_memory_allocated() / 2**30, 'gpu': torch.cuda.get_device_name(0),
            'check': {k_: {'mean': float(np.mean(x)), 'min': float(np.min(x)), 'max': float(np.max(x)), 'n': len(x)}
                      for k_, x in chk.items() if x}}
    log(json.dumps(info))
    if not args.max_frames:
        assert n_written == tb - ta, (n_written, ta, tb)
    (loc / 'info.json').write_text(json.dumps(info, indent=1))
    dst = (OUT / 'selftest' if args.max_frames else BEE / 'ctx') / f'shard_{args.shard:02d}'
    R4.atomic_copy(loc, dst)
    (OUT / 'logs' / 'json').mkdir(parents=True, exist_ok=True)
    (OUT / 'logs' / 'json' / f'ctx_{args.shard:02d}{"_selftest" if args.max_frames else ""}.json').write_text(
        json.dumps(info, indent=1))


# ---------------------------------------------------------------------------------------------- V-JEPA 2
class VJ:
    def __init__(self, dev, side, bf16=True, layers=VJ_LAYERS):
        import torch
        from transformers import AutoConfig, AutoModel
        self.torch, self.dev, self.S, self.bf16 = torch, dev, side, bf16
        self.G = side // 16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        cfg = AutoConfig.from_pretrained(VJ_MODEL, revision=VJ_REV, cache_dir=VJ_CACHE)
        assert cfg.crop_size // cfg.patch_size == VJ_G0 and cfg.num_hidden_layers == VJ_NBLK
        cfg.crop_size = side
        self.model = AutoModel.from_pretrained(VJ_MODEL, revision=VJ_REV, cache_dir=VJ_CACHE, config=cfg,
                                               attn_implementation='sdpa').to(dev).eval().requires_grad_(False)
        blocks, self.norm = self.model.encoder.layer, self.model.encoder.layernorm
        assert len(blocks) == VJ_NBLK
        self.pos_check = None
        if self.G != VJ_G0:
            sc = VJ_G0 / self.G
            for bi, blk in enumerate(blocks):
                att = blk.attention
                orig = att.get_position_ids

                def gp(x, masks=None, _o=orig, _s=sc, _b=bi):
                    f, h, w = _o(x, masks)
                    if _b == 0 and self.pos_check is None:
                        self.pos_check = {'h_max_raw': float(h.max()), 'w_max_raw': float(w.max()),
                                          'f_max': float(f.max()), 'h_max_scaled': float(h.max() * _s)}
                    return f, h.float() * _s, w.float() * _s
                att.get_position_ids = gp
        self.layers = list(layers)
        self.cap = {}
        for l_ in self.layers:
            blocks[l_ - 1].register_forward_hook(
                lambda m, i, o, _l=l_: self.cap.__setitem__(_l, o[0] if isinstance(o, tuple) else o))
        self.mean = torch.tensor(R4.MEAN, device=dev)[None, None, :, None, None]
        self.std = torch.tensor(R4.STD, device=dev)[None, None, :, None, None]
        self.gpu_s, self.n_clips = 0.0, 0

    def forward(self, clips):
        """clips (B, 16, 3, S, S) uint8 cuda (rotated) -> {layer: (B, G*G, 1408) fp32} (tubelet TUB, final LayerNorm)."""
        torch = self.torch
        t0 = time.time()
        self.cap.clear()
        x = (clips.float() / 255 - self.mean) / self.std
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
            out = self.model(pixel_values_videos=x, skip_predictor=True).last_hidden_state
            hs = {l_: self.norm(self.cap[l_]).float() for l_ in self.layers}
        GG = self.G * self.G
        res = {}
        for l_ in self.layers:
            h = hs[l_]
            if l_ == VJ_NBLK:
                assert torch.allclose(h, out.float(), atol=2e-2 if self.bf16 else 1e-3), 'last block != output'
            assert h.shape[1] == (CLIP // 2) * GG, h.shape
            h = h[:, TUB * GG:(TUB + 1) * GG]
            assert torch.isfinite(h).all()
            res[l_] = h
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        self.n_clips += len(clips)
        return res


def vj_clip_job(v, k, side):
    """Decode the 16-frame clip of 5 fps frame k (source frames s_t + 8 (j - 7)), clamped to the video's source range;
    -> (kwargs for R4.decode, index map j -> decoded frame)."""
    s_t = v['offset'] + C['step'] * k
    lo_s, hi_s = v['offset'], v['offset'] + C['step'] * (v['n'] - 1)
    src = s_t + VSTRIDE * (np.arange(CLIP) - TPOS)
    ok = (src >= lo_s) & (src <= hi_s)
    j0, j1 = int(np.flatnonzero(ok)[0]), int(np.flatnonzero(ok)[-1])
    idx = np.clip(np.arange(CLIP), j0, j1) - j0
    return dict(path=v['path'], s0=int(src[j0]), n=j1 - j0 + 1, step=VSTRIDE, S=side, hflip=v['hflip'], fps=C['fps'],
                threads=2, jpgs=None), idx


def vj_side():
    f = OUT / 'vj_smoke.json'
    if not f.exists():
        raise SystemExit('run vjsmoke first (feasibility rule)')
    j = json.loads(f.read_text())
    return int(j['decision']['S']), bool(j['decision']['bf16'])


def cmd_vjsmoke(args):
    import torch
    dev = torch.device('cuda')
    P = N.load_plan()
    _, Rr = R4.obs_ranges_of(REPO / C['ann'])
    pipe = D3(dev, Rr, encode=False)
    F = pd.read_parquet(DIAG / 'frames.parquet')
    sub = F.iloc[[0, 5000, 40000, 80000]]
    vi_of = {v['obs']: i for i, v in enumerate(P['videos'])}
    res = {'model': VJ_MODEL, 'revision': VJ_REV, 'gpu': torch.cuda.get_device_name(0), 'prec': {}, 'speed': {}}
    clips = {}
    for side in (512, 768, 1024):
        cl = []
        for r in sub.itertuples():
            v = P['videos'][vi_of[r.obs]]
            kw, idx = vj_clip_job(v, int(r.k), side)
            fr = R4.decode(**{k_: x for k_, x in kw.items() if k_ != 'jpgs'})[idx]
            cl.append((pipe.rot(torch.from_numpy(fr).to(dev), v['rot']), v['obs']))
        clips[side] = cl
    for side in (512, 768, 1024):
        for bf in (True, False):
            if not bf and side == 1024:
                continue
            enc = VJ(dev, side, bf16=bf)
            torch.cuda.reset_peak_memory_stats()
            outs = []
            for x, obs in clips[side]:
                o = enc.forward(x[None])
                m = pipe.dark(x[TPOS:TPOS + 1], obs) >= N.THR_TIGHT
                outs.append({l_: o[l_][0][m[0]] for l_ in VJ_LAYERS})
            # timing: 3 repeats on clip 0 (after warm-up), batch 1 and 2
            sp = {}
            for bs in (1, 2):
                xb = torch.stack([clips[side][0][0]] * bs)
                enc.forward(xb)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(3):
                    enc.forward(xb)
                torch.cuda.synchronize()
                sp[f'bs{bs}_clips_per_s'] = 3 * bs / (time.time() - t0)
            sp['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 2**30
            sp['pos_check'] = enc.pos_check
            res['speed'][f'{side}_{"bf16" if bf else "fp32"}'] = sp
            log(f'V-JEPA S {side} {"bf16" if bf else "fp32"}: {json.dumps(sp)}')
            if bf:
                res.setdefault('_bf', {})[side] = outs
            else:
                cos = {}
                for l_ in VJ_LAYERS:
                    a = torch.cat([o[l_] for o in res['_bf'][side]])
                    b = torch.cat([o[l_] for o in outs])
                    c = torch.nn.functional.cosine_similarity(a, b, dim=1)
                    cos[l_] = {'n_tokens': len(c), 'cos_mean': float(c.mean()), 'cos_p1': float(c.quantile(0.01)),
                               'cos_min': float(c.min())}
                res['prec'][side] = cos
                log(f'bf16 vs fp32 at S {side}: {json.dumps(cos)}')
            del enc
            torch.cuda.empty_cache()
    res.pop('_bf', None)
    n_clips = 55_296
    bf_ok = all(res['prec'][s][33]['cos_mean'] >= 0.98 for s in res['prec'])
    tag = 'bf16' if bf_ok else 'fp32'
    proj = {s: n_clips / res['speed'][f'{s}_{tag}']['bs2_clips_per_s'] / 3600 for s in (512, 768, 1024)
            if f'{s}_{tag}' in res['speed']}
    Sx = 1024 if proj.get(1024, 1e9) <= 16 else (768 if proj.get(768, 1e9) <= 16 else 512)
    res['decision'] = {'bf16': bf_ok, 'projected_gpu_h': proj, 'S': Sx,
                       'rule': 'bf16 iff cos mean >= 0.98 at block 33; S = 1024 iff projected <= 16 GPU-h else 768 '
                               'else 512 (GPU-only time at bs 2, smoke GPU)'}
    log(json.dumps(res['decision']))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'vj_smoke.json').write_text(json.dumps(res, indent=1, default=str))


def cmd_vj(args):
    import torch
    dev = torch.device('cuda')
    P, PL = N.load_plan(), load_plan()
    side, bf = vj_side()
    items = PL['vj_parts'][args.shard]
    if args.limit:
        items = items[:args.limit]
    _, Rr = R4.obs_ranges_of(REPO / C['ann'])
    pipe = D3(dev, Rr, encode=False)
    enc = VJ(dev, side, bf16=bf)
    loc = local_dir() / 'vj'
    loc.mkdir(parents=True, exist_ok=True)
    fo = {l_: open(loc / f'tok{l_}.f16', 'wb') for l_ in VJ_LAYERS}
    meta = {'tok_frame': [], 'tok_pos': [], 'tok_fdark': []}
    jobs, idxs = [], []
    for i, (st_, vi, k) in enumerate(items):
        kw, idx = vj_clip_job(P['videos'][vi], k, side)
        jobs.append(((i, st_, vi, k), kw))
        idxs.append(idx)
    pf = R4.Prefetch(jobs, max(4, ncpu() - 2))
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    buf = []

    def flush():
        x = torch.stack([b_[1] for b_ in buf])
        o = enc.forward(x)
        for j, (key, xc, obs) in enumerate(buf):
            fd = pipe.dark(xc[TPOS:TPOS + 1], obs)[0]
            pos = torch.nonzero(fd >= N.THR_TIGHT).flatten()
            for l_ in VJ_LAYERS:
                fo[l_].write(o[l_][j, pos].half().cpu().numpy().tobytes())
            meta['tok_frame'].append(np.full(len(pos), key[0], np.int32))
            meta['tok_pos'].append(pos.cpu().numpy().astype(np.int16))
            meta['tok_fdark'].append(fd[pos].cpu().numpy().astype(np.float16))
        buf.clear()

    for j, (key, (fr, _)) in enumerate(pf):
        v = P['videos'][key[2]]
        xc = pipe.rot(torch.from_numpy(np.ascontiguousarray(fr[idxs[key[0]]])).to(dev), v['rot'])
        buf.append((key, xc, v['obs']))
        if len(buf) == 2:
            flush()
        if j % 200 == 0:
            el = time.time() - t0
            log(f'vj part {args.shard}: {j}/{len(jobs)} clips, {el:.0f}s ({enc.n_clips / max(el, 1):.2f} clips/s wall, '
                f'GPU busy {enc.gpu_s:.0f}s, decode thread-s {pf.dec_s:.0f})')
    if buf:
        flush()
    for f in fo.values():
        f.close()
    for k_, x in meta.items():
        np.save(loc / f'{k_}.npy', np.concatenate(x))
    pd.DataFrame([{'set': s_, 'video': vi, 'k': k} for s_, vi, k in items]).to_parquet(loc / 'frames.parquet')
    el = time.time() - t0
    info = {'part': args.shard, 'S': side, 'bf16': bf, 'n_clips': len(items),
            'n_tokens': int(sum(len(x) for x in meta['tok_frame'])), 'wall_s': el,
            'clips_per_s_wall': len(items) / el, 'gpu_busy_s': enc.gpu_s, 'decode_thread_s': pf.dec_s,
            'peak_mem_gb': torch.cuda.max_memory_allocated() / 2**30, 'gpu': torch.cuda.get_device_name(0),
            'pos_check': enc.pos_check}
    log(json.dumps(info))
    dst = (OUT / 'selftest' if args.limit else BEE / 'vj') / f'part_{args.shard:02d}'
    R4.atomic_copy(loc, dst)
    (OUT / 'logs' / 'json').mkdir(parents=True, exist_ok=True)
    (OUT / 'logs' / 'json' / f'vj_{args.shard:02d}{"_selftest" if args.limit else ""}.json').write_text(
        json.dumps(info, indent=1))


# ---------------------------------------------------------------------------------------------- geometry
def geom_video(obs_n):
    """One video's tight masks -> per-frame ncomp, maxc, speed, flux, area."""
    from scipy import ndimage
    obs, nv = obs_n
    z = np.load(BEE / 'masks' / f'{obs}.npz')
    m = np.unpackbits(z['bits'], axis=1)[:, :G * G].astype(bool).reshape(-1, G, G)
    n = len(m)
    assert n == nv, (obs, n, nv)
    st = np.ones((3, 3), bool)
    yy, xx = np.mgrid[0:G, 0:G]
    cents = []
    nc, mc, area = np.zeros(n, np.int16), np.zeros(n, np.int16), m.reshape(n, -1).sum(1)
    for t in range(n):
        lab, nl = ndimage.label(m[t], structure=st)
        if nl == 0:
            cents.append(np.zeros((0, 3)))
            continue
        sz = np.bincount(lab.ravel())[1:]
        cy = ndimage.sum(yy, lab, np.arange(1, nl + 1)) / sz
        cx = ndimage.sum(xx, lab, np.arange(1, nl + 1)) / sz
        big = sz >= 5
        cents.append(np.stack([cy[big], cx[big], sz[big]], 1))
        nc[t], mc[t] = int(big.sum()), int(sz[big].max()) if big.any() else 0
    sp = np.full(n, np.nan)
    for t in range(n - 1):
        a, b = cents[t], cents[t + 1]
        if len(a) and len(b):
            d = np.sqrt(((a[:, None, :2] - b[None, :, :2]) ** 2).sum(-1)).min(1)
            sp[t] = float((d * a[:, 2]).sum() / a[:, 2].sum()) * 5.0
    sp[n - 1] = sp[n - 2]
    flux = np.r_[(m[1:] ^ m[:-1]).reshape(n - 1, -1).sum(1) / np.maximum(1, area[:-1]), np.nan]
    flux[-1] = flux[-2]
    return {'ncomp': nc, 'maxc': mc, 'speed': sp, 'flux': flux, 'area': area.astype(np.int16)}


def cmd_geom(args):
    """Every annotated frame (plan order of the 144 videos): >= 5-token 8-connected components of the tight mask ->
    ncomp, maxc, CONTACT (maxc >= 1.5 A1(video)), contact run length, blob SPEED (tokens / s) and |ACCEL|."""
    from multiprocessing import Pool
    P = N.load_plan()
    vids = [v for v in P['videos'] if v['annotated']]
    t0 = time.time()
    with Pool(max(1, ncpu() - 1)) as pool_:
        R_ = pool_.map(geom_video, [(v['obs'], v['n']) for v in vids], chunksize=1)
    log(f'geom: {len(R_)} videos in {time.time() - t0:.0f}s')
    g = {k_: np.concatenate([r[k_] for r in R_]) for k_ in R_[0]}
    g['vid'] = np.concatenate([np.full(v['n'], i, np.int16) for i, v in enumerate(vids)])
    g['k'] = np.concatenate([np.arange(v['n'], dtype=np.int32) for v in vids])
    four = g['ncomp'] == 4
    glob = float(np.median(g['maxc'][four]))
    A1 = np.full(len(vids), glob)
    for vi in range(len(vids)):
        s_ = four & (g['vid'] == vi)
        if s_.sum() >= 20:
            A1[vi] = float(np.median(g['maxc'][s_]))
    g['contact'] = g['maxc'] >= 1.5 * A1[g['vid']]
    run = np.zeros(len(g['k']), np.int32)
    for vi in range(len(vids)):
        ii = np.flatnonzero(g['vid'] == vi)
        c = g['contact'][ii].astype(np.int8)
        ch = np.flatnonzero(np.diff(np.r_[-1, c, -1]) != 0)
        for a, b in zip(ch[:-1], ch[1:]):
            run[ii[a:b]] = (b - a) if c[a] else 0
    g['run'] = run
    acc = np.abs(np.diff(np.r_[g['speed'][0], g['speed']])) * 5.0
    acc[np.r_[True, g['vid'][1:] != g['vid'][:-1]]] = np.nan
    g['accel'] = acc
    g['A1'] = A1
    np.savez(BEE / 'geom.npz', **g)
    info = {'n_frames': len(g['k']), 'wall_s': time.time() - t0, 'A1_global': glob,
            'contact_share': float(g['contact'].mean()),
            'run_frames_quantiles_of_contact_frames': np.quantile(run[g['contact']], [.1, .25, .5, .75, .9]).tolist(),
            'contact_frames_in_runs_le3': float((g['contact'] & (run <= 3)).sum() / g['contact'].sum()),
            'contact_frames_in_runs_ge10': float((g['contact'] & (run >= 10)).sum() / g['contact'].sum()),
            'speed_quantiles': np.nanquantile(g['speed'], [.1, .5, .9]).tolist(),
            'accel_quantiles': np.nanquantile(g['accel'], [.1, .5, .9]).tolist()}
    log(json.dumps(info))
    (OUT / 'geom.json').write_text(json.dumps(info, indent=1))


# ---------------------------------------------------------------------------------------------- data for SAEs
def t1_feats(torch, c, win):
    """c (n, 512) float32 cuda (one video's level-1 frame codes) -> (n, d) window features."""
    n = c.shape[0]
    t = torch.arange(n, device=c.device)
    if win == 'id':
        return c
    if win == 'ms':
        Sm = torch.cat([torch.zeros(1, c.shape[1], device=c.device, dtype=torch.float64), c.double().cumsum(0)])
        outs = [c]
        for w in MS_WIN:
            lo, hi = (t - w).clamp(min=0), (t + w).clamp(max=n - 1) + 1
            outs.append(((Sm[hi] - Sm[lo]) / (hi - lo)[:, None].double()).float())
        return torch.cat(outs, 1)
    if win == 'dil':
        return torch.cat([c[(t + o).clamp(0, n - 1)] for o in DIL], 1)
    raise ValueError(win)


def t1_dim(win):
    return {'id': 1, 'ms': 1 + len(MS_WIN), 'dil': len(DIL)}[win] * W


def level1(obs, seed):
    with np.load(NPZ / f'{obs}.npz') as z:
        return z[f'A3_w512_s{seed}|max']


def t1_data(torch, dev, win, seed, roles=('train', 'calib', 'eval')):
    """{role: (N, d) fp16 cuda} of non-D frames of the annotated videos by round-1 role."""
    P = N.load_plan()
    out = {r: [] for r in roles}
    for v in P['videos']:
        if not v['annotated'] or v['role'] not in roles:
            continue
        c = torch.from_numpy(level1(v['obs'], seed).astype(np.float32)).to(dev)
        x = t1_feats(torch, c, win)
        keep = torch.from_numpy((np.arange(v['n']) - v['fps1_off']) % 10 != 0).to(dev)
        out[v['role']].append(x[keep].half())
    return {r: torch.cat(x) for r, x in out.items()}


def pool_store(name='pool'):
    parts = sorted((BEE / name).glob('part_*'))
    return parts


def t2_data(torch, dev, ctx):
    """T2 pool tokens by role: x = z_t (ctx 0) or [z_t, ctx_k]."""
    out = {}
    for part in pool_store():
        Fp = pd.read_parquet(part / 'frames.parquet')
        tf = np.load(part / 'tok_frame.npy')
        n = len(tf)
        tok = np.fromfile(part / 'tok.f16', np.float16).reshape(n, D)
        cx = np.fromfile(part / f'ctx{ctx}.f16', np.float16).reshape(n, D) if ctx else None
        role = Fp.role.values[tf]
        for r in ('train', 'calib', 'eval'):
            ii = np.flatnonzero(role == r)
            x = torch.from_numpy(tok[ii]).to(dev)
            if ctx:
                x = torch.cat([x, torch.from_numpy(cx[ii]).to(dev)], 1)
            out.setdefault(r, []).append(x)
    return {r: torch.cat(x) for r, x in out.items()}


def vj_parts():
    return sorted((BEE / 'vj').glob('part_*'))


def t3_data(torch, dev, layer):
    out = {}
    P = N.load_plan()
    for part in vj_parts():
        Fp = pd.read_parquet(part / 'frames.parquet')
        if not (Fp.set == 'pool').any():
            continue
        tf = np.load(part / 'tok_frame.npy')
        tok = np.memmap(part / f'tok{layer}.f16', np.float16, 'r', shape=(len(tf), VJ_D))
        isp = (Fp.set.values == 'pool')[tf]
        role = np.array([P['videos'][v]['role'] for v in Fp.video.values])[tf]
        for r in ('train', 'calib', 'eval'):
            ii = np.flatnonzero(isp & (role == r))
            if len(ii):
                out.setdefault(r, []).append(torch.from_numpy(np.ascontiguousarray(tok[ii])).to(dev))
    return {r: torch.cat(x) for r, x in out.items()}


def blocks_of(arm):
    a = ARMS[arm]
    if a['kind'] == 't1':
        return [W] * (t1_dim(a['win']) // W)
    if a['kind'] == 't2':
        return [D] * (2 if a['ctx'] else 1)
    return [VJ_D]


# ---------------------------------------------------------------------------------------------- train
def train_one(cfg, seed, X, steps, out, dev, blocks, n_init=500_000, bs=4096, lr=5e-4, warmup=500):
    """nnnt.train_one with the TokenNorm blocks as an argument (A3 recipe otherwise identical)."""
    import torch
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    Nn, din = X.shape
    ii = torch.from_numpy(np.sort(g.choice(Nn, min(n_init, Nn), replace=False))).to(dev)
    init = X[ii].float()
    norm = TokenNorm(din).to(dev).fit_blocks(init, blocks, seed=seed)
    sae = MatryoshkaBatchTopKSAE(d_in=din, n_latents=W, prefixes=PREFIXES, k=K, seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    perm, p = torch.randperm(Nn, generator=gen, device=dev), 0
    history, t0 = [], time.time()
    for step in range(steps):
        if p + bs > Nn:
            perm, p = torch.randperm(Nn, generator=gen, device=dev), 0
        b = perm[p:p + bs]
        p += bs
        logs, gn = _train_step(sae, opt, norm(X[b]), step, steps, lr, warmup, 1.0)
        if step % 2000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            history.append(logs)
    sae.eval()
    extra = {'config': 'temporal', 'cfg': cfg, 'seed': seed}
    save_checkpoint(out / 'sae.pt', sae, norm, extra=extra)
    return {'train_time_s': time.time() - t0, 'n_train': Nn,
            'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()), 'history': history}


def cmd_train(args):
    import torch
    from src.eci.sae import load_sae
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    dev = torch.device('cuda')
    arms = [a for a in (args.arms.split(',') if args.arms else ARMS) if a != 'T0']
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else SEEDS
    root = OUT / 'sae'
    tj = OUT / f'train_{args.tag or "_".join(arms)}.json'
    res = json.loads(tj.read_text()) if tj.exists() else {}
    torch.cuda.reset_peak_memory_stats()
    for arm in arms:
        a = ARMS[arm]
        cache = None
        for s in seeds:
            name = f'{arm}_s{s}'
            od = root / name
            if (od / 'sae_theta.pt').exists() and not args.force:
                log(f'[SKIP] {name}')
                continue
            t0 = time.time()
            if a['kind'] == 't1':
                Xd = t1_data(torch, dev, a['win'], s)  # level-1 seed = level-2 seed
            elif cache is None:
                cache = t2_data(torch, dev, a['ctx']) if a['kind'] == 't2' else t3_data(torch, dev, a['layer'])
                Xd = cache
            else:
                Xd = cache
            log(f'{name}: data {({r: tuple(x.shape) for r, x in Xd.items()})} in {time.time() - t0:.0f}s')
            cfg = {'arm': arm, **a, 'width': W, 'prefixes': list(PREFIXES), 'k': K, 'blocks': blocks_of(arm)}
            ti = train_one(cfg, s, Xd['train'], args.steps, od, dev, blocks_of(arm))
            torch.backends.cuda.matmul.allow_tf32 = False
            sae, norm, ck = load_sae(od / 'sae.pt', dev)
            th, _ = calibrate_threshold(sae, Xd['calib'], norm=norm)
            set_threshold(sae, th)
            stt = sparsity_stats(sae, Xd['eval'], th, norm=norm)
            fire = torch.zeros(W, dtype=torch.float64, device=dev)
            with torch.no_grad():
                for b in range(0, len(Xd['eval']), 65536):
                    fire += (sae.encode(norm(Xd['eval'][b:b + 65536].float()), mode='threshold') > 0).sum(0).double()
            ck['state_dict']['threshold'] = torch.tensor(float(th))
            ck['theta_star'] = float(th)
            torch.save(ck, od / 'sae_theta.pt')
            res[name] = {'cfg': cfg, **{k_: v_ for k_, v_ in ti.items() if k_ != 'history'}, 'theta_star': th,
                         'eval_fve': stt['fve'], 'eval_l0': stt['l0'],
                         'eval_dead_frac': float((fire == 0).double().mean()), 'n_calib': len(Xd['calib']),
                         'n_eval': len(Xd['eval']), 'peak_mem_gb': torch.cuda.max_memory_allocated() / 2**30,
                         'gpu': torch.cuda.get_device_name(0)}
            log(f'{name}: {ti["train_time_s"]:.0f}s, theta* {th:.4f}, eval FVE {stt["fve"]:.4f} L0 {stt["l0"]:.2f} '
                f'dead {res[name]["eval_dead_frac"]:.3f}')
            tj.write_text(json.dumps(res, indent=1, default=float))
            if a['kind'] == 't1':
                del Xd
                torch.cuda.empty_cache()
        del cache
        torch.cuda.empty_cache()


def all_train():
    res = {}
    for f in sorted(OUT.glob('train_*.json')):
        res.update(json.loads(f.read_text()))
    return res


# ---------------------------------------------------------------------------------------------- codes
def frame_max(torch, sae, norm, X, tf_local, nf, chunk=262_144):
    """token rows X (n, d) fp16 cuda with frame ids tf_local (n,) non-decreasing -> (nf, W) fp16 frame max."""
    out = torch.zeros((nf, W), dtype=torch.float32, device=X.device)
    for a in range(0, len(X), chunk):
        z = sae.encode(norm(X[a:a + chunk].float()), mode='threshold')
        idx = tf_local[a:a + chunk].long()[:, None].expand(-1, W)
        out.scatter_reduce_(0, idx, z, 'amax', include_self=True)
    return out.half()


def cmd_codes(args):
    import torch
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    cd = BEE / 'codes'
    cd.mkdir(parents=True, exist_ok=True)
    P = N.load_plan()
    F, tf, st = diag_index()
    vi_of = {v['obs']: i for i, v in enumerate(P['videos'])}
    annv = [v for v in P['videos'] if v['annotated']]
    arms = args.arms.split(',') if args.arms else list(ARMS)
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else SEEDS
    lsae = lambda nm: load_sae(OUT / 'sae' / nm / 'sae_theta.pt', dev)[:2]  # noqa: E731
    t0 = time.time()
    # T0 / T1: per video, all frames + the D rows
    Drow = {v['obs']: F.k.values[F.obs.values == v['obs']] for v in annv}
    for arm in [a for a in arms if ARMS[a]['kind'] in ('t0', 't1')]:
        for s in seeds:
            nm = f'{arm}_s{s}'
            if (cd / f'{nm}__D.npy').exists() and not args.force:
                continue
            if arm != 'T0':
                sae, norm = lsae(nm)
            allc, Dc = [], []
            for v in annv:
                c = level1(v['obs'], s)
                if arm == 'T0':
                    x = c
                else:
                    with torch.no_grad():
                        f_ = t1_feats(torch, torch.from_numpy(c.astype(np.float32)).to(dev), ARMS[arm]['win'])
                        x = sae.encode(norm(f_), mode='threshold').half().cpu().numpy()
                allc.append(x)
                Dc.append(x[Drow[v['obs']]])
            np.save(cd / f'{nm}__all.npy', np.concatenate(allc))
            np.save(cd / f'{nm}__D.npy', np.concatenate(Dc))
            log(f'{nm}: codes all {sum(len(x) for x in allc):,} / D {sum(len(x) for x in Dc):,} ({time.time() - t0:.0f}s)')
    # T2 on D: centre tokens (diag) + ctx shards, frame-aligned chunks
    t2 = [a for a in arms if ARMS[a]['kind'] == 't2']
    if t2:
        todo = [(f'{a}_s{s}', a) for a in t2 for s in seeds if args.force or not (cd / f'{a}_s{s}__D.npy').exists()]
        if todo:
            dtok = np.memmap(DIAG / 'tok.f16', np.float16, 'r', shape=(len(tf), D))
            shards = sorted((BEE / 'ctx').glob('shard_*'))
            infos = [json.loads((s_ / 'info.json').read_text()) for s_ in shards]
            infos, shards = zip(*sorted(zip(infos, shards), key=lambda x: x[0]['token_range'][0]))
            assert infos[0]['token_range'][0] == 0 and infos[-1]['token_range'][1] == len(tf)
            assert all(a_['token_range'][1] == b_['token_range'][0] for a_, b_ in zip(infos[:-1], infos[1:]))
            S_ = {nm: lsae(nm) for nm, _ in todo}
            out = {nm: torch.zeros((len(F), W), dtype=torch.float16, device=dev) for nm, _ in todo}
            for inf, sh in zip(infos, shards):
                ta, tb = inf['token_range']
                ctx = {kc: np.memmap(sh / f'ctx{kc}.f16', np.float16, 'r', shape=(tb - ta, D)) for kc in CTX_K}
                fa = int(tf[ta])
                fb = int(tf[tb - 1]) + 1
                for a0 in range(fa, fb, 2000):
                    a1 = min(fb, a0 + 2000)
                    ra, rb = int(st[a0]), int(st[a1])
                    z = torch.from_numpy(np.ascontiguousarray(dtok[ra:rb])).to(dev)
                    cx = {kc: torch.from_numpy(np.ascontiguousarray(ctx[kc][ra - ta:rb - ta])).to(dev) for kc in CTX_K}
                    loc_f = torch.from_numpy(tf[ra:rb] - a0).to(dev)
                    with torch.no_grad():
                        for nm, arm in todo:
                            kc = ARMS[arm]['ctx']
                            X = z if not kc else torch.cat([z, cx[kc]], 1)
                            out[nm][a0:a1] = frame_max(torch, *S_[nm], X, loc_f, a1 - a0)
                log(f'T2 codes: shard {sh.name} done ({time.time() - t0:.0f}s)')
            for nm in out:
                np.save(cd / f'{nm}__D.npy', out[nm].cpu().numpy())
    # T3 on S20
    t3 = [a for a in arms if ARMS[a]['kind'] == 't3']
    if t3:
        todo = [(f'{a}_s{s}', a) for a in t3 for s in seeds if args.force or not (cd / f'{a}_s{s}__S20.npy').exists()]
        if todo:
            S20 = F[s20_mask(F)].reset_index(drop=True)
            key = {(vi_of[o], int(k)): i for i, (o, k) in enumerate(zip(S20.obs, S20.k))}
            S_ = {nm: lsae(nm) for nm, _ in todo}
            out = {nm: torch.zeros((len(S20), W), dtype=torch.float16, device=dev) for nm, _ in todo}
            seen = np.zeros(len(S20), bool)
            for part in vj_parts():
                Fp = pd.read_parquet(part / 'frames.parquet')
                if not (Fp.set == 's20').any():
                    continue
                tfp = np.load(part / 'tok_frame.npy')
                toks = {l_: np.memmap(part / f'tok{l_}.f16', np.float16, 'r', shape=(len(tfp), VJ_D)) for l_ in VJ_LAYERS}
                rows = np.array([key.get((int(v), int(k)), -1) if s_ == 's20' else -1
                                 for s_, v, k in zip(Fp.set, Fp.video, Fp.k)])
                ii = np.flatnonzero(rows[tfp] >= 0)
                if not len(ii):
                    continue
                fl = rows[tfp[ii]]
                seen[np.unique(fl)] = True
                # local contiguous frame ids
                uf, inv = np.unique(tfp[ii], return_inverse=True)
                loc_f = torch.from_numpy(inv).to(dev)
                X = {l_: torch.from_numpy(np.ascontiguousarray(toks[l_][ii])).to(dev) for l_ in VJ_LAYERS}
                with torch.no_grad():
                    for nm, arm in todo:
                        fm = frame_max(torch, *S_[nm], X[ARMS[arm]['layer']], loc_f, len(uf))
                        out[nm][torch.from_numpy(rows[uf]).to(dev)] = fm
                log(f'T3 codes: {part.name} ({time.time() - t0:.0f}s)')
            assert seen.all(), f'S20 frames without V-JEPA tokens: {(~seen).sum()}'
            for nm in out:
                np.save(cd / f'{nm}__S20.npy', out[nm].cpu().numpy())


def s20_mask(F):
    P = N.load_plan()
    off = {v['obs']: v['fps1_off'] for v in P['videos']}
    return ((F.k.values - F.obs.map(off).values) % 20) == 0


# ---------------------------------------------------------------------------------------------- score
def ranks_auc(s, y):
    from scipy.stats import rankdata
    P_, N_ = int(y.sum()), int((~y).sum())
    if P_ == 0 or N_ == 0:
        return float('nan')
    r = rankdata(s)
    return float((r[y].sum() - P_ * (P_ + 1) / 2) / (P_ * N_))


def bout_len(Yfull, vid):
    """Per frame: length of the run of consecutive positive frames containing it (0 if negative), per video."""
    out = np.zeros(len(Yfull), np.int32)
    for v in np.unique(vid):
        ii = np.flatnonzero(vid == v)
        c = Yfull[ii].astype(np.int8)
        ch = np.flatnonzero(np.diff(np.r_[-1, c, -1]) != 0)
        for a, b in zip(ch[:-1], ch[1:]):
            if c[a]:
                out[ii[a:b]] = b - a
    return out


def score_context():
    """Labels, splits, annotator / bout / geometry features for the D frames and all annotated frames."""
    P = N.load_plan()
    PL = load_plan()
    F = pd.read_parquet(DIAG / 'frames.parquet')
    F['sub20'] = s20_mask(F)
    vids = [v for v in P['videos'] if v['annotated']]
    a = pd.read_csv(REPO / C['ann'], usecols=['observation_id', 'Y_nn', 'Y_np', 'Y_nt'])
    rows_all = np.concatenate([np.arange(v['src_lo'], v['src_lo'] + v['n']) for v in vids])
    y = a.iloc[rows_all]
    Yall = np.stack([(y.Y_nn.values > 0) | (y.Y_np.values > 0), y.Y_nt.values > 0, y.Y_nn.values > 0,
                     y.Y_np.values > 0], 1)
    obs_all = y.observation_id.values
    unit_all = pd.Series(obs_all).str.rsplit('_', n=2).str[0].values
    units, SPL = N.splits_of(set(unit_all))
    uidx_all = pd.Series(range(len(units)), index=units).loc[unit_all].values
    bl = np.stack([bout_len(Yall[:, li], obs_all) for li in range(2)], 1)
    off = np.cumsum([0] + [v['n'] for v in vids])[:-1]
    vpos = {v['obs']: i for i, v in enumerate(vids)}
    Dall = off[F.obs.map(vpos).values] + F.k.values  # D frame -> index in the all-frame order
    assert (F.row.values == rows_all[Dall]).all()
    ann = PL['annotator']
    cons_all = np.isin(pd.Series(obs_all).map(ann).values, CONS)
    geom = None
    if (BEE / 'geom.npz').exists():
        z = np.load(BEE / 'geom.npz')
        geom = {k_: z[k_] for k_ in ('contact', 'run', 'speed', 'accel')}
        assert len(geom['run']) == len(rows_all)
    sets = {'D': Dall, 'S20': Dall[F.sub20.values], 'all': np.arange(len(rows_all))}
    sets['D_cons'] = sets['D'][cons_all[sets['D']]]
    sets['S20_cons'] = sets['S20'][cons_all[sets['S20']]]
    return dict(Y=Yall, uidx=uidx_all, SPL=SPL, units=units, bl=bl, sets=sets, geom=geom, n_all=len(rows_all),
                F=F)


def extras(X, ctx, ix, half_of, sel):
    """Secondary readouts of the cross-fitted neurons. X (len(ix), W) numpy, sel = score_set output."""
    Y, bl, geom = ctx['Y'][ix], ctx['bl'][ix], ctx['geom']
    gc = {k_: v_[ix] for k_, v_ in geom.items()} if geom is not None else None
    res = {}
    for li, lb in enumerate(('nose_nose', 'nose_tail')):
        acc = {k_: [] for k_ in ('auc_long', 'auc_short', 'generic_share', 'label_share', 'dur_auc', 'rho_speed',
                                 'rho_accel')}
        for sn, half in half_of.items():
            if not sn.startswith('rnd'):
                continue
            for d_, held in ((0, 1), (1, 0)):
                j = sel['splits'][sn][lb]['neurons'][d_]
                H = np.flatnonzero(half == held)
                x = X[H, j].astype(np.float32)
                yy, bb = Y[H, li], bl[H, li]
                neg = ~yy
                lng, sht = yy & (bb >= 5), yy & (bb < 5)
                acc['auc_long'].append(ranks_auc(x[lng | neg], lng[lng | neg]))
                acc['auc_short'].append(ranks_auc(x[sht | neg], sht[sht | neg]))
                npos = int(yy.sum())
                top = np.argsort(-x, kind='stable')[:npos]
                acc['label_share'].append(float(yy[top].mean()))
                if gc is not None:
                    unl = ~Y[H, :].any(1)
                    acc['generic_share'].append(float((gc['contact'][H][top] & unl[top]).mean()))
                    r = gc['run'][H]
                    sus, bri = r >= 10, (r >= 1) & (r <= 3)
                    m_ = sus | bri
                    acc['dur_auc'].append(ranks_auc(x[m_], sus[m_]))
                    from scipy.stats import spearmanr
                    for kn, kk in (('rho_speed', 'speed'), ('rho_accel', 'accel')):
                        q = gc[kk][H]
                        ok = np.isfinite(q)
                        acc[kn].append(float(spearmanr(x[ok], q[ok]).statistic))
        res[lb] = {k_: float(np.nanmean(v_)) if v_ else None for k_, v_ in acc.items()}
    return res


def kin_all(X, ctx, ix):
    """All neurons: Spearman rho with speed / |accel| and label-free contact-duration AUROC on the scored frames."""
    from scipy.stats import rankdata
    g = {k_: v_[ix] for k_, v_ in ctx['geom'].items()}
    R = rankdata(X.astype(np.float32), axis=0)
    R = (R - R.mean(0)) / np.maximum(R.std(0), 1e-9)
    out = {}
    for kk in ('speed', 'accel'):
        ok = np.isfinite(g[kk])
        q = rankdata(g[kk][ok])
        q = (q - q.mean()) / q.std()
        Rr = R[ok]
        Rr = (Rr - Rr.mean(0)) / np.maximum(Rr.std(0), 1e-9)
        rho = (Rr * q[:, None]).mean(0)
        out[kk] = {'max_abs_rho': float(np.nanmax(np.abs(rho))), 'argmax': int(np.nanargmax(np.abs(rho))),
                   'n_abs_rho_ge_0.3': int((np.abs(rho) >= 0.3).sum()), 'rho': rho.astype(np.float32).tolist()}
    r = g['run']
    sus, bri = r >= 10, (r >= 1) & (r <= 3)
    m_ = sus | bri
    Rm = rankdata(X[m_].astype(np.float32), axis=0)
    P_, N_ = int(sus[m_].sum()), int(bri[m_].sum())
    auc = (Rm[sus[m_]].sum(0) - P_ * (P_ + 1) / 2) / (P_ * N_)
    out['dur_auc'] = {'max': float(np.nanmax(auc)), 'argmax': int(np.nanargmax(auc)),
                      'n_ge_0.65': int((auc >= 0.65).sum()), 'auc': auc.astype(np.float32).tolist()}
    return out


def cmd_score(args):
    import torch
    dev = torch.device('cuda')
    ctx = score_context()
    T = all_train()
    cd = BEE / 'codes'
    sf = OUT / 'score.json'
    res = json.loads(sf.read_text()) if sf.exists() and not args.force else {}
    res.setdefault('sets', {})
    res['n'] = {k_: int(len(v_)) for k_, v_ in ctx['sets'].items()}
    res['base_rate'] = {k_: {lb: float(ctx['Y'][v_, i].mean()) for i, lb in enumerate(LABELS)}
                        for k_, v_ in ctx['sets'].items()}
    Yg = torch.from_numpy(ctx['Y']).to(dev)
    plan_sets = []
    for nm, arm, s in names():
        kind = ARMS[arm]['kind']
        if kind == 't3':
            plan_sets += [(nm, 'S20', 'S20'), (nm, 'S20_cons', 'S20')]
        else:
            plan_sets += [(nm, 'D', 'D'), (nm, 'D_cons', 'D')]
            if arm == 'T0':
                plan_sets += [(nm, 'S20', 'D'), (nm, 'S20_cons', 'D')]
            if kind in ('t0', 't1') and args.all_frames:
                plan_sets += [(nm, 'all', 'all')]
    Dpos = {}  # set -> positions inside the stored code file's row space

    def rows_in(fs, src):
        key = (fs, src)
        if key not in Dpos:
            base = ctx['sets'][src]
            m = pd.Series(np.arange(len(base)), index=base)
            Dpos[key] = m.loc[ctx['sets'][fs]].values
        return Dpos[key]

    loaded = {}
    for nm, fs, src in plan_sets:
        key = f'{nm}|{fs}'
        f = cd / f'{nm}__{src}.npy'
        if key in res['sets'] or not f.exists():
            continue
        if (nm, src) not in loaded:
            loaded.clear()
            loaded[(nm, src)] = np.load(f)
        X = loaded[(nm, src)][rows_in(fs, src)]
        ix = ctx['sets'][fs]
        t0 = time.time()
        Xg = torch.from_numpy(np.ascontiguousarray(X)).to(dev)
        half_of = {sn: h[ctx['uidx'][ix]] for sn, h in ctx['SPL'].items()}
        out, _ = N.score_set(torch, Xg, Yg[torch.from_numpy(ix).to(dev)], half_of, dev)
        del Xg
        torch.cuda.empty_cache()
        rec = {'mean': out['mean'], 'primary': out['primary'], 'specificity': out['specificity'],
               'neurons_rnd0': {lb: out['splits']['rnd0'][lb]['neurons'] for lb in ('nose_nose', 'nose_tail')}}
        if fs in ('D', 'S20') and ctx['geom'] is not None:
            rec['extras'] = extras(X, ctx, ix, half_of, out)
            ka = kin_all(X, ctx, ix)
            rec['kin_all'] = {k_: {q: v_ for q, v_ in d_.items() if q not in ('rho', 'auc')} for k_, d_ in ka.items()}
            sp = out['specificity']
            rec['kin_modal'] = {'nn': {'rho_speed': ka['speed']['rho'][sp['nn_neuron']],
                                       'rho_accel': ka['accel']['rho'][sp['nn_neuron']],
                                       'dur_auc': ka['dur_auc']['auc'][sp['nn_neuron']]},
                                'nt': {'rho_speed': ka['speed']['rho'][sp['nt_neuron']],
                                       'rho_accel': ka['accel']['rho'][sp['nt_neuron']],
                                       'dur_auc': ka['dur_auc']['auc'][sp['nt_neuron']]}}
        res['sets'][key] = rec
        m = out['mean']
        log(f'{key}: nn R~ {m["nose_nose"]["R"]:.4f} AUROC {m["nose_nose"]["A"]:.4f} | nt R~ {m["nose_tail"]["R"]:.4f} '
            f'AUROC {m["nose_tail"]["A"]:.4f} | primary {out["primary"]:.4f} ({time.time() - t0:.0f}s)')
        sf.write_text(json.dumps(res, indent=1, default=float))
    res['train'] = T
    sf.write_text(json.dumps(res, indent=1, default=float))
    report(res)


def report(res=None):
    res = res or json.loads((OUT / 'score.json').read_text())
    Sx = res['sets']

    def vals(arm, fs, q='primary', lb=None, seeds=SEEDS):
        out = []
        for s in seeds:
            r = Sx.get(f'{arm}_s{s}|{fs}')
            if r is None:
                continue
            out.append(r['primary'] if q == 'primary' else r['mean'][lb][q])
        return out

    def ex(arm, fs, lb, q, seeds=SEEDS):
        return [Sx[f'{arm}_s{s}|{fs}']['extras'][lb][q] for s in seeds
                if f'{arm}_s{s}|{fs}' in Sx and 'extras' in Sx[f'{arm}_s{s}|{fs}']]
    ms = lambda x: f'{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}' if len(x) > 1 else (f'{x[0]:.3f}' if x else '-')  # noqa
    L = ['# Temporal context for v2 (mice v1) — results', '',
         'R~ = cross-fitted normalised R-precision (10 random pool splits), AUROC = test AUROC of the AUROC-selected '
         'neuron; primary = mean(nn R~, nt R~); mean ± sd over seeds (n in the seeds column).', '',
         'Frames: ' + ', '.join(f'{k_} {v_:,}' for k_, v_ in res['n'].items()), '']
    for fs_group in (('D', 'D_cons'), ('S20', 'S20_cons'), ('all',)):
        for fs in fs_group:
            arms = [a for a in ARMS if vals(a, fs)]
            if not arms:
                continue
            L += [f'## Frames {fs}', '', '| arm | seeds | primary R~ | nn R~ | nn AUROC | nt R~ | nt AUROC | '
                  'nn_mutual R~ | np R~ |', '|---|---|---|---|---|---|---|---|---|']
            for a in arms:
                r = [vals(a, fs), vals(a, fs, 'R', 'nose_nose'), vals(a, fs, 'A', 'nose_nose'),
                     vals(a, fs, 'R', 'nose_tail'), vals(a, fs, 'A', 'nose_tail'), vals(a, fs, 'R', 'nn_mutual'),
                     vals(a, fs, 'R', 'np_directional')]
                L.append(f'| {a} | {len(r[0])} | ' + ' | '.join(ms(x) for x in r) + ' |')
            L.append('')
    # secondary
    for fs in ('D', 'S20'):
        arms = [a for a in ARMS if ex(a, fs, 'nose_nose', 'auc_long')]
        if not arms:
            continue
        L += [f'## Secondary readouts of the selected neurons (held-out half), frames {fs}', '',
              '| arm | label | long-bout AUROC | short-bout AUROC | label share top-N | generic-contact share top-N | '
              'contact-duration AUROC | rho speed | rho accel |', '|---|---|---|---|---|---|---|---|---|']
        for a in arms:
            for lb in ('nose_nose', 'nose_tail'):
                L.append(f'| {a} | {lb} | ' + ' | '.join(ms(ex(a, fs, lb, q)) for q in
                                                         ('auc_long', 'auc_short', 'label_share', 'generic_share',
                                                          'dur_auc', 'rho_speed', 'rho_accel')) + ' |')
        L += ['', f'### All-neuron kinematics, frames {fs} (mean over seeds)', '',
              '| arm | max abs rho speed | n abs rho >= 0.3 (speed) | max abs rho accel | max contact-duration AUROC | '
              'n duration AUROC >= 0.65 |', '|---|---|---|---|---|---|']
        for a in arms:
            k_ = [Sx[f'{a}_s{s}|{fs}']['kin_all'] for s in SEEDS if f'{a}_s{s}|{fs}' in Sx]
            L.append(f'| {a} | {ms([x["speed"]["max_abs_rho"] for x in k_])} | '
                     f'{ms([x["speed"]["n_abs_rho_ge_0.3"] for x in k_])} | {ms([x["accel"]["max_abs_rho"] for x in k_])} '
                     f'| {ms([x["dur_auc"]["max"] for x in k_])} | {ms([x["dur_auc"]["n_ge_0.65"] for x in k_])} |')
        L.append('')
    # decision
    dec = {'candidates': {}, 'tests': {}, 'keep': {}}
    fam = {'T1': (('T1m', 'T1d'), 'D', ('T0', 'T1id')), 'T2': (('T2a', 'T2b'), 'D', ('T0', 'T2id')),
           'T3': (('T3g33',), 'S20', ('T0',))}
    for F_, (cands, fs, refs) in fam.items():
        m3 = {c: float(np.mean(vals(c, fs, seeds=(0, 1, 2)))) for c in cands if len(vals(c, fs, seeds=(0, 1, 2))) == 3}
        if not m3:
            continue
        best = max(m3, key=m3.get)
        dec['candidates'][F_] = {'seeds0-2_primary': m3, 'best': best}
        for ref in refs:
            for lb, q in (('primary', 'primary'), ('nose_nose', 'R'), ('nose_tail', 'R'), ('nose_nose', 'A'),
                          ('nose_tail', 'A')):
                x = vals(best, fs, q, None if q == 'primary' else lb)
                y_ = vals(ref, fs, q, None if q == 'primary' else lb)
                if len(x) > 1 and len(y_) > 1:
                    dec['tests'][f'{best} - {ref} | {fs} | {lb} {"R~" if q != "A" else "AUROC"}'] = N.welch(x, y_)
            x, y_ = vals(best, fs + '_cons', 'R', 'nose_tail'), vals(ref, fs + '_cons', 'R', 'nose_tail')
            if len(x) > 1 and len(y_) > 1:
                dec['tests'][f'{best} - {ref} | {fs}_cons | nose_tail R~'] = N.welch(x, y_)
            for lb in ('nose_nose', 'nose_tail'):
                for q in ('auc_long', 'auc_short', 'generic_share'):
                    x, y_ = ex(best, fs, lb, q), ex(ref, fs, lb, q)
                    if len(x) > 1 and len(y_) > 1:
                        if F_ == 'T1' and len(x) == len(y_):  # level-1 seed = level-2 seed: paired
                            dec['tests'][f'{best} - {ref} | {fs} | {lb} {q} (paired by seed)'] = N.paired(x, y_)
                        else:
                            dec['tests'][f'{best} - {ref} | {fs} | {lb} {q} (Welch)'] = N.welch(x, y_)
        t = dec['tests'].get(f'{best} - T0 | {fs} | primary R~')
        if t:
            dec['keep'][F_] = bool(t[0] >= 0.02 and t[1] > 0)
    kept = [dec['candidates'][f]['best'] for f, k_ in dec['keep'].items() if k_]
    if kept:
        fsx = {a: ('S20' if ARMS[a]['kind'] == 't3' else 'D') for a in kept}
        gain = {a: dec['tests'][f'{a} - T0 | {fsx[a]} | primary R~'][0] for a in kept}
        dec['recommendation'] = max(gain, key=gain.get)
    else:
        dec['recommendation'] = 'none kept: stay with A3 (T0)'
    L += ['## Decision (pre-registered rule, module docstring)', '', '```', json.dumps(dec, indent=1), '```', '']
    # train summary
    T = res.get('train', {})
    if T:
        L += ['## SAE training (eval FVE, dead share, train time)', '', '| SAE | n train | eval FVE | eval L0 | dead | '
              'train s | peak GB |', '|---|---|---|---|---|---|---|']
        for nm in sorted(T):
            t = T[nm]
            L.append(f'| {nm} | {t["n_train"]:,} | {t["eval_fve"]:.4f} | {t["eval_l0"]:.2f} | {t["eval_dead_frac"]:.3f} '
                     f'| {t["train_time_s"]:.0f} | {t.get("peak_mem_gb", 0):.1f} |')
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    (OUT / 'decisions.json').write_text(json.dumps(dec, indent=1))
    log('\n'.join(L[:80]))


# ---------------------------------------------------------------------------------------------- sheets
def cmd_sheets(args):
    """Per arm: the seed with the highest eval FVE (T0: A3 round-1 train.json), nn / nt neuron chosen on pools half 0
    of split rnd0, its 24 highest held-out (half 1) frames, <= 1 per video, per-token heat maps re-encoded."""
    import torch
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    P = N.load_plan()
    SC = json.loads((OUT / 'score.json').read_text())
    T = SC['train']
    T1r = json.loads((N.OUT / 'train.json').read_text())
    ctx = score_context()
    Fd = ctx['F']
    vi_of = {v['obs']: i for i, v in enumerate(P['videos'])}
    half_unit = ctx['SPL']['rnd0']
    a_ = pd.read_csv(REPO / C['ann'], usecols=['Y_nn', 'Y_np', 'Y_nt'])
    _, Rr = R4.obs_ranges_of(REPO / C['ann'])
    pipe = D3(dev, Rr)
    arms = args.arms.split(',') if args.arms else list(ARMS)
    sheets = OUT / 'sheets'
    sheets.mkdir(parents=True, exist_ok=True)
    out = json.loads((OUT / 'locality.json').read_text()) if (OUT / 'locality.json').exists() else {}
    vj = None
    for arm in arms:
        kind = ARMS[arm]['kind']
        if arm == 'T0':
            s = max(SEEDS, key=lambda s_: T1r[f'A3_w512_s{s_}']['eval_fve'])
        else:
            ss = [s_ for s_ in SEEDS if f'{arm}_s{s_}' in T]
            if not ss:
                continue
            s = max(ss, key=lambda s_: T[f'{arm}_s{s_}']['eval_fve'])
        nm = f'{arm}_s{s}'
        fs = 'S20' if kind == 't3' else 'D'
        rec = SC['sets'].get(f'{nm}|{fs}')
        if rec is None:
            continue
        X = np.load(BEE / 'codes' / f'{nm}__{fs}.npy')
        ix = ctx['sets'][fs]
        Fs = Fd if fs == 'D' else Fd[Fd.sub20.values].reset_index(drop=True)
        hu = half_unit[ctx['uidx'][ix]]
        sae = norm = None
        if kind != 't0':
            sae, norm = load_sae(OUT / 'sae' / nm / 'sae_theta.pt', dev)[:2]
        l1 = None
        if kind in ('t0', 't1'):
            l1 = load_sae(N.OUT / 'sae' / f'A3_w512_s{s}' / 'sae_theta.pt', dev)[:2]
        if kind == 't3' and vj is None:
            side, bf = vj_side()
            vj = VJ(dev, side, bf16=bf)
        for lb in ('nose_nose', 'nose_tail'):
            j = rec['neurons_rnd0'][lb][0]  # selected on half 0
            cand = np.flatnonzero(hu == 1)
            o_ = np.argsort(-X[cand, j].astype(np.float32), kind='stable')
            picks, seen = [], set()
            for i in cand[o_]:
                ob = Fs.obs.values[i]
                if ob in seen:
                    continue
                seen.add(ob)
                picks.append(i)
                if len(picks) == 24:
                    break
            tiles, recs = [], []
            for i in picks:
                ob, k = Fs.obs.values[i], int(Fs.k.values[i])
                v = P['videos'][vi_of[ob]]
                if kind == 't3':
                    kw, idx = vj_clip_job(v, k, vj.S)
                    fr = R4.decode(**{k_: x for k_, x in kw.items() if k_ != 'jpgs'})[idx]
                    xc = pipe.rot(torch.from_numpy(np.ascontiguousarray(fr)).to(dev), v['rot'])
                    with torch.no_grad():
                        tok = vj.forward(xc[None])[ARMS[arm]['layer']][0]
                    fd = pipe.dark(xc[TPOS:TPOS + 1], ob)[0]
                    m = fd >= N.THR_TIGHT
                    grid = vj.G
                    x_img = xc[TPOS]
                    with torch.no_grad():
                        act = torch.zeros(grid * grid, device=dev)
                        act[m] = sae.encode(norm(tok[m].float()), mode='threshold')[:, j]
                    recomputed = float(act.max())
                else:
                    ks = [k] + ([max(k - ARMS[arm]['ctx'], 0), min(k + ARMS[arm]['ctx'], v['n'] - 1)]
                                if kind == 't2' and ARMS[arm]['ctx'] else [])
                    frs = [R4.decode(**{k_: x for k_, x in dec_job(v, kk, 1).items() if k_ != 'jpgs'})[0] for kk in ks]
                    xs = pipe.rot(torch.from_numpy(np.stack(frs)).to(dev), v['rot'])
                    z = pipe.tokens(xs)
                    fd = pipe.dark(xs[:1], ob)[0]
                    m = fd >= N.THR_TIGHT
                    grid = G
                    x_img = xs[0]
                    with torch.no_grad():
                        act = torch.zeros(grid * grid, device=dev)
                        if kind == 't2':
                            xin = z[0, m] if not ARMS[arm]['ctx'] else torch.cat(
                                [z[0, m], ((z[1, m].float() + z[2, m].float()) / 2).half()], 1)
                            act[m] = sae.encode(norm(xin.float()), mode='threshold')[:, j]
                            recomputed = float(act.max())
                        else:
                            a1 = l1[0].encode(l1[1](z[0, m].float()), mode='threshold')  # (n_kept, 512) level 1
                            if kind == 't0':
                                act[m] = a1[:, j]
                                recomputed = float(act.max())
                            else:  # centre-frame attribution of the level-2 neuron
                                cb = DIL.index(0) * W if ARMS[arm]['win'] == 'dil' else 0
                                sc_ = norm.scale[cb:cb + W] if norm.scale.dim() else norm.scale
                                wj = (sae.W_enc[cb:cb + W, j] * sc_).clamp(min=0)
                                act[m] = a1 @ wj
                                recomputed = float('nan')
                act = act.cpu().numpy()
                fdn = fd.cpu().numpy()
                kept = m.cpu().numpy()
                tot = act.sum()
                hi = act >= 0.5 * act.max() if act.max() > 0 else np.zeros_like(kept)
                yv = a_.iloc[int(v['src_lo'] + k)]
                lab = ''.join(c_ for c_, col in (('nn', 'Y_nn'), ('np', 'Y_np'), ('nt', 'Y_nt')) if yv[col] > 0) or '-'
                g_ = ctx['geom']
                gi = ix[i]
                recs.append({'obs': ob, 'k': k, 'stored': float(X[i, j]), 'recomputed': recomputed, 'label': lab,
                             'contact': bool(g_['contact'][gi]) if g_ is not None else None,
                             'contact_run': int(g_['run'][gi]) if g_ is not None else None,
                             'mass_animal_px': float((act * fdn).sum() / tot) if tot > 0 else np.nan,
                             'baseline_kept_animal_px': float(fdn[kept].mean()) if kept.any() else np.nan,
                             'n_half_max': int(hi.sum()), 'animal_tokens': int((fdn >= N.THR_TIGHT).sum()),
                             'spread_rel': float(hi.sum() / max(1, (fdn >= N.THR_TIGHT).sum())),
                             'n_active': int((act > 0).sum()), 'n_kept': int(kept.sum())})
                tiles.append((x_img.cpu().numpy(), act.reshape(grid, grid), recs[-1]))
            R_ = pd.DataFrame(recs)
            pos_l = {'nose_nose': ('nn', 'np'), 'nose_tail': ('nt',)}[lb]
            summ = {c_: float(R_[c_].mean()) for c_ in ('mass_animal_px', 'baseline_kept_animal_px', 'n_half_max',
                                                        'spread_rel', 'n_active', 'n_kept')}
            summ['share_labelled_' + lb] = float(R_.label.apply(lambda s_: any(p in s_ for p in pos_l)).mean())
            summ['share_contact_unlabelled'] = float((R_.contact.astype(bool) & (R_.label == '-')).mean())
            summ['median_contact_run'] = float(R_.contact_run.median())
            summ['label_counts'] = R_.label.value_counts().to_dict()
            if R_.recomputed.notna().any():
                summ['recompute_max_rel_err'] = float((np.abs(R_.recomputed - R_.stored) /
                                                       np.maximum(R_.stored, 1e-6)).max())
            fn = sheets / f'{arm}_{lb}_s{s}_n{j}.jpg'
            what = 'centre-frame attribution map (level-1 A3 tokens x positive level-2 weights)' if kind == 't1' \
                else 'per-token activation'
            N.contact_sheet(tiles, fn, f'{nm} neuron {j} ({lb}-selected on pools half 0 of rnd0; top held-out frames, '
                                       f'1 per video; {what})', grid)
            out[f'{arm}|{lb}'] = {'name': nm, 'neuron': j, 'summary': summ, 'frames': recs, 'sheet': str(fn)}
            log(f'{arm} {lb}: ' + json.dumps(summ))
            (OUT / 'locality.json').write_text(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- cost
def cmd_cost(args):
    out = {}
    for f in sorted((OUT / 'logs').glob('*.out')):
        ut, ts, name, mem = [], [], None, []
        for line in f.read_text(errors='replace').splitlines():
            if line.startswith('[gpu] '):
                p_ = [x.strip() for x in line[6:].split(',')]
                try:
                    ts.append(pd.Timestamp(p_[0].replace('/', '-')))
                    name = p_[1]
                    ut.append(float(p_[2].split()[0]))
                    mem.append(float(p_[3].split()[0]))
                except Exception:
                    pass
        if ut:
            out[f.name] = {'gpu': name, 'gpu_h': (ts[-1] - ts[0]).total_seconds() / 3600 + 1 / 60,
                           'util_mean': float(np.mean(ut)), 'mem_max_mib': float(max(mem)), 'samples': len(ut)}
    tot = {}
    for k_, v_ in out.items():
        st_ = k_.split('_')[1] if k_.startswith('tmp_') else k_.split('_')[0]
        t = tot.setdefault(st_, {'gpu_h': 0.0, 'util': [], 'mem_max_mib': 0.0, 'jobs': 0})
        t['gpu_h'] += v_['gpu_h']
        t['util'].append(v_['util_mean'])
        t['mem_max_mib'] = max(t['mem_max_mib'], v_['mem_max_mib'])
        t['jobs'] += 1
    summ = {k_: {'gpu_h': round(v_['gpu_h'], 2), 'util_mean': round(float(np.mean(v_['util'])), 1),
                 'mem_max_gib': round(v_['mem_max_mib'] / 1024, 1), 'jobs': v_['jobs']} for k_, v_ in tot.items()}
    (OUT / 'cost.json').write_text(json.dumps({'jobs': out, 'by_step': summ}, indent=1))
    print(json.dumps(summ, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('plan', 'vjsmoke', 'pool', 'ctx', 'vj', 'geom', 'train', 'codes', 'score',
                                    'report', 'sheets', 'cost'))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--videos', default='')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--arms', default='')
    p.add_argument('--seeds', default='')
    p.add_argument('--tag', default='')
    p.add_argument('--shard-frames', type=int, default=36_000)
    p.add_argument('--pool-parts', type=int, default=4)
    p.add_argument('--vj-part', type=int, default=2600)
    p.add_argument('--all-frames', action='store_true')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    {'plan': cmd_plan, 'vjsmoke': cmd_vjsmoke, 'pool': cmd_pool, 'ctx': cmd_ctx, 'vj': cmd_vj, 'geom': cmd_geom,
     'train': cmd_train, 'codes': cmd_codes, 'score': cmd_score, 'report': lambda a: report(), 'sheets': cmd_sheets,
     'cost': cmd_cost}[args.step](args)


if __name__ == '__main__':
    main()
