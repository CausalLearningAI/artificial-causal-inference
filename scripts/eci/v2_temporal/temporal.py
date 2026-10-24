"""
TEMPORAL HANDLING for the ECI v2 representation, re-tested fairly with today's recipe (2026-10-25).

==================================================================================================================
PRE-REGISTRATION (written 2026-10-25 ~02:00, BEFORE any token of this study was encoded and BEFORE any score of any arm
below existed. Known before writing: the 2026-10-20 temporal test (results/vision/eci/decisions/temporal/table.md: on
mice v1 with the older A3 recipe - 512-latent SAE, 6k steps, 10k-frame pool - every temporal addition lost: level-2
code dictionary -0.049 primary R~, a +-0.4 / 1 s token context-MEAN channel [z_t, (z_t-k + z_t+k)/2] -0.045, V-JEPA 2-g
at 768 px -0.039..-0.043; readout smoothing ~0.6 s helped nose_nose +0.014); MABe22 encoder study (V-JEPA 2-L at 512 px,
last / 5/6 layer, 1 fps frames, AUROC: -0.068 vs DINOv3-L); CalMS21 (VJ-64 0.811 vs D3L-64 0.881 primary AUROC); the v2
CRIM13 check (smoothing +0.028 mean R~ over 11 behaviours).)

QUESTIONS (the user's, 2026-10-25)
    Q1 "Is smoothing a posteriori better than ANY local temporal information?"
    Q2 "Would a video model (V-JEPA again) work better, solving local temporal context internally?"

FIXED PIPELINE (everything except the temporal handling is the current v2 recipe):
    encoder  DINOv3-L/16 block 20 + final LayerNorm (src/eci/v2/encode.py, rev ea8dc28), bf16, native side:
             MABe22 512 px (224 source upscaled, the side of every earlier MABe22 DINOv3 arm; 32 grid), CalMS21 1024 px
             (1024 x 570 padded to a square; 64 grid), CRIM13 640 px (40 grid), mice v1 1024 px, ants 768 / 832 / 816 px.
    mask     TIGHT: token kept iff >= 25% of its 16 x 16 input patch are animal pixels; animal pixel = grey < 60 AND
             background - grey > 40 (the 'dark' cue of every v2 domain). Background per video: MABe22 = 0.98 quantile
             (kthvalue 29 of 30 evenly spaced frames, the MABe22 plan's rule; no arena waiver), CalMS21 / CRIM13 = per-
             pixel median of 32 evenly spaced frames (their v2 / benchmark rule), lab = the frozen v2 series config.
             The mask of the CENTRE frame is used for every arm (fixed camera: the same token positions in the
             neighbouring frames).
    SAE      Matryoshka BatchTopK 2048 (prefixes 128 / 256 / 512 / 2048), k 16, 30,000 steps x 4096 tokens, lr 5e-4,
             TokenNorm per 1024-d block, theta* calibrated on calib tokens (mean L0 = k) (src/eci/v2/sae.py = V2-full
             grid pick 2026-10-24). SEEDS 1000-1004 (5 per arm, fresh), identical seeds across arms.
    readout  frame max over the kept tokens of the theta* codes.
    data     MATCHED: every arm trains on the SAME centre frames (the pool) and is scored on the SAME frames.

ARMS (only the temporal handling changes; 5 fps grid, s = one 5 fps step = 0.2 s)
    T0   v2 today: SAE on z_t; frame max; readout smoothing = centred mean of the frame codes over t - s, t, t + s
         (0.6 s; src/eci/v2/readout.smooth h 1).
    T1   token-level temporal pooling BEFORE the SAE: x_t = (z_t-s + z_t + z_t+s) / 3 at each kept position of t.
    T2   symmetric motion channel: x_t = [z_t, (z_t+s - z_t-s) / 2] (2048-d, TokenNorm per 1024 block).
         RUN CONDITION (user: "only if T1 is not enough to answer"): T2 is trained on a benchmark only if T1 is NOT
         adopted (rule below) on the design benchmarks; its tokens come from the same store, no extra encode.
    T3   video model: V-JEPA 2-L (facebook/vjepa2-vitl-fpc64-256, encoder only; 24 blocks) block 20 (label-free 5/6-depth
         rule) + the encoder's final LayerNorm (the model truncated after block 20). Clip = 16 frames at 5 fps (one
         5 fps step between clip frames; frames t + s (j - 7), j = 0..15, clamped to the video), token of t = tubelet 3 =
         clip frames (t - s, t) (the lab / MABe22 convention), RoPE height / width positions INTERPOLATED into the
         16 x 16 pretraining grid (the MABe22 encoder-study winner, 'interp'), ImageNet mean / std, SDPA.
         Same tight mask (centre frame) and SAE recipe.
    READOUTS: every arm is read out twice. 'raw' = frame max at t; 'sm' = the 0.6 s centred mean of the frame values at
         t - s, t, t + s. Named variants: T0 (= sm, the current v2), T0u (raw: no temporal step at all), T1 (raw: the
         temporal step moved from the readout to the tokens), T1s (sm: both), T2 (raw), T2s (sm), T3 (raw: the video
         model replaces the smoothing), T3s (sm; only where its extra clips fit the budget, see FEASIBILITY).
    FEASIBILITY (label-free, fixed now; step 'smoke' measures, before any T3 encode): V-JEPA in bf16 iff the mean token
         cosine of block 20 vs fp32 on real smoke clips is >= 0.98 (the 2026-10-19 lab rule), else fp32. T3 side =
         the benchmark's DINOv3 side (same grid, same mask) iff the projected T3 encode of the three design benchmarks
         (pool + eval clips, measured clips / GPU-s at that side) is <= 15 GPU-h; else CalMS21 (the largest) drops to
         768 px (48 grid, mask recomputed at 48 by the same rule; stated as a deviation) and then 512. T3s (2 extra clips
         per eval frame) is added benchmark by benchmark, cheapest first, while the T3 total stays <= 15 GPU-h.

DESIGN BENCHMARKS (labels used ONLY to score; the user allows CRIM13 for design from 2026-10-25)
    MABe22   SAE pool = the plan's userTrain frames (10,192 train / 2,352 calib); eval = the 18,496 IPW-weighted
             submission frames; behaviours oral_oral, oral_genital, oral_ear, approaches, huddles (eci_common.PRIMARY);
             cross-fit unit = clip, splits = eci_train.halves seeds 0-9.
    CalMS21  pool = the plan's 10,000 train / 2,000 calib frames of task-1 TRAIN videos; eval = the 10,000 IPW frames of
             the 19 test videos; attack, investigation, mount; unit = video, splits = calms21 eci_train.halves 0-9.
    CRIM13   pool = released-train-half 10,000 train / 2,000 calib (+ test-half 2,000 calib = held-out FVE); eval = the
             30,000 IPW frames of both halves; the 11 behaviours of the v2 check; unit = video, split = the released
             halves (both directions).
METRICS (side by side, per behaviour and as the benchmark mean = the PRIMARY value per (variant, seed))
    R~    normalised R-precision (R-prec - pi) / (1 - pi), weighted (IPW) on the benchmarks, of the R~-picked neuron:
          picked on one half, scored on the other, both directions, mean over splits (scripts/benchmark/crim13/
          align_train.align_w2 = the v2 CRIM13-check statistic; unweighted = src/eci/v2/score.py on the lab).
    AP    average precision of the AP-picked neuron (same cross-fit); AP lift = AP / base rate also stored.
    AUROC of the AUROC-picked neuron (reported only).
    VID r video-level agreement: per video, the (IPW-weighted) mean of the neuron's readout value over the scored frames
          vs the ANNOTATED share of time in the behaviour (benchmarks: all labelled frames of the video; lab: the scored
          frames); Pearson r across the videos of one half for every neuron, the best neuron picked on one half and its
          r on the other half (both directions, mean over splits). Note fixed now: per-video means of 'raw' and 'sm'
          differ only by sampling noise, so VID r mostly separates arms by their tokens, not by the readout.
    CAPACITY (behaviour-neuron suite, scripts/eci/v2_behav/behav.py) only if its functions and probe set are usable
          read-only within the budget; otherwise stated as not measured.
DECISION RULE (per variant X vs T0, design benchmarks; frame-level primary R~, 5 vs 5 seeds, Welch 95% CI)
    WIN on a benchmark  : mean difference >= +0.01 and CI lower bound > 0.   LOSS: CI upper bound < 0.
    X is ADOPTED iff WIN on >= 2 of the 3 benchmarks, LOSS on none, and no LOSS of VID r (same Welch rule) on any
    benchmark. Several adopted -> the highest mean primary over the 3 benchmarks; candidates within 0.01 of it ->
    the SIMPLEST (order: T0 < T0u < T1 < T1s < T2 < T2s < T3 < T3s; fewer stages / no second encoder first).
    None adopted -> T0 stays. AP is reported next to R~ for every cell and never overrides the rule.
    VERDICTS: Q1 = "post-hoc smoothing is at least as good as local temporal information" iff none of T1 / T1s / T2 /
    T2s is adopted; a variant that WINS on one benchmark only is reported as such. Q2 = "V-JEPA works better" iff T3 or
    T3s is adopted; T3 LOSS on >= 2 benchmarks = "worse".
LAB CHECK (pre-registered confirmation; labels only score)
    mice v1: the 32,400 atlas-window diagnosis frames (every 20th 5 fps frame), nose_nose / nose_tail (+ nn_mutual /
    np_directional reported); split = the atlas pool split (as the V2-full tables) + rnd0-9 reported. ants: scored
    frames of v2 (13,200) and v3 (32,100), groom_any (+ on_lid_any reported); split 'r4' (+ rnd0-9). Lab PRIMARY: mice =
    mean(nn, nt) R~; ants = mean(v2 groom_any, v3 groom_any) R~. POOL (matched, label-free): mice = every 50th 5 fps
    frame of all 432 videos ((k - fps1_off) % 50 == 7: never a scored frame or a scored frame's neighbour's centre);
    ants v2 + v3 + v4 every 25th ((k % 25) == 7); roles = the series plan's video roles (train / calib / eval -> FVE);
    training tokens capped at 6M per domain by one seeded uniform token subset shared by all arms.
    Variants run on the lab: T0, T0u, T1, T1s, and T2 / T2s iff T2 ran on the benchmarks; T3 / T3s iff T3 was NOT a
    LOSS on >= 2 design benchmarks and the remaining budget covers it (cost estimate stated before launching).
    CONFIRMED (adopted variant X): no LOSS in either lab domain primary and WIN (same rule) in >= 1. If no variant is
    adopted, the lab check is descriptive: per variant vs T0 with Welch CIs per cell, reported with the same words.
BUDGET <= 80 GPU-h for the whole study (encodes logged with GPU utilisation and throughput).
==================================================================================================================

Steps (scripts/eci/v2_temporal/job.sh runs each in Slurm; staging in $LOCAL_DIR under /localhome):
    smoke   (GPU) V-JEPA block-20 truncation identity, bf16 vs fp32 cosine, clips / GPU-s at 512 / 640 / 768 / 1024
            and DINOv3 frames / GPU-s -> OUT/smoke.json
    encode  --bench B --enc d3|vj --shard i --nshard n  (GPU) -> BEE/<B>/<enc>/s<i>/
    lab     --dom mice|ants --shard i --nshard n        (GPU) DINOv3 records of the lab pool + scored frames
    run     --bench B --arms T0:T1 [--seeds]            (GPU) train each (arm, seed) SAE and score it -> OUT/<B>/
    decide  (CPU) tables + rule -> OUT/decision.json, OUT/table.md
Store layout: recs.parquet (one row per RECORD = a centre frame with its kept positions: video, v (source frame or lab
k), kind pool | ev0 | evm | evp, role, eid (eval frame id), n, start), pos.npy (int16), tok.f16: d3 = (n_tok, 3, 1024)
tokens of frames (c - s, c, c + s) at the centre's kept positions; vj = (n_tok, 1024).
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = Path(os.environ.get('TMP_OUT') or REPO / 'results/vision/eci_v2_temporal')
BEE = Path(os.environ.get('TMP_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_temporal')
SEEDS = tuple(range(1000, 1005))
SAE_CFG = dict(W=2048, prefixes=(128, 256, 512, 2048), k=16, steps=30000)
VJ_MODEL = 'facebook/vjepa2-vitl-fpc64-256'
VJ_CACHE = '/mnt/beegfs/locatgrp/shared/rcadei/hf_cache'
VJ_LAYER, CLIP, TPOS = 20, 16, 7
TUB = TPOS // 2
D = 1024
THR, DARK_ABS, DARK_REL = 0.25, 60, 40
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
VARIANTS = {'T0': ('T0', 'sm'), 'T0u': ('T0', 'raw'), 'T1': ('T1', 'raw'), 'T1s': ('T1', 'sm'),
            'T2': ('T2', 'raw'), 'T2s': ('T2', 'sm'), 'T3': ('T3', 'raw'), 'T3s': ('T3', 'sm')}
ORDER = ['T0', 'T0u', 'T1', 'T1s', 'T2', 'T2s', 'T3', 'T3s']
LAB_POOL_EVERY = {'mice': 50, 'ants': 25}
LAB_CAP = 6_000_000


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def local_dir():
    d = Path(os.environ.get('LOCAL_DIR') or '')
    assert str(d).startswith('/localhome') or str(d).startswith(os.environ.get('TMPDIR') or '/nonexistent'), d
    d.mkdir(parents=True, exist_ok=True)
    return d


def gpu_line():
    import torch
    return f'{torch.cuda.get_device_name(0)}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB'


# ================================================================================================ encoders
class D3:
    def __init__(self, dev):
        from src.eci.v2.encode import Encoder
        self.e = Encoder(dev)

    def __call__(self, x):
        return self.e(x)

    @property
    def gpu_s(self):
        return self.e.gpu_s


class VJ:
    """V-JEPA 2-L truncated after block VJ_LAYER (+ the encoder's final LayerNorm), RoPE positions interpolated."""

    def __init__(self, S, dev, bf16=True, layer=VJ_LAYER, interp=True, truncate=True):
        import torch
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(VJ_MODEL, cache_dir=VJ_CACHE)
        cfg.crop_size = S
        m = AutoModel.from_pretrained(VJ_MODEL, config=cfg, cache_dir=VJ_CACHE, attn_implementation='sdpa').to(dev).eval()
        self.G = S // 16
        blocks = m.encoder.layer
        assert len(blocks) == 24
        if interp and self.G != 16:
            sc = 16.0 / self.G
            for blk in blocks:
                att = blk.attention
                orig = att.get_position_ids

                def gp(x, masks=None, _o=orig, _s=sc):
                    f, h, w = _o(x, masks)
                    return f, h.float() * _s, w.float() * _s
                att.get_position_ids = gp
        if truncate:
            m.encoder.layer = torch.nn.ModuleList(list(blocks)[:layer])
        self.m = m.requires_grad_(False)
        self.S, self.dev, self.bf16 = S, dev, bf16
        self.mean = torch.tensor(MEAN, device=dev)[None, None, :, None, None]
        self.std = torch.tensor(STD, device=dev)[None, None, :, None, None]
        self.gpu_s, self.n_clips = 0.0, 0

    def __call__(self, x):
        """x (B, 16, 3, S, S) uint8 cuda -> (B, G*G, 1024) fp16 tokens of tubelet TUB."""
        import torch
        t0 = time.time()
        with torch.inference_mode():
            xin = (x.float() / 255 - self.mean) / self.std
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
                h = self.m(pixel_values_videos=xin, skip_predictor=True).last_hidden_state
            GG = self.G * self.G
            assert h.shape[1] == (CLIP // 2) * GG, h.shape
            z = h[:, TUB * GG:(TUB + 1) * GG].float()
            assert torch.isfinite(z).all()
            z = z.half()
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        self.n_clips += x.shape[0]
        return z


def grey_t(x):
    from src.eci.v2.mask import grey_of
    return grey_of(x)


# ================================================================================================ benchmarks
class Bench:
    """Adapter: frames, canvas, background, labels, splits of one design benchmark."""
    name, S, step, workers = None, None, None, 4

    def bg_index(self, N):
        raise NotImplementedError

    def bg_reduce(self, g):
        raise NotImplementedError

    def cells(self, q):
        return None

    def records(self, enc):
        """-> DataFrame video, v, kind, role, eid (eval rows: ev0 / evm / evp centres; pool rows)."""
        E, P = self.eval_table(), self.pool_table()
        parts = [pd.DataFrame({'video': P.video.values, 'v': P.v.values, 'kind': 'pool', 'role': P.role.values,
                               'eid': -1})]
        kinds = (('ev0', 0), ('evm', -1), ('evp', 1))
        if enc == 'vj':
            kinds = tuple(k for k in kinds if k[0] in self.vj_kinds)
        for kd, o in kinds:
            parts.append(pd.DataFrame({'video': E.video.values, 'v': E.v.values + o * self.step, 'kind': kd,
                                       'role': 'eval', 'eid': np.arange(len(E))}))
        return pd.concat(parts, ignore_index=True)

    vj_kinds = ('ev0',)


class MABe22(Bench):
    name, S, step, workers = 'mabe22', 512, 6, 6

    def __init__(self):
        sys.path.insert(0, str(REPO / 'scripts/benchmark/mabe22'))
        import eci_common as EC
        self.EC = EC
        self.off = EC.offset()
        self.labels = list(EC.PRIMARY)

    def eval_table(self):
        E = pd.read_parquet(self.EC.PLAN / 'eval_frames.parquet').reset_index(drop=True)
        return E.assign(video=E.clip, v=E.i + self.off)

    def pool_table(self):
        P = pd.read_parquet(self.EC.PLAN / 'train_frames.parquet').reset_index(drop=True)
        return P.assign(video=P.clip, v=P.i + self.off)

    def splits(self, E):
        import importlib.util
        sp = importlib.util.spec_from_file_location('mabe_train', REPO / 'scripts/benchmark/mabe22/eci_train.py')
        M = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(M)
        return {f's{s}': M.halves(E, s) for s in range(10)}

    def truth(self, E):
        sub = np.load(self.EC.RAW / 'mouse_submission_labels.npy', allow_pickle=True).item()
        fm = np.load(self.EC.RAW / 'mouse_frame_number_map.npy', allow_pickle=True).item()
        voc = list(sub['vocabulary'])
        L = sub['label_array']
        rows = [voc.index(b) for b in self.labels]
        out = {}
        for c in E.video.unique():
            a, b = fm[c]
            out[c] = [float(np.nanmean(L[r, a:b] == 1)) for r in rows]
        return pd.DataFrame.from_dict(out, orient='index', columns=self.labels)

    def video_set(self, q, recs):
        return 'mouse_submission' if (recs.kind != 'pool').any() else 'mouse_userTrain'

    def decode(self, q, want, recs):
        fr = self.EC.decode_clip(self.video_set(q, recs), q, local_dir())
        return {i: fr[i] for i in range(len(fr))}, len(fr)

    def canvas(self, frames, dev):
        import torch
        import torch.nn.functional as F
        x = torch.from_numpy(np.ascontiguousarray(np.stack(frames))).to(dev).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(self.S, self.S), mode='bicubic', align_corners=False).round().clamp(0, 255)
        return x.to(torch.uint8)

    def bg_index(self, N):
        return np.linspace(0, N - 1, 30).round().astype(int)

    def bg_reduce(self, g):
        return g.float().kthvalue(29, dim=0).values


class CalMS21(Bench):
    name, S, step, workers = 'calms21', 1024, 6, 4

    def __init__(self):
        sys.path.insert(0, str(REPO / 'scripts/benchmark/calms21'))
        import calms_common as CC
        self.CC = CC
        self.off = CC.offset()
        self.labels = ['attack', 'investigation', 'mount']
        self.lens = {}
        for sp in ('train', 'test'):
            d = np.load(CC.DER / f'task1_{sp}.npz', allow_pickle=True)
            for nm, ln in zip(d['names'], d['lens']):
                self.lens[str(nm)] = (sp, int(ln))

    def eval_table(self):
        E = pd.read_parquet(self.CC.PLAN / 'eval_frames.parquet').reset_index(drop=True)
        return E.assign(v=E.i + self.off)

    def pool_table(self):
        P = pd.read_parquet(self.CC.PLAN / 'train_frames.parquet').reset_index(drop=True)
        return P.assign(v=P.i + self.off)

    def splits(self, E):
        import importlib.util
        sp = importlib.util.spec_from_file_location('calms_train', REPO / 'scripts/benchmark/calms21/eci_train.py')
        M = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(M)
        return {f's{s}': M.halves(E, s, self.labels) for s in range(10)}

    def truth(self, E):
        d = np.load(self.CC.DER / 'task1_test.npz', allow_pickle=True)
        st = np.r_[0, np.cumsum(d['lens'])]
        out = {}
        for j, nm in enumerate(d['names']):
            lab = d['labels'][st[j]:st[j + 1]]
            out[str(nm)] = [float(np.mean(lab == self.CC.TASK1.index(b))) for b in self.labels]
        T = pd.DataFrame.from_dict(out, orient='index', columns=self.labels)
        return T.loc[sorted(E.video.unique())]

    def decode(self, q, want, recs):
        sp, n = self.lens[q]
        src = self.CC.video_path(sp, q)
        loc = local_dir() / f'_{q}_{os.getpid()}.mp4'
        shutil.copyfile(src, loc)
        try:
            fr, nd = self.CC.decode_frames(loc, want, threads=3)
        finally:
            loc.unlink()
        return fr, n

    def canvas(self, frames, dev):
        import torch
        x = torch.from_numpy(np.ascontiguousarray(np.stack(frames))).to(dev).permute(0, 3, 1, 2)
        c = torch.empty((len(frames), 3, self.S, self.S), dtype=torch.uint8, device=dev)
        c[:] = torch.tensor(self.CC.PADV, dtype=torch.uint8, device=dev)[None, :, None, None]
        c[:, :, self.CC.PADY:self.CC.PADY + x.shape[2]] = x
        return c

    def bg_index(self, N):
        return np.linspace(0, N - 1, 32).round().astype(int)

    def bg_reduce(self, g):
        return g.float().median(0).values


class CRIM13(Bench):
    name, S, step, workers = 'crim13', 640, 5, 6

    def __init__(self):
        sys.path.insert(0, str(REPO / 'scripts/benchmark/crim13'))
        import align_common as AC
        self.AC = AC
        self.C = json.loads((REPO / 'configs/eci_v2/crim13.json').read_text())
        self.P = pd.read_parquet(REPO / self.C['plan']).reset_index(drop=True)
        self.V = pd.read_parquet(REPO / self.C['plan'].replace('frames.parquet', 'videos.parquet')).set_index('video')
        self.labels = list(self.C['score']['labels'])
        inv = json.loads(AC.INV.read_text())
        self.dir = {r['name']: r['dir'] for r in inv['videos']}

    def eval_table(self):
        return self.P[self.P.role == 'eval'].reset_index(drop=True)

    def pool_table(self):
        P = self.P
        a = P[(P.half == 'train') & P.role.isin(['train', 'calib'])]
        b = P[(P.half == 'test') & (P.role == 'calib')].assign(role='fve')
        return pd.concat([a, b], ignore_index=True)

    def splits(self, E):
        return {'released': (E.half.values == 'test').astype(np.int64)}

    def truth(self, E):
        out = {}
        for q in sorted(E.video.unique()):
            n = int(self.V.loc[q, 'frames'])
            lab = self.AC.read_labels(self.AC.ROOT / self.dir[q] / f'{q}_t.txt', n)
            out[q] = [float(np.mean(lab == self.AC.VOCAB.index(b))) for b in self.labels]
        return pd.DataFrame.from_dict(out, orient='index', columns=self.labels)

    def decode(self, q, want, recs):
        n = int(self.V.loc[q, 'frames'])
        d = self.AC.ROOT / self.V.loc[q, 'half']
        offs = self.AC.load_seek(d / f'{q}_t-seek.mat')
        assert len(offs) == n
        fr, _ = self.AC.read_frames(d / f'{q}_t.seq', offs, want, threads=3)
        return fr, n

    def canvas(self, frames, dev):
        import torch
        q = self._cur
        W, H = int(self.V.loc[q, 'W']), int(self.V.loc[q, 'H'])
        return self.AC.canvas_gpu(frames, W, H, dev).to(torch.uint8)

    def cells(self, q):
        W, H = int(self.V.loc[q, 'W']), int(self.V.loc[q, 'H'])
        return self.AC.video_cells(W, H)

    def bg_index(self, N):
        return np.linspace(0, N - 1, 32).round().astype(int)

    def bg_reduce(self, g):
        return g.float().median(0).values


BENCHES = {'mabe22': MABe22, 'calms21': CalMS21, 'crim13': CRIM13}


def shard_videos(R, n, i):
    c = R.groupby('video').size().reset_index(name='n').sort_values(['n', 'video'], ascending=[False, True],
                                                                    kind='stable')
    return [v for j, v in enumerate(c.video) if j % n == i]


def fraction(x, bg, cells=None):
    """x (B, 3, S, S) uint8 canvas, bg (S, S) float -> (B, G*G) animal-pixel fraction per 16 px patch."""
    import torch.nn.functional as F
    g = grey_t(x).float()
    px = (g < DARK_ABS) & ((bg[None] - g) > DARK_REL)
    fd = F.avg_pool2d(px.float()[:, None], 16).flatten(1)
    if cells is not None:
        fd = fd * cells[None]
    return fd


class Writer:
    def __init__(self, loc, shape_tail):
        self.loc = loc
        loc.mkdir(parents=True, exist_ok=True)
        self.f = open(loc / 'tok.f16', 'wb')
        self.pos, self.recs, self.n, self.tail = [], [], 0, shape_tail

    def add(self, rec, pos, tok):
        assert tok.shape[1:] == self.tail and len(tok) == len(pos), (tok.shape, self.tail)
        self.f.write(np.ascontiguousarray(tok.astype(np.float16)).tobytes())
        self.pos.append(pos.astype(np.int16))
        self.recs.append({**rec, 'n': int(len(pos)), 'start': self.n})
        self.n += len(pos)

    def close(self, info):
        self.f.close()
        np.save(self.loc / 'pos.npy', np.concatenate(self.pos) if self.pos else np.zeros(0, np.int16))
        pd.DataFrame(self.recs).to_parquet(self.loc / 'recs.parquet')
        (self.loc / 'info.json').write_text(json.dumps(info, indent=1, default=float))


def atomic_copy(src, dst):
    dst = Path(dst)
    tmp = dst.with_name(dst.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)


def cmd_encode(a):
    import torch
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    B = BENCHES[a.bench]()
    if a.enc == 'vj':
        fz = json.loads((OUT / 'feasibility.json').read_text())
        S = int(fz['side'][a.bench])
        B.vj_kinds = tuple(fz['kinds'][a.bench])
        bf16 = bool(fz['bf16'])
    else:
        S = B.S
    G = S // 16
    R = B.records(a.enc)
    vids = shard_videos(R, a.nshard, a.shard)
    if a.max_videos:
        vids = vids[:a.max_videos]
    tag = f's{a.shard}' + ('_selftest' if a.selftest else '')
    dst = BEE / a.bench / a.enc / tag
    if (dst / 'recs.parquet').exists() and not a.force:
        log(f'[SKIP] {dst}')
        return
    enc = D3(dev) if a.enc == 'd3' else VJ(S, dev, bf16=bf16)
    W = Writer(local_dir() / f'{a.bench}_{a.enc}_{tag}', (3, D) if a.enc == 'd3' else (D,))
    st = B.step
    bs = {512: 64, 640: 32, 768: 24, 1024: 16}[S] if a.enc == 'd3' else {512: 4, 640: 2, 768: 2, 1024: 1}[S]
    log(f'{a.bench} {a.enc} {tag}: {len(vids)} videos, {int(R.video.isin(vids).sum())} records, S {S}, bs {bs}, '
        f'{torch.cuda.get_device_name(0)}')

    def need_of(q):
        Rv = R[R.video == q]
        return Rv

    def dec(q):
        t_ = time.time()
        Rv = need_of(q)
        want = set()
        N = {'mabe22': 1801}.get(B.name) or (B.lens[q][1] if B.name == 'calms21' else int(B.V.loc[q, 'frames']))
        c = np.clip(Rv.v.values, 0, N - 1)
        if a.enc == 'd3':
            for o in (-1, 0, 1):
                want |= set(np.clip(c + o * st, 0, N - 1).tolist())
        else:
            for j in range(CLIP):
                want |= set(np.clip(c + st * (j - TPOS), 0, N - 1).tolist())
        want |= set(B.bg_index(N).tolist())
        fr, N2 = B.decode(q, want, Rv)
        if B.name == 'mabe22':
            assert N2 == 1801, (q, N2)
        missing = [w for w in want if w not in fr]
        assert not missing, (q, len(missing), missing[:5])
        return Rv, fr, N, time.time() - t_

    pool = ThreadPoolExecutor(B.workers)
    fut = {j: pool.submit(dec, vids[j]) for j in range(min(len(vids), B.workers + 1))}
    t00, tdec, twait, nkept = time.time(), 0.0, 0.0, []
    for k, q in enumerate(vids):
        t0 = time.time()
        Rv, fr, N, dt = fut.pop(k).result()
        twait += time.time() - t0
        tdec += dt
        if k + B.workers + 1 < len(vids):
            fut[k + B.workers + 1] = pool.submit(dec, vids[k + B.workers + 1])
        B._cur = q
        cl = B.cells(q)
        cells = None
        if cl is not None:
            cells = torch.from_numpy(cl.astype(np.float32)).to(dev)
            if G != B.S // 16:
                cells = torch.nn.functional.interpolate(cells.view(1, 1, B.S // 16, B.S // 16), size=(G, G),
                                                        mode='nearest').flatten()
        with torch.no_grad():
            bgi = B.bg_index(N)
            bg = B.bg_reduce(torch.cat([grey_t(canvas_at(B, [fr[i] for i in bgi[b0:b0 + 16]], dev, S))
                                        for b0 in range(0, len(bgi), 16)]))
            cs = np.clip(Rv.v.values, 0, N - 1)
            uc = np.unique(cs)
            K = {}
            for b0 in range(0, len(uc), 64):
                ub = uc[b0:b0 + 64]
                fd = fraction(canvas_at(B, [fr[i] for i in ub], dev, S), bg, cells)
                keep = (fd >= THR).cpu().numpy()
                for i, c in enumerate(ub):
                    K[int(c)] = np.flatnonzero(keep[i]).astype(np.int64)
            if a.enc == 'd3':
                U = {}
                for c in uc:
                    for o in (-1, 0, 1):
                        f = int(np.clip(c + o * st, 0, N - 1))
                        U.setdefault(f, []).append(K[int(c)])
                U = {f: np.unique(np.concatenate(v)) for f, v in U.items()}
                fl = sorted(U)
                Z = {}
                for b0 in range(0, len(fl), bs):
                    fb = fl[b0:b0 + bs]
                    z = enc(canvas_at(B, [fr[i] for i in fb], dev, S))
                    for i, f in enumerate(fb):
                        Z[f] = z[i, torch.from_numpy(U[f]).to(dev)].cpu().numpy()
                for r in Rv.itertuples():
                    c = int(np.clip(r.v, 0, N - 1))
                    kk = K[c]
                    tk = np.stack([Z[f][np.searchsorted(U[f], kk)] for f in
                                   (int(np.clip(c + o * st, 0, N - 1)) for o in (-1, 0, 1))], 1)
                    W.add({'video': q, 'v': int(r.v), 'kind': r.kind, 'role': r.role, 'eid': int(r.eid)}, kk, tk)
                    nkept.append(len(kk))
            else:
                rl = list(Rv.itertuples())
                for b0 in range(0, len(rl), bs):
                    rb = rl[b0:b0 + bs]
                    idx = np.stack([np.clip(int(np.clip(r.v, 0, N - 1)) + st * (np.arange(CLIP) - TPOS), 0, N - 1)
                                    for r in rb])
                    x = canvas_at(B, [fr[i] for i in idx.ravel()], dev, S).view(len(rb), CLIP, 3, S, S)
                    z = enc(x)
                    for i, r in enumerate(rb):
                        kk = K[int(np.clip(r.v, 0, N - 1))]
                        W.add({'video': q, 'v': int(r.v), 'kind': r.kind, 'role': r.role, 'eid': int(r.eid)}, kk,
                              z[i, torch.from_numpy(kk).to(dev)].cpu().numpy())
                        nkept.append(len(kk))
        del fr
        if k % 10 == 0 or k == len(vids) - 1:
            el = time.time() - t00
            log(f'{k + 1}/{len(vids)} videos, {len(W.recs)} records, {el:.0f}s wall, GPU busy {enc.gpu_s:.0f}s, '
                f'decode wait {twait:.0f}s; {gpu_line()}')
    el = time.time() - t00
    info = {'bench': a.bench, 'enc': a.enc, 'S': S, 'G': G, 'shard': a.shard, 'nshard': a.nshard, 'n_videos': len(vids),
            'n_records': len(W.recs), 'n_tokens': W.n, 'kept_mean': float(np.mean(nkept)) if nkept else 0,
            'records_zero_kept': int(np.sum(np.array(nkept) == 0)), 'wall_s': el, 'gpu_busy_s': enc.gpu_s,
            'decode_thread_s': tdec, 'decode_wait_s': twait, 'records_per_wall_s': len(W.recs) / max(el, 1),
            'gpu': torch.cuda.get_device_name(0), 'bf16': getattr(enc, 'bf16', True)}
    W.close(info)
    log(json.dumps(info))
    od = OUT / 'encode'
    od.mkdir(parents=True, exist_ok=True)
    (od / f'{a.bench}_{a.enc}_{tag}.json').write_text(json.dumps(info, indent=1, default=float))
    atomic_copy(W.loc, dst)


def canvas_at(B, frames, dev, S):
    """benchmark canvas at the encoder side S (the arm's side; resized when the T3 side differs from the DINOv3 side)."""
    import torch
    import torch.nn.functional as F
    x = B.canvas(frames, dev)
    if x.shape[-1] != S:
        x = F.interpolate(x.float(), size=(S, S), mode='bicubic', align_corners=False,
                          antialias=S < x.shape[-1]).round().clamp(0, 255).to(torch.uint8)
    return x


# ================================================================================================ lab
def lab_tables(dom):
    """-> (scored DataFrame series, obs, k, row, eid (+ labels / splits are built at scoring), pool centres)."""
    for p in ('scripts/eci', 'scripts/eci/nnnt', 'scripts/eci/v2', 'scripts/eci/v2_mask', 'scripts/eci/v2_mask2'):
        if str(REPO / p) not in sys.path:
            sys.path.insert(0, str(REPO / p))
    import mask_test as MT
    if dom == 'mice':
        F = MT.frame_list('mice', 'diag')
        keep = ~F.obs.str.endswith('_H').values | (F.k.values >= 4500)
        Sc = F[keep].assign(series='v1').reset_index(drop=True)
        sers = ('v1',)
    else:
        Sc = pd.concat([MT.frame_list('ants', f'scored_{s}').assign(series=s) for s in ('v2', 'v3')], ignore_index=True)
        sers = ('v2', 'v3', 'v4')
    Sc = Sc[['series', 'obs', 'k', 'row']].copy()
    Sc['eid'] = np.arange(len(Sc))
    return Sc, sers, MT


def cmd_lab(a):
    import torch
    from src.eci.v2.pipeline import Series
    from src.eci.v2.video import Prefetch
    dev = torch.device('cuda')
    dom = a.dom
    Sc, sers, MT = lab_tables(dom)
    enc = D3(dev)
    SE = {s: Series(MT.series_cfg(dom, s), dev, encoder=enc.e) for s in sers}
    off = {}
    if dom == 'mice':
        sys.path.insert(0, str(REPO / 'scripts/eci/nnnt'))
        import nnnt as N
        off = {v['obs']: v['fps1_off'] for v in N.load_plan()['videos']}
    ev = LAB_POOL_EVERY[dom]
    runs = []  # (series, video dict, k0, n, [(record dict, centre k)])
    for s, se in SE.items():
        scv = Sc[Sc.series == s]
        for v in se.videos:
            n = v['n']
            o = off.get(v['obs'], 0)
            for c in np.arange((o + 7) % ev, n, ev):
                k0, k1 = max(0, c - 1), min(n - 1, c + 1)
                role = {'train': 'train', 'calib': 'calib'}.get(v['role'], 'fve')
                runs.append((s, v, int(k0), int(k1 - k0 + 1), [({'kind': 'pool', 'role': role, 'eid': -1}, int(c))]))
            for r in scv[scv.obs == v['obs']].itertuples():
                k = int(r.k)
                k0, k1 = max(0, k - 2), min(n - 1, k + 2)
                recs = [({'kind': kd, 'role': 'eval', 'eid': int(r.eid)}, int(np.clip(k + o_, 0, n - 1)))
                        for kd, o_ in (('ev0', 0), ('evm', -1), ('evp', 1))]
                runs.append((s, v, int(k0), int(k1 - k0 + 1), recs))
    # shard by video (whole videos per shard), balanced by run count
    vids = pd.Series([f'{r[0]}|{r[1]["obs"]}' for r in runs])
    cnt = vids.value_counts().sort_values(ascending=False, kind='stable')
    mine = set(cnt.index[a.shard::a.nshard])
    runs = [r for r, vk in zip(runs, vids) if vk in mine]
    if a.selftest:
        runs = runs[:60]
    tag = f's{a.shard}' + ('_selftest' if a.selftest else '')
    dst = BEE / dom / 'd3' / tag
    if (dst / 'recs.parquet').exists() and not a.force:
        log(f'[SKIP] {dst}')
        return
    W = Writer(local_dir() / f'{dom}_{tag}', (3, D))
    jobs = [((j,), SE[r[0]].job(r[1], r[2], r[3], 1)) for j, r in enumerate(runs)]
    log(f'lab {dom} {tag}: {len(runs)} runs over {len(mine)} videos, {torch.cuda.get_device_name(0)}')
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=Series.run_job)
    t00, nk = time.time(), []
    pend = []  # batch several runs per encoder call

    def flush():
        if not pend:
            return
        xs, fds, meta = [], [], []
        for j, fr in pend:
            s, v, k0, n, recs = runs[j]
            ks = np.arange(k0, k0 + n)
            assert len(fr) == n, (v['obs'], len(fr), n)
            x, fd = SE[s].frames(v, fr, ks)
            xs.append(x)
            fds.append(fd)
            meta.append((j, len(ks)))
        x, fd = torch.cat(xs), torch.cat(fds)
        z, keep = SE[runs[pend[0][0]][0]].tokens(x, fd)
        keep = keep.cpu().numpy()
        o = 0
        for j, n in meta:
            s, v, k0, n_, recs = runs[j]
            for rec, c in recs:
                ic = c - k0
                kk = np.flatnonzero(keep[o + ic]).astype(np.int64)
                ii = [o + max(ic - 1, 0), o + ic, o + min(ic + 1, n - 1)]
                tk = z[ii][:, torch.from_numpy(kk).to(dev)].permute(1, 0, 2).cpu().numpy()
                W.add({'series': s, 'video': v['obs'], 'v': int(c), 'k': int(c), 'row': int(v['src_lo'] + c), **rec},
                      kk, tk)
                nk.append(len(kk))
            o += n
        pend.clear()

    nfr = 0
    for (j,), fr in pf:
        if pend and runs[pend[0][0]][0] != runs[j][0]:
            flush()
        pend.append((j, fr))
        nfr += len(fr)
        if sum(len(f) for _, f in pend) >= 15:
            flush()
        if j % 2000 == 0:
            el = time.time() - t00
            log(f'run {j}/{len(runs)}, {len(W.recs)} records, {nfr} frames, {el:.0f}s ({nfr / max(el, 1):.1f} frames/s); '
                f'GPU busy {enc.gpu_s:.0f}s; decode thread-s {pf.dec_s:.0f}; {gpu_line()}')
    flush()
    el = time.time() - t00
    info = {'dom': dom, 'shard': a.shard, 'nshard': a.nshard, 'n_runs': len(runs), 'n_records': len(W.recs),
            'n_frames_encoded': nfr, 'n_tokens': W.n, 'kept_mean': float(np.mean(nk)), 'wall_s': el,
            'gpu_busy_s': enc.gpu_s, 'decode_thread_s': pf.dec_s, 'frames_per_wall_s': nfr / max(el, 1),
            'gpu': torch.cuda.get_device_name(0)}
    W.close(info)
    log(json.dumps(info))
    od = OUT / 'encode'
    od.mkdir(parents=True, exist_ok=True)
    (od / f'lab_{dom}_{tag}.json').write_text(json.dumps(info, indent=1, default=float))
    atomic_copy(W.loc, dst)


# ================================================================================================ store / arms
class Store:
    """All shards of BEE/<bench>/<enc>/s* as one record table + token memmaps."""

    def __init__(self, bench, enc):
        root = BEE / bench / enc
        sh = sorted([p for p in root.glob('s*') if p.is_dir() and not p.name.endswith('.tmp')
                     and 'selftest' not in p.name], key=lambda p: int(p.name[1:]))
        assert sh, root
        Rs, toks, pos, o = [], [], [], 0
        tail = (3, D) if enc == 'd3' else (D,)
        for p in sh:
            R = pd.read_parquet(p / 'recs.parquet')
            nt = int(R.n.sum())
            R['start'] = R['start'] + o
            R['shard'] = p.name
            toks.append(np.memmap(p / 'tok.f16', np.float16, 'r', shape=(nt,) + tail))
            pos.append(np.load(p / 'pos.npy'))
            Rs.append(R)
            o += nt
        self.R = pd.concat(Rs, ignore_index=True)
        self.toks, self.tail, self.enc = toks, tail, enc
        self.shard_off = np.r_[0, np.cumsum([len(t) for t in toks])]
        self.info = [json.loads((p / 'info.json').read_text()) for p in sh]

    def gather(self, rec_idx, arm, keep_p=None, seed=0):
        """records -> (X (n_tok, d) fp16 numpy, frame index (n_tok,) into rec_idx order)."""
        Rr = self.R.iloc[rec_idx]
        xs, fl = [], []
        g = np.random.default_rng(seed) if keep_p else None
        for j, (st, n) in enumerate(zip(Rr.start.values, Rr.n.values)):
            if n == 0:
                continue
            si = int(np.searchsorted(self.shard_off, st, side='right') - 1)
            a = st - self.shard_off[si]
            t = np.asarray(self.toks[si][a:a + n])
            if keep_p is not None and keep_p < 1:
                m = g.random(n) < keep_p
                t = t[m]
                if not len(t):
                    continue
            xs.append(arm_tokens(t, arm))
            fl.append(np.full(len(t), j, np.int64))
        if not xs:
            return np.zeros((0, D), np.float16), np.zeros(0, np.int64)
        return np.concatenate(xs), np.concatenate(fl)


def arm_tokens(t, arm):
    if arm == 'T3':
        return t
    if arm == 'T0':
        return t[:, 1]
    if arm == 'T1':
        return ((t[:, 0].astype(np.float32) + t[:, 1] + t[:, 2]) / 3).astype(np.float16)
    if arm == 'T2':
        return np.concatenate([t[:, 1], ((t[:, 2].astype(np.float32) - t[:, 0]) / 2).astype(np.float16)], 1)
    raise KeyError(arm)


# ================================================================================================ scoring
def weighted_stats(X, Y, w, rows):
    """-> dict R (W, L) normalised R-precision, AP, A (AUROC), base (L,), rate (W,)."""
    for p in ('scripts/benchmark/crim13', 'scripts/eci', 'scripts/benchmark/mabe22'):
        if str(REPO / p) not in sys.path:
            sys.path.insert(0, str(REPO / p))
    import align_train as AT
    S = AT.align_w2(X, Y, w, rows)
    base = np.asarray(S['base'], np.float64)
    return {'R': (S['rprec'] - base[None]) / (1 - base[None]), 'AP': S['ap'], 'A': S['auc'], 'base': base,
            'rate': S['rate']}


def cf_frame(X, Y, w, halves, labels):
    """cross-fitted frame-level metrics: {label: {R, AP, AP_lift, A, R_neurons}} per split + mean over splits."""
    import torch
    dev = X.device
    out = {}
    for sn, h in halves.items():
        H = {k: weighted_stats(X, Y, w, torch.from_numpy(np.flatnonzero(h == k)).to(dev)) for k in (0, 1)}
        rec = {}
        for li, lb in enumerate(labels):
            d = []
            for s, t in ((0, 1), (1, 0)):
                e = {}
                for key in ('R', 'AP', 'A'):
                    sc = np.where(np.isfinite(H[s][key][:, li]), H[s][key][:, li], -np.inf)
                    j = int(np.argmax(sc))
                    e[key] = float(H[t][key][j, li])
                    e[key + '_neuron'] = j
                e['AP_lift'] = e['AP'] / max(H[t]['base'][li], 1e-12)
                d.append(e)
            rec[lb] = {k: float(np.mean([x[k] for x in d])) for k in ('R', 'AP', 'AP_lift', 'A')}
            rec[lb]['R_neurons'] = [x['R_neuron'] for x in d]
        out[sn] = rec
    return out


def cf_video(X, w, vid, truth, halves, labels):
    """video-level: per video weighted mean of each neuron vs the annotated share; neuron picked by Pearson r on one
    half of the videos, r on the other. X (n, W) float cuda, vid (n,) int index into truth rows (nV, L)."""
    import torch
    dev = X.device
    nV = truth.shape[0]
    wv = torch.zeros(nV, dtype=torch.float64, device=dev).index_add_(0, vid, w)
    M = torch.zeros((nV, X.shape[1]), dtype=torch.float64, device=dev).index_add_(0, vid, X.double() * w[:, None])
    M = M / wv[:, None].clamp_min(1e-12)
    T = torch.from_numpy(truth).to(dev).double()
    out = {}
    for sn, h in halves.items():
        vh = np.full(nV, -1)
        vh[vid.cpu().numpy()] = h
        rs = {}
        for k in (0, 1):
            ix = torch.from_numpy(np.flatnonzero(vh == k)).to(dev)
            Mk, Tk = M[ix], T[ix]
            Mc, Tc = Mk - Mk.mean(0), Tk - Tk.mean(0)
            r = (Mc.T @ Tc) / (Mc.norm(dim=0)[:, None] * Tc.norm(dim=0)[None]).clamp_min(1e-12)
            r[(Mc.norm(dim=0) == 0)] = float('nan')
            rs[k] = r.cpu().numpy()
        rec = {}
        for li, lb in enumerate(labels):
            d = []
            for s, t in ((0, 1), (1, 0)):
                sc = np.where(np.isfinite(rs[s][:, li]), rs[s][:, li], -np.inf)
                j = int(np.argmax(sc))
                d.append(float(np.nan_to_num(rs[t][j, li])))
            rec[lb] = {'r': float(np.mean(d))}
        rec['_n_videos'] = [int((vh == 0).sum()), int((vh == 1).sum())]
        out[sn] = rec
    return out


def eval_codes(St, sae, norm, arm, dev, kinds):
    """frame codes (theta*, max over kept tokens) of every eval record of the store -> {kind: (n_eval, W) float32 cuda}."""
    import torch
    from src.eci.v2.sae import frame_codes
    R = St.R
    E = R[R.kind != 'pool']
    neval = int(E.eid.max()) + 1
    out = {}
    for kd in kinds:
        ix = np.flatnonzero((R.kind == kd).values)
        eid = R.eid.values[ix]
        assert len(np.unique(eid)) == len(eid) == neval, (kd, len(eid), neval)
        C = torch.zeros((neval, sae.n_latents), dtype=torch.float32, device=dev)
        for b0 in range(0, len(ix), 4000):
            bi = ix[b0:b0 + 4000]
            x, fl = St.gather(bi, arm)
            cm, _, _ = frame_codes(sae, norm, torch.from_numpy(x).to(dev), torch.from_numpy(fl).to(dev), len(bi))
            C[torch.from_numpy(eid[b0:b0 + 4000]).to(dev)] = cm
        out[kd] = C
    return out


class EvalSpec:
    """labels / weights / splits / video truth of one scoring domain (benchmark or lab cell)."""

    def __init__(self, Y, w, halves, labels, vid, truth, primary_split=None, prim=None):
        self.Y, self.w, self.halves, self.labels = Y, w, halves, labels
        self.vid, self.truth, self.primary_split, self.prim = vid, truth, primary_split, prim


def bench_spec(B):
    E = B.eval_table()
    vids = sorted(E.video.unique())
    T = B.truth(E).loc[vids]
    return EvalSpec(E[B.labels].values.astype(bool), E.w.values.astype(np.float64), B.splits(E), B.labels,
                    pd.Series(range(len(vids)), index=vids).loc[E.video].values, T.values.astype(np.float64))


def lab_specs(dom, Sc, MT):
    """-> {cell name: (eid index array into Sc, EvalSpec)}."""
    from types import SimpleNamespace
    sys.path.insert(0, str(REPO / 'scripts/eci/v2_mask2'))
    import release_ab as RA
    out = {}
    if dom == 'mice':
        Fall = MT.frame_list('mice', 'diag')
        St = SimpleNamespace(F=Fall, nf=len(Fall))
        Y, sets, half, _, _ = MT.mice_sets(St)
        keep, h = RA.mice_window(St, half['all'])
        Fw = Fall.iloc[keep].reset_index(drop=True)
        assert (Fw.obs.values == Sc.obs.values).all() and (Fw.k.values == Sc.k.values).all()
        labs = list(MT.FZ.MLAB)
        Yw = Y[keep][:, :len(labs)].astype(bool)
        obs = sorted(Fw.obs.unique())
        vid = pd.Series(range(len(obs)), index=obs).loc[Fw.obs].values
        T = np.stack([Yw[vid == i].mean(0) for i in range(len(obs))])
        out['mice'] = (np.arange(len(Sc)), EvalSpec(Yw, np.ones(len(Fw)), h, labs, vid, T, 'atlas',
                                                        ['nose_nose', 'nose_tail']))
    else:
        for ser in ('v2', 'v3'):
            ix = np.flatnonzero(Sc.series.values == ser)
            F = Sc.iloc[ix].reset_index(drop=True)
            Fl = MT.frame_list('ants', f'scored_{ser}')
            assert (Fl.obs.values == F.obs.values).all() and (Fl.k.values == F.k.values).all()
            St = SimpleNamespace(F=Fl, nf=len(Fl))
            Y, hh, labs = MT.ants_set(St, ser)
            Y = Y.astype(bool)
            obs = sorted(F.obs.unique())
            vid = pd.Series(range(len(obs)), index=obs).loc[F.obs].values
            T = np.stack([Y[vid == i].mean(0) for i in range(len(obs))])
            out[f'ants_{ser}'] = (ix, EvalSpec(Y, np.ones(len(F)), hh, list(labs), vid, T, 'r4', ['groom_any']))
    return out


def score_spec(C, spec, dev):
    """C {readout: (n, W) float32 cuda} -> {readout: {'frame': ..., 'video': ...}}."""
    import torch
    Y = torch.from_numpy(spec.Y).to(dev)
    w = torch.from_numpy(spec.w).to(dev)
    vid = torch.from_numpy(np.asarray(spec.vid, np.int64)).to(dev)
    res = {}
    for ro, X in C.items():
        Xh = X.clamp(0, 65000).half()
        res[ro] = {'frame': cf_frame(Xh, Y, w, spec.halves, spec.labels),
                   'video': cf_video(X, w, vid, spec.truth, spec.halves, spec.labels)}
    return res


def summarise(res, labels, primary_split=None, prim_labels=None):
    """mean over splits (or the named primary split) -> per label R / AP / AP_lift / A / vid r and primaries."""
    out = {}
    for ro, r in res.items():
        sp = [primary_split] if primary_split else list(r['frame'])
        pl = prim_labels or labels
        m = {}
        for lb in labels:
            m[lb] = {k: float(np.mean([r['frame'][s][lb][k] for s in sp])) for k in ('R', 'AP', 'AP_lift', 'A')}
            m[lb]['vid_r'] = float(np.mean([r['video'][s][lb]['r'] for s in sp]))
            m[lb]['R_allsplits'] = float(np.mean([r['frame'][s][lb]['R'] for s in r['frame']]))
        m['_primary'] = {k: float(np.mean([m[lb][k] for lb in pl])) for k in ('R', 'AP', 'AP_lift', 'A', 'vid_r',
                                                                                 'R_allsplits')}
        out[ro] = m
    return out


def cmd_run(a):
    import torch
    from src.eci.v2 import sae as SAE
    dev = torch.device('cuda')
    seeds = [int(s) for s in a.seeds.replace(':', ',').split(',')] if a.seeds else list(SEEDS)
    lab = a.bench in ('mice', 'ants')
    od = OUT / a.bench
    (od / 'scores').mkdir(parents=True, exist_ok=True)
    if lab:
        Sc, _, MT = lab_tables(a.bench)
        specs = lab_specs(a.bench, Sc, MT)
    else:
        B = BENCHES[a.bench]()
        spec = bench_spec(B)
    for arm in a.arms.replace(':', ',').split(','):
        St = Store(a.bench, 'vj' if arm == 'T3' else 'd3')
        R = St.R
        kinds = [k for k in ('ev0', 'evm', 'evp') if (R.kind == k).any()]
        pool = R[R.kind == 'pool']
        roles = {r: np.flatnonzero((R.kind == 'pool').values & (R.role == r).values) for r in ('train', 'calib', 'fve')}
        if not len(roles['fve']):
            roles['fve'] = roles['calib']
        keep_p = None
        if lab:
            ntr = int(R.n.values[roles['train']].sum())
            keep_p = min(1.0, LAB_CAP / max(ntr, 1))
        t0 = time.time()
        Xtr, _ = St.gather(roles['train'], arm, keep_p=keep_p, seed=7)
        Xca, _ = St.gather(roles['calib'], arm, keep_p=keep_p, seed=8)
        Xev, _ = St.gather(roles['fve'], arm, keep_p=keep_p, seed=9)
        log(f'{a.bench} {arm}: tokens train {len(Xtr):,} calib {len(Xca):,} fve {len(Xev):,} (keep p {keep_p}), '
            f'{len(pool)} pool records, gather {time.time() - t0:.0f}s')
        Xtr_g = torch.from_numpy(Xtr).to(dev)
        for s in seeds:
            f = od / 'scores' / f'{arm}_s{s}.json'
            if f.exists() and not a.force:
                continue
            sd = BEE / a.bench / 'sae' / f'{arm}_s{s}'
            t1 = time.time()
            if (sd / 'sae_theta.pt').exists() and not a.force:
                rec = json.loads((sd / 'train.json').read_text())
            else:
                rec = SAE.train(SAE_CFG, s, Xtr_g, torch.from_numpy(Xca).to(dev), torch.from_numpy(Xev).to(dev), sd, dev)
            ttrain = time.time() - t1
            sae, norm = SAE.load(sd / 'sae_theta.pt', dev)
            C = eval_codes(St, sae, norm, arm, dev, kinds)
            ro = {'raw': C['ev0']}
            if 'evm' in C and 'evp' in C:
                ro['sm'] = (C['evm'] + C['ev0'] + C['evp']) / 3
            del C
            out = {'arm': arm, 'seed': s, 'bench': a.bench, 'train': {k: rec[k] for k in rec if k != 'history'},
                   'train_s': ttrain, 'score_s': None}
            t2 = time.time()
            if lab:
                out['cells'] = {}
                for cell, (ix, sp) in specs.items():
                    it = torch.from_numpy(ix).to(dev)
                    res = score_spec({k: v.index_select(0, it) for k, v in ro.items()}, sp, dev)
                    out['cells'][cell] = {'summary': summarise(res, sp.labels, sp.primary_split, sp.prim),
                                          'summary_rnd': summarise({k: {'frame': {s_: v['frame'][s_] for s_ in v['frame']
                                                                                   if s_.startswith('rnd')},
                                                                         'video': {s_: v['video'][s_] for s_ in v['video']
                                                                                   if s_.startswith('rnd')}}
                                                                    for k, v in res.items()}, sp.labels, None, sp.prim),
                                          'full': res}
            else:
                res = score_spec(ro, spec, dev)
                out['summary'] = summarise(res, spec.labels)
                out['full'] = res
            out['score_s'] = time.time() - t2
            f.write_text(json.dumps(out, default=float))
            log(f'{a.bench} {arm} s{s}: FVE {rec["eval_fve"]:.4f}, train {ttrain:.0f}s, score {out["score_s"]:.0f}s; ' +
                (json.dumps({c: {r: round(v['summary'][r]['_primary']['R'], 4) for r in v['summary']}
                             for c, v in out['cells'].items()}) if lab else
                 json.dumps({r: {k: round(v['_primary'][k], 4) for k in ('R', 'AP', 'vid_r')}
                             for r, v in out['summary'].items()})) + f'; {gpu_line()}')
            del sae, norm, ro
            torch.cuda.empty_cache()
        del Xtr_g
        torch.cuda.empty_cache()


# ================================================================================================ smoke
def cmd_smoke(a):
    import torch
    import torch.nn.functional as F
    dev = torch.device('cuda')
    B = MABe22()
    E = B.eval_table()
    clips = sorted(E.video.unique())[:2]
    fr = [B.decode(c, None, pd.DataFrame({'kind': ['ev0']}))[0] for c in clips]
    res = {'gpu': torch.cuda.get_device_name(0), 'vj': {}, 'd3': {}}
    for S in (512, 640, 768, 1024):
        # real clips: 5 fps clip around frame 900 of each decoded MABe22 clip, upscaled bicubic to S
        idx = np.clip(900 + 6 * (np.arange(CLIP) - TPOS), 0, 1800)
        x = torch.stack([F.interpolate(torch.from_numpy(np.stack([f[i] for i in idx])).to(dev).permute(0, 3, 1, 2)
                                       .float(), size=(S, S), mode='bicubic', align_corners=False).round().clamp(0, 255)
                         .to(torch.uint8) for f in fr])
        r = {}
        vb = VJ(S, dev, bf16=True)
        zb = vb(x[:1]).float()
        vf = VJ(S, dev, bf16=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        zf = vf(x[:1]).float()
        r['cos_bf16_vs_fp32_mean'] = float(F.cosine_similarity(zb, zf, dim=-1).mean())
        r['cos_bf16_vs_fp32_p1'] = float(F.cosine_similarity(zb, zf, dim=-1).flatten().quantile(0.01))
        if S == 512:  # truncation identity: full model + hook on block 20 + final LN == truncated model
            full = VJ(S, dev, bf16=False, truncate=False)
            cap = {}
            hk = full.m.encoder.layer[VJ_LAYER - 1].register_forward_hook(lambda m, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))
            with torch.inference_mode():
                xin = (x[:1].float() / 255 - full.mean) / full.std
                full.m(pixel_values_videos=xin, skip_predictor=True)
            h = full.m.encoder.layernorm(cap['h']).float()
            GG = (S // 16) ** 2
            r['truncation_max_abs_diff'] = float((h[:, TUB * GG:(TUB + 1) * GG] - zf).abs().max())
            hk.remove()
            del full
        del vf
        for bs in (1, 2, 4):
            try:
                xx = x[:1].repeat(bs, 1, 1, 1, 1)
                vb(xx)
                torch.cuda.synchronize()
                g0, n0 = vb.gpu_s, vb.n_clips
                for _ in range(3):
                    vb(xx)
                r[f'clips_per_s_bf16_bs{bs}'] = (vb.n_clips - n0) / (vb.gpu_s - g0)
                r[f'peak_gib_bs{bs}'] = torch.cuda.max_memory_allocated() / 2**30
            except torch.cuda.OutOfMemoryError:
                r[f'clips_per_s_bf16_bs{bs}'] = None
                torch.cuda.empty_cache()
        del vb
        torch.cuda.empty_cache()
        res['vj'][S] = r
        log(f'vj S {S}: {r}')
    d3 = D3(dev)
    for S, bs in ((512, 64), (640, 32), (1024, 16)):
        x = torch.randint(0, 255, (bs, 3, S, S), dtype=torch.uint8, device=dev)
        d3(x)
        g0, n0 = d3.gpu_s, d3.e.n_frames
        for _ in range(3):
            d3(x)
        res['d3'][S] = {'frames_per_s': (d3.e.n_frames - n0) / (d3.gpu_s - g0)}
    log(json.dumps(res['d3']))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'smoke.json').write_text(json.dumps(res, indent=1, default=float))


def cmd_feasibility(a):
    """the pre-registered FEASIBILITY rule applied to OUT/smoke.json -> OUT/feasibility.json (CPU)."""
    sm = json.loads((OUT / 'smoke.json').read_text())
    vj = {int(k): v for k, v in sm['vj'].items()}
    S0 = {'mabe22': 512, 'calms21': 1024, 'crim13': 640}
    bf16 = all(vj[S]['cos_bf16_vs_fp32_mean'] >= 0.98 for S in S0.values())
    n = {}
    for b in S0:
        B = BENCHES[b]()
        n[b] = (len(B.pool_table()), len(B.eval_table()))

    def rate(S):
        r = [v for k, v in vj[S].items() if k.startswith('clips_per_s_bf16') and v]
        best = max(r)
        return best if bf16 else best / 3.0  # fp32 ~3x slower (MABe22 encoder study measured 3.0x at 512)

    side = dict(S0)

    def cost(side, kinds):
        return sum((n[b][0] + n[b][1] * (3 if kinds[b] == 3 else 1)) / rate(side[b]) / 3600 for b in side)
    kinds = {b: 1 for b in S0}
    for alt in (768, 512):
        if cost(side, kinds) <= 15:
            break
        side['calms21'] = alt
    for b in sorted(S0, key=lambda b: (n[b][1] * 2) / rate(side[b])):
        k2 = {**kinds, b: 3}
        if cost(side, k2) <= 15:
            kinds = k2
    out = {'bf16': bf16, 'side': side, 'kinds': {b: (['ev0', 'evm', 'evp'] if kinds[b] == 3 else ['ev0']) for b in S0},
           'projected_gpu_h': cost(side, kinds), 'n_pool_eval': n,
           'clips_per_gpu_s': {S: rate(S) for S in vj}, 'rule': 'temporal.py docstring FEASIBILITY'}
    (OUT / 'feasibility.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('smoke', 'feasibility', 'encode', 'lab', 'run', 'decide'))
    p.add_argument('--bench')
    p.add_argument('--dom')
    p.add_argument('--enc', default='d3')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--max-videos', type=int, default=0)
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--arms', default='T0:T1')
    p.add_argument('--seeds', default='')
    a = p.parse_args()
    if a.step == 'decide':
        import decide
        decide.main()
        return
    {'smoke': cmd_smoke, 'feasibility': cmd_feasibility, 'encode': cmd_encode, 'lab': cmd_lab, 'run': cmd_run}[a.step](a)


if __name__ == '__main__':
    main()
