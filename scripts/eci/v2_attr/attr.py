"""
ECI v2: where does a neuron's evidence come from? Tight mask (V2F_M0) vs thin-part mask (V2F_N2), mice v1 tails.

User question (2026-10-25): with the tight mask, can tail behaviours still be captured through DINO's context mixing
into the kept body tokens, how would we know, and is the reported heat map just the activation of the max-pooled token?

==================================================================================================================
PRE-REGISTRATION (written 2026-10-25 before any frame of this study was encoded or scored; changes after this point
are dated amendments at the end of this docstring)

WHAT THE HEAT MAPS SHOW (task 1, code reading, no data): the v2 atlas / heat files draw a_t = the neuron's SAE code at
EVERY kept token t (zero on tokens the mask dropped), src/eci/v2/heatmap.py (neuron_maps) and scripts/eci/v2/tools.py
TokenCodes.heat / heat_rows (assert max_t a_t == the frame code); the page overlays it (scripts/eci/build_atlas_v37.py
step_v2heat -> build_explorer tile_frames). It is not only the argmax token, and it is a TOKEN-level quantity: a code at a
body token whose value is caused by tail pixels outside any kept token is drawn on the body token.

REPRESENTATIONS (frozen, no training here): DINOv3-L/16 block 20 at native 1024 px (src/eci/v2/encode.py), canonical
frame (odour corner top right); SAEs = deployment seeds of the V2-full candidates (highest held-out FVE among 900-902):
V2F_M0_s901 (tight mask: >= 25% frozen dark-cue pixels per patch) and V2F_N2_s900 (thin-part mask: mask2 cue >= 25% OR
>= 16 appendage pixels), theta* threshold, frame value = max over the arm's kept tokens (single frames, no temporal
smoothing).

FRAMES: mice v1, all 432 videos, 6 frames each at k = round(linspace(0.08, 0.92, 6) * (n - 1)) of the 5 fps grid (2,592
frames). Halves by pool (pool = obs without the last two fields; crc32(pool) % 2): SEL (= 0, neuron selection) and TEST
(= 1, attribution numbers and sheets). Probes use 5-fold GroupKFold by pool over all frames.

PIXEL SETS (label-free, src/eci/v2/mask2 with its pre-registered parameters, mice h from calib.json):
    pix  = mask2 animal pixels (cue()); app = appendage() thin parts (on mice: tails, plus occasional paws / snout tips).
    fillT    = app dilated by 2 px (square), so no tail halo stays;
    fillTout = fillT pixels inside patches the tight mask does NOT keep (tail pixels no M0 token can see as its own);
    fillB    = (pix dilated by 2 px) minus fillT (bodies).
CONDITIONS: orig; occT / occTout / occB = those pixels replaced by the video's median empty-arena background (the stored
mask2 background). Every arm keeps its ORIGINAL kept-token positions in every condition (only the evidence changes).

TOKEN SETS: strict body token = M0-kept AND zero fillT pixels in its 16 x 16 patch (its own pixels are IDENTICAL in orig
and occT, so any change of its representation under occT is context, i.e. information that reached it through
attention). Tail token = N2-kept AND not M0-kept AND >= 16 appendage pixels.

TARGETS (label-free, from app):
    token (strict body tokens):
      K1 = 1[nb_app >= 16], nb_app = appendage pixels in the 5 x 5 token neighbourhood minus the token itself (a tail
           within 2 tokens = 32 px);  K2 = log1p(nb_app);  K3 = (cos, sin) of the direction from the token centre to
           the neighbourhood appendage-pixel centroid (K1 = 1 tokens only).
    frame: F1 = log1p(appendage pixels in the frame); F2 = number of appendage components >= 64 px (8-connected);
           F3 = 1[F2 >= 1] (AUROC reported only if both classes >= 5% of frames).
PROBES: ridge (features standardised on the training fold, intercept, lambda in {0.1, 1, 10, 100, 1e3, 1e4} chosen by
    an inner 3-fold group CV on the training fold), 5-fold GroupKFold by pool; binary targets scored by AUROC of the
    ridge score, continuous by R^2 (pooled over held-out folds). Token level: <= 300k training tokens per fold (seeded
    subsample), all held-out tokens scored.
    Token features: z of the strict body token (orig). The SAME fitted probe is applied to the same token in occT.
    Frame features: mean of strict body tokens (orig; and the orig probe applied to the occT mean), mean of tail tokens
    (zeros + a has-tail-token indicator when there is none), mean of M0-kept tokens, mean of N2-kept tokens, M0 frame
    codes (2048), N2 frame codes (2048).
LEAK METRICS (task 2):
    token leak       = AUROC_orig(K1) - AUROC_occT(K1) with the same probe; 95% CI by pool bootstrap (500 draws);
    leak share       = (mean score of K1=1 tokens orig - occT) / (mean score K1=1 orig - mean score K1=0 orig);
    R^2 drop of K2 and K3 (orig vs occT);
    representation change = median ||z_occT - z|| / ||z|| of strict body tokens by token distance (Chebyshev) to the
    nearest patch with fillT pixels (1, 2, 3, 4-6, >= 7), and the share of strict body tokens whose M0 SAE active set
    changes.
DECISION Q2: "the tight mask's kept body tokens carry tail information through context" iff token leak >= 0.05 AND its
    CI lower bound > 0. "Most of the decodable tail signal is context" iff leak share >= 0.5.

NEURONS (selected on SEL frames only):
    N2 tail neurons   = top 5 by appendage share sum_t a_t app_t / sum_t a_t pix_t (mask2_test definition; app_t, pix_t
                        = appendage / animal pixel fractions of token t) among neurons firing on >= 20 SEL frames;
    M0 tail-predicting = top 5 by AUROC of the neuron's code at strict body tokens for K1 (neurons non-zero on >= 0.5%
                        of SEL strict body tokens), plus top 5 by Spearman(frame code, F1) over SEL frames (union);
    random baselines  = 20 random M0 and 20 random N2 neurons (seed 0) firing on >= 20 SEL frames.
ATTRIBUTION (task 3, TEST frames; per neuron its top 30 TEST frames by orig frame code):
    D_X = max(0, a_orig - a_X) for X in {occT, occTout, occB}; tail share = D_T / (D_T + D_B) (frames with D_T + D_B = 0
    dropped); relative drops D_T / a, D_Tout / a; heat-map tail mass = the appendage share above on the same frames.
    Medians over the frames are reported.
DECISION Q3: an M0 neuron "fires because of tail pixels it cannot see" iff median D_Tout / a >= 0.3 AND median tail
    share >= 0.5; its token heat map is "misleading" iff in addition its median heat-map tail mass <= 0.1. An N2 tail
    neuron's heat map is "faithful about tails" iff |median heat-map tail mass - median tail share| <= 0.2.
SHEETS: 3 top N2 tail neurons + 3 top M0 tail-predicting neurons (token-AUROC list), top 3 TEST frames each: block
    occlusion map (each 16 px token-aligned block within 2 tokens of an animal pixel replaced by the background, map =
    a_orig - a_block), next to the frame and the token heat map, thin-part pixels outlined. Agreement per frame: Spearman
    between the token heat map and the occlusion map over the ROI blocks, and the token distance between their peaks.
RECOMMENDATION RULES (task 4):
    mask: if Q2 is not met or no M0 neuron meets Q3, tail behaviours are not recoverable from the tight mask in a way the
          pipeline can attribute -> any tail claim needs thin-part tokens (N2-type); otherwise the tight mask carries
          tail information through context and is defensible for tails provided attribution is by occlusion.
    heat maps: complement token heat maps with occlusion maps iff any M0 neuron is 'misleading' (Q3) OR the median
          Spearman(heat, occlusion) over sheet frames < 0.5 for either arm; replace them only if the token heat map
          misses the occlusion peak (distance > 2 tokens) on most (> 50%) sheet frames of both arms.
BUDGET: <= 25 GPU-h (expected ~3).

AMENDMENT 2026-10-25 ~02:20 (after the first analyse run printed the token-probe block only; the attribution and frame
probes of that run were never written because it crashed; no neuron number had been seen): the first run showed that
tail occlusion changes strict body tokens even >= 7 tokens away (median relative change 0.07, M0 active set changed in
80% of tokens), so a noise floor is needed to read any change. Two CONTROL conditions are added, nothing else changes:
    occC  = the same fillT shape moved (torch.roll, shifts of 64-512 px, first with no animal pixel within 32 px) onto
            bedding and filled with the same background (a same-area pixel change that removes no animal part);
    orig2 = the original frame encoded again ALONE (batch of 1 instead of 5): bf16 numerical floor.
    Reported next to occT: token probe AUROC / R^2 on occC and orig2, representation change by distance for both, and
    per neuron D_C / a = max(0, a_orig - a_occC) / a and |a_orig - a_orig2| / a. Also tail-vs-control leak
    AUROC_occC - AUROC_occT with a pool-bootstrap CI. The decision rules are unchanged. Sheet occlusion maps compute
    the unoccluded value inside every batch (element 0), so the map is not shifted by batch-shape numerics.

Steps (scripts/eci/v2_attr/job.sh; staging in $LOCAL_DIR):
    extract --shard i --nshard n [--selftest]   (GPU) -> BEE/mice/s<i>/
    analyse                                    (GPU) -> OUT/attr.json, OUT/table.md
    sheets                                     (GPU) -> OUT/sheets/*.png, OUT/sheets.json
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
for p in ('scripts/eci', 'scripts/eci/nnnt', 'scripts/eci/v2', 'scripts/eci/v2_mask', 'scripts/eci/v2_mask2'):
    sys.path.insert(0, str(REPO / p))
import mask2_test as T2  # noqa: E402

N, R4 = T2.N, T2.R4
log = N.log
OUT = Path(os.environ.get('V2A_OUT') or REPO / 'results/vision/eci_v2_attr')
BEE = Path(os.environ.get('V2A_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_attr')
SAE_BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_diag/mice/sae')
DEPLOY = {'M0': 'V2F_M0_s901', 'N2': 'V2F_N2_s900'}
ARMS = ('M0', 'N2')
CONDS = ('orig', 'occT', 'occTout', 'occB', 'occC', 'orig2')
ZCONDS = ('orig', 'occT', 'occC', 'orig2')  # conditions whose union tokens are stored
NFR, FILL_PX, NB_R, MIN_COMP = 6, 2, 2, 64
LAMS = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4)
DOM, SER = 'mice', 'v1'


def pool_of(o):
    return o.rsplit('_', 2)[0]


def half_of(o):
    return zlib.crc32(pool_of(o).encode()) % 2  # 0 = SEL, 1 = TEST


def frame_list(se):
    rows = []
    for v in se.videos:
        for k in np.unique(np.round(np.linspace(0.08, 0.92, NFR) * (v['n'] - 1)).astype(int)):
            rows.append((v['obs'], int(k)))
    return rows


class FakeEncoder:
    """CPU dry run of the code paths only (V2A_FAKE_ENC=1): random tokens, never used for any reported number."""
    def __init__(self, dev):
        self.dev, self.gpu_s, self.n_frames = dev, 0.0, 0

    def __call__(self, x):
        import torch
        g = torch.Generator().manual_seed(int(x[:, :, ::64, ::64].float().sum()) % 1000)
        G = x.shape[-1] // 16
        return torch.randn(x.shape[0], G * G, 1024, generator=g).half().to(x.device)


def get_dev_enc():
    import torch
    if os.environ.get('V2A_FAKE_ENC'):
        dev = torch.device('cpu')
        return dev, FakeEncoder(dev)
    from src.eci.v2.encode import Encoder
    dev = torch.device('cuda')
    return dev, Encoder(dev)


def gpu_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'


def load_saes(dev):
    from src.eci.sae import load_sae
    out = {}
    for a, d in DEPLOY.items():
        sae, norm, _ = load_sae(SAE_BEE / d / 'sae_theta.pt', dev)
        out[a] = (sae.eval(), norm)
    return out


# ---------------------------------------------------------------------------------------------- per-frame geometry
class Geo:
    """Pixel sets, token keeps and token / frame targets of one canonical frame x (1, 3, S, S) uint8 cuda."""

    def __init__(self, se, v, x, k, pack, h):
        import torch
        import torch.nn.functional as F
        from scipy import ndimage
        from src.eci.v2 import mask2 as M2
        S = x.shape[-1]
        G = S // 16
        core0 = se.mask.pixels(x, v['src_lo'] + np.array([k]))
        r = M2.cue(x, pack, core0, h)
        pix = r['pix']
        _, app = M2.appendage(pix, h)
        dil = lambda m, r_: F.max_pool2d(m.float()[:, None], 2 * r_ + 1, 1, r_)[:, 0] > 0.5  # noqa: E731
        pool = lambda m: F.avg_pool2d(m.float()[:, None], 16).flatten(1)[0]  # noqa: E731
        self.raw, self.new, self.fapp = pool(core0), pool(pix), pool(app)
        self.napp = self.fapp * 256
        self.keep = {'M0': self.raw >= 0.25, 'N2': (self.new >= 0.25) | (self.napp >= M2.MIN_THIN_PX)}
        fillT = dil(app, FILL_PX)
        keep_px = self.keep['M0'].view(1, 1, G, G).float()
        keep_px = F.interpolate(keep_px, scale_factor=16, mode='nearest')[:, 0] > 0.5
        self.fill = {'occT': fillT, 'occTout': fillT & ~keep_px, 'occB': dil(pix, FILL_PX) & ~fillT}
        self.nfillT = pool(fillT) * 256
        bg = pack['bg'].round().clamp(0, 255).to(torch.uint8)[None]
        self.x = {'orig': x}
        # amendment 2026-10-25 (control): the fillT shape moved to bedding (no animal pixel within 32 px), same fill
        anim = dil(pix, 32)
        best = None
        for r_ in (64, 128, 192, 256, 320, 384, 448, 512):
            for dy, dx in ((r_, 0), (-r_, 0), (0, r_), (0, -r_), (r_, r_), (-r_, -r_), (r_, -r_), (-r_, r_)):
                sh = torch.roll(fillT, (dy, dx), (1, 2))
                if not bool((sh & anim).any()):
                    best = sh
                    break
            if best is not None:
                break
        self.fill['occC'] = best if best is not None else torch.roll(fillT, (256, 256), (1, 2)) & ~anim
        for c, m in self.fill.items():
            self.x[c] = torch.where(m[:, None], bg, x)
        self.x['orig2'] = x
        dpx = lambda c: float((self.x[c].float() - x.float()).abs().amax(1)[self.fill[c]].mean()) \
            if self.fill[c].any() else 0.0  # noqa: E731
        self.dpix = {'dpix_occT': dpx('occT'), 'dpix_occC': dpx('occC'), 'occC_px': int(self.fill['occC'].sum())}
        # token neighbourhood targets (5 x 5 minus self)
        yy, xx = torch.meshgrid(torch.arange(S, device=x.device, dtype=torch.float32),
                                torch.arange(S, device=x.device, dtype=torch.float32), indexing='ij')
        af = app[0].float()
        A = F.avg_pool2d(af[None, None], 16)[0, 0] * 256
        Ay = F.avg_pool2d((af * yy)[None, None], 16)[0, 0] * 256
        Ax = F.avg_pool2d((af * xx)[None, None], 16)[0, 0] * 256
        K = torch.ones(1, 1, 2 * NB_R + 1, 2 * NB_R + 1, device=x.device)
        box = lambda m: F.conv2d(m[None, None], K, padding=NB_R)[0, 0] - m  # noqa: E731
        nA, nY, nX = box(A), box(Ay), box(Ax)
        gy, gx = torch.meshgrid(torch.arange(G, device=x.device), torch.arange(G, device=x.device), indexing='ij')
        cy, cx = gy * 16 + 7.5, gx * 16 + 7.5
        dy = torch.where(nA > 0, nY / nA.clamp(min=1) - cy, torch.zeros_like(nA))
        dx = torch.where(nA > 0, nX / nA.clamp(min=1) - cx, torch.zeros_like(nA))
        self.nb_app = nA.flatten()
        self.nb_ang = torch.atan2(dy, dx).flatten()
        # token distance (Chebyshev) to the nearest patch with fillT pixels, capped at 9
        ft = (self.nfillT > 0).view(1, 1, G, G).float()
        dist = torch.full((G * G,), 9, dtype=torch.int8, device=x.device)
        cur = ft
        if ft.sum() > 0:
            dist[ft.flatten() > 0] = 0
            for d in range(1, 9):
                cur = F.max_pool2d(cur, 3, 1, 1)
                new = (cur.flatten() > 0) & (dist == 9)
                dist[new] = d
        self.dist = dist
        a_np = app[0].cpu().numpy()
        lab, n = ndimage.label(a_np, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0)
        self.frame = {'app_px': int(a_np.sum()), 'n_comp64': int((sizes >= MIN_COMP).sum()), 'pix_px': int(pix.sum()),
                      'core_px': int(core0.sum()), 'fillTout_px': int(self.fill['occTout'].sum()),
                      'fillT_px': int(fillT.sum()), 'fillB_px': int(self.fill['occB'].sum()),
                      'n_M0': int(self.keep['M0'].sum()), 'n_N2': int(self.keep['N2'].sum()), **self.dpix}
        self.app_px = app[0]
        self.S, self.G = S, G


def frame_codes(saes, z, keep):
    """z (C, G*G, d) fp16 tokens of the conditions, keep {arm: (G*G,) bool} -> {arm: (C, W) frame max codes,
    (W,) token code tensors of orig (n_kept, W)}."""
    import torch
    out = {}
    for a, (sae, norm) in saes.items():
        idx = torch.nonzero(keep[a]).flatten()
        if len(idx) == 0:
            out[a] = (torch.zeros(z.shape[0], sae.n_latents, device=z.device), None, idx)
            continue
        c = sae.encode(norm(z[:, idx].float().flatten(0, 1)), mode='threshold').view(z.shape[0], len(idx), -1)
        out[a] = (c.amax(1), c[0], idx)
    return out


# ---------------------------------------------------------------------------------------------- extract
def cmd_extract(args):
    import torch
    from src.eci.v2.encode import Encoder
    from src.eci.v2.pipeline import Series
    from src.eci.v2.video import Prefetch
    dev, enc = get_dev_enc()
    se = Series(T2.series_cfg(DOM, SER), dev, encoder=enc)
    h = T2.h_of(DOM, SER)
    saes = load_saes(dev)
    W = saes['M0'][0].n_latents
    vid = {v['obs']: i for i, v in enumerate(se.videos)}
    fl = [r for r in frame_list(se) if vid[r[0]] % args.nshard == args.shard]
    if args.selftest:
        fl = fl[::6][:int(os.environ.get('V2A_NSEL', 12))]  # one frame per video
    jobs = [((i,), se.job(se.V[o], k, 1, threads=2)) for i, (o, k) in enumerate(fl)]
    loc = N.local_dir() / f'attr_s{args.shard}'
    loc.mkdir(parents=True, exist_ok=True)
    fz = {c: open(loc / f'z_{c}.f16', 'wb') for c in ZCONDS}
    tm = {k: [] for k in ('tok_frame', 'tok_pos', 'raw', 'new', 'napp', 'nfillT', 'nb_app', 'nb_ang', 'dist',
                          'kM0', 'kN2')}
    FC = {a: np.zeros((len(fl), len(CONDS), W), np.float16) for a in ARMS}
    HN = {a: np.zeros((len(fl), W), np.float32) for a in ARMS}  # sum_t a_t app_t
    HD = {a: np.zeros((len(fl), W), np.float32) for a in ARMS}  # sum_t a_t pix_t
    ARG = {a: np.full((len(fl), W), -1, np.int16) for a in ARMS}
    recs, t0, packs = [], time.time(), {}
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=Series.run_job)
    for (i,), fr in pf:
        o, k = fl[i]
        v = se.V[o]
        if o not in packs:
            packs.clear()
            packs[o] = T2.load_pack(DOM, SER, o, 0, dev)
        with torch.no_grad():
            x, _ = se.frames(v, fr, np.array([k]))
            g = Geo(se, v, x, k, packs[o], h)
            z = enc(torch.cat([g.x[c] for c in CONDS[:-1]]))  # (5, G*G, 1024) fp16
            z = torch.cat([z, enc(g.x['orig2'])])  # orig again, alone in its batch: the numerical noise floor
            fc = frame_codes(saes, z, g.keep)
            for a in ARMS:
                m, tc, idx = fc[a]
                FC[a][i] = m.cpu().numpy().astype(np.float16)
                if tc is not None:
                    HN[a][i] = (tc * g.fapp[idx][:, None]).sum(0).cpu().numpy()
                    HD[a][i] = (tc * g.new[idx][:, None]).sum(0).cpu().numpy()
                    am = tc.argmax(0)
                    ARG[a][i] = np.where(m[0].cpu().numpy() > 0, idx[am].cpu().numpy(), -1)
            U = torch.nonzero(g.keep['M0'] | g.keep['N2']).flatten()
            for c in ZCONDS:
                fz[c].write(z[CONDS.index(c), U].cpu().numpy().tobytes())
            tm['tok_frame'].append(np.full(len(U), i, np.int32))
            tm['tok_pos'].append(U.cpu().numpy().astype(np.int16))
            for nm, t_ in (('raw', g.raw), ('new', g.new), ('napp', g.napp), ('nfillT', g.nfillT),
                           ('nb_app', g.nb_app), ('nb_ang', g.nb_ang)):
                tm[nm].append(t_[U].cpu().numpy().astype(np.float32))
            tm['dist'].append(g.dist[U].cpu().numpy())
            tm['kM0'].append(g.keep['M0'][U].cpu().numpy())
            tm['kN2'].append(g.keep['N2'][U].cpu().numpy())
        recs.append({'obs': o, 'k': k, 'pool': pool_of(o), 'half': half_of(o), 'role': v['role'], **g.frame})
        if i % 50 == 0:
            el = time.time() - t0
            log(f'extract s{args.shard}: {i + 1}/{len(fl)} frames, {el:.0f}s, GPU busy {enc.gpu_s:.0f}s, decode thread-s '
                f'{pf.dec_s:.0f}')
    for f in fz.values():
        f.close()
    for k_, x_ in tm.items():
        np.save(loc / f'{k_}.npy', np.concatenate(x_))
    for a in ARMS:
        np.save(loc / f'fc_{a}.npy', FC[a])
        np.save(loc / f'hn_{a}.npy', HN[a])
        np.save(loc / f'hd_{a}.npy', HD[a])
        np.save(loc / f'arg_{a}.npy', ARG[a])
    pd.DataFrame(recs).to_parquet(loc / 'frames.parquet')
    info = {'shard': args.shard, 'n_frames': len(recs), 'n_tokens': int(sum(len(x) for x in tm['tok_frame'])),
            'wall_s': time.time() - t0, 'gpu_s': enc.gpu_s, 'gpu': gpu_name()}
    log(json.dumps(info))
    if os.environ.get('V2A_FAKE_ENC'):
        R4.atomic_copy(loc, BEE / DOM / f's{args.shard}')
        return
    if args.selftest:
        selftest_panel(se, fl[:3], h, saes, dev)
        (OUT / 'selftest.json').write_text(json.dumps({'info': info, 'frames': recs}, indent=1, default=float))
        return
    R4.atomic_copy(loc, BEE / DOM / f's{args.shard}')
    (OUT / 'extract').mkdir(parents=True, exist_ok=True)
    (OUT / 'extract' / f's{args.shard}.json').write_text(json.dumps(info, indent=1))


def selftest_panel(se, fl, h, saes, dev):
    """Visual check of the pixel sets and conditions on 3 frames -> OUT/selftest_panel.png."""
    import torch
    from PIL import Image
    tiles = []
    for o, k in fl:
        v = se.V[o]
        fr = se.run_job(**se.job(v, k, 1))
        x, _ = se.frames(v, fr, np.array([k]))
        g = Geo(se, v, x, k, T2.load_pack(DOM, SER, o, 0, dev), h)
        row = []
        for c in CONDS:
            im = g.x[c][0].permute(1, 2, 0).cpu().numpy().copy()
            if c == 'orig':
                im[g.app_px.cpu().numpy()] = (255, 0, 0)
            row.append(np.asarray(Image.fromarray(im).resize((384, 384), Image.BILINEAR)))
        tiles.append(np.concatenate(row, 1))
    Image.fromarray(np.concatenate(tiles, 0)).save(OUT / 'selftest_panel.png')


# ---------------------------------------------------------------------------------------------- load
class Data:
    def __init__(self, root=None):
        root = Path(root or BEE / DOM)
        parts = sorted([p for p in root.glob('s*') if p.is_dir() and not p.name.endswith('.tmp')],
                       key=lambda p: int(p.name[1:]))
        Fs, T, self.fc, self.hn, self.hd, self.arg = [], {}, {a: [] for a in ARMS}, {a: [] for a in ARMS}, \
            {a: [] for a in ARMS}, {a: [] for a in ARMS}
        self.zfiles, off = [], 0
        for p in parts:
            F_ = pd.read_parquet(p / 'frames.parquet')
            n = len(F_)
            Fs.append(F_)
            for k in ('tok_frame', 'tok_pos', 'raw', 'new', 'napp', 'nfillT', 'nb_app', 'nb_ang', 'dist', 'kM0', 'kN2'):
                x = np.load(p / f'{k}.npy')
                if k == 'tok_frame':
                    x = x + off
                T.setdefault(k, []).append(x)
            for a in ARMS:
                self.fc[a].append(np.load(p / f'fc_{a}.npy'))
                self.hn[a].append(np.load(p / f'hn_{a}.npy'))
                self.hd[a].append(np.load(p / f'hd_{a}.npy'))
                self.arg[a].append(np.load(p / f'arg_{a}.npy'))
            self.zfiles.append((p, len(T['tok_frame'][-1])))
            off += n
        self.F = pd.concat(Fs, ignore_index=True)
        self.T = {k: np.concatenate(v) for k, v in T.items()}
        for d in (self.fc, self.hn, self.hd, self.arg):
            for a in ARMS:
                d[a] = np.concatenate(d[a])
        self.nf = len(self.F)
        self.nt = len(self.T['tok_frame'])

    def z(self, cond, sel=None):
        out, o = [], 0
        for p, n in self.zfiles:
            mm = np.memmap(p / f'z_{cond}.f16', np.float16, 'r', shape=(n, 1024))
            if sel is None:
                out.append(np.asarray(mm))
            else:
                s = sel[o:o + n]
                out.append(np.asarray(mm[s]))
            o += n
        return np.concatenate(out)


# ---------------------------------------------------------------------------------------------- probes
def ridge_fit(X, y, lam):
    """X (n, d) torch float32 standardised, y (n, m) -> (w (d, m), b (m,))."""
    import torch
    xm, ym = X.mean(0), y.mean(0)
    Xc, yc = X - xm, y - ym
    A = Xc.T @ Xc + lam * Xc.shape[0] / 1000.0 * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    w = torch.linalg.solve(A, Xc.T @ yc)
    return w, ym - xm @ w


def r2(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    if y.ndim == 1:
        y, p = y[:, None], p[:, None]
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float('nan')
    return float(roc_auc_score(y, s))


def cv_probe(X, Y, groups, dev, Xextra=None, cap=300000, seed=0):
    """Ridge probe, outer 5-fold GroupKFold, inner 3-fold lambda choice -> held-out predictions (n, m) and the same
    fold models applied to Xextra rows (same rows, other condition) -> pred_extra."""
    import torch
    from sklearn.model_selection import GroupKFold
    Y = Y.reshape(len(Y), -1).astype(np.float32)
    P = np.zeros_like(Y)
    if Xextra is not None and not isinstance(Xextra, dict):
        Xextra = {'x': Xextra}
        single = True
    else:
        single = False
    PE = {k: np.zeros_like(Y) for k in Xextra} if Xextra is not None else None
    rng = np.random.default_rng(seed)
    lams = []
    for tr, te in GroupKFold(5).split(X, Y, groups):
        if len(tr) > cap:
            tr = np.sort(rng.choice(tr, cap, replace=False))
        Xt = torch.from_numpy(X[tr].astype(np.float32)).to(dev)
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        Xt = (Xt - mu) / sd
        Yt = torch.from_numpy(Y[tr]).to(dev)
        best, bl = -1e18, LAMS[0]
        gi = groups[tr]
        for lam in LAMS:
            sc = []
            for a, b in GroupKFold(3).split(tr, groups=gi):
                w, c = ridge_fit(Xt[a], Yt[a], lam)
                pr = (Xt[b] @ w + c).cpu().numpy()
                sc.append(-((pr - Y[tr][b]) ** 2).mean())
            if np.mean(sc) > best:
                best, bl = np.mean(sc), lam
        lams.append(bl)
        w, c = ridge_fit(Xt, Yt, bl)
        for s0 in range(0, len(te), 200000):
            tt = te[s0:s0 + 200000]
            P[tt] = ((torch.from_numpy(X[tt].astype(np.float32)).to(dev) - mu) / sd @ w + c).cpu().numpy()
            if Xextra is not None:
                for k_, XE in Xextra.items():
                    PE[k_][tt] = ((torch.from_numpy(XE[tt].astype(np.float32)).to(dev) - mu) / sd @ w + c).cpu().numpy()
        del Xt, Yt
        torch.cuda.empty_cache()
    return P, (PE['x'] if single else PE), lams


def boot_auroc_diff(y, s1, s2, groups, n=500, seed=0):
    """Pool bootstrap of AUROC(y, s1) - AUROC(y, s2) -> (lo, hi)."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_of = [np.flatnonzero(inv == g) for g in range(len(ug))]
    d = []
    for _ in range(n):
        pick = rng.integers(0, len(ug), len(ug))
        ii = np.concatenate([idx_of[g] for g in pick])
        d.append(auroc(y[ii], s1[ii]) - auroc(y[ii], s2[ii]))
    return [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def frame_mean(D, Z, sel, nf, dev):
    """mean of the token rows sel (bool over D.T; Z = their tokens) per frame -> (nf, d) float32, count (nf,)."""
    import torch
    fr = D.T['tok_frame'][sel]
    S = torch.zeros((nf, Z.shape[1]), dtype=torch.float32, device=dev)
    for s0 in range(0, len(fr), 200000):
        S.index_add_(0, torch.from_numpy(fr[s0:s0 + 200000]).to(dev).long(),
                     torch.from_numpy(Z[s0:s0 + 200000]).to(dev).float())
    c = np.bincount(fr, minlength=nf)
    return (S.cpu().numpy() / np.maximum(c, 1)[:, None]).astype(np.float32), c


# ---------------------------------------------------------------------------------------------- analyse
def cmd_analyse(args):
    import torch
    from scipy.stats import spearmanr
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = Data()
    F_, T = D.F, D.T
    res = {'n_frames': D.nf, 'n_tokens': D.nt, 'deploy': DEPLOY}
    groups_f = F_.pool.values
    tf = T['tok_frame']
    strict = T['kM0'] & (T['nfillT'] == 0)
    tailtok = T['kN2'] & ~T['kM0'] & (T['napp'] >= 16)
    res['tokens'] = {'strict_body': int(strict.sum()), 'tail': int(tailtok.sum()), 'M0_kept': int(T['kM0'].sum()),
                     'N2_kept': int(T['kN2'].sum()),
                     'M0_kept_with_fillT_px': int((T['kM0'] & (T['nfillT'] > 0)).sum())}
    res['frames'] = {'app_px_median': float(F_.app_px.median()), 'n_comp64_mean': float(F_.n_comp64.mean()),
                     'share_F3': float((F_.n_comp64 >= 1).mean()),
                     'share_fillT_outside_M0': float(F_.fillTout_px.sum() / max(1, F_.fillT_px.sum()))}
    log(json.dumps(res))
    # ------------------------------------------------ token probes (strict body tokens)
    Zo = D.z('orig', strict)
    ZX = {c: D.z(c, strict) for c in ZCONDS[1:]}
    Zt = ZX['occT']
    nb = T['nb_app'][strict]
    K1 = (nb >= 16).astype(np.float32)
    K2 = np.log1p(nb).astype(np.float32)
    gt = groups_f[tf[strict]]
    P, PEd, lams = cv_probe(Zo, np.stack([K1, K2], 1), gt, dev, Xextra=ZX)
    PE = PEd['occT']
    a_o, a_t = auroc(K1, P[:, 0]), auroc(K1, PE[:, 0])
    pos, neg = K1 > 0, K1 == 0
    share = float((P[pos, 0].mean() - PE[pos, 0].mean()) / (P[pos, 0].mean() - P[neg, 0].mean()))
    ci = boot_auroc_diff(K1, P[:, 0], PE[:, 0], gt)
    tok = {'n': int(len(K1)), 'K1_base_rate': float(K1.mean()), 'K1_AUROC_orig': a_o, 'K1_AUROC_occT': a_t,
           'K1_leak_AUROC': a_o - a_t, 'K1_leak_CI95': ci, 'K1_leak_share_of_score': share,
           'K1_mean_score_pos_orig': float(P[pos, 0].mean()), 'K1_mean_score_pos_occT': float(PE[pos, 0].mean()),
           'K1_mean_score_neg_orig': float(P[neg, 0].mean()), 'K1_mean_score_neg_occT': float(PE[neg, 0].mean()),
           'K2_R2_orig': r2(K2, P[:, 1]), 'K2_R2_occT': r2(K2, PE[:, 1]), 'lambdas': lams}
    for c in ('occC', 'orig2'):
        tok[f'K1_AUROC_{c}'] = auroc(K1, PEd[c][:, 0])
        tok[f'K2_R2_{c}'] = r2(K2, PEd[c][:, 1])
    tok['K1_leak_vs_control_AUROC'] = tok['K1_AUROC_occC'] - a_t
    tok['K1_leak_vs_control_CI95'] = boot_auroc_diff(K1, PEd['occC'][:, 0], PE[:, 0], gt)
    ang = T['nb_ang'][strict][pos]
    K3 = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    P3, PE3, _ = cv_probe(Zo[pos], K3, gt[pos], dev, Xextra={c: X[pos] for c, X in ZX.items()})
    tok.update(K3_R2_orig=r2(K3, P3), K3_n=int(pos.sum()), **{f'K3_R2_{c}': r2(K3, PE3[c]) for c in ZX})
    # representation change by distance (tail occlusion, control occlusion, numerical floor)
    dist = T['dist'][strict].astype(int)
    bins = {'1': dist == 1, '2': dist == 2, '3': dist == 3, '4-6': (dist >= 4) & (dist <= 6), '>=7': dist >= 7}
    tok['n_by_dist'] = {b: int(m.sum()) for b, m in bins.items()}
    zo32 = Zo.astype(np.float32)
    no = np.linalg.norm(zo32, axis=1)
    sae, norm = load_saes(dev)['M0']
    with torch.no_grad():
        ao_ = [sae.encode(norm(torch.from_numpy(Zo[s0:s0 + 100000]).to(dev).float()), 'threshold') > 0
               for s0 in range(0, len(Zo), 100000)]
    for c, X in ZX.items():
        rel = np.linalg.norm(X.astype(np.float32) - zo32, axis=1) / no
        tok[f'rel_change_median_by_dist_{c}'] = {b: float(np.median(rel[m])) if m.any() else None for b, m in bins.items()}
        ch = []
        with torch.no_grad():
            for q, s0 in enumerate(range(0, len(Zo), 100000)):
                ct = sae.encode(norm(torch.from_numpy(X[s0:s0 + 100000]).to(dev).float()), 'threshold') > 0
                ch.append((ao_[q] != ct).any(1).cpu().numpy())
        ch = np.concatenate(ch)
        tok[f'sae_active_set_changed_by_dist_{c}'] = {b: float(ch[m].mean()) if m.any() else None for b, m in bins.items()}
    del ao_, zo32
    tok['rel_change_median_by_dist'] = tok['rel_change_median_by_dist_occT']
    tok['sae_active_set_changed_by_dist'] = tok['sae_active_set_changed_by_dist_occT']
    tok['Q2_context_carries_tail'] = bool(tok['K1_leak_AUROC'] >= 0.05 and ci[0] > 0)
    tok['Q2_mostly_context'] = bool(share >= 0.5)
    res['token_probe'] = tok
    log(json.dumps(tok, default=float))
    # ------------------------------------------------ M0 tail-predicting neurons (token AUROC on SEL strict tokens)
    selt = (F_.half.values[tf[strict]] == 0)
    rng = np.random.default_rng(0)
    ii = np.flatnonzero(selt)
    ii = np.sort(rng.choice(ii, min(len(ii), 150000), replace=False))
    with torch.no_grad():
        C = sae.encode(norm(torch.from_numpy(Zo[ii]).to(dev).float()), 'threshold').cpu().numpy()
    act = (C > 0).mean(0)
    from scipy.stats import rankdata
    yk = K1[ii] > 0
    au = np.full(C.shape[1], np.nan)
    okn = np.flatnonzero(act >= 0.005)
    R = rankdata(C[:, okn], axis=0)
    npos, nneg = yk.sum(), (~yk).sum()
    au[okn] = (R[yk].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)
    m0_tok = [int(j) for j in np.argsort(-np.nan_to_num(au, nan=-1))[:5]]
    del Zo, Zt, ZX, C, R
    sel_f = F_.half.values == 0
    test_f = ~sel_f
    F1 = np.log1p(F_.app_px.values)
    fcM0 = D.fc['M0'][:, 0].astype(np.float32)
    rho = np.full(fcM0.shape[1], np.nan)
    firing = {a: (D.fc[a][sel_f, 0] > 0).sum(0) for a in ARMS}
    for j in np.flatnonzero(firing['M0'] >= 20):
        rho[j] = spearmanr(fcM0[sel_f, j], F1[sel_f]).correlation
    m0_fr = [int(j) for j in np.argsort(-np.nan_to_num(rho, nan=-9))[:5]]
    hn, hd = D.hn['N2'][sel_f].sum(0), D.hd['N2'][sel_f].sum(0)
    apsh = np.where((firing['N2'] >= 20) & (hd > 0), hn / np.maximum(hd, 1e-9), np.nan)
    n2_tail = [int(j) for j in np.argsort(-np.nan_to_num(apsh, nan=-1))[:5]]
    rr = np.random.default_rng(0)
    rnd = {a: [int(j) for j in rr.choice(np.flatnonzero(firing[a] >= 20), 20, replace=False)] for a in ARMS}
    res['selection'] = {
        'M0_token_auroc': {j: float(au[j]) for j in m0_tok}, 'M0_frame_spearman': {j: float(rho[j]) for j in m0_fr},
        'N2_appendage_share': {j: float(apsh[j]) for j in n2_tail}, 'random': rnd,
        'M0_token_auroc_quantiles': {q: float(np.nanquantile(au, q)) for q in (0.5, 0.9, 0.99)},
        'N2_appendage_share_quantiles': {q: float(np.nanquantile(apsh, q)) for q in (0.5, 0.9, 0.99)},
        'M0_appendage_share_top': None}
    hnm, hdm = D.hn['M0'][sel_f].sum(0), D.hd['M0'][sel_f].sum(0)
    apm = np.where((firing['M0'] >= 20) & (hdm > 0), hnm / np.maximum(hdm, 1e-9), np.nan)
    res['selection']['M0_appendage_share_top'] = {int(j): float(apm[j]) for j in np.argsort(-np.nan_to_num(apm, nan=-1))[:5]}
    # ------------------------------------------------ attribution on TEST frames
    def attrib(a, j):
        fc = D.fc[a][:, :, j].astype(np.float32)
        ti = np.flatnonzero(test_f & (fc[:, 0] > 0))
        ti = ti[np.argsort(-fc[ti, 0])[:30]]
        if len(ti) == 0:
            return None
        ao = fc[ti, 0]
        dT, dTo, dB, dC = (np.maximum(0, ao - fc[ti, CONDS.index(c)]) for c in ('occT', 'occTout', 'occB', 'occC'))
        dN = np.abs(ao - fc[ti, CONDS.index('orig2')])
        den = dT + dB
        ok = den > 0
        hm = D.hn[a][ti, j] / np.maximum(D.hd[a][ti, j], 1e-9)
        return {'n_frames': int(len(ti)), 'a_median': float(np.median(ao)),
                'tail_share_median': float(np.median(dT[ok] / den[ok])) if ok.any() else None,
                'dT_rel_median': float(np.median(dT / ao)), 'dTout_rel_median': float(np.median(dTo / ao)),
                'dB_rel_median': float(np.median(dB / ao)), 'heat_tail_mass_median': float(np.median(hm)),
                'dC_rel_median': float(np.median(dC / ao)), 'dNoise_rel_median': float(np.median(dN / ao)),
                'frames': [int(x) for x in ti[:3]]}
    att = {'N2_tail': {}, 'M0_token': {}, 'M0_frame': {}, 'M0_random': {}, 'N2_random': {}}
    for j in n2_tail:
        att['N2_tail'][j] = attrib('N2', j)
    for j in m0_tok:
        att['M0_token'][j] = attrib('M0', j)
    for j in m0_fr:
        att['M0_frame'][j] = attrib('M0', j)
    for a in ARMS:
        for j in rnd[a]:
            att[f'{a}_random'][j] = attrib(a, j)
    for grp, d in att.items():
        for j, r in d.items():
            if r is None:
                continue
            if grp.startswith('M0'):
                r['Q3_fires_on_unseen_tail'] = bool(r['dTout_rel_median'] >= 0.3 and (r['tail_share_median'] or 0) >= 0.5)
                r['Q3_heat_misleading'] = bool(r['Q3_fires_on_unseen_tail'] and r['heat_tail_mass_median'] <= 0.1)
            else:
                r['Q3_heat_faithful'] = bool(r['tail_share_median'] is not None and
                                             abs(r['heat_tail_mass_median'] - r['tail_share_median']) <= 0.2)
    res['attribution'] = att
    # ------------------------------------------------ frame probes
    Zall = D.z('orig')
    Xb, cb = frame_mean(D, Zall[strict], strict, D.nf, dev)
    Xbt, _ = frame_mean(D, D.z('occT', strict), strict, D.nf, dev)
    Xtl, ct = frame_mean(D, Zall[tailtok], tailtok, D.nf, dev)
    Xtl = np.concatenate([Xtl, (ct > 0)[:, None].astype(np.float32)], 1)
    Xm0, _ = frame_mean(D, Zall[T['kM0']], T['kM0'], D.nf, dev)
    Xn2, _ = frame_mean(D, Zall[T['kN2']], T['kN2'], D.nf, dev)
    del Zall
    YF = np.stack([F1, F_.n_comp64.values.astype(np.float32), (F_.n_comp64.values >= 1).astype(np.float32)], 1)
    fr_res = {}
    for nm, X, XE in (('strict_body_mean', Xb, Xbt), ('tail_tokens_mean', Xtl, None), ('M0_kept_mean', Xm0, None),
                      ('N2_kept_mean', Xn2, None), ('M0_frame_codes', D.fc['M0'][:, 0].astype(np.float32), None),
                      ('N2_frame_codes', D.fc['N2'][:, 0].astype(np.float32), None)):
        P_, PE_, _ = cv_probe(X, YF, groups_f, dev, Xextra=XE)
        r = {'F1_R2': r2(YF[:, 0], P_[:, 0]), 'F2_R2': r2(YF[:, 1], P_[:, 1])}
        f3 = YF[:, 2].mean()
        r['F3_AUROC'] = auroc(YF[:, 2], P_[:, 2]) if 0.05 <= f3 <= 0.95 else None
        if PE_ is not None:
            r.update(F1_R2_occT=r2(YF[:, 0], PE_[:, 0]), F2_R2_occT=r2(YF[:, 1], PE_[:, 1]),
                     F1_mean_pred_orig=float(P_[:, 0].mean()), F1_mean_pred_occT=float(PE_[:, 0].mean()),
                     F1_mean_true=float(YF[:, 0].mean()))
        fr_res[nm] = r
        log(nm + ' ' + json.dumps(r, default=float))
    res['frame_probe'] = fr_res
    res['frame_probe_F3_base_rate'] = float(YF[:, 2].mean())
    (OUT / 'attr.json').write_text(json.dumps(res, indent=1, default=float))
    write_table(res)


def write_table(res):
    t = res['token_probe']
    L = ['# ECI v2 attribution: tight mask (V2F_M0 s901) vs thin-part mask (V2F_N2 s900), mice v1', '',
         f'{res["n_frames"]} frames; tokens {json.dumps(res["tokens"])}', '',
         '## Q2 context leak: token probe on strict body tokens (own pixels identical in orig / tail-occluded)', '',
         '| target | metric | orig | tail occluded | control occlusion (same shape on bedding) | orig re-encoded alone '
         '(numerical floor) | drop orig - tail occl. |', '|---|---|---|---|---|---|---|',
         f'| K1 tail within 2 tokens (base rate {t["K1_base_rate"]:.3f}) | AUROC | {t["K1_AUROC_orig"]:.3f} | '
         f'{t["K1_AUROC_occT"]:.3f} | {t["K1_AUROC_occC"]:.3f} | {t["K1_AUROC_orig2"]:.3f} | '
         f'{t["K1_leak_AUROC"]:+.3f} [{t["K1_leak_CI95"][0]:+.3f}, {t["K1_leak_CI95"][1]:+.3f}] |',
         f'| K2 log1p tail px nearby | R^2 | {t["K2_R2_orig"]:.3f} | {t["K2_R2_occT"]:.3f} | {t["K2_R2_occC"]:.3f} | '
         f'{t["K2_R2_orig2"]:.3f} | {t["K2_R2_orig"] - t["K2_R2_occT"]:+.3f} |',
         f'| K3 direction to tail (n {t["K3_n"]}) | R^2 | {t["K3_R2_orig"]:.3f} | {t["K3_R2_occT"]:.3f} | '
         f'{t["K3_R2_occC"]:.3f} | {t["K3_R2_orig2"]:.3f} | {t["K3_R2_orig"] - t["K3_R2_occT"]:+.3f} |', '',
         f'K1 leak share of the probe score: {t["K1_leak_share_of_score"]:.3f}; tail vs control occlusion AUROC '
         f'{t["K1_leak_vs_control_AUROC"]:+.3f} [{t["K1_leak_vs_control_CI95"][0]:+.3f}, '
         f'{t["K1_leak_vs_control_CI95"][1]:+.3f}]. Q2 context carries tail: '
         f'{t["Q2_context_carries_tail"]}; mostly context: {t["Q2_mostly_context"]}', '',
         '| token distance to nearest tail patch | ' + ' | '.join(t['n_by_dist']) + ' |',
         '|---|' + '---|' * len(t['n_by_dist']),
         '| n strict body tokens | ' + ' | '.join(str(v) for v in t['n_by_dist'].values()) + ' |']
    fm_ = lambda d: ' | '.join(f'{v:.3f}' if v is not None else '-' for v in d.values())  # noqa: E731
    for c, nm in (('occT', 'tail occluded'), ('occC', 'control occlusion'), ('orig2', 'numerical floor')):
        L.append(f'| median rel. change of z, {nm} | {fm_(t[f"rel_change_median_by_dist_{c}"])} |')
    for c, nm in (('occT', 'tail occluded'), ('occC', 'control occlusion'), ('orig2', 'numerical floor')):
        L.append(f'| share M0 SAE active set changed, {nm} | {fm_(t[f"sae_active_set_changed_by_dist_{c}"])} |')
    L += ['',
         '## Frame-level probes (5-fold by pool)', '',
         f'F3 base rate {res["frame_probe_F3_base_rate"]:.3f}', '',
         '| features | F1 log1p tail px R^2 | F2 n tails R^2 | F3 tail visible AUROC | F1 R^2 tail occluded |',
         '|---|---|---|---|---|']
    for nm, r in res['frame_probe'].items():
        f3 = f'{r["F3_AUROC"]:.3f}' if r.get('F3_AUROC') is not None else '-'
        oc = f'{r["F1_R2_occT"]:.3f} (mean pred {r["F1_mean_pred_orig"]:.2f} -> {r["F1_mean_pred_occT"]:.2f})' \
            if 'F1_R2_occT' in r else '-'
        L.append(f'| {nm} | {r["F1_R2"]:.3f} | {r["F2_R2"]:.3f} | {f3} | {oc} |')
    L += ['', '## Q3 occlusion attribution (TEST frames, top 30 per neuron; medians)', '',
          '| group | neuron | sel. score | a | tail share D_T/(D_T+D_B) | D_T/a | D_Tout/a | D_B/a | D_C/a (control) | '
          '|noise|/a | heat-map tail mass | flag |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    S = res['selection']
    score = {'N2_tail': S['N2_appendage_share'], 'M0_token': S['M0_token_auroc'], 'M0_frame': S['M0_frame_spearman']}
    f = lambda v: '-' if v is None else f'{v:.3f}'  # noqa: E731
    for grp in ('N2_tail', 'M0_token', 'M0_frame'):
        for j, r in res['attribution'][grp].items():
            if r is None:
                continue
            flag = (f'unseen-tail {r["Q3_fires_on_unseen_tail"]}, misleading {r["Q3_heat_misleading"]}'
                    if grp.startswith('M0') else f'faithful {r["Q3_heat_faithful"]}')
            L.append(f'| {grp} | {j} | {f(score[grp].get(j, score[grp].get(str(j))))} | {r["a_median"]:.3f} | '
                     f'{f(r["tail_share_median"])} | {r["dT_rel_median"]:.3f} | {r["dTout_rel_median"]:.3f} | '
                     f'{r["dB_rel_median"]:.3f} | {r["dC_rel_median"]:.3f} | {r["dNoise_rel_median"]:.3f} | '
                     f'{r["heat_tail_mass_median"]:.3f} | {flag} |')
    for grp in ('M0_random', 'N2_random'):
        rs = [r for r in res['attribution'][grp].values() if r]
        ts = [r['tail_share_median'] for r in rs if r['tail_share_median'] is not None]
        L.append(f'| {grp} (20) | median of medians | - | {np.median([r["a_median"] for r in rs]):.3f} | '
                 f'{np.median(ts):.3f} | {np.median([r["dT_rel_median"] for r in rs]):.3f} | '
                 f'{np.median([r["dTout_rel_median"] for r in rs]):.3f} | {np.median([r["dB_rel_median"] for r in rs]):.3f} | '
                 f'{np.median([r["dC_rel_median"] for r in rs]):.3f} | {np.median([r["dNoise_rel_median"] for r in rs]):.3f} | '
                 f'{np.median([r["heat_tail_mass_median"] for r in rs]):.3f} | |')
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------- sheets
def cmd_sheets(args):
    import torch
    from PIL import Image, ImageDraw
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use('Agg')
    cmap = matplotlib.colormaps['turbo']
    from src.eci.v2.encode import Encoder
    from src.eci.v2.pipeline import Series
    res = json.loads((OUT / 'attr.json').read_text())
    D = Data()
    dev, enc = get_dev_enc()
    se = Series(T2.series_cfg(DOM, SER), dev, encoder=enc)
    h = T2.h_of(DOM, SER)
    saes = load_saes(dev)
    picks = [('N2', int(j)) for j in list(res['selection']['N2_appendage_share'])[:3]] + \
            [('M0', int(j)) for j in list(res['selection']['M0_token_auroc'])[:3]]
    if args.selftest:
        picks = picks[:1]
    (OUT / 'sheets').mkdir(parents=True, exist_ok=True)
    out = {}
    for a, j in picks:
        grp = 'N2_tail' if a == 'N2' else 'M0_token'
        fr_ids = res['attribution'][grp][str(j)]['frames'][:1 if args.selftest else 3]
        rows, agr = [], []
        for fi in fr_ids:
            o, k = D.F.obs[fi], int(D.F.k[fi])
            v = se.V[o]
            fr = se.run_job(**se.job(v, k, 1))
            with torch.no_grad():
                x, _ = se.frames(v, fr, np.array([k]))
                pack = T2.load_pack(DOM, SER, o, 0, dev)
                g = Geo(se, v, x, k, pack, h)
                G, S = g.G, g.S
                sae, norm = saes[a]
                idx = torch.nonzero(g.keep[a]).flatten()
                z0 = enc(x)
                c0 = sae.encode(norm(z0[0, idx].float()), 'threshold')[:, j]
                heat = torch.zeros(G * G, device=dev)
                heat[idx] = c0
                a0 = float(c0.max())
                pixd = F_pool_dilate(g, G)
                blocks = torch.nonzero(pixd).flatten().tolist()
                bg = pack['bg'].round().clamp(0, 255).to(torch.uint8)
                occ = torch.zeros(G * G, device=dev)
                a0s = []
                for b0 in range(0, len(blocks), 15):
                    bb = blocks[b0:b0 + 15]
                    xs = x.repeat(16, 1, 1, 1)  # element 0 = the unoccluded frame: same batch shape, same numerics
                    for q, bl in enumerate(bb):
                        yy, xx_ = (bl // G) * 16, (bl % G) * 16
                        xs[q + 1, :, yy:yy + 16, xx_:xx_ + 16] = bg[:, yy:yy + 16, xx_:xx_ + 16]
                    zb = enc(xs)
                    cb = sae.encode(norm(zb[:, idx].float().flatten(0, 1)), 'threshold').view(16, len(idx), -1)
                    am = cb[:, :, j].amax(1)
                    a0s.append(float(am[0]))
                    occ[torch.tensor(bb, device=dev)] = am[0] - am[1:len(bb) + 1]
            hv, ov = heat.cpu().numpy(), occ.cpu().numpy()
            roi = np.array(blocks)
            sp = spearmanr(hv[roi], ov[roi]).correlation if len(roi) > 2 else np.nan
            ph, po = int(np.argmax(hv)), int(np.argmax(ov))
            pd_ = max(abs(ph // G - po // G), abs(ph % G - po % G))
            tail_occ = float(np.clip(ov, 0, None)[(g.napp.cpu().numpy() >= 16)].sum() / max(np.clip(ov, 0, None).sum(), 1e-9))
            agr.append({'obs': o, 'k': k, 'a': a0, 'a_batch_sd': float(np.std(a0s)), 'spearman_heat_occ': float(sp), 'peak_dist_tokens': pd_,
                        'occ_mass_on_tail_tokens': tail_occ, 'n_blocks': len(blocks),
                        'occ_min': float(ov.min()), 'occ_max': float(ov.max())})
            img = x[0].permute(1, 2, 0).cpu().numpy()
            app = g.app_px.cpu().numpy()
            edge = app & ~np.pad(app, 1)[2:, 1:-1] | app & ~np.pad(app, 1)[:-2, 1:-1] | \
                app & ~np.pad(app, 1)[1:-1, 2:] | app & ~np.pad(app, 1)[1:-1, :-2]

            def overlay(m, vmax):
                mm = np.clip(m.reshape(G, G) / max(vmax, 1e-9), 0, 1)
                up = np.asarray(Image.fromarray((mm * 255).astype(np.uint8)).resize((S, S), Image.BILINEAR)) / 255.
                col = (cmap(up)[..., :3] * 255).astype(np.float32)
                al = (0.65 * up)[..., None]
                o_ = (img * (1 - al) + col * al).astype(np.uint8)
                o_[edge] = (255, 255, 255)
                return o_
            raw = img.copy()
            raw[edge] = (255, 0, 0)
            pan = [raw, overlay(hv, hv.max()), overlay(np.clip(ov, 0, None), max(ov.max(), 1e-9))]
            rows.append(np.concatenate([np.asarray(Image.fromarray(p).resize((512, 512), Image.BILINEAR)) for p in pan], 1))
            log(f'sheet {a} n{j} {o} k{k}: a {a0:.3f}, spearman {sp:.3f}, peak dist {pd_}, blocks {len(blocks)}')
        im = Image.fromarray(np.concatenate(rows, 0))
        d = ImageDraw.Draw(im)
        for r_, ag in enumerate(agr):
            d.text((6, r_ * 512 + 4), f'{a} n{j} {ag["obs"]} k{ag["k"]} a={ag["a"]:.2f} | frame (thin parts red)',
                   fill=(255, 255, 0))
            d.text((518, r_ * 512 + 4), 'token heat map (SAE code per kept token)', fill=(255, 255, 0))
            d.text((1030, r_ * 512 + 4), f'occlusion map (drop of frame code), rho={ag["spearman_heat_occ"]:.2f}',
                   fill=(255, 255, 0))
        fn = OUT / 'sheets' / f'{a}_n{j:04d}.png'
        im.save(fn)
        out[f'{a}_n{j}'] = {'file': str(fn.relative_to(REPO)), 'frames': agr}
    if not args.selftest:
        (OUT / 'sheets.json').write_text(json.dumps(out, indent=1, default=float))
    else:
        log(json.dumps(out, default=float))


def F_pool_dilate(g, G):
    """ROI blocks: tokens within 2 tokens of any token with animal (mask2) pixels."""
    import torch.nn.functional as F
    m = (g.new > 0).view(1, 1, G, G).float()
    return (F.max_pool2d(m, 5, 1, 2).flatten() > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('extract', 'analyse', 'sheets'))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'extract': cmd_extract, 'analyse': cmd_analyse, 'sheets': cmd_sheets}[a.step](a)


if __name__ == '__main__':
    main()
