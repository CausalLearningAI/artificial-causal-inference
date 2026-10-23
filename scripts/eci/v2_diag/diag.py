"""
ECI v2 vs v1 DIAGNOSIS: why does the v2 representation not beat v1 on annotated behaviours, and which recipe does?

==================================================================================================================
PRE-REGISTRATION (written 2026-10-23 BEFORE any token of this study was encoded and before any score of any arm below
existed. Known before writing: the v2 thin-mask release check, results/vision/eci_v2_mask2/ab_table.md: M0 (= v2
recipe) minus v1, cross-fitted normalised R-precision R~ on the atlas split, full dictionaries: mice nose_nose +0.090,
mice nose_tail -0.007 [-0.018, +0.003], ants v2 groom_any -0.014, ants v3 groom_any -0.033, ants v3 on_lid_any +0.041;
at prefix 256: on_lid_any -0.29 (v2's lid neuron lies outside 256 in most seeds).)

QUESTION. Which of the pipeline differences between v1 (DINOv2-B/14 last layer at 448 px = 32 x 32 grid, dilated
feature+dark mask, Matryoshka BatchTopK 1024 trained on ~68M / ~7M tokens for 5 epochs, EMA threshold) and v2
(DINOv3-L/16 block 20 at native side = 64 grid mice / 48-52 grid ants, tight dark-pixel mask, Matryoshka 2048 on the
~4M / ~1.2M-token burst pool for 30k steps, calibrated theta*) explain the v2 - v1 differences per label, and does a
SIMPLER-or-equal recipe exist that is >= v1 - 0.01 on every label?

FRAMES, LABELS, SCORER: exactly those of the v2 release check (scripts/eci/v2_mask2/release_ab.py): mice = the 43,200
diagnosis frames restricted to the atlas window; ants = the step-2 scored frames of v2 (13,200) and v3 (32,100). SAE
training pools = the step-2 pools (mice 'pool' bursts; ants 'pool_v2' + 'pool_v3', one SAE per domain), roles
train / calib / eval as stored (theta* on calib videos, FVE on eval videos). Codes = frame max over the arm's kept
tokens, UNSMOOTHED (as release_ab). R~ = src/eci/v2/score.py; PRIMARY value = R~ of the cross-fitted best neuron on the
ATLAS split ('R_atlas', release_ab), the 10-random-halves mean 'R_rnd' reported.
CELLS (5): mice nose_nose, mice nose_tail, ants v2 groom_any, ants v3 groom_any, ants v3 on_lid_any; at the full
dictionary and at prefix 256 (both decide); nn_mutual / np_directional reported.

TOKEN SETS
    E0     the existing step-2 M0 store (eci_codes/v2_mask2): DINOv3-L/16 block 20 + final LN, native side (mice 1024 ->
           64 grid; ants 768 / 832 -> 48 / 52 grid), tight mask (>= 25% frozen-cue animal pixels per 16 px patch).
    32-grid sets, encoded here ONCE (step 'prep') on exactly the same frames, from the same decoded canonical frame
    (native side S) resized on the GPU (bicubic, antialias) to 512 px (patch-16 models) / 448 px (patch-14 models), so
    every set has a 32 x 32 grid = v1's token density; output of block L passed through the model's final LayerNorm,
    CLS / registers dropped:
        d3l16 / d3l20 / d3l24   DINOv3-L/16 (rev ea8dc28) blocks 16 / 20 / 24 (bf16 autocast, as v2)
        d3b12                   DINOv3-B/16 block 12 (last)               (bf16)
        d2b10 / d2b12           DINOv2-B/14 blocks 10 / 12 (12 = last = v1's encoder) (fp32, as v1)
        d2l24                   DINOv2-L/14 block 24 (last)               (bf16)
    32-grid masks: 'tight' = >= 25% frozen-cue animal pixels in the 32-grid cell (the v2 rule at v1's grid; pixel
    fraction pooled from the native-side pixel mask); 'dil' = tight dilated by one cell (3 x 3) = v1's one-patch ring.

ARMS (each: Matryoshka BatchTopK, frame max readout; the v2 SAE recipe W 2048, prefixes 128/512/2048, k 16, 30,000 steps
x 4096, lr 5e-4, global centring + scalar scale (TokenNorm), calibrated theta*, unless stated). SEEDS 300-304 (5 per
arm per domain).
  Stage A (SAE knobs, on E0):
    A_ref    the v2 recipe re-trained with this script (identity check against the 20 / 33 existing M0 seeds)
    A_w1024  W 1024, prefixes 128/256/512/1024 (v1's)
    A_w4096  W 4096, prefixes 128/512/2048/4096
    A_k8, A_k32           k 8 / 32
    A_s90k   90,000 steps (v1 mice saw ~82k steps)
    A_half   half of the training frames (seeded) -> data-size slope
    A_z      per-dimension z-scored tokens (mean / sd of 250k training tokens) instead of global centring + scale
    A_ema    the A_ref SAEs read out with their training EMA threshold instead of theta* (v1's readout); no new SAE
  Stage B (tokens, on the 32-grid sets, tight mask unless stated):
    B_d3l20  density: vs E0 (same encoder / layer, 32 vs native grid)
    B_d3l24, B_d3l16       layer (vs B_d3l20)
    B_d2b12  v1's encoder in the v2 pipeline (vs B_d3l20: DINOv3-L b20 -> DINOv2-B last)
    B_d2b10  layer for DINOv2-B (vs B_d2b12)
    B_d3b12  family at size B: vs B_d2b12          B_d2l24  family at size L (last layer): vs B_d3l24
    B_d3l20dil  mask: one-cell dilation (vs B_d3l20)
    B_v1like    d2b12 + dil + W 1024 (v1 prefixes) + EMA readout = v1's recipe on the v2 pool (vs v1 itself: the
                remaining difference is training data / steps)
CEILINGS (step 'probe', supervised, labels used only to measure): per token set (E0 and the seven 32-grid sets, tight
    mask) and cell: T-max = one linear direction on per-dimension standardised tokens, frame score = max over kept
    tokens (scripts/eci/nnnt/r2.py fit_probe 'max', 1500 steps, balanced batches); best raw channel = the best single
    standardised token dimension (either sign), frame max, selected by cross-fitting exactly like a neuron. Both on the
    atlas split (R~ held out, both directions). Reading: T-max(set) >> best neuron(set) = dictionary bottleneck;
    T-max(set) differs across sets = encoder / density bottleneck.

EFFECT OF A FACTOR on a cell = mean(arm) - mean(its reference) over seeds, Welch 95% CI. References: Stage A arms vs
    the existing M0 seeds (mice 70-89, ants 70-102; A_ref must lie inside their 95% range, else the A references
    switch to A_ref and this is reported); A_ema vs A_ref (paired by seed, t-CI); B arms as listed.
    A factor EXPLAINS part of the v1 - v2 gap on a cell iff moving it from the v2 level to the v1 level shifts R~
    toward v1 by >= 0.01 with the CI excluding 0.

COMBINED RECIPE (step 3; the recipe is fixed from the stage A/B results, then pre-registered in a second dated block
    here before any confirmation SAE is trained). Per factor, a non-v2 level is adopted iff (a) it improves at least one
    of the 10 decision cells (5 cells x {full, 256}) with the CI excluding 0, and (b) no decision cell gets worse with
    the CI upper bound < 0 AND the mean drop > 0.01, in BOTH domains' cells. TIE-BREAK (user rule 2026-10-23): when
    two levels are not separated by this rule, the SIMPLER one is chosen, simpler = (in order) fewer tokens per frame
    (coarser grid), smaller encoder, the encoder's last layer (no truncation), no mask dilation, global
    centring + scale (no per-dimension statistics), smaller dictionary, fewer training steps, fewer frames to encode.
    theta* stays (principled mean L0 = k, user 2026-10-14) unless EMA wins by the rule. The recipe must stay the
    one-line pipeline: encoder -> animal mask -> Matryoshka SAE -> readout -> NES/NESS (nothing added).
CONFIRMATION: fresh seeds 400-419 (mice) / 400-432 (ants) of the combined recipe vs v1 (deployed codes, release_ab
    refs): release criteria (A) distinct best nose-nose vs nose-tail neurons at full and prefix 256 (deployment seed =
    highest held-out FVE among 400-402, and >= 80% of seeds), (B) every decision cell's 95% t-CI lower bound of
    (recipe - v1) >= -0.01 at full and 256; background-neuron share <= 0.01 (scripts/eci/v2_mask2 definition);
    tail / odour-box discovery scores reported (scripts/eci/v2_mask2/discovery.py definitions).
DEVIATION FROM THE DEV / TEST POLICY (stated before scoring): factors are attributed on the lab labels, not MABe22 /
    CalMS21. Reasons: the question is why v2 trails v1 on these labels; the user allowed all mice v1 labels for this goal
    (project_nnnt_first_goal 2026-10-16); held-out evidence for the encoder factor already exists (CalMS21, 2026-10-15:
    size-matched DINOv3-L = DINOv2-L at a 64 grid, +0.017 primary AUROC; DINOv3-L wins at 32, +0.052). The confirmation
    uses fresh seeds but the same frames, so it is NOT an independent test of the recipe choice; this is said in the
    report. Rotation is identical in v1 and v2 (mice odour corner top right, ants none) and was measured neutral on
    2026-10-20, so it is not an arm here.
AMENDMENT 2026-10-23 ~04:30 (after the stage A/B scores of A_ref, A_w1024, A_w4096, A_s90k, A_ema (mice), B_d3l20,
    B_d3l24, B_d3l16, B_d2b12, B_d3b12, B_d2l24, B_v1like (mice) were seen; BEFORE any score of the arms below):
    (1) DROPPED for the GPU budget (25 GPU-h; ~15 used by the encode and the first arms): A_k8, A_k32 (k is not a
        v1 / v2 difference), B_d2b10. Not run.
    (2) BUG, no rule change: the first stage A/B jobs passed the arm list with commas, which sbatch --export splits,
        so only the first arm of each list ran; the rest were resubmitted with ':' separators.
    (3) ADDED, because B_v1like (v1's recipe on the v2 pool) does NOT reach v1 on mice nose_tail (0.047 vs v1 0.103)
        while width, steps and threshold do not explain it, leaving the training data / training mask:
        A_x2     the v2 recipe trained on the FULL reference pools (both burst halves: mice nnnt/r2/pool 62,208 frames,
                 ants v2/freeze pool_v2 + pool_v3 99,180 frames; tokens identical to M0, max |diff| 0 in the step-2
                 identity checks) = 2x the frames of the step-2 pools; same 30k steps, same roles. vs M0.
        B_v1sae  the DEPLOYED v1 SAE (mice fg448al_s0, ants antsfg_s0, its own stored threshold) read out on this
                 study's d2b12 'dil' tokens: a scoring identity check (does this token set reproduce v1's numbers?),
                 one SAE, reported next to v1 and B_v1like (B_v1like - B_v1sae = v1's training data / training mask /
                 steps, at identical scoring tokens).
==================================================================================================================

Steps (scripts/eci/v2_diag/job.sh; staging in $LOCAL_DIR):
    prep  --dom D --kind K --shard i --nshard n [--selftest]  (GPU) 32-grid multi-encoder store -> BEE/<D>/<K>/s<i>/
    train --dom D --arm A [--seeds a-b]                       (GPU) SAEs -> BEE/<D>/sae/<A>_s<seed>
    arm   --dom D --arms A,B [--seeds a-b]                    (GPU) train + score each arm
    score --dom D --arm A [--seeds]                           (GPU) -> OUT/<D>/ab/<A>_s<seed>.json
    probe --dom D --set S                                     (GPU) -> OUT/<D>/probe/<S>.json
    neurons --dom D --arm A [--seeds]                         (GPU) background / appendage / odour neuron counts
    decide                                                    (CPU) -> OUT/ablation.json, OUT/ablation_table.md
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
for p in ('scripts/eci', 'scripts/eci/nnnt', 'scripts/eci/v2', 'scripts/eci/v2_mask', 'scripts/eci/v2_mask2'):
    sys.path.insert(0, str(REPO / p))
import freeze as FZ  # noqa: E402
import mask_test as MT  # noqa: E402
import mask2_test as T2  # noqa: E402
import release_ab as RA  # noqa: E402

N, R2, R3, R4 = FZ.N, FZ.R2, FZ.R3, FZ.R4
log = N.log
OUT = Path(os.environ.get('V2D_OUT') or REPO / 'results/vision/eci_v2_diag')
BEE = Path(os.environ.get('V2D_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_diag')
G32 = 32
SEEDS = tuple(range(300, 305))
CONF_SEEDS = {'mice': tuple(range(400, 420)), 'ants': tuple(range(400, 433))}
KINDS = {'mice': ('pool', 'diag'), 'ants': ('pool_v2', 'pool_v3', 'scored_v2', 'scored_v3')}

# ---------------------------------------------------------------------------------------------- encoders (32 grid)
MODELS = {  # name -> (HF id, revision or None, input side, patch, dtype, layers kept)
    'd3l': ('facebook/dinov3-vitl16-pretrain-lvd1689m', 'ea8dc2863c51be0a264bab82070e3e8836b02d51', 512, 16, 'bf16',
            (16, 20, 24)),
    'd3b': ('facebook/dinov3-vitb16-pretrain-lvd1689m', None, 512, 16, 'bf16', (12,)),
    'd2b': ('facebook/dinov2-base', None, 448, 14, 'fp32', (10, 12)),
    'd2l': ('facebook/dinov2-large', None, 448, 14, 'bf16', (24,)),
}
SETS = {f'{m}{L}': (m, L) for m, c in MODELS.items() for L in c[5]}  # d3l16 d3l20 d3l24 d3b12 d2b10 d2b12 d2l24
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

BASE = dict(W=2048, prefixes=(128, 512, 2048), k=16, steps=30000, norm='global', frac=1.0, readout='theta')
ARMS = {
    'A_ref': dict(src='E0'), 'A_w1024': dict(src='E0', W=1024, prefixes=(128, 256, 512, 1024)),
    'A_w4096': dict(src='E0', W=4096, prefixes=(128, 512, 2048, 4096)), 'A_k8': dict(src='E0', k=8),
    'A_k32': dict(src='E0', k=32), 'A_s90k': dict(src='E0', steps=90000), 'A_half': dict(src='E0', frac=0.5),
    'A_z': dict(src='E0', norm='z'), 'A_ema': dict(src='E0', reuse='A_ref', readout='ema'),
    **{f'B_{s}': dict(src=s, mask='tight') for s in SETS},
    'B_d3l20dil': dict(src='d3l20', mask='dil'),
    'B_v1like': dict(src='d2b12', mask='dil', W=1024, prefixes=(128, 256, 512, 1024), readout='ema'),
    'A_x2': dict(src='E0full'),
    'B_v1sae': dict(src='d2b12', mask='dil', readout='own',
                    sae_path={'mice': 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448al_s0/sae.pt',
                              'ants': 'dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0/sae.pt'}),
}
REFS = {  # arm -> reference arm for the one-factor effect ('M0' = existing step-2 M0 seeds)
    **{a: 'M0' for a in ARMS if a.startswith('A_') and a != 'A_ema'}, 'A_ema': 'A_ref',
    'B_d3l20': 'M0', 'B_d3l24': 'B_d3l20', 'B_d3l16': 'B_d3l20', 'B_d2b12': 'B_d3l20', 'B_d2b10': 'B_d2b12',
    'B_d3b12': 'B_d2b12', 'B_d2l24': 'B_d3l24', 'B_d3l20dil': 'B_d3l20', 'B_v1like': 'v1', 'A_x2': 'M0',
    'B_v1sae': 'v1',
}


def cfg_of(arm):
    return {**BASE, **ARMS[arm]}


def gpu_line():
    return MT.gpu_line()


# ---------------------------------------------------------------------------------------------- prep
class Multi:
    """The four encoders; __call__(x at side S uint8) -> {set: (B, 1024, d) fp16} at the 32 grid."""

    def __init__(self, dev):
        import torch
        from transformers import AutoModel
        self.dev, self.m, self.gpu_s, self.snap = dev, {}, 0.0, {}
        for nm, (mid, rev, side, patch, dt, layers) in MODELS.items():
            kw = {'revision': rev} if rev else {}
            m = AutoModel.from_pretrained(mid, attn_implementation='sdpa', **kw).to(dev).eval().requires_grad_(False)
            blocks = m.layer if hasattr(m, 'layer') else m.encoder.layer
            normm = m.norm if hasattr(m, 'norm') else m.layernorm
            npre = 1 + int(getattr(m.config, 'num_register_tokens', 0) or 0)
            cap = {}
            for L in layers:
                blocks[L - 1].register_forward_hook(
                    lambda mod, i, o, L=L, cap=cap: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o))
            self.m[nm] = dict(m=m, side=side, patch=patch, dt=dt, layers=layers, norm=normm, npre=npre, cap=cap,
                              nblocks=len(blocks))
            self.snap[nm] = getattr(m.config, '_commit_hash', None)
        self.mean = torch.tensor(MEAN, device=dev)[None, :, None, None]
        self.std = torch.tensor(STD, device=dev)[None, :, None, None]

    def __call__(self, x):
        import torch
        import torch.nn.functional as F
        t0 = time.time()
        out = {}
        xf = x.float()
        with torch.no_grad():
            for nm, M in self.m.items():
                xi = F.interpolate(xf, size=(M['side'], M['side']), mode='bicubic', antialias=True,
                                   align_corners=False).clamp(0, 255)
                xin = (xi / 255 - self.mean) / self.std
                M['cap'].clear()
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=M['dt'] == 'bf16'):
                    last = M['m'](pixel_values=xin).last_hidden_state
                for L in M['layers']:
                    h = last if L == M['nblocks'] else M['norm'](M['cap'][L])
                    z = h[:, M['npre']:].float()
                    assert z.shape[1] == G32 * G32 and torch.isfinite(z).all(), (nm, L, z.shape)
                    out[f'{nm}{L}'] = z.half()
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        return out

    def last_check(self, x):
        """norm(hook of the last block) == last_hidden_state (the hook / LN convention used for the inner layers)."""
        import torch
        import torch.nn.functional as F
        res = {}
        with torch.no_grad():
            for nm, M in self.m.items():
                L = M['nblocks']
                blocks = M['m'].layer if hasattr(M['m'], 'layer') else M['m'].encoder.layer
                cap = {}
                hk = blocks[L - 1].register_forward_hook(
                    lambda mod, i, o, cap=cap: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))
                xi = F.interpolate(x[:2].float(), size=(M['side'], M['side']), mode='bicubic', antialias=True,
                                   align_corners=False).clamp(0, 255)
                xin = (xi / 255 - self.mean) / self.std
                last = M['m'](pixel_values=xin).last_hidden_state.float()
                hk.remove()
                res[nm] = float((M['norm'](cap['h']).float() - last).abs().max())
        return res


def cmd_prep(args):
    import torch
    import torch.nn.functional as F
    from src.eci.v2.pipeline import Series
    from src.eci.v2.video import Prefetch
    from src.eci.v2 import mask2 as M2
    dev = torch.device('cuda')
    dom, kind = args.dom, args.kind
    K = MT.KINDS[dom][kind]
    h = T2.h_of(dom, K['series'])
    enc = Multi(dev)
    se = Series(T2.series_cfg(dom, K['series']), dev, load_encoder=False)
    S = se.S
    cell = S // G32
    assert cell * G32 == S, (S, G32)
    Fr = MT.frame_list(dom, kind)
    obs_all = list(dict.fromkeys(Fr.obs.values))
    mine = set(obs_all[args.shard::args.nshard])
    Fr = Fr[Fr.obs.isin(mine)].reset_index(drop=True)
    items = []
    for o, g in Fr.groupby('obs', sort=False):
        idx = dict(zip(g.k.values, g.index.values))
        for w in sorted({T2.window_of(dom, k) for k in g.k.values}):
            kw = np.array([k for k in g.k.values if T2.window_of(dom, k) == w])
            for k0, n, every, ks in MT.runs_of(kw):
                items.append((o, w, k0, n, every, ks, [idx[k] for k in ks]))
    if args.selftest:
        items = items[:3]
    jobs = [((i,), se.job(se.V[o], k0, n, every)) for i, (o, w, k0, n, every, ks, _) in enumerate(items)]
    loc = N.local_dir() / f'v2d_{dom}_{kind}_s{args.shard}'
    loc.mkdir(parents=True, exist_ok=True)
    ftok = {s: open(loc / f'tok_{s}.f16', 'wb') for s in SETS}
    meta = {k: [] for k in ('tok_frame', 'tok_pos', 'fr_raw', 'fr_new', 'fr_app', 'fr_any')}
    recs, t0, bgc = [], time.time(), {}
    bs = 16 if S >= 1000 else 32
    checks = {}
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=se.run_job)
    for (i,), fr in pf:
        o, w, k0, n, every, ks, fidx = items[i]
        v = se.V[o]
        if (o, w) not in bgc:
            bgc.clear()
            bgc[(o, w)] = T2.load_pack(dom, K['series'], o, w, dev)
        pack = bgc[(o, w)]
        grid = k0 + every * np.arange(n)
        fr = fr[np.isin(grid, ks)]
        assert len(fr) == len(ks)
        for b in range(0, len(ks), bs):
            kb = ks[b:b + bs]
            with torch.no_grad():
                x, _ = se.frames(v, fr[b:b + bs], kb)
                core0 = se.mask.pixels(x, v['src_lo'] + kb)
                r = M2.cue(x, pack, core0, h)
                _, app = M2.appendage(r['pix'], h)
                anyp = core0 | (r['d'] > M2.T_LO)
                pool = lambda m: F.avg_pool2d(m.float()[:, None], cell).flatten(1)  # noqa: E731
                raw, frn, fapp, fany = pool(core0), pool(r['pix']), pool(app), pool(anyp)
                tight = raw >= 0.25
                dil = F.max_pool2d(tight.float().view(-1, 1, G32, G32), 3, 1, 1).flatten(1) > 0.5
                Z = enc(x)
                if not checks and args.shard == 0:
                    checks['last_layer_hook_vs_last_hidden_state_maxabs'] = enc.last_check(x)
                    if dom == 'mice':  # 32-grid tight fraction == 2 x 2 mean of the 64-grid M0 fractions
                        r64 = F.avg_pool2d(core0.float()[:, None], 16)
                        checks['pool_consistency_maxabs'] = float((F.avg_pool2d(r64, 2).flatten(1) - raw).abs().max())
            for ii in range(len(kb)):
                pos = torch.nonzero(dil[ii]).flatten()
                fid = len(recs)
                for s_ in SETS:
                    ftok[s_].write(Z[s_][ii, pos].cpu().numpy().tobytes())
                meta['tok_frame'].append(np.full(len(pos), fid, np.int32))
                meta['tok_pos'].append(pos.cpu().numpy().astype(np.int16))
                for nm, t_ in (('fr_raw', raw), ('fr_new', frn), ('fr_app', fapp), ('fr_any', fany)):
                    meta[nm].append(t_[ii, pos].cpu().numpy().astype(np.float16))
                r0 = Fr.iloc[fidx[b + ii]]
                rec = {'obs': o, 'k': int(kb[ii]), 'ref_f': int(r0.ref_f), 'n_tight': int(tight[ii].sum()),
                       'n_dil': int(len(pos))}
                for c_ in ('row', 'role', 'series'):
                    if c_ in Fr:
                        rec[c_] = r0[c_]
                recs.append(rec)
        if i % 200 == 0:
            el = time.time() - t0
            log(f'prep {dom} {kind} s{args.shard}: run {i}/{len(items)}, {len(recs):,} frames, {el:.0f}s '
                f'({len(recs) / max(el, 1):.1f} fps; GPU busy {enc.gpu_s:.0f}s; decode thread-s {pf.dec_s:.0f}); '
                f'{gpu_line()}')
    for f_ in ftok.values():
        f_.close()
    for k_, x_ in meta.items():
        np.save(loc / f'{k_}.npy', np.concatenate(x_))
    Fm = pd.DataFrame(recs)
    Fm.to_parquet(loc / 'frames.parquet')
    # same frames, same order as the step-2 store shard (identity of the frame sets)
    ref = T2.BEE / dom / kind / f's{args.shard}' / 'frames.parquet'
    if ref.exists() and not args.selftest and args.nshard == len(list((T2.BEE / dom / kind).glob('s*'))):
        Fr2 = pd.read_parquet(ref)
        same = len(Fr2) == len(Fm) and (Fr2.obs.values == Fm.obs.values).all() and (Fr2.k.values == Fm.k.values).all()
        checks['same_frames_as_step2_shard'] = bool(same)
        assert same, f'frame order differs from {ref}'
    el = time.time() - t0
    info = {'dom': dom, 'kind': kind, 'shard': args.shard, 'nshard': args.nshard, 'S': S, 'n_frames': len(Fm),
            'n_stored_tokens': int(Fm.n_dil.sum()), 'tight_per_frame': float(Fm.n_tight.mean()),
            'dil_per_frame': float(Fm.n_dil.mean()), 'wall_s': el, 'gpu_s': enc.gpu_s, 'fps_wall': len(Fm) / el,
            'decode_thread_s': pf.dec_s, 'gpu': torch.cuda.get_device_name(0), 'snapshots': enc.snap,
            'checks': checks, 'dims': {s: int(Z[s].shape[-1]) for s in SETS}}
    log(json.dumps(info, default=float))
    od = OUT / dom / 'prep'
    od.mkdir(parents=True, exist_ok=True)
    tag = '_selftest' if args.selftest else ''
    (od / f'{kind}_s{args.shard}{tag}.json').write_text(json.dumps(info, indent=1, default=float))
    (loc / 'info.json').write_text(json.dumps(info, default=float))
    if not args.selftest:
        R4.atomic_copy(loc, BEE / dom / kind / f's{args.shard}')


# ---------------------------------------------------------------------------------------------- stores
class Store:
    """A 32-grid token set on frames of one or more kinds: the FZ.TokStore interface (F, tf, pos, T, starts, nf, G,
    dev, chunks) plus fd / app / any / raw per token for neuron counting."""

    def __init__(self, dom, kinds, tset, mask, dev, on_gpu=True):
        Fs, parts, o = [], [], 0
        d = None
        for kind in kinds:
            root = BEE / dom / kind
            for sp in sorted([p for p in root.glob('s*') if not p.name.endswith('.tmp')], key=lambda p: int(p.name[1:])):
                info = json.loads((sp / 'info.json').read_text())
                d = info['dims'][tset]
                F_ = pd.read_parquet(sp / 'frames.parquet')
                tf = np.load(sp / 'tok_frame.npy')
                raw = np.load(sp / 'fr_raw.npy').astype(np.float32)
                keep = raw >= 0.25 if mask == 'tight' else np.ones(len(tf), bool)
                mm = np.memmap(sp / f'tok_{tset}.f16', np.float16, 'r', shape=(len(tf), d))
                parts.append(dict(tf=tf[keep] + o, pos=np.load(sp / 'tok_pos.npy').astype(np.int64)[keep],
                                  raw=raw[keep], **{k: np.load(sp / f'fr_{k}.npy').astype(np.float32)[keep]
                                                    for k in ('new', 'app', 'any')},
                                  tok=np.asarray(mm[keep])))
                Fs.append(F_)
                o += len(F_)
        assert parts, (dom, kinds)
        self.F = pd.concat(Fs, ignore_index=True)
        cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
        self.tf, self.pos, self.raw, self.fd, self.app, self.any = (cat(k) for k in ('tf', 'pos', 'raw', 'new', 'app',
                                                                                      'any'))
        self.tok = cat('tok')
        self.d, self.nf, self.G, self.dev = d, len(self.F), G32, dev
        self.starts = np.r_[0, np.cumsum(np.bincount(self.tf, minlength=self.nf))]
        self.dist = np.zeros(len(self.tf), np.int8)
        self.T = R2.tokens_to(__import__('torch'), self.tok, dev) if on_gpu else None

    chunks = FZ.TokStore.chunks


def e0_store(dom, kind, dev, on_gpu=True):
    St = T2.arm_store(T2.Union(dom, kind), 'M0', dev, on_gpu=on_gpu)
    St.d = 1024
    return St


def store_of(dom, kind, cfg, dev, on_gpu=True):
    if cfg['src'] == 'E0full' and kind.startswith('pool'):  # the full reference pool (both burst halves)
        St = FZ.TokStore(MT.KINDS[dom][kind]['ref'], 0, dev, on_gpu=on_gpu)
        St.d = 1024
        return St
    if cfg['src'] in ('E0', 'E0full'):
        return e0_store(dom, kind, dev, on_gpu)
    return Store(dom, (kind,), cfg['src'], cfg.get('mask', 'tight'), dev, on_gpu)


def inputs(St, sel):
    """fp16 cuda rows of the tokens where sel (numpy bool per token) holds."""
    import torch
    ix = np.flatnonzero(sel)
    out = torch.empty((len(ix), St.d), dtype=torch.float16, device=St.dev)
    for a in range(0, len(ix), 1_000_000):
        out[a:a + 1_000_000] = torch.from_numpy(np.ascontiguousarray(St.tok[ix[a:a + 1_000_000]])).to(St.dev)
    return out


def codes(St, sae, norm):
    """frame codes (nf, W) fp16 cuda = max over the store's kept tokens (threshold mode)."""
    import torch
    W = sae.n_latents
    X = torch.zeros((St.nf, W), dtype=torch.float16, device=St.dev)
    fp = max(4, min(48, int(2**27 // W // 160)))
    with torch.no_grad():
        for f0, f1, a, b, z in St.chunks(fp):
            fl = torch.from_numpy(St.tf[a:b] - f0).to(St.dev)
            c = sae.encode(norm(z.float()), mode='threshold')
            X[f0:f1] = torch.zeros((f1 - f0, W), device=St.dev).index_reduce_(0, fl, c.float(), 'amax',
                                                                               include_self=True).half()
    return X


# ---------------------------------------------------------------------------------------------- train
def sae_dir(dom, arm, s):
    return BEE / dom / 'sae' / f'{arm}_s{s}'


def train_one(torch, cfg, seed, Xtr, Xca, Xev, od, dev):
    """= scripts/eci/nnnt/r3.py train_arm (uniform sampler, 'id' prep) with: width / prefixes / k / steps from cfg,
    norm 'global' (TokenNorm, as r3) or 'z' (per-dimension mean / sd), and the training EMA threshold recorded."""
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    od.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    din = Xtr.shape[1]
    g = np.random.default_rng(seed)
    ii = torch.from_numpy(np.sort(g.choice(len(Xtr), min(250_000, len(Xtr)), replace=False))).to(dev)
    init = Xtr[ii].float()
    norm = TokenNorm(din).to(dev).fit_blocks(init, [din], seed=seed)
    if cfg['norm'] == 'z':
        with torch.no_grad():
            norm.mean.copy_(init.mean(0))
            norm.scale = (1.0 / (init.std(0) + 1e-6)).to(dev)
    sae = MatryoshkaBatchTopKSAE(d_in=din, n_latents=cfg['W'], prefixes=tuple(cfg['prefixes']), k=cfg['k'],
                                 seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    opt = torch.optim.Adam(sae.parameters(), lr=5e-4, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    Nn, bs, steps = len(Xtr), 4096, cfg['steps']
    perm, p, hist, t0 = torch.randperm(Nn, generator=gen, device=dev), 0, [], time.time()
    for step in range(steps):
        if p + bs > Nn:
            perm, p = torch.randperm(Nn, generator=gen, device=dev), 0
        b = perm[p:p + bs]
        p += bs
        logs, _ = _train_step(sae, opt, norm(Xtr[b].float()), step, steps, 5e-4, 500, 1.0)
        if step % 5000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            hist.append(logs)
    sae.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    tt = time.time() - t0
    th_ema = float(sae.threshold)
    full = lambda x: norm(x.float())  # noqa: E731
    th, _ = calibrate_threshold(sae, Xca, norm=full)
    set_threshold(sae, th)
    stt = sparsity_stats(sae, Xev, th, norm=full)
    save_checkpoint(od / 'sae_theta.pt', sae, norm, extra={'config': 'v2_diag', 'cfg': cfg, 'seed': seed,
                                                            'theta_ema': th_ema})
    return {'train_time_s': tt, 'steps_per_s': steps / tt, 'n_train_tokens': Nn, 'theta_star': th, 'theta_ema': th_ema,
            'eval_fve': stt['fve'], 'eval_l0': stt['l0'],
            'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()), 'history': hist}


def pool_kinds(dom):
    return ('pool',) if dom == 'mice' else ('pool_v2', 'pool_v3')


def cmd_train(args):
    import torch
    dev = torch.device('cuda')
    dom, arm = args.dom, args.arm
    cfg = cfg_of(arm)
    if 'reuse' in cfg or 'sae_path' in cfg:
        return
    seeds = FZ.parse_seeds(args.seeds, SEEDS)
    todo = [s for s in seeds if not (sae_dir(dom, arm, s) / 'sae_theta.pt').exists()]
    if not todo:
        return
    sts = [store_of(dom, k, cfg, dev, on_gpu=False) for k in pool_kinds(dom)]
    X = {}
    for r in ('calib', 'eval'):
        X[r] = torch.cat([inputs(St, St.F.role.values[St.tf] == r) for St in sts]).cpu()
    tr_sel = []
    for St in sts:
        m = St.F.role.values[St.tf] == 'train'
        tr_sel.append(m)
    log(f'{dom} {arm}: calib {len(X["calib"]):,}, eval {len(X["eval"]):,}; {gpu_line()}')
    od = OUT / dom / 'train'
    od.mkdir(parents=True, exist_ok=True)
    for s in todo:
        sel = []
        for St, m in zip(sts, tr_sel):
            if cfg['frac'] < 1:  # seeded half of the TRAINING FRAMES (all tokens of a kept frame)
                fk = np.random.default_rng(s).random(St.nf) < cfg['frac']
                m = m & fk[St.tf]
            sel.append(m)
        Xtr = torch.cat([inputs(St, m) for St, m in zip(sts, sel)])
        ti = train_one(torch, cfg, s, Xtr, X['calib'], X['eval'], sae_dir(dom, arm, s), dev)
        rec = {k: ti[k] for k in ('train_time_s', 'n_train_tokens', 'theta_star', 'theta_ema', 'eval_fve', 'eval_l0',
                                  'dead_at_end_train')}
        log(f'{dom} {arm}_s{s}: {ti["train_time_s"]:.0f}s, {ti["n_train_tokens"]:,} train tokens, eval FVE '
            f'{ti["eval_fve"]:.4f}, L0 {ti["eval_l0"]:.2f}, dead {ti["dead_at_end_train"]}, theta* '
            f'{ti["theta_star"]:.3f} / ema {ti["theta_ema"]:.3f}; {gpu_line()}')
        (od / f'{arm}_s{s}.json').write_text(json.dumps(rec, indent=1, default=float))
        del Xtr
        torch.cuda.empty_cache()


def load_arm_sae(dom, arm, s, dev):
    from src.eci.sae import load_sae
    from src.eci.sae_calibrate import set_threshold
    cfg = cfg_of(arm)
    src_arm = cfg.get('reuse', arm)
    if 'sae_path' in cfg:  # a fixed deployed SAE (its own stored threshold)
        sae, norm, _ = load_sae(REPO / cfg['sae_path'][dom], dev)
        return sae, norm
    sae, norm, _ = load_sae(sae_dir(dom, src_arm, s) / 'sae_theta.pt', dev)
    if cfg['readout'] == 'ema':
        set_threshold(sae, float(json.loads((OUT / dom / 'train' / f'{src_arm}_s{s}.json').read_text())['theta_ema']))
    return sae, norm


def fve_of(dom, arm, s):
    f = OUT / dom / 'train' / f'{cfg_of(arm).get("reuse", arm)}_s{s}.json'
    return json.loads(f.read_text()).get('eval_fve', np.nan) if f.exists() else np.nan


# ---------------------------------------------------------------------------------------------- score
def scored_stores(dom, cfg, dev):
    if dom == 'mice':
        return {'v1': store_of('mice', 'diag', cfg, dev)}
    return {ser: store_of('ants', f'scored_{ser}', cfg, dev) for ser in ('v2', 'v3')}


def cmd_score(args):
    import torch
    dev = torch.device('cuda')
    dom, arm = args.dom, args.arm
    cfg = cfg_of(arm)
    seeds = FZ.parse_seeds(args.seeds, SEEDS)
    od = OUT / dom / 'ab'
    od.mkdir(parents=True, exist_ok=True)
    src_arm = cfg.get('reuse', arm)
    if 'sae_path' in cfg:
        seeds = [] if (od / f'{arm}_s{SEEDS[0]}.json').exists() else [SEEDS[0]]
    else:
        seeds = [s for s in seeds if not (od / f'{arm}_s{s}.json').exists()
                 and (sae_dir(dom, src_arm, s) / 'sae_theta.pt').exists()]
    if not seeds:
        return
    sts = scored_stores(dom, cfg, dev)
    if dom == 'mice':
        St = sts['v1']
        Y, sets, half, _, _ = MT.mice_sets(St)
        Yg = torch.from_numpy(Y).to(dev)
    else:
        lab = {}
        for ser, St_ in sts.items():
            Y, hh, labs = MT.ants_set(St_, ser)
            lab[ser] = (torch.from_numpy(Y).to(dev), hh, labs)
    for s in seeds:
        sae, norm = load_arm_sae(dom, arm, s, dev)
        if dom == 'mice':
            ab = RA.ab_mice(codes(St, sae, norm), Yg, half['all'], St, FZ.MLAB)
            log(f'mice {arm}_s{s}: R_atlas full nn {ab["full"]["nose_nose"]["R_atlas"]:.4f} nt '
                f'{ab["full"]["nose_tail"]["R_atlas"]:.4f} | 256 nn {ab["256"]["nose_nose"]["R_atlas"]:.4f} nt '
                f'{ab["256"]["nose_tail"]["R_atlas"]:.4f}; {gpu_line()}')
        else:
            ab = {}
            for ser, St_ in sts.items():
                Yg_, hh, labs = lab[ser]
                ab[ser] = RA.ab_ants(codes(St_, sae, norm), Yg_, hh, labs)
            log(f'ants {arm}_s{s}: R_atlas full v2 groom {ab["v2"]["full"]["groom_any"]["R_atlas"]:.4f} v3 groom '
                f'{ab["v3"]["full"]["groom_any"]["R_atlas"]:.4f} lid {ab["v3"]["full"]["on_lid_any"]["R_atlas"]:.4f} '
                f'| 256 lid {ab["v3"]["256"]["on_lid_any"]["R_atlas"]:.4f}; {gpu_line()}')
        ab['eval_fve'] = fve_of(dom, arm, s)
        (od / f'{arm}_s{s}.json').write_text(json.dumps(ab, indent=1, default=float))
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- probe (ceilings)
def cmd_probe(args):
    import torch
    from src.eci.v2.score import crossfit
    dev = torch.device('cuda')
    dom, tset = args.dom, args.set
    cfg = {'src': tset, 'mask': 'tight'}
    sts = scored_stores(dom, cfg, dev)
    res = {'set': tset, 'note': 'T-max: one linear direction on standardised tokens, frame max (r2 fit_probe max); '
                               'chan: best single standardised dimension (either sign), frame max; atlas split'}
    jobs = []
    if dom == 'mice':
        St = sts['v1']
        Y, sets, half, _, _ = MT.mice_sets(St)
        keep, hh = RA.mice_window(St, half['all'])
        jobs.append(('mice', St, keep, Y[keep][:, :2], hh['atlas'], ['nose_nose', 'nose_tail'], hh))
    else:
        for ser, St_ in sts.items():
            Y, hh, labs = MT.ants_set(St_, ser)
            want = ['groom_any'] + (['on_lid_any'] if ser == 'v3' else [])
            jobs.append((ser, St_, np.arange(St_.nf), Y[:, [labs.index(w) for w in want]], hh['r4'], want, hh))
    for ser, St, keep, Yk, h_at, labs, hall in jobs:
        # frame subset (atlas window for mice) -> local token bags
        fsel = np.zeros(St.nf, bool)
        fsel[keep] = True
        newid = -np.ones(St.nf, np.int64)
        newid[keep] = np.arange(len(keep))
        tsel = fsel[St.tf]
        T = inputs(St, tsel)
        tf = newid[St.tf[tsel]]
        ii = torch.from_numpy(np.random.default_rng(0).choice(len(T), min(1_000_000, len(T)), replace=False)).to(dev)
        mu, sd = T[ii].float().mean(0), T[ii].float().std(0) + 1e-3
        for a in range(0, len(T), 500_000):
            T[a:a + 500_000] = ((T[a:a + 500_000].float() - mu) / sd).half()
        nfk = len(keep)
        bags = R2.Bags(torch, T, tf, nfk, dev)
        # best single channel: frame max of +z and of -z (shifted >= 0 for the fp16 scorer)
        tfg = torch.from_numpy(tf).to(dev)
        mx = torch.full((nfk, St.d), -1e4, device=dev)
        mn = torch.full((nfk, St.d), -1e4, device=dev)
        for a in range(0, len(T), 500_000):
            z = T[a:a + 500_000].float()
            mx.index_reduce_(0, tfg[a:a + 500_000], z, 'amax', include_self=True)
            mn.index_reduce_(0, tfg[a:a + 500_000], -z, 'amax', include_self=True)
        empty = bags.cnt == 0
        Xc = torch.cat([mx, mn], 1)
        Xc[empty] = float('nan')
        Xc = Xc - torch.nan_to_num(Xc, nan=1e4).min(0).values
        Xc = torch.nan_to_num(Xc, nan=0.0).clamp(0, 6e4).half()
        Ykg = torch.from_numpy(Yk).to(dev)
        cf = crossfit(Xc, Ykg, {'atlas': h_at, 'rnd_atlas': h_at}, labs)  # crossfit's mean needs an rnd* key
        out = {}
        for li, lb in enumerate(labs):
            y = Yk[:, li].astype(bool)
            recs = []
            for a_, b_ in ((0, 1), (1, 0)):
                tr = torch.from_numpy(np.flatnonzero(h_at == a_)).to(dev)
                te = np.flatnonzero(h_at == b_)
                f = R2.fit_probe(torch, 'max', bags, None, y, tr, dev, steps=1500)
                recs.append(R2.metrics(f(torch.from_numpy(te).to(dev)), y[te]))
            out[lb] = {'tmax_R': float(np.mean([r['R'] for r in recs])), 'tmax_A': float(np.mean([r['A'] for r in recs])),
                       'chan_R': cf['splits']['atlas'][lb]['R'], 'chan_A': cf['splits']['atlas'][lb]['A'],
                       'base': float(y.mean()), 'n': int(len(y))}
            # robustness (added after the first probe run, reported only): T-max on the random halves rnd0-4 too
            rr = []
            for sn in [f'rnd{i}' for i in range(5)]:
                hs = hall[sn]
                for a_, b_ in ((0, 1), (1, 0)):
                    tr = torch.from_numpy(np.flatnonzero(hs == a_)).to(dev)
                    te = np.flatnonzero(hs == b_)
                    f = R2.fit_probe(torch, 'max', bags, None, y, tr, dev, steps=1500)
                    rr.append(R2.metrics(f(torch.from_numpy(te).to(dev)), y[te])['R'])
            out[lb]['tmax_R_rnd5'] = float(np.mean(rr))
            out[lb]['tmax_R_rnd5_sd'] = float(np.std(rr))
            log(f'probe {dom} {ser} {tset} {lb}: T-max R~ {out[lb]["tmax_R"]:.4f} AUROC {out[lb]["tmax_A"]:.4f} | '
                f'best channel R~ {out[lb]["chan_R"]:.4f}; {gpu_line()}')
        res[ser] = out
        del T, bags, Xc, mx, mn
        torch.cuda.empty_cache()
    od = OUT / dom / 'probe5'
    od.mkdir(parents=True, exist_ok=True)
    (od / f'{tset}.json').write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- neurons
def cmd_neurons(args):
    """Background / appendage / odour-box neuron counts (scripts/eci/v2_mask2/mask2_test.py definitions, selection half
    rnd0 == 0); no sheets."""
    import torch
    dev = torch.device('cuda')
    dom, arm = args.dom, args.arm
    cfg = cfg_of(arm)
    sts = scored_stores(dom, cfg, dev)
    if dom == 'mice':
        Y, sets, half, uidx, SPL = MT.mice_sets(sts['v1'])
        hA = {'v1': SPL['rnd0'][uidx] == 0}
        odor = T2.odour_corner(sts['v1'].G * 16)
    else:
        hA = {ser: MT.ants_set(St, ser)[1]['rnd0'] == 0 for ser, St in sts.items()}
        odor = None
    od = OUT / dom / 'neurons'
    od.mkdir(parents=True, exist_ok=True)
    for s in FZ.parse_seeds(args.seeds, SEEDS):
        f = od / f'{arm}_s{s}.json'
        if f.exists() or not (sae_dir(dom, cfg.get('reuse', arm), s) / 'sae_theta.pt').exists():
            continue
        sae, norm = load_arm_sae(dom, arm, s, dev)
        accs = {ser: T2.token_stats(St, sae, norm, hA[ser], odor)[0] for ser, St in sts.items()}
        acc = {k: sum(accs[ser][k] for ser in sts) for k in accs[next(iter(sts))]}
        ok = acc['fire'] >= T2.MIN_FIRE
        off = np.where(ok, acc['off'] / np.maximum(acc['hit'], 1), np.nan)
        odr = np.where(ok, acc['odor'] / np.maximum(acc['hit'], 1), np.nan)
        app = np.where(ok & (acc['den'] > 0), acc['num'] / np.maximum(acc['den'], 1e-12), np.nan)
        r = {'n_scored': int(ok.sum()), 'n_background': int(np.nansum(off > 0.5)),
             'share_background': float(np.nansum(off > 0.5) / max(ok.sum(), 1)),
             'n_app': int(np.nansum(app > T2.APP_NEURON)), 'n_odor': int(np.nansum(odr > T2.ODOR_FRAC))}
        log(f'{dom} {arm}_s{s}: background {r["n_background"]}/{r["n_scored"]}, appendage {r["n_app"]}, odour '
            f'{r["n_odor"]}; {gpu_line()}')
        f.write_text(json.dumps(r, indent=1, default=float))
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- decide
DCELLS = [('mice', None, 'nose_nose'), ('mice', None, 'nose_tail'), ('ants', 'v2', 'groom_any'),
          ('ants', 'v3', 'groom_any'), ('ants', 'v3', 'on_lid_any')]
RCELLS = [('mice', None, 'nn_mutual'), ('mice', None, 'np_directional')]


def cell_vals(dom, arm, ser, lb, p, key='R_atlas', seeds=None):
    """per-seed values of one cell -> {seed: value}."""
    if arm == 'M0':
        d, ss = T2.OUT / dom / 'ab', T2.SEEDS[dom]
    else:
        d, ss = OUT / dom / 'ab', seeds or SEEDS
    out = {}
    for s in ss:
        f = d / f'{arm}_s{s}.json'
        if f.exists():
            r = json.loads(f.read_text())
            out[s] = (r[p] if ser is None else r[ser][p])[lb][key]
    return out


def v1_val(dom, ser, lb, p, key='R_atlas'):
    refs = json.loads((T2.OUT / 'ab_refs.json').read_text())
    r = refs[dom] if dom == 'mice' else refs[dom][ser]
    return r['v1'][p][lb][key]


def welch_ci(a, b):
    from src.eci.v2.score import welch
    return welch(a, b)


def cmd_decide(args):
    res, L = {}, []
    L.append('| arm | ref | cell | prefix | arm mean ± sd (n) | arm - ref (Welch 95% CI) | arm - v1 |')
    L.append('|---|---|---|---|---|---|---|')
    for arm in ARMS:
        ref = REFS[arm]
        for dom, ser, lb in DCELLS + RCELLS:
            for p in ('full', '256'):
                a = cell_vals(dom, arm, ser, lb, p)
                if len(a) < 2:
                    continue
                v1 = v1_val(dom, ser, lb, p)
                av = np.array(list(a.values()))
                rec = {'mean': float(av.mean()), 'sd': float(av.std(ddof=1)), 'n': len(av), 'v1': v1,
                       'minus_v1': RA.tci(av - v1)}
                if ref == 'v1':
                    rec['minus_ref'] = rec['minus_v1']
                elif ref == 'A_ref':  # paired by seed
                    b = cell_vals(dom, ref, ser, lb, p)
                    common = sorted(set(a) & set(b))
                    if len(common) >= 2:
                        rec['minus_ref'] = RA.tci(np.array([a[s] - b[s] for s in common]))
                else:
                    b = cell_vals(dom, ref, ser, lb, p)
                    if len(b) >= 2:
                        rec['minus_ref'] = list(welch_ci(av, np.array(list(b.values()))))
                nm = f'{dom}{" " + ser if ser else ""} {lb}'
                res.setdefault(arm, {})[f'{nm} | {p}'] = rec
                mr = rec.get('minus_ref')
                L.append(f'| {arm} | {ref} | {nm} | {p} | {rec["mean"]:.4f} ± {rec["sd"]:.4f} ({rec["n"]}) | '
                         + (f'{mr[0]:+.4f} [{mr[1]:+.4f}, {mr[2]:+.4f}]' if mr else '-') + ' | '
                         f'{rec["minus_v1"][0]:+.4f} [{rec["minus_v1"][1]:+.4f}, {rec["minus_v1"][2]:+.4f}] |')
    # M0 reference rows
    for dom, ser, lb in DCELLS:
        for p in ('full', '256'):
            a = np.array(list(cell_vals(dom, 'M0', ser, lb, p).values()))
            res.setdefault('M0', {})[f'{dom}{" " + ser if ser else ""} {lb} | {p}'] = {
                'mean': float(a.mean()), 'sd': float(a.std(ddof=1)), 'n': len(a), 'v1': v1_val(dom, ser, lb, p)}
    probes = {}
    for dom in ('mice', 'ants'):
        for f in sorted((OUT / dom / 'probe5').glob('*.json')) if (OUT / dom / 'probe5').exists() else []:
            probes.setdefault(dom, {})[f.stem] = json.loads(f.read_text())
    res['_probes'] = probes
    (OUT / 'ablation.json').write_text(json.dumps(res, indent=1, default=float))
    P = ['', '| token set | cell | T-max R~ (AUROC) | best channel R~ |', '|---|---|---|---|']
    for dom, d in probes.items():
        for tset, r in d.items():
            for ser, cells in r.items():
                if not isinstance(cells, dict):
                    continue
                for lb, c in cells.items():
                    P.append(f'| {tset} | {dom} {ser} {lb} | {c["tmax_R"]:.4f} ({c["tmax_A"]:.3f}) | {c["chan_R"]:.4f} |')
    (OUT / 'ablation_table.md').write_text('\n'.join(L + P))
    print('\n'.join(L + P))


def cmd_probes(args):
    """--set a,b,c: one probe per token set in sequence."""
    import torch
    for tset in args.set.split(','):
        a = argparse.Namespace(**{**vars(args), 'set': tset})
        if (OUT / args.dom / 'probe5' / f'{tset}.json').exists():
            continue
        cmd_probe(a)
        torch.cuda.empty_cache()


def cmd_arm(args):
    """train then score (one job); --arms a,b,c runs several arms in sequence."""
    import torch
    for arm in (args.arms or args.arm).split(','):
        args.arm = arm
        cmd_train(args)
        torch.cuda.empty_cache()
        cmd_score(args)
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('prep', 'train', 'score', 'arm', 'probe', 'neurons', 'decide'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--kind')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--arm')
    p.add_argument('--arms')
    p.add_argument('--set')
    p.add_argument('--seeds')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    {'prep': cmd_prep, 'train': cmd_train, 'score': cmd_score, 'arm': cmd_arm, 'probe': cmd_probes, 'neurons': cmd_neurons, 'decide': cmd_decide}[a.step](a)


if __name__ == '__main__':
    main()
