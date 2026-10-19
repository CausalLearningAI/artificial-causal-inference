"""
RESOLUTION SWEEP (2026-10-19): justify (or change) the v2 input size / token grid of the ECI pipeline with one clean,
pre-registered sweep. Current rule (scripts/eci/rollout_r4.py l.11-13): input sized to ~10 patches per body length,
never above native, multiple of 16. Only the input size changes between arms; every other part is a fixed recipe.

PATCHES PER BODY LENGTH (body length = results/vision/animal_scale/scale.md, label-free, native px; patches = body x
S / native / patch): mice v1 303 px at 2064 native; ants v2 64 px at 770; ants v3 76 px at 824.

==================================================================================================================
PRE-REGISTRATION (written 2026-10-19 ~20:30 CEST, BEFORE any token store, SAE or score of this sweep existed. The only
numbers seen beforehand are those quoted in the task: round-1 / round-2 nn/nt tables (results/vision/eci_nnnt/
table.md, r2/table.md: A3 / S0 at 1024 primary R~ 0.103 ± 0.039 over 3 seeds; seed sd ~0.03-0.04), CalMS21 32 vs 64,
mice round-1 A2 vs A4. The round-2 2048 px zoom arm's SCORES were NOT seen (its sweep was still running); only its
token / pool stores are reused here, read-only.)

MICE v1 (nose_nose = Y_nn | Y_np, nose_tail = Y_nt; secondary nn_mutual = Y_nn, np_directional = Y_np)
    RECIPE A (primary) = round-1 winner A3 = round-2 'S0' at resolutions != 1024: DINOv3-L/16 block 20 (bf16, SDPA),
        odour-corner rotation, TIGHT mask (token kept iff >= 25% of its patch is animal pixels by the lab pixel cue),
        no change channel (x = z_t), Matryoshka BatchTopK 512 (prefixes 128/256/512), k = 16, 6000 steps x 4096 tokens,
        theta* on the calibration videos (mean L0 = k), frame value = max over kept tokens.
    RECIPE B (secondary, robustness of the ranking to dictionary size) = round-2 'S2': Matryoshka 2048 (128/512/2048),
        k = 16, 30,000 steps; else as A.
    TRAIN POOL (identical frames for every arm) = round-2 2048 pool definition: 8 bursts of 10 consecutive 5 fps frames
        per video at random starts (rng 7), all 432 videos, diagnosis-grid frames removed (31,104 frames). All kept
        tokens are used (no cap); every arm runs the same number of SAE steps.
    SCORED FRAMES (identical for every arm) = every 20th 5 fps frame of the 144 annotated videos (k = fps1_off + 20 j;
        43,200 frames) = the round-2 2048 diagnosis set. Frame sets: 'all' (primary) and 'consistent' = videos of the
        annotators SG / SD / MR / CP (secondary; reported, never decides).
    ARMS (DINOv3-L/16): d3_512 (32 grid, 4.7 patches / body), d3_768 (48, 7.0), d3_1024 (64, 9.4 = current),
        d3_1408 (88, 12.9), d3_2048 (128, 18.8; round-2 stores reused). PATCH-SIZE ARMS (DINOv2-L/14 block 20 of 24,
        after the final LayerNorm, CLS dropped): d2_672 (48 grid, 7.0 patches / body = d3_768) and d2_896 (64 grid,
        9.4 = d3_1024), same mask rule at the 14 px patch, same recipe A.
    METRICS (round-1 scorer nnnt.score_set): R~ = cross-fitted normalised R-precision (best neuron picked on one half of
        the 24 pools, scored on the other, both directions, mean over the 10 random 12 / 12 pool splits) and the test
        AUROC of that neuron; primary per SAE seed = mean(R~ nose_nose, R~ nose_tail).
    SEEDS: recipe A seeds 0-23 for every arm (SAE training is ~1 min; the cost is the token stores): seeds 0-11 =
        SCREENING, seeds 12-23 = fresh CONFIRMATION. Recipe B seeds 0-5.
    DECISION RULE (recipe A, frames 'all'):
        1. Screening (seeds 0-11, DINOv3 arms): best = arm with the highest mean primary R~. For every arm a: delta_a =
           mean(a) - mean(best) with a Welch 95% CI. Candidate = the SMALLEST arm with delta_a >= -0.01 AND a CI that
           includes 0 (i.e. within 0.01 R~ of the best and not significantly worse).
        2. Confirmation (fresh seeds 12-23), candidate vs d3_1024 (the current rule):
           - candidate LARGER than 1024: adopt it iff (candidate - 1024) primary R~ >= +0.01 with the Welch 95% CI
             excluding 0, and neither nose_nose nor nose_tail R~ drops by more than 0.02;
           - candidate SMALLER than 1024: adopt it iff (candidate - 1024) primary R~ >= -0.01 and the CI lower bound
             >= -0.03 (non-inferiority margin 0.03 R~), and neither label's R~ drops by more than 0.02;
           - otherwise (or candidate = 1024) keep 1024.
        3. Patch size (seeds 0-23 pooled, 24 vs 24): the rule is stated in PATCHES (not pixels) iff the DINOv2/14 grid
           effect (d2_896 - d2_672) and the DINOv3/16 grid effect at the same patches per body (d3_1024 - d3_768) have
           the same verdict (WIN: delta >= +0.01 and CI excludes 0; LOSES: <= -0.01 and CI excludes 0; else NO
           DIFFERENCE). DINOv3 vs DINOv2 at matched patches (d3_1024 - d2_896, d3_768 - d2_672) is reported with the
           same verdicts (it mixes encoder and pixels; descriptive).
        Recipe B and the 'consistent' frame set are reported per arm with the same deltas; if they rank differently
        from recipe A / 'all' the report says so, they do not change the decision.

ANTS (v2 + v3, grooming labels; groom_any = Y_Y2F | Y_B2F primary, groom_yellow, groom_blue secondary)
    RECIPE = the r4 pooled ants recipe (the deployed ants model): DINOv3-L/16 block 20, the lab 'ants' mask rule on the
        512 px dataset JPG (32 grid, projected to the G grid by patch centre), change channel x = [z_t, z_t - z_{t-10}]
        ('d2', primary) and z_t only ('d0', secondary) from the same pass, Matryoshka BatchTopK 512 (128/256/512), k 16,
        6000 steps x 4096, theta* on calibration videos; frame value = max over kept tokens. Source offset 2 (r4p).
    TRAIN FRAMES = the r4 pooled prepass frames of v2 and v3 (5,320 each: 4,864 train / 304 calib / 152 eval), each with
        its t - 10 partner; a cap of 6M training tokens (uniform, seeded) applies identically to every arm (it binds
        only on the largest arm).
    SCORED FRAMES = every 10th 5 fps frame (k = 0, 10, 20, ...; 2 s apart, so each frame's t - 10 partner is the
        previous scored frame) of the labelled range of every v2 (44) / v3 (107) video of the r4 pooled plans.
    ARMS: aN = native-capped (v2 768 px = 4.0 patches / body, v3 832 = 4.8; = the current rule), a7 = UPSAMPLED to ~7
        (v2 1344 = 7.0, v3 1216 = 7.0), a10 = UPSAMPLED to ~10 (v2 1920 = 10.0, v3 1728 = 10.0). Upsampling adds no
        image detail: these arms test TOKEN DENSITY (more, smaller patches over the same pixels), not resolution.
    METRICS: per series, the round-1 scorer (cross-fitted best neuron over two halves of the VIDEOS, 10 random splits,
        rng 1000 + s): R~ and test AUROC per label. Primary per SAE seed = mean over v2 and v3 of R~(groom_any), d2.
    SEEDS 0-23 per variant (0-11 screening, 12-23 confirmation).
    DECISION RULE: default = native (current rule). Best upsampled arm on seeds 0-11 (higher mean primary R~ of a7 /
        a10); adopt it iff on the fresh seeds 12-23 (best - aN) primary R~ >= +0.01 with the Welch 95% CI excluding 0
        AND it also holds on each series separately with delta >= 0 (a gain carried by one series is reported as such
        and not adopted).

COST (all arms): encoder tokens per frame (G^2), SAE tokens per frame after the mask, encode frames / GPU-s, GPU-hours
from sacct; GPU utilisation logged every 60 s by the job wrapper ([gpu] lines).
==================================================================================================================

Steps (scripts/eci/resolution/res.sh runs each in Slurm; heavy inputs / outputs staged on $LOCAL_DIR):
    mtokens --arm A   (GPU) mice scored-frame token store (round-2 cmd_tokens, --every 20) -> BEE/mice/<arm>/diag
    mpool   --arm A   (GPU) mice train pool (round-2 cmd_pool, 8 bursts / video) -> BEE/mice/<arm>/pool
    msweep  --arm A   (GPU) train recipe A / B SAEs, theta*, score -> OUT/mice/sweep_<arm>.json
    aprep   --arm A   (GPU) ants train-frame tokens (both series) -> BEE/ants/<arm>/prep
    atrain  --arm A   (GPU) ants SAEs d2 / d0 x seeds -> BEE/ants/<arm>/sae
    aencode --arm A --series v2|v3  (GPU) scored frames, SAE codes on the fly -> BEE/ants/<arm>/codes_<series>.npz
    ascore  --arm A   (GPU) -> OUT/ants/score_<arm>.json
    decide            (CPU) -> OUT/decision.json, OUT/report.md
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
import nnnt as N  # noqa: E402  (Pipe, plan, scorer; read-only use)
import r2 as R2  # noqa: E402  (token / pool steps, SAE trainer; read-only use, module globals redirected)

R4, REPO, C = N.R4, N.REPO, N.C
OUTR = Path(os.environ.get('RES_OUT') or REPO / 'results/vision/eci_resolution')
BEER = Path(os.environ.get('RES_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/resolution')
log = N.log

# ---------------------------------------------------------------------------------------------- mice arms
MICE_BODY, MICE_NATIVE = 303, 2064
MARMS = {  # arm: (encoder, input px, patch px)
    'd3_512': ('d3', 512, 16), 'd3_768': ('d3', 768, 16), 'd3_1024': ('d3', 1024, 16), 'd3_1408': ('d3', 1408, 16),
    'd3_2048': ('d3', 2048, 16), 'd2_672': ('d2', 672, 14), 'd2_896': ('d2', 896, 14)}
RECIPES = {'A': (512, (128, 256, 512), 16, 6000), 'B': (2048, (128, 512, 2048), 16, 30000)}
SEEDS = {'A': tuple(range(24)), 'B': tuple(range(6))}
CONSISTENT = ('SG', 'SD', 'MR', 'CP')
D2L = 'facebook/dinov2-large'


def mice_dirs(arm):
    """-> (pool dir, diag dir) on BeeGFS. d3_2048 = the round-2 zoom stores; d3_1024 diag = the round-2 1024 diagnosis
    store (every 10th frame; the every-20th frames are selected at scoring time)."""
    if arm == 'd3_2048':
        return N.BEE / 'r2_s2048' / 'pool', N.BEE / 'r2_s2048' / 'diag'
    if arm == 'd3_1024':
        return BEER / 'mice' / arm / 'pool', N.BEE / 'r2' / 'diag'
    return BEER / 'mice' / arm / 'pool', BEER / 'mice' / arm / 'diag'


def patches_per_body(arm):
    enc, S, p = MARMS[arm]
    return MICE_BODY * S / MICE_NATIVE / p


class PipeD2(N.Pipe):
    """nnnt.Pipe with DINOv2-L/14 (blocks 1-20 of 24, then its final LayerNorm, CLS dropped) as the token encoder;
    mask cues unchanged (computed at the input side S with S // G = 14 px patches)."""

    def __init__(self, dev, ranges):
        import torch
        from transformers import AutoModel
        super().__init__(dev, ranges)
        del self.m3
        m = AutoModel.from_pretrained(D2L, attn_implementation='sdpa').to(dev).eval()
        assert len(m.encoder.layer) == 24
        m.encoder.layer = torch.nn.ModuleList(list(m.encoder.layer)[:20])
        self.m3 = m.requires_grad_(False)
        torch.cuda.empty_cache()


def setup_mice(arm):
    """Point nnnt / r2 module globals at this arm (input side, grid, encoder, output dirs)."""
    enc, S, p = MARMS[arm]
    N.S, N.G64 = S, S // p
    assert N.G64 * p == S
    R2.OUT2 = OUTR / 'mice' / arm
    R2.BEE2 = BEER / 'mice' / arm
    if enc == 'd2':
        R4.N_PRE = 1  # DINOv2: CLS only (read by nnnt.Pipe.run at call time)
        N.Pipe = PipeD2
    log(f'arm {arm}: encoder {enc}, input {S} px, grid {N.G64}, {patches_per_body(arm):.2f} patches / body length')


def cmd_mtokens(args):
    assert args.arm not in ('d3_1024', 'd3_2048'), 'reused stores'
    setup_mice(args.arm)
    R2.cmd_tokens(argparse.Namespace(every=20, max_videos=args.max_videos, max_frames=args.max_frames,
                                     selftest=args.selftest))


def cmd_mpool(args):
    assert args.arm != 'd3_2048', 'reused store'
    setup_mice(args.arm)
    R2.cmd_pool(argparse.Namespace(bursts_per_video=8, max_videos=args.max_videos, selftest=args.selftest))


def ref_rows():
    """Annotation rows of the scored frames = the every-20th diagnosis grid (sorted)."""
    P = N.load_plan()
    rows = [v['src_lo'] + np.arange(v['fps1_off'], v['n'], 20) for v in P['videos'] if v['annotated']]
    return np.sort(np.concatenate(rows))


def load_tokens_subset(torch, droot, dev, want_rows):
    """Diagnosis store -> (frame rows (nf,), token frame id (n_tok,) int64 cuda, tokens fp16 (GPU if they fit in half
    of its memory, else pinned host RAM)) restricted to the frames whose annotation row is in want_rows."""
    F, tf, tok = R2.load_diag(droot)
    keepf = np.isin(F.row.values, want_rows)
    assert keepf.sum() == len(want_rows), (int(keepf.sum()), len(want_rows))
    newid = np.full(len(F), -1, np.int64)
    newid[keepf] = np.arange(int(keepf.sum()))
    ti = np.flatnonzero(keepf[tf])
    nb = len(ti) * N.D * 2
    on_gpu = nb <= 0.5 * torch.cuda.get_device_properties(dev).total_memory
    T = torch.empty((len(ti), N.D), dtype=torch.float16, device=dev if on_gpu else 'cpu',
                    pin_memory=not on_gpu)
    for a in range(0, len(ti), 1_000_000):
        T[a:a + 1_000_000] = torch.from_numpy(np.ascontiguousarray(tok[ti[a:a + 1_000_000]]))
    log(f'scored tokens {len(ti):,} of {len(tf):,} ({nb / 2**30:.1f} GiB) on {"GPU" if on_gpu else "pinned host RAM"}; '
        f'{int(keepf.sum()):,} of {len(F):,} frames')
    tfg = torch.from_numpy(newid[tf[ti]]).to(dev)
    return F.row.values[keepf], tfg, T


def cmd_msweep(args):
    import torch
    from src.eci.sae import load_sae
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    setup_mice(args.arm)
    dev = torch.device('cuda')
    loc = N.local_dir()
    pool_b, diag_b = mice_dirs(args.arm)
    sroot = (Path(args.sae_dir) if args.sae_dir else BEER / 'mice' / 'sae')
    out_f = OUTR / 'mice' / (f'sweep_{args.arm}' + ('_selftest' if args.selftest else '') + '.json')
    res = json.loads(out_f.read_text()) if out_f.exists() else {'arm': args.arm, 'train': {}, 'sets': {}}
    enc, S, p = MARMS[args.arm]
    res.update({'encoder': enc, 'S': S, 'patch': p, 'grid': S // p, 'patches_per_body': patches_per_body(args.arm),
                'encoder_tokens_per_frame': (S // p) ** 2})
    recipes = args.recipes.split(',')
    todo = [(r, s) for r in recipes for s in (SEEDS[r] if not args.seeds else [int(x) for x in args.seeds.split(',')])]
    t0 = time.time()
    pdir = Path(args.pool_dir) if args.pool_dir else loc / 'pool'
    if not args.pool_dir:
        R4.atomic_copy(pool_b, pdir)
    PF = pd.read_parquet(pdir / 'frames.parquet')
    ptf = np.load(pdir / 'tok_frame.npy')
    ptok = np.memmap(pdir / 'tok.f16', np.float16, 'r', shape=(len(ptf), N.D))
    prole = PF.role.values[ptf]
    res['pool'] = {'n_frames': int(len(PF)), 'n_tokens': int(len(ptf)), 'tokens_per_frame': float(len(ptf) / len(PF)),
                   'roles': {k_: int(v_) for k_, v_ in pd.Series(prole).value_counts().items()}}
    log(f'pool {len(PF):,} frames, {len(ptf):,} tokens ({len(ptf) / len(PF):.1f} / frame), staged in '
        f'{time.time() - t0:.0f}s')

    def X_of(role):
        ii = np.flatnonzero(prole == role)
        X = torch.empty((len(ii), N.D), dtype=torch.float16, device=dev)
        for a in range(0, len(ii), 500_000):
            X[a:a + 500_000] = torch.from_numpy(np.ascontiguousarray(ptok[ii[a:a + 500_000]])).to(dev)
        return X
    need = [(r, s) for r, s in todo if args.force or not (sroot / f'{args.arm}_{r}_s{s}' / 'sae_theta.pt').exists()]
    if need:
        Xtr = X_of('train')
        log(f'{len(Xtr):,} train tokens')
        for r, s in need:
            W, pre, k, steps = RECIPES[r]
            name = f'{args.arm}_{r}_s{s}'
            cfg = {'arm': args.arm, 'recipe': r, 'width': W, 'prefixes': list(pre), 'k': k, 'steps': steps,
                   'encoder': enc, 'S': S}
            _, _, ti = R2.train_k(cfg, s, Xtr, args.steps or steps, sroot / name, dev, W, pre, k)
            res['train'][name] = {'cfg': cfg, **{q: ti[q] for q in ('train_time_s', 'n_pool_tokens', 'dead_at_end_train')}}
            log(f'{name}: trained {ti["train_time_s"]:.0f}s')
        del Xtr
        torch.cuda.empty_cache()
        Xca, Xev = X_of('calib'), X_of('eval')
        for r, s in need:
            name = f'{args.arm}_{r}_s{s}'
            sae, norm, ck = load_sae(sroot / name / 'sae.pt', dev)
            th, _ = calibrate_threshold(sae, Xca, norm=norm)
            set_threshold(sae, th)
            stt = sparsity_stats(sae, Xev, th, norm=norm)
            ck['state_dict']['threshold'] = torch.tensor(float(th))
            ck['theta_star'] = float(th)
            torch.save(ck, sroot / name / 'sae_theta.pt')
            res['train'].setdefault(name, {}).update({'theta_star': th, 'eval_fve': stt['fve'], 'eval_l0': stt['l0']})
            log(f'{name}: theta* {th:.4f} eval FVE {stt["fve"]:.4f} L0 {stt["l0"]:.2f}')
        del Xca, Xev
        torch.cuda.empty_cache()
        out_f.parent.mkdir(parents=True, exist_ok=True)
        out_f.write_text(json.dumps(res, indent=1, default=float))
    del ptok
    # ---- scored frames
    droot = Path(args.diag_dir) if args.diag_dir else loc / 'diag'
    if not args.diag_dir:
        R4.atomic_copy(diag_b, droot)
    want = ref_rows() if not args.selftest else np.sort(pd.read_parquet(droot / 'frames.parquet').row.values)
    rows, tfg, T = load_tokens_subset(torch, droot, dev, want)
    order = np.argsort(rows, kind='stable')
    assert (rows[order] == want).all()
    nf = len(rows)
    res['scored'] = {'n_frames': int(nf), 'sae_tokens_per_frame': float(len(tfg) / nf)}
    Y, unit = R2.labels_of(rows)
    a = pd.read_csv(REPO / C['ann'], usecols=['W_annotator'])
    ann = a.W_annotator.values[rows]
    sc = json.loads((N.OUT / 'score.json').read_text())
    units, SPL = sc['units'], {k_: np.array(v_) for k_, v_ in sc['splits'].items()}
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    sets = {'all': np.arange(nf), 'consistent': np.flatnonzero(pd.Series(ann).isin(CONSISTENT).values)}
    sets = {k_: ix for k_, ix in sets.items() if len(ix) >= 200 and Y[ix, :2].sum(0).min() >= 10}
    res['scored']['n_frames_sets'] = {k_: int(len(v_)) for k_, v_ in sets.items()}
    res['scored']['base_rate'] = {k_: {lb: float(Y[ix, i].mean()) for i, lb in enumerate(N.LABELS)}
                                  for k_, ix in sets.items()}
    res['scored']['pools_sets'] = {k_: int(len(set(unit[ix]))) for k_, ix in sets.items()}
    log(f'scored {nf:,} frames ({len(tfg) / nf:.1f} SAE tokens / frame); sets ' + json.dumps(res['scored']['n_frames_sets'])
        + ' base ' + json.dumps(res['scored']['base_rate']['all']))
    Yg = {k_: torch.from_numpy(Y[ix]).to(dev) for k_, ix in sets.items()}
    half_of = {k_: {s_: h[uidx[ix]] for s_, h in SPL.items()} for k_, ix in sets.items()}
    ixg = {k_: torch.from_numpy(ix).to(dev) for k_, ix in sets.items()}
    for r, s in todo:
        name = f'{args.arm}_{r}_s{s}'
        if name in res['sets'] and not args.force:
            continue
        sae, norm, _ = load_sae(sroot / name / 'sae_theta.pt', dev)
        X = torch.zeros((nf, sae.n_latents), dtype=torch.float32, device=dev)
        ch = R2.enc_chunk(sae.n_latents)
        t1 = time.time()
        with torch.no_grad():
            for c0 in range(0, len(tfg), ch):
                x = R2.chunk_of(T, c0, c0 + ch, dev).float()
                X.index_reduce_(0, tfg[c0:c0 + ch], sae.encode(norm(x), mode='threshold'), 'amax', include_self=True)
        te = time.time() - t1
        Xh = X.half()
        del X
        rec = {'encode_s': te, 'frame_fire_rate_mean': float((Xh > 0).float().mean())}
        for k_ in sets:
            out, _ = N.score_set(torch, Xh.index_select(0, ixg[k_]), Yg[k_], half_of[k_], dev)
            rec[k_] = {'primary': out['primary'], 'mean': out['mean'], 'specificity': out['specificity']}
        res['sets'][name] = rec
        m = rec['all']['mean']
        log(f'{name}: nn R~ {m["nose_nose"]["R"]:.4f} AUROC {m["nose_nose"]["A_at_R"]:.4f} | nt R~ '
            f'{m["nose_tail"]["R"]:.4f} AUROC {m["nose_tail"]["A_at_R"]:.4f} | primary R~ {rec["all"]["primary"]:.4f} '
            f'| consistent primary R~ {rec.get("consistent", {}).get("primary", float("nan")):.4f} (encode {te:.0f}s)')
        del Xh
        torch.cuda.empty_cache()
        out_f.write_text(json.dumps(res, indent=1, default=float))
    out_f.write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- ants
ANTS = {'v2': dict(dom='ants_v2p', base='ants_v2', native=770, body=64),
        'v3': dict(dom='ants_v3p', base='ants_v3', native=824, body=76)}
AARMS = {'aN': {'v2': 768, 'v3': 832}, 'a7': {'v2': 1344, 'v3': 1216}, 'a10': {'v2': 1920, 'v3': 1728}}
AVARS = ('d2', 'd0')
ALABELS = ('groom_any', 'groom_yellow', 'groom_blue')
ASEEDS = tuple(range(24))
TRAIN_CAP = 6_000_000
AW, APRE, AK, ASTEPS = 512, (128, 256, 512), 16, 6000


def ants_ppb(arm, ser):
    a = ANTS[ser]
    return a['body'] * AARMS[arm][ser] / a['native'] / 16


def ants_pipe(arm, ser, dev):
    """rollout_r4.Pipe of the pooled series at this arm's input side (mask = the lab 'ants' rule on the 512 px JPG,
    32 grid projected to G by patch centre)."""
    a = ANTS[ser]
    S = AARMS[arm][ser]
    assert S % 16 == 0
    R4.DOMS[a['dom']] = dict(R4.DOMS[a['dom']], S=S)
    c = R4.DOMS[a['dom']]
    A, Rr = R4.obs_ranges_of(REPO / c['ann'], ('frame_path',))
    return R4.Pipe(a['dom'], dev, Rr), c, A['frame_path'].values


def ants_bs(G):
    return max(2, int(32 * 48 * 48 / (G * G)))


def ants_jobs_pairs(c, V, items, fp, jdir):
    """items [(series video dict, k)] -> decode jobs of (t - 10, t) pairs (5 fps frames, source offset per video)."""
    jobs = []
    for v, k in items:
        assert k >= 10
        rr = [v['src_lo'] + k - 10, v['src_lo'] + k]
        jobs.append(((v['obs'], k), dict(path=v['path'], s0=R4.voff(c, v) + c['step'] * (k - 10), n=2,
                                         step=c['step'] * 10, S=c['S'], hflip=v['hflip'], fps=c['fps'], threads=2,
                                         jpgs=[jdir / fp[r] for r in rr])))
    return jobs


def cmd_aprep(args):
    """Train / calib / eval frames of the r4 pooled ants SAE (series v2, v3) with their t - 10 partners at this arm's
    input side: kept tokens of t (mask of t) and the partner tokens at the same positions -> BEE/ants/<arm>/prep."""
    import torch
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    PF = pd.read_parquet(R4.BEE / 'ants_pooled' / 'prepass' / 'frames.parquet')
    loc = N.local_dir() / 'prep'
    loc.mkdir(parents=True, exist_ok=True)
    ft, fpv = open(loc / 'tok.f16', 'wb'), open(loc / 'prev.f16', 'wb')
    meta, frames = {'tok_frame': [], 'tok_pos': []}, []
    info = {'arm': args.arm, 'series': {}}
    for ser, a in ANTS.items():
        t0 = time.time()
        pipe, c, fp = ants_pipe(args.arm, ser, dev)
        G = pipe.G
        P = R4.load_plan(a['dom'])
        V = {v['obs']: v for v in P['videos']}
        sub = PF[PF.series == a['base']]
        if args.max_frames:
            sub = sub.groupby('role', group_keys=False).head(args.max_frames)
        items = []
        for r in sub.itertuples():
            v = V[r.obs]
            assert v['src_lo'] + r.k == r.src_row and v['role'] == r.role, (r.obs, r.k)
            items.append((v, int(r.k)))
        jdir = N.local_dir() / 'jpg'
        jobs = ants_jobs_pairs(c, V, items, fp, jdir)
        R4.stage_jpgs(sorted({str(f.relative_to(jdir)) for _, kw in jobs for f in kw['jpgs']}), jdir)
        bs = max(1, ants_bs(G) // 2)
        pf = R4.Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4))
        buf, n_tok = [], 0

        def flush(buf):
            nonlocal n_tok
            fr = np.concatenate([b[1][0] for b in buf])
            jp = np.concatenate([b[1][1] for b in buf])
            rows = np.concatenate([[V[o]['src_lo'] + k - 10, V[o]['src_lo'] + k] for (o, k), _ in buf])
            with torch.no_grad():
                z, m, n32, _ = pipe.run(fr, rows, jp)
            for i, ((o, k), _) in enumerate(buf):
                pos = torch.nonzero(m[2 * i + 1]).flatten()
                ft.write(z[2 * i + 1, pos].cpu().numpy().tobytes())
                fpv.write(z[2 * i, pos].cpu().numpy().tobytes())
                fid = len(frames)
                meta['tok_frame'].append(np.full(len(pos), fid, np.int32))
                meta['tok_pos'].append(pos.cpu().numpy().astype(np.int16))
                frames.append({'series': ser, 'obs': o, 'k': k, 'role': V[o]['role'], 'n_kept': int(len(pos)),
                               'n32': int(n32[2 * i + 1])})
                n_tok += len(pos)
        for ji, (key, x) in enumerate(pf):
            buf.append((key, x))
            if len(buf) == bs:
                flush(buf)
                buf = []
            if ji % 1000 == 0:
                log(f'{ser}: {ji:,} / {len(jobs):,} frames, {time.time() - t0:.0f}s, GPU {pipe.gpu_s:.0f}s, '
                    f'decode thread-s {pf.dec_s:.0f}')
        if buf:
            flush(buf)
        n_ser = sum(1 for f in frames if f['series'] == ser)
        info['series'][ser] = {'S': c['S'], 'G': G, 'patches_per_body': ants_ppb(args.arm, ser), 'n_frames': n_ser,
                               'n_tokens': n_tok, 'tokens_per_frame': n_tok / max(n_ser, 1),
                               'wall_s': time.time() - t0, 'gpu_s': pipe.gpu_s,
                               'encodes_per_gpu_s': 2 * n_ser / max(pipe.gpu_s, 1e-9)}
        log(f'{ser}: ' + json.dumps(info['series'][ser]))
        del pipe
        torch.cuda.empty_cache()
    ft.close()
    fpv.close()
    for k_, x in meta.items():
        np.save(loc / f'{k_}.npy', np.concatenate(x))
    pd.DataFrame(frames).to_parquet(loc / 'frames.parquet')
    info['gpu'] = torch.cuda.get_device_name(0)
    od = OUTR / 'ants' / args.arm
    od.mkdir(parents=True, exist_ok=True)
    (od / ('prep_selftest.json' if args.selftest else 'prep.json')).write_text(json.dumps(info, indent=1))
    if not args.selftest:
        R4.atomic_copy(loc, BEER / 'ants' / args.arm / 'prep')


def cmd_atrain(args):
    import torch
    from src.eci.sae import load_sae
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    dev = torch.device('cuda')
    pdir = Path(args.pool_dir) if args.pool_dir else N.local_dir() / 'prep'
    if not args.pool_dir:
        R4.atomic_copy(BEER / 'ants' / args.arm / 'prep', pdir)
    F = pd.read_parquet(pdir / 'frames.parquet')
    tf = np.load(pdir / 'tok_frame.npy')
    tok = np.memmap(pdir / 'tok.f16', np.float16, 'r', shape=(len(tf), N.D))
    prv = np.memmap(pdir / 'prev.f16', np.float16, 'r', shape=(len(tf), N.D))
    role = F.role.values[tf]
    sroot = Path(args.sae_dir) if args.sae_dir else BEER / 'ants' / args.arm / 'sae'
    seeds = [int(x) for x in args.seeds.split(',')] if args.seeds else ASEEDS
    out_f = OUTR / 'ants' / args.arm / ('train_selftest.json' if args.selftest else 'train.json')
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    ntr = int((role == 'train').sum())
    tr_ii = np.flatnonzero(role == 'train')
    if ntr > TRAIN_CAP:
        tr_ii = np.sort(np.random.default_rng(0).choice(tr_ii, TRAIN_CAP, replace=False))
    res['_pool'] = {'n_frames': int(len(F)), 'n_tokens': int(len(tf)), 'n_train_tokens_available': ntr,
                    'n_train_tokens_used': int(len(tr_ii)),
                    'tokens_per_frame_by_series': F.groupby('series').n_kept.mean().to_dict()}
    log(json.dumps(res['_pool']))
    for var in AVARS:
        def X_of(ii):
            X = torch.empty((len(ii), N.D * (2 if var == 'd2' else 1)), dtype=torch.float16, device=dev)
            for a in range(0, len(ii), 500_000):
                s_ = ii[a:a + 500_000]
                t = torch.from_numpy(np.ascontiguousarray(tok[s_])).to(dev)
                X[a:a + len(s_), :N.D] = t
                if var == 'd2':
                    X[a:a + len(s_), N.D:] = (t.float() - torch.from_numpy(np.ascontiguousarray(prv[s_])).to(dev).float()
                                              ).half()
            return X
        need = [s for s in seeds if args.force or not (sroot / f'{var}_s{s}' / 'sae_theta.pt').exists()]
        if not need:
            continue
        Xtr = X_of(tr_ii)
        for s in need:
            cfg = {'arm': args.arm, 'variant': var, 'width': AW, 'prefixes': list(APRE), 'k': AK, 'steps': ASTEPS}
            _, _, ti = R2.train_k(cfg, s, Xtr, args.steps or ASTEPS, sroot / f'{var}_s{s}', dev, AW, APRE, AK)
            res[f'{var}_s{s}'] = {q: ti[q] for q in ('train_time_s', 'n_pool_tokens', 'dead_at_end_train')}
            log(f'{var}_s{s}: trained {ti["train_time_s"]:.0f}s')
        del Xtr
        torch.cuda.empty_cache()
        Xca, Xev = X_of(np.flatnonzero(role == 'calib')), X_of(np.flatnonzero(role == 'eval'))
        for s in need:
            od = sroot / f'{var}_s{s}'
            sae, norm, ck = load_sae(od / 'sae.pt', dev)
            th, _ = calibrate_threshold(sae, Xca, norm=norm)
            set_threshold(sae, th)
            stt = sparsity_stats(sae, Xev, th, norm=norm)
            ck['state_dict']['threshold'] = torch.tensor(float(th))
            ck['theta_star'] = float(th)
            torch.save(ck, od / 'sae_theta.pt')
            res[f'{var}_s{s}'].update({'theta_star': th, 'eval_fve': stt['fve'], 'eval_l0': stt['l0']})
            log(f'{var}_s{s}: theta* {th:.4f} eval FVE {stt["fve"]:.4f} L0 {stt["l0"]:.2f}')
        del Xca, Xev
        torch.cuda.empty_cache()
        out_f.parent.mkdir(parents=True, exist_ok=True)
        out_f.write_text(json.dumps(res, indent=1, default=float))


def ants_labels(dom):
    c = R4.DOMS[dom]
    a = pd.read_csv(REPO / c['eci'] / 'annotations.csv', usecols=['observation_id', 'Y_Y2F', 'Y_B2F'])
    ok = a[['Y_Y2F', 'Y_B2F']].notna().all(1).values
    y = (a.Y_Y2F.values > 0), (a.Y_B2F.values > 0)
    return ok, np.stack([y[0] | y[1], y[0], y[1]], 1), a.observation_id.values


def cmd_aencode(args):
    """Scored frames of one series: every 10th 5 fps frame of each video's labelled range (k = 0 mod 10; the t - 10
    partner = the previous grid frame, clipped to frame 0), codes (frame max) of every d2 / d0 SAE of the arm."""
    import torch
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    ser = args.series
    a = ANTS[ser]
    pipe, c, fp = ants_pipe(args.arm, ser, dev)
    G = pipe.G
    P = R4.load_plan(a['dom'])
    ok, Y, aobs = ants_labels(a['dom'])
    sroot = Path(args.sae_dir) if args.sae_dir else BEER / 'ants' / args.arm / 'sae'
    seeds = [int(x) for x in args.seeds.split(',')] if args.seeds else ASEEDS
    saes = []
    for var in AVARS:
        for s in seeds:
            sae, norm, ck = load_sae(sroot / f'{var}_s{s}' / 'sae_theta.pt', dev)
            assert abs(float(sae.threshold) - ck['theta_star']) < 1e-6
            saes.append((f'{var}_s{s}', var, sae, norm))
    vids = P['videos'][:args.max_videos or None]
    jdir = N.local_dir() / 'jpg'
    jobs, rels, plan = [], [], []
    for v in vids:
        lab = ok[v['lay_lo']:v['lay_lo'] + v['n']]
        assert (aobs[v['lay_lo']:v['lay_lo'] + v['n']] == v['obs']).all()
        kl = np.flatnonzero(lab)
        if not len(kl):
            continue
        assert lab[kl[0]:kl[-1] + 1].all(), (v['obs'], 'labelled range not contiguous')
        k0 = max(0, (kl[0] // 10) * 10 - 10)
        ks = np.arange(k0, kl[-1] + 1, 10)
        if args.max_frames:
            ks = ks[:args.max_frames]
        plan.append((v, ks))
        for c0 in range(0, len(ks), 60):
            kk = ks[c0:c0 + 60]
            rels += list(fp[v['src_lo'] + kk])
            jobs.append(((v['obs'], kk), dict(path=v['path'], s0=R4.voff(c, v) + c['step'] * int(kk[0]), n=len(kk),
                                              step=c['step'] * 10, S=c['S'], hflip=v['hflip'], fps=c['fps'], threads=2,
                                              jpgs=[jdir / f for f in fp[v['src_lo'] + kk]])))
    R4.stage_jpgs(sorted(set(rels)), jdir)
    V = {v['obs']: v for v in vids}
    bs = ants_bs(G)
    pf = R4.Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4))
    acc = {nm: [] for nm, *_ in saes}
    meta = {'obs': [], 'k': [], 'lay_row': [], 'n_kept': [], 'labelled': []}
    prev = {}  # obs -> (k, z (G*G, 1024) fp16) of the last grid frame
    t0, n_done = time.time(), 0
    for (o, kk), (fr, jp) in pf:
        v = V[o]
        for b in range(0, len(kk), bs):
            kb = kk[b:b + bs]
            with torch.no_grad():
                z, m, n32, _ = pipe.run(fr[b:b + bs], v['src_lo'] + kb, jp[b:b + bs])
                zp = []
                for i, k in enumerate(kb):
                    if k == 0 or (i == 0 and o not in prev):  # clipped (k = 0) or an unscored partner-only frame
                        zp.append(z[i])
                    else:
                        pk, pz = prev[o] if i == 0 else (kb[i - 1], z[i - 1])
                        assert pk == k - 10, (o, k, pk)
                        zp.append(pz)
                zp = torch.stack(zp)
                prev[o] = (int(kb[-1]), z[-1].clone())
                lab = ok[v['lay_lo'] + kb]
                bi, pi = m.nonzero(as_tuple=True)
                zt, zpt = z[bi, pi], zp[bi, pi]
                xin = {'d0': zt, 'd2': torch.cat([zt, (zt.float() - zpt.float()).half()], 1)}
                B = len(kb)
                for nm, var, sae, norm in saes:
                    x = xin[var]
                    cm = torch.zeros((B, AW), dtype=torch.float32, device=dev)
                    if len(x):
                        cm.index_reduce_(0, bi, sae.encode(norm(x.float()), mode='threshold'), 'amax',
                                         include_self=True)
                    acc[nm].append(cm[torch.from_numpy(lab).to(dev)].half().cpu().numpy())
            for i, k in enumerate(kb):
                if lab[i]:
                    meta['obs'].append(o)
                    meta['k'].append(int(k))
                    meta['lay_row'].append(int(v['lay_lo'] + k))
                    meta['n_kept'].append(int(m[i].sum()))
            n_done += len(kb)
        if len(meta['k']) and n_done % 3000 < 60:
            el = time.time() - t0
            log(f'{ser} {args.arm}: {n_done:,} frames ({len(meta["k"]):,} labelled), {el:.0f}s, GPU {pipe.gpu_s:.0f}s '
                f'({n_done / max(pipe.gpu_s, 1e-9):.1f} frames / GPU-s), decode thread-s {pf.dec_s:.0f}')
    out = {nm: np.concatenate(x) for nm, x in acc.items()}
    M = {k_: np.asarray(v_) for k_, v_ in meta.items() if k_ != 'labelled'}
    loc = N.local_dir() / f'codes_{ser}.npz'
    np.savez(loc, **{f'c|{k_}': v_ for k_, v_ in out.items()}, **{f'm|{k_}': v_ for k_, v_ in M.items()})
    info = {'arm': args.arm, 'series': ser, 'S': c['S'], 'G': G, 'patches_per_body': ants_ppb(args.arm, ser),
            'n_frames_encoded': n_done, 'n_scored': int(len(M['k'])), 'n_videos': len(plan),
            'sae_tokens_per_frame': float(np.mean(M['n_kept'])), 'encoder_tokens_per_frame': G * G,
            'wall_s': time.time() - t0, 'gpu_s': pipe.gpu_s, 'frames_per_gpu_s': n_done / max(pipe.gpu_s, 1e-9),
            'gpu': torch.cuda.get_device_name(0)}
    log(json.dumps(info))
    od = OUTR / 'ants' / args.arm
    od.mkdir(parents=True, exist_ok=True)
    (od / (f'encode_{ser}' + ('_selftest' if args.selftest else '') + '.json')).write_text(json.dumps(info, indent=1))
    if not args.selftest:
        R4.atomic_copy(loc, BEER / 'ants' / args.arm / loc.name)
    else:
        R4.atomic_copy(loc, N.local_dir() / 'out_codes' / loc.name)


def score_generic(torch, X, Y, labels, half_of):
    """Round-1 cross-fit (nnnt.score_set) for any labels: per split, the neuron with the highest R~ (AUROC) on one
    half is scored on the other, both directions averaged; mean over the 'rnd' splits."""
    H = {}
    out = {'splits': {}}
    for sn, half in half_of.items():
        for h in (0, 1):
            H[h] = N.half_stats(torch, X, Y, torch.from_numpy(np.flatnonzero(half == h)).to(X.device))
        rec = {}
        for li, lb in enumerate(labels):
            d = []
            for s_, t_ in ((0, 1), (1, 0)):
                sr = np.where(np.isfinite(H[s_]['R'][:, li]), H[s_]['R'][:, li], -np.inf)
                sa = np.where(np.isfinite(H[s_]['A'][:, li]), H[s_]['A'][:, li], -np.inf)
                j, ja = int(np.argmax(sr)), int(np.argmax(sa))
                d.append({'R': float(H[t_]['R'][j, li]), 'A_at_R': float(H[t_]['A'][j, li]),
                          'A': float(H[t_]['A'][ja, li]), 'neuron': j})
            rec[lb] = {q: float(np.mean([x[q] for x in d])) for q in ('R', 'A_at_R', 'A')}
            rec[lb]['neurons'] = [x['neuron'] for x in d]
        out['splits'][sn] = rec
    rnd = [s for s in half_of if s.startswith('rnd')]
    out['mean'] = {lb: {q: float(np.mean([out['splits'][s][lb][q] for s in rnd])) for q in ('R', 'A_at_R', 'A')}
                   for lb in labels}
    return out


def cmd_ascore(args):
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_f = OUTR / 'ants' / args.arm / ('score_selftest.json' if args.selftest else 'score.json')
    res = {'arm': args.arm, 'series': {}, 'sae': {}}
    for ser, a in ANTS.items():
        f = (N.local_dir() / 'out_codes' if args.selftest else BEER / 'ants' / args.arm) / f'codes_{ser}.npz'
        if not f.exists():
            log(f'missing {f}')
            continue
        z = np.load(f)
        obs = z['m|obs']
        _, Y, _ = ants_labels(a['dom'])
        Yr = Y[z['m|lay_row']]
        units, SPL = N.splits_of(set(obs))
        uidx = pd.Series(range(len(units)), index=units).loc[obs].values
        half_of = {s_: h[uidx] for s_, h in SPL.items()}
        Yg = torch.from_numpy(Yr).to(dev)
        res['series'][ser] = {'n_frames': int(len(obs)), 'n_videos': len(units),
                              'base_rate': {lb: float(Yr[:, i].mean()) for i, lb in enumerate(ALABELS)},
                              'sae_tokens_per_frame': float(z['m|n_kept'].mean())}
        for key in [k_ for k_ in z.files if k_.startswith('c|')]:
            nm = key[2:]
            X = torch.from_numpy(z[key]).to(dev)
            o = score_generic(torch, X, Yg, ALABELS, half_of)
            res['sae'].setdefault(nm, {})[ser] = o['mean']
            res['sae'][nm][ser]['r4_split'] = o['splits']['r4']
            del X
        log(f'{ser}: scored {len(obs):,} frames, {len(units)} videos')
    for nm, d in res['sae'].items():
        if all(s in d for s in ANTS):
            d['primary'] = float(np.mean([d[s]['groom_any']['R'] for s in ANTS]))
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(res, indent=1, default=float))
    for var in AVARS:
        pr = [d['primary'] for nm, d in res['sae'].items() if nm.startswith(var) and 'primary' in d]
        if pr:
            log(f'{args.arm} {var}: primary R~ (groom_any, mean of v2 / v3) {np.mean(pr):.4f} ± {np.std(pr):.4f} '
                f'over {len(pr)} seeds')


# ---------------------------------------------------------------------------------------------- decide
def verdict(d, lo, hi):
    return 'WIN' if d >= 0.01 and lo > 0 else 'LOSES' if d <= -0.01 and hi < 0 else 'NO DIFFERENCE'


def gpu_hours():
    """sacct elapsed of this sweep's GPU jobs (names res_*), per job name."""
    import subprocess
    r = subprocess.run(['sacct', '-u', os.environ.get('USER', ''), '-S', '2026-10-19', '-X', '-n', '-P', '-o',
                        'JobName,ElapsedRaw,State,AllocTRES'], capture_output=True, text=True).stdout
    out = {}
    for ln in r.splitlines():
        nm, el, st, tres = (ln.split('|') + ['', '', '', ''])[:4]
        if not nm.startswith('res_') or 'gpu' not in tres:
            continue
        out.setdefault(nm, 0.0)
        out[nm] += int(el or 0) / 3600
    return out


def cmd_decide(args):
    W = N.welch
    ms = lambda x: f'{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}' if len(x) > 1 else '-'  # noqa: E731
    dec, L = {}, ['# Resolution sweep — results', '',
                  'Pre-registered rule: scripts/eci/resolution/res.py docstring. R~ = cross-fitted normalised '
                  'R-precision; AUROC = cf AUROC (AUROC-selected neuron, as the round-1 / 2 tables); mean ± sd over SAE '
                  'seeds; deltas = Welch 95% CI.', '']
    gh = gpu_hours()
    # ------------------------------------------------------------------ mice
    M = {}
    for arm in MARMS:
        f = OUTR / 'mice' / f'sweep_{arm}.json'
        if f.exists():
            M[arm] = json.loads(f.read_text())

    def mv(arm, rec, seeds, fs='all', lb='primary', q='R'):
        out = []
        for s in seeds:
            x = M.get(arm, {}).get('sets', {}).get(f'{arm}_{rec}_s{s}', {}).get(fs)
            if x is None:
                continue
            out.append(x['primary'] if lb == 'primary' else x['mean'][lb][q])
        return out
    if M:
        L += ['## Mice v1 (43,200 scored frames, every 20th 5 fps frame of the 144 annotated videos)', '',
              '| arm | input px | grid | patches / body | encoder tokens / frame | SAE tokens / frame (scored) | recipe | '
              'seeds | primary R~ | nose_nose R~ | nose_nose AUROC | nose_tail R~ | nose_tail AUROC | '
              'consistent-annotator primary R~ | GPU-h (tokens + pool + sweep) |', '|' + '---|' * 15]
        for arm, d in M.items():
            g = sum(v for k_, v in gh.items() if k_.endswith('_' + arm))
            for rec, sds in (('A', range(12)), ('A', range(24)), ('B', range(6))):
                p_ = mv(arm, rec, sds)
                if not p_:
                    continue
                L.append(f'| {arm} | {d["S"]} | {d["grid"]} | {d["patches_per_body"]:.1f} | '
                         f'{d["encoder_tokens_per_frame"]:,} | {d.get("scored", {}).get("sae_tokens_per_frame", 0):.0f} | '
                         f'{rec} | {min(sds)}-{max(sds)} ({len(p_)}) | {ms(p_)} | '
                         f'{ms(mv(arm, rec, sds, lb="nose_nose"))} | {ms(mv(arm, rec, sds, lb="nose_nose", q="A"))} | '
                         f'{ms(mv(arm, rec, sds, lb="nose_tail"))} | {ms(mv(arm, rec, sds, lb="nose_tail", q="A"))} | '
                         f'{ms(mv(arm, rec, sds, fs="consistent"))} | {g:.2f} |')
        L.append('')
        d3 = [a for a in MARMS if a.startswith('d3') and len(mv(a, 'A', range(12))) >= 2]
        scr = {a: float(np.mean(mv(a, 'A', range(12)))) for a in d3}
        md = {'screening_mean_primary_R': scr}
        if scr:
            best = max(scr, key=scr.get)
            md['best'] = best
            md['deltas_vs_best'] = {a: W(mv(a, 'A', range(12)), mv(best, 'A', range(12))) for a in d3 if a != best}
            ok = [a for a in d3 if a == best or (md['deltas_vs_best'][a][0] >= -0.01 and md['deltas_vs_best'][a][2] >= 0)]
            cand = min(ok, key=lambda a: MARMS[a][1])
            md['candidate'] = cand
            conf = {}
            if cand != 'd3_1024' and mv(cand, 'A', range(12, 24)) and mv('d3_1024', 'A', range(12, 24)):
                for lb in ('primary', 'nose_nose', 'nose_tail'):
                    conf[lb] = W(mv(cand, 'A', range(12, 24), lb=lb), mv('d3_1024', 'A', range(12, 24), lb=lb))
                drop = min(conf['nose_nose'][0], conf['nose_tail'][0]) < -0.02
                if MARMS[cand][1] > 1024:
                    adopt = conf['primary'][0] >= 0.01 and conf['primary'][1] > 0 and not drop
                else:
                    adopt = conf['primary'][0] >= -0.01 and conf['primary'][1] >= -0.03 and not drop
                md['confirmation_cand_minus_1024_seeds12-23'] = conf
                md['decision'] = cand if adopt else 'd3_1024'
            else:
                md['decision'] = 'd3_1024' if cand == 'd3_1024' else 'PENDING (confirmation seeds missing)'
            # descriptive: every arm vs 1024 on all 24 seeds; recipe B and the consistent set vs 1024
            md['vs_1024_seeds0-23'] = {a: {lb: W(mv(a, 'A', range(24), lb=lb), mv('d3_1024', 'A', range(24), lb=lb))
                                           for lb in ('primary', 'nose_nose', 'nose_tail')}
                                       for a in M if a != 'd3_1024' and len(mv(a, 'A', range(24))) > 1}
            md['vs_1024_auroc_seeds0-23'] = {a: {lb: W(mv(a, 'A', range(24), lb=lb, q='A'),
                                                       mv('d3_1024', 'A', range(24), lb=lb, q='A'))
                                                 for lb in ('nose_nose', 'nose_tail')}
                                             for a in M if a != 'd3_1024' and len(mv(a, 'A', range(24))) > 1}
            md['recipe_B_vs_1024'] = {a: W(mv(a, 'B', range(6)), mv('d3_1024', 'B', range(6))) for a in M
                                      if a != 'd3_1024' and len(mv(a, 'B', range(6))) > 1}
            md['consistent_vs_1024'] = {a: W(mv(a, 'A', range(24), fs='consistent'),
                                             mv('d3_1024', 'A', range(24), fs='consistent')) for a in M
                                        if a != 'd3_1024' and len(mv(a, 'A', range(24), fs='consistent')) > 1}
        # patch-size question
        pq = {}
        for nm, a1, a2 in (('DINOv3/16 64 vs 48 grid (d3_1024 - d3_768)', 'd3_1024', 'd3_768'),
                           ('DINOv2/14 64 vs 48 grid (d2_896 - d2_672)', 'd2_896', 'd2_672'),
                           ('matched 9.4 patches: DINOv3 1024 px - DINOv2 896 px', 'd3_1024', 'd2_896'),
                           ('matched 7.0 patches: DINOv3 768 px - DINOv2 672 px', 'd3_768', 'd2_672')):
            x, y = mv(a1, 'A', range(24)), mv(a2, 'A', range(24))
            if len(x) > 1 and len(y) > 1:
                d_ = W(x, y)
                pq[nm] = {'delta_primary_R': d_, 'verdict': verdict(*d_)}
        k1, k2 = 'DINOv3/16 64 vs 48 grid (d3_1024 - d3_768)', 'DINOv2/14 64 vs 48 grid (d2_896 - d2_672)'
        if k1 in pq and k2 in pq:
            pq['rule_in_patches'] = pq[k1]['verdict'] == pq[k2]['verdict']
        md['patch_size'] = pq
        dec['mice'] = md
        L += ['### Mice decision', '', '```', json.dumps(md, indent=1), '```', '']
    # ------------------------------------------------------------------ ants
    A = {}
    for arm in AARMS:
        f = OUTR / 'ants' / arm / 'score.json'
        if f.exists():
            A[arm] = json.loads(f.read_text())

    def av(arm, var, seeds, ser=None, lb='groom_any', q='R'):
        out = []
        for s in seeds:
            d = A.get(arm, {}).get('sae', {}).get(f'{var}_s{s}')
            if d is None or (ser is None and 'primary' not in d) or (ser and ser not in d):
                continue
            out.append(d['primary'] if ser is None else d[ser][lb][q])
        return out
    if A:
        L += ['## Ants (v2 + v3 pooled SAE; grooming)', '',
              '| arm | v2 px / patches | v3 px / patches | variant | seeds | primary R~ (groom_any, mean v2 v3) | '
              'v2 groom_any R~ | v2 AUROC | v3 groom_any R~ | v3 AUROC | v2 yellow / blue R~ | v3 yellow / blue R~ | '
              'SAE tokens / frame v2 / v3 | GPU-h |', '|' + '---|' * 14]
        for arm, d in A.items():
            g = sum(v for k_, v in gh.items() if k_.endswith('_' + arm) or f'_{arm}_' in k_)
            tpf = ' / '.join(f'{d["series"][s]["sae_tokens_per_frame"]:.0f}' for s in ANTS if s in d['series'])
            for var in AVARS:
                for sds in (range(12), range(24)):
                    p_ = av(arm, var, sds)
                    if not p_:
                        continue
                    L.append(f'| {arm} | {AARMS[arm]["v2"]} / {ants_ppb(arm, "v2"):.1f} | {AARMS[arm]["v3"]} / '
                             f'{ants_ppb(arm, "v3"):.1f} | {var} | {min(sds)}-{max(sds)} ({len(p_)}) | {ms(p_)} | '
                             f'{ms(av(arm, var, sds, "v2"))} | {ms(av(arm, var, sds, "v2", q="A"))} | '
                             f'{ms(av(arm, var, sds, "v3"))} | {ms(av(arm, var, sds, "v3", q="A"))} | '
                             f'{ms(av(arm, var, sds, "v2", "groom_yellow"))} / {ms(av(arm, var, sds, "v2", "groom_blue"))} | '
                             f'{ms(av(arm, var, sds, "v3", "groom_yellow"))} / {ms(av(arm, var, sds, "v3", "groom_blue"))} | '
                             f'{tpf} | {g:.2f} |')
        L.append('')
        ad = {}
        up = [a for a in ('a7', 'a10') if len(av(a, 'd2', range(12))) > 1]
        if up and len(av('aN', 'd2', range(12))) > 1:
            scr = {a: float(np.mean(av(a, 'd2', range(12)))) for a in ['aN'] + up}
            best = max(up, key=lambda a: scr[a])
            ad.update(screening_mean_primary_R=scr, best_upsampled=best)
            if len(av(best, 'd2', range(12, 24))) > 1:
                c_ = W(av(best, 'd2', range(12, 24)), av('aN', 'd2', range(12, 24)))
                per = {s: W(av(best, 'd2', range(12, 24), s), av('aN', 'd2', range(12, 24), s)) for s in ANTS}
                ad['confirmation_best_minus_aN_seeds12-23'] = {'primary': c_, **per}
                ad['decision'] = best if (c_[0] >= 0.01 and c_[1] > 0 and all(per[s][0] >= 0 for s in ANTS)) else 'aN'
            ad['vs_aN_seeds0-23'] = {a: {v_: W(av(a, v_, range(24)), av('aN', v_, range(24))) for v_ in AVARS}
                                     for a in up}
            ad['vs_aN_auroc_seeds0-23'] = {a: {s: W(av(a, 'd2', range(24), s, q='A'), av('aN', 'd2', range(24), s, q='A'))
                                               for s in ANTS} for a in up}
        dec['ants'] = ad
        L += ['### Ants decision', '', '```', json.dumps(ad, indent=1), '```', '']
    dec['gpu_hours_by_job'] = gh
    L += ['## GPU-hours by job name (sacct elapsed)', '', '```', json.dumps(gh, indent=1), '```', '']
    (OUTR / 'decision.json').write_text(json.dumps(dec, indent=1, default=float))
    (OUTR / 'report.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('mtokens', 'mpool', 'msweep', 'aprep', 'atrain', 'aencode', 'ascore', 'decide'))
    p.add_argument('--arm', default='')
    p.add_argument('--series', default='')
    p.add_argument('--recipes', default='A,B')
    p.add_argument('--seeds', default='')
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--pool-dir', default='')
    p.add_argument('--diag-dir', default='')
    p.add_argument('--sae-dir', default='')
    p.add_argument('--max-videos', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--force', action='store_true')
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()
    g = globals()
    g[f'cmd_{args.step}'](args)


if __name__ == '__main__':
    main()
