"""
Second-level ("relational") SAEs for ECI: tracker-free behaviour neurons from WHICH first-level patch neurons fire next
to each other (src/eci/l2sae.py). Three levels, matching the individual / pair / collective framing:
    level 1  patch codes (parts, postures)  = the best held-out first-level SAE of scripts/eci/sae_levers.py
    level 2  local configurations of level-1 codes (interactions): one token per foreground patch p,
             [ code(p) | NEAR(p) = max of the codes at Chebyshev distance 1-2 | FAR(p) = max at distance 3..r_far ],
             p itself excluded from NEAR / FAR
    level 3  the frame's MEAN level-1 code over its foreground patches (collective), count-normalised by construction

Data, split, labels, cross-fit halves and the scoring harness are those of scripts/eci/sae_levers.py (split seed 0):
    mice  fg448 store; train = 100 unannotated videos (123,300 frames, 18.0M tokens); eval = the 172,800 1 fps store
          frames of the 144 annotated videos; labels nose_nose = Y_nn OR Y_np, nose_tail = Y_nt
    ants  antsfg store (v2 + v3); train = 128 random videos; eval = the other 128 (76,800 frames);
          labels groom_any, groom_yellow (Y2F), groom_blue (B2F), onlid_yellow (v3 frames only)

Level-1 SAE per domain (results/vision/eci_sae_levers/<domain>/sae/<cfg>_s<seed>/sae.pt; level-2 / level-3 seed s are
trained on level-1 seed s, so the seed spread includes the level-1 seed):
    mice  'w4096+cell' (4096 latents, per-(video, position) token centring): the best held-out lever (nose-nose cf AUROC
          0.737, AP 0.069). Centring means: train videos from their train frames, eval videos from their eval frames.
    ants  'base' (1024 latents, the deployed recipe re-trained on the 128 train videos only, 3 seeds). The deployed
          antsfg SAE (neuron 90) was trained on ALL 256 videos, including the eval videos, and has one seed, so it is
          kept as a reference row (dot-Voronoi read of n90), not used as level 1. Centring hurts ants (levers).
Sparse level-1 codes: threshold inference, then the top-K = 48 (value, latent) pairs per token (truncation logged).

Radii (animal size measured on train frames, src/eci/multiscale.py / sae_levers merged_meta: mice median single-animal
foreground component 45 patches, bbox ~8 x 6; ants 21 patches, bbox ~6 x 5):
    NEAR  Chebyshev 1-2 (both domains): touching / adjacent body parts
    FAR   mice 3-6, ants 3-5 (about one animal length)
    Level-2 dimension = 3 x level-1 width (mice 12,288, ants 3,072) -- full codes, no top-M projection.
    Block normalisation: one scale per block [code | NEAR | FAR] (TokenNorm.fit_blocks definition: mean subtracted,
    each block average norm sqrt(block size)), fitted on 100k pool tokens.
SAEs (src/eci/sae.py Matryoshka BatchTopK, the levers recipe: Adam lr 5e-4, 500 warmup, cosine to 10%, 6000 steps x
4096 tokens, AuxK, b_dec = geometric median, threshold inference), 3 seeds:
    level 2  4096 latents, prefixes 512/1024/2048/4096, k = 16, on a pool of 4M (mice) / 2M (ants) train tokens drawn
             uniformly without replacement (pool seed 0 = the levers 'uniform' pool); NEAR / FAR always read from the
             full train frames.
    level 3  1024 latents, prefixes 128/256/512/1024, k = 16, on all train frames (one token per frame).

Read-outs per frame (eval):
    level 1  max | top8 (mean of the 8 largest patch activations; zero-padded below 8 patches) | frac (fraction of the
             frame's foreground patches where the latent fires) | mean (= the level-3 input)
    level 2  max | top8 | frac
    level 3  code
Scores per read-out (sae_levers.score_model, unchanged): cross-fitted best-neuron AUROC (+ AP of that neuron),
best-AP, honest top-1% (neurons firing on >= 1% of the selection half's frames), size-controlled AUROC (AUROC-selected
neuron within foreground-count deciles of the test half).
Ants: dot-Voronoi read (sae_levers.column_codes rule: tokens within 2 grid units of the focal centroid, closer to the
yellow dot than the blue one, and the reverse) of the cross-fitted best groom_any neuron of the level-1 and level-2
'max' read-outs, and of the deployed n90.
Collective sanity (NOT a human label): Spearman of every level-3 neuron (and level-1 mean neuron) with mice foreground
spread (radius of gyration of the foreground patch centres, grid units) / ants tracker dispersion (mean pairwise
centroid distance, multiscale_sae.ants_dispersion); cross-fitted: neuron picked on |rho| of half A, rho on half B.
Video level (mice; step 'video', CPU): diag_switch_regions.video_metrics (the diag_video_level.py measurements) of the
per-video in-window means of the AUROC- and top1-selected neurons (neuron chosen on the other half), raw and after
regressing the per-video mean on the per-video mean foreground patches per frame within each half (d02bbbc).
Contact sheets (seed 0): level-2 max / top8 read-out, AUROC-selected on half 0, its top-16 frames of half 1 (<= 2 per
video), the argmax patch outlined (green = label positive).

Steps (STEP of scripts/eci/l2sae.sh): selftest | train | evaluate | video (CPU, mice) | summary (login-node safe)
Outputs: results/vision/eci_l2sae/<domain>/{sae,eval,codes_best,sheets,video}/, summary.json, table.md
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import multiscale_sae as ms  # noqa: E402
import sae_levers as sl  # noqa: E402
import spatial_sae_pilot as ssp  # noqa: E402
from src.eci.l2sae import (GRID, SparseCodes, fit_block_norm, frame_reads, level2_features, level2_reference,  # noqa: E402
                           ring_offsets)
from src.eci.levers import CellMeans  # noqa: E402
from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, load_sae, save_checkpoint  # noqa: E402

LEV = REPO / 'results/vision/eci_sae_levers'
DOM = {'mice': {'l1': 'w4096+cell', 'center': 'cell', 'r_far': 6},
       'ants': {'l1': 'base', 'center': None, 'r_far': 5}}
R_NEAR = (1, 2)
K_SPARSE = 48
L2_WIDTH, L3_WIDTH, K_L2 = 4096, 1024, 16
READS = {1: ('max', 'top8', 'frac', 'mean'), 2: ('max', 'top8', 'frac'), 3: ('code',)}
LABELS = sl.LABELS
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def rings_for(domain, dev):
    return [ring_offsets(*R_NEAR).to(dev), ring_offsets(R_NEAR[1] + 1, DOM[domain]['r_far']).to(dev)]


def l1_path(domain, seed):
    return LEV / domain / 'sae' / f"{DOM[domain]['l1']}_s{seed}" / 'sae.pt'


def load_ram(idx, frames, dst):
    """Stage frames to local disk (ms.stage), load the tokens into RAM, remove the staging dir."""
    _, pos, lens = ms.stage(idx, frames, dst)
    t0 = time.time()
    tok = np.load(dst / 'tok.npy')
    shutil.rmtree(dst)
    log(f'  tokens loaded into RAM ({tok.nbytes / 1e9:.1f} GB) in {time.time() - t0:.0f}s')
    return tok, pos, lens


def cell_means(tok, tvid, pos, n_videos, dev):
    cm = CellMeans(n_videos, sl.D, dev)
    for a in range(0, len(tok), 2_000_000):
        b = min(len(tok), a + 2_000_000)
        cm.add(torch.from_numpy(tok[a:b]).to(dev), torch.from_numpy(tvid[a:b]).to(dev),
               torch.from_numpy(pos[a:b].astype(np.int64)).to(dev))
    return cm.finalize(sl.MIN_CELL)


@torch.no_grad()
def encode_level1(sae, norm, mode, cm, tok, pos, tvid, lens, dev, want=('mean',), chunk=32768):
    """Level-1 codes of all tokens -> SparseCodes (GPU), per-frame read-outs {name: (F, m)} (GPU; 'mean' fp32, the
    others fp16) and encode stats (FVE in the model's input space, L0, dead, truncation)."""
    m = sae.n_latents
    F_ = len(lens)
    st = np.r_[0, np.cumsum(lens)]
    sc = SparseCodes(pos, lens, m, K_SPARSE, dev)
    reads = {k: torch.zeros(F_, m, dtype=torch.float32 if k == 'mean' else torch.float16, device=dev) for k in want}
    lens_t = torch.from_numpy(lens.astype(np.int64)).to(dev)
    s1 = torch.zeros(sl.D, dtype=torch.float64, device=dev)
    s2 = sse = l0 = 0.0
    fire = torch.zeros(m, dtype=torch.float64, device=dev)
    for f0, f1 in sl.frame_chunks(lens, chunk):
        lo, hi = st[f0], st[f1]
        if hi == lo:
            continue
        T = torch.from_numpy(np.ascontiguousarray(tok[lo:hi])).to(dev)
        if mode:
            T = cm.center(T, torch.from_numpy(tvid[lo:hi]).to(dev), torch.from_numpy(pos[lo:hi].astype(np.int64)).to(dev),
                          mode)
        x = norm(T)
        z = sae.encode(x, mode='threshold')
        s1 += x.double().sum(0)
        s2 += float(x.double().pow(2).sum())
        sse += float((x - sae.decode(z)).double().pow(2).sum())
        act = z > 0
        fire += act.sum(0).double()
        l0 += float(act.sum())
        sc.put(int(lo), z)
        fr = torch.repeat_interleave(torch.arange(f1 - f0, device=dev), lens_t[f0:f1])
        R = frame_reads(z, fr, lens_t[f0:f1], want=want)
        for k, v in R.items():
            reads[k][f0:f1] = v.to(reads[k].dtype)
        del T, x, z, R
    n = sc.n
    tss = s2 - float(s1.pow(2).sum()) / n
    fr_ = (fire / n).cpu().numpy()
    stats = {'fve': 1 - sse / tss, 'l0_per_token': l0 / n, 'dead_frac': float((fr_ == 0).mean()), 'n_tokens': n,
             'sparse': {**sc.stats, 'frac_tokens_over_K': sc.stats['tokens_over_K'] / n,
                        'frac_mass_dropped': sc.stats['mass_dropped'] / max(sc.stats['mass_total'], 1e-12)}}
    return sc, reads, stats


# ---------------------------------------------------------------------------------------------- selftest
def cmd_selftest(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = np.random.default_rng(0)
    F_, m, K = 40, 60, 8
    lens = g.integers(0, 160, F_)
    lens[3] = 0
    lens[5] = 1
    lens[7] = 1024  # a full frame (edges + every neighbour present)
    pos = np.concatenate([np.sort(g.choice(GRID * GRID, k, replace=False)) for k in lens]).astype(np.int16)
    n = int(lens.sum())
    Z = np.zeros((n, m), np.float32)
    for i in range(n):
        k = g.integers(0, 7)
        Z[i, g.choice(m, k, replace=False)] = g.gamma(1.0, 1.0, k) + 0.05
    sc = SparseCodes(pos, lens, m, K, dev)
    sc.put(0, torch.from_numpy(Z).to(dev))
    assert sc.stats['tokens_over_K'] == 0 and abs(sc.stats['mass_dropped']) < 1e-2 * sc.stats['mass_total']
    Zh = Z.astype(np.float16).astype(np.float32)  # codes are stored fp16
    for r_far in (6, 5):
        ref = level2_reference(Zh, pos, lens, (1, 2), (3, r_far))
        rings = [ring_offsets(1, 2).to(dev), ring_offsets(3, r_far).to(dev)]
        got = torch.cat([level2_features(sc, torch.arange(a, min(n, a + 777), device=dev), rings)
                         for a in range(0, n, 777)]).cpu().numpy()
        d = np.abs(got - ref).max()
        log(f'level2_features vs brute force (r_far {r_far}): max |diff| {d:.2e} over {n} tokens')
        assert d < 1e-6
    assert len(ring_offsets(1, 2)) == 24 and len(ring_offsets(3, 6)) == 13 ** 2 - 5 ** 2
    # truncation bookkeeping
    sc2 = SparseCodes(pos, lens, m, 3, dev)
    sc2.put(0, torch.from_numpy(Z).to(dev))
    over = int(((Z > 0).sum(1) > 3).sum())
    assert sc2.stats['tokens_over_K'] == over and sc2.stats['mass_dropped'] > 0
    log(f'truncation: {over} tokens over K=3, dropped mass fraction {sc2.stats["mass_dropped"] / sc2.stats["mass_total"]:.3f}')
    # frame read-outs vs numpy
    st = np.r_[0, np.cumsum(lens)]
    lt = torch.from_numpy(lens.astype(np.int64)).to(dev)
    fr = torch.repeat_interleave(torch.arange(F_, device=dev), lt)
    R = {k: v.cpu().numpy() for k, v in frame_reads(torch.from_numpy(Zh).to(dev), fr, lt).items()}
    for f in range(F_):
        z = Zh[st[f]:st[f + 1]]
        if len(z) == 0:
            for k in R:
                assert np.abs(R[k][f]).max() == 0
            continue
        top = np.sort(np.vstack([z, np.zeros((max(0, 8 - len(z)), m), np.float32)]), 0)[::-1][:8].mean(0)
        for k, ref_ in (('max', z.max(0)), ('top8', top), ('frac', (z > 0).mean(0)), ('mean', z.mean(0))):
            assert np.abs(R[k][f] - ref_).max() < 2e-3 * max(1, np.abs(ref_).max()), (k, f)
    log('frame_reads (max / top8 / frac / mean) OK, incl. empty and 1-patch frames')
    # block norm vs TokenNorm.fit_blocks
    X = torch.from_numpy(g.normal(size=(5000, 30)).astype(np.float32) * np.r_[np.ones(10), 3 * np.ones(10), 0.1 * np.ones(10)]
                         .astype(np.float32)).to(dev)
    a = fit_block_norm(lambda b: X[b], torch.arange(5000, device=dev), [10, 10, 10], chunk=1234)
    b_ = TokenNorm(30).to(dev).fit_blocks(X, [10, 10, 10])
    assert torch.allclose(a.mean, b_.mean, atol=1e-5) and torch.allclose(a.scale, b_.scale, rtol=1e-4)
    log('fit_block_norm == TokenNorm.fit_blocks')
    log('selftest OK')


# ---------------------------------------------------------------------------------------------- train
def train_sae(prep, N, d_in, width, prefixes, seed, steps, out, extra, dev, norm, init_x, bs=4096, lr=5e-4, warmup=500):
    """prep(b LongTensor of row indices into the pool 0..N-1) -> raw (unnormalised) inputs (B, d_in)."""
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    sae = MatryoshkaBatchTopKSAE(d_in=d_in, n_latents=width, prefixes=prefixes, k=K_L2, seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init_x)))
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    perm, p = torch.randperm(N, generator=gen, device=dev), 0
    history, t0 = [], time.time()
    for step in range(steps):
        if p + bs > N:
            perm, p = torch.randperm(N, generator=gen, device=dev), 0
        b = perm[p:p + bs]
        p += bs
        logs, gn = _train_step(sae, opt, norm(prep(b)), step, steps, lr, warmup, 1.0)
        if step % 1000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            history.append(logs)
            log(f"    step {step}/{steps} fve {logs['fve']:.4f} l0 {logs['l0']:.1f} dead {logs['n_dead']} {logs['elapsed_s']}s")
    sae.eval()
    dead = int((sae.tokens_since_fired >= sae.dead_tokens).sum())
    save_checkpoint(out / 'sae.pt', sae, norm, extra={**extra, 'seed': seed})
    (out / 'train.json').write_text(json.dumps({**extra, 'seed': seed, 'n_pool': N, 'steps': steps,
                                                'token_presentations': steps * bs, 'train_time_s': time.time() - t0,
                                                'dead_at_end_train': dead, 'dead_frac_train': dead / width,
                                                'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu',
                                                'history': history}, indent=1))
    log(f'    trained in {time.time() - t0:.0f}s, final fve {history[-1]["fve"]:.4f}, dead {dead}/{width}')


def cmd_train(args):
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out_dir)
    loc = ms.local_dir()
    D_ = DOM[args.domain]
    idx = ssp.StoreIndex(args.domain)
    tr, _, _, _, train_v, _ = idx.split(args.domain, sl.N_TRAIN[args.domain], 0)
    tr = np.sort(tr)
    if args.max_train_frames:
        tr = np.sort(np.random.default_rng(0).choice(tr, args.max_train_frames, replace=False))
    vids, fvid = sl.video_index(idx.obs[tr])
    log(f'{args.domain}: {len(train_v)} train videos, {len(tr):,} train frames; level 1 = {D_["l1"]}; '
        f'rings NEAR {R_NEAR}, FAR (3, {D_["r_far"]})')
    tok, pos, lens = load_ram(idx, tr, loc / 'train')
    n = len(pos)
    tvid = np.repeat(fvid, lens)
    cm = cell_means(tok, tvid, pos, len(vids), dev) if D_['center'] else None
    rng = np.random.default_rng(args.pool_seed)  # the levers 'uniform' pool
    cap = min(args.cap or sl.CAP[args.domain], n)
    pool = torch.from_numpy(np.sort(rng.choice(n, cap, replace=False))).to(dev)
    rings = rings_for(args.domain, dev)
    meta = {'domain': args.domain, 'n_train_frames': len(tr), 'n_train_tokens': n, 'cap': cap, 'pool_seed': args.pool_seed,
            'level1': D_['l1'], 'r_near': R_NEAR, 'r_far': [R_NEAR[1] + 1, D_['r_far']], 'K_sparse': K_SPARSE,
            'n_offsets': [len(r) for r in rings], 'steps': args.steps, 'seeds': {}}
    for seed in args.seeds:
        t0 = time.time()
        sae1, norm1, _ = load_sae(l1_path(args.domain, seed), dev)
        m = sae1.n_latents
        sc, reads, st1 = encode_level1(sae1, norm1, D_['center'], cm, tok, pos, tvid, lens, dev, want=('mean',))
        del sae1
        X3 = reads.pop('mean')
        log(f'  level-1 s{seed}: encoded {n:,} train tokens in {time.time() - t0:.0f}s, FVE {st1["fve"]:.4f}, L0 '
            f'{st1["l0_per_token"]:.2f}, tokens over K {100 * st1["sparse"]["frac_tokens_over_K"]:.3f}%, mass dropped '
            f'{100 * st1["sparse"]["frac_mass_dropped"]:.4f}%')
        # ---- level 2
        g = np.random.default_rng(seed)
        samp = pool[torch.from_numpy(np.sort(g.choice(cap, min(100_000, cap), replace=False))).to(dev)]
        feat = lambda b: level2_features(sc, b, rings)  # noqa: E731
        t1 = time.time()
        norm2 = fit_block_norm(feat, samp, [m, m, m])
        x0, cnts = level2_features(sc, samp[:20_000], rings, with_counts=True)
        nb = {'near_fg_neighbours_mean': float(cnts[0].float().mean()), 'far_fg_neighbours_mean': float(cnts[1].float().mean()),
              'frac_no_near_neighbour': float((cnts[0] == 0).float().mean()),
              'frac_no_far_neighbour': float((cnts[1] == 0).float().mean()),
              'nnz_per_block': [float((x0[:, k * m:(k + 1) * m] > 0).sum(1).float().mean()) for k in range(3)],
              'block_scales': [float(norm2.scale[k * m]) for k in range(3)]}
        del x0
        log(f'  level-2 block norm in {time.time() - t1:.0f}s; neighbourhood stats {json.dumps(nb)}')
        extra = {'level': 2, 'domain': args.domain, 'level1': str(l1_path(args.domain, seed).relative_to(REPO)),
                 'level1_seed': seed, 'r_near': R_NEAR, 'r_far': [R_NEAR[1] + 1, D_['r_far']], 'K_sparse': K_SPARSE,
                 'cap': cap, 'pool_seed': args.pool_seed, 'neighbourhood': nb}
        log(f'  level-2 s{seed}: d_in {3 * m}, {L2_WIDTH} latents')
        init = feat(samp[:32_768])
        train_sae(lambda b: feat(pool[b]), cap, 3 * m, L2_WIDTH, (L2_WIDTH // 8, L2_WIDTH // 4, L2_WIDTH // 2, L2_WIDTH),
                  seed, args.steps, out / 'sae' / f'l2_s{seed}', extra, dev, norm2, init)
        del init
        # ---- level 3
        norm3 = TokenNorm(m).to(dev).fit(X3, seed=seed)
        gi = torch.from_numpy(np.sort(g.choice(len(X3), min(100_000, len(X3)), replace=False))).to(dev)
        log(f'  level-3 s{seed}: {len(X3):,} frame tokens, d_in {m}, {L3_WIDTH} latents')
        train_sae(lambda b: X3[b], len(X3), m, L3_WIDTH, (L3_WIDTH // 8, L3_WIDTH // 4, L3_WIDTH // 2, L3_WIDTH), seed,
                  args.steps, out / 'sae' / f'l3_s{seed}',
                  {'level': 3, 'domain': args.domain, 'level1': str(l1_path(args.domain, seed).relative_to(REPO)),
                   'level1_seed': seed, 'input': 'mean level-1 code over the frame foreground patches'}, dev, norm3, X3[gi])
        meta['seeds'][seed] = {'level1_encode': st1, 'neighbourhood': nb}
        del sc, X3, reads
        torch.cuda.empty_cache()
        (out / 'train_meta.json').write_text(json.dumps(meta, indent=1))
    log('train done')


# ---------------------------------------------------------------------------------------------- evaluate
@torch.no_grad()
def encode_level2(sae, norm, sc, rings, lens, dev, chunk=8192):
    F_ = len(lens)
    m2 = sae.n_latents
    st = np.r_[0, np.cumsum(lens)]
    lens_t = torch.from_numpy(lens.astype(np.int64)).to(dev)
    reads = {k: torch.zeros(F_, m2, dtype=torch.float16, device=dev) for k in READS[2]}
    s1 = torch.zeros(sae.d_in, dtype=torch.float64, device=dev)
    s2 = sse = l0 = 0.0
    fire = torch.zeros(m2, dtype=torch.float64, device=dev)
    for f0, f1 in sl.frame_chunks(lens, chunk):
        lo, hi = int(st[f0]), int(st[f1])
        if hi == lo:
            continue
        x = norm(level2_features(sc, torch.arange(lo, hi, device=dev), rings))
        z = sae.encode(x, mode='threshold')
        s1 += x.double().sum(0)
        s2 += float(x.double().pow(2).sum())
        sse += float((x - sae.decode(z)).double().pow(2).sum())
        act = z > 0
        fire += act.sum(0).double()
        l0 += float(act.sum())
        fr = torch.repeat_interleave(torch.arange(f1 - f0, device=dev), lens_t[f0:f1])
        for k, v in frame_reads(z, fr, lens_t[f0:f1], want=READS[2]).items():
            reads[k][f0:f1] = v.half()
        del x, z
    n = int(st[-1])
    tss = s2 - float(s1.pow(2).sum()) / n
    fr_ = (fire / n).cpu().numpy()
    return reads, {'fve': 1 - sse / tss, 'l0_per_token': l0 / n, 'dead_frac': float((fr_ == 0).mean()), 'n_tokens': n}


@torch.no_grad()
def encode_level3(sae, norm, X, dev, chunk=65536):
    out = torch.zeros(len(X), sae.n_latents, dtype=torch.float16, device=dev)
    s1 = torch.zeros(sae.d_in, dtype=torch.float64, device=dev)
    s2 = sse = l0 = 0.0
    fire = torch.zeros(sae.n_latents, dtype=torch.float64, device=dev)
    for a in range(0, len(X), chunk):
        x = norm(X[a:a + chunk].to(dev).float())
        z = sae.encode(x, mode='threshold')
        s1 += x.double().sum(0)
        s2 += float(x.double().pow(2).sum())
        sse += float((x - sae.decode(z)).double().pow(2).sum())
        fire += (z > 0).sum(0).double()
        l0 += float((z > 0).sum())
        out[a:a + chunk] = z.half()
    n = len(X)
    tss = s2 - float(s1.pow(2).sum()) / n
    fr_ = (fire / n).cpu().numpy()
    return out, {'fve': 1 - sse / tss, 'l0_per_frame': l0 / n, 'dead_frac': float((fr_ == 0).mean()), 'n_frames': n}


@torch.no_grad()
def l2_token_acts(sae, norm, sc, rings, J, lo, hi, dev):
    """Level-2 activations (threshold inference) of neurons J for global tokens lo .. hi -> (n, |J|)."""
    J = torch.as_tensor(J, device=dev)
    x = norm(level2_features(sc, torch.arange(lo, hi, device=dev), rings))
    pre = torch.relu((x - sae.b_dec) @ sae.W_enc[:, J] + sae.b_enc[J])
    return pre * (pre > sae.threshold)


@torch.no_grad()
def l2_regional(sae, norm, sc, rings, J, pos, lens, anchors, dev, chunk=16384):
    """Ants dot-Voronoi read of level-2 neurons J (sae_levers.column_codes rule) -> yellow max, blue max, frame max."""
    st = np.r_[0, np.cumsum(lens)]
    F_ = len(lens)
    outs = [torch.zeros(F_, len(J), device=dev) for _ in range(3)]
    lens_t = torch.from_numpy(lens.astype(np.int64)).to(dev)
    for f0, f1 in sl.frame_chunks(lens, chunk):
        lo, hi = int(st[f0]), int(st[f1])
        if hi == lo:
            continue
        z = l2_token_acts(sae, norm, sc, rings, J, lo, hi, dev)
        P = torch.from_numpy(pos[lo:hi].astype(np.int64)).to(dev)
        fr = torch.repeat_interleave(torch.arange(f1 - f0, device=dev), lens_t[f0:f1])
        outs[2][f0:f1].index_reduce_(0, fr, z, 'amax', include_self=True)
        a = anchors[f0:f1][fr]
        cy, cx = (P // GRID).float() + 0.5, (P % GRID).float() + 0.5
        df = torch.hypot(cx - a[:, 0], cy - a[:, 1])
        dy = torch.hypot(cx - a[:, 2], cy - a[:, 3])
        db = torch.hypot(cx - a[:, 4], cy - a[:, 5])
        for o, msk in ((outs[0], (df <= ms.VOR_D) & (dy < db)), (outs[1], (df <= ms.VOR_D) & (db < dy))):
            if msk.any():
                o[f0:f1].index_reduce_(0, fr[msk], z[msk], 'amax', include_self=True)
    return [o.cpu().numpy() for o in outs]


def regional_read(ys, bs, J, j, rows, ok, ygt, bgt):
    c = J.index(j)
    r1 = rows & ok
    r2 = r1 & (ygt ^ bgt)
    return {'y2f_auroc': float(ms.rank_auc(ys[r1][:, [c]], ygt[r1])[0]),
            'b2f_auroc': float(ms.rank_auc(bs[r1][:, [c]], bgt[r1])[0]),
            'disc_auroc': float(ms.rank_auc((ys - bs)[r2][:, [c]], ygt[r2])[0]),
            'n_frames': int(r1.sum()), 'n_exactly_one': int(r2.sum())}


def regional_cf(ys, bs, J, res, half, ok, ygt, bgt):
    dirs = [{'neuron': d['auc']['neuron'], 'test_half': d['test_half'],
             **regional_read(ys, bs, J, d['auc']['neuron'], half == d['test_half'], ok, ygt, bgt)}
            for d in res['groom_any']['dirs']]
    o = {q: float(np.mean([d[q] for d in dirs])) for q in ('y2f_auroc', 'b2f_auroc', 'disc_auroc')}
    o['dirs'] = dirs
    return o


def collective(X, proxies, half):
    """Cross-fitted Spearman of every column of X (F, m) (np float32) with each proxy: neuron picked on |rho| of half
    A, signed rho of that neuron on half B (sign taken from half A), averaged over the two directions."""
    out = {}
    for name, (v, ok) in proxies.items():
        rho = {}
        for h in (0, 1):
            r = ok & (half == h)
            rho[h] = np.nan_to_num(ms.spearman_cols(X[r], v[r]))
        dirs = []
        for a, b in ((0, 1), (1, 0)):
            j = int(np.argmax(np.abs(rho[a])))
            dirs.append({'neuron': j, 'select_rho': float(rho[a][j]), 'test_rho': float(np.sign(rho[a][j]) * rho[b][j])})
        out[name] = {'cf_abs_rho': float(np.mean([d['test_rho'] for d in dirs])), 'dirs': dirs,
                     'n': int(ok.sum())}
    return out


def fg_spread(pos, lens):
    """Radius of gyration (grid units) of the foreground patch centres per frame (0 for < 2 patches)."""
    f = np.repeat(np.arange(len(lens)), lens)
    r, c = (pos // GRID).astype(np.float64), (pos % GRID).astype(np.float64)
    n = np.maximum(lens, 1).astype(np.float64)
    mr = np.bincount(f, r, len(lens)) / n
    mc = np.bincount(f, c, len(lens)) / n
    v = np.bincount(f, (r - mr[f]) ** 2 + (c - mc[f]) ** 2, len(lens)) / n
    return np.sqrt(v)


@torch.no_grad()
def make_sheets(key, ro, reads_ro, res, lab, labels_sheet, valid, half, J_acts, lens, sdir, per_video=2, n_show=16):
    """J_acts(j, lo, hi) -> per-token activation of neuron j (n,) for tokens lo..hi."""
    st = np.r_[0, np.cumsum(lens)]
    index = []
    for b in labels_sheet:
        e = res[b]['dirs'][0]
        j = e['auc']['neuron']
        x = reads_ro[:, j].astype(np.float32)
        x = np.where(valid[b] & (half == e['test_half']), x, 0)
        pick, seen = [], {}
        for i in np.argsort(-x, kind='stable'):
            if x[i] <= 0 or len(pick) == n_show:
                break
            v = lab.obs.iat[i]
            if seen.get(v, 0) >= per_video:
                continue
            seen[v] = seen.get(v, 0) + 1
            pick.append(int(i))
        boxes = []
        for i in pick:
            z = J_acts(j, int(st[i]), int(st[i + 1]))
            boxes.append(int(J_acts.pos[st[i] + int(z.argmax())]))
        fn = f'{key}__{b}__auc_n{j}.jpg'
        n_pos = sl.contact_sheet(sdir / fn, None, pick, lab, b, x, boxes)
        index.append({'model': key, 'readout': ro, 'label': b, 'neuron': j, 'n_shown': len(pick),
                      'n_label_positive': int(n_pos), 'precision_shown': n_pos / max(len(pick), 1),
                      'test_half_base_rate': res[b]['base_half'][e['test_half']],
                      'frames': [{'obs': lab.obs.iat[i], 'frame_idx': int(lab.frame_idx.iat[i]), 'pos': p,
                                  'label': bool(lab[b].iat[i]), 'act': float(x[i])} for i, p in zip(pick, boxes)],
                      'file': fn})
    return index


@torch.no_grad()
def cmd_evaluate(args):
    dev = torch.device('cuda')
    out = Path(args.out_dir)
    loc = ms.local_dir()
    lout = loc / 'out'
    for d in ('eval', 'codes_best', 'sheets'):
        (lout / d).mkdir(parents=True, exist_ok=True)
    D_ = DOM[args.domain]
    idx = ssp.StoreIndex(args.domain)
    _, ev, _, _, _, _ = idx.split(args.domain, sl.N_TRAIN[args.domain], 0)
    ev = np.sort(ev)
    if args.max_eval_frames:
        ev = ev[np.linspace(0, len(ev) - 1, args.max_eval_frames).astype(int)]
    lab, _, valid = ms.eval_labels(args.domain, idx, ev)
    labels = LABELS[args.domain]
    vids, fvid = sl.video_index(lab.obs.values)
    unit = lab.obs.str.rsplit('_', n=2).str[0].values if args.domain == 'mice' else lab.obs.values
    units = sorted(set(unit))
    half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
    log(f'{args.domain}: {len(ev):,} eval frames, {len(vids)} videos, {len(units)} cross-fit units, frames per half '
        f'{np.bincount(half).tolist()}; labels ' + json.dumps({b: [int(valid[b].sum()), round(float(lab[b][valid[b]].mean()), 4)]
                                                              for b in labels}))
    if not args.max_eval_frames:  # same frames / halves as sae_levers
        ref = pd.read_parquet(LEV / args.domain / 'labels.parquet')
        assert (ref['row'].values == lab['row'].values).all() and (np.load(LEV / args.domain / 'half.npy') == half).all()
        log('  eval frames and halves identical to sae_levers')
    tok, pos, lens = load_ram(idx, ev, loc / 'eval')
    tvid = np.repeat(fvid, lens)
    rank = np.argsort(np.argsort(lens + np.random.default_rng(0).random(len(lens)) * 1e-3))
    decile = np.minimum(rank * 10 // len(lens), 9)
    cm = cell_means(tok, tvid, pos, len(vids), dev) if D_['center'] else None
    Yt = torch.from_numpy(lab[labels].values.astype(bool)).to(dev)
    rings = rings_for(args.domain, dev)
    spread = fg_spread(pos, lens)
    if args.domain == 'mice':
        proxies = {'fg_spread': (spread, lens > 1), 'fg_count': (lens.astype(np.float64), np.ones(len(lens), bool))}
    else:
        disp, _ = ms.ants_dispersion(lab)
        proxies = {'tracker_dispersion': (disp, np.isfinite(disp)),
                   'fg_count': (lens.astype(np.float64), np.ones(len(lens), bool))}
    prox_meta = {k: {'n': int(ok.sum()), 'spearman_with_fg_count': float(ms.spearman_cols(
        lens[ok].astype(np.float64)[:, None], v[ok])[0])} for k, (v, ok) in proxies.items()}
    log(f'  collective proxies (NOT human labels): {json.dumps(prox_meta)}')
    anchors = None
    if args.domain == 'ants':
        A, ok = ms.ants_anchors(lab)
        At = torch.from_numpy(np.nan_to_num(A, nan=-1e4)).float().to(dev)
        ygt, bgt = lab['groom_yellow'].values.astype(bool), lab['groom_blue'].values.astype(bool)
        anchors = (At, ok)
        log(f'  ants anchors: {ok.sum():,}/{len(ok):,} frames with focal centroid + both colour dots')
    meta = {'domain': args.domain, 'n_eval_frames': len(ev), 'n_eval_videos': len(vids), 'units': len(units),
            'level1': D_['l1'], 'proxies': prox_meta, 'seeds': args.seeds}
    sheet_index = []

    def save_eval(key, level, ro, seed, res, stats, extra=None):
        e = {'key': key, 'level': level, 'readout': ro, 'seed': seed, 'stats': stats, 'labels': res, **(extra or {})}
        (lout / 'eval' / f'{key}.json').write_text(json.dumps(e, indent=1, default=float))
        f = lambda b: (f"{b}: cf AUROC {res[b]['cf_auroc']:.3f} cf AP {res[b]['cf_ap']:.4f} top1 {res[b]['cf_top1']:.3f} "  # noqa: E731
                       f"sc {res[b]['cf_size_ctrl_auroc']:.3f}")
        log(f'  {key}: ' + ' | '.join(f(b) for b in labels)
            + (f" | regional {json.dumps({k: round(v, 3) for k, v in e['regional'].items() if k != 'dirs'})}"
               if 'regional' in e else '')
            + (f" | collective {json.dumps({k: round(v['cf_abs_rho'], 3) for k, v in e['collective'].items()})}"
               if 'collective' in e else ''))

    def score(key, codes_np):
        t0 = time.time()
        c = torch.from_numpy(codes_np).to(dev)
        res, _ = sl.score_model(c, Yt, labels, valid, half, decile, dev)
        del c
        js = sorted({d[s]['neuron'] for b in labels for d in res[b]['dirs'] for s in ('auc', 'ap', 'top1')})
        np.savez_compressed(lout / 'codes_best' / f'{key}.npz', neurons=np.array(js), codes=codes_np[:, js])
        log(f'    scored {key} in {time.time() - t0:.0f}s')
        return res

    if args.domain == 'ants' and args.deployed:
        sae_d, norm_d, _ = load_sae(ms.DEPLOYED_ANTS, dev)
        At, ok = anchors
        ys, bs = sl.column_codes(sae_d, norm_d, None, None, [90], tok, pos, tvid, lens, dev, region=At)
        meta['deployed_n90'] = {'all_eval': regional_read(ys, bs, [90], 90, np.ones(len(ok), bool), ok, ygt, bgt),
                                'halves': [regional_read(ys, bs, [90], 90, half == h, ok, ygt, bgt) for h in (0, 1)]}
        log(f"  deployed antsfg n90 dot-Voronoi: {json.dumps(meta['deployed_n90'])}")
        del sae_d
    for seed in args.seeds:
        t0 = time.time()
        torch.backends.cuda.matmul.allow_tf32 = False  # level 1 as sae_levers evaluated it
        sae1, norm1, _ = load_sae(l1_path(args.domain, seed), dev)
        sc, reads1, st1 = encode_level1(sae1, norm1, D_['center'], cm, tok, pos, tvid, lens, dev, want=READS[1])
        X3 = reads1['mean'].cpu()
        reads1 = {k: v.half().cpu().numpy() for k, v in reads1.items()}
        torch.cuda.empty_cache()
        log(f'  level-1 s{seed} encoded in {time.time() - t0:.0f}s: FVE {st1["fve"]:.4f} L0 {st1["l0_per_token"]:.2f} '
            f'over-K {100 * st1["sparse"]["frac_tokens_over_K"]:.3f}% mass dropped {100 * st1["sparse"]["frac_mass_dropped"]:.4f}%')
        res1 = {}
        for ro in READS[1]:
            key = f'L1_{ro}_s{seed}'
            res1[ro] = score(key, reads1[ro])
            extra = {}
            if ro == 'max' and anchors is not None:
                At, ok = anchors
                J = sorted({d['auc']['neuron'] for d in res1[ro]['groom_any']['dirs']})
                ys, bs = sl.column_codes(sae1, norm1, None, None, J, tok, pos, tvid, lens, dev, region=At)
                extra['regional'] = regional_cf(ys, bs, J, res1[ro], half, ok, ygt, bgt)
            if ro == 'mean':
                extra['collective'] = collective(reads1[ro].astype(np.float32), proxies, half)
            save_eval(key, 1, ro, seed, res1[ro], st1, extra)
        del sae1
        # ---- level 2
        t0 = time.time()
        torch.backends.cuda.matmul.allow_tf32 = True  # level-2 inputs are 12,288-dim (mice)
        sae2, norm2, _ = load_sae(out / 'sae' / f'l2_s{seed}' / 'sae.pt', dev)
        reads2, st2 = encode_level2(sae2, norm2, sc, rings, lens, dev)
        reads2 = {k: v.cpu().numpy() for k, v in reads2.items()}
        torch.cuda.empty_cache()
        log(f'  level-2 s{seed} encoded in {time.time() - t0:.0f}s: FVE {st2["fve"]:.4f} L0 {st2["l0_per_token"]:.2f} '
            f'dead {100 * st2["dead_frac"]:.1f}%')
        tj = json.loads((out / 'sae' / f'l2_s{seed}' / 'train.json').read_text())
        st2.update(dead_frac_train=tj['dead_frac_train'], train_time_s=tj['train_time_s'])
        for ro in READS[2]:
            key = f'L2_{ro}_s{seed}'
            res = score(key, reads2[ro])
            extra = {}
            if ro == 'max' and anchors is not None:
                At, ok = anchors
                J = sorted({d['auc']['neuron'] for d in res['groom_any']['dirs']})
                ys, bs, fm = l2_regional(sae2, norm2, sc, rings, J, pos, lens, At, dev)
                ref_ = reads2['max'][:, J].astype(np.float32)
                chk = {'max_abs_diff': float(np.abs(fm - ref_).max()), 'frac_frames_diff_gt_1e-2':
                       float((np.abs(fm - ref_) > 1e-2).any(1).mean()), 'corr': float(np.corrcoef(fm.ravel(), ref_.ravel())[0, 1])}
                log(f'    level-2 per-column re-read vs frame max: {json.dumps(chk)}')
                extra['regional'] = regional_cf(ys, bs, J, res, half, ok, ygt, bgt)
                extra['regional_reread_check'] = chk
            save_eval(key, 2, ro, seed, res, st2, extra)
            if seed == args.seeds[0] and ro in ('max', 'top8'):
                def J_acts(j, lo, hi):
                    return l2_token_acts(sae2, norm2, sc, rings, [j], lo, hi, dev)[:, 0].cpu().numpy()
                J_acts.pos = pos
                sheet_index += make_sheets(key, ro, reads2[ro], res, lab,
                                           ['nose_nose', 'nose_tail'] if args.domain == 'mice' else ['groom_any'],
                                           valid, half, J_acts, lens, lout / 'sheets')
        del reads2, sae2
        # ---- level 3
        sae3, norm3, _ = load_sae(out / 'sae' / f'l3_s{seed}' / 'sae.pt', dev)
        c3, st3 = encode_level3(sae3, norm3, X3, dev)
        c3 = c3.cpu().numpy()
        tj = json.loads((out / 'sae' / f'l3_s{seed}' / 'train.json').read_text())
        st3.update(dead_frac_train=tj['dead_frac_train'], train_time_s=tj['train_time_s'])
        key = f'L3_code_s{seed}'
        res = score(key, c3)
        save_eval(key, 3, 'code', seed, res, st3, {'collective': collective(c3.astype(np.float32), proxies, half)})
        del sc, X3, c3, sae3, reads1
        torch.cuda.empty_cache()
    (lout / f'eval_meta.json').write_text(json.dumps(meta, indent=1, default=float))
    (lout / 'sheets' / 'index.json').write_text(json.dumps(sheet_index, indent=1))
    lab.drop(columns=['frame_path']).to_parquet(lout / 'labels.parquet')
    np.save(lout / 'half.npy', half)
    np.save(lout / 'lens.npy', lens.astype(np.int32))
    np.save(lout / 'fg_spread.npy', spread.astype(np.float32))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(lout, out, dirs_exist_ok=True)
    log(f'results copied to {out}')
    cmd_summary(args)


# ---------------------------------------------------------------------------------------------- video (mice, CPU)
def cmd_video(args):
    """diag_video_level measurements of the selected neurons of every read-out, raw and fg-count controlled."""
    import diag_switch_regions as dsr
    from src.eci.domain import MiceDomain
    assert args.domain == 'mice'
    out = Path(args.out_dir)
    vout = out / 'video'
    vout.mkdir(parents=True, exist_ok=True)
    loc = ms.local_dir()
    t0 = time.time()
    A = dsr.design()
    lab = pd.read_parquet(out / 'labels.parquet')
    half_f = np.load(out / 'half.npy')
    lens = np.load(out / 'lens.npy').astype(np.float64)
    F_ = len(lab)
    assert F_ == 172800
    vi = {o: i for i, o in enumerate(A.observation_id)}
    fv = lab.obs.map(vi).values.astype(np.int64)
    shutil.copyfile(dsr.ANN, loc / 'ann.csv')
    nfr = pd.read_csv(loc / 'ann.csv', usecols=['observation_id'])['observation_id'].value_counts()
    (loc / 'ann.csv').unlink()
    wstart = MiceDomain().window_start(nfr.loc[A.observation_id].values, A.stage.values)
    inwin = lab.frame_idx.values >= wstart[fv]
    nwin = np.bincount(fv[inwin], minlength=len(A))
    assert (nwin == 900).all(), np.unique(nwin)
    w = inwin.astype(np.float64)
    vh = pd.Series(half_f).groupby(fv).first()
    assert (vh.values == A.half.values).all(), 'cross-fit halves differ from diag_video_level'

    def vmean(fr):
        return np.bincount(fv, weights=np.asarray(fr, np.float64) * w, minlength=len(A)) / 900

    fg = vmean(lens)
    log(f'video step: {len(A)} videos, 900 in-window eval frames each; per-video fg patches/frame '
        f'{fg.min():.0f}..{fg.max():.0f}')

    def resid(x):
        o = np.empty_like(x)
        for h in (0, 1):
            m = (A.half == h).values
            X_ = np.c_[np.ones(m.sum()), fg[m]]
            o[m] = x[m] - X_ @ np.linalg.lstsq(X_, x[m], rcond=None)[0]
        return o

    R = {'fg_per_video': {'mean': float(fg.mean())}, 'candidates': {}}
    for b in ('nose_nose', 'nose_tail'):
        R['fg_per_video'][f'video_r_rate_{b}'] = float(np.mean([dsr.corr(fg[(A.half == h).values],
                                                                        A[f'{b}_rate'].values[(A.half == h).values])['pearson']
                                                               for h in (0, 1)]))
    PV = A[['observation_id', 'pool', 'stage', 'half']].copy()
    PV['fg_per_frame'] = fg
    for p in sorted((out / 'eval').glob('*.json')):
        e = json.loads(p.read_text())
        key = e['key']
        z = np.load(out / 'codes_best' / f'{key}.npz')
        neurons, codes = list(z['neurons']), z['codes']
        for sel in ('auc', 'top1'):
            for b in ('nose_nose', 'nose_tail'):
                x = np.full(len(A), np.nan)
                picks = {}
                for d in e['labels'][b]['dirs']:
                    j = d[sel]['neuron']
                    xv = vmean(codes[:, neurons.index(j)])
                    m = (A.half == d['test_half']).values
                    x[m] = xv[m]
                    picks[d['test_half']] = j
                assert np.isfinite(x).all()
                raw, _, _ = dsr.video_metrics(x, A, b)
                ctl, _, _ = dsr.video_metrics(resid(x), A, b)
                name = f'{key}|{sel}|{b}'
                PV[name] = x
                R['candidates'][name] = {'key': key, 'level': e['level'], 'readout': e['readout'], 'seed': e['seed'],
                                         'selection': sel, 'behaviour': b, 'neurons_by_test_half': picks,
                                         'r_with_fg_per_video': float(np.mean([dsr.corr(x[(A.half == h).values],
                                                                                       fg[(A.half == h).values])['pearson']
                                                                              for h in (0, 1)])),
                                         'raw': raw, 'fg_controlled': ctl,
                                         'nes_useful_controlled': bool(ctl['video_r_rate'] >= 0.7 and
                                                                       ctl['delta_r_pooled'] >= 0.5)}
    R['truth'] = {b: {q: {sw: dsr.ttest(dsr.plain_deltas(A[f'{b}_{q}'].values, sorted(A.pool.unique()),
                                                          {(p_, s_): i for i, (p_, s_) in enumerate(zip(A.pool, A.stage))})[sw])
                          for sw in dsr.SW} for q in ('rate', 'bpm')} for b in ('nose_nose', 'nose_tail')}
    (vout / 'results.json').write_text(json.dumps(R, indent=1, default=float))
    PV.to_parquet(vout / 'per_video.parquet')
    log(f'video step done in {time.time() - t0:.0f}s -> {vout}')
    video_table(out)


def video_table(out):
    import diag_switch_regions as dsr
    R = json.loads((out / 'video' / 'results.json').read_text())
    C_ = R['candidates']
    L = ['# Video level (mice), held-out halves; raw | after regressing the per-video mean on fg patches/frame\n',
         f"Per-video mean fg patches/frame itself: video r with rate nose_nose {R['fg_per_video']['video_r_rate_nose_nose']:.2f}, "
         f"nose_tail {R['fg_per_video']['video_r_rate_nose_tail']:.2f}\n"]
    for b in ('nose_nose', 'nose_tail'):
        tr = R['truth'][b]
        L.append(f'\n## {b}\n')
        L.append('annotated rate dz: ' + ', '.join(f"{sw} {tr['rate'][sw]['dz']:+.2f} (p {tr['rate'][sw]['p']:.3f})" for sw in dsr.SW))
        L.append('annotated bouts/min dz: ' + ', '.join(f"{sw} {tr['bpm'][sw]['dz']:+.2f} (p {tr['bpm'][sw]['p']:.3f})" for sw in dsr.SW))
        L.append('\n| model / read-out | sel | r(fg) | video r rate raw / ctl | r bouts raw / ctl | pooled d-r raw / ctl | '
                 + ' | '.join(f'{sw} dz raw / ctl (sign vs rate)' for sw in dsr.SW) + ' | NES-useful (ctl) |')
        L.append('|' + '---|' * (7 + len(dsr.SW)))
        groups = {}
        for n, c in C_.items():
            if c['behaviour'] == b:
                groups.setdefault((c['level'], c['readout'], c['selection']), []).append(c)
        for (lv, ro, sel), cs in sorted(groups.items()):
            cs = sorted(cs, key=lambda c: c['seed'])
            ms_ = lambda f: f'{np.mean([f(c) for c in cs]):.2f}+-{np.std([f(c) for c in cs], ddof=1) if len(cs) > 1 else 0:.2f}'  # noqa: E731
            cells = []
            for sw in dsr.SW:
                cells.append(ms_(lambda c: c['raw']['effects'][sw]['neuron']['dz']) + ' / '
                             + ms_(lambda c: c['fg_controlled']['effects'][sw]['neuron']['dz'])
                             + f" ({sum(c['fg_controlled']['effects'][sw]['sign_agree_rate'] for c in cs)}/{len(cs)})")
            L.append(f'| L{lv} {ro} | {sel} | {ms_(lambda c: c["r_with_fg_per_video"])} | '
                     f'{ms_(lambda c: c["raw"]["video_r_rate"])} / {ms_(lambda c: c["fg_controlled"]["video_r_rate"])} | '
                     f'{ms_(lambda c: c["raw"]["video_r_bpm"])} / {ms_(lambda c: c["fg_controlled"]["video_r_bpm"])} | '
                     f'{ms_(lambda c: c["raw"]["delta_r_pooled"])} / {ms_(lambda c: c["fg_controlled"]["delta_r_pooled"])} | '
                     + ' | '.join(cells) + f" | {sum(c['nes_useful_controlled'] for c in cs)}/{len(cs)} |")
    (out / 'video' / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------- summary
METRICS = ('cf_auroc', 'cf_ap', 'cf_ap_at_auroc', 'cf_top1', 'cf_size_ctrl_auroc', 'cf_auroc_rate', 'cf_top1_rate',
           'insample_auroc', 'base_rate')


def ms_(v):
    return {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 'per_seed': [float(x) for x in v]}


def gain(a, ref, b):
    d_auc = a[b]['cf_auroc']['mean'] - ref[b]['cf_auroc']['mean']
    r_ap = a[b]['cf_ap']['mean'] / ref[b]['cf_ap']['mean']
    ag = d_auc >= 0.05 and a[b]['cf_auroc']['mean'] > max(ref[b]['cf_auroc']['per_seed'])
    pg = r_ap >= 1.5 and a[b]['cf_ap']['mean'] > max(ref[b]['cf_ap']['per_seed'])
    return {'d_cf_auroc': d_auc, 'ratio_cf_ap': r_ap, 'auroc_gain': bool(ag), 'ap_gain': bool(pg), 'gain': bool(ag or pg)}


def cmd_summary(args):
    out = Path(args.out_dir)
    E = [json.loads(p.read_text()) for p in sorted((out / 'eval').glob('*.json'))]
    labels = LABELS[args.domain]
    agg = {}
    for name in dict.fromkeys(f"L{e['level']}_{e['readout']}" for e in E):
        es = sorted([e for e in E if f"L{e['level']}_{e['readout']}" == name], key=lambda e: e['seed'])
        a = {'seeds': [e['seed'] for e in es]}
        for q in ('fve', 'l0_per_token', 'l0_per_frame', 'dead_frac', 'dead_frac_train'):
            v = [e['stats'][q] for e in es if q in e['stats']]
            if v:
                a[q] = ms_(v)
        for b in labels:
            a[b] = {q: ms_([e['labels'][b][q] for e in es]) for q in METRICS}
        if 'regional' in es[0]:
            a['regional'] = {q: ms_([e['regional'][q] for e in es]) for q in ('y2f_auroc', 'b2f_auroc', 'disc_auroc')}
        if 'collective' in es[0]:
            a['collective'] = {k: ms_([e['collective'][k]['cf_abs_rho'] for e in es]) for k in es[0]['collective']}
        agg[name] = a
    crit = {}
    for name, a in agg.items():
        lv = int(name[1])
        if lv == 1:
            continue
        c = {}
        refs = {'vs_L1_same_readout': f"L1_{name.split('_', 1)[1]}"} if lv == 2 else {'vs_L1_mean': 'L1_mean'}
        refs['vs_L1_max'] = 'L1_max'
        for rn, ref in refs.items():
            if ref not in agg:
                continue
            c[rn] = {b: gain(a, agg[ref], b) for b in labels}
        if args.domain == 'mice':
            c['stretch_top1_ge_0.40'] = {b: bool(a[b]['cf_top1']['mean'] >= 0.40) for b in labels}
            c['frac_of_ceiling_ap_above_base'] = {b: (a[b]['cf_ap']['mean'] - a[b]['base_rate']['mean'])
                                                  / (sl.CEILING_AP[b] - a[b]['base_rate']['mean']) for b in labels}
        crit[name] = c
    meta = json.loads((out / 'eval_meta.json').read_text()) if (out / 'eval_meta.json').exists() else {}
    S = {'domain': args.domain, 'labels': labels, 'mean_sd': agg, 'criteria': crit, 'eval_meta': meta}
    (out / 'summary.json').write_text(json.dumps(S, indent=1))
    f = lambda d: f"{d['mean']:.3f}+-{d['sd']:.3f}"  # noqa: E731
    f4 = lambda d: f"{d['mean']:.4f}+-{d['sd']:.4f}"  # noqa: E731
    L = [f'# {args.domain}: level x read-out x label, cross-fitted (mean +- sd over 3 seeds)\n',
         '| label | level / read-out | cf AUROC | cf AP | AP @ AUROC-sel | honest top1% | size-ctrl AUROC | '
         'firing rate (AUROC-sel) | per-seed cf AUROC | per-seed cf AP |', '|' + '---|' * 10]
    for b in labels:
        for name, a in agg.items():
            d = a[b]
            L.append(f"| {b} | {name} | {f(d['cf_auroc'])} | {f4(d['cf_ap'])} | {f4(d['cf_ap_at_auroc'])} | "
                     f"{f(d['cf_top1'])} | {f(d['cf_size_ctrl_auroc'])} | {f(d['cf_auroc_rate'])} | "
                     f"{[round(x, 3) for x in d['cf_auroc']['per_seed']]} | {[round(x, 4) for x in d['cf_ap']['per_seed']]} |")
    L.append('\n| model | FVE | L0 | dead (eval) | dead (train) |\n|---|---|---|---|---|')
    for name, a in agg.items():
        l0 = a.get('l0_per_token', a.get('l0_per_frame'))
        L.append(f"| {name} | {f(a['fve']) if 'fve' in a else ''} | {f(l0) if l0 else ''} | "
                 f"{f(a['dead_frac']) if 'dead_frac' in a else ''} | {f(a['dead_frac_train']) if 'dead_frac_train' in a else ''} |")
    for name, a in agg.items():
        if 'regional' in a:
            r = a['regional']
            L.append(f"\nregional (dot-Voronoi, cf groom_any neuron) {name}: Y2F {f(r['y2f_auroc'])}, B2F {f(r['b2f_auroc'])}, "
                     f"disc {f(r['disc_auroc'])}")
        if 'collective' in a:
            L.append(f"\ncollective (NOT a human label; cf Spearman) {name}: "
                     + ', '.join(f"{k} {f(v)}" for k, v in a['collective'].items()))
    if 'deployed_n90' in meta:
        L.append(f"\ndeployed antsfg n90 dot-Voronoi (all eval frames; not held out): {json.dumps(meta['deployed_n90']['all_eval'])}")
    if 'proxies' in meta:
        L.append(f"\nproxies: {json.dumps(meta['proxies'])}")
    L.append('\n## criteria\n```\n' + json.dumps(crit, indent=1) + '\n```')
    (out / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['selftest', 'train', 'evaluate', 'video', 'summary', 'video_table'])
    p.add_argument('--domain', default='mice', choices=['mice', 'ants'])
    p.add_argument('--out-dir')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--cap', type=int, default=0)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--pool-seed', type=int, default=0)
    p.add_argument('--max-train-frames', type=int, default=0)
    p.add_argument('--max-eval-frames', type=int, default=0)
    p.add_argument('--deployed', action='store_true', help='ants: dot-Voronoi read of the deployed antsfg n90')
    args = p.parse_args()
    if args.cmd == 'video_table':
        return video_table(Path(args.out_dir))
    {'selftest': cmd_selftest, 'train': cmd_train, 'evaluate': cmd_evaluate, 'video': cmd_video,
     'summary': cmd_summary}[args.cmd](args)


if __name__ == '__main__':
    main()
