"""
SAE levers for ECI: can an UNSUPERVISED sparse autoencoder on single foreground patch tokens be pushed to spend
neurons on animal behaviour (mice nose-nose / nose-tail contact, ants grooming) instead of high-variance nuisance
(arena position, body part, setup, odour bag)? Three changes to what the SAE prioritises, each alone against a
baseline, then the ones that help combined.

Data (identical to scripts/eci/multiscale_sae.py / spatial_sae_pilot.py, split seed 0):
    mice  fg448 store; train = 100 videos WITHOUT behaviour annotation; eval = the 172,800 store frames (1 fps) of the
          144 annotated videos. Labels: nose_nose = Y_nn OR Y_np, nose_tail = Y_nt
    ants  antsfg store (v2 + v3); train = 128 random videos; eval = the store frames of the other 128.
          Labels: groom_any, groom_yellow (Y2F), groom_blue (B2F), onlid_yellow (v3 frames only)
Tokens: single raw DINOv2-base foreground patch tokens (768 dims, no position / window features). Training pool: at
most --cap tokens (mice 4M, ants 2M) drawn uniformly without replacement from all train-frame tokens (pool seed 0, so
every config and seed sees the same pool unless the lever changes it). No label is used in training.

BASELINE ('base') = the deployed fg448 / antsfg recipe: src/eci/sae.py Matryoshka BatchTopK, 1024 latents, prefixes
128/256/512/1024, k = 16, Adam lr 5e-4, 500 warmup, cosine to 10%, 6000 steps x 4096 tokens, AuxK on latents dead for
2M tokens, b_dec = geometric median, TokenNorm (mean subtraction + one scalar scale to average norm sqrt(768)).
(The multiscale pilot's S1 SAEs are NOT reused: their input is the 1539-dim window token [token | token | fg fraction,
centroid row, centroid col] with per-block norms, not the raw token.)

Levers (config names; combine with '+', e.g. 'w4096+cell+merged'):
    w4096, w16384  L1 width: 4096 / 16384 latents, Matryoshka prefixes width/8, /4, /2, /1, k = 16. Dead-latent
                   handling = the repo's AuxK (k_aux 512, coefficient 1/32, dead after 2M tokens), on for every width.
    cell           L2 remove what never changes: before TokenNorm, subtract from each token the mean token of the same
                   video at the same grid position over all frames of that video in the store split (train videos:
                   their train frames; eval videos: their eval frames; no labels used). Cells with < 20 foreground
                   tokens fall back to the per-video mean. TokenNorm is then fitted on the centred tokens (average
                   norm sqrt(768), i.e. the baseline's scale).
    video          L2 cheaper variant: subtract the per-video mean token only.
    merged         L3 show it more contact: a frame is 'merged' when its largest 8-connected foreground component on the
                   32 x 32 patch grid exceeds 1.5 x the median single-animal component area (median over components of
                   train frames whose mask splits into exactly one component per animal: mice 4, ants 3). The training
                   pool is drawn from all train tokens with replacement, every token of a merged frame having 5x the
                   sampling weight. TokenNorm and b_dec are fitted on the uniform pool, as for the baseline.
    hard           L3b hard-token mining: the first 2000 steps as the baseline; then every 500 steps the per-token
                   reconstruction error (threshold inference) over the uniform pool is recomputed and batches are
                   drawn with probability proportional to it.

Evaluation (per config x seed, threshold inference; per-frame code of a neuron = its max over the frame's tokens):
    FVE on all eval tokens in the model's input space (for 'cell' / 'video': the CENTRED tokens), L0 per token, dead
    fraction (latents that never fire on the eval tokens).
    Cross-fitted single-neuron scores: eval videos split into two halves (mice: by pool = the 6 videos of one
    cage group; ants: by video), alternating over the sorted ids. A neuron is selected on half A and scored on half B,
    then the reverse; the two test scores are averaged. Selections: best AUROC ('cf_auroc', with the AP of that same
    neuron 'cf_ap_at_auroc'), best AP ('cf_ap'), best top-1% precision among neurons firing on >= 1% of the half's
    frames ('cf_top1' = honest top-1%). Size-controlled AUROC: the AUROC-selected neuron on the test half, within
    foreground-count deciles (deciles over all eval frames; deciles with >= 10 positives and negatives), averaged.
    In-sample best (all eval frames, optional, labelled 'insample_*').
    Ants: the dot-Voronoi read (multiscale_sae.cmd_regional = diag_ants_pairs.py 'mark_vor2' rule: tokens within 2
    grid units of the focal body centroid, closer to the yellow dot than the blue one = yellow region, and the reverse)
    at the cross-fitted best groom_any neuron (selected on frame-max AUROC on half A, read on half B): Y2F AUROC with
    the yellow-region max, B2F AUROC with the blue-region max, Y2F-vs-B2F discrimination AUROC of their difference
    among frames with exactly one of the two. Reference: deployed antsfg neuron 90 (trained on all ants videos, so not
    held out).
    Merged-frame fraction on the eval frames and its overlap with the labels (information only).
    Contact sheets (seed 0): the neuron selected on half A (by AUROC, and by honest top-1%), its top-16 frames of half B
    (<= 2 per video), the argmax patch outlined (green = label positive, red = negative).

Steps (STEP of scripts/eci/sae_levers.sh):
    selftest  align_gpu vs spatial_sae_pilot.align_columns, CellMeans and frame_components vs brute force
    train     stage the train frames to $LOCAL_DIR, build pools / centring means, train configs x seeds -> OUT/sae/
    evaluate  stage the eval frames, encode, score -> OUT/{eval,align,codes_best,sheets}/, OUT/summary.json
    summary   aggregate OUT/eval/*.json (all configs evaluated so far) -> OUT/summary.json + table (login-node safe)
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
import spatial_sae_pilot as ssp  # noqa: E402
from src.eci.levers import GRID, CellMeans, align_gpu, frame_components  # noqa: E402
from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, load_sae, save_checkpoint  # noqa: E402

D = 768
N_TRAIN = ms.N_TRAIN
N_ANIMALS = {'mice': 4, 'ants': 3}
CAP = {'mice': 4_000_000, 'ants': 2_000_000}
LABELS = {'mice': ['nose_nose', 'nose_tail'], 'ants': ['groom_any', 'groom_yellow', 'groom_blue', 'onlid_yellow']}
CEILING_AP = {'nose_nose': 0.275, 'nose_tail': 0.266}  # supervised gated-attention probe (commit 980c52d)
LEVERS = {'base': {}, 'w4096': {'width': 4096}, 'w16384': {'width': 16384}, 'cell': {'center': 'cell'},
          'video': {'center': 'video'}, 'merged': {'sampler': 'merged'}, 'hard': {'sampler': 'hard'}}
MERGE_FACTOR, MERGE_WEIGHT, MIN_CELL = 1.5, 5.0, 20
HARD_START, HARD_EVERY = 2000, 500
MIN_RATE = 0.01
log = lambda s: print(s, flush=True)  # noqa: E731


def parse_cfg(name):
    c = {'width': 1024, 'center': None, 'sampler': 'uniform'}
    seen = set()
    for part in name.split('+'):
        if part not in LEVERS:
            raise ValueError(f'unknown lever {part!r} in {name!r}')
        for k in LEVERS[part]:
            if k in seen:
                raise ValueError(f'{name!r} sets {k} twice')
            seen.add(k)
        c.update(LEVERS[part])
    return c


def frame_chunks(lens, budget):
    """Frame ranges [f0, f1) holding about `budget` tokens each (at least one frame)."""
    st = np.r_[0, np.cumsum(lens)]
    out, f0 = [], 0
    while f0 < len(lens):
        f1 = int(np.searchsorted(st, st[f0] + budget, side='right')) - 1
        f1 = min(max(f1, f0 + 1), len(lens))
        out.append((f0, f1))
        f0 = f1
    return out


def video_index(obs):
    vids = sorted(set(obs))
    vmap = {v: i for i, v in enumerate(vids)}
    return vids, np.array([vmap[v] for v in obs], np.int64)


def merged_meta(domain, pos, lens):
    nc, largest, areas, cfr = frame_components(pos, lens)
    single = nc[cfr] == N_ANIMALS[domain]
    med = float(np.median(areas[single]))
    return nc, largest, {'n_animals': N_ANIMALS[domain], 'frames_with_one_comp_per_animal': int((nc == N_ANIMALS[domain]).sum()),
                         'median_single_animal_area': med, 'p25_p75_single_area': np.percentile(areas[single], [25, 75]).tolist(),
                         'threshold': MERGE_FACTOR * med, 'factor': MERGE_FACTOR,
                         'n_comp_hist': {int(k): int(v) for k, v in zip(*np.unique(np.minimum(nc, 10), return_counts=True))}}


@torch.no_grad()
def token_variance(T, transform, dev, chunk=250_000):
    """Total variance (sum over dims) of transform(T rows) -- T: CPU fp16 tensor / array."""
    s1 = torch.zeros(D, dtype=torch.float64, device=dev)
    s2, n = 0.0, 0
    for a in range(0, len(T), chunk):
        x = transform(a, min(len(T), a + chunk)).double()
        s1 += x.sum(0)
        s2 += float(x.pow(2).sum())
        n += len(x)
    return s2 / n - float((s1 / n).pow(2).sum())


# ---------------------------------------------------------------------------------------------- selftest
def cmd_selftest(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = np.random.default_rng(0)
    n, m, L = 3000, 40, 3
    X = np.where(g.random((n, m)) < 0.3, g.gamma(1.0, 1.0, (n, m)), 0).astype(np.float16)
    X[:, 5] = 0
    X[:, 6] = np.float16(1.5)  # all tied
    X[:, 7] = np.round(X[:, 7] * 2) / 2  # many ties
    X[:, 8] = np.where(g.random(n) < 0.004, 1, 0)  # fires on fewer frames than the top 1%
    Y = g.random((n, L)) < np.array([0.02, 0.2, 0.5])
    Y[:, 0] |= X[:, 1] > 2.5
    ref = ssp.align_columns(X.astype(np.float32), Y)
    Xt, Yt = torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev)
    got = align_gpu(Xt, Yt, block=16)
    for q in ('auc', 'ap'):
        d = np.abs(got[q] - ref[q]).max()
        log(f'align_gpu {q}: max |diff| vs align_columns {d:.2e}')
        assert d < 1e-9, q
    k_top = int(round(0.01 * n))
    clean = []  # columns whose k-th and (k+1)-th largest non-zero values differ (no tie at the top-1% cut-off)
    for j in range(m):
        v = np.sort(X[:, j][X[:, j] > 0].astype(np.float32))[::-1]
        if len(v) <= k_top or v[k_top - 1] != v[k_top]:
            clean.append(j)
    d = np.nanmax(np.abs(got['prec_top'] - ref['prec_top'])[clean])
    tied = sorted(set(range(m)) - set(clean))
    log(f'align_gpu prec_top on {len(clean)} columns without a tie at the cut-off: max |diff| {d:.2e}; '
        f'{len(tied)} tied columns (pro rata vs frame order) max |diff| '
        f'{np.nanmax(np.abs(got["prec_top"] - ref["prec_top"])[tied]):.3f}')
    assert d < 1e-9 and 6 in tied and 8 in clean
    assert np.abs(got['rate'] - ref['rate']).max() < 1e-12
    rows = np.sort(g.choice(n, 1200, replace=False))
    sub = align_gpu(Xt, Yt, rows=torch.from_numpy(rows).to(dev), cols=torch.tensor([1, 3, 7], device=dev))
    r2 = ssp.align_columns(X[rows][:, [1, 3, 7]].astype(np.float32), Y[rows])
    assert np.abs(sub['auc'] - r2['auc']).max() < 1e-9 and np.abs(sub['ap'] - r2['ap']).max() < 1e-9
    log('align_gpu rows / cols subset OK')
    # frame components vs brute force flood fill
    lens = g.integers(0, 120, 200)
    pos = np.concatenate([np.sort(g.choice(GRID * GRID, k, replace=False)) for k in lens]).astype(np.int16)
    nc, largest, areas, cfr = frame_components(pos, lens)
    st = np.r_[0, np.cumsum(lens)]
    for f in range(len(lens)):
        cells = set(pos[st[f]:st[f + 1]].tolist())
        sizes = []
        while cells:
            stack = [cells.pop()]
            s = 0
            while stack:
                p = stack.pop()
                s += 1
                r, c = divmod(p, GRID)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        q = (r + dr) * GRID + c + dc
                        if 0 <= r + dr < GRID and 0 <= c + dc < GRID and q in cells:
                            cells.remove(q)
                            stack.append(q)
            sizes.append(s)
        assert nc[f] == len(sizes) and (largest[f] == (max(sizes) if sizes else 0)), f
        assert sorted(areas[cfr == f].tolist()) == sorted(sizes), f
    log('frame_components OK (8-connected, vs flood fill)')
    # cell means vs numpy
    ntok = 5000
    tok = g.normal(size=(ntok, 6)).astype(np.float32)
    vid = g.integers(0, 3, ntok)
    pp = g.integers(0, 50, ntok)
    cm = CellMeans(3, 6, dev, grid=8)
    cm.add(torch.from_numpy(tok).to(dev), torch.from_numpy(vid).to(dev), torch.from_numpy(pp).to(dev))
    cm.finalize(min_count=20)
    out = cm.center(torch.from_numpy(tok).to(dev), torch.from_numpy(vid).to(dev), torch.from_numpy(pp).to(dev),
                    'cell').cpu().numpy()
    for i in g.choice(ntok, 50):
        same = (vid == vid[i]) & (pp == pp[i])
        mu = tok[same].mean(0) if same.sum() >= 20 else tok[vid == vid[i]].mean(0)
        assert np.abs(out[i] - (tok[i] - mu)).max() < 1e-4
    outv = cm.center(torch.from_numpy(tok).to(dev), torch.from_numpy(vid).to(dev), torch.from_numpy(pp).to(dev),
                     'video').cpu().numpy()
    assert np.abs(outv[0] - (tok[0] - tok[vid == vid[0]].mean(0))).max() < 1e-4
    log(f'CellMeans OK {cm.stats}')
    log('selftest OK')


# ---------------------------------------------------------------------------------------------- train
def gather(tok, sel, chunk=500_000):
    out = np.empty((len(sel), D), np.float16)
    for a in range(0, len(sel), chunk):
        out[a:a + chunk] = tok[sel[a:a + chunk]]
    return out


@torch.no_grad()
def recon_error(sae, X, prep, chunk=65536):
    err = torch.empty(len(X), device=X.device)
    for a in range(0, len(X), chunk):
        x = prep(slice(a, a + chunk))
        err[a:a + chunk] = (sae.decode(sae.encode(x, mode='threshold')) - x).pow(2).sum(1)
    return err


def train_one(name, cfg, seed, pools, cm, steps, out, extra, dev, bs=4096, lr=5e-4, warmup=500, hard_start=HARD_START):
    """pools: {'uniform' / 'merged': (X (N, 768) fp16 GPU, vid (N,) long GPU, pos (N,) long GPU, merged (N,) bool GPU)}."""
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    mode = cfg['center']

    def prep_from(pool):
        X, V, P, _ = pool
        if mode is None:
            return lambda i: X[i]
        return lambda i: cm.center(X[i], V[i], P[i], mode)

    uni = pools['uniform']
    g = np.random.default_rng(seed)
    ii = torch.from_numpy(np.sort(g.choice(len(uni[0]), min(500_000, len(uni[0])), replace=False))).to(dev)
    init = prep_from(uni)(ii).float()
    norm = TokenNorm(D).to(dev).fit(init, seed=seed)
    w = cfg['width']
    sae = MatryoshkaBatchTopKSAE(d_in=D, n_latents=w, prefixes=(w // 8, w // 4, w // 2, w), k=16, seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    pool = pools['merged'] if cfg['sampler'] == 'merged' else uni
    raw = prep_from(pool)
    prep = lambda i: norm(raw(i))  # noqa: E731
    N = len(pool[0])
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    perm, p = torch.randperm(N, generator=gen, device=dev), 0
    history, hard_log, t0 = [], [], time.time()
    hard_q = None
    for step in range(steps):
        if cfg['sampler'] == 'hard' and step >= hard_start:
            if (step - hard_start) % HARD_EVERY == 0:
                sae.eval()
                err = recon_error(sae, pool[0], prep)
                sae.train()
                hard_q = torch.multinomial(err, bs * HARD_EVERY, replacement=True, generator=gen)
                mshare = float(err[pool[3]].sum() / err.sum())
                hard_log.append({'step': step, 'mean_err': float(err.mean()), 'max_over_mean': float(err.max() / err.mean()),
                                 'merged_token_share_of_sampling': mshare, 'merged_token_share_uniform': float(pool[3].float().mean())})
                log(f'    hard mining @ {step}: mean err {float(err.mean()):.1f}, max/mean {float(err.max() / err.mean()):.1f}, '
                    f'merged-frame tokens get {100 * mshare:.1f}% of draws (uniform {100 * float(pool[3].float().mean()):.1f}%)')
            o = (step - hard_start) % HARD_EVERY
            b = hard_q[o * bs:(o + 1) * bs]
        else:
            if p + bs > N:
                perm, p = torch.randperm(N, generator=gen, device=dev), 0
            b = perm[p:p + bs]
            p += bs
        logs, gn = _train_step(sae, opt, prep(b), step, steps, lr, warmup, 1.0)
        if step % 1000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            history.append(logs)
            log(f"    step {step}/{steps} fve {logs['fve']:.4f} l0 {logs['l0']:.1f} dead {logs['n_dead']} "
                f"{logs['elapsed_s']}s")
    sae.eval()
    dead_train = int((sae.tokens_since_fired >= sae.dead_tokens).sum())
    save_checkpoint(out / 'sae.pt', sae, norm, extra={**extra, 'seed': seed})
    (out / 'train.json').write_text(json.dumps({**extra, 'seed': seed, 'n_pool': N, 'steps': steps,
                                                'token_presentations': steps * bs, 'train_time_s': time.time() - t0,
                                                'dead_at_end_train': dead_train, 'dead_frac_train': dead_train / w,
                                                'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', 'history': history,
                                                'hard_mining': hard_log}, indent=1))
    log(f'    {name} s{seed}: trained in {time.time() - t0:.0f}s, final fve {history[-1]["fve"]:.4f}, '
        f'dead (AuxK definition) {dead_train}/{w}')


def cmd_train(args):
    dev = torch.device(os.environ.get('LEVERS_DEVICE', 'cuda'))  # 'cpu' only for tiny dry runs
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out_dir)
    loc = ms.local_dir()
    cfgs = {n: parse_cfg(n) for n in args.configs}
    idx = ssp.StoreIndex(args.domain)
    tr, _, _, _, train_v, eval_v = idx.split(args.domain, N_TRAIN[args.domain], 0)
    tr = np.sort(tr)
    if args.max_train_frames:
        tr = np.sort(np.random.default_rng(0).choice(tr, args.max_train_frames, replace=False))
    vids, fvid = video_index(idx.obs[tr])
    log(f'{args.domain}: {len(train_v)} train videos ({len(vids)} with frames), {len(tr):,} train frames; '
        f'{len(eval_v)} eval videos; configs {list(cfgs)}')
    tok, pos, lens = ms.stage(idx, tr, loc / 'train')
    n = len(pos)
    tvid = np.repeat(fvid, lens)
    t0 = time.time()
    nc, largest, meta = merged_meta(args.domain, pos, lens)
    merged_f = largest > meta['threshold']
    merged_t = np.repeat(merged_f, lens)
    meta.update(merged_frame_frac=float(merged_f.mean()), merged_token_frac=float(merged_t.mean()))
    log(f'  components in {time.time() - t0:.0f}s: median single-animal area {meta["median_single_animal_area"]:.1f} '
        f'patches (from {meta["frames_with_one_comp_per_animal"]:,} frames), threshold {meta["threshold"]:.1f}; merged '
        f'frames {100 * meta["merged_frame_frac"]:.1f}% ({100 * meta["merged_token_frac"]:.1f}% of tokens); '
        f'n_comp hist {meta["n_comp_hist"]}')
    rng = np.random.default_rng(args.pool_seed)
    cap = min(args.cap or CAP[args.domain], n)
    sels = {'uniform': np.sort(rng.choice(n, cap, replace=False))}
    if any(c['sampler'] == 'merged' for c in cfgs.values()):
        w = np.where(merged_t, MERGE_WEIGHT, 1.0)
        sels['merged'] = np.sort(rng.choice(n, cap, replace=True, p=w / w.sum()))
        meta['merged_pool'] = {'merged_token_share': float(merged_t[sels['merged']].mean()),
                               'unique_tokens': int(len(np.unique(sels['merged'])))}
        log(f'  merged pool: {100 * meta["merged_pool"]["merged_token_share"]:.1f}% merged-frame tokens, '
            f'{meta["merged_pool"]["unique_tokens"]:,} unique of {cap:,}')
    cm = None
    if any(c['center'] for c in cfgs.values()):
        t0 = time.time()
        cm = CellMeans(len(vids), D, dev)
        for a in range(0, n, 2_000_000):
            b = min(n, a + 2_000_000)
            cm.add(torch.from_numpy(np.ascontiguousarray(tok[a:b])).to(dev), torch.from_numpy(tvid[a:b]).to(dev),
                   torch.from_numpy(pos[a:b].astype(np.int64)).to(dev))
        cm.finalize(MIN_CELL)
        meta['cell_means'] = cm.stats
        log(f'  centring means in {time.time() - t0:.0f}s: {cm.stats}')
    pools = {}
    for k, s in sels.items():
        t0 = time.time()
        pools[k] = (torch.from_numpy(gather(tok, s)).to(dev), torch.from_numpy(tvid[s]).to(dev),
                    torch.from_numpy(pos[s].astype(np.int64)).to(dev), torch.from_numpy(merged_t[s]).to(dev))
        log(f'  pool {k}: {len(s):,} tokens gathered in {time.time() - t0:.0f}s')
    del tok
    shutil.rmtree(loc / 'train')
    if cm is not None:  # variance removed by the centring, on the uniform pool
        X, V, P, _ = pools['uniform']
        vraw = token_variance(X, lambda a, b: X[a:b].float(), dev)
        meta['variance'] = {'raw_total_var_per_token': vraw}
        for mode in ('cell', 'video'):
            vc = token_variance(X, lambda a, b: cm.center(X[a:b], V[a:b], P[a:b], mode), dev)
            meta['variance'][f'{mode}_total_var_per_token'] = vc
            meta['variance'][f'{mode}_frac_removed'] = 1 - vc / vraw
        log(f'  train-pool variance: {meta["variance"]}')
    meta.update(domain=args.domain, train_videos=train_v, n_train_frames=len(tr), n_train_tokens=n, cap=cap,
                pool_seed=args.pool_seed, steps=args.steps, configs=cfgs)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'train_meta_{"_".join(cfgs)}.json').write_text(json.dumps(meta, indent=1))
    for name, c in cfgs.items():
        log(f'{name}: {c}')
        for seed in args.seeds:
            train_one(name, c, seed, pools, cm, args.steps, out / 'sae' / f'{name}_s{seed}',
                      {'config': name, 'lever': c, 'domain': args.domain, 'cap': cap, 'pool_seed': args.pool_seed,
                       'hard_start': args.hard_start}, dev, hard_start=args.hard_start)
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- evaluate
@torch.no_grad()
def encode_model(sae, norm, mode, cm, tok, pos, tvid, lens, dev):
    """Frame-max codes (F, m) fp16 on the GPU + FVE / L0 / firing stats over all eval tokens."""
    m = sae.n_latents
    F_ = len(lens)
    st = np.r_[0, np.cumsum(lens)]
    codes = torch.zeros(F_, m, dtype=torch.float16, device=dev)
    s1 = torch.zeros(D, dtype=torch.float64, device=dev)
    s2 = sse = l0 = 0.0
    fire = torch.zeros(m, dtype=torch.float64, device=dev)
    ntok = 0
    for f0, f1 in frame_chunks(lens, max(4096, 2 ** 27 // m)):
        lo, hi = st[f0], st[f1]
        if hi == lo:
            continue
        T = torch.from_numpy(tok[lo:hi]).to(dev)
        if mode:
            T = cm.center(T, torch.from_numpy(tvid[lo:hi]).to(dev), torch.from_numpy(pos[lo:hi].astype(np.int64)).to(dev),
                          mode)
        x = norm(T)
        z = sae.encode(x, mode='threshold')
        xh = sae.decode(z)
        s1 += x.double().sum(0)
        s2 += float(x.double().pow(2).sum())
        sse += float((x - xh).double().pow(2).sum())
        act = z > 0
        fire += act.sum(0).double()
        l0 += float(act.sum())
        ntok += len(x)
        fr = torch.repeat_interleave(torch.arange(f1 - f0, device=dev), torch.from_numpy(lens[f0:f1]).to(dev))
        mx = torch.zeros(f1 - f0, m, device=dev)
        mx.index_reduce_(0, fr, z, 'amax', include_self=True)
        codes[f0:f1] = mx.half()
    tss = s2 - float(s1.pow(2).sum()) / ntok
    fr_ = (fire / ntok).cpu().numpy()
    return codes, {'fve': 1 - sse / tss, 'total_var_per_token_normed': tss / ntok, 'l0_per_token': l0 / ntok,
                   'dead_frac': float((fr_ == 0).mean()), 'n_tokens': ntok, 'token_fire_rate': fr_}


@torch.no_grad()
def column_codes(sae, norm, mode, cm, J, tok, pos, tvid, lens, dev, region=None):
    """Frame-max codes of neurons J only (threshold inference is per latent). region: None, or (anchors (F, 6) tensor in
    grid units) -> returns (yellow-region max, blue-region max), each (F, |J|)."""
    J = torch.as_tensor(J, device=dev)
    st = np.r_[0, np.cumsum(lens)]
    F_ = len(lens)
    outs = [torch.zeros(F_, len(J), device=dev) for _ in range(1 if region is None else 2)]
    We, be, th = sae.W_enc[:, J], sae.b_enc[J], sae.threshold
    for f0, f1 in frame_chunks(lens, 1_000_000):
        lo, hi = st[f0], st[f1]
        if hi == lo:
            continue
        T = torch.from_numpy(tok[lo:hi]).to(dev)
        P = torch.from_numpy(pos[lo:hi].astype(np.int64)).to(dev)
        if mode:
            T = cm.center(T, torch.from_numpy(tvid[lo:hi]).to(dev), P, mode)
        pre = torch.relu((norm(T) - sae.b_dec) @ We + be)
        z = pre * (pre > th)
        fr = torch.repeat_interleave(torch.arange(f1 - f0, device=dev), torch.from_numpy(lens[f0:f1]).to(dev))
        if region is None:
            outs[0][f0:f1].index_reduce_(0, fr, z, 'amax', include_self=True)
            continue
        a = region[f0:f1][fr]
        cy, cx = (P // GRID).float() + 0.5, (P % GRID).float() + 0.5
        df = torch.hypot(cx - a[:, 0], cy - a[:, 1])
        dy = torch.hypot(cx - a[:, 2], cy - a[:, 3])
        db = torch.hypot(cx - a[:, 4], cy - a[:, 5])
        for o, msk in ((outs[0], (df <= ms.VOR_D) & (dy < db)), (outs[1], (df <= ms.VOR_D) & (db < dy))):
            if msk.any():
                o[f0:f1].index_reduce_(0, fr[msk], z[msk], 'amax', include_self=True)
    return [o.cpu().numpy() for o in outs]


def score_model(codes, Yt, labels, valid, half, decile, dev):
    """Per label: half-A / half-B / all-frames alignment of every neuron, cross-fitted selections, size control."""
    res, arrays = {}, {}
    groups = {}
    for li, b in enumerate(labels):
        groups.setdefault(valid[b].tobytes(), (valid[b], []))[1].append(li)
    for msk, lis in groups.values():
        R = {}
        for h in (0, 1, 'all'):
            rows = np.flatnonzero(msk & (half == h)) if h != 'all' else np.flatnonzero(msk)
            R[h] = align_gpu(codes, Yt[:, lis], rows=torch.from_numpy(rows).to(dev))
        for k, li in enumerate(lis):
            b = labels[li]
            for h in (0, 1, 'all'):
                for q in ('auc', 'ap', 'prec_top'):
                    arrays[f'{b}|{h}|{q}'] = R[h][q][:, k].astype(np.float32)
                arrays[f'{b}|{h}|rate'] = R[h]['rate'].astype(np.float32)
            dirs = []
            for s, t in ((0, 1), (1, 0)):
                S, T = R[s], R[t]
                e = {'select_half': s, 'test_half': t}
                for sel, q, honest in (('auc', 'auc', False), ('ap', 'ap', False), ('top1', 'prec_top', True)):
                    sc = S[q][:, k].copy()
                    if honest:
                        sc[S['rate'] < MIN_RATE] = -np.inf
                    sc[~np.isfinite(sc)] = -np.inf
                    j = int(np.argmax(sc))
                    e[sel] = {'neuron': j, 'select_score': float(sc[j]), 'test_auc': float(T['auc'][j, k]),
                              'test_ap': float(T['ap'][j, k]), 'test_top1': float(T['prec_top'][j, k]),
                              'test_rate': float(T['rate'][j]), 'select_rate': float(S['rate'][j])}
                # size control: AUROC-selected neuron within fg-count deciles of the test half
                j = e['auc']['neuron']
                rows_t = msk & (half == t)
                sc_d = []
                for d in range(10):
                    rr = np.flatnonzero(rows_t & (decile == d))
                    yy = Yt[torch.from_numpy(rr).to(dev), li] if len(rr) else None
                    if len(rr) == 0 or int(yy.sum()) < 10 or len(rr) - int(yy.sum()) < 10:
                        continue
                    sc_d.append(float(align_gpu(codes, Yt[:, [li]], rows=torch.from_numpy(rr).to(dev),
                                                cols=torch.tensor([j], device=dev))['auc'][0, 0]))
                e['auc']['test_size_ctrl_auc'] = float(np.mean(sc_d)) if sc_d else float('nan')
                e['auc']['size_ctrl_deciles'] = len(sc_d)
                dirs.append(e)
            A = R['all']
            ok = A['rate'] >= MIN_RATE
            top_h = np.where(ok, np.nan_to_num(A['prec_top'][:, k], nan=-1), -1)
            res[b] = {
                'base_rate': float(A['base'][k]), 'n': int(A['n']),
                'n_half': [int(R[0]['n']), int(R[1]['n'])], 'base_half': [float(R[0]['base'][k]), float(R[1]['base'][k])],
                'cf_auroc': float(np.mean([e['auc']['test_auc'] for e in dirs])),
                'cf_ap_at_auroc': float(np.mean([e['auc']['test_ap'] for e in dirs])),
                'cf_top1_at_auroc': float(np.mean([e['auc']['test_top1'] for e in dirs])),
                'cf_size_ctrl_auroc': float(np.nanmean([e['auc']['test_size_ctrl_auc'] for e in dirs])),
                'cf_ap': float(np.mean([e['ap']['test_ap'] for e in dirs])),
                'cf_top1': float(np.mean([e['top1']['test_top1'] for e in dirs])),
                'cf_top1_rate': float(np.mean([e['top1']['test_rate'] for e in dirs])),
                'cf_auroc_rate': float(np.mean([e['auc']['test_rate'] for e in dirs])),
                'insample_auroc': float(np.nanmax(A['auc'][:, k])), 'insample_ap': float(np.nanmax(A['ap'][:, k])),
                'insample_top1_honest': float(top_h.max()),
                'insample_top1_any': float(np.nanmax(A['prec_top'][:, k])),
                'dirs': dirs}
    return res, arrays


def contact_sheet(path, title_rows, frames, lab, label, x, jbox, n_show=16):
    from PIL import Image, ImageDraw
    th = 256
    sheet = Image.new('RGB', (4 * th, 4 * (th + 14)), 'white')
    dr = ImageDraw.Draw(sheet)
    n_pos = 0
    c = 512 // GRID
    for q, (i, p) in enumerate(zip(frames, jbox)):
        im = Image.open(REPO / 'dataset' / lab.frame_path.iat[i]).convert('RGB')
        if im.size != (512, 512):
            im = im.resize((512, 512))
        d2 = ImageDraw.Draw(im)
        yv = bool(lab[label].iat[i])
        n_pos += yv
        r_, c_ = divmod(int(p), GRID)
        d2.rectangle([c_ * c - 2, r_ * c - 2, (c_ + 1) * c + 1, (r_ + 1) * c + 1], outline=(0, 220, 0) if yv else (230, 0, 0),
                     width=3)
        im = im.resize((th, th))
        rr, cc = divmod(q, 4)
        sheet.paste(im, (cc * th, rr * (th + 14)))
        dr.text((cc * th + 2, rr * (th + 14) + th), f'{lab.obs.iat[i]} f{lab.frame_idx.iat[i]} '
                f'{"POS" if yv else "neg"} a={float(x[i]):.2f}', fill='black')
    sheet.save(path, quality=85)
    return n_pos


@torch.no_grad()
def make_sheets(name, sae, norm, mode, cm, res, lab, labels_sheet, valid, half, tok, pos, tvid, lens, dev, sdir,
                per_video=2, n_show=16):
    st = np.r_[0, np.cumsum(lens)]
    index = []
    for b in labels_sheet:
        e = res[b]['dirs'][0]  # selected on half 0, frames of half 1
        for sel in ('auc', 'top1'):
            j = e[sel]['neuron']
            x = column_codes(sae, norm, mode, cm, [j], tok, pos, tvid, lens, dev)[0][:, 0]
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
            We, be = sae.W_enc[:, [j]], sae.b_enc[[j]]
            for i in pick:
                T = torch.from_numpy(tok[st[i]:st[i + 1]]).to(dev)
                P = torch.from_numpy(pos[st[i]:st[i + 1]].astype(np.int64)).to(dev)
                if mode:
                    T = cm.center(T, torch.from_numpy(tvid[st[i]:st[i + 1]]).to(dev), P, mode)
                pre = ((norm(T) - sae.b_dec) @ We + be)[:, 0]
                boxes.append(int(P[int(pre.argmax())]))
            fn = f'{name}__{b}__{sel}_n{j}.jpg'
            n_pos = contact_sheet(sdir / fn, None, pick, lab, b, x, boxes)
            index.append({'config': name, 'label': b, 'selection': sel, 'neuron': j, 'n_shown': len(pick),
                          'n_label_positive': int(n_pos), 'precision_shown': n_pos / max(len(pick), 1),
                          'test_half_base_rate': res[b]['base_half'][e['test_half']], 'file': fn})
    return index


@torch.no_grad()
def cmd_evaluate(args):
    dev = torch.device(os.environ.get('LEVERS_DEVICE', 'cuda'))  # 'cpu' only for tiny dry runs
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out_dir)
    loc = ms.local_dir()
    lout = loc / 'out'
    for d in ('eval', 'align', 'codes_best', 'sheets'):
        (lout / d).mkdir(parents=True, exist_ok=True)
    cfgs = {n: parse_cfg(n) for n in args.configs}
    idx = ssp.StoreIndex(args.domain)
    _, ev, _, _, _, eval_v = idx.split(args.domain, N_TRAIN[args.domain], 0)
    ev = np.sort(ev)
    if args.max_eval_frames:
        ev = ev[np.linspace(0, len(ev) - 1, args.max_eval_frames).astype(int)]
    lab, _, valid = ms.eval_labels(args.domain, idx, ev)
    labels = LABELS[args.domain]
    vids, fvid = video_index(lab.obs.values)
    cover = {v: (int((idx.obs == v).sum()), int((lab.obs == v).sum())) for v in vids}
    n_short = sum(a != b for a, b in cover.values())
    log(f'{args.domain}: {len(ev):,} eval frames, {len(vids)} videos ({n_short} videos with store frames outside the eval '
        f'set); labels ' + json.dumps({b: [int(valid[b].sum()), round(float(lab[b][valid[b]].mean()), 4)] for b in labels}))
    if args.domain == 'mice':
        unit = lab.obs.str.rsplit('_', n=2).str[0].values  # pool = the 6 videos of one cage group
    else:
        unit = lab.obs.values
    units = sorted(set(unit))
    half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
    log(f'  cross-fit halves: {len(units)} units, frames per half {np.bincount(half).tolist()}, base rates per half '
        + json.dumps({b: [round(float(lab[b][valid[b] & (half == h)].mean()), 4) for h in (0, 1)] for b in labels}))
    tok_m, pos, lens = ms.stage(idx, ev, loc / 'eval')
    t0 = time.time()
    tok = np.load(loc / 'eval' / 'tok.npy')
    shutil.rmtree(loc / 'eval')
    log(f'  eval tokens loaded into RAM ({tok.nbytes / 1e9:.1f} GB) in {time.time() - t0:.0f}s')
    del tok_m
    tvid = np.repeat(fvid, lens)
    rank = np.argsort(np.argsort(lens + np.random.default_rng(0).random(len(lens)) * 1e-3))
    decile = np.minimum(rank * 10 // len(lens), 9)
    meta = {'domain': args.domain, 'n_eval_frames': len(ev), 'n_eval_videos': len(vids), 'units': len(units),
            'n_videos_with_store_frames_outside_eval': n_short}
    # merged frames (threshold measured on the train frames)
    tm = sorted(out.glob('train_meta_*.json'))
    if tm:
        thr = json.loads(tm[0].read_text())['threshold']
        _, largest, _, _ = frame_components(pos, lens)
        mg = largest > thr
        meta['merged'] = {'threshold': thr, 'eval_merged_frame_frac': float(mg.mean())}
        for b in labels:
            v = valid[b]
            y = lab[b].values.astype(bool)
            meta['merged'][b] = {'p_label_given_merged': float(y[v & mg].mean()), 'p_label_given_not_merged':
                                 float(y[v & ~mg].mean()), 'p_merged_given_label': float(mg[v & y].mean())}
        log(f'  merged frames (threshold {thr:.1f} patches): {json.dumps(meta["merged"])}')
    cm = None
    if any(c['center'] for c in cfgs.values()):
        cm = CellMeans(len(vids), D, dev)
        for a in range(0, len(tok), 2_000_000):
            b = min(len(tok), a + 2_000_000)
            cm.add(torch.from_numpy(tok[a:b]).to(dev), torch.from_numpy(tvid[a:b]).to(dev),
                   torch.from_numpy(pos[a:b].astype(np.int64)).to(dev))
        cm.finalize(MIN_CELL)
        Tt = torch.from_numpy(tok)
        Vt, Pt = torch.from_numpy(tvid), torch.from_numpy(pos.astype(np.int64))
        vraw = token_variance(Tt, lambda a, b: Tt[a:b].to(dev).float(), dev)
        meta['variance'] = {'raw_total_var_per_token': vraw, 'cell_means': cm.stats}
        for mode in ('cell', 'video'):
            vc = token_variance(Tt, lambda a, b: cm.center(Tt[a:b].to(dev), Vt[a:b].to(dev), Pt[a:b].to(dev), mode), dev)
            meta['variance'][f'{mode}_frac_removed'] = 1 - vc / vraw
        log(f'  eval-token variance removed by centring: {json.dumps(meta["variance"])}')
    (lout / f'eval_meta_{"_".join(cfgs)}.json').write_text(json.dumps(meta, indent=1))
    Yt = torch.from_numpy(lab[labels].values.astype(bool)).to(dev)
    anchors = None
    if args.domain == 'ants':
        A, ok = ms.ants_anchors(lab)
        anchors = (torch.from_numpy(np.nan_to_num(A, nan=-1e4)).float().to(dev), ok)
        ygt, bgt = lab['groom_yellow'].values.astype(bool), lab['groom_blue'].values.astype(bool)
        log(f'  ants anchors: {ok.sum():,}/{len(ok):,} frames with focal centroid + both colour dots')
    models = [(n, s) for n in cfgs for s in args.seeds]
    if args.deployed and args.domain == 'ants':
        models.append(('deployed', 0))
    sheet_index = []
    for name, seed in models:
        t0 = time.time()
        key = name if name == 'deployed' else f'{name}_s{seed}'
        pth = ms.DEPLOYED_ANTS if name == 'deployed' else out / 'sae' / key / 'sae.pt'
        sae, norm, ck = load_sae(pth, dev)
        mode = None if name == 'deployed' else cfgs[name]['center']
        codes, st = encode_model(sae, norm, mode, cm, tok, pos, tvid, lens, dev)
        t_enc = time.time() - t0
        res, arrays = score_model(codes, Yt, labels, valid, half, decile, dev)
        e = {'config': name, 'seed': seed, 'lever': None if name == 'deployed' else cfgs[name], 'width': sae.n_latents,
             'fve': st['fve'], 'fve_space': f'centred tokens ({mode})' if mode else 'raw tokens',
             'l0_per_token': st['l0_per_token'], 'dead_frac': st['dead_frac'], 'n_tokens': st['n_tokens'],
             'labels': res}
        if name != 'deployed':
            tj = json.loads((out / 'sae' / key / 'train.json').read_text())
            e['dead_frac_train'] = tj['dead_frac_train']
            e['train_time_s'] = tj['train_time_s']
        if anchors is not None:
            At, ok = anchors
            one = ygt ^ bgt
            J = sorted({d['auc']['neuron'] for d in res['groom_any']['dirs']} | ({90} if name == 'deployed' else set()))
            ys, bs = column_codes(sae, norm, mode, cm, J, tok, pos, tvid, lens, dev, region=At)
            reg = {}

            def read(j, rows):
                c = J.index(j)
                r1 = rows & ok
                r2 = r1 & one
                return {'y2f_auroc': float(ms.rank_auc(ys[r1][:, [c]], ygt[r1])[0]),
                        'b2f_auroc': float(ms.rank_auc(bs[r1][:, [c]], bgt[r1])[0]),
                        'disc_auroc': float(ms.rank_auc((ys - bs)[r2][:, [c]], ygt[r2])[0]),
                        'n_frames': int(r1.sum()), 'n_exactly_one': int(r2.sum())}
            if name == 'deployed':
                reg['n90_all_eval'] = read(90, np.ones(len(ok), bool))
                fm = column_codes(sae, norm, mode, cm, [90], tok, pos, tvid, lens, dev)[0]
                reg['n90_framemax_groom_any_auroc'] = float(ssp.align_columns(fm, lab[['groom_any']].values.astype(bool))
                                                            ['auc'][0, 0])
                reg['n90_halves'] = [read(90, half == h) for h in (0, 1)]
            dirs = [{'neuron': d['auc']['neuron'], **read(d['auc']['neuron'], half == d['test_half'])}
                    for d in res['groom_any']['dirs']]
            reg['cf_groom_any_neuron'] = {q: float(np.mean([d[q] for d in dirs])) for q in
                                          ('y2f_auroc', 'b2f_auroc', 'disc_auroc')}
            reg['cf_groom_any_neuron']['dirs'] = dirs
            e['regional'] = reg
        (lout / 'eval' / f'{key}.json').write_text(json.dumps(e, indent=1))
        np.savez_compressed(lout / 'align' / f'{key}.npz', token_fire_rate=st['token_fire_rate'].astype(np.float32),
                            **arrays)
        js = sorted({d[s]['neuron'] for b in labels for d in res[b]['dirs'] for s in ('auc', 'ap', 'top1')})
        np.savez_compressed(lout / 'codes_best' / f'{key}.npz', neurons=np.array(js),
                            codes=codes[:, torch.tensor(js, device=dev)].cpu().numpy())
        del codes
        torch.cuda.empty_cache()
        if seed == args.seeds[0] and name != 'deployed':
            sheet_index += make_sheets(name, sae, norm, mode, cm, res, lab,
                                       ['nose_nose', 'nose_tail'] if args.domain == 'mice' else ['groom_any'],
                                       valid, half, tok, pos, tvid, lens, dev, lout / 'sheets')
        f = lambda b: (f"{b}: cf AUROC {res[b]['cf_auroc']:.3f} AP@auc {res[b]['cf_ap_at_auroc']:.4f} cf AP "  # noqa: E731
                       f"{res[b]['cf_ap']:.4f} top1 {res[b]['cf_top1']:.3f} sc {res[b]['cf_size_ctrl_auroc']:.3f}")
        log(f'  {key}: FVE {st["fve"]:.4f} L0 {st["l0_per_token"]:.2f} dead {100 * st["dead_frac"]:.1f}% '
            f'[encode {t_enc:.0f}s, total {time.time() - t0:.0f}s] | ' + ' | '.join(f(b) for b in labels)
            + (f" | regional@cf groom neuron {json.dumps({k: round(v, 3) for k, v in e['regional']['cf_groom_any_neuron'].items() if k != 'dirs'})}"
               if 'regional' in e else ''))
        if name == 'deployed':
            log(f"  deployed n90: {json.dumps({k: v for k, v in e['regional'].items() if k.startswith('n90')})}")
    (lout / 'sheets' / f'index_{"_".join(cfgs)}.json').write_text(json.dumps(sheet_index, indent=1))
    lab.drop(columns=['frame_path']).to_parquet(lout / 'labels.parquet')
    np.save(lout / 'half.npy', half)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(lout, out, dirs_exist_ok=True)
    log(f'results copied to {out}')
    cmd_summary(args)


# ---------------------------------------------------------------------------------------------- summary
METRICS = ('cf_auroc', 'cf_ap', 'cf_ap_at_auroc', 'cf_top1', 'cf_size_ctrl_auroc', 'insample_auroc', 'insample_ap',
           'insample_top1_honest', 'cf_auroc_rate', 'cf_top1_rate', 'base_rate')


def cmd_summary(args):
    out = Path(args.out_dir)
    E = [json.loads(p.read_text()) for p in sorted((out / 'eval').glob('*.json'))]
    labels = LABELS[args.domain]
    agg = {}
    for cfg in dict.fromkeys(e['config'] for e in E):
        es = sorted([e for e in E if e['config'] == cfg], key=lambda e: e['seed'])
        a = {'seeds': [e['seed'] for e in es], 'width': es[0]['width'], 'lever': es[0]['lever'],
             'fve_space': es[0]['fve_space']}
        for q in ('fve', 'dead_frac', 'dead_frac_train', 'l0_per_token'):
            v = [e[q] for e in es if q in e]
            if v:
                a[q] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 'per_seed': v}
        for b in labels:
            a[b] = {}
            for q in METRICS:
                v = [e['labels'][b][q] for e in es]
                a[b][q] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                           'per_seed': v}
        if 'regional' in es[0]:
            a['regional'] = {}
            for q in ('y2f_auroc', 'b2f_auroc', 'disc_auroc'):
                v = [e['regional']['cf_groom_any_neuron'][q] for e in es]
                a['regional'][q] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                    'per_seed': v}
            if cfg == 'deployed':
                a['regional']['n90'] = {k: v for k, v in es[0]['regional'].items() if k.startswith('n90')}
        agg[cfg] = a
    crit = {}
    if 'base' in agg:
        B = agg['base']
        for cfg, a in agg.items():
            if cfg in ('base', 'deployed'):
                continue
            c = {}
            for b in labels:
                d_auc = a[b]['cf_auroc']['mean'] - B[b]['cf_auroc']['mean']
                r_ap = a[b]['cf_ap']['mean'] / B[b]['cf_ap']['mean']
                auc_gain = d_auc >= 0.05 and a[b]['cf_auroc']['mean'] > max(B[b]['cf_auroc']['per_seed'])
                ap_gain = r_ap >= 1.5 and a[b]['cf_ap']['mean'] > max(B[b]['cf_ap']['per_seed'])
                c[b] = {'d_cf_auroc': d_auc, 'ratio_cf_ap': r_ap, 'auroc_gain': bool(auc_gain), 'ap_gain': bool(ap_gain),
                        'gain': bool(auc_gain or ap_gain)}
                if b in CEILING_AP:
                    base_rate = a[b]['base_rate']['mean']
                    c[b]['frac_of_ceiling_ap_above_base'] = (a[b]['cf_ap']['mean'] - base_rate) / (CEILING_AP[b] - base_rate)
                    c[b]['stretch_top1_ge_0.40'] = bool(a[b]['cf_top1']['mean'] >= 0.40)
            if args.domain == 'ants':
                c['ants_no_loss_groom_any'] = bool(a['groom_any']['cf_auroc']['mean'] >= B['groom_any']['cf_auroc']['mean'] - 0.02)
            crit[cfg] = c
        for b in labels:
            if b in CEILING_AP:
                br = B[b]['base_rate']['mean']
                crit.setdefault('base', {})[b] = {'frac_of_ceiling_ap_above_base': (B[b]['cf_ap']['mean'] - br) / (CEILING_AP[b] - br)}
    S = {'domain': args.domain, 'labels': labels, 'mean_sd': agg, 'criteria': crit}
    (out / 'summary.json').write_text(json.dumps(S, indent=1))
    f = lambda d: f"{d['mean']:.3f}+-{d['sd']:.3f}"  # noqa: E731
    f4 = lambda d: f"{d['mean']:.4f}+-{d['sd']:.4f}"  # noqa: E731
    log(f'\n{args.domain}: config x label, cross-fitted (mean +- sd over seeds)')
    log('label | config | cf AUROC | cf AP (AP-sel) | AP@AUROC-sel | honest top1% | size-ctrl AUROC | in-sample AUROC | '
        'FVE | dead (eval) | per-seed cf AUROC | per-seed cf AP | per-seed top1')
    for b in labels:
        for cfg, a in agg.items():
            d = a[b]
            log(f"{b} | {cfg} | {f(d['cf_auroc'])} | {f4(d['cf_ap'])} | {f4(d['cf_ap_at_auroc'])} | {f(d['cf_top1'])} | "
                f"{f(d['cf_size_ctrl_auroc'])} | {f(d['insample_auroc'])} | {f(a['fve'])} | {f(a['dead_frac'])} | "
                f"{[round(x, 3) for x in d['cf_auroc']['per_seed']]} | {[round(x, 4) for x in d['cf_ap']['per_seed']]} | "
                f"{[round(x, 3) for x in d['cf_top1']['per_seed']]}")
    for cfg, a in agg.items():
        if 'regional' in a:
            r = a['regional']
            log(f"regional | {cfg} | Y2F {f(r['y2f_auroc'])} | B2F {f(r['b2f_auroc'])} | disc {f(r['disc_auroc'])}"
                + (f" | n90 {json.dumps(r['n90'])}" if 'n90' in r else ''))
    log('criteria: ' + json.dumps(crit, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['selftest', 'train', 'evaluate', 'summary'])
    p.add_argument('--domain', default='mice', choices=['mice', 'ants'])
    p.add_argument('--out-dir')
    p.add_argument('--configs', nargs='+', default=['base'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--cap', type=int, default=0)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--pool-seed', type=int, default=0)
    p.add_argument('--hard-start', type=int, default=HARD_START, help='L3b: first hard-mining step (smoke tests)')
    p.add_argument('--max-train-frames', type=int, default=0)
    p.add_argument('--max-eval-frames', type=int, default=0)
    p.add_argument('--deployed', action='store_true', help='ants: also score the deployed antsfg SAE (neuron 90)')
    args = p.parse_args()
    {'selftest': cmd_selftest, 'train': cmd_train, 'evaluate': cmd_evaluate, 'summary': cmd_summary}[args.cmd](args)


if __name__ == '__main__':
    main()
