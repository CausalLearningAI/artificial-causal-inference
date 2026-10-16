"""
NN/NT round 2 (2026-10-16). Step 1 = DIAGNOSIS: is the bottleneck the encoder tokens or the SAE / readout?

Round 1 (scripts/eci/nnnt/nnnt.py, results/vision/eci_nnnt/table.md) ended with A3|max (DINOv3-L/16 block 20, 1024 px,
odour-corner rotation, tight pixel mask, no change channel, Matryoshka BatchTopK 512 / k 16, frame max) at nose_nose
R~ 0.12 / cf AUROC 0.73 and nose_tail R~ 0.07 / 0.70 (R~ = cross-fitted normalised R-precision). Not a clear detector.

DIAGNOSIS SET (step 'tokens'): every 10th 5 fps frame of the 144 annotated mice v1 videos (k = fps1_off + 10 j, i.e.
every second r4 'fps1' frame; 86,400 frames), the A3 token stream exactly as nnnt.Pipe makes it (rotated, DINOv3-L
block 20 at 1024 px, tight mask = >= 25 % animal pixels in the 16 x 16 patch). Stored per kept token: the 1024-d token
(fp16), grid position, animal-pixel fraction; per frame: annotations row. -> BEE2/diag/ (about 27 GB).

CEILINGS (step 'probe'; all supervised, labels used ONLY here to measure what the tokens / codes can support; nothing
from here enters any SAE). Cross-fitting = the round-1 pool splits rnd0-rnd4 (12 / 12 pools), fit on one half, score on
the other, both directions, mean. Same metrics as round 1 (R~, AUROC; AP lift).
    T-mean   logistic regression on the frame mean of the kept tokens (the old 'mean-pool' ceiling).
    T-max    ONE linear direction w on tokens, frame score = max_t w . z_t (MIL max): the best possible single 'neuron'
             with the round-1 max readout if a dictionary had allocated it. The key number.
    T-att    gated attention MIL (token MLP 1024 -> 256, softmax attention): information ceiling of the tokens.
    C-lin    logistic regression on the A3 SAE frame codes (max readout, 512): is the label in the dictionary's span?
    C-max    one linear direction over A3 token CODES, max over tokens: a sparse combination of neurons at token level.
    N-best   the best single A3 neuron (cross-fitted R~ selection, nnnt.score_set) on the same frames = round 1.
Reading: T-max >> N-best means the encoder supports a single-direction detector and the SAE fails to allocate it;
T-max ~ N-best (low) means the tokens themselves are the bottleneck (change the input: zoom / resolution / encoder).

ROUND-2 SAE SWEEP (step 'sweep'; PRE-REGISTERED 2026-10-16 ~20:40 CEST, BEFORE any sweep score and before the token
ceilings T-max / T-att were known; only C-lin (codeprobe.json) had been seen: nose_nose R~ 0.20-0.22 / nose_tail
0.13-0.15 for a supervised readout of the whole 512-wide round-1 dictionaries vs 0.10-0.14 / 0.06-0.08 for their best
single neuron). Question: do label-free SAE knobs (more data, width, sparsity, centring, nesting) on the round-1 winner's
tokens (A3: rotated, tight mask, DINOv3-L block 20 at 1024 px, no change channel) raise single-neuron alignment?
    TRAIN POOL (step 'pool'): bursts of 10 consecutive 5 fps frames, 16 per video at random starts (rng 7), ALL 432
        videos, diagnosis-grid frames removed -> ~62k frames, ~9M tight tokens. Videos keep their round-1 roles:
        SAE on 'train' videos, theta* on 'calib' videos (mean L0 = k), FVE on 'eval' videos.
    ARMS (seeds 0-2; every arm k = 16 unless noted; Matryoshka BatchTopK = round-1 SAE class)
        S0  round-1 A3 recipe: Matryoshka 512 (128/256/512), 6000 steps x 4096, trained on the ROUND-1 store
            (r64T train tokens) = the round-1 reference re-scored on the diagnosis frames.
        S1  S0 recipe on the new pool (data effect).
        S2  Matryoshka 2048 (128/512/2048), 30,000 steps, pool.
        S3  Matryoshka 8192 (128/512/2048/8192), 30,000 steps, pool.
        S4  S2 with k = 8.          S5  S2 with k = 32.
        S6  S2 on per-video CENTRED tokens (each video's mean pool token subtracted, train and read-out; the fixed-camera
            prior; label-free).
        S7  S2 without nesting (plain BatchTopK 2048, prefix = 2048 only).
    SCORING: the 86,400 diagnosis frames (every 10th 5 fps frame of the 144 annotated videos, disjoint from the pool),
        frame value = max over kept tokens, round-1 scorer (nnnt.score_set) on all 10 random pool splits: R~ and cf
        AUROC for nose_nose and nose_tail (+ nn_mutual, np_directional).
    RULE: primary per (arm, seed) = mean(R~ nose_nose, R~ nose_tail) (round 1's). Winner = highest mean over seeds
        0-2 among S1-S7. Confirmation: seeds 3-5 of the winner and of S0 trained fresh; KEEP the winner iff the 6 vs 6
        Welch 95% CI of primary R~ (winner - S0) excludes 0 AND the mean gain is >= +0.02; nose_nose and nose_tail are
        always reported separately (a gain carried by one label only is said so). Heat maps (step 'sheets') for the
        winner's and S0's modal nn / nt neurons on held-out frames, with the round-1 locality numbers.
    CROSS-DATASET CHECK: the winning knob is re-run on the CalMS21 swap (sniff_face), CRIM13 (sniff) and MABe22
        (oral_oral, oral_genital) with their own frozen pipelines; reported next to the mice numbers.
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
sys.path.insert(0, str(HERE))
import nnnt as N  # noqa: E402  (Pipe, plan, scorer; read-only use)

R4, REPO, C = N.R4, N.REPO, N.C
OUT2 = Path(os.environ.get('R2_OUT') or N.OUT / 'r2')
BEE2 = Path(os.environ.get('R2_BEE') or N.BEE / 'r2')
log = N.log
SPL_DIAG = ('rnd0', 'rnd1', 'rnd2', 'rnd3', 'rnd4')


def labels_of(rows):
    a = pd.read_csv(REPO / C['ann'], usecols=['observation_id', 'Y_nn', 'Y_np', 'Y_nt'])
    y = a.iloc[rows]
    assert y[['Y_nn', 'Y_np', 'Y_nt']].notna().all().all()
    Y = np.stack([(y.Y_nn.values > 0) | (y.Y_np.values > 0), y.Y_nt.values > 0, y.Y_nn.values > 0,
                  y.Y_np.values > 0], 1)
    unit = pd.Series(y.observation_id.values).str.rsplit('_', n=2).str[0].values
    return Y, unit


# ---------------------------------------------------------------------------------------------- tokens
def cmd_tokens(args):
    import torch
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    P = N.load_plan()
    _, Rr = R4.obs_ranges_of(REPO / C['ann'])
    vids = [(i, v) for i, v in enumerate(P['videos']) if v['annotated']]
    if args.max_videos:
        vids = vids[:args.max_videos]
    fp = pd.read_csv(REPO / C['ann'], usecols=['frame_path'])['frame_path'].values
    jdir = N.local_dir() / 'jpg'
    loc = N.local_dir() / 'diag'
    loc.mkdir(parents=True, exist_ok=True)
    jobs, rels = [], []
    for vi, v in vids:
        ks = np.arange(v['fps1_off'], v['n'], 10)[:args.max_frames or None]
        for c0 in range(0, len(ks), 60):
            kk = ks[c0:c0 + 60]
            rels += list(fp[v['src_lo'] + kk])
            jobs.append(((vi, kk), dict(path=v['path'], s0=v['offset'] + C['step'] * int(kk[0]), n=len(kk),
                                        step=C['step'] * 10, S=N.S, hflip=v['hflip'], fps=C['fps'], threads=2,
                                        jpgs=[jdir / f for f in fp[v['src_lo'] + kk]])))
    st = ThreadPoolExecutor(1).submit(R4.stage_jpgs, rels, jdir)
    pipe = N.Pipe(dev, Rr)
    st.result()
    ftok = open(loc / 'tok.f16', 'wb')
    meta = {'tok_frame': [], 'tok_pos': [], 'tok_fdark': []}
    frows, fobs, fk = [], [], []
    t0 = time.time()
    pf = R4.Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 2))
    for bi_, ((vi, kk), (fr, jp)) in enumerate(pf):
        v = P['videos'][vi]
        for b in range(0, len(kk), 16):
            kb = kk[b:b + 16]
            rows = v['src_lo'] + kb
            with torch.no_grad():
                o = pipe.run(torch.from_numpy(fr[b:b + 16]).to(dev), torch.from_numpy(jp[b:b + 16]).to(dev), rows,
                             v['obs'], 'odor', v['rot'])
                m = o['dark64'] >= N.THR_TIGHT
            for i in range(len(kb)):
                pos = torch.nonzero(m[i]).flatten()
                fid = len(frows)
                ftok.write(o['z'][i, pos].cpu().numpy().tobytes())
                meta['tok_frame'].append(np.full(len(pos), fid, np.int32))
                meta['tok_pos'].append(pos.cpu().numpy().astype(np.int16))
                meta['tok_fdark'].append(o['dark64'][i, pos].cpu().numpy().astype(np.float16))
                frows.append(int(rows[i]))
                fobs.append(v['obs'])
                fk.append(int(kb[i]))
        if bi_ % 200 == 0:
            el = time.time() - t0
            log(f'tokens {bi_}/{len(jobs)} chunks, {len(frows):,} frames, {el:.0f}s ({len(frows) / max(el, 1):.1f} fps, '
                f'GPU busy {pipe.gpu_s:.0f}s, decode thread-s {pf.dec_s:.0f})')
    ftok.close()
    for k_, x in meta.items():
        np.save(loc / f'{k_}.npy', np.concatenate(x))
    pd.DataFrame({'row': frows, 'obs': fobs, 'k': fk}).to_parquet(loc / 'frames.parquet')
    el = time.time() - t0
    info = {'n_frames': len(frows), 'n_tokens': int(sum(len(x) for x in meta['tok_frame'])), 'wall_s': el,
            'fps': len(frows) / el, 'gpu_s': pipe.gpu_s, 'gpu': torch.cuda.get_device_name(0)}
    log(json.dumps(info))
    OUT2.mkdir(parents=True, exist_ok=True)
    if args.selftest:
        (OUT2 / 'tokens_selftest.json').write_text(json.dumps(info, indent=1))
        return
    (OUT2 / 'tokens.json').write_text(json.dumps(info, indent=1))
    R4.atomic_copy(loc, BEE2 / 'diag')
    log(f'-> {BEE2 / "diag"}')


def cmd_pool(args):
    """Label-free SAE TRAINING POOL for round 2: bursts of 10 consecutive 5 fps frames at random starts in ALL 432
    mice v1 videos (rng 7; per video n_bursts = args.bursts_per_video), dropping every frame of the diagnosis grid
    (k = fps1_off mod 10), so pool and scoring frames never coincide. Same A3 token stream (rotated, DINOv3-L block 20,
    1024 px, tight mask). -> BEE2/pool/ (tok.f16, tok_frame, tok_pos, frames.parquet with video / obs / role)."""
    import torch
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    P = N.load_plan()
    _, Rr = R4.obs_ranges_of(REPO / C['ann'])
    fp = pd.read_csv(REPO / C['ann'], usecols=['frame_path'])['frame_path'].values
    g = np.random.default_rng(7)
    jdir = N.local_dir() / 'jpg'
    jobs, rels = [], []
    for vi, v in list(enumerate(P['videos']))[:args.max_videos or None]:
        for a in np.sort(g.choice(v['n'] - 10, args.bursts_per_video, replace=False)):
            kk0 = np.arange(a, a + 10)  # decoded burst
            kk = kk0[(kk0 - v['fps1_off']) % 10 != 0]  # kept: never a diagnosis-grid frame
            rels += list(fp[v['src_lo'] + kk0])
            jobs.append(((vi, kk0, kk), dict(path=v['path'], s0=v['offset'] + C['step'] * int(a), n=10, step=C['step'],
                                             S=N.S, hflip=v['hflip'], fps=C['fps'], threads=2,
                                             jpgs=[jdir / f for f in fp[v['src_lo'] + kk0]])))
    rels = sorted(set(rels))
    st = ThreadPoolExecutor(1).submit(R4.stage_jpgs, rels, jdir)
    pipe = N.Pipe(dev, Rr)
    st.result()
    loc = N.local_dir() / 'pool'
    loc.mkdir(parents=True, exist_ok=True)
    ftok = open(loc / 'tok.f16', 'wb')
    meta = {'tok_frame': [], 'tok_pos': []}
    fr_ = []
    t0 = time.time()
    pf = R4.Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 2))
    for bi_, ((vi, kk0, kk), (fr, jp)) in enumerate(pf):
        v = P['videos'][vi]
        sel = np.isin(kk0, kk)
        with torch.no_grad():
            o = pipe.run(torch.from_numpy(fr[sel]).to(dev), torch.from_numpy(jp[sel]).to(dev), v['src_lo'] + kk,
                         v['obs'], 'odor', v['rot'])
            m = o['dark64'] >= N.THR_TIGHT
        for i in range(len(kk)):
            pos = torch.nonzero(m[i]).flatten()
            ftok.write(o['z'][i, pos].cpu().numpy().tobytes())
            meta['tok_frame'].append(np.full(len(pos), len(fr_), np.int32))
            meta['tok_pos'].append(pos.cpu().numpy().astype(np.int16))
            fr_.append({'video': vi, 'obs': v['obs'], 'k': int(kk[i]), 'role': v['role'], 'annotated': v['annotated']})
        if bi_ % 500 == 0:
            el = time.time() - t0
            log(f'pool {bi_}/{len(jobs)} bursts, {len(fr_):,} frames, {el:.0f}s ({len(fr_) / max(el, 1):.1f} fps, '
                f'GPU busy {pipe.gpu_s:.0f}s, decode thread-s {pf.dec_s:.0f})')
    ftok.close()
    for k_, x in meta.items():
        np.save(loc / f'{k_}.npy', np.concatenate(x))
    pd.DataFrame(fr_).to_parquet(loc / 'frames.parquet')
    el = time.time() - t0
    info = {'n_frames': len(fr_), 'n_tokens': int(sum(len(x) for x in meta['tok_frame'])), 'wall_s': el,
            'fps': len(fr_) / el, 'gpu_s': pipe.gpu_s, 'gpu': torch.cuda.get_device_name(0)}
    log(json.dumps(info))
    OUT2.mkdir(parents=True, exist_ok=True)
    if args.selftest:
        (OUT2 / 'pool_selftest.json').write_text(json.dumps(info, indent=1))
        return
    (OUT2 / 'pool.json').write_text(json.dumps(info, indent=1))
    R4.atomic_copy(loc, BEE2 / 'pool')
    log(f'-> {BEE2 / "pool"}')


def load_diag(root):
    F = pd.read_parquet(root / 'frames.parquet')
    tf = np.load(root / 'tok_frame.npy')
    tok = np.memmap(root / 'tok.f16', np.float16, 'r', shape=(len(tf), N.D))
    return F, tf, tok


# ---------------------------------------------------------------------------------------------- metrics
def metrics(s, y):
    """Single score vector s (n,) float64, labels y (n,) bool -> R~ (ties at the cut pro rata), AUROC, AP lift."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    n, k = len(s), int(y.sum())
    pi = k / n
    o = np.argsort(-s, kind='stable')
    ss, yy = s[o], y[o]
    cut = ss[k - 1]
    above = ss > cut
    at = ss == cut
    tp = yy[above].sum() + yy[at].sum() * (k - above.sum()) / max(1, at.sum())
    return {'R': float((tp / k - pi) / (1 - pi)), 'A': float(roc_auc_score(y, s)),
            'AP_lift': float(average_precision_score(y, s) / pi)}


# ---------------------------------------------------------------------------------------------- probes
class Bags:
    """Tokens of a frame subset on the GPU, grouped by frame (tokens are stored frame-contiguous)."""

    def __init__(self, torch, T, tf, n_frames, dev):
        self.torch = torch
        self.T = T  # (N, d) cuda (normalised, fp16)
        self.tf = torch.from_numpy(tf.astype(np.int64)).to(dev)
        cnt = torch.bincount(self.tf, minlength=n_frames)
        self.cnt, self.start = cnt, torch.cumsum(cnt, 0) - cnt
        self.nf = n_frames

    def gather(self, frames):
        """frames (B,) LongTensor -> token index (M,), bag id in 0..B-1 (M,)."""
        torch = self.torch
        c = self.cnt[frames]
        bag = torch.repeat_interleave(torch.arange(len(frames), device=frames.device), c)
        off = torch.cumsum(c, 0) - c
        idx = self.start[frames][bag] + torch.arange(len(bag), device=frames.device) - off[bag]
        return idx, bag


def seg_max(torch, v, bag, B):
    out = torch.full((B,), -1e4, device=v.device, dtype=v.dtype)
    return out.scatter_reduce(0, bag, v, 'amax', include_self=True)


def fit_probe(torch, kind, bags, Xf, y, tr, dev, steps=1500, bs=2048, lr=1e-3, wd=1e-4, seed=0):
    """kind 'mean' | 'clin' (frame features Xf) | 'max' | 'cmax' (token MIL max on bags) | 'att' (gated attention MIL).
    tr (n_tr,) frame ids, balanced batches (half positives). Returns score(frames) -> (B,) float."""
    nn = torch.nn
    g = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)
    d = Xf.shape[1] if kind in ('mean', 'clin') else bags.T[2] if kind == 'cmax' else bags.T.shape[1]
    if kind == 'att':
        net = nn.ModuleDict({'h': nn.Sequential(nn.Linear(d, 256), nn.GELU()), 'a': nn.Linear(256, 128),
                             'b': nn.Linear(256, 128), 'w': nn.Linear(128, 1), 'o': nn.Linear(256, 1)}).to(dev)
    else:
        net = nn.Linear(d, 1).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    ytr = torch.from_numpy(y[tr.cpu().numpy()]).to(dev)
    pos, neg = tr[ytr], tr[~ytr]

    def fwd(fr):
        if kind in ('mean', 'clin'):
            return net(Xf[fr]).squeeze(1)
        idx, bag = bags.gather(fr)
        if kind == 'cmax':  # sparse token codes: w . c_t = sum_j v_tj w[i_tj]
            CV, CI, _ = bags.T
            v = (CV[idx].float() * net.weight[0][CI[idx]]).sum(1) + net.bias
            return seg_max(torch, v, bag, len(fr))
        x = bags.T[idx].float()
        if kind == 'max':
            return seg_max(torch, net(x).squeeze(1), bag, len(fr))
        h = net['h'](x)
        a = net['w'](torch.tanh(net['a'](h)) * torch.sigmoid(net['b'](h))).squeeze(1)
        am = seg_max(torch, a, bag, len(fr))
        e = torch.exp(a - am[bag])
        den = torch.zeros(len(fr), device=dev).index_add(0, bag, e)
        hp = torch.zeros((len(fr), h.shape[1]), device=dev).index_add(0, bag, e[:, None] * h) / den[:, None]
        return net['o'](hp).squeeze(1)
    for step in range(steps):
        fr = torch.cat([pos[torch.randint(len(pos), (bs // 2,), generator=g, device=dev)],
                        neg[torch.randint(len(neg), (bs // 2,), generator=g, device=dev)]])
        lab = torch.cat([torch.ones(bs // 2, device=dev), torch.zeros(bs // 2, device=dev)])
        loss = nn.functional.binary_cross_entropy_with_logits(fwd(fr), lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def score(frames):
        out = []
        with torch.no_grad():
            for a in range(0, len(frames), 4096):
                out.append(fwd(frames[a:a + 4096]).double().cpu())
        return torch.cat(out).numpy()
    return score


def cmd_probe(args):
    import torch
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    root = Path(args.diag_dir) if args.diag_dir else N.local_dir() / 'diag'
    if not args.diag_dir:
        R4.atomic_copy(BEE2 / 'diag', root)
    F, tf, tok = load_diag(root)
    nf = len(F)
    Y, unit = labels_of(F.row.values)
    sc = json.loads((N.OUT / 'score.json').read_text())
    units, SPL = sc['units'], {k_: np.array(v_) for k_, v_ in sc['splits'].items()}
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    log(f'{nf:,} frames, {len(tf):,} tokens; base rates nn {Y[:, 0].mean():.4f} nt {Y[:, 1].mean():.4f}')
    # tokens on GPU, normalised per dim (label-free statistics of 1M random tokens)
    T = torch.empty((len(tf), N.D), dtype=torch.float16, device=dev)
    for a in range(0, len(tf), 1_000_000):
        T[a:a + 1_000_000] = torch.from_numpy(np.ascontiguousarray(tok[a:a + 1_000_000])).to(dev)
    ii = torch.from_numpy(np.random.default_rng(0).choice(len(tf), min(1_000_000, len(tf)), replace=False)).to(dev)
    mu, sd = T[ii].float().mean(0), T[ii].float().std(0) + 1e-3
    # A3 SAE token codes (seed chosen label-free: highest held-out FVE, as nnnt locality), frame max
    TR = json.loads((N.OUT / 'train.json').read_text())
    a3 = max([f'A3_w512_s{s}' for s in N.SEEDS6], key=lambda n_: TR[n_]['eval_fve'])
    sae, norm, _ = load_sae(N.OUT / 'sae' / a3 / 'sae_theta.pt', dev)
    W = sae.n_latents
    tfg = torch.from_numpy(tf.astype(np.int64)).to(dev)
    cmaxf = torch.zeros((nf, W), dtype=torch.float32, device=dev)
    # token codes kept sparse: the 32 largest per token (threshold-mode L0 is ~k = 16), log1p values
    CV = torch.empty((len(tf), 32), dtype=torch.float16, device=dev)
    CI = torch.empty((len(tf), 32), dtype=torch.int64, device=dev)
    l0 = 0.0
    with torch.no_grad():
        for a in range(0, len(tf), 200_000):
            cz = sae.encode(norm(T[a:a + 200_000].float()), mode='threshold')
            cmaxf.index_reduce_(0, tfg[a:a + 200_000], cz, 'amax', include_self=True)
            l0 += float((cz > 0).sum())
            v_, i_ = cz.topk(32, dim=1)
            CV[a:a + 200_000], CI[a:a + 200_000] = torch.log1p(v_).half(), i_
            del cz
    log(f'A3 token L0 {l0 / len(tf):.2f} (sparse store keeps 32)')
    # check against the stored round-1 codes of the same frames
    Z = {o: np.load(N.BEE / 'videos' / f'{o}.npz')[f'{a3}|max'] for o in F.obs.unique()}
    st = np.stack([Z[o][k] for o, k in zip(F.obs.values, F.k.values)]).astype(np.float32)
    rc = cmaxf.cpu().numpy()
    chk = {'sae': a3, 'max_abs_diff': float(np.abs(rc - st).max()), 'mean_abs_diff': float(np.abs(rc - st).mean()),
           'fire_agree': float(((rc > 0) == (st > 0)).mean())}
    log(f'stored vs recomputed A3 codes: {chk}')
    # N-best: round-1 scorer on these frames
    half_of = {s: SPL[s][uidx] for s in SPL_DIAG}
    Yg = torch.from_numpy(Y).to(dev)
    nb, _ = N.score_set(torch, cmaxf.half(), Yg, half_of, dev)
    res = {'n_frames': nf, 'n_tokens': int(len(tf)), 'base': {'nose_nose': float(Y[:, 0].mean()),
                                                                 'nose_tail': float(Y[:, 1].mean())},
           'code_check': chk, 'N-best': {lb: nb['mean'][lb] for lb in ('nose_nose', 'nose_tail')}, 'probes': {}}
    log(f'N-best {a3}: ' + json.dumps({lb: {q: round(nb['mean'][lb][q], 4) for q in ('R', 'A')}
                                       for lb in ('nose_nose', 'nose_tail')}))
    # token features
    Tn = T
    for a in range(0, len(tf), 1_000_000):
        Tn[a:a + 1_000_000] = ((T[a:a + 1_000_000].float() - mu) / sd).half()
    bags = Bags(torch, Tn, tf, nf, dev)
    meanf = torch.zeros((nf, N.D), device=dev).index_add_(0, tfg, Tn.float()) / bags.cnt.clamp_min(1)[:, None].float()
    cl = torch.log1p(cmaxf)
    cl = (cl - cl.mean(0)) / (cl.std(0) + 1e-3)
    cbag = Bags(torch, (CV, CI, W), tf, nf, dev)
    kinds = args.kinds.split(',')
    for li, lb in enumerate(('nose_nose', 'nose_tail')):
        y = Y[:, li]
        for kind in kinds:
            recs = []
            for s in SPL_DIAG[:args.n_splits]:
                h = SPL[s][uidx]
                for a_, b_ in ((0, 1), (1, 0)):
                    tr = torch.from_numpy(np.flatnonzero(h == a_)).to(dev)
                    te = np.flatnonzero(h == b_)
                    if y[te].sum() == 0 or y[tr.cpu().numpy()].sum() == 0:
                        continue
                    bg, Xf = {'max': (bags, None), 'att': (bags, None), 'cmax': (cbag, None),
                              'mean': (None, meanf), 'clin': (None, cl)}[kind]
                    f = fit_probe(torch, kind, bg, Xf, y, tr, dev, steps=args.steps or 1500)
                    recs.append(metrics(f(torch.from_numpy(te).to(dev)), y[te]))
            m = {q: float(np.mean([r[q] for r in recs])) for q in ('R', 'A', 'AP_lift')}
            m['R_sd'] = float(np.std([r['R'] for r in recs]))
            res['probes'].setdefault(kind, {})[lb] = m
            log(f'{lb} {kind}: R~ {m["R"]:.4f} (sd {m["R_sd"]:.4f}) AUROC {m["A"]:.4f} AP lift {m["AP_lift"]:.2f}')
            OUT2.mkdir(parents=True, exist_ok=True)
            (OUT2 / ('diag_selftest.json' if args.selftest else 'diag.json')).write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------------- code probe (CPU)
def cmd_codeprobe(args):
    """C-lin on ALL 864,000 annotated frames: logistic regression on the stored round-1 frame codes (max readout,
    log1p, standardised) of the listed SAEs, cross-fitted on pool splits rnd0-rnd2, vs that SAE's best single neuron
    (round-1 score.json, same splits). CPU only."""
    from sklearn.linear_model import LogisticRegression
    P = N.load_plan()
    vids = [v for v in P['videos'] if v['annotated']]
    rows = np.concatenate([np.arange(v['src_lo'], v['src_lo'] + v['n']) for v in vids])
    Y, unit = labels_of(rows)
    sc = json.loads((N.OUT / 'score.json').read_text())
    units, SPL = sc['units'], {k_: np.array(v_) for k_, v_ in sc['splits'].items()}
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    res = json.loads((OUT2 / 'codeprobe.json').read_text()) if (OUT2 / 'codeprobe.json').exists() else {}
    for name in args.saes.split(','):
        X = np.concatenate([np.load(N.BEE / 'videos' / f'{v["obs"]}.npz')[f'{name}|max'] for v in vids])
        X = np.log1p(X.astype(np.float32))
        X = (X - X.mean(0)) / (X.std(0) + 1e-3)
        out = {}
        for li, lb in enumerate(('nose_nose', 'nose_tail')):
            recs = []
            for s in ('rnd0', 'rnd1', 'rnd2'):
                h = SPL[s][uidx]
                for a_, b_ in ((0, 1), (1, 0)):
                    tr, te = np.flatnonzero(h == a_), np.flatnonzero(h == b_)
                    m = LogisticRegression(C=0.1, max_iter=300, class_weight='balanced').fit(X[tr], Y[tr, li])
                    recs.append(metrics(m.decision_function(X[te]).astype(np.float64), Y[te, li]))
            out[lb] = {q: float(np.mean([r[q] for r in recs])) for q in ('R', 'A', 'AP_lift')}
            nb = sc['sets'][f'{name}|max|all']['mean'][lb]
            out[lb]['N-best'] = {'R': nb['R'], 'A': nb['A'], 'AP_lift_at_R': nb['AP_lift_at_R']}
            log(f'{name} {lb}: C-lin R~ {out[lb]["R"]:.4f} AUROC {out[lb]["A"]:.4f} AP lift {out[lb]["AP_lift"]:.2f} | '
                f'N-best R~ {nb["R"]:.4f} AUROC {nb["A"]:.4f}')
        res[name] = out
        OUT2.mkdir(parents=True, exist_ok=True)
        (OUT2 / 'codeprobe.json').write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------------- sweep
SWEEP = {  # name: (store, width, prefixes, k, steps, centre)
    'S0': ('r1', 512, (128, 256, 512), 16, 6000, False),
    'S1': ('pool', 512, (128, 256, 512), 16, 6000, False),
    'S2': ('pool', 2048, (128, 512, 2048), 16, 30000, False),
    'S3': ('pool', 8192, (128, 512, 2048, 8192), 16, 30000, False),
    'S4': ('pool', 2048, (128, 512, 2048), 8, 30000, False),
    'S5': ('pool', 2048, (128, 512, 2048), 32, 30000, False),
    'S6': ('pool', 2048, (128, 512, 2048), 16, 30000, True),
    'S7': ('pool', 2048, (2048,), 16, 30000, False),
}


def train_k(cfg, seed, X, steps, out, dev, width, prefixes, k, n_init=500_000, bs=4096, lr=5e-4, warmup=500):
    """nnnt.train_one with k as an argument."""
    import torch
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    Nn, din = X.shape
    ii = torch.from_numpy(np.sort(g.choice(Nn, min(n_init, Nn), replace=False))).to(dev)
    init = X[ii].float()
    norm = TokenNorm(din).to(dev).fit_blocks(init, [N.D] * (din // N.D), seed=seed)
    sae = MatryoshkaBatchTopKSAE(d_in=din, n_latents=width, prefixes=prefixes, k=k, seed=seed).to(dev)
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
        if step % 5000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            history.append(logs)
    sae.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    extra = {'config': 'nnnt_round2', 'cfg': cfg, 'seed': seed}
    save_checkpoint(out / 'sae.pt', sae, norm, extra=extra)
    return sae, norm, {'train_time_s': time.time() - t0, 'n_pool_tokens': Nn,
                       'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()), 'history': history}


def video_means(torch, tok, tf, F, dev):
    """Per-video mean token of the pool (label-free) -> (obs list, (n_videos, D) float32 cuda)."""
    vid = F.obs.values[tf]
    obs = sorted(np.unique(F.obs.values))
    M = torch.zeros((len(obs), N.D), dtype=torch.float32, device=dev)
    for i, o in enumerate(obs):
        ii = np.flatnonzero(vid == o)
        M[i] = torch.from_numpy(np.asarray(tok[ii], dtype=np.float32).mean(0)).to(dev)
    return obs, M


def cmd_sweep(args):
    import torch
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    dev = torch.device('cuda')
    loc = N.local_dir()
    seeds = [int(x) for x in args.seeds.split(',')]
    arms = args.arms.split(',') if args.arms else list(SWEEP)
    sroot = OUT2 / ('sae_selftest' if args.selftest else 'sae')
    res_f = OUT2 / ('sweep_selftest.json' if args.selftest else 'sweep.json')
    res = json.loads(res_f.read_text()) if res_f.exists() else {'train': {}, 'sets': {}}
    t0 = time.time()
    if not args.pool_dir:
        R4.atomic_copy(BEE2 / 'pool', loc / 'pool')
    pdir = Path(args.pool_dir) if args.pool_dir else loc / 'pool'
    PF = pd.read_parquet(pdir / 'frames.parquet')
    ptf = np.load(pdir / 'tok_frame.npy')
    ptok = np.memmap(pdir / 'tok.f16', np.float16, 'r', shape=(len(ptf), N.D))
    prole = PF.role.values[ptf]
    log(f'pool {len(PF):,} frames {len(ptf):,} tokens staged in {time.time() - t0:.0f}s; roles ' +
        json.dumps(pd.Series(prole).value_counts().to_dict()))
    vobs, VM = video_means(torch, ptok, ptf, PF, dev) if any(SWEEP[a][5] for a in arms) else ([], None)
    vix = {o: i for i, o in enumerate(vobs)}
    pvid = torch.from_numpy(pd.Series(PF.obs.values).map(vix).fillna(-1).values.astype(np.int64)[ptf]).to(dev)

    def X_of(store, role, centre):
        if store == 'r1':
            tok, _, rl, sel = N.load_store('r64T', loc / 'stores')
            ii = np.flatnonzero((rl == role) & sel)
            src, vids_ = tok, None
        else:
            ii = np.flatnonzero(prole == role)
            src, vids_ = ptok, pvid
        X = torch.empty((len(ii), N.D), dtype=torch.float16, device=dev)
        for a in range(0, len(ii), 500_000):
            s_ = ii[a:a + 500_000]
            x = torch.from_numpy(np.ascontiguousarray(src[s_])).to(dev)
            if centre:
                x = (x.float() - VM[vids_[torch.from_numpy(s_).to(dev)]]).half()
            X[a:a + len(s_)] = x
        return X
    if any(SWEEP[a][0] == 'r1' for a in arms):
        R4.atomic_copy(N.BEE / 'prepass', loc / 'stores' / 'nnnt')
    # ---- train (grouped by input stream)
    saes = {}
    for key in sorted({(SWEEP[a][0], SWEEP[a][5]) for a in arms}):
        todo = [(a, s_) for a in arms for s_ in seeds if (SWEEP[a][0], SWEEP[a][5]) == key]
        Xtr = X_of(key[0], 'train', key[1])
        log(f'stream {key}: {len(Xtr):,} train tokens')
        for a, s_ in todo:
            store, W, pre, k, steps, centre = SWEEP[a]
            name = f'{a}_s{s_}'
            od = sroot / name
            if (od / 'sae_theta.pt').exists() and not args.force:
                continue
            cfg = {'arm': a, 'store': store, 'width': W, 'prefixes': list(pre), 'k': k, 'steps': steps,
                   'centre': centre, 'tokens': 'A3 (r64T, d0)'}
            st = args.steps or steps
            _, _, ti = train_k(cfg, s_, Xtr, st, od, dev, W, pre, k)
            res['train'][name] = {'cfg': cfg, **{q: ti[q] for q in ('train_time_s', 'n_pool_tokens',
                                                                     'dead_at_end_train')}}
            log(f'{name}: trained {ti["train_time_s"]:.0f}s ({st} steps)')
        del Xtr
        torch.cuda.empty_cache()
        Xca, Xev = X_of(key[0], 'calib', key[1]), X_of(key[0], 'eval', key[1])
        from src.eci.sae import load_sae
        for a, s_ in todo:
            name = f'{a}_s{s_}'
            od = sroot / name
            if (od / 'sae_theta.pt').exists() and not args.force:
                continue
            sae, norm, ck = load_sae(od / 'sae.pt', dev)
            th, _ = calibrate_threshold(sae, Xca, norm=norm)
            set_threshold(sae, th)
            stt = sparsity_stats(sae, Xev, th, norm=norm)
            ck['state_dict']['threshold'] = torch.tensor(float(th))
            ck['theta_star'] = float(th)
            torch.save(ck, od / 'sae_theta.pt')
            res['train'].setdefault(name, {}).update({'theta_star': th, 'eval_fve': stt['fve'], 'eval_l0': stt['l0']})
            log(f'{name}: theta* {th:.4f} eval FVE {stt["fve"]:.4f} L0 {stt["l0"]:.2f}')
        del Xca, Xev
        torch.cuda.empty_cache()
        res_f.write_text(json.dumps(res, indent=1, default=float))
    del ptok
    # ---- read-out on the diagnosis frames and score
    from src.eci.sae import load_sae
    droot = Path(args.diag_dir) if args.diag_dir else loc / 'diag'
    if not args.diag_dir:
        R4.atomic_copy(BEE2 / 'diag', droot)
    F, tf, tok = load_diag(droot)
    nf = len(F)
    Y, unit = labels_of(F.row.values)
    sc = json.loads((N.OUT / 'score.json').read_text())
    units, SPL = sc['units'], {k_: np.array(v_) for k_, v_ in sc['splits'].items()}
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    half_of = {s_: h[uidx] for s_, h in SPL.items()}
    T = torch.empty((len(tf), N.D), dtype=torch.float16, device=dev)
    for a in range(0, len(tf), 1_000_000):
        T[a:a + 1_000_000] = torch.from_numpy(np.ascontiguousarray(tok[a:a + 1_000_000])).to(dev)
    tfg = torch.from_numpy(tf.astype(np.int64)).to(dev)
    dvid = torch.from_numpy(pd.Series(F.obs.values).map(vix).fillna(-1).values.astype(np.int64)[tf]).to(dev) \
        if vix else None
    Yg = torch.from_numpy(Y).to(dev)
    for a in arms:
        for s_ in seeds:
            name = f'{a}_s{s_}'
            if name in res['sets'] and not args.force:
                continue
            sae, norm, _ = load_sae(sroot / name / 'sae_theta.pt', dev)
            X = torch.zeros((nf, sae.n_latents), dtype=torch.float32, device=dev)
            with torch.no_grad():
                for c0 in range(0, len(tf), 200_000):
                    x = T[c0:c0 + 200_000].float()
                    if SWEEP[a][5]:
                        assert int(dvid[c0:c0 + 200_000].min()) >= 0
                        x = x - VM[dvid[c0:c0 + 200_000]]
                    X.index_reduce_(0, tfg[c0:c0 + 200_000], sae.encode(norm(x), mode='threshold'), 'amax',
                                    include_self=True)
            t1 = time.time()
            out, A = N.score_set(torch, X.half(), Yg, half_of, dev)
            out['width'] = int(sae.n_latents)
            out['frame_fire_rate_mean'] = float((X > 0).float().mean())
            res['sets'][name] = out
            m = out['mean']
            log(f'{name}: nn R~ {m["nose_nose"]["R"]:.4f} AUROC {m["nose_nose"]["A"]:.4f} | nt R~ '
                f'{m["nose_tail"]["R"]:.4f} AUROC {m["nose_tail"]["A"]:.4f} | primary {out["primary"]:.4f} '
                f'({time.time() - t1:.0f}s)')
            np.save(sroot / name / 'diag_codes_max.npy', X.half().cpu().numpy()) if args.save_codes else None
            del X
            torch.cuda.empty_cache()
            res_f.write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- sheets
def cmd_sheets(args):
    """Heat maps of sweep SAEs (args.names): for nose_nose and nose_tail, the neuron selected by R~ on half 0 of split
    rnd0 (sweep.json), its 24 highest half-1 (held-out) diagnosis frames, at most one per video, per-token activation
    from the stored diagnosis tokens (no re-encode) over the decoded rotated 1024 px frame. Locality numbers as round 1:
    activation mass on animal pixels vs the kept-token baseline, tokens >= half max, label counts. -> OUT2/sheets/."""
    import torch
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    P = N.load_plan()
    V = {v['obs']: v for v in P['videos']}
    res = json.loads((OUT2 / 'sweep.json').read_text())
    sc = json.loads((N.OUT / 'score.json').read_text())
    half = dict(zip(sc['units'], sc['splits']['rnd0']))
    droot = Path(args.diag_dir) if args.diag_dir else N.local_dir() / 'diag'
    if not args.diag_dir:
        R4.atomic_copy(BEE2 / 'diag', droot)
    F, tf, tok = load_diag(droot)
    pos_all = np.load(droot / 'tok_pos.npy')
    fd_all = np.load(droot / 'tok_fdark.npy').astype(np.float32)
    starts = np.r_[0, np.cumsum(np.bincount(tf, minlength=len(F)))]
    a = pd.read_csv(REPO / C['ann'], usecols=['Y_nn', 'Y_np', 'Y_nt'])
    T = torch.empty((len(tf), N.D), dtype=torch.float16, device=dev)
    for c0 in range(0, len(tf), 1_000_000):
        T[c0:c0 + 1_000_000] = torch.from_numpy(np.ascontiguousarray(tok[c0:c0 + 1_000_000])).to(dev)
    tfg = torch.from_numpy(tf.astype(np.int64)).to(dev)
    VM = None
    hold = np.array([half[o.rsplit('_', 2)[0]] == 1 for o in F.obs.values])
    out = json.loads((OUT2 / 'locality.json').read_text()) if (OUT2 / 'locality.json').exists() else {}
    (OUT2 / 'sheets').mkdir(parents=True, exist_ok=True)
    for name in args.names.split(','):
        arm = name.split('_')[0]
        sae, norm, _ = load_sae(OUT2 / 'sae' / name / 'sae_theta.pt', dev)
        if SWEEP[arm][5] and VM is None:
            pdir = N.local_dir() / 'pool'
            R4.atomic_copy(BEE2 / 'pool', pdir)
            PF = pd.read_parquet(pdir / 'frames.parquet')
            ptf = np.load(pdir / 'tok_frame.npy')
            vobs, VM = video_means(torch, np.memmap(pdir / 'tok.f16', np.float16, 'r', shape=(len(ptf), N.D)), ptf, PF,
                                   dev)
            dvid = torch.from_numpy(pd.Series(F.obs.values).map({o: i for i, o in enumerate(vobs)}).values
                                    .astype(np.int64)[tf]).to(dev)
        for lb in ('nose_nose', 'nose_tail'):
            j = res['sets'][name]['splits']['rnd0'][lb]['neurons'][0]
            act = torch.empty(len(tf), dtype=torch.float32, device=dev)
            with torch.no_grad():
                for c0 in range(0, len(tf), 200_000):
                    x = T[c0:c0 + 200_000].float()
                    if SWEEP[arm][5]:
                        x = x - VM[dvid[c0:c0 + 200_000]]
                    act[c0:c0 + 200_000] = sae.encode(norm(x), mode='threshold')[:, j]
            fmax = torch.zeros(len(F), device=dev).index_reduce_(0, tfg, act, 'amax', include_self=True).cpu().numpy()
            act = act.cpu().numpy()
            cand = pd.DataFrame({'f': np.arange(len(F)), 'v': fmax, 'obs': F.obs.values})[hold]
            cand = cand.sort_values('v', ascending=False).drop_duplicates('obs').head(24)
            tiles, recs = [], []
            for f, val in zip(cand.f.values, cand.v.values):
                o, k, row = F.obs.values[f], int(F.k.values[f]), int(F.row.values[f])
                v = V[o]
                fr = R4.decode(v['path'], v['offset'] + C['step'] * k, 1, C['step'], N.S, v['hflip'], C['fps'],
                               threads=4)[0]
                x = np.rot90(fr, v['rot'], axes=(0, 1)).transpose(2, 0, 1).copy()
                h = np.zeros(N.G64 * N.G64, np.float32)
                fdk = np.zeros(N.G64 * N.G64, np.float32)
                ps = pos_all[starts[f]:starts[f + 1]].astype(np.int64)
                h[ps] = act[starts[f]:starts[f + 1]]
                fdk[ps] = fd_all[starts[f]:starts[f + 1]]
                y = a.iloc[row]
                tot = h.sum()
                hi = h >= 0.5 * h.max() if h.max() > 0 else np.zeros_like(h, bool)
                r = {'obs': o, 'k': k, 'stored': float(val), 'n_half_max': int(hi.sum()), 'n_active': int((h > 0).sum()),
                     'label': ''.join(c for c, col in (('nn', 'Y_nn'), ('np', 'Y_np'), ('nt', 'Y_nt')) if y[col] > 0)
                     or '-', 'mass_animal_px': float((h * fdk).sum() / tot) if tot > 0 else np.nan,
                     'baseline_kept_animal_px': float(fdk[ps].mean()) if len(ps) else np.nan}
                recs.append(r)
                tiles.append((x, h.reshape(N.G64, N.G64), r))
            R = pd.DataFrame(recs)
            summ = {c: float(R[c].mean()) for c in ('mass_animal_px', 'baseline_kept_animal_px', 'n_half_max',
                                                     'n_active')}
            summ['single_token_frames'] = int((R.n_half_max == 1).sum())
            summ['label_counts'] = R.label.value_counts().to_dict()
            fn = OUT2 / 'sheets' / f'{name}_{lb}_n{j}.jpg'
            N.contact_sheet(tiles, fn, f'round 2 {name} neuron {j} ({lb}-selected on pools half 0 of rnd0; top held-'
                                       f'out frames, 1 per video)', N.G64)
            out[f'{name}|{lb}'] = {'neuron': j, 'summary': summ, 'frames': recs, 'sheet': str(fn)}
            log(f'{name} {lb} n{j}: ' + json.dumps(summ))
            (OUT2 / 'locality.json').write_text(json.dumps(out, indent=1, default=float))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('tokens', 'probe', 'codeprobe', 'pool', 'sweep', 'sheets'))
    p.add_argument('--max-videos', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--bursts-per-video', type=int, default=16)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--arms', default='')
    p.add_argument('--pool-dir', default='')
    p.add_argument('--force', action='store_true')
    p.add_argument('--save-codes', action='store_true')
    p.add_argument('--names', default='')
    p.add_argument('--diag-dir', default='')
    p.add_argument('--kinds', default='mean,clin,max,cmax,att')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--steps', type=int, default=0, help='probe steps (default 1500) / sweep steps override')
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--saes', default='A3_w512_s2,A3_w512_s0,A2_w1024_s0,A2_w512_s0,A1_w512_s0')
    args = p.parse_args()
    {'tokens': cmd_tokens, 'probe': cmd_probe, 'codeprobe': cmd_codeprobe, 'pool': cmd_pool, 'sweep': cmd_sweep, 'sheets': cmd_sheets}[args.step](args)


if __name__ == '__main__':
    main()
