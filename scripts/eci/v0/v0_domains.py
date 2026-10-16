"""
ECI "v0" (the original ECI paper recipe, arXiv 2510.14073 v2 App. D.2; see scripts/eci/v0/ants_v2_v0.py, which
replicates it exactly on ants v2) on the other domains and contrasts of the do-pilot atlas:

  ants_v3   v3 videos of t = 2 / 6 / 8 (dataset/ants/eci30/experiment.csv), 30 min each.
            Contrasts v3_2_vs_6 and v3_2_vs_8, SEPARATELY (never pooled; 2 vs 8 is fully confounded with the
            recording day: t = 8 only on day C).
  ants_v4   v4 (healthy Exp4 colonies) + v4_infested videos of t = 2 / 6, 30 min each. Contrasts v4_2_vs_6,
            v4_infested_2_vs_6 and infested vs healthy at t = 2, at t = 6 and pooled over t = 2 and 6 (confounded
            with the recording day and colony: the series share only day F).
  mice_v1   all 432 videos (72 pools x 6 stages); habituation on its last 15 min, odor and post on their whole
            15 min (the atlas' current window). Contrasts as in the atlas: B_stage<s> het vs wt within stage
            (Welch), A_<g>_<tr> paired stage transition within genotype (paired t-test over pools: v0's single-pass
            test applied to the per-pool change, the nearest analogue of the paper's two-sample test), C_<tr> het
            vs wt on the per-pool change (Welch on differences).
  frogs_v1  35 juvenile froglets, hour 1. FoxP1_vs_WT, En1_vs_WT (group fully confounded with recording session).

Recipe (identical for every domain, as in ants_v2_v0.py): frames at 2 fps; the full frame resized to 224 x 224
(overcomplete DinoV2 preprocess, bicubic antialias, ImageNet norm) after an in-memory JPEG round trip (PIL default
quality); DINOv2-S/14 x_norm_patchtokens (256 x 384); TopK SAE 384 -> 4608, k = 20, Adam 5e-4, MSE, grad clip 1.0,
10 epochs, 128 frames per batch, no shuffle (video order), all frames, seed 0, trained on single patch tokens;
APPLIED to the mean of the 256 patch tokens of each frame; video code = mean over its frames; single-pass t-test per
code, raw p < 0.05, no covariates, no mask, no matching.

Frame source: data/<subject>/<v>/observations/full/ (5 fps, 512 px: the frames every atlas model uses; the ants v2 run
decoded the 30 fps sources at exactly 2 fps, which these domains do not all have locally). 2 fps = the 5 fps frames
start + floor(2.5 i), i = 0 .. n - 1 (n = 3600 for 30 min, 1800 for 15 min, 7200 for 1 h). Decoded sequentially
with cv2 (grab for skipped frames). Frame count shortfalls fail loud.

Deviations from the ants v2 run, forced by size: patch tokens are kept only on the job's local disk, as float16
(mice: 778k frames x 256 tokens), not on BeeGFS; the token-level FVE is measured on every 5th batch of 128 frames.
The SAE input of the paper (the mean-pooled token, float32) and every code are computed from the float32 encoder
output.

Gate checks per analysis: n per arm, testable codes, raw-p hits vs expected false positives (0.05 x testable codes),
Bonferroni and BH-FDR survivors; per hit (top 25 by p): effect, Cohen d, nuisance structure (share of the unit-level
variance explained by each nuisance field, eta2, raw and within-arm, with permutation p-values); per domain: FVE on
single tokens and on the mean-pooled input, alive codes (tokens / mean-pooled frames / video means), and where labels
exist (ants grooming, first 10 min of the annotated videos; mice nn / np / nt on the 144 annotated videos) the best
behaviour-aligned code (frame-level F1 of code > 0, AUROC) against the always-on F1 baseline.

Outputs: small -> results/vision/<subject>/eci_v0_<domain>/ (results.json, found_neurons.csv, pvalues_<analysis>.csv
in one wide csv, video_codes.npy, videos.csv, sae_train_log.json); big -> /mnt/beegfs/locatgrp/shared/rcadei/eci_v0/
<domain>/ (pooled tokens, top-k frame codes, SAE, frames.csv).

Usage (inside a Slurm job, scripts/eci/v0/v0_domains.sh):
    python scripts/eci/v0/v0_domains.py --domain mice_v1 --video-dir <staged videos> --stage-dir <local scratch> \
        --out-dir <local out>
"""
import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import ttest_1samp, ttest_ind
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ants_v2_v0 import FVE, eta2, f1_rule  # noqa: E402
from nes_original import neural_effect_search_base  # noqa: E402

N_TOKENS, DIM = 256, 384
N_CODES, TOP_K = 12 * DIM, 20
BIG_ROOT = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_v0')
SUBJECT = {'ants_v3': 'ants', 'ants_v4': 'ants', 'mice_v1': 'mice', 'frogs_v1': 'frogs'}
STAGES = {('S', 'H'): 1, ('S', 'O'): 2, ('S', 'P'): 3, ('F', 'H'): 4, ('F', 'O'): 5, ('F', 'P'): 6}
TRANSITIONS = {'1to2': (1, 2), '2to3': (2, 3), '4to5': (4, 5), '5to6': (5, 6)}
N_PERM = 2000
MAX_HITS_CHECKED = 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_line():
    if not torch.cuda.is_available():
        return 'no gpu'
    free, total = torch.cuda.mem_get_info()
    return f"gpu mem used {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB"


# ------------------------------------------------------------------------------------------------
# designs: one row per video, in the order of the frame table
# ------------------------------------------------------------------------------------------------
def load_design(domain):
    if domain in ('ants_v3', 'ants_v4'):
        exps = ('v3',) if domain == 'ants_v3' else ('v4', 'v4_infested')
        e = pd.read_csv(REPO / 'dataset/ants/eci30/experiment.csv', dtype={'observation_id': str})
        e = e[e.experiment.isin(exps)].reset_index(drop=True)
        rows = []
        for ex in exps:
            raw = pd.read_csv(REPO / f'data/ants/{ex}/experiment.csv', dtype={'observation_id': str})
            rows.append(raw[['observation_id', 'observation_file']])
        raw = pd.concat(rows)
        e = e.merge(raw, on='observation_id', how='left', validate='1:1')
        assert e.observation_file.notna().all()
        e['video_rel'] = [f'data/ants/{x}/observations/full/{f}' for x, f in zip(e.experiment, e.observation_file)]
        e['start'], e['n_keep'] = 0, 3600
        e['nestbox'] = e['nestbox'].fillna('none').astype(str)
        nuis = ['batch', 'position', 'recording_date', 'nestbox']
    elif domain == 'mice_v1':
        e = pd.read_csv(REPO / 'data/mice/v1/experiment.csv')
        e = e[e.valid == 1].reset_index(drop=True)
        e['stage'] = [STAGES[(o, p)] for o, p in zip(e.odor, e.phase)]
        e['video_rel'] = [f'data/mice/v1/observations/full/{f}' for f in e.observation_file]
        e['start'] = np.where(e.phase == 'H', 4500, 0)  # habituation: its last 15 min (frames 4500..8999 at 5 fps)
        e['n_keep'] = 1800
        oc = pd.read_csv(REPO / 'dataset/mice/v1/eci/odor_corner.csv')[['observation_id', 'odor_corner']]
        e = e.merge(oc, on='observation_id', how='left', validate='1:1')
        assert e.odor_corner.notna().all()
        nuis = ['odor_corner', 'line', 'sex', 'date']
    elif domain == 'frogs_v1':
        e = pd.read_csv(REPO / 'data/frogs/v1/experiment.csv')
        e = e[e.valid == 1].reset_index(drop=True)
        e['video_rel'] = [f'data/frogs/v1/observations/full/{f}' for f in e.observation_file]
        e['start'], e['n_keep'] = 0, 7200
        nuis = ['session', 'slot', 'date']
    else:
        raise ValueError(domain)
    for c in nuis:
        e[c] = e[c].astype(str)
    e['offset'] = np.concatenate([[0], np.cumsum(e.n_keep.to_numpy())[:-1]])
    return e, nuis


def analyses(domain, d):
    """-> list of dicts: id, kind ('welch' | 'paired' | 'did'), control / treatment masks over videos (welch) or the
    per-pool (a, b) video index pairs (paired / did), plus labels."""
    out = []
    if domain == 'ants_v3':
        for t in (6, 8):
            out.append(dict(id=f'v3_2_vs_{t}', kind='welch', control=(d['T'] == 2).to_numpy(),
                            treatment=(d['T'] == t).to_numpy(), label=f'v3 t={t} vs t=2',
                            confound='recording day: t=8 only on day C, t=2 only on days A/B' if t == 8 else ''))
    elif domain == 'ants_v4':
        for ex in ('v4', 'v4_infested'):
            m = (d.experiment == ex).to_numpy()
            out.append(dict(id=f'{ex}_2_vs_6', kind='welch', control=m & (d['T'] == 2).to_numpy(),
                            treatment=m & (d['T'] == 6).to_numpy(), label=f'{ex} t=6 vs t=2', confound=''))
        inf = (d.experiment == 'v4_infested').to_numpy()
        for aid, ts in (('infested_vs_healthy_t2', (2,)), ('infested_vs_healthy_t6', (6,)),
                        ('infested_vs_healthy_t2_6', (2, 6))):
            m = d['T'].isin(ts).to_numpy()
            out.append(dict(id=aid, kind='welch', control=m & ~inf, treatment=m & inf,
                            label=f'infested vs healthy t={"+".join(map(str, ts))}',
                            confound='recording day and colony (series share only day F)'))
    elif domain == 'mice_v1':
        for s in range(1, 7):
            m = (d.stage == s).to_numpy()
            out.append(dict(id=f'B_stage{s}', kind='welch', control=m & (d.genotype == 'wt').to_numpy(),
                            treatment=m & (d.genotype == 'het').to_numpy(), label=f'het vs wt, stage {s}',
                            confound=''))
        key = {(p, s): i for i, (p, s) in enumerate(zip(d.pool, d.stage))}
        for tr, (a, b) in TRANSITIONS.items():
            pools = sorted(set(d.pool))
            pairs = np.array([(key[(p, a)], key[(p, b)]) for p in pools])
            het = np.array([d.genotype.iloc[key[(p, a)]] == 'het' for p in pools])
            for g, gm in (('het', het), ('wt', ~het)):
                out.append(dict(id=f'A_{g}_{tr}', kind='paired', pairs=pairs[gm], label=f'{g}: stage {a} -> {b}',
                                confound=''))
            out.append(dict(id=f'C_{tr}', kind='did', pairs=pairs, treatment=het,
                            label=f'het vs wt change, stage {a} -> {b}', confound=''))
    elif domain == 'frogs_v1':
        for t in ('FoxP1', 'En1'):
            out.append(dict(id=f'{t}_vs_WT', kind='welch', control=(d.group == 'WT').to_numpy(),
                            treatment=(d.group == t).to_numpy(), label=f'{t} vs WT',
                            confound='recording session: every session holds one group only'))
    return out


def load_labels(domain, d):
    """(N, L) float32 frame labels (NaN = unlabelled) at the kept frames, and their names."""
    if domain.startswith('ants'):
        ann, cols = REPO / 'dataset/ants/eci30/annotations.csv', ['Y_Y2F', 'Y_B2F']
        names = ['groom_yellow', 'groom_blue']
    elif domain == 'mice_v1':
        ann, cols = REPO / 'dataset/mice/v1/annotations.csv', ['Y_nn', 'Y_np', 'Y_nt']
        names = ['nn', 'np', 'nt']
    else:
        return None, []
    a = pd.read_csv(ann, usecols=['observation_id', 'frame_idx'] + cols, dtype={'observation_id': str})
    a = a[a.observation_id.isin(set(d.observation_id))]
    fr = frame_table(d)
    j = fr[['observation_id', 'frame_idx']].merge(a, on=['observation_id', 'frame_idx'], how='left', validate='1:1')
    lab = j[cols].to_numpy(dtype=np.float32)
    anyl = np.where(np.isnan(lab).all(1), np.nan, np.nanmax(np.where(np.isnan(lab), 0, lab), 1))
    lab = np.column_stack([lab, anyl]).astype(np.float32)
    return lab, names + ['any']


def frame_table(d):
    oid, fidx = [], []
    for o, s, n in zip(d.observation_id, d.start, d.n_keep):
        oid.append(np.repeat(o, n))
        fidx.append(s + (5 * np.arange(n)) // 2)
    return pd.DataFrame(dict(observation_id=np.concatenate(oid), frame_idx=np.concatenate(fidx)))


# ------------------------------------------------------------------------------------------------
# stage 1: frames -> tokens (fp16, local disk) + mean-pooled tokens (fp32)
# ------------------------------------------------------------------------------------------------
class VideoChunks(IterableDataset):
    def __init__(self, paths, starts, n_keeps, preprocess, chunk=128):
        self.paths, self.starts, self.n_keeps, self.preprocess, self.chunk = paths, starts, n_keeps, preprocess, chunk

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (0, 1) if info is None else (info.id, info.num_workers)
        cv2.setNumThreads(1)
        for vi in range(wid, len(self.paths), nw):
            keep = self.starts[vi] + (5 * np.arange(self.n_keeps[vi])) // 2
            keep_set, last = set(keep.tolist()), int(keep[-1])
            cap = cv2.VideoCapture(str(self.paths[vi]))
            fc, kept, buf, start = 0, 0, [], 0
            while fc <= last:
                if not cap.grab():
                    break
                if fc in keep_set:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    img = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))  # BGR -> RGB
                    jpg = io.BytesIO()
                    img.save(jpg, format='JPEG')
                    jpg.seek(0)
                    buf.append(self.preprocess(Image.open(jpg).convert('RGB')))
                    kept += 1
                    if len(buf) == self.chunk:
                        yield vi, start, torch.stack(buf)
                        start, buf = start + len(buf), []
                fc += 1
            cap.release()
            if buf:
                yield vi, start, torch.stack(buf)
            if kept < len(keep):
                yield vi, -kept, None  # shortfall marker


def extract(d, video_dir, tok_path, device, num_workers):
    from overcomplete.models import DinoV2
    model = DinoV2(device=device)
    model.to(device)
    model.eval()
    N = int(d.n_keep.sum())
    import shutil
    need, free = N * N_TOKENS * DIM * 2, shutil.disk_usage(Path(tok_path).parent).free
    log(f"token store needs {need / 1e9:.0f} GB on local disk, {free / 1e9:.0f} GB free")
    if free < 1.05 * need:
        raise SystemExit(f'not enough local disk for the token store: {need / 1e9:.0f} GB needed, {free / 1e9:.0f} GB free')
    tokens = np.lib.format.open_memmap(tok_path, mode='w+', dtype=np.float16, shape=(N, N_TOKENS, DIM))
    pooled = np.empty((N, DIM), dtype=np.float32)
    counts = np.zeros(len(d), dtype=np.int64)
    paths = [Path(video_dir) / Path(r).name if video_dir else REPO / r for r in d.video_rel]
    loader = DataLoader(VideoChunks(paths, d.start.to_numpy(), d.n_keep.to_numpy(), model.preprocess),
                        batch_size=None, num_workers=num_workers, prefetch_factor=4)
    off = d.offset.to_numpy()
    t0, done, last, short = time.time(), 0, 0, {}
    with torch.inference_mode():
        for vi, start, x in loader:
            if x is None:
                short[d.observation_id.iloc[vi]] = -start
                continue
            feats = model.forward_features(x.to(device, non_blocking=True))
            assert feats.shape[1:] == (N_TOKENS, DIM), feats.shape
            g = off[vi] + start
            pooled[g:g + len(x)] = feats.float().mean(1).cpu().numpy()
            tokens[g:g + len(x)] = feats.half().cpu().numpy()
            counts[vi] += len(x)
            done += len(x)
            if done - last >= 20000:
                last = done
                log(f"tokens {done}/{N} frames, {done / (time.time() - t0):.0f} frames/s, {gpu_line()}")
    tokens.flush()
    if short or (counts != d.n_keep.to_numpy()).any():
        bad = {o: int(c) for o, c, n in zip(d.observation_id, counts, d.n_keep) if c != n}
        raise SystemExit(f'frame shortfall (kept frames per video): {bad} {short}')
    log(f"tokens done: {done} frames in {time.time() - t0:.0f}s ({done / (time.time() - t0):.0f} frames/s)")
    del tokens
    return pooled


# ------------------------------------------------------------------------------------------------
# stage 2: SAE (ants_v2_v0.train_sae, with float16 storage cast to float32 and a prefetching loader)
# ------------------------------------------------------------------------------------------------
class Batches(torch.utils.data.Dataset):
    def __init__(self, path, batch_frames):
        self.path, self.b = path, batch_frames
        self.n = np.load(path, mmap_mode='r').shape[0]
        self.starts = list(range(0, self.n, batch_frames))
        self.tok = None

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        if self.tok is None:
            self.tok = np.load(self.path, mmap_mode='r')
        s = self.starts[i]
        return torch.from_numpy(np.ascontiguousarray(self.tok[s:s + self.b]))


def train_sae(tok_path, device, epochs, batch_frames, seed):
    from overcomplete.sae import TopKSAE
    from overcomplete.sae.trackers import DeadCodeTracker
    from overcomplete.sae.train import _compute_reconstruction_error
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    sae = TopKSAE(input_shape=DIM, nb_concepts=N_CODES, top_k=TOP_K, device=device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=5e-4)
    ds = Batches(tok_path, batch_frames)
    loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=6, prefetch_factor=4, pin_memory=True,
                        persistent_workers=True)
    hist = []
    for epoch in range(epochs):
        sae.train()
        t0 = time.time()
        ep_loss = ep_r2 = ep_l0 = 0.0
        tracker = None
        for xb in loader:
            x = xb.to(device, non_blocking=True).float().reshape(-1, DIM)
            optimizer.zero_grad(set_to_none=True)
            z_pre, z, x_hat = sae(x)
            loss = (x - x_hat).square().mean()
            if tracker is None:
                tracker = DeadCodeTracker(z.shape[1], device)
            tracker.update(z.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                ep_loss += loss.item()
                ep_r2 += float(_compute_reconstruction_error(x, x_hat))
                ep_l0 += (z > 0).sum(1).float().mean().item()
        nb = len(ds)
        row = dict(epoch=epoch + 1, loss=ep_loss / nb, r2_overcomplete=ep_r2 / nb, l0=ep_l0 / nb,
                   dead_ratio=tracker.get_dead_ratio(), seconds=time.time() - t0,
                   tokens_per_s=ds.n * N_TOKENS / (time.time() - t0))
        hist.append(row)
        log(f"SAE epoch {epoch + 1}/{epochs}: loss {row['loss']:.5f}, R2 {row['r2_overcomplete']:.4f}, "
            f"L0 {row['l0']:.1f}, dead {100 * row['dead_ratio']:.1f}%, {row['seconds']:.0f}s "
            f"({row['tokens_per_s']:.0f} tokens/s), {gpu_line()}")
    sae.eval()
    return sae, hist


# ------------------------------------------------------------------------------------------------
# stage 3: encoding of the mean-pooled tokens (+ token-level quality on every 5th batch)
# ------------------------------------------------------------------------------------------------
def encode(sae, pooled, tok_path, d, device, batch_frames=128):
    N = len(pooled)
    top_idx = np.empty((N, TOP_K), dtype=np.int16)
    top_val = np.empty((N, TOP_K), dtype=np.float32)
    fve_pool, fve_tok = FVE(), FVE()
    alive_tok = torch.zeros(N_CODES, dtype=torch.bool, device=device)
    alive_pool = torch.zeros(N_CODES, dtype=torch.bool, device=device)
    l0_tok, n_tok, l0_pool = 0.0, 0, 0.0
    vid = np.repeat(np.arange(len(d)), d.n_keep.to_numpy())
    vsum = torch.zeros(len(d), N_CODES, dtype=torch.float64, device=device)
    vid_t = torch.from_numpy(vid).to(device)
    tok = np.load(tok_path, mmap_mode='r') if tok_path is not None else None
    t0 = time.time()
    with torch.inference_mode():
        for bi, s in enumerate(range(0, N, batch_frames)):
            p = torch.from_numpy(pooled[s:s + batch_frames]).to(device)
            _, zp, ph = sae(p)
            fve_pool.update(p, ph)
            alive_pool |= (zp > 0).any(0)
            l0_pool += (zp > 0).sum().item()
            v, i = zp.topk(TOP_K, dim=1)
            top_idx[s:s + len(p)] = i.short().cpu().numpy()
            top_val[s:s + len(p)] = v.float().cpu().numpy()
            vsum.index_add_(0, vid_t[s:s + len(p)], zp.double())
            if tok is not None and bi % 5 == 0:
                x = torch.from_numpy(np.ascontiguousarray(tok[s:s + batch_frames])).to(device).float().reshape(-1, DIM)
                _, z, xh = sae(x)
                fve_tok.update(x, xh)
                alive_tok |= (z > 0).any(0)
                l0_tok += (z > 0).sum().item()
                n_tok += len(x)
    vcodes = (vsum / torch.from_numpy(d.n_keep.to_numpy()).to(device)[:, None]).cpu().numpy()
    log(f"encoding done in {time.time() - t0:.0f}s, {gpu_line()}")
    q = dict(fve_meanpooled=fve_pool.value(), l0_meanpooled_frames=l0_pool / N,
             alive_codes_meanpooled_frames=int(alive_pool.sum().item()),
             alive_codes_video_means=int((vcodes > 0).any(0).sum()), n_frames=int(N))
    if n_tok:
        q.update(fve_tokens_every5th_batch=fve_tok.value(), alive_codes_tokens_every5th_batch=int(alive_tok.sum().item()),
                 l0_tokens=l0_tok / n_tok, n_tokens_checked=int(n_tok))
    return top_idx, top_val, vcodes, q


def dense_rows(top_idx, top_val, rows, cols):
    """(len(rows), len(cols)) dense codes of the given frames, restricted to the given code columns."""
    pos = np.full(N_CODES, -1, np.int64)
    pos[cols] = np.arange(len(cols))
    out = np.zeros((len(rows), len(cols)), np.float32)
    ii, vv = top_idx[rows].astype(np.int64), top_val[rows]
    for k in range(TOP_K):
        c = pos[ii[:, k]]
        m = (c >= 0) & (vv[:, k] > 0)
        out[np.where(m)[0], c[m]] = vv[m, k]
    return out


# ------------------------------------------------------------------------------------------------
# stage 4: search + gate checks
# ------------------------------------------------------------------------------------------------
def eta2_perm(y, groups, arm, rng):
    """eta2 of y by groups (raw, unrestricted permutation p) and within arm (y centred per arm, groups permuted within
    arm; arm None -> same as raw). Groups with one level -> None."""
    g = pd.factorize(pd.Series(groups).astype(str))[0]
    if len(np.unique(g)) < 2:
        return None
    G = np.eye(g.max() + 1)[g]
    nk = G.sum(0)

    def e2(v, perms):
        vp = v[perms]                      # (P, n)
        sst = ((vp - vp.mean(1, keepdims=True)) ** 2).sum(1)
        ssb = ((vp @ G) ** 2 / nk).sum(1) - vp.sum(1) ** 2 / len(v)
        return np.where(sst > 0, ssb / np.maximum(sst, 1e-300), np.nan)

    n = len(y)
    ident = np.arange(n)[None]
    perms = np.stack([rng.permutation(n) for _ in range(N_PERM)])
    obs = float(e2(y, ident)[0])
    null = e2(y, perms)
    res = dict(eta2=obs, perm_p=float((1 + np.sum(null >= obs - 1e-12)) / (1 + N_PERM)))
    if arm is not None:
        a = np.asarray(arm).astype(int)
        yc = y.astype(np.float64).copy()
        for k in (0, 1):
            yc[a == k] -= yc[a == k].mean()
        pw = np.tile(np.arange(n), (N_PERM, 1))
        for k in (0, 1):
            idx = np.where(a == k)[0]
            for r in range(N_PERM):
                pw[r, idx] = idx[rng.permutation(len(idx))]
        obs_w = float(e2(yc, ident)[0])
        null_w = e2(yc, pw)
        res.update(eta2_within_arm=obs_w,
                   perm_p_within_arm=float((1 + np.sum(null_w >= obs_w - 1e-12)) / (1 + N_PERM)))
    return res


def run_analysis(an, d, nuis, vcodes, rng):
    names = np.array([f'N_{j + 1}' for j in range(N_CODES)])
    if an['kind'] == 'welch':
        c, t = an['control'], an['treatment']
        X1, X0 = vcodes[t], vcodes[c]
        with np.errstate(invalid='ignore', divide='ignore'):
            tt, p = ttest_ind(X1, X0, axis=0, equal_var=False)
        units = d[c | t].copy()
        arm = t[c | t].astype(int)
        Y = vcodes[c | t]
        n = dict(n_control=int(c.sum()), n_treatment=int(t.sum()))
        eff = X1.mean(0) - X0.mean(0)
        sd = np.sqrt((X1.var(0, ddof=1) * (len(X1) - 1) + X0.var(0, ddof=1) * (len(X0) - 1)) / (len(X1) + len(X0) - 2))
    elif an['kind'] == 'paired':
        a, b = an['pairs'][:, 0], an['pairs'][:, 1]
        D = vcodes[b] - vcodes[a]
        with np.errstate(invalid='ignore', divide='ignore'):
            tt, p = ttest_1samp(D, 0.0, axis=0)
        units, arm, Y = d.iloc[a].copy(), None, D
        n = dict(n_pools=int(len(a)))
        eff, sd = D.mean(0), D.std(0, ddof=1)
    else:  # did
        a, b = an['pairs'][:, 0], an['pairs'][:, 1]
        D = vcodes[b] - vcodes[a]
        tm = an['treatment']
        with np.errstate(invalid='ignore', divide='ignore'):
            tt, p = ttest_ind(D[tm], D[~tm], axis=0, equal_var=False)
        units, arm, Y = d.iloc[a].copy(), tm.astype(int), D
        n = dict(n_control=int((~tm).sum()), n_treatment=int(tm.sum()))
        eff = D[tm].mean(0) - D[~tm].mean(0)
        sd = np.sqrt((D[tm].var(0, ddof=1) * (tm.sum() - 1) + D[~tm].var(0, ddof=1) * ((~tm).sum() - 1)) / (len(D) - 2))
    valid = np.isfinite(p)
    nt = int(valid.sum())
    sig = np.where(valid & (p < 0.05))[0]
    sig = sig[np.argsort(p[sig])]
    fdr = multipletests(p[valid], alpha=0.05, method='fdr_bh')[0].sum() if nt else 0
    res = dict(id=an['id'], kind=an['kind'], label=an['label'], confound=an['confound'], **n,
               n_testable_codes=nt, n_found_raw_p05=int(len(sig)), expected_false_positives=0.05 * nt,
               n_found_bonferroni=int((valid & (p < 0.05 / max(nt, 1))).sum()), n_found_fdr_bh=int(fdr),
               min_p=float(np.nanmin(p)) if nt else None)
    if an['kind'] == 'welch':  # cross-check against the original code path
        df = pd.DataFrame(np.vstack([vcodes[an['control']], vcodes[an['treatment']]]), columns=list(names))
        df.insert(0, 'T', np.r_[np.zeros(an['control'].sum(), int), np.ones(an['treatment'].sum(), int)])
        base = neural_effect_search_base(df, test='AD', method='t-test', alpha=0.05, k=2000, neuron_prefix='N_',
                                         fallback_start_neur_index=4, sn=[])
        assert sorted(int(s) - 1 for s in base['neurons']) == sorted(sig.tolist()), f"{an['id']}: mismatch vs nes.py"
        res['matches_original_nes_base'] = True
    hits = []
    for r, j in enumerate(sig):
        h = dict(analysis=an['id'], neuron=names[j], index0=int(j), p=float(p[j]), t=float(tt[j]), effect=float(eff[j]),
                 cohen_d=float(eff[j] / sd[j]) if sd[j] > 0 else float('nan'),
                 bonferroni=bool(p[j] < 0.05 / nt), units_active=int((Y[:, j] != 0).sum()))
        if r < MAX_HITS_CHECKED:
            for c in nuis:
                e = eta2_perm(Y[:, j].astype(np.float64), units[c].to_numpy(), arm, rng)
                if e is not None:
                    for k, v in e.items():
                        h[f'{c}_{k}'] = v
        hits.append(h)
    # nuisance summary over the checked hits: per field, hits with permutation p < 0.05 (expected 5% by chance)
    chk = hits[:MAX_HITS_CHECKED]
    summ = {}
    for c in nuis:
        key = f'{c}_perm_p_within_arm' if arm is not None else f'{c}_perm_p'
        vals = [h[key] for h in chk if key in h]
        if vals:
            summ[c] = dict(n_checked=len(vals), n_perm_p05=int(sum(v < 0.05 for v in vals)),
                           test='within arm' if arm is not None else 'raw (paired: per-pool change)')
    res['nuisance_summary'] = summ
    return res, hits, p, tt


def alignment(top_idx, top_val, labels, lab_names, alive):
    out = {}
    for li, ln in enumerate(lab_names):
        rows = np.where(~np.isnan(labels[:, li]))[0]
        if len(rows) == 0:
            continue
        y = labels[rows, li].astype(np.int8)
        X = dense_rows(top_idx, top_val, rows, alive)
        act = X > 0
        pos = y == 1
        tp = act[pos].sum(0).astype(np.float64)
        fp = act[~pos].sum(0).astype(np.float64)
        fn = pos.sum() - tp
        with np.errstate(invalid='ignore', divide='ignore'):
            pr, rc = np.where(tp + fp > 0, tp / (tp + fp), 0), np.where(tp + fn > 0, tp / (tp + fn), 0)
            f1 = np.where(pr + rc > 0, 2 * pr * rc / (pr + rc), 0)
        au = np.array([roc_auc_score(y, X[:, k]) if 0 < y.sum() < len(y) else np.nan for k in range(X.shape[1])])
        prev = float(y.mean())
        jf, ja = int(np.argmax(f1)), int(np.nanargmax(np.abs(au - 0.5)))
        out[ln] = dict(n_frames=int(len(rows)), prevalence=prev, always_on_f1=2 * prev / (1 + prev),
                       best_f1=dict(neuron=f'N_{alive[jf] + 1}', f1=float(f1[jf]), auroc=float(au[jf])),
                       best_auroc=dict(neuron=f'N_{alive[ja] + 1}', auroc=float(au[ja]), f1=float(f1[ja])),
                       auroc_by_code={int(alive[k]): float(au[k]) for k in range(len(alive))},
                       f1_by_code={int(alive[k]): float(f1[k]) for k in range(len(alive))})
        log(f"alignment {ln}: prevalence {prev:.4f}, always-on F1 {2 * prev / (1 + prev):.3f}, best F1 "
            f"{f1[jf]:.3f} (N_{alive[jf] + 1}), best |AUROC-0.5| AUROC {au[ja]:.3f} (N_{alive[ja] + 1})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=list(SUBJECT))
    ap.add_argument('--video-dir', default='', help='staged videos (flat dir); default = the repo paths')
    ap.add_argument('--stage-dir', required=True, help='local scratch for the float16 token store')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--big-dir', default=None)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-frames', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num-workers', type=int, default=14)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--smoke', type=int, default=0, help='keep only the first n videos of each arm value (test run)')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out_dir)
    big = Path(args.big_dir or BIG_ROOT / args.domain)
    out.mkdir(parents=True, exist_ok=True)
    big.mkdir(parents=True, exist_ok=True)
    log(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}")

    d, nuis = load_design(args.domain)
    if args.smoke:
        col = {'ants_v3': 'T', 'ants_v4': 'T', 'mice_v1': 'stage', 'frogs_v1': 'group'}[args.domain]
        if args.domain == 'mice_v1':
            pools = sorted(set(d[d.genotype == 'het'].pool))[:args.smoke] + sorted(set(d[d.genotype == 'wt'].pool))[:args.smoke]
            d = d[d.pool.isin(pools)]
        elif args.domain == 'ants_v4':
            d = d.groupby(['experiment', 'T'], group_keys=False).head(args.smoke)
        else:
            d = d.groupby(col, group_keys=False).head(args.smoke)
        d = d.reset_index(drop=True)
        d['offset'] = np.concatenate([[0], np.cumsum(d.n_keep.to_numpy())[:-1]])
    ans = analyses(args.domain, d)
    log(f"{args.domain}: {len(d)} videos, {int(d.n_keep.sum())} frames at 2 fps; analyses {[a['id'] for a in ans]}")
    labels, lab_names = load_labels(args.domain, d)
    if labels is not None:
        log(f"labels {lab_names}: labelled frames {[int((~np.isnan(labels[:, i])).sum()) for i in range(labels.shape[1])]}")
    frame_table(d).to_csv(big / 'frames.csv', index=False)

    tok_path = Path(args.stage_dir) / 'tokens_fp16.npy'
    cache = [big / f for f in ('frame_codes_top20_idx.npy', 'frame_codes_top20_val.npy', 'video_codes.npy',
                               'quality.json', 'sae_train_log.json')]
    if all(f.exists() for f in cache) and not args.overwrite:  # rerun of the search / checks only (CPU is enough)
        log(f'loading cached codes from {big}')
        top_idx, top_val, vcodes = (np.load(f) for f in cache[:3])
        q = json.loads(cache[3].read_text())
        (out / 'sae_train_log.json').write_text(cache[4].read_text())
        assert len(top_idx) == int(d.n_keep.sum()) and len(vcodes) == len(d)
    else:
        pooled = extract(d, args.video_dir, tok_path, device, args.num_workers)
        np.save(big / 'pooled_tokens.npy', pooled)
        sae, hist = train_sae(tok_path, device, args.epochs, args.batch_frames, args.seed)
        torch.save(sae, big / 'sae_topk_4608_k20_s0.pt')
        (out / 'sae_train_log.json').write_text(json.dumps(hist, indent=1))
        top_idx, top_val, vcodes, q = encode(sae, pooled, tok_path, d, device)
        tok_path.unlink()
        np.save(cache[0], top_idx)
        np.save(cache[1], top_val)
        np.save(cache[2], vcodes)
        cache[3].write_text(json.dumps(q, indent=1))
        cache[4].write_text(json.dumps(hist, indent=1))
    log(f"quality: {q}")

    rng = np.random.default_rng(0)
    res = dict(domain=args.domain, n_videos=len(d), quality=q, analyses=[], smoke=args.smoke,
               recipe='v0 = original ECI paper recipe; see the docstring of scripts/eci/v0/v0_domains.py')
    all_hits, pcols = [], {}
    for an in ans:
        r, hits, p, tt = run_analysis(an, d, nuis, vcodes, rng)
        res['analyses'].append(r)
        all_hits += hits
        pcols[f'p_{an["id"]}'] = p
        pcols[f't_{an["id"]}'] = tt
        log(f"{an['id']}: {json.dumps({k: v for k, v in r.items() if k not in ('nuisance_summary',)})} "
            f"nuisance {json.dumps(r['nuisance_summary'])}")
    alive_mask = np.zeros(N_CODES, bool)
    for k in range(TOP_K):
        alive_mask[np.unique(top_idx[top_val[:, k] > 0, k])] = True
    alive = np.where(alive_mask)[0]
    if labels is not None:
        al = alignment(top_idx, top_val, labels, lab_names, alive)
        for h in all_hits:
            for ln, a in al.items():
                h[f'auroc_{ln}'] = a['auroc_by_code'].get(h['index0'], float('nan'))
                h[f'f1_{ln}'] = a['f1_by_code'].get(h['index0'], 0.0)
        for a in al.values():
            a.pop('auroc_by_code')
            a.pop('f1_by_code')
        res['alignment'] = al
        # best-aligned code among the hits of each analysis (largest |AUROC - 0.5| on the 'any' label)
        for r in res['analyses']:
            hs = [h for h in all_hits if h['analysis'] == r['id'] and np.isfinite(h.get('auroc_any', np.nan))]
            if hs:
                b = max(hs, key=lambda h: abs(h['auroc_any'] - 0.5))
                bf = max(hs, key=lambda h: h['f1_any'])
                r['best_aligned_hit_any'] = dict(neuron=b['neuron'], auroc_any=b['auroc_any'], f1_any=b['f1_any'],
                                                 p=b['p'], best_f1_neuron=bf['neuron'], best_f1_any=bf['f1_any'])
    np.save(out / 'video_codes.npy', vcodes.astype(np.float32))
    d.drop(columns=['offset']).to_csv(out / 'videos.csv', index=False)
    pd.DataFrame(dict(neuron=[f'N_{j + 1}' for j in range(N_CODES)], index0=np.arange(N_CODES), **pcols)).to_csv(
        out / 'pvalues_all_codes.csv', index=False)
    pd.DataFrame(all_hits).to_csv(out / 'found_neurons.csv', index=False)
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    log('DONE')


if __name__ == '__main__':
    main()
