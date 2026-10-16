"""
ECI "v0" = an exact replication of the original ECI paper recipe ("Exploratory Causal Inference in
SAEnce", arXiv 2510.14073 v2, App. D.2) on ants v2 (ISTAnt). Reference code (read-only):
/nfs/scistore19/locatgrp/tmencatt/projects/neuralscience  (encoders.py, sae.py, nes.py,
runnable/s2-encode_dataset.py, s3-train_sae.py, s4-sae_encoding.py, notebooks/istant/istant-experiment.ipynb).

Stages (each one is skipped if its output already exists on BeeGFS, unless --overwrite):
  1. frames -> tokens. The 44 valid videos (data/ants/v2/experiment.csv, `valid` == 1, CSV order, b3 out).
     Frame picking = src/old/data.py: decode the 30 fps source .mkv sequentially with cv2, keep every
     15th frame (source frames 0, 15, ..., 17985 -> 1200 frames = 10 min at 2 fps), BGR -> RGB.
     The original manifest stored these frames as JPEGs (PIL default quality), so each frame does an
     in-memory JPEG round trip before the encoder. Encoder = overcomplete 0.3.0 DinoV2 wrapper
     (dinov2_vits14, Resize 224x224 bicubic antialias, ImageNet norm, final-layer x_norm_patchtokens)
     -> (256, 384) float32 tokens per frame.
  2. SAE. overcomplete TopKSAE(384 -> 4608 = 12 x 384, top_k 20), default Linear+BatchNorm encoder,
     Adam lr 5e-4, plain MSE, grad-norm clip 1.0, 10 epochs, batches of 128 FRAMES (= 32,768 tokens,
     the original collate flattens the tokens), NO shuffling (the original DataLoader has shuffle off,
     so batches follow manifest order: video by video), all frames of all 44 videos (no split). Seed 0
     (the original seed is unrecorded).
  3. Encoding (s4-sae_encoding.py): sae.eval(); per frame MEAN of the 256 patch tokens -> SAE -> 4608
     codes; per video the mean over its 1200 frames -> 44 x 4608 table.
  4. Search + gate checks: single-pass Welch t-test per code (treated = CSV treatment 2 -> T = 1, as in
     the original manifest), raw p < 0.05; the recursive neural_effect_search of the original nes.py
     (vendored verbatim in scripts/eci/v0/nes_original.py) with the notebook's arguments; alive codes,
     reconstruction quality, grooming F1 / AUROC per found code, background / position checks.

Neuron names follow the original export: column N_j holds code index j - 1 (so the paper's "394" is
N_394 = 0-based index 393). Both are written out.

Usage (inside a Slurm job, see ants_v2_v0.sh):
    python scripts/eci/v0/ants_v2_v0.py --video-dir /localhome/$USER/$SLURM_JOB_ID/videos \
        --out-dir /localhome/$USER/$SLURM_JOB_ID/out
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
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nes_original import neural_effect_search, neural_effect_search_base  # noqa: E402

N_FRAMES = 1200          # frames kept per video (10 min at 2 fps)
FPS_FACTOR = 15          # src/old/data.py reduce_fps_factor: 30 fps -> 2 fps
N_TOKENS, DIM = 256, 384  # DINOv2-S/14 at 224 px: 16 x 16 patches, 384 dims
N_CODES, TOP_K = 12 * DIM, 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_line():
    if not torch.cuda.is_available():
        return 'no gpu'
    free, total = torch.cuda.mem_get_info()
    return f"gpu mem used {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB, peak alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"


# ------------------------------------------------------------------------------------------------
# labels (copied logic of src/old/data.py: map_behaviour_to_label / label_frame / load_labels)
# ------------------------------------------------------------------------------------------------
def map_behaviour_to_label(behaviour):
    if behaviour in (' groom-yellow', ' groom-orange'):
        return (1, 0)
    if behaviour == ' groom-blue':
        return (0, 1)
    if behaviour in (' groom-yellowandblue', ' groom-orangeandblue'):
        return (1, 1)
    raise ValueError(f'Unknown behaviour: {behaviour}')


def load_labels(path):
    """(1200, 2) int labels [yellow->focal, blue->focal] at source frames 0, 15, ..., 17985."""
    beh = pd.read_csv(path, skiprows=3, skipfooter=1, engine='python')
    out = np.zeros((N_FRAMES, 2), dtype=np.int64)
    for _, row in beh.iterrows():
        y, b = map_behaviour_to_label(row[' Behavior'])
        for i in range(N_FRAMES):
            f = i * FPS_FACTOR
            if row[' Beginning-frame'] <= f < row[' End-frame']:
                out[i] += (y, b)
    return (out > 0).astype(np.int8)  # the original manifest labels are binary


# ------------------------------------------------------------------------------------------------
# stage 1: frames -> DINOv2 patch tokens
# ------------------------------------------------------------------------------------------------
class VideoChunks(IterableDataset):
    """Each worker decodes whole videos sequentially (like the old cv2 loader) and yields chunks of
    preprocessed frames: (video position, first frame index, (n, 3, 224, 224) float32)."""

    def __init__(self, paths, preprocess, chunk=128):
        self.paths, self.preprocess, self.chunk = paths, preprocess, chunk

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (0, 1) if info is None else (info.id, info.num_workers)
        cv2.setNumThreads(1)
        for vi in range(wid, len(self.paths), nw):
            cap = cv2.VideoCapture(str(self.paths[vi]))
            frame_count, kept, buf, start = 0, 0, [], 0
            while kept < N_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % FPS_FACTOR == 0:
                    img = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))  # BGR -> RGB
                    jpg = io.BytesIO()
                    img.save(jpg, format='JPEG')  # PIL default quality, like the original manifest
                    jpg.seek(0)
                    buf.append(self.preprocess(Image.open(jpg).convert('RGB')))
                    kept += 1
                    if len(buf) == self.chunk:
                        yield vi, start, torch.stack(buf)
                        start, buf = start + len(buf), []
                frame_count += 1
            cap.release()
            if buf:
                yield vi, start, torch.stack(buf)


def extract_tokens(videos, video_dir, device, num_workers):
    from overcomplete.models import DinoV2
    model = DinoV2(device=device)
    model.to(device)
    model.eval()
    paths = [Path(video_dir) / f for f in videos.observation_file]
    tokens = np.empty((len(videos) * N_FRAMES, N_TOKENS, DIM), dtype=np.float32)
    counts = np.zeros(len(videos), dtype=np.int64)
    loader = DataLoader(VideoChunks(paths, model.preprocess), batch_size=None,
                        num_workers=num_workers, prefetch_factor=2)
    t0, done, last = time.time(), 0, 0
    with torch.inference_mode():
        for vi, start, x in loader:
            feats = model.forward_features(x.to(device, non_blocking=True))  # (n, 256, 384)
            assert feats.shape[1:] == (N_TOKENS, DIM), feats.shape
            g = vi * N_FRAMES + start
            tokens[g:g + len(x)] = feats.float().cpu().numpy()
            counts[vi] += len(x)
            done += len(x)
            if done - last >= 5000:
                last = done
                log(f"tokens {done}/{len(tokens)} frames, {done / (time.time() - t0):.0f} frames/s, {gpu_line()}")
    log(f"tokens done: {done} frames in {time.time() - t0:.0f}s ({done / (time.time() - t0):.0f} frames/s)")
    return tokens, counts


# ------------------------------------------------------------------------------------------------
# stage 2: TopK SAE training (s3-train_sae.py + sae.py train_sae, without wandb)
# ------------------------------------------------------------------------------------------------
def train_sae(tokens, device, epochs, batch_frames, seed):
    from overcomplete.sae import TopKSAE
    from overcomplete.sae.trackers import DeadCodeTracker
    from overcomplete.sae.train import _compute_reconstruction_error
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    sae = TopKSAE(input_shape=DIM, nb_concepts=N_CODES, top_k=TOP_K, device=device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=5e-4)
    n = len(tokens)
    starts = list(range(0, n, batch_frames))  # in order, no shuffle; last batch is smaller
    hist = []
    for epoch in range(epochs):
        sae.train()
        t0 = time.time()
        ep_loss = ep_r2 = ep_l0 = 0.0
        tracker = None
        for s in starts:
            x = torch.from_numpy(tokens[s:s + batch_frames]).to(device, non_blocking=True)
            x = x.reshape(-1, DIM)  # 'n t d -> (n t) d', as WithFeatures.collate_fn
            optimizer.zero_grad(set_to_none=True)
            z_pre, z, x_hat = sae(x)
            loss = (x - x_hat).square().mean()  # sae.py Loss.criterion with no reanimation / sparsity terms
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
        nb = len(starts)
        row = dict(epoch=epoch + 1, loss=ep_loss / nb, r2_overcomplete=ep_r2 / nb, l0=ep_l0 / nb,
                   dead_ratio=tracker.get_dead_ratio(), seconds=time.time() - t0,
                   tokens_per_s=n * N_TOKENS / (time.time() - t0))
        hist.append(row)
        log(f"SAE epoch {epoch + 1}/{epochs}: loss {row['loss']:.5f}, R2 {row['r2_overcomplete']:.4f}, "
            f"L0 {row['l0']:.1f}, dead {100 * row['dead_ratio']:.1f}%, {row['seconds']:.0f}s "
            f"({row['tokens_per_s']:.0f} tokens/s), {gpu_line()}")
    sae.eval()
    return sae, hist


# ------------------------------------------------------------------------------------------------
# stage 3: encoding (s4-sae_encoding.py with pooling MEAN) + reconstruction quality
# ------------------------------------------------------------------------------------------------
class FVE:
    """Fraction of variance explained, 1 - SSE / SST with the per-dimension mean (float64 sums)."""

    def __init__(self):
        self.n, self.s, self.s2, self.sse = 0, 0.0, 0.0, 0.0

    def update(self, x, x_hat):
        x64 = x.double()
        self.n += x.shape[0]
        self.s = self.s + x64.sum(0)
        self.s2 = self.s2 + (x64 ** 2).sum(0)
        self.sse += ((x64 - x_hat.double()) ** 2).sum().item()

    def value(self):
        sst = (self.s2 - self.s ** 2 / self.n).sum().item()
        return 1.0 - self.sse / sst


def encode(sae, tokens, device, batch_frames=128):
    n = len(tokens)
    frame_codes = np.empty((n, N_CODES), dtype=np.float32)
    fve_tok, fve_pool = FVE(), FVE()
    alive_tok = torch.zeros(N_CODES, dtype=torch.bool, device=device)
    l0_tok = 0.0
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, n, batch_frames):
            feats = torch.from_numpy(tokens[s:s + batch_frames]).to(device)
            # token-level quality (the SAE's training distribution)
            x = feats.reshape(-1, DIM)
            _, z, x_hat = sae(x)
            fve_tok.update(x, x_hat)
            alive_tok |= (z > 0).any(0)
            l0_tok += (z > 0).sum().item()
            # what the paper actually uses: mean over the 256 patch tokens, then the SAE
            pooled = feats.mean(dim=1)
            _, zp, xp_hat = sae(pooled)
            fve_pool.update(pooled, xp_hat)
            frame_codes[s:s + len(feats)] = zp.float().cpu().numpy()
    log(f"encoding done in {time.time() - t0:.0f}s, {gpu_line()}")
    quality = dict(fve_tokens=fve_tok.value(), fve_meanpooled=fve_pool.value(),
                   alive_codes_tokens=int(alive_tok.sum().item()), l0_tokens=l0_tok / (n * N_TOKENS),
                   alive_codes_meanpooled_frames=int((frame_codes > 0).any(0).sum()),
                   l0_meanpooled_frames=float((frame_codes > 0).sum(1).mean()))
    return frame_codes, quality


# ------------------------------------------------------------------------------------------------
# stage 4: search + gate checks
# ------------------------------------------------------------------------------------------------
def eta2(values, groups):
    """Share of the variance of `values` explained by the categorical `groups` (one-way ANOVA R^2)."""
    v = np.asarray(values, dtype=np.float64)
    g = pd.Series(groups).astype(str).to_numpy()
    sst = ((v - v.mean()) ** 2).sum()
    if sst == 0:
        return float('nan')
    ssb = sum(len(v[g == k]) * (v[g == k].mean() - v.mean()) ** 2 for k in np.unique(g))
    return float(ssb / sst)


def f1_rule(active, y):
    """prf_rule_sign0_multi: prediction = code > 0."""
    tp = int((active & (y == 1)).sum())
    fp = int((active & (y == 0)).sum())
    fn = int((~active & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)


def analyse(videos, frame_codes, labels, quality):
    nv = len(videos)
    T = (videos.treatment.to_numpy() == 2).astype(int)  # original manifest: treatment 2 -> T = 1
    vcodes = frame_codes.reshape(nv, N_FRAMES, N_CODES).astype(np.float64).mean(1)
    names = [f'N_{j + 1}' for j in range(N_CODES)]
    res = dict(n_treated=int(T.sum()), n_control=int((1 - T).sum()),
               treated_definition='CSV treatment == 2 -> T = 1 (original manifest)')

    # --- primary: single-pass Welch t-test, raw p < 0.05 -------------------------------------------
    with np.errstate(invalid='ignore', divide='ignore'):
        t_stat, pvals = ttest_ind(vcodes[T == 1], vcodes[T == 0], axis=0, equal_var=False)
    valid = np.isfinite(pvals)
    n_valid = int(valid.sum())
    sig = np.where(valid & (pvals < 0.05))[0]
    m1, m0 = vcodes[T == 1].mean(0), vcodes[T == 0].mean(0)
    sd_pooled = np.sqrt((vcodes[T == 1].var(0, ddof=1) * (T.sum() - 1) + vcodes[T == 0].var(0, ddof=1) * ((1 - T).sum() - 1)) / (nv - 2))
    res['single_pass'] = dict(
        n_testable_codes=n_valid, n_found_raw_p05=int(len(sig)), expected_false_positives=0.05 * n_valid,
        bonferroni_threshold=0.05 / n_valid, n_found_bonferroni=int((valid & (pvals < 0.05 / n_valid)).sum()),
        min_p=float(np.nanmin(pvals)))
    # cross-check against the original code path (neural_effect_search_base, test AD, method t-test)
    df = pd.DataFrame(vcodes, columns=names)
    df.insert(0, 'T', T)
    base = neural_effect_search_base(df, test='AD', method='t-test', alpha=0.05, k=2000,
                                     neuron_prefix='N_', fallback_start_neur_index=4, sn=[])
    assert sorted(int(s) - 1 for s in base['neurons']) == sorted(sig.tolist()), 'single-pass mismatch vs nes.py'
    res['single_pass']['matches_original_nes_base'] = True

    # --- secondary: the original recursive neural_effect_search -----------------------------------
    res['recursive'] = {}
    for method in ['t-test', 'Bonferroni', 'FDR']:
        out = neural_effect_search(df, method=method, alpha=0.05, k=2000, neuron_prefix='N_',
                                   fallback_start_neur_index=4)
        res['recursive'][method] = out
        log(f"recursive NES ({method}): {out['neurons']}")

    # --- per found code: effect, grooming F1 / AUROC, background checks ---------------------------
    y_or = (labels.sum(1) > 0).astype(np.int8)
    y_y, y_b = labels[:, 0], labels[:, 1]
    vid = np.repeat(np.arange(nv), N_FRAMES)
    meta_frame = {f'batch_{b}': np.repeat((videos.batch == b).to_numpy(), N_FRAMES) for b in sorted(videos.batch.unique())}
    meta_frame.update({f'pos_{p}': np.repeat((videos.position == p).to_numpy(), N_FRAMES) for p in sorted(videos.position.unique())})
    meta_frame.update({f'day_{d}': np.repeat((videos.recording_date == d).to_numpy(), N_FRAMES) for d in sorted(videos.recording_date.unique())})
    rows = []
    for j in sorted(sig, key=lambda j: pvals[j]):
        c = frame_codes[:, j]
        act = c > 0
        row = dict(neuron=names[j], index0=int(j), mean_treated=float(m1[j]), mean_control=float(m0[j]),
                   diff=float(m1[j] - m0[j]), cohen_d=float((m1[j] - m0[j]) / sd_pooled[j]),
                   welch_t=float(t_stat[j]), p=float(pvals[j]), frac_frames_active=float(act.mean()),
                   n_videos_active=int((vcodes[:, j] > 0).sum()))
        for lab, y in [('groom_any', y_or), ('groom_yellow', y_y), ('groom_blue', y_b)]:
            row[f'f1_{lab}'] = f1_rule(act, y)['f1']
            row[f'auroc_{lab}'] = float(roc_auc_score(y, c))
        # background-like: is the code mostly a per-video constant, and does that constant follow the setup?
        row['eta2_frames_by_video'] = eta2(c, vid)
        row['eta2_videomean_by_batch'] = eta2(vcodes[:, j], videos.batch)
        row['eta2_videomean_by_position'] = eta2(vcodes[:, j], videos.position)
        row['eta2_videomean_by_day'] = eta2(vcodes[:, j], videos.recording_date)
        best = max(((k, f1_rule(act, v.astype(np.int8))['f1']) for k, v in meta_frame.items()), key=lambda kv: kv[1])
        row['best_setup_label'], row['best_setup_f1'] = best
        rows.append(row)
    res['found'] = rows

    # --- context: best grooming F1 / AUROC over ALL alive codes (as the notebook did for F1) ------
    act_all = frame_codes > 0
    alive = np.where(act_all.any(0))[0]
    pos = y_or == 1
    tp = act_all[pos].sum(0).astype(np.float64)
    fp = act_all[~pos].sum(0).astype(np.float64)
    fn = pos.sum() - tp
    with np.errstate(invalid='ignore', divide='ignore'):
        prec, rec = np.where(tp + fp > 0, tp / (tp + fp), 0), np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
    aurocs = np.full(N_CODES, np.nan)
    for j in alive:
        aurocs[j] = roc_auc_score(y_or, frame_codes[:, j])
    jf, ja = int(np.argmax(f1)), int(np.nanargmax(aurocs))
    res['all_codes_grooming'] = dict(
        best_f1=dict(neuron=names[jf], f1=float(f1[jf]), auroc=float(aurocs[jf]), p=float(pvals[jf])),
        best_auroc=dict(neuron=names[ja], auroc=float(aurocs[ja]), f1=float(f1[ja]), p=float(pvals[ja])),
        grooming_prevalence=float(y_or.mean()))

    # --- context: video-level grooming-rate t-test between arms -----------------------------------
    rate = labels.reshape(nv, N_FRAMES, 2).mean(1)
    rate_or = y_or.reshape(nv, N_FRAMES).mean(1)
    gt = {}
    for lab, r in [('any', rate_or), ('yellow', rate[:, 0]), ('blue', rate[:, 1]), ('yellow_plus_blue', rate.sum(1))]:
        t_, p_ = ttest_ind(r[T == 1], r[T == 0], equal_var=False)
        gt[lab] = dict(mean_treated=float(r[T == 1].mean()), mean_control=float(r[T == 0].mean()), t=float(t_), p=float(p_))
    res['grooming_rate_ttest'] = gt
    res['quality'] = quality
    return res, vcodes, pvals, t_stat


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video-dir', default=str(REPO / 'data/ants/v2/observations/source'))
    p.add_argument('--out-dir', required=True, help='small results (copied to results/vision/ants/eci_v0 by the job)')
    p.add_argument('--big-dir', default='/mnt/beegfs/locatgrp/shared/rcadei/eci_v0/ants_v2')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-frames', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=12)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    device = 'cuda'
    out, big = Path(args.out_dir), Path(args.big_dir)
    out.mkdir(parents=True, exist_ok=True)
    big.mkdir(parents=True, exist_ok=True)
    log(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}")

    exp = pd.read_csv(REPO / 'data/ants/v2/experiment.csv')
    videos = exp[exp.valid == 1].reset_index(drop=True)
    assert len(videos) == 44 and 'b3' not in set(videos.observation_id), videos.observation_id.tolist()
    labels = np.concatenate([load_labels(REPO / 'data/ants/v2/annotations' / f) for f in videos.annotation_file])
    log(f"{len(videos)} valid videos; labels {labels.shape}, grooming (any) prevalence {(labels.sum(1) > 0).mean():.4f}")

    # stage 1: tokens (5 large float32 shards on BeeGFS, one per batch a-e)
    shard_paths = {b: big / f'tokens_batch_{b}.npy' for b in sorted(videos.batch.unique())}
    if all(pth.exists() for pth in shard_paths.values()) and not args.overwrite:
        log('loading cached tokens from BeeGFS')
        tokens = np.concatenate([np.load(shard_paths[b]) for b in sorted(shard_paths)])
        counts = np.full(len(videos), N_FRAMES)
        assert len(tokens) == len(videos) * N_FRAMES
    else:
        tokens, counts = extract_tokens(videos, args.video_dir, device, args.num_workers)
        assert (counts == N_FRAMES).all(), dict(zip(videos.observation_id, counts))
        for b, pth in shard_paths.items():
            rows = np.where(np.repeat((videos.batch == b).to_numpy(), N_FRAMES))[0]
            assert (np.diff(rows) == 1).all()  # batches are contiguous in CSV order
            np.save(pth, tokens[rows[0]:rows[-1] + 1])
        log(f"saved token shards to {big}")
    frames = pd.DataFrame(dict(observation_id=np.repeat(videos.observation_id, N_FRAMES),
                               source_frame=np.tile(np.arange(N_FRAMES) * FPS_FACTOR, len(videos)),
                               groom_yellow=labels[:, 0], groom_blue=labels[:, 1]))
    frames.to_csv(big / 'frames.csv', index=False)
    frames_per_video = dict(zip(videos.observation_id, counts.tolist()))

    # stage 2: SAE
    sae_path = big / 'sae_topk_4608_k20_s0.pt'
    if sae_path.exists() and not args.overwrite:
        log('loading cached SAE')
        sae = torch.load(sae_path, map_location=device, weights_only=False)
        sae.eval()
        hist = json.loads((out / 'sae_train_log.json').read_text()) if (out / 'sae_train_log.json').exists() else []
    else:
        sae, hist = train_sae(tokens, device, args.epochs, args.batch_frames, args.seed)
        torch.save(sae, sae_path)  # same format as sae.py save_sae
    (out / 'sae_train_log.json').write_text(json.dumps(hist, indent=1))

    # stage 3: codes
    frame_codes, quality = encode(sae, tokens, device)
    del tokens
    np.save(big / 'frame_codes_meanpool.npy', frame_codes)
    log(f"quality: {quality}")

    # stage 4: search + gates
    res, vcodes, pvals, t_stat = analyse(videos, frame_codes, labels, quality)
    res['frames_per_video'] = frames_per_video
    res['recipe'] = dict(encoder='overcomplete 0.3.0 DinoV2 (torch hub dinov2_vits14), 224 bicubic, x_norm_patchtokens',
                         frames='cv2 sequential decode, every 15th source frame 0..17985, JPEG round trip (PIL default q)',
                         sae=f'overcomplete TopKSAE {DIM}->{N_CODES}, k={TOP_K}, Adam 5e-4, MSE, clip 1.0, '
                             f'{args.epochs} epochs, {args.batch_frames} frames/batch, no shuffle, seed {args.seed}',
                         pooling='MEAN of 256 patch tokens before the SAE; video = mean over 1200 frames',
                         search='Welch ttest_ind equal_var=False, raw p<0.05; recursive nes.py neural_effect_search')
    np.save(out / 'video_codes.npy', vcodes.astype(np.float32))
    vt = videos.copy()
    vt['T'] = (vt.treatment == 2).astype(int)
    vt.to_csv(out / 'videos.csv', index=False)
    pd.DataFrame(dict(neuron=[f'N_{j + 1}' for j in range(N_CODES)], index0=np.arange(N_CODES),
                      welch_t=t_stat, p=pvals)).to_csv(out / 'pvalues_all_codes.csv', index=False)
    pd.DataFrame(res['found']).to_csv(out / 'found_neurons.csv', index=False)
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    log(f"single pass: {res['single_pass']}")
    log(f"found: {pd.DataFrame(res['found']).to_string()}")
    log(f"grooming: {res['all_codes_grooming']} | rate t-test {res['grooming_rate_ttest']}")
    log('DONE')


if __name__ == '__main__':
    main()
