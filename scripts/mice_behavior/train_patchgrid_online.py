"""
"Online" DINOv2 patch-grid training: uses ALL raw patch tokens (256 per frame, no
pooling to 4x4) instead of the cached, adaptive-pooled patch_grid4 embeddings —
infeasible to permanently cache to disk at this resolution for reuse across many
future experiments (200,000 frames x 256 patches x 768dim x fp16 is ~78.6GB, vs.
~4.9GB for the pooled 16-token version this repo already keeps).

Design: re-decoding+re-encoding every frame on every batch (the first, naive version
of this script did exactly that) was far too slow — still mid-way through its first
training epoch after 15+ minutes. Instead: compute every candidate frame's patch
tokens through the frozen encoder exactly ONCE (the WHOLE max_train_frames-bounded
pool: all positives + all available negatives, not just one epoch's worth), via a
properly parallelized DataLoader (matching the ~450-500 frames/sec this repo's own
extraction scripts achieve), hold the result in CPU RAM only for the duration of this
one run (never written to disk), then train fast from that in-memory cache. This is
still "online" in the sense that matters (no permanent disk cache of the
full-resolution representation), just not recomputed on every single batch.

Because the FULL candidate pool is cached (not just one draw), each epoch can afford
to resample a fresh neg_ratio-sized subset of negatives from it (see the epoch loop) —
different negatives each epoch, zero extra encoding cost, instead of training on the
exact same fixed negative set for the whole run. This only actually varies anything
when the candidate pool exceeds what neg_ratio would draw; at small max_train_frames
bounds the negative pool is often already saturated (all available negatives get used
every epoch regardless — see the printed neg-pool message), in which case there is
nothing left to resample and this degrades gracefully to the old fixed-set behavior.

Reuses FrameBatchData's tested sample-selection/context-window/label logic (built
with a tiny placeholder load_embeddings_fn so its own memory footprint stays
negligible — n_patches=1, emb_dim=1 — then its `flat` placeholder is discarded
entirely); only `gi` (global frame index per sample), `offsets_grid`, `pad_mask`,
and `labels` are used to determine which raw frames actually need encoding.

Reuses the confirmed-best patch-grid hyperparameters (results/vision/mice/frame/
patchgrid/config.json) rather than re-searching — this is a representation-quality
test (pooled 16 tokens vs. all 256), not a hyperparameter test.

Capped at 30 epochs (patience 10), not the usual 100/15 — prior patch-grid runs have
consistently converged well before epoch 30 anyway, and this bounds the (much
cheaper, post-encode) training-loop cost.

Anti-overfitting knobs (train AP 0.91 vs. val AP 0.25 on the first patchgrid256 run —
train near-ceiling is the signature of the model memorizing train-pool idiosyncrasies
rather than the behavioral signal, given only ~19 unique train pools):
  - patch_dropout / patch_noise_std / frame_dropout: embedding-space augmentation
    (see MouseFrameClassifier docstring) — applied to the cached tokens on the fly
    inside the model, so it costs nothing extra on top of the one-time encode (no
    re-running the frozen encoder, unlike pixel-space augmentation which would).
  - cross_attn_dim: bottlenecks the temporal cross-attention module from 768 down to
    CROSS_ATTN_DIM — a full 768-dim attention module has no business fitting a 5-position
    context window (context_k=2); most of its ~2.4M params had nowhere useful to go.
  - PatchAttnPool's own attention now also gets `dropout` (previously only the head and
    cross_attn did — an oversight, now fixed for every caller of MouseFrameClassifier).

Usage:
    python scripts/mice_behavior/train_patchgrid_online.py
    python scripts/mice_behavior/train_patchgrid_online.py --smoke   # tiny local sanity check
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import get_fixed_val_pools
from src.mice_behavior.head_cfg import get_head_cfg
from src.dataset.get_dataset import load_dataset

DATA_DIR = gsf.DATA_DIR
DATASET_DIR = gsf.DATASET_DIR
SEED = gsf.SEED
MODEL_ID = 'facebook/dinov2-base'
EMB_DIM = 768
N_PATCHES_FULL = 256  # 16x16, no pooling, no register tokens — confirmed empirically


def dummy_loader(n_patches, emb_dim):
    def _load(obs_boundary):
        return {obs_s: np.zeros((obs_e - obs_s, n_patches, emb_dim), dtype=np.float16) for obs_s, obs_e in obs_boundary.values()}
    return _load


def named_group_grad_norms(model):
    """Per-top-level-submodule L2 grad norm (patch_pool, cross_attn, head, ...) -- call
    AFTER backward()+unscale_(), BEFORE clip_grad_norm_/zero_grad(). Cheap (grads already
    materialized, just a few reductions) -- localizes vanishing/exploding gradients to a
    specific module instead of only seeing one aggregate number."""
    groups = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        top = name.split('.')[0]
        groups.setdefault(top, []).append(p.grad.detach())
    return {g: torch.sqrt(sum((t.float() ** 2).sum() for t in tensors)).item() for g, tensors in groups.items()}


def focal_loss_with_pos_weight(logits, targets, pos_weight, gamma=2.0):
    """BCE-with-logits, but each element's loss is additionally scaled by (1-p_t)^gamma
    (Lin et al. 2017) so well-classified examples (easy negatives, and positives the
    model already nails) contribute almost nothing to the gradient/loss, while the
    still-hard examples dominate — unlike plain pos_weight, which upweights *every*
    positive uniformly (easy and hard alike), including the confidently-classified ones
    that don't need it. Keeps pos_weight's existing broadcasting convention (only scales
    the positive-class term) so it composes with the pos_weight already computed here."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits.float())
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_term = (1 - p_t).clamp(min=1e-6) ** gamma
    class_weight = pos_weight * targets + (1 - targets)
    return (class_weight * focal_term * bce).mean()


class _ImageDataset(Dataset):
    """Indexes the HF frame dataset directly by a list of global row indices, applying the
    DINOv2 processor per-item (so workers do the CPU-bound decode+resize in parallel).

    input_size: if set, overrides the processor's default resize+center-crop (which
    downsamples the native 512x512 frame to 224x224 -- a 2.3x linear / 5x area loss of
    detail) with a plain resize to (input_size, input_size), no cropping. Must be a
    multiple of the encoder's patch_size (14 for dinov2-base) for a clean patch grid.
    DINOv2 handles non-default resolutions natively (position embeddings interpolate
    internally, confirmed empirically -- no extra flag needed).

    blur_to: if set (must be <= input_size), the image is first downsized to
    (blur_to, blur_to) and then upsized back to (input_size, input_size) before the
    processor's normalization -- i.e. the SAME patch count as input_size, but with no
    real detail beyond what blur_to could represent (upsampling interpolates, it can't
    invent information). Isolates whether a resolution win is from genuine extra detail
    vs. just more attention "slots" (patch tokens) for patch_pool to work with."""

    def __init__(self, hf_dataset, global_indices, processor, input_size=None, blur_to=None):
        self.ds, self.indices, self.proc = hf_dataset, global_indices, processor
        self.input_size, self.blur_to = input_size, blur_to

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image = self.ds[int(self.indices[i])]['image']
        if self.blur_to is not None:
            image = image.resize((self.blur_to, self.blur_to)).resize((self.input_size, self.input_size))
        if self.input_size is not None:
            processed = self.proc(images=image, size={'height': self.input_size, 'width': self.input_size},
                                   do_center_crop=False, return_tensors='pt')
        else:
            processed = self.proc(images=image, return_tensors='pt')
        return processed['pixel_values'].squeeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true', help='tiny local sanity check: 1 epoch, small budgets')
    p.add_argument('--max-train-frames', type=int, default=200_000)
    # 20, not 30: with --lr-schedule cosine the LR floors at --lr-decay-epochs and runs have
    # consistently early-stopped by epoch ~14. 30 mostly bought TIMEOUTs (the single largest
    # category of wasted GPU time in this project so far: 68 of 138 GPU-hours).
    p.add_argument('--n-epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--encode-batch-size', type=int, default=256)
    p.add_argument('--context-k', type=int, default=None,
                    help='override best_cfg context_k (temporal half-window; T = 2k+1 frames). '
                         'NEVER swept before -- grid_search_frame.py hardcoded it to 2 and only '
                         'widened via stride, and every wider setting was worse (stride 2/3/4 -> '
                         '0.238/0.236/0.245 vs 0.288 at stride 1). Measured bout durations say the '
                         'window is too WIDE, not too narrow: at 5 fps the k=2 window spans 1.0 s '
                         'while the median bout is 3 frames (nt, 0.6 s) / 2 frames (nn, 0.4 s). '
                         'Smaller k is also cheaper -- cache and per-batch bytes scale with T.')
    p.add_argument('--val-monitor-size', type=int, default=12_500,
                    help='size of the per-epoch val monitor subsample. Uniform/prevalence-preserving '
                         '(see note at the sampling site) -- NOT rebalanced. Costs no extra encoding '
                         '(full val is encoded anyway for the final number), only forward passes. '
                         'Reduced from 50,000: that exceeded the 46,560-sample TRAINING set, so '
                         'over half of every epoch went on monitoring. 12.5k keeps ~122 nt / ~216 '
                         'nn positives (~9%% relative AP noise), enough to rank checkpoints.')
    p.add_argument('--neg-ratio', type=int, default=10,
                    help='negatives per positive, resampled fresh each epoch from the cached pool. '
                         '1 (pure balance) measurably hurt full-val AP despite the healthiest-looking '
                         'training curve of the day (0.2439, worse than the un-fixed 200k-frame '
                         'baseline) -- likely because training on a 1:1 mix is too easy relative to '
                         'the real, far-more-imbalanced ranking task, so the model never had to learn '
                         'to discriminate against the bulk of easy negatives that dominate real val. '
                         'No pos_weight regardless of this value -- resampling already handles the '
                         'imbalance without needing to also reweight the loss.')
    p.add_argument('--num-workers', type=int, default=8,
                    help='ENCODE workers. Reads are NFS-latency-bound (~100ms/frame), so this '
                         'should exceed the CPU count: 48 measured 4.8x over 16.')
    p.add_argument('--gather-workers', type=int, default=6,
                    help='TRAINING-gather workers -- deliberately SMALL and separate from '
                         '--num-workers. The gather reads from the in-RAM token cache, so it is '
                         'not latency-bound, and DataLoader holds workers*prefetch_factor batches '
                         'in flight at once. Reusing the encode value here OOMed a run: 48 workers '
                         'x prefetch 4 x 604MB/batch = 116GB of prefetch on top of a 163GiB cache, '
                         'against a 300G allocation.')
    p.add_argument('--tag', type=str, default='', help='suffix for RESULTS_DIR, e.g. "reg_a" -> patchgrid256_dinov2_reg_a')
    p.add_argument('--cross-attn-dim', type=int, default=192, help='0 = no bottleneck (full emb_dim)')
    p.add_argument('--patch-pool-dim', type=int, default=0, help='0 = no bottleneck (full emb_dim); '
                    'bottlenecks patch_pool itself, the largest capacity block, untouched by every '
                    'prior capacity-reduction experiment (only cross_attn_dim ever touched capacity before)')
    # Embedding-space augmentation knobs. Kept (plumbing is harmless and they default off) but note
    # a 4-way ablation found NO effect on full-val AP -- do not expect these to help.
    p.add_argument('--patch-dropout', type=float, default=0.0)
    p.add_argument('--patch-noise-std', type=float, default=0.0, help='fraction of the batch patch-token std')
    p.add_argument('--frame-dropout', type=float, default=0.0)
    p.add_argument('--use-layernorm', action='store_true',
                    help='LayerNorm around each attention module + inside the head. Tested once (0.2489 '
                         'vs 0.2875) but WITHOUT retuning lr, so that test was not a fair comparison.')
    p.add_argument('--loss', choices=['bce', 'focal'], default='bce')
    p.add_argument('--focal-gamma', type=float, default=2.0,
                    help='NOTE: this is gamma-only focal, with no alpha class-balancing term. That is '
                         'NOT the RetinaNet recipe (Lin et al. 2017 uses alpha AND gamma over the full '
                         'negative set), so the poor result from the one focal run here is not evidence '
                         'against focal loss as such.')
    p.add_argument('--input-size', type=int, default=None,
                    help='resize frames to this size instead of DINOv2\'s default 224x224 center crop '
                         '(must be a multiple of patch_size=14). Patch count = (input_size/14)^2, and '
                         'both encode time and cache RAM scale with it. A blur control (448px rendered '
                         'from an upsampled 224px image, 0.3492, vs genuine 448px, 0.3380) showed the '
                         'gain from raising this comes from having MORE PATCH TOKENS, not from finer '
                         'visual detail -- i.e. it is really about spatial addressing granularity in '
                         'patch_pool, which suggests better pooling may buy the same gain far cheaper.')
    p.add_argument('--lr-schedule', choices=['none', 'cosine'], default='none',
                    help='cosine: decays lr to 1%% of its initial value over --lr-decay-epochs, then '
                         'HOLDS at that floor for the rest of training (a plain torch CosineAnnealingLR '
                         'is periodic -- it would rise back toward the initial lr after --lr-decay-epochs '
                         'if left running past T_max, which is not what we want here, so this is a custom '
                         'decay-then-hold LambdaLR instead). No run in this project has ever used any LR '
                         'schedule -- suspected cause of val_loss bottoming out at epoch 1-2 and never '
                         'improving again (oversized, undecayed gradient steps from the large pos_weight '
                         'never get a chance to settle).')
    p.add_argument('--lr-decay-epochs', type=int, default=None,
                    help='epochs over which lr decays to its floor (default: --n-epochs). A first cosine-'
                         'schedule test (T_max=n_epochs=30) showed val_loss diverging starting ~epoch 5, '
                         'while lr had only decayed to 60%% of its initial value by epoch 13 -- decay too '
                         'slow to catch the divergence. Set this much smaller (e.g. 6-8) to decay fast '
                         'right where the divergence starts.')
    args = p.parse_args()
    PATCH_SIZE = 14
    n_patches_full = N_PATCHES_FULL if args.input_size is None else (args.input_size // PATCH_SIZE) ** 2
    cross_attn_dim = args.cross_attn_dim or None
    patch_pool_dim = args.patch_pool_dim or None

    if args.smoke:
        args.max_train_frames, args.n_epochs, args.patience, args.batch_size = 30_000, 2, 2, 64

    tag_suffix = '_smoke' if args.smoke else (f'_{args.tag}' if args.tag else '')
    RESULTS_DIR = gsf.FRAME_DIR / f'patchgrid256_dinov2{tag_suffix}'
    best_cfg = get_head_cfg()
    # stride stays at best_cfg's value (1): a 2/3/4 sweep was strictly worse (0.2376/0.2356/0.2450
    # vs 0.2875 at stride 1), so widening the temporal window is settled and no longer a knob.
    stride_val = best_cfg['stride']
    context_k = args.context_k if args.context_k is not None else best_cfg['context_k']
    print(f'Reusing confirmed-best patchgrid cfg: {best_cfg}', flush=True)
    print(f'max_train_frames={args.max_train_frames}  n_epochs={args.n_epochs}  patience={args.patience}', flush=True)
    print(f'reg: cross_attn_dim={cross_attn_dim}  patch_dropout={args.patch_dropout}  '
          f'patch_noise_std={args.patch_noise_std}  frame_dropout={args.frame_dropout}', flush=True)
    print(f'context_k={context_k} (T={2*context_k+1} frames = {(2*context_k+1)/5:.1f}s @5fps)  stride={stride_val}  input_size={args.input_size}  n_patches_full={n_patches_full}', flush=True)

    pair_labels_path = gsf.build_pair_labels(DATA_DIR, DATASET_DIR, overwrite=False)
    annotations_csv = DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    obs_to_pool = gsf.load_obs_to_pool_map(DATA_DIR)
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    pools = sorted({obs_to_pool[o] for o in all_obs})
    val_pool_set = get_fixed_val_pools(pools)
    train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]
    print(f'Split: {len(train_obs)} train obs / {len(val_obs)} val obs', flush=True)

    print('Building sample index (placeholder embeddings, cheap)...', flush=True)
    train_meta = FrameBatchData(
        str(annotations_csv), str(pair_labels_path), train_obs, context_k, 1,
        dummy_loader(1, 1), n_patches=1, stride=stride_val, max_frames=args.max_train_frames, seed=SEED,
    )
    val_meta = FrameBatchData(
        str(annotations_csv), str(pair_labels_path), val_obs, context_k, 1,
        dummy_loader(1, 1), n_patches=1, stride=stride_val,
    )
    del train_meta.flat, val_meta.flat  # never used — only gi/offsets_grid/pad_mask/labels matter

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {MODEL_ID} on {dev}...', flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    encoder = AutoModel.from_pretrained(MODEL_ID).to(dev)
    encoder.eval()
    encoder.requires_grad_(False)

    print('Loading full-frame HF dataset (raw JPEGs)...', flush=True)
    hf_dataset = load_dataset(subject='mice', version='v1', dataset_root=str(DATASET_DIR), frame_type='full')

    # Undersampling (neg_ratio negatives per positive, resampled fresh every epoch from the
    # cached pool) with NO pos_weight -- pos_weight and undersampling both correct the same
    # imbalance, and stacking both (as this script used to) meant partially undoing one fix
    # with the other. But going all the way to 1:1 (tested, worse) throws out too much: it
    # makes training too easy relative to the real, far-more-imbalanced ranking task, so the
    # model never learns to discriminate against the bulk of easy negatives real val is full
    # of. neg_ratio stays a real (moderate, not extreme) knob; only pos_weight is gone.
    labels_all = train_meta.labels
    pos_idx = np.where(labels_all.sum(axis=1) > 0)[0]
    neg_idx = np.where(labels_all.sum(axis=1) == 0)[0]
    n_pos_frames = max(len(pos_idx), 1)
    n_neg_draw = min(len(neg_idx), args.neg_ratio * n_pos_frames)
    saturated = n_neg_draw >= len(neg_idx)
    print(f'{args.neg_ratio}:1 sampling (no pos_weight): {n_pos_frames:,} positives, requesting '
          f'{args.neg_ratio * n_pos_frames:,} negatives, {len(neg_idx):,} available in the cached '
          f'candidate pool -> {"SATURATED (identical every epoch)" if saturated else f"resampling a fresh {n_neg_draw:,} of {len(neg_idx):,} negatives every epoch"}',
          flush=True)

    # PREVALENCE-PRESERVING monitor -- a uniform random subsample of val, NOT a rebalanced one.
    # Average Precision is prevalence-dependent by construction (unlike ROC-AUC; Davis & Goadrich
    # 2006), so a rebalanced monitor measures a *different quantity* than the reported full-val AP
    # and is a biased proxy for it. Every earlier version of this script rebalanced the monitor
    # (2:1, then neg_ratio:1), and it demonstrably lied: the balanced-1:1 run reached monitor AP
    # 0.726 -- the healthiest curve of the whole investigation -- while its actual full-val AP was
    # 0.0812, the worst result recorded. Checkpoint selection and early stopping were optimizing
    # the wrong objective. A uniform subsample keeps val's natural ~1.1% nt / ~3.2% nn prevalence,
    # so monitor AP now tracks the number we actually report.
    v_rng = np.random.default_rng(SEED)
    n_monitor = min(len(val_meta), args.val_monitor_size)
    val_keep = np.sort(v_rng.choice(len(val_meta), size=n_monitor, replace=False))
    v_prev = val_meta.labels[val_keep]
    print(f'val monitor (per-epoch only; final number uses FULL val): {len(val_meta):,} -> '
          f'{len(val_keep):,} uniform/prevalence-preserving '
          f'(nt {100*v_prev[:,0].mean():.2f}%, nn {100*v_prev[:,1].mean():.2f}% '
          f'vs full-val nt {100*val_meta.labels[:,0].mean():.2f}%, nn {100*val_meta.labels[:,1].mean():.2f}%)',
          flush=True)

    def needed_raw_indices(meta, sample_idx):
        gi = meta.gi[sample_idx]
        offsets = meta.offsets_grid
        abs_idx = gi[:, None] + offsets[None, :]
        mask = meta.pad_mask[sample_idx]
        return np.unique(abs_idx[~mask])

    print('Determining unique raw frames needed (full train candidate pool + val subsample + full val)...', flush=True)
    # cache the WHOLE bounded candidate pool (pos_idx + all of neg_idx), not just one epoch's
    # draw -- lets each epoch resample a fresh subset of negatives from what's already cached,
    # with zero extra encoding cost, instead of training on the exact same fixed negatives
    # every single epoch for the whole run.
    need_train = needed_raw_indices(train_meta, np.concatenate([pos_idx, neg_idx]))
    need_val_sub = needed_raw_indices(val_meta, val_keep)
    need_val_full = needed_raw_indices(val_meta, np.arange(len(val_meta)))
    all_needed = np.unique(np.concatenate([need_train, need_val_sub, need_val_full]))
    print(f'  {len(need_train):,} unique train frames, {len(need_val_full):,} unique val frames '
          f'(full val incl. subsample) -> {len(all_needed):,} total to encode once', flush=True)

    # Fail fast on an impossible cache instead of OOMing ~45 min into the encode (which cost ~4
    # GPU-hours across two runs). The footprint is exactly predictable, so check it up front.
    cache_gib = len(all_needed) * n_patches_full * EMB_DIM * 2 / 1024**3
    slurm_mem_mb = os.environ.get('SLURM_MEM_PER_NODE')
    limit_gib = int(slurm_mem_mb) / 1024 if slurm_mem_mb else None
    print(f'Token cache footprint: {cache_gib:.0f} GiB '
          f'({len(all_needed):,} frames x {n_patches_full} patches x {EMB_DIM} dim x fp16)'
          + (f'  |  job allocation: {limit_gib:.0f} GiB' if limit_gib else '  |  allocation unknown'), flush=True)
    # Thresholds calibrated on this project's own observed boundary: 703 GiB in an 850 GiB job
    # (ratio 0.83) ran fine; 820 GiB in an 880 GiB job (ratio 0.93) died OOM ~46 min into encoding.
    if limit_gib is not None and cache_gib > 0.90 * limit_gib:
        raise SystemExit(
            f'ABORT before encoding: cache needs {cache_gib:.0f} GiB of a {limit_gib:.0f} GiB allocation '
            f'({100*cache_gib/limit_gib:.0f}%). A run at 93% OOMed after ~46 min of wasted encoding. '
            f'Fix by raising --mem, lowering --max-train-frames, or lowering --input-size '
            f'(footprint scales as (input_size/14)^2, so 448->224 cuts it 4x).')
    if limit_gib is not None and cache_gib > 0.75 * limit_gib:
        print(f'  WARNING: cache is {100*cache_gib/limit_gib:.0f}% of the allocation -- little headroom.', flush=True)

    print(f'Encoding {len(all_needed):,} frames once via a parallel DataLoader '
          f'({args.num_workers} workers, batch {args.encode_batch_size})...', flush=True)
    t_encode0 = time.time()
    loader = DataLoader(
        _ImageDataset(hf_dataset, all_needed, processor, input_size=args.input_size), batch_size=args.encode_batch_size,
        num_workers=args.num_workers, pin_memory=(dev.type == 'cuda'), shuffle=False,
        prefetch_factor=4 if args.num_workers > 0 else None, persistent_workers=args.num_workers > 0,
    )
    cache = torch.empty((len(all_needed), n_patches_full, EMB_DIM), dtype=torch.float16)
    cursor = 0
    last_report = (0, 0.0)
    with torch.inference_mode():
        for pixel_values in loader:
            pixel_values = pixel_values.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
                out = encoder(pixel_values=pixel_values)
            tokens = out.last_hidden_state[:, 1:].half().cpu()  # dinov2-base: CLS only, no register tokens
            cache[cursor:cursor + tokens.shape[0]] = tokens
            cursor += tokens.shape[0]
            if cursor % (args.encode_batch_size * 20) == 0 or cursor == len(all_needed):
                elapsed = time.time() - t_encode0
                # Report a WINDOWED rate alongside the cumulative one. Cumulative average hides
                # OS page-cache decay: a run whose first ~66k frames were still cached from a
                # previous job read 877 frames/s cumulative while its true cold rate had already
                # fallen to ~230, which nearly caused a non-existent speedup to be reported.
                win_n, win_dt = cursor - last_report[0], elapsed - last_report[1]
                inst = win_n / win_dt if win_dt > 0 else float('nan')
                last_report = (cursor, elapsed)
                print(f'  encoded {cursor:,}/{len(all_needed):,} '
                      f'({inst:.1f} frames/s now, {cursor/elapsed:.1f} cumulative)', flush=True)
    print(f'Encoding done in {(time.time()-t_encode0)/60:.1f} min '
          f'({len(all_needed)/(time.time()-t_encode0):.1f} frames/s average)', flush=True)
    del encoder, processor, loader
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

    def build_batch_tensor(meta, sample_idx):
        gi = meta.gi[sample_idx]
        offsets = meta.offsets_grid
        abs_idx = gi[:, None] + offsets[None, :]
        mask = meta.pad_mask[sample_idx]
        B, T = abs_idx.shape
        valid = ~mask
        flat_idx = abs_idx[valid]
        # all_needed is sorted (np.unique), and cache[i] <-> all_needed[i] by construction (the
        # encode loader iterated it in order) -- searchsorted is an exact, vectorized replacement
        # for what used to be a per-element Python dict lookup on every single batch.
        positions = np.searchsorted(all_needed, flat_idx)
        gathered = cache[positions].to(dev, non_blocking=True)
        ctx = torch.zeros((B, T, n_patches_full, EMB_DIM), dtype=torch.float16, device=dev)
        ctx[torch.from_numpy(valid)] = gathered
        offsets_t = torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()).to(dev)
        labels_t = torch.from_numpy(meta.labels[sample_idx]).to(dev)
        mask_t = torch.from_numpy(mask).to(dev)
        return ctx, offsets_t, labels_t, mask_t

    class _GatherDataset(Dataset):
        """Builds one whole training batch per __getitem__, on a worker process.

        Profiling (scripts/mice_behavior/bench_batch_pipeline.py, L40S) showed the training
        loop spent 71% of each batch inside a blocking single-threaded `cache[positions]`
        gather in the MAIN process -- 1350 ms of gather against 40-105 ms of forward+backward,
        i.e. the GPU idled ~90% of training. The encode phase already used a parallel
        DataLoader; the training loop had none at all.

        Workers inherit `cache` through fork's copy-on-write, so the (hundreds of GiB) tensor
        is NOT duplicated per worker -- they only read it. Returns CPU fp16 tensors; pinning
        and the host->device copy are handled by the DataLoader (pin_memory=True) so the
        transfer overlaps with compute instead of serialising after it.
        """

        def __init__(self, meta, batches):
            self.meta, self.batches = meta, batches

        def __len__(self):
            return len(self.batches)

        def __getitem__(self, i):
            sample_idx = self.batches[i]
            gi = self.meta.gi[sample_idx]
            offsets = self.meta.offsets_grid
            abs_idx = gi[:, None] + offsets[None, :]
            mask = self.meta.pad_mask[sample_idx]
            B, T = abs_idx.shape
            valid = ~mask
            positions = np.searchsorted(all_needed, abs_idx[valid])
            ctx = torch.zeros((B, T, n_patches_full, EMB_DIM), dtype=torch.float16)
            ctx[torch.from_numpy(valid)] = cache[positions]
            return (ctx,
                    torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()),
                    torch.from_numpy(self.meta.labels[sample_idx]),
                    torch.from_numpy(mask))

    def make_batch_loader(meta, sample_order, batch_size, workers):
        batches = [sample_order[i:i + batch_size] for i in range(0, len(sample_order), batch_size)]
        in_flight = max(workers, 1) * 2
        est_gib = in_flight * (args.batch_size * (2 * context_k + 1) * n_patches_full * EMB_DIM * 2) / 1024**3
        if not hasattr(make_batch_loader, '_logged'):
            print(f'  train gather: {workers} workers x prefetch 2 = {in_flight} batches in flight '
                  f'~= {est_gib:.1f} GiB', flush=True)
            make_batch_loader._logged = True
        return DataLoader(
            _GatherDataset(meta, batches), batch_size=None, shuffle=False,
            num_workers=workers, pin_memory=(dev.type == 'cuda'),
            prefetch_factor=2 if workers > 0 else None, persistent_workers=False,
        )

    model = MouseFrameClassifier(
        emb_dim=EMB_DIM, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'],
        use_patch_grid=True, dropout=best_cfg['dropout'], cross_attn_dim=cross_attn_dim, patch_pool_dim=patch_pool_dim,
        patch_dropout=args.patch_dropout, patch_noise_std=args.patch_noise_std, frame_dropout=args.frame_dropout,
        use_layernorm=args.use_layernorm,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}  (cross_attn_dim={cross_attn_dim}, patch_pool_dim={patch_pool_dim}, '
          f'patch_dropout={args.patch_dropout}, patch_noise_std={args.patch_noise_std}, '
          f'frame_dropout={args.frame_dropout}, use_layernorm={args.use_layernorm})', flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_cfg['lr'], weight_decay=best_cfg['weight_decay'])
    lr_scheduler = None
    if args.lr_schedule == 'cosine':
        decay_epochs = args.lr_decay_epochs or args.n_epochs
        eta_min_ratio = 0.01

        def _decay_then_hold(epoch):
            # epoch is 0-indexed by LambdaLR's internal counter; decays smoothly from 1.0 to
            # eta_min_ratio over decay_epochs, then HOLDS at eta_min_ratio forever after --
            # unlike plain CosineAnnealingLR, which is periodic and would rise back up.
            if epoch >= decay_epochs:
                return eta_min_ratio
            progress = epoch / decay_epochs
            return eta_min_ratio + 0.5 * (1 - eta_min_ratio) * (1 + math.cos(math.pi * progress))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_decay_then_hold)
        print(f'lr_schedule: cosine decay-then-hold, {best_cfg["lr"]} -> {best_cfg["lr"] * eta_min_ratio} '
              f'over {decay_epochs} epochs, then held flat through epoch {args.n_epochs}', flush=True)

    # No pos_weight: the epoch's data is already 1:1 balanced by construction (see above), so
    # there's no residual class imbalance left to reweight for.
    if args.loss == 'focal':
        print(f'loss: focal (gamma={args.focal_gamma}), no class reweighting (already 1:1 balanced)', flush=True)
        no_reweight = torch.ones(2, device=dev)
        criterion = lambda logits, lbl: focal_loss_with_pos_weight(logits, lbl, no_reweight, gamma=args.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_ap = -1.0
    epochs_since_best = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'macro_ap': []}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shuffle_rng = np.random.default_rng(SEED)
    # GradScaler is required under fp16 autocast, not optional -- without it, gradients can
    # underflow to exact zero (float16's limited dynamic range), which is exactly what
    # happened on the first run of this script: train_loss stuck flat at ~1.01-1.02 for 16
    # straight epochs and macro_ap was bit-for-bit identical epoch 10 through 16 -- the
    # unmistakable signature of weights that had silently stopped updating. train.py's own
    # train()/train_frame() always pair autocast with a GradScaler for exactly this reason.
    amp_enabled = dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        if saturated:
            epoch_idx = np.concatenate([pos_idx, neg_idx])  # nothing to resample -- using every negative already
        else:
            neg_sample = shuffle_rng.choice(neg_idx, size=n_neg_draw, replace=False)  # fresh draw every epoch
            epoch_idx = np.concatenate([pos_idx, neg_sample])
        shuffle_rng.shuffle(epoch_idx)

        total_loss, n_seen = 0.0, 0
        grad_norms = []
        last_group_norms = {}
        t0 = time.time()
        for ctx, offs, lbl, mask in make_batch_loader(train_meta, epoch_idx, args.batch_size, args.gather_workers):
            ctx = ctx.to(dev, non_blocking=True)
            offs, lbl, mask = (offs.to(dev, non_blocking=True), lbl.to(dev, non_blocking=True),
                               mask.to(dev, non_blocking=True))
            batch_idx = lbl  # only its length is used below
            optimizer.zero_grad()
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp_enabled):
                # no .float(): autocast casts to fp16 for the matmuls anyway, so upcasting here
                # only materialised a second, 2x-larger fp32 copy of a multi-GiB tensor.
                logits = model(ctx, offsets=offs, key_padding_mask=mask)
                loss = criterion(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            grad_norms.append(pre_clip_norm.item())
            last_group_norms = named_group_grad_norms(model)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(batch_idx)
            n_seen += len(batch_idx)
        train_loss = total_loss / n_seen
        grad_norms = np.array(grad_norms)
        frac_clipped = float((grad_norms > 0.5).mean())
        print(f'  grad_norm (pre-clip): min={grad_norms.min():.3f} mean={grad_norms.mean():.3f} '
              f'max={grad_norms.max():.3f}  frac_clipped(>0.5)={frac_clipped:.2f}  '
              f'per-module(last batch)={ {k: round(v, 3) for k, v in last_group_norms.items()} }', flush=True)
        if lr_scheduler is not None:
            lr_scheduler.step()

        model.eval()
        all_probs, all_labels, val_loss_sum, v_n = [], [], 0.0, 0
        with torch.no_grad():
            for b0 in range(0, len(val_keep), args.batch_size):
                batch_idx = val_keep[b0:b0 + args.batch_size]
                ctx, offs, lbl, mask = build_batch_tensor(val_meta, batch_idx)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
                    logits = model(ctx.float(), offsets=offs, key_padding_mask=mask)
                    batch_loss = criterion(logits, lbl)
                all_probs.append(torch.sigmoid(logits).float().cpu())
                all_labels.append(lbl.cpu())
                val_loss_sum += batch_loss.item() * len(batch_idx)
                v_n += len(batch_idx)
        probs = torch.cat(all_probs).numpy()
        labels_np = torch.cat(all_labels).numpy()
        aps = [average_precision_score(labels_np[:, i], probs[:, i]) for i in range(2)]
        macro_ap = float(np.mean(aps))
        val_loss = val_loss_sum / v_n

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['macro_ap'].append(macro_ap)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'epoch {epoch:3d}/{args.n_epochs}  loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
              f'macro_ap={macro_ap:.4f}  nt={aps[0]:.3f} nn={aps[1]:.3f}  lr={cur_lr:.2e}  ({time.time()-t0:.1f}s)', flush=True)

        if macro_ap > best_ap:
            best_ap = macro_ap
            epochs_since_best = 0
            torch.save(model.state_dict(), RESULTS_DIR / 'best_model.pt')
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f'early stopping: no improvement for {args.patience} epochs', flush=True)
                break

    model.load_state_dict(torch.load(RESULTS_DIR / 'best_model.pt', map_location=dev, weights_only=True))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for b0 in range(0, len(val_meta), args.batch_size):
            batch_idx = np.arange(b0, min(b0 + args.batch_size, len(val_meta)))
            ctx, offs, lbl, mask = build_batch_tensor(val_meta, batch_idx)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
                logits = model(ctx.float(), offsets=offs, key_padding_mask=mask)
            all_probs.append(torch.sigmoid(logits).float().cpu())
            all_labels.append(lbl.cpu())
    probs = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    per_label = {name: average_precision_score(labels_np[:, i], probs[:, i]) for i, name in enumerate(['nt', 'nn'])}
    final_score = float(np.mean(list(per_label.values())))
    print(f'FULL-VAL macro AP: {final_score:.4f}  {per_label}', flush=True)

    with open(RESULTS_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(RESULTS_DIR / 'config.json', 'w') as f:
        json.dump({'cfg': best_cfg, 'val_pools': sorted(val_pool_set), 'n_epochs_cap': args.n_epochs,
                   'n_patches': n_patches_full, 'max_train_frames': args.max_train_frames,
                   'neg_pool_saturated': bool(saturated), 'balanced_1to1_sampling': True,
                   'cross_attn_dim': cross_attn_dim, 'patch_dropout': args.patch_dropout,
                   'patch_noise_std': args.patch_noise_std, 'frame_dropout': args.frame_dropout,
                   'use_layernorm': args.use_layernorm, 'loss': args.loss, 'focal_gamma': args.focal_gamma,
                   'stride': stride_val, 'context_k': context_k, 'input_size': args.input_size, 'neg_ratio': args.neg_ratio,
                   'val_monitor_size': args.val_monitor_size, 'val_monitor_prevalence_preserving': True,
                   'lr_schedule': args.lr_schedule, 'patch_pool_dim': patch_pool_dim,
                   'n_params': n_params,
                   'best_ap': final_score, 'best_per_label': per_label}, f, indent=2)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
