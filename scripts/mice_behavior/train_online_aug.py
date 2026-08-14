"""Online per-batch DINOv2 encoding from a JPEG-bytes RAM cache, with D4 augmentation.

Why this exists
---------------
train_patchgrid_online.py caches DINOv2 *tokens*, which is both the largest possible
representation and the one thing that makes pixel-space augmentation impossible: tokens are
computed from one fixed rendering of each frame, so a rotated image needs different tokens.
That restricted us to embedding-space augmentation (patch dropout / noise / frame dropout),
and a 4-way ablation of those found no effect.

Caching the JPEG BYTES instead and encoding per batch inverts every part of that trade:

    cache contents          size (444k frames)     augmentation?
    DINOv2 tokens @448px    650 GiB                no
    DINOv2 tokens @224px    163 GiB                no
    raw JPEG bytes           17 GiB (measured)     YES

Costs ~2-10x more encoder forward passes (each frame is re-encoded every epoch it is drawn
rather than once), but the encoder was never the bottleneck: frame reads are NFS-latency-bound
at ~100 ms/frame, and this cache pays that cost EXACTLY ONCE, after which every epoch reads
from RAM. It also removes the 300-700 GiB allocations that caused repeated OOM aborts.

Why D4 and not arbitrary rotation
---------------------------------
The cage is filmed top-down, so the dihedral group D4 (4 rotations x optional flip = 8
variants) is a genuine symmetry of the data: "nose-to-tail sniffing" does not depend on the
camera's orientation. Restricting to multiples of 90 degrees keeps the transform EXACT --
no interpolation blur, no black corners, no resampling artefacts, unlike arbitrary angles.
With only ~1,186 nt and ~3,696 nn independent bouts in train, this 8x label-preserving
expansion targets the actual constraint, which is effective sample size.

The SAME transform is applied to every frame in a sample's context window. Applying
independent transforms per frame would destroy the temporal relationship, which the
context_k sweep showed is what nose-tail detection depends on (nt full-val AP 0.0449 at
context_k=0 vs 0.1411 at context_k=2, a 3.1x gap; nn is nearly context-free by comparison).

Usage:
    python scripts/mice_behavior/train_online_aug.py --context-k 2 --augment d4
"""
import argparse
import io
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
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import get_fixed_val_pools
from src.mice_behavior.metrics import ap_report, format_ap_report, rate_report, format_rate_report
from src.mice_behavior.viz import plot_confusion_examples
from src.mice_behavior.head_cfg import get_head_cfg
from train_patchgrid_online import dummy_loader

MODEL_ID = 'facebook/dinov2-base'
EMB_DIM, PATCH_SIZE = 768, 14
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DATASET_ROOT = Path('dataset')


class _BytesReader(Dataset):
    """Reads raw JPEG bytes off NFS. Used once, with many workers: reads are latency-bound
    (~100 ms/frame; the decode itself is only ~2-3 ms), so workers sit blocked on I/O and
    should far exceed the CPU count -- 48 workers measured 4.8x over 16."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with open(DATASET_ROOT / self.paths[i], 'rb') as f:
            return i, np.frombuffer(f.read(), dtype=np.uint8)


def d4_transform(img: Image.Image, op: int) -> Image.Image:
    """One of the 8 dihedral-group symmetries. Exact: 90-degree rotations and flips are
    lossless pixel permutations, so no interpolation or padding is introduced."""
    if op & 4:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    r = op & 3
    if r == 1:
        img = img.transpose(Image.ROTATE_90)
    elif r == 2:
        img = img.transpose(Image.ROTATE_180)
    elif r == 3:
        img = img.transpose(Image.ROTATE_270)
    return img


class _SampleDataset(Dataset):
    """Decodes + augments + normalises one whole batch of samples per __getitem__.

    jpeg_cache is inherited by workers through fork copy-on-write, so the ~17 GiB is shared,
    not duplicated per worker.
    """

    def __init__(self, meta, batches, jpeg_cache, input_size, augment, seed,
                 photo_brightness=(0.80, 1.25), photo_contrast=(0.80, 1.25),
                 photo_gamma=(0.83, 1.20), fixed_op=None):
        # fixed_op pins the D4 transform instead of drawing it, which is what test-time
        # augmentation needs: the same deterministic rendering applied to every sample, then
        # averaged over ops. None keeps the training behaviour (draw per sample).
        self.fixed_op = fixed_op
        self.photo_brightness = photo_brightness
        self.photo_contrast = photo_contrast
        self.photo_gamma = photo_gamma
        # jpeg_cache is keyed by GLOBAL frame index and is populated lazily, epoch by epoch.
        # Workers fork per-epoch (a new DataLoader is built each epoch for the resampled
        # negatives), so each fork inherits whatever the main process has cached so far.
        self.meta, self.batches, self.cache = meta, batches, jpeg_cache
        self.input_size = input_size
        self.augment, self.seed = augment, seed

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, bi):
        sample_idx = self.batches[bi]
        offsets = self.meta.offsets_grid
        abs_idx = self.meta.gi[sample_idx][:, None] + offsets[None, :]
        mask = self.meta.pad_mask[sample_idx]
        B, T = abs_idx.shape
        S = self.input_size
        out = torch.zeros((B, T, 3, S, S), dtype=torch.float32)
        rng = np.random.default_rng((self.seed, bi))
        use_d4 = self.augment in ('d4', 'd4_photo')
        use_photo = self.augment == 'd4_photo'
        for b in range(B):
            # one transform per SAMPLE, shared by all T context frames -- per-frame transforms
            # would break the temporal relationship nose-tail detection relies on.
            op = (self.fixed_op if self.fixed_op is not None
                  else (int(rng.integers(0, 8)) if use_d4 else 0))
            # Photometric jitter, drawn once per sample for the same reason: independent
            # per-frame jitter would inject spurious frame-to-frame change into a window the
            # model reads temporally. D4 gives only 8 exact renderings of each positive, and
            # with every positive drawn every epoch that is the binding limit on effective
            # sample size -- this makes the rendering continuous instead. It also attacks
            # video-level appearance nuisance (lighting/exposure differs per recording), the
            # plausible reason frame ranking is much stronger than between-video rate ranking.
            # Frames are pure grayscale (R=G=B exactly, verified), so hue/saturation jitter
            # would be a no-op; brightness/contrast/gamma are the only meaningful axes.
            if use_photo:
                bright = float(rng.uniform(*self.photo_brightness))
                contrast = float(rng.uniform(*self.photo_contrast))
                gamma = float(rng.uniform(*self.photo_gamma))
            for t in range(T):
                if mask[b, t]:
                    continue
                buf = self.cache[int(abs_idx[b, t])]
                with Image.open(io.BytesIO(buf.tobytes())) as im:
                    im = im.convert('RGB')
                    if op:
                        im = d4_transform(im, op)
                    im = im.resize((S, S), Image.BILINEAR)
                    arr = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())
                x = arr.permute(2, 0, 1).float() / 255.0
                if use_photo:
                    x = x.pow(gamma).mul(bright)
                    m = x.mean()
                    x = ((x - m) * contrast + m).clamp_(0.0, 1.0)
                out[b, t] = (x - IMAGENET_MEAN) / IMAGENET_STD
        return (out,
                torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()),
                torch.from_numpy(self.meta.labels[sample_idx]),
                torch.from_numpy(mask))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--context-k', type=int, default=2)
    p.add_argument('--stride', type=int, default=1,
                    help='context positions sit at +-context_k*stride in steps of stride, so the '
                         'window widens WITHOUT adding positions: still 2k+1 frames encoded, same '
                         'token and attention cost. At 5 fps, k=2 spans only +-0.4 s at stride 1 '
                         'vs +-0.8 s at stride 2. nt is the context-dependent label (AP monotone '
                         'in k: 0.105/0.154/0.197 at 504px) and the one furthest from target, so '
                         'reach is worth probing separately from density.')
    p.add_argument('--neg-ratio', type=int, default=1)
    p.add_argument('--max-train-frames', type=int, default=300_000,
                    help='NOT a cap on training data in the way the name suggests, and easy to '
                         'misread: FrameBatchData keeps EVERY positive frame (+ its +-reach '
                         'context) unconditionally, BEFORE this budget is consulted, then tops up '
                         'with random NEGATIVE frames until the budget is reached. So all 23,280 '
                         'positive anchors (~4.9k bouts) are already in at the 300k default, and '
                         'raising this buys negative DIVERSITY only -- never more positives, never '
                         'more bouts. Epoch cost is unaffected too: with neg_ratio=1 an epoch is '
                         'n_pos positives + n_pos sampled negatives regardless of pool size. '
                         'The original rationale (embedding RAM: patch-grid tokens are 16x CLS) is '
                         'void on this online path, which passes a dummy 1-dim loader and deletes '
                         '.flat immediately -- no embeddings are ever stored. What still binds is '
                         'the JPEG-bytes cache at ~45 KB/frame: 300k -> ~19 GiB, 900k -> ~45 GiB, '
                         'unbounded (2.59M) -> ~114 GiB, plus the one-time NFS read of each frame.')
    p.add_argument('--input-size', type=int, default=224)
    p.add_argument('--augment', choices=['none', 'd4', 'd4_photo'], default='d4',
                    help="'d4_photo' adds per-sample brightness/contrast/gamma jitter on top of "
                         'the 8 exact D4 renderings. Every positive is drawn every epoch, so with '
                         'd4 alone each one is seen 40 times across only 8 distinct renderings; '
                         'the photometric draw is continuous, so no two are identical.')
    p.add_argument('--photo-strength', type=float, default=1.0,
                    help='scales the d4_photo jitter ranges about 1.0 (0.5 = half as strong, '
                         '2.0 = double). 1.0 = brightness/contrast in [0.80, 1.25], gamma in '
                         '[0.83, 1.20].')
    p.add_argument('--n-epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--read-workers', type=int, default=32, help='NFS reads: latency-bound, use many')
    p.add_argument('--decode-workers', type=int, default=16, help='decode+augment: CPU-bound, ~= n_cpus')
    p.add_argument('--val-monitor-size', type=int, default=12_500,
                    help='per-epoch monitor size. Was 25k, but the monitor pass ran 25k val '
                         'samples against a 46,560-sample training set -- a third of each epoch '
                         'spent monitoring rather than learning (visible as a 99%%->70%% GPU '
                         'utilisation dip). 12.5k still yields ~122 nt / ~216 nn positives, '
                         'about 9%% relative noise on the AP estimate, which is ample for RANKING '
                         'checkpoints; the reported number always comes from full val anyway.')
    p.add_argument('--cross-attn-dim', type=int, default=64)
    p.add_argument('--patch-pool-dim', type=int, default=256)
    p.add_argument('--use-motion', action='store_true',
                    help='pool the per-patch delta to the previous context position through a '
                         'second PatchAttnPool and concatenate it with the content pooling (see '
                         'MouseFrameClassifier.use_motion). Motivated by a zero-learning probe: '
                         'the mean patch-delta L2 norm alone reaches ROC-AUC 0.61 (nt) / 0.54 (nn) '
                         'vs 0.53/0.46 for a same-shape raw-content-magnitude baseline, so the '
                         'frame-to-frame change carries signal the content-only path has to infer '
                         'indirectly. Costs one extra pooling pass (~5%% of step time; the frozen '
                         'encoder dominates).')
    p.add_argument('--lr-decay-epochs', type=int, default=6)
    p.add_argument('--warmup-epochs', type=int, default=0,
                    help='linear LR ramp before the cosine decay starts. The trainable stack has '
                         'no normalization anywhere by default (raw DINO patch tokens -> '
                         'unnormalized MHA -> unnormalized MHA -> MLP), and every run to date '
                         'started at the full LR on step 1; the res448 run scored monitor AP 0.107 '
                         'in epoch 1 vs 0.242 in epoch 2, consistent with a wasted/unstable first '
                         'epoch. 0 reproduces the old schedule exactly.')
    p.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam',
                    help="'adam' applies weight_decay as L2 coupled to Adam's adaptive scaling, "
                         'which makes it a near-no-op as a regularizer -- it was inherited '
                         'unexamined from a config tuned on a 4x4 coarse patch grid. Now that the '
                         'model demonstrably overfits (train loss still falling at epoch 40 while '
                         'val AP is flat from ~24), decoupled AdamW decay is the knob that '
                         'actually bites.')
    p.add_argument('--lr', type=float, default=None, help='override inherited cfg lr')
    p.add_argument('--weight-decay', type=float, default=None, help='override inherited cfg')
    p.add_argument('--dropout', type=float, default=None, help='override inherited cfg')
    p.add_argument('--unfreeze-blocks', type=int, default=0,
                    help='SUPERVISED fine-tuning: unfreeze the last N of DINOv2-base\'s 12 '
                         'transformer blocks and train them with the same BCE loss. Every result '
                         'to date comes from a FROZEN encoder feeding a ~0.5M-param head, so the '
                         'representation itself has never been adapted to top-down grayscale '
                         'mouse video -- far from DINOv2\'s pretraining distribution. This is the '
                         'largest remaining capacity lever, and also the riskiest: the model '
                         'already overfits ~4.9k bouts, so the encoder gets its own much lower LR '
                         '(--encoder-lr) with layer-wise decay rather than the head\'s.')
    p.add_argument('--encoder-lr', type=float, default=1e-5,
                    help='LR for the outermost unfrozen block; deeper blocks are scaled down by '
                         '--layerwise-decay per block')
    p.add_argument('--layerwise-decay', type=float, default=0.65)
    p.add_argument('--seed', type=int, default=None,
                    help='seeds model init, dropout, negative resampling and the augmentation '
                         'draw. Defaults to gsf.SEED, reproducing every prior run. Until now '
                         'torch was never seeded on this path at all, so head init and dropout '
                         'varied run to run while the data order did not -- meaning "same config, '
                         'different result" was possible but a deliberate seed replicate was not. '
                         'Vary this (not the config) to separate a real effect from run-to-run '
                         'noise, which matters because the metric that ranks these models '
                         '(per-observation Pearson r at n=24) has SE ~0.2.')
    p.add_argument('--wandb', action='store_true', help='log the run to Weights & Biases')
    p.add_argument('--wandb-project', type=str, default='mice-behavior-frame')
    p.add_argument('--tag', type=str, default='online_aug')
    p.add_argument('--jpeg-cache-file', type=str, default=None,
                    help='persist/reuse the JPEG-bytes cache at this path (no extension). The read '
                         'phase is 444k RANDOM small-file NFS opens taking ~33 min at ~226 f/s, '
                         'during which the GPU sits at 0%% utilisation (measured). Written once as '
                         'ONE ~17 GiB file, it reloads as a single SEQUENTIAL read (or is memory-'
                         'mapped, so not read at all up front) in well under 2 min. Only the JPEG '
                         'representation is small enough for this -- the token cache is 163 GiB at '
                         '224px and 650 GiB at 448px. NOTE: shared storage was at 95%% capacity '
                         'when this was added, so this is opt-in rather than default.')
    p.add_argument('--smoke', action='store_true',
                    help='tiny end-to-end check: caps train AND val so the (NFS-bound) read phase '
                         'takes ~1 min instead of ~30, surfacing pipeline bugs before a real run')
    args = p.parse_args()
    if args.seed is None:
        args.seed = gsf.SEED
    torch.manual_seed(args.seed)
    if args.smoke:
        args.max_train_frames, args.n_epochs, args.batch_size = 4_000, 2, 32
        args.val_monitor_size, args.tag = 600, 'online_aug_smoke'

    n_patches = (args.input_size // PATCH_SIZE) ** 2
    # The old 'patchgrid256_dinov2_' prefix is gone: every run here is a 256-dim patch grid over
    # DINOv2, so it distinguished nothing and just pushed the informative part of the name out of
    # sight. Pass a --tag matching the scheme in rename_runs.py -- res<input>_k<context>_<encoder>
    # _<augment> -- or run that script afterwards to fold this run into it.
    OUT = gsf.FRAME_DIR / args.tag
    OUT.mkdir(parents=True, exist_ok=True)
    best_cfg = get_head_cfg()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'context_k={args.context_k} (T={2*args.context_k+1}, stride={args.stride}, '
          f'reach +-{args.context_k*args.stride} frames)  input_size={args.input_size} '
          f'({n_patches} patches)  augment={args.augment}  neg_ratio={args.neg_ratio}'
          f'  motion={args.use_motion}', flush=True)

    pair_labels = gsf.build_pair_labels(gsf.DATA_DIR, gsf.DATASET_DIR, overwrite=False)
    ann_csv = gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    o2p = gsf.load_obs_to_pool_map(gsf.DATA_DIR)
    all_obs = pd.read_parquet(pair_labels)['observation_id'].unique().tolist()
    pools = sorted({o2p[o] for o in all_obs})
    val_pools = get_fixed_val_pools(pools)
    train_obs = [o for o in all_obs if o2p[o] not in val_pools]
    val_obs = [o for o in all_obs if o2p[o] in val_pools]
    if args.smoke:
        # val is 144k frames and dominates the read phase, so cap it too or "smoke" isn't smoke
        train_obs, val_obs = train_obs[:2], val_obs[:1]
    print(f'train {len(train_obs)} obs / val {len(val_obs)} obs (pools {sorted(val_pools)})', flush=True)

    tm = FrameBatchData(str(ann_csv), str(pair_labels), train_obs, args.context_k, 1,
                        dummy_loader(1, 1), n_patches=1, stride=args.stride,
                        max_frames=args.max_train_frames, seed=args.seed)
    vm = FrameBatchData(str(ann_csv), str(pair_labels), val_obs, args.context_k, 1,
                        dummy_loader(1, 1), n_patches=1, stride=args.stride)
    del tm.flat, vm.flat

    pos_idx = np.where(tm.labels.sum(1) > 0)[0]
    neg_idx = np.where(tm.labels.sum(1) == 0)[0]
    n_neg = min(len(neg_idx), args.neg_ratio * max(len(pos_idx), 1))
    saturated = n_neg >= len(neg_idx)
    print(f'{args.neg_ratio}:1 sampling: {len(pos_idx):,} pos, {len(neg_idx):,} neg available -> '
          f'{"SATURATED" if saturated else f"fresh {n_neg:,} each epoch"}', flush=True)

    # deliberately gsf.SEED, not args.seed: the monitor subset is evaluation infrastructure, and
    # early stopping has to score every arm of a sweep against the identical val frames or the
    # arms are not comparable. --seed varies the training run, never what it is measured on.
    v_rng = np.random.default_rng(gsf.SEED)
    val_keep = np.sort(v_rng.choice(len(vm), size=min(len(vm), args.val_monitor_size), replace=False))
    vp = vm.labels[val_keep]
    print(f'val monitor {len(val_keep):,} (prevalence-preserving: nt {100*vp[:,0].mean():.2f}% '
          f'nn {100*vp[:,1].mean():.2f}%)', flush=True)

    def needed(meta, si):
        a = meta.gi[si][:, None] + meta.offsets_grid[None, :]
        return np.unique(a[~meta.pad_mask[si]])

    # Frames the run COULD touch. No longer pre-read -- used only to size the save manifest
    # and to report what fraction the lazy path actually avoided.
    all_needed = np.unique(np.concatenate([
        needed(tm, np.concatenate([pos_idx, neg_idx])),
        needed(vm, np.arange(len(vm)))]))
    val_full_frames = needed(vm, np.arange(len(vm)))

    ann = pd.read_csv(ann_csv, usecols=['frame_path'])
    frame_paths = ann.frame_path.values
    cache_bin = Path(f'{args.jpeg_cache_file}.bin') if args.jpeg_cache_file else None
    cache_idx = Path(f'{args.jpeg_cache_file}.npz') if args.jpeg_cache_file else None

    jpeg_cache = {}
    if cache_bin and cache_bin.exists() and cache_idx.exists():
        m = np.load(cache_idx)
        t0 = time.time()
        blob = np.memmap(cache_bin, dtype=np.uint8, mode='r')
        offs, keys = m['offsets'], m['all_needed']
        jpeg_cache = {int(k): blob[offs[i]:offs[i+1]] for i, k in enumerate(keys)}
        print(f'JPEG cache REUSED from {cache_bin} ({int(offs[-1])/1024**3:.1f} GiB, memory-mapped, '
              f'{len(jpeg_cache):,} frames) in {time.time()-t0:.1f}s', flush=True)

    def ensure_cached(frame_idx, what):
        """Read only the frames we do not already hold. Lazy by design: the upfront read of
        every candidate frame blocked training for ~33 min at 0% GPU, and ~32% of it was
        full-val frames that are not touched until the final evaluation."""
        missing = np.array(sorted(set(int(g) for g in frame_idx) - set(jpeg_cache)), dtype=np.int64)
        if not len(missing):
            return 0.0
        t0 = time.time()
        dl = DataLoader(_BytesReader(frame_paths[missing]), batch_size=None,
                        num_workers=args.read_workers, prefetch_factor=6, collate_fn=lambda x: x)
        for i, buf in dl:
            jpeg_cache[int(missing[i])] = buf
        dt = time.time() - t0
        print(f'  [{what}] read {len(missing):,} new frames in {dt/60:.1f} min '
              f'({len(missing)/max(dt,1e-9):.0f} f/s); cache now {len(jpeg_cache):,}', flush=True)
        return dt

    encoder = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    encoder.requires_grad_(False)
    # best_cfg comes from a search run on a 4x4 (16-token) coarse patch grid with CACHED tokens,
    # the old val split and neg_ratio=15 -- every one of those conditions has since changed, so
    # each field is overridable rather than inherited on faith.
    lr = args.lr if args.lr is not None else best_cfg['lr']
    wd = args.weight_decay if args.weight_decay is not None else best_cfg['weight_decay']
    dropout = args.dropout if args.dropout is not None else best_cfg['dropout']
    model = MouseFrameClassifier(
        emb_dim=EMB_DIM, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'],
        use_patch_grid=True, dropout=dropout, use_motion=args.use_motion,
        cross_attn_dim=args.cross_attn_dim or None, patch_pool_dim=args.patch_pool_dim or None).to(dev)
    print(f'classifier params: {sum(p.numel() for p in model.parameters()):,} '
          f'(DINOv2 frozen, {sum(p.numel() for p in encoder.parameters())/1e6:.0f}M)', flush=True)
    print(f'optimizer={args.optimizer} lr={lr:g} weight_decay={wd:g} dropout={dropout:g} '
          f'warmup={args.warmup_epochs} decay_epochs={args.lr_decay_epochs}', flush=True)
    opt_cls = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    groups = [{'params': list(model.parameters()), 'lr': lr}]
    if args.unfreeze_blocks > 0:
        blocks = encoder.encoder.layer            # DINOv2-base: 12 blocks
        n = min(args.unfreeze_blocks, len(blocks))
        for depth, blk in enumerate(reversed(blocks[-n:])):
            blk.requires_grad_(True)
            # depth 0 = outermost (closest to the head) and most task-specific, so it moves
            # fastest; deeper blocks hold the general features we do NOT want to disturb.
            blk_lr = args.encoder_lr * (args.layerwise_decay ** depth)
            groups.append({'params': list(blk.parameters()), 'lr': blk_lr})
        # LayerNorm after the last block feeds the tokens we actually consume
        encoder.layernorm.requires_grad_(True)
        groups.append({'params': list(encoder.layernorm.parameters()), 'lr': args.encoder_lr})
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f'fine-tuning last {n}/{len(blocks)} encoder blocks: {trainable/1e6:.1f}M params, '
              f'lr {args.encoder_lr:g} decaying x{args.layerwise_decay} per block inward',
              flush=True)
    opt = opt_cls(groups, lr=lr, weight_decay=wd)
    finetune = args.unfreeze_blocks > 0
    eta = 0.01

    def lr_factor(e):
        # warmup_epochs=0 reduces this EXACTLY to the previous cosine-to-floor schedule.
        if e < args.warmup_epochs:
            return (e + 1) / (args.warmup_epochs + 1)
        prog = (e - args.warmup_epochs) / max(args.lr_decay_epochs - args.warmup_epochs, 1)
        return eta if prog >= 1 else eta + 0.5 * (1 - eta) * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    run = None
    if args.wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=args.tag, config=vars(args) | {
                'n_patches': n_patches, 'lr': lr, 'weight_decay': wd, 'dropout': dropout,
                'n_heads': best_cfg['n_heads'], 'hidden_dim': best_cfg['hidden_dim'],
                'n_params': sum(p.numel() for p in model.parameters())})
            print(f'wandb: {run.url}', flush=True)
        except Exception as e:   # never let telemetry kill a multi-hour training run
            print(f'wandb disabled ({e.__class__.__name__}: {e})', flush=True)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')

    def scaled(lo, hi):
        return (1 - (1 - lo) * args.photo_strength, 1 + (hi - 1) * args.photo_strength)

    def make_loader(meta, order, augment, seed):
        batches = [order[i:i+args.batch_size] for i in range(0, len(order), args.batch_size)]
        return DataLoader(_SampleDataset(meta, batches, jpeg_cache,
                                         args.input_size, augment, seed,
                                         scaled(0.80, 1.25), scaled(0.80, 1.25), scaled(0.83, 1.20)),
                          batch_size=None, num_workers=args.decode_workers,
                          pin_memory=(dev.type == 'cuda'), prefetch_factor=4)

    @torch.no_grad()
    def evaluate(order):
        model.eval()
        P, L = [], []
        for imgs, offs, lbl, mask in make_loader(vm, order, 'none', 0):
            imgs = imgs.to(dev, non_blocking=True)
            B, T = imgs.shape[:2]
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                tok = encoder(pixel_values=imgs.view(B*T, *imgs.shape[2:])).last_hidden_state[:, 1:]
                logits = model(tok.view(B, T, n_patches, EMB_DIM),
                               offsets=offs.to(dev), key_padding_mask=mask.to(dev))
            P.append(torch.sigmoid(logits).float().cpu()); L.append(lbl)
        return torch.cat(P).numpy(), torch.cat(L).numpy()

    rng = np.random.default_rng(args.seed)
    best, since, hist = -1.0, 0, []
    for ep in range(1, args.n_epochs + 1):
        model.train()
        order = (np.concatenate([pos_idx, neg_idx]) if saturated else
                 np.concatenate([pos_idx, rng.choice(neg_idx, size=n_neg, replace=False)]))
        rng.shuffle(order)
        read_s = ensure_cached(needed(tm, order), f'epoch {ep} train')
        if ep == 1:
            read_s += ensure_cached(needed(vm, val_keep), 'val monitor')
        tot, seen, t0 = 0.0, 0, time.time()
        for imgs, offs, lbl, mask in make_loader(tm, order, args.augment, args.seed * 1000 + ep):
            imgs, lbl = imgs.to(dev, non_blocking=True), lbl.to(dev, non_blocking=True)
            B, T = imgs.shape[:2]
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                # frozen encoder: no graph, no backward through DINOv2. When fine-tuning we
                # must keep the graph, which is what makes the step ~2x more expensive.
                with torch.set_grad_enabled(finetune):
                    tok = encoder(pixel_values=imgs.view(B*T, *imgs.shape[2:])).last_hidden_state[:, 1:]
                tok = tok.view(B, T, n_patches, EMB_DIM)
                logits = model(tok if finetune else tok.detach(),
                               offsets=offs.to(dev), key_padding_mask=mask.to(dev))
                loss = crit(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g['params'] if p.grad is not None], 0.5)
            scaler.step(opt); scaler.update()
            tot += loss.item() * B; seen += B
        sched.step()
        probs, labs = evaluate(val_keep)
        per_ap = [float(average_precision_score(labs[:, i], probs[:, i])) for i in (0, 1)]
        per_auc = [float(roc_auc_score(labs[:, i], probs[:, i])) for i in (0, 1)]
        ap = float(np.mean(per_ap))
        # val BCE on the PREVALENCE-PRESERVING monitor set, whereas train loss is on balanced
        # 1:1 data -- the two are not comparable in absolute terms, but the val trend is the
        # overfitting signal that a plateauing AP alone does not distinguish from convergence.
        p = np.clip(probs, 1e-7, 1 - 1e-7)
        val_loss = float(-(labs * np.log(p) + (1 - labs) * np.log(1 - p)).mean())
        row = {'epoch': ep, 'train_loss': tot/seen, 'monitor_ap': ap, 'val_loss': val_loss,
               'ap_nt': per_ap[0], 'ap_nn': per_ap[1],
               'auc_nt': per_auc[0], 'auc_nn': per_auc[1],
               'lr': float(opt.param_groups[0]['lr'])}
        hist.append(row)
        if run is not None:
            try:
                run.log(row, step=ep)
            except Exception:
                pass
        print(f'epoch {ep:3d}/{args.n_epochs}  loss={tot/seen:.4f}  val_loss={val_loss:.4f}  '
              f'monitor_ap={ap:.4f}  nt={per_ap[0]:.4f}  nn={per_ap[1]:.4f}  '
              f'auc_nt={per_auc[0]:.4f}  auc_nn={per_auc[1]:.4f}  '
              f'lr={opt.param_groups[0]["lr"]:.2e}  ({time.time()-t0:.1f}s compute'
              f'{f", {read_s:.0f}s new-frame read" if read_s > 1 else ""})', flush=True)
        if ap > best:
            best, since = ap, 0
            torch.save(model.state_dict(), OUT / 'best_model.pt')
            if finetune:
                # The head alone is not a usable checkpoint once the encoder has moved -- but
                # only the unfrozen blocks moved. Saving the whole encoder wrote 346 MB per run
                # of which the frozen blocks are a byte-identical copy of facebook/dinov2-base:
                # at --unfreeze-blocks 2 that is 10 of 12 blocks, ~287 MB of pure duplication,
                # per saved epoch, per run. Store only the tensors that actually train; every
                # loader pairs this with a freshly pretrained encoder and applies it with
                # strict=False, so the frozen remainder comes from the hub as it always did.
                train_keys = {n for n, p in encoder.named_parameters() if p.requires_grad}
                torch.save({k: v for k, v in encoder.state_dict().items() if k in train_keys},
                           OUT / 'best_encoder.pt')
        else:
            since += 1
            if since >= args.patience:
                print(f'early stopping (no improvement for {args.patience})', flush=True)
                break

    model.load_state_dict(torch.load(OUT / 'best_model.pt', map_location=dev, weights_only=True))
    if finetune:
        # strict=False: best_encoder.pt holds only the unfrozen tensors (see the save above).
        # `encoder` already carries the pretrained weights, so this overlays the trained ones.
        # Also accepts the pre-2026-08-14 full-encoder files, where every key simply matches.
        encoder.load_state_dict(torch.load(OUT / 'best_encoder.pt', map_location=dev,
                                           weights_only=True), strict=False)
        encoder.eval()
    ensure_cached(val_full_frames, 'full-val (deferred to the end)')
    probs, labs = evaluate(np.arange(len(vm)))
    sample_obs = np.zeros(len(vm), dtype=object)   # observation id per val sample
    a2 = pd.read_csv(ann_csv, usecols=['observation_id', 'frame_idx'])
    a2 = a2[a2.observation_id.isin(set(val_obs))].reset_index()
    b = {o: int(g['index'].values[0]) for o, g in a2.groupby('observation_id', sort=False)}
    st = np.array(sorted(b.values())); nm = np.array([k for k, _ in sorted(b.items(), key=lambda x: x[1])])
    sample_obs = nm[np.searchsorted(st, vm.gi, side='right') - 1]

    print(f'\n{"="*70}\nFULL-VAL\n{"="*70}')
    apr = ap_report(probs, labs, sample_obs, tolerances=(0, 1, 2))
    print(format_ap_report(apr, tolerances=(0, 1, 2)))
    rr = rate_report(probs, labs, sample_obs)
    auc = {n: float(roc_auc_score(labs[:, i], probs[:, i])) for i, n in enumerate(('nt', 'nn'))}
    print(f'\nROC-AUC  nt {auc["nt"]:.4f}  nn {auc["nn"]:.4f}')
    print()
    print(format_rate_report(rr))

    # qualitative TP/FP/FN/TN grids for both behaviours. The val frames are already in the
    # JPEG cache from the full-val pass, so this costs decoding only -- no NFS re-read.
    viz_files = {}
    try:
        viz_files = plot_confusion_examples(
            probs, labs, sample_obs, vm.gi, frame_paths, OUT, jpeg_cache=jpeg_cache,
            title_prefix=f'{args.tag}  ')
    except Exception as e:                # never let a figure lose a finished training run
        print(f'  [viz] skipped ({e.__class__.__name__}: {e})', flush=True)

    if run is not None:
        try:
            import wandb
            run.log({f'confusion/{k}': wandb.Image(str(v)) for k, v in viz_files.items()})
        except Exception:
            pass
        try:
            # the whole suite, not just AP: plain + tolerant AP (bouts are 1-2 frames, so
            # tol1/tol2 separate genuine misses from boundary jitter), ROC-AUC, and the
            # per-observation rate agreement that metrics.py flags as the select-on quantity.
            run.summary.update(
                {f'fullval/ap_{k.replace("/", "_")}': v['ap'] for k, v in apr.items()}
                | {f'fullval/enrich_{k.replace("/", "_")}': v['enrichment']
                   for k, v in apr.items() if 'enrichment' in v}
                | {f'fullval/auc_{k}': v for k, v in auc.items()}
                | {f'fullval/rate_{n}_{k}': v for n, d in rr.items() for k, v in d.items()
                   if isinstance(v, (int, float))})
            run.finish()
        except Exception:
            pass
    json.dump({'cfg': best_cfg, 'context_k': args.context_k, 'input_size': args.input_size,
               'n_patches': n_patches, 'augment': args.augment, 'neg_ratio': args.neg_ratio,
               'use_motion': args.use_motion, 'lr_decay_epochs': args.lr_decay_epochs, 'stride': args.stride,
               'cross_attn_dim': args.cross_attn_dim, 'patch_pool_dim': args.patch_pool_dim,
               'unfreeze_blocks': args.unfreeze_blocks, 'encoder_lr': args.encoder_lr,
               'layerwise_decay': args.layerwise_decay,
               'photo_strength': args.photo_strength, 'optimizer': args.optimizer, 'seed': args.seed,
               'warmup_epochs': args.warmup_epochs, 'lr': lr, 'weight_decay': wd, 'dropout': dropout,
               'n_epochs': args.n_epochs, 'batch_size': args.batch_size,
               'max_train_frames': args.max_train_frames, 'val_pools': sorted(val_pools),
               'jpeg_cache_gib': sum(len(b) for b in jpeg_cache.values())/1024**3,
               'jpeg_cache_frames': len(jpeg_cache), 'ap_report': apr, 'rate_report': rr,
               'best_ap': apr['macro/tol0']['ap'], 'history': hist},
              open(OUT / 'config.json', 'w'), indent=2)
    if cache_bin and not cache_bin.exists():
        t0 = time.time()
        cache_bin.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(sorted(jpeg_cache), dtype=np.int64)
        offs = np.zeros(len(keys) + 1, dtype=np.int64)
        with open(cache_bin, 'wb') as fh:
            for i, k in enumerate(keys):
                b = jpeg_cache[int(k)]
                fh.write(b.tobytes() if hasattr(b, 'tobytes') else bytes(b))
                offs[i+1] = offs[i] + len(b)
        np.savez(cache_idx, all_needed=keys, offsets=offs)
        print(f'Saved reusable JPEG cache -> {cache_bin} ({offs[-1]/1024**3:.1f} GiB, '
              f'{len(keys):,} frames, {time.time()-t0:.0f}s)', flush=True)
    print(f'\nSaved {OUT}/', flush=True)


if __name__ == '__main__':
    main()
