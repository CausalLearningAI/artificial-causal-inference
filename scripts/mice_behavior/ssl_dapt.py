"""Domain-adaptive self-supervised pretraining of DINOv2 on unlabelled mice v1 frames.

This is the `ssl_dapt` arm that ablate_head_vs_encoder.sh names and deliberately does not
launch. It tests hypothesis (2) -- that supervised fine-tuning helped because DINOv2 needed
RECALIBRATION to top-down grayscale rodent video, not because it needed to learn new
task-specific computation. If that is what is going on, the adaptation should not require
behaviour labels at all: 1.73M frames nobody has ever annotated should suffice.

WHAT IT BUYS US THAT THE SUPERVISED ARMS CANNOT
-----------------------------------------------
The labelled corpus is 144 observations in 24 pools, and the head is trained on 120 of them
(77,613 anchor frames after bounding). Sitting unused on disk are 288 observations in 48
further pools -- different animals, different sessions, all three lines -- totalling 1,728,000
frames with no behaviour annotation whatsoever. Supervised adaptation cannot touch them. This
can, and that is the entire argument for running it: the axis it adds is SCENE DIVERSITY
(2.4x the pools, 3.4x the observations), not more gradient steps on the same 24 pools.

WHY NOT LITERALLY EVERY FRAME
-----------------------------
"Use all frames" is the obvious framing and it is the wrong one, for a reason specific to this
dataset rather than a compute excuse. The recordings are 5 fps, so consecutive frames are 200 ms
apart and a mouse barely moves between them; frame t+1 is very nearly a duplicate of frame t.
All 2,448,000 non-val frames would cost a 102.6 GiB JPEG cache (vs 18.6 GiB today, on a
filesystem at 96% capacity) and ~4.1 h per epoch, i.e. 40+ h for a normal SSL schedule --
past the wall clock, so it would need checkpoint/requeue plumbing -- and it would spend that
budget re-encoding near-duplicates. --frame-stride 10 keeps one frame every 2 s: 244,800
frames and a 10.3 GiB cache, and it still spans EVERY non-val observation and pool. Diversity
is preserved; redundancy is what gets dropped.

Measured throughput (smoke run, L40S, 448px, batch 64): 97 frames/s including teacher forward
and student forward/backward, i.e. ~42 min per epoch at stride 10, so ~8.4 h for 12 epochs on
an L40S and ~6 h on an A100. That fits one job; stride 1 would not.

VAL POOLS ARE EXCLUDED (rd11_2, rd13, rd14, rd18)
-------------------------------------------------
SSL uses no labels, so training on val-pool frames is tempting and would even be defensible as
transductive learning. It is excluded anyway. The whole point of the standing val set is to
estimate generalisation to unseen ANIMALS, and an encoder that has seen those animals' frames
-- however label-free the objective -- no longer supports that claim. Excluding them also keeps
Stage B's number directly comparable to every arm measured on this split since 2026-08-12.

THE OBJECTIVE: MASKED PATCH FEATURE PREDICTION AGAINST AN EMA TEACHER
--------------------------------------------------------------------
Teacher (EMA of the student, no grad) sees the clean image. Student sees the image with ~50% of
its patches replaced by a learned mask token, and must predict the teacher's patch features at
exactly the masked positions. Loss is smooth-L1 on instance-normalised targets averaged over
the top --target-layers blocks. This is the data2vec-2.0 recipe.

Two deliberate choices:

  PATCH-level, not CLS-level. The downstream head pools PATCH tokens (patch_pool_dim=256 over
  a 32x32 grid); the CLS token is not what it reads. A DINO-style image-level objective would
  shape a representation the classifier never consumes.

  Feature regression, not prototype distillation. DINOv2's own loss adds iBOT prototypes with
  centering, sharpening and Sinkhorn normalisation, plus KoLeo and multi-crop -- a dozen coupled
  knobs whose failure mode is silent collapse, and 10 forward passes per image. Regressing
  normalised teacher features needs two forwards, has no prototype bank to tune, and collapse
  is directly observable (see below). For CONTINUED pretraining under domain shift, that
  trade is worth making; we are not reproducing DINOv2 from scratch.

Masking is BLOCK-wise, not i.i.d. per patch. At 1024 patches an independent 50% mask leaves
almost every masked patch with 4 unmasked neighbours, and bilinear-interpolating a neighbour is
a near-perfect solution that teaches nothing. Contiguous rectangles remove that shortcut.

THE SAME 2 BLOCKS, SO THE COMPARISON IS ABOUT THE SIGNAL AND NOT THE CAPACITY
----------------------------------------------------------------------------
--unfreeze-blocks defaults to 2 to match res448_k2_ft2_d4photo and res448_k2_bit2_*. Held
fixed across arms, the only thing that varies is what moved those blocks: behaviour labels
(ft2), bias-only behaviour labels (bit2), or unlabelled frames (here). Resolution is 448 for
the same reason -- adapting blocks at 224 and deploying them at 448 would confound the arm
with a resolution shift, and 448 beat 224 by a wide margin (0.438 vs 0.296 macro AP).

GUARDS -- READ THESE BEFORE READING STAGE B'S AP
-------------------------------------------------
An SSL run that quietly did nothing and an SSL run that quietly collapsed both produce a Stage
B number that looks like an honest null. Three metrics separate those cases, logged every epoch:

  drift        1 - cos(adapted patch features, STOCK DINOv2 patch features) on a fixed probe
               batch drawn from the VAL pools (never trained on here). ~0 means Stage A was a
               no-op and Stage B is measuring nothing; near 1 means the representation was
               destroyed. This is the single number that decides whether Stage B is worth 5 h.
  target_std   per-dimension std of the teacher targets across the batch. Collapse drives this
               to 0. If it falls below --collapse-std the run aborts rather than burning hours
               producing an uninterpretable checkpoint.
  st_cos       student-teacher cosine at masked positions. Should climb from ~0; pinned at 1.0
               early means the task is trivially solvable and the mask is not biting.

There is no early stopping and no best-checkpoint selection, on purpose: Stage A has no labelled
validation signal, and SSL loss has no established monotone relation to downstream AP, so
selecting on it would be selection on a quantity we cannot interpret. The final encoder is what
Stage B consumes.

Stage B is train_online_aug.py --init-encoder <this run>/best_encoder.pt --unfreeze-blocks 0,
with every other flag identical to the frozen_ctrl_s42 arm. That keeps the encoder frozen during
head training, so the arm answers "did unlabelled adaptation improve the FROZEN representation?"
-- which is the question, and is also why it is comparable to frozen_ctrl_s42 rather than to a
fine-tuned run.

    python scripts/mice_behavior/ssl_dapt.py --smoke        # ~5 min end-to-end pipeline check
    bash scripts/mice_behavior/ablate_ssl_dapt.sh           # both stages, chained
"""
import argparse
import io
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.pools import VAL_POOLS_V1, load_obs_to_pool_map
from train_online_aug import (DATASET_ROOT, IMAGENET_MEAN, IMAGENET_STD, MODEL_ID,
                              PATCH_SIZE, _BytesReader, d4_transform)


class _FrameDataset(Dataset):
    """Decodes + augments + normalises one whole batch of single frames per __getitem__.

    One item is a batch (not a frame) for the same reason train_online_aug.py does it: the
    per-item Python/IPC overhead is a significant fraction of a ~3 ms JPEG decode, and batching
    inside the worker amortises it. jpeg_cache is inherited through fork copy-on-write.

    Augmentation is drawn PER FRAME here, unlike the supervised path which draws once per
    sample and shares it across the T context frames. There is no temporal window in this
    objective -- each frame is an independent image -- so the constraint that motivated sharing
    (not injecting spurious frame-to-frame change into a window read temporally) does not apply.
    """

    def __init__(self, gidx, batches, jpeg_cache, input_size, augment, seed, photo_strength=1.0):
        self.gidx, self.batches, self.cache = gidx, batches, jpeg_cache
        self.input_size, self.augment, self.seed = input_size, augment, seed
        s = photo_strength
        self.bright = (1 - 0.20 * s, 1 + 0.25 * s)
        self.contrast = (1 - 0.20 * s, 1 + 0.25 * s)
        self.gamma = (1 - 0.17 * s, 1 + 0.20 * s)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, bi):
        rows = self.batches[bi]
        S = self.input_size
        out = torch.zeros((len(rows), 3, S, S), dtype=torch.float32)
        rng = np.random.default_rng((self.seed, bi))
        use_d4 = self.augment in ('d4', 'd4_photo')
        use_photo = self.augment == 'd4_photo'
        for j, r in enumerate(rows):
            buf = self.cache[int(self.gidx[r])]
            with Image.open(io.BytesIO(buf.tobytes())) as im:
                im = im.convert('RGB')
                if use_d4:
                    im = d4_transform(im, int(rng.integers(0, 8)))
                if use_photo:
                    im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(*self.bright)))
                    im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(*self.contrast)))
                im = im.resize((S, S), Image.BILINEAR)
                a = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())
            a = a.permute(2, 0, 1).float().div_(255)
            if use_photo:
                a = a.pow_(float(rng.uniform(*self.gamma)))
            out[j] = (a - IMAGENET_MEAN) / IMAGENET_STD
        return out


def block_mask(batch, grid, ratio, min_patches, rng):
    """BEiT-style block masking: cover ~`ratio` of a grid x grid patch map with rectangles.

    i.i.d. masking is the wrong null here. At grid=32 a 50% independent mask leaves nearly every
    masked patch with unmasked neighbours, and DINOv2 patch features are locally smooth enough
    that copying a neighbour nearly solves the task -- the loss would fall without the encoder
    learning anything about this domain. Rectangles of >= min_patches force prediction across a
    span wider than the local smoothness.
    """
    target = int(ratio * grid * grid)
    m = np.zeros((batch, grid, grid), dtype=bool)
    for b in range(batch):
        while True:
            filled = int(m[b].sum())
            if filled >= target:
                break
            # Bound each rectangle by what is still unmasked. Drawing freely from
            # [min_patches, target/2] overshot the requested ratio by up to 28% (measured
            # 0.64 for ratio 0.5), which would make the task unevenly hard across images and
            # confound the loss curve with mask-area variance. Overshoot is now at most one
            # min_patches block.
            hi = max(min_patches + 1, min(target // 2, target - filled) + 1)
            # aspect ratio drawn in log space so wide and tall blocks are equally likely
            area = int(rng.integers(min_patches, hi))
            ar = math.exp(rng.uniform(math.log(0.3), math.log(1 / 0.3)))
            h = min(grid, max(1, int(round(math.sqrt(area * ar)))))
            w = min(grid, max(1, int(round(math.sqrt(area / ar)))))
            top = rng.integers(0, grid - h + 1)
            left = rng.integers(0, grid - w + 1)
            m[b, top:top + h, left:left + w] = True
    return torch.from_numpy(m.reshape(batch, -1))


@torch.no_grad()
def patch_features(encoder, x, chunk=32):
    """Final-layer patch tokens (CLS dropped), which is what the downstream head consumes."""
    outs = []
    for i in range(0, len(x), chunk):
        with torch.amp.autocast('cuda', enabled=x.is_cuda):
            outs.append(encoder(pixel_values=x[i:i + chunk]).last_hidden_state[:, 1:].float())
    return torch.cat(outs)


def make_targets(teacher, x, n_layers):
    """Average the top `n_layers` blocks' patch tokens, instance-normalising each first.

    Averaging several layers rather than taking the last one is the data2vec-2.0 target: a
    single layer is a narrower, noisier regression target, and with only the last 2 blocks
    trainable most of the averaged depth is frozen, which anchors the target and is a large part
    of why this variant does not need centering/sharpening to stay stable. The per-layer
    instance norm puts every layer on the same scale so the deepest one cannot dominate purely
    by having a larger activation magnitude.
    """
    with torch.no_grad():
        out = teacher(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states[-n_layers:]
        t = torch.stack([F.layer_norm(h[:, 1:].float(), (h.shape[-1],)) for h in hs]).mean(0)
        return F.layer_norm(t, (t.shape[-1],))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='ssl_dapt')
    p.add_argument('--input-size', type=int, default=448,
                   help='448 to match the downstream head. Adapting at 224 and deploying at 448 '
                        'would confound this arm with a resolution shift.')
    p.add_argument('--frame-stride', type=int, default=10,
                   help='keep every Nth frame of every non-val observation. At 5 fps, stride 10 '
                        'is one frame every 2 s. See the module docstring: stride 1 costs a '
                        '102.6 GiB cache and ~4.1 h/epoch to re-encode near-duplicates.')
    p.add_argument('--include-labeled-obs', type=int, default=1,
                   help='1 keeps the 120 non-val LABELLED observations in the SSL corpus '
                        'alongside the 288 unlabelled ones (they are still unlabelled AS FAR AS '
                        'THIS OBJECTIVE IS CONCERNED -- no Y is read here). 0 restricts to the '
                        '288, which is the stricter "labels bought us nothing" claim.')
    p.add_argument('--unfreeze-blocks', type=int, default=2,
                   help='matched to res448_k2_ft2_d4photo so the arms differ only in what moved '
                        'the blocks, not in how many moved.')
    p.add_argument('--encoder-lr', type=float, default=3e-5,
                   help='3x the supervised 1e-5, not the 10x this arm was first written with. '
                        'The smoke run (2 epochs, ~100 steps, peak lr 5e-5) already reached '
                        'drift 0.32, so the representation moves FAST under this objective; at '
                        '1e-4 over the real ~46k steps it would very likely overshoot into the '
                        'regime where Stage B measures a wrecked encoder rather than an adapted '
                        'one. The asymmetry matters: too little drift is cheap to detect and fix '
                        'by raising this, too much costs the full ~11 h before it shows up.')
    p.add_argument('--layerwise-decay', type=float, default=0.65,
                   help='same convention as every fine-tuning run in this repo (and equally '
                        'untuned there -- inherited for comparability, not because it is optimal).')
    p.add_argument('--mask-ratio', type=float, default=0.5)
    p.add_argument('--mask-min-patches', type=int, default=16)
    p.add_argument('--target-layers', type=int, default=4)
    p.add_argument('--ema-momentum', type=float, default=0.996,
                   help='start value; annealed to 1.0 on a cosine so the target stops moving as '
                        'the student converges')
    p.add_argument('--n-epochs', type=int, default=12)
    p.add_argument('--warmup-epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64,
                   help='64, matching the supervised runs, because the trainable mask token sits '
                        'at the EMBEDDING layer: autograd therefore stores activations for all 12 '
                        'blocks even though only the last 2 hold trainable weights.')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--augment', default='d4_photo', choices=['none', 'd4', 'd4_photo'])
    p.add_argument('--photo-strength', type=float, default=1.0)
    p.add_argument('--read-workers', type=int, default=32)
    p.add_argument('--decode-workers', type=int, default=16)
    p.add_argument('--probe-size', type=int, default=256,
                   help='frames from the VAL pools used only to measure drift. Never trained on.')
    p.add_argument('--collapse-std', type=float, default=0.05,
                   help='abort if the teacher targets stop varying across the batch')
    p.add_argument('--jpeg-cache-file', default='dataset/mice/v1/jpegcache_ssl',
                   help='built once, reused by every later SSL arm')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', default='mice-behavior-frame')
    p.add_argument('--smoke', action='store_true',
                   help='tiny end-to-end check: 3k frames, 2 epochs, own cache file')
    args = p.parse_args()
    if args.seed is None:
        args.seed = gsf.SEED
    if args.smoke:
        args.n_epochs, args.warmup_epochs, args.frame_stride = 2, 0, 400
        args.probe_size, args.tag = 64, 'ssl_dapt_smoke'
        args.jpeg_cache_file = 'dataset/mice/v1/jpegcache_ssl_smoke'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    grid = args.input_size // PATCH_SIZE
    OUT = gsf.FRAME_DIR / args.tag
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- frame selection -------------------------------------------------------------
    ann = pd.read_csv(gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv',
                      usecols=['observation_id', 'frame_idx', 'frame_path'])
    ann['g'] = np.arange(len(ann))            # global index: the JPEG cache's key
    o2p = load_obs_to_pool_map(gsf.DATA_DIR)
    ann['pool'] = ann.observation_id.map(o2p)
    if ann['pool'].isna().any():
        raise SystemExit(f'{ann["pool"].isna().sum()} frames have no pool mapping')
    is_val = ann['pool'].isin(VAL_POOLS_V1)

    # An observation counts as labelled if any behaviour column is non-null anywhere in it.
    lab_cols = ['Y_nn', 'Y_np', 'Y_nt']
    ycols = pd.read_csv(gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv', usecols=lab_cols)
    labeled_obs = set(ann.observation_id[ycols.notna().any(axis=1).values].unique())

    pool_ok = ~is_val
    if not args.include_labeled_obs:
        pool_ok &= ~ann.observation_id.isin(labeled_obs)
    sel = ann[pool_ok & (ann.frame_idx % args.frame_stride == 0)]
    gidx = sel['g'].values
    frame_paths = ann.frame_path.values

    n_unlab = int((~sel.observation_id.isin(labeled_obs)).sum())
    print(f'SSL corpus: {len(gidx):,} frames  '
          f'({sel.observation_id.nunique()} observations, {sel["pool"].nunique()} pools, '
          f'stride {args.frame_stride})', flush=True)
    print(f'  from never-annotated observations: {n_unlab:,} frames '
          f'({100*n_unlab/max(len(gidx),1):.0f}%)', flush=True)
    print(f'  val pools EXCLUDED: {sorted(VAL_POOLS_V1)}', flush=True)

    # probe frames: val pools only, so drift is measured where nothing was trained
    probe_pool = ann[is_val]['g'].values
    probe_gidx = rng.choice(probe_pool, size=min(args.probe_size, len(probe_pool)), replace=False)

    # ---- JPEG cache ------------------------------------------------------------------
    cache_bin = Path(f'{args.jpeg_cache_file}.bin')
    cache_idx = Path(f'{args.jpeg_cache_file}.npz')
    jpeg_cache = {}
    if cache_bin.exists() and cache_idx.exists():
        m = np.load(cache_idx)
        t0 = time.time()
        blob = np.memmap(cache_bin, dtype=np.uint8, mode='r')
        offs, keys = m['offsets'], m['all_needed']
        jpeg_cache = {int(k): blob[offs[i]:offs[i + 1]] for i, k in enumerate(keys)}
        print(f'JPEG cache REUSED from {cache_bin} ({int(offs[-1])/1024**3:.1f} GiB, '
              f'memory-mapped, {len(jpeg_cache):,} frames) in {time.time()-t0:.1f}s', flush=True)

    need = np.array(sorted(set(int(g) for g in np.concatenate([gidx, probe_gidx]))
                           - set(jpeg_cache)), dtype=np.int64)
    if len(need):
        t0 = time.time()
        dl = DataLoader(_BytesReader(frame_paths[need]), batch_size=None,
                        num_workers=args.read_workers, prefetch_factor=6, collate_fn=lambda x: x)
        for i, buf in dl:
            jpeg_cache[int(need[i])] = buf
        dt = time.time() - t0
        print(f'  read {len(need):,} new frames in {dt/60:.1f} min '
              f'({len(need)/max(dt,1e-9):.0f} f/s); cache now {len(jpeg_cache):,}', flush=True)
        keys = np.array(sorted(jpeg_cache), dtype=np.int64)
        offs = np.zeros(len(keys) + 1, dtype=np.int64)
        for i, k in enumerate(keys):
            offs[i + 1] = offs[i] + len(jpeg_cache[int(k)])
        with open(cache_bin, 'wb') as f:
            for k in keys:
                f.write(np.asarray(jpeg_cache[int(k)]).tobytes())
        np.savez(cache_idx, all_needed=keys, offsets=offs)
        print(f'  cache written to {cache_bin} ({offs[-1]/1024**3:.1f} GiB)', flush=True)

    # ---- student / teacher / stock reference ------------------------------------------
    student = AutoModel.from_pretrained(MODEL_ID).to(dev)
    student.requires_grad_(False)
    blocks = student.encoder.layer
    n = min(args.unfreeze_blocks, len(blocks))
    groups = []
    for depth, blk in enumerate(reversed(blocks[-n:])):
        blk.requires_grad_(True)
        groups.append({'params': list(blk.parameters()),
                       'lr': args.encoder_lr * (args.layerwise_decay ** depth)})
    student.layernorm.requires_grad_(True)
    groups.append({'params': list(student.layernorm.parameters()), 'lr': args.encoder_lr})
    # The released DINOv2 checkpoint ships mask_token at exactly zero -- it was never trained
    # and is not part of the distributed weights. Left frozen, masked positions would carry
    # only a positional embedding, an out-of-distribution input to the 10 frozen blocks that
    # the 2 trainable ones would have to repair. Training it (768 params) lets it settle on an
    # in-distribution "unknown patch" vector. It is saved with the checkpoint and is inert
    # downstream, where bool_masked_pos is never passed.
    student.embeddings.mask_token.requires_grad_(True)
    groups.append({'params': [student.embeddings.mask_token], 'lr': args.encoder_lr})

    teacher = deepcopy(student).eval()
    teacher.requires_grad_(False)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f'\nstudent: last {n}/{len(blocks)} blocks trainable, {trainable/1e6:.2f}M params '
          f'({100*trainable/sum(p.numel() for p in student.parameters()):.2f}% of the encoder)',
          flush=True)
    print(f'objective: masked patch feature regression vs EMA teacher | grid {grid}x{grid} '
          f'({grid*grid} patches), mask {args.mask_ratio:.0%} in blocks of >={args.mask_min_patches}, '
          f'targets = mean of top {args.target_layers} blocks', flush=True)
    print(f'encoder_lr={args.encoder_lr:g} layerwise_decay={args.layerwise_decay} '
          f'wd={args.weight_decay:g} epochs={args.n_epochs} warmup={args.warmup_epochs} '
          f'batch={args.batch_size} augment={args.augment}', flush=True)

    # Predictor: maps student patch tokens to the target space. Discarded after Stage A -- it is
    # part of the objective, not of the representation, and Stage B must not inherit it.
    predictor = nn.Sequential(nn.Linear(student.config.hidden_size, student.config.hidden_size),
                              nn.GELU(),
                              nn.Linear(student.config.hidden_size, student.config.hidden_size)).to(dev)
    groups.append({'params': list(predictor.parameters()), 'lr': args.encoder_lr * 10})

    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    steps_per_epoch = max(1, len(gidx) // args.batch_size)
    total_steps = steps_per_epoch * args.n_epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.01 + 0.5 * (1 - 0.01) * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    # ---- stock-encoder reference for the drift metric ---------------------------------
    # Computed once, then the reference encoder is freed: only the features are needed, and a
    # third 87M-param copy resident on the GPU for 5 h buys nothing.
    probe_ds = _FrameDataset(probe_gidx, [np.arange(len(probe_gidx))], jpeg_cache,
                             args.input_size, 'none', args.seed)
    probe_x = probe_ds[0].to(dev)
    stock = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    stock.requires_grad_(False)
    ref_feat = patch_features(stock, probe_x)
    ref_feat = F.normalize(ref_feat, dim=-1)
    del stock
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    print(f'drift probe: {len(probe_gidx)} val-pool frames, stock features cached\n', flush=True)

    run = None
    if args.wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=args.tag, config=vars(args) | {
                'n_frames': len(gidx), 'n_patches': grid * grid, 'trainable_params': trainable})
            print(f'wandb: {run.url}', flush=True)
        except Exception as e:        # never let telemetry kill a multi-hour run
            print(f'wandb disabled ({e.__class__.__name__}: {e})', flush=True)

    history = []
    gstep = 0
    for epoch in range(args.n_epochs):
        t0 = time.time()
        order = rng.permutation(len(gidx))
        batches = [order[i:i + args.batch_size]
                   for i in range(0, len(order) - args.batch_size + 1, args.batch_size)]
        loader = DataLoader(_FrameDataset(gidx, batches, jpeg_cache, args.input_size,
                                          args.augment, args.seed + epoch, args.photo_strength),
                            batch_size=None, num_workers=args.decode_workers,
                            pin_memory=(dev.type == 'cuda'), prefetch_factor=4)
        student.train()
        tot_loss = tot_cos = tot_std = nb = 0
        for x in loader:
            x = x.to(dev, non_blocking=True)
            mask = block_mask(len(x), grid, args.mask_ratio, args.mask_min_patches, rng).to(dev)
            with torch.amp.autocast('cuda', enabled=dev.type == 'cuda'):
                tgt = make_targets(teacher, x, args.target_layers)
                sh = student(pixel_values=x, bool_masked_pos=mask).last_hidden_state[:, 1:]
                pred = predictor(sh)
            pred, tgt = pred.float()[mask], tgt.float()[mask]
            loss = F.smooth_l1_loss(pred, tgt, beta=1.0)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g['params'] if p.requires_grad], 3.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            gstep += 1

            # EMA momentum -> 1.0 on a cosine: the target should stop moving as the student converges
            mom = 1 - (1 - args.ema_momentum) * 0.5 * (1 + math.cos(math.pi * gstep / total_steps))
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.mul_(mom).add_(sp.detach(), alpha=1 - mom)
                for tb, sb in zip(teacher.buffers(), student.buffers()):
                    tb.copy_(sb)

            with torch.no_grad():
                tot_loss += loss.item()
                tot_cos += F.cosine_similarity(pred, tgt, dim=-1).mean().item()
                tot_std += tgt.std(0).mean().item()
            nb += 1

        # ---- guards -------------------------------------------------------------------
        student.eval()
        cur = patch_features(student, probe_x)
        drift = float(1 - (F.normalize(cur, dim=-1) * ref_feat).sum(-1).mean())
        loss_e, cos_e, std_e = tot_loss / nb, tot_cos / nb, tot_std / nb
        print(f'epoch {epoch+1:>3}/{args.n_epochs}  loss={loss_e:.4f}  st_cos={cos_e:.4f}  '
              f'target_std={std_e:.4f}  drift={drift:.4f}  '
              f'lr={opt.param_groups[0]["lr"]:.2e}  ({time.time()-t0:.0f}s)', flush=True)
        history.append({'epoch': epoch + 1, 'loss': loss_e, 'st_cos': cos_e,
                        'target_std': std_e, 'drift': drift})
        if run is not None:
            run.log(history[-1])
        if std_e < args.collapse_std:
            raise SystemExit(
                f'COLLAPSE: target_std {std_e:.4f} < {args.collapse_std}. The teacher stopped '
                f'distinguishing patches, so every later epoch and Stage B would be measuring '
                f'a constant. Aborting instead of writing an uninterpretable checkpoint.')

        # Written every epoch: there is no best-checkpoint criterion (see docstring), so this is
        # the latest, and it doubles as a crash-resume point.
        train_keys = {k for k, prm in student.named_parameters() if prm.requires_grad}
        torch.save({k: v for k, v in student.state_dict().items() if k in train_keys},
                   OUT / 'best_encoder.pt')

    final = history[-1] if history else {}
    print(f'\nStage A done. drift={final.get("drift", float("nan")):.4f}', flush=True)
    # Both tails produce a Stage B number that reads like a clean result and is not one.
    if final.get('drift', 0) < 0.01:
        print('  WARNING: drift < 0.01 -- the representation barely moved. Stage B will very '
              'likely reproduce frozen_ctrl_s42 and would not be evidence about hypothesis (2). '
              'Raise --encoder-lr or --n-epochs before spending the GPU hours.', flush=True)
    elif final.get('drift', 0) > 0.60:
        print(f'  WARNING: drift {final["drift"]:.3f} is large -- the adapted features have '
              'little left in common with the ones every other arm was measured on. A Stage B '
              'number BELOW frozen_ctrl_s42 should be read as "this LR destroyed the encoder", '
              'not as "unlabelled adaptation does not help". Re-run at a lower --encoder-lr '
              'before concluding anything about hypothesis (2).', flush=True)
    with open(OUT / 'ssl_config.json', 'w') as f:
        json.dump({'cfg': vars(args), 'n_frames': int(len(gidx)),
                   'n_observations': int(sel.observation_id.nunique()),
                   'n_pools': int(sel['pool'].nunique()),
                   'n_frames_unlabeled_obs': n_unlab,
                   'val_pools_excluded': sorted(VAL_POOLS_V1),
                   'n_patches': grid * grid, 'trainable_params': int(trainable),
                   'history': history}, f, indent=2)
    print(f'Saved {OUT}/best_encoder.pt + ssl_config.json', flush=True)
    print(f'\nStage B:\n  TAG=res448_k2_frozen_d4photo_sslinit '
          f'INIT_ENCODER={OUT}/best_encoder.pt \\\n    UNFREEZE_BLOCKS=0 '
          f'sbatch scripts/mice_behavior/train_online_aug.sh', flush=True)
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
