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
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import get_fixed_val_pools
from src.mice_behavior.metrics import ap_report, format_ap_report, rate_report, format_rate_report
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

    def __init__(self, meta, batches, jpeg_cache, pos_of_global, input_size, augment, seed):
        self.meta, self.batches, self.cache = meta, batches, jpeg_cache
        self.pos_of_global, self.input_size = pos_of_global, input_size
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
        for b in range(B):
            # one transform per SAMPLE, shared by all T context frames -- per-frame transforms
            # would break the temporal relationship nose-tail detection relies on.
            op = int(rng.integers(0, 8)) if self.augment == 'd4' else 0
            for t in range(T):
                if mask[b, t]:
                    continue
                buf = self.cache[self.pos_of_global[int(abs_idx[b, t])]]
                with Image.open(io.BytesIO(buf.tobytes())) as im:
                    im = im.convert('RGB')
                    if op:
                        im = d4_transform(im, op)
                    im = im.resize((S, S), Image.BILINEAR)
                    arr = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())
                out[b, t] = ((arr.permute(2, 0, 1).float() / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
        return (out,
                torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()),
                torch.from_numpy(self.meta.labels[sample_idx]),
                torch.from_numpy(mask))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--context-k', type=int, default=2)
    p.add_argument('--neg-ratio', type=int, default=1)
    p.add_argument('--max-train-frames', type=int, default=300_000)
    p.add_argument('--input-size', type=int, default=224)
    p.add_argument('--augment', choices=['none', 'd4'], default='d4')
    p.add_argument('--n-epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--read-workers', type=int, default=32, help='NFS reads: latency-bound, use many')
    p.add_argument('--decode-workers', type=int, default=16, help='decode+augment: CPU-bound, ~= n_cpus')
    p.add_argument('--val-monitor-size', type=int, default=25_000)
    p.add_argument('--cross-attn-dim', type=int, default=64)
    p.add_argument('--patch-pool-dim', type=int, default=256)
    p.add_argument('--lr-decay-epochs', type=int, default=6)
    p.add_argument('--tag', type=str, default='online_aug')
    p.add_argument('--smoke', action='store_true',
                    help='tiny end-to-end check: caps train AND val so the (NFS-bound) read phase '
                         'takes ~1 min instead of ~30, surfacing pipeline bugs before a real run')
    args = p.parse_args()
    if args.smoke:
        args.max_train_frames, args.n_epochs, args.batch_size = 4_000, 2, 32
        args.val_monitor_size, args.tag = 600, 'online_aug_smoke'

    n_patches = (args.input_size // PATCH_SIZE) ** 2
    OUT = gsf.FRAME_DIR / f'patchgrid256_dinov2_{args.tag}'
    OUT.mkdir(parents=True, exist_ok=True)
    best_cfg = json.load(open(gsf.FRAME_DIR / 'patchgrid4x4_dinov2' / 'config.json'))['cfg']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'context_k={args.context_k} (T={2*args.context_k+1})  input_size={args.input_size} '
          f'({n_patches} patches)  augment={args.augment}  neg_ratio={args.neg_ratio}', flush=True)

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
                        dummy_loader(1, 1), n_patches=1, stride=1,
                        max_frames=args.max_train_frames, seed=gsf.SEED)
    vm = FrameBatchData(str(ann_csv), str(pair_labels), val_obs, args.context_k, 1,
                        dummy_loader(1, 1), n_patches=1, stride=1)
    del tm.flat, vm.flat

    pos_idx = np.where(tm.labels.sum(1) > 0)[0]
    neg_idx = np.where(tm.labels.sum(1) == 0)[0]
    n_neg = min(len(neg_idx), args.neg_ratio * max(len(pos_idx), 1))
    saturated = n_neg >= len(neg_idx)
    print(f'{args.neg_ratio}:1 sampling: {len(pos_idx):,} pos, {len(neg_idx):,} neg available -> '
          f'{"SATURATED" if saturated else f"fresh {n_neg:,} each epoch"}', flush=True)

    v_rng = np.random.default_rng(gsf.SEED)
    val_keep = np.sort(v_rng.choice(len(vm), size=min(len(vm), args.val_monitor_size), replace=False))
    vp = vm.labels[val_keep]
    print(f'val monitor {len(val_keep):,} (prevalence-preserving: nt {100*vp[:,0].mean():.2f}% '
          f'nn {100*vp[:,1].mean():.2f}%)', flush=True)

    def needed(meta, si):
        a = meta.gi[si][:, None] + meta.offsets_grid[None, :]
        return np.unique(a[~meta.pad_mask[si]])

    all_needed = np.unique(np.concatenate([
        needed(tm, np.concatenate([pos_idx, neg_idx])),
        needed(vm, np.arange(len(vm)))]))
    pos_of_global = {int(g): i for i, g in enumerate(all_needed)}

    ann = pd.read_csv(ann_csv, usecols=['frame_path'])
    paths = ann.frame_path.values[all_needed]
    print(f'Reading {len(all_needed):,} JPEGs into RAM ({args.read_workers} workers; '
          f'NFS-latency-bound, paid ONCE)...', flush=True)
    t0 = time.time()
    jpeg_cache = [None] * len(all_needed)
    loader = DataLoader(_BytesReader(paths), batch_size=None, num_workers=args.read_workers,
                        prefetch_factor=6, collate_fn=lambda x: x)
    nbytes, last = 0, (0, 0.0)
    for n, (i, buf) in enumerate(loader, 1):
        jpeg_cache[i] = buf
        nbytes += buf.nbytes
        if n % 20000 == 0 or n == len(all_needed):
            el = time.time() - t0
            inst = (n - last[0]) / max(el - last[1], 1e-9); last = (n, el)
            print(f'  {n:,}/{len(all_needed):,} ({inst:.0f} f/s now, {n/el:.0f} cum)', flush=True)
    print(f'JPEG cache: {nbytes/1024**3:.1f} GiB in {(time.time()-t0)/60:.1f} min', flush=True)

    encoder = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    encoder.requires_grad_(False)
    model = MouseFrameClassifier(
        emb_dim=EMB_DIM, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'],
        use_patch_grid=True, dropout=best_cfg['dropout'],
        cross_attn_dim=args.cross_attn_dim or None, patch_pool_dim=args.patch_pool_dim or None).to(dev)
    print(f'classifier params: {sum(p.numel() for p in model.parameters()):,} '
          f'(DINOv2 frozen, {sum(p.numel() for p in encoder.parameters())/1e6:.0f}M)', flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=best_cfg['lr'], weight_decay=best_cfg['weight_decay'])
    eta = 0.01
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: eta if e >= args.lr_decay_epochs else
                                              eta + 0.5*(1-eta)*(1+math.cos(math.pi*e/args.lr_decay_epochs)))
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')

    def make_loader(meta, order, augment, seed):
        batches = [order[i:i+args.batch_size] for i in range(0, len(order), args.batch_size)]
        return DataLoader(_SampleDataset(meta, batches, jpeg_cache, pos_of_global,
                                         args.input_size, augment, seed),
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

    rng = np.random.default_rng(gsf.SEED)
    best, since, hist = -1.0, 0, []
    for ep in range(1, args.n_epochs + 1):
        model.train()
        order = (np.concatenate([pos_idx, neg_idx]) if saturated else
                 np.concatenate([pos_idx, rng.choice(neg_idx, size=n_neg, replace=False)]))
        rng.shuffle(order)
        tot, seen, t0 = 0.0, 0, time.time()
        for imgs, offs, lbl, mask in make_loader(tm, order, args.augment, gsf.SEED * 1000 + ep):
            imgs, lbl = imgs.to(dev, non_blocking=True), lbl.to(dev, non_blocking=True)
            B, T = imgs.shape[:2]
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                with torch.no_grad():   # encoder frozen: no graph, no backward through DINOv2
                    tok = encoder(pixel_values=imgs.view(B*T, *imgs.shape[2:])).last_hidden_state[:, 1:]
                logits = model(tok.view(B, T, n_patches, EMB_DIM).detach(),
                               offsets=offs.to(dev), key_padding_mask=mask.to(dev))
                loss = crit(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt); scaler.update()
            tot += loss.item() * B; seen += B
        sched.step()
        probs, labs = evaluate(val_keep)
        ap = float(np.mean([average_precision_score(labs[:, i], probs[:, i]) for i in (0, 1)]))
        hist.append({'epoch': ep, 'train_loss': tot/seen, 'monitor_ap': ap})
        print(f'epoch {ep:3d}/{args.n_epochs}  loss={tot/seen:.4f}  monitor_ap={ap:.4f}  '
              f'lr={opt.param_groups[0]["lr"]:.2e}  ({time.time()-t0:.1f}s)', flush=True)
        if ap > best:
            best, since = ap, 0
            torch.save(model.state_dict(), OUT / 'best_model.pt')
        else:
            since += 1
            if since >= args.patience:
                print(f'early stopping (no improvement for {args.patience})', flush=True)
                break

    model.load_state_dict(torch.load(OUT / 'best_model.pt', map_location=dev, weights_only=True))
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
    print()
    print(format_rate_report(rr))
    json.dump({'cfg': best_cfg, 'context_k': args.context_k, 'input_size': args.input_size,
               'n_patches': n_patches, 'augment': args.augment, 'neg_ratio': args.neg_ratio,
               'max_train_frames': args.max_train_frames, 'val_pools': sorted(val_pools),
               'jpeg_cache_gib': nbytes/1024**3, 'ap_report': apr, 'rate_report': rr,
               'best_ap': apr['macro/tol0']['ap'], 'history': hist},
              open(OUT / 'config.json', 'w'), indent=2)
    print(f'\nSaved {OUT}/', flush=True)


if __name__ == '__main__':
    main()
