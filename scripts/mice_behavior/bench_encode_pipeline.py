"""Is the one-time encode phase GPU-bound or DataLoader-bound?

Evidence it is NOT GPU-bound: the same code at 448px achieves ~36% MFU on an H100 but only
7-11% on A100/L40S/A40/A10. If DINOv2 were the limit, all of them would sit near their own
compute roof. A 5x spread across GPUs whose fp16 peaks differ by ~3x points at the CPU-side
input pipeline (arrow random-access read -> JPEG decode -> resize -> normalise) instead.

Measures the two halves in isolation:
  A) DataLoader only  -- iterate batches, never touch the GPU, sweep num_workers
  B) GPU only         -- feed the encoder pre-made tensors, no dataloader at all
whichever is slower is the ceiling for the combined pipeline.

Usage: python scripts/mice_behavior/bench_encode_pipeline.py --input-size 448
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.dataset.get_dataset import load_dataset
from train_patchgrid_online import _ImageDataset

MODEL_ID = 'facebook/dinov2-base'

p = argparse.ArgumentParser()
p.add_argument('--input-size', type=int, default=448)
p.add_argument('--batch-size', type=int, default=256)
p.add_argument('--n-batches', type=int, default=12)
p.add_argument('--workers', type=int, nargs='+', default=[8, 16, 32])
args = p.parse_args()

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {torch.cuda.get_device_name(0) if dev.type=="cuda" else "cpu"}  '
      f'input_size={args.input_size}  batch={args.batch_size}', flush=True)
import os
print(f'CPUs visible to this job: {len(os.sched_getaffinity(0))}\n', flush=True)

processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
hf = load_dataset(subject='mice', version='v1', dataset_root=str(gsf.DATASET_DIR), frame_type='full')

# Each config MUST read a disjoint, previously-untouched slice of frames. An earlier version
# reused the same indices for every config, so configs 2+ were served from the OS page cache
# and reported ~4600 fps -- warm-RAM throughput, not the cold read the real 480k-frame encode
# performs. That produced an impossible 31x jump from 8 to 16 workers.
per_cfg = args.batch_size * (args.n_batches + 2)
n_avail = len(hf)
assert n_avail >= per_cfg * len(args.workers), f'need {per_cfg*len(args.workers):,} frames, have {n_avail:,}'
slices = {nw: np.arange(i * per_cfg, (i + 1) * per_cfg) for i, nw in enumerate(args.workers)}
print(f'(each config reads its own disjoint {per_cfg:,}-frame slice, cold)\n', flush=True)

# ---- A) DataLoader only: no GPU work at all ----
print('A) DataLoader ONLY (decode+resize+normalise, GPU untouched):', flush=True)
best_dl = 0.0
for nw in args.workers:
    dl = DataLoader(_ImageDataset(hf, slices[nw], processor, input_size=args.input_size),
                    batch_size=args.batch_size, num_workers=nw, shuffle=False,
                    pin_memory=False, prefetch_factor=4 if nw else None, persistent_workers=bool(nw))
    it = iter(dl)
    next(it)                                    # warm up workers
    t0, n = time.perf_counter(), 0
    for _ in range(args.n_batches):
        try:
            b = next(it)
        except StopIteration:
            break
        n += b.shape[0]
    dt = time.perf_counter() - t0
    fps = n / dt
    best_dl = max(best_dl, fps)
    print(f'   num_workers={nw:<3} -> {fps:7.1f} frames/s', flush=True)
    del it, dl

# ---- B) GPU only: synthetic tensors, no dataloader ----
print('\nB) GPU ONLY (encoder fed pre-made tensors, no dataloader):', flush=True)
enc = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
enc.requires_grad_(False)
x = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=dev, dtype=torch.float32)
with torch.inference_mode():
    for _ in range(3):
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
            enc(pixel_values=x)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.n_batches):
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
            enc(pixel_values=x)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    gpu_fps = args.batch_size * args.n_batches / (time.perf_counter() - t0)
print(f'   {gpu_fps:7.1f} frames/s', flush=True)

print(f'\nVERDICT: dataloader ceiling {best_dl:.0f} fps | GPU ceiling {gpu_fps:.0f} fps')
if best_dl < gpu_fps:
    print(f'  -> DATALOADER-BOUND: the GPU could do {gpu_fps/best_dl:.1f}x more work than the '
          f'input pipeline can feed it. Encode time is set by CPU decode, not by DINOv2.')
else:
    print(f'  -> GPU-BOUND: the input pipeline can feed {best_dl/gpu_fps:.1f}x what the GPU consumes.')
