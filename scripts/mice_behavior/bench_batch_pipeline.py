"""Locate the real bottleneck in the patchgrid training loop.

Motivation: the trainable model is only ~520k parameters, yet an epoch takes 5-20 minutes.
Per batch (512 x 5 frames x 1024 patches x 768 dim) the model does ~1 TFLOP -- roughly 20 ms
on an A40 -- while the batch tensor itself is 3.75 GiB, and the current code calls .float()
on it before the forward pass, materialising a second 7.5 GiB fp32 copy that torch.autocast
immediately casts back down to fp16. So the loop is almost certainly memory-bound, not
compute-bound, and the parameter count is irrelevant to its cost.

Times each stage separately and tests three candidate fixes:
  1. drop .float()          -- pass fp16 straight into autocast
  2. pinned staging buffer  -- non-pinned host memory forces a synchronous copy
  3. both

Usage: python scripts/mice_behavior/bench_batch_pipeline.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.model import MouseFrameClassifier

B, T, P, D = 512, 5, 1024, 768
N_CACHE = 40_000          # stand-in for the real cache (same access pattern, less RAM)
N_ITERS = 8


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {dev}  ({torch.cuda.get_device_name(0) if dev.type=="cuda" else "cpu"})', flush=True)
    print(f'batch tensor: {B*T*P*D*2/1024**3:.2f} GiB fp16 | {B*T*P*D*4/1024**3:.2f} GiB fp32\n', flush=True)

    cache = torch.empty((N_CACHE, P, D), dtype=torch.float16)
    cache.normal_()
    model = MouseFrameClassifier(emb_dim=D, n_heads=8, hidden_dim=384, use_patch_grid=True,
                                 dropout=0.4, cross_attn_dim=64, patch_pool_dim=256).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    crit = torch.nn.BCEWithLogitsLoss()
    offs = torch.from_numpy(np.broadcast_to(np.arange(-2, 3), (B, T)).copy()).to(dev)
    mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
    lbl = torch.randint(0, 2, (B, 2), dtype=torch.float32, device=dev)
    pinned = torch.empty((B * T, P, D), dtype=torch.float16, pin_memory=(dev.type == 'cuda'))

    def run(tag, use_pinned, upcast):
        t_gather = t_xfer = t_fwd = t_bwd = 0.0
        for it in range(N_ITERS):
            pos = np.random.randint(0, N_CACHE, size=B * T)
            sync(); t0 = time.perf_counter()
            g = cache[pos]                                   # CPU gather
            sync(); t1 = time.perf_counter()
            if use_pinned:
                pinned.copy_(g)
                ctx = pinned.to(dev, non_blocking=True).view(B, T, P, D)
            else:
                ctx = g.to(dev, non_blocking=True).view(B, T, P, D)
            sync(); t2 = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                out = model(ctx.float() if upcast else ctx, offsets=offs, key_padding_mask=mask)
                loss = crit(out, lbl)
            sync(); t3 = time.perf_counter()
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sync(); t4 = time.perf_counter()
            if it >= 2:                                       # skip warmup
                t_gather += t1-t0; t_xfer += t2-t1; t_fwd += t3-t2; t_bwd += t4-t3
        n = N_ITERS - 2
        tot = (t_gather+t_xfer+t_fwd+t_bwd)/n
        print(f'{tag:<28} gather {t_gather/n*1000:7.0f}ms | h2d {t_xfer/n*1000:7.0f}ms | '
              f'fwd {t_fwd/n*1000:6.0f}ms | bwd {t_bwd/n*1000:6.0f}ms | TOTAL {tot*1000:7.0f}ms '
              f'({148*tot/60:.1f} min/epoch @148 batches)', flush=True)
        return tot

    base = run('current (.float, unpinned)', False, True)
    a = run('no .float()', False, False)
    b = run('pinned only', True, True)
    c = run('no .float() + pinned', True, False)
    print(f'\nspeedups vs current: no-float {base/a:.2f}x | pinned {base/b:.2f}x | both {base/c:.2f}x', flush=True)
    if dev.type == 'cuda':
        print(f'peak GPU mem: {torch.cuda.max_memory_allocated()/1024**3:.1f} GiB', flush=True)


if __name__ == '__main__':
    main()
