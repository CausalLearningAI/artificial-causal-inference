"""
Train a Matryoshka BatchTopK SAE on final-layer DINOv2 patch tokens (ECI step 2).

Split by POOL: --n-val-pools-per-genotype het pools + the same number of wt pools
(chosen with --split-seed) are held out for validation; the SAE trains on the
patch tokens of all other pools.

Two token stores (src/eci/token_store.py):
    single file (v1, 64 frames/obs)  all train tokens are loaded on the GPU as fp16.
    sharded (v2, 1 fps, ~133M tokens) tokens are streamed from disk in chunks of
        --chunk-frames randomly ordered frames (a uniform random sample each), in a
        seeded random chunk order per epoch, shuffled within the chunk on the GPU.

Output (default):
    dataset/mice/v1/eci/sae/matryoshka_btk_{n_latents}_k{k}_{tag or ep{epochs}}_s{seed}/
        sae.pt        state dict + TokenNorm stats + hparams
        metrics.json  hparams, split, training history, validation metrics
                      (inference threshold and exact per-token top-k)

Usage:
    python scripts/eci/train_sae.py --seed 0
    python scripts/eci/train_sae.py --seed 0 --epochs 0.05 --out-dir /tmp/sae_test   # quick test
    python scripts/eci/train_sae.py --seed 0 --epochs 2 --tag fps1 \
        --tokens-dir dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fps1              # v2 (streamed)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.sae import (MatryoshkaBatchTopKSAE, TokenNorm, evaluate_sae, geometric_median,  # noqa: E402
                         save_checkpoint, train_sae, train_sae_stream)
from src.eci.token_store import TokenStore  # noqa: E402


def pool_split(meta, n_per_genotype=4, seed=0):
    pools = meta.groupby('pool').genotype.first()
    rng = np.random.default_rng(seed)
    val = []
    for gt in ('het', 'wt'):
        cand = sorted(pools[pools == gt].index.tolist())
        val += sorted(rng.choice(cand, n_per_genotype, replace=False).tolist())
    return val


def load_frames(path, frame_mask, device, chunk=1024):
    """Patch tokens of the frames where frame_mask is True -> (n*256, d) fp16 on device."""
    mm = np.load(path, mmap_mode='r')
    n_sel = int(frame_mask.sum())
    out = torch.empty((n_sel * mm.shape[1], mm.shape[2]), dtype=torch.float16, device=device)
    cur = 0
    for f0 in range(0, mm.shape[0], chunk):
        sel = np.nonzero(frame_mask[f0:f0 + chunk])[0]
        if len(sel) == 0:
            continue
        block = np.ascontiguousarray(mm[f0:f0 + chunk][sel]).reshape(-1, mm.shape[2])
        out[cur:cur + len(block)] = torch.from_numpy(block).to(device)
        cur += len(block)
    assert cur == out.shape[0]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens-dir', default=str(REPO / 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1'))
    p.add_argument('--out-dir', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--n-val-pools-per-genotype', type=int, default=4)
    p.add_argument('--n-latents', type=int, default=1024)
    p.add_argument('--prefixes', default='128,256,512,1024')
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--k-aux', type=int, default=512)
    p.add_argument('--aux-coef', type=float, default=1 / 32)
    p.add_argument('--dead-tokens', type=int, default=2_000_000)
    p.add_argument('--epochs', type=float, default=20)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--tag', default=None, help='name tag replacing ep{epochs} in the output dir (e.g. fps1)')
    p.add_argument('--chunk-frames', type=int, default=16384, help='(sharded store) frames per streamed chunk')
    p.add_argument('--norm-frames', type=int, default=4000, help='(sharded store) random train frames for norm/b_dec init')
    p.add_argument('--max-chunks', type=int, default=None, help='(sharded store) only the first K chunks (testing)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        REPO / 'dataset/mice/v1/eci/sae' / \
        f'matryoshka_btk_{args.n_latents}_k{args.k}_{args.tag or f"ep{args.epochs:g}"}_s{args.seed}'
    if (out_dir / 'metrics.json').exists() and not args.overwrite:
        print(f'[SKIP] {out_dir} exists (use --overwrite)')
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokens_dir = Path(args.tokens_dir)

    store = TokenStore(tokens_dir)
    meta = store.meta
    val_pools = pool_split(meta, args.n_val_pools_per_genotype, args.split_seed)
    is_val = meta.pool.isin(val_pools).values
    print(f'val pools: {val_pools}  ({is_val.sum()} val frames, {(~is_val).sum()} train frames)', flush=True)
    prefixes = [int(x) for x in args.prefixes.split(',')]
    t0 = time.time()

    if not store.sharded:  # ------------------------------------------------ v1: GPU-resident
        train_tok = load_frames(tokens_dir / 'patch_tokens.npy', ~is_val, device)
        val_tok = load_frames(tokens_dir / 'patch_tokens.npy', is_val, device)
        print(f'loaded {train_tok.shape[0]:,} train / {val_tok.shape[0]:,} val tokens in {time.time() - t0:.0f}s',
              flush=True)
        norm = TokenNorm(train_tok.shape[1]).to(device).fit(train_tok, seed=args.seed)
        n_epochs_full = int(np.ceil(args.epochs))
        # fractional epochs (testing): subsample the train tokens
        if args.epochs < 1:
            n_keep = int(train_tok.shape[0] * args.epochs)
            train_tok = train_tok[torch.randperm(train_tok.shape[0], device=device)[:n_keep]]
        n_train_tokens = int(train_tok.shape[0])
        sae = MatryoshkaBatchTopKSAE(d_in=train_tok.shape[1], n_latents=args.n_latents, prefixes=prefixes, k=args.k,
                                     k_aux=args.k_aux, aux_coef=args.aux_coef, dead_tokens=args.dead_tokens,
                                     seed=args.seed).to(device)
        g = torch.Generator(device='cpu').manual_seed(args.seed)
        sub = torch.randperm(train_tok.shape[0], generator=g)[:100_000].to(device)
        with torch.no_grad():
            sae.b_dec.copy_(geometric_median(norm(train_tok[sub])))
        print(f'norm scale {float(norm.scale):.4f}, |b_dec init| {float(sae.b_dec.norm()):.3f}', flush=True)
        history = train_sae(sae, norm, train_tok, epochs=n_epochs_full, batch_size=args.batch_size, lr=args.lr,
                            warmup=args.warmup, grad_clip=args.grad_clip, seed=args.seed,
                            log_fn=lambda s: print(s, flush=True))
        extra_metrics = {}
    else:  # ---------------------------------------------------------------- v2: streamed
        if args.epochs != int(args.epochs):
            raise ValueError('sharded store: --epochs must be an integer (use --max-chunks to test)')
        rng = np.random.default_rng(args.seed)
        train_idx = np.nonzero(~is_val)[0]
        init_idx = np.sort(rng.choice(train_idx, min(args.norm_frames, len(train_idx)), replace=False))
        init_tok = torch.from_numpy(store.frames(init_idx).reshape(-1, store.dim)).to(device)
        norm = TokenNorm(store.dim).to(device).fit(init_tok, seed=args.seed)
        sae = MatryoshkaBatchTopKSAE(d_in=store.dim, n_latents=args.n_latents, prefixes=prefixes, k=args.k,
                                     k_aux=args.k_aux, aux_coef=args.aux_coef, dead_tokens=args.dead_tokens,
                                     seed=args.seed).to(device)
        g = torch.Generator(device='cpu').manual_seed(args.seed)
        sub = torch.randperm(init_tok.shape[0], generator=g)[:100_000].to(device)
        with torch.no_grad():
            sae.b_dec.copy_(geometric_median(norm(init_tok[sub])))
        del init_tok
        print(f'norm/b_dec from {len(init_idx)} random train frames: norm scale {float(norm.scale):.4f}, '
              f'|b_dec init| {float(sae.b_dec.norm()):.3f}', flush=True)

        chunk_tokens = store.n_selected_tokens_per_chunk(~is_val, args.chunk_frames)
        n_chunks = len(chunk_tokens) if args.max_chunks is None else args.max_chunks

        def chunk_iter(ep):
            it = store.iter_chunks(~is_val, args.chunk_frames, order_seed=args.seed * 1000 + ep)
            for n, (ci, block) in enumerate(it):
                if n >= n_chunks:
                    break
                yield ci, block

        if args.max_chunks is not None:  # schedule length must match what is streamed
            order = [np.random.default_rng(args.seed * 1000 + ep).permutation(len(chunk_tokens))[:n_chunks]
                     for ep in range(int(args.epochs))]
            if any(set(o) != set(order[0]) for o in order):
                raise ValueError('--max-chunks with several epochs picks different chunks; use --epochs 1')
            ct = np.zeros_like(chunk_tokens)
            ct[order[0]] = chunk_tokens[order[0]]
            chunk_tokens = ct
        n_train_tokens = int(chunk_tokens.sum())
        print(f'{len(chunk_tokens)} chunks of {args.chunk_frames} frames, {n_train_tokens:,} train tokens/epoch, '
              f'{int(args.epochs) * int(sum(int(n) // args.batch_size for n in chunk_tokens)):,} steps', flush=True)
        history = train_sae_stream(sae, norm, chunk_iter, chunk_tokens, epochs=int(args.epochs),
                                   batch_size=args.batch_size, lr=args.lr, warmup=args.warmup,
                                   grad_clip=args.grad_clip, seed=args.seed, log_fn=lambda s: print(s, flush=True))
        tv = time.time()
        val_idx = np.nonzero(is_val)[0]
        val_tok = torch.from_numpy(store.frames(val_idx).reshape(-1, store.dim))
        print(f'loaded {val_tok.shape[0]:,} val tokens ({len(val_idx)} frames) in {time.time() - tv:.0f}s', flush=True)
        extra_metrics = {'n_steps': len(history) and history[-1]['step'] + 1, 'chunk_frames': args.chunk_frames}
    train_time = time.time() - t0
    sae.eval()
    save_checkpoint(out_dir / 'sae.pt', sae, norm, extra={'val_pools': val_pools, 'args': vars(args)})

    val_thr = evaluate_sae(sae, norm, val_tok, mode='threshold')
    val_topk = evaluate_sae(sae, norm, val_tok, mode='topk')
    for name, ev in (('threshold', val_thr), ('topk', val_topk)):
        print(f'VAL ({name}) on {ev["n_tokens"]:,} tokens / {ev["n_frames"]} frames:')
        for m, r in ev['prefixes'].items():
            print(f'  m={m:>5}  FVE {r["fve"]:.4f}  L0/token {r["l0_per_token"]:6.2f}  '
                  f'dead {100 * r["dead_frac"]:5.1f}% ({r["n_dead"]})  pooled-frame L0 {r["pooled_frame_l0_mean"]:.1f}')
    metrics = {'args': vars(args), 'val_pools': val_pools, 'n_train_tokens': n_train_tokens, **extra_metrics,
               'n_val_tokens': int(val_tok.shape[0]), 'norm_scale': float(norm.scale),
               'threshold': float(sae.threshold), 'train_time_s': round(train_time, 1),
               'val_threshold': val_thr, 'val_topk': val_topk, 'history': history}
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    print(f'Done -> {out_dir}')


if __name__ == '__main__':
    main()
