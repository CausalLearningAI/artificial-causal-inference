"""
Extract final-layer DINOv2 patch tokens + CLS for a training subset of mice v1 frames
(for training the ECI sparse autoencoder).

Two sampling modes:
    --n-per-obs N (default 64, v1)  N frames per observation, evenly spread in time with
                                    seeded jitter. One output directory:
        dataset/mice/v1/eci/train_tokens/dinov2_base_l-1/
            patch_tokens.npy  float16 (N, 256, 768)
            cls.npy           float16 (N, 768)
            metadata.parquet  row_idx (into annotations.csv), observation_id, frame_idx, pool, ...
            config.json
    --stride S (v2: S=5 = 1 fps)    every S-th frame of every observation (seeded offset).
                                    The frames are put in a seeded random order and split
                                    into --n-shards contiguous shards (one SLURM array task
                                    each), so every contiguous block of rows is a uniform
                                    random sample of frames. Then --finalize writes the
                                    concatenated metadata (see src/eci/token_store.py):
        dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fps1/
            shards/shard_XX/{patch_tokens.npy, cls.npy, metadata.parquet, config.json, DONE}
            metadata.parquet  all frames in stored order (+ shard, shard_row)
            config.json

Usage:
    python scripts/eci/extract_train_tokens.py                                   # v1
    python scripts/eci/extract_train_tokens.py --stride 5 --n-shards 16 --shard 3
    python scripts/eci/extract_train_tokens.py --stride 5 --n-shards 16 --finalize
    python scripts/eci/extract_train_tokens.py --max-obs 2 --out-dir /tmp/test   # quick test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.extract import build_frame_table, extract_tokens, sample_frames, sample_frames_stride  # noqa: E402


def sharded_frames(args):
    """All sampled frames in the stored (seeded random) order, with shard / shard_row."""
    table = build_frame_table(args.dataset_dir, args.data_dir)
    print(f'  {table.observation_id.nunique()} observations, {len(table):,} frames, '
          f'{(table.row_idx < 0).sum():,} frames not in annotations.csv', flush=True)
    if args.max_obs:
        keep = sorted(table.observation_id.unique())[:args.max_obs]
        table = table[table.observation_id.isin(keep)]
    frames = sample_frames_stride(table, args.stride, args.seed)
    perm = np.random.default_rng(args.seed + 1).permutation(len(frames))
    frames = frames.iloc[perm].reset_index(drop=True)
    edges = np.linspace(0, len(frames), args.n_shards + 1).round().astype(np.int64)
    shard = np.searchsorted(edges, np.arange(len(frames)), side='right') - 1
    frames['shard'] = shard
    frames['shard_row'] = np.arange(len(frames)) - edges[shard]
    print(f'  sampled {len(frames):,} frames (stride {args.stride}, seed {args.seed}), '
          f'{args.n_shards} shards', flush=True)
    return frames, edges


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--data-dir', default=str(REPO / 'data'))
    p.add_argument('--encoder', default='dinov2_base')
    p.add_argument('--resolution', type=int, default=224)
    p.add_argument('--n-per-obs', type=int, default=64)
    p.add_argument('--stride', type=int, default=None, help='every stride-th frame per obs (5 = 1 fps); sharded output')
    p.add_argument('--n-shards', type=int, default=16)
    p.add_argument('--shard', type=int, default=None)
    p.add_argument('--finalize', action='store_true', help='(stride mode) write the concatenated metadata')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-obs', type=int, default=None, help='only the first K observations (testing)')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    suffix = '' if args.resolution == 224 else f'_r{args.resolution}'
    if args.stride is not None:
        suffix += '_fps1' if args.stride == 5 else f'_stride{args.stride}'
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(args.dataset_dir) / 'mice' / 'v1' / 'eci' / 'train_tokens' / f'{args.encoder}_l-1{suffix}'

    if args.stride is None:  # ---------------------------------------------- v1 mode
        if out_dir.exists() and not args.overwrite:
            print(f'[SKIP] {out_dir} exists (use --overwrite)')
            return
        print('Scanning frames on disk...', flush=True)
        table = build_frame_table(args.dataset_dir, args.data_dir)
        print(f'  {table.observation_id.nunique()} observations, {len(table):,} frames, '
              f'{(table.row_idx < 0).sum():,} frames not in annotations.csv', flush=True)
        if args.max_obs:
            keep = sorted(table.observation_id.unique())[:args.max_obs]
            table = table[table.observation_id.isin(keep)]
        frames = sample_frames(table, args.n_per_obs, args.seed)
        print(f'  sampled {len(frames):,} frames ({args.n_per_obs}/obs, seed {args.seed})', flush=True)
        extract_tokens(frames, out_dir, dataset_dir=args.dataset_dir, encoder=args.encoder,
                       resolution=args.resolution, batch_size=args.batch_size,
                       num_workers=args.num_workers, device=args.device,
                       extra_config={'n_per_obs': args.n_per_obs, 'seed': args.seed})
        return

    # ------------------------------------------------------------------- sharded mode
    if (out_dir / 'metadata.parquet').exists() and not args.overwrite:
        print(f'[SKIP] {out_dir} already finalized (use --overwrite)')
        return
    print('Scanning frames on disk...', flush=True)
    frames, edges = sharded_frames(args)
    shard_cfg = {'stride': args.stride, 'seed': args.seed, 'order': 'seeded random permutation (seed+1)',
                 'n_shards': args.n_shards, 'n_frames_total': len(frames)}
    if args.shard is not None:
        shard_dir = out_dir / 'shards' / f'shard_{args.shard:02d}'
        if (shard_dir / 'DONE').exists() and not args.overwrite:
            print(f'[SKIP] {shard_dir} done')
        else:
            sel = frames[frames.shard == args.shard].reset_index(drop=True)
            print(f'shard {args.shard}: rows [{edges[args.shard]}, {edges[args.shard + 1]}) = {len(sel):,} frames',
                  flush=True)
            extract_tokens(sel, shard_dir, dataset_dir=args.dataset_dir, encoder=args.encoder,
                           resolution=args.resolution, batch_size=args.batch_size,
                           num_workers=args.num_workers, device=args.device,
                           extra_config={**shard_cfg, 'shard': args.shard})
            (shard_dir / 'DONE').touch()
    if args.finalize:
        parts = []
        for i in range(args.n_shards):
            d = out_dir / 'shards' / f'shard_{i:02d}'
            if not (d / 'DONE').exists():
                raise RuntimeError(f'{d} not done')
            m = pd.read_parquet(d / 'metadata.parquet')
            ref = frames[frames.shard == i].reset_index(drop=True)
            if not (m.frame_path.values == ref.frame_path.values).all():
                raise RuntimeError(f'shard {i} frame order does not match the sampling')
            n_tok = np.load(d / 'patch_tokens.npy', mmap_mode='r').shape[0]
            if n_tok != len(m):
                raise RuntimeError(f'shard {i}: {n_tok} token rows vs {len(m)} metadata rows')
            parts.append(m)
        meta = pd.concat(parts, ignore_index=True)
        meta.to_parquet(out_dir / 'metadata.parquet', index=False)
        cfg = json.loads((out_dir / 'shards' / 'shard_00' / 'config.json').read_text())
        cfg.pop('shard', None)
        cfg.pop('elapsed_s', None)
        cfg['n_frames'] = len(meta)
        cfg['shard_elapsed_s'] = [json.loads((out_dir / 'shards' / f'shard_{i:02d}' / 'config.json').read_text())
                                  ['elapsed_s'] for i in range(args.n_shards)]
        (out_dir / 'config.json').write_text(json.dumps(cfg, indent=2))
        print(f'finalized {len(meta):,} frames, {meta.observation_id.nunique()} observations -> {out_dir}')


if __name__ == '__main__':
    main()
