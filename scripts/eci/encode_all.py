"""
Encode ALL mice v1 frames (annotations.csv row order) with DINOv2-base + the ECI SAE.

Output (default): dataset/mice/v1/eci/codes/<sae name>/
    codes_mean.npy  float16 (2592000, 1024)  SAE codes mean-pooled over the 256 patches
    codes_max.npy   float16 (2592000, 1024)  SAE codes max-pooled over the patches
    cls_l-1.npy     float16 (2592000, 768)   DINOv2 CLS token, last_hidden_state
    config.json     model, SAE checkpoint, inference mode, shard ranges
    verify.json     checks (written by --verify)
    shards/shard_XX/{...npy, DONE}, DONE (after --merge)

Usage:
    python scripts/eci/encode_all.py --shard 3 --n-shards 16     # one shard (SLURM array task)
    python scripts/eci/encode_all.py --merge --verify --n-shards 16
    python scripts/eci/encode_all.py --sae matryoshka_btk_1024_k16_fps1_s0 --outputs codes_mean,codes_max \
        --tokens-dir dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fps1 --shard 3          # v2, no CLS
    python scripts/eci/encode_all.py --shard 0 --n-shards 20000 --out-dir /tmp/enc_test   # quick test
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.encode import encode_shard, merge_shards, shard_ranges, verify_codes  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--sae', default='matryoshka_btk_1024_k16_ep20_s0')
    p.add_argument('--tokens-dir', default=str(REPO / 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1'))
    p.add_argument('--out-dir', default=None)
    p.add_argument('--n-shards', type=int, default=16)
    p.add_argument('--shard', type=int, default=None)
    p.add_argument('--merge', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--device', default='cuda')
    p.add_argument('--outputs', default='codes_mean,codes_max,cls_l-1', help='comma list (drop cls_l-1 to skip CLS)')
    args = p.parse_args()
    outputs = tuple(args.outputs.split(','))

    ds = Path(args.dataset_dir)
    sae_path = ds / 'mice/v1/eci/sae' / args.sae / 'sae.pt'
    out_dir = Path(args.out_dir) if args.out_dir else ds / 'mice/v1/eci/codes' / args.sae
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = pd.read_csv(ds / 'mice/v1/annotations.csv', usecols=['frame_path'])['frame_path'].tolist()
    n_rows = len(frame_paths)
    ranges = shard_ranges(n_rows, args.n_shards)

    cfg_path = out_dir / 'config.json'
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps({
            'model_id': 'facebook/dinov2-base', 'resolution': 224,
            'layer': 'last_hidden_state (after final LayerNorm), rounded to float16 before the SAE',
            'preprocessing': 'src/eci/extract.py load_encoder (resize shortest edge 256 bicubic, center crop 224)',
            'sae_checkpoint': str(sae_path), 'sae_inference': 'global threshold (BatchTopK paper)',
            'pooling': {'codes_mean': 'mean over 256 patches', 'codes_max': 'max over 256 patches'},
            'outputs': list(outputs), 'tokens_dir_for_verify': args.tokens_dir,
            'row_order': 'dataset/mice/v1/annotations.csv', 'n_rows': n_rows,
            'n_shards': args.n_shards, 'shard_ranges': ranges}, indent=1))

    if args.shard is not None:
        lo, hi = ranges[args.shard]
        print(f'shard {args.shard}/{args.n_shards}: rows [{lo}, {hi})', flush=True)
        encode_shard(frame_paths, lo, hi, out_dir / 'shards' / f'shard_{args.shard:02d}', sae_path,
                     dataset_dir=ds, batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
                     outputs=outputs)
    if args.merge:
        merge_shards(out_dir, ranges, n_rows, outputs)
    if args.verify:
        res = verify_codes(out_dir, sae_path, args.tokens_dir, outputs=outputs)
        print(json.dumps(res, indent=1))
        (out_dir / 'verify.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
