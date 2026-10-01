"""
Spatial frame features of every frame (annotations.csv row order) for one foreground SAE, in one GPU pass:
pairwise co-activation of latents on touching patches, zone pooling, and blob measures of the foreground mask
(definitions: src/eci/spatial_encode.py). Tokens, mask and SAE patch codes are recomputed with the SAE's own
pipeline (src/eci/fg_encode.py _Runner); every batch re-derives codes_max and checks it against the stored rows.

Output (existing codes folders are only read):
    <codes>/<sae>_spatial/   config.json, shards/ (deleted after a passing verify), pair_frequency.csv, verify.json, DONE
    <codes>/<sae>_pairs/     codes_pairs.npy  float16 (N, n_pairs kept)  pair strength s_ab, pairs with s_ab > 0 in
                             >= 1% of all frames; features.csv (column, a, b, frame_frac, column_all)
    <codes>/<sae>_zones/     codes_zones.npy  float16 (N, n_zones * 128) zone-major max per zone and latent;
                             zone_nfg.npy int16 (N, n_zones); features.csv (column, zone, zone_name, latent)
    <codes>/<sae>_blobs/     blobs.npy float32 (N, 4) [n_blobs, mean_blob_distance, n_components, largest_component];
                             features.csv
    each with n_fg.npy (N,) int16 and DONE, so the NES runners read them like any codes folder
    (scripts/eci/run_nes.py --sae <sae>_pairs --primary-pooling pairs --poolings pairs --prefixes all).

Usage:
    python scripts/eci/spatial_encode_all.py --domain mice --sae matryoshka_btk_1024_k16_fg448_s0 --shard 3 --n-shards 24
    python scripts/eci/spatial_encode_all.py --domain mice --sae matryoshka_btk_1024_k16_fg448_s0 --merge --verify --delete-shards
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.domain import DOMAINS, get_domain  # noqa: E402
from src.eci.encode import shard_ranges  # noqa: E402
from src.eci.spatial_encode import (BLOB_COLUMNS, MIN_BLOB, PAIR_K, PREFIX, SpatialEncoder,  # noqa: E402
                                    encode_spatial_shard, merge_spatial, verify_spatial)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', default='mice', choices=DOMAINS)
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--sae', required=True)
    p.add_argument('--n-shards', type=int, default=24)
    p.add_argument('--shard', type=int, default=None)
    p.add_argument('--merge', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--delete-shards', action='store_true', help='after a passing verify')
    p.add_argument('--min-pair-frac', type=float, default=0.01)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=16)
    args = p.parse_args()

    dom = get_domain(args.domain)
    ds = Path(args.dataset_dir)
    codes_root = ds / dom.eci_rel / 'codes'
    src = codes_root / args.sae
    if not (src / 'DONE').exists():
        raise SystemExit(f'{src}/DONE missing: the SAE has no finished codes to align with')
    src_cfg = json.loads((src / 'config.json').read_text())
    if 'foreground_rule' not in src_cfg:
        raise SystemExit('spatial features need a foreground SAE (codes config has no foreground_rule)')
    bg_dir = src_cfg.get('backgrounds') or str(ds / dom.eci_rel / 'fg448/background')
    ann = ds / dom.ann_rel
    sae_path = ds / dom.eci_rel / 'sae' / args.sae / 'sae.pt'
    odor_csv = ds / dom.eci_rel / 'odor_corner.csv' if args.domain == 'mice' else None
    out_dir = codes_root / f'{args.sae}_spatial'
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    n_rows = len(frame_paths)
    ranges = shard_ranges(n_rows, args.n_shards)
    enc = SpatialEncoder(sae_path, frame_paths, ds, bg_dir, ann, args.domain, odor_csv) \
        if (args.shard is not None or args.verify) else None
    cfg = out_dir / 'config.json'
    if not cfg.exists():
        zone_names = enc.zone_names if enc else None
        cfg.write_text(json.dumps({
            'what': 'spatial frame features, src/eci/spatial_encode.py', 'prefix_latents': PREFIX,
            'pairs': 'unordered latent pairs a < b (< 128): s_ab = max over 8-neighbouring patch pairs p != q of '
                     'min(z_a(p), z_b(q)); a = b and the same patch excluded; per-patch top-%d codes' % PAIR_K,
            'zones': ('mice: near odor (patch centre within 0.5 x arena side of the arena corner on the odor-bag side, '
                      'dataset/mice/v1/eci/odor_corner.csv) / rest' if args.domain == 'mice'
                      else 'ants: 3 x 3 grid of patch rows / cols 0-10, 11-21, 22-31'),
            'zone_names': zone_names, 'zone_feature': 'max over the foreground patches of the zone, latent < 128',
            'blobs': f'connected components of the foreground mask on the 32 x 32 grid, 8-connectivity, blob = '
                     f'component >= {MIN_BLOB} patches; columns {list(BLOB_COLUMNS)}; distance in patch units, '
                     '0 when < 2 blobs',
            'source_codes': str(src), 'source_codes_config': src_cfg, 'sae_checkpoint': str(sae_path),
            'backgrounds': bg_dir, 'odor_corner': str(odor_csv) if odor_csv else None,
            'row_order': f'dataset/{dom.ann_rel}', 'n_rows': n_rows, 'n_shards': args.n_shards, 'shard_ranges': ranges},
            indent=1))
    if args.shard is not None:
        lo, hi = ranges[args.shard]
        print(f'{args.domain} {args.sae}: shard {args.shard}/{args.n_shards}: rows [{lo}, {hi})', flush=True)
        encode_spatial_shard(enc, lo, hi, out_dir / 'shards' / f'shard_{args.shard:02d}', src / 'codes_max.npy',
                             args.batch_size, args.num_workers)
    if args.merge:
        zn = json.loads(cfg.read_text())['zone_names'] or (enc.zone_names if enc else None)
        if zn is None:
            from src.eci.spatial_encode import zone_maps
            zn = zone_maps(args.domain, [], odor_csv)[1]
        merge_spatial(out_dir, ranges, n_rows, args.sae, codes_root, tuple(zn), args.min_pair_frac)
    if args.verify:
        res = verify_spatial(out_dir, enc, args.sae, codes_root, n_rows, ranges)
        print(json.dumps({k: v for k, v in res.items() if k != 'recompute'}, indent=1))
        print('recompute:', {k: v for k, v in res['recompute'].items() if k != 'rows'})
        (out_dir / 'verify.json').write_text(json.dumps(res, indent=1))
        if res['passed'] and args.delete_shards and (out_dir / 'DONE').exists():
            shutil.rmtree(out_dir / 'shards')
            print('shards deleted')
        elif not res['passed']:
            raise SystemExit('VERIFY FAILED')


if __name__ == '__main__':
    main()
