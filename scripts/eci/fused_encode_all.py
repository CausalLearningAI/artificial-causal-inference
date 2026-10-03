"""
Fused encoding of ALL frames (annotations.csv row order) for one foreground-pipeline SAE: one DINOv2 forward per
frame -> pooled SAE codes (as scripts/eci/fg_encode_all.py) AND SOMP codes (as scripts/eci/somp_encode_all.py),
src/eci/fused_encode.py. Saves re-running DINOv2 on every frame for SOMP.

Output: the same folders, files and shard layout as the two separate passes:
    <domain eci dir>/codes/<sae>/shards/shard_XX/       codes_max, codes_mean, n_fg
    <domain eci dir>/codes/<sae>_somp/shards/shard_XX/  codes_somp, somp_idx, energy, n_fg
config.json of both folders is written here exactly as the separate scripts write it (the codes config by
scripts/eci/fg_encode_all.py itself; the SOMP config with the same keys plus 'fused_with_codes').
Merge / verify / delete shards / the <sae>_mean link use the existing scripts unchanged
(scripts/eci/fused_encode_merge.sh).

Usage:
    python scripts/eci/fused_encode_all.py --sae matryoshka_btk_1024_k16_ff448al_s0 --shard 3 --n-shards 24
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.domain import DOMAINS, get_domain  # noqa: E402
from src.eci.encode import shard_ranges  # noqa: E402
from src.eci.fused_encode import FusedEncoder, encode_fused_shard  # noqa: E402


def write_codes_config(args):
    """config.json of <codes>/<sae>/ written by scripts/eci/fg_encode_all.py itself (no shard / merge / verify)."""
    spec = importlib.util.spec_from_file_location('fg_encode_all', REPO / 'scripts/eci/fg_encode_all.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    argv = ['fg_encode_all.py', '--domain', args.domain, '--dataset-dir', args.dataset_dir, '--sae', args.sae,
            '--n-shards', str(args.n_shards)] + (['--bg-dir', args.bg_dir] if args.bg_dir else [])
    old, sys.argv = sys.argv, argv
    try:
        mod.main()
    finally:
        sys.argv = old


def write_somp_config(cfg, args, src, src_cfg, sae_path, bg_dir, ann_rel, n_rows, ranges):
    """Same keys as scripts/eci/somp_encode_all.py (fg pipeline) + 'fused_with_codes'."""
    cfg.write_text(json.dumps({
        'aggregation': 'SOMP (simultaneous orthogonal matching pursuit), src/eci/somp.py', 'k_atoms': args.k,
        'pipeline': 'fg', 'tokens': 'foreground patches, whole frame at 448, rule ' + src_cfg.get('foreground_rule_name', 'fg448'),
        'token_space': 'x = TokenNorm(token fp16) - b_dec', 'dictionary': 'SAE decoder rows W_dec (unit norm)',
        'selection': 'argmax over atoms of sum_t |<residual_t, atom>|, then least-squares refit on all chosen atoms',
        'importance': 'RMS over the frame tokens of the atom coefficient (ResiDual norm / sqrt(n tokens)); '
                      'atoms chosen after residual <= 1e-6 * energy get 0 and index -1',
        'files': {'codes_somp': 'float16 (N, m) importance, 0 = not chosen',
                  'somp_idx': 'int16 (N, K) chosen atoms in selection order, -1 = unused',
                  'energy': 'float32 (N, 2+K): [sum ||x||^2, SAE per-token reconstruction residual energy, '
                            'SOMP residual energy after 1..K atoms]',
                  'n_fg': 'int16 (N,) foreground tokens per frame (fg pipeline only)'},
        'source_codes': str(src), 'source_codes_config': src_cfg, 'sae_checkpoint': str(sae_path),
        'backgrounds': bg_dir, 'row_order': f'dataset/{ann_rel}', 'n_rows': n_rows, 'n_shards': args.n_shards,
        'shard_ranges': ranges,
        'fused_with_codes': 'computed in the same DINOv2 pass as the source codes (scripts/eci/fused_encode_all.py); '
                            "shard codes_max_alignment_cos compares with that pass's pooled codes_max"}, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', default='mice', choices=DOMAINS)
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--bg-dir', default=None, help="default as scripts/eci/fg_encode_all.py (the SAE's align)")
    p.add_argument('--sae', required=True)
    p.add_argument('--k', type=int, default=16, help='SOMP atoms per frame')
    p.add_argument('--n-shards', type=int, default=24)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=16)
    args = p.parse_args()

    dom = get_domain(args.domain)
    ds = Path(args.dataset_dir)
    codes_root = ds / dom.eci_rel / 'codes'
    src = codes_root / args.sae
    somp_dir = codes_root / f'{args.sae}_somp'
    ann = ds / dom.ann_rel
    sae_path = ds / dom.eci_rel / 'sae' / args.sae / 'sae.pt'
    if not (src / 'config.json').exists():
        write_codes_config(args)
    src_cfg = json.loads((src / 'config.json').read_text())
    if 'foreground_rule' not in src_cfg:
        raise SystemExit(f'{src}/config.json is not a foreground-pipeline codes config')
    bg_dir = src_cfg['backgrounds']  # as somp_encode_all.py: the codes' own backgrounds
    frame_paths = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    n_rows = len(frame_paths)
    ranges = shard_ranges(n_rows, args.n_shards)
    if src_cfg['n_rows'] != n_rows or src_cfg['n_shards'] != args.n_shards:
        raise SystemExit(f'{src}/config.json has n_rows {src_cfg["n_rows"]}, n_shards {src_cfg["n_shards"]}')
    somp_dir.mkdir(parents=True, exist_ok=True)
    if not (somp_dir / 'config.json').exists():
        write_somp_config(somp_dir / 'config.json', args, src, src_cfg, sae_path, bg_dir, dom.ann_rel, n_rows, ranges)
    enc = FusedEncoder(sae_path, args.k, frame_paths, ds, bg_dir, ann)
    lo, hi = ranges[args.shard]
    print(f'fused codes + SOMP (K={args.k}), shard {args.shard}/{args.n_shards}: rows [{lo}, {hi})', flush=True)
    encode_fused_shard(enc, lo, hi, src / 'shards' / f'shard_{args.shard:02d}',
                       somp_dir / 'shards' / f'shard_{args.shard:02d}', args.batch_size, args.num_workers)


if __name__ == '__main__':
    main()
