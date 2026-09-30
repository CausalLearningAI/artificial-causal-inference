"""
SOMP spatial aggregation of every frame (annotations.csv row order) for one SAE: DINOv2 tokens of the frame
(recomputed, same model / preprocessing / token set as the SAE's existing codes) -> SOMP over the SAE decoder
(src/eci/somp.py, src/eci/somp_encode.py) -> one sparse vector over the SAE latents per frame.

SOMP as implemented (ResiDual's somp, criterion l1, adapted):
    tokens      the frame's foreground patches (foreground SAEs) or all 256 patches (full-frame SAE ep20)
    space       x = norm(token) - b_dec  (the SAE input space, minus the SAE's decoder bias)
    dictionary  SAE decoder rows W_dec (unit norm)
    selection   K = 16 steps (--k, = the SAE's per-token k); each adds the atom with the largest
                sum over tokens of |<residual token, atom>|, then all chosen atoms are refit by least squares
    importance  RMS over the frame's tokens of each chosen atom's coefficient (= ResiDual's coefficient
                norm / sqrt(n tokens)); atoms picked after the frame is fully explained get 0

Output: <domain eci dir>/codes/<sae>_somp/ (codes_somp, somp_idx, energy, [n_fg], config.json, verify.json,
DONE; see src/eci/somp_encode.py). Existing codes folders are only read (their config.json picks the
pipeline, their codes_max.npy is the alignment reference).

--link-mean makes <domain eci dir>/codes/<sae>_mean/ (symlinks to the existing codes_mean.npy [and n_fg.npy]
+ DONE), so the NES runners can run the mean-pooled codes as their own result set.

Usage:
    python scripts/eci/somp_encode_all.py --domain mice --sae matryoshka_btk_1024_k16_fg448_s0 --shard 3 --n-shards 24
    python scripts/eci/somp_encode_all.py --domain mice --sae matryoshka_btk_1024_k16_fg448_s0 --merge --verify --delete-shards
    python scripts/eci/somp_encode_all.py --domain mice --sae matryoshka_btk_1024_k16_fg448_s0 --link-mean
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.domain import DOMAINS, get_domain  # noqa: E402
from src.eci.encode import shard_ranges  # noqa: E402
from src.eci.fg_encode import merge_fg_shards  # noqa: E402
from src.eci.somp_encode import SOMP_OUTPUTS, SompEncoder, encode_somp_shard, verify_somp  # noqa: E402


def link_mean(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for n in ('codes_mean.npy', 'n_fg.npy'):
        if (src / n).exists() and not (dst / n).exists():
            os.symlink(src / n, dst / n)
    (dst / 'config.json').write_text(json.dumps({
        'aggregation': 'mean over the frame\'s patches (foreground patches for foreground SAEs)',
        'source': str(src), 'files': 'symlinks to the source codes_mean.npy [and n_fg.npy]'}, indent=1))
    (dst / 'DONE').touch()
    print(f'linked {dst}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', default='mice', choices=DOMAINS)
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--sae', required=True)
    p.add_argument('--k', type=int, default=16, help='atoms per frame')
    p.add_argument('--bg-dir', default=None, help='default <dataset dir>/<domain eci dir>/fg448/background')
    p.add_argument('--out-dir', default=None, help='default <codes>/<sae>_somp')
    p.add_argument('--n-shards', type=int, default=24)
    p.add_argument('--shard', type=int, default=None)
    p.add_argument('--merge', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--delete-shards', action='store_true', help='after a passing verify')
    p.add_argument('--link-mean', action='store_true')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=16)
    args = p.parse_args()

    dom = get_domain(args.domain)
    ds = Path(args.dataset_dir)
    codes_root = ds / dom.eci_rel / 'codes'
    src = codes_root / args.sae
    if not (src / 'DONE').exists():
        raise SystemExit(f'{src}/DONE missing: the SAE has no finished codes to align with')
    if args.link_mean:
        link_mean(src, codes_root / f'{args.sae}_mean')
        return
    src_cfg = json.loads((src / 'config.json').read_text())
    pipeline = 'fg' if 'foreground_rule' in src_cfg else 'full'
    if pipeline == 'full' and src_cfg.get('resolution') != 224:
        raise SystemExit(f'unexpected full-frame codes config: {src_cfg}')
    bg_dir = args.bg_dir or src_cfg.get('backgrounds') or str(ds / dom.eci_rel / 'fg448/background')
    ann = ds / dom.ann_rel
    sae_path = ds / dom.eci_rel / 'sae' / args.sae / 'sae.pt'
    out_dir = Path(args.out_dir) if args.out_dir else codes_root / f'{args.sae}_somp'
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    n_rows = len(frame_paths)
    ranges = shard_ranges(n_rows, args.n_shards)
    cfg = out_dir / 'config.json'
    if not cfg.exists():
        cfg.write_text(json.dumps({
            'aggregation': 'SOMP (simultaneous orthogonal matching pursuit), src/eci/somp.py', 'k_atoms': args.k,
            'pipeline': pipeline, 'tokens': ('foreground patches, whole frame at 448, rule ' + src_cfg.get('foreground_rule_name', 'fg448'))
            if pipeline == 'fg' else 'all 256 patches, 224 center crop',
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
            'backgrounds': bg_dir if pipeline == 'fg' else None,
            'row_order': f'dataset/{dom.ann_rel}', 'n_rows': n_rows, 'n_shards': args.n_shards, 'shard_ranges': ranges},
            indent=1))
    if args.shard is not None or args.verify:
        enc = SompEncoder(pipeline, sae_path, args.k, frame_paths, ds, bg_dir, ann)
    if args.shard is not None:
        lo, hi = ranges[args.shard]
        print(f'{pipeline} pipeline, K={args.k}, shard {args.shard}/{args.n_shards}: rows [{lo}, {hi})', flush=True)
        encode_somp_shard(enc, lo, hi, out_dir / 'shards' / f'shard_{args.shard:02d}', src / 'codes_max.npy',
                          args.batch_size, args.num_workers)
    if args.merge:
        merge_fg_shards(out_dir, ranges, n_rows, SOMP_OUTPUTS + (('n_fg',) if pipeline == 'fg' else ()))
    if args.verify:
        res = verify_somp(out_dir, enc, n_rows, ranges)
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
