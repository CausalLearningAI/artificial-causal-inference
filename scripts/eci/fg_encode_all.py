"""
Encode ALL 2,592,000 mice v1 frames (annotations.csv row order) with DINOv2-base on the whole
frame at 448 (no crop) + foreground mask (src/eci/foreground.py) + a foreground SAE.

Output (default): dataset/mice/v1/eci/codes/<sae name>/
    codes_max.npy   float16 (2592000, 1024)  max over foreground patches (0 if none)
    codes_mean.npy  float16 (2592000, 1024)  mean over foreground patches (0 if none)
    n_fg.npy        int16   (2592000,)       foreground patches per frame (of 1024)
    config.json, verify.json, shards/ (deleted after a successful --verify with --delete-shards)

--domain (default mice, src/eci/domain.py) picks annotations.csv, the SAE and codes dirs and the
default backgrounds (ants: dataset/ants/eci/..., 768,000 frames). The foreground rule is the SAE's.

Usage:
    python scripts/eci/fg_encode_all.py --shard 3 --n-shards 24
    python scripts/eci/fg_encode_all.py --merge --verify --delete-shards --n-shards 24
    python scripts/eci/fg_encode_all.py --domain ants --sae <ants sae> --shard 3 --n-shards 24
    python scripts/eci/fg_encode_all.py --sae matryoshka_btk_1024_k16_fg448al_s0 --shard 3 --n-shards 24
An SAE trained on odor-aligned tokens (checkpoint 'align' = 'odor') encodes rotated frames, with the default
backgrounds fg448al/background; config.json records 'align'.
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
from src.eci.fg_encode import encode_fg_shard, merge_fg_shards, verify_fg_codes  # noqa: E402
import torch  # noqa: E402

from src.eci.extract import MODEL_IDS  # noqa: E402
from src.eci.foreground import ENCODER_RESOLUTION, RULES, align_tag  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', default='mice', choices=DOMAINS)
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--bg-dir', default=None, help="default <dataset dir>/<domain eci dir>/fg448[al]/background (SAE's align)")
    p.add_argument('--sae', default='matryoshka_btk_1024_k16_fg448_s0')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--n-shards', type=int, default=24)
    p.add_argument('--shard', type=int, default=None)
    p.add_argument('--merge', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--delete-shards', action='store_true', help='after a passing verify')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=16)
    args = p.parse_args()

    dom = get_domain(args.domain)
    ds = Path(args.dataset_dir)
    ann = ds / dom.ann_rel
    sae_path = ds / dom.eci_rel / 'sae' / args.sae / 'sae.pt'
    out_dir = Path(args.out_dir) if args.out_dir else ds / dom.eci_rel / 'codes' / args.sae
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    n_rows = len(frame_paths)
    ranges = shard_ranges(n_rows, args.n_shards)
    cfg = out_dir / 'config.json'
    ck = torch.load(sae_path, map_location='cpu', weights_only=False)
    rule_name, motion_delta = ck.get('fg_rule', 'fg448'), int(ck.get('motion_delta', 0) or 0)
    encoder = ck.get('encoder', 'dinov2_base')
    bg_sub = bool(ck.get('bg_sub', False))
    align = ck.get('align', 'none')
    del ck
    args.bg_dir = args.bg_dir or str(ds / dom.eci_rel / align_tag(align) / 'background')
    if not cfg.exists():
        enc_cfg = {} if encoder == 'dinov2_base' else {  # SAE tokens from another encoder, mask from DINOv2 448
            'encoder': encoder, 'model_id': MODEL_IDS[encoder], 'resolution': ENCODER_RESOLUTION[encoder],
            'preprocessing': f'whole 512x512 frame at {ENCODER_RESOLUTION[encoder]} (no resize at 512), ImageNet '
                             'normalization, patch 16 (32x32 patches), CLS and register tokens dropped',
            'mask_encoder': 'facebook/dinov2-base at 448 (foreground rule and backgrounds unchanged)'}
        cfg.write_text(json.dumps({
            'model_id': 'facebook/dinov2-base', 'resolution': 448, 'center_crop': False,
            'layer': 'last_hidden_state (after final LayerNorm), fp32 forward rounded to float16 before the SAE',
            'preprocessing': 'whole 512x512 frame resized bicubic to 448x448, ImageNet normalization (32x32 patches)',
            'foreground_rule_name': rule_name, 'foreground_rule': RULES[rule_name], 'backgrounds': args.bg_dir,
            'motion_delta': motion_delta,
            'sae_input': 'token' if motion_delta == 0 else f'[token_t, token_t - token_(t-{motion_delta})] same patch',
            'sae_checkpoint': str(sae_path), 'sae_inference': 'global threshold',
            'pooling': {'codes_max': 'max over foreground patches', 'codes_mean': 'mean over foreground patches',
                        'n_fg': 'number of foreground patches'},
            'row_order': f'dataset/{dom.ann_rel}', 'n_rows': n_rows,
            'n_shards': args.n_shards, 'shard_ranges': ranges, **enc_cfg,
            **({'sae_input': 'token - background token (rule background, same patch position and time block)',
                'bg_sub': True} if bg_sub else {}),
            **({'align': align, 'alignment': 'every frame rotated by a multiple of 90 degrees (lossless) before the '
                'encoder so that the odor corner (odor_corner.csv) is at the top right; patch positions are '
                'aligned-frame positions'} if align != 'none' else {})}, indent=1))
    if args.shard is not None:
        lo, hi = ranges[args.shard]
        print(f'shard {args.shard}/{args.n_shards}: rows [{lo}, {hi})', flush=True)
        encode_fg_shard(frame_paths, lo, hi, out_dir / 'shards' / f'shard_{args.shard:02d}', sae_path, args.bg_dir,
                        ann, ds, args.batch_size, args.num_workers)
    if args.merge:
        merge_fg_shards(out_dir, ranges, n_rows)
    if args.verify:
        res = verify_fg_codes(out_dir, sae_path, args.bg_dir, ann, ds)
        a = res['alignment']
        ok = (res['codes_max']['shape'][0] == n_rows == res['n_rows_annotations']
              and res['codes_max']['rows_nonfinite'] == 0 and res['codes_mean']['rows_nonfinite'] == 0
              and a['codes_max']['cos_min'] > 0.99 and a['n_fg_max_abs_diff'] <= 5)
        res['passed'] = bool(ok)
        print(json.dumps(res, indent=1))
        (out_dir / 'verify.json').write_text(json.dumps(res, indent=1))
        if ok and args.delete_shards and (out_dir / 'DONE').exists():
            shutil.rmtree(out_dir / 'shards')
            print('shards deleted')
        elif not ok:
            raise SystemExit('VERIFY FAILED')


if __name__ == '__main__':
    main()
