"""
Crop prototype step 2: Matryoshka BatchTopK SAE (src/eci/sae.py, same class / loss / optimizer as
the patch SAEs) on one DINOv2 CLS token per crop. Separate SAEs for
    single   type-0 crops (one mouse)
    pair     type-1 (two blobs < pair_d apart) + type-2 (merged blob = contact / huddle) crops
Train = crops of the 20 training videos (1 fps, crops/train); validation (FVE / L0 / dead only) = a
random 100k subset of the held-out eval crops of the same kind (crops/eval). No labels.

Defaults: 512 latents, prefixes 64/128/256/512, k = 8, batch 1024, 60 epochs, lr 5e-4, dead after
100k crops, AuxK k_aux 256.
Output: dataset/mice/v1/eci/crops/sae/matryoshka_btk_512_k8_{kind}_s{seed}/ sae.pt, metrics.json
Usage: python scripts/eci/crops_train_sae.py --kind pair --seed 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, evaluate_sae, geometric_median, save_checkpoint, train_sae  # noqa: E402

CROPS = REPO / 'dataset/mice/v1/eci/crops'
TYPES = {'single': (0,), 'pair': (1, 2)}


def load(split, kind, feat='cls'):
    X, T = [], []
    for f in sorted((CROPS / split).glob('task_*.npz')):
        if '.tmp' in f.name:
            continue
        z = np.load(f)
        m = np.isin(z['crop_spec'][:, 0].astype(int), TYPES[kind])
        X.append(z[feat][m]); T.append(z['crop_spec'][m, 0].astype(np.int8))
    return np.concatenate(X), np.concatenate(T)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=tuple(TYPES), required=True)
    p.add_argument('--feat', default='cls', choices=('cls', 'mean'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-latents', type=int, default=512)
    p.add_argument('--prefixes', default='64,128,256,512')
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--k-aux', type=int, default=256)
    p.add_argument('--dead-tokens', type=int, default=100_000)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=200)
    args = p.parse_args()
    name = f'matryoshka_btk_{args.n_latents}_k{args.k}_{args.kind}' + ('' if args.feat == 'cls' else f'_{args.feat}')
    out = CROPS / 'sae' / f'{name}_s{args.seed}'
    if (out / 'metrics.json').exists():
        print(f'[SKIP] {out}'); return
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device('cuda')
    tr, ttr = load('train', args.kind, args.feat)
    va, _ = load('eval', args.kind, args.feat)
    va = va[np.sort(np.random.default_rng(0).choice(len(va), min(len(va), 100_000), replace=False))]
    print(f'{args.kind}: {len(tr):,} train crops (types {np.bincount(ttr).tolist()}), {len(va):,} val crops', flush=True)
    x = torch.from_numpy(tr).to(dev)
    norm = TokenNorm(x.shape[1]).to(dev).fit(x, seed=args.seed)
    sae = MatryoshkaBatchTopKSAE(d_in=x.shape[1], n_latents=args.n_latents, prefixes=[int(v) for v in args.prefixes.split(',')],
                                 k=args.k, k_aux=args.k_aux, dead_tokens=args.dead_tokens, seed=args.seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(x[:100_000])))
    t0 = time.time()
    hist = train_sae(sae, norm, x, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, warmup=args.warmup,
                     seed=args.seed, log_every=500, log_fn=lambda s: print(s, flush=True))
    sae.eval()
    save_checkpoint(out / 'sae.pt', sae, norm, extra={'args': vars(args)})
    vt = torch.from_numpy(va[:len(va) // 64 * 64])
    ev = evaluate_sae(sae, norm, vt, n_patches=64, mode='threshold')
    for r in ev['prefixes'].values():
        r.pop('pooled_frame_l0_mean', None)
    for m, r in ev['prefixes'].items():
        print(f'  VAL m={m:>4} FVE {r["fve"]:.4f} L0 {r["l0_per_token"]:.2f} dead {100 * r["dead_frac"]:.1f}%', flush=True)
    fire = np.array(ev.pop('fire_rate'))
    (out / 'metrics.json').write_text(json.dumps({'args': vars(args), 'n_train': int(len(tr)), 'n_val': int(len(vt)),
                                                  'train_time_s': round(time.time() - t0, 1), 'threshold': float(sae.threshold),
                                                  'val_threshold': ev, 'val_fire_rate_median': float(np.median(fire)),
                                                  'val_fire_rate_max': float(fire.max()), 'history': hist}, indent=1))
    print(f'Done -> {out}')


if __name__ == '__main__':
    main()
