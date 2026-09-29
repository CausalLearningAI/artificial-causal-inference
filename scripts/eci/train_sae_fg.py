"""
Train a Matryoshka BatchTopK SAE on FOREGROUND DINOv2 patch tokens (whole frame at 448,
no crop; src/eci/foreground.py), ECI mice v1. Same model / loss / optimizer / schedule as
scripts/eci/train_sae.py; same 8 held-out validation pools as the existing SAEs (read
from --val-from metrics.json). All train tokens are held in CPU RAM (~110 GB) and random
batches are gathered by a prefetch thread.

Output: dataset/mice/v1/eci/sae/matryoshka_btk_{n_latents}_k{k}_{tag}_s{seed}/
    sae.pt, metrics.json (args, split, history, validation metrics on all held-out
    foreground tokens: threshold inference and exact per-token top-k)

Usage:
    python scripts/eci/train_sae_fg.py --seed 0
    python scripts/eci/train_sae_fg.py --seed 0 --max-train-tokens 2000000 --epochs 1 --out-dir <test dir>
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.foreground import FgTokenStore  # noqa: E402
from src.eci.sae import (MatryoshkaBatchTopKSAE, TokenNorm, _log_step, _train_step,  # noqa: E402
                         evaluate_sae, geometric_median, save_checkpoint)


def row_pools(dataset_dir, data_dir):
    """Pool name of every annotations.csv row -> (codes (n_rows,) int16, names list)."""
    ann = pd.read_csv(Path(dataset_dir) / 'mice/v1/annotations.csv', usecols=['observation_id'])
    exp = pd.read_csv(Path(data_dir) / 'mice/v1/experiment.csv').set_index('observation_id')
    pool = exp.loc[ann.observation_id.values, 'pool'].values
    codes, names = pd.factorize(pool)
    return codes.astype(np.int16), list(names)


def load_split(store, is_val_row, n_threads=8):
    """-> (train tokens (n, d) fp16, val tokens, val rows, val pos) in CPU RAM."""
    sel_tr, sel_va, rows_va, pos_va = [], [], [], []
    for s in range(len(store.dirs)):
        v = is_val_row[store.row(s)]
        sel_tr.append(~v); sel_va.append(v)
    n_tr, n_va = sum(int(m.sum()) for m in sel_tr), sum(int(m.sum()) for m in sel_va)
    train = np.empty((n_tr, store.dim), np.float16)
    val = np.empty((n_va, store.dim), np.float16)
    offs = np.cumsum([0] + [int(m.sum()) for m in sel_tr]), np.cumsum([0] + [int(m.sum()) for m in sel_va])

    def job(s):
        t = np.asarray(store.tokens(s))
        train[offs[0][s]:offs[0][s + 1]] = t[sel_tr[s]]
        val[offs[1][s]:offs[1][s + 1]] = t[sel_va[s]]
    with ThreadPoolExecutor(n_threads) as ex:
        list(ex.map(job, range(len(store.dirs))))
    for s in range(len(store.dirs)):
        rows_va.append(store.row(s)[sel_va[s]]); pos_va.append(store.pos(s)[sel_va[s]])
    return train, val, np.concatenate(rows_va), np.concatenate(pos_va)


def batches(train, epochs, batch_size, seed):
    """Yield (epoch, fp16 pinned tensor) random batches (per-epoch permutation), prefetched."""
    rng = np.random.default_rng(seed)
    n = train.shape[0]
    spe = n // batch_size

    def get(idx):
        return torch.from_numpy(train[np.sort(idx)]).pin_memory()
    with ThreadPoolExecutor(4) as ex:
        for ep in range(epochs):
            perm = rng.permutation(n)
            futs = [ex.submit(get, perm[b * batch_size:(b + 1) * batch_size]) for b in range(min(8, spe))]
            for b in range(spe):
                x = futs[b].result()
                futs[b] = None
                nb = b + 8
                if nb < spe:
                    futs.append(ex.submit(get, perm[nb * batch_size:(nb + 1) * batch_size]))
                yield ep, x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens-dir', default=str(REPO / 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1'))
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--data-dir', default=str(REPO / 'data'))
    p.add_argument('--val-from', default=str(REPO / 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_ep20_s0/metrics.json'))
    p.add_argument('--out-dir', default=None)
    p.add_argument('--tag', default='fg448')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-latents', type=int, default=1024)
    p.add_argument('--prefixes', default='128,256,512,1024')
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--k-aux', type=int, default=512)
    p.add_argument('--aux-coef', type=float, default=1 / 32)
    p.add_argument('--dead-tokens', type=int, default=2_000_000)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--max-train-tokens', type=int, default=None, help='testing: random subset')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        REPO / 'dataset/mice/v1/eci/sae' / f'matryoshka_btk_{args.n_latents}_k{args.k}_{args.tag}_s{args.seed}'
    if (out_dir / 'metrics.json').exists() and not args.overwrite:
        print(f'[SKIP] {out_dir} exists')
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')

    val_pools = json.loads(Path(args.val_from).read_text())['val_pools']
    codes, names = row_pools(args.dataset_dir, args.data_dir)
    is_val_row = np.isin(np.array(names)[codes], val_pools)
    store = FgTokenStore(args.tokens_dir)
    t0 = time.time()
    train, val, val_rows, val_pos = load_split(store, is_val_row)
    print(f'val pools {val_pools}: {len(train):,} train / {len(val):,} val foreground tokens, loaded in '
          f'{time.time() - t0:.0f}s', flush=True)
    if args.max_train_tokens:
        keep = np.sort(np.random.default_rng(args.seed).choice(len(train), args.max_train_tokens, replace=False))
        train = train[keep]

    g = np.random.default_rng(args.seed)
    init = torch.from_numpy(train[np.sort(g.choice(len(train), 500_000, replace=False))]).to(device)
    norm = TokenNorm(store.dim).to(device).fit(init, seed=args.seed)
    prefixes = [int(x) for x in args.prefixes.split(',')]
    sae = MatryoshkaBatchTopKSAE(d_in=store.dim, n_latents=args.n_latents, prefixes=prefixes, k=args.k,
                                 k_aux=args.k_aux, aux_coef=args.aux_coef, dead_tokens=args.dead_tokens,
                                 seed=args.seed).to(device)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    print(f'norm scale {float(norm.scale):.4f}, |b_dec init| {float(sae.b_dec.norm()):.3f}', flush=True)

    opt = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999))
    total = args.epochs * (len(train) // args.batch_size)
    print(f'{total:,} steps ({args.epochs} epochs x {len(train) // args.batch_size:,})', flush=True)
    history, step, tt = [], 0, time.time()
    for ep, xb in batches(train, args.epochs, args.batch_size, args.seed):
        x = norm(xb.to(device, non_blocking=True))
        logs, gn = _train_step(sae, opt, x, step, total, args.lr, args.warmup, args.grad_clip)
        if step % 500 == 0 or step == total - 1:
            _log_step(sae, opt, logs, gn, step, total, ep, tt, history, lambda s: print(s, flush=True))
        step += 1
    assert step == total
    train_time = time.time() - tt
    sae.eval()
    save_checkpoint(out_dir / 'sae.pt', sae, norm, extra={'val_pools': val_pools, 'args': vars(args)})

    val_t = torch.from_numpy(val[: (len(val) // 256) * 256])  # evaluate_sae groups rows by 256 (ignored)
    ev = {}
    for mode in ('threshold', 'topk'):
        ev[mode] = evaluate_sae(sae, norm, val_t, n_patches=256, mode=mode)
        for r in ev[mode]['prefixes'].values():
            r.pop('pooled_frame_l0_mean', None)
        print(f'VAL ({mode}) on {ev[mode]["n_tokens"]:,} foreground tokens:')
        for m, r in ev[mode]['prefixes'].items():
            print(f'  m={m:>5}  FVE {r["fve"]:.4f}  L0/token {r["l0_per_token"]:6.2f}  dead {100 * r["dead_frac"]:5.1f}% '
                  f'({r["n_dead"]})', flush=True)
    metrics = {'args': vars(args), 'val_pools': val_pools, 'n_train_tokens': int(len(train)),
               'n_val_tokens': int(val_t.shape[0]), 'n_steps': total, 'norm_scale': float(norm.scale),
               'threshold': float(sae.threshold), 'train_time_s': round(train_time, 1),
               'val_threshold': ev['threshold'], 'val_topk': ev['topk'], 'history': history}
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    print(f'Done in {train_time:.0f}s -> {out_dir}')


if __name__ == '__main__':
    main()
