"""
Train a Matryoshka BatchTopK SAE on FOREGROUND DINOv2 patch tokens (whole frame at 448,
no crop; src/eci/foreground.py), ECI mice v1. Same model / loss / optimizer / schedule as
scripts/eci/train_sae.py; same 8 held-out validation pools as the existing SAEs (read
from --val-from metrics.json). All train tokens are held in CPU RAM (~110 GB) and random
batches are gathered by a prefetch thread.

Output: dataset/mice/v1/eci/sae/matryoshka_btk_{n_latents}_k{k}_{tag}_s{seed}/
    sae.pt, metrics.json (args, split, history, validation metrics on all held-out
    foreground tokens: threshold inference and exact per-token top-k)

Validation split per domain (--domain, src/eci/domain.py val_split): mice = the held-out pools of
--val-from; ants = about 10% of the videos of every (experiment, T), seeded (--split-seed, default 0).
metrics.json 'val_pools' lists the held-out units (ants: videos).

Usage:
    python scripts/eci/train_sae_fg.py --seed 0
    python scripts/eci/train_sae_fg.py --seed 0 --max-train-tokens 2000000 --epochs 1 --out-dir <test dir>
    python scripts/eci/train_sae_fg.py --domain ants --tokens-dir dataset/ants/eci/train_tokens/<store> --tag antfg448
A token store written with fg_extract_train.py --encoder dinov3_base (shard.json 'encoder') makes the checkpoint and
metrics.json record 'encoder'; fg_encode.py then encodes with that encoder.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.domain import DOMAINS, get_domain  # noqa: E402
from src.eci.foreground import FgTokenStore  # noqa: E402
from src.eci.sae import (MatryoshkaBatchTopKSAE, TokenNorm, _log_step, _train_step,  # noqa: E402
                         evaluate_sae, geometric_median, save_checkpoint)


def load_split(store, is_val_row, n_threads=8, motion=False):
    """-> (train tokens (n, d) fp16, val tokens, val rows, val pos) in CPU RAM.
    motion=True: each row is [token_t, token_t - token_{t-D}] (2 d), from tokens.f16 / prev.f16."""
    sel_tr, sel_va, rows_va, pos_va = [], [], [], []
    for s in range(len(store.dirs)):
        v = is_val_row[store.row(s)]
        sel_tr.append(~v); sel_va.append(v)
    n_tr, n_va = sum(int(m.sum()) for m in sel_tr), sum(int(m.sum()) for m in sel_va)
    d_out = store.dim * (2 if motion else 1)
    train = np.empty((n_tr, d_out), np.float16)
    val = np.empty((n_va, d_out), np.float16)
    offs = np.cumsum([0] + [int(m.sum()) for m in sel_tr]), np.cumsum([0] + [int(m.sum()) for m in sel_va])

    def job(s):
        t = np.asarray(store.tokens(s))
        if motion:
            t = np.concatenate([t, (t.astype(np.float32) - store.prev(s)).astype(np.float16)], 1)
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


@torch.no_grad()
def block_fve(sae, norm, tokens, blocks, names, chunk=65536):
    """FVE per block of input dims (normalized space, all latents, threshold inference)."""
    dev = sae.W_dec.device
    sse, s1, s2, n = [0.0] * len(blocks), [0.0] * len(blocks), [0.0] * len(blocks), 0
    for a in range(0, tokens.shape[0], chunk):
        x = norm(tokens[a:a + chunk].to(dev))
        r = sae.decode(sae.encode(x, mode='threshold'))
        o = 0
        for i, b in enumerate(blocks):
            xs, rs = x[:, o:o + b].double(), r[:, o:o + b].double()
            sse[i] += float((xs - rs).pow(2).sum()); s1[i] = s1[i] + xs.sum(0); s2[i] += float(xs.pow(2).sum())
            o += b
        n += x.shape[0]
    return {nm: float(1 - sse[i] / (s2[i] - float((s1[i] ** 2).sum()) / n)) for i, nm in enumerate(names)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', default='mice', choices=DOMAINS)
    p.add_argument('--tokens-dir', default=None, help='default <domain eci dir>/train_tokens/dinov2_base_l-1_fg448_fps1')
    p.add_argument('--val-from', default=str(REPO / 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_ep20_s0/metrics.json'),
                   help='mice: the SAE whose held-out pools are reused')
    p.add_argument('--split-seed', type=int, default=0, help='ants: seed of the held-out videos')
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
    p.add_argument('--motion', action='store_true',
                   help='SAE input = [token_t, token_t - token_{t-D}] (store written with --motion-delta D); '
                        'each half normalized to the same average norm (TokenNorm.fit_blocks)')
    args = p.parse_args()
    dom = get_domain(args.domain)
    args.tokens_dir = args.tokens_dir or str(dom.eci_dir / 'train_tokens/dinov2_base_l-1_fg448_fps1')

    out_dir = Path(args.out_dir) if args.out_dir else \
        dom.eci_dir / 'sae' / f'matryoshka_btk_{args.n_latents}_k{args.k}_{args.tag}_s{args.seed}'
    if (out_dir / 'metrics.json').exists() and not args.overwrite:
        print(f'[SKIP] {out_dir} exists')
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')

    val_pools, is_val_row = dom.val_split(args.val_from, args.split_seed)
    store = FgTokenStore(args.tokens_dir)
    t0 = time.time()
    train, val, val_rows, val_pos = load_split(store, is_val_row, motion=args.motion)
    shard_info = store.info[0]
    motion_delta = int(shard_info.get('motion_delta', 0)) if args.motion else 0
    if args.motion and motion_delta <= 0:
        raise SystemExit('--motion needs a token store written with --motion-delta')
    print(f'val pools {val_pools}: {len(train):,} train / {len(val):,} val foreground tokens, loaded in '
          f'{time.time() - t0:.0f}s', flush=True)
    if args.max_train_tokens:
        keep = np.sort(np.random.default_rng(args.seed).choice(len(train), args.max_train_tokens, replace=False))
        train = train[keep]

    g = np.random.default_rng(args.seed)
    init = torch.from_numpy(train[np.sort(g.choice(len(train), 500_000, replace=False))]).to(device)
    d_in = train.shape[1]
    norm = TokenNorm(d_in).to(device)
    norm = norm.fit_blocks(init, [store.dim, store.dim], seed=args.seed) if args.motion else norm.fit(init, seed=args.seed)
    prefixes = [int(x) for x in args.prefixes.split(',')]
    sae = MatryoshkaBatchTopKSAE(d_in=d_in, n_latents=args.n_latents, prefixes=prefixes, k=args.k,
                                 k_aux=args.k_aux, aux_coef=args.aux_coef, dead_tokens=args.dead_tokens,
                                 seed=args.seed).to(device)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    print(f'norm scale {norm.scale.unique()[:4].tolist()}, |b_dec init| {float(sae.b_dec.norm()):.3f}', flush=True)

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
    extra = {'val_pools': val_pools, 'args': vars(args), 'fg_rule': shard_info.get('rule_name', 'fg448'),
             'motion_delta': motion_delta}
    if 'encoder' in shard_info:  # token stores of a non-DINOv2 encoder (absent = dinov2_base)
        extra['encoder'] = shard_info['encoder']
    save_checkpoint(out_dir / 'sae.pt', sae, norm, extra=extra)

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
    if args.motion:  # FVE of each half (static token / token change), prefix 1024, threshold inference
        ev['blocks'] = block_fve(sae, norm, val_t, [store.dim, store.dim], ('static', 'delta'))
        print('VAL per-block FVE (m=1024):', ev['blocks'], flush=True)
    metrics = {'args': vars(args), 'val_pools': val_pools, 'fg_rule': extra['fg_rule'], 'motion_delta': motion_delta,
               'val_blocks': ev.get('blocks'), 'n_train_tokens': int(len(train)),
               'n_val_tokens': int(val_t.shape[0]), 'n_steps': total,
               'threshold': float(sae.threshold), 'norm_scale_blocks': norm.scale.unique().tolist(), 'train_time_s': round(train_time, 1),
               'val_threshold': ev['threshold'], 'val_topk': ev['topk'], 'history': history}
    if 'encoder' in extra:
        metrics['encoder'] = extra['encoder']
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    print(f'Done in {train_time:.0f}s -> {out_dir}')


if __name__ == '__main__':
    main()
