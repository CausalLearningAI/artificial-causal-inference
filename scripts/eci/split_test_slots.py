"""
Instance-split GATE test, method A (GPU): unsupervised slot attention (src/eci/slots.py, DINOSAUR-style) on the frozen
DINOv2-base 448 foreground tokens of the 1 fps token store, trained on TRAIN videos only (the pilots' split), then
read on the eval frames of split_test_select.py.

Training tokens: per train video one contiguous block of up to --block store frames (random start, seed 0), frames
with 1..--max-tokens foreground tokens (mice 360 = 2 x four mice of ~45 patches; ants 200); loaded once into RAM.
Variants (all d = 256, 3 iterations, decoder MLP 3 x 1024, Adam lr 4e-4, 1000 warm-up steps, cosine to 0, grad clip
1, batches of 64 frames, bf16 autocast, --steps steps):
    A_k{N+1}       K = N + 1 random slots
    A_k{N+2}       K = N + 2 random slots
    A_k{N+1}seed   K = N + 1, N slots seeded from k-means (k = N) of the frame's foreground patch positions
Inference: fixed noise seed, per token argmax over the decoder alphas -> 32 x 32 label map (slot + 1; 0 = not a
foreground patch).

Output results/vision/eci_split_test/<domain>/slots/<variant>/{model.pt, train.json}, labels_<variant>.npz
(labels (F, 32, 32) uint8 in eval.parquet order, runtime).
Usage: python scripts/eci/split_test_slots.py --domain mice [--steps 20000] [--variants A_k5 ...]
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
sys.path.insert(0, str(REPO / 'scripts/eci'))
from split_test_common import DOMAINS, OUT  # noqa: E402
from src.eci.slots import SlotModel  # noqa: E402

log = lambda s: print(s, flush=True)  # noqa: E731


def pad_batch(tok, pos, starts, lens, fr, dev, mean, std):
    """frames fr (indices into starts / lens) -> padded (x (B, L, d), pos (B, L), valid (B, L)) on dev."""
    ln = lens[fr]
    L = int(ln.max())
    B = len(fr)
    rep = torch.repeat_interleave(torch.arange(B), ln)
    off = torch.arange(int(ln.sum())) - torch.repeat_interleave(torch.cumsum(ln, 0) - ln, ln)
    ti = starts[fr][rep] + off
    x = torch.zeros(B, L, tok.shape[1], device=dev)
    p = torch.zeros(B, L, dtype=torch.long, device=dev)
    v = torch.zeros(B, L, dtype=torch.bool, device=dev)
    xs = (tok[ti].to(dev, non_blocking=True).float() - mean) / std
    x[rep.to(dev), off.to(dev)] = xs
    p[rep.to(dev), off.to(dev)] = pos[ti].to(dev)
    v[rep.to(dev), off.to(dev)] = True
    return x, p, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=list(DOMAINS))
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--block', type=int, default=300)
    ap.add_argument('--variants', nargs='*', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--infer-only', action='store_true', help='load slots/<variant>/model.pt, redo the eval read only')
    args = ap.parse_args()
    dom = DOMAINS[args.domain]
    N = dom['N']
    variants = {f'A_k{N + 1}': dict(K=N + 1, n_seeded=0), f'A_k{N + 2}': dict(K=N + 2, n_seeded=0),
                f'A_k{N + 1}seed': dict(K=N + 1, n_seeded=N)}
    if args.variants:
        variants = {k: variants[k] for k in args.variants}
    out = OUT / args.domain / 'slots'
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    from spatial_sae_pilot import StoreIndex
    t0 = time.time()
    idx = StoreIndex(args.domain)
    tr, _, _, _, train_v, _ = idx.split(args.domain, dom['n_train'], 0)
    cap = 360 if args.domain == 'mice' else 200
    rng = np.random.default_rng(args.seed)
    if args.infer_only:
        pick, tok = np.zeros(0, int), np.zeros((0, 768), np.float16)
    else:
        pick = []
        obs_tr = idx.obs[tr]
        for v in train_v:
            f = tr[obs_tr == v]
            f = f[(idx.nfg[f] > 0) & (idx.nfg[f] <= cap)]
            if len(f) == 0:
                continue
            s = rng.integers(0, max(1, len(f) - args.block + 1))
            pick.append(f[s:s + args.block])
        pick = np.sort(np.concatenate(pick))
        tok, pos, lens = idx.load(pick)
        log(f'{args.domain}: {len(train_v)} train videos, {len(pick):,} train frames, {len(tok):,} tokens '
            f'({tok.nbytes / 1e9:.1f} GB) loaded in {time.time() - t0:.0f}s; mean {lens.mean():.0f} tokens/frame')
        tok_t = torch.from_numpy(tok)
        pos_t = torch.from_numpy(pos.astype(np.int64))
        lens_t = torch.from_numpy(lens.astype(np.int64))
        starts = torch.from_numpy(np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64))
        sub = torch.from_numpy(rng.choice(len(tok), min(300_000, len(tok)), replace=False))
        samp = tok_t[sub].float()
        mean, std = samp.mean(0).to(dev), samp.std(0).clamp_min(1e-3).to(dev)
        del samp
    # eval frames
    E = pd.read_parquet(OUT / args.domain / 'eval.parquet')
    row2f = {int(r): i for i, r in enumerate(idx.rows)}
    ef = np.array([row2f[int(r)] for r in E.row])
    order = np.argsort(ef)
    etok, epos, elens = idx.load(ef[order])  # load() sorts; ef[order] is sorted
    etok_t, epos_t = torch.from_numpy(etok), torch.from_numpy(epos.astype(np.int64))
    elens_t = torch.from_numpy(elens.astype(np.int64))
    estarts = torch.from_numpy(np.r_[0, np.cumsum(elens)[:-1]].astype(np.int64))
    log(f'eval: {len(E)} frames, {len(etok):,} tokens, max {elens.max()} tokens/frame')

    for name, cfg in variants.items():
        torch.manual_seed(args.seed)
        model = SlotModel(K=cfg['K'], n_seeded=cfg['n_seeded']).to(dev)
        d = out / name
        if args.infer_only:
            ck = torch.load(d / 'model.pt', map_location=dev)
            model.load_state_dict(ck['state_dict'])
            mean, std = ck['mean'].to(dev), ck['std'].to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=4e-4)
        g = torch.Generator().manual_seed(args.seed)
        hist = []
        tt = time.time()
        model.train()
        for step in range(0 if args.infer_only else args.steps):
            lr = 4e-4 * min(1.0, (step + 1) / 1000) * 0.5 * (1 + np.cos(np.pi * step / args.steps))
            for pg in opt.param_groups:
                pg['lr'] = lr
            fr = torch.randint(0, len(lens_t), (args.batch,), generator=g)
            x, p, v = pad_batch(tok_t, pos_t, starts, lens_t, fr, dev, mean, std)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                recon, alpha, _ = model(x, p, v)
            loss = SlotModel.loss(recon, x, v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 500 == 0 or step == args.steps - 1:
                with torch.no_grad():
                    am = alpha.argmax(1)
                    used = torch.stack([(torch.bincount(am[b][v[b]], minlength=cfg['K']) > 0).sum() for b in range(len(am))])
                    fve = 1 - float(loss) / float(((x - 0) ** 2).mean(-1)[v].mean())
                h = dict(step=step, loss=float(loss), fve=round(fve, 4), slots_used=float(used.float().mean()),
                         grad_norm=float(gn), elapsed_s=round(time.time() - tt, 1))
                hist.append(h)
                log(f'  {name} ' + ' '.join(f'{k} {v_}' for k, v_ in h.items()))
        train_s = time.time() - tt
        model.eval()
        eg = torch.Generator(device=dev).manual_seed(1234)
        labs = np.zeros((len(E), 32, 32), np.uint8)
        conf = np.zeros(len(E), np.float32)
        torch.cuda.synchronize()
        te = time.time()
        with torch.no_grad():
            for b0 in range(0, len(E), args.batch):
                fr = torch.arange(b0, min(b0 + args.batch, len(E)))
                x, p, v = pad_batch(etok_t, epos_t, estarts, elens_t, fr, dev, mean, std)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, alpha, _ = model(x, p, v, gen=eg)
                am = alpha.argmax(1)
                mx = alpha.max(1).values
                for j in range(len(fr)):
                    vj = v[j]
                    lab = np.zeros(1024, np.uint8)
                    lab[p[j][vj].cpu().numpy()] = am[j][vj].cpu().numpy() + 1
                    k = int(fr[j])
                    labs[order[k]] = lab.reshape(32, 32)  # k = position in store order -> eval.parquet row order[k]
                    conf[order[k]] = float(mx[j][vj].mean())
        torch.cuda.synchronize()
        inf_s = time.time() - te
        d.mkdir(exist_ok=True)
        if not args.infer_only:
            torch.save({'state_dict': model.state_dict(), 'cfg': cfg, 'mean': mean.cpu(), 'std': std.cpu()}, d / 'model.pt')
            (d / 'train.json').write_text(json.dumps({'variant': name, 'cfg': cfg, 'steps': args.steps, 'batch': args.batch,
                                                  'n_train_frames': int(len(pick)), 'n_train_tokens': int(len(tok)),
                                                  'train_videos': [str(v) for v in train_v], 'train_s': round(train_s, 1),
                                                  'infer_s_per_frame': inf_s / len(E), 'gpu': torch.cuda.get_device_name(0),
                                                  'history': hist}, indent=1))
        np.savez_compressed(OUT / args.domain / f'labels_{name}.npz', labels=labs, conf=conf,
                            runtime_s_per_frame=inf_s / len(E), level='patch')
        log(f'{name}: trained {train_s:.0f}s, inference {1000 * inf_s / len(E):.2f} ms/frame (batch {args.batch})')


if __name__ == '__main__':
    main()
