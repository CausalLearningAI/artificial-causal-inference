"""Successive-halving hyperparameter search for the online-augmented per-frame classifier.

Why a new search script
-----------------------
grid_search_frame.py and search_patchgrid_online.py both rest on one efficiency insight:
the frozen encoder's output is independent of every hyperparameter being searched, so the
whole candidate pool is encoded ONCE into a token cache and each trial trains a fresh head
on top of it. Online D4/photometric augmentation destroys that — tokens now depend on which
rendering was drawn, so there is nothing stable to cache.

This script recovers the same win from the other side. Trials that share an augmentation
setting can share the *augmented batch stream*: encode each batch once under no_grad, then
run every trial's head on those same tokens. The encoder is ~90% of step cost and the heads
are 0.5-3M params, so a rung of 8 trials costs barely more than a single training run.
Trials are therefore GROUPED by (augment, photo_strength), and only the number of distinct
augmentation settings — not the number of trials — multiplies encoder cost.

That is what makes searching at the full 504px resolution affordable, rather than searching
at 224px and hoping the optima transfer.

Successive halving
------------------
Trials train in rungs (default cumulative epochs 4 / 8 / 14 / 24), and the bottom half is
cut at each rung boundary. This is affordable because the ranking settles early: on the
res448 run, epoch 12 already held 92% of the final AP. Every surviving trial keeps its
optimizer and LR-schedule state across rungs — a rung boundary is a checkpoint, not a
restart — and all trials share one cosine schedule length (the final rung total) so the
schedule means the same thing for every trial regardless of when it dies.

What is deliberately NOT searched
---------------------------------
neg_ratio: fixed at 1. Above ~2.33 it saturates the negative pool (54,333 negatives vs
23,280 positives) so the distinct settings collapse, and its real effect is on prior shift,
which is corrected analytically downstream rather than by training. Leaving it out keeps
every trial's batch stream identical within a group, which is what makes stream sharing work.
context_k / input_size: fixed per search job, since they change the batch stream shape.

Selection
---------
Rung promotion uses monitor macro AP, the only thing cheap enough to compute every epoch.
The FINAL report deliberately does not stop there: survivors get the full suite on full val
(plain + tolerant AP with prevalence and enrichment, ROC-AUC, and the per-observation rate
agreement metrics.py flags as the select-on quantity), and the top-5 under EACH metric are
printed side by side so a winner that only wins on plain AP is visible as such. Bouts are
1-2 frames, so plain frame-exact AP spends much of its range on boundary jitter; tol1/tol2
are the honest frame metrics, and rate correlation r is what actually drives PPI variance
reduction — though at n=24 val observations its SE is ~0.2, so it discriminates weakly.

Usage:
    python scripts/mice_behavior/search_online_aug.py --n-configs 32 --input-size 504
    sbatch scripts/mice_behavior/search_online_aug.sh
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
import train_online_aug as toa
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import get_fixed_val_pools
from src.mice_behavior.metrics import ap_report, rate_report, format_ap_report, format_rate_report
from train_patchgrid_online import dummy_loader

EMB_DIM, PATCH_SIZE = toa.EMB_DIM, toa.PATCH_SIZE


def sample_cfg(rng: random.Random) -> dict:
    """Search space for the CURRENT regime, which differs from the inherited one.

    The live best_cfg (n_heads=8, hidden_dim=384, dropout=0.4, wd=1e-4, lr=3e-4) was tuned on
    a 4x4 (16-token) coarse patch grid with cached tokens, the old val split and neg_ratio=15.
    None of those hold now, so nothing here is anchored to it.

    weight_decay is conditioned on the optimizer: 1e-4 under Adam is L2 coupled to the
    adaptive scaling and barely regularizes, whereas the same number under AdamW is decoupled
    and real. Sampling one shared range would have made the two optimizers incomparable.
    lr is capped at 1e-3: the pairwise search saw PatchAttnPool collapse above that.
    """
    optimizer = rng.choice(['adam', 'adamw'])
    weight_decay = (rng.choice([0.0, 1e-5, 1e-4, 1e-3]) if optimizer == 'adam'
                    else rng.choice([0.01, 0.05, 0.1, 0.2]))
    augment = rng.choice(['d4', 'd4_photo', 'd4_photo', 'd4_photo'])
    return dict(
        n_heads=rng.choice([1, 2, 4, 8]),
        hidden_dim=rng.choice([128, 256, 384, 512]),
        # all multiples of 8 so any n_heads divides them (MultiheadAttention requires it);
        # None = run at full width (patch_pool_dim None -> 768).
        cross_attn_dim=rng.choice([32, 64, 128, None]),
        patch_pool_dim=rng.choice([128, 256, 384, None]),
        dropout=rng.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        weight_decay=weight_decay,
        optimizer=optimizer,
        lr=rng.choice([1e-4, 3e-4, 1e-3]),
        warmup_epochs=rng.choice([0, 2, 4]),
        augment=augment,
        photo_strength=(rng.choice([0.5, 1.0, 1.5]) if augment == 'd4_photo' else 1.0),
    )


def group_key(cfg):
    """Trials sharing this key see an identical batch stream and share encoder forwards."""
    return (cfg['augment'], cfg['photo_strength'] if cfg['augment'] == 'd4_photo' else 0.0)


class Trial:
    def __init__(self, tid, cfg, total_epochs, dev):
        self.tid, self.cfg, self.dev = tid, cfg, dev
        self.model = MouseFrameClassifier(
            emb_dim=EMB_DIM, n_heads=cfg['n_heads'], hidden_dim=cfg['hidden_dim'],
            use_patch_grid=True, dropout=cfg['dropout'],
            cross_attn_dim=cfg['cross_attn_dim'], patch_pool_dim=cfg['patch_pool_dim'])
        self.n_params = sum(p.numel() for p in self.model.parameters())
        opt_cls = torch.optim.AdamW if cfg['optimizer'] == 'adamw' else torch.optim.Adam
        self.opt = opt_cls(self.model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        eta, warm = 0.01, cfg['warmup_epochs']

        def factor(e):
            if e < warm:
                return (e + 1) / (warm + 1)
            prog = (e - warm) / max(total_epochs - warm, 1)
            return eta if prog >= 1 else eta + 0.5 * (1 - eta) * (1 + math.cos(math.pi * prog))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, factor)
        self.scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
        self.epochs, self.best_ap, self.history, self.alive = 0, -1.0, [], True

    def to(self, device):
        self.model.to(device)
        # optimizer state tensors must follow the model or step() mixes devices
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        return self


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-configs', type=int, default=32)
    p.add_argument('--rungs', type=int, nargs='+', default=[4, 8, 14, 24],
                    help='CUMULATIVE epochs at each rung; the bottom --cut-fraction is dropped '
                         'at every boundary. The last value is also the cosine schedule length.')
    p.add_argument('--cut-fraction', type=float, default=0.5)
    p.add_argument('--input-size', type=int, default=504)
    p.add_argument('--context-k', type=int, default=2)
    p.add_argument('--max-train-frames', type=int, default=300_000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--read-workers', type=int, default=32)
    p.add_argument('--decode-workers', type=int, default=8)
    p.add_argument('--val-monitor-size', type=int, default=12_500)
    p.add_argument('--final-full-val', type=int, default=3,
                    help='how many survivors get the expensive full-val multi-metric report')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', type=str, default='mice-behavior-frame')
    p.add_argument('--tag', type=str, default='search_online_aug')
    p.add_argument('--smoke', action='store_true',
                    help='tiny end-to-end check of the rung/grouping/promotion machinery: caps '
                         'obs, configs, epochs and resolution so the whole search runs in a few '
                         'minutes instead of ~10 h')
    args = p.parse_args()
    if args.smoke:
        args.n_configs, args.rungs, args.input_size = 4, [1, 2], 224
        args.max_train_frames, args.val_monitor_size = 4_000, 600
        args.final_full_val, args.tag = 2, 'search_smoke'

    n_patches = (args.input_size // PATCH_SIZE) ** 2
    total_epochs = args.rungs[-1]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUT = gsf.FRAME_DIR / args.tag
    OUT.mkdir(parents=True, exist_ok=True)
    LOG = gsf.SEARCH_DIR / f'log_{args.tag}.jsonl'
    gsf.SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    print(f'search: {args.n_configs} configs  input_size={args.input_size} ({n_patches} patches)  '
          f'context_k={args.context_k}  rungs={args.rungs}', flush=True)

    # ---------------------------------------------------------------- data (once, shared)
    pair_labels = gsf.build_pair_labels(gsf.DATA_DIR, gsf.DATASET_DIR, overwrite=False)
    ann_csv = gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    o2p = gsf.load_obs_to_pool_map(gsf.DATA_DIR)
    all_obs = pd.read_parquet(pair_labels)['observation_id'].unique().tolist()
    pools = sorted({o2p[o] for o in all_obs})
    val_pools = get_fixed_val_pools(pools)
    train_obs = [o for o in all_obs if o2p[o] not in val_pools]
    val_obs = [o for o in all_obs if o2p[o] in val_pools]
    if args.smoke:
        # val dominates the read phase, so cap it too or "smoke" is not smoke
        train_obs, val_obs = train_obs[:3], val_obs[:2]
    print(f'train {len(train_obs)} obs / val {len(val_obs)} obs (pools {sorted(val_pools)})', flush=True)

    tm = FrameBatchData(str(ann_csv), str(pair_labels), train_obs, args.context_k, 1,
                        dummy_loader(1, 1), n_patches=1, stride=1,
                        max_frames=args.max_train_frames, seed=gsf.SEED)
    vm = FrameBatchData(str(ann_csv), str(pair_labels), val_obs, args.context_k, 1,
                        dummy_loader(1, 1), n_patches=1, stride=1)
    del tm.flat, vm.flat

    pos_idx = np.where(tm.labels.sum(1) > 0)[0]
    neg_idx = np.where(tm.labels.sum(1) == 0)[0]
    n_neg = min(len(neg_idx), len(pos_idx))    # neg_ratio fixed at 1, see module docstring
    v_rng = np.random.default_rng(gsf.SEED)
    val_keep = np.sort(v_rng.choice(len(vm), size=min(len(vm), args.val_monitor_size), replace=False))
    print(f'{len(pos_idx):,} pos / {len(neg_idx):,} neg available; monitor {len(val_keep):,}', flush=True)

    def needed(meta, si):
        a = meta.gi[si][:, None] + meta.offsets_grid[None, :]
        return np.unique(a[~meta.pad_mask[si]])

    ann = pd.read_csv(ann_csv, usecols=['frame_path'])
    frame_paths = ann.frame_path.values
    jpeg_cache = {}

    def ensure_cached(frame_idx, what):
        missing = np.array(sorted(set(int(g) for g in frame_idx) - set(jpeg_cache)), dtype=np.int64)
        if not len(missing):
            return
        t0 = time.time()
        dl = DataLoader(toa._BytesReader(frame_paths[missing]), batch_size=None,
                        num_workers=args.read_workers, prefetch_factor=6, collate_fn=lambda x: x)
        for i, buf in dl:
            jpeg_cache[int(missing[i])] = buf
        print(f'  [{what}] read {len(missing):,} frames in {(time.time()-t0)/60:.1f} min; '
              f'cache {len(jpeg_cache):,}', flush=True)

    # ---------------------------------------------------------------- trials
    rng = random.Random(gsf.SEED)
    seen, cfgs = set(), []
    while len(cfgs) < args.n_configs and len(seen) < 4000:
        c = sample_cfg(rng)
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            cfgs.append(c)
    trials = [Trial(i, c, total_epochs, dev) for i, c in enumerate(cfgs)]
    for t in trials:
        print(f'  trial {t.tid:2d} {t.n_params/1e6:5.2f}M  {t.cfg}', flush=True)

    encoder = AutoModel.from_pretrained(toa.MODEL_ID).to(dev).eval()
    encoder.requires_grad_(False)
    crit = nn.BCEWithLogitsLoss()
    ep_rng = np.random.default_rng(gsf.SEED)

    def make_loader(meta, order, augment, photo_strength, seed):
        batches = [order[i:i+args.batch_size] for i in range(0, len(order), args.batch_size)]

        def sc(lo, hi):
            return (1 - (1 - lo) * photo_strength, 1 + (hi - 1) * photo_strength)

        return DataLoader(toa._SampleDataset(meta, batches, jpeg_cache, args.input_size,
                                             augment, seed, sc(0.80, 1.25), sc(0.80, 1.25),
                                             sc(0.83, 1.20)),
                          batch_size=None, num_workers=args.decode_workers,
                          pin_memory=(dev.type == 'cuda'), prefetch_factor=4)

    @torch.no_grad()
    def encode(imgs):
        B, T = imgs.shape[:2]
        tok = encoder(pixel_values=imgs.view(B*T, *imgs.shape[2:])).last_hidden_state[:, 1:]
        return tok.view(B, T, n_patches, EMB_DIM)

    def run_epoch(group, epoch_seed):
        """One epoch for every trial in `group`, sharing ONE encoder forward per batch."""
        order = np.concatenate([pos_idx, ep_rng.choice(neg_idx, size=n_neg, replace=False)])
        ep_rng.shuffle(order)
        ensure_cached(needed(tm, order), f'train epoch (group {group[0].cfg["augment"]})')
        for t in group:
            t.model.train()
        tot = {t.tid: 0.0 for t in group}
        seen_n = 0
        cfg0 = group[0].cfg
        for imgs, offs, lbl, mask in make_loader(tm, order, cfg0['augment'],
                                                 cfg0['photo_strength'], epoch_seed):
            imgs, lbl = imgs.to(dev, non_blocking=True), lbl.to(dev, non_blocking=True)
            offs, mask = offs.to(dev), mask.to(dev)
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                tok = encode(imgs).detach()      # <- shared by every trial in the group
            for t in group:
                t.opt.zero_grad()
                with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                    loss = crit(t.model(tok, offsets=offs, key_padding_mask=mask), lbl)
                t.scaler.scale(loss).backward()
                t.scaler.unscale_(t.opt)
                torch.nn.utils.clip_grad_norm_(t.model.parameters(), 0.5)
                t.scaler.step(t.opt)
                t.scaler.update()
                tot[t.tid] += loss.item() * imgs.size(0)
            seen_n += imgs.size(0)
        for t in group:
            t.sched.step()
            t.epochs += 1
        return {k: v / max(seen_n, 1) for k, v in tot.items()}

    @torch.no_grad()
    def evaluate(group, order):
        """Monitor pass, also sharing one encoder forward across the group. Always unaugmented."""
        for t in group:
            t.model.eval()
        P = {t.tid: [] for t in group}
        L = []
        for imgs, offs, lbl, mask in make_loader(vm, order, 'none', 1.0, 0):
            imgs = imgs.to(dev, non_blocking=True)
            offs, mask = offs.to(dev), mask.to(dev)
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                tok = encode(imgs)
                for t in group:
                    P[t.tid].append(torch.sigmoid(t.model(tok, offsets=offs,
                                                          key_padding_mask=mask)).float().cpu())
            L.append(lbl)
        return {k: torch.cat(v).numpy() for k, v in P.items()}, torch.cat(L).numpy()

    def log(rec):
        rec['ts'] = time.time()
        with open(LOG, 'a') as f:
            f.write(json.dumps(rec, default=str) + '\n')

    # ---------------------------------------------------------------- successive halving
    ensure_cached(needed(vm, val_keep), 'monitor')
    # ONE run for the whole search, with per-trial metric prefixes (t07/monitor_ap), rather
    # than 32 concurrent wandb runs -- lighter, and the sweep reads as a single chart.
    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, name=args.tag,
                            config={'input_size': args.input_size, 'context_k': args.context_k,
                                    'n_configs': args.n_configs, 'rungs': args.rungs,
                                    'trials': {f't{t.tid}': t.cfg for t in trials}})
            print(f'wandb: {wb.url}', flush=True)
        except Exception as e:
            print(f'wandb disabled ({e.__class__.__name__}: {e})', flush=True)

    prev = 0
    for rung, target in enumerate(args.rungs):
        alive = [t for t in trials if t.alive]
        groups = {}
        for t in alive:
            groups.setdefault(group_key(t.cfg), []).append(t)
        print(f'\n{"="*70}\nRUNG {rung+1}: {len(alive)} trials -> epochs {prev+1}..{target} '
              f'across {len(groups)} augmentation group(s)\n{"="*70}', flush=True)
        for gkey, group in groups.items():
            for t in group:
                t.to(dev)
            for ep in range(prev, target):
                t0 = time.time()
                losses = run_epoch(group, gsf.SEED * 1000 + ep)
                probs, labs = evaluate(group, val_keep)
                for t in group:
                    pr = probs[t.tid]
                    per_ap = [float(average_precision_score(labs[:, i], pr[:, i])) for i in (0, 1)]
                    per_auc = [float(roc_auc_score(labs[:, i], pr[:, i])) for i in (0, 1)]
                    q = np.clip(pr, 1e-7, 1 - 1e-7)
                    vl = float(-(labs * np.log(q) + (1 - labs) * np.log(1 - q)).mean())
                    ap = float(np.mean(per_ap))
                    t.best_ap = max(t.best_ap, ap)
                    row = {'epoch': t.epochs, 'train_loss': losses[t.tid], 'val_loss': vl,
                           'monitor_ap': ap, 'ap_nt': per_ap[0], 'ap_nn': per_ap[1],
                           'auc_nt': per_auc[0], 'auc_nn': per_auc[1],
                           'lr': float(t.opt.param_groups[0]['lr'])}
                    t.history.append(row)
                    if wb is not None:
                        try:
                            wb.log({f't{t.tid:02d}/{k}': v for k, v in row.items()
                                    if k != 'epoch'} | {'epoch': t.epochs})
                        except Exception:
                            pass
                print(f'  [{gkey}] epoch {ep+1}/{target} in {time.time()-t0:.0f}s  ' +
                      '  '.join(f't{t.tid}:{t.history[-1]["monitor_ap"]:.4f}' for t in group),
                      flush=True)
            for t in group:
                t.to(torch.device('cpu'))     # free GPU before the next group trains
                torch.cuda.empty_cache() if dev.type == 'cuda' else None
        alive.sort(key=lambda t: -t.best_ap)
        for t in alive:
            log({'rung': rung + 1, 'trial': t.tid, 'epochs': t.epochs,
                 'best_monitor_ap': t.best_ap, 'cfg': t.cfg})
        if rung < len(args.rungs) - 1:
            keep = max(1, int(round(len(alive) * (1 - args.cut_fraction))))
            for t in alive[keep:]:
                t.alive = False
            print(f'  -> keeping {keep}: ' +
                  ', '.join(f't{t.tid}({t.best_ap:.4f})' for t in alive[:keep]), flush=True)
        prev = target

    # ---------------------------------------------------------------- final multi-metric report
    survivors = sorted([t for t in trials if t.alive], key=lambda t: -t.best_ap)
    print(f'\n{"="*70}\nFULL-VAL MULTI-METRIC REPORT ({min(len(survivors), args.final_full_val)} '
          f'survivors)\n{"="*70}', flush=True)
    val_full = needed(vm, np.arange(len(vm)))
    ensure_cached(val_full, 'full-val')
    a2 = pd.read_csv(ann_csv, usecols=['observation_id', 'frame_idx'])
    a2 = a2[a2.observation_id.isin(set(val_obs))].reset_index()
    b = {o: int(g['index'].values[0]) for o, g in a2.groupby('observation_id', sort=False)}
    st = np.array(sorted(b.values()))
    nm = np.array([k for k, _ in sorted(b.items(), key=lambda x: x[1])])
    sample_obs = nm[np.searchsorted(st, vm.gi, side='right') - 1]

    finals = []
    for t in survivors[:args.final_full_val]:
        t.to(dev)
        probs, labs = evaluate([t], np.arange(len(vm)))
        pr = probs[t.tid]
        apr = ap_report(pr, labs, sample_obs, tolerances=(0, 1, 2))
        rr = rate_report(pr, labs, sample_obs)
        auc = {n: float(roc_auc_score(labs[:, i], pr[:, i])) for i, n in enumerate(('nt', 'nn'))}
        print(f'\n--- trial {t.tid}  {t.cfg}')
        print(format_ap_report(apr, tolerances=(0, 1, 2)))
        print(f'ROC-AUC  nt {auc["nt"]:.4f}  nn {auc["nn"]:.4f}')
        print(format_rate_report(rr))
        rec = {'trial': t.tid, 'cfg': t.cfg, 'n_params': t.n_params, 'epochs': t.epochs,
               'ap_report': apr, 'rate_report': rr, 'roc_auc': auc,
               'macro_ap_tol0': apr['macro/tol0']['ap'], 'macro_ap_tol1': apr['macro/tol1']['ap'],
               'mean_auc': float(np.mean(list(auc.values()))),
               'mean_rate_r': float(np.mean([rr[n]['pearson_r'] for n in ('nt', 'nn')]))}
        finals.append(rec)
        torch.save(t.model.state_dict(), OUT / f'trial{t.tid:02d}_model.pt')
        log({'final': True, **rec})
        if wb is not None:
            try:
                p_ = f'fullval/t{t.tid:02d}'
                wb.summary.update(
                    {f'{p_}/ap_{k.replace("/", "_")}': v['ap'] for k, v in apr.items()}
                    | {f'{p_}/auc_{k}': v for k, v in auc.items()}
                    | {f'{p_}/rate_{n}_{k}': v for n, d in rr.items() for k, v in d.items()
                       if isinstance(v, (int, float))})
            except Exception:
                pass
        t.to(torch.device('cpu'))

    # rank under each metric separately -- a config that only wins on plain AP should be
    # visible as such rather than silently crowned (see module docstring on selection).
    print(f'\n{"="*70}\nRANKING BY EACH METRIC (disagreement is the point)\n{"="*70}')
    for key, label in [('macro_ap_tol0', 'macro AP (tol0)'), ('macro_ap_tol1', 'macro AP (tol1)'),
                       ('mean_auc', 'mean ROC-AUC'), ('mean_rate_r', 'mean per-obs rate r')]:
        rank = sorted(finals, key=lambda r: -r[key])
        print(f'  {label:<22} ' + '  '.join(f'#{i+1} t{r["trial"]}={r[key]:.4f}'
                                            for i, r in enumerate(rank)))
    print('\n  (n=24 val observations -> SE on r is ~0.2; treat rate-r ordering as weak evidence)')

    json.dump({'args': vars(args), 'finals': finals,
               'all_trials': [{'trial': t.tid, 'cfg': t.cfg, 'best_monitor_ap': t.best_ap,
                               'epochs': t.epochs, 'history': t.history} for t in trials]},
              open(OUT / 'search.json', 'w'), indent=2, default=str)
    if wb is not None:
        try:
            wb.finish()
        except Exception:
            pass
    print(f'\nSaved {OUT}/search.json', flush=True)


if __name__ == '__main__':
    main()
