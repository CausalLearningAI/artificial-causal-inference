"""
CalMS21 ECI - SUPERVISED CEILING per token config (DIAGNOSTIC ONLY, never a selection criterion; GPU job).

Same probe code and settings as mabe22/eci_enc_probe.py: scripts/eci/diag_mice_patches.cross_validate, probes mean_lin
and mil_ctx (3 x 3 context on the config's own grid), 5 VIDEO-grouped folds over the 19 task1 test videos, inner
validation = max(4, 17%) of the training videos, IPW-weighted metrics, MIL 3000 steps, seeds 0 1 2, on the masked z_t
tokens of the eval frames (no temporal difference) and the scored labels of plan/eval_plan.json.
Outputs: OUT/probe/probe_<config>.json
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2] / 'scripts/eci'))
from calms_common import OUT, PLAN, TOK, log, stage_dir  # noqa: E402
from eci_train import G64, labels, shards  # noqa: E402


def main():
    import torch
    import diag_mice_patches as dmp
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--probes', nargs='+', default=['mean_lin', 'mil_ctx'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=3000)
    args = ap.parse_args()
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    stage = stage_dir()
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    labs = labels()
    od = OUT / 'probe'
    od.mkdir(parents=True, exist_ok=True)
    f = od / f'probe_{args.config}.json'
    G = 64 if args.config in G64 else 32

    # label-STRATIFIED video folds (attack is in only 5 of the 19 test videos: a plain random split leaves folds without
    # a positive and AUROC undefined): positive videos of each label, rarest label first, dealt round robin over the 5
    # folds; the rest at random. Inner validation: one not-yet-chosen positive video per label (rarest first), filled
    # at random to max(4, 17%) of the training videos.
    posv = {b: set(E.video[E[b]].unique()) for b in labs}
    order_l = sorted(labs, key=lambda b: len(posv[b]))

    def pool_folds(pools, k=5, seed=0):
        g = np.random.default_rng(seed)
        side = {}
        for b in order_l:
            todo = [v for v in sorted(posv[b]) if v not in side]
            g.shuffle(todo)
            cnt = [sum(side.get(v) == f for v in posv[b]) for f in range(k)]
            start = int(np.argmin(cnt))
            for j, v in enumerate(todo):
                side[v] = (start + j) % k
        rest = [v for v in sorted(set(pools)) if v not in side]
        g.shuffle(rest)
        for j, v in enumerate(rest):
            side[v] = j % k
        return [sorted(v for v in side if side[v] == f) for f in range(k)]

    def inner_val(train_pools, fold, n_val_=None):
        g = np.random.default_rng(100 + fold)
        k = max(4, int(round(0.17 * len(train_pools))))
        val = []
        for b in order_l:
            c = [v for v in sorted(train_pools) if v in posv[b] and v not in val]
            if c and not any(v in posv[b] for v in val):
                val.append(c[int(g.integers(len(c)))])
        rest = [v for v in sorted(train_pools) if v not in val]
        val += list(g.choice(rest, max(0, k - len(val)), replace=False))
        return set(val)
    dmp.pool_folds = pool_folds
    dmp.inner_val = inner_val
    dmp.BEH = list(labs)
    t0 = time.time()
    toks, poss, lens, Fs = [], [], [], []
    for base in shards('eval', args.config):
        Fm = pd.read_parquet(TOK / f'{base}_frames.parquet')
        src = TOK / f'{base}_{args.config}_t.npy'
        loc = stage / src.name
        shutil.copyfile(src, loc)
        a = np.load(loc)
        loc.unlink()
        p = np.load(TOK / f'{base}_{args.config}_pos.npy')
        ln = Fm.n32.values.astype(np.int64) * (4 if G == 64 else 1)
        assert len(a) == ln.sum() == len(p)
        toks.append(a)
        poss.append(p)
        lens.append(ln)
        Fs.append(Fm)
    Fm = pd.concat(Fs).reset_index(drop=True)
    lens = np.concatenate(lens)
    st = np.r_[0, np.cumsum(lens)]
    order = pd.MultiIndex.from_frame(Fm[['video', 'i']]).get_indexer(pd.MultiIndex.from_frame(E[['video', 'i']]))
    assert (order >= 0).all()
    tok, pos = np.concatenate(toks), np.concatenate(poss)
    idx = np.concatenate([np.arange(st[o], st[o + 1]) for o in order])
    tok, pos, lens = tok[idx], pos[idx], lens[order]
    assert pos.max() < G * G
    lab = E.copy()
    lab['pool'] = lab['video']
    lab['n_fg'] = lens
    bag = dmp.Bag(tok, pos, lens, G, dev)
    res = dmp.cross_validate(bag, lab, lab.w.values.astype(np.float64), dev, args.probes, args.seeds, None, args.config,
                             args.steps)
    summ = dmp.summarise(res)
    f.write_text(json.dumps({'config': args.config, 'grid': G, 'dim': int(tok.shape[1]), 'n_tokens': int(len(tok)),
                             'labels': labs, 'summary': summ, 'runs': res, 'wall_s': time.time() - t0,
                             'gpu': torch.cuda.get_device_name(0)}, indent=1, default=float))
    for s in summ:
        if s['probe'] in args.probes:
            log(f"{args.config} {s['probe']} {s['behaviour']}: AUROC {s['auroc_mean']:.4f} AP {s['ap_mean']:.4f}")


if __name__ == '__main__':
    main()
