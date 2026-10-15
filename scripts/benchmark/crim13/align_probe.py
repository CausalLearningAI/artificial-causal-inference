"""
CRIM13 alignment test - SUPERVISED CEILING (DIAGNOSTIC ONLY, never a selection criterion; GPU job).

Probe code and settings of calms21/eci_probe.py: scripts/eci/diag_mice_patches.cross_validate, probe mil_ctx (attention
MIL with 3 x 3 context on the 40 grid), MIL 3000 steps, seeds 0 1 2, IPW-weighted metrics, on the masked z_t tokens of
d3l_640_L20 (the C1 tokens, no temporal difference) of the eval frames of both halves. Folds = the released halves (2
folds, roles swapped as in the SAE protocol); inner validation = 15% of the training half's videos (rng 100 + fold).
Behaviours: --labels (default sniff).
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
from align_common import GRID, OUT, PLAN, TOK, log, stage_dir  # noqa: E402
from align_train import shard_bases  # noqa: E402


def main():
    import torch
    import diag_mice_patches as dmp
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='d3l_640_L20')
    ap.add_argument('--labels', nargs='+', default=['sniff'])
    ap.add_argument('--probes', nargs='+', default=['mil_ctx'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=3000)
    args = ap.parse_args()
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    stage = stage_dir()
    P = pd.read_parquet(PLAN / 'frames.parquet')
    E = P[P.role == 'eval'].reset_index(drop=True)
    half_of = dict(zip(E.video, E.half))

    def pool_folds(pools, k=2, seed=0):
        return [sorted(v for v in set(pools) if half_of[v] == 'test'), sorted(v for v in set(pools) if half_of[v] == 'train')]

    def inner_val(train_pools, fold, n_val_=None):
        g = np.random.default_rng(100 + fold)
        tp = sorted(train_pools)
        return set(g.choice(tp, max(4, int(round(0.15 * len(tp)))), replace=False).tolist())
    dmp.pool_folds, dmp.inner_val, dmp.BEH = pool_folds, inner_val, list(args.labels)
    t0 = time.time()
    toks, poss, lens, Fs = [], [], [], []
    for half in ('train', 'test'):
        for base in shard_bases(half):
            Fm = pd.read_parquet(TOK / f'{base}_frames.parquet')
            src = TOK / f'{base}_{args.config}_t.npy'
            loc = stage / src.name
            shutil.copyfile(src, loc)
            a = np.load(loc)
            loc.unlink()
            p = np.load(TOK / f'{base}_pos40.npy')
            ln = Fm.n40.values.astype(np.int64)
            st = np.r_[0, np.cumsum(ln)]
            ev = np.flatnonzero((Fm.role == 'eval').values)
            ti = np.concatenate([np.arange(st[i], st[i + 1]) for i in ev])
            toks.append(a[ti])
            poss.append(p[ti])
            lens.append(ln[ev])
            Fs.append(Fm.iloc[ev])
            del a
    Fm = pd.concat(Fs).reset_index(drop=True)
    lens = np.concatenate(lens)
    st = np.r_[0, np.cumsum(lens)]
    order = pd.MultiIndex.from_frame(Fm[['video', 'v']]).get_indexer(pd.MultiIndex.from_frame(E[['video', 'v']]))
    assert (order >= 0).all()
    tok, pos = np.concatenate(toks), np.concatenate(poss)
    idx = np.concatenate([np.arange(st[o], st[o + 1]) for o in order])
    tok, pos, lens = tok[idx], pos[idx], lens[order]
    lab = E.copy()
    lab['pool'] = lab['video']
    lab['n_fg'] = lens
    bag = dmp.Bag(tok, pos, lens, GRID, dev)
    res = dmp.cross_validate(bag, lab, lab.w.values.astype(np.float64), dev, args.probes, args.seeds, None,
                             args.config, args.steps)
    summ = dmp.summarise(res)
    od = OUT / 'probe'
    od.mkdir(parents=True, exist_ok=True)
    (od / f'probe_{args.config}.json').write_text(json.dumps(
        {'config': args.config, 'grid': GRID, 'labels': args.labels, 'folds': 'released halves (fold 0 tests the test '
         'half)', 'summary': summ, 'runs': res, 'wall_s': time.time() - t0, 'gpu': torch.cuda.get_device_name(0)},
        indent=1, default=float))
    for s in summ:
        log(f"{args.config} {s['probe']} {s['behaviour']}: AUROC {s['auroc_mean']:.4f} AP {s['ap_mean']:.4f}")


if __name__ == '__main__':
    main()
