"""
MABe22 mouse ECI evaluation: SUPERVISED CEILING on the baseline (A0) eval tokens (GPU Slurm job).

Reuses scripts/eci/diag_mice_patches.cross_validate (probes mean_lin and mil_ctx) on the masked 448 tokens z_t of the
eval frames, clip-grouped: 'pool' = clip, 5 folds over clips (dmp.pool_folds), inner validation = 17% of the training
clips (dmp.inner_val with n_val scaled from the lab's 4 of 24 pools; 4 clips would hold almost no rare positives),
weighted metrics with the eval inverse-probability weights. MIL: 3000 steps, seeds --seeds.
Outputs: OUT/probe/probe_<tag>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/eci'))
from eci_common import BEHAVIOURS, OUT, PLAN, TOK, log, stage_dir  # noqa: E402
from eci_train import load_shard, shards  # noqa: E402


def main():
    import torch
    import diag_mice_patches as dmp
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', nargs='+', default=BEHAVIOURS)
    ap.add_argument('--probes', nargs='+', default=['mean_lin', 'mil_ctx'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--tag', default='A0')
    ap.add_argument('--tok', default='m448', help='m448 (A0 tokens) | m448o (A7 orbit-averaged)')
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    stage = stage_dir()
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    toks, poss, lens_all = [], [], []
    for key in shards('eval'):
        X, lens, Fm = load_shard(key, args.tok, False, stage)
        toks.append(X)
        lens_all.append(lens)
        poss.append(np.load(TOK / f'{key}_pos448.npy'))
    tok, pos, lens = np.concatenate(toks), np.concatenate(poss), np.concatenate(lens_all)
    assert len(lens) == len(E) and len(pos) == len(tok)
    lab = E.copy()
    lab['pool'] = lab['clip']
    lab['n_fg'] = lens
    for b in args.labels:
        lab[b] = lab[b].astype(bool)
    dmp.BEH = list(args.labels)
    n_val = None

    def inner_val(train_pools, fold, n_val_=None):
        k = max(4, int(round(0.17 * len(train_pools))))
        return set(np.random.default_rng(100 + fold).choice(sorted(train_pools), k, replace=False).tolist())
    dmp.inner_val = inner_val
    bag = dmp.Bag(tok, pos, lens, 32, dev)
    res = dmp.cross_validate(bag, lab, lab.w.values.astype(np.float64), dev, args.probes, args.seeds, None, args.tag,
                             args.steps)
    summ = dmp.summarise(res)
    (OUT / 'probe').mkdir(parents=True, exist_ok=True)
    name = f"probe_{args.tag}_{'_'.join(b[:6] for b in args.labels)}.json"
    (OUT / 'probe' / name).write_text(json.dumps({'labels': args.labels, 'summary': summ, 'runs': res, 'n_val': n_val,
                                                  'wall_s': time.time() - t0, 'gpu': torch.cuda.get_device_name(0)},
                                                 indent=1, default=float))
    for s in summ:
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
    log(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
