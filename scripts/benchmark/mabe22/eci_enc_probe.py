"""
MABe22 mouse ECI - ENCODER STUDY (rules: eci_encoders.sh): per-layer SUPERVISED CEILING (diagnostic only, GPU job).

Same probe code and settings as eci_probe.py (scripts/eci/diag_mice_patches.cross_validate, probes mean_lin and
mil_ctx, 5 clip-grouped folds, inner validation = 17% of the training clips, IPW-weighted metrics, MIL 3000 steps,
seeds 0 1 2) on the masked t tokens of the eval frames of one encoder tag (eci_enc_encode.py store OUT/encoders/tok/)
and each of --layers. mil_ctx's 3 x 3 context uses the encoder's own grid (--grid).
--tag r3 reads the round-3 store's plain DINOv2 448 tokens (m448_t, last layer) instead (reference = eci_probe A0).
Outputs: OUT/encoders/probe/probe_<tag>_L<layer>.json
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/eci'))
from eci_common import OUT, PLAN, log, stage_dir  # noqa: E402

R3 = OUT / 'tok_r3'
ETOK = OUT / 'encoders' / 'tok'


def eval_keys(store):
    js = [j for j in store.glob('eval_*of*.json') if re.fullmatch(r'eval_\d+of\d+(_.+)?\.json', j.name)]
    n = sorted({int(re.match(r'eval_\d+of(\d+)', j.name).group(1)) for j in js})
    assert len(n) == 1, n
    return [f'eval_{i}of{n[0]}' for i in range(n[0])]


def load(tag, layer, stage):
    toks, poss, lens_all = [], [], []
    for base in eval_keys(R3):
        if tag == 'r3':
            Fm = pd.read_parquet(R3 / f'{base}_frames.parquet')
            files, pf, ln = [R3 / f'{base}_m448_t.npy'], R3 / f'{base}_pos448.npy', Fm.n448.values
        else:
            Fm = pd.read_parquet(ETOK / f'{base}_{tag}_frames.parquet')
            files, pf, ln = [ETOK / f'{base}_{tag}_L{layer}_t.npy'], ETOK / f'{base}_{tag}_pos.npy', Fm.n.values
        loc = stage / files[0].name
        shutil.copyfile(files[0], loc)
        a = np.load(loc)
        loc.unlink()
        a = a.reshape(len(a), -1)
        p = np.load(pf)
        assert len(a) == ln.sum() == len(p), (base, len(a), ln.sum(), len(p))
        toks.append(a)
        poss.append(p)
        lens_all.append(ln.astype(np.int64))
    return np.concatenate(toks), np.concatenate(poss), np.concatenate(lens_all)


def main():
    import torch
    import diag_mice_patches as dmp
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--layers', nargs='+', type=int, default=[0])
    ap.add_argument('--grid', type=int, default=32)
    ap.add_argument('--labels', nargs='+', default=['oral_oral_contact', 'oral_genital_contact'])
    ap.add_argument('--probes', nargs='+', default=['mean_lin', 'mil_ctx'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=3000)
    args = ap.parse_args()
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    stage = stage_dir()
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    out = OUT / 'encoders' / 'probe'
    out.mkdir(parents=True, exist_ok=True)

    def inner_val(train_pools, fold, n_val_=None):
        k = max(4, int(round(0.17 * len(train_pools))))
        return set(np.random.default_rng(100 + fold).choice(sorted(train_pools), k, replace=False).tolist())
    dmp.inner_val = inner_val
    dmp.BEH = list(args.labels)
    for layer in args.layers:
        f = out / f'probe_{args.tag}_L{layer}.json'
        if f.exists():
            log(f'[SKIP] {f.name}')
            continue
        t0 = time.time()
        tok, pos, lens = load(args.tag, layer, stage)
        assert len(lens) == len(E)
        assert pos.max() < args.grid ** 2
        lab = E.copy()
        lab['pool'] = lab['clip']
        lab['n_fg'] = lens
        for b in args.labels:
            lab[b] = lab[b].astype(bool)
        bag = dmp.Bag(tok, pos, lens, args.grid, dev)
        res = dmp.cross_validate(bag, lab, lab.w.values.astype(np.float64), dev, args.probes, args.seeds, None,
                                 f'{args.tag}_L{layer}', args.steps)
        summ = dmp.summarise(res)
        f.write_text(json.dumps({'tag': args.tag, 'layer': layer, 'grid': args.grid, 'dim': int(tok.shape[1]),
                                 'n_tokens': int(len(tok)), 'labels': args.labels, 'summary': summ, 'runs': res,
                                 'wall_s': time.time() - t0, 'gpu': torch.cuda.get_device_name(0)}, indent=1,
                                default=float))
        for s in summ:
            if s['probe'] in args.probes:
                log(f"{args.tag} L{layer} {s['probe']} {s['behaviour']}: AUROC {s['auroc_mean']:.4f} AP {s['ap_mean']:.4f}")
        del bag, tok
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
