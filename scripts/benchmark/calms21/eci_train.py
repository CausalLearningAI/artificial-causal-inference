"""
CalMS21 ECI - SAE training, theta*, eval read-out and cross-fitted scoring for one token config x a list of seeds
(rules: eci_res.sh; GPU Slurm job). Labels are read ONLY in the scoring step.

The machinery is MABe22's (scripts/benchmark/mabe22/eci_train.py, unchanged functions): train_one (TokenNorm.fit_blocks
per D-dim block, b_dec = geometric median, Adam 5e-4, 500 warm-up, cosine, 6000 steps x 4096 tokens, TF32 on for
training, off for calibration / read-out), frame_max (frame value = max over the frame's kept tokens, 0 if none),
theta* (src/eci/sae_calibrate.calibrate_threshold on ALL tokens of the held-out calibration frames), and
contact_sae.cross_fit (best neuron chosen on one half by AUROC / AP / honest top-1% among neurons firing on >= 1% of that
half's weighted frames, scored on the other half, both directions averaged; inverse-probability weights).
Recipe (fixed in eci_res.sh): plain BatchTopK (one prefix, NO Matryoshka) d 4096, k 16, theta*;
inputs x = [z_t, z_t - z_{t-60}] (DINO configs, 2 s at 30 fps) or z_t only (V-JEPA configs, --no-delta).
Train pool = plan/train_frames.parquet role 'train' (task1 train videos, labels never read), calibration = role 'calib';
eval = plan/eval_frames.parquet (task1 test videos), cross-fitted at the VIDEO level: halves(E, seed) for split seeds
0 and 1 (videos holding positives of each scored label dealt alternately, rarest label first, the rest at random).
Outputs: OUT/sae/<config>_s<seed>/{sae.pt, train.json}; OUT/eval/<config>_s<seed>.json
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
sys.path.insert(0, str(HERE.parent / 'mabe22'))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2] / 'scripts/eci'))
from calms_common import OUT, PLAN, TOK, log, stage_dir  # noqa: E402

DIM = {'d2_448': 768, 'd2_896': 768, 'd3_512': 768, 'd3_1024': 768, 'vj_512': 1024, 'vj_1024': 1024,
       'd3_1024lo': 768, 'd2_896lo': 768}
G64 = ('d2_896', 'd3_1024', 'vj_1024', 'd3_1024lo', 'd2_896lo')


def labels():
    return json.loads((PLAN / 'eval_plan.json').read_text())['scored_labels']


def shards(set_, cfg):
    js = sorted(TOK.glob(f'{set_}_*of*_*.json'))
    keys = sorted({j.name.split('_')[1] for j in js if 'smoke' not in j.name and cfg in j.name})
    n = sorted({int(k.split('of')[1]) for k in keys})
    assert len(n) == 1 and len(keys) == n[0], (set_, cfg, keys)
    return [f'{set_}_{i}of{n[0]}' for i in range(n[0])]


def load_shard(base, cfg, delta, stage):
    Fm = pd.read_parquet(TOK / f'{base}_frames.parquet')
    arr = {}
    for k in ('t', 'p60') if delta else ('t',):
        src = TOK / f'{base}_{cfg}_{k}.npy'
        loc = stage / src.name
        shutil.copyfile(src, loc)
        arr[k] = np.load(loc)
        loc.unlink()
    pos = np.load(TOK / f'{base}_{cfg}_pos.npy')
    G = 64 if cfg in G64 else 32
    lens = Fm.n32.values.astype(np.int64) * (4 if G == 64 else 1)
    assert lens.sum() == len(arr['t']) == len(pos), (base, cfg, lens.sum(), arr['t'].shape, pos.shape)
    if not delta:
        return arr['t'], lens, Fm
    D = arr['t'].shape[1]
    X = np.empty((len(arr['t']), 2 * D), np.float16)
    for c in range(0, len(X), 400_000):
        ta = arr['t'][c:c + 400_000]
        X[c:c + len(ta), :D] = ta
        X[c:c + len(ta), D:] = (ta.astype(np.float32) - arr['p60'][c:c + 400_000].astype(np.float32)).astype(np.float16)
    return X, lens, Fm


def halves(E, seed, labs):
    """video-level 2-fold split (as mabe22 eci_train.halves, clips -> videos)."""
    g = np.random.default_rng(seed)
    pos = {b: set(E['video'][E[b]].unique()) for b in labs}
    side = {}
    for b in sorted(labs, key=lambda b: len(pos[b])):
        todo = [c for c in sorted(pos[b]) if c not in side]
        g.shuffle(todo)
        have = [sum(side.get(c) == h for c in pos[b]) for h in (0, 1)]
        h = int(have[1] < have[0])
        for c in todo:
            side[c] = h
            h = 1 - h
    rest = [c for c in sorted(E['video'].unique()) if c not in side]
    g.shuffle(rest)
    for j, c in enumerate(rest):
        side[c] = j % 2
    return E['video'].map(side).values.astype(np.int64)


def score(codes, E, Ht, dev, labs):
    import torch
    import contact_sae as cs
    cs.LABELS = list(labs)
    Y = torch.from_numpy(E[labs].values.astype(bool)).to(dev)
    w = torch.from_numpy(E.w.values.astype(np.float64)).to(dev)
    out = {}
    for s, half in Ht.items():
        r = cs.cross_fit(codes, Y, w, half, dev)
        for b in labs:
            r[b]['cf_rate_auc_pick'] = float(np.mean([d['auc']['test_rate'] for d in r[b]['dirs']]))
        out[f'split{s}'] = r
    return out


def main():
    import torch
    import eci_train as ET
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=list(DIM))
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--run', default='', help='run name (default = config)')
    args = ap.parse_args()
    cfg = args.config
    delta = not cfg.startswith('vj')
    name = args.run or cfg
    ET.D = DIM[cfg]
    ET.WIDTH = 4096
    sae_cfg = dict(sp='btk', k=16, matry=False, delta=delta, config=cfg, width=4096, lag=60 if delta else None)
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    for d in ('sae', 'eval'):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(PLAN / 'train_frames.parquet')
    out = {'train': ([], []), 'calib': ([], [])}
    for base in shards('train', cfg):
        X, lens, Fm = load_shard(base, cfg, delta, stage)
        role = Fm.merge(P, on=['video', 'i'], how='left').role.values
        assert not pd.isna(role).any()
        tr = np.repeat(role, lens)
        for r in ('train', 'calib'):
            out[r][0].append(X[tr == r])
            out[r][1].append(lens[role == r])
        del X
    Xtr, ltr, Xca, lca = (np.concatenate(v) for r in ('train', 'calib') for v in out[r])
    del out
    log(f'{name}: train {Xtr.shape} on {len(ltr)} frames ({ltr.mean():.1f} tokens / frame), calib {Xca.shape} on '
        f'{len(lca)} frames')
    todo = [s for s in args.seeds if not (OUT / 'sae' / f'{name}_s{s}' / 'sae.pt').exists()]
    if todo:
        Xg = torch.from_numpy(Xtr).to(dev)
        for seed in todo:
            ET.train_one(name, sae_cfg, seed, Xg, ltr, args.steps, OUT / 'sae' / f'{name}_s{seed}', dev)
            torch.cuda.empty_cache()
        del Xg
    del Xtr
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    models, calib = {}, {}
    Xca_g = torch.from_numpy(Xca).to(dev)
    for seed in args.seeds:
        sae, norm, ck = ET.load_any(OUT / 'sae' / f'{name}_s{seed}' / 'sae.pt', dev)
        assert sae.n_latents == 4096
        ema = float(sae.threshold)
        th, info = calibrate_threshold(sae, Xca_g, norm=norm)
        set_threshold(sae, th)
        info['theta_ema'] = ema
        info['calib_stats'] = sparsity_stats(sae, Xca_g, th, norm=norm)
        calib[seed] = info
        log(f'  s{seed} theta* {th:.4f} (EMA {ema:.4f}) calib L0 {info["l0_calib_theta_star"]:.3f}')
        models[seed] = (sae, norm)
    del Xca_g
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    labs = labels()
    parts = {s: ([], []) for s in args.seeds}
    Fall, lens_all = [], []
    for base in shards('eval', cfg):
        X, lens, Fm = load_shard(base, cfg, delta, stage)
        Fall.append(Fm)
        lens_all.append(lens)
        for seed, (sae, norm) in models.items():
            c, a, stt = ET.frame_max(sae, norm, X, lens, dev, False)
            parts[seed][0].append(c)
            parts[seed][1].append(stt)
        del X
    Fm = pd.concat(Fall).reset_index(drop=True)
    # eval shards are video-sorted inside; re-order the codes to the plan order
    key = pd.MultiIndex.from_frame(Fm[['video', 'i']])
    order = key.get_indexer(pd.MultiIndex.from_frame(E[['video', 'i']]))
    assert (order >= 0).all() and len(np.unique(order)) == len(E) == len(Fm)
    lens = np.concatenate(lens_all)[order]
    Ht = {s: halves(E, s, labs) for s in (0, 1)}
    for s, h in Ht.items():
        miss = [b for b in labs if min(E[b][h == 0].sum(), E[b][h == 1].sum()) == 0]
        assert not miss, (s, miss)
    w = torch.from_numpy(E.w.values).to(dev).double()
    oi = torch.from_numpy(order).to(dev)
    for seed in args.seeds:
        codes = torch.cat(parts[seed][0])[oi].contiguous()
        sts = parts[seed][1]
        ntot = sum(x['n_tokens'] for x in sts)
        stats = {'l0': sum(x['l0'] * x['n_tokens'] for x in sts) / ntot, 'n_tokens': ntot,
                 'fve_shard_mean': float(np.mean([x['fve'] for x in sts]))}
        fr = (((codes > 0).double() * w[:, None]).sum(0) / w.sum()).cpu().numpy()
        res = {'config': cfg, 'run': name, 'cfg': sae_cfg, 'seed': seed, 'calib': calib[seed], 'eval_stats': stats,
               'eval_frames': len(E), 'eval_frames_0_tokens': int((lens == 0).sum()), 'labels': labs,
               'firing': {'dead_frac_eval': float((fr == 0).mean()), 'n_eligible_1pct': int((fr >= 0.01).sum()),
                          'median_rate_alive': float(np.median(fr[fr > 0])) if (fr > 0).any() else None},
               'halves': {f'split{s}': {'videos_half0': sorted(set(E['video'][h == 0])),
                                        'videos_half1': sorted(set(E['video'][h == 1]))} for s, h in Ht.items()},
               'm4096': score(codes, E, Ht, dev, labs), 'full': 'm4096'}
        r0 = res['m4096']['split0']
        log(f'  {name} s{seed}: ' + ', '.join(f"{b} {r0[b]['cf_auroc']:.3f}/{r0[b]['cf_ap']:.4f}/{r0[b]['cf_top1']:.3f}"
                                              for b in labs) + ' (cf AUROC / AP / honest top-1%, split 0)')
        (OUT / 'eval' / f'{name}_s{seed}.json').write_text(json.dumps(res, indent=1, default=float))
        del codes
        torch.cuda.empty_cache()
    log(f'{name} done in {time.time() - t00:.0f}s')


if __name__ == '__main__':
    main()
