"""
Mask-U test (rules: scripts/benchmark/eci_mask_u.sh) - SAE training, theta*, eval read-out and cross-fitted scoring of
one arm x a list of seeds (GPU job). Labels are read ONLY in the scoring step.

Arms: --mask cur (the benchmark's current rule) | U (rule U). Both read the SAME union token store (encode_mabe.py /
encode_calms.py) and keep the tokens flagged in_cur / in_u; frame value = max over the frame's kept tokens (0 if none).
Inputs x = [z_t, z_t - z_{t-60}] (2 s), TokenNorm per 1024-d block. The machinery is unchanged from the benchmarks:
    mabe   scripts/benchmark/mabe22/eci_train.py (train_one, Matryoshka BatchTopK d 4096 prefixes 512/1024/2048/4096,
           k 16, 6000 steps x 4096 tokens; theta* = sae_calibrate on the held-out userTrain calib frames; frame_max;
           score = contact_sae.cross_fit, clip-level halves, split seeds 0, 1, IPW weights) = the encoder study's
           D3L-L20 recipe (eci_enc_train.py), here with bf16 tokens
    calms  scripts/benchmark/calms21/eci_train.py (plain BatchTopK d 4096, k 16, same train_one; frame_pool; video-level
           halves, split seeds 0, 1; labels attack / investigation / mount) = the phase-2 base D3L-64 recipe
Outputs: OUT/<bench>/sae/<arm>_s<seed>/{sae.pt, train.json}, OUT/<bench>/eval/<arm>_s<seed>.json
"""
import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from common import OUT, TOK, log, stage_dir  # noqa: E402


def load_mod(name, path, extra_path):
    sys.path.insert(0, str(extra_path))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def shard_bases(bench, set_):
    js = sorted((TOK / bench).glob(f'{set_}_*of*.json'))
    keys = [j.stem for j in js if 'smoke' not in j.name]
    n = {int(k.split('of')[1]) for k in keys}
    assert len(n) == 1 and len(keys) == list(n)[0], (bench, set_, keys)
    return [f'{set_}_{i}of{list(n)[0]}' for i in range(list(n)[0])]


def load_shard(bench, base, arm, stage):
    d = TOK / bench
    Fm = pd.read_parquet(d / f'{base}_frames.parquet')
    keep = np.load(d / f'{base}_{"incur" if arm == "cur" else "inu"}.npy')
    arr = {}
    for k in ('t', 'p60'):
        src = d / f'{base}_{k}.npy'
        loc = stage / src.name
        shutil.copyfile(src, loc)
        a = np.load(loc, mmap_mode='r')
        assert len(a) == len(keep) == Fm.n.sum(), (base, a.shape, keep.shape)
        arr[k] = np.ascontiguousarray(a[keep])
        del a
        loc.unlink()
    lens = Fm['n_cur' if arm == 'cur' else 'n_u'].values.astype(np.int64)
    # per-frame kept counts recomputed from the flags (must equal the recorded counts)
    seg = np.repeat(np.arange(len(Fm)), Fm.n.values)
    chk = np.bincount(seg, weights=keep.astype(np.float64), minlength=len(Fm)).astype(np.int64)
    assert (chk == lens).all(), (base, arm)
    D = arr['t'].shape[1]
    X = np.empty((len(arr['t']), 2 * D), np.float16)
    for c in range(0, len(X), 400_000):
        ta = arr['t'][c:c + 400_000]
        X[c:c + len(ta), :D] = ta
        X[c:c + len(ta), D:] = (ta.astype(np.float32) - arr['p60'][c:c + 400_000].astype(np.float32)).astype(np.float16)
    return X, lens, Fm


def main():
    import torch
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', choices=['mabe', 'calms'], required=True)
    ap.add_argument('--mask', choices=['cur', 'U'], required=True)
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--steps', type=int, default=6000)
    args = ap.parse_args()
    if args.bench == 'mabe':
        ET = load_mod('mabe_eci_train', REPO / 'scripts/benchmark/mabe22/eci_train.py', REPO / 'scripts/benchmark/mabe22')
        PLAN = REPO / 'results/vision/benchmark_mabe22/eci/plan'
        key = 'clip'
    else:
        ET = load_mod('mabe_eci_train', REPO / 'scripts/benchmark/mabe22/eci_train.py', REPO / 'scripts/benchmark/mabe22')
        CT = load_mod('calms_eci_train', REPO / 'scripts/benchmark/calms21/eci_train.py', REPO / 'scripts/benchmark/calms21')
        PLAN = CT.PLAN
        key = 'video'
    ET.D, ET.WIDTH = 1024, 4096
    if args.bench == 'calms':
        def build(cfg_, din, seed):
            from src.eci.sae import MatryoshkaBatchTopKSAE
            return MatryoshkaBatchTopKSAE(d_in=din, n_latents=4096, prefixes=(4096,), k=16, seed=seed)
        ET.build = build
    matry = args.bench == 'mabe'
    name = args.mask
    sae_cfg = dict(sp='btk', k=16, matry=matry, delta=True, bench=args.bench, mask=args.mask, lag=60,
                   tokens=str(TOK / args.bench), encoder='dinov3l block 20 bf16', width=4096)
    run = OUT / args.bench
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    for d in ('sae', 'eval'):
        (run / d).mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(PLAN / 'train_frames.parquet')
    out = {'train': ([], []), 'calib': ([], [])}
    for base in shard_bases(args.bench, 'train'):
        X, lens, Fm = load_shard(args.bench, base, args.mask, stage)
        role = Fm.merge(P, on=[key, 'i'], how='left').role.values
        assert not pd.isna(role).any()
        tr = np.repeat(role, lens)
        for r in ('train', 'calib'):
            out[r][0].append(X[tr == r])
            out[r][1].append(lens[role == r])
        del X
    Xtr, ltr, Xca, lca = (np.concatenate(v) for r in ('train', 'calib') for v in out[r])
    del out
    log(f'{args.bench} {name}: train {Xtr.shape} on {len(ltr)} frames ({ltr.mean():.1f} tokens / frame), calib '
        f'{Xca.shape} on {len(lca)} frames')
    todo = [s for s in args.seeds if not (run / 'sae' / f'{name}_s{s}' / 'sae.pt').exists()]
    if todo:
        Xg = torch.from_numpy(Xtr).to(dev)
        for seed in todo:
            ET.train_one(name, sae_cfg, seed, Xg, ltr, args.steps, run / 'sae' / f'{name}_s{seed}', dev)
            torch.cuda.empty_cache()
        del Xg
    del Xtr
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    models, calib = {}, {}
    Xca_g = torch.from_numpy(Xca).to(dev)
    for seed in args.seeds:
        sae, norm, ck = ET.load_any(run / 'sae' / f'{name}_s{seed}' / 'sae.pt', dev)
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
    parts = {s: ([], []) for s in args.seeds}
    Fall, lens_all = [], []
    for base in shard_bases(args.bench, 'eval'):
        X, lens, Fm = load_shard(args.bench, base, args.mask, stage)
        Fall.append(Fm)
        lens_all.append(lens)
        for seed, (sae, norm) in models.items():
            c, _, stt = ET.frame_max(sae, norm, X, lens, dev, False)
            parts[seed][0].append(c)
            parts[seed][1].append(stt)
        del X
    Fm = pd.concat(Fall).reset_index(drop=True)
    kidx = pd.MultiIndex.from_frame(Fm[[key, 'i']])
    order = kidx.get_indexer(pd.MultiIndex.from_frame(E[[key, 'i']]))
    assert (order >= 0).all() and len(np.unique(order)) == len(E) == len(Fm)
    lens = np.concatenate(lens_all)[order]
    if args.bench == 'mabe':
        labs = None
        Ht = {s: ET.halves(E, s) for s in (0, 1)}
    else:
        labs = CT.labels()
        Ht = {s: CT.halves(E, s, labs) for s in (0, 1)}
    w = torch.from_numpy(E.w.values).to(dev).double()
    oi = torch.from_numpy(order).to(dev)
    for seed in args.seeds:
        codes = torch.cat(parts[seed][0])[oi].contiguous()
        sts = parts[seed][1]
        ntot = sum(x['n_tokens'] for x in sts)
        stats = {'l0': sum(x['l0'] * x['n_tokens'] for x in sts) / max(ntot, 1), 'n_tokens': ntot,
                 'fve_shard_mean': float(np.mean([x['fve'] for x in sts]))}
        fr = (((codes > 0).double() * w[:, None]).sum(0) / w.sum()).cpu().numpy()
        res = {'bench': args.bench, 'arm': name, 'cfg': sae_cfg, 'seed': seed, 'calib': calib[seed], 'eval_stats': stats,
               'eval_frames': len(E), 'eval_frames_0_tokens': int((lens == 0).sum()),
               'kept_per_eval_frame_w': float(np.average(lens, weights=E.w.values)),
               'firing': {'dead_frac_eval': float((fr == 0).mean()), 'n_eligible_1pct': int((fr >= 0.01).sum()),
                          'median_rate_alive': float(np.median(fr[fr > 0])) if (fr > 0).any() else None},
               'full': 'm4096'}
        res['m4096'] = ET.score(codes, E, Ht, dev) if args.bench == 'mabe' else CT.score(codes, E, Ht, dev, labs)
        r0 = res['m4096']['split0']
        log(f'  {args.bench} {name} s{seed}: ' + ', '.join(f"{b[:12]} {r0[b]['cf_auroc']:.3f}/{r0[b]['cf_ap']:.4f}"
                                                         for b in list(r0)[:5]) + ' (cf AUROC / AP, split 0)')
        (run / 'eval' / f'{name}_s{seed}.json').write_text(json.dumps(res, indent=1, default=float))
        del codes
        torch.cuda.empty_cache()
    log(f'{args.bench} {name} done in {time.time() - t00:.0f}s')


if __name__ == '__main__':
    main()
