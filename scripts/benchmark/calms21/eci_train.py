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
       'd3_512_L10': 768, 'd3_1024_L10': 768, 'd3l_512_L20': 1024, 'd3l_1024_L20': 1024, 'd2l_448': 1024, 'd2l_448_L20': 1024, 'd2l_896_L20': 1024,
       'd3_1024lo_L10': 768, 'd3l_1024lo_L20': 1024, 'd2_896lo': 768}
G64 = ('d2l_896_L20', 'd2_896', 'd3_1024', 'vj_1024', 'd3_1024_L10', 'd3l_1024_L20', 'd3_1024lo_L10', 'd3l_1024lo_L20', 'd2_896lo')


def labels():
    return json.loads((PLAN / 'eval_plan.json').read_text())['scored_labels']


def shards(set_, cfg):
    js = sorted(TOK.glob(f'{set_}_*of*_*.json'))
    keys = sorted({j.name.split('_')[1] for j in js if 'smoke' not in j.name and cfg in j.name})
    n = sorted({int(k.split('of')[1]) for k in keys})
    assert len(n) == 1 and len(keys) == n[0], (set_, cfg, keys)
    return [f'{set_}_{i}of{n[0]}' for i in range(n[0])]


def load_shard(base, cfg, lag, stage, tok='mask', set_='train'):
    """-> X (N, D or 2D) fp16, lens (F,), frames. tok: 'mask' (pass A / D masked tokens; lag = 0 (z_t only), 30, 60,
    120), 'nomask' (phase 2: train = random video cells of the masked count, eval = all video cells; lag 60 only),
    'orbit' (phase 2: D4-averaged masked tokens; lag 60 only)."""
    Fm = pd.read_parquet(TOK / f'{base}_frames.parquet')
    G = 64 if cfg in G64 else 32
    f = 4 if G == 64 else 1
    if tok == 'mask':
        kt, kp = f'{cfg}_t', f'{cfg}_p{lag}'
    elif tok == 'orbit':
        assert lag in (0, 60)
        kt, kp = f'{cfg}_orb_t', f'{cfg}_orb_p60'
    else:
        assert lag in (0, 60)
        kt, kp = (f'{cfg}_rnd_t', f'{cfg}_rnd_p60') if set_ == 'train' else (f'{cfg}_all_t', f'{cfg}_all_p60')
    arr = {}
    for k, kk in (('t', kt), ('p', kp)) if lag else (('t', kt),):
        src = TOK / f'{base}_{kk}.npy'
        loc = stage / src.name
        shutil.copyfile(src, loc)
        arr[k] = np.load(loc)
        loc.unlink()
    if tok == 'nomask' and set_ != 'train':
        nv = int(len(arr['t']) // len(Fm))
        lens = np.full(len(Fm), nv, np.int64)
    else:
        lens = Fm.n32.values.astype(np.int64) * f
    assert lens.sum() == len(arr['t']), (base, cfg, tok, lens.sum(), arr['t'].shape)
    if tok == 'mask':
        assert len(np.load(TOK / f'{base}_{cfg}_pos.npy')) == lens.sum()
    if not lag:
        return arr['t'], lens, Fm
    D = arr['t'].shape[1]
    X = np.empty((len(arr['t']), 2 * D), np.float16)
    for c in range(0, len(X), 400_000):
        ta = arr['t'][c:c + 400_000]
        X[c:c + len(ta), :D] = ta
        X[c:c + len(ta), D:] = (ta.astype(np.float32) - arr['p'][c:c + 400_000].astype(np.float32)).astype(np.float16)
    return X, lens, Fm


def frame_pool(sae, norm, X, lens, dev, max_tok=16384):
    """-> (max codes, mean codes) (F, m) fp16 GPU over each frame's tokens (0 for a frame without tokens), stats; the
    max is mabe22 eci_train.frame_max's construction, the mean divides the same padded sum by the frame's token count."""
    import torch
    Fn, m = len(lens), sae.n_latents
    st = np.r_[0, np.cumsum(lens)]
    cmax = torch.zeros(Fn, m, dtype=torch.float16, device=dev)
    cmean = torch.zeros(Fn, m, dtype=torch.float16, device=dev)
    s1 = s2 = sse = 0.0
    l0 = 0
    with torch.no_grad():
        f0 = 0
        while f0 < Fn:
            f1, Lm = f0 + 1, int(lens[f0])
            while (f1 < Fn and st[f1 + 1] - st[f0] <= max_tok
                   and (f1 + 1 - f0) * max(Lm, int(lens[f1]), 1) * m <= (1 << 27)):
                Lm = max(Lm, int(lens[f1]))
                f1 += 1
            a, b = st[f0], st[f1]
            if b > a:
                x = norm(torch.from_numpy(np.ascontiguousarray(X[a:b])).to(dev))
                ln = torch.from_numpy(lens[f0:f1]).to(dev)
                seg = torch.repeat_interleave(torch.arange(f1 - f0, device=dev), ln)
                z = sae.encode(x, mode='threshold')
                xh = sae.decode(z)
                s1 = s1 + x.double().sum(0)
                s2 = s2 + x.double().pow(2).sum(0)
                sse = sse + (x - xh).double().pow(2).sum(0)
                l0 += int((z > 0).sum())
                Lm = int(ln.max())
                off = torch.arange(len(z), device=dev) - torch.repeat_interleave(torch.cumsum(ln, 0) - ln, ln)
                pad = torch.zeros(f1 - f0, Lm, m, device=dev)
                pad[seg, off] = z
                cmax[f0:f1] = pad.max(1).values.half()
                cmean[f0:f1] = (pad.sum(1) / ln.clamp_min(1)[:, None].float()).half()
                del pad, z, xh, x
            f0 = f1
    N = int(st[-1])
    tss = s2 - s1.pow(2) / N
    return cmax, cmean, {'fve': float(1 - sse.sum() / tss.sum()), 'l0': l0 / N, 'n_tokens': N}


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
    # phase 2 (eci_recheck.sh); defaults = the grid recipe
    ap.add_argument('--lag', type=int, default=-1, help='-1: 60 for image encoders, 0 for V-JEPA; 0 = z_t only')
    ap.add_argument('--width', type=int, default=4096)
    ap.add_argument('--prefixes', type=int, nargs='*', default=[], help='Matryoshka prefixes (empty = plain)')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--tok', choices=['mask', 'nomask', 'orbit'], default='mask')
    ap.add_argument('--nsplits', type=int, default=2, help='video-level split seeds 0..n-1 (descriptive robustness: 10, '
                    'written to OUT/eval_splits/ instead of OUT/eval/)')
    args = ap.parse_args()
    EVD = 'eval' if args.nsplits == 2 else f'eval_splits{args.nsplits}'
    cfg = args.config
    lag = (0 if cfg.startswith('vj') else 60) if args.lag < 0 else args.lag
    delta = lag > 0
    name = args.run or cfg
    W = args.width
    ET.D = DIM[cfg]
    ET.WIDTH = W
    pre = tuple(args.prefixes) if args.prefixes else (W,)
    assert pre[-1] == W

    def build(cfg_, din, seed):
        from src.eci.sae import MatryoshkaBatchTopKSAE
        return MatryoshkaBatchTopKSAE(d_in=din, n_latents=W, prefixes=pre, k=cfg_['k'], seed=seed)
    ET.build = build
    sae_cfg = dict(sp='btk', k=args.k, matry=len(pre) > 1, prefixes=list(pre), delta=delta, config=cfg, width=W,
                   lag=lag if delta else None, tok=args.tok)
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    for d in ('sae', EVD):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(PLAN / 'train_frames.parquet')
    out = {'train': ([], []), 'calib': ([], [])}
    for base in shards('train', cfg):
        X, lens, Fm = load_shard(base, cfg, lag, stage, args.tok, 'train')
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
        assert sae.n_latents == W
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
    parts = {s: ([], [], []) for s in args.seeds}
    Fall, lens_all = [], []
    for base in shards('eval', cfg):
        X, lens, Fm = load_shard(base, cfg, lag, stage, args.tok, 'eval')
        Fall.append(Fm)
        lens_all.append(lens)
        for seed, (sae, norm) in models.items():
            c, cm, stt = frame_pool(sae, norm, X, lens, dev)
            parts[seed][0].append(c)
            parts[seed][1].append(stt)
            parts[seed][2].append(cm)
        del X
    Fm = pd.concat(Fall).reset_index(drop=True)
    # eval shards are video-sorted inside; re-order the codes to the plan order
    key = pd.MultiIndex.from_frame(Fm[['video', 'i']])
    order = key.get_indexer(pd.MultiIndex.from_frame(E[['video', 'i']]))
    assert (order >= 0).all() and len(np.unique(order)) == len(E) == len(Fm)
    lens = np.concatenate(lens_all)[order]
    Ht = {s: halves(E, s, labs) for s in range(args.nsplits)}
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
                                        'videos_half1': sorted(set(E['video'][h == 1])),
                                        'pos_videos_per_half': {b: [int(E[b][h == x].groupby(E.video[h == x]).any().sum())
                                                                    for x in (0, 1)] for b in labs}}
                          for s, h in Ht.items()},
               f'm{W}': score(codes, E, Ht, dev, labs), 'full': f'm{W}',
               f'mean_m{W}': score(torch.cat(parts[seed][2])[oi].contiguous(), E, Ht, dev, labs)}
        r0 = res[f'm{W}']['split0']
        log(f'  {name} s{seed}: ' + ', '.join(f"{b} {r0[b]['cf_auroc']:.3f}/{r0[b]['cf_ap']:.4f}/{r0[b]['cf_top1']:.3f}"
                                              for b in labs) + ' (cf AUROC / AP / honest top-1%, split 0)')
        (OUT / EVD / f'{name}_s{seed}.json').write_text(json.dumps(res, indent=1, default=float))
        del codes
        torch.cuda.empty_cache()
    log(f'{name} done in {time.time() - t00:.0f}s')


if __name__ == '__main__':
    main()
