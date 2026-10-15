"""
MABe22 mouse ECI round 3 (rules: eci_round3.sh): SAE training on the round-3 token store (MABE_TOK = tok_r3),
theta* calibration, eval read-out, cross-fitted scoring. GPU Slurm job, one arm x a list of seeds.

Same machinery as eci_train.py (train_one, frame_max, score, halves, load_any; the SAE recipe, 6000 steps x 4096 tokens,
theta* on the held-out userTrain calibration frames, max over each frame's kept tokens), generalised to:
    width     d (Matryoshka prefixes d/8, d/4, d/2, d, or one prefix d without Matryoshka); --width overrides the arm's
              width and the run is then named <arm>_w<d>
    inputs    blocks concatenated per token (TokenNorm per 768 block): ('t', kind) / ('d', kind_t, kind_lag) = z_t - z_lag
              / ('m', [kinds]) = mean of those tokens
    store     key suffix '' (main store), '_s336_672' (E3 scale store), '_adapt' (E6 adapted-encoder store)
    mix       E5: the mouse train pool + an equal number of beetle train tokens (btrain_* files, uniform subsample, rng 0),
              theta* on mouse calib + an equal number of beetle calib tokens
Arms (orbit tokens unless named):
    R          [z_t, z_t - z_{t-60}], d 4096 (read-outs 512 / 1024 / 2048 / 4096)        E1 / E2 / E3 / E5 / E6 control
    R512       d 512 Matryoshka (read-outs 64 / 128 / 256 / 512); R512plain d 512 single prefix      E1
    R512m128 / R512m128b   d 512, prefixes 128/256/512 / 128/512                                     E1b
    L15 L30 L120   [z_t, z_t - z_{t-L}] (L60 = R)                                                     E2 per-lag
    CM         [z_t, mean(z_{t-15}, z_{t-30}, z_{t-45}, z_{t-60})]                                   E2 context-mean
    ML         [z_t, z_t - z_{t-30}, z_t - z_{t-60}, z_t - z_{t-120}]                                E2 multi-lag concat
    R336 R672  R on the 336 / 672-px orbit tokens (mask transferred by patch centre)                  E3
    RA         R on the adapted-encoder tokens                                                        E6
    RMIX       R trained on mouse + beetle tokens                                                     E5
Outputs: OUT/r3/sae/<run>_s<seed>/{sae.pt, train.json}; OUT/r3/eval/<run>_s<seed>.json (eci_train format);
         --save-codes: OUT/r3/codes/<run>_s<seed>.npy (eval frame codes, fp16) for the E2 pooled selection.
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
from eci_common import BEHAVIOURS, OUT, PLAN, TOK, log, stage_dir  # noqa: E402
import eci_train as ET  # noqa: E402

RUN = OUT / 'r3'
D = 768
o = 'm448o'
ARMS = {
    'R': dict(inp=[('t', f'{o}_t'), ('d', f'{o}_t', f'{o}_p60')], width=4096, matry=True),
    'R512': dict(inp=[('t', f'{o}_t'), ('d', f'{o}_t', f'{o}_p60')], width=512, matry=True),
    'R512plain': dict(inp=[('t', f'{o}_t'), ('d', f'{o}_t', f'{o}_p60')], width=512, matry=False),
    'L15': dict(inp=[('t', f'{o}_t'), ('d', f'{o}_t', f'{o}_p15')], width=4096, matry=True),
    'L30': dict(inp=[('t', f'{o}_t'), ('d', f'{o}_t', f'{o}_p')], width=4096, matry=True),
    'L120': dict(inp=[('t', f'{o}_t'), ('d', f'{o}_t', f'{o}_p120')], width=4096, matry=True),
    'CM': dict(inp=[('t', f'{o}_t'), ('m', [f'{o}_p15', f'{o}_p', f'{o}_p45', f'{o}_p60'])], width=4096, matry=True),
    'ML': dict(inp=[('t', f'{o}_t'), ('d', f'{o}_t', f'{o}_p'), ('d', f'{o}_t', f'{o}_p60'),
                    ('d', f'{o}_t', f'{o}_p120')], width=4096, matry=True),
    'R336': dict(inp=[('t', 'm336o_t'), ('d', 'm336o_t', 'm336o_p60')], width=4096, matry=True, store='_s336_672',
                 lens='n336'),
    'R672': dict(inp=[('t', 'm672o_t'), ('d', 'm672o_t', 'm672o_p60')], width=4096, matry=True, store='_s336_672',
                 lens='n672'),
    'RA': dict(inp=[('t', 'ma448o_t'), ('d', 'ma448o_t', 'ma448o_p60')], width=4096, matry=True, store='_adapt'),
    'R512m128': dict(inp=[('t', f'{o}_t'), ('d', f'{o}_t', f'{o}_p60')], width=512, matry=True, prefixes=(128, 256, 512)),
    'R512m128b': dict(inp=[('t', f'{o}_t'), ('d', f'{o}_t', f'{o}_p60')], width=512, matry=True, prefixes=(128, 512)),
    'RMIX': dict(inp=[('t', f'{o}_t'), ('d', f'{o}_t', f'{o}_p60')], width=4096, matry=True, mix=True),
}
for a in ARMS.values():
    a.update(sp='btk', k=16, delta=True, tok=a['inp'][0][1])


def shards(set_):
    js = [j for j in TOK.glob(f'{set_}_*of*.json') if re.fullmatch(rf'{set_}_\d+of\d+\.json', j.name)]
    meta = sorted((json.loads(j.read_text()) for j in js), key=lambda m: m['shard'])
    assert meta and len(meta) == meta[0]['nshard'], [j.name for j in js]
    return [f"{set_}_{m['shard']}of{m['nshard']}" for m in meta]


def kinds(cfg):
    ks = []
    for b in cfg['inp']:
        for k in ([b[1]] if b[0] == 't' else [b[1], b[2]] if b[0] == 'd' else b[1]):
            if k not in ks:
                ks.append(k)
    return ks


def load_shard(key, cfg, stage):
    """-> X (N, 768 x blocks) fp16, lens (F,), frames DataFrame (tokens copied to local disk first)."""
    Fm = pd.read_parquet(TOK / f'{key}_frames.parquet')
    arr = {}
    for k in kinds(cfg):
        src = TOK / f'{key}_{k}.npy'
        loc = stage / src.name
        shutil.copyfile(src, loc)
        arr[k] = np.load(loc).reshape(-1, D)
        loc.unlink()
    lens = Fm[cfg.get('lens', 'n448')].values.astype(np.int64)
    N = len(next(iter(arr.values())))
    assert lens.sum() == N and all(len(a) == N for a in arr.values()), (key, lens.sum(), N)
    X = np.empty((N, D * len(cfg['inp'])), np.float16)
    for c in range(0, N, 400_000):
        sl = slice(c, c + 400_000)
        for j, b in enumerate(cfg['inp']):
            if b[0] == 't':
                v = arr[b[1]][sl]
            elif b[0] == 'd':
                v = (arr[b[1]][sl].astype(np.float32) - arr[b[2]][sl].astype(np.float32)).astype(np.float16)
            else:
                v = np.mean([arr[k][sl].astype(np.float32) for k in b[1]], 0).astype(np.float16)
            X[sl, j * D:(j + 1) * D] = v
    return X, lens, Fm


def load_train(cfg, stage):
    P = pd.read_parquet(PLAN / 'train_frames.parquet')
    st = cfg.get('store', '')
    out = {'train': ([], []), 'calib': ([], [])}
    for key in shards('train'):
        X, lens, Fm = load_shard(key + st, cfg, stage)
        role = Fm.merge(P, on=['clip', 'i'], how='left').role.values
        assert not pd.isna(role).any() and (Fm[['clip', 'i']].values == P[P['clip'].isin(Fm['clip'])][['clip', 'i']].values).all()
        tr = np.repeat(role, lens)
        for r in ('train', 'calib'):
            out[r][0].append(X[tr == r])
            out[r][1].append(lens[role == r])
    Xtr, ltr, Xca, lca = (np.concatenate(v) for r in ('train', 'calib') for v in out[r])
    mixinfo = None
    if cfg.get('mix'):
        bt, bc = [], []
        for key in shards('btrain'):
            X, lens, Fm = load_shard(key, cfg, stage)
            tr = np.repeat(Fm.role.values, lens)
            bt.append(X[tr == 'train'])
            bc.append(X[tr == 'calib'])
        bt, bc = np.concatenate(bt), np.concatenate(bc)
        g = np.random.default_rng(0)
        assert len(bt) >= len(Xtr) and len(bc) >= len(Xca), (len(bt), len(Xtr), len(bc), len(Xca))
        bt = bt[np.sort(g.choice(len(bt), len(Xtr), replace=False))]
        bc = bc[np.sort(g.choice(len(bc), len(Xca), replace=False))]
        mixinfo = {'mouse_train_tokens': len(Xtr), 'beetle_train_tokens': len(bt), 'mouse_calib_tokens': len(Xca),
                   'beetle_calib_tokens': len(bc)}
        Xtr, Xca = np.concatenate([Xtr, bt]), np.concatenate([Xca, bc])
        ltr = np.r_[ltr, np.ones(len(bt), np.int64)]  # lens unused for BatchTopK training
    return Xtr, ltr, Xca, lca, mixinfo


def main():
    import torch
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=list(ARMS), required=True)
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--width', type=int, default=0)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--save-codes', action='store_true')
    args = ap.parse_args()
    cfg = dict(ARMS[args.arm])
    name = args.arm
    if args.width:
        cfg['width'], name = args.width, f'{args.arm}_w{args.width}'
    W = cfg['width']
    ET.WIDTH = W  # eci_train.build reads the module-level width (prefixes W/8, W/4, W/2, W)
    readouts = {4096: (512, 1024, 2048, 4096), 512: (64, 128, 256, 512)}.get(W, (W,)) if cfg['matry'] else (W,)
    if cfg.get('prefixes'):  # E1b: explicit Matryoshka prefixes
        from src.eci.sae import MatryoshkaBatchTopKSAE
        pre = tuple(cfg['prefixes'])
        ET.build = lambda c, din, seed: MatryoshkaBatchTopKSAE(d_in=din, n_latents=W, prefixes=pre, k=c['k'], seed=seed)
        readouts = pre
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    for d in ('sae', 'eval', 'codes'):
        (RUN / d).mkdir(parents=True, exist_ok=True)
    todo = [s for s in args.seeds if not (RUN / 'sae' / f'{name}_s{s}' / 'sae.pt').exists()]
    Xtr, ltr, Xca, lca, mixinfo = load_train(cfg, stage)
    log(f'{name} {cfg}: train {Xtr.shape}, calib {Xca.shape}, mix {mixinfo}')
    if todo:
        Xg = torch.from_numpy(Xtr).to(dev)
        for seed in todo:
            ET.train_one(name, cfg, seed, Xg, ltr, args.steps, RUN / 'sae' / f'{name}_s{seed}', dev)
            torch.cuda.empty_cache()
        del Xg
    del Xtr
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    models, calib = {}, {}
    Xca_g = torch.from_numpy(Xca).to(dev)
    for seed in args.seeds:
        sae, norm, ck = ET.load_any(RUN / 'sae' / f'{name}_s{seed}' / 'sae.pt', dev)
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
    parts = {s: ([], []) for s in args.seeds}
    Fall, lens_all = [], []
    for key in shards('eval'):
        X, lens, Fm = load_shard(key + cfg.get('store', ''), cfg, stage)
        Fall.append(Fm)
        lens_all.append(lens)
        for seed, (sae, norm) in models.items():
            c, a, stt = ET.frame_max(sae, norm, X, lens, dev, False)
            parts[seed][0].append(c)
            parts[seed][1].append(stt)
        del X
    Fm = pd.concat(Fall).reset_index(drop=True)
    assert (Fm['clip'].values == E['clip'].values).all() and (Fm.i.values == E.i.values).all()
    lens = np.concatenate(lens_all)
    Ht = {s: ET.halves(E, s) for s in (0, 1)}
    w = torch.from_numpy(E.w.values).to(dev).double()
    for seed in args.seeds:
        codes = torch.cat(parts[seed][0])
        sts = parts[seed][1]
        ntot = sum(x['n_tokens'] for x in sts)
        stats = {'l0': sum(x['l0'] * x['n_tokens'] for x in sts) / ntot, 'n_tokens': ntot,
                 'fve_shard_mean': float(np.mean([x['fve'] for x in sts]))}
        fr = (((codes > 0).double() * w[:, None]).sum(0) / w.sum()).cpu().numpy()
        res = {'arm': args.arm, 'run': name, 'cfg': cfg, 'seed': seed, 'calib': calib[seed], 'eval_stats': stats,
               'mix': mixinfo, 'eval_frames': len(E), 'eval_frames_0_tokens': int((lens == 0).sum()),
               'firing': {'dead_frac_eval': float((fr == 0).mean()), 'n_eligible_1pct': int((fr >= 0.01).sum()),
                          'median_rate_alive': float(np.median(fr[fr > 0])) if (fr > 0).any() else None}}
        for m in readouts:
            res[f'm{m}'] = ET.score(codes[:, :m].contiguous(), E, Ht, dev)
            r0 = res[f'm{m}']['split0']
            log(f'  {name} s{seed} m{m}: ' + ', '.join(f"{b[:10]} {r0[b]['cf_auroc']:.3f}/{r0[b]['cf_ap']:.4f}"
                                                       for b in BEHAVIOURS[:5]))
        res['full'] = f'm{W}'
        (RUN / 'eval' / f'{name}_s{seed}.json').write_text(json.dumps(res, indent=1, default=float))
        if args.save_codes:
            np.save(RUN / 'codes' / f'{name}_s{seed}.npy', codes.cpu().numpy())
        del codes
        torch.cuda.empty_cache()
    log(f'{name} done in {time.time() - t00:.0f}s')


if __name__ == '__main__':
    main()
