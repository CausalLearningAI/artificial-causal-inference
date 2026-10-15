"""
MABe22 mouse ECI - ENCODER STUDY (rules: eci_encoders.sh): SAE training, theta*, eval read-out, cross-fitted scoring
for the encoder arms. GPU Slurm job, one arm x a list of seeds.

The machinery is eci_train.py's / eci_r3_train.py's, unchanged: Matryoshka BatchTopK d 4096 (prefixes 512 / 1024 / 2048 /
4096), k 16, TokenNorm.fit_blocks per D-dim block, b_dec = geometric median, Adam 5e-4, 500 warm-up, cosine, 6000 steps
x 4096 tokens (TF32 on for training, off for calibration / read-out); theta* (src/eci/sae_calibrate) on ALL tokens of
the held-out userTrain calibration frames; frame value = max over the frame's kept tokens; eci_train.score =
contact_sae.cross_fit (clip-level halves, split seeds 0 and 1, IPW weights). Train pool = plan/train_frames.parquet
'train' rows, calibration = its 'calib' rows, eval = plan/eval_frames.parquet (the round-3 frames).
Arms (tokens of block 'L' after the final LayerNorm; inputs per token):
    D2       DINOv2-base 448, last layer, [z_t, z_t - z_t-60] (seeds 300-305: the round-3 store's m448_t / m448_p60;
             later seeds: this study's identical re-encode, tag d2_448)
    D2L      DINOv2-large 448 (1024-d), layer --layer, [z_t, z_t - z_t-60]                         (C2 size control)
    D3LO     DINOv3-L/16 512 with D4 orbit tokens (tag d3l_512o), layer 20, [z_t, z_t - z_t-60]     (C3)
    D3B      DINOv3-B/16 512, layer --layer (default 12 = last), [z_t, z_t - z_t-60]
    D3L      DINOv3-L/16 512, layer --layer (default 24 = last), [z_t, z_t - z_t-60]
    VJ       V-JEPA 2-L, --tag (vj_512x / vj_512i / vj_256) and --layer, z_t only
    VJD      the same tokens, [z_t, z_t - z_t-60] (z_t-60 = the token of the clip centred on t - 60)
Outputs: OUT/encoders/sae/<run>_s<seed>/{sae.pt, train.json}; OUT/encoders/eval/<run>_s<seed>.json (eci_train format),
run = <arm>[_<tag>]_L<layer> (D2: 'D2').
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/eci'))
from eci_common import BEHAVIOURS, OUT, PLAN, log, stage_dir  # noqa: E402
import eci_train as ET  # noqa: E402

ETOK = Path(os.environ.get('MABE_ENC_TOK') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/encoders')  # BeeGFS
RUN = OUT / 'encoders'
NSH = {'train': 8, 'eval': 12}


ARMS = ['D2', 'D2L', 'D3B', 'D3L', 'D3LO', 'VJ', 'VJD']


def arm_cfg(arm, tag, layer):
    # 'D2' runs of seeds 300-305 (run name 'D2') used the round-3 store's m448_t / m448_p60 (deleted 12:46 by its owner);
    # later D2 runs use this study's re-encode of the same pipeline (tag d2_448, block 12; max abs diff to m448_t 0.0078,
    # fp16 rounding, all 12 eval shards), run name D2_d2_448_L12.
    D = {'D2': 768, 'D2L': 1024, 'D3B': 768, 'D3L': 1024, 'D3LO': 1024, 'VJ': 1024, 'VJD': 1024}[arm]
    tag = tag or {'D2': 'd2_448', 'D2L': 'd2l_448', 'D3B': 'd3b_512', 'D3L': 'd3l_512', 'D3LO': 'd3l_512o'}[arm]
    layer = layer or {'D2': 12, 'D2L': 24, 'D3B': 12, 'D3L': 24, 'D3LO': 20}[arm]
    return dict(store=ETOK, suffix=f'_{tag}', t=f'L{layer}_t', p=f'L{layer}_p60', lens='n', D=D, delta=arm != 'VJ',
                run=f'{arm}_{tag}_L{layer}')


def load_shard(set_, sh, cfg, stage):
    base = f'{set_}_{sh}of{NSH[set_]}{cfg["suffix"]}'
    Fm = pd.read_parquet(cfg['store'] / f'{base}_frames.parquet')
    arr = {}
    for k in ('t', 'p') if cfg['delta'] else ('t',):
        src = cfg['store'] / f'{base}_{cfg[k]}.npy'
        loc = stage / src.name
        shutil.copyfile(src, loc)
        arr[k] = np.load(loc).reshape(-1, cfg['D'])
        loc.unlink()
    lens = Fm[cfg['lens']].values.astype(np.int64)
    assert lens.sum() == len(arr['t']), (base, lens.sum(), arr['t'].shape)
    if not cfg['delta']:
        return arr['t'], lens, Fm
    D = cfg['D']
    X = np.empty((len(arr['t']), 2 * D), np.float16)
    for c in range(0, len(X), 400_000):
        ta = arr['t'][c:c + 400_000]
        X[c:c + len(ta), :D] = ta
        X[c:c + len(ta), D:] = (ta.astype(np.float32) - arr['p'][c:c + 400_000].astype(np.float32)).astype(np.float16)
    return X, lens, Fm


def main():
    import torch
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--steps', type=int, default=6000)
    args = ap.parse_args()
    cfg = arm_cfg(args.arm, args.tag, args.layer)
    name = cfg['run']
    ET.D = cfg['D']  # TokenNorm.fit_blocks block size in eci_train.train_one
    ET.WIDTH = 4096
    sae_cfg = dict(sp='btk', k=16, matry=True, delta=cfg['delta'], encoder_arm=args.arm, tokens=str(cfg['store']),
                   suffix=cfg['suffix'], kinds=[cfg['t'], cfg['p']] if cfg['delta'] else [cfg['t']], D=cfg['D'])
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    for d in ('sae', 'eval'):
        (RUN / d).mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(PLAN / 'train_frames.parquet')
    out = {'train': ([], []), 'calib': ([], [])}
    for sh in range(NSH['train']):
        X, lens, Fm = load_shard('train', sh, cfg, stage)
        role = Fm.merge(P, on=['clip', 'i'], how='left').role.values
        assert not pd.isna(role).any()
        tr = np.repeat(role, lens)
        for r in ('train', 'calib'):
            out[r][0].append(X[tr == r])
            out[r][1].append(lens[role == r])
        del X
    Xtr, ltr, Xca, lca = (np.concatenate(v) for r in ('train', 'calib') for v in out[r])
    del out
    log(f'{name}: train {Xtr.shape} on {len(ltr)} frames, calib {Xca.shape} on {len(lca)} frames')
    todo = [s for s in args.seeds if not (RUN / 'sae' / f'{name}_s{s}' / 'sae.pt').exists()]
    if todo:
        Xg = torch.from_numpy(Xtr).to(dev)
        for seed in todo:
            ET.train_one(name, sae_cfg, seed, Xg, ltr, args.steps, RUN / 'sae' / f'{name}_s{seed}', dev)
            torch.cuda.empty_cache()
        del Xg
    del Xtr
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    models, calib = {}, {}
    Xca_g = torch.from_numpy(Xca).to(dev)
    for seed in args.seeds:
        sae, norm, ck = ET.load_any(RUN / 'sae' / f'{name}_s{seed}' / 'sae.pt', dev)
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
    parts = {s: ([], []) for s in args.seeds}
    Fall, lens_all = [], []
    for sh in range(NSH['eval']):
        X, lens, Fm = load_shard('eval', sh, cfg, stage)
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
        res = {'arm': args.arm, 'run': name, 'cfg': sae_cfg, 'seed': seed, 'calib': calib[seed], 'eval_stats': stats,
               'eval_frames': len(E), 'eval_frames_0_tokens': int((lens == 0).sum()),
               'firing': {'dead_frac_eval': float((fr == 0).mean()), 'n_eligible_1pct': int((fr >= 0.01).sum()),
                          'median_rate_alive': float(np.median(fr[fr > 0])) if (fr > 0).any() else None}}
        for m in (512, 1024, 2048, 4096):
            res[f'm{m}'] = ET.score(codes[:, :m].contiguous(), E, Ht, dev)
            r0 = res[f'm{m}']['split0']
            log(f'  {name} s{seed} m{m}: ' + ', '.join(f"{b[:10]} {r0[b]['cf_auroc']:.3f}/{r0[b]['cf_ap']:.4f}"
                                                       for b in BEHAVIOURS[:5]))
        res['full'] = 'm4096'
        (RUN / 'eval' / f'{name}_s{seed}.json').write_text(json.dumps(res, indent=1, default=float))
        del codes
        torch.cuda.empty_cache()
    log(f'{name} done in {time.time() - t00:.0f}s')


if __name__ == '__main__':
    main()
