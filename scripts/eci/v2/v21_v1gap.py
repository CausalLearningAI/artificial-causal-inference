"""
ECI v2.1 ants v1-gap decomposition: why does v2 not beat v1 on ants? Pre-registration: scripts/eci/v2/v21_v1gap_prereg.md
(committed before any score of this file existed).

Chain (every rung scored on the SAME 127,200 ants eval frames, labels, cells, splits and scorer as the ladder,
scripts/eci/v2/v21_ladder.py; readout = RAW frame max of the threshold-mode token codes, the readout v1 is scored with):
    v1      DINOv2-B/14 @448 (32 x 32 grid, from the 512 px JPG), last layer, v1 mask (FgBackgrounds rule 'ants'),
            v1 SAE (1024, 5 epochs on v1's v2 + v3 store), 11 seeds (existing ladder rows v1_s0-10)
    D2r1    v1's tokens of the eval frames (same encoder / grid / mask), SAE 'r1' (1024 latents, prefixes 128-1024,
            k 16, 8,380 steps x 4096 = v1's step count, global TokenNorm) trained on these frames' tokens
    D2r2    same tokens, SAE 'r2' = the frozen v2 recipe (2048, prefixes 128/256/512/2048, k 16, 30k x 4096)
    D3r1/r2 DINOv3-L/16 block 20 (+ final LN; the v2 encoder) on the SAME 512 px JPG = the same 32 x 32 grid and the SAME
            v1 mask positions; recipes r1 / r2
    SN2raw  the existing SN2 models (DINOv3 native side 768-832, N2 mask, v2.1 training sample at 5 fps, r2) under the raw
            frame max (no HMM)
    SN2     the ladder row (P(on) readout)
SAE training sample (D2*, D3*): the kept tokens of the eval frames themselves (1 fps, all 4 series; v1 and v2 also train
on most scored videos), videos split by rng 0 into train 90% / calibration 5% / evaluation 5% (theta* / FVE); every
video is scored. Seeds 1, 2, 3.

Steps
  tok                 (GPU) tokens of both encoders at the v1 mask positions -> BEE/tok_{d2,d3}.f16, frame.npy, pos.npy
  train --arm A       (GPU) seeds 1-3 of arm A + frame-max codes of all eval frames -> BEE/codes/<A>_s<seed>.npy
  cols --keys k,...   (CPU) ladder scorer (v21_ladder.score2: gate + focal cells) -> OUT/ladder/ants/<key>.json, BEE npz
  boot                (CPU) the ladder bootstrap (same replicates as boot_ants.npz) for the new keys + report
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
import v21_ladder as L  # noqa: E402

T = L.T
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_1/v1gap')
OUT = T.OUT / 'v1gap'
LB0, LO0 = L.LB, L.LO  # the ladder's own rows (v1, SN2)
L.LB, L.LO = BEE / 'ladder', OUT / 'ladder'
R1 = dict(W=1024, prefixes=[128, 256, 512, 1024], k=16, steps=8380, norm='global')
R2 = dict(W=2048, prefixes=[128, 256, 512, 2048], k=16, steps=30000, norm='global')
ARMS = {'D2r1': ('d2', R1), 'D2r2': ('d2', R2), 'D3r1': ('d3', R1), 'D3r2': ('d3', R2)}
SEEDS = (1, 2, 3)
log = L.log


def eval_frames():
    import eval as EV
    return pd.read_parquet(EV.BEE / 'ants' / 'frames.parquet')


# ---------------------------------------------------------------------------------------------- tokens
class RawRGB:
    """processor2 shim for FrameDatasetFG: the 512 px RGB frame as uint8 (3, 512, 512) (the v2 Encoder normalises)."""

    def __call__(self, images, return_tensors='pt'):
        import torch
        return {'pixel_values': torch.from_numpy(np.asarray(images, np.uint8).copy()).permute(2, 0, 1)[None]}


def cmd_tok(a):
    import torch
    import eval as EV
    from src.eci.foreground import RULES, FgBackgrounds, FgEncoder, FrameDatasetFG, encode_batch, obs_rows
    from src.eci.sae import load_sae
    from src.eci.v2.encode import Encoder
    dev = torch.device('cuda')
    F = eval_frames()
    enc = FgEncoder('dinov2_base', dev)
    e3 = Encoder(dev)
    s0, n0 = load_sae(REPO / EV.V1SAE['ants'], dev)[:2]
    ref = np.load(EV.BEE / 'ants/codes/v1_s0.npy', mmap_mode='r')
    loc = Path(os.environ.get('LOCAL_DIR', '/tmp')) / 'v1gap_tok'
    loc.mkdir(parents=True, exist_ok=True)
    X2, X3, FR, PO = [], [], [], []
    chk = {'frames': 0, 'fire_agree': 0.0, 'max_abs_diff': 0.0}
    t0 = time.time()
    for src, g in F.groupby('src'):
        ann = REPO / EV.ANN[src]
        fp = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
        fb = FgBackgrounds(REPO / 'dataset/ants/eci30/fg448/background', obs_rows(ann), RULES['ants'], dev, align='none',
                           cache=8)
        rows = g.row.values.astype(np.int64)
        ds = FrameDatasetFG([str(REPO / 'dataset' / p) for p in fp[rows]], enc.processor, rows, RawRGB())
        dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, pin_memory=True, prefetch_factor=4,
                                         num_workers=max(2, int(os.environ.get('SLURM_CPUS_PER_TASK', 8)) - 2))
        gi, o = g.index.values, 0
        for pix, grey, rr, rgb in dl:
            with torch.no_grad():
                tok = encode_batch(enc.model, pix, dev)
                mk, _ = fb.mask(tok, grey.to(dev), rr.numpy())
                fi, pi = torch.nonzero(mk, as_tuple=True)
                z3 = e3(rgb.to(dev))
                assert z3.shape[1] == tok.shape[1] == 1024
                X2.append(tok[fi, pi].cpu())
                X3.append(z3[fi, pi].cpu())
                fglob = torch.from_numpy(gi[o:o + len(tok)]).to(dev)[fi]
                FR.append(fglob.int().cpu())
                PO.append(pi.short().cpu())
                if chk['frames'] < 20_000:  # identity: the deployed v1 SAE on these tokens vs the stored v1_s0 codes
                    z = s0.encode(n0(tok[fi, pi]), mode='threshold').float()
                    mx = torch.zeros(len(tok), z.shape[1], device=dev).index_reduce_(0, fi, z, 'amax', include_self=True)
                    rc = torch.from_numpy(np.asarray(ref[gi[o:o + len(tok)]], np.float32)).to(dev)
                    chk['fire_agree'] += float(((mx > 0) == (rc > 0)).float().mean()) * len(tok)
                    chk['max_abs_diff'] = max(chk['max_abs_diff'], float((mx - rc).abs().max()))
                    chk['frames'] += len(tok)
            o += len(tok)
        assert o == len(g)
        log(f'{src}: {len(g):,} frames, {time.time() - t0:.0f}s; d3 GPU {e3.gpu_s:.0f}s')
    chk['fire_agree'] /= max(1, chk['frames'])
    if chk['fire_agree'] < 0.999:
        raise SystemExit(f'IDENTITY CHECK FAILED (v1 tokens): {chk}')
    x2, x3 = torch.cat(X2).numpy(), torch.cat(X3).numpy()
    fr, po = torch.cat(FR).numpy(), torch.cat(PO).numpy()
    o = np.argsort(fr, kind='stable')
    np.save(loc / 'tok_d2.npy', x2[o])
    del x2
    np.save(loc / 'tok_d3.npy', x3[o])
    np.save(loc / 'frame.npy', fr[o])
    np.save(loc / 'pos.npy', po[o])
    meta = {'frames': len(F), 'tokens': int(len(fr)), 'tokens_per_frame': float(len(fr) / len(F)),
            'frames_without_token': int((np.bincount(fr, minlength=len(F)) == 0).sum()), 'identity_v1_s0': chk,
            'wall_s': time.time() - t0, 'd3_gpu_s': e3.gpu_s}
    BEE.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in loc.iterdir():
        shutil.copy2(f, BEE / f.name)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'tok_meta.json').write_text(json.dumps(meta, indent=1))
    log(json.dumps(meta))


# ---------------------------------------------------------------------------------------------- train + codes
def roles(F):
    obs = np.array(sorted(F.obs.unique()))
    p = np.random.default_rng(0).permutation(len(obs))
    n = max(1, int(round(0.05 * len(obs))))
    r = pd.Series('train', index=obs)
    r.iloc[p[:n]] = 'calib'
    r.iloc[p[n:2 * n]] = 'eval'
    return r


def cmd_train(a):
    import shutil
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    F = eval_frames()
    enc, cfg = ARMS[a.arm]
    loc = Path(os.environ.get('LOCAL_DIR', '/tmp')) / 'v1gap_tr'
    loc.mkdir(parents=True, exist_ok=True)
    for f in (f'tok_{enc}.npy', 'frame.npy'):
        if not (loc / f).exists():
            shutil.copy2(BEE / f, loc / f)
    X = np.load(loc / f'tok_{enc}.npy', mmap_mode='r')
    fr = np.load(loc / 'frame.npy')
    rv = roles(F).reindex(F.obs.values).values[fr]
    rng = np.random.default_rng(0)
    sel = {}
    for r, cap in (('train', None), ('calib', 1_500_000), ('eval', 1_500_000)):
        ix = np.flatnonzero(rv == r)
        if cap and len(ix) > cap:
            ix = np.sort(rng.choice(ix, cap, replace=False))
        sel[r] = ix
    Xtr = torch.from_numpy(np.ascontiguousarray(X[sel['train']])).to(dev)
    Xca = torch.from_numpy(np.ascontiguousarray(X[sel['calib']]))
    Xev = torch.from_numpy(np.ascontiguousarray(X[sel['eval']]))
    log(f'{a.arm}: train {len(Xtr):,} / calib {len(Xca):,} / eval {len(Xev):,} tokens, d {X.shape[1]}')
    (BEE / 'codes').mkdir(parents=True, exist_ok=True)
    (OUT / 'train').mkdir(parents=True, exist_ok=True)
    fl = torch.from_numpy(fr.astype(np.int64)).to(dev)
    for s in a.seeds:
        key = f'{a.arm}_s{s}'
        od = BEE / 'sae' / key
        rec = SA.train(cfg, s, Xtr, Xca, Xev, od, dev)
        (OUT / 'train' / f'{key}.json').write_text(json.dumps({k: v for k, v in rec.items() if k != 'history'}, indent=1,
                                                              default=float))
        sae, norm = SA.load(od / 'sae_theta.pt', dev)
        cm = torch.zeros((len(F), cfg['W']), dtype=torch.float32, device=dev)
        with torch.no_grad():
            for b in range(0, len(X), 500_000):
                x = torch.from_numpy(np.ascontiguousarray(X[b:b + 500_000])).to(dev)
                c = sae.encode(norm(x.float()), mode='threshold').float()
                cm.index_reduce_(0, fl[b:b + 500_000], c, 'amax', include_self=True)
        np.save(BEE / 'codes' / f'{key}.npy', cm.half().cpu().numpy())
        log(f'{key}: fve {rec["eval_fve"]:.4f}, theta* {rec["theta_star"]:.3f}, active share '
            f'{(cm > 0).float().mean().item():.4f}, {rec["train_time_s"]:.0f}s')
        del cm, sae
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- scoring
def readout(dom, key, agg, F):
    """new keys: <arm>_s<seed> (codes of this file) or SN2raw_s<seed> (the ladder SN2 model, raw frame max)."""
    assert dom == 'ants' and not agg
    if key.startswith('SN2raw_'):
        base = key.replace('SN2raw', 'SN2')
        od = T.BEE / dom / 'codes'
        R = pd.read_parquet(od / 'eval_rows_n2.parquet')
        E = T.frame_sets(dom)['eval']
        rix = pd.Series(np.arange(len(R)), index=pd.MultiIndex.from_frame(R)).reindex(pd.MultiIndex.from_frame(E)).values
        assert not np.isnan(rix).any()
        raw = np.load(od / f'{base}_eval.npy').astype(np.float32)
        return raw[rix.astype(np.int64)], None
    X = np.load(BEE / 'codes' / f'{key}.npy').astype(np.float32)
    assert len(X) == len(F)
    return X, None


def cmd_cols(a):
    L.readout = readout
    for k in a.keys.split(','):
        L.cmd_cols(argparse.Namespace(dom='ants', model=k, agg=None))


RUNGS = {'v1': [f'v1_s{i}' for i in range(11)], 'SN2': [f'SN2_s{i}' for i in (1001, 1002, 1003)],
         'SN2raw': [f'SN2raw_s{i}' for i in (1001, 1002, 1003)],
         **{arm: [f'{arm}_s{s}' for s in SEEDS] for arm in ARMS}}
CONTRASTS = [('D2r1', 'v1'), ('D2r2', 'D2r1'), ('D3r1', 'D2r1'), ('D3r2', 'D2r2'), ('D3r2', 'D3r1'), ('SN2raw', 'D3r2'),
             ('SN2', 'SN2raw'), ('SN2raw', 'v1'), ('SN2', 'v1'), ('D3r2', 'v1'), ('D3r1', 'v1'), ('D2r2', 'v1'),
             ('SN2', 'D3r2')]
MN = ('AUROC', 'nAP', 'top1_prec')


def cmd_boot(a):
    from multiprocessing import get_context
    F, SP = L.frames('ants')
    rng = np.random.default_rng(L.SEED)  # identical replicates to boot_ants.npz (v21_ladder.cmd_boot)
    Wser = {}
    for ser in sorted(F.series.unique()):
        units = np.array(sorted(set(F.unit.values[F.series.values == ser])))
        W = np.stack([np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)) for _ in range(L.NB)])
        Wser[ser] = (units, W.astype(np.float64))
    new = [k for r in RUNGS for k in RUNGS[r] if (L.LO / 'ants' / f'{k}.json').exists()]
    bf = BEE / 'boot_new.npz'
    old = dict(np.load(bf)) if bf.exists() else {}
    todo = [k for k in new if not any(x.startswith(f'{k}|') for x in old)]
    jobs = [('ants', t, p, Wser) for t in todo for p in ('gate', 'focal')]
    with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        for tag, part, out in pool.imap_unordered(L.model_boot, jobs):
            for c, m in out.items():
                if m is not None:
                    old[f'{tag}|{part}|{c}'] = m
            log(f'boot {tag} {part} done')
    # identity: column 0 vs the scorer, new keys
    mx = 0.0
    for k in todo:
        J = json.loads((L.LO / 'ants' / f'{k}.json').read_text())
        for part in ('gate', 'focal'):
            for c, r in J[part].items():
                if f'{k}|{part}|{c}' in old:
                    for i, M in enumerate(MN):
                        if r[M] is not None:
                            mx = max(mx, abs(old[f'{k}|{part}|{c}'][i, 0] - r[M]))
    if mx > 1e-6:
        raise SystemExit(f'BOOT IDENTITY CHECK FAILED {mx}')
    np.savez_compressed(bf, **old)
    z0 = np.load(LB0 / 'boot_ants.npz')
    Z = {**{k: z0[k] for k in z0.files}, **old}
    # replicate check: the ladder's stored v1 row must be reproduced by these Wser (same replicates)
    report(Z, mx)


def report(Z, idmax):
    lad = json.loads((T.OUT / 'ladder' / 'ladder.json').read_text())['ants']
    gcells = lad['gate_cells']
    sers = sorted({c.split('|')[0] for c in gcells})
    groups = {'all': gcells, **{s: [c for c in gcells if c.startswith(s + '|')] for s in sers},
              'Y2F+B2F': [c for c in gcells if not c.endswith('groom_any')]}
    fz = {'single': [f'{s}|single' for s in sers], 'double': ['v3|double', 'v4|double']}
    dm = {}
    for rung, keys in RUNGS.items():
        for gname, cs in list(groups.items()) + [(f'focal_{k}', v) for k, v in fz.items()]:
            part = 'focal' if gname.startswith('focal_') else 'gate'
            ks = [k for k in keys if all(f'{k}|{part}|{c}' in Z for c in cs)]
            if ks:
                dm[(rung, gname)] = np.stack([np.mean([Z[f'{k}|{part}|{c}'] for c in cs], 0) for k in ks])
    rng = np.random.default_rng(L.SEED + 7)
    res = {'identity_max_abs_diff_new': idmax, 'levels': {}, 'contrasts': {}}
    for (rung, g), A in dm.items():
        res['levels'].setdefault(rung, {})[g] = {M: {'mean': float(A[:, i, 0].mean()), 'median': float(np.median(A[:, i, 0])),
                                                      'seeds': A[:, i, 0].tolist()} for i, M in enumerate(MN)}
    for x, y in CONTRASTS:
        for g in list(groups) + ['focal_single', 'focal_double']:
            if (x, g) not in dm or (y, g) not in dm:
                continue
            ax, ay = dm[(x, g)], dm[(y, g)]
            r = {}
            for i, M in enumerate(MN):
                bx, by = ax[:, i, 1:], ay[:, i, 1:]
                ix = rng.integers(0, len(bx), (L.NB, len(bx)))
                iy = rng.integers(0, len(by), (L.NB, len(by)))
                rr = np.arange(L.NB)[:, None]
                d = np.nanmean(bx[ix, rr], 1) - np.nanmean(by[iy, rr], 1)
                r[M] = {'diff': float(ax[:, i, 0].mean() - ay[:, i, 0].mean()),
                        'ci': np.nanpercentile(d, [2.5, 97.5]).tolist(),
                        'ci_units_only': np.nanpercentile(bx.mean(0) - by.mean(0), [2.5, 97.5]).tolist()}
            res['contrasts'].setdefault(f'{x} - {y}', {})[g] = r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(res, indent=1))
    f3 = lambda v: f'{v:+.3f}'  # noqa: E731
    md = ['# ants v1-gap decomposition (raw frame max unless the rung is SN2 = P(on))', '',
          'Levels: seed mean (median) over the gate cells; per-seed values in report.json.', '',
          '| rung | n | mean AUROC | top-1% prec | nAP | Y2F+B2F AUROC | Y2F+B2F top-1% | focal single top-1% | focal double top-1% |',
          '|---|---|---|---|---|---|---|---|---|']
    for rung in RUNGS:
        if (rung, 'all') not in dm:
            continue
        lv = res['levels'][rung]
        c = lambda g, M: (f"{lv[g][M]['mean']:.3f} ({lv[g][M]['median']:.3f})" if g in lv else '-')  # noqa: E731
        md.append(f"| {rung} | {dm[(rung, 'all')].shape[0]} | {c('all', 'AUROC')} | {c('all', 'top1_prec')} | {c('all', 'nAP')} | "
                  f"{c('Y2F+B2F', 'AUROC')} | {c('Y2F+B2F', 'top1_prec')} | {c('focal_single', 'top1_prec')} | "
                  f"{c('focal_double', 'top1_prec')} |")
    md += ['', '| contrast | group | d mean AUROC [95% CI videos+seeds] | d top-1% prec [CI] | d nAP [CI] |', '|---|---|---|---|---|']
    for name, gs in res['contrasts'].items():
        for g, r in gs.items():
            md.append(f'| {name} | {g} | ' + ' | '.join(f"{f3(r[M]['diff'])} [{f3(r[M]['ci'][0])}, {f3(r[M]['ci'][1])}]"
                                                         for M in MN) + ' |')
    (OUT / 'report.md').write_text('\n'.join(md) + '\n')
    log('\n'.join(md))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('tok', 'train', 'cols', 'boot'))
    p.add_argument('--arm', choices=tuple(ARMS))
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')], default=list(SEEDS))
    p.add_argument('--keys', default='')
    a = p.parse_args()
    {'tok': cmd_tok, 'train': cmd_train, 'cols': cmd_cols, 'boot': cmd_boot}[a.step](a)
