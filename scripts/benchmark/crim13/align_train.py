"""
CRIM13 alignment test - SAE training, theta*, eval read-out and fold scoring for one ARM x 6 seeds x 2 folds (rules:
eci_align.sh; GPU Slurm job). Labels are read ONLY in the scoring step.

Arms: C1 = d3l_640_L20, x = [z_t, z_t - z_{t-50}];  C2 = d3l_640_L20, x = z_t;  C3 = d2l_560_L20, x = [z_t, z_t - z_{t-50}];
      C4 (EXPLORATORY, addendum 1) = C1 trained on random video cells (same count as the mask) - its eval read-out is
      align_c4.py (all video cells), not this file.
Folds: A = SAE trained on the TRAIN half's pool (role 'train', theta* on role 'calib'), neuron picked on the train half's
       eval frames, scored on the test half's eval frames; B = roles swapped. Machinery (unchanged functions): mabe22
       eci_train.train_one (TokenNorm per 1024-d block, geometric-median b_dec, Adam 5e-4, 500 warm-up, cosine,
       6000 steps x 4096 tokens), frame_max (frame value = max over the frame's kept tokens, 0 if none), theta*
       (src/eci/sae_calibrate.calibrate_threshold on all calib tokens), contact_sae.align_w (weighted AUROC / AP /
       top-1% / firing rate, inverse-probability weights). Picks per behaviour: AUROC, AP, honest top-1% (neurons
       firing on >= 1% of the selection half's weighted frames).
Also (descriptive): scoring restricted to BC- / C57-intruder videos of the scoring half (pick unchanged), and leave one
resident out (pick on the selection half's frames of the other 3 residents, score on the held-out resident's frames of
the scoring half).
Outputs: OUT/sae/<arm>_<fold>_s<seed>/{sae.pt, train.json}; OUT/eval/<arm>_<fold>_s<seed>.json; OUT/eval/<arm>_picks.npz
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
sys.path.insert(0, str(HERE.parent / 'mabe22'))
from align_common import LAG, OUT, PLAN, RESIDENTS, SCORED, TOK, log, stage_dir  # noqa: E402

ARMS = {'C1': ('d3l_640_L20', LAG, 'mask'), 'C2': ('d3l_640_L20', 0, 'mask'), 'C3': ('d2l_560_L20', LAG, 'mask'),
        'C4': ('d3l_640_L20', LAG, 'rnd')}
FOLDS = {'A': ('train', 'test'), 'B': ('test', 'train')}   # fold: (SAE + selection half, scoring half)
D = 1024


def shard_bases(half):
    js = sorted(TOK.glob(f'{half}_*of*.json'))
    keys = sorted({j.stem for j in js if 'smoke' not in j.stem and j.stem.count('_') == 1})
    n = sorted({int(k.split('of')[1]) for k in keys})
    assert len(n) == 1 and len(keys) == n[0], (half, keys)
    return [f'{half}_{i}of{n[0]}' for i in range(n[0])]


def load(base, cfg, lag, stage, tok='mask', rows=None):
    """-> X (N, D or 2D) fp16, lens per frame of the kept frames, frames table. tok 'rnd': the random-cell pool tokens
    (pool frames only). rows: boolean over the shard's frames to keep (default all)."""
    Fm = pd.read_parquet(TOK / f'{base}_frames.parquet')
    if tok == 'rnd':
        Fm = Fm[Fm.role != 'eval'].reset_index(drop=True)
    pre = 'rnd_' if tok == 'rnd' else ''
    arr = {}
    for k in (('t', 'p50') if lag else ('t',)):
        src = TOK / f'{base}_{cfg}_{pre}{k}.npy'
        loc = stage / src.name
        shutil.copyfile(src, loc)
        arr[k] = np.load(loc)
        loc.unlink()
    lens = Fm.n40.values.astype(np.int64)
    assert lens.sum() == len(arr['t'])
    st = np.r_[0, np.cumsum(lens)]
    if rows is None:
        rows = np.ones(len(Fm), bool)
    rows = np.asarray(rows)
    ti = np.concatenate([np.arange(st[i], st[i + 1]) for i in np.flatnonzero(rows)] or [np.zeros(0, np.int64)])
    t = arr['t'][ti]
    if not lag:
        return t, lens[rows], Fm[rows].reset_index(drop=True)
    X = np.empty((len(t), 2 * D), np.float16)
    X[:, :D] = t
    X[:, D:] = (t.astype(np.float32) - arr['p50'][ti].astype(np.float32)).astype(np.float16)
    return X, lens[rows], Fm[rows].reset_index(drop=True)


NB = 0x7C01


def align_w2(X, Y, w, rows, block=512):
    """contact_sae.align_w (weighted AUROC / AP / honest-top-1% precision / firing rate; fp16 bit-pattern histograms =
    exact tie groups) plus, per neuron and label (addendum 2): R-precision = weighted precision among the top-N
    weighted frames, N = the label's weighted positive mass (ties at the cut pro rata, NOT capped at the firing
    frames) and F1-max = the best F1 over all thresholds of the neuron (tie-group boundaries, 'fire on everything'
    included)."""
    import torch
    import contact_sae as cs
    R = cs.align_w(X, Y, w, rows, block)
    dev = X.device
    Y, w = Y[rows], w[rows].double()
    n, L = Y.shape
    m = X.shape[1]
    P = (Y.double() * w[:, None]).sum(0)
    rp = torch.full((m, L), float('nan'), dtype=torch.float64, device=dev)
    f1 = rp.clone()
    for j0 in range(0, m, block):
        b = min(block, m - j0)
        xb = X[rows, j0:j0 + b]
        bits = (xb.contiguous().view(torch.int16).long() & 0x7FFF) + NB * torch.arange(b, device=dev)[None]
        wb = w[:, None].expand(n, b)
        tot_d = torch.bincount(bits.flatten(), weights=wb.flatten(), minlength=b * NB).view(b, NB).flip(1)
        ccnt = tot_d.cumsum(1)
        for li in range(L):
            if P[li] <= 0:
                continue
            yl = Y[:, li]
            pos = torch.bincount(bits[yl].flatten(), weights=wb[yl].flatten(), minlength=b * NB).view(b, NB).flip(1)
            ctp = pos.cumsum(1)
            f1[j0:j0 + b, li] = (2 * ctp / (ccnt + P[li])).max(1).values
            k = torch.full((b,), float(P[li]), dtype=torch.float64, device=dev)
            ib = torch.searchsorted(ccnt.contiguous(), k[:, None] - 1e-9).clamp_max(NB - 1)
            cb, tb, pb = ccnt.gather(1, ib)[:, 0], tot_d.gather(1, ib)[:, 0], pos.gather(1, ib)[:, 0]
            tp = ctp.gather(1, ib)[:, 0] - pb + pb * (k - (cb - tb)) / tb.clamp_min(1e-12)
            rp[j0:j0 + b, li] = tp / k
    R['rprec'], R['f1'] = rp.cpu().numpy(), f1.cpu().numpy()
    return R


def pick_and_test(S, T, k):
    e = {}
    for sel, q, honest in (('auc', 'auc', False), ('ap', 'ap', False), ('top1', 'prec_top', True),
                           ('rprec', 'rprec', False), ('f1', 'f1', False)):
        sc = S[q][:, k].copy()
        if honest:
            sc[S['rate'] < 0.01] = -np.inf
        sc[~np.isfinite(sc)] = -np.inf
        j = int(np.argmax(sc))
        e[sel] = {'neuron': j, 'select_score': float(sc[j]), 'test_auc': float(T['auc'][j, k]),
                  'test_ap': float(T['ap'][j, k]), 'test_top1': float(T['prec_top'][j, k]),
                  'test_rate': float(T['rate'][j]), 'test_rprec': float(T['rprec'][j, k]),
                  'test_f1': float(T['f1'][j, k])}
    return e


def score(codes, E, Y, w, sel_half, test_half, dev):
    import torch
    import contact_sae as cs
    rows = lambda m: torch.from_numpy(np.flatnonzero(m)).to(dev)  # noqa: E731
    hs, ht = (E.half.values == sel_half), (E.half.values == test_half)
    S = align_w2(codes, Y, w, rows(hs))
    T = align_w2(codes, Y, w, rows(ht))
    out = {'main': {}, 'strain': {}, 'loro': {}}
    for k, b in enumerate(SCORED):
        e = pick_and_test(S, T, k)
        e['base_sel'], e['base_test'] = float(S['base'][k]), float(T['base'][k])
        out['main'][b] = e
    for st in ('BC', 'C57'):
        Tm = align_w2(codes, Y, w, rows(ht & (E.strain.values == st)))
        out['strain'][st] = {b: {'test_auc': float(Tm['auc'][out['main'][b]['auc']['neuron'], k]),
                                 'test_ap_at_auc': float(Tm['ap'][out['main'][b]['auc']['neuron'], k]),
                                 'test_rprec_at_auc': float(Tm['rprec'][out['main'][b]['auc']['neuron'], k]),
                                 'base_test': float(Tm['base'][k]), 'n': int(Tm['n'])} for k, b in enumerate(SCORED)}
    for r in RESIDENTS:
        Sr = align_w2(codes, Y, w, rows(hs & (E.resident.values != r)))
        Tr = align_w2(codes, Y, w, rows(ht & (E.resident.values == r)))
        out['loro'][r] = {'n_test': int(Tr['n'])}
        for k, b in enumerate(SCORED):
            e = pick_and_test(Sr, Tr, k)
            out['loro'][r][b] = {'test_auc': e['auc']['test_auc'], 'base_test': float(Tr['base'][k]),
                                 'n_pos_test': int(E[b].values[ht & (E.resident.values == r)].sum())}
    return out


def main():
    import torch
    import eci_train as ET
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(ARMS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[500, 501, 502, 503, 504, 505])
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--folds', nargs='+', default=['A', 'B'])
    ap.add_argument('--train-only', action='store_true')
    args = ap.parse_args()
    cfg, lag, tok = ARMS[args.arm]
    W = 4096
    ET.D, ET.WIDTH = D, W

    def build(c_, din, seed):
        from src.eci.sae import MatryoshkaBatchTopKSAE
        return MatryoshkaBatchTopKSAE(d_in=din, n_latents=W, prefixes=(W,), k=c_['k'], seed=seed)
    ET.build = build
    sae_cfg = dict(sp='btk', k=16, matry=False, prefixes=[W], delta=lag > 0, config=cfg, width=W, lag=lag or None,
                   tok=tok, arm=args.arm)
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    for d in ('sae', 'eval'):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    models, calib = {}, {}
    for fold in args.folds:
        half = FOLDS[fold][0]
        out = {'train': ([], []), 'calib': ([], [])}
        for base in shard_bases(half):
            Fm0 = pd.read_parquet(TOK / f'{base}_frames.parquet')
            if tok == 'rnd':
                Fm0 = Fm0[Fm0.role != 'eval'].reset_index(drop=True)
            for r in ('train', 'calib'):
                X, lens, _ = load(base, cfg, lag, stage, tok, (Fm0.role == r).values)
                out[r][0].append(X)
                out[r][1].append(lens)
        Xtr, ltr, Xca, lca = (np.concatenate(v) for r in ('train', 'calib') for v in out[r])
        del out
        log(f'{args.arm} fold {fold}: train {Xtr.shape} on {len(ltr)} frames ({ltr.mean():.1f} tokens / frame), '
            f'calib {Xca.shape} on {len(lca)} frames')
        name = lambda s: f'{args.arm}_{fold}_s{s}'  # noqa: E731
        todo = [s for s in args.seeds if not (OUT / 'sae' / name(s) / 'sae.pt').exists()]
        if todo:
            torch.backends.cuda.matmul.allow_tf32 = True
            Xg = torch.from_numpy(Xtr).to(dev)
            for seed in todo:
                ET.train_one(name(seed), sae_cfg, seed, Xg, ltr, args.steps, OUT / 'sae' / name(seed), dev)
                torch.cuda.empty_cache()
            del Xg
        del Xtr
        torch.cuda.empty_cache()
        if args.train_only:
            continue
        torch.backends.cuda.matmul.allow_tf32 = False
        Xca_g = torch.from_numpy(Xca).to(dev)
        for seed in args.seeds:
            sae, norm, ck = ET.load_any(OUT / 'sae' / name(seed) / 'sae.pt', dev)
            ema = float(sae.threshold)
            th, info = calibrate_threshold(sae, Xca_g, norm=norm)
            set_threshold(sae, th)
            info['theta_ema'] = ema
            info['calib_stats'] = sparsity_stats(sae, Xca_g, th, norm=norm)
            calib[(fold, seed)] = info
            models[(fold, seed)] = (sae, norm)
            log(f'  {name(seed)} theta* {th:.4f} (EMA {ema:.4f}) calib L0 {info["l0_calib_theta_star"]:.3f}')
        del Xca_g, Xca
        torch.cuda.empty_cache()
    if tok == 'rnd':  # C4: theta* (on the random-cell calib tokens) saved for align_c4.py
        (OUT / 'sae' / f'{args.arm}_calib.json').write_text(json.dumps(
            {f'{f}_s{sd}': {'theta': float(models[(f, sd)][0].threshold), **v} for (f, sd), v in calib.items()},
            indent=1, default=float))
    if args.train_only or tok == 'rnd':
        log(f'{args.arm} trained in {time.time() - t00:.0f}s (no read-out here)')
        return
    # eval read-out: every model on the eval frames of BOTH halves (masked tokens)
    P = pd.read_parquet(PLAN / 'frames.parquet')
    E = P[P.role == 'eval'].reset_index(drop=True)
    parts = {k: ([], [], []) for k in models}
    Fall, lens_all, pos_all = [], [], []
    for half in ('train', 'test'):
        for base in shard_bases(half):
            Fm0 = pd.read_parquet(TOK / f'{base}_frames.parquet')
            ev = (Fm0.role == 'eval').values
            X, lens, Fm = load(base, cfg, lag, stage, 'mask', ev)
            p40 = np.load(TOK / f'{base}_pos40.npy')
            st0 = np.r_[0, np.cumsum(Fm0.n40.values)]
            pos_all += [p40[st0[i]:st0[i + 1]] for i in np.flatnonzero(ev)]
            Fm['half'] = half
            Fall.append(Fm)
            lens_all.append(lens)
            for k, (sae, norm) in models.items():
                c, a, stt = ET.frame_max(sae, norm, X, lens, dev, False)
                parts[k][0].append(c)
                parts[k][1].append(a)
                parts[k][2].append(stt)
            del X
    Fm = pd.concat(Fall).reset_index(drop=True)
    order = pd.MultiIndex.from_frame(Fm[['video', 'v']]).get_indexer(pd.MultiIndex.from_frame(E[['video', 'v']]))
    assert (order >= 0).all() and len(np.unique(order)) == len(E) == len(Fm)
    assert (Fm.half.values[order] == E.half.values).all()
    lens = np.concatenate(lens_all)[order]
    pos_all = [pos_all[o] for o in order]
    oi = torch.from_numpy(order).to(dev)
    Y = torch.from_numpy(E[SCORED].values.astype(bool)).to(dev)
    w = torch.from_numpy(E.w.values.astype(np.float64)).to(dev)
    picks = {}
    for (fold, seed), pr in parts.items():
        codes = torch.cat(pr[0])[oi].contiguous()
        arg = torch.cat(pr[1])[oi].contiguous()
        sts = pr[2]
        ntot = sum(x['n_tokens'] for x in sts)
        stats = {'l0': sum(x['l0'] * x['n_tokens'] for x in sts) / ntot, 'n_tokens': ntot,
                 'fve_shard_mean': float(np.mean([x['fve'] for x in sts]))}
        sel, tst = FOLDS[fold]
        res = {'arm': args.arm, 'config': cfg, 'lag': lag, 'fold': fold, 'sel_half': sel, 'test_half': tst,
               'seed': seed, 'cfg': sae_cfg, 'calib': calib[(fold, seed)], 'eval_stats': stats,
               'eval_frames': len(E), 'eval_frames_0_tokens': int((lens == 0).sum()), 'labels': SCORED,
               **score(codes, E, Y, w, sel, tst, dev)}
        fr = (((codes > 0).double() * w[:, None]).sum(0) / w.sum()).cpu().numpy()
        res['firing'] = {'dead_frac_eval': float((fr == 0).mean()), 'n_eligible_1pct': int((fr >= 0.01).sum())}
        m = res['main']
        log(f'  {args.arm} {fold} s{seed}: ' + ', '.join(
            f"{b} {m[b]['auc']['test_auc']:.3f}/{m[b]['ap']['test_ap']:.3f}/{m[b]['top1']['test_top1']:.3f}"
            for b in SCORED) + ' (test AUROC / AP / honest top-1%)')
        (OUT / 'eval' / f'{args.arm}_{fold}_s{seed}.json').write_text(json.dumps(res, indent=1, default=float))
        for b in SCORED:  # per-frame value and peak cell of the AUROC- and top-1%-picked neurons (contact sheets)
            for kind in ('auc', 'top1'):
                j = m[b][kind]['neuron']
                av = arg[:, j].long().cpu().numpy()
                pk = np.array([pos_all[i][av[i]] if lens[i] > 0 else -1 for i in range(len(E))], np.int16)
                picks[f'{fold}_s{seed}_{b}_{kind}'] = np.stack([codes[:, j].float().cpu().numpy(), pk.astype(np.float32)])
        del codes, arg
        torch.cuda.empty_cache()
    np.savez_compressed(OUT / 'eval' / f'{args.arm}_picks.npz', **picks)
    log(f'{args.arm} done in {time.time() - t00:.0f}s')


if __name__ == '__main__':
    main()
