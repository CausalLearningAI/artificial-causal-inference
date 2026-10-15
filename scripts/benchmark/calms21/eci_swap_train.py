"""
CalMS21 ECI - ROLE SWAP (rules: eci_swap.sh): SAE training on the task1 TEST videos, theta*, read-out on the task1
TRAIN videos, cross-fitted single-neuron scoring of the task-3 pair labels (each on its own sequences) and of task 1.
GPU job, one arm x 6 seeds. Run with CALMS_OUT=results/vision/benchmark_calms21/swap and
CALMS_TOK=/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/pairpush/calms21.

SET NAMES ARE SWAPPED (see eci_swap_plan.py): store set 'eval' = SAE pool (test videos), store set 'train' = eval frames
(train videos). The SAE recipe and the frame read-out are the grid's, unchanged (calms21 eci_train.load_shard, mabe22
eci_train.train_one / load_any, theta* = src.eci.sae_calibrate.calibrate_threshold, BatchTopK d 4096 k 16, 6000 steps x
4096 tokens, frame value = max over the frame's kept tokens). Scoring = contact_sae.cross_fit (best neuron picked on one
half by AUROC / AP / honest top-1% among neurons firing on >= 1% of that half's weighted frames, scored on the other,
both directions averaged), run PER LABEL on that label's frames only, video-level halves (calms21 eci_train.halves over
all 7 labels, rarest first) for split seeds 0-9.
--pool orbit: SAE trained / calibrated on the plan's orbit_pool subset only (4,000 + 1,000 frames); with --tok orbit the
pool tokens are read from --pool-tok (a store holding only that subset).
Outputs: OUT/sae/<arm>_s<seed>/{sae.pt, train.json}; OUT/eval/<arm>_s<seed>.json; with --sheet: OUT/sheets/<arm>_s<seed>
_sniff_face.json (the top-16 frames of the sniff_face neuron and their peak patch, for eci_swap_sheet.py).
"""
import argparse
import importlib.util
import json
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

_spec = importlib.util.spec_from_file_location('calms_eci_train', HERE / 'eci_train.py')
CT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CT)

T1 = ['attack', 'investigation', 'mount']
T3 = ['sniff_face', 'approach', 'mount_attempt', 'intromission']
LABS = T1 + T3


def frame_pool_arg(sae, norm, X, lens, dev, max_tok=16384):
    """as calms21 eci_train.frame_pool (max only) plus the within-frame argmax token of every neuron (int16)."""
    import torch
    Fn, m = len(lens), sae.n_latents
    st = np.r_[0, np.cumsum(lens)]
    cmax = torch.zeros(Fn, m, dtype=torch.float16, device=dev)
    amax = torch.zeros(Fn, m, dtype=torch.int16, device=dev)
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
                Lm = int(ln.max())
                off = torch.arange(len(z), device=dev) - torch.repeat_interleave(torch.cumsum(ln, 0) - ln, ln)
                pad = torch.zeros(f1 - f0, Lm, m, device=dev)
                pad[seg, off] = z
                mx = pad.max(1)
                cmax[f0:f1] = mx.values.half()
                amax[f0:f1] = mx.indices.to(torch.int16)
                del pad, z, x
            f0 = f1
    return cmax, amax


def rprec_f1(X, y, w, block=512):
    """Weighted R-precision (precision among the top frames holding the positives' weight mass P, ties pro rata) and
    F1-max (best F1 over all thresholds; thresholds between distinct values, incl. 'all frames') of every column of
    non-negative codes X (n, m) against bool y (n,), weights w (n,) double. -> (rprec (m,), f1max (m,)) numpy."""
    import torch
    n, m = X.shape
    P = float((w * y.double()).sum())
    rp, f1 = np.zeros(m), np.zeros(m)
    ar = torch.arange(n, device=X.device)
    for j0 in range(0, m, block):
        xs, o = torch.sort(X[:, j0:j0 + block].float(), dim=0, descending=True)
        ws, ys = w[o], y[o].double()
        CW, TP = ws.cumsum(0), (ws * ys).cumsum(0)
        end = torch.ones_like(xs, dtype=torch.bool)
        end[:-1] = xs[:-1] != xs[1:]
        f = torch.where(end, 2 * TP / (CW + P), torch.zeros_like(TP))
        f1[j0:j0 + block] = f.max(0).values.cpu().numpy()
        hit = end & (CW >= P - 1e-9)
        idx = torch.argmax(hit.int(), 0)                                    # first group end reaching P
        pe = torch.where(end, ar[:, None].expand_as(end), torch.full_like(end, -1, dtype=torch.long))
        pe = torch.cummax(pe, 0).values                                     # last group end at or before each row
        cols = torch.arange(xs.shape[1], device=X.device)
        prev = torch.where(idx > 0, pe[(idx - 1).clamp_min(0), cols], torch.full_like(idx, -1))
        cwp = torch.where(prev >= 0, CW[prev.clamp_min(0), cols], torch.zeros_like(CW[0]))
        tpp = torch.where(prev >= 0, TP[prev.clamp_min(0), cols], torch.zeros_like(TP[0]))
        cwi, tpi = CW[idx, cols], TP[idx, cols]
        tp = tpp + (tpi - tpp) * (P - cwp) / (cwi - cwp).clamp_min(1e-12)
        rp[j0:j0 + block] = (tp / P).cpu().numpy()
    return rp, f1


def extra_cf(cb, Y, w, h, r):
    """addendum 2 metrics for one label / split: R-precision and F1-max of (a) the AUROC-picked neuron of each direction
    and (b) the neuron picked by the same metric on the selection half; scored on the other half, directions averaged."""
    import torch
    M = {}
    for x in (0, 1):
        it = torch.from_numpy(np.flatnonzero(h == x)).to(cb.device)
        M[x] = rprec_f1(cb[it], Y[it, 0], w[it])
    out = {}
    for k, nm in ((0, 'rprec'), (1, 'f1max')):
        a, b_ = [], []
        for d, (s, t) in enumerate(((0, 1), (1, 0))):
            a.append(M[t][k][r['dirs'][d]['auc']['neuron']])
            b_.append(M[t][k][int(np.argmax(M[s][k]))])
        out[f'cf_{nm}_auc_pick'] = float(np.mean(a))
        out[f'cf_{nm}'] = float(np.mean(b_))
    return out


def compact(r):
    return {k: v for k, v in r.items() if k != 'dirs'} | {
        'picks': [{s: {'neuron': e[s]['neuron'], 'test_auc': e[s]['test_auc'], 'test_ap': e[s]['test_ap'],
                       'test_top1': e[s]['test_top1'], 'test_rate': e[s]['test_rate']} for s in ('auc', 'ap', 'top1')}
                  for e in r['dirs']]}


def main():
    import torch
    import contact_sae as cs
    import eci_train as ET  # mabe22's (its directory precedes this one on sys.path)
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    assert hasattr(ET, 'train_one') and ET.__file__.endswith('mabe22/eci_train.py'), ET.__file__
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--lag', type=int, default=60)
    ap.add_argument('--tok', choices=['mask', 'orbit', 'nomask'], default='mask')
    ap.add_argument('--pool', choices=['full', 'orbit'], default='full')
    ap.add_argument('--pool-tok', default='')
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--nsplits', type=int, default=10)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--sheet', action='store_true')
    args = ap.parse_args()
    assert str(OUT).endswith('benchmark_calms21/swap') and 'pairpush' in str(TOK), (OUT, TOK)
    cfg, lag, W = args.config, args.lag, 4096
    ET.D = CT.DIM[cfg]
    ET.WIDTH = W

    def build(cfg_, din, seed):
        from src.eci.sae import MatryoshkaBatchTopKSAE
        return MatryoshkaBatchTopKSAE(d_in=din, n_latents=W, prefixes=(W,), k=cfg_['k'], seed=seed)
    ET.build = build
    sae_cfg = dict(sp='btk', k=16, matry=False, prefixes=[W], delta=lag > 0, config=cfg, width=W,
                   lag=lag if lag else None, tok=args.tok, pool=args.pool, swap=True)
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    for d in ('sae', 'eval', 'sheets'):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    # ---------------------------------------------------------------- SAE pool (store set 'eval' = TEST videos)
    P = pd.read_parquet(PLAN / 'eval_frames.parquet')
    CT.TOK = Path(args.pool_tok) if args.pool_tok else TOK
    out = {'train': ([], []), 'calib': ([], [])}
    for base in CT.shards('eval', cfg):
        X, lens, Fm = CT.load_shard(base, cfg, lag, stage, args.tok, 'train' if args.tok == 'nomask' else 'eval')
        M = Fm.merge(P, on=['video', 'i'], how='left')
        assert not M.role.isna().any()
        role = M.role.values.copy()
        if args.pool == 'orbit':
            role[~M.orbit_pool.values.astype(bool)] = 'drop'
        tr = np.repeat(role, lens)
        for r in ('train', 'calib'):
            out[r][0].append(X[tr == r])
            out[r][1].append(lens[role == r])
        del X
    CT.TOK = TOK
    Xtr, ltr, Xca, lca = (np.concatenate(v) for r in ('train', 'calib') for v in out[r])
    del out
    log(f'{args.arm}: SAE pool train {Xtr.shape} on {len(ltr)} frames ({ltr.mean():.1f} tokens / frame), calib '
        f'{Xca.shape} on {len(lca)} frames')
    todo = [s for s in args.seeds if not (OUT / 'sae' / f'{args.arm}_s{s}' / 'sae.pt').exists()]
    if todo:
        Xg = torch.from_numpy(Xtr).to(dev)
        for seed in todo:
            ET.train_one(args.arm, sae_cfg, seed, Xg, ltr, args.steps, OUT / 'sae' / f'{args.arm}_s{seed}', dev)
            torch.cuda.empty_cache()
        del Xg
    del Xtr
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    models, calib = {}, {}
    Xca_g = torch.from_numpy(Xca).to(dev)
    for seed in args.seeds:
        sae, norm, ck = ET.load_any(OUT / 'sae' / f'{args.arm}_s{seed}' / 'sae.pt', dev)
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
    # ---------------------------------------------------------------- read-out (store set 'train' = TRAIN videos)
    E = pd.read_parquet(PLAN / 'train_frames.parquet')
    parts = {s: ([], []) for s in args.seeds}
    Fall, lens_all, pos_all = [], [], []
    for base in CT.shards('train', cfg):
        X, lens, Fm = CT.load_shard(base, cfg, lag, stage, args.tok, 'eval' if args.tok == 'nomask' else 'train')
        Fall.append(Fm)
        lens_all.append(lens)
        pos = np.load(TOK / f'{base}_{cfg}_pos.npy')
        assert args.tok == 'nomask' or len(pos) == lens.sum()
        st = np.r_[0, np.cumsum(lens)][:-1]
        pos_all.append((pos, st))
        for seed, (sae, norm) in models.items():
            c, am = frame_pool_arg(sae, norm, X, lens, dev)
            parts[seed][0].append(c)
            parts[seed][1].append(am)
        del X
    Fm = pd.concat(Fall).reset_index(drop=True)
    key = pd.MultiIndex.from_frame(Fm[['video', 'i']])
    order = key.get_indexer(pd.MultiIndex.from_frame(E[['video', 'i']]))
    assert (order >= 0).all() and len(np.unique(order)) == len(E) == len(Fm)
    lens = np.concatenate(lens_all)[order]
    # frame -> (pos array, start) for the peak-patch lookup of the sheets
    shard_of = np.concatenate([np.full(len(l_), k) for k, l_ in enumerate(lens_all)])[order]
    start_of = np.concatenate([p[1] for p in pos_all])[order]
    Ht = {s: CT.halves(E, s, LABS) for s in range(args.nsplits)}
    sub = {b: (np.ones(len(E), bool) if b in T1 else E['in_' + b].values.astype(bool)) for b in LABS}
    hinfo = {}
    for s, h in Ht.items():
        hinfo[f'split{s}'] = {b: [int(E[b][sub[b] & (h == x)].groupby(E.video[sub[b] & (h == x)]).any().sum())
                                  for x in (0, 1)] for b in LABS}
        miss = [b for b in LABS if min(E[b][h == 0].sum(), E[b][h == 1].sum()) == 0]
        assert not miss, (s, miss)
    oi = torch.from_numpy(order).to(dev)
    for seed in args.seeds:
        codes = torch.cat(parts[seed][0])[oi].contiguous()
        res = {'arm': args.arm, 'config': cfg, 'cfg': sae_cfg, 'seed': seed, 'calib': calib[seed],
               'eval_frames': len(E), 'eval_frames_0_tokens': int((lens == 0).sum()), 'labels': LABS,
               'pos_videos_per_half': hinfo, 'scores': {}}
        for b in LABS:
            idx = np.flatnonzero(sub[b])
            it = torch.from_numpy(idx).to(dev)
            cs.LABELS = [b]
            Y = torch.from_numpy(E[b].values[idx].astype(bool)[:, None]).to(dev)
            w = torch.from_numpy(E.w.values[idx].astype(np.float64)).to(dev)
            cb = codes[it].contiguous()
            wr = (((cb > 0).double() * w[:, None]).sum(0) / w.sum()).cpu().numpy()
            res['scores'][b] = {'n_frames': int(len(idx)), 'n_pos': int(Y.sum()),
                                'weighted_base_rate': float((w * Y[:, 0].double()).sum() / w.sum()),
                                'n_eligible_1pct': int((wr >= 0.01).sum()), 'dead_frac': float((wr == 0).mean()),
                                'splits': {}}
            for s, h in Ht.items():
                r = cs.cross_fit(cb, Y, w, h[idx], dev)[b]
                r['cf_rate_auc_pick'] = float(np.mean([d['auc']['test_rate'] for d in r['dirs']]))
                r.update(extra_cf(cb, Y, w, h[idx], r))
                res['scores'][b]['splits'][f'split{s}'] = compact(r)
            sp = res['scores'][b]['splits']
            for k in ('cf_auroc', 'cf_ap', 'cf_top1', 'cf_top1_rate', 'cf_rate_auc_pick', 'cf_ap_at_auroc', 'cf_rprec',
                      'cf_rprec_auc_pick', 'cf_f1max', 'cf_f1max_auc_pick'):
                res['scores'][b][k] = float(np.mean([sp[x][k] for x in sp]))
            del cb
        log(f'  {args.arm} s{seed}: ' + ', '.join(
            f"{b} {res['scores'][b]['cf_auroc']:.3f}/{res['scores'][b]['cf_ap']:.4f}/{res['scores'][b]['cf_top1']:.3f}"
            for b in LABS) + ' (cf AUROC / AP / honest top-1%, mean of 10 split seeds)')
        (OUT / 'eval' / f'{args.arm}_s{seed}.json').write_text(json.dumps(res, indent=1, default=float))
        if args.sheet and seed == args.seeds[0]:
            # descriptive: the sniff_face neuron picked most often by AUROC over the 10 splits x 2 directions
            b = 'sniff_face'
            picks = [p['auc']['neuron'] for x in res['scores'][b]['splits'].values() for p in x['picks']]
            n = int(pd.Series(picks).mode().iloc[0])
            idx = np.flatnonzero(sub[b])
            v = codes[torch.from_numpy(idx).to(dev), n].float().cpu().numpy()
            top = idx[np.argsort(-v, kind='stable')[:16]]
            am = torch.cat(parts[seed][1])[oi][:, n].cpu().numpy()
            rows = []
            for j in top:
                pos, _ = pos_all[shard_of[j]]
                p = int(pos[start_of[j] + int(am[j])])
                rows.append({'video': E.video.iloc[j], 'i': int(E.i.iloc[j]), 'value': float(codes[j, n]),
                             'label': bool(E[b].iloc[j]), 'stratum': E.stratum.iloc[j], 'peak_pos': p,
                             'grid': 64 if cfg in CT.G64 else 32})
            js = {'arm': args.arm, 'seed': seed, 'label': b, 'neuron': n, 'pick_count': int(picks.count(n)),
                  'n_picks': len(picks), 'frames': rows,
                  'note': 'top 16 of the frames in sniff_face sequences in the stratified eval sample (positives '
                          'over-sampled; not a population top-16)'}
            (OUT / 'sheets' / f'{args.arm}_s{seed}_sniff_face.json').write_text(json.dumps(js, indent=1))
            log(f'  sheet: neuron {n} ({picks.count(n)}/{len(picks)} picks), top-16 positives '
                f'{sum(r["label"] for r in rows)}')
        del codes
        torch.cuda.empty_cache()
    log(f'{args.arm} done in {time.time() - t00:.0f}s')


if __name__ == '__main__':
    main()
