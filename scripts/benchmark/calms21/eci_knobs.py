"""
CalMS21 ECI - ROLE SWAP, SAE KNOB SWEEP for a clean SNIFFING neuron (launcher: eci_knobs.sh). Mirrors the label-free
SAE knob sweep (round 2) on mice v1 so a recipe win can be checked on a second, public dataset.
Run with CALMS_OUT=results/vision/benchmark_calms21/swap and CALMS_TOK=/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/
pairpush/calms21 (asserted). SET NAMES ARE SWAPPED (eci_swap_plan.py): store set 'eval' = SAE pool (19 task1 TEST
videos, labels never read), store set 'train' = scoring frames (70 task1 TRAIN videos).

PRE-REGISTERED RULE, written 2026-10-16 BEFORE any SAE of this sweep was trained or scored (only the existing S1-S3
results of eci_swap.sh had been seen). Anything added after a sweep score was seen is labelled EXPLORATORY.
======================================================================================================================
SHARED (all arms): config d3l_1024_L20 tokens (DINOv3-L/16 block 20, 1024 px canvas, 64^2 grid), the swap's 'dark'
  mask as stored, NO change channel (x = z_t only, --lag 0, as arm S2), frame value = max over the frame's kept tokens,
  theta* = src.eci.sae_calibrate.calibrate_threshold on the 2,000 calibration frames' tokens, SAE training =
  mabe22 eci_train.train_one (TokenNorm per 1024 block, b_dec = geometric median, Adam 5e-4, 500 warm-up, cosine,
  batches of 4096 tokens, AuxK), class src.eci.sae.MatryoshkaBatchTopKSAE (plain = one prefix = the width).
  SAE seeds 600, 601, 602 for every arm. Scoring = eci_swap_train's (contact_sae.cross_fit + addendum-2 R-precision /
  F1-max, video-level halves, split seeds 0-9 x 2 directions, inverse-probability weights, each label on its frames).
ARMS:                 width  prefixes              k   steps   tokens
  B0 (control = S2)   4096   (4096)                16   6000   raw
  B1                   512   128/256/512           16   6000   raw
  B2                  2048   128/512/2048          16  30000   raw
  B3                  8192   128/512/2048/8192     16  30000   raw
  B4                  2048   128/512/2048           8  30000   raw
  B5                  2048   128/512/2048          32  30000   raw
  B6                  2048   128/512/2048          16  30000   per-video CENTRED
  B7                  2048   (2048)                16  30000   raw
  CENTRED: from every token subtract the mean token of its video, computed over all stored tokens of that video's
  frames in its own set (pool: train + calibration frames of that test video; scoring: that train video's scoring
  frames), unweighted. No label is read for it, BUT the scoring frames were sampled with label-defined strata
  (positives over-sampled), so the scoring-set video mean is computed on a label-dependent frame sample (caveat,
  stated in the report).
PRIMARY (sniff_face, on the frames of its own sequences):
  (P1) weighted R-precision, neuron picked by R-precision on the selection half (cf_rprec);
  (P2) honest top-1% (top-1%-picked among neurons firing on >= 1% of the selection half's weighted frames; cf_top1).
  Reported next to them: R-precision of the AUROC-picked neuron (cf_rprec_auc_pick), cf AUROC (AUROC pick), cf AP
  (AP pick), AP lift = cf AP / weighted base rate, F1-max; investigation AUROC and R-precision (both picks); SAE FVE on
  the held-out calibration tokens (theta*), FVE on the scoring tokens, dead fraction (latents with zero weighted firing
  rate on all scoring frames; and dead at the end of training).
CONTRAST vs B0, per primary metric: delta of the 3 vs 3 per-seed values (each = mean over split seeds 0-9), Welch 95%
  CI. WIN = delta >= +0.010 AND CI excludes 0; LOSES = delta <= -0.010 AND CI excludes 0; else EQUIVALENT.
  ARM VERDICT: BETTER iff P1 WINS and P2 does not LOSE (or P2 WINS and P1 does not LOSE); WORSE iff a primary LOSES
  and neither WINS; MIXED iff one WINS and the other LOSES; else EQUIVALENT. 3 seeds give a weak test (df ~2-4,
  wide CIs): EQUIVALENT means "not resolved", not "same".
CONTACT SHEET (descriptive): best arm by mean P1, its first seed, the sniff_face neuron picked most often by R-precision
  over the 20 selections (and the AUROC-mode one), top-16 frames among the sniff_face-sequence scoring frames.
======================================================================================================================
Outputs: SAE checkpoints on BeeGFS (--sae-dir, default /mnt/beegfs/locatgrp/shared/rcadei/eci_knobs_calms21/sae/
<arm>_s<seed>/sae.pt), train.json copied to OUT/knobs/sae/<arm>_s<seed>/; OUT/knobs/eval/<arm>_s<seed>.json;
OUT/knobs/sheets/<arm>_s<seed>_sniff_face_{rprec,auc}.json (for eci_swap_sheet.py). Custom arms: --arm NAME --width
--prefixes --k --steps [--center]. --selftest: 1 seed, few steps, 2 split seeds, outputs under OUT/knobs_selftest.
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
_spec = importlib.util.spec_from_file_location('calms_eci_swap_train', HERE / 'eci_swap_train.py')
SW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SW)          # sets sys.path like the swap trainer (mabe22 eci_train before calms21's)
CT = SW.CT
from calms_common import OUT, PLAN, TOK, log, stage_dir  # noqa: E402

CFG = 'd3l_1024_L20'
ARMS = {
    'B0': dict(width=4096, prefixes=[4096], k=16, steps=6000, center=False),
    'B1': dict(width=512, prefixes=[128, 256, 512], k=16, steps=6000, center=False),
    'B2': dict(width=2048, prefixes=[128, 512, 2048], k=16, steps=30000, center=False),
    'B3': dict(width=8192, prefixes=[128, 512, 2048, 8192], k=16, steps=30000, center=False),
    'B4': dict(width=2048, prefixes=[128, 512, 2048], k=8, steps=30000, center=False),
    'B5': dict(width=2048, prefixes=[128, 512, 2048], k=32, steps=30000, center=False),
    'B6': dict(width=2048, prefixes=[128, 512, 2048], k=16, steps=30000, center=True),
    'B7': dict(width=2048, prefixes=[2048], k=16, steps=30000, center=False),
}
SEEDS = [600, 601, 602]


def video_center(Xs, Fs, lens_s):
    """in place: subtract from every token its video's mean token (over all the given shards' tokens of that video).
    Xs / Fs / lens_s: lists per shard of tokens (N, D) fp16, frames DataFrame, lens. -> {video: n_tokens}."""
    vids = sorted(set().union(*[set(F.video) for F in Fs]))
    vi = {v: j for j, v in enumerate(vids)}
    D = Xs[0].shape[1]
    S, n = np.zeros((len(vids), D)), np.zeros(len(vids))
    tv = []
    for X, F, ln in zip(Xs, Fs, lens_s):
        t = np.repeat(F.video.map(vi).values.astype(np.int64), ln)
        tv.append(t)
        for c in range(0, len(X), 400_000):
            np.add.at(S, t[c:c + 400_000], X[c:c + 400_000].astype(np.float64))
        n += np.bincount(t, minlength=len(vids))
    mu = (S / np.maximum(n, 1)[:, None]).astype(np.float32)
    for X, t in zip(Xs, tv):
        for c in range(0, len(X), 400_000):
            X[c:c + 400_000] = (X[c:c + 400_000].astype(np.float32) - mu[t[c:c + 400_000]]).astype(np.float16)
    return {v: int(n[j]) for v, j in vi.items()}


def frame_pool_stats(sae, norm, X, lens, dev, max_tok=16384):
    """SW.frame_pool_arg (max over the frame's tokens + argmax token) plus FVE / L0 of the full dictionary under the
    calibrated threshold on these tokens (sums for a later pooled FVE)."""
    import torch
    Fn, m = len(lens), sae.n_latents
    st = np.r_[0, np.cumsum(lens)]
    cmax = torch.zeros(Fn, m, dtype=torch.float16, device=dev)
    amax = torch.zeros(Fn, m, dtype=torch.int16, device=dev)
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
                mx = pad.max(1)
                cmax[f0:f1] = mx.values.half()
                amax[f0:f1] = mx.indices.to(torch.int16)
                del pad, z, x, xh
            f0 = f1
    return cmax, amax, (s1, s2, sse, l0, int(st[-1]))


def extra_cf_picks(cb, Y, w, h, r):
    """SW.extra_cf plus the R-precision-picked neuron of each direction (for the sheets)."""
    import torch
    M = {}
    for x in (0, 1):
        it = torch.from_numpy(np.flatnonzero(h == x)).to(cb.device)
        M[x] = SW.rprec_f1(cb[it], Y[it, 0], w[it])
    out, picks = {}, []
    for k, nm in ((0, 'rprec'), (1, 'f1max')):
        a, b_ = [], []
        for d, (s, t) in enumerate(((0, 1), (1, 0))):
            a.append(M[t][k][r['dirs'][d]['auc']['neuron']])
            j = int(np.argmax(M[s][k]))
            b_.append(M[t][k][j])
            if nm == 'rprec':
                picks.append(j)
        out[f'cf_{nm}_auc_pick'] = float(np.mean(a))
        out[f'cf_{nm}'] = float(np.mean(b_))
    out['rprec_picks'] = picks
    return out


def load_set(store_set, plan, stage, center):
    """all shards of one store set (z_t only). -> list of (X, lens, Fm) per shard, centring info."""
    sh = [CT.load_shard(base, CFG, 0, stage, 'mask', store_set) for base in CT.shards(store_set, CFG)]
    info = None
    if center:
        info = video_center([s[0] for s in sh], [s[2] for s in sh], [s[1] for s in sh])
    return sh, info


def main():
    import torch
    import contact_sae as cs
    import eci_train as ET  # mabe22's
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    assert hasattr(ET, 'train_one') and ET.__file__.endswith('mabe22/eci_train.py'), ET.__file__
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='*', default=[], help='arms of the pre-registered table')
    ap.add_argument('--arm', default='', help='custom arm name (with --width --prefixes --k --steps [--center])')
    ap.add_argument('--width', type=int)
    ap.add_argument('--prefixes', nargs='+', type=int)
    ap.add_argument('--k', type=int)
    ap.add_argument('--steps', type=int)
    ap.add_argument('--center', action='store_true')
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--nsplits', type=int, default=10)
    ap.add_argument('--sae-dir', default='/mnt/beegfs/locatgrp/shared/rcadei/eci_knobs_calms21/sae')
    ap.add_argument('--selftest', action='store_true', help='300 steps, first seed, 2 split seeds, OUT/knobs_selftest')
    args = ap.parse_args()
    assert str(OUT).endswith('benchmark_calms21/swap') and 'pairpush' in str(TOK), (OUT, TOK)
    arms = {a: dict(ARMS[a]) for a in args.arms}
    if args.arm:
        arms[args.arm] = dict(width=args.width, prefixes=args.prefixes, k=args.k, steps=args.steps, center=args.center)
    assert arms, 'no arm given'
    for a, c in arms.items():
        assert c['prefixes'][-1] == c['width'] and c['prefixes'] == sorted(c['prefixes']), (a, c)
    cen = {c['center'] for c in arms.values()}
    assert len(cen) == 1, 'one job = one token variant (centred or raw)'
    center = cen.pop()
    seeds, nsplits = args.seeds, args.nsplits
    od = OUT / 'knobs'
    sae_root = Path(args.sae_dir)
    if args.selftest:
        seeds, nsplits, od, sae_root = seeds[:1], 2, OUT / 'knobs_selftest', sae_root.parent / 'sae_selftest'
        for c in arms.values():
            c['steps'] = 300
    for d in ('sae', 'eval', 'sheets'):
        (od / d).mkdir(parents=True, exist_ok=True)
    ET.D = CT.DIM[CFG]
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    gpu = torch.cuda.get_device_name(0)
    # ---------------------------------------------------------------- SAE pool (store set 'eval' = TEST videos)
    P = pd.read_parquet(PLAN / 'eval_frames.parquet')
    sh, cinfo_pool = load_set('eval', P, stage, center)
    out = {'train': ([], []), 'calib': ([], [])}
    for X, lens, Fm in sh:
        M = Fm.merge(P, on=['video', 'i'], how='left')
        assert not M.role.isna().any()
        role = M.role.values
        tr = np.repeat(role, lens)
        for r in ('train', 'calib'):
            out[r][0].append(X[tr == r])
            out[r][1].append(lens[role == r])
    del sh
    Xtr, ltr, Xca, lca = (np.concatenate(v) for r in ('train', 'calib') for v in out[r])
    del out
    log(f'pool train {Xtr.shape} on {len(ltr)} frames, calib {Xca.shape} on {len(lca)} frames, centred {center}, '
        f'load {time.time() - t00:.0f}s, {gpu}')
    Xg = torch.from_numpy(Xtr).to(dev)
    del Xtr
    for a, c in arms.items():
        W = c['width']

        def build(cfg_, din, seed, W=W, pre=tuple(c['prefixes'])):
            from src.eci.sae import MatryoshkaBatchTopKSAE
            return MatryoshkaBatchTopKSAE(d_in=din, n_latents=W, prefixes=pre, k=cfg_['k'], seed=seed)
        ET.build, ET.WIDTH = build, W
        c['sae_cfg'] = dict(sp='btk', k=c['k'], matry=len(c['prefixes']) > 1, prefixes=c['prefixes'], delta=False,
                            config=CFG, width=W, lag=None, tok='mask', pool='full', swap=True, center=center,
                            steps=c['steps'], sweep='knobs')
        for seed in seeds:
            o = sae_root / f'{a}_s{seed}'
            if (o / 'sae.pt').exists():
                log(f'[SKIP] {o}')
                continue
            ET.train_one(a, c['sae_cfg'], seed, Xg, ltr, c['steps'], o, dev)
            ti = json.loads((o / 'train.json').read_text())
            log(f'  {a} s{seed}: throughput {ti["token_presentations"] / ti["train_time_s"] / 1e6:.2f} M tokens / s')
            (od / 'sae' / f'{a}_s{seed}').mkdir(parents=True, exist_ok=True)
            shutil.copyfile(o / 'train.json', od / 'sae' / f'{a}_s{seed}' / 'train.json')
            torch.cuda.empty_cache()
    del Xg
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    models, calib = {}, {}
    Xca_g = torch.from_numpy(Xca).to(dev)
    del Xca
    for a, c in arms.items():
        for seed in seeds:
            sae, norm, ck = ET.load_any(sae_root / f'{a}_s{seed}' / 'sae.pt', dev)
            assert sae.n_latents == c['width'] and list(sae.prefixes) == c['prefixes'] and sae.k == c['k']
            ema = float(sae.threshold)
            th, info = calibrate_threshold(sae, Xca_g, norm=norm)
            set_threshold(sae, th)
            info['theta_ema'] = ema
            info['calib_stats'] = sparsity_stats(sae, Xca_g, th, norm=norm)
            calib[(a, seed)] = info
            log(f'  {a} s{seed} theta* {th:.4f} (EMA {ema:.4f}) calib L0 {info["l0_calib_theta_star"]:.3f} '
                f'calib FVE {info["calib_stats"]["fve"]:.4f}')
            models[(a, seed)] = (sae, norm)
    del Xca_g
    torch.cuda.empty_cache()
    # ---------------------------------------------------------------- read-out (store set 'train' = TRAIN videos)
    E = pd.read_parquet(PLAN / 'train_frames.parquet')
    t1 = time.time()
    sh, cinfo_eval = load_set('train', E, stage, center)
    log(f'scoring set loaded in {time.time() - t1:.0f}s ({sum(len(s[0]) for s in sh)} tokens)')
    Fm = pd.concat([s[2] for s in sh]).reset_index(drop=True)
    lens_all = [s[1] for s in sh]
    key = pd.MultiIndex.from_frame(Fm[['video', 'i']])
    order = key.get_indexer(pd.MultiIndex.from_frame(E[['video', 'i']]))
    assert (order >= 0).all() and len(np.unique(order)) == len(E) == len(Fm)
    lens = np.concatenate(lens_all)[order]
    pos_all = [(np.load(TOK / f'{b}_{CFG}_pos.npy'), np.r_[0, np.cumsum(l_)][:-1])
               for b, l_ in zip(CT.shards('train', CFG), lens_all)]
    shard_of = np.concatenate([np.full(len(l_), k) for k, l_ in enumerate(lens_all)])[order]
    start_of = np.concatenate([p[1] for p in pos_all])[order]
    Ht = {s: CT.halves(E, s, SW.LABS) for s in range(nsplits)}
    sub = {b: (np.ones(len(E), bool) if b in SW.T1 else E['in_' + b].values.astype(bool)) for b in SW.LABS}
    oi = torch.from_numpy(order).to(dev)
    wall = torch.from_numpy(E.w.values.astype(np.float64)).to(dev)
    for (a, seed), (sae, norm) in models.items():
        t2 = time.time()
        cl, al, acc = [], [], [0.0, 0.0, 0.0, 0, 0]
        for X, ln, _ in sh:
            cm, am, st_ = frame_pool_stats(sae, norm, X, ln, dev)
            cl.append(cm)
            al.append(am)
            acc = [p + q for p, q in zip(acc, st_)]
        s1, s2, sse, l0, N = acc
        fve_eval = float(1 - sse.sum() / (s2 - s1.pow(2) / N).sum())
        codes = torch.cat(cl)[oi].contiguous()
        amx = torch.cat(al)[oi]
        del cl, al
        wr_all = (((codes > 0).double() * wall[:, None]).sum(0) / wall.sum()).cpu().numpy()
        tj = json.loads((sae_root / f'{a}_s{seed}' / 'train.json').read_text())
        res = {'arm': a, 'config': CFG, 'cfg': arms[a]['sae_cfg'], 'seed': seed, 'calib': calib[(a, seed)],
               'fve_calib': calib[(a, seed)]['calib_stats']['fve'], 'fve_eval': fve_eval, 'l0_eval': l0 / N,
               'dead_frac_eval': float((wr_all == 0).mean()), 'dead_at_end_train': tj['dead_at_end_train'],
               'dead_frac_train': tj['dead_at_end_train'] / arms[a]['width'], 'train_time_s': tj['train_time_s'],
               'train_gpu': tj['gpu'], 'centring_tokens_per_video': {'pool': cinfo_pool, 'eval': cinfo_eval},
               'eval_frames': len(E), 'nsplits': nsplits, 'selftest': args.selftest, 'scores': {}}
        rp_picks = {}
        for b in SW.LABS:
            idx = np.flatnonzero(sub[b])
            it = torch.from_numpy(idx).to(dev)
            cs.LABELS = [b]
            Y = torch.from_numpy(E[b].values[idx].astype(bool)[:, None]).to(dev)
            w = torch.from_numpy(E.w.values[idx].astype(np.float64)).to(dev)
            cb = codes[it].contiguous()
            wr = (((cb > 0).double() * w[:, None]).sum(0) / w.sum()).cpu().numpy()
            base = float((w * Y[:, 0].double()).sum() / w.sum())
            res['scores'][b] = {'n_frames': int(len(idx)), 'n_pos': int(Y.sum()), 'weighted_base_rate': base,
                                'n_eligible_1pct': int((wr >= 0.01).sum()), 'dead_frac': float((wr == 0).mean()),
                                'splits': {}}
            rp_picks[b] = []
            for s, h in Ht.items():
                r = cs.cross_fit(cb, Y, w, h[idx], dev)[b]
                r['cf_rate_auc_pick'] = float(np.mean([d['auc']['test_rate'] for d in r['dirs']]))
                r.update(extra_cf_picks(cb, Y, w, h[idx], r))
                rp_picks[b] += r['rprec_picks']
                res['scores'][b]['splits'][f'split{s}'] = SW.compact(r)
            sp = res['scores'][b]['splits']
            for k in ('cf_auroc', 'cf_ap', 'cf_top1', 'cf_top1_rate', 'cf_rate_auc_pick', 'cf_ap_at_auroc', 'cf_rprec',
                      'cf_rprec_auc_pick', 'cf_f1max', 'cf_f1max_auc_pick'):
                res['scores'][b][k] = float(np.mean([sp[x][k] for x in sp]))
            res['scores'][b]['ap_lift'] = res['scores'][b]['cf_ap'] / base
            del cb
        sf, iv = res['scores']['sniff_face'], res['scores']['investigation']
        log(f'  {a} s{seed}: sniff_face AUROC {sf["cf_auroc"]:.3f} AP {sf["cf_ap"]:.4f} top-1% {sf["cf_top1"]:.3f} '
            f'R-prec(R-pick) {sf["cf_rprec"]:.3f} R-prec(AUROC-pick) {sf["cf_rprec_auc_pick"]:.3f}; investigation '
            f'AUROC {iv["cf_auroc"]:.3f} R-prec {iv["cf_rprec"]:.3f}; FVE calib {res["fve_calib"]:.4f} eval '
            f'{fve_eval:.4f}; dead {res["dead_frac_eval"]:.3f}; scored in {time.time() - t2:.0f}s')
        (od / 'eval' / f'{a}_s{seed}.json').write_text(json.dumps(res, indent=1, default=float))
        # sheet JSONs (descriptive): sniff_face neuron picked most often by R-precision and by AUROC
        b = 'sniff_face'
        idx = np.flatnonzero(sub[b])
        auc_picks = [p['auc']['neuron'] for x in res['scores'][b]['splits'].values() for p in x['picks']]
        for tag, picks in (('rprec', rp_picks[b]), ('auc', auc_picks)):
            n = int(pd.Series(picks).mode().iloc[0])
            v = codes[torch.from_numpy(idx).to(dev), n].float().cpu().numpy()
            top = idx[np.argsort(-v, kind='stable')[:16]]
            am = amx[:, n].cpu().numpy()
            rows = []
            for j in top:
                pos, _ = pos_all[shard_of[j]]
                rows.append({'video': E.video.iloc[j], 'i': int(E.i.iloc[j]), 'value': float(codes[j, n]),
                             'label': bool(E[b].iloc[j]), 'stratum': E.stratum.iloc[j],
                             'peak_pos': int(pos[start_of[j] + int(am[j])]), 'grid': 64})
            js = {'arm': a, 'seed': seed, 'label': b, 'pick': tag, 'neuron': n, 'pick_count': int(picks.count(n)),
                  'n_picks': len(picks), 'frames': rows,
                  'note': 'top 16 of the frames in sniff_face sequences in the stratified eval sample (positives '
                          'over-sampled; not a population top-16)'}
            (od / 'sheets' / f'{a}_s{seed}_sniff_face_{tag}.json').write_text(json.dumps(js, indent=1))
        del codes, amx
        torch.cuda.empty_cache()
    log(f'done in {time.time() - t00:.0f}s')


if __name__ == '__main__':
    main()
