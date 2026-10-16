"""
CRIM13 SAE-KNOB TEST: does a label-free SAE recipe change make a cleaner single SNIFFING neuron? The CRIM13 mirror of
the mice v1 round-2 knob sweep, so that a recipe win can be confirmed across datasets. GPU Slurm job per arm
(launcher: align_knobs.sh). Labels are read ONLY in the scoring step (align_train.score, unchanged).

PRE-REGISTERED RULES, written 2026-10-16 BEFORE any SAE of this test was trained or scored (only the token-store
rebuild and a smoke encode had been submitted). Anything added after a knob score was seen is labelled EXPLORATORY.
========================================================================================================================
FIXED (everything as arm C2 of eci_align.sh, whose sniff numbers were known when these rules were written: AUROC 0.771,
  AP 0.372, honest top-1% 0.591, R-precision 0.421 of the AUROC pick; mean over 6 seeds x 2 folds):
  tokens  d3l_640_L20 z_t (DINOv3-L/16, 640 px canvas, block 20), NO change channel, the eci_align.sh 'dark' mask as
          stored; rebuilt by align_knobs_encode.py (= align_encode.cmd_encode, d3l store only).
  SAE     src/eci/sae.MatryoshkaBatchTopKSAE (plain = one prefix = the width), trained by mabe22 eci_train.train_one
          (TokenNorm, geometric-median b_dec, Adam 5e-4, 500 warm-up, cosine, batches of 4096 tokens); inference
          threshold theta* (src/eci/sae_calibrate.calibrate_threshold on the held-out calib tokens).
  read-out frame value = max over the frame's kept tokens (eci_train.frame_max).
  folds   A: SAE on the released train half's pool, neuron picked on the train half's eval frames, scored on the test
          half's eval frames; B: swapped (align_train.FOLDS). Picks and metrics: align_train.score / pick_and_test.
  seeds   600, 601, 602 for every arm (3 seeds x 2 folds).
ARMS (knobs = width, Matryoshka prefixes, k, steps, centring):
  B0  plain BatchTopK 4096, k 16, 6000 steps                    = the C2 recipe with new seeds (CONTROL)
  B1  Matryoshka 512, prefixes 128/256/512, k 16, 6000 steps
  B2  Matryoshka 2048, prefixes 128/512/2048, k 16, 30000 steps
  B3  Matryoshka 8192, prefixes 128/512/2048/8192, k 16, 30000 steps
  B4  B2 with k 8
  B5  B2 with k 32
  B6  B2 on per-video CENTRED tokens: each video's mean token (fp32 mean over ALL stored kept tokens of that video's
      plan frames, every role; label-free) is subtracted before TokenNorm, in training, theta* and the read-out alike
  B7  plain (non-nested) BatchTopK 2048, k 16, 30000 steps
METRICS (sniff; per seed x fold on the scoring half, inverse-probability weights):
  PRIMARY  sniff R-precision of the AUROC-picked neuron, sniff R-precision of the R-precision-picked neuron, sniff
           honest top-1% (top-1%-precision pick among neurons firing on >= 1% of the selection half's weighted frames).
  ALSO     sniff AUROC (AUROC pick), sniff AP (AP pick), AP lift (AP / the scoring half's weighted sniff base rate),
           pair primary AUROC of eci_align.sh (per seed: mean over the 7 pair behaviours of the fold-averaged test
           AUROC of the AUROC-picked neuron), SAE eval FVE (eci_train.frame_max FVE, mean over the 4 shards' eval
           tokens), firing rate of the AUROC pick.
RULE, each arm vs B0, per metric: per-seed value = mean over folds A and B; delta = mean(arm) - mean(B0); Welch 95% CI
  on the 3 vs 3 per-seed values (3 seeds is WEAK: wide CIs, the verdict is conservative).
  sniff primaries and sniff AUROC:  WIN delta >= +0.020 and CI excludes 0;  LOSES delta <= -0.020 and CI excludes 0;
  pair primary AUROC:               WIN delta >= +0.010 and CI excludes 0;  LOSES delta <= -0.010 and CI excludes 0;
  else EQUIVALENT. ARM VERDICT (sniff detector): BETTER if at least one of the 3 sniff primaries WINS and none LOSES;
  WORSE if at least one LOSES and none WINS; MIXED if both; else EQUIVALENT. Pair primary AUROC and the rest are
  reported next to it and do not change the arm verdict.
SANITY: B0 vs the published C2 numbers (seeds 500-505) - B0 is the same recipe, so its sniff means should fall within
  the seed spread; a large gap means the rebuilt store or this script differs from the pipeline (fail loud).
CONTACT SHEET: the arm with the highest mean sniff R-precision (R-prec pick); align_sheets.sniff_sheet on its eval
  JSONs (the model with the highest selection-half sniff AUROC; 16 top scoring-half frames, AUROC and honest picks).
========================================================================================================================
Outputs: OUT/knobs/sae/<arm>_<fold>_s<seed>/{sae.pt, train.json}; OUT/knobs/eval/<arm>_<fold>_s<seed>.json and
<arm>_picks.npz (the align_train.py formats); OUT/knobs/res/knobs.{md,json} (--report, login safe);
OUT/knobs/sheets/ (--sheets <arm>). --selftest: smoke store, 200 steps, fold A, seed 600 -> OUT/knobs_selftest/.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2] / 'scripts/eci'))
sys.path.insert(0, str(HERE.parent / 'mabe22'))
import align_train as AT  # noqa: E402
from align_common import OUT, PAIR, PLAN, SCORED, TOK, log, stage_dir  # noqa: E402

CFG = 'd3l_640_L20'
D = 1024
ARMS = {
    'B0': dict(width=4096, prefixes=[4096], k=16, steps=6000, centre=False),
    'B1': dict(width=512, prefixes=[128, 256, 512], k=16, steps=6000, centre=False),
    'B2': dict(width=2048, prefixes=[128, 512, 2048], k=16, steps=30000, centre=False),
    'B3': dict(width=8192, prefixes=[128, 512, 2048, 8192], k=16, steps=30000, centre=False),
    'B4': dict(width=2048, prefixes=[128, 512, 2048], k=8, steps=30000, centre=False),
    'B5': dict(width=2048, prefixes=[128, 512, 2048], k=32, steps=30000, centre=False),
    'B6': dict(width=2048, prefixes=[128, 512, 2048], k=16, steps=30000, centre=True),
    'B7': dict(width=2048, prefixes=[2048], k=16, steps=30000, centre=False),
}
LABEL = {'B0': 'B0 plain 4096 k16 6k (control)', 'B1': 'B1 Matryoshka 512 k16 6k',
         'B2': 'B2 Matryoshka 2048 k16 30k', 'B3': 'B3 Matryoshka 8192 k16 30k', 'B4': 'B4 = B2 with k8',
         'B5': 'B5 = B2 with k32', 'B6': 'B6 = B2 on per-video centred tokens', 'B7': 'B7 plain 2048 k16 30k'}
SEEDS = [600, 601, 602]


def bases(half, smoke):
    if smoke:
        return [f'{half}_0of2_smoke3']
    return AT.shard_bases(half)


def load_shard(base, stage, centre):
    """-> X (N, 1024) fp16 z_t kept tokens of the whole shard (centred per video if asked), lens, frames table, info."""
    Fm = pd.read_parquet(TOK / f'{base}_frames.parquet')
    src = TOK / f'{base}_{CFG}_t.npy'
    loc = stage / src.name
    shutil.copyfile(src, loc)
    X = np.load(loc)
    loc.unlink()
    lens = Fm.n40.values.astype(np.int64)
    assert lens.sum() == len(X), (base, lens.sum(), X.shape)
    info = {}
    if centre:
        # frames of one video are contiguous in the store (align_encode appends video by video): assert it
        vv = Fm.video.values
        starts = np.flatnonzero(np.r_[True, vv[1:] != vv[:-1]])
        assert len(starts) == len(set(vv)), f'{base}: videos not contiguous'
        st = np.r_[0, np.cumsum(lens)]
        fe = np.r_[starts[1:], len(Fm)]
        norms, resid = [], []
        for a, b in zip(starts, fe):
            ta, tb = st[a], st[b]
            if tb == ta:
                continue
            x = X[ta:tb].astype(np.float32)
            mu = x.mean(0)
            x -= mu
            X[ta:tb] = x.astype(np.float16)
            norms.append(float(np.linalg.norm(mu)))
            resid.append(float(np.abs(X[ta:tb].astype(np.float32).mean(0)).max()))
        info = {'videos': len(norms), 'mean_token_norm_mean': float(np.mean(norms)),
                'max_abs_residual_mean_after_centring': float(np.max(resid))}
        log(f'  {base}: centred {len(norms)} videos, |video mean| {np.mean(norms):.2f}, '
            f'max |residual mean| {np.max(resid):.2e}')
    return X, lens, Fm, info


def subset(X, lens, rows):
    st = np.r_[0, np.cumsum(lens)]
    ti = np.concatenate([np.arange(st[i], st[i + 1]) for i in np.flatnonzero(rows)] or [np.zeros(0, np.int64)])
    return X[ti], lens[rows]


def run(arm, kn, seeds, folds, smoke, out_root, steps=None):
    import torch
    import eci_train as ET
    from src.eci.sae import MatryoshkaBatchTopKSAE
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    W, pre, k = kn['width'], tuple(kn['prefixes']), kn['k']
    steps = steps or kn['steps']
    ET.D, ET.WIDTH = D, W

    def build(c_, din, seed):
        return MatryoshkaBatchTopKSAE(d_in=din, n_latents=W, prefixes=pre, k=c_['k'], seed=seed)
    ET.build = build
    sae_cfg = dict(sp='btk', k=k, matry=len(pre) > 1, prefixes=list(pre), delta=False, config=CFG, width=W, lag=None,
                   tok='mask', arm=arm, steps=steps, centre=kn['centre'], test='align_knobs')
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    for d in ('sae', 'eval'):
        (out_root / d).mkdir(parents=True, exist_ok=True)
    # one read per shard: pool (train / calib) tokens and eval tokens
    H, cinfo = {}, {}
    for half in ('train', 'test'):
        H[half] = []
        for base in bases(half, smoke):
            X, lens, Fm, ci = load_shard(base, stage, kn['centre'])
            cinfo[base] = ci
            role = Fm.role.values
            part = {r: subset(X, lens, role == r) for r in ('train', 'calib', 'eval')}
            if smoke and not kn['centre']:   # selftest: equal to align_train.load (the pipeline's own loader)
                Xr, lr, _ = AT.load(base, CFG, 0, stage, 'mask', role == 'eval')
                assert np.array_equal(Xr, part['eval'][0]) and np.array_equal(lr, part['eval'][1])
                log(f'  {base}: loader == align_train.load on {len(Xr)} eval tokens')
            p40 = np.load(TOK / f'{base}_pos40.npy')
            st0 = np.r_[0, np.cumsum(Fm.n40.values)]
            ev = np.flatnonzero(role == 'eval')
            H[half].append(dict(base=base, part=part, Fe=Fm[role == 'eval'].reset_index(drop=True),
                                pos=[p40[st0[i]:st0[i + 1]] for i in ev]))
            del X
        log(f'  loaded {half}: ' + ', '.join(f"{h['base']} pool {len(h['part']['train'][0])}+"
                                               f"{len(h['part']['calib'][0])} eval {len(h['part']['eval'][0])} tokens"
                                               for h in H[half]))
    models, calib, tinfo = {}, {}, {}
    for fold in folds:
        half = AT.FOLDS[fold][0]
        Xtr = np.concatenate([h['part']['train'][0] for h in H[half]])
        ltr = np.concatenate([h['part']['train'][1] for h in H[half]])
        Xca = np.concatenate([h['part']['calib'][0] for h in H[half]])
        log(f'{arm} fold {fold}: train {Xtr.shape} on {len(ltr)} frames, calib {Xca.shape}; W {W} prefixes {pre} '
            f'k {k} steps {steps} centre {kn["centre"]}')
        name = lambda s: f'{arm}_{fold}_s{s}'  # noqa: E731
        todo = [s for s in seeds if not (out_root / 'sae' / name(s) / 'sae.pt').exists()]
        if todo:
            torch.backends.cuda.matmul.allow_tf32 = True
            Xg = torch.from_numpy(Xtr).to(dev)
            for seed in todo:
                t0 = time.time()
                ET.train_one(name(seed), sae_cfg, seed, Xg, ltr, steps, out_root / 'sae' / name(seed), dev)
                tinfo[name(seed)] = {'train_s': time.time() - t0, 'tokens_per_s': steps * 4096 / (time.time() - t0)}
                log(f'    throughput {tinfo[name(seed)]["tokens_per_s"]:.0f} tokens/s')
                torch.cuda.empty_cache()
            del Xg
        del Xtr
        torch.backends.cuda.matmul.allow_tf32 = False
        Xca_g = torch.from_numpy(Xca).to(dev)
        for seed in seeds:
            sae, norm, ck = ET.load_any(out_root / 'sae' / name(seed) / 'sae.pt', dev)
            assert tuple(sae.prefixes) == pre and sae.n_latents == W and sae.k == k, (sae.prefixes, sae.k)
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
    # eval read-out: every model on the eval frames of BOTH halves
    P = pd.read_parquet(PLAN / 'frames.parquet')
    E = P[P.role == 'eval'].reset_index(drop=True)
    parts = {kk: ([], [], []) for kk in models}
    Fall, lens_all, pos_all = [], [], []
    for half in ('train', 'test'):
        for h in H[half]:
            X, lens = h['part']['eval']
            Fm = h['Fe'].copy()
            Fm['half'] = half
            Fall.append(Fm)
            lens_all.append(lens)
            pos_all += h['pos']
            for kk, (sae, norm) in models.items():
                c, a, stt = ET.frame_max(sae, norm, X, lens, dev, False)
                parts[kk][0].append(c)
                parts[kk][1].append(a)
                parts[kk][2].append(stt)
    del H
    Fm = pd.concat(Fall).reset_index(drop=True)
    if smoke:   # selftest: only the smoke videos' eval frames exist
        key = pd.MultiIndex.from_frame(Fm[['video', 'v']])
        E = E[pd.MultiIndex.from_frame(E[['video', 'v']]).isin(key)].reset_index(drop=True)
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
                 'fve_shard_mean': float(np.mean([x['fve'] for x in sts])), 'fve_per_shard': [x['fve'] for x in sts]}
        sel, tst = AT.FOLDS[fold]
        res = {'arm': arm, 'knobs': kn, 'config': CFG, 'lag': 0, 'fold': fold, 'sel_half': sel, 'test_half': tst,
               'seed': seed, 'cfg': sae_cfg, 'calib': calib[(fold, seed)], 'eval_stats': stats,
               'train_info': tinfo.get(f'{arm}_{fold}_s{seed}'), 'centring': cinfo if kn['centre'] else None,
               'eval_frames': len(E), 'eval_frames_0_tokens': int((lens == 0).sum()), 'labels': SCORED,
               **AT.score(codes, E, Y, w, sel, tst, dev)}
        fr = (((codes > 0).double() * w[:, None]).sum(0) / w.sum()).cpu().numpy()
        res['firing'] = {'dead_frac_eval': float((fr == 0).mean()), 'n_eligible_1pct': int((fr >= 0.01).sum())}
        m = res['main']['sniff']
        log(f'  {arm} {fold} s{seed}: sniff AUROC {m["auc"]["test_auc"]:.3f} AP {m["ap"]["test_ap"]:.3f} honest top-1% '
            f'{m["top1"]["test_top1"]:.3f} R-prec(AUROC pick) {m["auc"]["test_rprec"]:.3f} R-prec(R-prec pick) '
            f'{m["rprec"]["test_rprec"]:.3f}; eval FVE {stats["fve_shard_mean"]:.4f} L0 {stats["l0"]:.2f}')
        (out_root / 'eval' / f'{arm}_{fold}_s{seed}.json').write_text(json.dumps(res, indent=1, default=float))
        for b in SCORED:
            for kind in ('auc', 'top1', 'rprec'):
                j = res['main'][b][kind]['neuron']
                av = arg[:, j].long().cpu().numpy()
                pk = np.array([pos_all[i][av[i]] if lens[i] > 0 else -1 for i in range(len(E))], np.int16)
                picks[f'{fold}_s{seed}_{b}_{kind}'] = np.stack([codes[:, j].float().cpu().numpy(), pk.astype(np.float32)])
        del codes, arg
        torch.cuda.empty_cache()
    np.savez_compressed(out_root / 'eval' / f'{arm}_picks.npz', **picks)
    log(f'{arm} done in {time.time() - t00:.0f}s')


# ------------------------------------------------------------------------------------------------ report (login safe)
def load_arm(root, arm):
    R = {}
    for f in sorted((root / 'eval').glob(f'{arm}_?_s*.json')):
        m = re.fullmatch(rf'{arm}_([AB])_s(\d+)\.json', f.name)
        if m:
            R.setdefault(int(m.group(2)), {})[m.group(1)] = json.loads(f.read_text())
    return {s: v for s, v in R.items() if len(v) == 2}


METRICS = {  # name: (function of one fold result, kind)
    'sniff R-precision (AUROC pick)': (lambda r: r['main']['sniff']['auc']['test_rprec'], 'sniff'),
    'sniff R-precision (R-prec pick)': (lambda r: r['main']['sniff']['rprec']['test_rprec'], 'sniff'),
    'sniff honest top-1%': (lambda r: r['main']['sniff']['top1']['test_top1'], 'sniff'),
    'sniff AUROC': (lambda r: r['main']['sniff']['auc']['test_auc'], 'sniff_also'),
    'sniff AP': (lambda r: r['main']['sniff']['ap']['test_ap'], 'info'),
    'sniff AP lift': (lambda r: r['main']['sniff']['ap']['test_ap'] / r['main']['sniff']['base_test'], 'info'),
    'firing rate (AUROC pick)': (lambda r: r['main']['sniff']['auc']['test_rate'], 'info'),
    'SAE eval FVE': (lambda r: r['eval_stats']['fve_shard_mean'], 'info'),
}
PRIMARY = ['sniff R-precision (AUROC pick)', 'sniff R-precision (R-prec pick)', 'sniff honest top-1%']


def report():
    from align_score import verdict as _v, welch  # noqa: F401
    root = OUT / 'knobs'
    (root / 'res').mkdir(parents=True, exist_ok=True)
    D_ = {a: load_arm(root, a) for a in ARMS}
    D_ = {a: v for a, v in D_.items() if v}
    out = {'arms': {}, 'contrasts': {}}

    def vals(R, f):     # per seed x fold, and per seed (fold mean)
        sf = [f(v[fo]) for v in R.values() for fo in 'AB']
        ps = [np.mean([f(v[fo]) for fo in 'AB']) for _, v in sorted(R.items())]
        return np.array(sf), np.array(ps)

    def pair(R):
        ps = [np.mean([np.mean([v[fo]['main'][b]['auc']['test_auc'] for fo in 'AB']) for b in PAIR])
              for _, v in sorted(R.items())]
        return np.array(ps)
    hdr = list(METRICS) + ['pair primary AUROC']
    L = ['# CRIM13 SAE-knob test (pre-registered rules: scripts/benchmark/crim13/align_knobs.py docstring)', '',
         'd3l_640_L20 z_t tokens, no change channel, mask as stored, frame value = max over kept tokens; seeds 600-602 x '
         'folds A/B. Cells: mean ± sd over seeds x folds (pair primary AUROC: over the 3 per-seed values). Sniff '
         'weighted base rate of the scoring half: 0.1514 (fold A, test half) / 0.1377 (fold B, train half).', '',
         '| arm | n seeds | ' + ' | '.join(hdr) + ' |', '|' + '---|' * (len(hdr) + 2)]
    for a, R in D_.items():
        row, o = [], {'seeds': sorted(R)}
        for nm, (f, _) in METRICS.items():
            sf, ps = vals(R, f)
            o[nm] = {'mean': float(sf.mean()), 'sd': float(sf.std(ddof=1)) if len(sf) > 1 else float('nan'),
                     'per_seed': ps.tolist(), 'per_seed_fold': sf.tolist()}
            row.append(f'{sf.mean():.3f} ± {o[nm]["sd"]:.3f}')
        pp = pair(R)
        o['pair primary AUROC'] = {'mean': float(pp.mean()), 'sd': float(pp.std(ddof=1)) if len(pp) > 1 else float('nan'),
                                   'per_seed': pp.tolist()}
        row.append(f'{pp.mean():.4f} ± {o["pair primary AUROC"]["sd"]:.4f}')
        out['arms'][a] = o
        L.append(f'| {LABEL[a]} | {len(R)} | ' + ' | '.join(row) + ' |')
    if 'B0' in D_:
        L += ['', '## each arm vs B0 (Welch 95% CI on 3 vs 3 per-seed fold-mean values; 3 seeds is weak)', '',
              '| arm | ' + ' | '.join(PRIMARY + ['sniff AUROC', 'pair primary AUROC']) + ' | arm verdict |',
              '|' + '---|' * 7]
        B = out['arms']['B0']
        for a in D_:
            if a == 'B0':
                continue
            cells, vs = [], {}
            for nm in PRIMARY + ['sniff AUROC', 'pair primary AUROC']:
                thr = 0.010 if nm == 'pair primary AUROC' else 0.020
                d, lo, hi, df = welch(out['arms'][a][nm]['per_seed'], B[nm]['per_seed'])
                v = 'WIN' if (d >= thr and lo > 0) else ('LOSES' if (d <= -thr and hi < 0) else 'EQUIVALENT')
                vs[nm] = v
                out['contrasts'].setdefault(a, {})[nm] = {'delta': d, 'lo': lo, 'hi': hi, 'df': df, 'verdict': v}
                cells.append(f'{d:+.3f} [{lo:+.3f}, {hi:+.3f}] {v}')
            pv = [vs[n] for n in PRIMARY]
            av = ('MIXED' if 'WIN' in pv and 'LOSES' in pv else 'BETTER' if 'WIN' in pv else
                  'WORSE' if 'LOSES' in pv else 'EQUIVALENT')
            out['contrasts'][a]['arm_verdict'] = av
            L.append(f'| {LABEL[a]} | ' + ' | '.join(cells) + f' | {av} |')
        L += ['', 'SANITY: published C2 (same recipe, seeds 500-505): sniff AUROC 0.771, AP 0.372, honest top-1% 0.591, '
              'R-precision (AUROC pick) 0.421; B0 above: AUROC ' + f"{B['sniff AUROC']['mean']:.3f}, AP "
              f"{B['sniff AP']['mean']:.3f}, honest top-1% {B['sniff honest top-1%']['mean']:.3f}, R-precision (AUROC "
              f"pick) {B['sniff R-precision (AUROC pick)']['mean']:.3f}."]
    (root / 'res' / 'knobs.md').write_text('\n'.join(L) + '\n')
    (root / 'res' / 'knobs.json').write_text(json.dumps(out, indent=1, default=float))
    print('\n'.join(L))


def sheets(arm):
    import align_sheets as AS
    AS.OUT = OUT / 'knobs'
    AS.SH = OUT / 'knobs' / 'sheets'
    AS.SH.mkdir(parents=True, exist_ok=True)
    AS.sniff_sheet(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=list(ARMS))
    ap.add_argument('--width', type=int)
    ap.add_argument('--prefixes', nargs='+', type=int)
    ap.add_argument('--k', type=int)
    ap.add_argument('--steps', type=int)
    ap.add_argument('--centre', type=int, choices=[0, 1])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--folds', nargs='+', default=['A', 'B'])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--sheets')
    a = ap.parse_args()
    if a.report:
        return report()
    if a.sheets:
        return sheets(a.sheets)
    if a.selftest:
        # smoke store (3 videos per half, shard 0); fold A only (the smoke train half has no C57 video); residents
        # present in both smoke halves only (leave-one-resident-out needs them); 200 steps; two arms: centring + widest
        root = OUT / 'knobs_selftest'
        AT.RESIDENTS = ['A25', 'A29']
        for arm in ('B6', 'B3', 'B0'):
            run(arm, ARMS[arm], [600], ['A'], True, root, steps=200)
            r = json.loads((root / 'eval' / f'{arm}_A_s600.json').read_text())
            l0 = r['calib']['l0_calib_theta_star']
            assert abs(l0 - ARMS[arm]['k']) < 0.5, (arm, l0)
            assert r['cfg']['prefixes'] == ARMS[arm]['prefixes'] and r['cfg']['centre'] == ARMS[arm]['centre']
            if ARMS[arm]['centre']:
                c = r['centring']
                assert all(v['max_abs_residual_mean_after_centring'] < 5e-3 for v in c.values()), c
            log(f'SELFTEST {arm} ok: calib L0 {l0:.3f}, eval FVE {r["eval_stats"]["fve_shard_mean"]:.4f}, '
                f'eval frames {r["eval_frames"]}')
        log('SELFTEST PASSED')
        return
    kn = dict(ARMS[a.arm]) if a.arm else {}
    for f in ('width', 'prefixes', 'k', 'steps'):
        if getattr(a, f) is not None:
            kn[f] = getattr(a, f)
    if a.centre is not None:
        kn['centre'] = bool(a.centre)
    assert set(kn) == {'width', 'prefixes', 'k', 'steps', 'centre'}, kn
    name = a.arm if a.arm and kn == ARMS[a.arm] else (
        f"X_w{kn['width']}_p{'-'.join(map(str, kn['prefixes']))}_k{kn['k']}_s{kn['steps']}_c{int(kn['centre'])}")
    run(name, kn, a.seeds, a.folds, False, OUT / 'knobs')


if __name__ == '__main__':
    main()
