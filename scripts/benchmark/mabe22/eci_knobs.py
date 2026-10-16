"""
MABe22 mouse ECI - KNOB SWEEP on the P2 tokens (DINOv3-L/16 512 px, block 20, 32 x 32 grid, round-3 'arena' mask;
store tag p2d3l_512 built by `eci_p2.sh encode`). Cross-dataset check of the label-free SAE knob sweep run on mice v1
(round 2): does a dictionary width / Matryoshka prefix / k / steps / per-clip centring / change-channel choice make a
cleaner SNIFFING neuron (oral_oral_contact ~ nose-nose, oral_genital_contact ~ nose-tail)?

PRE-REGISTERED RULES, written and committed 2026-10-16 BEFORE any knob-sweep SAE was trained or scored. FROZEN; anything
added after a knob score was seen is labelled EXPLORATORY.

MACHINERY (unchanged, imported): eci_train.train_one (TokenNorm.fit_blocks per 1024-d block, b_dec = geometric median,
Adam 5e-4, 500 warm-up, cosine, batches of 4096 tokens, AuxK, TF32 on for training only), src.eci.sae
MatryoshkaBatchTopKSAE (prefixes given here; plain BatchTopK = one prefix = the width), theta* =
src.eci.sae_calibrate.calibrate_threshold on ALL tokens of the held-out userTrain calibration frames, eval frame value =
max over the frame's kept tokens (eci_train.frame_max), eci_train.score = contact_sae.cross_fit (clip-level halves
eci_train.halves, split seeds 0 and 1, both directions averaged, eval inverse-probability weights), read-out at the full
width. Train pool = plan/train_frames.parquet 'train' rows, calibration = its 'calib' rows, eval = plan/eval_frames.parquet.

ARMS (SAE seeds 600, 601, 602 for every arm; input x = z_t unless 'D'):
    B0   Matryoshka BatchTopK 4096, prefixes 512/1024/2048/4096, k 16, 6000 steps (= the P2N recipe; control)
    B1   Matryoshka 512, prefixes 128/256/512, k 16, 6000 steps
    B2   Matryoshka 2048, prefixes 128/512/2048, k 16, 30000 steps
    B3   Matryoshka 8192, prefixes 128/512/2048/8192, k 16, 30000 steps
    B4   = B2 with k 8
    B5   = B2 with k 32
    B6   = B2 on per-clip CENTRED tokens: from every stored token of a clip subtract that clip's mean stored token
         (mean over all its kept tokens of all its stored frames; train clips: train + calib frames together; eval clips:
         their eval frames). Label-free. Applied identically at training, theta* and read-out.
    B7   plain (non-nested) BatchTopK 2048, k 16, 30000 steps
    B0D  = B0 with x = [z_t, z_t - z_{t-60}] (2 s change channel, = the P2D recipe)
    B2D  = B2 with x = [z_t, z_t - z_{t-60}]

METRICS (per SAE seed: mean over split seeds 0 / 1 and the two cross-fit directions; labels oral_oral_contact and
oral_genital_contact; weighted by the eval IPW weights):
    cf R-precision (NEW): for one neuron on one clip half, N = the weighted positive count of the label in that half;
        precision = weighted positives among the highest-valued frames holding N of the weight mass (ties, including
        the zero block, counted pro rata - no cap at the firing frames); normalised R = (precision - base) / (1 - base),
        base = the half's weighted base rate (0 = chance, 1 = perfect). The neuron with the highest R on one half is
        picked (any neuron, no rate filter), its R on the other half is the score.
    cf AUROC, cf AP (pipeline picks: by AUROC / by AP), AP lift = cf AP / the label's weighted eval base rate,
    honest top-1% (pipeline: top-1% pick among neurons firing on >= 1% of the selection half),
    pipeline primary AUROC = mean cf AUROC over oral_oral, oral_genital, approaches, huddles,
    SAE eval FVE (mean over the 12 eval shards; in each arm's own normalised input space - not comparable between z_t,
    [z_t, delta] and centred inputs).
PRIMARY (the decision) = per seed, the mean over oral_oral and oral_genital of cf R-precision.
CONTRAST arm - B0: Welch 95% CI on the 3 vs 3 per-seed primaries.
    WIN iff the CI lies above 0; LOSES iff it lies below 0; else NOT RESOLVED (equivalent at this power).
    Reported next to it, same Welch CI, descriptive: mean (oo, og) cf AUROC, mean AP lift, mean honest top-1%, pipeline
    primary AUROC. Caveat stated with every verdict: 3 seeds per arm is weak - the CI uses df ~2-4 (t ~ 2.8-4.3), so
    only large, seed-stable effects resolve; seed variation is SAE seed only (same tokens, same eval clips).
    Also descriptive (same Welch CI): B4, B5, B6, B7, B2D vs B2 (each is one knob away from B2) and B0D vs B0.
    A recipe 'win' worth transferring needs the same direction on mice v1; this script makes no transfer decision.

OUTPUTS: SAEs (large) -> BeeGFS $MABE_KNOB_STORE/sae/<arm>_s<seed>/; eval codes of seed 600 (for sheets) ->
$MABE_KNOB_STORE/codes/; small JSON / MD / JPG -> results/vision/benchmark_mabe22/eci/knobs/{eval,knobs.md,knobs.json,
sheets/}. Submit: scripts/benchmark/mabe22/eci_knobs.sh.
Commands:  run --arms B0 B1 ... [--seeds ..] [--steps_scale f] [--smoke]   (GPU)
           selftest (GPU: R-precision vs a numpy brute-force reference on random data with ties)
           score (login safe: reads the eval JSONs)   sheet --arm <arm> (CPU: contact sheet of seed 600)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/eci'))
from eci_common import BEHAVIOURS, OUT, PLAN, SHORT, log, stage_dir  # noqa: E402
import eci_train as ET  # noqa: E402
import eci_enc_train as T  # noqa: E402

T.ETOK = Path(os.environ.get('MABE_ENC_TOK') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/pairpush/mabe22')
STORE = Path(os.environ.get('MABE_KNOB_STORE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/knobs_mabe22')
KOUT = OUT / 'knobs'
TAG, D = 'p2d3l_512', 1024
SEEDS = [600, 601, 602]
SN = ['oral_oral_contact', 'oral_genital_contact']
P4 = ['oral_oral_contact', 'oral_genital_contact', 'approaches', 'huddles']

ARMS = {
    'B0': dict(width=4096, prefixes=(512, 1024, 2048, 4096), k=16, steps=6000, centre=False, delta=False),
    'B1': dict(width=512, prefixes=(128, 256, 512), k=16, steps=6000, centre=False, delta=False),
    'B2': dict(width=2048, prefixes=(128, 512, 2048), k=16, steps=30000, centre=False, delta=False),
    'B3': dict(width=8192, prefixes=(128, 512, 2048, 8192), k=16, steps=30000, centre=False, delta=False),
    'B4': dict(width=2048, prefixes=(128, 512, 2048), k=8, steps=30000, centre=False, delta=False),
    'B5': dict(width=2048, prefixes=(128, 512, 2048), k=32, steps=30000, centre=False, delta=False),
    'B6': dict(width=2048, prefixes=(128, 512, 2048), k=16, steps=30000, centre=True, delta=False),
    'B7': dict(width=2048, prefixes=(2048,), k=16, steps=30000, centre=False, delta=False),
    'B0D': dict(width=4096, prefixes=(512, 1024, 2048, 4096), k=16, steps=6000, centre=False, delta=True),
    'B2D': dict(width=2048, prefixes=(128, 512, 2048), k=16, steps=30000, centre=False, delta=True),
}
VS_B2 = ['B4', 'B5', 'B6', 'B7', 'B2D']


# ---------------------------------------------------------------------------------------------- R-precision
def rprec_cols(X, y, w):
    """X (n, m) float GPU (>= 0), y (n,) bool, w (n,) double -> normalised weighted R-precision per column (m,) double.
    Top set = highest-valued frames holding P (weighted positives) of the weight mass; the tie group at the boundary
    (incl. the zero block) is counted pro rata."""
    import torch
    W = w.sum()
    yw = w * y.double()
    P = yw.sum()
    base = P / W
    out = torch.empty(X.shape[1], dtype=torch.float64, device=X.device)
    for j0 in range(0, X.shape[1], 512):
        xb = X[:, j0:j0 + 512].float().t().contiguous()  # (b, n)
        v, o = torch.sort(xb, dim=1, descending=True)
        cw = torch.cumsum(w[o], 1)
        ib = torch.searchsorted(cw, torch.full((xb.shape[0], 1), float(P) - 1e-9, dtype=cw.dtype, device=cw.device))
        ib = ib.clamp_max(xb.shape[1] - 1)
        vstar = v.gather(1, ib)  # (b, 1) boundary value
        gt, eq = xb > vstar, xb == vstar
        tp_gt = (gt.double() * yw[None]).sum(1)
        w_gt = (gt.double() * w[None]).sum(1)
        tp_eq = (eq.double() * yw[None]).sum(1)
        w_eq = (eq.double() * w[None]).sum(1)
        tp = tp_gt + (P - w_gt) * tp_eq / w_eq
        out[j0:j0 + xb.shape[0]] = (tp / P - base) / (1 - base)
    return out


def rprec_ref(x, y, w):
    """numpy brute-force reference for one column (tie groups explicit)."""
    P, W = (w * y).sum(), w.sum()
    tp, got = 0.0, 0.0
    for val in np.unique(x)[::-1]:
        g = x == val
        wg, pg = w[g].sum(), (w[g] * y[g]).sum()
        if got + wg <= P:
            tp, got = tp + pg, got + wg
        else:
            tp += (P - got) * pg / wg
            got = P
        if got >= P - 1e-12:
            break
    base = P / W
    return (tp / P - base) / (1 - base)


def rprec_cf(codes, E, Ht, dev, labels=BEHAVIOURS):
    """cross-fitted R-precision: pick the best neuron on one half, score it on the other; both directions; per split."""
    import torch
    w = torch.from_numpy(E.w.values.astype(np.float64)).to(dev)
    out = {}
    for s, half in Ht.items():
        res = {}
        rows = {h: torch.from_numpy(np.flatnonzero(half == h)).to(dev) for h in (0, 1)}
        for b in labels:
            y = torch.from_numpy(E[b].values.astype(bool)).to(dev)
            R = {h: rprec_cols(codes[rows[h]], y[rows[h]], w[rows[h]]) for h in (0, 1)}
            dirs = []
            for a, t in ((0, 1), (1, 0)):
                sc = R[a].clone()
                sc[~torch.isfinite(sc)] = -float('inf')
                j = int(torch.argmax(sc))
                dirs.append({'neuron': j, 'select': float(sc[j]), 'test': float(R[t][j])})
            res[b] = {'cf_rprec': float(np.mean([d['test'] for d in dirs])), 'dirs': dirs}
        out[f'split{s}'] = res
    return out


# ---------------------------------------------------------------------------------------------- data
def centre_clips(X, lens, Fm):
    """subtract each clip's mean stored token (fp32 accumulation) in place; X rows are contiguous per clip."""
    clips = Fm['clip'].values
    st = np.r_[0, np.cumsum(lens)]
    chg = np.r_[True, clips[1:] != clips[:-1]]
    starts = np.flatnonzero(chg)
    assert len(starts) == len(set(clips)), 'clip frames not contiguous'
    ends = np.r_[starts[1:], len(clips)]
    for a, b in zip(starts, ends):
        ta, tb = st[a], st[b]
        if tb > ta:
            mu = X[ta:tb].astype(np.float32).mean(0)
            X[ta:tb] = (X[ta:tb].astype(np.float32) - mu).astype(np.float16)
    return X


def cfg_of(delta):
    return dict(store=T.ETOK, suffix=f'_{TAG}', t='L20_t', p='L20_p60', lens='n', D=D, delta=delta)


def load_set(set_, delta, centre, stage, smoke):
    shs = range(T.NSH[set_])[:2 if smoke else None]
    Xs, Ls, Fs = [], [], []
    seen = set()
    for sh in shs:
        X, lens, Fm = T.load_shard(set_, sh, cfg_of(delta), stage)
        cl = set(Fm['clip'])
        assert not (cl & seen), 'a clip spans two shards'
        seen |= cl
        if centre:
            X = centre_clips(X, lens, Fm)
        Xs.append(X)
        Ls.append(lens)
        Fs.append(Fm)
    return Xs, Ls, Fs


def build_factory(cfg):
    def build(_cfg, din, seed):
        from src.eci.sae import MatryoshkaBatchTopKSAE
        return MatryoshkaBatchTopKSAE(d_in=din, n_latents=cfg['width'], prefixes=cfg['prefixes'], k=cfg['k'], seed=seed)
    return build


# ---------------------------------------------------------------------------------------------- run
def run_arm(arm, seeds, dev, stage, smoke, cache):
    import torch
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    cfg = dict(ARMS[arm])
    steps = 300 if smoke else cfg['steps']
    sfx = '_smoke' if smoke else ''
    t00 = time.time()
    key = (cfg['delta'], cfg['centre'])
    if key not in cache:
        cache.clear()
        cache[key] = {s: load_set(s, cfg['delta'], cfg['centre'], stage, smoke) for s in ('train', 'eval')}
    (Xs, Ls, Fs), (Xe, Le, Fe) = cache[key]['train'], cache[key]['eval']
    P = pd.read_parquet(PLAN / 'train_frames.parquet')
    parts = {'train': ([], []), 'calib': ([], [])}
    for X, lens, Fm in zip(Xs, Ls, Fs):
        role = Fm.merge(P, on=['clip', 'i'], how='left').role.values
        assert not pd.isna(role).any()
        tr = np.repeat(role, lens)
        for r in parts:
            parts[r][0].append(X[tr == r])
            parts[r][1].append(lens[role == r])
    Xtr, ltr, Xca, lca = (np.concatenate(v) for r in ('train', 'calib') for v in parts[r])
    del parts
    log(f'{arm} {cfg} steps {steps}: train {Xtr.shape} on {len(ltr)} frames, calib {Xca.shape} on {len(lca)} frames')
    ET.D = D
    ET.build = build_factory(cfg)
    sae_cfg = dict(sp='btk', k=cfg['k'], matry=len(cfg['prefixes']) > 1, delta=cfg['delta'], centre=cfg['centre'],
                   width=cfg['width'], prefixes=list(cfg['prefixes']), steps=steps, tokens=str(T.ETOK), tag=TAG)
    sdir = STORE / f'sae{sfx}'
    todo = [s for s in seeds if not (sdir / f'{arm}_s{s}' / 'sae.pt').exists()]
    if todo:
        Xg = torch.from_numpy(Xtr).to(dev)
        for seed in todo:
            t0 = time.time()
            ET.train_one(arm, sae_cfg, seed, Xg, ltr, steps, sdir / f'{arm}_s{seed}', dev)
            dt = time.time() - t0
            log(f'  throughput {arm} s{seed}: {steps * 4096 / dt / 1e6:.2f} M tokens/s ({dt:.0f}s for {steps} steps)')
            torch.cuda.empty_cache()
        del Xg
    del Xtr
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    models, calib = {}, {}
    Xca_g = torch.from_numpy(Xca).to(dev)
    for seed in seeds:
        sae, norm, ck = ET.load_any(sdir / f'{arm}_s{seed}' / 'sae.pt', dev)
        assert sae.n_latents == cfg['width'] and tuple(sae.prefixes) == tuple(cfg['prefixes']) and sae.k == cfg['k']
        ema = float(sae.threshold)
        th, info = calibrate_threshold(sae, Xca_g, norm=norm)
        set_threshold(sae, th)
        info['theta_ema'] = ema
        info['calib_stats'] = sparsity_stats(sae, Xca_g, th, norm=norm)
        calib[seed] = info
        log(f'  {arm} s{seed} theta* {th:.4f} (EMA {ema:.4f}) calib L0 {info["l0_calib_theta_star"]:.3f}')
        models[seed] = (sae, norm)
    del Xca_g
    torch.backends.cuda.matmul.allow_tf32 = False
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    Fm = pd.concat(Fe).reset_index(drop=True)
    if smoke:
        E = E.merge(Fm[['clip', 'i']], on=['clip', 'i'], how='inner').reset_index(drop=True)
    assert (Fm['clip'].values == E['clip'].values).all() and (Fm.i.values == E.i.values).all()
    lens = np.concatenate(Le)
    Ht = {s: ET.halves(E, s) for s in (0, 1)}
    w = torch.from_numpy(E.w.values).to(dev).double()
    (KOUT / f'eval{sfx}').mkdir(parents=True, exist_ok=True)
    (STORE / f'codes{sfx}').mkdir(parents=True, exist_ok=True)
    for seed, (sae, norm) in models.items():
        t0 = time.time()
        cs, ag, sts = [], [], []
        for X, ln in zip(Xe, Le):
            c, a, stt = ET.frame_max(sae, norm, X, ln, dev, False)
            cs.append(c)
            ag.append(a)
            sts.append(stt)
        codes = torch.cat(cs)
        ntot = sum(x['n_tokens'] for x in sts)
        stats = {'l0': sum(x['l0'] * x['n_tokens'] for x in sts) / ntot, 'n_tokens': ntot,
                 'fve_shard_mean': float(np.mean([x['fve'] for x in sts]))}
        log(f'  read-out {arm} s{seed}: {ntot / (time.time() - t0) / 1e6:.2f} M tokens/s')
        fr = (((codes > 0).double() * w[:, None]).sum(0) / w.sum()).cpu().numpy()
        res = {'arm': arm, 'cfg': sae_cfg, 'seed': seed, 'calib': calib[seed], 'eval_stats': stats,
               'eval_frames': len(E), 'eval_frames_0_tokens': int((lens == 0).sum()),
               'firing': {'dead_frac_eval': float((fr == 0).mean()), 'n_eligible_1pct': int((fr >= 0.01).sum()),
                          'median_rate_alive': float(np.median(fr[fr > 0])) if (fr > 0).any() else None},
               'full': ET.score(codes, E, Ht, dev), 'rprec': rprec_cf(codes, E, Ht, dev)}
        r0, q0 = res['full']['split0'], res['rprec']['split0']
        log(f'  {arm} s{seed}: fve {stats["fve_shard_mean"]:.4f} ' + ', '.join(
            f"{SHORT[b]} auc {r0[b]['cf_auroc']:.3f} ap {r0[b]['cf_ap']:.4f} top1 {r0[b]['cf_top1']:.3f} "
            f"R {q0[b]['cf_rprec']:.3f}" for b in SN))
        (KOUT / f'eval{sfx}' / f'{arm}_s{seed}.json').write_text(json.dumps(res, indent=1, default=float))
        if seed == seeds[0]:
            np.save(STORE / f'codes{sfx}' / f'{arm}_s{seed}_codes.npy', codes.cpu().numpy())
            np.save(STORE / f'codes{sfx}' / f'{arm}_s{seed}_argtok.npy', torch.cat(ag).cpu().numpy())
            np.save(STORE / f'codes{sfx}' / f'{arm}_lens.npy', lens)
        del codes, cs, ag
        torch.cuda.empty_cache()
    log(f'{arm} done in {time.time() - t00:.0f}s')


def cmd_run(a):
    import torch
    dev = torch.device('cuda')
    stage = stage_dir()
    cache = {}
    # group arms by input so each token set is loaded once
    arms = sorted(a.arms, key=lambda x: (ARMS[x]['delta'], ARMS[x]['centre']))
    for arm in arms:
        run_arm(arm, a.seeds, dev, stage, a.smoke, cache)


def cmd_selftest(a):
    import torch
    dev = torch.device('cuda')
    g = np.random.default_rng(0)
    n, m = 3000, 40
    X = np.where(g.random((n, m)) < 0.7, 0.0, np.round(g.random((n, m)) * 5, 1)).astype(np.float16)  # zeros + ties
    y = g.random(n) < 0.05
    X[y, :5] += np.float16(1.0)  # some informative columns
    w = g.random(n) * 3 + 0.1
    got = rprec_cols(torch.from_numpy(X).to(dev), torch.from_numpy(y).to(dev), torch.from_numpy(w).to(dev)).cpu().numpy()
    ref = np.array([rprec_ref(X[:, j].astype(np.float32), y, w) for j in range(m)])
    err = float(np.abs(got - ref).max())
    perfect = rprec_ref(y.astype(np.float32), y, w)
    log(f'R-precision selftest: max |gpu - ref| {err:.2e}; informative cols {got[:5].round(3)}, noise mean '
        f'{got[5:].mean():.3f}; perfect column {perfect:.3f}')
    assert err < 1e-6 and abs(perfect - 1) < 1e-9, 'R-precision selftest FAILED'
    log('R-precision selftest PASSED')


# ---------------------------------------------------------------------------------------------- score
def load_arm(arm, sfx=''):
    R = {}
    for s in SEEDS:
        f = KOUT / f'eval{sfx}' / f'{arm}_s{s}.json'
        if f.exists():
            R[s] = json.loads(f.read_text())
    return R


def seed_metrics(j, base):
    F, Q = j['full'], j['rprec']
    sp = list(F)
    m = lambda b, k: float(np.mean([F[s][b][k] for s in sp]))  # noqa: E731
    out = {'fve': j['eval_stats']['fve_shard_mean'], 'l0': j['eval_stats']['l0'],
           'primary_auroc_pipeline': float(np.mean([m(b, 'cf_auroc') for b in P4]))}
    for b in SN:
        out[f'{SHORT[b]}_auroc'] = m(b, 'cf_auroc')
        out[f'{SHORT[b]}_ap'] = m(b, 'cf_ap')
        out[f'{SHORT[b]}_aplift'] = m(b, 'cf_ap') / base[b]
        out[f'{SHORT[b]}_top1'] = m(b, 'cf_top1')
        out[f'{SHORT[b]}_rprec'] = float(np.mean([Q[s][b]['cf_rprec'] for s in Q]))
    for k in ('auroc', 'aplift', 'top1', 'rprec'):
        out[f'sniff_{k}'] = float(np.mean([out[f'{SHORT[b]}_{k}'] for b in SN]))
    return out


def cmd_score(a):
    from eci_confirm import welch
    sfx = '_smoke' if a.smoke else ''
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    base = {b: float((E[b] * E.w).sum() / E.w.sum()) for b in SN}
    S = {}
    for arm in ARMS:
        R = load_arm(arm, sfx)
        if R:
            S[arm] = {s: seed_metrics(j, base) for s, j in sorted(R.items())}
    cols = [('oral_oral_auroc', 'oo AUROC', 3), ('oral_oral_ap', 'oo AP', 4), ('oral_oral_aplift', 'oo AP lift', 2),
            ('oral_oral_top1', 'oo honest top-1%', 3), ('oral_oral_rprec', 'oo R-prec', 3),
            ('oral_genital_auroc', 'og AUROC', 3), ('oral_genital_ap', 'og AP', 4),
            ('oral_genital_aplift', 'og AP lift', 2), ('oral_genital_top1', 'og honest top-1%', 3),
            ('oral_genital_rprec', 'og R-prec', 3), ('sniff_rprec', 'PRIMARY mean R-prec', 3),
            ('primary_auroc_pipeline', 'pipeline primary AUROC', 3), ('fve', 'eval FVE', 4)]
    L = ['# MABe22 ECI knob sweep on P2 tokens (DINOv3-L/16 512 px block 20; rules: scripts/benchmark/mabe22/'
         'eci_knobs.py)', '',
         f'SAE seeds {SEEDS}; mean ± sd over seeds (each seed = mean over split seeds 0/1 and both cross-fit '
         f'directions). Weighted eval base rate oral_oral (oo) {base[SN[0]]:.4f}, oral_genital (og) {base[SN[1]]:.4f}; '
         'AP lift = cf AP / base rate. R-prec = normalised weighted R-precision (0 chance, 1 perfect), cross-fitted. '
         'eval FVE is in each arm\'s own input space (not comparable across z_t / delta / centred inputs).', '',
         '| arm | n | ' + ' | '.join(c[1] for c in cols) + ' |', '|---|---|' + '---|' * len(cols)]
    for arm, V in S.items():
        cells = []
        for k, _, dp in cols:
            v = np.array([x[k] for x in V.values()])
            cells.append(f'{v.mean():.{dp}f} ± {v.std(ddof=1) if len(v) > 1 else 0:.{dp}f}')
        L.append(f'| {arm} | {len(V)} | ' + ' | '.join(cells) + ' |')
    out = {'base_rate_weighted': base, 'per_seed': S, 'contrasts': {}}
    keys = [('sniff_rprec', 'PRIMARY mean R-prec'), ('sniff_auroc', 'mean cf AUROC'), ('sniff_aplift', 'mean AP lift'),
            ('sniff_top1', 'mean honest top-1%'), ('primary_auroc_pipeline', 'pipeline primary AUROC')]

    def contrast_rows(pairs, title):
        rows = ['', f'## {title}', '', '| contrast | ' + ' | '.join(f'{n} delta [95% CI]' for _, n in keys)
                + ' | verdict (PRIMARY) |', '|---|' + '---|' * (len(keys) + 1)]
        for arm, ref in pairs:
            if arm not in S or ref not in S or len(S[arm]) < 2 or len(S[ref]) < 2:
                continue
            C = {}
            for k, _ in keys:
                d, lo, hi, se, df = welch([x[k] for x in S[arm].values()], [x[k] for x in S[ref].values()])
                C[k] = {'delta': d, 'ci': [lo, hi], 'df': df}
            lo, hi = C['sniff_rprec']['ci']
            v = 'WIN' if lo > 0 else ('LOSES' if hi < 0 else 'NOT RESOLVED')
            C['verdict'] = v
            out['contrasts'][f'{arm}-{ref}'] = C
            rows.append(f'| {arm} - {ref} | ' + ' | '.join(
                f"{C[k]['delta']:+.3f} [{C[k]['ci'][0]:+.3f}, {C[k]['ci'][1]:+.3f}]" for k, _ in keys) + f' | {v} |')
        return rows
    L += contrast_rows([(x, 'B0') for x in ARMS if x != 'B0'], 'Each arm vs B0 (pre-registered decision), Welch 95% CI, '
                       '3 vs 3 seeds (weak: df ~2-4)')
    L += contrast_rows([(x, 'B2') for x in VS_B2] + [('B0D', 'B0')], 'One knob away (descriptive)')
    (KOUT / f'knobs{sfx}.md').write_text('\n'.join(L) + '\n')
    (KOUT / f'knobs{sfx}.json').write_text(json.dumps(out, indent=1, default=float))
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------- sheet
def cmd_sheet(a):
    """contact sheet (seed 600, split 0): oral_oral neuron picked on half 0 by R-precision and by AUROC, its 16
    highest frames of half 1. Tile = 224 frame x2, red box = the patch of the frame's max, green border = positive."""
    from PIL import Image, ImageDraw
    from eci_common import decode_clip, offset
    stage = stage_dir()
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    h = ET.halves(E, 0)
    off = offset()
    seed = SEEDS[0]
    j_ = json.loads((KOUT / 'eval' / f'{a.arm}_s{seed}.json').read_text())
    codes = np.load(STORE / 'codes' / f'{a.arm}_s{seed}_codes.npy', mmap_mode='r')
    argt = np.load(STORE / 'codes' / f'{a.arm}_s{seed}_argtok.npy', mmap_mode='r')
    lens = np.load(STORE / 'codes' / f'{a.arm}_lens.npy')
    pos = np.concatenate([np.load(T.ETOK / f'eval_{s}of12_{TAG}_pos.npy') for s in range(12)]).astype(np.int64)
    st = np.r_[0, np.cumsum(lens)]
    assert st[-1] == len(pos)
    G, ps = 32, 7  # 32 x 32 grid over the 224 frame
    od = KOUT / 'sheets'
    od.mkdir(parents=True, exist_ok=True)
    index = {}
    for lab in a.labels:
        for pick in ('rprec', 'auc'):
            if pick == 'rprec':
                j = j_['rprec']['split0'][lab]['dirs'][0]['neuron']
                e = j_['rprec']['split0'][lab]['dirs'][0]
            else:
                e = j_['full']['split0'][lab]['dirs'][0]['auc']
                j = e['neuron']
            rows = np.flatnonzero(h == 1)
            v = np.asarray(codes[rows, j], np.float32)
            top = rows[np.argsort(-v, kind='stable')[:16]]
            img, by = {}, {}
            for f in top:
                by.setdefault(E['clip'][f], []).append(f)
            for c, fs in by.items():
                fr = decode_clip('mouse_submission', c, stage, 'rgb24')
                for f in fs:
                    img[f] = fr[E.i[f] + off]
            Wd = Image.new('RGB', (4 * 448, 4 * 448))
            for q, f in enumerate(top):
                im = Image.fromarray(img[f]).resize((448, 448), Image.BICUBIC)
                d = ImageDraw.Draw(im)
                if lens[f] > 0 and codes[f, j] > 0:
                    p = int(pos[st[f] + int(argt[f, j])])
                    r, cc = divmod(p, G)
                    d.rectangle([cc * ps * 2, r * ps * 2, (cc + 1) * ps * 2, (r + 1) * ps * 2], outline=(255, 0, 0),
                                width=3)
                pz = bool(E[lab][f])
                d.rectangle([0, 0, 447, 447], outline=(0, 220, 0) if pz else (128, 128, 128), width=6)
                d.text((8, 8), f'{E["clip"][f][:10]} i{E.i[f]} v{float(codes[f, j]):.2f} {"POS" if pz else "neg"}',
                       fill=(255, 255, 0))
                Wd.paste(im, ((q % 4) * 448, (q // 4) * 448))
            path = od / f'{a.arm}_s{seed}_{lab.replace("_contact", "")}_{pick}.jpg'
            Wd.save(path, quality=85)
            npos = int(E[lab].values[top].sum())
            index[path.name] = {'arm': a.arm, 'seed': seed, 'label': lab, 'pick': pick, 'neuron': j,
                                'pick_detail': e, 'positives_in_16': npos}
            log(f'{path.name}: neuron {j}, {npos}/16 positive in the top 16 frames of half 1')
    (od / f'index_{a.arm}.json').write_text(json.dumps(index, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--arms', nargs='+', choices=list(ARMS), required=True)
    r.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    r.add_argument('--smoke', action='store_true', help='2 shards per set, 300 steps, outputs suffixed _smoke')
    sp.add_parser('selftest')
    s = sp.add_parser('score')
    s.add_argument('--smoke', action='store_true')
    h = sp.add_parser('sheet')
    h.add_argument('--arm', required=True)
    h.add_argument('--labels', nargs='+', default=['oral_oral_contact'])
    a = ap.parse_args()
    {'run': cmd_run, 'selftest': cmd_selftest, 'score': cmd_score, 'sheet': cmd_sheet}[a.cmd](a)


if __name__ == '__main__':
    main()
