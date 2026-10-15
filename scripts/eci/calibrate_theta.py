"""
CALIBRATE_THETA: replace the EMA inference threshold of the ECI Matryoshka BatchTopK SAEs by a calibrated one.

The SAE (src/eci/sae.py) encodes at inference z = pre * 1[pre > theta] with one global theta, estimated in training as
an EMA (rate 0.01) of the smallest activation BatchTopK kept. Calibrated alternative (src/eci/sae_calibrate.py):
theta* = the (N k + 1)-th largest of the N x m pooled post-ReLU pre-activations of N held-out TRAINING tokens
(= empirical (1 - k/m) quantile), so the mean L0 per held-out train token is exactly k. No rate, no retraining.

SAEs
    deployed C1 (seed 0): dataset/<d>/eci/sae/matryoshka_btk_4096_k16_<tag>_s0, d = mice (fg448m5), ants (antsfgm5),
        frogs (frogsfgm5); trained on a uniform token pool (rollout_c1: seed 0, mice 4M / ants 2M / frogs 2M tokens) of
        the domain's 1 fps motion store (<eci>/train_tokens/dinov2_base_l-1_<rule>_fps1_d5).
    harness (mice, 3 seeds): c448_s* (results/vision/eci_res_pool/mice/sae), btk16_s*, btk8_s*
        (results/vision/eci_sparsity/sae); trained on ALL tokens of the 10,000 res_pool train frames (100 unannotated
        videos, every ~9th 1 fps store frame).
Held-out train tokens (from the same 1 fps d5 store, x = [token_t, token_t - token_{t-5}], read with pread of whole
    store frames, never staged in bulk):
    deployed  store tokens NOT in the training pool (pool rebuilt as rollout_c1: default_rng(0).choice(N, cap)).
    harness   tokens of store frames of the 100 harness train videos that are NOT among the 10,000 train frames.
    Store frames are shuffled (seed 2) and split into two disjoint sets: CAL (calibration, theta* is fitted here) and
    CHK (out-of-sample check of the train L0), ~1M tokens each (frogs: all 278,886 held-out tokens, split by frames).
Eval tokens: mice = the 20,000-frame annotated B subset of the sparsity_test / quick_repr harness (re-encoded from the
    staged JPGs here, checked against the store); ants / frogs have no eval harness, CHK plays that role.

Mice harness scoring (sparsity_test.eval_arm line for line, threshold = EMA theta, then theta*): frame value = MAX over
    kept tokens, cross-fitted best neuron (vjepa.score, weighted), usable / all frames, prefixes 512 / 4096, FVE, L0.
    The EMA run is checked against the stored results/vision/eci_sparsity/eval/<arm>_s<seed>.json.

PRE-REGISTERED VERDICT (fixed before any result): theta* is acceptable iff for every harness arm (c448, btk16, btk8;
    3-seed means, usable frames, prefix 4096) |dAUROC| <= 0.01 and |dAP| <= 0.005 on BOTH nose_nose and nose_tail
    (d = theta* - EMA). Then theta* replaces the EMA theta (no retraining).

RESULTS (job 827764, A10, 2026-10-15, 38 min; selftest job 827760: theta* == brute-force sort on 4 random SAEs)
    theta* is BELOW the EMA theta on all 12 SAEs, by 0.0003 (ants) to 0.009 (btk8): the EMA ran slightly high, so
    the EMA held-out train L0 was 15.93-16.00 (k = 16) / 7.97-7.99 (k = 8) instead of exactly k.
    SAE        EMA theta  theta*   L0 CAL EMA / theta*   L0 CHK EMA / theta*   FVE CHK EMA / theta*
    dep_mice   2.8346     2.8277   15.949 / 16.000       16.066 / 16.118       0.7911 / 0.7913
    dep_ants   2.7009     2.7006   15.997 / 16.000       16.018 / 16.021       0.8407 / 0.8407
    dep_frogs  3.0188     3.0140   15.969 / 16.000       15.954 / 15.986       0.8327 / 0.8328   (139,444 tokens/set)
    c448 s0-2  2.843-2.844 2.838-2.840  15.95-15.98 / 16  15.92-15.95 / 15.97-15.98   0.7974-0.7976 / +0.0001-0.0002
    btk16 s0-2 2.843-2.851 2.839-2.843  15.93-15.97 / 16  15.92-15.95 / 15.97-15.98   0.7971-0.7977 / +0.0001-0.0003
    btk8 s0-2  4.602-4.668 4.595-4.659  7.97-7.99 / 8     7.96-7.97 / 7.98-7.99       0.7417-0.7439 / +0.0002
    Mice B-subset eval tokens: L0 is ~1 above k under BOTH thresholds (EMA 16.94-17.02, theta* 16.98-17.05; btk8
    8.39-8.42 / 8.42-8.44; dep_mice 16.64 / 16.70): a train -> eval distribution shift, not a calibration error.
    EMA eval reproduces the stored sparsity_test JSONs (max |diff| AUROC / AP / top-1% 5.2e-4, |dFVE| 2e-7).
    theta* - EMA, usable m4096, 3-seed means (max |per-seed delta| AUROC / AP):
        c448   nn dAUROC +0.0000 dAP +0.0000, nt +0.0000 / +0.0000; max seed 7e-5 / 5e-6
        btk16  nn +0.0001 / +0.0000, nt +0.0001 / +0.0000; max seed 3.8e-4 / 2.5e-5
        btk8   nn +0.0001 / +0.0000, nt +0.0001 / +0.0000; max seed 2.9e-4 / 7.1e-5
        dep_mice (1 seed) |dAUROC| <= 4e-5, |dAP| <= 4e-5; top-1% unchanged everywhere; FVE +0.0002 everywhere.
    VERDICT: theta* acceptable (all arms far inside |dAUROC| <= 0.01, |dAP| <= 0.005) -> replaces the EMA theta.

Steps
    selftest  GPU  calibrate_threshold vs brute-force sort on small random SAEs (exact rank check)
    run       GPU  held-out sets -> theta* for every SAE -> train / eval L0 + FVE -> mice harness eval under both thetas
    summary   any  OUT/table.md, OUT/summary.json (reads JSONs only)
Outputs: results/vision/eci_calibrate_theta/ (gitignored).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
OUT = Path(os.environ.get('CT_OUT') or REPO / 'results/vision/eci_calibrate_theta')
D = 768
LABELS = ['nose_nose', 'nose_tail']
STORE = {'mice': 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1_d5',
         'ants': 'dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfg_fps1_d5',
         'frogs': 'dataset/frogs/eci/train_tokens/dinov2_base_l-1_frogsfg_fps1_d5'}
ECI = {'mice': 'dataset/mice/v1/eci', 'ants': 'dataset/ants/eci', 'frogs': 'dataset/frogs/eci'}
TAG = {'mice': 'fg448m5', 'ants': 'antsfgm5', 'frogs': 'frogsfgm5'}
CAP = {'mice': 4_000_000, 'ants': 2_000_000, 'frogs': 2_000_000}  # rollout_c1.CAP
N_SET = 1_000_000
HARNESS = [f'{a}_s{s}' for a in ('c448', 'btk16', 'btk8') for s in (0, 1, 2)]
SP_EVAL = REPO / 'results/vision/eci_sparsity/eval'
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def sae_path(name):
    if name.startswith('dep_'):
        d = name[4:]
        return REPO / ECI[d] / 'sae' / f'matryoshka_btk_4096_k16_{TAG[d]}_s0' / 'sae.pt'
    arm = name.split('_s')[0]
    base = REPO / ('results/vision/eci_res_pool/mice/sae' if arm == 'c448' else 'results/vision/eci_sparsity/sae')
    return base / name / 'sae.pt'


def stage_dir():
    d = Path(os.environ.get('LOCAL_DIR') or '')
    tmp = os.environ.get('TMPDIR') or '/nonexistent'
    if not (str(d).startswith('/localhome') or str(d).startswith(tmp)):
        raise RuntimeError(f'LOCAL_DIR must be under /localhome or $TMPDIR, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------- held-out sets
class Store:
    """The d5 store: per-frame token layout over all shards (global token index = shard offset + frame offset)."""

    def __init__(self, domain):
        from src.eci.foreground import FgTokenStore
        self.st = FgTokenStore(REPO / STORE[domain])
        for i in self.st.info:
            assert int(i['motion_delta']) == 5 and int(i['stride']) == 5, i
        self.N = int(self.st.sizes.sum())
        self.shard_start = np.r_[0, np.cumsum(self.st.sizes)]
        sh, f0, nfg, rows = [], [], [], []
        for s, d in enumerate(self.st.dirs):
            z = np.load(d / 'frames.npz')
            n = z['n_fg'].astype(np.int64)
            assert n.sum() == self.st.sizes[s], (d, n.sum(), self.st.sizes[s])
            sh.append(np.full(len(n), s)); f0.append(self.shard_start[s] + np.r_[0, np.cumsum(n)[:-1]])
            nfg.append(n); rows.append(z['rows'].astype(np.int64))
        self.shard, self.f0, self.nfg, self.rows = map(np.concatenate, (sh, f0, nfg, rows))

    def read(self, frames, drop=None):
        """tokens of store frames (any order; read sorted) -> X (n, 1536) fp16 = [tok, tok - prev], minus the global
        token indices in `drop` (sorted). Also returns the global token indices kept."""
        import rollout_c1 as rc
        frames = np.sort(frames)
        gidx = np.concatenate([np.arange(self.f0[f], self.f0[f] + self.nfg[f]) for f in frames])
        keep = np.ones(len(gidx), bool) if drop is None else ~np.isin(gidx, drop, assume_unique=False)
        parts, kept = [], []
        for s in np.unique(self.shard[frames]):
            m = (gidx >= self.shard_start[s]) & (gidx < self.shard_start[s + 1])
            loc = gidx[m] - self.shard_start[s]
            tok = rc.pread_gather(self.st.dirs[s] / 'tokens.f16', loc, D)
            prv = rc.pread_gather(self.st.dirs[s] / 'prev.f16', loc, D)
            k = keep[m]
            X = np.empty((int(k.sum()), 2 * D), np.float16)
            X[:, :D] = tok[k]
            X[:, D:] = (tok[k].astype(np.float32) - prv[k].astype(np.float32)).astype(np.float16)
            parts.append(X); kept.append(gidx[m][k])
        return np.concatenate(parts), np.concatenate(kept)


def split_frames(cand, nfg_kept, n_set, seed=2):
    """shuffle candidate frames, take frames until each of CAL / CHK holds >= n_set tokens (or split all in half)."""
    g = np.random.default_rng(seed)
    order = cand[g.permutation(len(cand))]
    c = np.cumsum(nfg_kept[order])
    if c[-1] < 2 * n_set:  # not enough: all frames, split in half by tokens
        a = int(np.searchsorted(c, c[-1] / 2)) + 1
        return order[:a], order[a:]
    a = int(np.searchsorted(c, n_set)) + 1
    b = int(np.searchsorted(c, c[a - 1] + n_set)) + 1
    return order[:a], order[a:b]


def heldout_sets(kind, S):
    """kind: 'dep_<domain>' or 'harness'. -> {'cal': X, 'chk': X}, info."""
    import pandas as pd
    if kind.startswith('dep_'):
        d = kind[4:]
        cap = min(CAP[d], S.N)
        t0 = time.time()
        sel = np.sort(np.random.default_rng(0).choice(S.N, cap, replace=False))  # rollout_c1 cmd_train pool
        log(f'{kind}: pool rebuilt ({cap:,} of {S.N:,} tokens) in {time.time() - t0:.0f}s')
        # kept tokens per frame (frame f owns global tokens [f0, f0 + nfg))
        in_pool = np.bincount(np.searchsorted(S.f0, sel, side='right') - 1, minlength=len(S.f0))
        nk = S.nfg - in_pool
        cand = np.flatnonzero(nk > 0)
        drop = sel
        info = {'pool': f'rollout_c1 pool default_rng(0).choice({S.N}, {cap}) excluded', 'n_store_tokens': S.N,
                'n_pool': cap, 'n_heldout_available': int(nk.sum())}
    else:
        T = pd.read_parquet(REPO / 'results/vision/eci_res_pool/mice/train/frames.parquet')
        assert (S.rows[T.store_frame.values] == T.row.values).all(), 'store_frame -> row mismatch'
        import src.eci.domain as dm
        obs = pd.read_csv(dm.get_domain('mice').ann_path, usecols=['observation_id'])['observation_id'].values
        fobs = obs[S.rows]
        cand = np.flatnonzero(np.isin(fobs, T.obs.unique()) & (S.nfg > 0))
        cand = np.setdiff1d(cand, T.store_frame.values)
        nk = S.nfg.copy()
        drop = None
        info = {'pool': 'store frames of the 100 harness train videos minus the 10,000 train frames',
                'n_train_videos': int(T.obs.nunique()), 'n_candidate_frames': len(cand),
                'n_heldout_available': int(nk[cand].sum())}
    a, b = split_frames(cand, nk, N_SET)
    assert len(np.intersect1d(a, b)) == 0
    out = {}
    for nm, fr in (('cal', a), ('chk', b)):
        t0 = time.time()
        X, gk = S.read(fr, drop)
        if drop is not None:
            assert not np.isin(gk, drop).any()
        out[nm] = X
        info[nm] = {'n_frames': len(fr), 'n_tokens': len(X), 'read_s': round(time.time() - t0, 1)}
        log(f'{kind} {nm}: {len(fr):,} frames, {len(X):,} tokens read in {time.time() - t0:.0f}s')
    return out, info


# ---------------------------------------------------------------------------------------------- theta
def theta_one(name, sets, dev, extra_info, eval_X=None):
    from src.eci.sae_calibrate import calibrate_threshold, sparsity_stats
    from src.eci.sae import load_sae
    sae, norm, ck = load_sae(sae_path(name), dev)
    ema = float(sae.threshold)
    t0 = time.time()
    th, ci = calibrate_threshold(sae, sets['cal'], norm=norm)
    log(f'{name}: EMA theta {ema:.4f} -> theta* {th:.4f} ({time.time() - t0:.0f}s); cal L0 theta* '
        f'{ci["l0_calib_theta_star"]:.4f}, EMA {ci["l0_calib_theta_current"]:.4f}')
    res = {'sae': name, 'path': str(sae_path(name).relative_to(REPO)), 'k': sae.k, 'n_latents': sae.n_latents,
           'theta_ema': ema, 'theta_star': th, 'calibration': ci, 'sets': extra_info, 'stats': {}}
    for nm in ('cal', 'chk'):
        for tn, t in (('ema', ema), ('star', th)):
            res['stats'][f'{nm}_{tn}'] = sparsity_stats(sae, sets[nm], t, norm=norm)
    log(f'{name}: ' + ', '.join(f"{k} L0 {v['l0']:.3f} FVE {v['fve']:.4f}" for k, v in res['stats'].items()))
    (OUT / 'theta').mkdir(parents=True, exist_ok=True)
    (OUT / 'theta' / f'{name}.json').write_text(json.dumps(res, indent=1))
    return th


# ---------------------------------------------------------------------------------------------- harness eval
def eval_harness(name, theta_name, theta, Xev, tb, dev, half_all):
    """sparsity_test.eval_arm with the threshold set to `theta` (written to OUT/eval/<name>_<theta_name>.json)."""
    import torch
    import vjepa as vj
    import sparsity_test as spt
    from src.eci.sae import load_sae
    from src.eci.sae_calibrate import set_threshold
    torch.backends.cuda.matmul.allow_tf32 = False
    sae, norm, _ = load_sae(sae_path(name), dev)
    ema = float(sae.threshold)
    set_threshold(sae, theta)
    F, sub = tb['eval']['F'], tb['eval']['sub']
    codes, st, _ = vj.frame_max_tokens(sae, norm, Xev, tb['eval']['lens'], dev)
    rows = {'all': np.arange(len(F)), 'usable': np.flatnonzero(F.usable.values)}
    res = {'stats': st, 'k': sae.k, 'theta': float(sae.threshold), 'theta_ema': ema, 'theta_name': theta_name}
    for fs, rr in rows.items():
        for p in (512, 4096):
            res[f'{fs}_m{p}'] = vj.score(codes[:, :p].contiguous(), sub, rr, half_all, dev)
    r = res['usable_m4096']
    w = sub.w.values.astype(np.float64)
    fr = spt.firing(codes, w, rows['usable'])
    elig = fr >= spt.MIN_RATE
    res['fire_rate_usable_w'] = fr.tolist()
    res['firing'] = {'dead_frac_eval': float((fr == 0).mean()), 'n_eligible': int(elig.sum()),
                     'median_rate_eligible': float(np.median(fr[elig])) if elig.any() else float('nan')}
    log(f"  {name} [{theta_name} {theta:.4f}] fve {st['fve']:.4f} l0 {st['l0']:.3f} usable m4096: " + ', '.join(
        f"{b} AUROC {r[b]['cf_auroc']:.4f} AP {r[b]['cf_ap']:.4f} top1 {r[b]['cf_top1']:.3f}" for b in LABELS))
    ref = SP_EVAL / f'{name}.json'
    if theta_name == 'ema' and ref.exists():  # reproduction check vs the stored sparsity_test eval
        R = json.loads(ref.read_text())
        dd = {f'{key}/{b}/{q}': res[key][b][q] - R[key][b][q] for key in ('usable_m4096', 'usable_m512', 'all_m4096')
              for b in LABELS for q in ('cf_auroc', 'cf_ap', 'cf_top1')}
        res['repro_vs_sparsity_test'] = {'max_abs_diff': float(max(abs(v) for v in dd.values())),
                                         'd_fve': st['fve'] - R['stats']['fve'], 'd_l0': st['l0'] - R['stats']['l0'],
                                         'theta_stored': R['theta']}
        log(f"  {name} EMA reproduction vs {ref.name}: {res['repro_vs_sparsity_test']}")
    (OUT / 'eval').mkdir(parents=True, exist_ok=True)
    (OUT / 'eval' / f'{name}_{theta_name}.json').write_text(json.dumps(res, indent=1, default=float))
    del codes
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- steps
def cmd_selftest(args):
    """calibrate_threshold vs brute force (full sort) on random SAEs: theta* must equal the (N k + 1)-th largest."""
    import torch
    from src.eci.sae import MatryoshkaBatchTopKSAE
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    out = []
    for seed, (N, d, m, k, nb) in enumerate([(3000, 64, 512, 16, 1 << 20), (5000, 128, 1024, 8, 64),
                                              (2000, 32, 256, 4, 7), (4000, 96, 2048, 32, 1 << 12)]):
        sae = MatryoshkaBatchTopKSAE(d_in=d, n_latents=m, prefixes=(m // 2, m), k=k, seed=seed).to(dev)
        with torch.no_grad():
            sae.b_enc.normal_(0, 0.3, generator=torch.Generator(device=dev).manual_seed(seed))
            sae.threshold.fill_(0.5)
        x = torch.randn(N, d, generator=torch.Generator().manual_seed(seed))
        th, info = calibrate_threshold(sae, x, k=k, batch=777, n_bins=nb)  # odd batch: many partial batches
        with torch.no_grad():  # same batches as calibrate_threshold, so the fp32 values are bit-identical
            a = torch.cat([torch.relu(sae.pre_acts(x[i:i + 777].to(dev))).flatten() for i in range(0, N, 777)])
            a = a.sort(descending=True).values
        brute = float(a[N * k])  # (N k + 1)-th largest, 0-based index N k
        l0 = float((a > th).sum()) / N
        ok = th == brute and abs(l0 - k) < 1e-9
        out.append({'N': N, 'm': m, 'k': k, 'n_bins': nb, 'theta_star': th, 'brute': brute, 'l0': l0, 'ok': ok,
                    'l0_calib_reported': info['l0_calib_theta_star']})
        log(f'selftest {out[-1]}')
        assert ok, out[-1]
        set_threshold(sae, th)
        assert float(sae.threshold) == np.float32(th)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'selftest.json').write_text(json.dumps(out, indent=1))
    log('selftest passed')


def cmd_run(args):
    import torch
    dev = torch.device('cuda')
    t00 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cmd_selftest(args)
    thetas = {}
    # ---- deployed C1 + harness: held-out train tokens from the 1 fps d5 stores
    for d in args.domains:
        S = Store(d)
        log(f'{d}: store {STORE[d]}: {len(S.st.dirs)} shards, {S.N:,} tokens, {len(S.f0):,} frames')
        kinds = [f'dep_{d}'] + (['harness'] if d == 'mice' and not args.no_harness else [])
        for kind in kinds:
            sets, info = heldout_sets(kind, S)
            names = [kind] if kind.startswith('dep_') else HARNESS
            for nm in names:
                thetas[nm] = theta_one(nm, sets, dev, info)
            del sets
    (OUT / 'thetas.json').write_text(json.dumps(thetas, indent=1))
    log(f'thetas done in {time.time() - t00:.0f}s: {thetas}')
    if args.no_harness or 'mice' not in args.domains:
        return
    # ---- mice B-subset harness under both thresholds
    import contact_sae as cs
    import quick_repr as qr
    from src.eci.sae import load_sae
    stage = stage_dir()
    tb = qr.tables()
    tb = {'train': qr.take(tb['train'], np.arange(64)), 'eval': tb['eval']}  # train: only the 64 store-check frames
    half_all = cs.video_units(tb['eval']['sub'].obs.values)
    jpg = qr.stage_jpgs(tb, stage)
    X, einfo = qr.encode_sets('base', ['last'], tb, jpg, dev, args.bs, args.workers)
    X = X['last']
    einfo['store_check'] = qr.check_vs_store(X, tb)
    (OUT / 'encode.json').write_text(json.dumps({'gpu': torch.cuda.get_device_name(0), **einfo,
                                                 'n_eval_tokens': int(tb['eval']['lens'].sum())}, indent=1, default=float))
    for nm in HARNESS + ['dep_mice']:
        ema = float(load_sae(sae_path(nm), 'cpu')[0].threshold)
        for tn, t in (('ema', ema), ('star', thetas[nm])):
            eval_harness(nm, tn, t, X['eval'], tb, dev, half_all)
    log(f'run done in {time.time() - t00:.0f}s')


def cmd_summary(args):
    ms = lambda a, n=4: f'{np.mean(a):.{n}f}+-{np.std(a):.{n}f}' if len(a) > 1 else f'{np.mean(a):.{n}f}'  # noqa: E731
    TH = {f.stem: json.loads(f.read_text()) for f in sorted((OUT / 'theta').glob('*.json'))}
    lines = ['# calibrate_theta: EMA theta vs calibrated theta* (mean L0 = k on held-out train tokens)', '',
             '## thresholds and L0 / FVE per token (CAL = calibration set, CHK = disjoint held-out train set)', '',
             '| SAE | EMA theta | theta* | theta* - EMA | CAL tokens | L0 CAL EMA | L0 CAL theta* | L0 CHK EMA | '
             'L0 CHK theta* | FVE CHK EMA | FVE CHK theta* |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for n, r in TH.items():
        s = r['stats']
        lines.append(f"| {n} | {r['theta_ema']:.4f} | {r['theta_star']:.4f} | {r['theta_star'] - r['theta_ema']:+.4f} | "
                     f"{s['cal_ema']['n_tokens']:,} | {s['cal_ema']['l0']:.3f} | {s['cal_star']['l0']:.3f} | "
                     f"{s['chk_ema']['l0']:.3f} | {s['chk_star']['l0']:.3f} | {s['chk_ema']['fve']:.4f} | "
                     f"{s['chk_star']['fve']:.4f} |")
    E = {f.stem: json.loads(f.read_text()) for f in sorted((OUT / 'eval').glob('*.json'))}
    S, V = {'theta': TH}, {}
    if E:
        lines += ['', '## mice B-subset harness (usable frames, prefix 4096; eval-token L0 / FVE)', '',
                  '| SAE | theta | eval L0 | eval FVE | nn AUROC | nn AP | nn top-1% | nt AUROC | nt AP | nt top-1% |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for n in HARNESS + ['dep_mice']:
            for tn in ('ema', 'star'):
                r = E.get(f'{n}_{tn}')
                if r is None:
                    continue
                u = r['usable_m4096']
                lines.append(f"| {n} | {tn} {r['theta']:.4f} | {r['stats']['l0']:.3f} | {r['stats']['fve']:.4f} | " + ' | '.join(
                    f"{u[b]['cf_auroc']:.4f} | {u[b]['cf_ap']:.4f} | {u[b]['cf_top1']:.3f}" for b in LABELS) + ' |')
        rep = {n: E[f'{n}_ema']['repro_vs_sparsity_test'] for n in HARNESS if 'repro_vs_sparsity_test' in E.get(f'{n}_ema', {})}
        if rep:
            lines += ['', f"EMA reproduction vs stored sparsity_test evals: max |diff| over AUROC / AP / top-1% = "
                      f"{max(v['max_abs_diff'] for v in rep.values()):.2e} (max |dFVE| "
                      f"{max(abs(v['d_fve']) for v in rep.values()):.2e}, max |dL0| {max(abs(v['d_l0']) for v in rep.values()):.2e})"]
        lines += ['', '## deltas theta* - EMA (usable, m4096) and pre-registered verdict '
                  '(|dAUROC| <= 0.01 and |dAP| <= 0.005, both labels, 3-seed means)', '',
                  '| arm | seeds | nn dAUROC | nn dAP | nn dtop-1% | nt dAUROC | nt dAP | nt dtop-1% | dFVE | d eval L0 | pass |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for arm in ('c448', 'btk16', 'btk8', 'dep_mice'):
            ns = [n for n in HARNESS + ['dep_mice'] if (n.split('_s')[0] == arm or n == arm)
                  and f'{n}_ema' in E and f'{n}_star' in E]
            if not ns:
                continue
            dl = {b: {q: [E[f'{n}_star']['usable_m4096'][b][q] - E[f'{n}_ema']['usable_m4096'][b][q] for n in ns]
                      for q in ('cf_auroc', 'cf_ap', 'cf_top1')} for b in LABELS}
            dfve = [E[f'{n}_star']['stats']['fve'] - E[f'{n}_ema']['stats']['fve'] for n in ns]
            dl0 = [E[f'{n}_star']['stats']['l0'] - E[f'{n}_ema']['stats']['l0'] for n in ns]
            ok = all(abs(np.mean(dl[b]['cf_auroc'])) <= 0.01 and abs(np.mean(dl[b]['cf_ap'])) <= 0.005 for b in LABELS)
            V[arm] = {'n_seeds': len(ns), 'pass': bool(ok), 'deltas': dl, 'd_fve': dfve, 'd_eval_l0': dl0,
                      'max_abs_seed_d_auroc': float(max(abs(v) for b in LABELS for v in dl[b]['cf_auroc'])),
                      'max_abs_seed_d_ap': float(max(abs(v) for b in LABELS for v in dl[b]['cf_ap']))}
            lines.append(f"| {arm} | {len(ns)} | " + ' | '.join(
                f"{ms(dl[b]['cf_auroc'])} | {ms(dl[b]['cf_ap'])} | {ms(dl[b]['cf_top1'], 3)}" for b in LABELS) +
                f" | {ms(dfve)} | {ms(dl0, 3)} | {ok} |")
        reg = [a for a in ('c448', 'btk16', 'btk8') if a in V]
        if reg:
            V['accept_theta_star'] = bool(len(reg) == 3 and all(V[a]['pass'] for a in reg))
            lines += ['', f"VERDICT (pre-registered arms c448, btk16, btk8): theta* acceptable = {V['accept_theta_star']}"
                      f" (dep_mice reported, not part of the rule)"]
    S['verdict'] = V
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['selftest', 'run', 'summary'])
    ap.add_argument('--domains', nargs='+', default=['mice', 'ants', 'frogs'])
    ap.add_argument('--no-harness', action='store_true')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=14)
    args = ap.parse_args()
    {'selftest': cmd_selftest, 'run': cmd_run, 'summary': cmd_summary}[args.step](args)


if __name__ == '__main__':
    main()
