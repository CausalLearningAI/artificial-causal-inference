"""
ECI v2.1 independent benchmark check (AMENDMENT 4 wrap-up, task 5c): does temporal context in the SAE input improve
neuron-behaviour alignment over the static input on CalMS21, where nothing in v2.1 was tuned?

==================================================================================================================
PRE-REGISTRATION (written 2026-11-02 BEFORE any CalMS21 v2.1 SAE, code or score existed).
DATA: the v2.1 store of CalMS21 (eci_tokens/v2_full/cal/v2.1_m5; DINOv3-L/16 block 20, 1024 canvas, 5 fps = source
    frame k x 6, offset 0 (calms_common.offset()); flag bit0 = mask v5 M_t; U_t = union of M over t-2..t+2). Task-1
    videos: 70 train (label-free SAE data), 19 test (scoring only). Verified before writing: every store video has
    ceil(N / 6) frames.
ARMS (as v21_train.py; x = the token at an M_t position, neighbours at the same position, same edge rule):
    S x; V x + d1, d1 = (x_{t+1} - x_{t-1}) / 2; W x + d1 + (s_t - mean(s)), s_t = per-dimension std over t-2..t+2,
    mean(s) = its mean over the training sample.
SAE: the frozen v2.0 recipe through src/eci/v2/sae.train (v20.R.spec()['sae']: Matryoshka BatchTopK 2048, k16,
    prefixes 128/256/512/2048, 30k steps, TokenNorm). Train videos split by rng 0 into 58 train / 6 calib / 6 eval;
    up to 8M train, 1M calib, 1M eval M_t tokens (uniform subsample, rng 0). Seeds 2001, 2002, 2003.
CODES: frame max over the M_t tokens at every 5 fps frame of the 19 test videos (frames without a token = 0).
LABELS: task1_test.npz labels at source frame k x 6 (attack / investigation / mount; one-vs-rest; 'other' unused).
READOUTS: raw frame max, and P(on) = per-neuron 3-state HMM (src/eci/v2/hmm.py) fitted label-free on the test spans
    (as v21_train.score). PRIMARY = P(on).
METRIC: cross-fitted best single neuron. Video halves = scripts/benchmark/calms21/eci_train.halves(E, s, labels),
    s = 0..4 (balances the positive videos of each behaviour across halves), both directions (10 directions). Neuron
    eligibility: fires on >= 0.2% of the behaviour's frames (eval.py rule). Per direction: AUROC of the AUROC-picked
    neuron, nAP = (AP - base) / (1 - base) of the nAP-picked neuron, top-1% precision of the AUROC-picked neuron (cut
    ceil(1%), fixed random tie-break rng 0); mean over directions; reported per behaviour, as the mean over the three
    behaviours, and the mean over investigation + mount (attack has only 5 positive test videos: flagged).
BOOTSTRAP: paired over the 19 test videos, 2000 replicates (rng 20261102), picks fixed at the full-data picks; an
    arm = mean over its seeds (primary: seeds fixed; secondary: seeds also resampled). Contrasts V - S, W - S, W - V on
    mean AUROC (P(on) primary, raw reported), 95% percentile CI.
DECISION: 'temporal supported on CalMS21' for an arm (V or W) iff its mean AUROC (P(on), three behaviours) minus S has
    a 95% CI excluding 0 on the positive side. Reported, not a gate.
==================================================================================================================
Steps: prep (GPU) | train --arm X --seed s (GPU) | codes (GPU) | score (CPU)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
import v21_train as T  # noqa: E402  (store access, feature kernels; sys.path / env set there)

sys.path.insert(0, str(REPO / 'scripts/benchmark/calms21'))
DOM = 'cal'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_1/bench_cal')
OUT = REPO / 'results/vision/eci_v2_1/bench'
ARMS, SEEDS = ('S', 'V', 'W'), (2001, 2002, 2003)
LABS = ['attack', 'investigation', 'mount']
CAP = {'train': 8_000_000, 'calib': 1_000_000, 'eval': 1_000_000}
NB, BSEED = 2000, 20261102
log = T.log


def video_lists():
    import calms_common as CC
    L = {}
    for sp in ('train', 'test'):
        d = np.load(CC.DER / f'task1_{sp}.npz', allow_pickle=True)
        L[sp] = [str(x) for x in d['names']]
    tr = list(np.random.default_rng(0).permutation(sorted(L['train'])))
    return {'calib': tr[:6], 'eval': tr[6:12], 'train': tr[12:]}, L['test']


def feats(V_, b0, b1, dev):
    """x, d1, s (per-dim 5-frame std) for the M_t tokens of frames b0..b1-1 and their frame index."""
    import torch
    W = V_.window(b0, b1)
    c, nb = T.centre_rows(W, 1)
    if not len(c):
        return None
    x, cd, rs, cnt = T.raw_feats(W['tok'], c, nb, dev)
    acc = torch.zeros_like(x)
    for r, m in rs:
        acc += (r ** 2) * m[:, None]
    return x, cd, torch.sqrt(acc / cnt[:, None]), W['fi'][c]


def cmd_prep(a):
    import torch
    import shutil
    dev = torch.device('cuda')
    roles, _ = video_lists()
    rng = np.random.default_rng(0)
    od = BEE / 'data'
    od.mkdir(parents=True, exist_ok=True)
    loc = Path(os.environ['LOCAL_DIR'])
    info, t0 = {}, time.time()
    for role, vids in roles.items():
        Vs = [T.Video(DOM, DOM, v) for v in vids]
        avail = sum(int(((V_.flag & 1) > 0).sum()) for V_ in Vs)
        p = min(1.0, 1.05 * CAP[role] / avail)
        parts = {'x': [], 'd': [], 's': []}
        for V_ in Vs:
            for b0 in range(0, len(V_.k), 600):
                f = feats(V_, b0, min(len(V_.k), b0 + 600), dev)
                if f is None:
                    continue
                keep = torch.from_numpy(rng.random(len(f[0])) < p).to(dev)
                for nm, z in zip('xds', f[:3]):
                    parts[nm].append(z[keep].half().cpu())
        n = sum(len(z) for z in parts['x'])
        sel = np.sort(rng.choice(n, min(n, CAP[role]), replace=False))
        info[role] = {'videos': len(vids), 'M_tokens_available': avail, 'used': int(len(sel))}
        for nm in 'xds':
            X = torch.cat(parts[nm])[torch.from_numpy(sel)]
            parts[nm] = None
            if role == 'train' and nm == 's':
                torch.save({'mean_s': X.float().mean(0)}, od / 'w_stats.pt')
            np.save(loc / f'{role}_{nm}.npy', X.numpy())
            shutil.copyfile(loc / f'{role}_{nm}.npy', od / f'{role}_{nm}.npy')
            (loc / f'{role}_{nm}.npy').unlink()
        log(f'prep {role}: {json.dumps(info[role])} ({time.time() - t0:.0f}s)')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cal_prep.json').write_text(json.dumps({'roles': roles, 'tokens': info}, indent=1))


def arm_x(arm, x, d, s, mean_s):
    if arm == 'S':
        return x
    if arm == 'V':
        return x + d
    return x + d + (s - mean_s)


def cmd_train(a):
    import torch
    import v20 as V
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    od = BEE / 'data'
    ms = torch.load(od / 'w_stats.pt')['mean_s']
    X = {}
    for r in CAP:
        x, d, s = (torch.from_numpy(np.load(od / f'{r}_{nm}.npy')) for nm in 'xds')
        X[r] = arm_x(a.arm, x.float(), d.float(), s.float(), ms).half() if a.arm != 'S' else x
        del x, d, s
    cfg = {**V.R.spec()['sae'], 'context_window': 0, 'arm': a.arm, 'store': T.VER, 'bench': 'calms21'}
    rec = SA.train(cfg, a.seed, V.HostTok(X['train'], dev), X['calib'], X['eval'], BEE / 'sae' / f'{a.arm}_s{a.seed}', dev)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'cal_train_{a.arm}_s{a.seed}.json').write_text(json.dumps(rec, indent=1, default=float))
    log(f'{a.arm} s{a.seed}: eval FVE {rec["eval_fve"]:.5f} L0 {rec["eval_l0"]:.2f}')


def cmd_codes(a):
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    _, test = video_lists()
    ms = torch.load(BEE / 'data' / 'w_stats.pt')['mean_s'].to(dev)
    saes = {f'{arm}_s{s}': SA.load(BEE / 'sae' / f'{arm}_s{s}' / 'sae_theta.pt', dev) for arm in ARMS for s in SEEDS}
    rows = []
    Vs = {v: T.Video(DOM, DOM, v) for v in sorted(test)}
    for v, V_ in Vs.items():
        rows += [(v, int(k)) for k in V_.k]
    R = pd.DataFrame(rows, columns=['obs', 'k'])
    Xc = {k: np.zeros((len(R), 2048), np.float16) for k in saes}
    base, t0 = 0, time.time()
    for v, V_ in Vs.items():
        n = len(V_.k)
        for b0 in range(0, n, 600):
            b1 = min(n, b0 + 600)
            f = feats(V_, b0, b1, dev)
            if f is None:
                continue
            x, d, s, fi = f
            fl = torch.from_numpy(fi - b0).to(dev)
            for k, (sae, norm) in saes.items():
                with torch.no_grad():
                    cc = sae.encode(norm(arm_x(k.split('_')[0], x, d, s, ms)), mode='threshold').float()
                m = torch.zeros((b1 - b0, 2048), device=dev).index_reduce_(0, fl, cc, 'amax', include_self=True)
                Xc[k][base + b0: base + b1] = m.half().cpu().numpy()
        base += n
        log(f'codes {v}: {n} frames ({time.time() - t0:.0f}s)')
    od = BEE / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    R.to_parquet(od / 'rows.parquet')
    for k, x in Xc.items():
        np.save(od / f'{k}.npy', x)


# ---------------------------------------------------------------------------------------------- score
def labels(R):
    import calms_common as CC
    d = np.load(CC.DER / 'task1_test.npz', allow_pickle=True)
    st = np.r_[0, np.cumsum(d['lens'])]
    pos = {str(nm): (st[j], int(d['lens'][j])) for j, nm in enumerate(d['names'])}
    E = R.copy()
    E['video'] = E.obs
    for b in LABS:
        E[b] = False
    for v, g in E.groupby('obs', sort=False):
        s0, n = pos[v]
        src = g.k.values * 6 + CC.offset()
        assert src.max() < n and len(g) == -(-n // 6), v
        lab = d['labels'][s0 + src]
        for b in LABS:
            E.loc[g.index, b] = lab == CC.TASK1.index(b)
    return E


def score_model(X, E, H, starts_all):
    """cross-fitted best single neuron per behaviour -> {b: {AUROC, nAP, top1, picks [(dir, jA, jN, heldout idx)]}}"""
    import analysis as AN
    out = {}
    vcode = pd.factorize(E.obs.values)[0].astype(np.int64)
    nvid = int(vcode.max()) + 1
    rng = np.random.default_rng(0)
    for b in LABS:
        y = E[b].values.astype(bool)
        tb = rng.random(len(y))
        elig = np.flatnonzero((X > 0).mean(0) >= AN.ELIG)
        Xc = np.asfortranarray(X[:, elig])
        per = {}
        for sn, h_ in H.items():
            for h in (0, 1):
                sel = np.flatnonzero(h_ == h)
                if y[sel].sum() == 0 or y[sel].all():
                    per[(sn, h)] = None
                    continue
                _, st_h = AN.sub_starts(h_ == h, starts_all)
                per[(sn, h)] = AN.metrics_cols(Xc, sel.astype(np.int64), y[sel], st_h, vcode[sel], nvid)[0]
        A, N, P1, picks = [], [], [], []
        for sn, h_ in H.items():
            for h in (0, 1):
                a_, b_ = per[(sn, h)], per[(sn, 1 - h)]
                if a_ is None or b_ is None:
                    continue
                qa, qn = int(np.nanargmax(a_[:, 2])), int(np.nanargmax(a_[:, 1]))
                A.append(float(b_[qa, 2]))
                N.append(float(b_[qn, 1]))
                o = np.flatnonzero(h_ == 1 - h)
                c = X[o, elig[qa]].astype(np.float64)
                n = max(1, int(np.ceil(0.01 * len(o))))
                P1.append(float(y[o][np.lexsort((tb[o], -c))[:n]].mean()))
                picks.append((sn, h, int(elig[qa]), int(elig[qn])))
        out[b] = {'AUROC': float(np.mean(A)), 'nAP': float(np.mean(N)), 'top1_prec': float(np.mean(P1)),
                  'base': float(y.mean()), 'n_dirs': len(A), 'picks': picks, 'tb': tb}
    return out


def cmd_score(a):
    from multiprocessing import get_context
    import analysis as AN
    import importlib.util
    from src.eci.v2 import hmm as HM
    import v20 as V20
    sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
    from v21_ladder import wmetrics
    sp = importlib.util.spec_from_file_location('calms_train', REPO / 'scripts/benchmark/calms21/eci_train.py')
    M = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(M)
    od = BEE / 'codes'
    R = pd.read_parquet(od / 'rows.parquet')
    E = labels(R)
    H = {f's{s}': M.halves(E, s, LABS) for s in range(5)}
    st = np.r_[0, np.flatnonzero(E.obs.values[1:] != E.obs.values[:-1]) + 1, len(E)].astype(np.int64)
    units = np.array(sorted(E.obs.unique()))
    u = pd.Series(np.arange(len(units)), index=units).reindex(E.obs.values).values.astype(np.int64)
    rng = np.random.default_rng(BSEED)
    Wb = np.stack([np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)) for _ in range(NB)])
    Wt = np.concatenate([np.ones((len(units), 1)), Wb.T.astype(np.float64)], 1)
    res, boot = {}, {}
    for arm in ARMS:
        for s in SEEDS:
            k = f'{arm}_s{s}'
            raw = np.load(od / f'{k}.npy').astype(np.float32)
            Xt = np.ascontiguousarray(raw.T)
            V20._G.update(Xc=Xt, sc=st)
            with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
                prm = [None] * 2048
                for j, p in pool.imap_unordered(V20._fit_one, range(2048), chunksize=16):
                    prm[j] = p
            pon = np.zeros_like(raw)
            for j in range(2048):
                pon[:, j] = HM.apply(Xt[j], st, prm[j])[0]
            for rd, X in (('pon', pon), ('raw', raw)):
                sc = score_model(X, E, H, st)
                bb = {}
                for b, r in sc.items():
                    y = E[b].values.astype(bool)
                    acc = []
                    for sn, h, ja, jn in r['picks']:
                        o = np.flatnonzero(H[sn] == 1 - h)
                        cnt = np.bincount(u[o], minlength=len(units)).astype(np.float64)
                        m = wmetrics(X[o, ja], y[o], u[o], r['tb'][o], Wt, cnt)
                        if jn != ja:
                            m[1] = wmetrics(X[o, jn], y[o], u[o], r['tb'][o], Wt, cnt)[1]
                        acc.append(m)
                    bb[b] = np.nanmean(np.stack(acc), 0)  # (3, NB + 1)
                    for i, Mn in enumerate(('AUROC', 'nAP', 'top1_prec')):
                        assert abs(bb[b][i, 0] - r[Mn]) < 1e-6, ('boot identity', k, rd, b, Mn, bb[b][i, 0], r[Mn])
                    r.pop('tb')
                res.setdefault(k, {})[rd] = sc
                boot.setdefault(k, {})[rd] = bb
            log(f'{k}: ' + ', '.join(f'{rd} mean AUROC {np.mean([res[k][rd][b]["AUROC"] for b in LABS]):.4f}' for rd in ('pon', 'raw')))
    np.savez_compressed(BEE / 'boot.npz', **{f'{k}|{rd}|{b}': m for k, d in boot.items() for rd, bs in d.items() for b, m in bs.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cal_scores.json').write_text(json.dumps(res, default=float))
    report(res, boot)


def report(res, boot):
    rng = np.random.default_rng(BSEED + 1)
    fve = {k: json.loads((OUT / f'cal_train_{k}.json').read_text())['eval_fve'] for k in res}
    J = {'per_model': {}, 'arms': {}, 'contrasts': {}}
    mr = lambda v: f'{np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]'  # noqa: E731
    L = ['## CalMS21 (19 test videos; median [range] over seeds 2001-2003; P(on) primary)', '',
         '| arm | mean AUROC P(on) | mean AUROC raw | mean AUROC inv+mount P(on) | mean nAP P(on) | mean top-1% prec P(on) | '
         'AUROC attack P(on) | AUROC investigation P(on) | AUROC mount P(on) | SAE eval FVE |', '|' + '---|' * 10]
    for arm in ARMS:
        ks = [f'{arm}_s{s}' for s in SEEDS]
        g = lambda rd, M, bs=LABS: [float(np.mean([res[k][rd][b][M] for b in bs])) for k in ks]  # noqa: E731
        row = {'auroc_pon': g('pon', 'AUROC'), 'auroc_raw': g('raw', 'AUROC'), 'auroc_im_pon': g('pon', 'AUROC', LABS[1:]),
               'nap_pon': g('pon', 'nAP'), 'top1_pon': g('pon', 'top1_prec'), 'nap_raw': g('raw', 'nAP'),
               'top1_raw': g('raw', 'top1_prec'), 'fve': [fve[k] for k in ks],
               **{f'auroc_{b}_pon': g('pon', 'AUROC', [b]) for b in LABS}}
        J['arms'][arm] = row
        L.append(f'| {arm} | {mr(row["auroc_pon"])} | {mr(row["auroc_raw"])} | {mr(row["auroc_im_pon"])} | {mr(row["nap_pon"])} | '
                 f'{mr(row["top1_pon"])} | {mr(row["auroc_attack_pon"])} | {mr(row["auroc_investigation_pon"])} | '
                 f'{mr(row["auroc_mount_pon"])} | {mr(row["fve"])} |')
    L += ['', '| contrast | readout | labels | metric | point | 95% CI (seeds fixed) | 95% CI (seeds resampled) | verdict |',
          '|---|---|---|---|---|---|---|---|']
    for a1, a0 in (('V', 'S'), ('W', 'S'), ('W', 'V')):
        for rd in ('pon', 'raw'):
            for nm, bs in (('all 3', LABS), ('inv+mount', LABS[1:])):
                for i, Mn in enumerate(('AUROC', 'nAP', 'top1_prec')):
                    arr = {arm: np.stack([np.mean([boot[f'{arm}_s{s}'][rd][b][i] for b in bs], 0) for s in SEEDS])
                           for arm in (a1, a0)}  # (seeds, NB + 1)
                    d_fix = arr[a1][:, 1:].mean(0) - arr[a0][:, 1:].mean(0)
                    pt = float(arr[a1][:, 0].mean() - arr[a0][:, 0].mean())
                    i1 = rng.integers(0, len(SEEDS), (NB, len(SEEDS)))
                    i0 = rng.integers(0, len(SEEDS), (NB, len(SEEDS)))
                    cols = np.arange(1, NB + 1)
                    d_rs = arr[a1][i1, cols[:, None]].mean(1) - arr[a0][i0, cols[:, None]].mean(1)
                    c1 = [float(x) for x in np.nanpercentile(d_fix, [2.5, 97.5])]
                    c2 = [float(x) for x in np.nanpercentile(d_rs, [2.5, 97.5])]
                    v = ''
                    if Mn == 'AUROC' and rd == 'pon' and nm == 'all 3' and a0 == 'S':
                        v = f'temporal {"SUPPORTED" if c1[0] > 0 else "NOT supported"} ({a1})'
                    J['contrasts'][f'{a1}-{a0}|{rd}|{nm}|{Mn}'] = {'point': pt, 'ci_seeds_fixed': c1, 'ci_seeds_resampled': c2}
                    L.append(f'| {a1} - {a0} | {rd} | {nm} | mean {Mn} | {pt:+.4f} | [{c1[0]:+.4f}, {c1[1]:+.4f}] | '
                             f'[{c2[0]:+.4f}, {c2[1]:+.4f}] | {v} |')
    L += ['', 'Attack is positive in only 5 of the 19 test videos (mouse075-079); the inv+mount rows exclude it.',
          f'Bootstrap: {NB} replicates over test videos (rng {BSEED}), neuron picks fixed at the full-data picks.']
    (OUT / 'cal_report.json').write_text(json.dumps(J, indent=1, default=float))
    (OUT / 'cal_report.md').write_text('\n'.join(L))
    print('\n'.join(L))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('prep', 'train', 'codes', 'score'))
    p.add_argument('--arm', default='S')
    p.add_argument('--seed', type=int, default=2001)
    a = p.parse_args()
    {'prep': cmd_prep, 'train': cmd_train, 'codes': cmd_codes, 'score': cmd_score}[a.step](a)


if __name__ == '__main__':
    main()
