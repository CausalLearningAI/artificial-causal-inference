"""
ECI v2.10: PRED-contact (PREDk16) vs recipe V on the validation benchmarks (pre-registration scripts/eci/v2/v210_prereg.md).
Sets: cal (CalMS21), mab (MABe22 mouse triplets), crim (CRIM13); frogs = health + codes only.
Recipe V on cal = the v2.1 bench SAEs / codes (eci_codes/v2_1/bench_cal, reused unchanged); on mab / crim = the v2.1 bench
code path (prep / train copied from v21_bench.py with the set as a parameter), seeds 2001-2003.
Steps:
  roles  --dom D             (CPU) write the pool / scored video lists -> OUT/roles_<D>.json
  prep   --dom D             (GPU) V token arrays (x, d1, s) of the pool roles (mab / crim)
  trainV --dom D             (GPU) recipe V, seeds 2001-2003 (mab / crim)
  codes  --dom D             (GPU) frame-max codes of every scored frame (V on mab / crim, PREDk16 s3001-3003 everywhere;
                             frogs: all frames of all videos) + PREDk health on V's held-out eval tokens
  score  --dom D --key K     (CPU) P(on) HMM + cross-fitted best-neuron metrics -> OUT/bench/<D>_score_<K>.json
  report                     (CPU) tables.md, summary.json
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
import v21_train as T  # noqa: E402  (store access; sys.path / env set there)
import v21_bench as B21  # noqa: E402  (v2.1 CalMS21 bench: roles, features, labels, scorer)

BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_10')
OUT = REPO / 'results/vision/eci_v2_10'
BENCH = ('cal', 'mab', 'crim')
SEEDS_V, SEEDS_P = (2001, 2002, 2003), (3001, 3002, 3003)
KEYS_P = [f'PREDk16_s{s}' for s in SEEDS_P]
KEYS_V = [f'V_s{s}' for s in SEEDS_V]
STEP = {'cal': 6, 'mab': 6, 'crim': 5}
NSPLIT = 5
CAP = B21.CAP
log = T.log


def manifest(dom):
    m = json.loads((T.STORE / dom / T.VER / 'manifest.json').read_text())
    assert not m['missing_videos'] and m['videos'] == m['expected_videos'] == len(m['videos_list'])
    return [(v['series'], v['obs'], int(v['frames'])) for v in m['videos_list']]


def mab_ec():
    sys.path.insert(0, str(REPO / 'scripts/benchmark/mabe22'))
    import eci_common as EC
    return EC


def crim_ac():
    sys.path.insert(0, str(REPO / 'scripts/benchmark/crim13'))
    import align_common as AC
    return AC


def _split_pool(pool):
    p = list(np.random.default_rng(0).permutation(sorted(pool)))
    nc = int(round(0.1 * len(p)))
    return {'calib': p[:nc], 'eval': p[nc:2 * nc], 'train': p[2 * nc:]}


def roles(dom):
    """-> {'train', 'calib', 'eval', 'scored'}: lists of obs (series = dom for the benchmarks); frogs: train = [(ser, obs)]."""
    if dom == 'frogs':
        return {'train': [(s, o) for s, o, _ in manifest(dom)], 'calib': [], 'eval': [], 'scored': []}
    obs = [o for _, o, _ in manifest(dom)]
    if dom == 'cal':
        r, test = B21.video_lists()
        assert set(test) | set(sum(r.values(), [])) == set(obs)
        return {**r, 'scored': sorted(test)}
    if dom == 'mab':
        fm = np.load(mab_ec().RAW / 'mouse_frame_number_map.npy', allow_pickle=True).item()
        pool = [o for o in obs if o not in fm]
        return {**_split_pool(pool), 'scored': sorted(o for o in obs if o in fm)}
    inv = json.loads(crim_ac().INV.read_text())['videos']
    half = {v['name']: v['split'] for v in inv}
    pool = [o for o in obs if half[o] == 'train']
    return {**_split_pool(pool), 'scored': sorted(obs)}


def nframes(dom):
    return {o: n for _, o, n in manifest(dom)}


def vdir(dom):
    return Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_1/bench_cal') if dom == 'cal' else BEE / dom / 'V'


def cmd_roles(a):
    r = roles(a.dom)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'roles_{a.dom}.json').write_text(json.dumps({k: [list(x) if isinstance(x, tuple) else x for x in v] for k, v in r.items()}, indent=1))
    log(f'roles {a.dom}: ' + ', '.join(f'{k} {len(v)}' for k, v in r.items()))


# ---------------------------------------------------------------------------------------------- V (= v21_bench prep / train)
def cmd_prep(a):
    """= v21_bench.cmd_prep with the set as a parameter (roles from roles(dom))."""
    import torch
    import shutil
    dom = a.dom
    assert dom in ('mab', 'crim')
    dev = torch.device('cuda')
    R = roles(dom)
    rl = {k: R[k] for k in ('calib', 'eval', 'train')}
    rng = np.random.default_rng(0)
    od = vdir(dom) / 'data'
    od.mkdir(parents=True, exist_ok=True)
    loc = Path(os.environ['LOCAL_DIR'])
    info, t0 = {}, time.time()
    for role, vids in rl.items():
        Vs = [T.Video(dom, dom, v) for v in vids]
        avail = sum(int(((V_.flag & 1) > 0).sum()) for V_ in Vs)
        p = min(1.0, 1.05 * CAP[role] / avail)
        parts = {'x': [], 'd': [], 's': []}
        for V_ in Vs:
            for b0 in range(0, len(V_.k), 600):
                f = B21.feats(V_, b0, min(len(V_.k), b0 + 600), dev)
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
        log(f'prep {dom} {role}: {json.dumps(info[role])} ({time.time() - t0:.0f}s)')
    (OUT / 'bench').mkdir(parents=True, exist_ok=True)
    (OUT / 'bench' / f'{dom}_prepV.json').write_text(json.dumps({'roles': rl, 'tokens': info}, indent=1))


def cmd_trainV(a):
    """= v21_bench.cmd_train (arm V), seeds 2001-2003 in one job."""
    import torch
    import v20 as V
    from src.eci.v2 import sae as SA
    dom = a.dom
    assert dom in ('mab', 'crim')
    dev = torch.device('cuda')
    od = vdir(dom) / 'data'
    ms = torch.load(od / 'w_stats.pt')['mean_s']
    X = {}
    for r in CAP:
        x, d, s = (torch.from_numpy(np.load(od / f'{r}_{nm}.npy')) for nm in 'xds')
        X[r] = B21.arm_x('V', x.float(), d.float(), s.float(), ms).half()
        del x, d, s
    for seed in SEEDS_V:
        cfg = {**V.R.spec()['sae'], 'context_window': 0, 'arm': 'V', 'store': T.VER, 'bench': dom}
        rec = SA.train(cfg, seed, V.HostTok(X['train'], dev), X['calib'], X['eval'], vdir(dom) / 'sae' / f'V_s{seed}', dev)
        (OUT / 'bench').mkdir(parents=True, exist_ok=True)
        (OUT / 'bench' / f'{dom}_trainV_s{seed}.json').write_text(json.dumps(rec, indent=1, default=float))
        log(f'{dom} V s{seed}: eval FVE {rec["eval_fve"]:.5f} L0 {rec["eval_l0"]:.2f} steps/s {rec.get("steps_per_s", float("nan")):.2f}')


# ---------------------------------------------------------------------------------------------- codes
def x_only(V_, b0, b1, dev):
    """M_t tokens of frames b0..b1-1 (same rows as v21_bench.feats) -> x float32, frame index."""
    import torch
    W = V_.window(b0, b1)
    c = np.flatnonzero(((W['flag'] & 1) > 0) & (W['fi'] >= W['i0']) & (W['fi'] < W['i1']))
    if not len(c):
        return None
    return torch.from_numpy(W['tok'][c]).to(dev).float(), W['fi'][c]


def cmd_codes(a):
    import torch
    from src.eci.v2 import sae as SA
    from src.eci.sae_calibrate import sparsity_stats
    dom = a.dom
    dev = torch.device('cuda')
    keysV = KEYS_V if dom in ('mab', 'crim') else []
    saes = {k: SA.load(vdir(dom) / 'sae' / k / 'sae_theta.pt', dev) for k in keysV}
    saes.update({k: SA.load(BEE / dom / 'sae' / k / 'sae_theta.pt', dev) for k in KEYS_P})
    wid = {k: s[0].n_latents for k, s in saes.items()}
    od = BEE / dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    # health of PREDk on V's held-out eval tokens (identical tokens to V's eval FVE)
    if dom in BENCH:
        xe = torch.from_numpy(np.load(vdir(dom) / 'data' / 'eval_x.npy'))
        for k in KEYS_P:
            sae, norm = saes[k]
            th = float(sae.threshold)
            st = sparsity_stats(sae, xe, th, norm=norm)
            fired = torch.zeros(sae.n_latents, dtype=torch.long, device=dev)
            with torch.no_grad():
                for b0 in range(0, len(xe), 65536):
                    fired += (sae.encode(norm(xe[b0:b0 + 65536].to(dev).float()), mode='threshold') > 0).sum(0)
            h = {'key': k, 'tokens': int(len(xe)), 'tokens_source': f'{vdir(dom)}/data/eval_x.npy (V eval role)', 'theta_star': th,
                 'fve': float(st['fve']), 'l0': float(st['l0']), 'never_fire_frac': float((fired == 0).float().mean())}
            (OUT / 'bench').mkdir(parents=True, exist_ok=True)
            (OUT / 'bench' / f'{dom}_health_{k}.json').write_text(json.dumps(h, indent=1))
            log(f'health {dom} {k}: {h}')
        del xe
    if dom == 'frogs':
        vids = [(s, o) for s, o, _ in manifest(dom)]
    else:
        vids = [(dom, o) for o in roles(dom)['scored']]
    Vs = [(s, o, T.Video(dom, s, o)) for s, o in vids]
    rows = []
    for s, o, V_ in Vs:
        rows += [(s, o, int(k)) for k in V_.k]
    R = pd.DataFrame(rows, columns=['series', 'obs', 'k'])
    if dom == 'cal':  # same row order as the v2.1 V codes
        R0 = pd.read_parquet(vdir('cal') / 'codes' / 'rows.parquet')
        assert R0.obs.tolist() == R.obs.tolist() and R0.k.tolist() == R.k.tolist(), 'cal rows != v2.1 rows'
    R.to_parquet(od / 'rows.parquet')
    mm = {k: np.lib.format.open_memmap(od / f'{k}.npy.part', 'w+', np.float16, (len(R), wid[k])) for k in saes}
    base, t0, ntok = 0, time.time(), 0
    for vi, (s, o, V_) in enumerate(Vs):
        n = len(V_.k)
        for b0 in range(0, n, 600):
            b1 = min(n, b0 + 600)
            if keysV:
                f = B21.feats(V_, b0, b1, dev)
                if f is None:
                    continue
                x, d, sd, fi = f
            else:
                f = x_only(V_, b0, b1, dev)
                if f is None:
                    continue
                x, fi = f
            ntok += len(x)
            fl = torch.from_numpy(fi - b0).to(dev)
            for k, (sae, norm) in saes.items():
                inp = B21.arm_x('V', x, d, sd, None) if k.startswith('V_') else x
                m = torch.zeros((b1 - b0, wid[k]), device=dev)
                with torch.no_grad():
                    for c0 in range(0, len(inp), 65536):
                        cc = sae.encode(norm(inp[c0:c0 + 65536]), mode='threshold').float()
                        m.index_reduce_(0, fl[c0:c0 + 65536], cc, 'amax', include_self=True)
                mm[k][base + b0: base + b1] = m.half().cpu().numpy()
        base += n
        if vi % 50 == 0 or vi == len(Vs) - 1:
            log(f'codes {dom} {vi + 1}/{len(Vs)} videos, {base:,} frames, {ntok:,} tokens ({time.time() - t0:.0f}s)')
    for k, m in mm.items():
        m.flush()
        del m
        os.replace(od / f'{k}.npy.part', od / f'{k}.npy')
    (OUT / 'bench').mkdir(parents=True, exist_ok=True)
    (OUT / 'bench' / f'{dom}_codes.json').write_text(json.dumps({'dom': dom, 'videos': len(Vs), 'frames': len(R), 'tokens': ntok,
                                                                 'keys': list(saes), 'path': str(od), 'time_s': time.time() - t0}, indent=1))
    log(f'codes {dom} done: {len(R):,} frames, keys {list(saes)}')


# ---------------------------------------------------------------------------------------------- labels / halves
def frame_table(dom, R):
    """-> E (rows of R that are scored: video, obs, k, label columns), valid (len(R),) bool, labs, halves dict."""
    import importlib.util
    if dom == 'cal':
        E = B21.labels(R)
        valid = np.ones(len(R), bool)
        labs = list(B21.LABS)
        sp = importlib.util.spec_from_file_location('calms_train', REPO / 'scripts/benchmark/calms21/eci_train.py')
        M = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(M)
        return E, valid, labs, {f's{s}': M.halves(E, s, labs) for s in range(NSPLIT)}
    if dom == 'mab':
        EC = mab_ec()
        sub = np.load(EC.RAW / 'mouse_submission_labels.npy', allow_pickle=True).item()
        fm = np.load(EC.RAW / 'mouse_frame_number_map.npy', allow_pickle=True).item()
        voc = list(sub['vocabulary'])
        L = sub['label_array']
        off = EC.offset()
        assert off == 1
        E = R.copy()
        E['video'] = E.obs
        i = E.k.values * 6 - off
        a0 = np.array([fm[o][0] for o in E.obs.values])
        a1 = np.array([fm[o][1] for o in E.obs.values])
        valid = (i >= 0) & (i < a1 - a0)
        for b in EC.BEHAVIOURS:
            yb = np.zeros(len(E), bool)
            yb[valid] = L[voc.index(b), (a0 + i)[valid]] == 1
            E[b] = yb
        E = E[valid].reset_index(drop=True)
        E['clip'] = E.video
        sp = importlib.util.spec_from_file_location('mabe_train', REPO / 'scripts/benchmark/mabe22/eci_train.py')
        M = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(M)
        return E, valid, list(EC.PRIMARY), {f's{s}': M.halves(E, s) for s in range(NSPLIT)}
    AC = crim_ac()
    inv = {v['name']: v for v in json.loads(AC.INV.read_text())['videos']}
    E = R.copy()
    E['video'] = E.obs
    lab = np.full(len(E), -1, np.int64)
    for o, g in E.groupby('obs', sort=False):
        n = int(inv[o]['frames'])
        lb = AC.read_labels(AC.ROOT / inv[o]['dir'] / f'{o}_t.txt', n)
        src = g.k.values * 5
        ok = src < n
        lab[g.index.values[ok]] = lb[src[ok]]
    valid = lab >= 0
    for b in AC.SCORED:
        E[b] = lab == AC.VOCAB.index(b)
    E['half'] = [inv[o]['split'] for o in E.obs.values]
    E = E[valid].reset_index(drop=True)
    return E, valid, list(AC.SCORED), {'released': (E.half.values == 'test').astype(np.int64)}


def starts_of(obs):
    return np.r_[0, np.flatnonzero(obs[1:] != obs[:-1]) + 1, len(obs)].astype(np.int64)


# ---------------------------------------------------------------------------------------------- score
def top3_video(c, y, vc, o):
    """precision@top3/video on held-out rows o: raw code c (n,), y bool, video code vc -> (precision, chance, n videos)."""
    pr, ch = [], []
    for v in np.unique(vc[o]):
        r = o[vc[o] == v]
        if not y[r].any():
            continue
        top = r[np.lexsort((r, -c[r].astype(np.float64)))[:3]]
        pr.append(float(y[top].mean()))
        ch.append(float(y[r].mean()))
    return (float(np.mean(pr)), float(np.mean(ch)), len(pr)) if pr else (float('nan'), float('nan'), 0)


def score_model(X, Xraw, E, H, starts_all, labs):
    """= v21_bench.score_model (AUROC of the AUROC pick, nAP of the nAP pick, top-1% of the AUROC pick) + AP of the AP
    pick (= nAP pick) and precision@top3/video of the AP pick on the raw code."""
    import analysis as AN
    out = {}
    vcode = pd.factorize(E.obs.values)[0].astype(np.int64)
    nvid = int(vcode.max()) + 1
    rng = np.random.default_rng(0)
    for b in labs:
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
        A, N, P1, AP, PI, T3, T3c, T3n, picks = [], [], [], [], [], [], [], [], []
        for sn, h_ in H.items():
            for h in (0, 1):
                a_, b_ = per[(sn, h)], per[(sn, 1 - h)]
                if a_ is None or b_ is None:
                    continue
                qa, qn = int(np.nanargmax(a_[:, 2])), int(np.nanargmax(a_[:, 1]))
                o = np.flatnonzero(h_ == 1 - h)
                pi = float(y[o].mean())
                A.append(float(b_[qa, 2]))
                N.append(float(b_[qn, 1]))
                AP.append(float(b_[qn, 1]) * (1 - pi) + pi)
                PI.append(pi)
                c = X[o, elig[qa]].astype(np.float64)
                n = max(1, int(np.ceil(0.01 * len(o))))
                P1.append(float(y[o][np.lexsort((tb[o], -c))[:n]].mean()))
                t3 = top3_video(Xraw[:, elig[qn]], y, vcode, o)
                T3.append(t3[0])
                T3c.append(t3[1])
                T3n.append(t3[2])
                picks.append((sn, h, int(elig[qa]), int(elig[qn])))
        out[b] = {'AUROC': float(np.mean(A)), 'nAP': float(np.mean(N)), 'top1_prec': float(np.mean(P1)), 'AP': float(np.mean(AP)),
                  'AP_chance_heldout_prev': float(np.mean(PI)), 'top3vid': float(np.nanmean(T3)), 'top3vid_chance': float(np.nanmean(T3c)),
                  'top3vid_videos_mean': float(np.mean(T3n)), 'base': float(y.mean()), 'n_pos': int(y.sum()), 'n_dirs': len(A),
                  'picks': picks, 'per_dir': {'AUROC': A, 'AP': AP, 'top3vid': T3}}
    return out


def cmd_score(a):
    from multiprocessing import get_context
    from src.eci.v2 import hmm as HM
    import v20 as V20
    dom, k = a.dom, a.key
    cdir = vdir(dom) / 'codes' if (dom == 'cal' and k.startswith('V_')) else BEE / dom / 'codes'
    R = pd.read_parquet(BEE / dom / 'codes' / 'rows.parquet')
    E, valid, labs, H = frame_table(dom, R)
    raw_all = np.load(cdir / f'{k}.npy').astype(np.float32)
    assert len(raw_all) == len(R)
    W = raw_all.shape[1]
    st_all = starts_of(R.obs.values)
    Xt = np.ascontiguousarray(raw_all.T)
    V20._G.update(Xc=Xt, sc=st_all)
    t0 = time.time()
    with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        prm = [None] * W
        for j, p in pool.imap_unordered(V20._fit_one, range(W), chunksize=16):
            prm[j] = p
    pon = np.zeros_like(raw_all)
    for j in range(W):
        pon[:, j] = HM.apply(Xt[j], st_all, prm[j])[0]
    del Xt
    log(f'{dom} {k}: HMM done ({time.time() - t0:.0f}s), {W} latents, {len(R):,} frames, {int(valid.sum()):,} scored')
    raw, pon = raw_all[valid], pon[valid]
    del raw_all
    st = starts_of(E.obs.values)
    res = {'dom': dom, 'key': k, 'frames_rows': len(R), 'frames_scored': int(valid.sum()), 'videos_scored': int(E.obs.nunique()),
           'labels': labs, 'halves': list(H)}
    for rd, X in (('pon', pon), ('raw', raw)):
        sc = score_model(X, raw, E, H, st, labs)
        res[rd] = sc
        log(f'{dom} {k} {rd}: ' + ', '.join(f'{b} AUROC {sc[b]["AUROC"]:.3f} AP {sc[b]["AP"]:.3f} (prev {sc[b]["base"]:.4f}) '
                                             f'top3/vid {sc[b]["top3vid"]:.3f}' for b in labs))
    if dom == 'cal' and k.startswith('V_'):  # scorer identity vs the stored v2.1 scores
        old = json.loads((B21.OUT / 'cal_scores.json').read_text())[k]
        bad = [(rd, b, m, res[rd][b][m], old[rd][b][m]) for rd in ('pon', 'raw') for b in labs for m in ('AUROC', 'nAP', 'top1_prec')
               if abs(res[rd][b][m] - old[rd][b][m]) > 1e-9]
        res['identity_vs_v21_cal_scores'] = {'checked': 2 * len(labs) * 3, 'mismatches': bad}
        log(f'IDENTITY vs v2.1 cal_scores.json: {len(bad)} mismatches {bad}')
        if bad:
            (OUT / 'bench').mkdir(parents=True, exist_ok=True)
            (OUT / 'bench' / f'{dom}_score_{k}.FAILED.json').write_text(json.dumps(res, default=float))
            raise SystemExit('SCORER IDENTITY FAILED')
    (OUT / 'bench').mkdir(parents=True, exist_ok=True)
    (OUT / 'bench' / f'{dom}_score_{k}.json').write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- report
NAME = {'cal': 'CalMS21', 'mab': 'MABe22 mouse triplets', 'crim': 'CRIM13'}
PRIM = (('AUROC', 'AUROC'), ('AP', 'AP'), ('top3vid', 'precision@top3/video'))


def cmd_report(a):
    S, Jall = {}, {'question': 'is PREDk16 at least as good as recipe V on single-neuron behaviour alignment (v210_prereg.md)',
                   'benchmarks': {}, 'health': {}, 'frogs': {}}
    L = ['# ECI v2.10 - PRED-contact (PREDk16) vs recipe V on the validation benchmarks', '',
         'Pre-registration: scripts/eci/v2/v210_prereg.md. Readout P(on) for AUROC / AP, raw code for precision@top3/video. '
         'Neuron picked on the selection half, measured on the other half, mean over directions. '
         'best = max over the 3 dictionaries of an arm; spread = the larger seed range (max - min) of the two arms; '
         'PASS = best PREDk >= best V - spread. v1 has no benchmark dictionaries (not compared).', '']
    for dom in BENCH:
        sc = {}
        for k in KEYS_V + KEYS_P:
            f = OUT / 'bench' / f'{dom}_score_{k}.json'
            if f.exists():
                sc[k] = json.loads(f.read_text())
        if len(sc) < 6:
            L += [f'## {NAME[dom]}: MISSING scores ({sorted(set(KEYS_V + KEYS_P) - set(sc))})', '']
            continue
        any_ = next(iter(sc.values()))
        labs = any_['labels']
        J = {'frames_scored': any_['frames_scored'], 'videos_scored': any_['videos_scored'], 'behaviours': {}}
        L += [f'## {NAME[dom]} ({any_["videos_scored"]} scored videos, {any_["frames_scored"]:,} scored 5 fps frames; '
              f'directions per behaviour = {len(any_["halves"]) * 2})', '',
              '| behaviour | prevalence | metric (chance) | V s2001 / s2002 / s2003 | PREDk16 s3001 / s3002 / s3003 | best V | best PREDk | spread | verdict |',
              '|---|---|---|---|---|---|---|---|---|']
        nb_ok = 0
        for b in labs:
            ok_all, Jb = True, {'prevalence': any_['pon'][b]['base'], 'n_pos': any_['pon'][b]['n_pos']}
            for m, mn in PRIM:
                v = [sc[k]['pon'][b][m] for k in KEYS_V]
                p = [sc[k]['pon'][b][m] for k in KEYS_P]
                if m == 'AUROC':
                    ch = '0.5'
                elif m == 'AP':
                    ch = f'{np.mean([sc[k]["pon"][b]["AP_chance_heldout_prev"] for k in KEYS_V]):.4f}'
                else:
                    ch = f'{np.mean([sc[k]["pon"][b]["top3vid_chance"] for k in KEYS_V]):.4f}'
                bv, bp = float(np.nanmax(v)), float(np.nanmax(p))
                spr = float(max(np.nanmax(v) - np.nanmin(v), np.nanmax(p) - np.nanmin(p)))
                ok = bp >= bv - spr
                ok_all &= ok
                Jb[m] = {'V': v, 'PREDk16': p, 'best_V': bv, 'best_PREDk16': bp, 'spread': spr, 'chance': float(ch), 'pass': bool(ok)}
                L.append(f'| {b} | {Jb["prevalence"]:.4f} | {mn} ({ch}) | {" / ".join(f"{x:.3f}" for x in v)} | '
                         f'{" / ".join(f"{x:.3f}" for x in p)} | {bv:.3f} | {bp:.3f} | {spr:.3f} | {"PASS" if ok else "FAIL"} |')
            Jb['at_least_as_good'] = bool(ok_all)
            nb_ok += ok_all
            J['behaviours'][b] = Jb
        mean = {arm: {m: [float(np.mean([sc[k]['pon'][b][m] for b in labs])) for k in ks] for m, _ in PRIM}
                for arm, ks in (('V', KEYS_V), ('PREDk16', KEYS_P))}
        J['mean_over_behaviours_per_dictionary'] = mean
        J['behaviours_at_least_as_good'] = f'{nb_ok}/{len(labs)}'
        J['answer'] = 'at least as good' if nb_ok == len(labs) else 'NOT at least as good on every behaviour'
        L += ['', f'Behaviours "at least as good" (all three metrics PASS): **{nb_ok}/{len(labs)}** -> {J["answer"]}.', '',
              '| mean over behaviours | ' + ' | '.join(f'{arm} {mn}' for arm in ('V', 'PREDk16') for _, mn in PRIM) + ' |',
              '|---|' + '---|' * 6]
        for i in range(3):
            L.append(f'| dictionary {i + 1} (V s{SEEDS_V[i]}, PREDk16 s{SEEDS_P[i]}) | ' +
                     ' | '.join(f'{mean[arm][m][i]:.3f}' for arm in ('V', 'PREDk16') for m, _ in PRIM) + ' |')
        L.append('')
        Jall['benchmarks'][dom] = J
        S[dom] = sc
    # health
    L += ['## Health per dictionary', '',
          '| set | dictionary | FVE (own eval tokens) | L0 / token | dead latents (train) | FVE on V held-out eval tokens | L0 there | never fire there |',
          '|---|---|---|---|---|---|---|---|']
    for dom in BENCH + ('frogs',):
        for k in (KEYS_V if dom in BENCH else []) + KEYS_P:
            if k.startswith('V_'):
                f = (B21.OUT / f'cal_train_{k}.json') if dom == 'cal' else (OUT / 'bench' / f'{dom}_trainV_s{k[3:]}.json')
            else:
                f = OUT / 'train' / f'{dom}_{k}.json'
            if not f.exists():
                L.append(f'| {dom} | {k} | MISSING | | | | | |')
                continue
            r = json.loads(f.read_text())
            hf = OUT / 'bench' / f'{dom}_health_{k}.json'
            h = json.loads(hf.read_text()) if hf.exists() else {}
            row = {'eval_fve': r['eval_fve'], 'eval_l0': r['eval_l0'], 'dead_train': r.get('dead_at_end_train'),
                   'never_fire_eval': r.get('never_fire_on_eval_frac'), 'heldout_V_eval': h or None,
                   'steps': r.get('steps_run', r.get('cfg', {}).get('steps')), 'steps_per_s': r.get('steps_per_s'),
                   'gpu_util_median': r.get('gpu_util_median'), 'contact_w': r.get('contact_w')}
            Jall['health'].setdefault(dom, {})[k] = row
            hv = (f'{r["eval_fve"]:.4f}', f'{r["eval_l0"]:.2f}') if k.startswith('V_') else (f'{h["fve"]:.4f}', f'{h["l0"]:.2f}') if h else ('-', '-')
            L.append(f'| {dom} | {k} | {r["eval_fve"]:.4f} | {r["eval_l0"]:.2f} | {r.get("dead_at_end_train")} | {hv[0]} | {hv[1]} | '
                     f'{h.get("never_fire_frac", "-") if h else "-"} |')
    cj = OUT / 'bench' / 'frogs_codes.json'
    Jall['frogs'] = {'dictionaries': {k: str(BEE / 'frogs' / 'sae' / k / 'sae_theta.pt') for k in KEYS_P},
                     'codes': json.loads(cj.read_text()) if cj.exists() else None}
    L += ['', f'Frogs dictionaries: {BEE}/frogs/sae/<key>/sae_theta.pt; codes {BEE}/frogs/codes/<key>.npy (rows.parquet: series, obs, k).']
    (OUT / 'tables.md').write_text('\n'.join(L) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(Jall, indent=1, default=float))
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('roles', 'prep', 'trainV', 'codes', 'score', 'report'))
    p.add_argument('--dom', default='cal', choices=BENCH + ('frogs',))
    p.add_argument('--key', default='')
    a = p.parse_args()
    {'roles': cmd_roles, 'prep': cmd_prep, 'trainV': cmd_trainV, 'codes': cmd_codes, 'score': cmd_score, 'report': cmd_report}[a.step](a)


if __name__ == '__main__':
    main()
