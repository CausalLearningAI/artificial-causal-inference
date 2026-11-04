"""
ECI v2.2 FULL WINDOW re-score (v22_prereg.md AMENDMENT 2). Every mice number since v2.1 scored only the 2-min Tier B
probe window per annotated video (144 x 600 = 86,400 frames, ~13% of the annotated data). This file builds the full
atlas-window frame set (Tier A of scripts/eci/v2_prob/analysis.py: all 144 annotated videos, rows with labels;
habituation '_H' videos frames >= 4500 = the last 15 min; 144 x 4500 = 648,000 frames at 5 fps), the codes on it, the
label-free HMM P(on) and the identity checks. Scoring re-uses v22_eval.py with V22_FRAMES=full (or =fullsub: the full
code path restricted to the old 86,400 frames with the OLD caches, the pipeline identity check).

Steps
  frames                      (CPU) -> BEEF/frames.parquet (+ OUTF/frames.json, frames.md: counts, window check)
  codes2 --task i --ntask n   (GPU or CPU) v2 codes of the videos[i::n]: ONE token read, all V seeds 1001-1009, frame
                              max + top10 (definitions of v22_encode.cmd_codes) -> BEEF/codes/parts/t<i>_<KEY>_<ro>.npy
  codes1 --task i --ntask n   (GPU) v1 s0-s10 codes of the videos[i::n] with v1's exact token pipeline
                              (scripts/eci/v2_prob/study.py enc1 + codes: DINOv2-base 448, fg448al mask, fp16 tokens,
                              threshold-mode SAE, frame max) -> BEEF/codes/parts/t<i>_v1_s<s>.npy
  hmmcheck                    (CPU) fast HMM (this file) vs src/eci/v2/hmm.py on the OLD 2-min V_s1001 codes: max abs
                              diff of P(on) over all neurons -> OUTF/hmmcheck.json (fail loud > 1e-6)
  cache --model KEY           (CPU) assemble the parts in full-table order; raw codes fp16; v2: P(on) of every neuron
                              (3-state HMM, one sequence per video = the full window) -> C.cache_paths(KEY)
  idcodes                     (CPU) identity of the new codes / labels on the old frames -> OUTF/identity_codes.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v22_core as C  # noqa: E402

REPO, T, log = C.REPO, C.T, C.log
BEEF = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2/full')
OUTF = REPO / 'results/vision/eci_v2_2/full'
OLD_FRAMES = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_eval/mice/frames.parquet')
TIERA = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_prob/mice/tierA/frames.parquet')
V_KEYS = [f'V_s{s}' for s in range(1001, 1010)]
V1_KEYS = [f'v1_s{s}' for s in range(11)]
LABS = {'nose_nose': ['Y_nn', 'Y_np'], 'nn': ['Y_nn'], 'np': ['Y_np'], 'nt': ['Y_nt']}


def local_dir():
    d = Path(os.environ.get('LOCAL_DIR', f'/localhome/{os.environ.get("USER")}/{os.environ.get("SLURM_JOB_ID", "x")}_v22f'))
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------- frames
def bouts(y, br):
    """number of bouts (rising edges; a positive first frame of a sequence counts) of bool y with sequence breaks br."""
    prev = np.r_[False, y[:-1]]
    return int((y & (br | ~prev)).sum())


def counts(G):
    br = np.r_[True, G.obs.values[1:] != G.obs.values[:-1]]
    npv = G.groupby('obs').size()
    d = {'pools': int(G.unit.nunique()), 'videos': int(G.obs.nunique()), 'frames': int(len(G)),
         'frames_per_video_min_median_max': [int(npv.min()), float(npv.median()), int(npv.max())]}
    for b in LABS:
        y = G[f'y_{b}'].values
        vb = pd.Series(y & (br | ~np.r_[False, y[:-1]])).groupby(G.obs.values).sum()
        d[b] = {'frames_on': int(y.sum()), 'bouts': bouts(y, br), 'videos_with_bout': int((vb > 0).sum()),
                'pools_with_bout': int(G.unit[G.obs.isin(vb.index[vb > 0])].nunique()), 'rate': float(y.mean())}
    return d


def cmd_frames(a):
    t0 = time.time()
    ann = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv', usecols=['observation_id', 'frame_idx', 'Y_nn', 'Y_np', 'Y_nt'])
    ann['row'] = np.arange(len(ann))
    isH = ann.observation_id.str.endswith('_H').values
    inwin = ann.Y_nn.notna().values & (~isH | (ann.frame_idx.values >= 4500))
    A = ann[inwin]
    TA = pd.read_parquet(TIERA)
    assert len(TA) == len(A) and (np.sort(TA.row.values) == np.sort(A.row.values)).all(), 'window != Tier A frames'
    O = pd.read_parquet(OLD_FRAMES)
    order = pd.unique(O.obs.values)
    assert set(order) == set(A.observation_id.unique()) and len(order) == 144
    A = A.rename(columns={'observation_id': 'obs', 'frame_idx': 'k'})
    A['vo'] = A.obs.map({o: i for i, o in enumerate(order)})
    A = A.sort_values(['vo', 'k']).reset_index(drop=True).drop(columns='vo')
    g = A.groupby('obs', sort=False).k.agg(['min', 'max', 'size'])
    assert (g['size'] == 4500).all() and (g['max'] - g['min'] + 1 == g['size']).all(), 'window not 4500 contiguous frames'
    key = pd.MultiIndex.from_arrays([A.obs.values, A.k.values])
    cr = pd.Series(np.arange(len(O)), index=pd.MultiIndex.from_arrays([O.obs.values, O.k.values])).reindex(key)
    A['code_row'] = cr.fillna(-1).astype(np.int64).values
    sub = A[A.code_row >= 0]
    assert len(sub) == len(O) == 86400 and (sub.code_row.values == np.arange(len(O))).all(), 'old frames not an ordered subset'
    assert (sub.row.values == O.row.values).all()
    for c in ('Y_nn', 'Y_np', 'Y_nt'):  # labels on the old frames: identical (NaN-safe)
        assert np.array_equal(sub[c].fillna(-9).values, O[c].fillna(-9).values), c
    BEEF.mkdir(parents=True, exist_ok=True)
    A[['obs', 'k', 'row', 'Y_nn', 'Y_np', 'Y_nt', 'code_row']].to_parquet(BEEF / 'frames.parquet')
    # ---- labelled frames OUTSIDE the window
    out = {}
    lab = ann[~inwin]
    lab_any = lab[['Y_nn', 'Y_np', 'Y_nt']].notna().any(axis=1)
    out['rows_outside_window'] = int(len(lab))
    out['rows_outside_window_with_any_label_value'] = int(lab_any.sum())
    out['videos_with_labels_not_in_window_set'] = sorted(set(ann.observation_id[ann.Y_nn.notna()]) - set(order))
    for b, cols in LABS.items():
        y_all = (ann[cols].fillna(0).to_numpy() > 0).any(1)
        br = np.r_[True, ann.observation_id.values[1:] != ann.observation_id.values[:-1]]
        start = y_all & (br | ~np.r_[False, y_all[:-1]])
        # a bout = maximal run in a video; it is 'inside' iff all its frames are in the window
        bid = np.cumsum(start) * y_all
        nb = int(start.sum())
        inside = pd.Series(inwin[y_all]).groupby(bid[y_all]).agg(['all', 'any'])
        out[b] = {'bouts_annotations_csv': nb, 'bouts_fully_inside': int(inside['all'].sum()),
                  'bouts_fully_outside': int((~inside['any']).sum()),
                  'bouts_straddling_boundary': int((inside['any'] & ~inside['all']).sum()),
                  'positive_frames_outside': int((y_all & ~inwin).sum()),
                  'positive_frames_outside_H_before_4500': int((y_all & ~inwin & isH & (ann.frame_idx.values < 4500)).sum())}
    # ---- count tables (old 2-min windows vs full)
    F = C.mice_table_full(A)
    res = {'outside_window': out, 'counts': {}}
    for fs, G0 in (('old_2min', F[F.code_row.values >= 0]), ('full', F)):
        for part, m in (('all', np.ones(len(G0), bool)), ('sg', G0.sg.values), ('nonsg', G0.nonsg.values),
                        ('unknown', G0.annot.values == 'NA')):
            res['counts'][f'{fs}|{part}'] = counts(G0[m])
    OUTF.mkdir(parents=True, exist_ok=True)
    (OUTF / 'frames.json').write_text(json.dumps(res, indent=1))
    md = ['| frame set | pools | videos | frames (per video min/median/max) | ' +
          ' | '.join(f'{C.NAME[b]}: positive frames / bouts / videos with >= 1 bout' for b in LABS) + ' |',
          '|---|---|---|---|' + '---|' * len(LABS)]
    for k, d in res['counts'].items():
        fpv = d['frames_per_video_min_median_max']
        md.append(f"| {k} | {d['pools']} | {d['videos']} | {d['frames']:,} ({fpv[0]}/{fpv[1]:.0f}/{fpv[2]}) | " +
                  ' | '.join(f"{d[b]['frames_on']:,} / {d[b]['bouts']:,} / {d[b]['videos_with_bout']}" for b in LABS) + ' |')
    (OUTF / 'frames.md').write_text('\n'.join(md) + '\n\n```\n' + json.dumps(out, indent=1) + '\n```\n')
    log(f'frames done ({time.time() - t0:.0f}s): {len(A):,} frames')


# ---------------------------------------------------------------------------------------------- v2 codes
def videos():
    F = pd.read_parquet(BEEF / 'frames.parquet', columns=['obs', 'k'])
    g = F.groupby('obs', sort=False).k.agg(['min', 'max'])
    return [(o, int(r['min']), int(r['max']) + 1) for o, r in g.iterrows()]


def feats_v(W, dev):
    """V input x_t + (x_{t+1} - x_{t-1}) / 2 of the M_t (flag bit0) tokens of frames i0..i1-1 (v21_train.centre_rows +
    raw_feats edge rule, the +-1 neighbours only; same float32 arithmetic) -> f (n, 1024) torch, frame index (n,) np."""
    import torch
    fi, pos = W['fi'], W['pos']
    keyv = fi * 65536 + pos
    s = pd.Series(np.arange(len(fi)), index=keyv)
    c = np.flatnonzero(((W['flag'] & 1) > 0) & (fi >= W['i0']) & (fi < W['i1']))
    nb = {d: s.reindex(keyv[c] + d * 65536).fillna(-1).values.astype(np.int64) for d in (-1, 1)}
    tok = W['tok']
    x = torch.from_numpy(tok[c]).to(dev).float()
    nx = {d: torch.from_numpy(tok[np.where(nb[d] >= 0, nb[d], 0)]).to(dev).float() for d in nb}
    mp, mm = torch.from_numpy(nb[1] >= 0).to(dev)[:, None], torch.from_numpy(nb[-1] >= 0).to(dev)[:, None]
    cd = torch.where(mp & mm, (nx[1] - nx[-1]) / 2, torch.where(mp, nx[1] - x, torch.where(mm, x - nx[-1], torch.zeros_like(x))))
    return x + cd, fi[c]


def cmd_codes2(a):
    import torch
    from src.eci.v2 import sae as SA
    import v22_encode as EN
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    keys = a.models.split(',') if a.models else V_KEYS
    saes = {k: SA.load(EN.sae_path('mice', k), dev) for k in keys}
    vids = videos()[a.task::a.ntask]
    nfr = sum(k1 - k0 for _, k0, k1 in vids)
    X = {(k, r): np.zeros((nfr, 2048), np.float16) for k in keys for r in ('max', 'top10')}
    rows, base, ntok, t0 = [], 0, 0, time.time()
    CH = int(a.chunk)
    for vi, (obs, k0, k1) in enumerate(vids):
        V_ = T.Video('mice', 'v1', obs)
        kpos = {int(k): i for i, k in enumerate(V_.k)}
        i0, i1 = kpos[k0], kpos[k1 - 1] + 1
        assert i1 - i0 == k1 - k0, (obs, 'store frames != window')
        rows.append(pd.DataFrame({'obs': obs, 'k': np.asarray(V_.k[i0:i1], np.int64)}))
        for b0 in range(i0, i1, CH):
            b1 = min(i1, b0 + CH)
            W = V_.window(b0, b1)
            f, fi = feats_v(W, dev)
            if not len(fi):
                continue
            ntok += len(fi)
            lf = torch.from_numpy(fi - b0).to(dev)
            n = b1 - b0
            o_ = torch.argsort(lf, stable=True)
            lfs = lf[o_]
            cnt_ = torch.bincount(lfs, minlength=n)
            st_ = torch.cumsum(cnt_, 0) - cnt_
            rk = torch.arange(len(lfs), device=dev) - st_[lfs]
            sl = slice(base + b0 - i0, base + b1 - i0)
            for k, (sae, norm) in saes.items():
                with torch.no_grad():
                    cc = sae.encode(norm(f), mode='threshold').float()
                    m = torch.zeros((n, 2048), device=dev).index_reduce_(0, lf, cc, 'amax', include_self=True)
                    X[(k, 'max')][sl] = m.half().cpu().numpy()
                    D = torch.zeros((n, int(cnt_.max()), 2048), device=dev)
                    D[lfs, rk] = cc[o_]
                    v_ = D.topk(min(10, D.shape[1]), dim=1).values.sum(1) / cnt_.clamp(min=1).float()[:, None].clamp(max=10)
                    X[(k, 'top10')][sl] = v_.half().cpu().numpy()
                    del D, cc
        base += i1 - i0
        el = time.time() - t0
        log(f'task {a.task}: video {vi + 1}/{len(vids)} {obs}: {ntok:,} tokens, {el:.0f}s, {ntok / el:,.0f} tokens/s '
            f'({len(keys)} SAEs), projected task wall {el / (vi + 1) * len(vids):.0f}s')
    assert base == nfr
    od = BEEF / 'codes' / 'parts'
    od.mkdir(parents=True, exist_ok=True)
    R = pd.concat(rows, ignore_index=True)
    loc = local_dir()
    R.to_parquet(od / f't{a.task}_rows2.parquet')
    import shutil
    for (k, r), x in X.items():
        np.save(loc / 'tmp.npy', x)
        shutil.copyfile(loc / 'tmp.npy', od / f't{a.task}_{k}_{r}.npy')
    (loc / 'tmp.npy').unlink()
    meta = {'task': a.task, 'videos': len(vids), 'frames': nfr, 'tokens': ntok, 'seconds': time.time() - t0,
            'tokens_per_s': ntok / (time.time() - t0), 'device': str(dev),
            'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}
    (OUTF / 'codes').mkdir(parents=True, exist_ok=True)
    (OUTF / 'codes' / f'codes2_t{a.task}.json').write_text(json.dumps(meta, indent=1))
    log(f'codes2 done {json.dumps(meta)}')


# ---------------------------------------------------------------------------------------------- v1 codes
def cmd_codes1(a):
    import shutil
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from src.eci.foreground import RULES, FgBackgrounds, FgEncoder, align_rot90, encode_batch, obs_rows
    from src.eci.sae import load_sae
    import eval as EV
    dev = torch.device('cuda')
    c = dict(ann='dataset/mice/v1/annotations.csv', bg='dataset/mice/v1/eci/fg448al/background', rule='fg448', align='odor')
    ann = REPO / c['ann']
    F = pd.read_parquet(BEEF / 'frames.parquet', columns=['obs', 'k', 'row'])
    vids = [o for o, _, _ in videos()][a.task::a.ntask]
    Fr = F[F.obs.isin(vids)].reset_index(drop=True)
    rows = Fr.row.values.astype(np.int64)
    fp = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    rot = align_rot90(c['align'], ann)
    # stage the JPGs of these frames on local disk (one read of each file from NFS)
    loc = local_dir() / 'frames'
    t0 = time.time()
    src = [REPO / 'dataset' / p for p in fp[rows]]
    dst = [loc / p for p in fp[rows]]
    for d in {p.parent for p in dst}:
        d.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda sd: shutil.copyfile(sd[0], sd[1]), zip(src, dst)))
    log(f'task {a.task}: staged {len(dst):,} JPGs in {time.time() - t0:.0f}s')
    enc = FgEncoder('dinov2_base', dev)
    fb = FgBackgrounds(REPO / c['bg'], obs_rows(ann), RULES[c['rule']], dev, align=c['align'], cache=8)
    ds = enc.dataset([str(p) for p in dst], rows, None if rot is None else rot[rows])
    dl = torch.utils.data.DataLoader(ds, batch_size=50, num_workers=max(2, int(os.environ.get('SLURM_CPUS_PER_TASK', 8)) - 2),
                                     shuffle=False, pin_memory=True, prefetch_factor=4)
    saes = {'v1_s0': load_sae(REPO / EV.V1SAE['mice'], dev)[:2]}
    for s in range(1, 11):
        saes[f'v1_s{s}'] = load_sae(EV.EC / f'v2_diag/v1retrain/mice/V1R_full_s{s}/sae.pt', dev)[:2]
    X = {n_: torch.zeros((len(Fr), 1024), dtype=torch.float16, device=dev) for n_ in saes}
    o, ntok, t1 = 0, 0, time.time()
    for bi, (pix, grey, rr) in enumerate(dl):
        with torch.no_grad():
            tok = encode_batch(enc.model, pix.to(dev, non_blocking=True), dev)
            mk, _ = fb.mask(tok, grey.to(dev), rr.numpy())
            fi, pi = torch.nonzero(mk, as_tuple=True)
            z = tok[fi, pi].float()  # fp16 tokens, as stored by the probe pipeline, read back as float32
            ntok += len(z)
            for n_, (sae, norm) in saes.items():
                cc = sae.encode(norm(z), mode='threshold').float()
                fm = torch.zeros((len(tok), cc.shape[1]), device=dev).index_reduce_(0, fi, cc, 'amax', include_self=True)
                X[n_][o:o + len(tok)] = fm.half()
        assert (rr.numpy() == rows[o:o + len(tok)]).all()
        o += len(tok)
        if bi % 100 == 0:
            el = time.time() - t1
            log(f'task {a.task}: {o:,}/{len(Fr):,} frames, {o / max(el, 1):.1f} fps, projected {len(Fr) / max(o, 1) * el:.0f}s')
    assert o == len(Fr)
    od = BEEF / 'codes' / 'parts'
    od.mkdir(parents=True, exist_ok=True)
    Fr[['obs', 'k']].to_parquet(od / f't{a.task}_rows1.parquet')
    for n_, x in X.items():
        np.save(od / f't{a.task}_{n_}.npy', x.cpu().numpy())
    meta = {'task': a.task, 'frames': len(Fr), 'tokens_per_frame': ntok / len(Fr), 'seconds': time.time() - t0,
            'fps': len(Fr) / (time.time() - t1), 'gpu': torch.cuda.get_device_name(0)}
    (OUTF / 'codes').mkdir(parents=True, exist_ok=True)
    (OUTF / 'codes' / f'codes1_t{a.task}.json').write_text(json.dumps(meta, indent=1))
    shutil.rmtree(loc, ignore_errors=True)
    log(f'codes1 done {json.dumps(meta)}')


def assemble(key, ro=None):
    """parts of KEY (readout ro for v2) -> (n_full, W) float16 in full-table order."""
    F = pd.read_parquet(BEEF / 'frames.parquet', columns=['obs', 'k'])
    idx = pd.Series(np.arange(len(F)), index=pd.MultiIndex.from_arrays([F.obs.values, F.k.values]))
    od = BEEF / 'codes' / 'parts'
    v1 = key.startswith('v1_')
    out, seen = None, np.zeros(len(F), bool)
    t = 0
    while (od / f't{t}_rows{1 if v1 else 2}.parquet').exists():
        R = pd.read_parquet(od / f't{t}_rows{1 if v1 else 2}.parquet')
        x = np.load(od / (f't{t}_{key}.npy' if v1 else f't{t}_{key}_{ro}.npy'))
        r = idx.reindex(pd.MultiIndex.from_arrays([R.obs.values, R.k.values])).values
        assert not np.isnan(r).any()
        r = r.astype(np.int64)
        if out is None:
            out = np.zeros((len(F), x.shape[1]), np.float16)
        assert not seen[r].any()
        out[r], seen[r] = x, True
        t += 1
    assert seen.all(), f'{key}: {(~seen).sum()} frames without codes ({t} parts)'
    return out


# ---------------------------------------------------------------------------------------------- fast HMM
LOG2PI = np.log(2 * np.pi)


@njit(cache=True)
def _fb_core(x, starts, pi0, A, z, mu, var, al, sc, Bs, g):
    """scaled forward-backward of src/eci/v2/hmm.py (_logB + _fb fused, same formulas) -> fills g (n, K); returns xis,
    log-likelihood."""
    n, K = g.shape
    c0 = np.log(z)
    c1 = np.log(1 - z) - 0.5 * (LOG2PI + np.log(var))
    ll = 0.0
    lb = np.empty(K)
    for t in range(n):
        if x[t] <= 0:
            for k in range(K):
                lb[k] = c0[k]
        else:
            lx = np.log(x[t])
            for k in range(K):
                lb[k] = c1[k] - 0.5 * ((lx - mu[k]) ** 2 / var[k]) - lx
        m = lb[0]
        for k in range(1, K):
            if lb[k] > m:
                m = lb[k]
        for k in range(K):
            Bs[t, k] = np.exp(lb[k] - m)
        ll += m
    xis = np.zeros((K, K))
    be = np.empty(K)
    be2 = np.empty(K)
    for s in range(starts.shape[0] - 1):
        a, b = starts[s], starts[s + 1]
        if b <= a:
            continue
        tot = 0.0
        for k in range(K):
            al[a, k] = pi0[k] * Bs[a, k]
            tot += al[a, k]
        sc[a] = tot
        for k in range(K):
            al[a, k] /= tot
        for t in range(a + 1, b):
            tot = 0.0
            for k in range(K):
                v = 0.0
                for j in range(K):
                    v += al[t - 1, j] * A[j, k]
                al[t, k] = v * Bs[t, k]
                tot += al[t, k]
            sc[t] = tot
            for k in range(K):
                al[t, k] /= tot
        for k in range(K):
            be[k] = 1.0
        # t = b - 1
        gg = 0.0
        for k in range(K):
            g[b - 1, k] = al[b - 1, k]
            gg += g[b - 1, k]
        for k in range(K):
            g[b - 1, k] /= gg
        for t in range(b - 2, a - 1, -1):
            # xi of the transition t -> t + 1 (be = beta_{t+1})
            tot = 0.0
            for j in range(K):
                for k in range(K):
                    tot += al[t, j] * A[j, k] * Bs[t + 1, k] * be[k]
            for j in range(K):
                for k in range(K):
                    xis[j, k] += al[t, j] * A[j, k] * Bs[t + 1, k] * be[k] / tot
            for j in range(K):
                v = 0.0
                for k in range(K):
                    v += A[j, k] * Bs[t + 1, k] * be[k]
                be2[j] = v / sc[t + 1]
            gg = 0.0
            for k in range(K):
                be[k] = be2[k]
                g[t, k] = al[t, k] * be[k]
                gg += g[t, k]
            for k in range(K):
                g[t, k] /= gg
        for t in range(a, b):
            ll += np.log(sc[t])
    return xis, ll


@njit(cache=True)
def _fit_one_fast(x, starts, mu0, z0, var0, iters, tol, out):
    """Baum-Welch of src/eci/v2/hmm.fit (K = len(mu0)), then P(on) of every frame under the fitted params -> out (n,).
    Returns (on state, iterations, ll)."""
    n = x.shape[0]
    K = mu0.shape[0]
    mu, z, var = mu0.copy(), z0.copy(), var0.copy()
    A = np.full((K, K), 0.05 / (K - 1))
    for k in range(K):
        A[k, k] = 0.95
    pi0 = np.full(K, 1.0 / K)
    al = np.empty((n, K))
    sc = np.empty(n)
    Bs = np.empty((n, K))
    g = np.empty((n, K))
    nseq = 0
    for s in range(starts.shape[0] - 1):
        if starts[s + 1] > starts[s]:
            nseq += 1
    prev = -np.inf
    ll = 0.0
    it = 0
    for it in range(iters):
        xis, ll = _fb_core(x, starts, pi0, A, z, mu, var, al, sc, Bs, g)
        p0 = np.zeros(K)
        for s in range(starts.shape[0] - 1):
            if starts[s + 1] > starts[s]:
                for k in range(K):
                    p0[k] += g[starts[s], k]
        ps = 0.0
        for k in range(K):
            p0[k] = max(p0[k] / nseq, 1e-6)
            ps += p0[k]
        for k in range(K):
            pi0[k] = p0[k] / ps
        for j in range(K):
            rs = 0.0
            for k in range(K):
                rs += xis[j, k]
            rs = max(rs, 1e-300)
            r2 = 0.0
            for k in range(K):
                A[j, k] = max(xis[j, k] / rs, 1e-9)
                r2 += A[j, k]
            for k in range(K):
                A[j, k] /= r2
        gs = np.zeros(K)
        gp = np.zeros(K)
        gl = np.zeros(K)
        for t in range(n):
            for k in range(K):
                gs[k] += g[t, k]
            if x[t] > 0:
                lx = np.log(x[t])
                for k in range(K):
                    gp[k] += g[t, k]
                    gl[k] += g[t, k] * lx
        for k in range(K):
            z[k] = min(max((gs[k] - gp[k]) / max(gs[k], 1e-300), 1e-6), 1 - 1e-6)
            if gp[k] > 1e-9:
                mu[k] = gl[k] / max(gp[k], 1e-300)
        gv = np.zeros(K)
        for t in range(n):
            if x[t] > 0:
                lx = np.log(x[t])
                for k in range(K):
                    gv[k] += g[t, k] * (lx - mu[k]) ** 2
        for k in range(K):
            if gp[k] > 1e-9:
                var[k] = gv[k] / max(gp[k], 1e-300)
            var[k] = max(var[k], 1e-3)
        if (ll - prev) / n < tol:
            break
        prev = ll
    on = 0
    best = -np.inf
    for k in range(K):
        ec = (1 - z[k]) * np.exp(mu[k] + var[k] / 2)
        if ec > best:
            best, on = ec, k
    _fb_core(x, starts, pi0, A, z, mu, var, al, sc, Bs, g)
    for t in range(n):
        out[t] = g[t, on]
    return on, it + 1, ll


@njit(parallel=True, cache=True)
def _fit_all(Xt, starts, mu0, z0, var0, ok, out):
    """Xt (W, n) float32 codes; init params (W, K); ok (W,) = fit (else P(on) = 1[c > 0]); out (W, n) float32."""
    W, n = Xt.shape
    iters = np.zeros(W, np.int64)
    for j in prange(W):
        x = Xt[j].astype(np.float64)
        if not ok[j]:
            for t in range(n):
                out[j, t] = 1.0 if x[t] > 0 else 0.0
            continue
        o = np.empty(n)
        _, it, _ = _fit_one_fast(x, starts, mu0[j], z0[j], var0[j], 200, 1e-7, o)
        iters[j] = it
        for t in range(n):
            out[j, t] = o[t]
    return iters


def pon_fast(raw, starts):
    """raw (n, W) codes, starts (m + 1,) -> P(on) (n, W) float32 (src/eci/v2/hmm.py K = 3, fit + apply), iterations."""
    raw = np.asarray(raw, np.float32)
    Xt = np.ascontiguousarray(raw.T)
    W = Xt.shape[0]
    mu0, z0, var0, ok = np.zeros((W, 3)), np.tile([0.95, 0.5, 0.05], (W, 1)), np.zeros((W, 3)), np.zeros(W, bool)
    for j in range(W):
        x = Xt[j].astype(np.float64)
        pos = x[x > 0]
        if len(pos) < 200:
            continue
        lx = np.log(pos)
        mu0[j] = np.percentile(lx, [10, 50, 90])
        var0[j] = max(lx.var(), 1e-3)
        ok[j] = True
    out = np.empty((W, len(raw)), np.float32)
    it = _fit_all(Xt, np.asarray(starts, np.int64), mu0, z0, var0, ok, out)
    return np.ascontiguousarray(out.T), it


def cmd_hmmcheck(a):
    """fast HMM vs the stored v2.2 P(on) cache of V_s1001 on the old 2-min windows (sequences = the 144 windows)."""
    from src.eci.v2 import hmm as HM
    pr, pp = C.cache_paths('V_s1001')  # V22_FRAMES=old paths
    raw = np.load(pr).astype(np.float32)
    ref = np.load(pp)
    st = np.arange(0, len(raw) + 1, 600)
    t0 = time.time()
    P, it = pon_fast(raw, st)
    el = time.time() - t0
    d = np.abs(P - ref)
    res = {'neurons': int(raw.shape[1]), 'frames': int(len(raw)), 'max_abs_diff_pon': float(d.max()),
           'mean_abs_diff_pon': float(d.mean()), 'n_neurons_diff_gt_1e-6': int((d.max(0) > 1e-6).sum()),
           'seconds_fast': el, 'cpus': int(os.environ.get('SLURM_CPUS_PER_TASK', 1)),
           'iterations_median_max': [float(np.median(it)), int(it.max())]}
    # reference re-fit of a few neurons with the original code (params + timing)
    t1 = time.time()
    js = list(range(0, raw.shape[1], 256))
    mx = 0.0
    for j in js:
        p = HM.fit(raw[:, j].astype(np.float64), st, K=3)
        q = HM.apply(raw[:, j].astype(np.float64), st, p)[0]
        mx = max(mx, float(np.abs(q - P[:, j]).max()))
    res['orig_refit_neurons'] = js
    res['orig_refit_max_abs_diff'] = mx
    res['orig_seconds_per_neuron'] = (time.time() - t1) / len(js)
    res['fast_cpu_seconds_per_neuron'] = el * res['cpus'] / raw.shape[1]
    OUTF.mkdir(parents=True, exist_ok=True)
    (OUTF / 'hmmcheck.json').write_text(json.dumps(res, indent=1))
    log(f'hmmcheck {json.dumps(res)}')
    if res['max_abs_diff_pon'] > 1e-6 or mx > 1e-6:
        raise SystemExit('FAST HMM IDENTITY FAILED')


def cmd_cache(a):
    assert C.FRAMES == 'full'
    key = a.model
    t0 = time.time()
    v1 = key.startswith('v1_')
    base, ro = (key.split('@') + ['max'])[:2]
    raw = assemble(base if v1 else base, None if v1 else ro)
    pr, pp = C.cache_paths(key)
    pr.parent.mkdir(parents=True, exist_ok=True)
    loc = local_dir()
    import shutil
    np.save(loc / 'raw.npy', raw)
    shutil.copyfile(loc / 'raw.npy', pr)
    (loc / 'raw.npy').unlink()
    log(f'{key}: raw {raw.shape} assembled ({time.time() - t0:.0f}s)')
    if v1:
        return
    F = pd.read_parquet(BEEF / 'frames.parquet', columns=['obs'])
    o = F.obs.values
    st = np.r_[np.flatnonzero(np.r_[True, o[1:] != o[:-1]]), len(o)].astype(np.int64)
    t1 = time.time()
    P, it = pon_fast(raw, st)
    log(f'{key}: HMM P(on) of {raw.shape[1]} neurons on {len(raw):,} frames in {time.time() - t1:.0f}s '
        f'(iterations median {np.median(it):.0f}, max {it.max()})')
    np.save(loc / 'pon.npy', P)
    shutil.copyfile(loc / 'pon.npy', pp)
    (loc / 'pon.npy').unlink()
    log(f'cache {key} done ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- identity of codes
def cmd_idcodes(a):
    """new codes on the old 86,400 frames vs the stored old codes."""
    F = pd.read_parquet(BEEF / 'frames.parquet', columns=['code_row'])
    m = F.code_row.values >= 0
    rr = F.code_row.values[m]
    assert (rr == np.arange(len(rr))).all()
    res = {}
    E2 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
    refs = {}
    for s in range(1001, 1010):
        refs[(f'V_s{s}', 'max')] = [E2 / f'v2_2/mice/codes/V_s{s}_max_eval.npy', E2 / f'v2_1/mice/codes/V_s{s}_eval.npy']
        refs[(f'V_s{s}', 'top10')] = [E2 / f'v2_2/mice/codes/V_s{s}_top10_eval.npy', E2 / f'v2_1/mice/codes/V_s{s}_eval_top10.npy']
    for s in range(11):
        refs[(f'v1_s{s}', None)] = [E2 / f'v2_prob/mice/codes/v1_s{s}.npy']
    # old rows of the v2 eval codes = v22 eval_rows (same order as the old frame table, checked in v22_core.build_cache)
    R0 = pd.read_parquet(E2 / 'v2_2/mice/codes/eval_rows.parquet')
    Fo = pd.read_parquet(OLD_FRAMES)
    rix = pd.Series(np.arange(len(R0)), index=pd.MultiIndex.from_arrays([R0.obs.values, R0.k.values])).reindex(
        pd.MultiIndex.from_arrays([Fo.obs.values, Fo.k.values])).values.astype(np.int64)
    for (key, ro), paths in refs.items():
        try:
            new = assemble(key, ro)[m].astype(np.float32)
        except AssertionError as e:
            res[f'{key}|{ro}'] = {'missing': str(e)}
            continue
        for p in paths:
            if not p.exists():
                continue
            old = np.load(p, mmap_mode='r')
            old = np.asarray(old[rix] if key.startswith('V_') else old[Fo.code_row.values], np.float32)
            d = np.abs(new - old)
            thr = 2.24 if key.startswith('V_') else None
            res[f'{key}|{ro}|{p.parent.parent.parent.name}'] = {
                'max_abs_diff': float(d.max()), 'mean_abs_diff': float(d.mean()),
                'share_entries_diff_gt_0.01': float((d > 0.01).mean()), 'share_entries_diff_gt_0.5': float((d > 0.5).mean()),
                'fire_agree': float(((new > 0) == (old > 0)).mean()),
                'entries_diff_gt_0.01_that_are_threshold_flips': float((((new == 0) | (old == 0)) & (d > 0.01)).sum() / max((d > 0.01).sum(), 1))}
            log(f'{key}|{ro} vs {p}: {res[f"{key}|{ro}|{p.parent.parent.parent.name}"]}')
    (OUTF / 'identity_codes.json').write_text(json.dumps(res, indent=1))


def cmd_idpipe(a):
    """pipeline identity: report.json of V22_FRAMES=fullsub vs the v2.2 report.json (picks, every level value per seed,
    contrast point differences)."""
    old = json.loads((REPO / 'results/vision/eci_v2_2/report.json').read_text())
    new = json.loads((OUTF / 'fullsub' / 'report.json').read_text())
    res = {}
    for dset in ('sg2non', 'sgcf'):
        So, Sn = old['sets'][dset], new['sets'][dset]
        same, mx, mx_v1, n, worst = True, 0.0, 0.0, 0, None
        for arm in Sn['levels']:
            same &= Sn['picks'][arm] == So['picks'][arm]
            for mk, v in Sn['levels'][arm].items():
                d = float(np.nanmax(np.abs(np.array(v['seeds'], float) - np.array(So['levels'][arm][mk]['seeds'], float))))
                n += 1
                if np.isnan(d):
                    continue
                if arm.startswith('v1'):  # v1 F1 / time use the HMM on the picked neuron (fast implementation here)
                    mx_v1 = max(mx_v1, d)
                elif d > mx:
                    mx, worst = d, (arm, mk)
        auc = max(abs(Sn['levels'][arm][f'{b}|auc']['seeds'][i] - So['levels'][arm][f'{b}|auc']['seeds'][i])
                  for arm in Sn['levels'] for b in ('nose_nose', 'nt') for i in range(len(Sn['levels'][arm][f'{b}|auc']['seeds'])))
        cd = max(abs(r['diff'] - So['contrasts'][c][k]['diff']) for c, rr in Sn['contrasts'].items() if c in So['contrasts'] for k, r in rr.items()
                 if np.isfinite(r['diff']))
        res[dset] = {'arms': len(Sn['levels']), 'picks_identical': bool(same), 'auroc_max_abs_diff': float(auc),
                     'all_level_values_max_abs_diff_v2': mx, 'worst_v2': worst, 'all_level_values_max_abs_diff_v1': mx_v1,
                     'contrast_point_max_abs_diff': float(cd), 'n_level_metrics': n}
    (OUTF / 'identity_pipeline.json').write_text(json.dumps(res, indent=1))
    log(f'idpipe {json.dumps(res)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('frames', 'codes2', 'codes1', 'hmmcheck', 'cache', 'idcodes', 'idpipe'))
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--ntask', type=int, default=1)
    p.add_argument('--models', default='')
    p.add_argument('--model', default='')
    p.add_argument('--chunk', default='100')
    a = p.parse_args()
    {'frames': cmd_frames, 'codes2': cmd_codes2, 'codes1': cmd_codes1, 'hmmcheck': cmd_hmmcheck, 'cache': cmd_cache,
     'idcodes': cmd_idcodes, 'idpipe': cmd_idpipe}[a.step](a)
