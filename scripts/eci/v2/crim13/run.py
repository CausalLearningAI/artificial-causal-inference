"""
ECI v2 on CRIM13 - VALIDATION CHECK of the FROZEN representation (configs/eci_v2/representation.json) on a public
benchmark. Config: configs/eci_v2/crim13.json. Nothing is tuned here: every representation choice is the spec's.

Why a CRIM13 entry point next to scripts/eci/v2/run.py: CRIM13 frames live in Norpix .seq files (JPEG records read by
seek offset, not ffmpeg-decodable lab videos with an r4 plan / annotations.csv layout), the frame plan is the CRIM13
alignment test's (stratified eval frames with weights) and the folds are the released halves. Everything else is the
v2 library / run.py code, imported unchanged:
    decode / canvas   scripts/benchmark/crim13/align_common.read_frames, canvas_gpu (the 640 canvas of arms B0-B7 / C1-C4)
    mask              src/eci/v2/mask.grey_of + THR (tight: >= 25% animal pixels per 16 x 16 patch, no dilation) with the
                      CRIM13 'dark' pixel cue (grey < 60 and > 40 below the per-pixel median of 32 frames); no token cue
    encoder           src/eci/v2/encode.Encoder (DINOv3-L/16 @ ea8dc28, block 20 + final LayerNorm, bf16 + SDPA)
    token store       run.py write_store / Store (frames.parquet, tok_frame / tok_pos / tok_fdark / tok_dist .npy, tok.f16)
    SAE               src/eci/v2/sae.train / deploy_seed / frame_codes (Matryoshka BatchTopK 2048, k 16, 30000 steps,
                      seeds 6 / 7 / 8, theta* on calib tokens, deployment seed = highest held-out FVE)
    readout           frame max over kept tokens, src/eci/v2/readout.smooth h 1 (0.6 s centred mean on the 5 fps grid)
    scoring           scripts/benchmark/crim13/align_train.score (weighted align_w2 + pick_and_test, unchanged); R~ =
                      (R-precision - pi) / (1 - pi) of the R-precision-picked neuron
    heat maps         src/eci/v2/heatmap.neuron_maps (per-token activation of the frame)

Steps (scripts/eci/v2/crim13/run.sh runs each in Slurm; staging in $LOCAL_DIR under /localhome):
    encode  (GPU) --shard i of n: tight-mask kept tokens of every needed frame of the shard's videos -> BEE/tok/shard<i>/
            needed = train-half pool (role train / calib), test-half role calib (held-out FVE) and every eval frame of
            both halves with its 5 fps neighbours v - 5, v + 5 (smoothing); + the mask check against the released tracks
    train   (GPU) one SAE on the train half's pool, all spec seeds -> BEE/sae/<name>/, OUT/train.json, OUT/best.json
    score   (GPU) every seed: frame codes, smoothing, both directions (and the unsmoothed codes as a sensitivity)
            -> OUT/score_s<seed>.json; deployed seed: contact sheets -> OUT/sheets/
    report  (CPU, small JSONs) table vs the stored CRIM13 arms on the same frames / weights / halves -> OUT/report.{md,json}
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/benchmark/crim13'))
sys.path.insert(0, str(REPO / 'scripts/eci'))
sys.path.insert(0, str(REPO / 'scripts/benchmark/mabe22'))
from align_common import (GRID, PAIR, ROOT, SCORED, canvas_gpu, load_seek, load_track, read_frames,  # noqa: E402
                          track_cell, video_cells)

_s = importlib.util.spec_from_file_location('v2_run', REPO / 'scripts/eci/v2/run.py')
RUN = importlib.util.module_from_spec(_s)
_s.loader.exec_module(RUN)

C = json.loads((REPO / 'configs/eci_v2/crim13.json').read_text())
SPEC = json.loads((REPO / 'configs/eci_v2/representation.json').read_text())
OUT = REPO / C['out']
BEE = Path(C['bee'])
G = C['geometry']['G']
S = C['geometry']['S']
NBG = C['mask']['n_bg_frames']
STEP = C['step']
log = RUN.log
assert G == GRID and S == 640 and SCORED == C['score']['labels'] and PAIR == C['score']['pair']


def sae_name(seed):
    return f'v2_crim13_c{SPEC["context_window"]}_w{SPEC["sae"]["W"]}_s{seed}'


def plan():
    P = pd.read_parquet(REPO / C['plan'])
    V = pd.read_parquet(REPO / C['plan'].replace('frames.parquet', 'videos.parquet')).set_index('video')
    return P, V


def needed(P, V):
    """(video, v) of every frame to encode with its uses (flags)."""
    parts = []
    pool = P[(P.half == 'train') & P.role.isin(['train', 'calib'])]
    parts.append(pd.DataFrame({'video': pool.video, 'v': pool.v, 'is_train': pool.role.values == 'train',
                               'is_calib': pool.role.values == 'calib'}))
    fv = P[(P.half == 'test') & (P.role == 'calib')]
    parts.append(pd.DataFrame({'video': fv.video, 'v': fv.v, 'is_fve': True}))
    E = P[P.role == 'eval']
    parts.append(pd.DataFrame({'video': E.video, 'v': E.v, 'is_eval': True, 'is_evn': True}))
    N = V.frames.reindex(E.video).values
    for d in (-STEP, STEP):
        ok = (E.v.values + d >= 0) & (E.v.values + d < N)
        parts.append(pd.DataFrame({'video': E.video.values[ok], 'v': E.v.values[ok] + d, 'is_evn': True}))
    A = pd.concat(parts, ignore_index=True)
    for f in ('is_train', 'is_calib', 'is_fve', 'is_eval', 'is_evn'):
        A[f] = A[f].astype('boolean').fillna(False).astype(bool)
    A = A.groupby(['video', 'v'], as_index=False).max()
    A['half'] = V.half.reindex(A.video).values
    return A.sort_values(['video', 'v']).reset_index(drop=True)


def shard_videos(A, n, i):
    c = A.groupby('video').size().reset_index(name='n').sort_values(['n', 'video'], ascending=[False, True],
                                                                     kind='stable')
    return [v for j, v in enumerate(c.video) if j % n == i]


# ---------------------------------------------------------------------------------------------- encode
class Masker:
    """Tight CRIM13 mask: per-video background = per-pixel median grey of NBG evenly spaced frames on the canvas."""

    def __init__(self, dev):
        self.dev = dev

    def background(self, frames, W, H):
        from src.eci.v2.mask import grey_of
        cb = canvas_gpu(frames, W, H, self.dev).to(__import__('torch').uint8)
        return grey_of(cb).float().median(0).values

    def fraction(self, x, bg, vc):
        """x (B, 3, S, S) uint8 canvas -> (B, G*G) animal-pixel fraction per patch (0 on pure-padding cells)."""
        import torch.nn.functional as F
        from src.eci.v2.mask import grey_of
        g = grey_of(x).float()
        px = (g < C['mask']['dark_abs']) & ((bg[None] - g) > C['mask']['dark_rel'])
        return F.avg_pool2d(px.float()[:, None], 16).flatten(1) * vc[None]


def cmd_encode(a):
    import torch
    from src.eci.v2.encode import Encoder
    from src.eci.v2.mask import THR, mask_sheet
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    P, V = plan()
    A = needed(P, V)
    vids = shard_videos(A, a.nshard, a.shard)
    if a.max_videos:
        vids = vids[:a.max_videos]
    tag = f'shard{a.shard}of{a.nshard}' + ('_selftest' if a.selftest else '')
    dst = BEE / ('tok_selftest' if a.selftest else 'tok') / tag
    if (dst / 'frames.parquet').exists() and not a.force:
        log(f'[SKIP] {dst}')
        return
    enc = Encoder(dev)
    mk = Masker(dev)
    loc = RUN.local_dir() / tag
    loc.mkdir(parents=True, exist_ok=True)
    ftok = open(loc / 'tok.f16', 'wb')
    toks, poss, fds, frames, sheet = [], [], [], [], []
    ce = C['encode']
    t00 = time.time()
    log(f'{tag}: {len(vids)} videos, {int(A.video.isin(vids).sum())} frames, {torch.cuda.get_device_name(0)}')

    def dec(q):
        t_ = time.time()
        R = A[A.video == q]
        N, d = int(V.loc[q, 'frames']), ROOT / V.loc[q, 'half']
        offs = load_seek(d / f'{q}_t-seek.mat')
        assert len(offs) == N, (q, len(offs), N)
        bgi = np.linspace(0, N - 1, NBG).round().astype(int)
        fr, nb = read_frames(d / f'{q}_t.seq', offs, set(bgi.tolist()) | set(R.v.tolist()), threads=ce['threads'])
        return fr, load_track(d / f'{q}_t-track.mat'), nb, time.time() - t_

    pool = ThreadPoolExecutor(ce['workers'])
    fut = {j: pool.submit(dec, vids[j]) for j in range(min(len(vids), ce['workers'] + 1))}
    tdec = twait = 0.0
    nbytes = 0
    for k, q in enumerate(vids):
        t0 = time.time()
        fr, (tpos, nm), nb, dt = fut.pop(k).result()
        twait += time.time() - t0
        tdec, nbytes = tdec + dt, nbytes + nb
        if k + ce['workers'] + 1 < len(vids):
            fut[k + ce['workers'] + 1] = pool.submit(dec, vids[k + ce['workers'] + 1])
        R = A[A.video == q].reset_index(drop=True)
        N, W, H = int(V.loc[q, 'frames']), int(V.loc[q, 'W']), int(V.loc[q, 'H'])
        vc = torch.from_numpy(video_cells(W, H)).to(dev).float()
        bgi = np.linspace(0, N - 1, NBG).round().astype(int)
        with torch.no_grad():
            bg = mk.background([fr[x] for x in bgi], W, H)
            vv = R.v.values
            keep_all = []
            for b0 in range(0, len(vv), ce['bs']):
                x = canvas_gpu([fr[x_] for x_ in vv[b0:b0 + ce['bs']]], W, H, dev).to(torch.uint8)
                fd = mk.fraction(x, bg, vc)
                keep = fd >= THR
                z = enc(x)
                for ii in range(len(x)):
                    p_ = torch.nonzero(keep[ii]).flatten()
                    fid = len(frames) + b0 + ii
                    ftok.write(z[ii, p_].cpu().numpy().tobytes())
                    toks.append(np.full(len(p_), fid, np.int32))
                    poss.append(p_.cpu().numpy().astype(np.int16))
                    fds.append(fd[ii, p_].cpu().numpy().astype(np.float16))
                keep_all.append(keep.cpu().numpy())
                if len(sheet) < 8 and b0 == 0 and len(sheet) < k + 1:
                    sheet.append((x[0].cpu().numpy(), keep[0].cpu().numpy(), f'{q[:22]} v{vv[0]} kept {int(keep[0].sum())}'))
                del x, z
        km = np.concatenate(keep_all)
        cell = track_cell(tpos[vv], W, H)
        for j, r in enumerate(R.itertuples()):
            row = {'video': q, 'v': int(r.v), 'half': r.half, 'is_train': r.is_train, 'is_calib': r.is_calib,
                   'is_fve': r.is_fve, 'is_eval': r.is_eval, 'is_evn': r.is_evn, 'n_kept': int(km[j].sum()),
                   'nm': int(nm[r.v]), 'strain': V.loc[q, 'strain']}
            for a_, nmx in ((0, 'res'), (1, 'int')):
                cc = cell[j, a_]
                ok = (nm[r.v] > a_) and cc >= 0
                row[f'has_{nmx}'] = bool(ok)
                row[f'rec_{nmx}'] = bool(km[j, cc]) if ok else False
            frames.append(row)
        el = time.time() - t00
        log(f'  video {k + 1}/{len(vids)} {q}: {len(vv)} frames, kept/frame {km.sum(1).mean():.1f}, '
            f'{len(frames) / el:.1f} frames/s wall, encoder GPU busy {enc.gpu_s:.0f}s ({enc.n_frames / max(enc.gpu_s, 1e-9):.1f} '
            f'frames per GPU-s); read+decode {tdec:.0f}s summed, waited {twait:.0f}s, {nbytes / 2**30:.2f} GB read')
        del fr
    pool.shutdown()
    ftok.close()
    Fm, tf = RUN.write_store(loc, frames, toks, poss, fds, G)
    wall = time.time() - t00
    info = {'shard': a.shard, 'nshard': a.nshard, 'videos': len(vids), 'frames': len(Fm), 'n_tokens': int(len(tf)),
            'tokens_per_frame': float(Fm.n_kept.mean()), 'frames_0_kept': int((Fm.n_kept == 0).sum()),
            'wall_s': wall, 'frames_per_wall_s': len(Fm) / wall, 'encoder_gpu_s': enc.gpu_s,
            'frames_per_encoder_gpu_s': enc.n_frames / max(enc.gpu_s, 1e-9), 'read_decode_s_summed': tdec,
            'decode_wait_s': twait, 'gb_read': nbytes / 2 ** 30, 'gpu': torch.cuda.get_device_name(0),
            'peak_mem_gb': torch.cuda.max_memory_allocated() / 2 ** 30}
    (loc / 'info.json').write_text(json.dumps(info, indent=1))
    od = OUT / ('encode_selftest' if a.selftest else 'encode')
    od.mkdir(parents=True, exist_ok=True)
    (od / f'{tag}.json').write_text(json.dumps(info, indent=1))
    mask_sheet(sheet, od / f'mask_sheet_{tag}.jpg', G)
    RUN.atomic_copy(loc, dst)
    log('done: ' + json.dumps(info))


# ---------------------------------------------------------------------------------------------- stores
def stores(selftest=False):
    """Stage every shard store to $LOCAL_DIR -> list of run.Store (frames table gets a global key)."""
    root = BEE / ('tok_selftest' if selftest else 'tok')
    out = []
    for d in sorted(root.glob('shard*')):
        if d.name.endswith('.tmp'):
            continue
        loc = RUN.local_dir() / d.name
        if not loc.exists():
            RUN.atomic_copy(d, loc)
        out.append(RUN.Store(loc, G))
    assert out, root
    if not selftest:
        n = {int(d.root.name.split('of')[1]) for d in out}
        assert len(n) == 1 and len(out) == n.pop(), [d.root.name for d in out]
    return out


def cmd_train(a):
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    St = stores(a.selftest)
    X = {}
    for role, col in (('train', 'is_train'), ('calib', 'is_calib'), ('eval', 'is_fve')):
        X[role] = torch.cat([s.inputs(0, s.F[col].values, dev) for s in St])
    info = {r: int(len(x)) for r, x in X.items()}
    info['frames'] = {r: int(sum(s.F[c].sum() for s in St)) for r, c in (('train', 'is_train'), ('calib', 'is_calib'),
                                                                         ('eval', 'is_fve'))}
    log(f'SAE inputs (tokens): {json.dumps(info)}')
    cfg = {**SPEC['sae'], 'context_window': SPEC['context_window']}
    if a.steps:
        cfg['steps'] = a.steps
    tag = '_selftest' if a.selftest else ''
    tf = OUT / f'train{tag}.json'
    recs = {}
    if tf.exists() and not a.force:
        recs = {int(k): v for k, v in json.loads(tf.read_text()).get('seeds', {}).items()}
    for seed in SPEC['seeds']:
        od = BEE / f'sae{tag}' / sae_name(seed)
        if (od / 'sae_theta.pt').exists() and seed in recs and not a.force:
            continue
        recs[seed] = SA.train(cfg, seed, X['train'], X['calib'], X['eval'], od, dev)
        r = recs[seed]
        log(f'{sae_name(seed)}: {r["train_time_s"]:.0f}s ({r["steps_per_s"]:.1f} steps/s), theta* {r["theta_star"]:.4f}, '
            f'held-out FVE {r["eval_fve"]:.4f}, L0 {r["eval_l0"]:.2f}, dead at end {r["dead_at_end_train"]}')
        OUT.mkdir(parents=True, exist_ok=True)
        tf.write_text(json.dumps({'tokens': info, 'seeds': recs}, indent=1, default=float))
    best = SA.deploy_seed(recs)
    b = {'name': sae_name(best), 'seed': best, 'eval_fve': recs[best]['eval_fve'], 'theta_star': recs[best]['theta_star'],
         'rule': 'highest held-out FVE among the spec seeds (label-free); held-out = test-half role calib frames',
         'dir': str(BEE / f'sae{tag}' / sae_name(best))}
    (OUT / f'best{tag}.json').write_text(json.dumps(b, indent=1))
    log(f'deployed {b}')


# ---------------------------------------------------------------------------------------------- score
def frame_codes_of(St, sae, norm, dev, col='is_evn', chunk=256):
    """-> codes (n, W) fp16 numpy of the frames of each store where col holds, keys DataFrame (video, v)."""
    import torch
    from src.eci.v2 import sae as SA
    W = sae.n_latents
    outs, keys = [], []
    for s in St:
        sel = np.flatnonzero(s.F[col].values)
        cm = np.zeros((len(sel), W), np.float16)
        for c0 in range(0, len(sel), chunk):
            ff = sel[c0:c0 + chunk]
            parts, fl = [], []
            for i, f in enumerate(ff):
                a_, b_ = int(s.starts[f]), int(s.starts[f + 1])
                parts.append(np.asarray(s.tok[a_:b_]))
                fl.append(np.full(b_ - a_, i, np.int64))
            x = torch.from_numpy(np.concatenate(parts)).to(dev).float()
            fl_ = torch.from_numpy(np.concatenate(fl)).to(dev)
            c, _, _ = SA.frame_codes(sae, norm, x, fl_, len(ff))
            cm[c0:c0 + len(ff)] = c.half().cpu().numpy()
        outs.append(cm)
        keys.append(s.F.iloc[sel][['video', 'v']])
    return np.concatenate(outs), pd.concat(keys, ignore_index=True)


def smoothed_eval(codes, keys, E, V):
    """Eval frames E (plan order) -> raw (n, W) and smoothed (n, W) fp16: centred mean over v - 5, v, v + 5 (those
    inside the video) = readout.smooth h 1 on the 5 fps grid."""
    from src.eci.v2 import readout as RO
    idx = pd.MultiIndex.from_frame(keys)
    ix = {d: idx.get_indexer(pd.MultiIndex.from_arrays([E.video.values, E.v.values + d])) for d in (-STEP, 0, STEP)}
    assert (ix[0] >= 0).all() and (ix[-STEP] >= 0).all()
    N = V.frames.reindex(E.video).values
    assert ((ix[STEP] >= 0) == (E.v.values + STEP < N)).all()
    raw = codes[ix[0]].astype(np.float32)
    acc = raw + codes[ix[-STEP]].astype(np.float32)
    has = ix[STEP] >= 0
    acc[has] += codes[ix[STEP][has]].astype(np.float32)
    sm = acc / np.where(has, 3, 2)[:, None]
    # check against the library smoothing on a few triplets
    for i in np.r_[np.flatnonzero(has)[:5], np.flatnonzero(~has)[:2]]:
        seq = [codes[ix[-STEP][i]], codes[ix[0][i]]] + ([codes[ix[STEP][i]]] if has[i] else [])
        ref = RO.smooth(np.stack(seq).astype(np.float32), 1)[1]
        assert np.allclose(ref, sm[i], atol=1e-6), i
    return raw.astype(np.float16), sm.astype(np.float16), int((~has).sum())


def rtilde(m):
    """one behaviour's AT.score 'main' entry -> R~ of its R-precision-picked neuron on the scoring half."""
    return (m['rprec']['test_rprec'] - m['base_test']) / (1 - m['base_test'])


def summarise(res):
    """two direction results (AT.score) -> per behaviour: R~ (R-prec pick), AUROC (AUROC pick), both directions."""
    out = {}
    for b in SCORED:
        d = [res[k]['main'][b] for k in ('A', 'B')]
        out[b] = {'R': float(np.mean([rtilde(x) for x in d])), 'R_dirs': [rtilde(x) for x in d],
                  'auroc': float(np.mean([x['auc']['test_auc'] for x in d])),
                  'R_at_auroc_pick': float(np.mean([(x['auc']['test_rprec'] - x['base_test']) / (1 - x['base_test'])
                                                    for x in d])),
                  'neurons_R': [x['rprec']['neuron'] for x in d], 'neurons_auroc': [x['auc']['neuron'] for x in d],
                  'rate_R_pick': float(np.mean([x['rprec']['test_rate'] for x in d])),
                  'base_test': [x['base_test'] for x in d]}
    out['_mean_all'] = float(np.mean([out[b]['R'] for b in SCORED]))
    out['_mean_pair'] = float(np.mean([out[b]['R'] for b in PAIR]))
    out['_auroc_mean_pair'] = float(np.mean([out[b]['auroc'] for b in PAIR]))
    return out


def cmd_score(a):
    import torch
    import align_train as AT
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    P, V = plan()
    St = stores(a.selftest)
    tag = '_selftest' if a.selftest else ''
    E = P[P.role == 'eval'].reset_index(drop=True)
    if a.selftest:
        have = pd.concat([s.F[s.F.is_eval][['video', 'v']] for s in St])
        E = E.merge(have, on=['video', 'v']).reset_index(drop=True)
    Y = torch.from_numpy(E[SCORED].values.astype(bool)).to(dev)
    w = torch.from_numpy(E.w.values.astype(np.float64)).to(dev)
    tr = json.loads((OUT / f'train{tag}.json').read_text())['seeds']
    best = json.loads((OUT / f'best{tag}.json').read_text())
    for seed in SPEC['seeds']:
        t0 = time.time()
        sae, norm = SA.load(BEE / f'sae{tag}' / sae_name(seed) / 'sae_theta.pt', dev)
        codes, keys = frame_codes_of(St, sae, norm, dev)
        raw, sm, n_trunc = smoothed_eval(codes, keys, E, V)
        rec = {'seed': seed, 'sae': sae_name(seed), 'deployed': seed == best['seed'], 'eval_frames': len(E),
               'eval_frames_truncated_window': n_trunc, 'train': {k: tr[str(seed)][k] for k in
                                                                 ('eval_fve', 'eval_l0', 'theta_star', 'dead_at_end_train',
                                                                  'train_time_s', 'n_train_tokens')}}
        wt = E.w.values / E.w.values.sum()
        for nm, X in (('smooth', sm), ('raw', raw)):
            Xg = torch.from_numpy(X).to(dev)
            res = {}
            for fold, (sel, tst) in {'A': ('train', 'test'), 'B': ('test', 'train')}.items():
                if a.selftest and not ((E.half == sel).any() and (E.half == tst).any()):
                    continue
                res[fold] = AT.score(Xg, E, Y, w, sel, tst, dev)
            rec[nm] = {'per_direction': res, 'summary': summarise(res) if len(res) == 2 else None}
            fr = ((X > 0) * wt[:, None]).sum(0)
            rec[nm]['neurons_used'] = {'active_on_any_eval_frame': int((X > 0).any(0).sum()),
                                       'weighted_rate_ge_1pct': int((fr >= 0.01).sum()), 'W': int(X.shape[1]),
                                       'mean_active_per_eval_frame': float((X > 0).sum(1).mean())}
            del Xg
        log(f'seed {seed}: ' + (f'R~ pair {rec["smooth"]["summary"]["_mean_pair"]:.3f}, all '
                                f'{rec["smooth"]["summary"]["_mean_all"]:.3f}; sniff R~ {rec["smooth"]["summary"]["sniff"]["R"]:.3f} '
                                f'AUROC {rec["smooth"]["summary"]["sniff"]["auroc"]:.3f}' if rec['smooth']['summary'] else '')
            + f' ({time.time() - t0:.0f}s)')
        (OUT / f'score{tag}_s{seed}.json').write_text(json.dumps(rec, indent=1, default=float))
        if seed == best['seed'] and rec['smooth']['summary']:
            sheets(St, sae, norm, E, sm, rec, V, dev, tag)
        del codes, raw, sm
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- contact sheets
def sheets(St, sae, norm, E, sm, rec, V, dev, tag, n_top=24, crop=224, tile=300):
    """Deployed seed; behaviours sniff / attack / approach: the R~-picked neuron of direction A (picked on the TRAIN
    half's eval frames); its n_top highest smoothed values on the TEST half's eval frames (held out), <= 1 frame per
    video; per tile the frame's per-token ACTIVATION map (heatmap.neuron_maps, frame v itself, no smoothing) over a
    crop x crop canvas-px crop centred on the peak token; green border = labelled positive, red = not (label written)."""
    import torch
    from matplotlib import cm as mcm
    from PIL import Image, ImageDraw
    from align_common import geom
    from src.eci.v2.heatmap import neuron_maps
    sd = OUT / f'sheets{tag}'
    sd.mkdir(parents=True, exist_ok=True)
    loc = {}
    for si, s in enumerate(St):
        for fi, (q, v) in enumerate(zip(s.F.video.values, s.F.v.values)):
            loc[(q, int(v))] = (si, fi)
    lab = E[SCORED + ['other', 'human']].values.argmax(1)
    names = SCORED + ['other', 'human']
    out = {}
    for b in ('sniff', 'attack', 'approach'):
        j = rec['smooth']['per_direction']['A']['main'][b]['rprec']['neuron']
        ht = np.flatnonzero(E.half.values == 'test')
        order = ht[np.argsort(-sm[ht, j].astype(np.float32), kind='stable')]
        pick, seen = [], set()
        for i in order:
            if E.video.values[i] in seen:
                continue
            seen.add(E.video.values[i])
            pick.append(i)
            if len(pick) == n_top:
                break
        tiles, meta = [], []
        for i in pick:
            q, v = E.video.values[i], int(E.v.values[i])
            si, fi = loc[(q, v)]
            s = St[si]
            a_, b_ = int(s.starts[fi]), int(s.starts[fi + 1])
            z = torch.from_numpy(np.asarray(s.tok[a_:b_])).to(dev).float()
            pos = torch.from_numpy(s.pos[a_:b_]).to(dev)
            A_, _ = neuron_maps(sae, norm, z, pos, G, 0, j) if b_ > a_ else (np.zeros(G * G, np.float32), None)
            W, H = int(V.loc[q, 'W']), int(V.loc[q, 'H'])
            d = ROOT / V.loc[q, 'half']
            fr, _ = read_frames(d / f'{q}_t.seq', load_seek(d / f'{q}_t-seek.mat'), [v])
            cv = canvas_gpu([fr[v]], W, H, dev)[0, 0].cpu().numpy()       # 640 x 640 canvas grey
            hm = np.kron(A_.reshape(G, G), np.ones((16, 16), np.float32))
            pk = int(np.argmax(A_)) if A_.max() > 0 else None
            cy, cx = ((pk // G) * 16 + 8, (pk % G) * 16 + 8) if pk is not None else (320, 320)
            pt, _ = geom(W, H)
            y0 = int(np.clip(cy - crop // 2, 0, 640 - crop))
            x0 = int(np.clip(cx - crop // 2, 0, 640 - crop))
            g = cv[y0:y0 + crop, x0:x0 + crop]
            h = hm[y0:y0 + crop, x0:x0 + crop]
            im = np.repeat(g[:, :, None], 3, 2)
            if A_.max() > 0:
                al = (h / A_.max())[:, :, None]
                col = mcm.inferno(np.clip(h / A_.max(), 0, 1))[:, :, :3] * 255
                im = np.where(al > 0, (1 - 0.55 * al) * im + 0.55 * al * col, im)
            t = Image.fromarray(im.clip(0, 255).astype(np.uint8)).resize((tile, tile), Image.BILINEAR)
            pos_ = bool(E[b].values[i])
            dr = ImageDraw.Draw(t)
            dr.rectangle([0, 0, tile - 1, tile - 1], outline=(0, 220, 0) if pos_ else (230, 0, 0), width=5)
            dr.text((8, 6), f'{q[:24]}', fill=(255, 255, 0))
            dr.text((8, 20), f'v{v} s={float(sm[i, j]):.2f} max a={A_.max():.2f}', fill=(255, 255, 0))
            dr.text((8, tile - 18), f'label: {names[lab[i]]}', fill=(0, 255, 0) if pos_ else (255, 120, 120))
            tiles.append(t)
            meta.append({'video': q, 'v': v, 'smoothed': float(sm[i, j]), 'frame_max_activation': float(A_.max()),
                         'label': names[lab[i]], 'positive': pos_, 'kept_tokens': int(b_ - a_),
                         'tokens_firing': int((A_ > 0).sum())})
        cols = 6
        sh = Image.new('RGB', (cols * tile, tile * int(np.ceil(len(tiles) / cols)) + 28), (20, 20, 20))
        ImageDraw.Draw(sh).text((8, 8), f'CRIM13 ECI v2 {rec["sae"]}: {b} neuron {j} (R~ pick on the train half); '
                                        f'{len(tiles)} top held-out test-half eval frames, <= 1 per video; per-token '
                                        f'activation of frame v (no smoothing in the map); green = {b}',
                                fill=(255, 255, 255))
        for k, t in enumerate(tiles):
            sh.paste(t, ((k % cols) * tile, 28 + (k // cols) * tile))
        p = sd / f'{b}_neuron{j}_{rec["sae"]}.jpg'
        sh.save(p, quality=88)
        out[b] = {'path': str(p), 'neuron': j, 'n_positive_of_top': int(sum(m['positive'] for m in meta)),
                  'n_tiles': len(tiles), 'frames': meta,
                  'note': 'eval frames are stratified by label (enriched for rare behaviours), so the share of '
                          'positives in the sheet is not a precision at the natural base rate'}
        log(f'sheet {b}: neuron {j}, {out[b]["n_positive_of_top"]}/{len(tiles)} labelled {b} -> {p}')
    (sd / 'sheets.json').write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------- report
def stored_arm(arm, root):
    """Stored CRIM13 arm (per-fold SAEs, unsmoothed frame max): per seed the fold-mean R~ / AUROC per behaviour."""
    import re
    R = {}
    for f in sorted((root / 'eval').glob(f'{arm}_?_s*.json')):
        m = re.fullmatch(rf'{arm}_([AB])_s(\d+)\.json', f.name)
        if m:
            R.setdefault(int(m.group(2)), {})[m.group(1)] = json.loads(f.read_text())
    R = {s: v for s, v in R.items() if len(v) == 2}
    if not R:
        return None
    out = {}
    for b in SCORED:
        rs = [np.mean([rtilde(v[fo]['main'][b]) for fo in 'AB']) for v in R.values()]
        au = [np.mean([v[fo]['main'][b]['auc']['test_auc'] for fo in 'AB']) for v in R.values()]
        out[b] = {'R_mean': float(np.mean(rs)), 'R_sd': float(np.std(rs, ddof=1)), 'auroc_mean': float(np.mean(au)),
                  'n_seeds': len(rs)}
    for grp, bs in (('_all', SCORED), ('_pair', PAIR)):
        per = [np.mean([np.mean([rtilde(v[fo]['main'][b]) for fo in 'AB']) for b in bs]) for v in R.values()]
        out[grp] = {'R_mean': float(np.mean(per)), 'R_sd': float(np.std(per, ddof=1))}
    return out


def cmd_report(a):
    P, V = plan()
    sc = {s: json.loads((OUT / f'score_s{s}.json').read_text()) for s in SPEC['seeds']}
    tr = json.loads((OUT / 'train.json').read_text())
    best = json.loads((OUT / 'best.json').read_text())
    E = P[P.role == 'eval']
    base = {b: {h: float(np.average(E[E.half == h][b], weights=E[E.half == h].w)) for h in ('train', 'test')}
            for b in SCORED}
    v2 = {}
    for nm in ('smooth', 'raw'):
        v2[nm] = {}
        for b in SCORED + ['_mean_all', '_mean_pair']:
            vals = [sc[s][nm]['summary'][b]['R'] if not b.startswith('_') else sc[s][nm]['summary'][b]
                    for s in SPEC['seeds']]
            au = [sc[s][nm]['summary'][b]['auroc'] for s in SPEC['seeds']] if not b.startswith('_') else None
            v2[nm][b] = {'R_mean': float(np.mean(vals)), 'R_sd': float(np.std(vals, ddof=1)), 'R_per_seed': vals,
                         'R_deployed': sc[best['seed']][nm]['summary'][b]['R'] if not b.startswith('_')
                         else sc[best['seed']][nm]['summary'][b]}
            if au:
                v2[nm][b].update(auroc_mean=float(np.mean(au)), auroc_sd=float(np.std(au, ddof=1)),
                                 auroc_deployed=sc[best['seed']][nm]['summary'][b]['auroc'])
    root = REPO / 'results/vision/benchmark_crim13/align'
    arms = {}
    for arm, r in [(x, root / 'knobs') for x in ('B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7')] + \
                  [(x, root) for x in ('C1', 'C2', 'C3', 'C4')]:
        s_ = stored_arm(arm, r)
        if s_:
            arms[arm] = s_
    prev_best = {}
    for b in SCORED + ['_all', '_pair']:
        k = 'R_mean'
        a_ = max(arms, key=lambda x: arms[x][b][k])
        prev_best[b] = (a_, arms[a_][b][k])
    # mask check (the tight mask vs the released tracks, eval frames of both halves)
    Fm = pd.concat([pd.read_parquet(d / 'frames.parquet') for d in sorted((BEE / 'tok').glob('shard*'))
                    if not d.name.endswith('.tmp')])
    Fe = Fm[Fm.is_eval]
    mask = {'tokens_per_frame_eval': float(Fe.n_kept.mean()), 'eval_frames_0_kept': int((Fe.n_kept == 0).sum()),
            'resident_recall': float(Fe.rec_res[Fe.has_res].mean()),
            'intruder_recall_BC': float(Fe.rec_int[Fe.has_int & (Fe.strain == 'BC')].mean()),
            'intruder_recall_C57': float(Fe.rec_int[Fe.has_int & (Fe.strain == 'C57')].mean()),
            'n_eval_frames': int(len(Fe))}
    old_gate = json.loads((root / 'res/mask_gate.json').read_text())
    enc = [json.loads(f.read_text()) for f in sorted((OUT / 'encode').glob('shard*.json'))]
    sd = tr['seeds']
    health = {s: {'held_out_FVE': sd[str(s)]['eval_fve'], 'L0': sd[str(s)]['eval_l0'],
                  'dead_at_end_train': sd[str(s)]['dead_at_end_train'],
                  'neurons_active_on_eval_frames': sc[s]['raw']['neurons_used']['active_on_any_eval_frame'],
                  'neurons_weighted_rate_ge_1pct': sc[s]['raw']['neurons_used']['weighted_rate_ge_1pct'],
                  'mean_active_neurons_per_eval_frame_raw': sc[s]['raw']['neurons_used']['mean_active_per_eval_frame']}
              for s in SPEC['seeds']}
    rep = {'v2': v2, 'stored_arms': arms, 'previous_best': prev_best, 'base_rate': base, 'mask': mask,
           'old_mask_gate': old_gate, 'sae_health': health, 'deployed': best, 'tokens': tr['tokens'], 'encode': enc}
    (OUT / 'report.json').write_text(json.dumps(rep, indent=1, default=float))
    L = ['# ECI v2 (frozen) on CRIM13 - validation check', '',
         f'Deployed SAE {best["name"]} (held-out FVE {best["eval_fve"]:.4f}). R~ = normalised R-precision (P - pi) / '
         '(1 - pi) of the R-precision-picked neuron, picked on one released half and scored on the other, mean of both '
         'directions; AUROC of the AUROC-picked neuron, same cross-fit. Weighted (inverse-probability) eval frames of '
         'the CRIM13 alignment plan. v2 cells: mean ± sd over the 3 SAE seeds (smoothed 0.6 s readout). Stored arms: '
         'per-fold SAEs, unsmoothed frame max, mean over their seeds.', '',
         '| behaviour | base rate (train / test half, weighted) | v2 R~ | v2 R~ deployed | v2 R~ unsmoothed | v2 AUROC | '
         'B0 R~ | B2 R~ | C1 R~ | previous best R~ (arm) |', '|' + '---|' * 10]
    for b in SCORED:
        L.append(f'| {b} | {base[b]["train"]:.4f} / {base[b]["test"]:.4f} | {v2["smooth"][b]["R_mean"]:.3f} ± '
                 f'{v2["smooth"][b]["R_sd"]:.3f} | {v2["smooth"][b]["R_deployed"]:.3f} | {v2["raw"][b]["R_mean"]:.3f} | '
                 f'{v2["smooth"][b]["auroc_mean"]:.3f} | {arms["B0"][b]["R_mean"]:.3f} | {arms["B2"][b]["R_mean"]:.3f} | '
                 f'{arms["C1"][b]["R_mean"]:.3f} | {prev_best[b][1]:.3f} ({prev_best[b][0]}) |')
    for g, k in (('mean over 11 behaviours', ('_mean_all', '_all')), ('mean over 7 pair behaviours', ('_mean_pair', '_pair'))):
        L.append(f'| **{g}** | | {v2["smooth"][k[0]]["R_mean"]:.3f} ± {v2["smooth"][k[0]]["R_sd"]:.3f} | '
                 f'{v2["smooth"][k[0]]["R_deployed"]:.3f} | {v2["raw"][k[0]]["R_mean"]:.3f} | | '
                 f'{arms["B0"][k[1]]["R_mean"]:.3f} | {arms["B2"][k[1]]["R_mean"]:.3f} | {arms["C1"][k[1]]["R_mean"]:.3f} | '
                 f'{prev_best[k[1]][1]:.3f} ({prev_best[k[1]][0]}) |')
    L += ['', '## SAE health', '', '| seed | held-out FVE | L0 | dead at end of training | neurons active on >= 1 eval '
          'frame | neurons with weighted rate >= 1% |', '|---|---|---|---|---|---|']
    for s, h in health.items():
        L.append(f'| {s}{" (deployed)" if s == best["seed"] else ""} | {h["held_out_FVE"]:.4f} | {h["L0"]:.2f} | '
                 f'{h["dead_at_end_train"]} | {h["neurons_active_on_eval_frames"]} | {h["neurons_weighted_rate_ge_1pct"]} |')
    L += ['', '## tight mask (eval frames, released track points; never used to build the mask)', '',
          f'kept tokens per eval frame {mask["tokens_per_frame_eval"]:.1f}; eval frames with 0 kept {mask["eval_frames_0_kept"]}; '
          f'own-cell recall: resident {mask["resident_recall"]:.3f}, BC (pale) intruder {mask["intruder_recall_BC"]:.3f}, '
          f'C57 (dark) intruder {mask["intruder_recall_C57"]:.3f}. Old dilated dark mask (all plan frames): resident '
          f'{old_gate["resident"]:.3f}, BC {old_gate["BC"]:.3f}, C57 {old_gate["C57"]:.3f}.']
    (OUT / 'report.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('encode', 'train', 'score', 'report'))
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=C['encode']['n_shards'])
    ap.add_argument('--max-videos', type=int, default=0)
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'encode': cmd_encode, 'train': cmd_train, 'score': cmd_score, 'report': cmd_report}[a.step](a)


if __name__ == '__main__':
    main()
