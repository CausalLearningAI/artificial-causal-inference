"""
Native-resolution pilot for mice ECI: does feeding DINOv2 the videos' REAL pixel detail (instead of the 512 px
standardised frames) give cleaner behaviour neurons with the same clean pipeline?
    frames -> DINOv2-base patch tokens -> the deployed fg448 animal mask -> Matryoshka BatchTopK SAE per patch
    -> frame code of a neuron = its max over the frame's kept patches.
Only the input changes (the 'scale prior'). Three arms on IDENTICAL frames and IDENTICAL kept regions:
    R448  512 px frame -> 448 px (32 x 32 patches, 1 patch = 16 px of the 512 frame = ~64 native px). Deployed input.
    U896  512 px frame upsampled to 896 (64 x 64): finer grid, NO new detail (the control).
    N896  native 2064 px source frame downscaled to 896 (64 x 64): finer grid WITH real detail (~32 native px / patch).
Mask: the store's fg448 mask of the frame; each 448 patch -> its 2 x 2 children at 896 (same frame area in every arm).

Frames
    train  100 store frames (1 fps, evenly spaced) of each of the 100 UNANNOTATED train videos of the SAE-lever split
           (spatial_sae_pilot.StoreIndex.split('mice', 100, seed 0)) = 10,000 frames. No label is used in training.
    eval   the 20,000-frame annotated B subset of scripts/eci/diag_mice_patches.py (results/vision/eci_repr_diag/
           mice/b: all 5,394 positive frames of either behaviour + 14,606 random negatives, inverse-sampling weights
           restore the natural base rate; 144 videos, 24 pools). Its R448 / U896 tokens are reused from there
           (tok448.npy = the store tokens re-encoded, tok896.npy = the 512 JPG bicubic-upsampled by the HF processor).
Native decoding (CPU job array): source frame of 512 px frame k = experiment.csv start_frame + 6 k + 2 (offset
verified by scripts/eci/split_test_native.py: 23/23 checks). Per video the source mp4 is copied to the job's local
disk, the wanted frames are grouped into clusters (consecutive wanted frames <= 350 source frames apart: decoding 350
frames sequentially costs about one seek), and each cluster is one ffmpeg call (accurate input seek, select the
wanted frames, the standardisation's scale+pad chain at 896, JPEG q 2, single-threaded, a pool of processes).
Check per video: 3 frames resized 896 -> 512 vs the dataset JPEG of the same frame and of its neighbours k +- 1
(mean absolute grey difference must be lowest at k).

SAEs: the levers recipe (sae_levers.train_one, config 'w4096': 4096 latents, prefixes 512/1024/2048/4096, k = 16,
6000 steps x 4096 tokens, Adam 5e-4, TokenNorm, geometric-median b_dec, AuxK), 3 seeds per arm, ALL train tokens of
the arm as the pool (R448 ~1.5M unique tokens, the 896 arms ~6M; same steps and batch, so presentations are equal).
Eval (weighted to the natural base rate): per-neuron AUROC / AP / top-1% precision with weights, cross-fitted over two
pool halves (24 pools sorted, alternating): select on half A (best AUROC, best AP, best top-1% among neurons firing on
>= 1% of the half's weighted frames = honest top-1%), score on half B and the reverse, average. Size-controlled AUROC:
the AUROC-selected neuron inside fg-count deciles of the test half (deciles with >= 10 positive and >= 10 negative
frames), averaged. FVE on the eval tokens in each arm's normalised input space.
Supervised ceiling: diag_mice_patches mil_ctx (gated-attention MIL + 3x3 context, 5 pool-grouped folds, 3 seeds) on
the N896 tokens; R448 / U896 ceilings are that script's existing B results on the same frames, folds and seeds.

Steps (scripts/eci/native_res.sh):
    plan      CPU  frame lists, train R448 tokens + masks from the store, decode plan, align_w self-test
    decode    CPU  array task SLURM_ARRAY_TASK_ID of the decode plan -> OUT/native/task_XX.{tar,json}
    encode    GPU  U896 / N896 tokens of the train frames, N896 tokens of the B frames (+ per-patch grey of N896)
    train     GPU  3 arms x 3 seeds SAEs -> OUT/sae/<arm>_s<seed>/
    evaluate  GPU  codes, weighted cross-fitted scores, contact sheets (N896 from the native frames), identity screen
    mil       GPU  mil_ctx ceiling on the N896 B tokens
    summary   any  OUT/summary.json + table (login-node safe: reads JSONs only)
Outputs in results/vision/eci_native_res/ (gitignored).
"""
import argparse
import io
import json
import os
import resource
import shutil
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
OUT = REPO / 'results/vision/eci_native_res'
BDIR = REPO / 'results/vision/eci_repr_diag/mice/b'
ANN = REPO / 'dataset/mice/v1/annotations.csv'
EXP = REPO / 'data/mice/v1/experiment.csv'
SRC = REPO / 'data/mice/source'
FPS, STEP, OFFSET = 30.0, 6, 2
GAP = 350  # max source-frame gap inside one decode cluster (smoke test: a seek costs 6.8 CPU-s, sequential decode 56 frames/CPU-s)
SIZE = 896
ARMS = ['r448', 'u896', 'n896']
BEH = ['nose_nose', 'nose_tail']
MIN_RATE = 0.01
D = 768
NB = 0x7C01
log = lambda s: print(s, flush=True)  # noqa: E731


def stage_dir():
    d = Path(os.environ.get('STAGE') or '')
    if not str(d) or not str(d).startswith('/localhome'):
        raise RuntimeError(f'STAGE must be the job dir under /localhome, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


def children_896(pos448):
    """448 grid positions (sorted) -> their 2x2 children on the 64 x 64 grid, sorted (diag_mice_patches mapping)."""
    p = np.asarray(pos448, np.int64)
    r, c = p // 32, p % 32
    return np.sort(np.concatenate([(2 * r + dr) * 64 + 2 * c + dc for dr in (0, 1) for dc in (0, 1)]))


# ---------------------------------------------------------------------------------------------- weighted alignment
def align_w(X, Y, w, rows=None, top_frac=0.01, block=256, cols=None):
    """Weighted version of src.eci.levers.align_gpu (same histogram-over-fp16-bits method, ties averaged).
    X (n, m) fp16 >= 0 GPU, Y (n, L) bool, w (n,) float weights. Top-1% = the highest-coded frames holding 1% of the
    total weight (frames tied at the cut-off pro rata); rate = weighted fraction of frames with a non-zero code."""
    import torch
    with torch.no_grad():
        dev = X.device
        if rows is not None:
            Y, w = Y[rows], w[rows]
        n, L = Y.shape
        w = w.double()
        W = float(w.sum())
        cols = torch.arange(X.shape[1], device=dev) if cols is None else cols
        m = len(cols)
        P = (w[:, None] * Y.double()).sum(0)
        N = W - P
        auc = torch.full((m, L), float('nan'), dtype=torch.float64, device=dev)
        ap, ptop = auc.clone(), auc.clone()
        rate = torch.zeros(m, dtype=torch.float64, device=dev)
        for j0 in range(0, m, block):
            c = cols[j0:j0 + block]
            b = len(c)
            xb = X.index_select(1, c)
            if rows is not None:
                xb = xb.index_select(0, rows)
            if (xb < 0).any() or not torch.isfinite(xb).all():
                raise ValueError('codes must be finite and >= 0')
            bits = (xb.contiguous().view(torch.int16).long() & 0x7FFF) + NB * torch.arange(b, device=dev)[None]
            del xb
            wb = w[:, None].expand(n, b)
            tot = torch.bincount(bits.flatten(), weights=wb.flatten(), minlength=b * NB).view(b, NB).double()
            rate[j0:j0 + b] = 1 - tot[:, 0] / W
            tot_d = tot.flip(1)
            ccnt = tot_d.cumsum(1)
            nnz = (W - tot[:, 0]).clamp_min(0)
            k = torch.minimum(torch.full_like(nnz, top_frac * W), nnz * (1 - 1e-9))
            ib = torch.searchsorted(ccnt.contiguous(), k[:, None]).clamp_max(NB - 1)
            for li in range(L):
                if P[li] <= 0 or N[li] <= 0:
                    continue
                yl = Y[:, li]
                pos = torch.bincount(bits[yl].flatten(), weights=wb[yl].flatten(), minlength=b * NB).view(b, NB)
                pos = pos.double().flip(1)
                neg = tot_d - pos
                ctp, cfp = pos.cumsum(1), neg.cumsum(1)
                auc[j0:j0 + b, li] = (pos * (N[li] - cfp) + 0.5 * pos * neg).sum(1) / (P[li] * N[li])
                prec = torch.where(ctp + cfp > 0, ctp / (ctp + cfp).clamp_min(1e-300), torch.zeros_like(ctp))
                ap[j0:j0 + b, li] = (pos / P[li] * prec).sum(1)
                cb, tb, pb = ccnt.gather(1, ib)[:, 0], tot_d.gather(1, ib)[:, 0], pos.gather(1, ib)[:, 0]
                tp = ctp.gather(1, ib)[:, 0] - pb + pb * (k - (cb - tb)) / tb.clamp_min(1e-300)
                pt = tp / k.clamp_min(1e-300)
                ptop[j0:j0 + b, li] = torch.where(k > 0, pt, torch.full_like(pt, float('nan')))
            del bits
        return {'auc': auc.cpu().numpy(), 'ap': ap.cpu().numpy(), 'prec_top': ptop.cpu().numpy(),
                'rate': rate.cpu().numpy(), 'base': (P / W).cpu().numpy(), 'n': n, 'W': W}


def selftest_align():
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from src.eci.levers import align_gpu
    g = np.random.default_rng(0)
    n, m = 4000, 12
    X = np.where(g.random((n, m)) < 0.3, g.gamma(1.0, 1.0, (n, m)), 0).astype(np.float16)
    X[:, 3] = np.round(X[:, 3] * 2) / 2  # ties
    Y = g.random((n, 2)) < np.array([0.03, 0.2])
    Y[:, 0] |= X[:, 1] > 2.5
    w = np.where(Y.any(1), 1.0, 11.46)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    a1 = align_w(Xt, Yt, torch.ones(n))
    a0 = align_gpu(Xt, Yt)
    for q in ('auc', 'ap', 'prec_top', 'rate'):
        d = np.nanmax(np.abs(a1[q] - a0[q]))
        assert d < 1e-9, (q, d)
    a = align_w(Xt, Yt, torch.from_numpy(w))
    for j in range(m):
        for li in range(2):
            ra = roc_auc_score(Y[:, li], X[:, j].astype(np.float32), sample_weight=w)
            pa = average_precision_score(Y[:, li], X[:, j].astype(np.float32), sample_weight=w)
            assert abs(ra - a['auc'][j, li]) < 1e-9 and abs(pa - a['ap'][j, li]) < 1e-9, (j, li)
    # weighted top-1% by brute force on a tie-free column
    x = X[:, 1].astype(np.float64)
    o = np.argsort(-x, kind='stable')
    k = 0.01 * w.sum()
    cw = np.cumsum(w[o])
    i = int(np.searchsorted(cw, k))
    tp = (w[o][:i] * Y[o][:i, 0]).sum() + (k - (cw[i - 1] if i else 0)) * Y[o][i, 0]
    assert abs(tp / k - a['prec_top'][1, 0]) < 1e-9, (tp / k, a['prec_top'][1, 0])
    assert abs(a['rate'][1] - np.average(X[:, 1] > 0, weights=w)) < 1e-12
    log('align_w self-test OK (w = 1 equals align_gpu; weighted AUROC / AP equal sklearn; weighted top-1% brute force)')


# ---------------------------------------------------------------------------------------------- plan
def clusters(src_frames, max_len=25):
    """Runs of wanted source frames <= GAP apart, at most max_len frames each (so long runs still spread over the
    process pool; the pilot ran without the max_len cap and single-cluster videos then decoded on one core)."""
    out, cur = [], [src_frames[0]]
    for s in src_frames[1:]:
        if s - cur[-1] <= GAP and len(cur) < max_len:
            cur.append(s)
        else:
            out.append(cur)
            cur = [s]
    out.append(cur)
    return out


def cmd_plan(args):
    import spatial_sae_pilot as ssp
    import multiscale_sae as ms
    selftest_align()
    t0 = time.time()
    (OUT / 'train').mkdir(parents=True, exist_ok=True)
    (OUT / 'b').mkdir(parents=True, exist_ok=True)
    idx = ssp.StoreIndex('mice')
    tr, _, _, _, train_v, eval_v = idx.split('mice', ms.N_TRAIN['mice'], 0)
    ann = pd.read_csv(ANN, usecols=['observation_id', 'frame_idx', 'frame_path'])
    exp = pd.read_csv(EXP).set_index('observation_id')
    log(f'store: {len(idx.rows):,} frames; {len(train_v)} train videos, {len(eval_v)} eval videos [{time.time() - t0:.0f}s]')
    sel = []
    for v in train_v:
        f = np.sort(tr[idx.obs[tr] == v])
        k = np.unique(np.round(np.linspace(0, len(f) - 1, args.per_video)).astype(int))
        sel.append(f[k])
    sel = np.sort(np.concatenate(sel))
    tok, pos, lens = idx.load(sel)
    rows = idx.rows[sel]
    T = pd.DataFrame({'store_frame': sel, 'row': rows, 'obs': ann.observation_id.values[rows],
                      'frame_idx': ann.frame_idx.values[rows], 'frame_path': ann.frame_path.values[rows], 'n_fg': lens})
    for c in ('phase', 'odor', 'line', 'genotype', 'pool'):
        T[c] = exp.loc[T.obs.values, c].values
    np.save(OUT / 'train/tok448.npy', tok)
    pos.astype(np.int16).tofile(OUT / 'train/pos448.i16')
    T.to_parquet(OUT / 'train/frames.parquet')
    log(f'train: {len(T):,} frames of {T.obs.nunique()} videos, {len(tok):,} R448 tokens ({tok.nbytes / 1e9:.1f} GB); '
        f'phase {T.drop_duplicates("obs").phase.value_counts().to_dict()} videos, odour '
        f'{T.drop_duplicates("obs").odor.value_counts().to_dict()}; tokens/frame mean {lens.mean():.1f} '
        f'[{time.time() - t0:.0f}s]')
    S = pd.read_parquet(BDIR / 'subset.parquet')
    S['frame_idx'] = ann.frame_idx.values[S.row.values]
    S['frame_path'] = ann.frame_path.values[S.row.values]
    assert (ann.observation_id.values[S.row.values] == S.obs.values).all()
    zb = np.load(BDIR / 'frames.npz')
    assert (zb['lens448'] == S.n_fg.values).all()
    S.to_parquet(OUT / 'b/frames.parquet')
    # decode plan
    videos = {}
    for name, F in (('train', T), ('b', S)):
        for o, d in F.groupby('obs'):
            k = np.sort(d.frame_idx.unique())
            st = int(exp.loc[o, 'start_frame'])
            s = (st + STEP * k + OFFSET).tolist()
            cl = clusters(s)
            fd = str(Path(d.frame_path.iat[0]).parent)
            assert (d.frame_path == [f'{fd}/frame_{x:06d}.jpg' for x in d.frame_idx]).all(), o
            videos[o] = {'set': name, 'src': exp.loc[o, 'observation_file'], 'start': st, 'k': k.tolist(), 'frame_dir': fd,
                         'n_clusters': len(cl), 'span_frames': int(sum(c[-1] - c[0] + 1 for c in cl)),
                         'size_gb': os.path.getsize(SRC / exp.loc[o, 'observation_file']) / 1e9}
    cost = {o: 15 + 4.4 * v['n_clusters'] + v['span_frames'] / 35 for o, v in videos.items()}
    tasks = [[] for _ in range(args.tasks)]
    load = np.zeros(args.tasks)
    for o in sorted(cost, key=lambda o: -cost[o]):
        t = int(np.argmin(load))
        tasks[t].append(o)
        load[t] += cost[o]
    n_wanted = sum(len(v['k']) for v in videos.values())
    n_cl = sum(v['n_clusters'] for v in videos.values())
    plan = {'videos': videos, 'tasks': tasks, 'gap': GAP, 'offset': OFFSET, 'n_videos': len(videos),
            'n_frames': n_wanted, 'n_clusters': n_cl, 'copy_gb': sum(v['size_gb'] for v in videos.values()),
            'est_cpu_h': float(load.sum() / 3600), 'est_task_cpu_h': (load / 3600).round(2).tolist(),
            'rollout': {'store_frames_1fps': int(len(idx.rows)),
                        'store_tokens448': int(idx.nfg.sum()),
                        'store_videos': int(len(set(idx.obs))),
                        'source_frames_store_videos': int(sum(int(exp.loc[o, 'end_frame']) - int(exp.loc[o, 'start_frame'])
                                                              for o in set(idx.obs))),
                        'source_gb_store_videos': float(sum(os.path.getsize(SRC / exp.loc[o, 'observation_file'])
                                                            for o in set(idx.obs)) / 1e9)}}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=1))
    log(f'decode plan: {len(videos)} videos, {n_wanted:,} frames in {n_cl:,} clusters, {plan["copy_gb"]:.0f} GB to stage, '
        f'estimated {plan["est_cpu_h"]:.1f} CPU-h over {args.tasks} tasks; rollout counts {plan["rollout"]} '
        f'[{time.time() - t0:.0f}s]')


# ---------------------------------------------------------------------------------------------- decode
def _vf(sel_expr):
    return (f"select='{sel_expr}',scale={SIZE}:{SIZE}:force_original_aspect_ratio=decrease,"
            f"pad={SIZE}:{SIZE}:(ow-iw)/2:(oh-ih)/2")


def decode_cluster(a):
    src, s, ks, dst, tmp = a
    s0 = s[0]
    expr = '+'.join(f'eq(n\\,{x - s0})' for x in s)
    td = Path(tmp) / f'c{s0}'
    td.mkdir(parents=True, exist_ok=True)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    cmd = ['ffmpeg', '-v', 'error', '-threads', '1', '-filter_threads', '1', '-ss', f'{s0 / FPS:.4f}', '-i', str(src),
           '-vf', _vf(expr), '-vsync', '0', '-frames:v', str(len(s)), '-q:v', '2', f'{td}/%05d.jpg']
    subprocess.run(cmd, check=True)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    got = 0
    for i, k in enumerate(ks):
        f = td / f'{i + 1:05d}.jpg'
        if f.exists():
            f.rename(Path(dst) / f'{k:06d}.jpg')
            got += 1
    shutil.rmtree(td)
    return got, len(s), (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), time.time() - t0


def mad_check(o, v, dst):
    """3 frames: native 896 -> 512 vs dataset JPEG of frames k-1, k, k+1 (mean absolute grey difference)."""
    from PIL import Image
    ks = [v['k'][i] for i in np.linspace(0, len(v['k']) - 1, 3).astype(int)]
    out = []
    for k in ks:
        nat = np.asarray(Image.open(Path(dst) / f'{k:06d}.jpg').convert('L').resize((512, 512), Image.BICUBIC), np.float32)
        row = []
        for kk in (k - 1, k, k + 1):
            p = REPO / 'dataset' / v['frame_dir'] / f'frame_{kk:06d}.jpg'
            row.append(float(np.abs(nat - np.asarray(Image.open(p).convert('L'), np.float32)).mean()) if p.exists() else None)
        out.append({'k': k, 'mad_km1_k_kp1': row})
    return out


def cmd_decode(args):
    t_all = time.time()
    t = int(os.environ['SLURM_ARRAY_TASK_ID']) if args.task < 0 else args.task
    plan = json.loads((OUT / 'plan.json').read_text())
    vids = plan['tasks'][t][:args.max_videos or None]
    stage = stage_dir()
    fr = stage / 'frames'
    tmp = stage / 'tmp'
    tmp.mkdir(exist_ok=True)
    rep = {'task': t, 'host': os.uname().nodename, 'videos': {}, 'workers': args.workers}
    for vi, o in enumerate(vids):
        v = plan['videos'][o]
        dst = fr / o
        dst.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        loc = stage / 'src.mp4'
        shutil.copyfile(SRC / v['src'], loc)
        t_copy = time.time() - t0
        s = [v['start'] + STEP * k + OFFSET for k in v['k']]
        cl = clusters(s)
        kk = iter(v['k'])
        jobs = [(str(loc), c, [next(kk) for _ in c], str(dst), str(tmp)) for c in cl]
        t1 = time.time()
        with Pool(args.workers) as pool:
            res = pool.map(decode_cluster, jobs, chunksize=1)
        t_dec = time.time() - t1
        got = sum(r[0] for r in res)
        e = {'set': v['set'], 'n_wanted': len(s), 'n_got': got, 'n_clusters': len(cl), 'copy_s': round(t_copy, 1),
             'size_gb': round(v['size_gb'], 2), 'decode_wall_s': round(t_dec, 1),
             'decode_cpu_s': round(sum(r[2] for r in res), 1), 'mad': mad_check(o, v, dst) if got == len(s) else None}
        if vi == 0 and t == 0:  # sequential-decode benchmark for the rollout estimate: 1800 source frames, keep 1 in 30
            r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
            tb = time.time()
            subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-filter_threads', '1', '-ss', f'{v["start"] / FPS:.4f}',
                            '-i', str(loc), '-vf', _vf('not(mod(n\\,30))'), '-vsync', '0', '-frames:v', '60', '-q:v', '2',
                            f'{tmp}/seq_%05d.jpg'], check=True)
            r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            rep['sequential_benchmark'] = {'source_frames': 1800, 'frames_out': 60, 'wall_s': round(time.time() - tb, 1),
                                           'cpu_s': round((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), 1)}
            for f in tmp.glob('seq_*.jpg'):
                f.unlink()
            log(f'  sequential benchmark: {rep["sequential_benchmark"]}')
        loc.unlink()
        rep['videos'][o] = e
        log(f'  [{vi + 1}/{len(vids)}] {o}: {got}/{len(s)} frames, {len(cl)} clusters, copy {t_copy:.0f}s '
            f'({v["size_gb"]:.1f} GB), decode {t_dec:.0f}s wall {e["decode_cpu_s"]:.0f} CPU-s; MAD '
            + (str([m['mad_km1_k_kp1'] for m in e['mad']]) if e['mad'] else 'skipped (frames missing)'))
    tpath = stage / f'task_{t:02d}.tar'
    with tarfile.open(tpath, 'w') as tf:
        tf.add(fr, arcname='.')
    (OUT / 'native').mkdir(parents=True, exist_ok=True)
    tsuf = '' if args.max_videos == 0 else '_smoke'
    shutil.copyfile(tpath, OUT / 'native' / f'task_{t:02d}{tsuf}.tar')
    rep['tar_gb'] = os.path.getsize(tpath) / 1e9
    rep['wall_s'] = round(time.time() - t_all, 1)
    (OUT / 'native' / f'task_{t:02d}{tsuf}.json').write_text(json.dumps(rep, indent=1))
    log(f'task {t}: {len(vids)} videos, {sum(e["n_got"] for e in rep["videos"].values())} frames, tar '
        f'{rep["tar_gb"]:.2f} GB, {rep["wall_s"]:.0f}s')


# ---------------------------------------------------------------------------------------------- encode
def cmd_encode(args):
    import torch
    from PIL import Image
    from src.eci.extract import load_encoder
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32, as the store and the B tokens
    torch.backends.cudnn.allow_tf32 = False
    stage = stage_dir()
    t0 = time.time()
    tars = sorted((OUT / 'native').glob('task_??.tar'))
    plan = json.loads((OUT / 'plan.json').read_text())
    assert len(tars) == len(plan['tasks']), f'{len(tars)} tars for {len(plan["tasks"])} tasks'
    nat = stage / 'native'
    nat.mkdir(exist_ok=True)
    for tp in tars:
        subprocess.run(['tar', '-C', str(nat), '-xf', str(tp)], check=True)
    log(f'staged {len(tars)} native tars in {time.time() - t0:.0f}s')
    T = pd.read_parquet(OUT / 'train/frames.parquet')
    S = pd.read_parquet(OUT / 'b/frames.parquet')
    chk = S.sample(64, random_state=0).sort_index()
    lst = stage / 'jpg512.txt'
    lst.write_text('\n'.join(list(T.frame_path) + list(chk.frame_path)) + '\n')
    t1 = time.time()
    (stage / 'jpg512').mkdir(exist_ok=True)
    subprocess.run(f'tar -C {REPO / "dataset"} -cf - -T {lst} | tar -C {stage / "jpg512"} -xf -', shell=True, check=True)
    log(f'staged {len(T) + len(chk):,} 512 px JPGs in {time.time() - t1:.0f}s')
    _, proc, model = load_encoder('dinov2_base', SIZE, dev, center_crop=False)
    G = SIZE // 14

    class DS(torch.utils.data.Dataset):
        def __init__(self, paths, grey):
            self.paths, self.grey = paths, grey

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            with Image.open(self.paths[i]) as im:
                im = im.convert('RGB')
            g = (np.asarray(im.convert('L'), np.float32).reshape(G, 14, G, 14).mean((1, 3)) if self.grey
                 else np.zeros((G, G), np.float32))
            return proc(images=im, return_tensors='pt')['pixel_values'][0], torch.from_numpy(g)

    def encode(paths, pos448, lens448, name, grey=False):
        n896 = 4 * lens448
        st4, s8 = np.r_[0, np.cumsum(lens448)], np.r_[0, np.cumsum(n896)]
        tok = np.lib.format.open_memmap(stage / f'{name}.npy', 'w+', np.float16, (int(n896.sum()), D))
        pos = np.empty(int(n896.sum()), np.int16)
        gr = np.empty(int(n896.sum()), np.float16)
        dl = torch.utils.data.DataLoader(DS(paths, grey), batch_size=args.bs, num_workers=8, shuffle=False,
                                         pin_memory=True)
        gpu_s, i0, tw = 0.0, 0, time.time()
        for bi, (pix, g) in enumerate(dl):
            torch.cuda.synchronize()
            tt = time.time()
            with torch.inference_mode():
                hs = model(pixel_values=pix.to(dev, non_blocking=True)).last_hidden_state.float()[:, 1:].half()
            torch.cuda.synchronize()
            gpu_s += time.time() - tt
            assert hs.shape[1] == G * G, hs.shape
            hs = hs.cpu().numpy()
            g = g.numpy().reshape(len(pix), -1)
            for j in range(len(pix)):
                f = i0 + j
                q = children_896(pos448[st4[f]:st4[f + 1]])
                tok[s8[f]:s8[f + 1]] = hs[j, q]
                pos[s8[f]:s8[f + 1]] = q
                gr[s8[f]:s8[f + 1]] = g[j, q]
            i0 += len(pix)
            if bi % 100 == 0:
                log(f'  {name}: {i0:,}/{len(paths):,} frames [{time.time() - tw:.0f}s]')
        tok.flush()
        del tok
        info = {'frames': len(paths), 'tokens': int(n896.sum()), 'gpu_s': round(gpu_s, 1), 'wall_s': round(time.time() - tw, 1),
                'gpu_s_per_1k_frames': round(1000 * gpu_s / len(paths), 1)}
        log(f'  {name}: {info}')
        return pos, gr, info

    info = {'gpu': torch.cuda.get_device_name(0), 'batch_size': args.bs}
    # 1) consistency: our U896 path on 64 B frames vs the existing B tok896 (same JPGs, same processor)
    zb = np.load(BDIR / 'frames.npz')
    l448 = zb['lens448'].astype(np.int64)
    pB448 = np.fromfile(BDIR / 'pos448.i16', np.int16)
    pB896 = np.fromfile(BDIR / 'pos896.i16', np.int16)
    s448 = np.r_[0, np.cumsum(l448)]
    s896 = np.r_[0, np.cumsum(4 * l448)]
    ci = chk.index.values
    cp = np.concatenate([pB448[s448[i]:s448[i + 1]] for i in ci])
    pos_c, _, _ = encode([stage / 'jpg512' / p for p in chk.frame_path], cp, l448[ci], 'chk_u896')
    a = np.load(stage / 'chk_u896.npy').astype(np.float32)
    ref896 = np.load(BDIR / 'tok896.npy', mmap_mode='r')
    b = np.concatenate([np.asarray(ref896[s896[i]:s896[i + 1]], np.float32) for i in ci])
    refp = np.concatenate([pB896[s896[i]:s896[i + 1]] for i in ci])
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    info['check_u896_vs_B'] = {'n_frames': len(ci), 'n_tokens': len(a), 'pos_equal': bool((pos_c == refp).all()),
                               'cos_min': float(cos.min()), 'cos_mean': float(cos.mean()),
                               'max_abs_diff': float(np.abs(a - b).max())}
    log(f'U896 path vs existing B tok896: {info["check_u896_vs_B"]}')
    assert info['check_u896_vs_B']['pos_equal'] and cos.min() > 0.99, 'U896 re-encode does not match the B tokens'
    # processor on an 896 input is (near) identity resampling: compare with plain normalisation
    im = Image.open(nat / S.obs.iat[0] / f'{S.frame_idx.iat[0]:06d}.jpg').convert('RGB')
    assert im.size == (SIZE, SIZE), im.size
    px = proc(images=im, return_tensors='pt')['pixel_values'][0].numpy()
    man = (np.asarray(im, np.float32) / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    info['processor_identity_max_abs'] = float(np.abs(px - man.transpose(2, 0, 1)).max())
    log(f'processor on a {SIZE} px native frame vs plain normalisation: max |diff| {info["processor_identity_max_abs"]:.4f}')
    (OUT / 'train').mkdir(exist_ok=True)
    pT448 = np.fromfile(OUT / 'train/pos448.i16', np.int16)
    lT = T.n_fg.values.astype(np.int64)
    # 2) train U896, train N896, B N896
    miss = [p for p in [nat / o / f'{k:06d}.jpg' for o, k in zip(T.obs, T.frame_idx)] +
            [nat / o / f'{k:06d}.jpg' for o, k in zip(S.obs, S.frame_idx)] if not p.exists()]
    assert not miss, f'{len(miss)} native frames missing, e.g. {miss[:3]}'
    jobs = [('train_u896', [stage / 'jpg512' / p for p in T.frame_path], pT448, lT, False),
            ('train_n896', [nat / o / f'{k:06d}.jpg' for o, k in zip(T.obs, T.frame_idx)], pT448, lT, False),
            ('b_n896', [nat / o / f'{k:06d}.jpg' for o, k in zip(S.obs, S.frame_idx)], pB448, l448, True)]
    for name, paths, p448, ln, grey in jobs:
        pos, gr, inf = encode(paths, p448, ln, name, grey)
        info[name] = inf
        sub, fn = name.split('_')
        shutil.copyfile(stage / f'{name}.npy', OUT / sub / f'tok{fn[1:]}{fn[0]}.npy')
        if sub == 'train':
            pos.tofile(OUT / 'train/pos896.i16')
        else:
            assert (pos == pB896).all()
            gr.tofile(OUT / 'b/grey896n.f16')
        (stage / f'{name}.npy').unlink()
    info['wall_s'] = round(time.time() - t0, 1)
    (OUT / 'encode.json').write_text(json.dumps(info, indent=1))
    log(f'encode done {info}')


# ---------------------------------------------------------------------------------------------- data access
def arm_tokens(arm, split):
    """-> (tok path, pos (N,) int16, lens (F,) int64, grid)."""
    if split == 'b':
        z = np.load(BDIR / 'frames.npz')
        if arm == 'r448':
            return BDIR / 'tok448.npy', np.fromfile(BDIR / 'pos448.i16', np.int16), z['lens448'].astype(np.int64), 32
        p = BDIR / 'tok896.npy' if arm == 'u896' else OUT / 'b/tok896n.npy'
        return p, np.fromfile(BDIR / 'pos896.i16', np.int16), z['lens896'].astype(np.int64), 64
    T = pd.read_parquet(OUT / 'train/frames.parquet')
    ln = T.n_fg.values.astype(np.int64)
    if arm == 'r448':
        return OUT / 'train/tok448.npy', np.fromfile(OUT / 'train/pos448.i16', np.int16), ln, 32
    return OUT / f'train/tok896{arm[0]}.npy', np.fromfile(OUT / 'train/pos896.i16', np.int16), 4 * ln, 64


def load_ram(path, stage):
    """Copy a token .npy to local disk once, then read it into RAM (one sequential NFS read)."""
    t0 = time.time()
    loc = stage / path.name
    if not loc.exists():
        shutil.copyfile(path, loc)
    x = np.load(loc)
    loc.unlink()
    log(f'  loaded {path.name} ({x.nbytes / 1e9:.1f} GB) in {time.time() - t0:.0f}s')
    return x


# ---------------------------------------------------------------------------------------------- train
def cmd_train(args):
    import torch
    import sae_levers as sl
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True  # as sae_levers train
    stage = stage_dir()
    cfg = sl.parse_cfg('w4096')
    meta = {}
    for arm in args.arms:
        path, pos, lens, grid = arm_tokens(arm, 'train')
        X = torch.from_numpy(load_ram(path, stage)).to(dev)
        assert len(X) == len(pos) == lens.sum()
        n = len(X)
        z = torch.zeros(n, dtype=torch.long, device=dev)
        pools = {'uniform': (X, z, z, torch.zeros(n, dtype=torch.bool, device=dev))}
        meta[arm] = {'unique_tokens': n, 'frames': int(len(lens)), 'tokens_per_frame': float(lens.mean()),
                     'presentations': args.steps * 4096, 'presentations_per_token': args.steps * 4096 / n}
        log(f'{arm}: {meta[arm]}')
        for seed in args.seeds:
            sl.train_one('w4096', cfg, seed, pools, None, args.steps, OUT / 'sae' / f'{arm}_s{seed}',
                         {'config': 'w4096', 'arm': arm, 'lever': cfg, 'domain': 'mice', 'cap': n, 'pool_seed': None},
                         dev)
            torch.cuda.empty_cache()
        del X, pools, z
        torch.cuda.empty_cache()
    p = OUT / 'train_meta.json'
    old = json.loads(p.read_text()) if p.exists() else {}
    p.write_text(json.dumps({**old, **meta}, indent=1))


# ---------------------------------------------------------------------------------------------- evaluate
def deciles(nfg):
    rank = np.argsort(np.argsort(nfg + np.random.default_rng(0).random(len(nfg)) * 1e-3))
    return np.minimum(rank * 10 // len(nfg), 9)


def score(codes, Yt, Wt, half, dec, dev):
    import torch
    res = {}
    R = {h: align_w(codes, Yt, Wt, rows=torch.from_numpy(np.flatnonzero(half == h)).to(dev)) for h in (0, 1)}
    A = align_w(codes, Yt, Wt)
    for li, b in enumerate(BEH):
        dirs = []
        for s, t in ((0, 1), (1, 0)):
            S_, T_ = R[s], R[t]
            e = {'select_half': s, 'test_half': t}
            for sel, q, honest in (('auc', 'auc', False), ('ap', 'ap', False), ('top1', 'prec_top', True)):
                sc = S_[q][:, li].copy()
                if honest:
                    sc[S_['rate'] < MIN_RATE] = -np.inf
                sc[~np.isfinite(sc)] = -np.inf
                j = int(np.argmax(sc))
                e[sel] = {'neuron': j, 'select_score': float(sc[j]), 'test_auc': float(T_['auc'][j, li]),
                          'test_ap': float(T_['ap'][j, li]), 'test_top1': float(T_['prec_top'][j, li]),
                          'test_rate': float(T_['rate'][j]), 'select_rate': float(S_['rate'][j])}
            j = e['auc']['neuron']
            sc_d = []
            yb = Yt[:, li].cpu().numpy()
            for d in range(10):
                rr = np.flatnonzero((half == t) & (dec == d))
                npos = int(yb[rr].sum())
                if npos < 10 or len(rr) - npos < 10:
                    continue
                sc_d.append(float(align_w(codes, Yt[:, [li]], Wt, rows=torch.from_numpy(rr).to(dev),
                                          cols=torch.tensor([j], device=dev))['auc'][0, 0]))
            e['auc']['test_size_ctrl_auc'] = float(np.mean(sc_d)) if sc_d else float('nan')
            e['auc']['size_ctrl_deciles'] = len(sc_d)
            dirs.append(e)
        ok = A['rate'] >= MIN_RATE
        res[b] = {'base_rate': float(A['base'][li]), 'base_half': [float(R[h]['base'][li]) for h in (0, 1)],
                  'cf_auroc': float(np.mean([e['auc']['test_auc'] for e in dirs])),
                  'cf_ap_at_auroc': float(np.mean([e['auc']['test_ap'] for e in dirs])),
                  'cf_size_ctrl_auroc': float(np.nanmean([e['auc']['test_size_ctrl_auc'] for e in dirs])),
                  'cf_ap': float(np.mean([e['ap']['test_ap'] for e in dirs])),
                  'cf_top1': float(np.mean([e['top1']['test_top1'] for e in dirs])),
                  'cf_top1_rate': float(np.mean([e['top1']['test_rate'] for e in dirs])),
                  'insample_auroc': float(np.nanmax(A['auc'][:, li])), 'insample_ap': float(np.nanmax(A['ap'][:, li])),
                  'insample_top1_honest': float(np.where(ok, np.nan_to_num(A['prec_top'][:, li], nan=-1), -1).max()),
                  'dirs': dirs}
    return res


def argmax_patch(sae, norm, j, tok, pos, st, frames, dev):
    import torch
    out = []
    We, be = sae.W_enc[:, [j]], sae.b_enc[[j]]
    for i in frames:
        T_ = torch.from_numpy(np.asarray(tok[st[i]:st[i + 1]])).to(dev)
        pre = ((norm(T_) - sae.b_dec) @ We + be)[:, 0]
        out.append(int(pos[st[i] + int(pre.argmax())]))
    return out


def pick_frames(x, mask, obs, n_show=16, per_video=2):
    x = np.where(mask, x, 0)
    pick, seen = [], {}
    for i in np.argsort(-x, kind='stable'):
        if x[i] <= 0 or len(pick) == n_show:
            break
        if seen.get(obs[i], 0) >= per_video:
            continue
        seen[obs[i]] = seen.get(obs[i], 0) + 1
        pick.append(int(i))
    return pick


def render_sheet(path, items, title, ncol=4, th=256):
    """items: (image path, image size, grid, patch pos, label positive (bool or None), caption)."""
    from PIL import Image, ImageDraw
    rows = int(np.ceil(len(items) / ncol))
    W, H = ncol * 2 * th, rows * (th + 14) + 16
    sheet = Image.new('RGB', (W, H), 'white')
    dr = ImageDraw.Draw(sheet)
    dr.text((4, 2), title, fill='black')
    for q, (ip, size, grid, p, yv, cap) in enumerate(items):
        im = Image.open(ip).convert('RGB')
        if im.size != (size, size):
            im = im.resize((size, size))
        ps = size / grid
        r_, c_ = divmod(int(p), grid)
        x0, y0 = c_ * ps, r_ * ps
        col = (0, 200, 0) if yv else ((230, 0, 0) if yv is not None else (0, 120, 255))
        full = im.copy()
        ImageDraw.Draw(full).rectangle([x0 - 3, y0 - 3, x0 + ps + 2, y0 + ps + 2], outline=col, width=max(2, size // 200))
        half = size / 16  # zoom crop: 1/8 of the frame width (~258 native px), centred on the patch
        cx, cy = x0 + ps / 2, y0 + ps / 2
        cx, cy = min(max(cx, half), size - half), min(max(cy, half), size - half)
        zoom = im.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((th, th), Image.LANCZOS)
        zs = th / (2 * half)
        ImageDraw.Draw(zoom).rectangle([(x0 - (cx - half)) * zs, (y0 - (cy - half)) * zs,
                                        (x0 + ps - (cx - half)) * zs, (y0 + ps - (cy - half)) * zs], outline=col, width=2)
        rr, cc = divmod(q, ncol)
        X0, Y0 = cc * 2 * th, 16 + rr * (th + 14)
        sheet.paste(full.resize((th, th), Image.LANCZOS), (X0, Y0))
        sheet.paste(zoom, (X0 + th, Y0))
        dr.text((X0 + 2, Y0 + th), cap, fill='black')
    sheet.save(path, quality=88)


def cmd_evaluate(args):
    import torch
    import sae_levers as sl
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False  # as sae_levers evaluate
    stage = stage_dir()
    for d in ('eval', 'sheets'):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    S = pd.read_parquet(OUT / 'b/frames.parquet')
    pools = sorted(S.pool.unique())
    half = S.pool.map({p: i % 2 for i, p in enumerate(pools)}).values
    dec = deciles(S.n_fg.values)
    Yt = torch.from_numpy(S[BEH].values.astype(bool)).to(dev)
    Wt = torch.from_numpy(S.w.values.astype(np.float64)).to(dev)
    log(f'B: {len(S):,} frames, {len(pools)} pools; halves {np.bincount(half).tolist()} frames; weighted base rates '
        + json.dumps({b: [round(float(np.average(S[b][half == h], weights=S.w[half == h])), 4) for h in (0, 1)] for b in BEH}))
    native = {}
    if args.sheets:
        t0 = time.time()
        nat = stage / 'native'
        nat.mkdir(exist_ok=True)
        for tp in sorted((OUT / 'native').glob('task_??.tar')):
            subprocess.run(['tar', '-C', str(nat), '-xf', str(tp)], check=True)
        log(f'  native frames staged in {time.time() - t0:.0f}s')
        native = {'dir': nat}
    obs = S.obs.values
    for arm in args.arms:
        path, pos, lens, grid = arm_tokens(arm, 'b')
        tok = load_ram(path, stage)
        st = np.r_[0, np.cumsum(lens)]
        assert len(tok) == len(pos) == st[-1]
        tvid = np.zeros(len(tok), np.int64)
        for seed in args.seeds:
            t0 = time.time()
            key = f'{arm}_s{seed}'
            sae, norm, _ = load_sae(OUT / 'sae' / key / 'sae.pt', dev)
            codes, stt = sl.encode_model(sae, norm, None, None, tok, pos, tvid, lens, dev)
            res = score(codes, Yt, Wt, half, dec, dev)
            tj = json.loads((OUT / 'sae' / key / 'train.json').read_text())
            e = {'arm': arm, 'seed': seed, 'fve': stt['fve'], 'l0_per_token': stt['l0_per_token'],
                 'dead_frac': stt['dead_frac'], 'n_tokens': stt['n_tokens'], 'tokens_per_frame': float(lens.mean()),
                 'train_fve_last': tj['history'][-1]['fve'], 'dead_frac_train': tj['dead_frac_train'],
                 'train_time_s': tj['train_time_s'], 'labels': res}
            (OUT / 'eval' / f'{key}.json').write_text(json.dumps(e, indent=1))
            log(f'{key}: FVE {stt["fve"]:.4f} L0 {stt["l0_per_token"]:.1f} dead {stt["dead_frac"]:.3f} | ' + ' | '.join(
                f'{b}: cfAUROC {res[b]["cf_auroc"]:.3f} cfAP {res[b]["cf_ap"]:.3f} top1 {res[b]["cf_top1"]:.3f} '
                f'size {res[b]["cf_size_ctrl_auroc"]:.3f} (in-sample AUROC {res[b]["insample_auroc"]:.3f} AP '
                f'{res[b]["insample_ap"]:.3f})' for b in BEH) + f' [{time.time() - t0:.0f}s]')
            if args.sheets and seed == args.seeds[0]:
                index = []
                for b in BEH:
                    e0 = res[b]['dirs'][0]  # selected on half 0, frames of half 1
                    for sel in ('auc', 'top1'):
                        j = e0[sel]['neuron']
                        x = sl.column_codes(sae, norm, None, None, [j], tok, pos, tvid, lens, dev)[0][:, 0]
                        pick = pick_frames(x, half == e0['test_half'], obs)
                        boxes = argmax_patch(sae, norm, j, tok, pos, st, pick, dev)
                        if arm == 'n896':
                            ims = [(native['dir'] / obs[i] / f'{S.frame_idx.iat[i]:06d}.jpg', SIZE) for i in pick]
                        else:
                            ims = [(REPO / 'dataset' / S.frame_path.iat[i], 512) for i in pick]
                        yv = [bool(S[b].iat[i]) for i in pick]
                        items = [(ip, sz, grid, p, y_, f'{obs[i]} f{S.frame_idx.iat[i]} {"POS" if y_ else "neg"} '
                                  f'a={x[i]:.2f}') for (ip, sz), p, y_, i in zip(ims, boxes, yv, pick)]
                        fn = f'{key}__{b}__{sel}_n{j}.jpg'
                        render_sheet(OUT / 'sheets' / fn, items, f'{key} {b} neuron {j} (selected on half 0 by {sel}; '
                                     f'top frames of half 1, <= 2 per video; green = label positive)')
                        index.append({'arm': arm, 'label': b, 'selection': sel, 'neuron': j, 'n_shown': len(pick),
                                      'n_pos_shown': int(sum(yv)), 'file': fn})
                        log(f'  sheet {fn}: {sum(yv)}/{len(pick)} label-positive')
                p = OUT / 'sheets/index.json'
                old = json.loads(p.read_text()) if p.exists() else []
                p.write_text(json.dumps([r for r in old if r['arm'] != arm] + index, indent=1))
                if arm == 'n896':
                    identity_screen(sae, norm, tok, pos, lens, st, S, native['dir'], dev)
            del codes
            torch.cuda.empty_cache()
        del tok


def identity_screen(sae, norm, tok, pos, lens, st, S, nat, dev, n_cand=6):
    """Exploratory: N896 neurons that fire on PALE patches INSIDE a mouse body (shave marks are pale fur on dark
    mice). Per token: grey of its 14 x 14 px patch on the native 896 frame minus the mean grey of the frame's kept
    patches; interior = the parent 448 patch has all 8 neighbours in the fg448 mask. Per neuron: activation-weighted
    means over the B tokens. Candidates: token fire rate 0.05% - 3%, interior share >= 0.8, highest relative grey.
    Sheets of their top tokens (one per frame, <= 2 per video). Qualitative: no identity labels exist on these frames."""
    import torch
    gr = np.fromfile(OUT / 'b/grey896n.f16', np.float16).astype(np.float32)
    fmean = np.repeat(np.add.reduceat(gr, st[:-1]) / np.maximum(lens, 1), lens)
    rel = gr - fmean
    p448 = np.fromfile(BDIR / 'pos448.i16', np.int16).astype(np.int64)
    l448 = lens // 4
    s4 = np.r_[0, np.cumsum(l448)]
    inter = np.zeros(len(pos), np.float32)
    for f in range(len(lens)):
        m = np.zeros((34, 34), bool)
        q = p448[s4[f]:s4[f + 1]]
        m[q // 32 + 1, q % 32 + 1] = True
        nb = sum(np.roll(np.roll(m, dr, 0), dc, 1) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        ok = (nb == 9)[1:33, 1:33].ravel()
        pp = pos[st[f]:st[f + 1]].astype(np.int64)
        inter[st[f]:st[f + 1]] = ok[(pp // 64 // 2) * 32 + (pp % 64) // 2]
    m = sae.n_latents
    acc = {k: torch.zeros(m, dtype=torch.float64, device=dev) for k in ('z', 'zg', 'zi', 'fire')}
    R, I = torch.from_numpy(rel).to(dev), torch.from_numpy(inter).to(dev)
    with torch.no_grad():
        for a in range(0, len(tok), 200_000):
            b = min(len(tok), a + 200_000)
            z = sae.encode(norm(torch.from_numpy(np.asarray(tok[a:b])).to(dev)), mode='threshold').float()
            acc['z'] += z.sum(0).double()
            acc['zg'] += (z * R[a:b, None]).sum(0).double()
            acc['zi'] += (z * I[a:b, None]).sum(0).double()
            acc['fire'] += (z > 0).sum(0).double()
    zs = acc['z'].clamp_min(1e-9)
    g_w, i_w = (acc['zg'] / zs).cpu().numpy(), (acc['zi'] / zs).cpu().numpy()
    rate = (acc['fire'] / len(tok)).cpu().numpy()
    ok = (rate >= 5e-4) & (rate <= 0.03) & (i_w >= 0.8)
    cand = np.argsort(-np.where(ok, g_w, -np.inf))[:n_cand]
    info = {'all_tokens_rel_grey_sd': float(rel.std()), 'interior_token_frac': float(inter.mean()),
            'n_neurons_passing_rate_interior': int(ok.sum()), 'candidates': []}
    obs = S.obs.values
    We_all = sae.W_enc
    for j in cand:
        j = int(j)
        # top tokens of neuron j: per frame max activation and its patch
        x = np.zeros(len(lens), np.float32)
        arg = np.zeros(len(lens), np.int64)
        with torch.no_grad():
            for f0 in range(0, len(lens), 2000):
                f1 = min(len(lens), f0 + 2000)
                T_ = torch.from_numpy(np.asarray(tok[st[f0]:st[f1]])).to(dev)
                pre = torch.relu((norm(T_) - sae.b_dec) @ We_all[:, j] + sae.b_enc[j])
                pre = pre * (pre > sae.threshold)
                fr = torch.repeat_interleave(torch.arange(f1 - f0, device=dev), torch.from_numpy(lens[f0:f1]).to(dev))
                mx = torch.zeros(f1 - f0, device=dev).index_reduce_(0, fr, pre, 'amax', include_self=True)
                x[f0:f1] = mx.cpu().numpy()
                pre_np, fr_np = pre.cpu().numpy(), fr.cpu().numpy()
                o = np.lexsort((-pre_np, fr_np))
                first = np.r_[0, np.flatnonzero(np.diff(fr_np[o])) + 1]
                arg[f0 + fr_np[o][first]] = st[f0] + o[first]
        pick = pick_frames(x, np.ones(len(lens), bool), obs, n_show=12)
        items = [(nat / obs[i] / f'{S.frame_idx.iat[i]:06d}.jpg', SIZE, 64, int(pos[arg[i]]), None,
                  f'{obs[i]} f{S.frame_idx.iat[i]} a={x[i]:.2f} relg={rel[arg[i]]:.0f}') for i in pick]
        fn = f'identity_n896_s0_n{j}.jpg'
        render_sheet(OUT / 'sheets' / fn, items, f'N896 s0 neuron {j}: token rate {rate[j]:.4f}, interior '
                     f'{i_w[j]:.2f}, activation-weighted relative grey {g_w[j]:.1f} (blue box = argmax patch)', ncol=4)
        info['candidates'].append({'neuron': j, 'token_rate': float(rate[j]), 'interior': float(i_w[j]),
                                   'rel_grey': float(g_w[j]), 'file': fn})
        log(f'  identity candidate n{j}: rate {rate[j]:.4f} interior {i_w[j]:.2f} rel grey {g_w[j]:.1f} -> {fn}')
    (OUT / 'eval/identity_screen.json').write_text(json.dumps(info, indent=1))


# ---------------------------------------------------------------------------------------------- MIL ceiling
def cmd_mil(args):
    import torch
    import diag_mice_patches as dmp
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True  # as diag_mice_patches b_probe
    stage = stage_dir()
    S = pd.read_parquet(OUT / 'b/frames.parquet')
    path, pos, lens, grid = arm_tokens('n896', 'b')
    tok = load_ram(path, stage)
    bag = dmp.Bag(tok, pos, lens, grid, dev)
    t0 = time.time()
    res = dmp.cross_validate(bag, S, S.w.values, dev, ['mil_ctx'], args.seeds, None, '896n_subset', args.steps)
    summ = dmp.summarise(res)
    (OUT / 'eval').mkdir(parents=True, exist_ok=True)
    (OUT / 'eval/mil_n896.json').write_text(json.dumps({'summary': summ, 'runs': res, 'gpu': torch.cuda.get_device_name(0),
                                                        'wall_s': round(time.time() - t0, 1)}, indent=1))
    for s in summ:
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))


# ---------------------------------------------------------------------------------------------- summary
def cmd_summary(args):
    rows = []
    for p in sorted((OUT / 'eval').glob('*_s?.json')):
        e = json.loads(p.read_text())
        for b in BEH:
            r = e['labels'][b]
            rows.append({'arm': e['arm'], 'seed': e['seed'], 'label': b, 'fve': e['fve'], 'l0': e['l0_per_token'],
                         'dead': e['dead_frac'], **{k: r[k] for k in ('cf_auroc', 'cf_ap', 'cf_ap_at_auroc', 'cf_top1',
                                                                       'cf_top1_rate', 'cf_size_ctrl_auroc',
                                                                       'insample_auroc', 'insample_ap', 'base_rate')}})
    df = pd.DataFrame(rows)
    mets = ['cf_auroc', 'cf_ap', 'cf_top1', 'cf_size_ctrl_auroc', 'fve', 'cf_ap_at_auroc', 'insample_ap']
    agg = df.groupby(['arm', 'label'])[mets].agg(['mean', 'std', 'min', 'max'])
    # supervised ceilings: existing B results (R448 = 448_subset, U896 = 896_subset) + ours (N896)
    ceil = {}
    old = json.loads((BDIR / 'results.json').read_text())['summary']
    new = json.loads((OUT / 'eval/mil_n896.json').read_text())['summary'] if (OUT / 'eval/mil_n896.json').exists() else []
    for s in old + new:
        if s['probe'] != 'mil_ctx':
            continue
        arm = {'448_subset': 'r448', '896_subset': 'u896', '896n_subset': 'n896'}[s['res']]
        ceil[(arm, s['behaviour'])] = s
    verdict = {}
    for b in BEH:
        v = {}
        for m, rule in (('cf_ap', 'ratio'), ('cf_auroc', 'diff')):
            d = {a: df[(df.arm == a) & (df.label == b)][m].values for a in ARMS}
            if any(len(x) == 0 for x in d.values()):
                continue
            nm = d['n896'].mean()
            beats = all(nm > d[a].max() for a in ('r448', 'u896'))
            eff = nm / d['r448'].mean() if rule == 'ratio' else nm - d['r448'].mean()
            v[m] = {'n896_mean': nm, 'r448_mean': d['r448'].mean(), 'u896_mean': d['u896'].mean(),
                    'n896_mean_above_every_seed_of_both': bool(beats), 'effect_vs_r448': eff,
                    'passes_size': bool(eff >= (1.5 if rule == 'ratio' else 0.05)),
                    'passes': bool(beats and eff >= (1.5 if rule == 'ratio' else 0.05))}
        v['real_detail_helps'] = any(x['passes'] for x in v.values() if isinstance(x, dict))
        top1 = df[(df.arm == 'n896') & (df.label == b)].cf_top1
        v['stretch_top1_ge_040'] = bool(len(top1) and top1.mean() >= 0.40)
        if all((a, b) in ceil for a in ARMS):
            c = {a: ceil[(a, b)] for a in ARMS}
            v['ceiling'] = {a: {'auroc': c[a]['auroc_mean'], 'ap': c[a]['ap_mean'], 'ap_sd_seeds': c[a].get('ap_sd_seeds'),
                                'size_ctrl': c[a]['auroc_size_ctrl_mean']} for a in ARMS}
            v['ceiling']['n896_ap_over_r448'] = c['n896']['ap_mean'] / c['r448']['ap_mean']
            v['ceiling']['n896_ap_over_u896'] = c['n896']['ap_mean'] / c['u896']['ap_mean']
            v['ceiling']['n896_auroc_minus_r448'] = c['n896']['auroc_mean'] - c['r448']['auroc_mean']
        verdict[b] = v
    out = {'per_seed': rows, 'agg': {f'{a}|{b}': {f'{m}_{s}': float(agg.loc[(a, b), (m, s)]) for m in mets
                                                   for s in ('mean', 'std', 'min', 'max')} for a, b in agg.index},
           'verdict': verdict}
    for f in ('plan.json', 'encode.json', 'train_meta.json'):
        if (OUT / f).exists():
            j = json.loads((OUT / f).read_text())
            out[f[:-5]] = {k: v for k, v in j.items() if k not in ('videos', 'tasks')}
    (OUT / 'summary.json').write_text(json.dumps(out, indent=1, default=float))
    pd.set_option('display.width', 250)
    print(agg[['cf_auroc', 'cf_ap', 'cf_top1', 'cf_size_ctrl_auroc', 'fve']].xs('mean', axis=1, level=1).round(4))
    print(agg[['cf_auroc', 'cf_ap', 'cf_top1', 'cf_size_ctrl_auroc', 'fve']].xs('std', axis=1, level=1).round(4))
    print(json.dumps(verdict, indent=1, default=float))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['plan', 'decode', 'encode', 'train', 'evaluate', 'mil', 'summary', 'selftest'])
    p.add_argument('--per-video', type=int, default=100)
    p.add_argument('--tasks', type=int, default=16)
    p.add_argument('--task', type=int, default=-1)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-videos', type=int, default=0, help='decode: first N videos of the task only (smoke test)')
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--arms', nargs='+', default=ARMS)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--steps', type=int, default=6000, help='train: SAE steps; mil: MIL steps (use 3000)')
    p.add_argument('--sheets', action='store_true')
    args = p.parse_args()
    if args.cmd == 'selftest':
        selftest_align()
        return
    {'plan': cmd_plan, 'decode': cmd_decode, 'encode': cmd_encode, 'train': cmd_train, 'evaluate': cmd_evaluate,
     'mil': cmd_mil, 'summary': cmd_summary}[args.cmd](args)


if __name__ == '__main__':
    main()
