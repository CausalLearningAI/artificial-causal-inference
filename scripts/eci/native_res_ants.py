"""
Native-resolution test for ANTS ECI (the ants counterpart of the mice T1 pilot, scripts/eci/native_res.py): does
feeding DINOv2 the videos' real pixels (instead of the 512 px standardised frames) give better behaviour neurons with
the same clean pipeline?
    frames -> DINOv2-base patch tokens -> today's antsfg animal mask -> Matryoshka BatchTopK SAE per patch
    -> frame code of a neuron = its max over the frame's kept patches.
Only the input changes. Three arms on IDENTICAL frames and IDENTICAL kept regions:
    R448  512 px frame -> 448 px (32 x 32 patches; 1 patch = 16 px of the 512 frame = ~26 native px). Today's input;
          tokens straight from the antsfg store.
    U896  512 px frame upsampled to 896 (64 x 64): finer grid, NO new detail (the control).
    N896  native source frame (v2 770 px, v3 824 px, H.264, 30 fps) upsampled to 896: finer grid WITH the real
          detail (~13 native px per patch).
Mask: the store's antsfg mask of the frame; each 448 patch -> its 2 x 2 children at 896 (same frame area in every arm).
Reference arm L448 = the levers w4096 checkpoints (trained on 2M tokens of ALL 76,800 train frames) scored with this
harness on the same eval frames: checks that the harness reproduces the levers / T2 numbers.

Frames (antsfg store, split of sae_levers / motion_t3 / combined: spatial_sae_pilot.StoreIndex.split('ants', 128, 0))
    train  --per-video store frames (1 fps, evenly spaced) of each of the 128 train videos. No label used in training.
    eval   ALL store frames of the 128 eval videos = 76,800 frames (600 per video, 1 fps, whole annotated 10 min).
Native frame of 512 px frame k = source frame start_frame (0 for every v2 / v3 video) + 6 k + OFFSET, decoded with the
ffmpeg 6.1.1 that made the standardised videos (~/.conda/envs/ffmpeg611; select the wanted frames, then the
standardisation's scale + pad chain at 896, rawvideo rgb24 through a pipe: no intermediate JPEG). OFFSET is measured by
the 'offset' step (two independent checks, see cmd_offset).

SAEs: the levers recipe (sae_levers.train_one, 'w4096': 4096 latents, prefixes 512/1024/2048/4096, k = 16, 6000 steps x
4096 tokens, Adam 5e-4, TokenNorm, geometric-median b_dec, AuxK), 3 seeds per arm, all train tokens of the arm as the
pool (same steps and batch in every arm, so token presentations are equal).
Eval: sae_levers.score_model (the levers harness): per label, cross-fitted best neuron (eval videos split into two
halves alternating over sorted ids; select on one half, score on the other, average): AUROC, AP (best-AP pick), honest
top-1% (neurons firing on >= 1% of the half's frames), size-controlled AUROC (AUROC pick within fg-count deciles).
Labels: groom_any (Y2F or B2F), groom_yellow (Y2F), groom_blue (B2F), onlid_yellow (Y_YOL, v3 frames only).
Video level (scripts/eci/readouts.py, ants): per held-out half, the AUROC pick's per-video mean presence (1 fps, whole
video) vs the annotated per-video rate (5 fps), Pearson, mean over halves; raw and with the foreground-count control
(residual on the per-video mean 448 kept-patch count, OLS within the half; the 896 count is exactly 4x).

PRE-REGISTERED RULE (same as the combined test): N896 WINS over R448 iff
    (i)  no label's cross-fitted AUROC (3-seed mean) drops more than 0.02, AND
    (ii) at least one label gains: frame level (cf AUROC 3-seed mean >= R448 + 0.03, or cf AP 3-seed mean >= 1.3 x
         R448, the 3-seed mean also above R448's best seed) or video level (count-controlled presence-mean r, 3-seed
         mean, >= R448 + 0.10).
Also reported: N896 vs U896 (real detail vs finer grid), same rule.

Steps (scripts/eci/native_res_ants.sh)
    plan      CPU  frame tables + pos448 of train / eval frames, rollout counts
    offset    CPU  frame-mapping check (exact reproduction of the standardisation + dataset-JPEG check)
    bench     GPU  DINOv2 at 896: speed and token agreement of fp32 vs fp16 (decides the encode precision)
    encode    GPU  --arm u896 | n896: train + eval tokens -> OUT/{train,eval}/tok_<arm>.npy
    train     GPU  arms x 3 seeds -> OUT/sae/<arm>_s<seed>/
    evaluate  GPU  codes, cross-fitted scores, video level, contact sheets -> OUT/{eval,codes_best,sheets}/
    summary   any  OUT/summary.json + table + decision (login-node safe: reads JSONs)
Outputs in results/vision/eci_native_res_ants/ (gitignored).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
OUT = Path(os.environ.get('NRA_OUT') or REPO / 'results/vision/eci_native_res_ants')  # NRA_OUT: smoke tests
DEV = os.environ.get('NRA_DEV', 'cuda')  # 'cpu' only for the CPU smoke test
LEV = REPO / 'results/vision/eci_sae_levers/ants'
FFMPEG = Path.home() / '.conda/envs/ffmpeg611/bin/ffmpeg'
FPS, STEP = 30.0, 6
SIZE, D = 896, 768
ARMS = ['r448', 'u896', 'n896']
LABELS = ['groom_any', 'groom_yellow', 'groom_blue', 'onlid_yellow']
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def stage_dir():
    d = Path(os.environ.get('STAGE') or '')
    if not str(d) or not str(d).startswith('/localhome'):
        raise RuntimeError(f'STAGE must be the job dir under /localhome, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


def offsets():
    """{observation_id: o}: source frame of 512 px frame k = 6 k + o, measured per video by cmd_offset_all."""
    p = OUT / 'offset.json'
    if not p.exists():
        raise RuntimeError('run the offset and offset_all steps first')
    return {k: int(v) for k, v in json.loads(p.read_text())['per_video'].items()}


def children_896(pos448):
    """448 grid positions (sorted) -> their 2x2 children on the 64 x 64 grid, sorted (native_res.children_896)."""
    p = np.asarray(pos448, np.int64)
    r, c = p // 32, p % 32
    return np.sort(np.concatenate([(2 * r + dr) * 64 + 2 * c + dc for dr in (0, 1) for dc in (0, 1)]))


def part_range(n, part, nparts):
    b = np.linspace(0, n, nparts + 1).round().astype(int)
    return b[part], b[part + 1]


def part_suffix(part, nparts):
    return '' if nparts == 1 else f'.part{part}of{nparts}'


def children_all(p448, l448):
    st = np.r_[0, np.cumsum(l448)]
    return np.concatenate([children_896(p448[st[f]:st[f + 1]]) for f in range(len(l448))]).astype(np.int16)


def source_of(version, obs):
    return REPO / f'data/ants/{version}/observations/source/{obs}.mkv'


# ---------------------------------------------------------------------------------------------- native decoding
def decode_native(src, src_frames, size=SIZE):
    """Source frames (sorted, unique) of one video -> uint8 (n, size, size, 3) via the standardisation's scale + pad
    chain (ffmpeg 6.1.1, decode from the start of the stream, select by decoded-frame index n)."""
    src_frames = [int(x) for x in src_frames]
    assert src_frames == sorted(set(src_frames)) and src_frames[0] >= 0
    expr = '+'.join(f'eq(n\\,{x})' for x in src_frames)
    vf = (f"select='{expr}',scale={size}:{size}:force_original_aspect_ratio=decrease,"
          f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2")
    cmd = [str(FFMPEG), '-v', 'error', '-threads', '1', '-filter_threads', '1', '-i', str(src), '-vf', vf,
           '-vsync', '0', '-frames:v', str(len(src_frames)), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    r = subprocess.run(cmd, check=True, capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8)
    if a.size != len(src_frames) * size * size * 3:
        raise RuntimeError(f'{src}: got {a.size // (size * size * 3)} of {len(src_frames)} frames')
    return a.reshape(len(src_frames), size, size, 3)


# ---------------------------------------------------------------------------------------------- plan
def cmd_plan(args):
    import spatial_sae_pilot as ssp
    import multiscale_sae as ms
    t0 = time.time()
    for d in ('train', 'eval'):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    idx = ssp.StoreIndex('ants')
    tr, ev, _, _, train_v, eval_v = idx.split('ants', ms.N_TRAIN['ants'], 0)
    ann = pd.read_csv(REPO / 'dataset/ants/eci/annotations.csv',
                      usecols=['observation_id', 'frame_idx', 'frame_path', 'experiment'])
    log(f'store: {len(idx.rows):,} frames; {len(train_v)} train videos, {len(eval_v)} eval videos [{time.time() - t0:.0f}s]')
    sel = []
    for v in train_v:
        f = np.sort(tr[idx.obs[tr] == v])
        k = np.unique(np.round(np.linspace(0, len(f) - 1, args.per_video)).astype(int))
        sel.append(f[k])
    sets = {'train': np.sort(np.concatenate(sel)), 'eval': np.sort(ev)}
    lab = pd.read_parquet(LEV / 'labels.parquet')
    assert (lab.row.values == idx.rows[sets['eval']]).all(), 'eval frames differ from the levers eval frames'
    if args.smoke:  # tiny CPU smoke test: 8 train videos x 4 frames; 3 v3 eval videos per half, ~12 frames each
        g = np.random.default_rng(0)
        tv = sorted(set(idx.obs[sets['train']]))[:8]
        sets['train'] = np.sort(np.concatenate([sets['train'][idx.obs[sets['train']] == v][:: 40][:4] for v in tv]))
        units = sorted(set(lab.obs))
        lab['half'] = lab.obs.map({u: i % 2 for i, u in enumerate(units)})
        pick = []
        for h in (0, 1):
            L = lab[(lab.half == h) & (lab.experiment == 'v3')]
            r = L.groupby('obs').onlid_yellow.sum().sort_values(ascending=False)
            for v in r.index[:3]:
                q = L[L.obs == v]
                ix = []
                for c, k in (('groom_yellow', 2), ('groom_blue', 2), ('onlid_yellow', 3)):
                    w = np.flatnonzero(q[c].values)
                    ix += list(g.choice(w, min(k, len(w)), replace=False))
                ix += list(g.choice(len(q), 5, replace=False))
                pick += list(q.index.values[sorted(set(ix))])
        sets['eval'] = np.sort(sets['eval'][np.array(sorted(pick))])
    for name, fr in sets.items():
        rows = idx.rows[fr]
        T = pd.DataFrame({'store_frame': fr, 'row': rows, 'obs': ann.observation_id.values[rows],
                          'frame_idx': ann.frame_idx.values[rows], 'frame_path': ann.frame_path.values[rows],
                          'version': ann.experiment.values[rows], 'n_fg': idx.nfg[fr].astype(np.int64)})
        _, pos, lens = idx.load(fr)
        assert (lens == T.n_fg.values).all()
        pos.astype(np.int16).tofile(OUT / name / 'pos448.i16')
        T.to_parquet(OUT / name / 'frames.parquet')
        log(f'{name}: {len(T):,} frames of {T.obs.nunique()} videos ({T.drop_duplicates("obs").version.value_counts().to_dict()}), '
            f'R448 tokens {int(lens.sum()):,} ({lens.mean():.1f}/frame, 896 arms 4x), frames with 0 tokens '
            f'{int((lens == 0).sum())} [{time.time() - t0:.0f}s]')
    roll = {'store_frames_1fps': int(len(idx.rows)), 'store_tokens448': int(idx.nfg.sum()),
            'store_videos': int(len(set(idx.obs))), 'tokens448_per_frame': float(idx.nfg.mean())}
    (OUT / 'plan.json').write_text(json.dumps({'per_video': args.per_video, 'rollout': roll, 'smoke': bool(args.smoke),
                                               'n_train_frames': int(len(sets['train'])),
                                               'n_eval_frames': int(len(sets['eval']))}, indent=1))
    log(f'rollout counts {roll}')


# ---------------------------------------------------------------------------------------------- offset
def cmd_offset(args):
    """Two checks of 'source frame = 6 k + o' on a sample of videos (v2 and v3):
    (a) exact: run the standardisation's own ffmpeg chain (fps=5 + scale + pad to 512, ffmpeg 6.1.1) on the first 20 s,
        LOSSLESS output, and the same scale chain on every source frame; output frame k must equal one source frame
        exactly (mean abs difference 0) -> the source index of each k.
    (b) independent, against the stored dataset JPEGs (which went through the libopenh264 encode + JPEG): the 12
        highest-motion frames per video (largest JPEG difference between k-1 and k+1, among 60 random k in the
        annotated 10 min); decode_native(6 k + o) for o in -3..3 at 512 -> mean abs grey difference to JPEG k; the
        argmin and its margin over the second best."""
    from PIL import Image
    T = pd.concat([pd.read_parquet(OUT / d / 'frames.parquet') for d in ('train', 'eval')])
    vids = T.drop_duplicates('obs')[['obs', 'version']]
    g = np.random.default_rng(0)
    pick = pd.concat([vids[vids.version == 'v2'].sample(4, random_state=0),
                      vids[vids.version == 'v3'].sample(8, random_state=0)])
    res = {'videos': {}}
    chain = 'scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2'
    all_a, all_b = [], []
    for obs, ver in zip(pick.obs, pick.version):
        src = source_of(ver, obs)
        r = {}
        # (a) exact reproduction (start_frame = 0: '-ss 0' as standardize.py)
        o5 = subprocess.run([str(FFMPEG), '-v', 'error', '-i', str(src), '-ss', '0', '-t', '20', '-vf', f'fps=5,{chain}',
                             '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True).stdout
        o30 = subprocess.run([str(FFMPEG), '-v', 'error', '-i', str(src), '-t', '21', '-vf', chain, '-vsync', '0',
                              '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True).stdout
        A = np.frombuffer(o5, np.uint8).reshape(-1, 512 * 512 * 3)[:90].astype(np.int16)
        B = np.frombuffer(o30, np.uint8).reshape(-1, 512 * 512 * 3).astype(np.int16)
        idx, mad0, mad2 = [], [], []
        for k in range(len(A)):
            lo, hi = max(0, 6 * k - 6), min(len(B), 6 * k + 7)
            m = np.array([np.abs(A[k] - B[j]).mean() for j in range(lo, hi)])
            o = np.argsort(m)
            idx.append(lo + int(o[0]))
            mad0.append(float(m[o[0]]))
            mad2.append(float(m[o[1]]))
        off = np.array(idx) - 6 * np.arange(len(A))
        r['exact'] = {'n_frames': len(A), 'offsets': {int(k): int(v) for k, v in zip(*np.unique(off, return_counts=True))},
                      'max_best_mad': max(mad0), 'min_second_best_mad': min(mad2)}
        all_a += off.tolist()
        # (b) dataset JPEGs, highest-motion frames
        fd = REPO / 'dataset' / f'ants/{ver}/frames/full/{obs}'
        cand = np.sort(g.choice(np.arange(5, 2995), 60, replace=False))
        grey = lambda k: np.asarray(Image.open(fd / f'frame_{k:06d}.jpg').convert('L'), np.float32)  # noqa: E731
        mot = np.array([np.abs(grey(k - 1) - grey(k + 1)).mean() for k in cand])
        ks = cand[np.argsort(-mot)[:12]]
        rows = []
        for k in sorted(ks):
            src_f = [6 * k + o for o in range(-3, 4)]
            nat = decode_native(src, src_f, size=512)
            ref = grey(k)
            m = [float(np.abs(np.asarray(Image.fromarray(x).convert('L'), np.float32) - ref).mean()) for x in nat]
            o = np.argsort(m)
            rows.append({'k': int(k), 'motion': float(mot[list(cand).index(k)]), 'mad_o_m3_p3': [round(x, 3) for x in m],
                         'best_offset': int(o[0]) - 3, 'margin': float(m[o[1]] - m[o[0]])})
            all_b.append(int(o[0]) - 3)
        r['jpeg'] = rows
        r['jpeg_best_offsets'] = {int(k): int(v) for k, v in zip(*np.unique([x['best_offset'] for x in rows],
                                                                           return_counts=True))}
        res['videos'][obs] = r
        log(f'{obs} ({ver}): exact offsets {r["exact"]["offsets"]} (best MAD max {r["exact"]["max_best_mad"]:.3f}, 2nd best '
            f'min {r["exact"]["min_second_best_mad"]:.2f}); JPEG best offsets {r["jpeg_best_offsets"]}, median margin '
            f'{np.median([x["margin"] for x in rows]):.2f} grey levels')
    ua, ca = np.unique(all_a, return_counts=True)
    ub, cb = np.unique(all_b, return_counts=True)
    res['exact_offsets'] = {int(k): int(v) for k, v in zip(ua, ca)}
    res['jpeg_offsets'] = {int(k): int(v) for k, v in zip(ub, cb)}
    res['offset'] = int(ua[np.argmax(ca)])
    res['margins_all'] = [x['margin'] for v in res['videos'].values() for x in v['jpeg']]
    (OUT / 'offset_sample.json').write_text(json.dumps(res, indent=1))
    log(f'exact: {res["exact_offsets"]}; JPEG: {res["jpeg_offsets"]} (most common {res["offset"]})')


def exact_offsets(a):
    """One video: the standardisation chain (fps=5 + scale + pad 512, ffmpeg 6.1.1, lossless) vs the scale chain on every
    source frame, for 512 px frames k in [k0, k0 + 100) at the start (k0 = 0) and at the end (k0 = 2900) of the
    annotated 10 min -> the source index of each k (exact match: mean abs difference 0)."""
    obs, ver = a
    src = source_of(ver, obs)
    chain = 'scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2'
    r = {}
    for k0 in (0, 2900):
        o5 = subprocess.run([str(FFMPEG), '-v', 'error', '-threads', '1', '-i', str(src), '-ss', '0', '-vf',
                             f"fps=5,select='between(n\\,{k0}\\,{k0 + 99})',{chain}", '-vsync', '0', '-frames:v', '100',
                             '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True).stdout
        s0 = max(0, STEP * k0 - 6)
        o30 = subprocess.run([str(FFMPEG), '-v', 'error', '-threads', '1', '-i', str(src), '-vf',
                              f"select='between(n\\,{s0}\\,{STEP * (k0 + 99) + 6})',{chain}", '-vsync', '0',
                              '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True).stdout
        A = np.frombuffer(o5, np.uint8).reshape(-1, 512 * 512 * 3).astype(np.int16)
        B = np.frombuffer(o30, np.uint8).reshape(-1, 512 * 512 * 3).astype(np.int16)
        offs, best, second = [], [], []
        for i in range(len(A)):
            c = STEP * (k0 + i) - s0
            js = list(range(max(0, c - 6), min(len(B), c + 7)))
            m = np.array([np.abs(A[i] - B[j]).mean() for j in js])
            o = np.argsort(m)
            offs.append(js[o[0]] - c)
            best.append(float(m[o[0]]))
            second.append(float(m[o[1]]))
        u, n = np.unique(offs, return_counts=True)
        r[k0] = {'n': len(A), 'offsets': {int(x): int(y) for x, y in zip(u, n)}, 'max_best_mad': max(best),
                 'min_second_mad': min(second)}
    return obs, ver, r


def cmd_offset_all(args):
    from multiprocessing import Pool
    T = pd.concat([pd.read_parquet(OUT / d / 'frames.parquet') for d in ('train', 'eval')])
    vids = T.drop_duplicates('obs')[['obs', 'version']].values.tolist()
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = pool.map(exact_offsets, vids, chunksize=1)
    per, bad, detail = {}, [], {}
    for obs, ver, r in res:
        allo = set(o for k0 in r for o in r[k0]['offsets'])
        exact = all(r[k0]['max_best_mad'] == 0 for k0 in r) and all(r[k0]['n'] == 100 for k0 in r)
        if len(allo) != 1 or not exact:
            bad.append(obs)
        per[obs] = int(min(allo, key=lambda o: -sum(r[k0]['offsets'].get(o, 0) for k0 in r)))
        detail[obs] = {'version': ver, **{str(k): v for k, v in r.items()}}
    by_ver = {}
    for obs, ver, _ in res:
        by_ver.setdefault(ver, {}).setdefault(per[obs], 0)
        by_ver[ver][per[obs]] += 1
    out = {'per_video': per, 'by_version': by_ver, 'videos_not_single_exact_offset': bad,
           'min_second_best_mad': min(v[k]['min_second_mad'] for v in detail.values() for k in ('0', '2900')),
           'detail': detail, 'wall_s': round(time.time() - t0, 1)}
    (OUT / 'offset.json').write_text(json.dumps(out, indent=1))
    log(f'{len(per)} videos in {time.time() - t0:.0f}s: offsets by version {by_ver}; videos without one exact offset '
        f'at start and end: {bad}; smallest second-best MAD {out["min_second_best_mad"]:.3f}')


# ---------------------------------------------------------------------------------------------- bench
def cmd_bench(args):
    import torch
    from PIL import Image
    from transformers import AutoModel
    from src.eci.extract import MODEL_IDS
    dev = torch.device('cuda')
    T = pd.read_parquet(OUT / 'eval/frames.parquet').sample(32, random_state=0)
    pix = []
    for ver, obs, k in zip(T.version, T.obs, T.frame_idx):
        a = decode_native(source_of(ver, obs), [6 * k + offsets()[obs]])[0]
        pix.append(((a.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))
    pix = torch.from_numpy(np.stack(pix)).to(dev)
    out, ref = {'gpu': torch.cuda.get_device_name(0)}, None
    for impl, dt, tf32 in (('eager', 'fp32', False), ('sdpa', 'fp32', False), ('sdpa', 'fp32', True),
                           ('sdpa', 'fp16', False), ('sdpa', 'bf16', False)):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        model = AutoModel.from_pretrained(MODEL_IDS['dinov2_base'], attn_implementation=impl).to(dev).eval()
        cast = {'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[dt]
        hs = []
        with torch.inference_mode():
            for rep in range(2):
                torch.cuda.synchronize()
                t0 = time.time()
                for b in range(0, 32, 16):
                    with torch.autocast('cuda', dtype=cast, enabled=cast is not None):
                        h = model(pixel_values=pix[b:b + 16]).last_hidden_state.float()[:, 1:]
                    if rep == 1:
                        hs.append(h.half().float())
                torch.cuda.synchronize()
                dt_s = (time.time() - t0) / 32
        h = torch.cat(hs).reshape(-1, D)
        key = f'{impl}_{dt}' + ('_tf32' if tf32 else '')
        if ref is None:
            ref = h
        cos = torch.nn.functional.cosine_similarity(h, ref, dim=1)
        out[key] = {'s_per_frame': round(dt_s, 4), 'cos_vs_eager_fp32_min': float(cos.min()),
                    'cos_mean': float(cos.mean()), 'rel_err_mean': float(((h - ref).norm(dim=1) / ref.norm(dim=1)).mean())}
        log(f'{key}: {out[key]}')
        del model
        torch.cuda.empty_cache()
    (OUT / 'bench.json').write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------- encode
def cmd_encode(args):
    """Tokens of one arm (u896 / n896) for the given splits. Precision: --dtype fp32 | fp16 | auto. auto = the rule fixed
    before any result: on the first batch run both fp32 and fp16 (autocast, SDPA attention); fp16 iff the cosine between
    the two is >= 0.999 for EVERY token of the batch (all 64 x 64 positions), else fp32. Both timings are recorded."""
    import torch
    from PIL import Image
    from src.eci.extract import MODEL_IDS, load_encoder
    from transformers import AutoModel
    dev = torch.device(DEV)
    cuda = dev.type == 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stage = stage_dir()
    arm = args.arm
    off = offsets()
    _, proc, _ = load_encoder('dinov2_base', SIZE, 'cpu', center_crop=False)
    model = AutoModel.from_pretrained(MODEL_IDS['dinov2_base'], attn_implementation='sdpa').to(dev).eval()
    G = SIZE // 14
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    def forward(pix, dt):
        if os.environ.get('NRA_FAKE_MODEL'):  # smoke test only: per-patch mean colour tiled to 768 dims + noise
            m = torch.nn.functional.avg_pool2d(pix, 14).flatten(2).transpose(1, 2)
            return m.repeat(1, 1, D // 3) + 0.1 * torch.randn(len(pix), G * G, D)
        cast = {'fp32': None, 'fp16': torch.float16}[dt]
        with torch.inference_mode(), torch.autocast(dev.type, dtype=cast, enabled=cast is not None):
            return model(pixel_values=pix).last_hidden_state.float()[:, 1:]

    class JpgDS(torch.utils.data.Dataset):  # U896: the 512 dataset JPEG, bicubic-upsampled by the HF processor
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            with Image.open(self.paths[i]) as im:
                im = im.convert('RGB')
            return i, proc(images=im, return_tensors='pt')['pixel_values'][0]

    class NatDS(torch.utils.data.IterableDataset):  # N896: native frames decoded per video (worker-sharded videos)
        def __init__(self, F):
            self.groups = [(o, v, d.index.values, d.frame_idx.values) for (o, v), d in F.groupby(['obs', 'version'])]

        def __iter__(self):
            wi = torch.utils.data.get_worker_info()
            for gi, (o, v, ix, ks) in enumerate(self.groups):
                if wi is not None and gi % wi.num_workers != wi.id:
                    continue
                order = np.argsort(ks)
                fr = decode_native(source_of(v, o), (STEP * ks[order] + off[o]).tolist())
                for j, a in zip(ix[order], fr):
                    yield int(j), torch.from_numpy(((a.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())

    dtype = args.dtype
    for split in args.splits:
        info = {'device': torch.cuda.get_device_name(0) if cuda else 'cpu', 'arm': arm, 'split': split, 'bs': args.bs,
                'host': os.uname().nodename}
        F = pd.read_parquet(OUT / split / 'frames.parquet').reset_index(drop=True)
        p448 = np.fromfile(OUT / split / 'pos448.i16', np.int16)
        f0, f1 = part_range(len(F), args.part, args.nparts)
        sfx = part_suffix(args.part, args.nparts)
        st_all = np.r_[0, np.cumsum(F.n_fg.values.astype(np.int64))]
        p448 = p448[st_all[f0]:st_all[f1]]
        F = F.iloc[f0:f1].reset_index(drop=True)
        info.update(part=args.part, nparts=args.nparts, frame_range=[int(f0), int(f1)])
        l448 = F.n_fg.values.astype(np.int64)
        st4, s8 = np.r_[0, np.cumsum(l448)], np.r_[0, np.cumsum(4 * l448)]
        if arm == 'u896':
            t1 = time.time()
            lst = stage / 'jpg512.txt'
            lst.write_text('\n'.join(F.frame_path) + '\n')
            (stage / 'jpg512').mkdir(exist_ok=True)
            subprocess.run(f'tar -C {REPO / "dataset"} -cf - -T {lst} | tar -C {stage / "jpg512"} -xf -', shell=True,
                           check=True)
            info['stage_jpg_s'] = round(time.time() - t1, 1)
            log(f'{split}: staged {len(F):,} 512 px JPGs in {time.time() - t1:.0f}s')
            ds = JpgDS([stage / 'jpg512' / p for p in F.frame_path])
        else:
            ds = NatDS(F)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers, pin_memory=cuda,
                                         prefetch_factor=4 if args.workers else None)
        tok = np.lib.format.open_memmap(stage / f'tok_{arm}.npy', 'w+', np.float16, (int(s8[-1]), D))
        pos = np.empty(int(s8[-1]), np.int16)
        done = np.zeros(len(F), bool)
        gpu_s, n, tw = 0.0, 0, time.time()
        for bi, (ii, pix) in enumerate(dl):
            pix = pix.to(dev, non_blocking=True)
            if dtype == 'auto':
                tm = {}
                for dt in ('fp32', 'fp16', 'fp32', 'fp16'):  # second pass = timed after warm-up
                    sync()
                    tt = time.time()
                    h = forward(pix, dt)
                    sync()
                    tm[dt] = (time.time() - tt) / len(pix)
                    if dt == 'fp32':
                        h32 = h
                    else:
                        h16 = h
                cos = torch.nn.functional.cosine_similarity(h16.reshape(-1, D), h32.reshape(-1, D), dim=1)
                dtype = 'fp16' if float(cos.min()) >= 0.999 else 'fp32'
                info['precision_check'] = {'n_frames': len(pix), 'cos_min': float(cos.min()), 'cos_mean': float(cos.mean()),
                                           's_per_frame_fp32': round(tm['fp32'], 4), 's_per_frame_fp16': round(tm['fp16'], 4),
                                           'chosen': dtype}
                log(f'precision check: {info["precision_check"]}')
                del h32, h16
            sync()
            tt = time.time()
            hs = forward(pix, dtype).half().cpu().numpy()
            gpu_s += time.time() - tt
            assert hs.shape[1] == G * G and np.isfinite(hs).all()
            for j, f in enumerate(ii.numpy()):
                q = children_896(p448[st4[f]:st4[f + 1]])
                tok[s8[f]:s8[f + 1]] = hs[j, q]
                pos[s8[f]:s8[f + 1]] = q
                done[f] = True
            n += len(ii)
            if bi % 200 == 0:
                log(f'  {split} {arm}: {n:,}/{len(F):,} frames [{time.time() - tw:.0f}s]')
        assert done.all(), f'{(~done).sum()} frames not encoded'
        tok.flush()
        del tok
        t1 = time.time()
        shutil.copyfile(stage / f'tok_{arm}.npy', OUT / split / f'tok_{arm}{sfx}.npy')
        pos.tofile(OUT / split / f'pos896_{arm}{sfx}.i16')
        (stage / f'tok_{arm}.npy').unlink()
        if (stage / 'jpg512').exists():
            shutil.rmtree(stage / 'jpg512')
        info.update(dtype=dtype, frames=len(F), tokens=int(s8[-1]), gpu_s=round(gpu_s, 1), wall_s=round(t1 - tw, 1),
                    s_per_frame_wall=round((t1 - tw) / len(F), 4), copy_s=round(time.time() - t1, 1))
        (OUT / f'encode_{arm}_{split}{sfx}.json').write_text(json.dumps(info, indent=1))
        log(f'{split} {arm}: {info}')


# ---------------------------------------------------------------------------------------------- data access
def arm_tokens(arm, split, stage):
    """-> (tokens (N, 768) fp16 in RAM, pos (N,) int16, lens (F,) int64, grid)."""
    import spatial_sae_pilot as ssp
    import multiscale_sae as ms
    F = pd.read_parquet(OUT / split / 'frames.parquet')
    l448 = F.n_fg.values.astype(np.int64)
    t0 = time.time()
    if arm in ('r448', 'l448'):
        idx = ssp.StoreIndex('ants')
        _, pos, lens = ms.stage(idx, F.store_frame.values, stage / f'{split}448')
        tok = np.load(stage / f'{split}448' / 'tok.npy')
        shutil.rmtree(stage / f'{split}448')
        assert (lens == l448).all() and (pos == np.fromfile(OUT / split / 'pos448.i16', np.int16)).all()
        grid = 32
    else:
        whole = OUT / split / f'tok_{arm}.npy'
        sfxs = [''] if whole.exists() else [part_suffix(i, n) for n in (int(sorted((OUT / split).glob(f'tok_{arm}.part*of*.npy'))[0]
                                                                          .stem.split('of')[1]),) for i in range(n)]
        lens = 4 * l448
        tok = np.empty((int(lens.sum()), D), np.float16)
        pos, o = [], 0
        for sx in sfxs:
            loc = stage / f'tok_{arm}{sx}.npy'
            shutil.copyfile(OUT / split / f'tok_{arm}{sx}.npy', loc)
            x = np.load(loc, mmap_mode='r')
            tok[o:o + len(x)] = x
            o += len(x)
            del x
            loc.unlink()
            pos.append(np.fromfile(OUT / split / f'pos896_{arm}{sx}.i16', np.int16))
        pos, grid = np.concatenate(pos), 64
        assert o == len(tok), f'{arm} {split}: parts hold {o} of {len(tok)} tokens'
        assert (pos == children_all(np.fromfile(OUT / split / 'pos448.i16', np.int16), l448)).all()
    assert len(tok) == len(pos) == lens.sum()
    log(f'  {arm} {split}: {len(tok):,} tokens ({tok.nbytes / 1e9:.1f} GB) in RAM [{time.time() - t0:.0f}s]')
    return tok, pos, lens, grid


# ---------------------------------------------------------------------------------------------- train
def cmd_train(args):
    import torch
    import sae_levers as sl
    dev = torch.device(DEV)
    torch.backends.cuda.matmul.allow_tf32 = True  # as sae_levers train
    stage = stage_dir()
    cfg = sl.parse_cfg('w4096')
    p = OUT / 'train_meta.json'
    meta = json.loads(p.read_text()) if p.exists() else {}
    for arm in args.arms:
        tok, pos, lens, grid = arm_tokens(arm, 'train', stage)
        X = torch.from_numpy(tok).to(dev)
        del tok
        n = len(X)
        z = torch.zeros(n, dtype=torch.long, device=dev)
        pools = {'uniform': (X, z, z, torch.zeros(n, dtype=torch.bool, device=dev))}
        meta[arm] = {'unique_tokens': n, 'frames': int(len(lens)), 'tokens_per_frame': float(lens.mean()),
                     'presentations': args.steps * 4096, 'presentations_per_token': args.steps * 4096 / n}
        log(f'{arm}: {meta[arm]}')
        for seed in args.seeds:
            sl.train_one('w4096', cfg, seed, pools, None, args.steps, OUT / 'sae' / f'{arm}_s{seed}',
                         {'config': 'w4096', 'arm': arm, 'lever': cfg, 'domain': 'ants', 'cap': n, 'pool_seed': None},
                         dev)
            torch.cuda.empty_cache()
        del X, pools, z
        torch.cuda.empty_cache()
        p.write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------------------------------------- evaluate
_CACHE = {}


def video_level(codes_np, neurons, res, lab, half, l448):
    """Per label: AUROC pick of each direction -> per-video presence mean on the test half vs annotated rate (r), raw
    and fg-count controlled (readouts.py ants). Returns {label: {...}} and the per-video tables."""
    from readouts import ants_truth, resid, video_readouts
    from diag_video_level import corr
    if 'gt' not in _CACHE:
        _CACHE['gt'] = ants_truth()
    gt = _CACHE['gt']
    out, tabs = {}, {}
    for b in LABELS:
        rr, rc, rb, rbc = [], [], [], []
        for d in res[b]['dirs']:
            j, h = d['auc']['neuron'], d['test_half']
            m = half == h
            x = codes_np[m, neurons.index(j)].astype(np.float32)
            V = video_readouts(x, x, l448[m].astype(np.float32), lab[m], gt['wstart'])
            V = V[V.index.isin(gt.index[gt[f'{b}_rate'].notna()])]
            v = V.presence_mean.values.astype(float)
            vc = resid(v, V.fg_mean.values)
            rr.append(corr(v, gt.loc[V.index, f'{b}_rate'])['pearson'])
            rc.append(corr(vc, gt.loc[V.index, f'{b}_rate'])['pearson'])
            rb.append(corr(v, gt.loc[V.index, f'{b}_bpm'])['pearson'])
            rbc.append(corr(vc, gt.loc[V.index, f'{b}_bpm'])['pearson'])
            tabs[(b, h)] = V
        out[b] = {'video_r_rate': float(np.mean(rr)), 'video_r_rate_fgctrl': float(np.mean(rc)),
                  'video_r_bpm': float(np.mean(rb)), 'video_r_bpm_fgctrl': float(np.mean(rbc)),
                  'n_videos_per_half': [int(len(tabs[(b, h)])) for h in (0, 1)]}
    return out, tabs


def cmd_evaluate(args):
    import torch
    import sae_levers as sl
    import multiscale_sae as ms
    import spatial_sae_pilot as ssp
    from native_res import pick_frames, render_sheet
    from src.eci.sae import load_sae
    dev = torch.device(DEV)
    torch.backends.cuda.matmul.allow_tf32 = False  # as sae_levers evaluate
    stage = stage_dir()
    for d in ('eval', 'codes_best', 'sheets', 'per_video'):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    F = pd.read_parquet(OUT / 'eval/frames.parquet')
    idx = ssp.StoreIndex('ants')
    lab, _, valid = ms.eval_labels('ants', idx, F.store_frame.values)
    assert (lab.row.values == F.row.values).all() and (lab.frame_idx.values == F.frame_idx.values).all()
    units = sorted(set(lab.obs))
    half = lab.obs.map({u: i % 2 for i, u in enumerate(units)}).values
    if not json.loads((OUT / 'plan.json').read_text())['smoke']:
        assert (half == np.load(LEV / 'half.npy')).all(), 'halves differ from levers'
    l448 = F.n_fg.values.astype(np.int64)
    rank = np.argsort(np.argsort(l448 + np.random.default_rng(0).random(len(l448)) * 1e-3))
    decile = np.minimum(rank * 10 // len(l448), 9)  # as sae_levers (448 counts; the 896 counts are exactly 4x)
    Yt = torch.from_numpy(lab[LABELS].values.astype(bool)).to(dev)
    obs = lab.obs.values
    off = offsets()
    log(f'eval: {len(lab):,} frames, {len(units)} videos, halves {np.bincount(half).tolist()}; labels '
        + json.dumps({b: [int(valid[b].sum()), round(float(lab[b][valid[b]].mean()), 4)] for b in LABELS}))
    for arm in args.arms:
        tok, pos, lens, grid = arm_tokens(arm, 'eval', stage)
        st = np.r_[0, np.cumsum(lens)]
        tvid = np.zeros(len(tok), np.int64)
        for seed in args.seeds:
            t0 = time.time()
            key = f'{arm}_s{seed}'
            pth = (LEV / 'sae' / f'w4096_s{seed}' / 'sae.pt') if arm == 'l448' else OUT / 'sae' / key / 'sae.pt'
            sae, norm, _ = load_sae(pth, dev)
            codes, stt = sl.encode_model(sae, norm, None, None, tok, pos, tvid, lens, dev)
            res, _ = sl.score_model(codes, Yt, LABELS, valid, half, decile, dev)
            js = sorted({d[s]['neuron'] for b in LABELS for d in res[b]['dirs'] for s in ('auc', 'ap', 'top1')})
            cb = codes[:, torch.tensor(js, device=dev)].cpu().numpy()
            np.savez_compressed(OUT / 'codes_best' / f'{key}.npz', neurons=np.array(js), codes=cb)
            vid, tabs = video_level(cb, js, res, lab, half, l448)
            if seed == args.seeds[0]:
                pd.concat({f'{b}|h{h}': V for (b, h), V in tabs.items()}).to_parquet(OUT / 'per_video' / f'{key}.parquet')
            e = {'arm': arm, 'seed': seed, 'fve': stt['fve'], 'l0_per_token': stt['l0_per_token'],
                 'dead_frac': stt['dead_frac'], 'n_tokens': stt['n_tokens'], 'tokens_per_frame': float(lens.mean()),
                 'labels': res, 'video': vid}
            (OUT / 'eval' / f'{key}.json').write_text(json.dumps(e, indent=1))
            log(f'{key}: FVE {stt["fve"]:.4f} L0 {stt["l0_per_token"]:.1f} dead {stt["dead_frac"]:.3f} | ' + ' | '.join(
                f'{b}: cfAUROC {res[b]["cf_auroc"]:.3f} cfAP {res[b]["cf_ap"]:.3f} top1 {res[b]["cf_top1"]:.3f} size '
                f'{res[b]["cf_size_ctrl_auroc"]:.3f} vid r {vid[b]["video_r_rate"]:.3f} / ctrl '
                f'{vid[b]["video_r_rate_fgctrl"]:.3f}' for b in LABELS) + f' [{time.time() - t0:.0f}s]')
            if args.sheets and seed == args.seeds[0] and arm in ('r448', 'n896'):
                index = []
                for b in ('groom_any', 'groom_yellow', 'groom_blue', 'onlid_yellow'):
                    e0 = res[b]['dirs'][0]  # selected on half 0, frames of half 1
                    for sel in ('auc', 'top1'):
                        j = e0[sel]['neuron']
                        x = sl.column_codes(sae, norm, None, None, [j], tok, pos, tvid, lens, dev)[0][:, 0]
                        pick = pick_frames(x, (half == e0['test_half']) & valid[b], obs)
                        boxes = []
                        We, be = sae.W_enc[:, [j]], sae.b_enc[[j]]
                        with torch.no_grad():
                            for i in pick:
                                T_ = torch.from_numpy(np.asarray(tok[st[i]:st[i + 1]])).to(dev)
                                pre = ((norm(T_) - sae.b_dec) @ We + be)[:, 0]
                                boxes.append(int(pos[st[i] + int(pre.argmax())]))
                        if arm == 'n896':
                            ims = []
                            for i in pick:
                                fp = stage / 'sheet_frames' / f'{obs[i]}_{F.frame_idx.iat[i]:06d}.png'
                                fp.parent.mkdir(exist_ok=True)
                                if not fp.exists():
                                    from PIL import Image
                                    a = decode_native(source_of(F.version.iat[i], obs[i]),
                                                      [STEP * int(F.frame_idx.iat[i]) + off[obs[i]]])[0]
                                    Image.fromarray(a).save(fp)
                                ims.append((fp, SIZE))
                        else:
                            ims = [(REPO / 'dataset' / F.frame_path.iat[i], 512) for i in pick]
                        yv = [bool(lab[b].iat[i]) for i in pick]
                        items = [(ip, sz, grid, p, y_, f'{obs[i]} f{F.frame_idx.iat[i]} {"POS" if y_ else "neg"} '
                                  f'a={x[i]:.2f}') for (ip, sz), p, y_, i in zip(ims, boxes, yv, pick)]
                        fn = f'{key}__{b}__{sel}_n{j}.jpg'
                        render_sheet(OUT / 'sheets' / fn, items, f'{key} {b} neuron {j} (selected on half 0 by {sel}; '
                                     f'top frames of half 1, <= 2 per video; green = label positive)')
                        index.append({'arm': arm, 'label': b, 'selection': sel, 'neuron': j, 'n_shown': len(pick),
                                      'n_pos_shown': int(sum(yv)), 'test_half_base_rate': res[b]['base_half'][e0['test_half']],
                                      'file': fn})
                        log(f'  sheet {fn}: {sum(yv)}/{len(pick)} label-positive')
                p = OUT / 'sheets/index.json'
                old = json.loads(p.read_text()) if p.exists() else []
                p.write_text(json.dumps([r for r in old if r['arm'] != arm] + index, indent=1))
            del codes
            torch.cuda.empty_cache()
        del tok


# ---------------------------------------------------------------------------------------------- summary
FMETS = ('cf_auroc', 'cf_ap', 'cf_top1', 'cf_size_ctrl_auroc', 'cf_ap_at_auroc')
VMETS = ('video_r_rate', 'video_r_rate_fgctrl', 'video_r_bpm', 'video_r_bpm_fgctrl')


def rule(df, new, old):
    """Pre-registered rule: arm `new` vs arm `old` (3-seed means)."""
    per, noloss, gains = {}, True, []
    for b in LABELS:
        a = df[(df.arm == new) & (df.label == b)]
        o = df[(df.arm == old) & (df.label == b)]
        d_auc = a.cf_auroc.mean() - o.cf_auroc.mean()
        r_ap = a.cf_ap.mean() / o.cf_ap.mean()
        d_vid = a.video_r_rate_fgctrl.mean() - o.video_r_rate_fgctrl.mean()
        g_auc = bool(d_auc >= 0.03 and a.cf_auroc.mean() > o.cf_auroc.max())
        g_ap = bool(r_ap >= 1.3 and a.cf_ap.mean() > o.cf_ap.max())
        g_vid = bool(d_vid >= 0.10)
        nl = bool(d_auc >= -0.02)
        noloss &= nl
        per[b] = {'d_cf_auroc': d_auc, 'ratio_cf_ap': r_ap, 'd_video_r_fgctrl': d_vid, 'no_loss': nl,
                  'gain_auroc': g_auc, 'gain_ap': g_ap, 'gain_video': g_vid}
        if g_auc or g_ap or g_vid:
            gains.append(b)
    return {'per_label': per, 'no_loss_all': bool(noloss), 'labels_with_gain': gains,
            'WIN': bool(noloss and len(gains) > 0)}


def cmd_summary(args):
    rows = []
    for p in sorted((OUT / 'eval').glob('*_s?.json')):
        e = json.loads(p.read_text())
        for b in LABELS:
            r = e['labels'][b]
            rows.append({'arm': e['arm'], 'seed': e['seed'], 'label': b, 'fve': e['fve'], 'l0': e['l0_per_token'],
                         'dead': e['dead_frac'], 'tokens_per_frame': e['tokens_per_frame'],
                         **{k: r[k] for k in FMETS + ('base_rate',)}, **{k: e['video'][b][k] for k in VMETS}})
    df = pd.DataFrame(rows)
    mets = list(FMETS) + list(VMETS) + ['fve']
    agg = df.groupby(['arm', 'label'])[mets].agg(['mean', 'std', 'min', 'max'])
    out = {'per_seed': rows,
           'agg': {f'{a}|{b}': {f'{m}_{s}': float(agg.loc[(a, b), (m, s)]) for m in mets for s in ('mean', 'std', 'min', 'max')}
                   for a, b in agg.index}}
    arms = set(df.arm)
    if {'n896', 'r448'} <= arms:
        out['decision_n896_vs_r448'] = rule(df, 'n896', 'r448')
    if {'n896', 'u896'} <= arms:
        out['n896_vs_u896'] = rule(df, 'n896', 'u896')
    if {'u896', 'r448'} <= arms:
        out['u896_vs_r448'] = rule(df, 'u896', 'r448')
    for f in ['plan.json', 'offset.json', 'offset_sample.json', 'train_meta.json'] + sorted(p.name for p in OUT.glob('encode_*.json')):
        if (OUT / f).exists():
            j = json.loads((OUT / f).read_text())
            out[f[:-5]] = {k: v for k, v in j.items() if k not in ('videos', 'margins_all')}
    (OUT / 'summary.json').write_text(json.dumps(out, indent=1, default=float))
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    for s in ('mean', 'std'):
        print(f'--- {s} over seeds')
        print(agg[['cf_auroc', 'cf_ap', 'cf_top1', 'cf_size_ctrl_auroc', 'video_r_rate', 'video_r_rate_fgctrl', 'fve']]
              .xs(s, axis=1, level=1).round(4))
    print('--- per seed cf_auroc / cf_ap')
    print(df.pivot_table(index=['label', 'arm'], columns='seed', values=['cf_auroc', 'cf_ap']).round(4))
    for k in ('decision_n896_vs_r448', 'n896_vs_u896', 'u896_vs_r448'):
        if k in out:
            print(k, json.dumps(out[k], indent=1, default=float))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['plan', 'offset', 'offset_all', 'bench', 'encode', 'train', 'evaluate', 'summary'])
    p.add_argument('--per-video', type=int, default=80)
    p.add_argument('--arm', default='n896', choices=['u896', 'n896'])
    p.add_argument('--splits', nargs='+', default=['train', 'eval'])
    p.add_argument('--part', type=int, default=0, help='encode: frame-range part of the split (contiguous frames)')
    p.add_argument('--nparts', type=int, default=1)
    p.add_argument('--dtype', default='auto', choices=['auto', 'fp32', 'fp16'])
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--arms', nargs='+', default=ARMS)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--sheets', action='store_true')
    p.add_argument('--smoke', action='store_true', help='plan: tiny frame sets for the CPU smoke test (with NRA_OUT)')
    args = p.parse_args()
    {'plan': cmd_plan, 'offset': cmd_offset, 'offset_all': cmd_offset_all, 'bench': cmd_bench, 'encode': cmd_encode, 'train': cmd_train,
     'evaluate': cmd_evaluate, 'summary': cmd_summary}[args.cmd](args)


if __name__ == '__main__':
    main()
