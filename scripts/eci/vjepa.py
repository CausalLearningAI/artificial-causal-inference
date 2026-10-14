"""
V-JEPA pilot (ECI, mice v1): swap the frozen IMAGE encoder (DINOv2-base, 448 px) for a frozen VIDEO encoder (V-JEPA 2
ViT-L, facebook/vjepa2-vitl-fpc64-256) and keep the rest of the pipeline: today's fg448 animal mask, the Matryoshka
BatchTopK SAE of the levers recipe (4096 latents, k = 16, contact_sae.train_one = motion_t3 / sae_levers recipe),
frame read-out = MAX over kept patches.

Encoder setup
    model     facebook/vjepa2-vitl-fpc64-256 (HF transformers 4.57.3 VJEPA2Model, encoder only: skip_predictor=True),
              loaded with config crop_size=448: the HF RoPE derives (t, h, w) of every token from crop_size //
              patch_size, so without the override a 448 input gets WRONG 2D positions (16-wide rows instead of 28).
              fp32 weights with TF32 matmuls, SDPA attention (bf16 weights tested in the smoke test: token cosine
              vs fp32 mean 0.985, min 0.43, so not used); normalised pixels held in fp16; tokens rounded to fp16.
    input     the standardised 5 fps 512 x 512 mp4 (data/mice/v1/observations/full), decoded sequentially in the job
              with ffmpeg 6.1.1 (-vf fps=5,format=rgb24, the get_frames.py chain, raw RGB instead of the q2 JPEG),
              whole frame resized 512 -> 448 (bicubic, antialias; no crop, as the DINOv2 448 store), ImageNet mean/std.
    clip      16 frames at 5 fps (3.2 s) for target frame t: frames t-7 .. t+8, indices clamped to the video
              (edge frame repeated), so t is always clip position 7.
    token     tubelets are 2 frames x 16 x 16 px -> 8 x 28 x 28 tokens; the per-frame token = tubelet 3 = clip
              positions (6, 7) = frames (t-1, t). Last-layer (post-LayerNorm) encoder output.
    mask      the fg448 store mask (32 x 32 grid, 16 px cells of the 512 frame = 14 px at 448) rasterised at 448;
              a V-JEPA patch (16 px at 448) is kept iff >= 50% of its area is covered by kept DINOv2 patches.
Frames
    train     the 10,000 unlabelled train frames of the native_res / res_pool pilots (results/vision/eci_res_pool/mice/
              train/frames.parquet, 100 per unannotated train video).
    eval      the 20,000-frame annotated B subset (results/vision/eci_repr_diag/mice/b/subset.parquet; inverse-sampling
              weights w; nose_nose = Y_nn OR Y_np, nose_tail = Y_nt). Scored on all 20,000 and on the 16,775 'usable'
              frames of the contact_sae table (results/vision/eci_mice_pairs/bsub/geom.npz idx) where C1 was reported.
Steps (scripts/eci/vjepa.sh)
    plan      CPU  frame tables, 28-grid masks, encode shards -> OUT/plan/
    smoke     GPU  2 clips: shapes, crop_size override check, bf16 vs fp32, decoded frame vs dataset JPG, timing
    encode    GPU  array task: decode + encode the task's videos -> OUT/tokens/<split>_partXXofYY.{npy,npz}
    probe     GPU  supervised ceiling: diag_mice_patches mean_lin + mil_ctx (pool-grouped 5-fold CV, 3 seeds)
    train     GPU  SAE seeds -> OUT/sae/vjepa_s<seed>/
    c1        GPU  C1 (DINOv2 + motion, results/vision/eci_t3_motion/mice/sae/M5_s*) frame-max codes on all 20,000
                   B frames -> OUT/c1_codes/
    eval      GPU  cross-fitted best neuron (weighted, contact_sae.cross_fit) + size-controlled AUROC at prefixes
                   512 / 4096, V-JEPA and C1, frames all / usable -> OUT/eval/
    sheets    GPU  contact sheets: best nose_nose / nose_tail neurons, 30 random eligible neurons -> OUT/sheets/
    summary   any  OUT/table.md, OUT/summary.json
Outputs: results/vision/eci_vjepa/ (gitignored).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
OUT = Path(os.environ.get('VJ_OUT') or REPO / 'results/vision/eci_vjepa')
RP = REPO / 'results/vision/eci_res_pool/mice/train'
BDIR = REPO / 'results/vision/eci_repr_diag/mice/b'
BSUB = REPO / 'results/vision/eci_mice_pairs/bsub'
C1DIR = REPO / 'results/vision/eci_t3_motion/mice/sae'
ANN = REPO / 'dataset/mice/v1/annotations.csv'
VID = REPO / 'data/mice/v1/observations/full'
FFMPEG = '/nfs/scistore19/locatgrp/rcadei/.conda/envs/ffmpeg611/bin/ffmpeg'
MODEL = 'facebook/vjepa2-vitl-fpc64-256'
S, G, PS = 448, 28, 16
CLIP, TPOS = 16, 7  # clip length, position of t in the clip
TUB = TPOS // 2  # tubelet holding t (frames t-1, t)
D = 1024
LABELS = ['nose_nose', 'nose_tail']
PREFIXES = (512, 4096)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def TAG(args):
    return '' if args.layer in (None, 24) else f'_L{args.layer}'


def stage_dir():
    d = Path(os.environ.get('LOCAL_DIR') or '')
    if not str(d).startswith('/localhome'):
        raise RuntimeError(f'LOCAL_DIR must be under /localhome, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------- plan
def map_mask(p448):
    """kept 32 x 32 positions -> kept 28 x 28 positions (>= 50% of the 16 px patch at 448 covered)."""
    m = np.zeros(32 * 32, np.float32)
    m[np.asarray(p448, np.int64)] = 1
    r = np.kron(m.reshape(32, 32), np.ones((14, 14), np.float32))  # 448 x 448
    cov = r.reshape(G, PS, G, PS).mean((1, 3))
    return np.flatnonzero(cov.ravel() >= 0.5).astype(np.int16)


def cmd_plan(args):
    out = OUT / 'plan'
    out.mkdir(parents=True, exist_ok=True)
    fp_all = pd.read_csv(ANN, usecols=['frame_path'])['frame_path'].values
    T = pd.read_parquet(RP / 'frames.parquet')
    tp = np.fromfile(RP / 'pos448.i16', np.int16)
    B = pd.read_parquet(BDIR / 'subset.parquet')
    bp = np.fromfile(BDIR / 'pos448.i16', np.int16)
    bl = np.load(BDIR / 'frames.npz')['lens448']
    assert (bl == B.n_fg.values).all() and len(bp) == bl.sum() and len(tp) == T.n_fg.sum()
    B['frame_path'] = fp_all[B.row.values]
    usable = np.zeros(len(B), bool)
    usable[np.load(BSUB / 'geom.npz')['idx']] = True
    B['usable'] = usable
    info = {}
    for name, F, p448 in (('train', T, tp), ('eval', B, bp)):
        F = F.copy()
        F['video'] = [Path(p).parent.name for p in F.frame_path]
        F['k'] = [int(Path(p).stem.split('_')[1]) for p in F.frame_path]
        if 'frame_idx' in F:
            assert (F.k.values == F.frame_idx.values).all()
        st = np.r_[0, np.cumsum(F.n_fg.values)]
        pos28 = [map_mask(p448[st[i]:st[i + 1]]) for i in range(len(F))]
        F['n28'] = [len(p) for p in pos28]
        assert (F.n28 > 0).all(), 'a frame keeps no 28-grid patch'
        np.concatenate(pos28).astype(np.int16).tofile(out / f'{name}_pos28.i16')
        F.to_parquet(out / f'{name}_frames.parquet')
        vids = F.groupby('video').size().sort_values(ascending=False)
        assert all((VID / f'{v}.mp4').exists() for v in vids.index)
        # greedy balance of clips over shards
        nsh = args.shards_train if name == 'train' else args.shards_eval
        load, shard = np.zeros(nsh), {}
        for v, n in vids.items():
            s = int(np.argmin(load)); shard[v] = s; load[s] += n + 40  # + per-video decode overhead
        (out / f'{name}_shards.json').write_text(json.dumps({'n_shards': nsh, 'shard_of_video': shard}))
        info[name] = {'n_frames': len(F), 'n_videos': int(len(vids)), 'n_shards': nsh,
                      'kept448_mean': float(F.n_fg.mean()), 'kept28_mean': float(F.n28.mean()),
                      'kept28_median': float(F.n28.median()), 'kept28_min': int(F.n28.min()),
                      'kept28_max': int(F.n28.max()), 'kept_area_ratio_28_vs_448': float(
                          (F.n28 * 16 ** 2).sum() / (F.n_fg * 14 ** 2).sum()), 'tokens28': int(F.n28.sum())}
        if name == 'eval':
            info[name]['n_usable'] = int(F.usable.sum())
        log(f'{name}: {json.dumps(info[name])}')
    (out / 'plan.json').write_text(json.dumps(info, indent=1))


def frames_table(split):
    F = pd.read_parquet(OUT / 'plan' / f'{split}_frames.parquet')
    pos = np.fromfile(OUT / 'plan' / f'{split}_pos28.i16', np.int16)
    return F, pos


# ---------------------------------------------------------------------------------------------- decode / encode
def decode_video(video, need, threads=2):
    """Sequential decode of the 5 fps mp4 -> ({frame index: uint8 (512, 512, 3)} for the needed frames, n_frames)."""
    need = set(int(x) for x in need)
    cmd = [FFMPEG, '-nostdin', '-loglevel', 'error', '-threads', str(threads), '-i', str(VID / f'{video}.mp4'),
           '-vf', 'fps=5,format=rgb24', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 24)
    nb = 512 * 512 * 3
    out, n, early = {}, 0, False
    buf = bytearray(nb)
    mv = memoryview(buf)
    while True:
        got = 0
        while got < nb:
            r = p.stdout.readinto(mv[got:])
            if not r:
                break
            got += r
        if got < nb:
            break
        if n in need:
            out[n] = np.frombuffer(bytes(buf), np.uint8).reshape(512, 512, 3)
        n += 1
        if len(out) == len(need):  # every needed frame decoded: stop (stream length unknown, not needed)
            early = True
            break
    p.stdout.close()
    if early:
        p.kill()
    p.wait()
    return out, (None if early else n)


def clip_index(k, n_frames):
    idx = np.arange(k - TPOS, k - TPOS + CLIP)
    return np.clip(idx, 0, (n_frames - 1) if n_frames else 10 ** 9)


def load_model(dev, crop=S, dtype=None):
    import torch
    from transformers import VJEPA2Model
    torch.backends.cuda.matmul.allow_tf32 = True  # fp32 weights + TF32 matmuls (bf16 gave token cosine min 0.43)
    m = VJEPA2Model.from_pretrained(MODEL, crop_size=crop, dtype=dtype or torch.float32,
                                    attn_implementation='sdpa').to(dev).eval()
    assert m.config.crop_size == crop
    return m


def prep(frames_u8, dev):
    """(n, 512, 512, 3) uint8 numpy -> (n, 3, 448, 448) normalised float32 GPU."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(frames_u8).to(dev).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(S, S), mode='bicubic', antialias=True, align_corners=False).clamp_(0, 1)
    m = torch.tensor(MEAN, device=dev)[None, :, None, None]
    s = torch.tensor(STD, device=dev)[None, :, None, None]
    return (x - m) / s


def encode_video(model, video, ks, need_frames, n_frames, dev, bs, gpu_t, layers=None):
    """targets ks of one video -> list of (G*G, D) fp16 numpy (tubelet TUB, all positions)."""
    import torch
    keys = np.array(sorted(need_frames))
    X = torch.empty((len(keys), 3, S, S), dtype=torch.float16, device=dev)  # normalised pixels, fp16 storage
    for a in range(0, len(keys), 256):
        X[a:a + 256] = prep(np.stack([need_frames[int(k)] for k in keys[a:a + 256]]), dev).half()
    lut = {int(k): i for i, k in enumerate(keys)}
    res = []
    for a in range(0, len(ks), bs):
        ci = np.array([[lut[int(j)] for j in clip_index(k, n_frames)] for k in ks[a:a + bs]])
        torch.cuda.synchronize(); t0 = time.time()
        with torch.inference_mode():
            v = X[torch.from_numpy(ci).to(dev)]  # (b, 16, 3, S, S)
            if layers is None:
                h = model(pixel_values_videos=v, skip_predictor=True).last_hidden_state
                hs = {24: h}
            else:  # hidden_states[L] = raw residual output of block L (no final LayerNorm), L < n blocks run
                o = model.encoder(pixel_values_videos=v, output_hidden_states=True)
                hs = {L: o.hidden_states[L] for L in layers}
            for L, h in hs.items():
                assert h.shape[1] == (CLIP // 2) * G * G and h.shape[2] == D, h.shape
                hs[L] = h[:, TUB * G * G:(TUB + 1) * G * G].float().half().cpu().numpy()
        torch.cuda.synchronize(); gpu_t[0] += time.time() - t0
        if layers is None:
            res.extend(list(hs[24]))
        else:
            res.extend([{L: hs[L][i] for L in layers} for i in range(len(hs[layers[0]]))])
    return res


def needed(ks, n_guess=None):
    s = set()
    for k in ks:
        s.update(int(j) for j in np.clip(np.arange(k - TPOS, k - TPOS + CLIP), 0, None))
    return s


def cmd_encode(args):
    import torch
    dev = torch.device('cuda')
    F, pos = frames_table(args.split)
    sh = json.loads((OUT / 'plan' / f'{args.split}_shards.json').read_text())
    nsh = sh['n_shards']
    vids = sorted(v for v, s in sh['shard_of_video'].items() if s == args.task)
    if args.max_videos:
        vids = vids[:args.max_videos]
    st = np.r_[0, np.cumsum(F.n28.values)]
    model = load_model(dev)
    layers = args.layers or None
    if layers:  # run only the blocks needed: hidden_states[L] for L <= max(layers) needs max(layers) + 1 blocks
        model.encoder.layer = model.encoder.layer[:max(layers) + 1]
    log(f'layers {layers or "last (24, post-LN)"}')
    log(f'{args.split} task {args.task}/{nsh}: {len(vids)} videos, '
        f'{int(F.video.isin(vids).sum())} clips; model {MODEL} crop {model.config.crop_size}')
    stage = stage_dir()
    toks, fids, lens = [], [], []
    gpu_t, t0, nclip = [0.0], time.time(), 0
    nf_of = {}

    def dec(v):
        ks = F.k.values[F.video.values == v]
        fr, n = decode_video(v, needed(ks))
        # clamp at the video end: the needed set assumed frames up to k+8 exist; if the stream ended, map them
        if n is not None:
            last = max(j for j in fr)
            for j in needed(ks):
                if j >= n:
                    fr[j] = fr[last]
        missing = needed(ks) - set(fr)
        assert not missing, (v, sorted(missing)[:5], n)
        return v, fr, n

    with ThreadPoolExecutor(args.decoders) as ex:
        futs = [ex.submit(dec, v) for v in vids[:args.decoders + 1]]
        nxt = len(futs)
        while futs:
            v, fr, n = futs.pop(0).result()
            if nxt < len(vids):
                futs.append(ex.submit(dec, vids[nxt])); nxt += 1
            rows = np.flatnonzero(F.video.values == v)
            ks = F.k.values[rows]
            # frames past the stream end were filled with the last frame, so clamp indices with n=None is exact
            H = encode_video(model, v, ks, fr, None, dev, args.bs, gpu_t, layers)
            for r, h in zip(rows, H):
                p = pos[st[r]:st[r + 1]].astype(np.int64)
                toks.append({L: h[L][p] for L in layers} if layers else h[p]); fids.append(r); lens.append(len(p))
            nf_of[v] = n
            nclip += len(rows)
            del fr
            log(f'  {v}: {len(rows)} clips (stream frames {n}); total {nclip} clips [{time.time() - t0:.0f}s, '
                f'gpu {gpu_t[0]:.0f}s]')
    name = f'{args.split}_part{args.task:02d}of{nsh:02d}'
    meta = {'frame': np.array(fids), 'lens': np.array(lens)}
    info = {'n_clips': nclip, 'wall_s': round(time.time() - t0, 1),
            'gpu_s': round(gpu_t[0], 1), 'gpu_s_per_clip': gpu_t[0] / max(nclip, 1), 'bs': args.bs,
            'gpu': torch.cuda.get_device_name(0), 'layers': layers, 'n_frames_stream': nf_of}
    for L in (layers or [None]):
        tok = np.concatenate([t[L] for t in toks] if L is not None else toks)
        info['n_tokens'] = int(len(tok))
        sd = stage / (f'L{L}' if L is not None else 'last')
        sd.mkdir(exist_ok=True)
        np.save(sd / f'{name}.npy', tok)
        np.savez(sd / f'{name}.npz', **meta)
        (sd / f'{name}.json').write_text(json.dumps(info, indent=1))
        od = OUT / ('tokens' + (f'_L{L}' if L is not None else ''))
        od.mkdir(parents=True, exist_ok=True)
        for ext in ('npy', 'npz', 'json'):
            subprocess.run(['cp', str(sd / f'{name}.{ext}'), str(od / f'{name}.{ext}')], check=True)
        del tok
    log(f'done: {json.dumps({k: v for k, v in info.items() if k != "n_frames_stream"})}')


def load_tokens(split, tag=''):
    """-> (tok (N, D) fp16, pos28 (N,), lens (F,)) in the plan's frame order."""
    F, pos = frames_table(split)
    parts = sorted((OUT / ('tokens' + tag)).glob(f'{split}_part*of*.npz'))
    nsh = json.loads((OUT / 'plan' / f'{split}_shards.json').read_text())['n_shards']
    assert len(parts) == nsh, f'{len(parts)}/{nsh} {split} parts'
    fr, ln, tk = [], [], []
    for p in parts:
        z = np.load(p)
        fr.append(z['frame']); ln.append(z['lens']); tk.append(np.load(p.with_suffix('.npy')))
    fr, ln = np.concatenate(fr), np.concatenate(ln)
    assert len(fr) == len(F) and len(set(fr)) == len(F)
    assert (ln == F.n28.values[fr]).all()
    tk = np.concatenate(tk)
    st = np.r_[0, np.cumsum(ln)]
    order = np.argsort(fr)
    tok = np.concatenate([tk[st[i]:st[i + 1]] for i in order])
    lens = F.n28.values
    assert len(tok) == lens.sum() == len(pos)
    return tok, pos, lens


# ---------------------------------------------------------------------------------------------- smoke
def cmd_smoke(args):
    import torch
    from PIL import Image
    dev = torch.device('cuda')
    F, pos = frames_table('eval')
    r = int(np.flatnonzero(F.k.values > 20)[0])
    v, k = F.video.iat[r], int(F.k.iat[r])
    t0 = time.time()
    fr, n = decode_video(v, needed([k, 3]))
    log(f'decoded {len(fr)} frames of {v} in {time.time() - t0:.1f}s (stream ended: {n})')
    jpg = np.asarray(Image.open(REPO / 'dataset' / F.frame_path.iat[r]).convert('RGB'), np.int16)
    dd = np.abs(jpg - fr[k].astype(np.int16))
    log(f'decoded frame {k} vs dataset JPG: max abs {dd.max()}, mean abs {dd.mean():.3f}')
    for crop, dtype in ((S, torch.bfloat16), (S, torch.float32), (256, torch.bfloat16)):
        model = load_model(dev, crop, dtype)
        g = [0.0]
        H = encode_video(model, v, [k, 3], fr, None, dev, 2, g)
        log(f'crop_size {crop} {dtype}: per-frame token block {H[0].shape}, |h| {np.linalg.norm(H[0].astype(np.float32), axis=1).mean():.2f}, gpu {g[0]:.2f}s')
        if crop == S and dtype == torch.bfloat16:
            ref = H
        else:
            a, b = ref[0].astype(np.float32), H[0].astype(np.float32)
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            log(f'  vs crop 448 bf16: token cosine mean {cos.mean():.4f} min {cos.min():.4f}')
        del model
        torch.cuda.empty_cache()
    model = load_model(dev)
    clips = needed([k + i for i in range(16)])
    fr2, _ = decode_video(v, clips)
    for bs in (4, 8, 16):
        g = [0.0]
        encode_video(model, v, [k + i for i in range(16)], fr2, None, dev, bs, g)
        log(f'bs {bs}: {g[0] / 16:.3f} GPU-s / clip, peak mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB')
    st = np.r_[0, np.cumsum(F.n28.values)]
    log(f'kept 28-grid patches of frame {r}: {F.n28.iat[r]} (448 grid {F.n_fg.iat[r]}); positions {pos[st[r]:st[r] + 8]}')


# ---------------------------------------------------------------------------------------------- probe
def cmd_probe(args):
    import torch
    import diag_mice_patches as dmp
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    F, _ = frames_table('eval')
    sub = pd.read_parquet(BDIR / 'subset.parquet')
    assert (sub.row.values == F.row.values).all()
    tag = TAG(args)
    tok, pos, lens = load_tokens('eval', tag)
    bag = dmp.Bag(np.ascontiguousarray(tok), pos, lens, G, dev)
    res = dmp.cross_validate(bag, sub, sub.w.values, dev, args.probes, args.seeds, None, 'vjepa448', args.steps)
    summ = dmp.summarise(res)
    (OUT / ('probe' + tag)).mkdir(parents=True, exist_ok=True)
    (OUT / ('probe' + tag) / 'results.json').write_text(json.dumps({'summary': summ, 'runs': res, 'wall_s': time.time() - t0,
                                                            'gpu': torch.cuda.get_device_name(0)}, indent=1))
    for s in summ:
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))


# ---------------------------------------------------------------------------------------------- SAE
def cmd_train(args):
    import torch
    import contact_sae as cs
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    tag = TAG(args)
    tok, _, _ = load_tokens('train', tag)
    X = torch.from_numpy(tok).to(dev)
    log(f'train tokens {tuple(X.shape)}')
    for seed in args.seeds:
        o = OUT / 'sae' / f'vjepa{tag}_s{seed}'
        if (o / 'sae.pt').exists() and not args.overwrite:
            log(f'[SKIP] {o}'); continue
        cs.train_one(X, [D], 4096, seed, args.steps, o, {'config': 'vjepa_vitl_448_tub3', 'width': 4096}, dev)
        torch.cuda.empty_cache()


def b_store_frames(idx, sub):
    f_of_row = pd.Series(np.arange(len(idx.rows)), index=idx.rows)
    fr = f_of_row.loc[sub.row.values].values
    assert (idx.rows[fr] == sub.row.values).all() and (idx.nfg[fr] == sub.n_fg.values).all()
    return fr


def cmd_c1(args):
    import torch
    import contact_sae as cs
    import motion_t3 as mt
    import spatial_sae_pilot as ssp
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    od = OUT / 'c1_codes'
    od.mkdir(parents=True, exist_ok=True)
    sub = pd.read_parquet(BDIR / 'subset.parquet')
    idx = ssp.StoreIndex('mice')
    d5m = mt.D5Matcher(idx, 'mice')
    fr = b_store_frames(idx, sub)
    pre = cs.Preloaded(idx, d5m, fr)
    order = np.argsort(fr)
    info = {'c1_d5_unmatched_tokens': int((~pre.ok).sum())}
    for seed in (0, 1, 2):
        sae, norm, _ = load_sae(C1DIR / f'M5_s{seed}' / 'sae.pt', dev)
        codes, st = mt.encode_codes(sae, norm, 'M5', pre.tok, {-5: pre.prev}, pre.lens, dev)
        c = np.empty((len(fr), sae.n_latents), np.float16)
        c[order] = codes.cpu().numpy()
        np.save(od / f'M5_s{seed}.npy', c)
        info[f's{seed}'] = {'fve': st['fve']}
        log(f'C1 s{seed}: FVE {st["fve"]:.4f}')
        del codes
        torch.cuda.empty_cache()
    (od / 'info.json').write_text(json.dumps(info, indent=1))


def frame_max_tokens(sae, norm, tok, lens, dev, chunk=65536, want_arg=None):
    """per-token codes -> frame-max codes (F, m) fp16 GPU; FVE, L0; want_arg: neuron list -> per-frame argmax token."""
    import torch
    m = sae.n_latents
    Fn = len(lens)
    st = np.r_[0, np.cumsum(lens)]
    frame = np.repeat(np.arange(Fn), lens)
    codes = torch.zeros(Fn, m, dtype=torch.float16, device=dev)
    s1 = s2 = sse = 0
    l0 = 0
    arg = None
    if want_arg is not None:
        J = torch.tensor(want_arg, device=dev)
        best = torch.full((Fn, len(want_arg)), -1.0, device=dev)
        arg = torch.zeros((Fn, len(want_arg)), dtype=torch.long, device=dev)
    with torch.no_grad():
        for a in range(0, len(tok), chunk):
            x = norm(torch.from_numpy(np.ascontiguousarray(tok[a:a + chunk])).to(dev))
            z = sae.encode(x, mode='threshold')
            xh = sae.decode(z)
            s1 = s1 + x.double().sum(0); s2 = s2 + x.double().pow(2).sum(0); sse = sse + (x - xh).double().pow(2).sum(0)
            l0 += float((z > 0).sum())
            fr = torch.from_numpy(frame[a:a + chunk]).to(dev)
            mx = torch.zeros(Fn, m, device=dev)
            mx.index_reduce_(0, fr, z, 'amax', include_self=True)
            codes = torch.maximum(codes, mx.half())
            if want_arg is not None:
                zz = z[:, J]
                ti = torch.arange(a, a + len(z), device=dev)
                for q in range(len(want_arg)):
                    # per frame argmax over this chunk
                    v = zz[:, q]
                    cur = torch.full((Fn,), -1.0, device=dev).index_reduce_(0, fr, v, 'amax', include_self=True)
                    hit = (v == cur[fr]) & (v > best[fr, q])
                    upd_f = fr[hit]
                    arg[upd_f, q] = ti[hit]
                    best[:, q] = torch.maximum(best[:, q], cur)
    tss = s2 - s1.pow(2) / len(tok)
    return codes, {'fve': float(1 - sse.sum() / tss.sum()), 'l0': l0 / len(tok)}, arg


def size_ctrl(codes_col, y, w, dec):
    import contact_sae as cs
    v = []
    for d in np.unique(dec):
        m = dec == d
        if y[m].sum() < 10 or (~y[m]).sum() < 10:
            continue
        v.append(cs.wmetrics(codes_col[m], y[m], w[m])[0])
    return float(np.mean(v)) if v else float('nan')


def score(codes, lab, rows, half_all, dev):
    """cross_fit (weighted, contact_sae) on `rows` + size-controlled AUROC of the AUROC-picked neuron on its test half."""
    import torch
    import contact_sae as cs
    import diag_mice_patches as dmp
    L = lab.iloc[rows].reset_index(drop=True)
    half = half_all[rows]
    Y = L[LABELS].values.astype(bool)
    c = codes[torch.from_numpy(rows).to(dev)]
    r = cs.cross_fit(c, torch.from_numpy(Y).to(dev), torch.from_numpy(L.w.values).to(dev), half, dev)
    dec = dmp.size_deciles(L.n_fg.values)
    for k, b in enumerate(LABELS):
        sc = []
        for di, e in enumerate(r[b]['dirs']):
            t = half == 1 - di
            j = e['auc']['neuron']
            col = c[:, j].float().cpu().numpy()
            e['auc']['test_size_ctrl_auc'] = size_ctrl(col[t], Y[t, k], L.w.values[t], dec[t])
            sc.append(e['auc']['test_size_ctrl_auc'])
        r[b]['cf_size_ctrl_auroc'] = float(np.nanmean(sc))
    return r


def cmd_eval(args):
    import torch
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    F, _ = frames_table('eval')
    sub = pd.read_parquet(BDIR / 'subset.parquet')
    assert (sub.row.values == F.row.values).all()
    rows = {'all': np.arange(len(F)), 'usable': np.flatnonzero(F.usable.values)}
    od = OUT / 'eval'
    od.mkdir(parents=True, exist_ok=True)
    (OUT / 'codes').mkdir(exist_ok=True)
    import contact_sae as cs
    half_all = cs.video_units(sub.obs.values)
    log(f'halves (all frames) {np.bincount(half_all).tolist()}')
    tag = TAG(args)
    tok, pos, lens = load_tokens('eval', tag)
    jobs = [(f'vjepa{tag}_s{s}', OUT / 'sae' / f'vjepa{tag}_s{s}' / 'sae.pt') for s in args.seeds]
    if args.c1:
        jobs += [(f'C1_M5_s{s}', None) for s in (0, 1, 2)]
    for key, ck in jobs:
        if (od / f'{key}.json').exists() and not args.overwrite:
            log(f'[SKIP] {key}'); continue
        if ck is None:
            codes = torch.from_numpy(np.load(OUT / 'c1_codes' / f'{key[3:]}.npy')).to(dev)
            st = {}
        else:
            sae, norm, _ = load_sae(ck, dev)
            codes, st, _ = frame_max_tokens(sae, norm, tok, lens, dev)
            np.save(OUT / 'codes' / f'{key}.npy', codes.cpu().numpy())
        res = {'stats': st}
        for fs, rr in rows.items():
            for p in PREFIXES:
                r = score(codes[:, :p].contiguous(), sub, rr, half_all, dev)
                res[f'{fs}_m{p}'] = r
                log(f'  {key} {fs} prefix {p}: ' + ', '.join(
                    f"{b} AUROC {r[b]['cf_auroc']:.3f} AP {r[b]['cf_ap']:.4f} top1 {r[b]['cf_top1']:.3f} "
                    f"size-ctrl {r[b]['cf_size_ctrl_auroc']:.3f}" for b in LABELS))
        rate = (codes > 0).float()
        w = torch.from_numpy(sub.w.values).to(dev).float()
        res['fire_rate_w'] = ((rate * w[:, None]).sum(0) / w.sum()).cpu().numpy().tolist()
        (od / f'{key}.json').write_text(json.dumps(res, indent=1, default=float))
        del codes
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- sheets
def cmd_sheets(args):
    import torch
    from PIL import Image, ImageDraw
    from src.eci.sae import load_sae
    import contact_sae as cs
    dev = torch.device('cuda')
    F, pos = frames_table('eval')
    sub = pd.read_parquet(BDIR / 'subset.parquet')
    tag = TAG(args)
    key = f'vjepa{tag}_s{args.sheet_seed}'
    R = json.loads((OUT / 'eval' / f'{key}.json').read_text())
    picks = []  # (name, neuron, frame rows restricted to, note)
    half = cs.video_units(sub.obs.values)
    for b in LABELS:
        for sel in ('auc', 'top1'):
            e = R['all_m4096'][b]['dirs'][0]  # selected on half 0, shown on half 1
            picks.append((f'best_{b}_{sel}', e[sel]['neuron'], half == 1,
                          f'{sel}-selected on half 0 for {b}; test half 1 AUROC {e[sel]["test_auc"]:.3f} AP '
                          f'{e[sel]["test_ap"]:.3f} top1 {e[sel]["test_top1"]:.3f}'))
    rate = np.array(R['fire_rate_w'])
    el = np.flatnonzero(rate >= 0.01)
    rg = np.random.default_rng(0)
    for j in sorted(rg.choice(el, min(args.n_random, len(el)), replace=False).tolist()):
        picks.append((f'rand_n{j:04d}', j, np.ones(len(F), bool), f'random eligible neuron, weighted fire rate {rate[j]:.3f}'))
    for j in args.extra_neurons:
        picks.append((f'extra_n{j:04d}', j, np.ones(len(F), bool), f'neuron {j}, weighted fire rate {rate[j]:.3f}'))
    tok, _, lens = load_tokens('eval', tag)
    sae, norm, _ = load_sae(OUT / 'sae' / key / 'sae.pt', dev)
    J = sorted({p[1] for p in picks})
    codes, _, arg = frame_max_tokens(sae, norm, tok, lens, dev, want_arg=J)
    codes = codes[:, J].float().cpu().numpy()
    arg = arg.cpu().numpy()
    st = np.r_[0, np.cumsum(lens)]
    tile_plan = []
    for name, j, msk, note in picks:
        q = J.index(j)
        a = codes[:, q].copy()
        a[~msk] = -1
        o = np.argsort(-a)
        chosen, per_v = [], {}
        for f in o:
            if a[f] <= 0 or len(chosen) >= args.n_tiles:
                break
            v = F.video.iat[f]
            if per_v.get(v, 0) >= 2:
                continue
            per_v[v] = per_v.get(v, 0) + 1
            chosen.append(int(f))
        tile_plan.append((name, j, note, chosen, [int(pos[arg[f, q]]) for f in chosen], [float(a[f]) for f in chosen]))
        assert all(st[f] <= arg[f, J.index(j)] < st[f + 1] for f in chosen)
    need = {}
    for _, _, _, ch, _, _ in tile_plan:
        for f in ch:
            k = int(F.k.iat[f])
            need.setdefault(F.video.iat[f], set()).update([max(k - 4, 0), k, k + 4])
    frames = {}

    def dec(v):
        fr, n = decode_video(v, need[v])
        if n is not None:
            last = max(fr)
            for x in need[v]:
                if x not in fr:
                    fr[x] = fr[last]
        return v, fr
    with ThreadPoolExecutor(12) as ex:
        for v, fr in ex.map(dec, sorted(need)):
            frames[v] = fr
    sd = OUT / 'sheets'
    sd.mkdir(parents=True, exist_ok=True)
    th = 192
    index = []
    for name, j, note, ch, pp, av in tile_plan:
        nc = 4
        nr = (len(ch) + nc - 1) // nc
        sheet = Image.new('RGB', (nc * (2 * th + 6), nr * (2 * th + 20) + 20), 'white')
        dr = ImageDraw.Draw(sheet)
        npos = {b: 0 for b in LABELS}
        dr.text((4, 4), f'{key} neuron {j}: {note}. Tile: full frame t (box = argmax patch) | zoom t-4 / t / t+4 '
                        f'(0.8 s steps). Border: green nose_nose, blue nose_tail, grey neither', fill='black')
        for i, (f, p, a) in enumerate(zip(ch, pp, av)):
            v, k = F.video.iat[f], int(F.k.iat[f])
            r_, c_ = divmod(p, G)
            cy, cx = (r_ + 0.5) * 512 / G, (c_ + 0.5) * 512 / G
            side = 160
            x0 = int(np.clip(cx - side / 2, 0, 512 - side)); y0 = int(np.clip(cy - side / 2, 0, 512 - side))
            full = Image.fromarray(frames[v][k]).copy()
            d2 = ImageDraw.Draw(full)
            s_ = 512 / G
            d2.rectangle([c_ * s_, r_ * s_, (c_ + 1) * s_, (r_ + 1) * s_], outline=(255, 40, 40), width=3)
            d2.rectangle([x0, y0, x0 + side, y0 + side], outline=(255, 220, 0), width=2)
            ims = [full.resize((th, th))]
            for kk in (max(k - 4, 0), k, k + 4):
                ims.append(Image.fromarray(frames[v][kk]).crop((x0, y0, x0 + side, y0 + side)).resize((th, th)))
            yn, yt = bool(sub.nose_nose.iat[f]), bool(sub.nose_tail.iat[f])
            npos['nose_nose'] += yn; npos['nose_tail'] += yt
            col = (0, 200, 0) if yn else (40, 110, 255) if yt else (150, 150, 150)
            tile = Image.new('RGB', (2 * th + 6, 2 * th + 6), col)
            for q_, im in enumerate(ims):
                tile.paste(im, (3 + (q_ % 2) * th, 3 + (q_ // 2) * th))
            rr, cc = divmod(i, nc)
            X0, Y0 = cc * (2 * th + 6), 20 + rr * (2 * th + 20)
            sheet.paste(tile, (X0, Y0))
            dr.text((X0 + 2, Y0 + 2 * th + 6), f'{F.obs.iat[f][:20]} k{k} a={a:.2f}', fill='black')
        fn = f'{key}_{name}.jpg'
        sheet.save(sd / fn, quality=85)
        index.append({'sheet': fn, 'neuron': j, 'note': note, 'n_tiles': len(ch), **{f'n_pos_{b}': npos[b] for b in LABELS},
                      'obs': [F.obs.iat[f] for f in ch], 'k': [int(F.k.iat[f]) for f in ch]})
        log(f'sheet {fn}: {len(ch)} tiles, positives {npos}')
    (sd / f'index_{key}.json').write_text(json.dumps(index, indent=1))


# ---------------------------------------------------------------------------------------------- summary
CFGS = ('C1_M5', 'vjepa', 'vjepa_L12', 'vjepa_L18')


def cmd_summary(args):
    E = OUT / 'eval'
    lines = ['# V-JEPA 2 ViT-L (448, tubelet t-1..t) SAE vs C1 (DINOv2 448 + motion) on the B subset (weighted)', '']
    summ = {}
    for fs in ('usable', 'all'):
        for p in PREFIXES:
            lines += ['', f'## frames: {fs}, prefix {p}', '',
                      '| label | config | n seeds | cf AUROC | cf AP | honest top-1% | size-ctrl AUROC | AP / C1 |',
                      '|---|---|---|---|---|---|---|---|']
            for b in LABELS:
                c1ap = None
                for cfg in CFGS:
                    fsn = sorted(E.glob(f'{cfg}_s*.json'))
                    if not fsn:
                        continue
                    rr = [json.loads(f.read_text())[f'{fs}_m{p}'][b] for f in fsn]
                    v = {q: [x[q] for x in rr] for q in ('cf_auroc', 'cf_ap', 'cf_top1', 'cf_size_ctrl_auroc')}
                    if cfg == 'C1_M5':
                        c1ap = np.mean(v['cf_ap'])
                    ratio = np.mean(v['cf_ap']) / c1ap if c1ap else float('nan')
                    ms = lambda a, n=3: f'{np.mean(a):.{n}f} +- {np.std(a):.{n}f}'  # noqa: E731
                    lines.append(f"| {b} | {cfg} | {len(rr)} | {ms(v['cf_auroc'])} | {ms(v['cf_ap'], 4)} | "
                                 f"{ms(v['cf_top1'])} | {ms(v['cf_size_ctrl_auroc'])} | {ratio:.2f} |")
                    summ.setdefault(f'{fs}_m{p}', {}).setdefault(b, {})[cfg] = {**v, 'ap_over_c1': ratio}
    ref = json.loads((BDIR / 'results.json').read_text())['summary']
    prs = {t: json.loads((OUT / f'probe{t}' / 'results.json').read_text())['summary']
           for t in ('', '_L12', '_L18') if (OUT / f'probe{t}' / 'results.json').exists()}
    if prs:
        lines += ['', '## supervised ceiling (pool-grouped 5-fold CV, weighted, all 20,000 B frames): AUROC / AP / size-ctrl',
                  '', '| label | probe | DINOv2 448 | ' + ' | '.join(f'V-JEPA {t or "last (24)"}' for t in prs) + ' |',
                  '|---|---|---|' + '---|' * len(prs)]
        f = lambda s: (f"{s[0]['auroc_mean']:.3f} / {s[0]['ap_mean']:.3f} / {s[0]['auroc_size_ctrl_mean']:.3f}"
                       if s else '-')  # noqa: E731
        for b in LABELS:
            for p_ in ('mean_lin', 'mil_ctx', 'n_fg_only'):
                a = [x for x in ref if x['res'] == '448_subset' and x['probe'] == p_ and x['behaviour'] == b]
                cs_ = {t: [x for x in pr if x['probe'] == p_ and x['behaviour'] == b] for t, pr in prs.items()}
                if a or any(cs_.values()):
                    lines.append(f'| {b} | {p_} | {f(a)} | ' + ' | '.join(f(c) for c in cs_.values()) + ' |')
                    summ.setdefault('ceiling', {}).setdefault(b, {})[p_] = {
                        'dinov2': a[0] if a else None, **{f'vjepa{t}': (c[0] if c else None) for t, c in cs_.items()}}
    # pre-registered reading on the usable frames (where C1's reference was reported), prefix 4096 and 512
    verdict = {}
    for key in ('usable_m4096', 'usable_m512', 'all_m4096', 'all_m512'):
        if key not in summ:
            continue
        for cfg in CFGS[1:]:
            v = {}
            for b in LABELS:
                s_ = summ[key][b].get(cfg)
                if s_ is None:
                    continue
                v[b] = {'ap_ge_1.5x_c1': bool(s_['ap_over_c1'] >= 1.5),
                        'top1_ge_0.30': bool(np.mean(s_['cf_top1']) >= 0.30),
                        'dream_top1_ge_0.40': bool(np.mean(s_['cf_top1']) >= 0.40)}
            if v:
                v['helps'] = any(x['ap_ge_1.5x_c1'] or x['top1_ge_0.30'] for x in v.values())
                verdict.setdefault(key, {})[cfg] = v
    summ['verdict'] = verdict
    lines += ['', 'Pre-registered: "V-JEPA helps" iff best-neuron cf AP >= 1.5x C1 on nose_nose or nose_tail, OR honest '
              'top-1% >= 0.30 (same frames); dream: top-1% >= 0.40', '', '```', json.dumps(verdict, indent=1), '```']
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['plan', 'smoke', 'encode', 'probe', 'train', 'c1', 'eval', 'sheets', 'summary'])
    ap.add_argument('--split', default='eval')
    ap.add_argument('--layers', nargs='*', type=int, default=[], help='encode: save these block outputs instead of last')
    ap.add_argument('--layer', type=int, default=24, help='probe / train / eval / sheets: which token set (24 = last)')
    ap.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    ap.add_argument('--shards-train', type=int, default=4)
    ap.add_argument('--shards-eval', type=int, default=8)
    ap.add_argument('--max-videos', type=int, default=0)
    ap.add_argument('--decoders', type=int, default=4)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--probes', nargs='+', default=['mean_lin', 'mil_ctx'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--c1', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--sheet-seed', type=int, default=0)
    ap.add_argument('--n-random', type=int, default=30)
    ap.add_argument('--n-tiles', type=int, default=12)
    ap.add_argument('--extra-neurons', nargs='*', type=int, default=[])
    args = ap.parse_args()
    if args.steps is None:
        args.steps = 3000 if args.step == 'probe' else 6000
    {'plan': cmd_plan, 'smoke': cmd_smoke, 'encode': cmd_encode, 'probe': cmd_probe, 'train': cmd_train,
     'c1': cmd_c1, 'eval': cmd_eval, 'sheets': cmd_sheets, 'summary': cmd_summary}[args.step](args)


if __name__ == '__main__':
    main()
