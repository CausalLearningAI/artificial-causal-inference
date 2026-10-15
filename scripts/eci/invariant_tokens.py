"""
INVARIANT_TOKENS (idea A) - orbit-averaged, rotation-invariant DINOv2 tokens for the mice ECI SAE; also the shared
encode / train / eval harness of BIG_ENCODER (idea B, scripts/eci/big_encoder.py). Question: can an UNSUPERVISED SAE
neuron capture nose-nose (nn = Y_nn OR Y_np) / nose-tail (nt = Y_nt) sniffing? No labels enter the representation.

Idea A: "a behaviour is the same in any orientation". DINOv2 tokens change a lot under rotation / flip (cos ~0.78
between a patch and its rotated counterpart, commit 44e9cd5), so one contact appears as several orientation variants.
    For every frame, the 448 px encoder input (HF dinov2-base processor of the 512 px frame, as quick_repr) is
    transformed by all 8 elements g of the dihedral group D4 (k quarter-turns counter-clockwise, then a left-right flip
    for g >= 4; src/eci/priors.d4_apply convention), each encoded by DINOv2-base, and the token of original patch p is
    read at d4_pos_map(g)[p] in transformed frame g (the 32 x 32 grid of 14 px patches maps exactly). The 8 tokens of a
    patch are AVERAGED -> an orientation-invariant token (no loss term). The same for frame t-5 (1 s);
    motion = orb(t) - orb(t-5). SAE input x = [orb_t, orb_t - orb_{t-5}].
    (The transform acts on the 448 tensor rather than the 512 JPEG: the bicubic resize commutes with D4 up to
    rounding; measured in the encode log, 'jpg512_vs_tensor448'.)
Idea B: DINOv2-large (ViT-L/14, d 1024) and DINOv2-giant (ViT-g/14, d 1536), same 448 input / 32 x 32 grid / mask /
    motion / SAE recipe, identity transform only (big_encoder.py).

Everything else = the quick_repr / sparsity_test C1 harness (reused by import):
    mask     the fg448 store positions (32 x 32 grid); the t-5 token is read at frame t's positions
    SAE      Matryoshka BatchTopK, 4096 latents, k 16, prefixes 512 / 1024 / 2048 / 4096, TokenNorm.fit_blocks([d, d]),
             AuxK, b_dec = geometric median, Adam 5e-4, 500 warm-up, cosine, 6000 steps x 4096 (motion_t3.train_one,
             with the block size d as a parameter)
    read-out frame value = MAX over kept tokens; sparsity_test.eval_arm (cross-fitted best neuron, vjepa.score, weighted
             B-subset labels, usable 16,775 frames, prefixes 512 / 4096) and sparsity_test.sheets
    ceiling  diag_mice_patches mean_lin + mil_ctx on the eval t tokens (pool-grouped 5-fold CV, 3 seeds, 3000 steps),
             as quick_repr.probe
Frames
    pool0    the 10,000 unlabelled res_pool train frames (100 evenly spaced store frames per unannotated train video)
    pool1    a second training pool: 100 store frames per SAME train video drawn uniformly at random (rng 1) among that
             video's store frames with >= 1 kept token, excluding pool0 frames (plan step)
    eval     the 20,000-frame annotated B subset (weights w), t-5 = row - 5 clipped to the video's first row
    All frames are DECODED LOCALLY from the standardised mp4s (ffmpeg 6.1.1 of the ffmpeg611 env, the get_frames.py
    chain fps=5 / rgb24 / JPEG q 2 = scripts/eci/priors.py; one frame per video byte-checked against the dataset JPG).
Arms (SAE seeds 0 1 2 x pools 0 1 = 6 runs per arm)
    id       DINOv2-base identity tokens = the c448 / btk16 recipe retrained on the same pools (the matched control)
    orb      idea A
    vitl / vitg   idea B (big_encoder.py)

PRE-REGISTERED RULE (fixed before any result; usable frames, prefix 4096, mean over the 6 runs)
    an arm 'helps' iff on nn or nt: cf AP >= 1.5 x ref OR honest top-1% >= 0.25, AND the OTHER label's cf AUROC is not
    more than 0.02 below ref's. Primary ref = the matched control 'id' (same 2 pools x 3 seeds); also reported vs the
    task-quoted c448 / btk16 (nn 0.690 / 0.0599 / 0.141, nt 0.719 / 0.0398 / 0.089). Dream target: top-1% >= 0.40.

Steps
    plan     CPU  pool1 frame table + positions -> results/vision/eci_invariant/plan/
    encode   GPU  --model base|large|giant [--orbit] --set pool0|pool1|eval --shard i --nshard n: local decode + encode
                  -> OUT/tok/<model>[_orb]/<set>_<i>of<n>_<kind>_<t|p>.npy (fp16 (N, d)), kind = id | orb
    train    GPU  --model ... --kinds id orb: SAEs (pools x seeds) -> eval JSONs -> sheets (seed 0, both pools) -> probe
    summary  any  OUT/table.md, OUT/summary.json (reads JSONs only)
    clean    any  deletes OUT/tok (the large token files)
Outputs: results/vision/eci_invariant/ (idea A), results/vision/eci_big_encoder/ (idea B); gitignored.
"""
import argparse
import json
import os
import re
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
OUT_INV = Path(os.environ.get('INV_OUT') or REPO / 'results/vision/eci_invariant')
OUT_BIG = Path(os.environ.get('BIG_OUT') or REPO / 'results/vision/eci_big_encoder')
PLAN = REPO / 'results/vision/eci_invariant/plan'  # the pool1 plan (shared by both ideas, also in smoke runs)
BDIR = REPO / 'results/vision/eci_repr_diag/mice/b'
G = 32
LABELS = ['nose_nose', 'nose_tail']
MODELS = {'base': 'facebook/dinov2-base', 'large': 'facebook/dinov2-large', 'giant': 'facebook/dinov2-giant'}
DIMS = {'base': 768, 'large': 1024, 'giant': 1536}
ARMNAME = {('base', 'id'): 'id', ('base', 'orb'): 'orb', ('large', 'id'): 'vitl', ('giant', 'id'): 'vitg'}
REF = {'nose_nose': {'auroc': 0.690, 'ap': 0.0599, 'top1': 0.141},
       'nose_tail': {'auroc': 0.719, 'ap': 0.0398, 'top1': 0.089}}  # task-quoted c448 / btk16 (usable, m4096)
SETS = ('pool0', 'pool1', 'eval')
FFMPEG = os.environ.get('PR_FFMPEG') or str(Path.home() / '.conda/envs/ffmpeg611/bin/ffmpeg')
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def stage_dir():
    d = Path(os.environ.get('LOCAL_DIR') or '')
    tmp = os.environ.get('TMPDIR') or '/nonexistent'
    if not (str(d).startswith('/localhome') or str(d).startswith(tmp)):
        raise RuntimeError(f'LOCAL_DIR must be under /localhome or $TMPDIR, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir(model):
    return OUT_INV if model == 'base' else OUT_BIG


# ---------------------------------------------------------------------------------------------- frames
def cmd_plan(args):
    """pool1: 100 random store frames per res_pool train video (rng 1), not in pool0, >= 1 kept token."""
    import motion_t3 as mt
    import multiscale_sae as ms
    import spatial_sae_pilot as ssp
    from src.eci.foreground import obs_rows
    t0 = time.time()
    PLAN.mkdir(parents=True, exist_ok=True)
    ann_p = REPO / 'dataset/mice/v1/annotations.csv'
    T = pd.read_parquet(REPO / 'results/vision/eci_res_pool/mice/train/frames.parquet')
    idx = ssp.StoreIndex('mice')
    tr, _, _, _, train_v, _ = idx.split('mice', ms.N_TRAIN['mice'], 0)
    assert np.isin(T.store_frame.values, tr).all() and sorted(set(T.obs)) == sorted(train_v)
    g = np.random.default_rng(1)
    sel = []
    for v in sorted(train_v):
        f = tr[idx.obs[tr] == v]
        f = f[~np.isin(f, T.store_frame.values)]
        sel.append(np.sort(g.choice(f, 100, replace=False)))
    sel = np.sort(np.concatenate(sel))
    assert not np.isin(sel, T.store_frame.values).any() and (idx.nfg[sel] > 0).all()
    ann = pd.read_csv(ann_p, usecols=['observation_id', 'frame_idx', 'frame_path'])
    R = obs_rows(ann_p)
    rows, obs = idx.rows[sel], idx.obs[sel]
    lo = np.array([R[o][0] for o in obs])
    prow = np.maximum(rows - 5, lo)
    F = pd.DataFrame({'store_frame': sel, 'row': rows, 'obs': obs, 'frame_idx': ann.frame_idx.values[rows],
                      'frame_path': ann.frame_path.values[rows], 'n_fg': idx.nfg[sel].astype(np.int64),
                      'prow': prow, 'pfi': ann.frame_idx.values[prow], 'ppath': ann.frame_path.values[prow]})
    assert (ann.observation_id.values[rows] == obs).all() and (ann.observation_id.values[prow] == obs).all()
    assert ((F.frame_idx - F.pfi).values == (F.row - F.prow).values).all()
    pa, gs = mt.all_pos(idx)
    pos = np.concatenate([pa[gs[f]:gs[f] + idx.nfg[f]] for f in sel]).astype(np.int16)
    pos.tofile(PLAN / 'pool1_pos448.i16')
    F.to_parquet(PLAN / 'pool1_frames.parquet')
    info = {'n_frames': len(F), 'n_videos': int(F.obs.nunique()), 'n_tokens': int(F.n_fg.sum()),
            'tokens_per_frame': float(F.n_fg.mean()), 'pool0_tokens_per_frame': float(T.n_fg.mean()),
            't5_is_self': int((F.prow == F.row).sum()), 'elapsed_s': round(time.time() - t0, 1)}
    (PLAN / 'pool1.json').write_text(json.dumps(info, indent=1))
    log(f'pool1 plan: {json.dumps(info)}')


def tables():
    """-> {set: dict(F, pos, lens, path_t, path_p[, sub])}; pool0 / eval = quick_repr.tables()."""
    import quick_repr as qr
    tb = qr.tables()
    P = pd.read_parquet(PLAN / 'pool1_frames.parquet')
    ppos = np.fromfile(PLAN / 'pool1_pos448.i16', np.int16)
    assert len(ppos) == P.n_fg.sum()
    return {'pool0': tb['train'], 'eval': tb['eval'],
            'pool1': dict(F=P, pos=ppos, lens=P.n_fg.values.astype(np.int64), path_t=P.frame_path.values,
                          path_p=P.ppath.values)}


def shard(t, i, n):
    import quick_repr as qr
    if n == 1:
        return t, np.arange(len(t['F']))
    ix = np.array_split(np.arange(len(t['F'])), n)[i]
    return qr.take(t, ix), ix


# ---------------------------------------------------------------------------------------------- decode
def video_file(frame_path):
    p = Path(frame_path)
    for ext in ('.mp4', '.mkv'):
        f = REPO / 'data' / p.parts[0] / p.parts[1] / 'observations' / 'full' / f'{p.parent.name}{ext}'
        if f.exists():
            return f
    raise FileNotFoundError(frame_path)


def _decode(job):
    """one standardised video -> the needed 5 fps frames as <jpg>/<frame_dir>/frame_<k>.jpg (= the dataset layout)."""
    fdir, ks, jpg = job
    dst = Path(jpg) / fdir
    tmp = dst / '_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src = video_file(f'{fdir}/frame_{ks[0]:06d}.jpg')
    loc = Path(jpg) / f'_{Path(fdir).name}{src.suffix}'
    shutil.copyfile(src, loc)
    sel = '+'.join(f'eq(n\\,{k})' for k in ks)
    subprocess.run([FFMPEG, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(loc), '-vf',
                    f'fps=5,select={sel},format=rgb24', '-vsync', '0', '-q:v', '2', '-start_number', '0',
                    str(tmp / '%06d.jpg')], check=True)
    loc.unlink()
    files = sorted(tmp.glob('*.jpg'))
    if len(files) != len(ks):
        raise RuntimeError(f'{fdir}: decoded {len(files)} of {len(ks)} frames')
    for f, k in zip(files, ks):
        f.rename(dst / f'frame_{k:06d}.jpg')
    tmp.rmdir()
    k0 = ks[len(ks) // 2]
    same = (REPO / 'dataset' / fdir / f'frame_{k0:06d}.jpg').read_bytes() == (dst / f'frame_{k0:06d}.jpg').read_bytes()
    return fdir, {'n': len(ks), 's': round(time.time() - t0, 1), 'bytes_equal': bool(same)}


def decode_paths(paths, stage, workers):
    """decode every frame in `paths` (dataset-relative JPG paths) locally -> stage/jpg (dataset layout)."""
    from multiprocessing import Pool
    jpg = stage / 'jpg'
    jpg.mkdir(parents=True, exist_ok=True)
    by = {}
    for p in sorted(set(paths)):
        m = re.fullmatch(r'frame_(\d+)\.jpg', Path(p).name)
        by.setdefault(str(Path(p).parent), []).append(int(m.group(1)))
    jobs = [(d, sorted(ks), str(jpg)) for d, ks in by.items() if not all((jpg / d / f'frame_{k:06d}.jpg').exists()
                                                                          for k in ks)]
    t0 = time.time()
    with Pool(workers) as pool:
        res = dict(pool.imap_unordered(_decode, jobs))
    nb = sum(v['bytes_equal'] for v in res.values())
    info = {'videos': len(res), 'frames': int(sum(v['n'] for v in res.values())), 'bytes_equal_videos': int(nb),
            'wall_s': round(time.time() - t0, 1)}
    log(f'decoded {json.dumps(info)}')
    if nb != len(res):
        raise RuntimeError(f'decoded JPG differs from the dataset JPG in {len(res) - nb} videos')
    return jpg, info


# ---------------------------------------------------------------------------------------------- D4
def d4_torch(x, g):
    """(B, C, H, W) -> the D4 element g of src/eci/priors.d4_apply on the (H, W) axes."""
    import torch
    y = torch.rot90(x, int(g) % 4, dims=(2, 3))
    return torch.flip(y, dims=(3,)) if g >= 4 else y


def check_d4_torch():
    """index image with one value per 14 px patch: transformed patch at d4_pos_map(g)[p] holds p, and the pixels equal
    priors.d4_apply of the numpy image (exact)."""
    import torch
    from src.eci.priors import d4_apply, d4_pos_map
    a = np.repeat(np.repeat(np.arange(G * G).reshape(G, G), 14, 0), 14, 1).astype(np.float32)
    x = torch.from_numpy(a)[None, None]
    for g in range(8):
        y = d4_torch(x, g)[0, 0].numpy()
        assert np.array_equal(y, d4_apply(a, g)), g
        J = y[7::14, 7::14].ravel().astype(np.int64)  # patch q of transformed frame holds original patch J[q]
        m = d4_pos_map(g)
        assert np.array_equal(J[m], np.arange(G * G)), g
    return True


# ---------------------------------------------------------------------------------------------- encode
def cmd_encode(args):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from src.eci.priors import d4_apply, d4_pos_map
    check_d4_torch()
    log('D4 torch self-test passed (8 elements, exact pixels and patch map)')
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    dev = torch.device('cuda')
    stage = stage_dir()
    OUT = out_dir(args.model)
    D = DIMS[args.model]
    tag = args.model + ('_orb' if args.orbit else '')
    od = OUT / 'tok' / tag
    od.mkdir(parents=True, exist_ok=True)
    key = f'{args.set}_{args.shard}of{args.nshard}'
    if (od / f'{key}.json').exists():
        log(f'[SKIP] {od / key}')
        return
    t00 = time.time()
    t, ix = shard(tables()[args.set], args.shard, args.nshard)
    if args.smoke:
        import quick_repr as qr
        step = max(1, len(t['F']) // 64)
        t, ix = qr.take(t, np.arange(0, len(t['F']), step)), ix[::step]
    st = np.r_[0, np.cumsum(t['lens'])]
    N = int(st[-1])
    jpg, dinfo = decode_paths(np.concatenate([t['path_t'], t['path_p']]), stage, args.workers)
    items = sorted(set(t['path_t']) | set(t['path_p']))
    iid = {p: j for j, p in enumerate(items)}
    cons = [[] for _ in items]
    for i in range(len(t['F'])):
        cons[iid[t['path_t'][i]]].append(('t', i))
        cons[iid[t['path_p'][i]]].append(('p', i))
    kinds = ['id', 'orb'] if args.orbit else ['id']
    arr = {(k, r): np.lib.format.open_memmap(stage / f'{key}_{k}_{r}.npy', 'w+', np.float16, (N, D))
           for k in kinds for r in ('t', 'p')}
    proc = AutoImageProcessor.from_pretrained(MODELS['base'], use_fast=True, size={'height': 448, 'width': 448},
                                              do_center_crop=False)  # the same pixels for every model
    model = AutoModel.from_pretrained(MODELS[args.model]).to(dev).eval()
    model.requires_grad_(False)
    npre = 1 + (getattr(model.config, 'num_register_tokens', 0) or 0)
    assert model.config.hidden_size == D and model.config.patch_size == 14
    maps = torch.stack([torch.from_numpy(d4_pos_map(g)) for g in range(8)]).to(dev)
    gs = range(8) if args.orbit else [0]
    log(f'{MODELS[args.model]} (d {D}, prefix {npre}, {model.config.num_hidden_layers} layers): {args.set} shard '
        f'{args.shard}/{args.nshard}: {len(t["F"]):,} frames, {len(items):,} images x {len(gs)} transforms, {N:,} tokens')

    class Imgs(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, j):
            with Image.open(jpg / items[j]) as im:
                im = im.convert('RGB')
            return j, proc(images=im, return_tensors='pt')['pixel_values'][0]

    dl = torch.utils.data.DataLoader(Imgs(), batch_size=args.bs, num_workers=args.workers, shuffle=False,
                                     pin_memory=True, prefetch_factor=4)
    diag = {'cos_g_vs_id': np.zeros(8), 'cos_orb_vs_id': 0.0, 'n': 0}
    t0, gpu, n = time.time(), 0.0, 0
    for b, (jj, pix) in enumerate(dl):
        torch.cuda.synchronize(); g0 = time.time()
        with torch.inference_mode():
            pix = pix.to(dev, non_blocking=True)
            x = torch.cat([d4_torch(pix, g) for g in gs]) if args.orbit else pix
            h = torch.cat([model(pixel_values=x[a:a + args.fwd]).last_hidden_state for a in range(0, len(x), args.fwd)])
            assert h.shape[1] == npre + G * G and torch.isfinite(h).all()
            h = h[:, npre:].float().reshape(len(gs), len(pix), G * G, D)
            H = {'id': h[0]}
            if args.orbit:
                al = torch.stack([h[g][:, maps[g]] for g in gs])  # (8, B, 1024, D): token of ORIGINAL patch p
                H['orb'] = al.mean(0)
                if b < 40:  # orbit spread on all patches of the first 40 batches
                    cs = torch.nn.functional.cosine_similarity(al, al[0:1], dim=-1).mean((1, 2))
                    diag['cos_g_vs_id'] += cs.cpu().numpy()
                    diag['cos_orb_vs_id'] += float(torch.nn.functional.cosine_similarity(H['orb'], al[0], dim=-1).mean())
                    diag['n'] += 1
            H = {k: v.half().cpu().numpy() for k, v in H.items()}
        torch.cuda.synchronize(); gpu += time.time() - g0
        for q, j in enumerate(jj.tolist()):
            for r, i in cons[j]:
                p = t['pos'][st[i]:st[i + 1]].astype(np.int64)
                for k in kinds:
                    arr[(k, r)][st[i]:st[i + 1]] = H[k][q, p]
        n += len(jj)
        if b % 100 == 0:
            log(f'  batch {b}: {n:,}/{len(items):,} images, {n / (time.time() - t0):.1f} img/s (gpu {gpu:.0f}s)')
    for a in arr.values():
        a.flush()
    enc = {'images': len(items), 'transforms': len(gs), 'wall_s': round(time.time() - t0, 1), 'gpu_s': round(gpu, 1),
           'img_per_s': round(len(items) / (time.time() - t0), 2), 'tf32': bool(args.tf32)}
    if args.orbit:
        enc['orbit_cos_mean_patch_g_vs_identity'] = (diag['cos_g_vs_id'] / max(diag['n'], 1)).round(4).tolist()
        enc['orbit_cos_mean_avg_vs_identity'] = round(diag['cos_orb_vs_id'] / max(diag['n'], 1), 4)
        enc['jpg512_vs_tensor448'] = jpg_vs_tensor(model, proc, jpg, items[:8], dev, npre, D, maps, d4_apply)
    log(f'encode: {json.dumps(enc)}')
    checks = store_check(args, t, ix, arr, D)
    del model
    torch.cuda.empty_cache()
    for (k, r), a in arr.items():
        del a
        shutil.copyfile(stage / f'{key}_{k}_{r}.npy', od / f'{key}_{k}_{r}.npy')
        (stage / f'{key}_{k}_{r}.npy').unlink()
    info = {'model': MODELS[args.model], 'set': args.set, 'shard': args.shard, 'nshard': args.nshard,
            'frames': int(len(t['F'])), 'frame_ix0': int(ix[0]), 'frame_ix1': int(ix[-1]) + 1, 'n_tokens': N, 'd': D,
            'kinds': kinds, 'gpu': torch.cuda.get_device_name(0), 'host': os.uname().nodename, 'decode': dinfo,
            'encode': enc, 'store_check': checks, 'wall_s': round(time.time() - t00, 1), 'smoke': bool(args.smoke)}
    (od / f'{key}.json').write_text(json.dumps(info, indent=1, default=float))
    log(f'encode done: {json.dumps({k: v for k, v in info.items() if k not in ("decode",)}, default=float)}')


def jpg_vs_tensor(model, proc, jpg, paths, dev, npre, D, maps, d4_apply):
    """the transform on the 448 tensor vs on the 512 JPEG array (then resized): per-g token cosine (a few frames)."""
    import torch
    from PIL import Image
    out = []
    with torch.inference_mode():
        for p in paths:
            a = np.asarray(Image.open(jpg / p).convert('RGB'))
            x0 = proc(images=Image.fromarray(a), return_tensors='pt')['pixel_values'].to(dev)
            xa = torch.cat([d4_torch(x0, g) for g in range(8)])
            xb = torch.cat([proc(images=Image.fromarray(d4_apply(a, g)), return_tensors='pt')['pixel_values']
                            for g in range(8)]).to(dev)
            ha = model(pixel_values=xa).last_hidden_state[:, npre:].float()
            hb = model(pixel_values=xb).last_hidden_state[:, npre:].float()
            out.append(torch.nn.functional.cosine_similarity(ha, hb, dim=-1).mean(1).cpu().numpy())
    c = np.stack(out)
    return {'frames': len(paths), 'cos_mean_per_g': c.mean(0).round(6).tolist(), 'cos_min_frame_mean': float(c.min())}


def store_check(args, t, ix, arr, D):
    """base identity tokens vs the store: eval t vs BDIR tok448; pool t vs StoreIndex (first 64 frames of the shard)."""
    if args.model != 'base':
        return None
    cos = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)  # noqa: E731
    a = arr[('id', 't')]
    if args.set == 'eval':
        ref = np.load(BDIR / 'tok448.npy', mmap_mode='r')
        st_all = np.r_[0, np.cumsum(tables()['eval']['lens'])]
        o = int(st_all[ix[0]])
        n = min(300_000, len(a))
        if args.smoke:
            rr = np.concatenate([np.arange(st_all[f], st_all[f + 1]) for f in ix])
            c = cos(np.asarray(a, np.float32), np.asarray(ref[rr], np.float32))
        else:
            c = cos(np.asarray(a[:n], np.float32), np.asarray(ref[o:o + n], np.float32))
        r = {'vs': 'BDIR tok448', 'n': int(len(c)), 'cos_min': float(c.min()), 'cos_mean': float(c.mean())}
    else:
        import spatial_sae_pilot as ssp
        idx = ssp.StoreIndex('mice')
        fr = t['F'].store_frame.values[:64]
        tok, pos, lens = idx.load(fr)
        m = int(lens.sum())
        assert (pos == t['pos'][:m]).all()
        c = cos(np.asarray(a[:m], np.float32), tok.astype(np.float32))
        r = {'vs': 'store', 'n': m, 'cos_min': float(c.min()), 'cos_mean': float(c.mean())}
    log(f'store check: {json.dumps(r)}')
    if r['cos_min'] < 0.999:
        raise RuntimeError(f'identity tokens differ from the store: {r}')
    return r


# ---------------------------------------------------------------------------------------------- train / eval
def load_X(model, kind, set_, stage, smoke=False):
    """-> X (N, 2d) fp16 = [x_t, x_t - x_{t-5}] (difference in fp32 of the fp16 tokens, quick_repr's rule)."""
    tag = model + ('_orb' if kind == 'orb' else '')
    od = out_dir(model) / 'tok' / tag
    js = sorted(od.glob(f'{set_}_*of*.json'), key=lambda f: json.loads(f.read_text())['shard'])
    if not js:
        tag = model + '_orb'  # base identity tokens live next to the orbit tokens
        od = out_dir(model) / 'tok' / tag
        js = sorted(od.glob(f'{set_}_*of*.json'), key=lambda f: json.loads(f.read_text())['shard'])
    meta = [json.loads(f.read_text()) for f in js]
    assert meta and len(meta) == meta[0]['nshard'] and all(m['smoke'] == smoke for m in meta), [str(f) for f in js]
    parts = []
    for m, f in zip(meta, js):
        k = f.stem
        loc = {}
        for r in ('t', 'p'):
            src = od / f'{k}_{kind}_{r}.npy'
            loc[r] = stage / f'{tag}_{k}_{kind}_{r}.npy'
            if not loc[r].exists():
                shutil.copyfile(src, loc[r])
        a, b = np.load(loc['t'], mmap_mode='r'), np.load(loc['p'], mmap_mode='r')
        d = a.shape[1]
        x = np.empty((len(a), 2 * d), np.float16)
        for c in range(0, len(a), 500_000):
            ta = np.asarray(a[c:c + 500_000])
            x[c:c + len(ta), :d] = ta
            x[c:c + len(ta), d:] = (ta.astype(np.float32) - np.asarray(b[c:c + 500_000], np.float32)).astype(np.float16)
        parts.append(x)
        del a, b
        for r in ('t', 'p'):
            loc[r].unlink()
    X = np.concatenate(parts) if len(parts) > 1 else parts[0]
    log(f'loaded {tag} {kind} {set_}: {X.shape} from {len(js)} shard(s)')
    return X


def train_one(name, seed, X, d, steps, out, extra, dev, bs=4096, lr=5e-4, warmup=500, width=4096, k=16):
    """motion_t3.train_one (= sparsity_test btk16) with the block size d as a parameter."""
    import torch
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    N, din = X.shape
    assert din == 2 * d
    ii = torch.from_numpy(np.sort(g.choice(N, min(500_000, N), replace=False))).to(dev)
    init = X[ii].float()
    norm = TokenNorm(din).to(dev).fit_blocks(init, [d, d], seed=seed)
    sae = MatryoshkaBatchTopKSAE(d_in=din, n_latents=width, prefixes=(width // 8, width // 4, width // 2, width), k=k,
                                 seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    perm, p = torch.randperm(N, generator=gen, device=dev), 0
    history, t0 = [], time.time()
    for step in range(steps):
        if p + bs > N:
            perm, p = torch.randperm(N, generator=gen, device=dev), 0
        b = perm[p:p + bs]
        p += bs
        logs, gn = _train_step(sae, opt, norm(X[b]), step, steps, lr, warmup, 1.0)
        if step % 1000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            history.append(logs)
            log(f"    {name} s{seed} step {step}/{steps} fve {logs['fve']:.4f} l0 {logs['l0']:.2f} dead {logs['n_dead']} "
                f"thr {logs['threshold']:.3f} {logs['elapsed_s']}s")
    sae.eval()
    dead = int((sae.tokens_since_fired >= sae.dead_tokens).sum())
    save_checkpoint(out / 'sae.pt', sae, norm, extra={**extra, 'seed': seed})
    (out / 'train.json').write_text(json.dumps({**extra, 'seed': seed, 'n_pool': N, 'steps': steps,
                                                'token_presentations': steps * bs, 'train_time_s': time.time() - t0,
                                                'dead_at_end_train': dead, 'dead_frac_train': dead / width,
                                                'threshold': float(sae.threshold),
                                                'norm_scale_blocks': [float(norm.scale[0]), float(norm.scale[d])],
                                                'gpu': torch.cuda.get_device_name(0), 'history': history}, indent=1))
    log(f'    {name} s{seed}: trained in {time.time() - t0:.0f}s, final fve {history[-1]["fve"]:.4f}, dead {dead}/{width}')


def cmd_train(args):
    import torch
    import contact_sae as cs
    import diag_mice_patches as dmp
    import sparsity_test as sp
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    OUT = out_dir(args.model)
    D = DIMS[args.model]
    # reuse sparsity_test's eval / sheets (BatchTopK path) with this harness's paths
    sp.OUT = OUT
    sp.sae_dir = lambda arm, seed: OUT / 'sae' / f'{arm}_s{seed}'
    tb = tables()
    if args.smoke:
        import quick_repr as qr
        for s in SETS:
            n = len(tb[s]['F'])
            tb[s] = qr.take(tb[s], np.arange(0, n, max(1, n // 64)))
        args.steps, args.probe_seeds = 100, [0]
    half_all = cs.video_units(tb['eval']['sub'].obs.values)
    jpg, _ = decode_paths(np.concatenate([tb['eval']['path_t'], tb['eval']['path_p']]), stage, args.workers)
    for kind in args.kinds:
        arm = ARMNAME[(args.model, kind)]
        Xev = load_X(args.model, kind, 'eval', stage, args.smoke)
        assert len(Xev) == tb['eval']['lens'].sum()
        for pool in args.pools:
            name = f'{arm}_p{pool}'
            todo = [s for s in args.seeds if not (OUT / 'sae' / f'{name}_s{s}' / 'sae.pt').exists()]
            if todo:
                Xn = load_X(args.model, kind, f'pool{pool}', stage, args.smoke)
                assert len(Xn) == tb[f'pool{pool}']['lens'].sum()
                Xtr = torch.from_numpy(Xn).to(dev)
                del Xn
                for seed in todo:
                    torch.backends.cuda.matmul.allow_tf32 = True  # as motion_t3 train
                    train_one(name, seed, Xtr, D, args.steps, OUT / 'sae' / f'{name}_s{seed}',
                              {'config': 'invariant_tokens', 'arm': arm, 'kind': kind, 'model': MODELS[args.model],
                               'pool': pool, 'd_block': D}, dev)
                    torch.cuda.empty_cache()
                del Xtr
                torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = False
            for seed in args.seeds:
                sp.eval_arm(name, seed, Xev, tb, dev, half_all)
            if not (OUT / 'sheets' / f'index_{name}_s{args.seeds[0]}.json').exists():
                sp.sheets(name, args.seeds[0], Xev, tb, jpg, dev)
        if args.probe:
            o = OUT / f'probe_{arm}.json'
            if o.exists():
                log(f'[SKIP] {o}')
            else:
                t0 = time.time()
                torch.backends.cuda.matmul.allow_tf32 = True
                sub = tb['eval']['sub']
                tok = np.ascontiguousarray(Xev[:, :D])
                bag = dmp.Bag(tok, tb['eval']['pos'], tb['eval']['lens'], G, dev)
                res = dmp.cross_validate(bag, sub, sub.w.values, dev, ['mean_lin', 'mil_ctx'], args.probe_seeds, None,
                                         arm, 50 if args.smoke else 3000)
                summ = dmp.summarise(res)
                o.write_text(json.dumps({'summary': summ, 'runs': res, 'wall_s': time.time() - t0,
                                         'gpu': torch.cuda.get_device_name(0)}, indent=1, default=float))
                for s in summ:
                    log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
                del bag, tok
                torch.cuda.empty_cache()
        del Xev
    log(f'train done in {time.time() - t00:.0f}s')


# ---------------------------------------------------------------------------------------------- summary
def collect(eval_dirs, arms):
    """-> {arm: {'runs': [(pool, seed, json)], ...}} from the eval JSONs of every dir (first hit wins)."""
    R = {}
    for a in arms:
        for d in eval_dirs:
            fs = sorted(d.glob(f'{a}_p*_s*.json'))
            if fs:
                R[a] = [(int(re.search(r'_p(\d+)_s', f.name).group(1)), int(re.search(r'_s(\d+)\.json', f.name).group(1)),
                         json.loads(f.read_text())) for f in fs]
                break
    return R


def summarise(OUT, arms, ctl_dirs, probe_dirs, title):
    R = collect([OUT / 'eval'] + ctl_dirs, arms)
    u = 'usable_m4096'
    ms = lambda a, n=3: f'{np.mean(a):.{n}f}+-{np.std(a):.{n}f}' if len(a) > 1 else f'{np.mean(a):.{n}f}'  # noqa: E731
    S = {}
    for a, rr in R.items():
        s = {'n_runs': len(rr), 'runs': [(p, sd) for p, sd, _ in rr]}
        for key in ('usable_m4096', 'usable_m512', 'all_m4096'):
            for b in LABELS:
                s[f'{key}/{b}'] = {q: [r[key][b][q] for _, _, r in rr] for q in ('cf_auroc', 'cf_ap', 'cf_top1')}
                for pool in sorted({p for p, _, _ in rr}):
                    s[f'{key}/{b}/pool{pool}'] = {q: float(np.mean([r[key][b][q] for p, _, r in rr if p == pool]))
                                                  for q in ('cf_auroc', 'cf_ap', 'cf_top1')}
        s['fve'] = [r['stats']['fve'] for _, _, r in rr]
        s['dead_frac_eval'] = [r['firing']['dead_frac_eval'] for _, _, r in rr]
        s['median_rate_eligible'] = [r['firing']['median_rate_eligible'] for _, _, r in rr]
        S[a] = s
    L = [f'# {title}', '', 'mice B subset, weighted, cross-fitted best neuron; runs = 2 training pools x 3 SAE seeds; '
         'mean+-sd over runs', '']
    for key in ('usable_m4096', 'usable_m512', 'all_m4096'):
        L += [f'## {key}', '', '| arm | runs | nn AUROC | nn AP | nn top-1% | nt AUROC | nt AP | nt top-1% | '
              'nn AP / id | nt AP / id |', '|---|---|---|---|---|---|---|---|---|---|']
        for a, s in S.items():
            nn, nt = s[f'{key}/nose_nose'], s[f'{key}/nose_tail']
            rn = np.mean(nn['cf_ap']) / np.mean(S['id'][f'{key}/nose_nose']['cf_ap']) if 'id' in S else float('nan')
            rt = np.mean(nt['cf_ap']) / np.mean(S['id'][f'{key}/nose_tail']['cf_ap']) if 'id' in S else float('nan')
            L.append(f"| {a} | {s['n_runs']} | {ms(nn['cf_auroc'])} | {ms(nn['cf_ap'], 4)} | {ms(nn['cf_top1'])} | "
                     f"{ms(nt['cf_auroc'])} | {ms(nt['cf_ap'], 4)} | {ms(nt['cf_top1'])} | {rn:.2f} | {rt:.2f} |")
        L.append('')
    L += ['## per-pool means (usable_m4096): AUROC / AP / top-1%', '', '| arm | pool | nn | nt |', '|---|---|---|---|']
    for a, s in S.items():
        for pool in sorted({p for p, _ in s['runs']}):
            f_ = lambda b: '{cf_auroc:.3f} / {cf_ap:.4f} / {cf_top1:.3f}'.format(**s[f'{u}/{b}/pool{pool}'])  # noqa: E731
            L.append(f'| {a} | {pool} | {f_("nose_nose")} | {f_("nose_tail")} |')
    L += ['', '## reconstruction / firing (eval)', '', '| arm | FVE | dead frac | median rate (eligible) |', '|---|---|---|---|']
    for a, s in S.items():
        L.append(f"| {a} | {ms(s['fve'], 4)} | {ms(s['dead_frac_eval'], 4)} | {ms(s['median_rate_eligible'])} |")
    V = {}
    refs = {'quoted_c448': REF}
    if 'id' in S:
        refs['id'] = {b: {'auroc': float(np.mean(S['id'][f'{u}/{b}']['cf_auroc'])),
                          'ap': float(np.mean(S['id'][f'{u}/{b}']['cf_ap'])),
                          'top1': float(np.mean(S['id'][f'{u}/{b}']['cf_top1']))} for b in LABELS}
    L += ['', '## pre-registered verdicts (usable, m4096, run means): helps iff on nn or nt AP >= 1.5 x ref OR top-1% >= '
          "0.25, AND the other label's AUROC drops <= 0.02; dream top-1% >= 0.40", '']
    for a, s in S.items():
        if a == 'id':
            continue
        for rn_, ref in refs.items():
            v = {}
            for b, ob in (('nose_nose', 'nose_tail'), ('nose_tail', 'nose_nose')):
                apr = np.mean(s[f'{u}/{b}']['cf_ap']) / ref[b]['ap']
                t1 = np.mean(s[f'{u}/{b}']['cf_top1'])
                drop = ref[ob]['auroc'] - np.mean(s[f'{u}/{ob}']['cf_auroc'])
                v[b] = {'ap_over_ref': float(apr), 'top1': float(t1), 'pass': bool(apr >= 1.5 or t1 >= 0.25),
                        'other_label_auroc_drop': float(drop), 'dream': bool(t1 >= 0.40)}
            v['helps'] = any(v[b]['pass'] and v[b]['other_label_auroc_drop'] <= 0.02 for b in LABELS)
            V.setdefault(a, {})[f'vs_{rn_}'] = v
            L.append(f"- {a} vs {rn_}: helps = {v['helps']}; " + '; '.join(
                f"{b}: AP x{v[b]['ap_over_ref']:.2f}, top-1% {v[b]['top1']:.3f}, other-label AUROC drop "
                f"{v[b]['other_label_auroc_drop']:+.3f}" for b in LABELS))
    # supervised ceilings
    ref = json.loads((BDIR / 'results.json').read_text())['summary']
    prs = {}
    for a in arms:
        for d in [OUT] + probe_dirs:
            if (d / f'probe_{a}.json').exists():
                prs[a] = json.loads((d / f'probe_{a}.json').read_text())['summary']
                break
    L += ['', '## supervised ceiling (eval t tokens only; pool-grouped 5-fold CV, weighted, all 20,000 B frames): '
          'AUROC / AP', '', '| label | probe | DINOv2-base (ref) | ' + ' | '.join(prs) + ' |',
          '|---|---|---|' + '---|' * len(prs)]
    f = lambda x: f"{x[0]['auroc_mean']:.3f} / {x[0]['ap_mean']:.3f}" if x else '-'  # noqa: E731
    C = {}
    for b in LABELS:
        for p_ in ('mean_lin', 'mil_ctx'):
            a0 = [x for x in ref if x['res'] == '448_subset' and x['probe'] == p_ and x['behaviour'] == b]
            cs_ = {t: [x for x in pr if x['probe'] == p_ and x['behaviour'] == b] for t, pr in prs.items()}
            L.append(f'| {b} | {p_} | {f(a0)} | ' + ' | '.join(f(c) for c in cs_.values()) + ' |')
            C.setdefault(b, {})[p_] = {'dinov2_base': a0[0] if a0 else None, **{t: (c[0] if c else None)
                                                                               for t, c in cs_.items()}}
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    (OUT / 'summary.json').write_text(json.dumps({'arms': S, 'verdict': V, 'ceiling': C, 'refs': refs}, indent=1,
                                                 default=float))
    print('\n'.join(L))


def cmd_summary(args):
    summarise(OUT_INV, ['id', 'orb'], [], [], 'invariant_tokens: orbit-averaged (D4) DINOv2-base tokens vs identity')


def cmd_clean(args):
    for OUT in (OUT_INV, OUT_BIG):
        if (OUT / 'tok').exists():
            shutil.rmtree(OUT / 'tok')
            log(f'removed {OUT / "tok"}')


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['plan', 'encode', 'train', 'summary', 'clean'])
    ap.add_argument('--model', default='base', choices=list(MODELS))
    ap.add_argument('--orbit', action='store_true')
    ap.add_argument('--set', default='pool0', choices=list(SETS))
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--kinds', nargs='+', default=['id', 'orb'])
    ap.add_argument('--pools', nargs='+', type=int, default=[0, 1])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--probe-seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--bs', type=int, default=8, help='images per loader batch (x 8 transforms with --orbit)')
    ap.add_argument('--fwd', type=int, default=64, help='images per encoder forward')
    ap.add_argument('--tf32', action='store_true')
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--smoke', action='store_true', help='~64 frames per set, 100 steps; set INV_SMOKE=1')
    return ap


def main():
    args = parser().parse_args()
    if args.smoke:
        assert os.environ.get('INV_SMOKE') == '1'
    {'plan': cmd_plan, 'encode': cmd_encode, 'train': cmd_train, 'summary': cmd_summary,
     'clean': cmd_clean}[args.step](args)


if __name__ == '__main__':
    main()
