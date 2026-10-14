"""
TEMPORAL_CTX: pre-registered test of the TEMPORAL CONTEXT of the ECI input. Question: is the 1 s look-back of C1 really
better than 0.2 s for SHORT behaviours (mouse sniffing lasts a fraction of a second), and does a frame AFTER t help?

Everything except the input blocks is the C1 harness (scripts/eci/quick_repr.py / sparsity_test.py, reused by import):
    tokens   DINOv2-base 448 px last-layer patch tokens (HF processor, whole frame bicubic -> 448, fp32, TF32 off, fp16
             rounding = the store rule) at the patch positions of frame t's mask (mice fg448, ants antsfg), for the frames
             t-1, t, t+1 (0.2 s) and t-5, t+5 (1 s) at 5 fps. Neighbours are CLIPPED to the video's rows (a clipped
             neighbour that falls on t itself gives zero change; t-5 = quick_repr / motion_t3.nbr_rows exactly).
             Frames are DECODED in the job from the standardised videos (data/<d>/<v>/observations/full) with the
             ffmpeg611 env and the exact src/dataset/get_frames.py command (fps=5, format=rgb24, -q:v 2); a sample of
             decoded frames is compared byte- and pixel-wise with the dataset JPGs, and the tokens with the store.
             Each unique JPG is encoded ONCE per job; every arm trains on the same tokens.
    norm     TokenNorm.fit_blocks with one 768-d block per input block (each block average norm sqrt(768))
    SAE      Matryoshka BatchTopK, 4096 latents, k = 16, prefixes (512, 1024, 2048, 4096), AuxK, b_dec = geometric
             median, Adam 5e-4, 500 warm-up, cosine to 10%, 6000 steps x 4096 tokens (= motion_t3.train_one)
    read-out frame value = MAX over kept tokens; cross-fitted best neuron
    mice     train = the 10,000 unlabelled res_pool train frames; eval = the 20,000-frame annotated B subset (weights w,
             nn = Y_nn OR Y_np, nt = Y_nt), scored on the 16,775 usable frames and all 20,000 (vjepa.score), prefixes 512
             and 4096
    ants     train = the 20,480 res_pool train frames; eval = the 76,800 res_pool eval frames (all 1 fps store frames of
             the 128 eval videos), labels groom_any / groom_yellow (Y2F) / groom_blue (B2F) / onlid_yellow,
             sae_levers.score_model (unweighted, res_pool's valid masks and halves), prefixes 512 and 4096

Arms (3 seeds each)
    M5   [x_t, x_t - x_{t-5}]                          the current C1 input, RETRAINED here on the same tokens
    S1   [x_t, x_t - x_{t-1}, x_{t+1} - x_t]           symmetric 0.2 s (back and forward)
    S5   [x_t, x_t - x_{t-5}, x_{t+5} - x_t]           symmetric 1 s
    ML   [x_t, x_t - x_{t-1}, x_t - x_{t-5}]           multi-lag back (0.2 s and 1 s)
    refs c448 (mice: results/vision/eci_sparsity/eval/c448_s*.json, the task-quoted numbers, same B frames; ants:
         results/vision/eci_res_pool/ants/eval/c448_max_s*.json, same frames) are reported alongside, not trained here.

PRE-REGISTERED (fixed before any result; 3-seed means; mice = usable frames, prefix 4096; ants = all eval frames with
the res_pool valid masks, prefix 4096; reference = M5 retrained in THIS job on the same tokens)
    beats M5   an arm beats M5 iff no label's cf AUROC drops > 0.02 below M5 (mice nn, nt and, when ants is run, all
               4 ants labels), AND at least one mice label gains: cf AP >= 1.3 x M5 OR honest top-1% >= M5 + 0.05.
    S1 on nn   reported separately: S1 vs M5 on nose_nose (AUROC delta, AP ratio, top-1% delta).
    best arm   (sheets) = the non-M5 arm with the highest mean over nn / nt of AP / M5 AP; sheets of its seed-0 best nn
               and nt neurons (AUROC- and top-1%-selected on half 0, shown on half 1), crops at t-1, t, t+1. S1 sheets
               are drawn too.

Steps
    run      GPU  decode the needed frames from the videos -> encode once -> train all arms x seeds -> eval -> sheets
    summary  any  OUT/table.md, OUT/summary.json (reads JSONs only)
Outputs: results/vision/eci_temporal_ctx/<domain>/ (gitignored).
"""
import argparse
import hashlib
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
OUT = Path(os.environ.get('TC_OUT') or REPO / 'results/vision/eci_temporal_ctx')
RP = REPO / 'results/vision/eci_res_pool'
SPE = REPO / 'results/vision/eci_sparsity/eval'
FFMPEG = Path.home() / '.conda/envs/ffmpeg611/bin/ffmpeg'
D, G, WIDTH, K = 768, 32, 4096, 16
LABELS = {'mice': ['nose_nose', 'nose_tail'], 'ants': ['groom_any', 'groom_yellow', 'groom_blue', 'onlid_yellow']}
OFF = {'t': 0, 'm1': -1, 'm5': -5, 'p1': 1, 'p5': 5}
ROLES = tuple(OFF)
# block (a, b) = x_a - x_b; (a, None) = x_a
ARMS = {'M5': [('t', None), ('t', 'm5')],
        'S1': [('t', None), ('t', 'm1'), ('p1', 't')],
        'S5': [('t', None), ('t', 'm5'), ('p5', 't')],
        'ML': [('t', None), ('t', 'm1'), ('t', 'm5')]}
MIN_RATE = 0.01
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def stage_dir():
    """LOCAL_DIR (must be under /localhome, or under $TMPDIR when the node has no /localhome)."""
    d = Path(os.environ.get('LOCAL_DIR') or '')
    tmp = os.environ.get('TMPDIR') or '/nonexistent'
    if not (str(d).startswith('/localhome') or str(d).startswith(tmp)):
        raise RuntimeError(f'LOCAL_DIR must be under /localhome or $TMPDIR, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


def fidx(p):
    return int(Path(p).stem.split('_')[-1])


# ---------------------------------------------------------------------------------------------- frames
def neighbours(F, rows, obs, R, fp_all):
    """-> {role: rows}, {role: paths}: row + offset clipped to the video's rows [lo, hi)."""
    lo = np.array([R[o][0] for o in obs])
    hi = np.array([R[o][1] for o in obs])
    assert ((rows >= lo) & (rows < hi)).all()
    out_r, out_p = {}, {}
    for r, o in OFF.items():
        rr = np.clip(rows + o, lo, hi - 1)
        pp = fp_all[rr]
        assert all(Path(a).parent == Path(b).parent for a, b in zip(fp_all[rows], pp)), r
        assert (np.array([fidx(a) - fidx(b) for a, b in zip(pp, fp_all[rows])]) == rr - rows).all(), r
        out_r[r], out_p[r] = rr, pp
    return out_r, out_p


def tables(d, smoke):
    """-> {split: dict(F, pos, lens, rows{role}, paths{role}[, sub / lab])} in the harness frame order."""
    from src.eci.foreground import obs_rows
    if d == 'mice':
        import quick_repr as qr
        tb = qr.tables()
        ann_p = qr.ANN
        if smoke:
            B = tb['eval']['F']
            vids = sorted(B.video.unique())[::12]
            tb = {'train': qr.take(tb['train'], np.arange(300)),
                  'eval': qr.take(tb['eval'], np.flatnonzero(B.video.isin(vids).values))}
        fp_all = pd.read_csv(ann_p, usecols=['frame_path'])['frame_path'].values
        R = obs_rows(ann_p)
        for s in ('train', 'eval'):
            t = tb[s]
            rows = t['F'].row.values.astype(np.int64)
            t['rows'], t['paths'] = neighbours(t['F'], rows, t['F'].obs.values, R, fp_all)
            assert (t['paths']['t'] == t['path_t']).all() and (t['paths']['m5'] == t['path_p']).all()
            if s == 'train':
                assert (t['rows']['m5'] == t['F'].prow.values).all()
        return tb
    from src.eci.domain import get_domain
    ann_p = get_domain('ants').ann_path
    ann = pd.read_csv(ann_p, usecols=['observation_id', 'frame_path'])
    fp_all = ann.frame_path.values
    R = obs_rows(ann_p)
    tb = {}
    for s in ('train', 'eval'):
        F = pd.read_parquet(RP / 'ants' / s / 'frames.parquet')
        pos = np.fromfile(RP / 'ants' / s / 'pos448.i16', np.int16)
        lens = F.n_fg.values.astype(np.int64)
        assert len(pos) == lens.sum()
        t = dict(F=F, pos=pos, lens=lens)
        if s == 'eval':
            lab = pd.read_parquet(RP / 'ants' / 'labels.parquet')
            assert (lab.row.values == F.row.values).all()
            V = np.load(RP / 'ants' / 'valid.npz')
            t.update(lab=lab, half=np.load(RP / 'ants' / 'half.npy'), valid={b: V[b] for b in LABELS['ants']})
        tb[s] = t
    if smoke:
        st = {s: np.r_[0, np.cumsum(tb[s]['lens'])] for s in tb}
        ev = tb['eval']['F']
        vids = list(dict.fromkeys(ev.obs))
        pick = vids[::16]
        sel = {'train': np.arange(300), 'eval': np.flatnonzero(ev.obs.isin(pick).values)}
        for s, ix in sel.items():
            t = tb[s]
            t2 = {'F': t['F'].iloc[ix].reset_index(drop=True), 'lens': t['lens'][ix],
                  'pos': np.concatenate([t['pos'][st[s][i]:st[s][i + 1]] for i in ix])}
            if s == 'eval':
                t2.update(lab=t['lab'].iloc[ix].reset_index(drop=True), half=t['half'][ix],
                          valid={b: v[ix] for b, v in t['valid'].items()})
            tb[s] = t2
    for s in ('train', 'eval'):
        t = tb[s]
        rows = t['F'].row.values.astype(np.int64)
        assert (fp_all[rows] == t['F'].frame_path.values).all()
        t['rows'], t['paths'] = neighbours(t['F'], rows, t['F'].obs.values, R, fp_all)
        assert (t['rows']['m5'] == t['F'].prow.values).all() and (t['paths']['m5'] == t['F'].ppath.values).all()
    return tb


def video_file(frame_dir):
    p = Path(frame_dir).parts  # <subject>/<version>/frames/full/<stem>
    assert p[2:4] == ('frames', 'full'), frame_dir
    base = REPO / 'data' / p[0] / p[1] / 'observations' / 'full' / p[4]
    for ext in ('.mp4', '.mkv'):
        if base.with_suffix(ext).exists():
            return base.with_suffix(ext)
    raise FileNotFoundError(base)


def _decode(a):
    """one video: copy to local disk -> get_frames.py command (ffmpeg611) up to the last needed frame -> keep the needed
    frames at stage/jpg/<frame_path>; returns timing + the byte / pixel check of up to 3 frames vs the dataset JPGs."""
    from PIL import Image
    frame_dir, ks, stage = a
    stage = Path(stage)
    vf = video_file(frame_dir)
    tag = frame_dir.replace('/', '__')
    loc = stage / 'vid' / (tag + vf.suffix)
    tmp = stage / 'dec' / tag
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shutil.copyfile(vf, loc)
    t1 = time.time()
    cmd = [str(FFMPEG), '-v', 'error', '-y', '-i', str(loc), '-vf', 'fps=5,format=rgb24', '-frames:v', str(max(ks) + 1),
           '-start_number', '0', '-q:v', '2', str(tmp / 'frame_%06d.jpg')]
    subprocess.run(cmd, check=True)
    dst = stage / 'jpg' / frame_dir
    dst.mkdir(parents=True, exist_ok=True)
    for k in ks:
        os.replace(tmp / f'frame_{k:06d}.jpg', dst / f'frame_{k:06d}.jpg')
    shutil.rmtree(tmp)
    loc.unlink()
    chk = []
    for k in sorted({ks[0], ks[len(ks) // 2], ks[-1]}):
        a_, b_ = dst / f'frame_{k:06d}.jpg', REPO / 'dataset' / frame_dir / f'frame_{k:06d}.jpg'
        ba, bb = a_.read_bytes(), b_.read_bytes()
        same_bytes = hashlib.md5(ba).digest() == hashlib.md5(bb).digest()
        if same_bytes:
            same_px, mad = True, 0.0
        else:
            with Image.open(a_) as ia, Image.open(b_) as ib:
                xa, xb = np.asarray(ia.convert('RGB'), np.int16), np.asarray(ib.convert('RGB'), np.int16)
            same_px, mad = bool(np.array_equal(xa, xb)), float(np.abs(xa - xb).mean())
        chk.append({'k': k, 'bytes': same_bytes, 'pixels': same_px, 'mad': mad})
    return frame_dir, {'n': len(ks), 'copy_s': round(t1 - t0, 1), 'decode_s': round(time.time() - t1, 1), 'check': chk}


def decode_frames(tb, stage, workers):
    from multiprocessing import Pool
    paths = sorted(set(np.concatenate([tb[s]['paths'][r] for s in tb for r in ROLES]).tolist()))
    by_dir = {}
    for p in paths:
        by_dir.setdefault(str(Path(p).parent), []).append(fidx(p))
    (stage / 'vid').mkdir(parents=True, exist_ok=True)
    jobs = [(d_, sorted(ks), str(stage)) for d_, ks in sorted(by_dir.items(), key=lambda x: -max(x[1]))]
    t0 = time.time()
    rep = {}
    with Pool(workers) as pool:
        for d_, e in pool.imap_unordered(_decode, jobs):
            rep[d_] = e
    ck = [c for e in rep.values() for c in e['check']]
    info = {'ffmpeg': str(FFMPEG), 'n_videos': len(rep), 'n_frames': len(paths), 'wall_s': round(time.time() - t0, 1),
            'decode_s_sum': round(sum(e['decode_s'] for e in rep.values()), 1),
            'check_n': len(ck), 'check_bytes_equal': int(sum(c['bytes'] for c in ck)),
            'check_pixels_equal': int(sum(c['pixels'] for c in ck)), 'check_mad_max': max(c['mad'] for c in ck)}
    log(f'decoded {len(paths):,} frames of {len(rep)} videos in {info["wall_s"]:.0f}s; check vs dataset JPGs: '
        f'{info["check_bytes_equal"]}/{len(ck)} byte-identical, {info["check_pixels_equal"]}/{len(ck)} pixel-identical, '
        f'max MAD {info["check_mad_max"]:.3f}')
    return stage / 'jpg', info


# ---------------------------------------------------------------------------------------------- encode
def encode(tb, jpg, dev, bs, workers):
    """-> {split: {role: (N, 768) fp16}}: every unique JPG encoded once, gathered at each requesting frame's positions."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=True, size={'height': 448, 'width': 448},
                                              do_center_crop=False)
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(dev).eval()
    model.requires_grad_(False)

    class Imgs(torch.utils.data.Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            with Image.open(jpg / self.paths[i]) as im:
                im = im.convert('RGB')
            return i, proc(images=im, return_tensors='pt')['pixel_values'][0]

    X, info = {}, {}
    for split in ('train', 'eval'):
        t = tb[split]
        st = np.r_[0, np.cumsum(t['lens'])]
        N = int(st[-1])
        X[split] = {r: np.empty((N, D), np.float16) for r in ROLES}
        req = {}
        for r in ROLES:
            for f, p in enumerate(t['paths'][r]):
                req.setdefault(p, []).append((r, f))
        paths = sorted(req)
        dl = torch.utils.data.DataLoader(Imgs(paths), batch_size=bs, num_workers=workers, shuffle=False,
                                         pin_memory=True, prefetch_factor=4)
        t0, gpu, n = time.time(), 0.0, 0
        filled = {r: np.zeros(len(t['lens']), bool) for r in ROLES}
        for b, (ii, pix) in enumerate(dl):
            torch.cuda.synchronize(); g0 = time.time()
            with torch.inference_mode():
                h = model(pixel_values=pix.to(dev, non_blocking=True)).last_hidden_state
                assert h.shape[1] == 1 + G * G and torch.isfinite(h).all()
                H = h[:, 1:].float().half().cpu().numpy()
            torch.cuda.synchronize(); gpu += time.time() - g0
            for k, i in enumerate(ii.numpy()):
                for r, f in req[paths[i]]:
                    X[split][r][st[f]:st[f + 1]] = H[k, t['pos'][st[f]:st[f + 1]].astype(np.int64)]
                    filled[r][f] = True
            n += len(ii)
            if b % 500 == 0:
                log(f'  {split} batch {b}: {n:,}/{len(paths):,} JPGs, {n / (time.time() - t0):.0f} f/s')
        assert all(v.all() for v in filled.values())
        info[split] = {'unique_jpgs': len(paths), 'requests': len(t['lens']) * len(ROLES), 'tokens': N,
                       'wall_s': round(time.time() - t0, 1), 'gpu_s': round(gpu, 1)}
        log(f'  {split}: {len(paths):,} unique JPGs for {len(t["lens"]) * len(ROLES):,} (frame, role) requests in '
            f'{time.time() - t0:.0f}s (gpu {gpu:.0f}s)')
    del model
    torch.cuda.empty_cache()
    return X, info


def check_vs_store(d, X, tb, smoke):
    """re-encoded tokens vs the stores: t vs the base store and t-5 vs the d5 store on 64 train frames (+ 64 ants eval
    frames); mice eval t vs BDIR tok448 (first 300k tokens)."""
    import motion_t3 as mt
    import spatial_sae_pilot as ssp
    cos = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)  # noqa: E731
    r = {}
    if d == 'mice' and not smoke:
        import quick_repr as qr
        ref = np.load(qr.BDIR / 'tok448.npy', mmap_mode='r')
        n = 300_000
        c = cos(np.asarray(X['eval']['t'][:n], np.float32), np.asarray(ref[:n], np.float32))
        r['eval_t_vs_bdir_tok448'] = {'n': n, 'cos_min': float(c.min()), 'cos_mean': float(c.mean())}
    idx = ssp.StoreIndex(d)
    dm = mt.D5Matcher(idx, d)
    for s in (['train', 'eval'] if d == 'ants' else ['train']):
        F = tb[s]['F']
        fr = F.store_frame.values[:64]
        assert (np.diff(fr) > 0).all()
        tok, pos, lens = idx.load(fr)
        m = int(lens.sum())
        assert (pos == tb[s]['pos'][:m]).all()
        prev, _, ok = dm.fetch(fr)
        c = cos(np.asarray(X[s]['t'][:m], np.float32), tok.astype(np.float32))
        r[f'{s}_t_vs_store'] = {'n': m, 'cos_min': float(c.min()), 'cos_mean': float(c.mean())}
        c = cos(np.asarray(X[s]['m5'][:m], np.float32)[ok], prev[ok].astype(np.float32))
        r[f'{s}_m5_vs_d5'] = {'n': int(ok.sum()), 'cos_min': float(c.min()), 'cos_mean': float(c.mean())}
    log(f'store check: {json.dumps(r)}')
    bad = {k: v for k, v in r.items() if v['cos_min'] < 0.99 or v['cos_mean'] < 0.9999}
    if bad:
        raise RuntimeError(f're-encoded tokens do not match the store: {bad}')
    return r


# ---------------------------------------------------------------------------------------------- arm inputs
class ArmX:
    """lazy (N, 768 * n_blocks) fp16 view of one arm's input over the role arrays (slices only)."""

    def __init__(self, R, arm):
        self.R, self.spec = R, ARMS[arm]
        self.n = len(R['t'])
        self.shape = (self.n, D * len(self.spec))

    def __len__(self):
        return self.n

    def __getitem__(self, s):
        assert isinstance(s, slice)
        out = []
        for a, b in self.spec:
            xa = self.R[a][s]
            out.append(xa if b is None else (xa.astype(np.float32) - self.R[b][s].astype(np.float32)).astype(np.float16))
        return np.concatenate(out, 1)


def to_gpu(R, arm, dev, chunk=500_000):
    import torch
    A = ArmX(R, arm)
    X = torch.empty(A.shape, dtype=torch.float16, device=dev)
    for a in range(0, len(A), chunk):
        X[a:a + chunk] = torch.from_numpy(A[a:a + chunk]).to(dev)
    return X


def block_stats(X, nb, n=200_000):
    """fraction of tokens whose change block is exactly zero (clipped neighbour) + median change / token norm."""
    x = X[:n].float()
    o = {}
    for j in range(1, nb):
        c = x[:, j * D:(j + 1) * D]
        o[f'block{j}_zero_frac'] = float((c.abs().sum(1) == 0).float().mean())
        o[f'block{j}_median_norm_over_token'] = float((c.norm(dim=1) / x[:, :D].norm(dim=1)).median())
    return o


# ---------------------------------------------------------------------------------------------- train
def train_one(arm, seed, X, steps, out, dev, bs=4096, lr=5e-4, warmup=500):
    """motion_t3.train_one with one TokenNorm block per input block (otherwise line for line the same)."""
    import torch
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    N, d = X.shape
    nb = d // D
    ii = torch.from_numpy(np.sort(g.choice(N, min(500_000, N), replace=False))).to(dev)
    init = X[ii].float()
    norm = TokenNorm(d).to(dev).fit_blocks(init, [D] * nb, seed=seed)
    sae = MatryoshkaBatchTopKSAE(d_in=d, n_latents=WIDTH, prefixes=(WIDTH // 8, WIDTH // 4, WIDTH // 2, WIDTH), k=K,
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
            log(f"    {arm} s{seed} step {step}/{steps} fve {logs['fve']:.4f} l0 {logs['l0']:.2f} dead {logs['n_dead']} "
                f"{logs['elapsed_s']}s")
    sae.eval()
    dead = int((sae.tokens_since_fired >= sae.dead_tokens).sum())
    extra = {'config': 'temporal_ctx', 'arm': arm, 'blocks': ARMS[arm], 'seed': seed}
    save_checkpoint(out / 'sae.pt', sae, norm, extra=extra)
    (out / 'train.json').write_text(json.dumps({**extra, 'n_pool': N, 'steps': steps, 'token_presentations': steps * bs,
                                                'train_time_s': time.time() - t0, 'dead_at_end_train': dead,
                                                'dead_frac_train': dead / WIDTH, 'threshold': float(sae.threshold),
                                                'norm_scale_blocks': [float(norm.scale[j * D]) for j in range(nb)],
                                                'gpu': torch.cuda.get_device_name(0), 'history': history}, indent=1))
    log(f'    {arm} s{seed}: trained in {time.time() - t0:.0f}s, final fve {history[-1]["fve"]:.4f}, dead {dead}/{WIDTH}')


# ---------------------------------------------------------------------------------------------- eval
def eval_arm(d, arm, seed, R, tb, dev, half_all):
    import torch
    import vjepa as vj
    from src.eci.sae import load_sae
    od = OUT / d / 'eval'
    od.mkdir(parents=True, exist_ok=True)
    key = f'{arm}_s{seed}'
    if (od / f'{key}.json').exists():
        log(f'[SKIP] eval {d} {key}')
        return
    sae, norm, _ = load_sae(OUT / d / 'sae' / key / 'sae.pt', dev)
    t = tb['eval']
    codes, st, _ = vj.frame_max_tokens(sae, norm, ArmX(R, arm), t['lens'], dev)
    res = {'stats': st, 'theta': float(sae.threshold)}
    if d == 'mice':
        F, sub = t['F'], t['sub']
        rows = {'all': np.arange(len(F)), 'usable': np.flatnonzero(F.usable.values)}
        for fs, rr in rows.items():
            for p in (512, 4096):
                r = vj.score(codes[:, :p].contiguous(), sub, rr, half_all, dev)
                res[f'{fs}_m{p}'] = r
                log(f'  mice {key} {fs} m{p}: ' + ', '.join(
                    f"{b} AUROC {r[b]['cf_auroc']:.3f} AP {r[b]['cf_ap']:.4f} top1 {r[b]['cf_top1']:.3f}"
                    for b in LABELS[d]))
        w = sub.w.values.astype(np.float64)
        u = rows['usable']
        rr_ = torch.from_numpy(u).to(dev)
        ww = torch.from_numpy(w[u]).to(dev).float()
        fr = (((codes[rr_] > 0).float() * ww[:, None]).sum(0) / ww.sum()).cpu().numpy()
        main = 'usable_m4096'
    else:
        import sae_levers as sl
        lab, half, valid = t['lab'], t['half'], t['valid']
        l448 = t['lens']
        rank = np.argsort(np.argsort(l448 + np.random.default_rng(0).random(len(l448)) * 1e-3))
        decile = np.minimum(rank * 10 // len(l448), 9)
        Yt = torch.from_numpy(lab[LABELS[d]].values.astype(bool)).to(dev)
        for p in (512, 4096):
            r, _ = sl.score_model(codes[:, :p].contiguous(), Yt, LABELS[d], valid, half, decile, dev)
            res[f'all_m{p}'] = r
            log(f'  ants {key} m{p}: ' + ', '.join(
                f"{b} AUROC {r[b]['cf_auroc']:.3f} AP {r[b]['cf_ap']:.4f} top1 {r[b]['cf_top1']:.3f}" for b in LABELS[d]))
        fr = (codes > 0).float().mean(0).cpu().numpy()
        main = 'all_m4096'
    elig = fr >= MIN_RATE
    res['fire_rate'] = fr.tolist()
    sel = {b: {s: [x[s]['test_rate'] for x in res[main][b]['dirs']] for s in ('auc', 'ap', 'top1')} for b in LABELS[d]}
    res['firing'] = {'dead_frac_eval': float((fr == 0).mean()), 'n_eligible': int(elig.sum()),
                     'median_rate_eligible': float(np.median(fr[elig])) if elig.any() else float('nan'),
                     'median_rate_alive': float(np.median(fr[fr > 0])) if (fr > 0).any() else float('nan'),
                     'frac_rate_gt_0.5': float((fr > 0.5).mean()), 'selected_neuron_rates': sel}
    log(f"  {d} {key}: fve {st['fve']:.4f} l0 {st['l0']:.2f} firing "
        f"{ {k: v for k, v in res['firing'].items() if k != 'selected_neuron_rates'} } selected {sel}")
    (od / f'{key}.json').write_text(json.dumps(res, indent=1, default=float))
    del codes
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- sheets (mice)
def sheets(arm, seed, R, tb, jpg, dev, n_tiles=16):
    """best nn / nt neurons (AUROC- and top-1%-selected on half 0, shown on half 1): full frame t (red = argmax patch,
    yellow = crop) | crop t-1 | crop t | crop t+1. Border green = nn, blue = nt, grey = neither."""
    from PIL import Image, ImageDraw
    import contact_sae as cs
    import vjepa as vj
    from src.eci.sae import load_sae
    d = 'mice'
    key = f'{arm}_s{seed}'
    Rj = json.loads((OUT / d / 'eval' / f'{key}.json').read_text())
    t = tb['eval']
    F, sub, pos, lens = t['F'], t['sub'], t['pos'], t['lens']
    half = cs.video_units(sub.obs.values)
    fr = Rj['fire_rate']
    picks = []
    for b in LABELS[d]:
        for sel in ('auc', 'top1'):
            e = Rj['usable_m4096'][b]['dirs'][0]
            j = e[sel]['neuron']
            picks.append((f'best_{b}_{sel}', j, (half == 1) & F.usable.values,
                          f'{sel}-selected on half 0 for {b}; half 1: AUROC {e[sel]["test_auc"]:.3f} AP '
                          f'{e[sel]["test_ap"]:.3f} top1 {e[sel]["test_top1"]:.3f}; fires on {fr[j]:.1%} of usable frames'))
    sae, norm, _ = load_sae(OUT / d / 'sae' / key / 'sae.pt', dev)
    J = sorted({p[1] for p in picks})
    codes, _, arg = vj.frame_max_tokens(sae, norm, ArmX(R, arm), lens, dev, want_arg=J)
    codes = codes[:, J].float().cpu().numpy()
    arg = arg.cpu().numpy()
    st = np.r_[0, np.cumsum(lens)]
    sd = OUT / d / 'sheets'
    sd.mkdir(parents=True, exist_ok=True)
    th, index = 200, []
    for name, j, msk, note in picks:
        q = J.index(j)
        a = codes[:, q].copy()
        a[~msk] = -1
        chosen, per_v = [], {}
        for f in np.argsort(-a):
            if a[f] <= 0 or len(chosen) >= n_tiles:
                break
            v = F.obs.iat[f]
            if per_v.get(v, 0) >= 2:
                continue
            per_v[v] = per_v.get(v, 0) + 1
            chosen.append(int(f))
        nc = 2
        nr = max(1, (len(chosen) + nc - 1) // nc)
        TW = 4 * th + 9
        sheet = Image.new('RGB', (nc * (TW + 8), nr * (th + 26) + 24), 'white')
        dr = ImageDraw.Draw(sheet)
        dr.text((4, 4), f'{arm} s{seed} neuron {j}: {note}', fill='black')
        dr.text((4, 13), 'tile: full t (red = argmax patch, yellow = crop) | crop t-1 | crop t | crop t+1 (0.2 s apart). '
                'border green nn, blue nt, grey none', fill='black')
        npos = {b: 0 for b in LABELS[d]}
        for i, f in enumerate(chosen):
            ti = int(arg[f, q])
            assert st[f] <= ti < st[f + 1]
            p = int(pos[ti])
            r_, c_ = divmod(p, G)
            s_ = 512 / G
            cy, cx = (r_ + 0.5) * s_, (c_ + 0.5) * s_
            side = 160
            x0 = int(np.clip(cx - side / 2, 0, 512 - side)); y0 = int(np.clip(cy - side / 2, 0, 512 - side))
            ims = {}
            for role in ('m1', 't', 'p1'):
                with Image.open(jpg / t['paths'][role][f]) as im:
                    ims[role] = im.convert('RGB')
            crops = [ims[r].crop((x0, y0, x0 + side, y0 + side)).resize((th, th)) for r in ('m1', 't', 'p1')]
            full = ims['t']
            d2 = ImageDraw.Draw(full)
            d2.rectangle([c_ * s_, r_ * s_, (c_ + 1) * s_, (r_ + 1) * s_], outline=(255, 40, 40), width=3)
            d2.rectangle([x0, y0, x0 + side, y0 + side], outline=(255, 220, 0), width=2)
            yn, yt = bool(sub.nose_nose.iat[f]), bool(sub.nose_tail.iat[f])
            npos['nose_nose'] += yn; npos['nose_tail'] += yt
            col = (0, 200, 0) if yn else (40, 110, 255) if yt else (150, 150, 150)
            tile = Image.new('RGB', (TW, th + 6), col)
            for k_, im in enumerate([full.resize((th, th))] + crops):
                tile.paste(im, (3 + k_ * (th + 1), 3))
            rr, cc = divmod(i, nc)
            X0, Y0 = cc * (TW + 8), 24 + rr * (th + 26)
            sheet.paste(tile, (X0, Y0))
            dr.text((X0 + 2, Y0 + th + 8), f'{F.obs.iat[f][:22]} k{int(F.k.iat[f])} a={a[f]:.2f}', fill='black')
        fn = f'{key}_{name}.jpg'
        sheet.save(sd / fn, quality=85)
        index.append({'sheet': fn, 'neuron': j, 'note': note, 'n_tiles': len(chosen),
                      **{f'n_pos_{b}': npos[b] for b in LABELS[d]}})
        log(f'sheet {fn}: {len(chosen)} tiles, positives {npos}')
    (sd / f'index_{key}.json').write_text(json.dumps(index, indent=1))


def best_arm(d, arms):
    E = OUT / d / 'eval'
    ap = {a: {b: np.mean([json.loads(f.read_text())['usable_m4096'][b]['cf_ap'] for f in sorted(E.glob(f'{a}_s*.json'))])
              for b in LABELS[d]} for a in arms}
    rat = {a: float(np.mean([ap[a][b] / ap['M5'][b] for b in LABELS[d]])) for a in arms if a != 'M5'}
    return max(rat, key=rat.get), rat


# ---------------------------------------------------------------------------------------------- run
def cmd_run(args):
    import torch
    dev = torch.device('cuda')
    t00 = time.time()
    d = args.domain
    stage = stage_dir()
    tb = tables(d, args.smoke)
    if args.smoke:
        args.steps = 100
    half_all = None
    if d == 'mice':
        import contact_sae as cs
        half_all = cs.video_units(tb['eval']['sub'].obs.values)
    od = OUT / d
    od.mkdir(parents=True, exist_ok=True)
    nbr = {s: {r: {'clipped_to_t': int((tb[s]['rows'][r] == tb[s]['F'].row.values).sum()) if r != 't' else 0,
                   'clipped': int((tb[s]['rows'][r] - tb[s]['F'].row.values != OFF[r]).sum())} for r in ROLES}
           for s in tb}
    log(f'{d}: train {len(tb["train"]["lens"]):,} frames / {int(tb["train"]["lens"].sum()):,} tokens; eval '
        f'{len(tb["eval"]["lens"]):,} frames / {int(tb["eval"]["lens"].sum()):,} tokens; neighbour clipping {nbr}')
    jpg, dinfo = decode_frames(tb, stage, args.decode_workers)
    X, einfo = encode(tb, jpg, dev, args.bs, args.workers)
    einfo['store_check'] = check_vs_store(d, X, tb, args.smoke)
    tok_meta = {'gpu': torch.cuda.get_device_name(0), 'host': os.uname().nodename, 'decode': dinfo, 'encode': einfo,
                'neighbour_clipping': nbr, 'wall_s_to_here': round(time.time() - t00, 1)}
    (od / f'encode_{"_".join(args.arms)}.json').write_text(json.dumps(tok_meta, indent=1, default=float))
    for seeds in ([args.seeds[0]], args.seeds[1:]):  # seed 0 of every arm first (interim read-out)
        if not seeds:
            continue
        for arm in args.arms:
            Xtr = None
            for seed in seeds:
                if not (od / 'sae' / f'{arm}_s{seed}' / 'sae.pt').exists():
                    if Xtr is None:
                        Xtr = to_gpu(X['train'], arm, dev)
                        bs_ = block_stats(Xtr, len(ARMS[arm]))
                        log(f'  {arm} train input {tuple(Xtr.shape)}: {bs_}')
                        (od / f'train_meta_{arm}.json').write_text(json.dumps(
                            {'arm': arm, 'blocks': ARMS[arm], 'shape': list(Xtr.shape), **bs_}, indent=1))
                    torch.backends.cuda.matmul.allow_tf32 = True  # as motion_t3 train
                    train_one(arm, seed, Xtr, args.steps, od / 'sae' / f'{arm}_s{seed}', dev)
                    torch.cuda.empty_cache()
                else:
                    log(f'[SKIP] train {arm}_s{seed}')
                torch.backends.cuda.matmul.allow_tf32 = False
                eval_arm(d, arm, seed, X['eval'], tb, dev, half_all)
            del Xtr
            torch.cuda.empty_cache()
    if d == 'mice' and 'M5' in args.arms and len(args.arms) > 1:
        best, rat = best_arm(d, args.arms)
        log(f'best arm (mean AP / M5 over nn, nt): {best}; ratios {rat}')
        for arm in dict.fromkeys([best, 'S1']):
            if arm in args.arms:
                sheets(arm, args.seeds[0], X['eval'], tb, jpg, dev)
    log(f'run done in {time.time() - t00:.0f}s')


# ---------------------------------------------------------------------------------------------- summary
def ms(a, n=3):
    return f'{np.mean(a):.{n}f}+-{np.std(a):.{n}f}' if len(a) > 1 else f'{np.mean(a):.{n}f}'


def load_refs(d):
    if d == 'mice':
        fs = sorted(SPE.glob('c448_s*.json'))
        return [json.loads(f.read_text()) for f in fs]
    fs = sorted((RP / 'ants' / 'eval').glob('c448_max_s*.json'))
    out = []
    for f in fs:
        e = json.loads(f.read_text())
        out.append({'all_m4096': e['labels'], 'stats': {'fve': e['fve'], 'l0': e['l0_per_token']}})
    return out


def collect(d):
    E = OUT / d / 'eval'
    R = {a: [json.loads(f.read_text()) for f in sorted(E.glob(f'{a}_s*.json'))] for a in ARMS}
    R = {a: v for a, v in R.items() if v}
    ref = load_refs(d)
    if ref:
        R = {'c448ref': ref, **R}
    keys = ['usable_m4096', 'usable_m512', 'all_m4096'] if d == 'mice' else ['all_m4096', 'all_m512']
    S = {}
    for a, rr in R.items():
        s = {'n_seeds': len(rr)}
        for key in keys:
            for b in LABELS[d]:
                if key not in rr[0]:
                    continue
                s[f'{key}/{b}'] = {q: [r[key][b][q] for r in rr] for q in ('cf_auroc', 'cf_ap', 'cf_top1', 'cf_top1_rate')}
                s[f'{key}/{b}']['auc_sel_rate'] = [float(np.mean([x['auc']['test_rate'] for x in r[key][b]['dirs']]))
                                                   for r in rr]
        s['fve'] = [r['stats']['fve'] for r in rr]
        s['l0'] = [r['stats']['l0'] for r in rr]
        if 'firing' in rr[0]:
            for q in ('dead_frac_eval', 'median_rate_eligible', 'median_rate_alive', 'frac_rate_gt_0.5', 'n_eligible'):
                s[q] = [r['firing'][q] for r in rr]
        S[a] = s
    return S, keys


def cmd_summary(args):
    lines = ['# temporal_ctx: temporal context of the ECI input (cross-fitted best neuron, 3 seeds mean+-sd; '
             'reference M5 = retrained in the same job on the same tokens; c448ref = earlier harness numbers, same '
             'frames)', '']
    SS, V = {}, {}
    for d in ('mice', 'ants'):
        if not (OUT / d / 'eval').exists():
            continue
        S, keys = collect(d)
        SS[d] = S
        for key in keys:
            lines += [f'## {d} {key}', '', '| arm | seeds | ' + ' | '.join(
                f'{b} AUROC | {b} AP | {b} top-1%' for b in LABELS[d]) + ' |', '|---|---|' + '---|---|---|' * len(LABELS[d])]
            for a, s in S.items():
                cells = []
                for b in LABELS[d]:
                    x = s.get(f'{key}/{b}')
                    cells += ([ms(x['cf_auroc']), ms(x['cf_ap'], 4), ms(x['cf_top1'])] if x else ['-'] * 3)
                lines.append(f"| {a} | {s['n_seeds']} | " + ' | '.join(cells) + ' |')
            lines.append('')
        main = 'usable_m4096' if d == 'mice' else 'all_m4096'
        lines += [f'## {d} reconstruction and firing ({"usable frames, weighted" if d == "mice" else "all eval frames"}; '
                  'a frame fires iff max over kept tokens > 0)', '',
                  '| arm | eval FVE | token L0 | dead frac | median rate (eligible >= 1%) | frac neurons rate > 50% | '
                  + ' | '.join(f'rate of AUROC-picked {b} | rate of top-1%-picked {b}' for b in LABELS[d]) + ' |',
                  '|---|---|---|---|---|---|' + '---|---|' * len(LABELS[d])]
        for a, s in S.items():
            g = lambda q, n=3: ms(s[q], n) if q in s else '-'  # noqa: E731
            cells = []
            for b in LABELS[d]:
                cells += [ms(s[f'{main}/{b}']['auc_sel_rate'], 2), ms(s[f'{main}/{b}']['cf_top1_rate'], 2)]
            lines.append(f"| {a} | {g('fve', 4)} | {g('l0', 2)} | {g('dead_frac_eval', 4)} | {g('median_rate_eligible')} | "
                         f"{g('frac_rate_gt_0.5')} | " + ' | '.join(cells) + ' |')
        lines.append('')
    # pre-registered verdicts vs M5
    if 'mice' in SS and 'M5' in SS['mice']:
        lines += ['## pre-registered verdicts (3-seed means; mice usable m4096, ants all m4096; reference M5 of this job)',
                  '']
        M = {d: SS[d]['M5'] for d in SS if 'M5' in SS[d]}
        main = {'mice': 'usable_m4096', 'ants': 'all_m4096'}
        for a in [x for x in ARMS if x != 'M5' and x in SS['mice']]:
            v = {'auroc_drop': {}, 'gain': {}}
            for d in M:
                if a not in SS[d]:
                    continue
                for b in LABELS[d]:
                    v['auroc_drop'][f'{d}/{b}'] = float(np.mean(M[d][f'{main[d]}/{b}']['cf_auroc'])
                                                        - np.mean(SS[d][a][f'{main[d]}/{b}']['cf_auroc']))
            for b in LABELS['mice']:
                x, r = SS['mice'][a][f'usable_m4096/{b}'], M['mice'][f'usable_m4096/{b}']
                apr = float(np.mean(x['cf_ap']) / np.mean(r['cf_ap']))
                dt = float(np.mean(x['cf_top1']) - np.mean(r['cf_top1']))
                v['gain'][b] = {'ap_ratio': apr, 'top1_delta': dt, 'pass': bool(apr >= 1.3 or dt >= 0.05)}
            v['no_auroc_drop'] = all(x <= 0.02 for x in v['auroc_drop'].values())
            v['domains_checked'] = sorted({k.split('/')[0] for k in v['auroc_drop']})
            v['beats_M5'] = bool(v['no_auroc_drop'] and any(g['pass'] for g in v['gain'].values()))
            v['mean_ap_ratio'] = float(np.mean([g['ap_ratio'] for g in v['gain'].values()]))
            V[a] = v
            worst = max(v['auroc_drop'], key=v['auroc_drop'].get)
            lines.append(f"- {a}: beats M5 = {v['beats_M5']} (domains {v['domains_checked']}); largest AUROC drop "
                         f"{worst} {v['auroc_drop'][worst]:+.3f}; " + '; '.join(
                             f"{b}: AP x{g['ap_ratio']:.2f}, top-1% {g['top1_delta']:+.3f}" for b, g in v['gain'].items()))
        if 'S1' in V:
            x, r = SS['mice']['S1']['usable_m4096/nose_nose'], M['mice']['usable_m4096/nose_nose']
            V['S1_on_nn'] = {'auroc_delta': float(np.mean(x['cf_auroc']) - np.mean(r['cf_auroc'])),
                             'ap_ratio': float(np.mean(x['cf_ap']) / np.mean(r['cf_ap'])),
                             'top1_delta': float(np.mean(x['cf_top1']) - np.mean(r['cf_top1']))}
            lines.append(f"- S1 on nose_nose vs M5: AUROC {V['S1_on_nn']['auroc_delta']:+.3f}, AP "
                         f"x{V['S1_on_nn']['ap_ratio']:.2f}, top-1% {V['S1_on_nn']['top1_delta']:+.3f}")
        cand = {a: v['mean_ap_ratio'] for a, v in V.items() if a in ARMS}
        if cand:
            best = max(cand, key=cand.get)
            V['best_arm'] = best
            lines += ['', f'Best arm (highest mean AP / M5 over nn, nt): {best} (x{cand[best]:.2f}); sheets: '
                      f'mice/sheets/{best}_s0_best_*.jpg (and S1_s0_best_*.jpg)']
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'summary.json').write_text(json.dumps({'arms': SS, 'verdict': V}, indent=1, default=float))
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['run', 'summary'])
    ap.add_argument('--domain', choices=['mice', 'ants'], default='mice')
    ap.add_argument('--arms', nargs='+', default=list(ARMS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--decode-workers', type=int, default=12)
    ap.add_argument('--smoke', action='store_true', help='300 train frames, a few eval videos, 100 steps; set TC_OUT')
    args = ap.parse_args()
    for a in args.arms:
        assert a in ARMS, a
    if args.smoke:
        assert os.environ.get('TC_OUT'), 'smoke: set TC_OUT to a scratch results dir'
    {'run': cmd_run, 'summary': cmd_summary}[args.step](args)


if __name__ == '__main__':
    main()
