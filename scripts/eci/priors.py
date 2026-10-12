"""
PRIORS test: two label-free behavioural priors added to the SAE objective of the final ECI pipeline C1 (ants + mice).

C1 (scripts/eci/combined.py, commit 5d7b8c0): 5 fps frames -> DINOv2-base 448 px patch tokens -> today's foreground
mask (mice fg448, ants antsfg) -> M5 input x = [token_t, token_t - token_{t-5}] (TokenNorm.fit_blocks([768, 768])) ->
Matryoshka BatchTopK SAE (4096 latents, prefixes 512 / 1024 / 2048 / 4096, k = 16, motion_t3.train_one recipe) ->
frame read-out = max over kept patches. The two loss terms are defined in src/eci/priors.py:
    INV   lambda * mean ||a(x) - a(x')||^2 / (||a(x)||^2 + 1), a = dense pre-TopK ReLU activations, x' = the same
          patch in ONE fixed D4-augmented copy of the frame pair (random non-identity element of rot90 x flip per
          training frame, rng 0), encoded by the same FgEncoder at 448. Reconstruction on x only.
    PRED  mu * ||d+ - P z_t||^2 / Var(d+), d+ = token_{t+5} - token_t (static block, same patch, 1 s ahead), P linear
          (+ bias, zero init) on the full BatchTopK training code, masked where t+5 is past the video end.

Arms (3 seeds each; identical train frames, splits, recipe, steps (6000), batch (4096) and seeds per domain)
    C1      res_pool 'c448': the C1 recipe trained on exactly these train frames (results/vision/eci_res_pool/<d>/sae/
            c448_s*), RE-EVALUATED here (its cf metrics must reproduce res_pool's eval JSONs). C1r = the same recipe
            retrained by this harness at lambda = mu = 0 (seed 0) as the identity check of the training loop.
    C1dep   reference only: the deployed C1 checkpoints (results/vision/eci_t3_motion/<d>/sae/M5_s*; other train frames).
    INV, PRED, INVPRED   C1 + the term(s).
Train frames = the res_pool train frames (mice 10,000 = 100 per unannotated train video; ants 20,480 = 160 per train
    video), t-5 from the d5 store (as C1). New encodes: the augmented t and t-5 frames and the t+5 frames, decoded
    LOCALLY from the standardised videos (data/<subject>/<version>/observations/full/<video>.mp4|mkv, the
    src/dataset/get_frames.py chain: fps=5, format=rgb24, JPEG q 2, with ffmpeg 6.1.1 of the ffmpeg611 env; byte-checked
    against one dataset JPEG per video).
Eval frames = ALL store frames of the annotated eval videos (mice 172,800 / 144 videos, ants 76,800 / 128 videos).

LAMBDA / MU RULE (fixed before any evaluation): per domain, train the plain model (C1, seed 0) for the first 1000
    steps of the 6000-step schedule; on 25 held batches of the training pool measure L_rec (the Matryoshka loss = sum
    over the 4 prefixes of the per-token SSE), L_inv raw, and L_pred raw (least-squares head with intercept on the
    warm-up code, fitted on 60 other batches). lambda = 0.25 L_rec / L_inv, mu = 0.25 L_rec / L_pred, so each term is
    25% of the reconstruction loss at the end of the warm-up. INVPRED uses both values.

Evaluation per arm x seed x Matryoshka PREFIX m in (512, 1024, 4096) (best-neuron search restricted to latents < m):
    frame level   sae_levers.score_model: cross-fitted best-neuron AUROC, AP (best-AP pick), honest top-1% (neurons
                  firing on >= 1% of frames), size-controlled AUROC
    video level   res_pool.video_level (scripts/eci/readouts.py): held-out per-video mean of the AUROC pick vs the
                  annotated rate and bouts / min, raw and fg-count controlled; mice phase switches (pooled d-r, dz)
    composition   alive (fires on >= 1 eval token) / eligible (frame value > 0 on >= 1% of eval frames) latents < m;
                  POSITION-BOUND share of eligible = conc > 0.3, conc = 1 - H / H_ref(K) of the argmax patches of the
                  neuron's top 1% eval frames on the 8 x 8 cell grid (neuron_map_v2 / tight_mask_sae rule, unweighted);
                  PREDICTIVE: cross-fitted single-neuron R^2 of d+ on held-out PROBE tokens (store frames of the train
                  videos NOT used in training, t+5 encoded), split by video halves; share of eligible neurons with R^2
                  above tau = the 90th percentile of C1 seed 0's eligible neurons at 4096 (so C1 s0 = 10% by
                  construction); also the same R^2 of d+ after removing its ridge prediction from the current input x
                  ('beyond-input'), and the whole-code ridge R^2 per prefix.
    sheets        best mice nose_nose / nose_tail and ants groom_any neuron of every arm (seed with the best cf AUROC,
                  prefix 4096): top test-half frames, <= 2 per video (frame t | zoom t | zoom t-5); 20 random eligible
                  mice neurons of the best arm (seed 0), 8 top frames each.

PRE-REGISTERED DECISIONS
    PRIMARY   some arm, at some prefix, has a mice neuron with honest top-1% >= 0.40 for nose_nose or nose_tail
              (cross-fitted, 3-seed mean of cf_top1).
    KEEP      per prior (INV, PRED; INVPRED reported with the same rule), at prefix 4096 vs C1 at 4096: in BOTH domains
              no label's cf AUROC (3-seed mean) drops > 0.02, AND >= 1 label gains: cf AUROC >= +0.03 and above C1's
              best seed, OR cf AP >= 1.3 x and above C1's best seed, OR count-controlled video r (rate or bouts / min)
              >= +0.10 (res_pool.res_rule). Also reported at 512 / 1024 vs C1 at the same prefix.
    WIDTH     for each kept arm (and C1): is 512 or 1024 within 0.02 cf AUROC of 4096 on every label (3-seed means)?

Steps (scripts/eci/priors.sh)
    plan      CPU  frame tables (aug element, t+5), probe frames, video decode lists, store tokens + d5 prev of the
                   train and probe frames -> OUT/<d>/{plan,base}/
    encode    GPU  local decode + encode of the augmented t / t-5, t+5 (train) and t+5 (probe) frames -> OUT/<d>/enc/
    warmup    GPU  the lambda / mu rule -> OUT/<d>/weights.json
    train     GPU  arms x seeds -> OUT/<d>/sae/<arm>_s<seed>/ (+ measure.json: the 3 loss values on held batches)
    evaluate  GPU  eval codes (MAX + argmax cell) of every SAE, score per prefix, composition -> OUT/<d>/{eval,comp,
                   codes_best}/
    sheets    CPU  contact sheets -> OUT/<d>/sheets/
    summary   login safe: tables + decisions -> OUT/summary.json, OUT/tables.md
Outputs: results/vision/eci_priors/ (gitignored). The large token files (base/, enc/) are deleted at the end (clean).
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
OUT = Path(os.environ.get('PR_OUT') or REPO / 'results/vision/eci_priors')
RP = REPO / 'results/vision/eci_res_pool'
T3 = REPO / 'results/vision/eci_t3_motion'
D = 768
SEEDS = (0, 1, 2)
PREFIXES = (512, 1024, 4096)
DOMAINS = ('mice', 'ants')
LABELS = {'mice': ['nose_nose', 'nose_tail'], 'ants': ['groom_any', 'groom_yellow', 'groom_blue', 'onlid_yellow']}
ARMS = {'C1': (0, 0), 'C1r': (0, 0), 'INV': (1, 0), 'PRED': (0, 1), 'INVPRED': (1, 1)}  # (uses lambda, uses mu)
EVAL_ARMS = ('C1', 'C1dep', 'INV', 'PRED', 'INVPRED')
N_PROBE = {'mice': 1700, 'ants': 5000}
WARM_STEPS, TOTAL_STEPS, SHARE = 1000, 6000, 0.25
TOP_FRAC, MIN_FIRE, PLACE_CONC = 0.01, 0.01, 0.3
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731
# ffmpeg 6.1.1 (conda env ffmpeg611) reproduces the dataset JPGs bit for bit; crl's ffmpeg 4.4 shifts pixels by ~1 grey level
FFMPEG = os.environ.get('PR_FFMPEG') or str(Path.home() / '.conda/envs/ffmpeg611/bin/ffmpeg')


def stage_dir():
    d = Path(os.environ.get('LOCAL_DIR') or '')
    if not str(d).startswith('/localhome'):
        raise RuntimeError(f'LOCAL_DIR must be the job dir under /localhome, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


def video_file(frame_path):
    """dataset frame path <subject>/<version>/frames/full/<video>/frame_k.jpg -> the standardised video."""
    p = Path(frame_path)
    subj, ver, name = p.parts[0], p.parts[1], p.parent.name
    for ext in ('.mp4', '.mkv'):
        f = REPO / 'data' / subj / ver / 'observations' / 'full' / f'{name}{ext}'
        if f.exists():
            return f
    raise FileNotFoundError(f'no standardised video for {frame_path}')


def sae_path(d, arm, seed):
    if arm == 'C1':
        return RP / d / 'sae' / f'c448_s{seed}' / 'sae.pt'
    if arm == 'C1dep':
        return T3 / d / 'sae' / f'M5_s{seed}' / 'sae.pt'
    return OUT / d / 'sae' / f'{arm}_s{seed}' / 'sae.pt'


def copy_local(src, stage):
    dst = stage / src.name
    if not dst.exists():
        shutil.copyfile(src, dst)
    return dst


# ---------------------------------------------------------------------------------------------- plan (CPU)
def cmd_plan(args):
    import motion_t3 as mt
    import res_pool as rp
    import spatial_sae_pilot as ssp
    from src.eci.domain import get_domain
    from src.eci.foreground import obs_rows
    from src.eci.priors import check_d4
    check_d4()
    log('D4 self-test passed (8 elements x 64 patches, pixel blocks exact)')
    for d in args.domains.split(','):
        t0 = time.time()
        od = OUT / d / 'plan'
        ob = OUT / d / 'base'
        od.mkdir(parents=True, exist_ok=True)
        ob.mkdir(parents=True, exist_ok=True)
        F, p448, l448 = rp.table(d, 'train')
        ann_p = get_domain(d).ann_path
        ann = pd.read_csv(ann_p, usecols=['observation_id', 'frame_idx', 'frame_path'])
        R = obs_rows(ann_p)
        hi = np.array([R[o][1] for o in F.obs])
        F['g'] = np.random.default_rng(0).integers(1, 8, len(F))  # non-identity D4 element per frame
        F['next_ok'] = (F.row.values + 5) < hi
        nrow = np.where(F.next_ok, F.row.values + 5, F.row.values)
        F['nrow'] = nrow
        F['nfi'] = np.where(F.next_ok, ann.frame_idx.values[nrow], -1)
        assert (ann.observation_id.values[nrow] == F.obs.values).all()
        assert ((F.nfi - F.frame_idx).values[F.next_ok.values] == 5).all()
        vmap = {o: str(video_file(p).relative_to(REPO)) for o, p in F.groupby('obs').frame_path.first().items()}
        F['video'] = F.obs.map(vmap)
        # probe frames: store frames of the train videos that are not training frames, with t+5 inside the video
        idx = ssp.StoreIndex(d)
        cand = np.flatnonzero(np.isin(idx.obs, list(vmap)) & (idx.nfg > 0) & ~np.isin(np.arange(len(idx.obs)), F.store_frame.values))
        hic = np.array([R[o][1] for o in idx.obs[cand]])
        cand = cand[idx.rows[cand] + 5 < hic]
        pf = np.sort(np.random.default_rng(1).choice(cand, N_PROBE[d], replace=False))
        rows = idx.rows[pf]
        Pq = pd.DataFrame({'store_frame': pf, 'row': rows, 'obs': idx.obs[pf], 'frame_idx': ann.frame_idx.values[rows],
                           'frame_path': ann.frame_path.values[rows], 'n_fg': idx.nfg[pf].astype(np.int64),
                           'nrow': rows + 5, 'nfi': ann.frame_idx.values[rows + 5]})
        assert (ann.observation_id.values[rows + 5] == Pq.obs.values).all() and ((Pq.nfi - Pq.frame_idx) == 5).all()
        Pq['video'] = Pq.obs.map(vmap)
        units = sorted(vmap)
        Pq['half'] = Pq.obs.map({u: i % 2 for i, u in enumerate(units)}).astype(int)
        pa, gs = mt.all_pos(idx)
        ppos = np.concatenate([pa[gs[f]:gs[f] + idx.nfg[f]] for f in pf]).astype(np.int16)
        ppos.tofile(od / 'probe_pos448.i16')
        F.to_parquet(od / 'train.parquet')
        Pq.to_parquet(od / 'probe.parquet')
        vids = {}
        for o in units:
            g = F[F.obs == o]
            q = Pq[Pq.obs == o]
            ks = set(g.frame_idx) | set(g.pfi) | set(g.nfi[g.next_ok]) | set(q.nfi)
            vids[o] = {'video': vmap[o], 'ks': sorted(int(k) for k in ks),
                       'check_k': int(g.frame_idx.iat[0]), 'check_path': g.frame_path.iat[0]}
        (od / 'videos.json').write_text(json.dumps(vids, indent=1))
        log(f'{d}: {len(F):,} train frames ({int(F.next_ok.sum()):,} with t+5), {len(Pq):,} probe frames '
            f'({int(Pq.n_fg.sum()):,} tokens) from {len(cand):,} candidates, {len(vids)} videos, '
            f'{sum(len(v["ks"]) for v in vids.values()):,} frames to decode; aug elements {np.bincount(F.g, minlength=8).tolist()}')
        # store tokens + d5 prev of the train and probe frames (the C1 input)
        md = mt.D5Matcher(idx, d)
        for name, fr, pos_ref in (('train', F.store_frame.values, p448), ('probe', pf, ppos)):
            tok, pos, lens = idx.load(fr)
            assert (pos == pos_ref).all() and (lens == idx.nfg[fr]).all()
            prev, _, ok = md.fetch(fr)
            np.save(ob / f'{name}_tok.npy', tok)
            np.save(ob / f'{name}_prev.npy', prev)
            log(f'  {d} {name}: {len(tok):,} tokens, {int((~ok).sum())} without a d5 counterpart (zero change) '
                f'[{time.time() - t0:.0f}s]')
        (od / 'plan.json').write_text(json.dumps({'n_train': len(F), 'n_train_tokens': int(l448.sum()),
                                                  'n_next_ok': int(F.next_ok.sum()), 'n_probe': len(Pq),
                                                  'n_probe_tokens': int(Pq.n_fg.sum()), 'n_videos': len(vids),
                                                  'aug_counts': np.bincount(F.g, minlength=8).tolist(),
                                                  'elapsed_s': round(time.time() - t0, 1)}, indent=1))


# ---------------------------------------------------------------------------------------------- encode (GPU)
def _decode(job):
    """Decode the needed 5 fps frames of one standardised video locally (get_frames.py chain + select) -> JPGs."""
    o, v, stage = job
    dst = Path(stage) / 'frames' / o
    dst.mkdir(parents=True, exist_ok=True)
    tmp = dst / 'tmp'
    tmp.mkdir()
    ks = v['ks']
    t0 = time.time()
    loc = Path(stage) / f'{o}{Path(v["video"]).suffix}'
    shutil.copyfile(REPO / v['video'], loc)
    sel = '+'.join(f'eq(n\\,{k})' for k in ks)
    subprocess.run([FFMPEG, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(loc), '-vf',
                    f'fps=5,select={sel},format=rgb24', '-vsync', '0', '-q:v', '2', '-start_number', '0',
                    str(tmp / '%06d.jpg')], check=True)
    loc.unlink()
    files = sorted(tmp.glob('*.jpg'))
    if len(files) != len(ks):
        raise RuntimeError(f'{o}: decoded {len(files)} of {len(ks)} frames')
    for f, k in zip(files, ks):
        f.rename(dst / f'{k:06d}.jpg')
    tmp.rmdir()
    same = (REPO / 'dataset' / v['check_path']).read_bytes() == (dst / f'{v["check_k"]:06d}.jpg').read_bytes()
    return o, {'n': len(ks), 's': round(time.time() - t0, 1), 'bytes_equal_dataset_jpg': bool(same)}


def cmd_encode(args):
    import torch
    from multiprocessing import Pool

    from PIL import Image

    from src.eci.foreground import FgEncoder, encode_batch
    from src.eci.priors import d4_apply, d4_pos_map
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device('cuda')
    stage = stage_dir()
    d = args.domain
    od = OUT / d / 'plan'
    F = pd.read_parquet(od / 'train.parquet')
    Pq = pd.read_parquet(od / 'probe.parquet')
    vids = json.loads((od / 'videos.json').read_text())
    p448 = np.fromfile(RP / d / 'train' / 'pos448.i16', np.int16).astype(np.int64)
    ppos = np.fromfile(od / 'probe_pos448.i16', np.int16).astype(np.int64)
    st = np.r_[0, np.cumsum(F.n_fg.values)]
    sp = np.r_[0, np.cumsum(Pq.n_fg.values)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        dec = dict(pool.imap_unordered(_decode, [(o, v, str(stage)) for o, v in vids.items()]))
    nb = sum(v['bytes_equal_dataset_jpg'] for v in dec.values())
    log(f'{d}: decoded {sum(v["n"] for v in dec.values()):,} frames of {len(dec)} videos in {time.time() - t0:.0f}s; '
        f'check frame byte-identical to the dataset JPG in {nb}/{len(dec)} videos')
    # items: unique (obs, k, g) to encode; consumers: (kind, row, positions)
    maps = [d4_pos_map(g) for g in range(8)]
    items, cons = {}, []

    def add(key, kind, i, pos):
        j = items.setdefault(key, len(items))
        cons.append((j, kind, i, pos))
    first = F.groupby('obs').head(args.n_check).index
    for i, r in enumerate(F.itertuples(index=False)):
        pos = p448[st[i]:st[i + 1]]
        mp = maps[r.g][pos]
        add((r.obs, int(r.frame_idx), int(r.g)), 'aug', i, mp)
        add((r.obs, int(r.pfi), int(r.g)), 'augp', i, mp)
        if r.next_ok:
            add((r.obs, int(r.nfi), 0), 'next', i, pos)
    for i in first:
        add((F.obs.iat[i], int(F.frame_idx.iat[i]), 0), 'chk', i, p448[st[i]:st[i + 1]])
    for i, r in enumerate(Pq.itertuples(index=False)):
        add((r.obs, int(r.nfi), 0), 'pnext', i, ppos[sp[i]:sp[i + 1]])
    keys = sorted(items, key=items.get)
    by_item = [[] for _ in keys]
    for j, kind, i, pos in cons:
        by_item[j].append((kind, i, pos))
    n_tr, n_pr = int(st[-1]), int(sp[-1])
    arrs = {k: np.lib.format.open_memmap(stage / f'{k}.npy', 'w+', np.float16, (n_tr, D)) for k in ('aug', 'augp', 'next')}
    arrs['pnext'] = np.lib.format.open_memmap(stage / 'pnext.npy', 'w+', np.float16, (n_pr, D))
    arrs['next'][:] = 0
    chk = {}
    enc = FgEncoder('dinov2_base', dev)
    proc = enc.processor

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(keys)

        def __getitem__(self, j):
            o, k, g = keys[j]
            a = np.asarray(Image.open(stage / 'frames' / o / f'{k:06d}.jpg').convert('RGB'))
            assert a.shape == (512, 512, 3), a.shape
            return j, proc(images=Image.fromarray(d4_apply(a, g)), return_tensors='pt')['pixel_values'][0]

    dl = torch.utils.data.DataLoader(DS(), batch_size=args.bs, num_workers=args.workers, shuffle=False,
                                     pin_memory=True, prefetch_factor=4)
    t1, gpu_s, done = time.time(), 0.0, 0
    sto = {k: st if k != 'pnext' else sp for k in ('aug', 'augp', 'next', 'pnext')}
    for bi, (jj, pix) in enumerate(dl):
        torch.cuda.synchronize()
        tt = time.time()
        tok = encode_batch(enc.model, pix, dev)
        torch.cuda.synchronize()
        gpu_s += time.time() - tt
        tok = tok.cpu().numpy()
        for q, j in enumerate(jj.tolist()):
            for kind, i, pos in by_item[j]:
                t = tok[q, pos]
                if kind == 'chk':
                    chk[i] = t
                else:
                    arrs[kind][sto[kind][i]:sto[kind][i + 1]] = t
        done += len(jj)
        if bi % 200 == 0:
            log(f'  batch {bi}: {done:,}/{len(keys):,} frames [{time.time() - t1:.0f}s, GPU {gpu_s:.0f}s]')
    for a in arrs.values():
        a.flush()
    # identity check: re-encoded frame t vs the store token
    base = np.load(OUT / d / 'base' / 'train_tok.npy', mmap_mode='r')
    ci = sorted(chk)
    a = np.concatenate([chk[i] for i in ci]).astype(np.float32)
    b = np.concatenate([np.asarray(base[st[i]:st[i + 1]]) for i in ci]).astype(np.float32)
    rel = float(np.linalg.norm(a - b, axis=1).mean() / np.linalg.norm(b, axis=1).mean())
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    # how much the static token changes under the symmetry (aug token at the mapped patch vs the store token)
    aug = np.asarray(arrs['aug'][:200_000], np.float32)
    bb = np.asarray(base[:200_000], np.float32)
    cos_aug = (aug * bb).sum(1) / (np.linalg.norm(aug, axis=1) * np.linalg.norm(bb, axis=1) + 1e-9)
    info = {'domain': d, 'gpu': torch.cuda.get_device_name(0), 'host': os.uname().nodename,
            'n_frames_encoded': len(keys), 'gpu_s': round(gpu_s, 1), 'wall_s': round(time.time() - t0, 1),
            'decode': {'videos': len(dec), 'frames': int(sum(v['n'] for v in dec.values())),
                       'bytes_equal_dataset_jpg_videos': int(nb), 'sum_s': round(sum(v['s'] for v in dec.values()), 1)},
            'identity_check': {'n_frames': len(ci), 'n_tokens': int(len(a)), 'max_abs': float(np.abs(a - b).max()),
                               'rel_l2': rel, 'cos_min': float(cos.min()), 'cos_mean': float(cos.mean()),
                               'exact_token_frac': float((a == b).all(1).mean())},
            'aug_vs_store_static_cos': {'mean': float(cos_aug.mean()), 'median': float(np.median(cos_aug)),
                                        'p05': float(np.percentile(cos_aug, 5))},
            'zero_next_tokens': int((~np.repeat(F.next_ok.values, F.n_fg.values)).sum())}
    log(f'{d} encode: {json.dumps(info)}')
    oe = OUT / d / 'enc'
    oe.mkdir(parents=True, exist_ok=True)
    arrs.clear()
    del aug, base
    for k in ('aug', 'augp', 'next', 'pnext'):
        shutil.copyfile(stage / f'{k}.npy', oe / f'{k}.npy')
        (stage / f'{k}.npy').unlink()
    (oe / 'encode.json').write_text(json.dumps(info, indent=1))
    shutil.rmtree(stage / 'frames')


# ---------------------------------------------------------------------------------------------- train data
def load_data(d, dev, stage, need_aug=True, need_next=True):
    import torch
    from src.eci.priors import PriorData
    F = pd.read_parquet(OUT / d / 'plan' / 'train.parquet')
    t0 = time.time()

    def ld(p):
        q = copy_local(p, stage)
        a = np.load(q)
        q.unlink()
        return a

    def pair(a, b):
        n = len(a)
        X = torch.empty((n, 2 * D), dtype=torch.float16, device=dev)
        for s in range(0, n, 1_000_000):
            t = torch.from_numpy(np.ascontiguousarray(a[s:s + 1_000_000])).to(dev)
            X[s:s + len(t), :D] = t
            X[s:s + len(t), D:] = (t.float() - torch.from_numpy(np.ascontiguousarray(b[s:s + 1_000_000])).to(dev).float()).half()
        return X
    tok = ld(OUT / d / 'base' / 'train_tok.npy')
    X = pair(tok, ld(OUT / d / 'base' / 'train_prev.npy'))
    Xa = pair(ld(OUT / d / 'enc' / 'aug.npy'), ld(OUT / d / 'enc' / 'augp.npy')) if need_aug else None
    Dr = ok = None
    if need_next:
        nx = ld(OUT / d / 'enc' / 'next.npy')
        Dr = torch.empty((len(tok), D), dtype=torch.float16, device=dev)
        for s in range(0, len(tok), 1_000_000):
            Dr[s:s + 1_000_000] = (torch.from_numpy(np.ascontiguousarray(nx[s:s + 1_000_000])).to(dev).float()
                                   - torch.from_numpy(np.ascontiguousarray(tok[s:s + 1_000_000])).to(dev).float()).half()
        ok = torch.from_numpy(np.repeat(F.next_ok.values, F.n_fg.values)).to(dev)
    assert len(tok) == int(F.n_fg.sum())
    log(f'{d}: train data on GPU: {len(tok):,} tokens, aug {Xa is not None}, next {Dr is not None} '
        f'[{time.time() - t0:.0f}s]')
    return PriorData(X, Xa, Dr, ok)


# ---------------------------------------------------------------------------------------------- warmup (GPU)
def cmd_warmup(args):
    import torch
    from src.eci.priors import measure_terms, train_prior
    torch.backends.cuda.matmul.allow_tf32 = True  # as motion_t3 / res_pool train
    dev = torch.device('cuda')
    stage = stage_dir()
    d = args.domain
    data = load_data(d, dev, stage)
    sae, norm, _, _, hist = train_prior(data, 0, WARM_STEPS, 0.0, 0.0, dev, log, total=TOTAL_STEPS)
    m = measure_terms(sae, norm, data, dev)
    lam = SHARE * m['L_rec'] / m['L_inv_raw']
    mu = SHARE * m['L_rec'] / m['L_pred_raw_ridge']
    W = {'domain': d, 'rule': f'term = {SHARE} x L_rec at step {WARM_STEPS} of the {TOTAL_STEPS}-step schedule of '
                              f'the plain model (seed 0), on held batches of the training pool',
         'lambda': lam, 'mu': mu, 'warmup_measure': m, 'warmup_history': hist,
         'gpu': torch.cuda.get_device_name(0)}
    (OUT / d / 'weights.json').write_text(json.dumps(W, indent=1))
    log(f'{d}: lambda {lam:.4g}, mu {mu:.4g}; {json.dumps(m)}')


# ---------------------------------------------------------------------------------------------- train (GPU)
def cmd_train(args):
    import torch
    from src.eci.priors import measure_terms, train_prior
    from src.eci.sae import load_sae
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device('cuda')
    stage = stage_dir()
    d = args.domain
    W = json.loads((OUT / d / 'weights.json').read_text())
    data = load_data(d, dev, stage)
    for arm in args.arms:
        ul, um = ARMS[arm]
        lam, mu = W['lambda'] * ul, W['mu'] * um
        seeds = [0] if arm == 'C1r' else args.seeds
        for seed in seeds:
            out = OUT / d / 'sae' / f'{arm}_s{seed}'
            if arm == 'C1':  # measure the reused res_pool checkpoints only
                sae, norm, _ = load_sae(sae_path(d, 'C1', seed), dev)
                out.mkdir(parents=True, exist_ok=True)
            else:
                if (out / 'measure.json').exists() and not args.force:
                    log(f'[SKIP] {out}')
                    continue
                sae, norm, _, _, _ = train_prior(data, seed, args.steps, lam, mu, dev, log, out=out,
                                                 extra={'config': 'C1+priors', 'arm': arm, 'domain': d, 'motion_delta': 5,
                                                        'blocks': [D, D], 'train_frames': 'res_pool train'})
            m = measure_terms(sae, norm, data, dev)
            if arm == 'C1r':  # identity of the training loop with motion_t3.train_one (res_pool c448_s0)
                ref, rn, _ = load_sae(sae_path(d, 'C1', seed), dev)
                m['repro_vs_res_pool_c448'] = {k: float((sae.state_dict()[k].float() - ref.state_dict()[k].float()).abs().max())
                                               for k in ('W_enc', 'W_dec', 'b_enc', 'b_dec', 'threshold')}
                m['repro_norm_scale_max_abs'] = float((norm.scale - rn.scale).abs().max())
            (out / 'measure.json').write_text(json.dumps({'arm': arm, 'seed': seed, 'lambda': lam, 'mu': mu, **m}, indent=1))
            log(f'{d} {arm} s{seed}: {json.dumps(m)}')
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- evaluate (GPU)
def cmd_evaluate(args):
    import torch
    import motion_t3 as mt
    import multiscale_sae as ms
    import readouts as rd
    import res_pool as rp
    import sae_levers as sl
    import spatial_sae_pilot as ssp
    from src.eci.priors import code_r2, entropy_rows, href, neuron_r2, ridge_head
    from src.eci.sae import load_sae
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device('cuda')
    stage = stage_dir()
    d = args.domain
    t0 = time.time()
    E, p448e, l448e = rp.table(d, 'eval')
    idx = ssp.StoreIndex(d)
    fr = E.store_frame.values
    tok, pos, lens = ms.stage(idx, fr, stage / 'eval')
    assert (lens == l448e).all() and (pos == p448e).all()
    prev = mt.stage_prev5(idx, d, fr, stage / 'prev.npy')
    keys = [(a, s) for a in args.arms for s in args.seeds]
    saes = {k: load_sae(sae_path(d, *k), dev)[:2] for k in keys}
    nF, m = len(lens), 4096
    MX = {k: np.empty((nF, m), np.float16) for k in keys}
    CE = {k: np.empty((nF, m), np.uint8) for k in keys}
    z64 = lambda q: torch.zeros(q, dtype=torch.float64, device=dev)  # noqa: E731
    ST = {k: {'s1': z64(2 * D), 's2': z64(2 * D), 'sse': z64(2 * D), 'fire': z64(m), 'l0': 0.0, 'ntok': 0} for k in keys}
    st = np.r_[0, np.cumsum(lens)]
    cell_all = torch.from_numpy(((pos // 32) // 4 * 8 + (pos % 32) // 4).astype(np.int16)).to(dev)
    with torch.no_grad():
        for ci, (f0, f1) in enumerate(rp.chunks_448(lens, max_cells=40000, max_tok=60_000)):
            T = torch.from_numpy(np.ascontiguousarray(tok[st[f0]:st[f1]])).to(dev).float()
            xr = torch.cat([T, T - torch.from_numpy(np.ascontiguousarray(prev[st[f0]:st[f1]])).to(dev).float()], 1)
            L = torch.from_numpy(lens[f0:f1].astype(np.int64)).to(dev)
            B = f1 - f0
            fi = torch.repeat_interleave(torch.arange(B, device=dev), L)
            cell = cell_all[st[f0]:st[f1]].long()
            for k in keys:
                sae, norm = saes[k]
                x = norm(xr)
                z = sae.encode(x, mode='threshold').float()
                S = ST[k]
                S['s1'] += x.double().sum(0)
                S['s2'] += x.double().pow(2).sum(0)
                S['sse'] += (x - sae.decode(z)).double().pow(2).sum(0)
                act = z > 0
                S['fire'] += act.sum(0).double()
                S['l0'] += float(act.sum())
                S['ntok'] += int(len(x))
                mx = torch.zeros(B, m, device=dev).index_reduce_(0, fi, z, 'amax', include_self=True)
                hit = act & (z == mx[fi])
                cv = torch.where(hit, cell[:, None].float(), torch.full((1, 1), 255.0, device=dev))
                ce = torch.full((B, m), 255.0, device=dev).index_reduce_(0, fi, cv, 'amin', include_self=True)
                MX[k][f0:f1] = mx.half().cpu().numpy()
                CE[k][f0:f1] = ce.to(torch.uint8).cpu().numpy()
                del z, x, mx, hit, cv, ce
            if ci % 500 == 0:
                log(f'  chunk {ci}: frames {f1:,}/{nF:,} [{time.time() - t0:.0f}s]')
    del tok, prev
    shutil.rmtree(stage / 'eval')
    (stage / 'prev.npy').unlink()
    log(f'{d}: codes of {len(keys)} SAEs on {nF:,} frames [{time.time() - t0:.0f}s]')
    # ------------------------------------------------ probe tokens (held-out frames of the train videos)
    Pq = pd.read_parquet(OUT / d / 'plan' / 'probe.parquet')
    ptok = np.load(copy_local(OUT / d / 'base' / 'probe_tok.npy', stage))
    pprev = np.load(copy_local(OUT / d / 'base' / 'probe_prev.npy', stage))
    pnext = np.load(copy_local(OUT / d / 'enc' / 'pnext.npy', stage))
    PT = torch.from_numpy(ptok).to(dev).float()
    PX = torch.cat([PT, PT - torch.from_numpy(pprev).to(dev).float()], 1)
    PD = torch.from_numpy(pnext).to(dev).float() - PT
    phalf = torch.from_numpy(np.repeat(Pq.half.values, Pq.n_fg.values).astype(bool)).to(dev)
    # ------------------------------------------------ labels / truth (res_pool tables)
    lab = pd.read_parquet(RP / d / 'labels.parquet')
    assert (lab.row.values == E.row.values).all()
    half = np.load(RP / d / 'half.npy')
    V = np.load(RP / d / 'valid.npz')
    valid = {b: V[b] for b in LABELS[d]}
    rank = np.argsort(np.argsort(l448e + np.random.default_rng(0).random(len(l448e)) * 1e-3))
    decile = np.minimum(rank * 10 // len(l448e), 9)
    Yt = torch.from_numpy(lab[LABELS[d]].values.astype(bool)).to(dev)
    gt = rd.mice_truth(stage) if d == 'mice' else rd.ants_truth()
    K = int(np.ceil(TOP_FRAC * nF))
    cells_np = ((pos // 32) // 4 * 8 + (pos % 32) // 4).astype(np.int64)
    HREF = href(cells_np, K)
    for dd in ('eval', 'comp', 'codes_best'):
        (OUT / d / dd).mkdir(parents=True, exist_ok=True)
    for k in keys:
        arm, seed = k
        t1 = time.time()
        sae, norm = saes[k]
        S = ST[k]
        tss = (S['s2'] - S['s1'] ** 2 / S['ntok'])
        fv = lambda a, b: float(1 - S['sse'][a:b].sum() / tss[a:b].sum())  # noqa: E731
        fve = {'fve': fv(0, 2 * D), 'fve_static': fv(0, D), 'fve_change': fv(D, 2 * D),
               'l0_per_token': S['l0'] / S['ntok'], 'n_tokens': S['ntok']}
        fire_tok = S['fire'].cpu().numpy()
        Xt = torch.from_numpy(MX[k]).to(dev)
        fire_frac = (Xt > 0).float().mean(0).cpu().numpy()
        # position-boundness: argmax cells of the top-1% frames
        tv, ti = Xt.float().topk(K, dim=0)  # (K, m)
        cg = torch.from_numpy(CE[k]).to(dev).gather(0, ti).long()  # (K, m)
        okc = (tv > 0) & (cg < 64)
        H = torch.zeros(m, 65, device=dev).scatter_add_(1, torch.where(okc, cg, 64).T, torch.ones_like(tv).T)[:, :64]
        conc = (1 - entropy_rows(H) / HREF).cpu().numpy()
        del tv, ti, cg, okc, H
        # predictive: single-neuron R^2 of d+ (normalised static units) on the probe tokens
        with torch.no_grad():
            Zp = sae.encode(norm(PX), mode='threshold').float()
            Yp = PD * norm.scale[:D]
            Wx, bx = ridge_head(norm(PX), Yp)
            Yr = Yp - (norm(PX) @ Wx + bx)  # beyond-input residual
            r2 = neuron_r2(Zp, Yp, phalf)
            r2r = neuron_r2(Zp, Yr, phalf)
            cr2 = {str(p): code_r2(Zp[:, :p].contiguous(), Yp, phalf) for p in PREFIXES}
            cr2r = {str(p): code_r2(Zp[:, :p].contiguous(), Yr, phalf) for p in PREFIXES}
            xr2 = code_r2(norm(PX), Yp, phalf)
        del Zp
        np.savez_compressed(OUT / d / 'comp' / f'{arm}_s{seed}.npz', fire_frac=fire_frac, fire_tok=fire_tok, conc=conc,
                            r2=r2, r2_resid=r2r)
        if seed == 0:  # frame codes of random eligible neurons for the qualitative browse (20 of all, 10 of < 512)
            el = np.flatnonzero(fire_frac >= MIN_FIRE)
            rg = np.random.default_rng(7)
            br = np.r_[rg.choice(el, min(20, len(el)), replace=False), rg.choice(el[el < 512], min(10, int((el < 512).sum())), replace=False)]
            np.savez_compressed(OUT / d / 'codes_best' / f'{arm}_s0_browse.npz', neurons=br, codes=MX[k][:, br])
        for p in PREFIXES:
            Xp = Xt[:, :p].contiguous()
            res, _ = sl.score_model(Xp, Yt, LABELS[d], valid, half, decile, dev)
            js = sorted({q[s]['neuron'] for b in LABELS[d] for q in res[b]['dirs'] for s in ('auc', 'ap', 'top1')})
            cb = MX[k][:, js]
            vid = rp.video_level(d, lambda j: cb[:, js.index(j)], res, lab, half, l448e, gt)
            el = fire_frac[:p] >= MIN_FIRE
            comp = {'alive': int((fire_tok[:p] > 0).sum()), 'eligible': int(el.sum()),
                    'position_bound_share': float((conc[:p][el] > PLACE_CONC).mean()) if el.any() else None,
                    'conc_median': float(np.nanmedian(conc[:p][el])) if el.any() else None,
                    'r2_median': float(np.median(r2[:p][el])) if el.any() else None,
                    'r2_resid_median': float(np.median(r2r[:p][el])) if el.any() else None,
                    'code_r2_dplus': cr2[str(p)], 'code_r2_dplus_beyond_input': cr2r[str(p)],
                    'input_r2_dplus': xr2, 'K_top': K, 'H_ref': HREF}
            e = {'arm': arm, 'seed': seed, 'prefix': p, 'ckpt': str(sae_path(d, arm, seed)), **fve, 'comp': comp,
                 'labels': res, 'video': vid}
            if arm == 'C1' and p == 4096:
                ref = json.loads((RP / d / 'eval' / f'c448_max_s{seed}.json').read_text())['labels']
                e['repro_max_abs_diff_vs_res_pool_c448'] = {q: max(abs(ref[b][q] - res[b][q]) for b in LABELS[d])
                                                            for q in ('cf_auroc', 'cf_ap', 'cf_top1')}
            if arm == 'C1dep' and p == 4096:
                ref = json.loads((RP / d / 'eval' / f'c448dep_max_s{seed}.json').read_text())['labels']
                e['repro_max_abs_diff_vs_res_pool_c448dep'] = {q: max(abs(ref[b][q] - res[b][q]) for b in LABELS[d])
                                                               for q in ('cf_auroc', 'cf_ap', 'cf_top1')}
            key = f'{arm}_s{seed}_m{p}'
            (OUT / d / 'eval' / f'{key}.json').write_text(json.dumps(e, indent=1, default=float))
            np.savez_compressed(OUT / d / 'codes_best' / f'{key}.npz', neurons=np.array(js), codes=cb)
            f = lambda b: (f"{b} {res[b]['cf_auroc']:.3f}/{res[b]['cf_ap']:.4f}/{res[b]['cf_top1']:.3f} "  # noqa: E731
                           f"r {vid[b]['presence_mean']['video_r_rate']:.2f}/{vid[b]['presence_mean|fgctrl']['video_r_rate']:.2f}")
            log(f'{d} {key}: FVE {fve["fve"]:.4f} elig {comp["eligible"]} posb {comp["position_bound_share"]:.3f} '
                f'codeR2 {cr2[str(p)]:.3f} | ' + ' | '.join(f(b) for b in LABELS[d])
                + (f" | repro {e.get('repro_max_abs_diff_vs_res_pool_c448') or e.get('repro_max_abs_diff_vs_res_pool_c448dep')}"
                   if p == 4096 and arm in ('C1', 'C1dep') else ''))
        del Xt
        MX[k] = CE[k] = None
        torch.cuda.empty_cache()
        log(f'  {arm} s{seed} scored [{time.time() - t1:.0f}s]')


# ---------------------------------------------------------------------------------------------- sheets (CPU)
def _decode_frames(job):
    """Decode the given 5 fps frame indices of one standardised video locally -> {k: (512, 512, 3) uint8}."""
    from PIL import Image
    video, ks, stage = job
    ks = sorted(set(int(k) for k in ks))
    tmp = Path(stage) / f'sh_{Path(video).stem}'
    tmp.mkdir(parents=True, exist_ok=True)
    sel = '+'.join(f'eq(n\\,{k})' for k in ks)
    subprocess.run([FFMPEG, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(REPO / video), '-vf',
                    f'fps=5,select={sel},format=rgb24', '-vsync', '0', '-q:v', '2', '-start_number', '0',
                    str(tmp / '%06d.jpg')], check=True)
    files = sorted(tmp.glob('*.jpg'))
    assert len(files) == len(ks), (video, len(files), len(ks))
    out = {k: np.asarray(Image.open(f).convert('RGB')) for k, f in zip(ks, files)}
    shutil.rmtree(tmp)
    return video, out


def cmd_sheets(args):
    import torch
    from multiprocessing import Pool
    import motion_t3 as mt
    import native_res as nr
    import res_pool as rp
    import spatial_sae_pilot as ssp
    from src.eci.sae import load_sae
    stage = stage_dir()
    torch.set_num_threads(args.workers)
    jobs = []
    for d in args.domains.split(','):
        E, _, _ = rp.table(d, 'eval')
        lab = pd.read_parquet(RP / d / 'labels.parquet')
        half = np.load(RP / d / 'half.npy')
        labs = ['nose_nose', 'nose_tail'] if d == 'mice' else ['groom_any']
        for arm in args.arms:
            evs = {s: json.loads((OUT / d / 'eval' / f'{arm}_s{s}_m4096.json').read_text()) for s in SEEDS}
            for b in labs:
                s = max(SEEDS, key=lambda s: evs[s]['labels'][b]['cf_auroc'])
                e0 = evs[s]['labels'][b]['dirs'][0]
                for sel in ('auc', 'top1'):
                    j = e0[sel]['neuron']
                    z = np.load(OUT / d / 'codes_best' / f'{arm}_s{s}_m4096.npz')
                    x = z['codes'][:, list(z['neurons']).index(j)].astype(np.float32)
                    pick = nr.pick_frames(x, half == e0['test_half'], E.obs.values)
                    jobs.append({'domain': d, 'arm': arm, 'label': b, 'seed': s, 'sel': sel, 'neuron': j, 'pick': pick,
                                 'x': x, 'test': e0[sel], 'kind': 'best'})
        if d == 'mice' and args.browse_arm:  # random eligible neurons of the browse arm (seed 0), 8 top frames each
            if args.browse_arm == 'auto':  # the best arm = highest 3-seed mean mice cf AUROC (both labels) at 4096
                sc = {a: np.mean([json.loads((OUT / d / 'eval' / f'{a}_s{s}_m4096.json').read_text())['labels'][b]['cf_auroc']
                                  for s in SEEDS for b in LABELS[d]]) for a in ('C1', 'INV', 'PRED', 'INVPRED')}
                args.browse_arm = max(sc, key=sc.get)
                log(f'browse arm = {args.browse_arm} (mean mice cf AUROC {json.dumps(sc)})')
            z = np.load(OUT / d / 'codes_best' / f'{args.browse_arm}_s0_browse.npz')
            picks = [nr.pick_frames(z['codes'][:, q].astype(np.float32), np.ones(len(E), bool), E.obs.values, n_show=8)
                     for q in range(len(z['neurons']))]
            jobs.append({'domain': d, 'arm': args.browse_arm, 'kind': 'browse', 'neurons': z['neurons'].tolist(),
                         'picks': picks, 'seed': 0})
    need = {}
    for jb in jobs:
        frames = [i for p in jb['picks'] for i in p] if jb['kind'] == 'browse' else jb['pick']
        for i in frames:
            need.setdefault(jb['domain'], set()).add(i)
    # decode t and t-5 of every needed frame
    imgs = {}
    for d, fs in need.items():
        E, _, _ = rp.table(d, 'eval')
        dj = {}
        for i in fs:
            v = str(video_file(E.frame_path.iat[i]).relative_to(REPO))
            dj.setdefault(v, set()).update({int(E.frame_idx.iat[i]), int(E.pfi.iat[i])})
        t0 = time.time()
        with Pool(args.workers) as pool:
            for v, out in pool.imap_unordered(_decode_frames, [(v, ks, str(stage)) for v, ks in dj.items()]):
                for k, a in out.items():
                    imgs[(d, v, k)] = a
        log(f'{d}: decoded {sum(len(k) for k in dj.values())} frames of {len(dj)} videos [{time.time() - t0:.0f}s]')
    # patch argmax of the neuron in each shown frame (CPU SAE on the store tokens of those frames)
    index, IDX = [], {}
    for jb in jobs:
        d = jb['domain']
        E, _, _ = rp.table(d, 'eval')
        lab = pd.read_parquet(RP / d / 'labels.parquet')
        if d not in IDX:
            IDX[d] = (ssp.StoreIndex(d),)
            IDX[d] += (mt.D5Matcher(IDX[d][0], d),)
        idx, md = IDX[d]
        sae, norm, _ = load_sae(sae_path(d, jb['arm'], jb['seed']), 'cpu')
        (OUT / d / 'sheets').mkdir(parents=True, exist_ok=True)
        groups = ([(jb['neuron'], jb['pick'], jb['x'], jb['label'])] if jb['kind'] == 'best' else
                  [(n, p, None, None) for n, p in zip(jb['neurons'], jb['picks'])])
        for j, pick, x, b in groups:
            if not pick:
                continue
            fr = E.store_frame.values[pick]
            o = np.argsort(fr)
            tok, pos, lens = idx.load(fr)
            prev, _, _ = md.fetch(fr)
            st = np.r_[0, np.cumsum(lens)]
            items = []
            ordmap = {int(fr[o[r]]): r for r in range(len(fr))}
            for i in pick:
                r = ordmap[int(E.store_frame.values[i])]
                T = torch.from_numpy(tok[st[r]:st[r + 1]]).float()
                with torch.no_grad():
                    zz = sae.encode(norm(torch.cat([T, T - torch.from_numpy(prev[st[r]:st[r + 1]]).float()], 1)))[:, j]
                q = int(zz.argmax())
                v = str(video_file(E.frame_path.iat[i]).relative_to(REPO))
                yv = bool(lab[b].iat[i]) if b else False
                tags = '' if b else ' '.join(f'{c}' for c in LABELS[d] if bool(lab[c].iat[i]))
                items.append((imgs[(d, v, int(E.frame_idx.iat[i]))], imgs[(d, v, int(E.pfi.iat[i]))], 32,
                              int(pos[st[r] + q]), yv,
                              f'{E.obs.iat[i]} f{int(E.frame_idx.iat[i])} ' + (f'{"POS" if yv else "neg"} max={x[i]:.2f}' if b
                                                                              else f'[{tags}] act {int((zz > 0).sum())}/{len(zz)}')))
            if jb['kind'] == 'best':
                t = jb['test']
                fn = f'best_{jb["arm"]}_s{jb["seed"]}__{b}__{jb["sel"]}_n{j}.jpg'
                title = (f'{d} {jb["arm"]} seed {jb["seed"]} {b} neuron {j} (selected on half 0 by '
                         f'{jb["sel"]}; test AUROC {t["test_auc"]:.3f} AP {t["test_ap"]:.3f} top1 {t["test_top1"]:.3f}); '
                         f'top test-half frames <= 2 per video; green = label positive')
            else:
                fn = f'browse_{jb["arm"]}_s0_n{j}.jpg'
                title = f'{d} {jb["arm"]} seed 0 random eligible neuron {j}: top eval frames, <= 2 per video (annotated labels in [])'
            rp.render(OUT / d / 'sheets' / fn, items, title)
            index.append({'domain': d, 'arm': jb['arm'], 'kind': jb['kind'], 'neuron': int(j), 'label': b,
                          'seed': jb['seed'], 'sel': jb.get('sel'), 'file': fn, 'n_shown': len(items),
                          'n_pos_shown': int(sum(it[4] for it in items)) if b else None})
            log(f'sheet {d}/{fn}: {len(items)} frames' + (f', {index[-1]["n_pos_shown"]} label-positive' if b else ''))
    for d in need:
        (OUT / d / 'sheets' / 'index.json').write_text(json.dumps([e for e in index if e['domain'] == d], indent=1))


# ---------------------------------------------------------------------------------------------- summary (login safe)
FMETS = ('cf_auroc', 'cf_ap', 'cf_top1', 'cf_size_ctrl_auroc')


def ms_(v):
    v = np.asarray(v, float)
    return {'mean': float(np.nanmean(v)), 'sd': float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
            'per_seed': [float(x) for x in v]}


def collect():
    S = {}
    for d in DOMAINS:
        Dd = {}
        tau = None
        c0 = OUT / d / 'comp' / 'C1_s0.npz'
        if c0.exists():
            c = np.load(c0)
            el = c['fire_frac'] >= MIN_FIRE
            tau = float(np.percentile(c['r2'][el], 90))
            tau_r = float(np.percentile(c['r2_resid'][el], 90))
        for arm in EVAL_ARMS:
            for p in PREFIXES:
                ps = [OUT / d / 'eval' / f'{arm}_s{s}_m{p}.json' for s in SEEDS]
                if not all(x.exists() for x in ps):
                    continue
                E = [json.loads(x.read_text()) for x in ps]
                c = {q: ms_([e[q] for e in E]) for q in ('fve', 'fve_static', 'fve_change', 'l0_per_token')}
                comp = {q: ms_([e['comp'][q] for e in E]) for q in E[0]['comp'] if q not in ('K_top', 'H_ref')}
                if tau is not None:
                    sh, shr = [], []
                    for s in SEEDS:
                        z = np.load(OUT / d / 'comp' / f'{arm}_s{s}.npz')
                        el = z['fire_frac'][:p] >= MIN_FIRE
                        sh.append(float((z['r2'][:p][el] > tau).mean()))
                        shr.append(float((z['r2_resid'][:p][el] > tau_r).mean()))
                    comp['predictive_share'] = ms_(sh)
                    comp['predictive_share_beyond_input'] = ms_(shr)
                    comp['tau_r2'], comp['tau_r2_resid'] = tau, tau_r
                c['comp'] = comp
                for k in ('repro_max_abs_diff_vs_res_pool_c448', 'repro_max_abs_diff_vs_res_pool_c448dep'):
                    if k in E[0]:
                        c[k] = {q: max(e[k][q] for e in E) for q in ('cf_auroc', 'cf_ap', 'cf_top1')}
                for b in LABELS[d]:
                    c[b] = {q: ms_([e['labels'][b][q] for e in E]) for q in FMETS}
                    c[b]['base_rate'] = E[0]['labels'][b]['base_rate']
                    for w in ('presence_mean', 'presence_mean|fgctrl'):
                        Vv = [e['video'][b][w] for e in E]
                        c[b][w] = {q: ms_([v[q] for v in Vv]) for q in Vv[0] if q != 'effects' and not isinstance(Vv[0][q], dict)}
                        if 'effects' in Vv[0]:
                            c[b][w]['effects'] = {sw: {'dz': ms_([v['effects'][sw]['dz'] for v in Vv]),
                                                       'truth_dz': Vv[0]['effects'][sw]['truth_dz'],
                                                       'sign_agree_seeds': int(sum(v['effects'][sw]['sign_agree'] for v in Vv))}
                                                  for sw in Vv[0]['effects']}
                Dd[f'{arm}|{p}'] = c
        S[d] = Dd
    return S


def decide(S):
    import res_pool as rp
    R = {'primary': {}, 'keep': {}, 'width': {}}
    best = None
    for key, c in S['mice'].items():
        for b in LABELS['mice']:
            v = c[b]['cf_top1']['mean']
            if best is None or v > best[0]:
                best = (v, key, b)
    R['primary'] = {'best_mice_cf_top1_mean': best[0], 'at': best[1], 'label': best[2], 'target': 0.40,
                    'MET': bool(best[0] >= 0.40)}
    for arm in ('INV', 'PRED', 'INVPRED'):
        R['keep'][arm] = {}
        for p in PREFIXES:
            cells = {}
            for d in DOMAINS:
                if f'{arm}|{p}' in S[d] and f'C1|{p}' in S[d]:
                    cells[d] = rp.res_rule(S, d, f'{arm}|{p}', f'C1|{p}')
            R['keep'][arm][str(p)] = {'domains': cells, 'KEEP': bool(len(cells) == 2 and all(v['WIN'] for v in cells.values()))}
    kept = [a for a in ('INV', 'PRED', 'INVPRED') if R['keep'][a]['4096']['KEEP']]
    R['kept_arms_at_4096'] = kept
    for arm in ['C1'] + kept:
        R['width'][arm] = {}
        for p in (512, 1024):
            per = {f'{d}|{b}': S[d][f'{arm}|{p}'][b]['cf_auroc']['mean'] - S[d][f'{arm}|4096'][b]['cf_auroc']['mean']
                   for d in DOMAINS for b in LABELS[d] if f'{arm}|{p}' in S[d]}
            R['width'][arm][str(p)] = {'d_cf_auroc_vs_4096': per, 'within_0.02_all': bool(per and all(v >= -0.02 for v in per.values()))}
    return R


def costs():
    C = {}
    for d in DOMAINS:
        p = OUT / d / 'enc' / 'encode.json'
        if p.exists():
            C[f'{d}_encode'] = {k: v for k, v in json.loads(p.read_text()).items() if k in ('gpu', 'gpu_s', 'wall_s', 'n_frames_encoded', 'decode')}
        ts = sorted((OUT / d / 'sae').glob('*_s?/train.json')) if (OUT / d / 'sae').exists() else []
        if ts:
            C[f'{d}_train_h'] = sum(json.loads(t.read_text())['train_time_s'] for t in ts) / 3600
    return C


def cmd_summary(args):
    S = collect()
    S['decision'] = decide(S)
    S['weights'] = {d: {k: v for k, v in json.loads((OUT / d / 'weights.json').read_text()).items()
                        if k in ('lambda', 'mu', 'warmup_measure', 'rule')} for d in DOMAINS if (OUT / d / 'weights.json').exists()}
    S['measure'] = {d: {p.parent.name: json.loads(p.read_text()) for p in sorted((OUT / d / 'sae').glob('*/measure.json'))}
                    for d in DOMAINS if (OUT / d / 'sae').exists()}
    S['encode'] = {d: json.loads((OUT / d / 'enc' / 'encode.json').read_text()) for d in DOMAINS
                   if (OUT / d / 'enc' / 'encode.json').exists()}
    S['costs'] = costs()
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    f = lambda x, k=3: f"{x['mean']:.{k}f} ± {x['sd']:.{k}f}"  # noqa: E731
    L = ['# PRIORS: INV / PRED loss terms on the C1 SAE (mean ± sd over 3 seeds)\n']
    for d, w in S['weights'].items():
        L.append(f"- {d}: lambda {w['lambda']:.4g}, mu {w['mu']:.4g} (warm-up L_rec {w['warmup_measure']['L_rec']:.1f}, "
                 f"L_inv {w['warmup_measure']['L_inv_raw']:.4f}, L_pred {w['warmup_measure']['L_pred_raw_ridge']:.4f})")
    for d in DOMAINS:
        L.append(f'\n## {d}\n')
        L.append('| label | arm | prefix | cf AUROC | cf AP | honest top-1% | size-ctrl AUROC | video r rate raw / ctl | '
                 'video r bouts raw / ctl |' + (' pooled d-r raw / ctl |' if d == 'mice' else ''))
        L.append('|---|---|---|---|---|---|---|---|---|' + ('---|' if d == 'mice' else ''))
        for b in LABELS[d]:
            for key, c in S[d].items():
                arm, p = key.split('|')
                x = c[b]
                v, w = x['presence_mean'], x['presence_mean|fgctrl']
                row = (f"| {b} ({x['base_rate']:.4f}) | {arm} | {p} | {f(x['cf_auroc'])} | {f(x['cf_ap'], 4)} | "
                       f"{f(x['cf_top1'])} | {f(x['cf_size_ctrl_auroc'])} | {v['video_r_rate']['mean']:.2f} / "
                       f"{w['video_r_rate']['mean']:.2f} | {v['video_r_bpm']['mean']:.2f} / {w['video_r_bpm']['mean']:.2f} |")
                if d == 'mice':
                    row += f" {v['delta_r_pooled']['mean']:.2f} / {w['delta_r_pooled']['mean']:.2f} |"
                L.append(row)
        L.append('\n| arm | prefix | FVE | alive | eligible | position-bound share | predictive share | pred. share beyond input | median R2 d+ | code R2 d+ | code R2 beyond input |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for key, c in S[d].items():
            arm, p = key.split('|')
            q = c['comp']
            g = lambda n, k=3: f(q[n], k) if n in q else '-'  # noqa: E731
            L.append(f"| {arm} | {p} | {f(c['fve'], 4)} | {q['alive']['mean']:.0f} | {q['eligible']['mean']:.0f} | "
                     f"{g('position_bound_share')} | {g('predictive_share')} | {g('predictive_share_beyond_input')} | "
                     f"{g('r2_median', 5)} | {g('code_r2_dplus')} | {g('code_r2_dplus_beyond_input')} |")
        for key, c in S[d].items():
            for k in ('repro_max_abs_diff_vs_res_pool_c448', 'repro_max_abs_diff_vs_res_pool_c448dep'):
                if k in c:
                    L.append(f'\n{key} {k}: {c[k]}')
    L.append('\n## Pre-registered decisions\n')
    L.append('```\n' + json.dumps(S['decision'], indent=1, default=float) + '\n```')
    L.append('\n## Train-pool loss values per checkpoint (measure.json)\n')
    for d, M in S['measure'].items():
        for k, m in M.items():
            L.append(f"- {d} {k}: L_rec {m['L_rec']:.1f}, L_inv {m['L_inv_raw']:.4f}, L_pred(ridge) {m['L_pred_raw_ridge']:.4f}, "
                     f"FVE {m['fve_train_code']:.4f}" + (f", repro {m['repro_vs_res_pool_c448']}" if 'repro_vs_res_pool_c448' in m else ''))
    L.append('\n## Encode checks / costs\n')
    L.append('```\n' + json.dumps({'encode': S['encode'], 'costs': S['costs']}, indent=1, default=float) + '\n```')
    (OUT / 'tables.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def cmd_clean(args):
    for d in DOMAINS:
        for sub in ('base', 'enc'):
            for p in sorted((OUT / d / sub).glob('*.npy')) if (OUT / d / sub).exists() else []:
                log(f'rm {p} ({p.stat().st_size / 1e9:.1f} GB)')
                p.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['plan', 'encode', 'warmup', 'train', 'evaluate', 'sheets', 'summary', 'clean'])
    p.add_argument('--domain', default='mice', choices=list(DOMAINS))
    p.add_argument('--domains', default='mice,ants')
    p.add_argument('--arms', nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--n-check', type=int, default=2)
    p.add_argument('--steps', type=int, default=TOTAL_STEPS)
    p.add_argument('--force', action='store_true')
    p.add_argument('--browse-arm', default=None)
    p.add_argument('--browse-frames', type=int, default=20000)
    a = p.parse_args()
    if a.arms is None:
        a.arms = {'train': ['C1r', 'INV', 'PRED', 'INVPRED', 'C1'], 'evaluate': list(EVAL_ARMS),
                  'sheets': ['C1', 'INV', 'PRED', 'INVPRED']}.get(a.cmd, [])
    {'plan': cmd_plan, 'encode': cmd_encode, 'warmup': cmd_warmup, 'train': cmd_train, 'evaluate': cmd_evaluate,
     'sheets': cmd_sheets, 'summary': cmd_summary, 'clean': cmd_clean}[a.cmd](a)


if __name__ == '__main__':
    main()
