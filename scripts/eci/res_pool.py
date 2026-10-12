"""
RES_POOL test: input RESOLUTION and the spatial POOLING read-out inside the final ECI pipeline C1 (ants + mice).

C1 (scripts/eci/combined.py, commit 5d7b8c0): frames at 5 fps -> DINOv2-base patch tokens -> today's foreground mask
(mice fg448, ants antsfg) -> M5 input [token_t, token_t - token_{t-5}] at the same patch, TokenNorm.fit_blocks([768,
768]) -> Matryoshka BatchTopK SAE (4096 latents, k = 16, levers recipe = motion_t3.train_one) -> frame read-out.

Arms (3 seeds each; identical train frames per domain, recipe, steps (6000) and batch (4096): equal token
presentations; the pool is ALL train tokens of the arm, as scripts/eci/native_res*.py)
    c448     448 px store tokens (32 x 32 grid) + token_{t-5} of the d5 store (motion_t3.D5Matcher), RETRAINED on the
             train frames below (the deployed C1 SAEs saw other frames: all train frames, capped pool).
    n896     native source frames -> 896 px (64 x 64 grid). Kept region = the frame's store mask, each 448 patch -> its
             2 x 2 children (same frame area in every arm). token_t and token_{t-5} at the same 896 patch, both native.
             mice: 2064 px mp4 -> scale + pad to 896, JPEG q 2 (ffmpeg chain of scripts/eci/native_res.py), decoded
             SEQUENTIALLY from the start of the stream (select by decoded-frame index);
             ants: 770 / 824 px source -> scale + pad to 896, raw RGB pipe (native_res_ants.decode_native).
    u896     the 512 px dataset JPEGs bicubic-upsampled to 896 by the HF processor (finer grid, no new detail).
    c448dep  reference only: the DEPLOYED C1 checkpoints (results/vision/eci_t3_motion/<d>/sae/M5_s*) on the c448 eval
             tokens; its MAX read-out must reproduce results/vision/eci_combined/<d>/eval/C1_s*.json.
Train frames: mice = the 10,000 frames of scripts/eci/native_res.py (100 per unannotated train video), ants = the 20,480
    frames of scripts/eci/native_res_ants.py (160 per train video), plus their t-5 frames.
Eval frames: ALL store frames (1 fps) of the annotated eval videos = the combined.py eval set: mice 172,800 frames of
    144 videos, ants 76,800 frames of 128 videos. Store frames are exactly 5 rows (1 s) apart (checked), so token_{t-5}
    of a store frame is the previous store frame, except a video's first store frame (row - 5 clipped to the video's
    first row, as motion_t3.nbr_rows; zero change when that is the frame itself).
Frame mapping: source frame of 5 fps frame k = start_frame + 6 k + 2 (mice, native_res), 6 k + o_video (ants,
    native_res_ants offset.json: v3 1, v2 2).
Precision: every 896 encode is an fp32 forward with TF32 off, tokens rounded to fp16 (the store's rule), one rule for
    all arms and jobs.
Read-outs (no retraining, the same patch codes): MAX over kept patches (today); TOPK = (sum of the k largest patch
    activations) / k with k fixed in animal-area units, k = round(0.25 x median single-animal area in patches at the
    grid) (levers train meta: mice a1 = 45 at 448 -> k 11 at 448, 45 at 896; ants a1 = 21 -> k 5 / 21; a frame with
    fewer than k kept patches counts the missing ones as 0); MEAN over kept patches.
Evaluation (combined.py harness) per arm x read-out x seed: sae_levers.score_model (cross-fitted best-neuron AUROC, AP
    of the best-AP pick, honest top-1%, size-controlled AUROC over 448 fg-count deciles); video level
    (scripts/eci/readouts.py): the AUROC pick's held-out per-video mean of the read-out vs the annotated rate and
    bouts / min, raw and fg-count controlled (residual on the per-video mean 448 kept-patch count; 896 = exactly 4x);
    mice also pooled d-r at the phase switches and per-switch dz / sign agreement (habituation = its last 15 min).

PRE-REGISTERED DECISIONS (fixed before any result)
  1 Resolution, per domain: n896 beats c448 (both MAX) iff no label's cf AUROC (3-seed mean) drops > 0.02 AND >= 1
    label gains: cf AUROC mean >= +0.03 and above c448's best seed, OR cf AP mean >= 1.3 x and above c448's best seed,
    OR count-controlled presence-mean video r (3-seed mean) >= +0.10 (annotated rate or bouts / min, as combined.py).
    Also reported with the same rule: u896 vs c448, n896 vs u896.
  2 Pooling: TOPK (or MEAN) replaces MAX iff in BOTH domains and at BOTH resolutions (c448, n896): no label's cf AUROC
    (3-seed mean) drops > 0.02 below MAX of the same arm, AND the count-controlled video r with the annotated RATE
    (3-seed mean), averaged over the domain's labels, rises by >= 0.05 (bouts / min reported alongside). If both pass:
    the one with the larger mean gain over the 4 domain x resolution cells. Otherwise MAX stays.
  3 Pooling artefact (ants): n896 - c448 count-controlled video r (rate) per label under MAX and under TOPK / MEAN;
    'the 896 video-level drop disappears' iff its mean over labels is < 0 under MAX and >= 0 under TOPK.

Steps (scripts/eci/res_pool.sh)
    plan          CPU   frame tables, pos448, labels, halves, mice decode plan, eval shards
    decode        CPU   (mice) array task: sequential native decode of the task's videos -> OUT/mice/native/*.tar
    encode_train  GPU   n896 / u896 train tokens + t-5 tokens -> OUT/<d>/train/<arm>_{tok,prev}.npy
    train         GPU   arm x 3 seeds -> OUT/<d>/sae/<arm>_s<seed>/
    evaluate      GPU   eval shard: streamed encode -> SAE codes -> MAX / TOPK / MEAN per frame for all latents ->
                        OUT/<d>/codes/<arm>/part_XXofYY/ (no eval tokens are written to NFS)
    score         GPU   frame + video level per arm x read-out x seed -> OUT/<d>/eval/<arm>_<ro>_s<seed>.json
    sheets        GPU   (mice n896) contact sheets of the best nose_nose / nose_tail neurons (t | crop t | crop t-5)
    summary       any   tables + decisions -> OUT/summary.json, OUT/tables.md (login-safe: reads JSONs)
Outputs: results/vision/eci_res_pool/ (gitignored); RP_OUT overrides it (smoke test: plan --smoke).
"""
import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
OUT = Path(os.environ.get('RP_OUT') or REPO / 'results/vision/eci_res_pool')
NRM = REPO / 'results/vision/eci_native_res'
NRA = REPO / 'results/vision/eci_native_res_ants'
T3 = REPO / 'results/vision/eci_t3_motion'
COMB = REPO / 'results/vision/eci_combined'
LEV = REPO / 'results/vision/eci_sae_levers'
MICE_SRC = REPO / 'data/mice/source'
MICE_EXP = REPO / 'data/mice/v1/experiment.csv'
D, SIZE, G = 768, 896, 64
SEEDS = (0, 1, 2)
ARMS = ('c448', 'c448dep', 'n896', 'u896')
READOUTS = ('max', 'topk', 'mean')
LABELS = {'mice': ['nose_nose', 'nose_tail'], 'ants': ['groom_any', 'groom_yellow', 'groom_blue', 'onlid_yellow']}
DOMAINS = ('mice', 'ants')
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def stage_dir():
    d = Path(os.environ.get('LOCAL_DIR') or '')
    if not str(d).startswith('/localhome'):
        raise RuntimeError(f'LOCAL_DIR must be the job dir under /localhome, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


def a1(d):
    return float(json.loads((LEV / d / 'train_meta_w4096_w16384.json').read_text())['median_single_animal_area'])


def ktop(d, grid):
    return int(round(0.25 * a1(d) * (grid // 32) ** 2))


def children_896(pos448):
    p = np.asarray(pos448, np.int64)
    r, c = p // 32, p % 32
    return np.sort(np.concatenate([(2 * r + dr) * 64 + 2 * c + dc for dr in (0, 1) for dc in (0, 1)]))


def plan():
    return json.loads((OUT / 'plan.json').read_text())


def table(d, split):
    F = pd.read_parquet(OUT / d / split / 'frames.parquet')
    p448 = np.fromfile(OUT / d / split / 'pos448.i16', np.int16)
    l448 = F.n_fg.values.astype(np.int64)
    assert len(p448) == l448.sum()
    return F, p448, l448


def sae_path(d, arm, seed):
    if arm == 'c448dep':
        return T3 / d / 'sae' / f'M5_s{seed}' / 'sae.pt'
    return OUT / d / 'sae' / f'{arm}_s{seed}' / 'sae.pt'


# ---------------------------------------------------------------------------------------------- plan
def cmd_plan(args):
    import motion_t3 as mt
    import multiscale_sae as ms
    import spatial_sae_pilot as ssp
    from src.eci.domain import get_domain
    from src.eci.foreground import obs_rows
    P = {'smoke': bool(args.smoke), 'ktop': {}, 'a1': {}}
    for d in args.domains.split(','):
        t0 = time.time()
        idx = ssp.StoreIndex(d)
        tr, ev, _, _, train_v, eval_v = idx.split(d, ms.N_TRAIN[d], 0)
        ann_p = get_domain(d).ann_path
        ann = pd.read_csv(ann_p, usecols=['observation_id', 'frame_idx', 'frame_path'] + (['experiment'] if d == 'ants' else []))
        R = obs_rows(ann_p)
        T = pd.read_parquet((NRM if d == 'mice' else NRA) / 'train/frames.parquet')
        assert np.isin(T.store_frame.values, tr).all() and (np.diff(T.store_frame.values) > 0).all()
        sets = {'train': T.store_frame.values.astype(np.int64), 'eval': np.sort(ev)}
        lab_all, _, _ = ms.eval_labels(d, idx, sets['eval'])
        if args.smoke:  # mice: 2 train videos x 4 frames, 1 pool per cross-fit half (12 videos), every 20th store frame
            o_tr = idx.obs[sets['train']]
            keep_tr = np.isin(o_tr, sorted(set(o_tr))[:2])
            sets['train'] = sets['train'][keep_tr][np.r_[0:4, -4:0]] if d == 'mice' else sets['train'][
                np.isin(o_tr, sorted(set(o_tr))[:3])][::40][:12]
            obs_e = idx.obs[sets['eval']]
            if d == 'mice':
                unit = pd.Series(obs_e).str.rsplit('_', n=2).str[0].values
                us = sorted(set(unit))
                pick = [us[0], us[1]]  # alternate halves
                m = np.isin(unit, pick)
            else:
                on = lab_all[lab_all.experiment == 'v3'].groupby('obs').onlid_yellow.sum().sort_values(ascending=False)
                m = np.isin(obs_e, on.index[:8].tolist())
            sel = np.flatnonzero(m)
            sel = np.concatenate([g.values[::20] for _, g in pd.Series(sel).groupby(obs_e[sel])])
            sets['eval'] = np.sort(sets['eval'][sel])
        pa, gs = mt.all_pos(idx)
        for name, fr in sets.items():
            rows, obs = idx.rows[fr], idx.obs[fr]
            lo = np.array([R[o][0] for o in obs])
            prow = np.maximum(rows - 5, lo)
            F = pd.DataFrame({'store_frame': fr, 'row': rows, 'obs': obs, 'frame_idx': ann.frame_idx.values[rows],
                              'frame_path': ann.frame_path.values[rows], 'n_fg': idx.nfg[fr].astype(np.int64),
                              'prow': prow, 'pfi': ann.frame_idx.values[prow], 'ppath': ann.frame_path.values[prow]})
            assert (ann.observation_id.values[rows] == obs).all() and (ann.observation_id.values[prow] == obs).all()
            assert ((F.frame_idx - F.pfi).values == (F.row - F.prow).values).all()
            assert (F.n_fg > 0).all()
            if d == 'ants':
                F['version'] = ann.experiment.values[rows]
            pos = np.concatenate([pa[gs[f]:gs[f] + idx.nfg[f]] for f in fr]).astype(np.int16)
            (OUT / d / name).mkdir(parents=True, exist_ok=True)
            pos.tofile(OUT / d / name / 'pos448.i16')
            F.to_parquet(OUT / d / name / 'frames.parquet')
            log(f'{d} {name}: {len(F):,} frames of {F.obs.nunique()} videos, {int(F.n_fg.sum()):,} tokens at 448 '
                f'({F.n_fg.mean():.1f}/frame); t-5 = the frame itself (zero change) {int((F.prow == F.row).sum())}, '
                f't-5 clipped to the video start {int(((F.row - F.prow) % 5 != 0).sum())}')
        # labels, cross-fit halves, size deciles
        E = pd.read_parquet(OUT / d / 'eval' / 'frames.parquet')
        lab, _, valid = ms.eval_labels(d, idx, E.store_frame.values)
        assert (lab.row.values == E.row.values).all()
        unit = lab.obs.str.rsplit('_', n=2).str[0].values if d == 'mice' else lab.obs.values
        units = sorted(set(unit))
        half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
        if not args.smoke:
            assert np.array_equal(half, np.load(LEV / d / 'half.npy')), 'halves differ from sae_levers'
        lab.drop(columns=['frame_path']).to_parquet(OUT / d / 'labels.parquet')
        np.save(OUT / d / 'half.npy', half)
        np.savez(OUT / d / 'valid.npz', **{b: valid[b] for b in LABELS[d]})
        P['ktop'][d] = {'448': ktop(d, 32), '896': ktop(d, 64)}
        P['a1'][d] = a1(d)
        vids = list(dict.fromkeys(E.obs))
        if d == 'mice':
            exp = pd.read_csv(MICE_EXP).set_index('observation_id')
            Tr = pd.read_parquet(OUT / d / 'train' / 'frames.parquet')
            videos = {}
            for split, F in (('eval', E), ('train', Tr)):
                for o, g in F.groupby('obs', sort=True):
                    ks = sorted(set(g.frame_idx.astype(int)) | set(g.pfi.astype(int)))
                    st = int(exp.loc[o, 'start_frame'])
                    videos[o] = {'set': split, 'src': exp.loc[o, 'observation_file'], 'start': st, 'ks': ks,
                                 't_ks': sorted(g.frame_idx.astype(int)), 'frame_dir': str(Path(g.frame_path.iat[0]).parent),
                                 'last_src': st + 6 * ks[-1] + 2}
            n_ev_tasks, n_tr_tasks = (1, 0) if args.smoke else (args.decode_tasks // 2, args.decode_tasks // 2)
            tasks = []
            for split, nt in (('eval', n_ev_tasks), ('train', n_tr_tasks)):
                vs = [o for o, v in videos.items() if v['set'] == split] if not args.smoke else list(videos)
                if nt == 0:
                    continue
                tk, load = [[] for _ in range(nt)], np.zeros(nt)
                for o in sorted(vs, key=lambda o: -videos[o]['last_src']):
                    t = int(np.argmin(load))
                    tk[t].append(o)
                    load[t] += videos[o]['last_src']
                tasks += tk
            n_sh = 1 if args.smoke else args.eval_shards_mice
            per = len([t for t in tasks if any(videos[o]['set'] == 'eval' for o in t)]) // n_sh
            shards = [[o for t in tasks[j * per:(j + 1) * per] for o in t if videos[o]['set'] == 'eval'] for j in range(n_sh)]
            assert sorted(o for s in shards for o in s) == sorted(vids)
            P['mice'] = {'videos': videos, 'decode_tasks': tasks, 'eval_shards': shards,
                         'source_frames_to_decode': int(sum(v['last_src'] + 1 for v in videos.values())),
                         'n_frames_to_decode': int(sum(len(v['ks']) for v in videos.values()))}
            log(f'mice decode plan: {len(videos)} videos, {P["mice"]["n_frames_to_decode"]:,} frames, '
                f'{P["mice"]["source_frames_to_decode"]:,} source frames decoded sequentially, {len(tasks)} tasks; '
                f'eval shards {[len(s) for s in shards]} videos')
        else:
            n_sh = 1 if args.smoke else args.eval_shards_ants
            P['ants'] = {'eval_shards': [vids[j::n_sh] for j in range(n_sh)]}
        log(f'{d}: plan done [{time.time() - t0:.0f}s]; k_top {P["ktop"][d]}')
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((OUT / 'plan.json').read_text()) if (OUT / 'plan.json').exists() else {}
    for k in ('ktop', 'a1'):
        P[k] = {**old.get(k, {}), **P[k]}
    (OUT / 'plan.json').write_text(json.dumps({**old, **P}, indent=1))


# ---------------------------------------------------------------------------------------------- decode (mice)
def _decode_video(a):
    import native_res as nr
    obs, v, stage = a
    stage = Path(stage)
    t0 = time.time()
    loc = stage / f'src_{obs}.mp4'
    shutil.copyfile(MICE_SRC / v['src'], loc)
    t_copy = time.time() - t0
    s = [v['start'] + 6 * k + 2 for k in v['ks']]
    tmp, dst = stage / 'tmp' / obs, stage / 'frames' / v['set'] / obs
    tmp.mkdir(parents=True, exist_ok=True)
    dst.mkdir(parents=True, exist_ok=True)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t1 = time.time()
    subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-filter_threads', '1', '-i', str(loc),
                    '-vf', nr._vf('+'.join(f'eq(n\\,{x})' for x in s)), '-vsync', '0', '-frames:v', str(len(s)),
                    '-q:v', '2', f'{tmp}/%06d.jpg'], check=True)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    files = sorted(tmp.glob('*.jpg'))
    got = len(files)
    for f, k in zip(files, v['ks']):
        f.rename(dst / f'{k:06d}.jpg')
    shutil.rmtree(tmp)
    loc.unlink()
    mad = nr.mad_check(obs, {'k': v['t_ks'], 'frame_dir': v['frame_dir']}, dst) if got == len(s) else None
    return obs, {'set': v['set'], 'n_wanted': len(s), 'n_got': got, 'last_src': v['last_src'], 'copy_s': round(t_copy, 1),
                 'decode_wall_s': round(time.time() - t1, 1),
                 'decode_cpu_s': round((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), 1), 'mad': mad}


def cmd_decode(args):
    from multiprocessing import Pool
    t_all = time.time()
    P = plan()['mice']
    vids = P['decode_tasks'][args.task]
    stage = stage_dir()
    jobs = [(o, P['videos'][o], str(stage)) for o in sorted(vids, key=lambda o: -P['videos'][o]['last_src'])]
    rep = {'task': args.task, 'host': os.uname().nodename, 'workers': args.workers, 'videos': {}}
    with Pool(args.workers) as pool:
        for o, e in pool.imap_unordered(_decode_video, jobs):
            rep['videos'][o] = e
            log(f'  {o}: {e["n_got"]}/{e["n_wanted"]} frames, copy {e["copy_s"]}s, decode {e["decode_wall_s"]}s wall '
                f'{e["decode_cpu_s"]} CPU-s; MAD k-1/k/k+1 ' + (str([[round(x, 2) if x else x for x in m['mad_km1_k_kp1']]
                                                                  for m in e['mad']]) if e['mad'] else 'skipped'))
            if e['n_got'] != e['n_wanted']:
                raise RuntimeError(f'{o}: decoded {e["n_got"]} of {e["n_wanted"]} frames')
    (OUT / 'mice' / 'native').mkdir(parents=True, exist_ok=True)
    for split in ('eval', 'train'):
        src = stage / 'frames' / split
        if not src.exists():
            continue
        tp = stage / f'task_{args.task:02d}_{split}.tar'
        subprocess.run(['tar', '-C', str(src), '-cf', str(tp), '.'], check=True)
        shutil.copyfile(tp, OUT / 'mice' / 'native' / tp.name)
        rep[f'tar_gb_{split}'] = os.path.getsize(tp) / 1e9
        tp.unlink()
    rep['wall_s'] = round(time.time() - t_all, 1)
    (OUT / 'mice' / 'native' / f'task_{args.task:02d}.json').write_text(json.dumps(rep, indent=1))
    log(f'task {args.task}: {len(vids)} videos in {rep["wall_s"]:.0f}s, '
        f'{sum(e["decode_cpu_s"] for e in rep["videos"].values()) / 3600:.2f} CPU-h of ffmpeg')


def stage_native(d_split_videos, stage):
    """Extract the decode tars holding the given (split, video) frames -> stage/native/<obs>/<k>.jpg."""
    P = plan()['mice']
    nat = stage / 'native'
    nat.mkdir(exist_ok=True)
    t0 = time.time()
    need = {}
    for split, o in d_split_videos:
        t = [i for i, tk in enumerate(P['decode_tasks']) if o in tk]
        assert len(t) == 1, o
        need.setdefault((t[0], split), set()).add(o)
    for (t, split), os_ in sorted(need.items()):
        tp = OUT / 'mice' / 'native' / f'task_{t:02d}_{split}.tar'
        subprocess.run(['tar', '-C', str(nat), '-xf', str(tp)] + [f'./{o}' for o in sorted(os_)], check=True)
    log(f'  staged native frames of {len(d_split_videos)} videos from {len(need)} tars in {time.time() - t0:.0f}s')
    return nat


# ---------------------------------------------------------------------------------------------- streamed 896 encoding
class Stream:
    """t frames = rows `sel` of a frames table F; items = the unique (video, k) frames to encode (t frames and their t-5
    frames), sorted per video by k so that a t-5 frame is always encoded before the frame that uses it. run() yields per
    batch the t frames done: (positions in sel, token_t list, token_t-5 list), both at the frame's 896 kept patches."""

    def __init__(self, F, sel, p448, l448, dev):
        import torch
        sub = F.iloc[sel].reset_index(drop=True)
        self.sub, self.dev = sub, dev
        st = np.r_[0, np.cumsum(l448)]
        self.pos = [children_896(p448[st[i]:st[i + 1]]) for i in sel]
        self.pos_t = [torch.from_numpy(p).to(dev) for p in self.pos]
        kt = list(zip(sub.obs, sub.frame_idx.astype(int)))
        kp = list(zip(sub.obs, sub.pfi.astype(int)))
        path = {}
        for k, p in zip(kt, sub.frame_path):
            path[k] = p
        for k, p in zip(kp, sub.ppath):
            path.setdefault(k, p)
        vord = {o: i for i, o in enumerate(dict.fromkeys(sub.obs))}
        self.items = sorted(path, key=lambda x: (vord[x[0]], x[1]))
        self.paths = [path[x] for x in self.items]
        ii = {x: i for i, x in enumerate(self.items)}
        self.t_item = np.array([ii[x] for x in kt])
        self.p_item = np.array([ii[x] for x in kp])
        self.t_at = defaultdict(list)
        for t, i in enumerate(self.t_item):
            self.t_at[int(i)].append(t)
        self.uses = Counter(int(p) for t, p in enumerate(self.p_item) if p != self.t_item[t])
        self.lens896 = np.array([len(p) for p in self.pos], np.int64)

    def run(self, model, ds, bs, workers):
        import torch
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False, pin_memory=True,
                                         prefetch_factor=4 if workers else None)
        cache, done, uses = {}, np.zeros(len(self.sub), bool), Counter(self.uses)
        self.gpu_s, self.n_items, t0 = 0.0, 0, time.time()
        for bi, (ii, pix) in enumerate(dl):
            torch.cuda.synchronize()
            tt = time.time()
            with torch.inference_mode():
                hs = model(pixel_values=pix.to(self.dev, non_blocking=True)).last_hidden_state.float()[:, 1:]
                assert hs.shape[1] == G * G and bool(torch.isfinite(hs).all()), hs.shape
                grid = hs.half()
            torch.cuda.synchronize()
            self.gpu_s += time.time() - tt
            ts, toks, prevs = [], [], []
            for j, it in enumerate(ii.tolist()):
                if uses.get(it):
                    cache[it] = grid[j].clone()
                for t in self.t_at.get(it, ()):
                    p = self.pos_t[t]
                    tok = grid[j, p]
                    pi = int(self.p_item[t])
                    if pi == it:
                        prev = tok
                    else:
                        prev = cache[pi][p]
                        uses[pi] -= 1
                        if uses[pi] == 0:
                            del cache[pi]
                    assert not done[t]
                    done[t] = True
                    ts.append(t)
                    toks.append(tok)
                    prevs.append(prev)
            self.n_items += len(ii)
            if ts:
                yield np.array(ts), toks, prevs
            if bi % 250 == 0:
                log(f'    batch {bi}: {self.n_items:,}/{len(self.items):,} frames encoded, {int(done.sum()):,} t frames '
                    f'[{time.time() - t0:.0f}s, GPU {self.gpu_s:.0f}s]; cache {len(cache)}')
        assert done.all() and not cache, (int((~done).sum()), len(cache))
        self.wall_s = time.time() - t0


def dataset(arm, d, stream, stage, proc, split):
    """torch Dataset of the stream's items: (item index, pixel_values (3, 896, 896))."""
    import torch
    from PIL import Image

    class Jpg(torch.utils.data.Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            with Image.open(self.paths[i]) as im:
                im = im.convert('RGB')
            return i, proc(images=im, return_tensors='pt')['pixel_values'][0]

    class AntsNative(torch.utils.data.IterableDataset):  # native_res_ants.NatDS: one ffmpeg call per video
        def __init__(self, groups):
            self.groups = groups

        def __iter__(self):
            import native_res_ants as nra
            wi = torch.utils.data.get_worker_info()
            off = nra.offsets()
            for gi, (o, ver, ix, ks) in enumerate(self.groups):
                if wi is not None and gi % wi.num_workers != wi.id:
                    continue
                fr = nra.decode_native(nra.source_of(ver, o), [6 * int(k) + off[o] for k in ks])
                for j, a in zip(ix, fr):
                    yield int(j), torch.from_numpy(((a.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())

    if arm == 'u896':
        lst = stage / 'jpg512.txt'
        lst.write_text('\n'.join(sorted(set(stream.paths))) + '\n')
        t0 = time.time()
        (stage / 'jpg512').mkdir(exist_ok=True)
        subprocess.run(f'tar -C {REPO / "dataset"} -cf - -T {lst} | tar -C {stage / "jpg512"} -xf -', shell=True, check=True)
        log(f'  staged {len(set(stream.paths)):,} 512 px JPGs in {time.time() - t0:.0f}s')
        return Jpg([stage / 'jpg512' / p for p in stream.paths])
    if d == 'mice':
        nat = stage_native([(split, o) for o in dict.fromkeys(stream.sub.obs)], stage)
        return Jpg([nat / o / f'{k:06d}.jpg' for o, k in stream.items])
    ver = dict(zip(stream.sub.obs, stream.sub.version))
    groups = []
    for o in dict.fromkeys(o for o, _ in stream.items):
        ix = [i for i, (oo, _) in enumerate(stream.items) if oo == o]
        groups.append((o, ver[o], ix, [stream.items[i][1] for i in ix]))
    return AntsNative(groups)


def load_dino(dev):
    import torch
    from src.eci.extract import load_encoder
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, proc, model = load_encoder('dinov2_base', SIZE, dev, center_crop=False)
    return proc, model


# ---------------------------------------------------------------------------------------------- encode_train
def cmd_encode_train(args):
    import torch
    dev = torch.device('cuda')
    stage = stage_dir()
    d, arm = args.domain, args.arm
    assert arm in ('n896', 'u896')
    F, p448, l448 = table(d, 'train')
    proc, model = load_dino(dev)
    S = Stream(F, np.arange(len(F)), p448, l448, dev)
    ds = dataset(arm, d, S, stage, proc, 'train')
    s8 = np.r_[0, np.cumsum(S.lens896)]
    n = int(s8[-1])
    tok = np.lib.format.open_memmap(stage / f'{arm}_tok.npy', 'w+', np.float16, (n, D))
    prev = np.lib.format.open_memmap(stage / f'{arm}_prev.npy', 'w+', np.float16, (n, D))
    log(f'{d} {arm} train: {len(F):,} t frames, {len(S.items):,} frames to encode, {n:,} tokens')
    for ts, tk, pv in S.run(model, ds, args.bs, args.workers):
        for t, a, b in zip(ts, tk, pv):
            tok[s8[t]:s8[t + 1]] = a.cpu().numpy()
            prev[s8[t]:s8[t + 1]] = b.cpu().numpy()
    tok.flush()
    prev.flush()
    info = {'domain': d, 'arm': arm, 'gpu': torch.cuda.get_device_name(0), 'n_t_frames': len(F),
            'n_frames_encoded': len(S.items), 'n_tokens': n, 'gpu_s': round(S.gpu_s, 1), 'wall_s': round(S.wall_s, 1),
            'gpu_s_per_1k_frames': round(1000 * S.gpu_s / len(S.items), 1), 'precision': 'fp32 forward, TF32 off, fp16 tokens',
            'zero_change_frames': int((S.t_item == S.p_item).sum())}
    ref = (NRM / 'train' / f'tok896{arm[0]}.npy') if d == 'mice' else None
    if ref is not None and ref.exists() and not plan()['smoke']:  # identity with native_res (same frames, A40 fp32)
        r = np.load(ref, mmap_mode='r')
        m = int(s8[64])
        a, b = np.asarray(tok[:m], np.float32), np.asarray(r[:m], np.float32)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        info['check_vs_native_res_first64'] = {'n_tokens': m, 'cos_min': float(cos.min()), 'cos_mean': float(cos.mean()),
                                               'max_abs': float(np.abs(a - b).max())}
        log(f'  check vs {ref.name}: {info["check_vs_native_res_first64"]}')
    del tok, prev
    (OUT / d / 'train').mkdir(parents=True, exist_ok=True)
    for f in (f'{arm}_tok.npy', f'{arm}_prev.npy'):
        shutil.copyfile(stage / f, OUT / d / 'train' / f)
        (stage / f).unlink()
    (OUT / d / f'encode_train_{arm}.json').write_text(json.dumps(info, indent=1))
    log(f'done: {json.dumps(info)}')


# ---------------------------------------------------------------------------------------------- train
def cmd_train(args):
    import torch
    import motion_t3 as mt
    import spatial_sae_pilot as ssp
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True  # as motion_t3 / sae_levers train
    stage = stage_dir()
    d, arm = args.domain, args.arm
    F, p448, l448 = table(d, 'train')
    t0 = time.time()
    if arm == 'c448':
        idx = ssp.StoreIndex(d)
        fr = F.store_frame.values
        tok, pos, lens = idx.load(fr)
        assert (pos == p448).all() and (lens == l448).all()
        prev, _, ok = mt.D5Matcher(idx, d).fetch(fr)
        extra = {'d5_unmatched_tokens': int((~ok).sum())}
    else:
        extra = {}
        loc = {}
        for k in ('tok', 'prev'):
            shutil.copyfile(OUT / d / 'train' / f'{arm}_{k}.npy', stage / f'{arm}_{k}.npy')
            loc[k] = np.load(stage / f'{arm}_{k}.npy')
            (stage / f'{arm}_{k}.npy').unlink()
        tok, prev = loc['tok'], loc['prev']
        assert len(tok) == 4 * l448.sum()
    n = len(tok)
    X = torch.empty((n, 2 * D), dtype=torch.float16, device=dev)
    for a in range(0, n, 1_000_000):
        t = torch.from_numpy(np.ascontiguousarray(tok[a:a + 1_000_000])).to(dev)
        X[a:a + len(t), :D] = t
        X[a:a + len(t), D:] = (t.float() - torch.from_numpy(np.ascontiguousarray(prev[a:a + 1_000_000])).to(dev).float()).half()
    del tok, prev
    zero = float(sum(int((X[a:a + 500_000, D:].float().abs().sum(1) == 0).sum()) for a in range(0, n, 500_000)) / n)
    rn = float((X[:200_000, D:].float().norm(dim=1) / X[:200_000, :D].float().norm(dim=1)).median())
    meta = {'domain': d, 'arm': arm, 'n_train_frames': len(F), 'n_train_tokens': n, 'tokens_per_frame': n / len(F),
            'steps': args.steps, 'presentations': args.steps * 4096, 'presentations_per_token': args.steps * 4096 / n,
            'zero_change_frac': zero, 'median_change_over_token_norm': rn, 'stage_s': round(time.time() - t0, 1), **extra}
    log(f'{d} {arm}: {json.dumps(meta)}')
    (OUT / d).mkdir(parents=True, exist_ok=True)
    (OUT / d / f'train_meta_{arm}.json').write_text(json.dumps(meta, indent=1))
    for seed in args.seeds:
        mt.train_one(arm, seed, X, args.steps, OUT / d / 'sae' / f'{arm}_s{seed}',
                     {'config': 'C1', 'arm': arm, 'motion_delta': 5, 'domain': d, 'cap': n, 'pool_seed': None,
                      'blocks': [D, D], 'train_frames': 'native_res' if d == 'mice' else 'native_res_ants'}, dev)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- evaluate
class Pooler:
    """Frame read-outs of every latent for several SAEs on M5 inputs: MAX, TOPK (sum of the k largest / k), MEAN, written
    to fp16 memmaps (n_frames, m) per (seed, read-out); FVE accumulators (whole input, static block, change block)."""

    def __init__(self, saes, k, n, dst, dev):
        import torch
        self.saes, self.k, self.dev = saes, k, dev
        self.m = saes[0][1].n_latents
        dst.mkdir(parents=True, exist_ok=True)
        self.mm = {(s, ro): np.lib.format.open_memmap(dst / f's{s}_{ro}.npy', 'w+', np.float16, (n, self.m))
                   for s, _, _ in saes for ro in READOUTS}
        z = lambda q: torch.zeros(q, dtype=torch.float64, device=dev)  # noqa: E731
        self.st = {s: {'s1': z(2 * D), 's2': z(2 * D), 'sse': z(2 * D), 'fire': z(self.m), 'l0': 0.0, 'ntok': 0}
                   for s, _, _ in saes}
        self.done = np.zeros(n, bool)

    def add(self, fidx, tok, prev, lens):
        import torch
        dev, m = self.dev, self.m
        B, N = len(lens), int(tok.shape[0])
        L = torch.from_numpy(np.asarray(lens, np.int64)).to(dev)
        fr = torch.repeat_interleave(torch.arange(B, device=dev), L)
        rank = torch.arange(N, device=dev) - (torch.cumsum(L, 0) - L)[fr]
        Lmax = int(np.max(lens))
        t = tok.float()
        xr = torch.cat([t, t - prev.float()], 1)
        contig = bool((np.diff(fidx) == 1).all())
        for s, sae, norm in self.saes:
            x = norm(xr)
            z = sae.encode(x, mode='threshold').float()
            xh = sae.decode(z)
            S = self.st[s]
            S['s1'] += x.double().sum(0)
            S['s2'] += x.double().pow(2).sum(0)
            S['sse'] += (x - xh).double().pow(2).sum(0)
            act = z > 0
            S['fire'] += act.sum(0).double()
            S['l0'] += float(act.sum())
            S['ntok'] += N
            mx = torch.zeros(B, m, device=dev).index_reduce_(0, fr, z, 'amax', include_self=True)
            mean = torch.zeros(B, m, device=dev).index_add_(0, fr, z) / L[:, None].float()
            Z = torch.zeros(B, Lmax, m, device=dev)
            Z[fr, rank] = z
            tk = Z.topk(min(self.k, Lmax), dim=1).values.sum(1) / self.k
            del Z
            for ro, v in (('max', mx), ('topk', tk), ('mean', mean)):
                a = v.half().cpu().numpy()
                if contig:
                    self.mm[(s, ro)][fidx[0]:fidx[-1] + 1] = a
                else:
                    self.mm[(s, ro)][fidx] = a
        assert not self.done[fidx].any()
        self.done[fidx] = True

    def finish(self, dst):
        assert self.done.all(), f'{int((~self.done).sum())} frames without codes'
        for v in self.mm.values():
            v.flush()
        np.savez(dst / 'stats.npz', **{f's{s}_{k}': (v.cpu().numpy() if hasattr(v, 'cpu') else np.array(v))
                                       for s, S in self.st.items() for k, v in S.items()})


def chunks_448(lens, max_cells=65536, max_tok=300_000):
    out, f0 = [], 0
    while f0 < len(lens):
        f1, lm, nt = f0, 0, 0
        while f1 < len(lens):
            lm2, nt2 = max(lm, lens[f1]), nt + lens[f1]
            if f1 > f0 and (lm2 * (f1 - f0 + 1) > max_cells or nt2 > max_tok):
                break
            lm, nt, f1 = lm2, nt2, f1 + 1
        out.append((f0, f1))
        f0 = f1
    return out


def cmd_evaluate(args):
    import torch
    import motion_t3 as mt
    import multiscale_sae as ms
    import spatial_sae_pilot as ssp
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    stage = stage_dir()
    d = args.domain
    E, p448, l448 = table(d, 'eval')
    P = plan()
    shards = P[d]['eval_shards']
    vids = shards[args.task]
    sel = np.flatnonzero(E.obs.isin(vids).values)
    tag = f'part_{args.task:02d}of{len(shards):02d}'
    info = {'domain': d, 'shard': args.task, 'n_shards': len(shards), 'n_videos': len(vids), 'n_frames': int(len(sel)),
            'gpu': torch.cuda.get_device_name(0), 'host': os.uname().nodename}
    t0 = time.time()
    arms = args.arms
    saes = {a: [(s, *load_sae(sae_path(d, a, s), dev)[:2]) for s in args.seeds] for a in arms}
    for a in arms:
        assert all(x[1].d_in == 2 * D for x in saes[a])
    if arms[0] in ('c448', 'c448dep'):
        assert all(a in ('c448', 'c448dep') for a in arms)
        k = ktop(d, 32)
        idx = ssp.StoreIndex(d)
        fr = E.store_frame.values[sel]
        tok, pos, lens = ms.stage(idx, fr, stage / 'eval')
        assert (lens == l448[sel]).all()
        prev = mt.stage_prev5(idx, d, fr, stage / 'prev.npy')
        info['d5_unmatched_tokens'] = int(mt.STAGE_INFO.get('d5_unmatched', -1))
        pools = {a: Pooler(saes[a], k, len(sel), stage / 'codes' / a, dev) for a in arms}
        st = np.r_[0, np.cumsum(lens)]
        with torch.no_grad():
            for ci, (f0, f1) in enumerate(chunks_448(lens)):
                T = torch.from_numpy(np.ascontiguousarray(tok[st[f0]:st[f1]])).to(dev)
                Pv = torch.from_numpy(np.ascontiguousarray(prev[st[f0]:st[f1]])).to(dev)
                for a in arms:
                    pools[a].add(np.arange(f0, f1), T, Pv, lens[f0:f1])
                if ci % 200 == 0:
                    log(f'  chunk {ci}: frames {f1:,}/{len(lens):,} [{time.time() - t0:.0f}s]')
        del tok, prev
    else:
        assert len(arms) == 1 and arms[0] in ('n896', 'u896')
        a = arms[0]
        k = ktop(d, 64)
        proc, model = load_dino(dev)
        S = Stream(E, sel, p448, l448, dev)
        ds = dataset(a, d, S, stage, proc, 'eval')
        pools = {a: Pooler(saes[a], k, len(sel), stage / 'codes' / a, dev)}
        info['n_frames_encoded'] = len(S.items)
        for ts, tk, pv in S.run(model, ds, args.bs, args.workers):
            o = np.argsort(ts)
            ts = ts[o]
            with torch.no_grad():
                pools[a].add(ts, torch.cat([tk[i] for i in o]), torch.cat([pv[i] for i in o]), S.lens896[ts])
        info.update(encode_gpu_s=round(S.gpu_s, 1), encode_wall_s=round(S.wall_s, 1),
                    gpu_s_per_1k_frames=round(1000 * S.gpu_s / len(S.items), 1))
    info['k_top'] = k
    info['wall_s'] = round(time.time() - t0, 1)
    for a in arms:
        dst = stage / 'codes' / a
        pools[a].finish(dst)
        np.save(dst / 'eidx.npy', sel)
        (dst / 'info.json').write_text(json.dumps({**info, 'arm': a}, indent=1))
        fin = OUT / d / 'codes' / a / tag
        fin.mkdir(parents=True, exist_ok=True)
        for f in dst.iterdir():
            shutil.copyfile(f, fin / f.name)
        shutil.rmtree(dst)
        log(f'{a} -> {fin}')
    log(f'done: {json.dumps(info)}')


# ---------------------------------------------------------------------------------------------- score
def video_level(d, col, res, lab, half, l448, gt):
    import readouts as rd
    out = {}
    for b in LABELS[d]:
        R = {}
        for dd in res[b]['dirs']:
            h, j = dd['test_half'], dd['auc']['neuron']
            m = half == h
            x = col(j)[m].astype(np.float32)
            V = rd.video_readouts(x, x, l448[m].astype(np.float32), lab[m], gt['wstart'])
            R[('presence', h)] = R[('extent', h)] = V
        v = rd.video_check_mice(R, gt, b) if d == 'mice' else rd.video_check_ants(R, gt, b)
        out[b] = {k: v[k] for k in ('presence_mean', 'presence_mean|fgctrl')}
    return out


def load_codes(d, arm, seed, ro, n, stage):
    parts = sorted((OUT / d / 'codes' / arm).glob('part_*'))
    nt = int(parts[0].name.split('of')[1])
    assert len(parts) == nt, f'{d} {arm}: {len(parts)} of {nt} eval parts'
    X = np.empty((n, 4096), np.float16)
    seen = np.zeros(n, bool)
    for p in parts:
        e = np.load(p / 'eidx.npy')
        shutil.copyfile(p / f's{seed}_{ro}.npy', stage / 'c.npy')
        X[e] = np.load(stage / 'c.npy')
        (stage / 'c.npy').unlink()
        assert not seen[e].any()
        seen[e] = True
    assert seen.all()
    return X, parts


def fve_stats(parts, seed):
    S = [np.load(p / 'stats.npz') for p in parts]
    g = lambda k: sum(s[f's{seed}_{k}'] for s in S)  # noqa: E731
    s1, s2, sse, n = g('s1'), g('s2'), g('sse'), float(g('ntok'))
    tss = s2 - s1 ** 2 / n
    f = lambda a, b: float(1 - sse[a:b].sum() / tss[a:b].sum())  # noqa: E731
    return {'fve': f(0, 2 * D), 'fve_static': f(0, D), 'fve_change': f(D, 2 * D), 'l0_per_token': float(g('l0')) / n,
            'dead_frac': float((g('fire') == 0).mean()), 'n_tokens': int(n)}


def cmd_score(args):
    import torch
    import readouts as rd
    import sae_levers as sl
    dev = torch.device('cuda')
    stage = stage_dir()
    d = args.domain
    E, _, l448 = table(d, 'eval')
    lab = pd.read_parquet(OUT / d / 'labels.parquet')
    assert (lab.row.values == E.row.values).all()
    half = np.load(OUT / d / 'half.npy')
    V = np.load(OUT / d / 'valid.npz')
    valid = {b: V[b] for b in LABELS[d]}
    rank = np.argsort(np.argsort(l448 + np.random.default_rng(0).random(len(l448)) * 1e-3))
    decile = np.minimum(rank * 10 // len(l448), 9)
    Yt = torch.from_numpy(lab[LABELS[d]].values.astype(bool)).to(dev)
    gt = rd.mice_truth(stage) if d == 'mice' else rd.ants_truth()
    for dd in ('eval', 'codes_best'):
        (OUT / d / dd).mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for seed in args.seeds:
            for ro in READOUTS:
                t0 = time.time()
                X, parts = load_codes(d, arm, seed, ro, len(E), stage)
                Xt = torch.from_numpy(X).to(dev)
                res, _ = sl.score_model(Xt, Yt, LABELS[d], valid, half, decile, dev)
                js = sorted({q[s]['neuron'] for b in LABELS[d] for q in res[b]['dirs'] for s in ('auc', 'ap', 'top1')})
                cb = X[:, js]
                vid = video_level(d, lambda j: cb[:, js.index(j)], res, lab, half, l448, gt)
                key = f'{arm}_{ro}_s{seed}'
                info = json.loads((parts[0] / 'info.json').read_text())
                e = {'arm': arm, 'readout': ro, 'seed': seed, 'k_top': info['k_top'], 'ckpt': str(sae_path(d, arm, seed)),
                     **fve_stats(parts, seed), 'labels': res, 'video': vid}
                if arm == 'c448dep' and ro == 'max' and (COMB / d / 'eval' / f'C1_s{seed}.json').exists():
                    ref = json.loads((COMB / d / 'eval' / f'C1_s{seed}.json').read_text())['presence']
                    e['repro_max_abs_diff_vs_combined_C1'] = {q: max(abs(ref[b][q] - res[b][q]) for b in LABELS[d])
                                                              for q in ('cf_auroc', 'cf_ap', 'cf_top1')}
                (OUT / d / 'eval' / f'{key}.json').write_text(json.dumps(e, indent=1, default=float))
                np.savez_compressed(OUT / d / 'codes_best' / f'{key}.npz', neurons=np.array(js), codes=cb)
                f = lambda b: (f"{b} {res[b]['cf_auroc']:.3f}/{res[b]['cf_ap']:.4f}/{res[b]['cf_top1']:.3f} "  # noqa: E731
                               f"r {vid[b]['presence_mean']['video_r_rate']:.2f}/{vid[b]['presence_mean|fgctrl']['video_r_rate']:.2f}")
                log(f'{d} {key}: FVE {e["fve"]:.4f} | ' + ' | '.join(f(b) for b in LABELS[d])
                    + (f' | repro {e["repro_max_abs_diff_vs_combined_C1"]}' if 'repro_max_abs_diff_vs_combined_C1' in e else '')
                    + f' [{time.time() - t0:.0f}s]')
                del Xt, X
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- sheets (mice n896)
def render(path, items, title, th=256):
    """items: (img t (H, W, 3) uint8, img t-5, grid, patch pos, label positive, caption). Tile: frame t with the patch
    box | zoom on the patch at t | the same zoom at t-5 (1/8 of the frame width)."""
    from PIL import Image, ImageDraw
    ncol = 2
    rows = int(np.ceil(len(items) / ncol))
    W, H = ncol * 3 * th, rows * (th + 14) + 16
    sheet = Image.new('RGB', (W, H), 'white')
    dr = ImageDraw.Draw(sheet)
    dr.text((4, 2), title, fill='black')
    for q, (a, b, grid, p, yv, cap) in enumerate(items):
        size = a.shape[0]
        ps = size / grid
        r_, c_ = divmod(int(p), grid)
        x0, y0 = c_ * ps, r_ * ps
        col = (0, 200, 0) if yv else (230, 0, 0)
        ims = [Image.fromarray(a), Image.fromarray(b)]
        full = ims[0].copy()
        ImageDraw.Draw(full).rectangle([x0 - 3, y0 - 3, x0 + ps + 2, y0 + ps + 2], outline=col, width=4)
        half = size / 16
        cx, cy = min(max(x0 + ps / 2, half), size - half), min(max(y0 + ps / 2, half), size - half)
        rr, cc = divmod(q, ncol)
        X0, Y0 = cc * 3 * th, 16 + rr * (th + 14)
        sheet.paste(full.resize((th, th), Image.LANCZOS), (X0, Y0))
        zs = th / (2 * half)
        for k, im in enumerate(ims):
            z = im.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((th, th), Image.LANCZOS)
            ImageDraw.Draw(z).rectangle([(x0 - (cx - half)) * zs, (y0 - (cy - half)) * zs, (x0 + ps - (cx - half)) * zs,
                                         (y0 + ps - (cy - half)) * zs], outline=col, width=2)
            ImageDraw.Draw(z).text((3, 3), 't' if k == 0 else 't-5 (1 s)', fill=(255, 255, 0))
            sheet.paste(z, (X0 + (k + 1) * th, Y0))
        dr.text((X0 + 2, Y0 + th), cap, fill='black')
    sheet.save(path, quality=88)


def cmd_sheets(args):
    import torch
    from PIL import Image
    import native_res as nr
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    stage = stage_dir()
    d, arm = 'mice', 'n896'
    E, p448, l448 = table(d, 'eval')
    lab = pd.read_parquet(OUT / d / 'labels.parquet')
    half = np.load(OUT / d / 'half.npy')
    (OUT / d / 'sheets').mkdir(parents=True, exist_ok=True)
    jobs = []
    for b in LABELS[d]:
        for ro in args.readouts:
            ev = {s: json.loads((OUT / d / 'eval' / f'{arm}_{ro}_s{s}.json').read_text()) for s in SEEDS}
            s = max(SEEDS, key=lambda s: ev[s]['labels'][b]['cf_auroc'])
            e0 = ev[s]['labels'][b]['dirs'][0]
            for sel in (('auc', 'top1') if ro == 'max' else ('auc',)):
                j = e0[sel]['neuron']
                z = np.load(OUT / d / 'codes_best' / f'{arm}_{ro}_s{s}.npz')
                x = z['codes'][:, list(z['neurons']).index(j)].astype(np.float32)
                pick = nr.pick_frames(x, half == e0['test_half'], E.obs.values)
                jobs.append({'label': b, 'readout': ro, 'seed': s, 'sel': sel, 'neuron': j, 'pick': pick, 'x': x,
                             'test': e0[sel]})
    frames = sorted({i for jb in jobs for i in jb['pick']})
    proc, model = load_dino(dev)
    S = Stream(E, np.array(frames), p448, l448, dev)
    ds = dataset(arm, d, S, stage, proc, 'eval')
    TK = {}
    for ts, tk, pv in S.run(model, ds, 8, 4):
        for t, a, b_ in zip(ts, tk, pv):
            TK[frames[t]] = (a, b_, S.pos[t])
    nat = stage / 'native'
    k896 = ktop(d, 64)
    index = []
    for jb in jobs:
        sae, norm, _ = load_sae(sae_path(d, arm, jb['seed']), dev)
        j = jb['neuron']
        items = []
        for i in jb['pick']:
            a, b_, pos = TK[i]
            with torch.no_grad():
                xx = norm(torch.cat([a.float(), a.float() - b_.float()], 1))
                zz = sae.encode(xx, mode='threshold')[:, j].float()
            q = int(zz.argmax())
            im_t = np.asarray(Image.open(nat / E.obs.iat[i] / f'{int(E.frame_idx.iat[i]):06d}.jpg').convert('RGB'))
            im_p = np.asarray(Image.open(nat / E.obs.iat[i] / f'{int(E.pfi.iat[i]):06d}.jpg').convert('RGB'))
            yv = bool(lab[jb['label']].iat[i])
            items.append((im_t, im_p, G, int(pos[q]), yv, f'{E.obs.iat[i]} f{int(E.frame_idx.iat[i])} '
                          f'{"POS" if yv else "neg"} {jb["readout"]}={jb["x"][i]:.2f} active {int((zz > 0).sum())}/{len(zz)}'))
        fn = f'{arm}_{jb["readout"]}_s{jb["seed"]}__{jb["label"]}__{jb["sel"]}_n{j}.jpg'
        t = jb['test']
        render(OUT / d / 'sheets' / fn, items,
               f'mice N896 C1 seed {jb["seed"]} {jb["label"]} neuron {j} ({jb["readout"]} read-out, k_top {k896}; selected '
               f'on half 0 by {jb["sel"]}; test-half AUROC {t["test_auc"]:.3f} AP {t["test_ap"]:.3f} top1 {t["test_top1"]:.3f}); '
               f'top frames of half 1, <= 2 per video; green = label positive')
        index.append({k: jb[k] for k in ('label', 'readout', 'seed', 'sel', 'neuron')} |
                     {'n_shown': len(items), 'n_pos_shown': int(sum(it[4] for it in items)), 'file': fn,
                      'test_auc': t['test_auc'], 'test_ap': t['test_ap'], 'test_top1': t['test_top1']})
        log(f'sheet {fn}: {index[-1]["n_pos_shown"]}/{len(items)} label-positive')
    (OUT / d / 'sheets' / 'index.json').write_text(json.dumps(index, indent=1))


# ---------------------------------------------------------------------------------------------- summary
FMETS = ('cf_auroc', 'cf_ap', 'cf_top1', 'cf_size_ctrl_auroc', 'cf_ap_at_auroc')


def ms_(v):
    v = np.asarray(v, float)
    return {'mean': float(np.nanmean(v)), 'sd': float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
            'per_seed': [float(x) for x in v]}


def collect():
    S = {}
    for d in DOMAINS:
        Dd = {}
        for arm in ARMS:
            for ro in READOUTS:
                ps = [OUT / d / 'eval' / f'{arm}_{ro}_s{s}.json' for s in SEEDS]
                if not all(p.exists() for p in ps):
                    continue
                E = [json.loads(p.read_text()) for p in ps]
                c = {'k_top': E[0]['k_top'], **{q: ms_([e[q] for e in E]) for q in ('fve', 'fve_static', 'fve_change',
                                                                                     'l0_per_token', 'dead_frac')}}
                if 'repro_max_abs_diff_vs_combined_C1' in E[0]:
                    c['repro_vs_combined_C1'] = {q: max(e['repro_max_abs_diff_vs_combined_C1'][q] for e in E)
                                                 for q in ('cf_auroc', 'cf_ap', 'cf_top1')}
                for b in LABELS[d]:
                    c[b] = {q: ms_([e['labels'][b][q] for e in E]) for q in FMETS}
                    c[b]['base_rate'] = E[0]['labels'][b]['base_rate']
                    for w in ('presence_mean', 'presence_mean|fgctrl'):
                        V = [e['video'][b][w] for e in E]
                        c[b][w] = {q: ms_([v[q] for v in V]) for q in V[0] if q not in ('effects',) and
                                   not isinstance(V[0][q], dict)}
                        if 'effects' in V[0]:
                            c[b][w]['effects'] = {sw: {'dz': ms_([v['effects'][sw]['dz'] for v in V]),
                                                       'truth_dz': V[0]['effects'][sw]['truth_dz'],
                                                       'truth_p': V[0]['effects'][sw]['truth_p'],
                                                       'sign_agree_seeds': int(sum(v['effects'][sw]['sign_agree'] for v in V))}
                                                  for sw in V[0]['effects']}
                Dd[f'{arm}|{ro}'] = c
        S[d] = Dd
    return S


def res_rule(S, d, new, old):
    per = {}
    for b in LABELS[d]:
        A, B = S[d][new][b], S[d][old][b]
        da = A['cf_auroc']['mean'] - B['cf_auroc']['mean']
        ra = A['cf_ap']['mean'] / B['cf_ap']['mean']
        g_auc = bool(da >= 0.03 and A['cf_auroc']['mean'] > max(B['cf_auroc']['per_seed']))
        g_ap = bool(ra >= 1.3 and A['cf_ap']['mean'] > max(B['cf_ap']['per_seed']))
        va, vb = A['presence_mean|fgctrl'], B['presence_mean|fgctrl']
        dr = va['video_r_rate']['mean'] - vb['video_r_rate']['mean']
        dbp = va['video_r_bpm']['mean'] - vb['video_r_bpm']['mean']
        per[b] = {'d_cf_auroc': da, 'ratio_cf_ap': ra, 'drop_gt_0.02': bool(da < -0.02), 'gain_auroc': g_auc,
                  'gain_ap': g_ap, 'd_ctl_video_r_rate': dr, 'd_ctl_video_r_bpm': dbp,
                  'gain_video': bool(dr >= 0.10 or dbp >= 0.10)}
        per[b]['gain'] = bool(g_auc or g_ap or per[b]['gain_video'])
    no_drop = not any(v['drop_gt_0.02'] for v in per.values())
    gains = [b for b, v in per.items() if v['gain']]
    return {'per_label': per, 'no_drop': no_drop, 'labels_with_gain': gains, 'WIN': bool(no_drop and gains)}


def pool_rule(S, d, arm, ro):
    per = {}
    for b in LABELS[d]:
        A, B = S[d][f'{arm}|{ro}'][b], S[d][f'{arm}|max'][b]
        per[b] = {'d_cf_auroc': A['cf_auroc']['mean'] - B['cf_auroc']['mean'],
                  'd_ctl_video_r_rate': A['presence_mean|fgctrl']['video_r_rate']['mean'] - B['presence_mean|fgctrl']['video_r_rate']['mean'],
                  'd_ctl_video_r_bpm': A['presence_mean|fgctrl']['video_r_bpm']['mean'] - B['presence_mean|fgctrl']['video_r_bpm']['mean'],
                  'd_raw_video_r_rate': A['presence_mean']['video_r_rate']['mean'] - B['presence_mean']['video_r_rate']['mean']}
    mr = float(np.mean([v['d_ctl_video_r_rate'] for v in per.values()]))
    mb = float(np.mean([v['d_ctl_video_r_bpm'] for v in per.values()]))
    no_drop = all(v['d_cf_auroc'] >= -0.02 for v in per.values())
    return {'per_label': per, 'no_drop': bool(no_drop), 'mean_d_ctl_video_r_rate': mr, 'mean_d_ctl_video_r_bpm': mb,
            'pass': bool(no_drop and mr >= 0.05)}


def decide(S):
    R = {'resolution': {}, 'pooling': {}, 'artefact_ants': {}}
    for d in DOMAINS:
        R['resolution'][d] = {}
        for new, old in (('n896', 'c448'), ('u896', 'c448'), ('n896', 'u896')):
            if f'{new}|max' in S[d] and f'{old}|max' in S[d]:
                R['resolution'][d][f'{new}_vs_{old}'] = res_rule(S, d, f'{new}|max', f'{old}|max')
    for ro in ('topk', 'mean'):
        cells = {}
        for d in DOMAINS:
            for arm in ('c448', 'n896'):
                if f'{arm}|{ro}' in S[d] and f'{arm}|max' in S[d]:
                    cells[f'{d}|{arm}'] = pool_rule(S, d, arm, ro)
        R['pooling'][ro] = {'cells': cells, 'complete': len(cells) == 4,
                            'pass_all': bool(len(cells) == 4 and all(c['pass'] for c in cells.values())),
                            'mean_gain_rate': float(np.mean([c['mean_d_ctl_video_r_rate'] for c in cells.values()])) if cells else None}
        for arm in ('c448dep', 'u896'):  # reported, not part of the rule
            for d in DOMAINS:
                if f'{arm}|{ro}' in S[d] and f'{arm}|max' in S[d]:
                    R['pooling'][ro].setdefault('reported_other_arms', {})[f'{d}|{arm}'] = pool_rule(S, d, arm, ro)
    ok = [ro for ro in ('topk', 'mean') if R['pooling'][ro]['pass_all']]
    R['pooling']['FINAL'] = (max(ok, key=lambda r: R['pooling'][r]['mean_gain_rate']) if ok else 'max')
    d = 'ants'
    for ro in READOUTS:
        if f'n896|{ro}' in S[d] and f'c448|{ro}' in S[d]:
            per = {b: S[d][f'n896|{ro}'][b]['presence_mean|fgctrl']['video_r_rate']['mean']
                   - S[d][f'c448|{ro}'][b]['presence_mean|fgctrl']['video_r_rate']['mean'] for b in LABELS[d]}
            per_b = {b: S[d][f'n896|{ro}'][b]['presence_mean|fgctrl']['video_r_bpm']['mean']
                     - S[d][f'c448|{ro}'][b]['presence_mean|fgctrl']['video_r_bpm']['mean'] for b in LABELS[d]}
            R['artefact_ants'][ro] = {'d_ctl_r_rate_n896_minus_c448': per, 'mean': float(np.mean(list(per.values()))),
                                      'd_ctl_r_bpm_n896_minus_c448': per_b, 'mean_bpm': float(np.mean(list(per_b.values())))}
    A = R['artefact_ants']
    if 'max' in A and 'topk' in A:
        A['drop_under_max'] = bool(A['max']['mean'] < 0)
        A['drop_disappears_under_topk'] = bool(A['max']['mean'] < 0 and A['topk']['mean'] >= 0)
    return R


def costs():
    C = {}
    js = sorted((OUT / 'mice' / 'native').glob('task_??.json')) if (OUT / 'mice' / 'native').exists() else []
    if js:
        T = [json.loads(p.read_text()) for p in js]
        V = [v for t in T for v in t['videos'].values()]
        mads = [m['mad_km1_k_kp1'] for v in V for m in (v['mad'] or [])]
        C['mice_decode'] = {'tasks': len(T), 'videos': len(V), 'frames': int(sum(v['n_got'] for v in V)),
                            'cpu_h': sum(v['decode_cpu_s'] for v in V) / 3600, 'copy_h': sum(v['copy_s'] for v in V) / 3600,
                            'max_task_wall_h': max(t['wall_s'] for t in T) / 3600,
                            'mad_checks': len(mads), 'mad_lowest_at_k': int(sum(m[1] <= min(x for x in m if x is not None) for m in mads)),
                            'tar_gb': sum(t.get('tar_gb_eval', 0) + t.get('tar_gb_train', 0) for t in T)}
    for d in DOMAINS:
        for p in sorted((OUT / d).glob('encode_train_*.json')) if (OUT / d).exists() else []:
            C[f'{d}_{p.stem}'] = json.loads(p.read_text())
        for arm in ARMS:
            parts = sorted((OUT / d / 'codes' / arm).glob('part_*/info.json')) if (OUT / d / 'codes' / arm).exists() else []
            if parts:
                I = [json.loads(p.read_text()) for p in parts]
                C[f'{d}_evaluate_{arm}'] = {'parts': len(I), 'frames': sum(i['n_frames'] for i in I),
                                            'frames_encoded': sum(i.get('n_frames_encoded', 0) for i in I),
                                            'wall_h': sum(i['wall_s'] for i in I) / 3600,
                                            'encode_gpu_h': sum(i.get('encode_gpu_s', 0) for i in I) / 3600,
                                            'gpus': sorted({i['gpu'] for i in I})}
        for arm in ARMS:
            ps = sorted((OUT / d / 'sae').glob(f'{arm}_s?/train.json')) if (OUT / d / 'sae').exists() else []
            if ps:
                C[f'{d}_train_{arm}'] = {'seeds': len(ps), 'train_h': sum(json.loads(p.read_text())['train_time_s'] for p in ps) / 3600}
            if (OUT / d / f'train_meta_{arm}.json').exists():
                C.setdefault(f'{d}_train_{arm}', {})['meta'] = json.loads((OUT / d / f'train_meta_{arm}.json').read_text())
    return C


def cmd_summary(args):
    S = collect()
    S['decision'] = decide(S)
    S['costs'] = costs()
    if (OUT / 'plan.json').exists():
        P = plan()
        S['plan'] = {'ktop': P['ktop'], 'a1': P['a1'], 'smoke': P['smoke']}
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    f = lambda x, k=3: f"{x['mean']:.{k}f} ± {x['sd']:.{k}f}"  # noqa: E731
    L = ['# RES_POOL: resolution x pooling in C1 (mean ± sd over 3 seeds)\n']
    for d in DOMAINS:
        L.append(f'\n## {d}\n')
        L.append('| label | arm | read-out | cf AUROC | cf AP | honest top-1% | size-ctrl AUROC | video r rate raw / ctl | '
                 'video r bouts raw / ctl |' + (' pooled d-r raw / ctl |' if d == 'mice' else ''))
        L.append('|---|---|---|---|---|---|---|---|---|' + ('---|' if d == 'mice' else ''))
        for b in LABELS[d]:
            for key, c in S[d].items():
                arm, ro = key.split('|')
                x = c[b]
                v, w = x['presence_mean'], x['presence_mean|fgctrl']
                row = (f"| {b} ({x['base_rate']:.4f}) | {arm} | {ro} | {f(x['cf_auroc'])} | {f(x['cf_ap'], 4)} | "
                       f"{f(x['cf_top1'])} | {f(x['cf_size_ctrl_auroc'])} | {v['video_r_rate']['mean']:.2f} / "
                       f"{w['video_r_rate']['mean']:.2f} | {v['video_r_bpm']['mean']:.2f} / {w['video_r_bpm']['mean']:.2f} |")
                if d == 'mice':
                    row += f" {v['delta_r_pooled']['mean']:.2f} / {w['delta_r_pooled']['mean']:.2f} |"
                L.append(row)
        if d == 'mice':
            L.append('\nper-switch dz of the held-out read-out mean (seed mean; annotated-rate dz and p in brackets; sign '
                     'agreement over 3 seeds), raw | fg-controlled:\n')
            for b in LABELS[d]:
                for key, c in S[d].items():
                    E0, E1 = c[b]['presence_mean']['effects'], c[b]['presence_mean|fgctrl']['effects']
                    L.append(f'- {b} {key}: ' + '; '.join(
                        f"{sw} {E0[sw]['dz']['mean']:+.2f} ({E0[sw]['truth_dz']:+.2f}, p {E0[sw]['truth_p']:.3f}) "
                        f"{E0[sw]['sign_agree_seeds']}/3 | {E1[sw]['dz']['mean']:+.2f} {E1[sw]['sign_agree_seeds']}/3"
                        for sw in E0))
        L.append('\n' + '; '.join(f"{k}: k_top {c['k_top']}, FVE {c['fve']['mean']:.4f} (static {c['fve_static']['mean']:.4f}, "
                                  f"change {c['fve_change']['mean']:.4f})" for k, c in S[d].items() if k.endswith('|max')))
        for k, c in S[d].items():
            if 'repro_vs_combined_C1' in c:
                L.append(f'\n{k} reproduction of combined C1 (max abs diff over seeds / labels): {c["repro_vs_combined_C1"]}')
    L.append('\n## Pre-registered decisions\n')
    L.append('```\n' + json.dumps(S['decision'], indent=1, default=float) + '\n```')
    L.append('\n## Costs\n')
    L.append('```\n' + json.dumps(S['costs'], indent=1, default=float) + '\n```')
    (OUT / 'tables.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['plan', 'decode', 'encode_train', 'train', 'evaluate', 'score', 'sheets', 'summary'])
    p.add_argument('--domain', default='mice', choices=list(DOMAINS))
    p.add_argument('--domains', default='mice,ants')
    p.add_argument('--arm', default='n896', choices=list(ARMS))
    p.add_argument('--arms', nargs='+', default=['c448', 'c448dep'])
    p.add_argument('--readouts', nargs='+', default=['max', 'topk'])
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--decode-tasks', type=int, default=16)
    p.add_argument('--eval-shards-mice', type=int, default=4)
    p.add_argument('--eval-shards-ants', type=int, default=2)
    a = p.parse_args()
    {'plan': cmd_plan, 'decode': cmd_decode, 'encode_train': cmd_encode_train, 'train': cmd_train,
     'evaluate': cmd_evaluate, 'score': cmd_score, 'sheets': cmd_sheets, 'summary': cmd_summary}[a.cmd](a)


if __name__ == '__main__':
    main()
