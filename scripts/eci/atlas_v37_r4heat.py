"""
Real per-patch heat maps for the atlas v37 R4 builds (scripts/eci/build_atlas_v37.py --model new), GPU.

The R4 codes store per frame and neuron only the pooled code (codes_max) and the argmax token, so the builder's patch
step (step_patch_new) fakes the heat from the argmax cell. This step recomputes the full per-token SAE codes of every
clip frame of one build with the R4 recipe itself (scripts/eci/rollout_r4.py, imported read only: decode, Pipe =
DINOv3-L/16 block 20 + lab mask, the d2 input [z_t, z_t - z_(t-10)], the kept SAE in threshold mode) and writes the
build_explorer patch files (sig 'r4heat-' + json of the clip rows, vmax, one (rows, G, G) float16 array per clip row
set). The heat shows the fnorm rule of the build's codes set <sae>_fnorm: a token's code is kept iff its fp16 value >
threshold_per_neuron[j] (the comparison write_rule_codes makes on codes_max), so the max of a heat map equals the
frame's fnorm codes_max.

Reproduction of the full pass (cmd_encode): a frame is encoded inside its ORIGINAL batch (the encode's decode chunk of
`chunk` frames, cut into `bs` frame batches; same frames, same batch shape), its t-10 frame inside that frame's own
original batch, and the SAE runs on all kept tokens of the original batch (same GEMM shape; rows of other frames do not
change a row's result, so their t-10 half is not needed and is set to zero). Each video runs on the GPU model its
encode ran on (parsed from the R4 encode logs; --gpu selects that set of videos and the job asserts its device).

Modes
  --gpu NAME [--task i --ntasks n]   encode the videos of the key whose full pass ran on GPU NAME (one of
                                     enc_gpus()), every n-th from i -> <work>/r4heat_parts/<gpu>_<i>.npz (per clip frame
                                     the unthresholded per-token codes of its neurons + check-1 stats)
  --assemble [--move]                (CPU) all parts -> checks 1-2, <work>/patch_r4heat/x<j>.npz, <work>/r4heat_report.json;
                                     --move: if the checks pass, back up patch/ peak files to patch_peak/ and install
  --dry                              print the per-GPU work (batches, frames) and exit
Checks
  1. per clip frame and clip neuron: fp16(max over tokens of the unthresholded code) vs the parent codes_max (max abs
     diff, active / inactive mismatches) and the argmax token vs the parent argmax; the same over all 512 neurons of
     these frames; the kept-token count vs the parent n_fg.
  2. max over tokens of the thresholded heat vs the fnorm codes_max: max abs diff, fire (active / inactive) agreement,
     exact fp16 match on the active pairs.
Acceptance (--move): exact (checks 1-2 zero diff, zero mismatches, argmax 100 %) OR fnorm fire agreement >= 0.999 and
exact match on >= 99.9 % of the active (row, neuron) pairs of check 2.

  KEY=ants_v3 GPU="NVIDIA L40S" sbatch ... scripts/eci/atlas_v37_r4heat.sh       (scripts/eci/atlas_v37_r4heat_key.sh
  submits every GPU part + the assemble of a key)
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LAYOUT = {'ants_v2': 'eci_r4v2', 'ants_v3': 'eci_r4v3', 'ants_v4': 'eci_r4v4', 'mice': None, 'frogs': None}
FP16_KEYS = {'ants_v2'}  # rollout_r4.FP16_DOMAINS (asserted below)

ap = argparse.ArgumentParser()
ap.add_argument('--key', required=True, choices=list(LAYOUT))
ap.add_argument('--variant', default='d2', choices=('d2', 'd0'), help='d2: x = [z_t, z_t - z_(t-10)]; d0: x = z_t')
ap.add_argument('--kr', type=int, default=8, help='clips per row rendered by the build (8 for every domain)')
ap.add_argument('--gpu', default=None, help='encode the videos whose full pass ran on this GPU model')
ap.add_argument('--task', type=int, default=0)
ap.add_argument('--ntasks', type=int, default=1)
ap.add_argument('--par', type=int, default=6, help='decode threads (videos in flight)')
ap.add_argument('--gap', type=int, default=8, help='merge needed batches of a video into one decode run when at most '
                                                   'this many frames apart')
ap.add_argument('--stage-gb', type=float, default=60.0, help='stage the part\'s source videos to $LOCAL_DIR when they '
                                                            'total at most this (else ffmpeg seeks them on NFS)')
ap.add_argument('--assemble', action='store_true')
ap.add_argument('--move', action='store_true')
ap.add_argument('--dry', action='store_true')
A = ap.parse_args()

if A.key in FP16_KEYS:
    os.environ['R4_AMP'] = 'fp16'
else:
    os.environ.pop('R4_AMP', None)
if LAYOUT[A.key]:
    os.environ['ANTS_ECI'] = LAYOUT[A.key]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import rollout_r4 as R4  # noqa: E402
import build_explorer as BE  # noqa: E402

log = R4.log
OUT37 = REPO / 'results/vision/eci_atlas_v37'


def enc_gpus(dom):
    """obs -> GPU model of the encode job that wrote it (R4 encode logs: 'GPU 0: <name> (UUID' + per video lines)."""
    out = {}
    for f in sorted(glob.glob(str(R4.OUT / 'logs' / '*')), key=os.path.getmtime):
        t = open(f, errors='replace').read()
        m = re.search(r'step encode domain (\S+)', t)
        if not m or m.group(1) != dom:
            continue
        g = re.search(r'^GPU 0: (.+?) \(UUID', t, re.M)
        for mm in re.finditer(rf'^\d\d:\d\d:\d\d {re.escape(dom)} (\S+) \((?:train|calib|eval)\): \d+ frames', t, re.M):
            out[mm.group(1)] = g.group(1) if g else None  # the latest log of a video wins (mtime order)
    return out


def gtag(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name.replace('NVIDIA ', '')).strip('_')


def batch_of(k, n, chunk, bs):
    """(first frame, length) of the full-pass batch holding 5 fps frame k of an n-frame video (cmd_encode)."""
    k0 = (k // chunk) * chunk
    m_ = min(chunk, n - k0)
    b = ((k - k0) // bs) * bs
    return k0 + b, min(bs, m_ - b)


def setup():
    dom = A.key
    assert (dom in R4.FP16_DOMAINS) == (dom in FP16_KEYS), R4.FP16_DOMAINS
    R4.check_amp(dom)
    c = R4.DOMS[dom]
    P = R4.load_plan(dom)
    pre_amp = json.loads((R4.OUT / dom / 'prepass.json').read_text()).get('amp', 'bf16')
    assert pre_amp == R4.AMP, (pre_amp, R4.AMP)
    best = json.loads((R4.OUT / dom / 'sae' / 'best.json').read_text())[A.variant]
    name = best['name']
    fname = f'{name}_fnorm'
    work = OUT37 / 'res' / f'{dom}_{A.variant}' / fname / '_cache' / 'explorer'
    cdir = REPO / c['eci'] / 'codes' / fname
    cfg = json.loads((cdir / 'config.json').read_text())
    parent = Path(cfg['parent_codes'])
    assert cfg['amp'] == R4.AMP and cfg['variant'] == A.variant and cfg['lag'] == (R4.LAG if A.variant == 'd2' else None), cfg
    G = int(cfg['G'])
    assert G == c['S'] // 16
    picks = json.loads((work / 'picks.json').read_text())
    neurons = picks['by_key'].get('max', {})
    if not neurons:
        raise SystemExit(f'{work}/picks.json: no max neurons')
    js = sorted(int(j) for j in neurons)
    RR = {j: BE.clip_rows(neurons[str(j)], A.kr, BE.SHOW_LENGTHS) for j in js}
    need = {}  # layout row -> neurons whose rendered clips show it
    for j, rr in RR.items():
        for rows in rr.values():
            for r in rows:
                need.setdefault(int(r), set()).add(j)
    U = np.array(sorted(need), np.int64)
    vids = P['videos']
    lay_lo = np.array([v['lay_lo'] for v in vids], np.int64)
    assert (np.diff(lay_lo) > 0).all()
    vi_of = np.searchsorted(lay_lo, U, side='right') - 1
    k_of = U - lay_lo[vi_of]
    assert (k_of < np.array([v['n'] for v in vids])[vi_of]).all()
    per_video = {}
    for r, vi, k in zip(U, vi_of, k_of):
        per_video.setdefault(int(vi), []).append((int(k), int(r)))
    return dict(dom=dom, c=c, P=P, best=best, name=name, fname=fname, work=work, cdir=cdir, cfg=cfg, parent=parent, G=G,
                picks=picks, js=js, RR=RR, need=need, U=U, vi_of=vi_of, k_of=k_of, per_video=per_video)


def video_batches(c, v, clip):
    """clip [(k, row)] of one video -> (sorted needed batches [(kb, len)], clip batches set, needed prev frames set)."""
    n = v['n']
    cb = {batch_of(k, n, c['chunk'], c['bs']) for k, _ in clip}
    prev = {max(k - R4.LAG, 0) for k, _ in clip} if A.variant == 'd2' else set()  # d0: no t-10 frame
    pb = {batch_of(k, n, c['chunk'], c['bs']) for k in prev}
    return sorted(cb | pb), cb, prev


def runs_of(batches, gap):
    """sorted [(kb, len)] -> decode runs [(a, b)] inclusive (batches at most gap frames apart merged)."""
    out = []
    for kb, ln in batches:
        if out and kb - out[-1][1] - 1 <= gap:
            out[-1][1] = max(out[-1][1], kb + ln - 1)
        else:
            out.append([kb, kb + ln - 1])
    return [tuple(x) for x in out]


def plan_gpu(S):
    eg = enc_gpus(S['dom'])
    vids = S['P']['videos']
    miss = [vids[vi]['obs'] for vi in S['per_video'] if eg.get(vids[vi]['obs']) is None]
    if miss:
        raise SystemExit(f'{S["dom"]}: no encode GPU known for {len(miss)} clip videos, e.g. {miss[:5]}')
    work = {}
    for vi, clip in S['per_video'].items():
        bt, _, _ = video_batches(S['c'], vids[vi], clip)
        work.setdefault(eg[vids[vi]['obs']], []).append((vi, len(bt), sum(ln for _, ln in bt)))
    return eg, work


def gpu_part(S):
    import torch
    dom, c, P, G = S['dom'], S['c'], S['P'], S['G']
    GG, W = G * G, R4.WIDTH
    vids = P['videos']
    dev = torch.device('cuda')
    gname = torch.cuda.get_device_name(0)
    if gname != A.gpu:
        raise SystemExit(f'device {gname!r} is not the requested encode GPU {A.gpu!r}')
    eg, work = plan_gpu(S)
    mine = sorted(work.get(A.gpu, []), key=lambda x: -x[2])
    load_ = np.zeros(A.ntasks)
    sel = []
    for vi, nb, nf in mine:  # balanced split by frames (largest first)
        t = int(np.argmin(load_))
        load_[t] += nf
        if t == A.task:
            sel.append(vi)
    sel = sorted(sel)
    log(f'{dom} part {A.gpu} {A.task}/{A.ntasks}: {len(sel)} videos, {int(load_[A.task])} frames to encode')
    pmax = np.load(S['parent'] / 'codes_max.npy', mmap_mode='r')
    parg = np.load(S['parent'] / 'argmax.npy', mmap_mode='r')
    pnfg = np.load(S['parent'] / 'n_fg.npy', mmap_mode='r')
    src_ann, Rsrc = R4.obs_ranges_of(REPO / c['ann'], ('frame_path',))
    src_fp = src_ann['frame_path'].values
    lay_fp = pd.read_csv(S['cfg']['layout_ann'], usecols=['frame_path'])['frame_path'].values
    VB = {}
    for vi in sel:
        v = vids[vi]
        assert Rsrc[v['obs']] == (v['src_lo'], v['src_lo'] + v['n']), v['obs']
        for k, r in S['per_video'][vi]:
            assert lay_fp[r] == S['picks']['frame_path'][str(r)] == src_fp[v['src_lo'] + k], (v['obs'], k, r)
        VB[vi] = video_batches(c, v, S['per_video'][vi])
    T0 = time.time()
    loc = R4.local_dir()
    rels = sorted({src_fp[vids[vi]['src_lo'] + kb + i] for vi in sel for kb, ln in VB[vi][0] for i in range(ln)})
    R4.stage_jpgs(rels, loc / 'jpg')
    paths = sorted({vids[vi]['path'] for vi in sel})
    tot_gb = sum(os.path.getsize(p) for p in paths) / 1e9
    free_gb = shutil.disk_usage(loc).free / 1e9
    staged = tot_gb <= A.stage_gb and tot_gb < 0.5 * free_gb
    if staged:
        (loc / 'vid').mkdir(parents=True, exist_ok=True)
        vmap = {p: str(loc / 'vid' / f'{i:04d}_{Path(p).name}') for i, p in enumerate(paths)}
        t0 = time.time()
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda kv: shutil.copyfile(*kv), vmap.items()))
        log(f'  staged {len(paths)} videos ({tot_gb:.2f} GB) in {time.time() - t0:.0f}s')
    else:
        vmap = {p: p for p in paths}
        log(f'  videos not staged ({tot_gb:.1f} GB, {free_gb:.0f} GB free, cap {A.stage_gb}): ffmpeg seeks on NFS')
    t_stage = time.time() - T0

    torch.backends.cuda.matmul.allow_tf32 = False  # as rollout_r4 cmd_encode
    pipe = R4.Pipe(dom, dev, Rsrc)
    from src.eci.sae import load_sae
    sae, norm, _ = load_sae(R4.OUT / dom / 'sae' / S['name'] / 'sae.pt', dev)
    assert abs(float(sae.threshold) - S['best']['theta_star']) < 1e-6
    dec_s, n_dec = [0.0], [0]

    def load(vi):
        t0 = time.time()
        v = vids[vi]
        fr = {}
        for a, b in runs_of(VB[vi][0], A.gap):
            x = R4.decode(vmap[v['path']], R4.voff(c, v) + c['step'] * a, b - a + 1, c['step'], c['S'], v['hflip'],
                          c['fps'], threads=4)
            n_dec[0] += len(x)
            out = []
            for kb, ln in VB[vi][0]:
                if a <= kb <= b:
                    fr[kb] = x[kb - a:kb - a + ln]
                    assert len(fr[kb]) == ln
            del out
        jp = {kb: R4.load_jpgs([loc / 'jpg' / src_fp[v['src_lo'] + kb + i] for i in range(ln)]) for kb, ln in VB[vi][0]}
        dec_s[0] += time.time() - t0
        return fr, jp

    S1 = {'pairs': 0, 'max_abs_diff': 0.0, 'mism_active': 0, 'arg_agree': 0, 'arg_active_pairs': 0,
          'arg_agree_active': 0, 'exact_pairs': 0, 'all_cells': 0, 'all_max_abs_diff': 0.0, 'all_mism_active': 0,
          'all_arg_agree_active': 0, 'all_active': 0, 'all_exact': 0, 'nfg_exact': 0, 'nfg_absdiff_max': 0,
          'frames': 0}
    out_rows, out_js, out_heat = [], [], []
    t_gpu, n_enc = 0.0, 0
    with ThreadPoolExecutor(A.par) as ex:
        futs, nxt = {}, 0
        for i, vi in enumerate(sel):
            while nxt < len(sel) and nxt < i + 2 * A.par:
                futs[sel[nxt]] = ex.submit(load, sel[nxt])
                nxt += 1
            fr, jp = futs.pop(vi).result()
            v = vids[vi]
            batches, cb, prev = VB[vi]
            clip = dict(S['per_video'][vi])  # k -> row
            tg = time.time()
            store = {}  # prev frame k -> its tokens (G*G, 1024) fp16, as encoded in its own full-pass batch
            for kb, ln in batches:
                ks = np.arange(kb, kb + ln)
                z, m, _, _ = pipe.run(fr[kb], v['src_lo'] + ks, jp[kb])
                n_enc += ln
                for t, k in enumerate(ks):
                    if int(k) in prev:
                        store[int(k)] = z[t].clone()
                if (kb, ln) not in cb:
                    continue
                Pv = z.clone()  # rows of non-clip frames: t-10 half = 0 (their values do not change clip rows)
                ci = [t for t, k in enumerate(ks) if int(k) in clip]
                for t in ci if A.variant == 'd2' else ():
                    Pv[t] = store[max(int(ks[t]) - R4.LAG, 0)]
                with torch.no_grad():
                    bi, pi = m.nonzero(as_tuple=True)
                    x = z[bi, pi]
                    if A.variant == 'd2':  # sae_codes
                        x = torch.cat([x, (x.float() - Pv[bi, pi].float()).half()], 1)
                    zz = sae.encode(norm(x.float()), mode='threshold') if len(x) else \
                        torch.zeros((0, W), device=dev)
                    full = torch.zeros((ln, GG, W), dtype=torch.float32, device=dev)
                    full[bi, pi] = zz
                    full = full[ci]
                    mx = full.amax(1)
                    hit = (full == mx[:, None]) & (full > 0)
                    cand = torch.where(hit, torch.arange(GG, device=dev)[None, :, None], GG)
                    am = cand.amin(1)
                    am[am == GG] = -1
                rows = np.array([clip[int(ks[t])] for t in ci])
                st_mx = np.asarray(pmax[rows], np.float32)  # rows increasing within a batch
                st_am = np.asarray(parg[rows], np.int64)
                my_mx = mx.half().float().cpu().numpy()
                my_am = am.cpu().numpy()
                nfg = m[ci].sum(1).cpu().numpy()
                st_nfg = np.asarray(pnfg[rows], np.int64)
                B = len(ci)
                S1['frames'] += B
                S1['nfg_exact'] += int((nfg == st_nfg).sum())
                S1['nfg_absdiff_max'] = max(S1['nfg_absdiff_max'], int(np.abs(nfg - st_nfg).max()))
                act_me, act_st = my_mx > 0, st_mx > 0
                both = act_me & act_st
                S1['all_cells'] += my_mx.size
                S1['all_max_abs_diff'] = max(S1['all_max_abs_diff'], float(np.abs(my_mx - st_mx).max()))
                S1['all_mism_active'] += int((act_me != act_st).sum())
                S1['all_exact'] += int((my_mx == st_mx).sum())
                S1['all_active'] += int(both.sum())
                S1['all_arg_agree_active'] += int((my_am[both] == st_am[both]).sum())
                for t, r in enumerate(rows):
                    jj = sorted(S['need'][int(r)])
                    S1['pairs'] += len(jj)
                    S1['max_abs_diff'] = max(S1['max_abs_diff'], float(np.abs(my_mx[t, jj] - st_mx[t, jj]).max()))
                    S1['exact_pairs'] += int((my_mx[t, jj] == st_mx[t, jj]).sum())
                    S1['mism_active'] += int((act_me[t, jj] != act_st[t, jj]).sum())
                    S1['arg_agree'] += int((my_am[t, jj] == st_am[t, jj]).sum())
                    bb = both[t, jj]
                    S1['arg_active_pairs'] += int(bb.sum())
                    S1['arg_agree_active'] += int((my_am[t, jj][bb] == st_am[t, jj][bb]).sum())
                    out_rows += [int(r)] * len(jj)
                    out_js += jj
                    out_heat.append(full[t][:, jj].T.half().cpu().numpy())  # (len(jj), GG)
                for k in [k for k in store if k < kb + ln - R4.LAG]:  # no later clip frame needs it
                    del store[k]
            del store
            torch.cuda.synchronize()
            t_gpu += time.time() - tg
            if (i + 1) % max(1, len(sel) // 10) == 0 or i + 1 == len(sel):
                log(f'  {i + 1}/{len(sel)} videos, {S1["frames"]} clip frames, {n_enc} frames encoded, '
                    f'{time.time() - T0:.0f}s')
    wall = time.time() - T0
    part = {'key': dom, 'gpu': A.gpu, 'task': A.task, 'ntasks': A.ntasks, 'videos': len(sel), 'S1': S1,
            'frames_encoded': n_enc, 'frames_decoded': n_dec[0], 'videos_staged': staged,
            'time_s': {'stage': t_stage, 'total': wall, 'gpu': t_gpu, 'decode_thread_s': dec_s[0], 'pipe_gpu': pipe.gpu_s},
            'fps_wall': n_enc / max(wall - t_stage, 1e-9), 'job': os.environ.get('SLURM_JOB_ID')}
    pd_ = S['work'] / 'r4heat_parts'
    pd_.mkdir(parents=True, exist_ok=True)
    f = pd_ / f'{gtag(A.gpu)}_{A.task}of{A.ntasks}.npz'
    tmp = f.with_name(f.stem + '.tmp.npz')
    np.savez(tmp, rows=np.array(out_rows, np.int64), js=np.array(out_js, np.int64),
             heat=np.concatenate(out_heat) if out_heat else np.zeros((0, GG), np.float16), info=json.dumps(part))
    os.replace(tmp, f)
    log(f'{dom} part {A.gpu} {A.task}/{A.ntasks} check 1: {json.dumps(S1)}')
    log(f'{dom} part: {n_enc} frames encoded ({n_dec[0]} decoded) in {wall - t_stage:.0f}s '
        f'({part["fps_wall"]:.1f} fps), stage {t_stage:.0f}s, total {wall:.0f}s, {gname} -> {f}')


def assemble(S):
    dom, G, js, RR, work = S['dom'], S['G'], S['js'], S['RR'], S['work']
    eg, wk = plan_gpu(S)
    files = sorted((work / 'r4heat_parts').glob('*.npz'))
    files = [f for f in files if '.tmp' not in f.name]
    parts = [np.load(f) for f in files]
    infos = [json.loads(str(p['info'])) for p in parts]
    want = {g: max(i['ntasks'] for i in infos if i['gpu'] == g) if any(i['gpu'] == g for i in infos) else None
            for g in wk}
    for g, n in want.items():
        got = sorted(i['task'] for i in infos if i['gpu'] == g)
        if n is None or got != list(range(n)):
            raise SystemExit(f'{dom}: parts of GPU {g}: tasks {got} of {n}')
    heat = {}
    for p in parts:
        for r, j, h in zip(p['rows'], p['js'], p['heat']):
            key = (int(r), int(j))
            if key in heat:
                raise SystemExit(f'{dom}: pair {key} in two parts')
            heat[key] = h
    n_need = sum(len(s) for s in S['need'].values())
    if len(heat) != n_need:
        raise SystemExit(f'{dom}: {len(heat)} of {n_need} frame-neuron pairs in the parts')
    S1 = {}
    for i in infos:
        for k, x in i['S1'].items():
            S1[k] = max(S1.get(k, 0), x) if 'max' in k else S1.get(k, 0) + x
    c1 = {**S1, 'arg_agree_frac': S1['arg_agree'] / max(S1['pairs'], 1),
          'arg_agree_active_frac': S1['arg_agree_active'] / max(S1['arg_active_pairs'], 1),
          'exact_frac': S1['exact_pairs'] / max(S1['pairs'], 1),
          'all_arg_agree_active_frac': S1['all_arg_agree_active'] / max(S1['all_active'], 1),
          'all_exact_frac': S1['all_exact'] / max(S1['all_cells'], 1),
          'nfg_exact_frac': S1['nfg_exact'] / max(S1['frames'], 1)}
    thr = np.load(S['cdir'] / 'threshold_per_neuron.npy').astype(np.float32)
    fmax = np.load(S['cdir'] / 'codes_max.npy', mmap_mode='r')
    outd = work / 'patch_r4heat'
    outd.mkdir(parents=True, exist_ok=True)
    S2 = {'pairs': 0, 'max_abs_diff': 0.0, 'fire_mism': 0, 'active_pairs': 0, 'active_exact': 0, 'exact': 0}
    for j in js:
        rr = RR[j]
        ur = sorted({int(r) for rows in rr.values() for r in rows})
        H16 = np.stack([heat[(r, j)] for r in ur])  # (n, GG) fp16 unthresholded
        Ht = np.where(H16.astype(np.float32) > thr[j], H16, np.float16(0))
        got = Ht.max(1).astype(np.float32)
        ref = np.asarray(fmax[np.array(ur), j], np.float32)
        act = (got > 0) | (ref > 0)
        S2['pairs'] += len(ur)
        S2['max_abs_diff'] = max(S2['max_abs_diff'], float(np.abs(got - ref).max()))
        S2['fire_mism'] += int(((got > 0) != (ref > 0)).sum())
        S2['active_pairs'] += int(act.sum())
        S2['active_exact'] += int((got[act] == ref[act]).sum())
        S2['exact'] += int((got == ref).sum())
        posv = Ht[Ht > 0].astype(np.float32)
        vmax = float(np.quantile(posv, 0.99)) if len(posv) else 1.0
        lp = {r: t for t, r in enumerate(ur)}
        arrs = {k: Ht[[lp[int(r)] for r in rows]].reshape(len(rows), G, G) for k, rows in rr.items()}
        np.savez(outd / f'{BE.pre("max")}{j:04d}.npz', sig='r4heat-' + json.dumps(rr), vmax=vmax, **arrs)
    S2['fire_agreement'] = 1 - S2['fire_mism'] / max(S2['pairs'], 1)
    S2['active_exact_frac'] = S2['active_exact'] / max(S2['active_pairs'], 1)
    exact = (c1['max_abs_diff'] == 0 and c1['mism_active'] == 0 and c1['arg_agree'] == c1['pairs'] and
             S2['max_abs_diff'] == 0 and S2['fire_mism'] == 0)
    ok = exact or (S2['fire_agreement'] >= 0.999 and S2['active_exact_frac'] >= 0.999)
    rep = {'key': dom, 'sae': S['fname'], 'parent': str(S['parent']), 'G': G, 'amp': R4.AMP, 'kr': A.kr,
           'neurons': len(js), 'clip_frames': len(S['U']), 'pairs': n_need, 'check1': c1, 'check2': S2,
           'exact': bool(exact), 'pass': bool(ok), 'parts': infos,
           'frames_encoded': sum(i['frames_encoded'] for i in infos),
           'gpu_videos': {g: len(v) for g, v in wk.items()}, 'out': str(outd), 'moved': False}
    log(f'{dom} check 1 (unthresholded vs parent codes_max / argmax): {json.dumps(c1)}')
    log(f'{dom} check 2 (fnorm heat max vs fnorm codes_max): {json.dumps(S2)}')
    log(f'{dom}: exact {exact}, pass {ok}; parts ' + '; '.join(
        f'{i["gpu"]} {i["task"]}/{i["ntasks"]}: {i["frames_encoded"]} frames {i["time_s"]["total"]:.0f}s '
        f'({i["fps_wall"]:.1f} fps)' for i in infos))
    if A.move:
        if not ok:
            (work / 'r4heat_report.json').write_text(json.dumps(rep, indent=1))
            raise SystemExit(f'{dom}: checks did not pass, patch/ NOT replaced (files left in {outd})')
        bak = work / 'patch_peak'
        bak.mkdir(exist_ok=True)
        (work / 'patch').mkdir(exist_ok=True)
        for j in js:
            fn = f'{BE.pre("max")}{j:04d}.npz'
            if (work / 'patch' / fn).exists() and str(np.load(work / 'patch' / fn)['sig']).startswith('peak-'):
                shutil.copyfile(work / 'patch' / fn, bak / fn)
        for j in js:
            fn = f'{BE.pre("max")}{j:04d}.npz'
            os.replace(outd / fn, work / 'patch' / fn)
        n_real = sum(str(np.load(work / 'patch' / f'{BE.pre("max")}{j:04d}.npz')['sig']) == 'r4heat-' + json.dumps(RR[j])
                     for j in js)
        assert n_real == len(js), f'{n_real}/{len(js)} r4heat files in patch/ after the move'
        rep['moved'] = True
        log(f'{dom}: patch/ replaced ({n_real} r4heat files; peak-patch files backed up in {bak})')
    (work / 'r4heat_report.json').write_text(json.dumps(rep, indent=1))


def main():
    S = setup()
    if A.dry:
        eg, wk = plan_gpu(S)
        log(f'{S["dom"]}: {len(S["js"])} neurons, {len(S["U"])} clip frames, '
            f'{sum(len(s) for s in S["need"].values())} pairs in {len(S["per_video"])} videos')
        for g, v in wk.items():
            log(f'  {g}: {len(v)} videos, {sum(x[1] for x in v)} batches, {sum(x[2] for x in v)} frames')
        print('GPUS=' + json.dumps({g: sum(x[2] for x in v) for g, v in wk.items()}))
        print('WORK=' + str(S['work']))
        return
    if A.assemble:
        assemble(S)
    else:
        if not A.gpu:
            raise SystemExit('--gpu NAME, --assemble or --dry')
        gpu_part(S)


if __name__ == '__main__':
    main()
