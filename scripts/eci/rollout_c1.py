"""
C1 ROLLOUT: the final ECI behaviour representation (decided by the pre-registered combined test, scripts/eci/combined.py,
results/vision/eci_combined/summary.json) deployed on every domain, then NES on it.

Pipeline C1 (per domain): 5 fps frames -> DINOv2-base at 448 (32 x 32 patches, src/eci/foreground.py FgEncoder) ->
TODAY's foreground mask (mice 'fg448' unaligned, ants 'ants', frogs 'frogs') -> SAE input M5 = [token_t, token_t -
token_{t-5}] at the same patch (5 frames = 1 s earlier, clipped to the video's first frame), TokenNorm.fit_blocks(
[768, 768]) -> Matryoshka BatchTopK SAE, 4096 latents, prefixes 512/1024/2048/4096, k = 16, the levers / T3 recipe
(scripts/eci/motion_t3.py train_one: Adam 5e-4, 500 warmup, cosine, 6000 steps x 4096 tokens, AuxK, b_dec = geometric
median) -> frame read-out codes_max (max over the mask patches) + codes_mean + n_fg for EVERY 5 fps frame, in the
existing codes layout (<domain eci dir>/codes/<sae>/, src/eci/fg_encode.py), so NES / atlas code runs unchanged.

DEPLOYMENT SAE (recorded choice): ONE model per domain, seed 0, trained on a token pool drawn uniformly without
replacement (pool seed 0) from ALL tokens of ALL videos of the domain's 1 fps motion store (every 5th frame, seeded
offset per video, scripts/eci/fg_extract_train.py --motion-delta 5): mice dinov2_base_l-1_fg448_fps1_d5 (432 videos),
ants dinov2_base_l-1_antsfg_fps1_d5 (256 videos), frogs dinov2_base_l-1_frogsfg_fps1_d5 (35 videos, built here with
the frogs rule at stride 5 like the other domains). Pool size and steps = the validated recipe: mice 4M, ants 2M
tokens (sae_levers.CAP); frogs 2M (no validated frogs recipe: the ants cap, the smaller of the two). No labels.
SAE tags: mice fg448m5, ants antsfgm5, frogs frogsfgm5 -> matryoshka_btk_4096_k16_<tag>_s0.
Training FVE / dead fraction are also measured on 1M held-out store tokens (uniform, not in the pool, seed 1).

Steps (scripts/eci/rollout_c1.sh runs each in Slurm; every heavy input is staged to $LOCAL_DIR first)
    plan      (login safe) cost estimate from the frame counts and measured throughputs -> OUT/plan.json
    tokens    (GPU, frogs only, one task per video) stage the task's JPGs + annotations.csv locally, run
              fg_extract_train.py (--rule frogs --stride 5 --motion-delta 5) on the local copy, copy the shard back
    train     (GPU) pool gather (pread of the selected tokens from the NFS store), training, held-out FVE / dead,
              -> <eci>/sae/<sae>/{sae.pt, train.json, metrics.json}; writes <eci>/codes/<sae>/config.json
    encode    (GPU, array over 24 shards) stage the shard's JPGs (+ the 5 frames before it) locally, encode_fg_shard
              on the local copy, copy the shard back to <codes>/<sae>/shards/shard_XX
    merge     (GPU for the recompute) fg_encode_all.py merge; verify_fg_codes with 100 random frames recomputed from
              JPEG; n_fg compared on EVERY frame with an existing codes set of the same mask rule (mice fg448mot,
              ants antsfgmot, frogs frogsfg); shards deleted after a pass -> verify.json
    align     (GPU, mice / ants) in-sample label alignment (the SAE saw every video; neuron selection cross-fitted
              over video halves as sae_levers.score_model) of the new SAE and the deployed SAE (mice fg448al, ants
              antsfg) on the same labelled 1 fps frames (fps1 rows of every labelled video) -> OUT/<d>/align.json
    levels    (GPU, mice / ants) two-axis level map of every neuron on the C1 evaluation frames (combined.py's C1
              input path) + presence / extent scores there (in-sample) -> OUT/<d>/levels.csv, levels_eval.json
    summary   (login safe) NES selections per domain / contrast vs the deployed atlas SAE -> OUT/summary.json, .md

Outputs: dataset/<..>/eci/{sae,codes}/matryoshka_btk_4096_k16_<tag>_s0/, results/vision/eci_rollout_c1/
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from src.eci.domain import get_domain  # noqa: E402
from src.eci.foreground import obs_rows  # noqa: E402

OUT = REPO / 'results/vision/eci_rollout_c1'
D, WIDTH, K, DELTA, N_SHARDS = 768, 4096, 16, 5, 24
DOMAINS = ('mice', 'ants', 'frogs')
TAG = {'mice': 'fg448m5', 'ants': 'antsfgm5', 'frogs': 'frogsfgm5'}
RULE = {'mice': 'fg448', 'ants': 'ants', 'frogs': 'frogs'}
STORE = {'mice': 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1_d5',
         'ants': 'dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfg_fps1_d5',
         'frogs': 'dataset/frogs/eci/train_tokens/dinov2_base_l-1_frogsfg_fps1_d5'}
CAP = {'mice': 4_000_000, 'ants': 2_000_000, 'frogs': 2_000_000}
N_HELDOUT = 1_000_000
DEPLOYED = {'mice': 'matryoshka_btk_1024_k16_fg448al_s0', 'ants': 'matryoshka_btk_1024_k16_antsfg_s0',
            'frogs': 'matryoshka_btk_1024_k16_frogsfg_s0'}
SAME_MASK = {'mice': 'matryoshka_btk_1024_k16_fg448mot_s0', 'ants': 'matryoshka_btk_1024_k16_antsfgmot_s0',
             'frogs': 'matryoshka_btk_1024_k16_frogsfg_s0'}  # existing codes with the same mask rule (n_fg check)
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def sae_name(d):
    return f'matryoshka_btk_{WIDTH}_k{K}_{TAG[d]}_s0'


def sae_dir(d):
    return get_domain(d).eci_dir / 'sae' / sae_name(d)


def codes_dir(d, name=None):
    return get_domain(d).codes_root / (name or sae_name(d))


def local_dir():
    d = os.environ.get('LOCAL_DIR')
    if not d:
        raise SystemExit('LOCAL_DIR is not set (the job staging dir /localhome/$USER/$SLURM_JOB_ID)')
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def frame_paths(d):
    return pd.read_csv(get_domain(d).ann_path, usecols=['frame_path'])['frame_path'].values


def stage_jpgs(rels, ds_loc, workers=16):
    """Copy dataset-relative JPG paths from REPO/dataset to ds_loc (same relative layout)."""
    t0 = time.time()
    for p in sorted({str(Path(r).parent) for r in rels}):
        (ds_loc / p).mkdir(parents=True, exist_ok=True)
    src = REPO / 'dataset'
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda r: shutil.copyfile(src / r, ds_loc / r), rels, chunksize=64))
    dt = time.time() - t0
    log(f'  staged {len(rels):,} JPGs in {dt:.0f}s ({len(rels) / max(dt, 1e-9):.0f} files/s)')


def copy_back(src_dir, dst_dir):
    """Local finished shard dir -> NFS dst_dir (via dst_dir.tmp + rename, so a partial copy is never DONE)."""
    tmp = Path(str(dst_dir) + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    t0 = time.time()
    shutil.copytree(src_dir, tmp)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    tmp.rename(dst_dir)
    log(f'  copied {src_dir} -> {dst_dir} in {time.time() - t0:.0f}s')


def pread_gather(path, idx, dim, workers=32, chunk=20_000):
    """Rows idx (sorted int64) of a raw float16 (n, dim) file, read with os.pread of contiguous runs (only the selected
    bytes cross NFS) -> np.ndarray (len(idx), dim) float16."""
    out = np.empty((len(idx), dim), np.float16)
    rb = dim * 2
    fd = os.open(path, os.O_RDONLY)

    def work(a):
        ii = idx[a:a + chunk]
        brk = np.flatnonzero(np.diff(ii) != 1) + 1
        for s, e in zip(np.r_[0, brk], np.r_[brk, len(ii)]):
            n = int(e - s)
            buf = os.pread(fd, n * rb, int(ii[s]) * rb)
            if len(buf) != n * rb:
                raise IOError(f'{path}: short read at row {ii[s]}')
            out[a + s:a + e] = np.frombuffer(buf, np.float16).reshape(n, dim)
    try:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, range(0, len(idx), chunk)))
    finally:
        os.close(fd)
    return out


# ---------------------------------------------------------------------------------------------- plan
def cmd_plan(args):
    """Cost estimate (login safe: reads annotations row counts from existing codes configs and store shard.json)."""
    P = {'throughput_frames_per_s_per_gpu': {'conservative (orchestrator)': 23.0,
                                             'measured fg448al full encode 2026-10 (median of 24 shards)': None},
         'domains': {}}
    logs = sorted((REPO / 'logs').glob('eci_fg_enc_208776_*.out'))
    fps = []
    for f in logs:
        for line in f.read_text().splitlines():
            if line.startswith('Done rows') and 'frames/s' in line:
                fps.append(float(line.split('(')[-1].split()[0]))
    if fps:
        P['throughput_frames_per_s_per_gpu']['measured fg448al full encode 2026-10 (median of 24 shards)'] = float(np.median(fps))
    tot = {'frames': 0, 'gpu_h_conservative': 0.0, 'gpu_h_measured': 0.0, 'storage_gb': 0.0}
    meas = float(np.median(fps)) if fps else 23.0
    for d in DOMAINS:
        c = json.loads((codes_dir(d, DEPLOYED[d]) / 'config.json').read_text())
        n = int(c['n_rows'])
        enc_c, enc_m = n / 23.0 / 3600, n / meas / 3600
        tok = 0.0
        if d == 'frogs':  # stride-5 store with motion: 1/5 of the frames, each encoded twice (frame t and t - 5)
            tok = n / 5 * 2 / 23.0 / 3600
        gb = n * WIDTH * 2 * 2 / 1e9 + n * 2 / 1e9
        P['domains'][d] = {'frames': n, 'encode_gpu_h_conservative': enc_c, 'encode_gpu_h_measured': enc_m,
                           'token_store_gpu_h_conservative': tok, 'train_gpu_h': 0.25, 'verify_align_gpu_h': 0.5,
                           'codes_storage_gb': gb, 'shard_storage_peak_gb': gb}
        tot['frames'] += n
        tot['gpu_h_conservative'] += enc_c + tok + 0.75
        tot['gpu_h_measured'] += enc_m + tok * 23.0 / meas + 0.75
        tot['storage_gb'] += gb
    tot['cpu_h_nes'] = 'about 1 h per domain and runner on 8 cores (first run streams codes_max / codes_mean once): ' \
                       '3 domains x 3 runners x (1-2 sets) x 8 cores <= ~100 core-h'
    P['total'] = tot
    P['caps'] = {'gpu_h': 150, 'cpu_h': 300}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'plan.json').write_text(json.dumps(P, indent=1))
    print(json.dumps(P, indent=1))


# ---------------------------------------------------------------------------------------------- tokens (frogs)
def cmd_tokens(args):
    """Frogs 1 fps motion store, one task = one video: stage, extract with fg_extract_train.py, copy back."""
    from fg_extract_train import fps1_rows
    d = args.domain
    dom = get_domain(d)
    store = REPO / STORE[d]
    shard = store / 'shards' / f'shard_{args.task:02d}'
    if (shard / 'DONE').exists():
        log(f'[SKIP] {shard}')
        return
    loc = local_dir()
    ds_loc = loc / 'dataset'
    ranges = obs_rows(dom.ann_path)
    sel = fps1_rows(ranges, 5, 0)
    ids = sorted(ranges)[args.task::args.n_tasks]
    rows = np.concatenate([sel[o] for o in ids])
    lo = np.concatenate([np.full(len(sel[o]), ranges[o][0]) for o in ids])
    need = np.union1d(rows, np.maximum(rows - DELTA, lo))
    (ds_loc / dom.ann_rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dom.ann_path, ds_loc / dom.ann_rel)
    log(f'task {args.task}: videos {ids}, {len(rows):,} frames, {len(need):,} JPGs to stage')
    stage_jpgs(list(frame_paths(d)[need]), ds_loc)
    cmd = [sys.executable, '-u', str(REPO / 'scripts/eci/fg_extract_train.py'), '--domain', d, '--dataset-dir',
           str(ds_loc), '--bg-dir', str(dom.eci_dir / 'fg448' / 'background'), '--out-dir', str(loc / 'store'),
           '--task', str(args.task), '--n-tasks', str(args.n_tasks), '--rule', RULE[d], '--stride', '5',
           '--motion-delta', str(DELTA), '--num-workers', str(args.num_workers)]
    log(' '.join(cmd))
    subprocess.run(cmd, check=True)
    (store / 'shards').mkdir(parents=True, exist_ok=True)
    copy_back(loc / 'store' / 'shards' / f'shard_{args.task:02d}', shard)


# ---------------------------------------------------------------------------------------------- train
def gather_pool(st, sel, shard_start):
    """(len(sel), 1536) fp16 torch tensor [token, token - prev] of global token indices sel (sorted)."""
    import torch
    X = torch.empty((len(sel), 2 * D), dtype=torch.float16)
    o = 0
    for s in range(len(st.dirs)):
        g0, g1 = shard_start[s], shard_start[s + 1]
        loc_i = sel[(sel >= g0) & (sel < g1)] - g0
        if not len(loc_i):
            continue
        t0 = time.time()
        tok = torch.from_numpy(pread_gather(st.dirs[s] / 'tokens.f16', loc_i, D))
        prv = torch.from_numpy(pread_gather(st.dirs[s] / 'prev.f16', loc_i, D))
        X[o:o + len(loc_i), :D] = tok
        X[o:o + len(loc_i), D:] = (tok.float() - prv.float()).half()  # as train_sae_fg.load_split / fg_sae_pool
        o += len(loc_i)
        log(f'  shard {s}: {len(loc_i):,} tokens gathered in {time.time() - t0:.0f}s')
    assert o == len(sel)
    return X


def cmd_train(args):
    import torch
    import motion_t3 as mt
    from src.eci.foreground import FgTokenStore
    from src.eci.sae import load_sae
    d = args.domain
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    st = FgTokenStore(REPO / STORE[d])
    bad = [i for i in st.info if (i['rule_name'], int(i['motion_delta']), int(i['stride']), float(i['patch_frac']))
           != (RULE[d], DELTA, 5, 1.0)]
    if bad:
        raise SystemExit(f'{STORE[d]}: shards with another rule / delta / stride: {bad[:2]}')
    videos = sorted(obs_rows(get_domain(d).ann_path))
    covered = {o for i in st.info for o in i['observations']}
    if covered != set(videos):
        raise SystemExit(f'{STORE[d]}: covers {len(covered)} of {len(videos)} videos')
    N = int(st.sizes.sum())
    shard_start = np.r_[0, np.cumsum(st.sizes)]
    cap = min(args.cap or CAP[d], N)
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(N, cap, replace=False))
    cand = np.random.default_rng(1).choice(N, min(N, cap + 2 * N_HELDOUT), replace=False)
    held = np.sort(cand[~np.isin(cand, sel)][:N_HELDOUT])
    log(f'{d}: store {STORE[d]}: {len(st.dirs)} shards, {N:,} tokens, {len(videos)} videos; pool {cap:,} '
        f'(seed 0), held-out {len(held):,} (seed 1)')
    t0 = time.time()
    X = gather_pool(st, sel, shard_start)
    log(f'pool gathered in {time.time() - t0:.0f}s')
    rows = np.concatenate([st.row(s) for s in range(len(st.dirs))])
    obs = pd.read_csv(get_domain(d).ann_path, usecols=['observation_id'])['observation_id'].values
    pool_videos = len(set(obs[rows[sel]]))
    zero = float((X[:, D:].float().abs().sum(1) == 0).float().mean())
    rn = float((X[:200_000, D:].float().norm(dim=1) / X[:200_000, :D].float().norm(dim=1)).median())
    log(f'pool: {pool_videos} videos, zero-change tokens {100 * zero:.2f}%, median |change|/|token| {rn:.3f}')
    loc = local_dir() / 'sae'
    extra = {'config': 'C1', 'arm': {'delta': DELTA, 'kind': 'back'}, 'motion_delta': DELTA, 'fg_rule': RULE[d],
             'encoder': 'dinov2_base', 'align': 'none', 'bg_sub': False, 'domain': d, 'cap': cap, 'pool_seed': 0,
             'blocks': [D, D], 'tokens_dir': STORE[d], 'pool': 'uniform over all tokens of all videos of the store',
             'deployment': 'C1 rollout (scripts/eci/rollout_c1.py)'}
    Xd = X.to(dev)
    del X
    mt.train_one(TAG[d], 0, Xd, args.steps, loc, extra, dev)
    del Xd
    torch.cuda.empty_cache()
    # held-out FVE / L0 / dead fraction
    sae, norm, _ = load_sae(loc / 'sae.pt', dev)
    Xh = gather_pool(st, held, shard_start)
    s1 = torch.zeros(2 * D, dtype=torch.float64, device=dev)
    s2, sse = torch.zeros_like(s1), torch.zeros_like(s1)
    fire = torch.zeros(WIDTH, dtype=torch.float64, device=dev)
    l0 = 0.0
    with torch.no_grad():
        for a in range(0, len(Xh), 100_000):
            x = norm(Xh[a:a + 100_000].to(dev))
            z = sae.encode(x, mode='threshold')
            xh = sae.decode(z)
            s1 += x.double().sum(0)
            s2 += x.double().pow(2).sum(0)
            sse += (x - xh).double().pow(2).sum(0)
            fire += (z > 0).sum(0).double()
            l0 += float((z > 0).sum())
    n = len(Xh)
    tss = s2 - s1.pow(2) / n
    fve = lambda sl_: float(1 - sse[sl_].sum() / tss[sl_].sum())  # noqa: E731
    fr = (fire / n).cpu().numpy()
    tj = json.loads((loc / 'train.json').read_text())
    ev = {'n_tokens': n, 'fve': fve(slice(None)), 'fve_static': fve(slice(0, D)), 'fve_change': fve(slice(D, None)),
          'l0_per_token': l0 / n, 'dead_frac': float((fr == 0).mean()),
          'dead_frac_by_prefix': {str(p): float((fr[:p] == 0).mean()) for p in (512, 1024, 2048, 4096)},
          'var_share_change': float(tss[D:].sum() / tss.sum())}
    log(f'held-out: {json.dumps(ev)}')
    metrics = {'args': {'domain': d, 'tokens_dir': STORE[d], 'tag': TAG[d], 'seed': 0, 'n_latents': WIDTH,
                        'prefixes': f'{WIDTH // 8},{WIDTH // 4},{WIDTH // 2},{WIDTH}', 'k': K, 'batch_size': 4096,
                        'lr': 5e-4, 'warmup': 500, 'steps': args.steps, 'cap': cap, 'pool_seed': 0, 'motion': True},
               'fg_rule': RULE[d], 'motion_delta': DELTA, 'recipe': 'C1 = combined test arm C1 (motion_t3 M5)',
               'n_store_tokens': N, 'n_store_videos': len(videos), 'n_pool_videos': pool_videos,
               'zero_change_frac': zero, 'median_change_over_token_norm': rn,
               'train': {k: tj[k] for k in ('n_pool', 'steps', 'token_presentations', 'train_time_s', 'dead_at_end_train',
                                            'dead_frac_train', 'norm_scale_blocks', 'gpu')},
               'train_final': tj['history'][-1], 'heldout': ev}
    (loc / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    out = sae_dir(d)
    out.mkdir(parents=True, exist_ok=True)
    for f in ('sae.pt', 'train.json', 'metrics.json'):
        shutil.copyfile(loc / f, out / f)
    log(f'-> {out}')
    # codes config.json (fg_encode_all.py writes it when missing; no encode without --shard)
    subprocess.run([sys.executable, '-u', str(REPO / 'scripts/eci/fg_encode_all.py'), '--domain', d, '--sae',
                    sae_name(d), '--n-shards', str(N_SHARDS)], check=True)


# ---------------------------------------------------------------------------------------------- encode
def cmd_encode(args):
    from src.eci.encode import shard_ranges
    from src.eci.fg_encode import encode_fg_shard
    d = args.domain
    dom = get_domain(d)
    cdir = codes_dir(d)
    cfg = json.loads((cdir / 'config.json').read_text())
    dst = cdir / 'shards' / f'shard_{args.shard:02d}'
    if (dst / 'DONE').exists():
        log(f'[SKIP] {dst}')
        return
    fp = frame_paths(d)
    ranges = shard_ranges(len(fp), N_SHARDS)
    assert [list(r) for r in ranges] == cfg['shard_ranges'], 'shard ranges differ from config.json'
    lo, hi = ranges[args.shard]
    starts = np.array(sorted(a for a, _ in obs_rows(dom.ann_path).values()))
    v0 = starts[np.searchsorted(starts, lo, side='right') - 1]  # first row of the video holding row lo
    first = max(lo - DELTA, int(v0))
    loc = local_dir()
    ds_loc = loc / 'dataset'
    log(f'{d} shard {args.shard}: rows [{lo}, {hi}), staging rows [{first}, {hi})')
    stage_jpgs(list(fp[first:hi]), ds_loc)
    encode_fg_shard(fp, lo, hi, loc / 'shard', sae_dir(d) / 'sae.pt', cfg['backgrounds'], dom.ann_path, ds_loc,
                    args.batch_size, args.num_workers)
    shutil.rmtree(ds_loc)
    dst.parent.mkdir(parents=True, exist_ok=True)
    copy_back(loc / 'shard', dst)


# ---------------------------------------------------------------------------------------------- merge + verify
def cmd_merge(args):
    import torch  # noqa: F401
    from src.eci.encode import shard_ranges
    from src.eci.fg_encode import merge_fg_shards, verify_fg_codes
    d = args.domain
    dom = get_domain(d)
    cdir = codes_dir(d)
    cfg = json.loads((cdir / 'config.json').read_text())
    n = len(frame_paths(d))
    t0 = time.time()
    merge_fg_shards(cdir, shard_ranges(n, N_SHARDS), n)
    log(f'merged in {time.time() - t0:.0f}s')
    res = verify_fg_codes(cdir, sae_dir(d) / 'sae.pt', cfg['backgrounds'], dom.ann_path, REPO / 'dataset',
                          n_check=args.n_check)
    nf = np.load(cdir / 'n_fg.npy')
    ref = np.load(codes_dir(d, SAME_MASK[d]) / 'n_fg.npy')
    res['n_fg_vs_same_rule_codes'] = {'reference': SAME_MASK[d], 'n_frames': int(len(nf)),
                                      'exact_match_frac': float((nf == ref).mean()),
                                      'max_abs_diff': int(np.abs(nf.astype(int) - ref.astype(int)).max())}
    a = res['alignment']
    ok = (res['codes_max']['shape'][0] == n == res['n_rows_annotations'] and res['codes_max']['rows_nonfinite'] == 0
          and res['codes_mean']['rows_nonfinite'] == 0 and a['codes_max']['cos_min'] > 0.99
          and a['n_fg_max_abs_diff'] <= 5 and res['n_fg_vs_same_rule_codes']['exact_match_frac'] > 0.999)
    res['passed'] = bool(ok)
    print(json.dumps(res, indent=1))
    (cdir / 'verify.json').write_text(json.dumps(res, indent=1))
    if not ok:
        raise SystemExit('VERIFY FAILED (shards kept)')
    shutil.rmtree(cdir / 'shards')
    log('verify passed, shards deleted')


# ---------------------------------------------------------------------------------------------- align
def labelled_fps1(d):
    """Labels of the fps1 rows (stride 5, seed 0 offsets = the token stores' frames) of every labelled video ->
    DataFrame (row, obs, labels...), labels, valid masks."""
    import spatial_sae_pilot as ssp
    from fg_extract_train import fps1_rows
    from sae_levers import LABELS
    dom = get_domain(d)
    lab, _ = ssp.labelled_frames(d)
    rows = np.sort(np.concatenate(list(fps1_rows(obs_rows(dom.ann_path), 5, 0).values())))
    lab = lab.set_index('row')
    lab = lab.loc[np.intersect1d(rows, lab.index.values)].reset_index()
    labels = list(LABELS[d])
    valid = {b: np.ones(len(lab), bool) for b in labels}
    if d == 'ants':
        v = pd.read_csv(dom.ann_path, usecols=['Y_YOL'])['Y_YOL'].values[lab['row'].values].astype(float)
        valid['onlid_yellow'] = ~np.isnan(v)
        lab['onlid_yellow'] = np.nan_to_num(v) > 0
    return lab, labels, valid


def cmd_align(args):
    import torch
    from sae_levers import score_model
    d = args.domain
    dev = torch.device('cuda')
    lab, labels, valid = labelled_fps1(d)
    unit = lab.obs.str.rsplit('_', n=2).str[0].values if d == 'mice' else lab.obs.values
    units = sorted(set(unit))
    half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
    rows = lab['row'].values
    log(f'{d}: {len(lab):,} labelled fps1 frames of {lab.obs.nunique()} videos, {len(units)} units; base rates '
        + json.dumps({b: [int(valid[b].sum()), round(float(lab[b][valid[b]].mean()), 4)] for b in labels}))
    Yt = torch.from_numpy(lab[labels].values.astype(bool)).to(dev)
    R = {'domain': d, 'n_frames': int(len(lab)), 'n_videos': int(lab.obs.nunique()), 'n_units': len(units),
         'frames': 'every 5th frame (fps1 rows, seed 0 offsets) of every labelled video',
         'note': 'IN-SAMPLE: both SAEs were trained on tokens of all videos of the domain (no labels). Neuron '
                 'selection is cross-fitted over two halves of the units (mice: pools, ants: videos); insample_* = '
                 'best neuron on all frames.', 'models': {}}
    for name in (sae_name(d), DEPLOYED[d]):
        t0 = time.time()
        cm = np.load(codes_dir(d, name) / 'codes_max.npy', mmap_mode='r')
        codes = torch.from_numpy(np.ascontiguousarray(cm[rows])).to(dev)
        nf = np.load(codes_dir(d, name) / 'n_fg.npy', mmap_mode='r')[rows].astype(np.float64)
        rank = np.argsort(np.argsort(nf + np.random.default_rng(0).random(len(nf)) * 1e-3))
        decile = np.minimum(rank * 10 // len(nf), 9)
        res, arrays = score_model(codes, Yt, labels, valid, half, decile, dev)
        if name == sae_name(d):  # per-neuron in-sample AUROC / AP / firing rate on all labelled frames (summary)
            (OUT / d).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OUT / d / 'align_neurons.npz', **{k: v for k, v in arrays.items() if '|all|' in k})
        R['models'][name] = {b: {k: v for k, v in r.items() if k != 'dirs'} | {
            'cf_neurons_auroc': [e['auc']['neuron'] for e in r['dirs']],
            'cf_neurons_top1': [e['top1']['neuron'] for e in r['dirs']]} for b, r in res.items()}
        log(f'{name}: scored in {time.time() - t0:.0f}s ' + json.dumps(
            {b: {k: round(r[k], 4) for k in ('insample_auroc', 'insample_ap', 'insample_top1_honest', 'cf_auroc',
                                             'cf_ap', 'cf_top1')} for b, r in res.items()}))
        del codes
        torch.cuda.empty_cache()
    (OUT / d).mkdir(parents=True, exist_ok=True)
    (OUT / d / 'align.json').write_text(json.dumps(R, indent=1))


# ---------------------------------------------------------------------------------------------- levels
def cmd_levels(args):
    """Level map (scripts/eci/tight_mask_sae.py level_map: two axes, social = self / pair / collective / off-animal and
    place-bound, plus plain-background) of every neuron of the deployment SAE on the C1 evaluation frames of
    scripts/eci/combined.py (the levers eval frames, T2 site maps), with the C1 arm's exact input path (base-store
    tokens + d5 prev, MotionTokens). Also the presence / extent scores there (IN-SAMPLE: the deployment SAE trained on
    these videos) next to the C1 3-seed validation numbers. -> OUT/<d>/levels.csv, OUT/<d>/levels_eval.json"""
    import torch
    import combined as cb
    import motion_t3 as mt
    import multiscale_sae as ms
    import sae_levers as sl
    import spatial_sae_pilot as ssp
    import tight_mask_sae as tms
    from src.eci.sae import load_sae
    d = args.domain
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    loc = local_dir()
    idx = ssp.StoreIndex(d)
    _, ev, _, _, _, _ = idx.split(d, ms.N_TRAIN[d], 0)
    ev = np.sort(ev)
    S = np.load(cb.T2 / d / 'mask' / 'sites.npz')
    assert (S['frames'] == ev).all()
    site, bcount = S['site'], S['bcount']
    lab, _, valid = ms.eval_labels(d, idx, ev)
    labels = sl.LABELS[d]
    unit = lab.obs.str.rsplit('_', n=2).str[0].values if d == 'mice' else lab.obs.values
    units = sorted(set(unit))
    half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
    assert (half == np.load(cb.LEV / d / 'half.npy')).all(), 'halves differ from sae_levers'
    Yt = torch.from_numpy(lab[labels].values.astype(bool)).to(dev)
    a1 = json.loads((cb.LEV / d / 'train_meta_w4096_w16384.json').read_text())['median_single_animal_area']
    _, pos, lens = ms.stage(idx, ev, loc / 'eval')
    tok = np.load(loc / 'eval' / 'tok.npy')
    shutil.rmtree(loc / 'eval')
    prev = mt.stage_prev5(idx, d, ev, loc / 'prev_eval.npy')
    X = cb.MotionTokens(tok, prev)
    rank = np.argsort(np.argsort(lens + np.random.default_rng(0).random(len(lens)) * 1e-3))
    decile = np.minimum(rank * 10 // len(lens), 9)
    sae, norm, _ = load_sae(sae_dir(d) / 'sae.pt', dev)
    t0 = time.time()
    pres, ext, stt = cb.encode_pe(sae, norm, X, lens, dev)
    resP, _ = sl.score_model(pres, Yt, labels, valid, half, decile, dev)
    resE, _ = sl.score_model(ext, Yt, labels, valid, half, decile, dev)
    with torch.no_grad():
        T, _, _, chk = tms.level_map(sae, norm, X, pos, lens, pres, ext, site, bcount, a1, dev, bcount.shape[1])
    (OUT / d).mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / d / 'levels.csv', index=False)
    strip = lambda R: {b: {k: v for k, v in r.items() if k != 'dirs'} | {  # noqa: E731
        'cf_neurons_auroc': [e['auc']['neuron'] for e in r['dirs']]} for b, r in R.items()}
    e = {'domain': d, 'n_eval_frames': int(len(ev)), 'note': 'IN-SAMPLE (deployment SAE trained on all videos)',
         **stt, 'presence': strip(resP), 'extent': strip(resE), 'levels': tms.level_counts(T), 'level_checks': chk,
         'stage': dict(mt.STAGE_INFO), 'elapsed_s': time.time() - t0}
    (OUT / d / 'levels_eval.json').write_text(json.dumps(e, indent=1, default=float))
    log(f'{d}: FVE {stt["fve"]:.4f}; ' + ' | '.join(f'{b} cf AUROC {resP[b]["cf_auroc"]:.3f} AP {resP[b]["cf_ap"]:.4f}'
                                                   for b in labels) + f'; levels {json.dumps(e["levels"])}')


# ---------------------------------------------------------------------------------------------- summary
PRIMARY = {  # outcome -> (subdir of the result set, summary.csv filter of the primary setting)
    'activation': ('', {'pooling': 'max', 'outcome_type': 'mean', 'test': 't', 'correction': 'bonferroni',
                        'window': 'full'}),
    'bout_rate': ('maxpool_bouts', {'outcome_type': 'bout_rate', 'threshold_q': 0.95, 'merge_gap': 0, 'bout_rule': 0,
                                    'test': 't', 'correction': 'bonferroni', 'window': 'full'}),
    'latency': ('maxpool_latency', {'outcome_type': 'latency', 'threshold_q': 0.95, 'transform': 'none', 'test': 't',
                                    'correction': 'bonferroni', 'window': 'common'})}


def selections(res):
    """Primary-setting NES selections of a result set -> {outcome: {analysis: {prefix: [rounds]}}}, skipped."""
    out, skipped = {}, {}
    for oc, (sub, flt) in PRIMARY.items():
        f = res / sub / 'summary.csv'
        if not f.exists():
            continue
        t = pd.read_csv(f)
        for k, v in flt.items():
            if k in t:
                t = t[np.isclose(t[k], v)] if isinstance(v, (int, float)) else t[t[k] == v]
        out[oc] = {}
        for (aid, pf), g in t.groupby(['analysis_id', 'prefix']):
            g = g[g['round'] > 0].sort_values('round')
            out[oc].setdefault(aid, {})[int(pf)] = [
                {'round': int(r['round']), 'neuron': int(r['neuron']), 'tau': float(r['tau']), 'p': float(r['p']),
                 'direction': r['direction']} for _, r in g.iterrows()]
        sj = res / sub / 'sanity.json'
        if sj.exists():
            skipped[oc] = [x.get('analysis_id') for x in json.loads(sj.read_text()).get('skipped', [])]
    return out, skipped


def change_share(d):
    """Decoder weight share of every latent on the change half of the input (normalised space)."""
    import torch
    ck = torch.load(sae_dir(d) / 'sae.pt', map_location='cpu', weights_only=False)
    W = ck['state_dict']['W_dec'].float()
    if W.shape[0] != WIDTH:
        W = W.T
    e = W.pow(2)
    return (e[:, D:].sum(1) / e.sum(1)).numpy()


def cmd_summary(args):
    from sae_levers import LABELS
    S = {}
    md = ['# C1 rollout: NES discoveries per domain', '',
          'Primary settings: codes_max frame values; per-video mean activation (run_nes.py), bout rate q 0.95 '
          '(run_nes_bouts.py), latency q 0.95 common window (run_nes_latency.py); t-test, Bonferroni 0.05; '
          'neurons picked by smallest p; nuisance = per-video mean foreground patch count (all domains) + recording '
          'day indicators (ants). Each cell: neuron (direction, tau, p) in selection order. "deployed, same settings" '
          '= the deployed SAE re-run with these settings; "atlas" = the deployed SAE as in the current atlas (|tau| '
          'pick, no nuisance). Neuron ids are SAE specific: only the counts and the contrasts are comparable.', '']
    for d in DOMAINS:
        dom = get_domain(d, 'pairs' if d == 'ants' else 'core')
        sub = 'pairs' if d == 'ants' else ''
        sets = {'C1': (dom.nes_root / sae_name(d) / sub, (512, 1024)),
                'deployed, same settings': (OUT / 'nes_deployed' / d / DEPLOYED[d] / sub, (128, 256)),
                'atlas': (dom.nes_root / DEPLOYED[d] / sub, (128, 256))}
        S[d] = {'sae': sae_name(d), 'sets': {}}
        for k, (res, pfs) in sets.items():
            if (res / 'summary.csv').exists():
                sel, sk = selections(res)
                S[d]['sets'][k] = {'dir': str(res.relative_to(REPO)), 'prefixes': pfs, 'selections': sel, 'skipped': sk}
        ann = {}
        if (sae_dir(d) / 'sae.pt').exists():
            ann['change_share'] = change_share(d)
        if (OUT / d / 'levels.csv').exists():
            ann['levels'] = pd.read_csv(OUT / d / 'levels.csv').set_index('neuron')
        if (OUT / d / 'align_neurons.npz').exists():
            ann['auc'] = dict(np.load(OUT / d / 'align_neurons.npz'))

        def describe(j):
            x = []
            if 'change_share' in ann:
                x.append(f'change {ann["change_share"][j]:.2f}')
            if 'levels' in ann and j in ann['levels'].index:
                L = ann['levels'].loc[j]
                x.append('plain-bg' if L['plain_background'] else
                         f'{L["social"]}{"/place" if L["place_bound"] else ""}' if L['eligible'] else 'rare')
            if 'auc' in ann:
                a = {b: ann['auc'][f'{b}|all|auc'][j] for b in LABELS.get(d, []) if f'{b}|all|auc' in ann['auc']}
                if a:
                    b = max(a, key=lambda q: abs(a[q] - 0.5))
                    x.append(f'{b} AUROC {a[b]:.2f}')
            return ', '.join(x)
        S[d]['neurons'] = {}
        md += [f'## {d} ({sae_name(d)})', '']
        C1 = S[d]['sets'].get('C1')
        if not C1:
            md += ['(no C1 NES results yet)', '']
            continue
        for oc in PRIMARY:
            md += [f'### {oc}', '', '| analysis | C1 p512 | C1 p1024 | deployed, same settings p128 / p256 | atlas p128 / p256 |',
                   '|---|---|---|---|---|']
            for aid in sorted(C1['selections'].get(oc, {})):
                cell = lambda key, pf: '; '.join(  # noqa: E731
                    f'{r["neuron"]} ({r["direction"]}, {r["tau"]:.3g}, {r["p"]:.1e})'
                    for r in S[d]['sets'].get(key, {}).get('selections', {}).get(oc, {}).get(aid, {}).get(pf, [])) or '-'
                md.append(f'| {aid} | {cell("C1", 512)} | {cell("C1", 1024)} | {cell("deployed, same settings", 128)} / '
                          f'{cell("deployed, same settings", 256)} | {cell("atlas", 128)} / {cell("atlas", 256)} |')
                for pf in (512, 1024):
                    for r in C1['selections'][oc][aid].get(pf, []):
                        S[d]['neurons'][r['neuron']] = describe(r['neuron'])
            sk = sorted(set(C1['skipped'].get(oc, [])) - {None})
            md += ['', f'Skipped (C1): {", ".join(sk) if sk else "none"}', '']
        if S[d]['neurons']:
            md += ['C1 neurons selected anywhere above: ' + '; '.join(f'{j}: {v}' for j, v in sorted(S[d]['neurons'].items())), '']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1, default=str))
    (OUT / 'summary.md').write_text('\n'.join(md))
    print('\n'.join(md))


# ---------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('plan', 'tokens', 'train', 'encode', 'merge', 'align', 'levels', 'summary'))
    p.add_argument('--domain', default='mice', choices=DOMAINS)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--n-tasks', type=int, default=35)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--cap', type=int, default=0)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--n-check', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=14)
    args = p.parse_args()
    {'plan': cmd_plan, 'tokens': cmd_tokens, 'train': cmd_train, 'encode': cmd_encode, 'merge': cmd_merge,
     'align': cmd_align, 'levels': cmd_levels, 'summary': cmd_summary}[args.step](args)


if __name__ == '__main__':
    main()
