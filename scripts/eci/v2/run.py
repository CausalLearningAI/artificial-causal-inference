"""
ECI v2 rollout: ONE entry point for every domain (configs/eci_v2/<domain>.json + the frozen representation spec
configs/eci_v2/representation.json). The library is src/eci/v2/ (README in its __init__.py).

Steps (scripts/eci/v2/run.sh runs each in Slurm; heavy inputs / outputs staged on $LOCAL_DIR)
    pool      (GPU) --series S: tight-mask token store of bursts of 10 consecutive 5 fps frames (k = 0 mod 10 frames
              dropped) at random starts (rng 7), per experiment frames_per_experiment -> BEE/<domain>/pool/<series>/
              (a series with 'pool_reuse' links an existing store of identical construction instead)
    train     (GPU) one SAE per domain (all experiments pooled, the same number of frames per experiment), seeds of the
              spec; theta* on calibration videos; FVE on evaluation videos; deployment seed = highest FVE -> BEE/<domain>/
              sae/<name>/, OUT/<domain>/train.json, OUT/<domain>/best.json
    encode    (GPU) --shard i of the full pass: every 5 fps frame of every video -> BEE/<domain>/videos/<series>/<obs>.npz
              (frame max codes, argmax token, kept count, sparse per-token codes)
    merge     (CPU) per series: raw frame codes in layout row order -> BEE/<domain>/codes/<series>/<name>_raw/
    fire      (CPU) readout smoothing (centred 0.6 s mean), 'fnorm' firing rule calibrated on the calibration videos
              of the domain, thresholded smoothed codes = the NES input set <name> (per series layout), per-video
              summaries (time spent, bout rate, latency) -> OUT/<domain>/fire.json, summaries_<series>.npz
    link      (CPU) symlink the codes / SAE dirs into the domain layouts read by the NES runners
    align     (GPU) alignment tables per behaviour (labelled series), v2 vs the v1 online model (and r4) on the same
              frames -> OUT/<domain>/align.json, align.md
    selftest  (GPU) one video per series end to end (decode -> mask -> tokens -> codes) + library equalities
    manifest  (CPU) results/vision/eci_v2/manifest.json
NES / NESS: scripts/eci/v2/nes.sh (the existing runners and atlas_v37_methods.py from a git archive of a pinned commit).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.eci.v2 import readout as RO  # noqa: E402
from src.eci.v2.context import edge_distance, sae_input  # noqa: E402

OUT = Path(os.environ.get('V2_OUT') or REPO / 'results/vision/eci_v2')
BEE = Path(os.environ.get('V2_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2')
D = 1024
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def spec():
    return json.loads((REPO / 'configs/eci_v2/representation.json').read_text())


def dcfg(domain):
    c = json.loads((REPO / f'configs/eci_v2/{domain}.json').read_text())
    c['spec'] = spec()
    return c


def local_dir():
    d = os.environ.get('LOCAL_DIR')
    if not d:
        raise SystemExit('LOCAL_DIR is not set (the job staging dir /localhome/$USER/$SLURM_JOB_ID)')
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)


def atomic_copy(src, dst):
    import shutil
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.tmp')
    if Path(src).is_dir():
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp)
        if dst.exists():
            shutil.rmtree(dst)
    else:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def sae_name(c, seed):
    s = c['spec']
    return f'v2_{c["domain"]}_c{s["context_window"]}_w{s["sae"]["W"]}_s{seed}'


def best_name(c):
    return json.loads((OUT / c['domain'] / 'best.json').read_text())['name']


def layout_ann(sc):
    """annotations.csv of the series layout (the row space of its codes)."""
    return REPO / sc.get('layout_ann', f'{sc["layout"]}/annotations.csv')


def series_of(c, name):
    return next(s for s in c['series'] if s['name'] == name)


def experiment_of(sc):
    """obs -> experiment of a series (layout experiment.csv when the series has several)."""
    if isinstance(sc.get('experiment'), str):
        return lambda o: sc['experiment']
    e = pd.read_csv(REPO / sc['layout'] / 'experiment.csv', usecols=['observation_id', 'experiment'])
    m = dict(zip(e.observation_id, e.experiment.astype(str)))
    return lambda o: m[o]


# ---------------------------------------------------------------------------------------------- token stores
class Store:
    """Token store (frames.parquet, tok_frame / tok_pos / tok_fdark / tok_dist .npy, tok.f16)."""

    def __init__(self, root, G):
        self.root, self.G = Path(root), G
        self.F = pd.read_parquet(self.root / 'frames.parquet')
        self.tf = np.load(self.root / 'tok_frame.npy')
        self.pos = np.load(self.root / 'tok_pos.npy').astype(np.int64)
        self.tok = np.memmap(self.root / 'tok.f16', np.float16, 'r', shape=(len(self.tf), D))
        self.nf = len(self.F)
        self.starts = np.r_[0, np.cumsum(np.bincount(self.tf, minlength=self.nf))]

    def inputs(self, w, frame_sel, dev, frames_per=48):
        """SAE inputs (fp16 cuda) of every token of the frames where frame_sel (bool (nf,)) holds, with context w."""
        import torch
        n = int(np.diff(self.starts)[frame_sel].sum())
        out = torch.empty((n, D * (2 if w else 1)), dtype=torch.float16, device=dev)
        o = 0
        for f0 in range(0, self.nf, frames_per):
            f1 = min(self.nf, f0 + frames_per)
            a, b = int(self.starts[f0]), int(self.starts[f1])
            if a == b or not frame_sel[f0:f1].any():
                continue
            z = torch.from_numpy(np.ascontiguousarray(self.tok[a:b])).to(dev)
            fl = torch.from_numpy(self.tf[a:b] - f0).to(dev)
            ps = torch.from_numpy(self.pos[a:b]).to(dev)
            m = torch.from_numpy(frame_sel[self.tf[a:b]]).to(dev)
            x = sae_input(z, ps, fl, f1 - f0, self.G, w)[m].half()
            out[o:o + len(x)] = x
            o += len(x)
        assert o == n
        return out


def write_store(loc, frames, toks, pos, fdark, G):
    """Finish a token store in loc (tok.f16 already written by the caller)."""
    Fm = pd.DataFrame(frames)
    Fm.to_parquet(loc / 'frames.parquet')
    tf = np.concatenate(toks)
    ps = np.concatenate(pos)
    np.save(loc / 'tok_frame.npy', tf)
    np.save(loc / 'tok_pos.npy', ps)
    np.save(loc / 'tok_fdark.npy', np.concatenate(fdark))
    np.save(loc / 'tok_dist.npy', edge_distance(tf, ps.astype(np.int64), len(Fm), G))
    return Fm, tf


def cmd_pool(args):
    import torch
    from src.eci.v2.pipeline import Series
    from src.eci.v2.video import Prefetch
    c = dcfg(args.domain)
    sc = series_of(c, args.series)
    if sc.get('pool_reuse'):
        log(f'{args.domain} {args.series}: pool reused from {sc["pool_reuse"]} (link step)')
        return
    dev = torch.device('cuda')
    Se = Series(sc, dev)
    exp_of = experiment_of(sc)
    vids = Se.videos[:args.max_videos or None]
    per_exp = {}
    for v in vids:
        per_exp.setdefault(exp_of(v['obs']), []).append(v)
    g = np.random.default_rng(7)
    items = []
    for ex, vs in sorted(per_exp.items()):
        nb = int(np.ceil(c['pool']['frames_per_experiment'] / (9 * len(vs))))
        for v in vs:
            for a_ in np.sort(g.choice(v['n'] - 10, min(nb, v['n'] - 10), replace=False)):
                kk0 = np.arange(a_, a_ + 10)
                items.append((v, kk0, ex))
    if args.max_frames:
        items = items[:args.max_frames]
    jobs = [((i,), Se.job(v, int(kk0[0]), 10)) for i, (v, kk0, ex) in enumerate(items)]
    loc = local_dir() / f'pool_{args.series}'
    loc.mkdir(parents=True, exist_ok=True)
    ftok = open(loc / 'tok.f16', 'wb')
    toks, poss, fds, frames = [], [], [], []
    sheet, sheet_obs = [], set()
    t0 = time.time()
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=Se.run_job)
    for (i,), fr in pf:
        v, kk0, ex = items[i]
        sel = kk0 % 10 != 0
        kk = kk0[sel]
        x, fd = Se.frames(v, fr[sel], kk)
        z, keep = Se.tokens(x, fd)
        if len(sheet) < 8 and v['obs'] not in sheet_obs:
            sheet_obs.add(v['obs'])
            sheet.append((x[0].cpu().numpy(), keep[0].cpu().numpy(), f'{v["obs"]} k{kk[0]} kept {int(keep[0].sum())}'))
        for ii in range(len(kk)):
            p_ = torch.nonzero(keep[ii]).flatten()
            fid = len(frames)
            ftok.write(z[ii, p_].cpu().numpy().tobytes())
            toks.append(np.full(len(p_), fid, np.int32))
            poss.append(p_.cpu().numpy().astype(np.int16))
            fds.append(fd[ii, p_].cpu().numpy().astype(np.float16))
            frames.append({'series': args.series, 'experiment': ex, 'obs': v['obs'], 'k': int(kk[ii]),
                           'row': int(v['lay_lo'] + kk[ii]), 'src_row': int(v['src_lo'] + kk[ii]), 'role': v['role'],
                           'n_kept': int(len(p_))})
        if i % 500 == 0:
            el = time.time() - t0
            log(f'pool {args.domain} {args.series}: {i}/{len(items)} bursts, {len(frames):,} frames, {el:.0f}s '
                f'({len(frames) / max(el, 1):.1f} fps; GPU busy {Se.enc.gpu_s:.0f}s; decode thread-s {pf.dec_s:.0f})')
    ftok.close()
    Fm, tf = write_store(loc, frames, toks, poss, fds, Se.G)
    info = {'domain': args.domain, 'series': args.series, 'S': Se.S, 'G': Se.G, 'n_frames': len(Fm),
            'n_tokens': int(len(tf)), 'tokens_per_frame': float(Fm.n_kept.mean()),
            'frames_0_kept': int((Fm.n_kept == 0).sum()), 'frames_by_experiment': Fm.experiment.value_counts().to_dict(),
            'roles': Fm.role.value_counts().to_dict(), 'wall_s': time.time() - t0, 'gpu_s': Se.enc.gpu_s,
            'gpu': torch.cuda.get_device_name(0)}
    log(json.dumps(info))
    od = OUT / args.domain
    od.mkdir(parents=True, exist_ok=True)
    tag = "_selftest" if args.selftest else ""
    (od / f'pool_{args.series}{tag}.json').write_text(json.dumps(info, indent=1))
    from src.eci.v2.mask import mask_sheet
    mask_sheet(sheet, od / f'mask_sheet_pool_{args.series}{tag}.jpg', Se.G)
    if not args.selftest:
        atomic_copy(loc, BEE / args.domain / 'pool' / args.series)


def pool_dir(c, sc):
    return Path(sc['pool_reuse']) if sc.get('pool_reuse') else BEE / c['domain'] / 'pool' / sc['name']


def cmd_train(args):
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    c = dcfg(args.domain)
    s = c['spec']
    w = s['context_window']
    stores, sel = {}, {}
    rng = np.random.default_rng(0)
    for sc in c['series']:
        dst = local_dir() / f'pool_{sc["name"]}'
        if not dst.exists():
            atomic_copy(pool_dir(c, sc), dst)
        St = Store(dst, sc['S'] // 16)
        if 'experiment' not in St.F:
            exp_of = experiment_of(sc)
            St.F['experiment'] = St.F.obs.map(exp_of)
        stores[sc['name']] = St
    # the same number of frames per experiment (all roles), seeded
    allF = pd.concat([St.F.assign(_s=n, _f=np.arange(St.nf)) for n, St in stores.items()])
    n_e = c['pool'].get('frames_per_experiment')
    counts = allF.groupby('experiment').size()
    n_take = int(min(counts.min(), n_e)) if n_e else int(counts.min())
    for n, St in stores.items():
        sel[n] = np.zeros(St.nf, bool)
    for ex, g_ in allF.groupby('experiment'):
        pick = g_.iloc[np.sort(rng.choice(len(g_), min(n_take, len(g_)), replace=False))]
        for n, ff in pick.groupby('_s'):
            sel[n][ff._f.values] = True
    info = {'frames_per_experiment': n_take, 'available': counts.to_dict()}
    X = {}
    for role in ('train', 'calib', 'eval'):
        parts = [St.inputs(w, sel[n] & (St.F.role.values == role), dev) for n, St in stores.items()]
        X[role] = torch.cat(parts)
        del parts
    X['calib'], X['eval'] = X['calib'].cpu(), X['eval'].cpu()
    info['tokens'] = {r: int(len(x)) for r, x in X.items()}
    log(f'{args.domain}: ' + json.dumps(info))
    cfg = {**s['sae'], 'context_window': w}
    if args.steps:
        cfg['steps'] = args.steps
    recs = {}
    tf = OUT / args.domain / ('train_selftest.json' if args.selftest else 'train.json')
    if tf.exists() and not args.selftest:
        recs = {int(k): v for k, v in json.loads(tf.read_text()).get('seeds', {}).items()}
    for seed in s['seeds']:
        od = BEE / args.domain / ('sae_selftest' if args.selftest else 'sae') / sae_name(c, seed)
        if (od / 'sae_theta.pt').exists() and seed in recs and not args.force:
            continue
        recs[seed] = SA.train(cfg, seed, X['train'], X['calib'], X['eval'], od, dev)
        r = recs[seed]
        log(f'{sae_name(c, seed)}: {r["train_time_s"]:.0f}s, theta* {r["theta_star"]:.4f}, eval FVE {r["eval_fve"]:.4f}, '
            f'L0 {r["eval_l0"]:.2f}, dead {r["dead_at_end_train"]}')
        tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text(json.dumps({'pool': info, 'seeds': recs}, indent=1, default=float))
    best = SA.deploy_seed(recs)
    b = {'name': sae_name(c, best), 'seed': best, 'eval_fve': recs[best]['eval_fve'],
         'theta_star': recs[best]['theta_star'], 'rule': 'highest held-out FVE among the spec seeds (label-free)',
         'dir': str(BEE / args.domain / 'sae' / sae_name(c, best))}
    if not args.selftest:
        (OUT / args.domain / 'best.json').write_text(json.dumps(b, indent=1))
    log(f'{args.domain}: deployed {b}')


# ---------------------------------------------------------------------------------------------- full pass
def shards_of(c, n_shards):
    """[(series, video index)] balanced by frames into n_shards (consecutive videos per series)."""
    items = []
    for sc in c['series']:
        P = json.loads((REPO / sc['plan']).read_text())
        for i, v in enumerate(P['videos']):
            items.append((sc['name'], i, v['n']))
    n = np.array([x[2] for x in items])
    cuts = np.searchsorted(np.cumsum(n), np.linspace(0, n.sum(), n_shards + 1)[1:-1])
    return [[(x[0], x[1]) for x in part] for part in np.split(np.array(items, dtype=object), cuts + 1) if len(part)]


def cmd_encode(args):
    import torch
    from src.eci.v2 import sae as SA
    from src.eci.v2.pipeline import Series
    from src.eci.v2.video import Prefetch
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    c = dcfg(args.domain)
    s = c['spec']
    w, sel = s['context_window'], s['readout']
    b = json.loads((OUT / args.domain / 'best.json').read_text())
    sae, norm = SA.load(Path(b['dir']) / 'sae_theta.pt', dev)
    W = sae.n_latents
    shards = shards_of(c, c['encode']['n_shards'])
    todo = shards[args.shard] if args.videos is None else [(args.series, int(x)) for x in args.videos.split(',')]
    from src.eci.v2.encode import Encoder
    enc = Encoder(dev)
    SE = {}
    T0 = time.time()
    tot = 0
    for ser, vi in todo:
        sc = series_of(c, ser)
        if ser not in SE:
            SE[ser] = Series(sc, dev, encoder=enc)
        Se = SE[ser]
        v = Se.videos[vi]
        outd = BEE / args.domain / ('videos_selftest' if args.selftest else 'videos') / ser
        if (outd / f'{v["obs"]}.npz').exists() and not args.force:
            continue
        n = min(v['n'], args.max_frames) if args.max_frames else v['n']
        ch, bs = c['encode']['chunk'], c['encode']['bs']
        jobs = [((k0,), Se.job(v, k0, min(ch, n - k0), threads=c['encode'].get('threads', 4))) for k0 in range(0, n, ch)]
        pf = Prefetch(jobs, c['encode'].get('par', 4), fn=Se.run_job)
        acc = {'cm': [], 'am': [], 'nk': [], 'tp': [], 'ac': [], 'ai': [], 'av': []}
        t0 = time.time()
        g0 = enc.gpu_s
        for (k0,), fr in pf:
            for b0 in range(0, len(fr), bs):
                kk = k0 + b0 + np.arange(len(fr[b0:b0 + bs]))
                with torch.no_grad():
                    x, fd = Se.frames(v, fr[b0:b0 + bs], kk)
                    z, keep = Se.tokens(x, fd)
                    if sel == 'edge':
                        raise SystemExit('edge readout is not part of v2')
                    bi, pi = keep.nonzero(as_tuple=True)
                    xin = sae_input(z[bi, pi], pi, bi, len(kk), Se.G, w)
                    cm, am, cz = SA.frame_codes(sae, norm, xin, bi, len(kk), pos=pi)
                    acc['cm'].append(cm.half().cpu().numpy())
                    acc['am'].append(am.cpu().numpy())
                    acc['nk'].append(keep.sum(1).to(torch.int16).cpu().numpy())
                    acc['tp'].append(pi.to(torch.int16).cpu().numpy())
                    nz = cz > 0
                    acc['ac'].append(nz.sum(1).to(torch.int16).cpu().numpy())
                    ti, li = nz.nonzero(as_tuple=True)
                    acc['ai'].append(li.to(torch.int16).cpu().numpy())
                    acc['av'].append(cz[ti, li].half().cpu().numpy())
        arr = {k_: np.concatenate(x_) for k_, x_ in acc.items()}
        assert len(arr['cm']) == n and int(arr['nk'].sum()) == len(arr['tp']) == len(arr['ac'])
        loc = local_dir() / f'{v["obs"]}.npz'
        np.savez(loc, codes_max=arr['cm'], argmax=arr['am'], n_kept=arr['nk'], tok_pos=arr['tp'], act_count=arr['ac'],
                 act_idx=arr['ai'], act_val=arr['av'], obs=v['obs'], n=n, G=Se.G, W=W, sae=b['name'])
        atomic_copy(loc, outd / loc.name)
        loc.unlink()
        dt = time.time() - t0
        tot += n
        log(f'{ser} {v["obs"]} ({v["role"]}): {n} frames in {dt:.0f}s ({n / dt:.1f} fps; GPU {enc.gpu_s - g0:.0f}s) | '
            f'kept/frame {arr["nk"].mean():.1f} | token L0 {arr["ac"].mean():.2f} | frame L0 '
            f'{(arr["cm"] > 0).sum(1).mean():.1f} | 0-kept frames {(arr["nk"] == 0).sum()}')
    wall = time.time() - T0
    log(f'{args.domain} shard {args.shard} DONE: {tot:,} frames in {wall:.0f}s ({tot / max(wall, 1):.1f} fps wall), '
        f'GPU busy (encoder) {enc.gpu_s:.0f}s, {torch.cuda.get_device_name(0)}')


# ---------------------------------------------------------------------------------------------- merge / fire / link
def codes_dir(c, sc, name):
    return BEE / c['domain'] / 'codes' / sc['name'] / name


def cmd_merge(args):
    c = dcfg(args.domain)
    name = best_name(c)
    W = c['spec']['sae']['W']
    for sc in c['series']:
        P = json.loads((REPO / sc['plan']).read_text())
        n_rows = len(pd.read_csv(layout_ann(sc), usecols=['observation_id']))
        cd = codes_dir(c, sc, f'{name}_raw')
        tmp = cd.with_name(cd.name + '.tmp')
        tmp.mkdir(parents=True, exist_ok=True)
        cm = np.lib.format.open_memmap(tmp / 'codes_max.npy', 'w+', np.float16, (n_rows, W))
        am = np.lib.format.open_memmap(tmp / 'argmax.npy', 'w+', np.int16, (n_rows, W))
        nf = np.zeros(n_rows, np.int16)
        done = np.zeros(n_rows, bool)
        for v in P['videos']:
            z = np.load(BEE / c['domain'] / 'videos' / sc['name'] / f'{v["obs"]}.npz')
            assert int(z['n']) == v['n'] and str(z['sae']) == name, (v['obs'], int(z['n']), str(z['sae']))
            r = slice(v['lay_lo'], v['lay_lo'] + v['n'])
            cm[r], am[r], nf[r] = z['codes_max'], z['argmax'], z['n_kept']
            done[r] = True
        cm.flush()
        am.flush()
        del cm, am
        np.save(tmp / 'n_fg.npy', nf)
        cfg = {'sae': f'{name}_raw', 'v2_spec': c['spec'], 'series': sc['name'], 'S': sc['S'], 'G': sc['S'] // 16,
               'rows': int(n_rows), 'rows_filled': int(done.sum()),
               'files': {'codes_max': 'per 5 fps frame and neuron: max over the kept tokens of the token code (theta*), '
                                      'NO smoothing, NO firing rule', 'argmax': 'grid position of the max token (-1 '
                                      'silent)', 'n_fg': 'kept (tight-mask) tokens per frame'}}
        (tmp / 'config.json').write_text(json.dumps(cfg, indent=1))
        (tmp / 'DONE').touch()
        if cd.exists():
            import shutil
            shutil.rmtree(cd)
        tmp.rename(cd)
        log(f'{args.domain} {sc["name"]}: {cd} ({done.sum():,} of {n_rows:,} rows)')


def cmd_fire(args):
    """Smoothing + fnorm calibration (calibration videos of the whole domain) + thresholded set + summaries."""
    c = dcfg(args.domain)
    s = c['spec']
    name = best_name(c)
    h = s['smooth_h']
    S_cal = []
    for sc in c['series']:
        P = json.loads((REPO / sc['plan']).read_text())
        mx = np.load(codes_dir(c, sc, f'{name}_raw') / 'codes_max.npy', mmap_mode='r')
        for v in P['videos']:
            if v['role'] == 'calib':
                S_cal.append(RO.smooth(np.asarray(mx[v['lay_lo']:v['lay_lo'] + v['n']]), h))
    sig, th, info = RO.calibrate_fnorm(S_cal, k=s['sae']['k'])
    del S_cal
    thr = (th * sig).astype(np.float32)
    out = {'sae': name, 'smooth_h': h, 'theta_n': th, 'calib': info, 'series': {}}
    for sc in c['series']:
        P = json.loads((REPO / sc['plan']).read_text())
        src = codes_dir(c, sc, f'{name}_raw')
        mx = np.load(src / 'codes_max.npy', mmap_mode='r')
        cd = codes_dir(c, sc, name)
        tmp = cd.with_name(cd.name + '.tmp')
        tmp.mkdir(parents=True, exist_ok=True)
        y = np.lib.format.open_memmap(tmp / 'codes_max.npy', 'w+', np.float16, mx.shape)
        W = mx.shape[1]
        Q = ('time_spent', 'n_bouts', 'bout_rate_per_min', 'latency_s', 'censored')
        summ = {q: np.zeros((len(P['videos']), W), np.float32) for q in Q}
        act = np.zeros((len(P['videos']), W), np.float32)
        fires = []
        for i, v in enumerate(P['videos']):
            r = slice(v['lay_lo'], v['lay_lo'] + v['n'])
            Sv = RO.smooth(np.asarray(mx[r]), h)
            Fv = Sv > thr[None]
            y[r] = np.where(Fv, Sv, 0).astype(np.float16)
            act[i] = Sv.mean(0)
            for q, val in RO.video_summaries(Fv, c.get('fps5', 5.0)).items():
                summ[q][i] = val
            fires.append(Fv.sum(1).mean())
        y.flush()
        del y
        import shutil
        shutil.copyfile(src / 'n_fg.npy', tmp / 'n_fg.npy')
        np.save(tmp / 'threshold_per_neuron.npy', thr)
        np.save(tmp / 'scale_sigma.npy', sig)
        cfg = json.loads((src / 'config.json').read_text())
        cfg.update({'sae': name, 'parent_codes': str(src), 'rule': f'smoothed (centred mean +-{h} frames) codes_max_j / '
                    f'sigma_j > theta_n = {th:.6g}; value = the smoothed code where it fires, else 0',
                    'files': {'codes_max': 'smoothed frame code where the neuron fires under fnorm, else 0',
                              'n_fg': 'kept tokens per frame', 'threshold_per_neuron': 'theta_n sigma_j'}})
        (tmp / 'config.json').write_text(json.dumps(cfg, indent=1))
        (tmp / 'DONE').touch()
        if cd.exists():
            shutil.rmtree(cd)
        tmp.rename(cd)
        f = OUT / args.domain / f'summaries_{sc["name"]}.npz'
        np.savez_compressed(f, obs=np.array([v['obs'] for v in P['videos']]), n_frames=np.array([v['n'] for v in P['videos']]),
                            mean_smoothed=act, sae=name, theta_n=th, sigma=sig, **summ,
                            rule='fnorm on the 0.6 s centred mean of the frame max codes; bout = maximal run of firing '
                                 'frames; latency = first firing frame / 5 fps, censored at the video length')
        out['series'][sc['name']] = {'codes': str(cd), 'summaries': str(f),
                                     'mean_fires_per_frame': float(np.mean(fires)),
                                     'median_time_spent': float(np.median(summ['time_spent'].mean(0))),
                                     'never_firing_video_neuron_frac': float(summ['censored'].mean())}
        log(f'{args.domain} {sc["name"]}: {json.dumps(out["series"][sc["name"]])}')
    (OUT / args.domain / 'fire.json').write_text(json.dumps(out, indent=1, default=float))


def cmd_link(args):
    """Symlinks: <layout>/codes/<name>[_raw] -> BeeGFS codes dirs; <layout>/sae/<name> -> the SAE dir (sae.pt =
    sae_theta.pt link, read by the atlas builder)."""
    c = dcfg(args.domain)
    name = best_name(c)
    b = json.loads((OUT / args.domain / 'best.json').read_text())
    sd = Path(b['dir'])
    if not (sd / 'sae.pt').exists():
        os.symlink(sd / 'sae_theta.pt', sd / 'sae.pt')
    links = []
    for sc in c['series']:
        lay = REPO / sc['layout']
        for nm in (name, f'{name}_raw'):
            l_ = lay / 'codes' / nm
            if not (l_.is_symlink() or l_.exists()):
                os.symlink(codes_dir(c, sc, nm), l_)
            links.append(str(l_))
        l_ = lay / 'sae' / name
        if not (l_.is_symlink() or l_.exists()):
            os.symlink(sd, l_)
        links.append(str(l_))
    (OUT / args.domain / 'links.json').write_text(json.dumps(links, indent=1))
    log('\n'.join(links))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('pool', 'train', 'encode', 'merge', 'fire', 'link', 'align', 'selftest',
                                    'manifest', 'heat'))
    p.add_argument('--picks', default='')
    p.add_argument('--out-dir', default='')
    p.add_argument('--domain', default='mice')
    p.add_argument('--series', default=None)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--videos', default=None)
    p.add_argument('--max-videos', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--force', action='store_true')
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()
    if args.step in ('align', 'selftest', 'manifest', 'heat'):
        import importlib.util
        spec_ = importlib.util.spec_from_file_location('v2_tools', Path(__file__).parent / 'tools.py')
        T = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(T)
        return getattr(T, f'cmd_{args.step}')(args)
    {'pool': cmd_pool, 'train': cmd_train, 'encode': cmd_encode, 'merge': cmd_merge, 'fire': cmd_fire,
     'link': cmd_link}[args.step](args)


if __name__ == '__main__':
    main()
