"""
ECI v2.0 rollout (spec configs/eci_v2/representation.json, frozen 2026-10-25): thin-part mask N2 + the V2-full SAEs.

Reuses scripts/eci/v2/run.py (encode / merge / smooth / link) with three in-memory changes: the spec / output roots
(V2_SPEC, V2_OUT = results/vision/eci_v2_0, V2_BEE = eci_codes/v2_0), the Series class (N2Series: token kept iff
mask2 cue fraction >= 0.25 OR >= 16 appendage pixels, exactly scripts/eci/v2_diag/diag.py cmd_full1's N2 arm) and the
shard counts. Mice / ants backgrounds = the stored mask2 packs (eci_codes/v2_mask2/bg); frogs packs are built here.

Steps (scripts/eci/v2/v20.sh; staging in $LOCAL_DIR)
    bgs      (CPU) --domain frogs --series juv|tad --shard i --nshard n: per-video mask2 background packs (48 frames
             evenly spread; juveniles canonical at side S like mask2_test.cmd_bgs, tadpoles in NATIVE coordinates,
             cropped per frame like the frame); 'hcal' = tadpole body half-width (label-free, mask_test.cmd_calib rule)
    store    (GPU) --domain frogs --shard i --nshard n: 1 fps (k % 5 == 2) of every frog video, N2 tokens, keep p = 1
    train    (GPU) --domain frogs: Matryoshka 2048 (128/256/512/2048) k16 30k, seeds of the spec, deploy = best FVE
    best     (login) --domain mice|ants: copy the V2F_N2 deployment SAE (s900, highest held-out FVE of 900-902)
    selftest (GPU) --domain D: separable max filter == reference; mice / ants: the encode path on full1 store frames
             reproduces the stored N2 token set and the SAE codes of the stored tokens; throughput
    encode   (GPU) --domain D --shard i: run.py cmd_encode with N2Series
    merge / smooth (CPU): run.py
    hmm      (CPU) --domain D: per-neuron 3-state HMM (src/eci/v2/hmm.py) fitted on the calibration videos' raw frame
             max, applied to every video -> codes sets <name>_hmm (P(on)), <name>_hmmv (Viterbi on = 1), summaries
    heatcheck (CPU) --domain D: frame codes == max of the stored per-token heat map (sampled frames, every neuron)
    link     (login) layout symlinks for <name>_{raw,smooth,hmm,hmmv}
"""
import json
import os
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
os.environ.setdefault('V2_SPEC', 'configs/eci_v2/representation.json')
os.environ.setdefault('V2_OUT', str(REPO / 'results/vision/eci_v2_0'))
os.environ.setdefault('V2_BEE', '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_0')
sys.path.insert(0, str(REPO))
for p in ('scripts/eci', 'scripts/eci/nnnt', 'scripts/eci/v2', 'scripts/eci/v2_mask', 'scripts/eci/v2_mask2'):
    sys.path.insert(0, str(REPO / p))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run as R  # noqa: E402

OUT, BEE = R.OUT, R.BEE
log = R.log
BG_M2 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_mask2/bg')
V2F = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_diag')
BG_V20 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_0/bg')  # frogs / mice v2 packs (fixed: PASS B uses another V2_BEE)
NOSTORE = os.environ.get('V20_NOSTORE') == '1'  # PASS B: no training store
NSH = {'mice': 32, 'ants': 16, 'frogs': 16}
NSH_STORE1 = 8  # frogs 1 fps store shards
N_BG = 48
H_FIXED = {('mice', 'v1'): 32.0, ('mice', 'v2'): 32.0, ('ants', 'v2'): 13.92838827718412, ('ants', 'v3'): 16.15549442140351,
           ('ants', 'v4'): 16.15549442140351, ('frogs', 'juv'): 15.811388300841896}  # calib.json; v4 = v3 (h_ser);
# mice v2 = v1 (same rig, same 2064 px source, same S 1024)
_dcfg = R.dcfg


DCFG_V20 = {'mice': 'configs/eci_v2/mice_v20.json'}  # mice: v1 + v2 series (configs/eci_v2/mice.json = v1 only)


def dcfg(domain):
    if domain in DCFG_V20:
        c = json.loads((REPO / DCFG_V20[domain]).read_text())
        c['spec'] = R.spec()
    else:
        c = _dcfg(domain)
    c['encode'] = {**c['encode'], 'n_shards': NSH[domain]}
    return c


R.dcfg = dcfg


def h_of(dom, ser):
    if (dom, ser) in H_FIXED:
        return H_FIXED[(dom, ser)]
    return float(json.loads((OUT / dom / f'hcal_{ser}.json').read_text())['body_halfwidth_px'])


def bg_file(dom, ser, obs, w):
    # stored mask2 packs for mice v1 / ants; mice v2 and frogs packs are built here (cmd_bgs)
    root = BG_M2 if dom in ('mice', 'ants') and (dom, ser) != ('mice', 'v2') else BG_V20
    return root / f'{dom}_{ser}' / (f'{obs}.npz' if w == 0 else f'{obs}__w{w}.npz')


def window_of(dom, k):
    return int(k) // 3000 if dom == 'ants' else 0


# ---------------------------------------------------------------------------------------------- N2 series
def make_series_class(dom):
    import torch
    import torch.nn.functional as F
    from src.eci.v2 import mask2 as M2
    from src.eci.v2.mask import grey_of
    from src.eci.v2.pipeline import Series, crop_np

    class N2Series(Series):
        """Series whose frames() returns fd = 1.0 on N2-kept tokens, 0.0 elsewhere (tokens() keeps fd >= 0.25)."""

        def __init__(self, cfg, dev, encoder=None, load_encoder=True):
            super().__init__(cfg, dev, encoder=encoder, load_encoder=load_encoder)
            self.dom, self.ser = dom, cfg['name']
            self.h = h_of(dom, self.ser)
            self._packs = {}
            self.stats = {'frames': 0, 'kept': 0, 'm0': 0}

        def pack(self, obs, w):
            key = (obs, w)
            if key not in self._packs:
                if len(self._packs) > 4:
                    self._packs.clear()
                z = np.load(bg_file(self.dom, self.ser, obs, w))
                m = self.cfg['mask']
                dv = self.dev
                if self.geometry == 'crop':  # native coordinates, cropped per frame
                    self._packs[key] = {'bg': z['bg'], 'mad': z['mad_s15'].astype(np.float32), 'q98': z['q98']}
                else:  # = mask2_test.load_pack
                    bg = torch.from_numpy(z['bg']).to(dv).float()
                    q98 = torch.from_numpy(z['q98']).to(dv).float()
                    gb = M2.grey_f(bg[None])[0]
                    self._packs[key] = {'bg': bg, 'mad': torch.from_numpy(z['mad_s15']).to(dv).float(),
                                        'q02': torch.from_numpy(z['q02']).to(dv).float(), 'q98': q98,
                                        'ghost': (gb < m.get('dark_abs', 60)) & ((q98 - gb) > m.get('dark_rel', 40))}
            return self._packs[key]

        @torch.no_grad()
        def n2(self, x, core0, pack, roi=None):
            """-> keep (B, G*G) bool (N2), m0 (B, G*G) bool (tight rule on core0)."""
            r = M2.hyst_pixels(x, pack['bg'], core0, self.h, t_lo=M2.T_LO, t_hi=M2.T_HI, k_noise=M2.K_NOISE,
                               reach_h=M2.REACH_H, sigma=M2.SIGMA, strong=True, mad=pack['mad'], k_mad=M2.K_MAD,
                               bad=pack['ghost'], roi=roi)
            _, app = M2.appendage(r['pix'], self.h)
            pool = lambda m: F.avg_pool2d(m.float()[:, None], 16).flatten(1)  # noqa: E731
            frn, ng = pool(r['pix']), pool(app) * 256
            return (frn >= 0.25) | (ng >= M2.MIN_THIN_PX), pool(core0) >= 0.25

        @torch.no_grad()
        def frames(self, v, fr, ks):
            ks = np.asarray(ks)
            if self.geometry == 'crop':
                x, b, roi = self.crops(v, fr, ks)
                core0 = ((b - grey_of(x).float()) > self.dark_rel) & roi
                P = self.pack(v['obs'], 0)
                L = self.locate(v['obs'])
                side = self.cfg['crop']['side']
                cr = lambda a: np.stack([crop_np(a, int(L['x0'][k]), int(L['y0'][k]), side) for k in ks])  # noqa
                bgc = torch.from_numpy(np.ascontiguousarray(cr(P['bg'].transpose(1, 2, 0)))).to(self.dev)
                bgc = F.interpolate(bgc.permute(0, 3, 1, 2).float(), size=(self.S, self.S), mode='bicubic',
                                    align_corners=False).round().clamp(0, 255)
                rs = lambda a: F.interpolate(torch.from_numpy(np.ascontiguousarray(cr(a))).to(self.dev).float()[:, None],  # noqa
                                             size=(self.S, self.S), mode='bilinear', align_corners=False)[:, 0]
                mad, q98 = rs(P['mad']), rs(P['q98'].astype(np.float32))
                gb = M2.grey_f(bgc)
                ghost = (gb < self.cfg['mask'].get('dark_abs', 60)) & ((q98 - gb) > self.dark_rel)
                keep, m0 = self.n2(x, core0, {'bg': bgc, 'mad': mad, 'ghost': ghost}, roi=roi)
            else:
                x, _ = super().frames(v, fr, ks)
                rows = v['src_lo'] + ks
                core0 = self.mask.pixels(x, rows)
                roi = None
                if self.mask.cue == 'frog':
                    roi = torch.ones_like(core0)
                    for sel, _pb, ro in self.mask.planes(rows, self.S):
                        if ro is not None:
                            roi[torch.from_numpy(sel).to(x.device)] = ro[None]
                keep = torch.zeros((len(ks), self.G * self.G), dtype=torch.bool, device=x.device)
                m0 = torch.zeros_like(keep)
                wins = np.array([window_of(self.dom, k) for k in ks])
                for w in np.unique(wins):
                    s = torch.from_numpy(np.flatnonzero(wins == w)).to(x.device)
                    keep[s], m0[s] = self.n2(x[s], core0[s], self.pack(v['obs'], int(w)),
                                             roi=None if roi is None else roi[s])
            self.stats['frames'] += len(ks)
            self.stats['kept'] += int(keep.sum())
            self.stats['m0'] += int(m0.sum())
            return x, keep.float()

    return N2Series


def patch_series(dom):
    import src.eci.v2.pipeline as PL
    PL.Series = make_series_class(dom)
    return PL.Series


# ---------------------------------------------------------------------------------------------- frogs backgrounds
def cmd_bgs(a):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from src.eci.v2 import mask2 as M2
    from src.eci.v2.pipeline import Series, decode_native
    dom, ser = a.domain, a.series
    c = dcfg(dom)
    sc = R.series_of(c, ser)
    se = Series(sc, torch.device('cpu'), load_encoder=False)
    vids = se.videos[a.shard::a.nshard]
    torch.set_num_threads(4)

    def one(v):
        p = bg_file(dom, ser, v['obs'], 0)
        if p.exists():
            return 0
        ks = np.unique(np.linspace(0, v['n'] - 1, N_BG).round().astype(int))
        if se.geometry == 'crop':
            xs = [decode_native(v['path'], se.voff(v) + sc['step'] * int(k), 1, sc['step'], v['w'], v['h'], sc['fps'], 1)
                  for k in ks]
            xs = torch.from_numpy(np.concatenate(xs)).permute(0, 3, 1, 2).contiguous()
        else:
            xs = torch.cat([se.frames(v, se.run_job(**se.job(v, int(k), 1, threads=1)), [int(k)])[0] for k in ks])
        bgf = M2.median_background(xs)
        mad = M2.background_mad(xs, bgf, sigma=M2.SIGMA)
        q02, q98 = M2.grey_quantiles(xs)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + '.tmp.npz')
        np.savez(tmp, bg=bgf.round().clamp(0, 255).to(torch.uint8).numpy(), n=len(ks), mad_s15=mad.half().numpy(),
                 q02=q02.round().to(torch.uint8).numpy(), q98=q98.round().to(torch.uint8).numpy())
        tmp.rename(p)
        return len(ks)
    t0 = time.time()
    with ThreadPoolExecutor(max(2, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) // 4)) as ex:
        for i, n in enumerate(ex.map(one, vids)):
            log(f'bgs {dom} {ser} {i + 1}/{len(vids)} ({time.time() - t0:.0f}s)')
    if ser == 'tad' and a.shard == 0:
        hcal(dom, ser, se)


def hcal(dom, ser, se):
    """Body half-width of the tadpole crops (mask_test.cmd_calib rule: median over components >= 100 px of the max
    distance transform of the frozen-cue pixels; 24 train videos x 10 frames, rng 0)."""
    import torch
    from scipy import ndimage
    from src.eci.v2.mask import grey_of
    g = np.random.default_rng(0)
    tv = [v for v in se.videos if v['role'] == 'train']
    obs = g.choice(len(tv), min(24, len(tv)), replace=False)
    hmax, npx = [], []
    for i in obs:
        v = tv[i]
        for k in np.sort(g.choice(v['n'], 10, replace=False)):
            fr = se.run_job(**se.job(v, int(k), 1))
            x, b, roi = se.crops(v, fr, [int(k)])
            px = (((b - grey_of(x).float()) > se.dark_rel) & roi)[0].numpy()
            npx.append(int(px.sum()))
            lab, n = ndimage.label(px)
            if n == 0:
                continue
            edt = ndimage.distance_transform_edt(px)
            area = np.bincount(lab.ravel(), minlength=n + 1)
            mx = ndimage.maximum(edt, lab, np.arange(1, n + 1))
            hmax += [float(m) for m, a_ in zip(mx, area[1:]) if a_ >= 100]
    out = {'S': se.S, 'n_frames': len(npx), 'n_components': len(hmax), 'body_halfwidth_px': float(np.median(hmax)),
           'animal_px_per_frame_median': float(np.median(npx)), 'rule': hcal.__doc__}
    (OUT / dom).mkdir(parents=True, exist_ok=True)
    (OUT / dom / f'hcal_{ser}.json').write_text(json.dumps(out, indent=1))
    log(f'hcal {dom} {ser}: {out}')


# ---------------------------------------------------------------------------------------------- frogs 1 fps store
def cmd_store(a):
    import torch
    from src.eci.v2.encode import Encoder
    from src.eci.v2.video import Prefetch
    dev = torch.device('cuda')
    dom = a.domain
    c = dcfg(dom)
    NS = patch_series(dom)
    enc = Encoder(dev)
    SE = {sc['name']: NS(sc, dev, encoder=enc) for sc in c['series']}
    vids = [(s, v) for s in SE for v in SE[s].videos][a.shard::a.nshard]
    if a.selftest:
        vids = vids[:2]
    jobs, meta = [], []
    for s, v in vids:
        ks = np.arange(2, v['n'], 5)
        if a.selftest:
            ks = ks[:120]
        for c0 in range(0, len(ks), 240):
            kk = ks[c0:c0 + 240]
            jobs.append(((len(meta),), SE[s].job(v, int(kk[0]), len(kk), 5)))
            meta.append((s, v, kk))
    loc = R.local_dir() / f'store_{dom}_s{a.shard}'
    loc.mkdir(parents=True, exist_ok=True)
    ftok = open(loc / 'tok.f16', 'wb')
    tfr, tps, recs = [], [], []
    t0 = time.time()
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=SE[vids[0][0]].run_job)
    for (j,), frm in pf:
        s, v, kk = meta[j]
        se = SE[s]
        bs = 16 if se.S >= 1000 else 32
        for b in range(0, len(kk), bs):
            kb = kk[b:b + bs]
            with torch.no_grad():
                x, fd = se.frames(v, frm[b:b + bs], kb)
                z, keep = se.tokens(x, fd)
            for ii in range(len(kb)):
                pt = torch.nonzero(keep[ii]).flatten()
                ftok.write(z[ii, pt].cpu().numpy().tobytes())
                tfr.append(np.full(len(pt), len(recs), np.int32))
                tps.append(pt.cpu().numpy().astype(np.int16))
                recs.append({'series': s, 'obs': v['obs'], 'k': int(kb[ii]), 'role': v['role'], 'n_kept': int(len(pt))})
        if j % 50 == 0:
            el = time.time() - t0
            log(f'store {dom} s{a.shard}: {j}/{len(jobs)} runs, {len(recs):,} frames, {el:.0f}s ({len(recs) / max(el, 1):.1f} '
                f'fps; GPU {enc.gpu_s:.0f}s; decode thread-s {pf.dec_s:.0f})')
    ftok.close()
    np.save(loc / 'tok_frame.npy', np.concatenate(tfr))
    np.save(loc / 'tok_pos.npy', np.concatenate(tps))
    Fm = pd.DataFrame(recs)
    Fm.to_parquet(loc / 'frames.parquet')
    el = time.time() - t0
    info = {'dom': dom, 'shard': a.shard, 'nshard': a.nshard, 'n_frames': len(Fm), 'n_tokens': int(Fm.n_kept.sum()),
            'tokens_per_frame': {s: float(g.n_kept.mean()) for s, g in Fm.groupby('series')}, 'wall_s': el,
            'gpu_s': enc.gpu_s, 'fps_wall': len(Fm) / max(el, 1), 'gpu': torch.cuda.get_device_name(0),
            'm0_tokens': {s: SE[s].stats['m0'] for s in SE}, 'n2_tokens': {s: SE[s].stats['kept'] for s in SE}}
    log(json.dumps(info))
    od = OUT / dom / 'store'
    od.mkdir(parents=True, exist_ok=True)
    (od / f's{a.shard}{"_selftest" if a.selftest else ""}.json').write_text(json.dumps(info, indent=1))
    if not a.selftest:
        R.atomic_copy(loc, BEE / dom / 'store' / f's{a.shard}')


def cmd_train(a):
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    dom = a.domain
    s = R.spec()
    shards = sorted((BEE / dom / 'store').glob('s[0-9]*'), key=lambda p: int(p.name[1:]))
    assert len(shards) == NSH_STORE1, (len(shards), NSH_STORE1)
    Fs, tfs, toks, o = [], [], [], 0
    for sp in shards:
        dst = R.local_dir() / sp.name
        if not dst.exists():
            R.atomic_copy(sp, dst)
        F_ = pd.read_parquet(dst / 'frames.parquet')
        tf = np.load(dst / 'tok_frame.npy')
        toks.append(np.memmap(dst / 'tok.f16', np.float16, 'r', shape=(len(tf), 1024)))
        tfs.append(tf + o)
        Fs.append(F_)
        o += len(F_)
    Fa = pd.concat(Fs, ignore_index=True)
    tf = np.concatenate(tfs)
    role = Fa.role.values[tf]
    cap = 16_000_000
    X, info = {}, {'frames': Fa.role.value_counts().to_dict(), 'series_frames': Fa.series.value_counts().to_dict()}
    for r in ('train', 'calib', 'eval'):
        idx = np.flatnonzero(role == r)
        if r == 'train' and len(idx) > cap:
            idx = np.sort(np.random.default_rng(0).choice(idx, cap, replace=False))
            info['train_capped_from'] = int((role == 'train').sum())
        parts, o = [], 0
        for T in toks:
            m = idx[(idx >= o) & (idx < o + len(T))] - o
            parts.append(np.asarray(T[m]))
            o += len(T)
        x = torch.from_numpy(np.concatenate(parts))
        X[r] = x.to(dev) if r == 'train' else x
        info[f'tokens_{r}'] = int(len(x))
    log(json.dumps(info))
    cfg = {**s['sae'], 'context_window': 0}
    recs = {}
    for seed in s['seeds'][dom]:
        od = BEE / dom / 'sae' / f'v20_{dom}_N2_s{seed}'
        recs[seed] = SA.train(cfg, seed, X['train'], X['calib'], X['eval'], od, dev)
        r = recs[seed]
        log(f'{od.name}: {r["train_time_s"]:.0f}s theta* {r["theta_star"]:.4f} eval FVE {r["eval_fve"]:.5f} '
            f'L0 {r["eval_l0"]:.2f} dead {r["dead_at_end_train"]}')
        (OUT / dom / 'train.json').write_text(json.dumps({'data': info, 'seeds': recs}, indent=1, default=float))
    best = SA.deploy_seed(recs)
    write_best(dom, f'v20_{dom}_N2_s{best}', best, recs[best]['eval_fve'], recs[best]['theta_star'],
               BEE / dom / 'sae' / f'v20_{dom}_N2_s{best}', {k: v['eval_fve'] for k, v in recs.items()})


def write_best(dom, name, seed, fve, th, d, cands):
    b = {'name': name, 'seed': seed, 'eval_fve': fve, 'theta_star': th, 'candidates_eval_fve': cands,
         'rule': 'highest held-out FVE among the spec seeds (label-free)', 'dir': str(d)}
    sp = Path(d) / 'sae.pt'
    if not (sp.exists() or sp.is_symlink()):
        os.symlink('sae_theta.pt', sp)
    (OUT / dom).mkdir(parents=True, exist_ok=True)
    (OUT / dom / 'best.json').write_text(json.dumps(b, indent=1))
    log(f'{dom}: deployed {b}')


def cmd_best(a):
    import shutil
    dom = a.domain
    recs = {s: json.loads((REPO / f'results/vision/eci_v2_diag/{dom}/train/V2F_N2_s{s}.json').read_text())
            for s in (900, 901, 902)}
    best = max(recs, key=lambda s: recs[s]['eval_fve'])
    src = V2F / dom / 'sae' / f'V2F_N2_s{best}' / 'sae_theta.pt'
    d = BEE / dom / 'sae' / f'v20_{dom}_N2_s{best}'
    d.mkdir(parents=True, exist_ok=True)
    if not (d / 'sae_theta.pt').exists():
        shutil.copyfile(src, d / 'sae_theta.pt')
    import hashlib
    h = [hashlib.sha1(Path(p).read_bytes()).hexdigest() for p in (src, d / 'sae_theta.pt')]
    assert h[0] == h[1]
    write_best(dom, d.name, best, recs[best]['eval_fve'], recs[best]['theta_star'], d,
               {s: r['eval_fve'] for s, r in recs.items()})


# ---------------------------------------------------------------------------------------------- selftest
def cmd_selftest(a):
    import torch
    import torch.nn.functional as F
    from src.eci.v2 import mask2 as M2
    from src.eci.v2 import sae as SA
    from src.eci.v2.encode import Encoder
    dev = torch.device('cuda')
    dom = a.domain
    res = {}
    g = torch.Generator(device=dev).manual_seed(0)
    mx = 0
    for r in (1, 3, 16, 64, 192):
        m = (torch.rand((3, 1, 300, 280), generator=g, device=dev) < 0.002).float()
        mx = max(mx, float((M2._maxpool(m, r) - M2._maxpool_ref(m, r)).abs().max()))
        mx = max(mx, float((M2._maxpool(-m, r) - M2._maxpool_ref(-m, r)).abs().max()))
    res['separable_maxpool_max_abs_diff'] = mx
    assert mx == 0
    c = dcfg(dom)
    NS = patch_series(dom)
    enc = Encoder(dev)
    b = json.loads((OUT / dom / 'best.json').read_text())
    sae, norm = SA.load(Path(b['dir']) / 'sae_theta.pt', dev)
    if dom in ('mice', 'ants'):
        # stored V2-full tokens (full1 shard 0; N2 arm = fr_new >= 0.25 | n_app >= 16) vs this encode path
        root = V2F / dom / 'full1' / 's0'
        Fm = pd.read_parquet(root / 'frames.parquet')
        tf = np.load(root / 'tok_frame.npy')
        pos = np.load(root / 'tok_pos.npy').astype(np.int64)
        arm = (np.load(root / 'fr_new.npy').astype(np.float32) >= 0.25) | (np.load(root / 'n_app.npy') >= M2.MIN_THIN_PX)
        tok = np.memmap(root / 'tok.f16', np.float16, 'r', shape=(len(tf), 1024))
        starts = np.r_[0, np.cumsum(np.bincount(tf, minlength=len(Fm)))]
        rng = np.random.default_rng(0)
        pick = []
        for o in rng.choice(Fm.obs.unique(), 3, replace=False):
            ii = np.flatnonzero(Fm.obs.values == o)
            pick += list(rng.choice(ii, 8, replace=False))
        SE = {}
        st = {'frames': 0, 'stored_tokens': 0, 'stored_in_keep': 0, 'keep_tokens': 0, 'keep_eq_stored_frames': 0,
              'tok_code_maxabs': 0.0, 'tok_code_exact_frac': [], 'frame_code_maxabs': 0.0, 'tok_maxabs': 0.0}
        for fi in pick:
            r = Fm.iloc[fi]
            sc = R.series_of(c, r.series)
            se = SE.setdefault(r.series, NS(sc, dev, encoder=enc))
            v = se.V[r.obs]
            k = int(r.k)
            k0 = (k // c['encode']['chunk']) * c['encode']['chunk']  # the encode step's chunk start
            fr = se.run_job(**se.job(v, k0, min(c['encode']['chunk'], v['n'] - k0)))
            with torch.no_grad():
                x, fd = se.frames(v, fr[k - k0:k - k0 + 1], [k])
                z, keep = se.tokens(x, fd)
            a0, a1 = starts[fi], starts[fi + 1]
            sp = pos[a0:a1][arm[a0:a1]]
            kp = torch.nonzero(keep[0]).flatten().cpu().numpy()
            inn = np.isin(sp, kp)
            st['frames'] += 1
            st['stored_tokens'] += len(sp)
            st['stored_in_keep'] += int(inn.sum())
            st['keep_tokens'] += len(kp)
            st['keep_eq_stored_frames'] += int(set(sp.tolist()) == set(kp.tolist()))
            if len(st.setdefault('app_eq', [])) < 6:  # fast disk ops == disk convolutions on real cue pixels
                core0 = se.mask.pixels(x, v['src_lo'] + np.array([k]))
                rr = M2.cue(x, se.pack(v['obs'], window_of(dom, k)), core0, se.h)
                q1, q0 = M2.appendage(rr['pix'], se.h), M2.appendage_ref(rr['pix'], se.h)
                st['app_eq'].append(bool(torch.equal(q1[0], q0[0]) and torch.equal(q1[1], q0[1])))
                st.setdefault('app_px', []).append(int(q1[1].sum()))
            if len(sp):
                zs = torch.from_numpy(np.asarray(tok[a0:a1][arm[a0:a1]])).to(dev)
                zm = z[0, torch.from_numpy(sp).to(dev)]
                st['tok_maxabs'] = max(st['tok_maxabs'], float((zs.float() - zm.float()).abs().max()))
                with torch.no_grad():
                    cs = sae.encode(norm(zs), mode='threshold')
                    cmn = sae.encode(norm(zm), mode='threshold')
                st['tok_code_maxabs'] = max(st['tok_code_maxabs'], float((cs - cmn).abs().max()))
                st['tok_code_exact_frac'].append(float((cs == cmn).float().mean()))
                st['frame_code_maxabs'] = max(st['frame_code_maxabs'], float((cs.max(0).values - cmn.max(0).values).abs().max()))
        st['tok_code_exact_frac'] = float(np.mean(st['tok_code_exact_frac']))
        st['stored_in_keep_frac'] = st['stored_in_keep'] / max(st['stored_tokens'], 1)
        st['note'] = ('full1 keep p = 0.2 for mice (stored = a seeded subset of the N2 set: expect stored_in_keep_frac 1, '
                      'keep_tokens ~5x stored); p = 1 for ants (expect keep == stored)')
        res['identity_full1'] = st
        log(json.dumps(st))
    (OUT / dom).mkdir(parents=True, exist_ok=True)
    (OUT / dom / 'selftest.json').write_text(json.dumps(res, indent=1, default=float))
    log(json.dumps(res, default=float))


P_STORE = 0.2


def cmd_encode(a):
    """PASS A: run.py cmd_encode (same per-video npz) + a 5 fps training store of a fixed random 20% of each frame's
    kept tokens (seeded per (obs, k): rng [crc32(obs), k]) -> BEE/<domain>/store5/s<shard>/ (one large shard)."""
    import torch
    from src.eci.v2 import sae as SA
    from src.eci.v2.encode import Encoder
    from src.eci.v2.video import Prefetch
    from src.eci.v2.context import sae_input
    Series = patch_series(a.domain)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    c = dcfg(a.domain)
    b = json.loads((OUT / a.domain / 'best.json').read_text())
    sae, norm = SA.load(Path(b['dir']) / 'sae_theta.pt', dev)
    W = sae.n_latents
    tag = f's{a.shard}'
    if a.series:  # shard within one series (--nshard shards), e.g. mice v1 while v2 is being set up
        todo = R.shards_of({**c, 'series': [R.series_of(c, a.series)]}, a.nshard)[a.shard]
        tag = f'{a.series}_s{a.shard}'
    else:
        todo = R.shards_of(c, c['encode']['n_shards'])[a.shard]
    enc = Encoder(dev)
    SE = {}
    loc_s = R.local_dir() / f'store5_{tag}'
    loc_s.mkdir(parents=True, exist_ok=True)
    ftok = open(loc_s / 'tok.f16', 'ab')
    st_tf, st_pos, recs = [], [], []
    T0, tot = time.time(), 0
    for ser, vi in todo:
        sc = R.series_of(c, ser)
        if ser not in SE:
            SE[ser] = Series(sc, dev, encoder=enc)
        Se = SE[ser]
        v = Se.videos[vi]
        outd = BEE / a.domain / 'videos' / ser
        n = min(v['n'], a.max_frames) if a.max_frames else v['n']
        ch, bs = c['encode']['chunk'], c['encode']['bs']
        jobs = [((k0,), Se.job(v, k0, min(ch, n - k0), threads=c['encode'].get('threads', 4))) for k0 in range(0, n, ch)]
        pf = Prefetch(jobs, c['encode'].get('par', 4), fn=Se.run_job)
        acc = {'cm': [], 'am': [], 'nk': [], 'tp': [], 'ac': [], 'ai': [], 'av': []}
        t0, g0 = time.time(), enc.gpu_s
        crc = zlib.crc32(v['obs'].encode())
        for (k0,), fr in pf:
            for b0 in range(0, len(fr), bs):
                kk = k0 + b0 + np.arange(len(fr[b0:b0 + bs]))
                with torch.no_grad():
                    x, fd = Se.frames(v, fr[b0:b0 + bs], kk)
                    z, keep = Se.tokens(x, fd)
                    bi, pi = keep.nonzero(as_tuple=True)
                    xin = sae_input(z[bi, pi], pi, bi, len(kk), Se.G, 0)
                    cm, am, cz = SA.frame_codes(sae, norm, xin, bi, len(kk), pos=pi)
                    acc['cm'].append(cm.half().cpu().numpy())
                    acc['am'].append(am.cpu().numpy())
                    nk = keep.sum(1).cpu().numpy()
                    acc['nk'].append(nk.astype(np.int16))
                    pic = pi.to(torch.int16).cpu().numpy()
                    acc['tp'].append(pic)
                    nz = cz > 0
                    acc['ac'].append(nz.sum(1).to(torch.int16).cpu().numpy())
                    ti, li = nz.nonzero(as_tuple=True)
                    acc['ai'].append(li.to(torch.int16).cpu().numpy())
                    acc['av'].append(cz[ti, li].half().cpu().numpy())
                    # training store: fixed random 20% of each frame's kept tokens
                    if NOSTORE:
                        continue
                    sel = np.concatenate([np.random.default_rng([crc, int(k)]).random(int(m)) < P_STORE
                                          for k, m in zip(kk, nk)]) if len(pic) else np.zeros(0, bool)
                    ft = bi.cpu().numpy()[sel]
                    ftok.write(xin[torch.from_numpy(sel).to(dev)].half().cpu().numpy().tobytes())
                    st_pos.append(pic[sel])
                    st_tf.append(len(recs) + ft.astype(np.int32))
                    cnt = np.bincount(ft, minlength=len(kk))
                    for i_, k in enumerate(kk):
                        recs.append((ser, v['obs'], int(k), v['role'], int(nk[i_]), int(cnt[i_])))
        arr = {k_: np.concatenate(x_) for k_, x_ in acc.items()}
        assert len(arr['cm']) == n and int(arr['nk'].sum()) == len(arr['tp']) == len(arr['ac'])
        loc = R.local_dir() / f'{v["obs"]}.npz'
        np.savez(loc, codes_max=arr['cm'], argmax=arr['am'], n_kept=arr['nk'], tok_pos=arr['tp'], act_count=arr['ac'],
                 act_idx=arr['ai'], act_val=arr['av'], obs=v['obs'], n=n, G=Se.G, W=W, sae=b['name'])
        R.atomic_copy(loc, outd / loc.name)
        loc.unlink()
        dt = time.time() - t0
        tot += n
        log(f'{ser} {v["obs"]} ({v["role"]}): {n} frames in {dt:.0f}s ({n / dt:.1f} fps; GPU {enc.gpu_s - g0:.0f}s) | '
            f'kept/frame {arr["nk"].mean():.1f} | token L0 {arr["ac"].mean():.2f} | frame L0 '
            f'{(arr["cm"] > 0).sum(1).mean():.1f} | 0-kept frames {(arr["nk"] == 0).sum()} | store tokens {sum(r[5] for r in recs):,}')
    ftok.close()
    np.save(loc_s / 'tok_frame.npy', np.concatenate(st_tf) if st_tf else np.zeros(0, np.int32))
    np.save(loc_s / 'tok_pos.npy', np.concatenate(st_pos) if st_pos else np.zeros(0, np.int16))
    Fm = pd.DataFrame(recs, columns=['series', 'obs', 'k', 'role', 'n_kept', 'n_store'])
    Fm.to_parquet(loc_s / 'frames.parquet')
    nt = int(Fm.n_store.sum())
    assert NOSTORE or (loc_s / 'tok.f16').stat().st_size == nt * 1024 * 2
    wall = time.time() - T0
    info = {'domain': a.domain, 'shard': tag, 'frames': tot, 'store_tokens': nt, 'store_bytes': nt * 2048,
            'kept_tokens': int(Fm.n_kept.sum()), 'p_store': P_STORE, 'wall_s': wall, 'gpu_s': enc.gpu_s,
            'fps_wall': tot / max(wall, 1), 'gpu': torch.cuda.get_device_name(0), 'sae': b['name']}
    (OUT / a.domain / 'encode').mkdir(parents=True, exist_ok=True)
    (OUT / a.domain / 'encode' / f'{tag}.json').write_text(json.dumps(info, indent=1))
    if not NOSTORE:
        R.atomic_copy(loc_s, BEE / a.domain / 'store5' / tag)
    log(f'{a.domain} shard {a.shard} DONE: ' + json.dumps(info))


# ---------------------------------------------------------------------------------------------- HMM readout
_G = {}


def _fit_one(j):
    from src.eci.v2 import hmm as HM
    p = HM.fit(_G['Xc'][j], _G['sc'], K=3)
    return j, p


def _apply_block(args):
    from src.eci.v2 import hmm as HM
    j0, j1, prm, starts, src, dst_p, dst_v, n = args
    X = np.load(src, mmap_mode='r')
    Pm = np.load(dst_p, mmap_mode='r+')
    Vm = np.load(dst_v, mmap_mode='r+')
    for j in range(j0, j1):
        x = np.asarray(X[j], np.float64)
        p, vv = HM.apply(x, starts, prm[j])
        Pm[j] = p.astype(np.float16)
        Vm[j] = vv.astype(np.float16)
    Pm.flush()
    Vm.flush()
    return j1 - j0


def cmd_hmm(a):
    import shutil
    from multiprocessing import get_context
    from src.eci.v2 import readout as RO
    dom = a.domain
    c = dcfg(dom)
    name = R.best_name(c)
    ncpu = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
    loc = R.local_dir()
    raws, plans = {}, {}
    cal, cs = [], [0]
    for sc in c['series']:
        P = json.loads((REPO / sc['plan']).read_text())
        plans[sc['name']] = P
        src = R.codes_dir(c, sc, f'{name}_raw') / 'codes_max.npy'
        dst = loc / f'raw_{sc["name"]}.npy'
        shutil.copyfile(src, dst)
        raws[sc['name']] = dst
        mx = np.load(dst, mmap_mode='r')
        for v in P['videos']:
            if v['role'] == 'calib':
                cal.append(np.asarray(mx[v['lay_lo']:v['lay_lo'] + v['n']], np.float32))
                cs.append(cs[-1] + v['n'])
    Xc = np.ascontiguousarray(np.concatenate(cal).T)  # (W, n_cal)
    del cal
    W = Xc.shape[0]
    _G.update(Xc=Xc, sc=np.array(cs, np.int64))
    t0 = time.time()
    prm = [None] * W
    with get_context('fork').Pool(ncpu) as pool:
        for j, p in pool.imap_unordered(_fit_one, range(W), chunksize=8):
            prm[j] = p
    log(f'hmm {dom}: fitted {W} neurons on {Xc.shape[1]:,} calibration frames ({len(cs) - 1} videos) in {time.time() - t0:.0f}s')
    fb = np.array([p is None for p in prm])
    keys = ('pi0', 'A', 'z', 'mu', 'var', 'ec')
    np.savez(OUT / dom / 'hmm_params.npz', fallback=fb, on=np.array([-1 if p is None else p['on'] for p in prm]),
             iters=np.array([0 if p is None else p['iters'] for p in prm]),
             **{k: np.stack([np.full_like(prm[np.flatnonzero(~fb)[0]][k], np.nan) if p is None else p[k] for p in prm])
                for k in keys})
    from src.eci.v2 import hmm as HM
    stat = np.array([HM.stationary_on(p) for p in prm])
    summ_all = {}
    for sc in c['series']:
        P = plans[sc['name']]
        mx = np.load(raws[sc['name']], mmap_mode='r')
        n = mx.shape[0]
        T = np.lib.format.open_memmap(loc / 'T.npy', 'w+', np.float16, (W, n))
        for r0 in range(0, n, 200_000):
            T[:, r0:r0 + 200_000] = np.asarray(mx[r0:r0 + 200_000]).T
        T.flush()
        del T
        vids = sorted(P['videos'], key=lambda v: v['lay_lo'])
        starts = np.array([v['lay_lo'] for v in vids] + [vids[-1]['lay_lo'] + vids[-1]['n']], np.int64)
        assert np.all(np.diff(starts) == [v['n'] for v in vids]) and starts[0] == 0 and starts[-1] == n, 'layout gaps'
        for nm in ('TP', 'TV'):
            np.lib.format.open_memmap(loc / f'{nm}.npy', 'w+', np.float16, (W, n)).flush()
        t0 = time.time()
        blocks = [(j0, min(W, j0 + 16), prm, starts, str(loc / 'T.npy'), str(loc / 'TP.npy'), str(loc / 'TV.npy'), n)
                  for j0 in range(0, W, 16)]
        with get_context('fork').Pool(ncpu) as pool:
            done = sum(pool.imap_unordered(_apply_block, blocks))
        assert done == W
        log(f'hmm {dom} {sc["name"]}: applied to {n:,} frames in {time.time() - t0:.0f}s')
        TP, TV = np.load(loc / 'TP.npy', mmap_mode='r'), np.load(loc / 'TV.npy', mmap_mode='r')
        for suf, Tm, rule in (('_hmm', TP, 'per-neuron 3-state HMM posterior P(on) (src/eci/v2/hmm.py), fitted on the '
                               'calibration videos of the domain'), ('_hmmv', TV, 'Viterbi path of the same HMM: 1 = on')):
            cd = R.codes_dir(c, sc, name + suf)
            tmp = loc / f'out{suf}'
            tmp.mkdir(exist_ok=True)
            y = np.lib.format.open_memmap(tmp / 'codes_max.npy', 'w+', np.float16, (n, W))
            for r0 in range(0, n, 200_000):
                y[r0:r0 + 200_000] = np.asarray(Tm[:, r0:r0 + 200_000]).T
            y.flush()
            del y
            src = R.codes_dir(c, sc, f'{name}_raw')
            shutil.copyfile(src / 'n_fg.npy', tmp / 'n_fg.npy')
            cfg = json.loads((src / 'config.json').read_text())
            cfg.update({'sae': name + suf, 'parent_codes': str(src), 'rule': rule})
            (tmp / 'config.json').write_text(json.dumps(cfg, indent=1))
            (tmp / 'DONE').touch()
            R.atomic_copy(tmp, cd)
            shutil.rmtree(tmp)
        # per-video summaries
        Q = ('mean_p_on', 'time_on_viterbi', 'n_bouts', 'bout_rate_per_min', 'latency_s', 'censored', 'mean_continuous')
        S = {q: np.zeros((len(vids), W), np.float32) for q in Q}
        for i, v in enumerate(vids):
            r = slice(v['lay_lo'], v['lay_lo'] + v['n'])
            Pv = np.asarray(TP[:, r], np.float32).T
            Vv = np.asarray(TV[:, r]).T > 0.5
            S['mean_p_on'][i] = Pv.mean(0)
            for q, val in RO.video_summaries(Vv, 5.0).items():
                S[{'time_spent': 'time_on_viterbi'}.get(q, q)][i] = val
            S['mean_continuous'][i] = RO.smooth(np.asarray(mx[r]), 1).mean(0)
        f = OUT / dom / f'summaries_{sc["name"]}.npz'
        np.savez_compressed(f, obs=np.array([v['obs'] for v in vids]), n_frames=np.array([v['n'] for v in vids]),
                            sae=name, **S, rule='mean_p_on = mean HMM P(on) (primary); bouts / latency from the Viterbi '
                            'on-state (bout = maximal on run; latency = first on frame / 5 fps, censored = never on); '
                            'mean_continuous = mean of the 0.6 s centred mean of the frame max')
        summ_all[sc['name']] = {'summaries': str(f), 'median_mean_p_on': float(np.median(S['mean_p_on'])),
                                'median_time_on_viterbi': float(np.median(S['time_on_viterbi'])),
                                'never_on_video_neuron_frac': float(S['censored'].mean())}
        del TP, TV
        for nm in ('T', 'TP', 'TV'):
            (loc / f'{nm}.npy').unlink()
    out = {'sae': name, 'K': 3, 'fit': 'calibration videos of the domain', 'n_fallback': int(fb.sum()),
           'stationary_on_share_median': float(np.nanmedian(stat[:, 0])),
           'mean_on_bout_frames_median': float(np.nanmedian(stat[:, 1])), 'series': summ_all}
    (OUT / dom / 'hmm.json').write_text(json.dumps(out, indent=1, default=float))
    log(json.dumps(out, default=float))


# ---------------------------------------------------------------------------------------------- checks / links
def cmd_heatcheck(a):
    sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
    import tools as TL
    dom = a.domain
    c = dcfg(dom)
    rng = np.random.default_rng(0)
    res = {'frames': 0, 'max_abs_diff': 0.0, 'n_nonzero_checked': 0}
    for sc in c['series']:
        P = json.loads((REPO / sc['plan']).read_text())
        for vi in rng.choice(len(P['videos']), 3, replace=False):
            v = P['videos'][vi]
            T = TL.TokenCodes(BEE / dom / 'videos' / sc['name'] / f'{v["obs"]}.npz')
            for k in rng.choice(v['n'], 20, replace=False):
                H = np.zeros((T.G * T.G, T.cm.shape[1]), np.float32)
                for t in range(T.tok0[k], T.tok0[k + 1]):
                    a0, a1 = T.act0[t], T.act0[t + 1]
                    H[T.pos[t], T.idx[a0:a1].astype(np.int64)] = T.val[a0:a1]
                hm = H.max(0)
                cm = T.cm[k].astype(np.float32)
                res['max_abs_diff'] = max(res['max_abs_diff'], float(np.abs(hm - cm).max()))
                res['frames'] += 1
                res['n_nonzero_checked'] += int((cm > 0).sum())
                j = int(np.argmax(cm))
                if cm[j] > 0:  # the atlas helper itself on the frame's top neuron
                    assert abs(T.heat(k, j).max() - cm[j]) < 1e-3
    res['PASS'] = res['max_abs_diff'] == 0.0
    (OUT / dom / 'heatcheck.json').write_text(json.dumps(res, indent=1))
    log(json.dumps(res))
    assert res['PASS']


def cmd_link(a):
    c = dcfg(a.domain)
    name = R.best_name(c)
    b = json.loads((OUT / a.domain / 'best.json').read_text())
    links = []
    for sc in c['series']:
        lay = REPO / sc['layout']
        for nm in (f'{name}_raw', f'{name}_smooth', f'{name}_hmm', f'{name}_hmmv'):
            for kind, tgt in (('codes', R.codes_dir(c, sc, nm)), ('sae', Path(b['dir']))):
                l_ = lay / kind / nm
                if not (l_.is_symlink() or l_.exists()):
                    os.symlink(tgt, l_)
                links.append(str(l_))
    (OUT / a.domain / 'links.json').write_text(json.dumps(links, indent=1))
    log('\n'.join(links))


# ---------------------------------------------------------------------------------------------- 5 fps retrain
SEEDS5 = (950, 951, 952, 953, 954)


class HostTok:
    """(n, d) fp16 host tensor indexed by device indices -> device rows (same values as a device tensor)."""

    def __init__(self, x, dev):
        self.x, self.dev, self.shape = x.pin_memory(), dev, x.shape

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i.cpu()].to(self.dev, non_blocking=True)
CAP5 = {'train': 24_000_000, 'calib': 2_000_000, 'eval': 2_000_000}


def cmd_train5(a):
    """SAE retrain on the PASS A 5 fps stores (20% of each frame's N2 tokens; mice v1 + v2, ants v2 + v3 + v4, frogs
    juv + tad), the v2.0 recipe (2048, 128/256/512/2048, k16, 30k x 4096), seeds SEEDS5. Tokens per role drawn
    uniformly without replacement (rng 0) up to CAP5 (the GPU cap; V2F_N2 mice trained on 23.0M tokens)."""
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    dom = a.domain
    s = R.spec()
    shards = sorted(p_ for p_ in (BEE / dom / 'store5').iterdir() if not p_.name.endswith('.tmp'))
    parts = []
    for sp in shards:
        F_ = pd.read_parquet(sp / 'frames.parquet')
        tf = np.load(sp / 'tok_frame.npy')
        parts.append((sp, F_, F_.role.values[tf]))
    info = {'shards': [p_.name for p_ in shards], 'frames': int(sum(len(x[1]) for x in parts)),
            'frames_by_series': pd.concat([x[1].series for x in parts]).value_counts().to_dict()}
    rng = np.random.default_rng(0)
    X = {}
    for r in ('train', 'calib', 'eval'):
        n_r = [int((x[2] == r).sum()) for x in parts]
        tot = sum(n_r)
        take = min(tot, CAP5[r])
        g = np.sort(rng.choice(tot, take, replace=False))
        out = np.empty((take, 1024), np.float16)
        o, base = 0, 0
        t0 = time.time()
        for (sp, F_, role), n_ in zip(parts, n_r):
            loc = np.flatnonzero(role == r)
            sel = loc[g[(g >= base) & (g < base + n_)] - base]
            base += n_
            mm = np.memmap(sp / 'tok.f16', np.float16, 'r', shape=(len(role), 1024))
            for c0 in range(0, len(sel), 2_000_000):
                q = sel[c0:c0 + 2_000_000]
                out[o:o + len(q)] = mm[q]
                o += len(q)
        assert o == take
        info[f'tokens_{r}'] = {'available': tot, 'used': take, 'read_s': round(time.time() - t0)}
        log(f'train5 {dom} {r}: {take:,} of {tot:,} tokens ({time.time() - t0:.0f}s)')
        if r == 'train' and os.environ.get('V20_HOSTTOK') == '1':  # tokens stay in pinned host memory (48 GB GPUs)
            X[r] = HostTok(torch.from_numpy(out), dev)
        else:
            X[r] = torch.from_numpy(out).to(dev) if r == 'train' else torch.from_numpy(out)
    cfg = {**s['sae'], 'context_window': 0}
    if a.steps:
        cfg['steps'] = a.steps
    recs = {}
    tf_ = OUT / dom / f'train5{"_" + str(a.steps) if a.steps else ""}.json'
    for seed in SEEDS5:
        od = BEE / dom / 'sae5' / f'v20_{dom}_N2f5_s{seed}{"_" + str(a.steps) if a.steps else ""}'
        recs[seed] = SA.train(cfg, seed, X['train'], X['calib'], X['eval'], od, dev)
        r = recs[seed]
        log(f'{od.name}: {r["train_time_s"]:.0f}s theta* {r["theta_star"]:.4f} eval FVE {r["eval_fve"]:.5f} '
            f'L0 {r["eval_l0"]:.2f} dead {r["dead_at_end_train"]}')
        tf_.write_text(json.dumps({'data': info, 'seeds': recs}, indent=1, default=float))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step')
    p.add_argument('--domain', default='mice')
    p.add_argument('--series', default=None)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--videos', default=None)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--force', action='store_true')
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--steps', type=int, default=0)
    a = p.parse_args()
    steps = {'bgs': cmd_bgs, 'store': cmd_store, 'train': cmd_train, 'best': cmd_best, 'selftest': cmd_selftest,
             'encode': cmd_encode, 'merge': R.cmd_merge, 'smooth': R.cmd_smooth, 'hmm': cmd_hmm,
             'heatcheck': cmd_heatcheck, 'link': cmd_link, 'train5': cmd_train5}
    steps[a.step](a)


if __name__ == '__main__':
    main()
