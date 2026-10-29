"""
ECI animal mask v3 candidate (src/eci/v2/mask3.py) vs the frozen v2.0 N2 mask: development panels, pre-registered
validation, approval sheets. CPU for the lab, GPU for the benchmarks. No token is encoded or stored.

Lab path: OLD = scripts/eci/v2/v20.py N2Series exactly (its frames(): decode, canonical rotation, frozen core cue,
frog ROI planes, tadpole crops, ant background windows; asserted equal to N2Series.n2 on the first frame). NEW = a
subclass that changes only (1) the background pack (mask3.fix_pack: ghost region filled with q98, MAD inpainted) and
(2) the per-frame rule (mask3.keep3: no ghost guard). Everything else is literally the same code path.

DEVELOPMENT (before this pre-registration; step dev, OUT/dev/): diagnosis of the pale-tail miss on mice v1
wt_ash1l_m_1_S_H k 6001 (mask3_diag.py) and the fill choice ('q98' vs 'telea') on 10 dev frames (4 mice ghost videos,
1 ant, 1 juvenile frog, 1 tadpole): q98 chosen (no residual ghost, bedding texture kept; telea left 0.12% residual ghost
on het_kmt5b_m_2_F_H). Dev frames are reused in the sheets as named hard cases, never in the random metric frames
(excluded by obs + k).

==================================================================================================================
PRE-REGISTRATION (2026-10-29, written before any metric below was computed; no adoption rule, the user decides)
FRAMES (lab, per series mice v1 / v2, ants v2 / v3 / v4, frogs juv / tad): rng = default_rng(0); min(40, n) videos
    chosen without replacement; per video 3 frames k drawn uniformly from [0, n - 1] (sorted). Decode failures are
    counted and reported, not replaced.
LAB METRICS (old N2 vs new mask3, same frames):
    tokens per frame (mean); added / removed share = |new & ~old| / |old|, |old & ~new| / |old| (summed);
    LEAK A = share of kept tokens with < 5% of their pixels in (frozen core cue | d_raw > T_LO), d_raw = max_c
        |blur(x) - blur(stored per-video median)| (the frozen v2_mask2 lab-leak measure, unchanged);
    LEAK B = the same with d against the background each arm actually uses (old: median; new: fixed);
    GHOST AREA (step ghost, ALL videos of each series): share of the frame where the frozen dark rule holds on the
        background used (old = stored median; new = fixed background); median, p90, max, n videos > 2%; plus the
        fixed region area (median, max).
TAIL / APPENDAGE RECALL (lab mice, eye check): the first 15 sampled frames of v1 and of v2 (rng order) rendered as
    frame | old | new (OUT/tailcheck_*.png). Claude (not code) counts, per mouse whose tail is visible, whether at least
    half of the visible tail length lies in kept tokens, for old and new, and counts kept off-animal blobs (kept
    tokens on bedding / wall not touching an animal, and bedding blobs attached to an animal). Reported as eye counts
    with this procedure. (The AMADEUS pilot keypoints, 3 videos, are not used.)
BENCHMARKS (GPU, step bench, scripts/eci/v2_mask2/validate.py frames, packs, h and score_frame unchanged): N2 =
    validate.masks_all 'N2'; NEW = mask3.fix_pack on the validation pack (dark_abs 60, dark_rel 40, h = validate h) +
    mask3.keep3 with the same core (validate.dark_cue, pack q98). MABe22 tail_seg_tok, tail_tip_tok, tail_base_tok,
    body_tok; CalMS21 pale_tok, pale_missed (= white-intruder miss rate), body_tok, tail_base_tok_*; CRIM13 pale_tok,
    pale_tok_tol1, res_tok, c57_tok; keypoint-region leak and tokens / frame for all; ghost area of the validation
    packs old / new. IDENTITY: the N2 numbers must equal results/vision/eci_v2_mask2/validate/<set>_final.json.
APPROVAL SHEETS (OUT/<domain>.png; frame | old N2 (blue) | new (green)): per series 2 frames drawn with rng 1 from
    the metric frames (mice 2 x 2 + 1 = 5 incl. one extra v1, ants 2 per series, frogs 3 per series), plus hard cases:
    mice huddle (dev ghost videos het_kmt5b_m_2_F_H k 1000, het_ash1l_m_3_F_O k 1500, v2 mixed_kmt5b_m_1_F_H k 1000),
    pale tail (wt_ash1l_m_1_S_H k 6001), odour box (the metric frame of v1 with the most old-N2 tokens in the
    canonical top-right quarter); ants: the middle frame of the video with the largest ghost area per series; frogs:
    dish rim = the juv metric frame with the most old-N2 tokens in the outer 10% border of the frame; benchmarks: 2
    validate frames per set (rng 1) -> OUT/benchmarks.png.
POST-HOC AMENDMENT 1 (2026-10-29 ~17:00, AFTER the pre-registered mask3 lab metrics of all 7 series, the mice v2
    eye check and 3 mice v1 eye-check frames were seen; before any mask3b number): mice show bedding halos and wall
    blocks kept next to old huddle spots (mice v1 / v2 +19% / +18% tokens, LEAK A 0.079). Cause: inside the fixed
    region the q98 fill differs from the current frame where the median was right (wall corners that the frozen ghost
    rule flags, displaced bedding). Rule mask3b (mask3.keep3b): candidate / strong-core pixels must differ from BOTH the
    fixed background and the stored median, d = min(d_fixed, d_median); fix_pack unchanged (q98 fill). Also found:
    mask2.hyst_pixels overwrites its per-pixel MAD argument with the per-frame scalar noise MAD inside its loop, so
    the per-pixel background-MAD term never acts in N2 (mask3's MAD inpainting therefore changes nothing). Same frames,
    metrics and sheets for mask3b (suffix _m3b); the benchmark step reports N2, mask3 and mask3b (bench_<set>_v2.json).
    A development sweep (frames outside the metric sample, OUT/sweep*/) found REACH_H 10 and T_LO 25 add wall /
    bedding blocks; they are NOT adopted. mask3 numbers stay reported as pre-registered.
==================================================================================================================

Steps (scripts/eci/v2/mask3.sh, PY = this file):
    dev   --cases dom/ser/obs/k,...  [--fills q98,telea]   development panels (no metric) -> OUT/dev/
    lab   --dom D --ser S          (CPU) lab metrics + sheet rows -> OUT/lab_<D>_<S>.json, OUT/rows/
    ghost --dom D --ser S          (CPU) ghost area, all videos -> OUT/ghost_<D>_<S>.json
    bench --set mab|cal|crim       (GPU) -> OUT/bench_<set>.json, OUT/rows/
    sheet                          (CPU) -> OUT/{mice,ants,frogs,benchmarks}.png, OUT/table.md
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
OUT = REPO / 'results/vision/eci_v2_0B/mask_approval'
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def env():
    os.environ.setdefault('V2_OUT', str(REPO / 'results/vision/eci_v2_0B'))
    os.environ.setdefault('V2_BEE', '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_0B')
    import v20 as V
    return V


def series_pair(dom, ser, dev, fill=None, rule='m3'):
    """-> (old N2 series, new mask3 series) sharing nothing but the config; both capture per-frame intermediates."""
    import torch
    import torch.nn.functional as F
    from src.eci.v2 import mask2 as M2
    from src.eci.v2 import mask3 as M3
    V = env()
    NS = V.make_series_class(dom)
    sc = V.R.series_of(V.dcfg(dom), ser)
    pool = lambda m: F.avg_pool2d(m.float()[:, None], 16).flatten(1)  # noqa: E731

    class Old(NS):
        def n2(self, x, core0, pack, roi=None):
            r = M2.hyst_pixels(x, pack['bg'], core0, self.h, t_lo=M2.T_LO, t_hi=M2.T_HI, k_noise=M2.K_NOISE,
                               reach_h=M2.REACH_H, sigma=M2.SIGMA, strong=True, mad=pack['mad'], k_mad=M2.K_MAD,
                               bad=pack['ghost'], roi=roi)
            _, app = M2.appendage(r['pix'], self.h)
            keep = (pool(r['pix']) >= 0.25) | (pool(app) * 256 >= M2.MIN_THIN_PX)
            if not getattr(self, '_checked', False):  # identity with the frozen N2Series.n2 on the first call
                k0, _ = NS.n2(self, x, core0, pack, roi=roi)
                assert torch.equal(k0, keep), 'capture path != N2Series.n2'
                self._checked = True
            self.cap.append({'x': x, 'core0': core0, 'bg_raw': pack['bg'], 'roi': roi, 'pix': r['pix'], 'app': app,
                             'keep': keep})
            return keep, pool(core0) >= 0.25

    class New(NS):
        def pack(self, obs, w):
            key = ('m3', obs, w)
            if key not in self._packs:
                P = NS.pack(self, obs, w)
                m = self.cfg['mask']
                da, dr = m.get('dark_abs', 60), m.get('dark_rel', 40)
                if self.geometry == 'crop':
                    fx = M3.fix_pack(P['bg'], P['q98'], P['mad'], self.h, da, dr, fill=fill)
                    bgc = fx['bg'] if rule == 'm3' else np.concatenate([fx['bg'], np.asarray(P['bg'], np.float32)], 0)
                    # m3b: channels 3-5 carry the median through the frozen crop code (grey_f reads channels 0-2 only)
                    out = {'bg': bgc, 'mad': fx['mad'], 'q98': P['q98'], 'bg_raw': P['bg']}
                else:
                    tn = lambda t: t.cpu().numpy()  # noqa: E731
                    fx = M3.fix_pack(tn(P['bg']), tn(P['q98']), tn(P['mad']), self.h, da, dr, fill=fill)
                    dv = self.dev
                    out = {'bg': torch.from_numpy(fx['bg']).to(dv), 'mad': torch.from_numpy(fx['mad']).to(dv),
                           'q98': P['q98'], 'ghost': torch.zeros_like(P['ghost']), 'bg_raw': P['bg']}
                g2 = M3.grey_np(fx['bg'])
                out['ghost_area_old'] = float(fx['ghost'].mean())
                out['ghost_area_new'] = float(((g2 < da) & ((np.asarray(P['q98'], np.float32) - g2) > dr)).mean())
                out['region_area'] = float(fx['region'].mean())
                if len(self._packs) > 8:
                    self._packs.clear()
                self._packs[key] = out
            return self._packs[key]

        def n2(self, x, core0, pack, roi=None):
            bgf = pack['bg']
            if rule == 'm3':
                keep, pix, app = M3.keep3(x, bgf, pack['mad'], core0, self.h, roi=roi)
            else:
                if self.geometry == 'crop':
                    bgf, bgm = bgf[:, :3], bgf[:, 3:]
                else:
                    bgm = pack['bg_raw']
                keep, pix, app = M3.keep3b(x, bgf, bgm, core0, self.h, roi=roi)
            self.cap.append({'x': x, 'core0': core0, 'bg_new': bgf, 'roi': roi, 'pix': pix, 'app': app,
                             'keep': keep})
            return keep, pool(core0) >= 0.25

    o, n = Old(sc, dev, load_encoder=False), New(sc, dev, load_encoder=False)
    o.cap, n.cap = [], []
    return o, n


def masks_for(o, n, v, ks):
    """decode once, run both -> (old caps, new caps) lists (one entry per background window / crop batch)."""
    out = []
    for k in ks:
        try:
            fr = o.run_job(**o.job(v, int(k), 1, threads=2))
        except RuntimeError as e:  # fail loud in the log, keep going (counted by the caller)
            log(f'DECODE FAIL {v["obs"]} k {k}: {str(e)[:200]}')
            continue
        o.cap.clear()
        n.cap.clear()
        o.frames(v, fr, [int(k)])
        n.frames(v, fr, [int(k)])
        assert len(o.cap) == 1 and len(n.cap) == 1
        out.append((int(k), o.cap[0], n.cap[0]))
    return out


def tint(im, keep, G, col, a=0.4):
    S = im.shape[0]
    K = np.kron(keep.reshape(G, G), np.ones((S // G, S // G), bool))
    o = im.astype(np.float32).copy()
    o[K] = (1 - a) * o[K] + a * np.array(col)
    return o.clip(0, 255).astype(np.uint8)


def panel(rows, path, tile=512):
    """rows [(frame uint8 HWC, old keep, new keep, G, title)] -> PNG: frame | old N2 (blue) | new (green)."""
    from PIL import Image, ImageDraw
    sh = Image.new('RGB', (3 * tile, tile * len(rows)))
    for r, (im, ko, kn, G, title) in enumerate(rows):
        for j, (a, name) in enumerate(((im, 'frame'), (tint(im, ko, G, (0, 90, 255)), f'old N2 {int(ko.sum())} tok'),
                                       (tint(im, kn, G, (0, 220, 0)), f'new mask3 {int(kn.sum())} tok'))):
            t = Image.fromarray(a).resize((tile, tile), Image.BILINEAR)
            ImageDraw.Draw(t).text((5, 5), f'{name} | {title}', fill=(255, 255, 0))
            sh.paste(t, (j * tile, r * tile))
    path.parent.mkdir(parents=True, exist_ok=True)
    sh.save(path)


def cmd_dev(a):
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    dev = torch.device('cpu')
    cases = [c.split('/') for c in a.cases.split(',')]
    for fill in a.fills.split(','):
        rows, info = [], []
        (OUT / 'dev').mkdir(parents=True, exist_ok=True)
        pairs = {}
        for dom, ser, obs, k in cases:
            if (dom, ser) not in pairs:
                pairs[(dom, ser)] = series_pair(dom, ser, dev, fill=fill)
            o, n = pairs[(dom, ser)]
            v = o.V[obs]
            for kk, co, cn in masks_for(o, n, v, [int(k)]):
                G = o.G
                im = co['x'][0].permute(1, 2, 0).numpy()
                ko, kn = co['keep'][0].numpy(), cn['keep'][0].numpy()
                P = n.pack(obs, env().window_of(dom, kk))
                rows.append((im, ko, kn, G, f'{dom} {ser} {obs} k{kk} ghost {P["ghost_area_old"]:.3f}'))
                from PIL import Image
                Image.fromarray(np.concatenate([im, tint(im, ko, G, (0, 90, 255)), tint(im, kn, G, (0, 220, 0))], 1)).save(
                    OUT / 'dev' / f'case_{fill}_{obs}_{kk}.png')
                info.append({'case': f'{dom}/{ser}/{obs}/{kk}', 'old': int(ko.sum()), 'new': int(kn.sum()),
                             'added': int((kn & ~ko).sum()), 'removed': int((ko & ~kn).sum()),
                             'ghost_area_old': P['ghost_area_old'], 'ghost_area_new': P['ghost_area_new'],
                             'region_area': P['region_area']})
                if a.bg:
                    from PIL import Image
                    hw = lambda t: (t if isinstance(t, np.ndarray) else t.cpu().numpy()).transpose(1, 2, 0)  # noqa
                    Image.fromarray(np.concatenate([hw(P['bg_raw']), hw(P['bg'])], 1).clip(0, 255).astype(np.uint8)).save(
                        OUT / 'dev' / f'bg_{fill}_{obs}.png')
                log(json.dumps(info[-1]))
        panel(rows, OUT / 'dev' / f'dev_{fill}.png', tile=448)
        (OUT / 'dev' / f'dev_{fill}.json').write_text(json.dumps(info, indent=1))


HARD = {('mice', 'v1'): [('het_kmt5b_m_2_F_H', 1000, 'huddle'), ('het_ash1l_m_3_F_O', 1500, 'huddle'),
                         ('wt_ash1l_m_1_S_H', 6001, 'pale tail')],
        ('mice', 'v2'): [('mixed_kmt5b_m_1_F_H', 1000, 'huddle')]}
N_SHEET = {'mice': 2, 'ants': 2, 'frogs': 3}


def sample_frames(se):
    g = np.random.default_rng(0)
    vids = se.videos
    pick = g.choice(len(vids), min(40, len(vids)), replace=False)
    out = []
    for i in pick:
        v = vids[i]
        for k in np.sort(g.integers(0, v['n'], 3)):
            out.append((v['obs'], int(k)))
    dev = {(o, k) for L in HARD.values() for o, k, _ in L}
    return [f for f in out if f not in dev]


SUF = {'m3': '', 'm3b': '_m3b'}
RULE = ['m3']


def save_row(im, ko, kn, G, title, name):
    name = name + SUF[RULE[0]]
    from PIL import Image
    (OUT / 'rows').mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([im, tint(im, ko, G, (0, 90, 255)), tint(im, kn, G, (0, 220, 0))], 1)).save(
        OUT / 'rows' / f'{name}.png')
    (OUT / 'rows' / f'{name}.txt').write_text(f'{title} | old {int(ko.sum())} tok | new {int(kn.sum())} tok')


def cmd_lab(a):
    import torch
    import torch.nn.functional as F
    from src.eci.v2 import mask2 as M2
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    dev = torch.device('cpu')
    o, n = series_pair(a.dom, a.ser, dev, rule=a.rule)
    fr = sample_frames(o)
    t0 = time.time()
    acc = {'old': {'tok': 0, 'leakA': 0, 'leakB': 0}, 'new': {'tok': 0, 'leakA': 0, 'leakB': 0}, 'add': 0, 'rem': 0,
           'frames': 0, 'decode_fail': 0}
    pool = lambda m: F.avg_pool2d(m.float()[:, None], 16).flatten(1)  # noqa: E731

    def dmap(x, bg):
        xf = M2.gauss_blur(x.float(), M2.SIGMA)
        bf = M2.gauss_blur((bg if bg.dim() == 4 else bg[None]).float(), M2.SIGMA)
        return (xf - bf).abs().amax(1)
    kept = []  # (obs, k, old keep, new keep, image) for the sheet / eye check selections
    for j, (obs, k) in enumerate(fr):
        res = masks_for(o, n, o.V[obs], [k])
        if not res:
            acc['decode_fail'] += 1
            continue
        _, co, cn = res[0]
        x, c0 = co['x'], co['core0']
        bgr = co['bg_raw'] if torch.is_tensor(co['bg_raw']) else torch.from_numpy(np.asarray(co['bg_raw'])).float()
        bgn = cn['bg_new'] if torch.is_tensor(cn['bg_new']) else torch.from_numpy(np.asarray(cn['bg_new'])).float()
        dr = dmap(x, bgr)
        dn = dmap(x, bgn)
        anyA = pool(c0 | (dr > M2.T_LO))[0] >= 0.05
        anyBo = anyA
        anyBn = pool(c0 | (dn > M2.T_LO))[0] >= 0.05
        ko, kn = co['keep'][0], cn['keep'][0]
        acc['old']['tok'] += int(ko.sum())
        acc['new']['tok'] += int(kn.sum())
        acc['old']['leakA'] += int((ko & ~anyA).sum())
        acc['new']['leakA'] += int((kn & ~anyA).sum())
        acc['old']['leakB'] += int((ko & ~anyBo).sum())
        acc['new']['leakB'] += int((kn & ~anyBn).sum())
        acc['add'] += int((kn & ~ko).sum())
        acc['rem'] += int((ko & ~kn).sum())
        acc['frames'] += 1
        kept.append((obs, k, ko.numpy(), kn.numpy(), x[0].permute(1, 2, 0).numpy().copy()))
        if j % 20 == 0:
            log(f'{a.dom} {a.ser} {j + 1}/{len(fr)} {time.time() - t0:.0f}s')
    G = o.G
    nf = max(acc['frames'], 1)
    res = {'dom': a.dom, 'ser': a.ser, 'frames': acc['frames'], 'decode_fail': acc['decode_fail'], 'h': o.h, 'G': G,
           'wall_s': time.time() - t0}
    for arm in ('old', 'new'):
        A = acc[arm]
        res[arm] = {'tokens_per_frame': A['tok'] / nf, 'leakA': A['leakA'] / max(A['tok'], 1),
                    'leakB': A['leakB'] / max(A['tok'], 1)}
    res['added_share'] = acc['add'] / max(acc['old']['tok'], 1)
    res['removed_share'] = acc['rem'] / max(acc['old']['tok'], 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'lab_{a.dom}_{a.ser}{SUF[a.rule]}.json').write_text(json.dumps(res, indent=1))
    log(json.dumps(res))
    # ---- sheet rows: random (rng 1), hard cases, rule-picked cases
    g1 = np.random.default_rng(1)
    nr = N_SHEET[a.dom] + (1 if (a.dom, a.ser) == ('mice', 'v1') else 0)
    for i in g1.choice(len(kept), min(nr, len(kept)), replace=False):
        ob, k, ko, kn, im = kept[i]
        save_row(im, ko, kn, G, f'random | {a.dom} {a.ser} {ob} k{k}', f'{a.dom}_{a.ser}_rand_{ob}_{k}')
    hard = list(HARD.get((a.dom, a.ser), []))
    if (a.dom, a.ser) == ('mice', 'v1'):
        tr = np.zeros((G, G), bool)
        tr[:G // 4, -(G // 4):] = True
        i = int(np.argmax([(t[2].reshape(G, G) & tr).sum() for t in kept]))
        ob, k, ko, kn, im = kept[i]
        save_row(im, ko, kn, G, f'odour box | {a.dom} {a.ser} {ob} k{k}', f'{a.dom}_{a.ser}_odour_{ob}_{k}')
    if a.dom == 'frogs' and a.ser == 'juv':
        b = np.ones((G, G), bool)
        w = max(1, G // 10)
        b[w:-w, w:-w] = False
        i = int(np.argmax([(t[2].reshape(G, G) & b).sum() for t in kept]))
        ob, k, ko, kn, im = kept[i]
        save_row(im, ko, kn, G, f'dish rim | {a.dom} {a.ser} {ob} k{k}', f'{a.dom}_{a.ser}_rim_{ob}_{k}')
    if a.dom == 'ants':
        gh = json.loads((OUT / f'ghost_{a.dom}_{a.ser}.json').read_text())
        ob = max(gh['videos'], key=lambda r: r['old'])['obs']
        hard.append((ob, int(o.V[ob]['n'] // 2), 'largest ghost'))
    for ob, k, why in hard:
        r = masks_for(o, n, o.V[ob], [k])
        if r:
            _, co, cn = r[0]
            save_row(co['x'][0].permute(1, 2, 0).numpy(), co['keep'][0].numpy(), cn['keep'][0].numpy(), G,
                     f'{why} | {a.dom} {a.ser} {ob} k{k}', f'{a.dom}_{a.ser}_hard_{ob}_{k}')
    if a.dom == 'mice':  # eye-check sheets: first 15 metric frames, 5 per image
        from PIL import Image, ImageDraw
        for b0 in range(0, 15, 5):
            rows = kept[b0:b0 + 5]
            sh = Image.new('RGB', (3 * 640, 640 * len(rows)))
            for r_, (ob, k, ko, kn, im) in enumerate(rows):
                for jj, a_ in enumerate((im, tint(im, ko, G, (0, 90, 255)), tint(im, kn, G, (0, 220, 0)))):
                    t = Image.fromarray(a_).resize((640, 640), Image.BILINEAR)
                    ImageDraw.Draw(t).text((5, 5), f'{["frame", "old", "new"][jj]} {ob} k{k}', fill=(255, 255, 0))
                    sh.paste(t, (jj * 640, r_ * 640))
            sh.save(OUT / f'tailcheck_{a.ser}_{b0 // 5}{SUF[a.rule]}.png')


def cmd_ghost(a):
    from src.eci.v2 import mask3 as M3
    V = env()
    sc = V.R.series_of(V.dcfg(a.dom), a.ser)
    P = json.loads((REPO / sc['plan']).read_text())
    m = sc['mask']
    da, dr = m.get('dark_abs', 60), m.get('dark_rel', 40)
    h = V.h_of(a.dom, a.ser)
    rows = []
    for v in P['videos']:
        nw = int(np.ceil(v['n'] / 3000)) if a.dom == 'ants' else 1
        go, gn, ra = [], [], []
        for w in range(nw):
            f = V.bg_file(a.dom, a.ser, v['obs'], w)
            if not f.exists():
                continue
            z = np.load(f)
            fx = M3.fix_pack(z['bg'], z['q98'], z['mad_s15'].astype(np.float32), h, da, dr)
            g2 = M3.grey_np(fx['bg'])
            go.append(float(fx['ghost'].mean()))
            gn.append(float(((g2 < da) & ((z['q98'].astype(np.float32) - g2) > dr)).mean()))
            ra.append(float(fx['region'].mean()))
        if go:
            rows.append({'obs': v['obs'], 'old': float(np.mean(go)), 'new': float(np.mean(gn)), 'region': float(np.mean(ra))})
    res = {'dom': a.dom, 'ser': a.ser, 'n_videos': len(rows)}
    for key in ('old', 'new', 'region'):
        x = np.array([r[key] for r in rows])
        res[key] = {'median': float(np.median(x)), 'p90': float(np.quantile(x, 0.9)), 'max': float(x.max()),
                    'n_gt_2pct': int((x > 0.02).sum())}
    res['videos'] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'ghost_{a.dom}_{a.ser}.json').write_text(json.dumps(res, indent=1))
    log(json.dumps({k: v for k, v in res.items() if k != 'videos'}))


def cmd_bench(a):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(REPO / 'scripts/eci/v2_mask2'))
    import validate as VA
    from src.eci.v2 import mask2 as M2
    from src.eci.v2 import mask3 as M3
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    kind = a.set
    t0 = time.time()
    if kind == 'cal':
        items = VA.cal_videos('test', 24, 0)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(6) as ex:
            loaded = list(ex.map(lambda it: VA.load_cal(it, 'test'), items))
        S, G = 1024, 64
        vids = [(nm, torch.from_numpy(xb).to(dev), torch.from_numpy(x).to(dev), ks, kps) for nm, xb, x, ks, kps in loaded]
    elif kind == 'crim':
        from concurrent.futures import ThreadPoolExecutor
        items = VA.crim_videos(12, 16, 0)
        with ThreadPoolExecutor(6) as ex:
            v0 = list(ex.map(lambda it: VA.load_crim(it, dev), items))
        vids = [(nm, xb, x, ks, [(xy[i], st) for i in range(len(ks))]) for nm, st, xb, x, ks, xy in v0]
        S, G = 640, 40
    else:
        sys.path.insert(0, str(REPO / 'scripts/eci/v2_mask2'))
        from common import mab_decode
        vids = []
        for c, ks, kps in VA.mab_clips(120, 8, 0):
            arr = mab_decode('mouse_submission', c)
            ib = np.linspace(0, len(arr) - 1, VA.N_BG).round().astype(int)
            vids.append((c, VA.up4(arr[ib], dev), VA.up4(arr[ks + 1], dev), ks, kps))
        S, G = 896, 56
    packs = [M2.background_pack(xb) for _, xb, _, _, _ in vids]
    h = VA.body_h([c.cpu().numpy() for (_, xb, _, _, _), pk in zip(vids, packs) for c in VA.dark_cue(xb[::6], pk['q98'])])
    rows = {'N2': [], 'NEW': [], 'NEWB': [], 'M0': []}
    gho, ghn, chg = [], [], {'old': 0, 'add': 0, 'rem': 0, 'addb': 0, 'remb': 0}
    sheet = []
    g1 = np.random.default_rng(1)
    pickv = set(g1.choice(len(vids), 2, replace=False).tolist())
    for vi, ((nm, xb, x, ks, kps), pk) in enumerate(zip(vids, packs)):
        tn = lambda t: t.cpu().numpy()  # noqa: E731
        fx = M3.fix_pack(tn(pk['bg']), tn(pk['q98']), tn(pk['mad']), h, 60, 40)
        gho.append(float(fx['ghost'].mean()))
        g2 = M3.grey_np(fx['bg'])
        ghn.append(float(((g2 < 60) & ((tn(pk['q98']) - g2) > 40)).mean()))
        bgn, madn = torch.from_numpy(fx['bg']).to(dev), torch.from_numpy(fx['mad']).to(dev)
        for b0 in range(0, len(x), 8):
            xx = x[b0:b0 + 8]
            ms, _ = VA.masks_all(xx, pk, h)
            core0 = VA.dark_cue(xx, pk['q98'])
            kn, pixn, _ = M3.keep3(xx, bgn, madn, core0, h)
            knb, pixb, _ = M3.keep3b(xx, bgn, pk['bg'], core0, h)
            for i in range(len(xx)):
                kk = kps[b0 + i]
                for arm, (kp_, px_) in (('M0', ms['M0']), ('N2', ms['N2'])):
                    rows[arm].append(VA.score_frame(kp_[i], px_[i], kk, S, G, h, kind))
                rows['NEW'].append(VA.score_frame(kn[i], pixn[i], kk, S, G, h, kind))
                rows['NEWB'].append(VA.score_frame(knb[i], pixb[i], kk, S, G, h, kind))
                chg['addb'] += int((knb[i] & ~ms['N2'][0][i]).sum())
                chg['remb'] += int((ms['N2'][0][i] & ~knb[i]).sum())
                ko_ = ms['N2'][0][i]
                chg['old'] += int(ko_.sum())
                chg['add'] += int((kn[i] & ~ko_).sum())
                chg['rem'] += int((ko_ & ~kn[i]).sum())
                if vi in pickv and b0 + i == 0:
                    sheet.append((xx[i].cpu().numpy().transpose(1, 2, 0), ko_.cpu().numpy(), kn[i].cpu().numpy(), nm,
                                  knb[i].cpu().numpy()))
    out = {'set': kind, 'S': S, 'G': G, 'h': h, 'n_videos': len(vids), 'n_frames': len(rows['N2']),
           'ghost_area_old_mean': float(np.mean(gho)), 'ghost_area_new_mean': float(np.mean(ghn)),
           'ghost_area_old_max': float(np.max(gho)), 'added_share': chg['add'] / max(chg['old'], 1),
           'removed_share': chg['rem'] / max(chg['old'], 1), 'added_share_b': chg['addb'] / max(chg['old'], 1),
           'removed_share_b': chg['remb'] / max(chg['old'], 1), 'wall_s': time.time() - t0,
           'device': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}
    for arm, R in rows.items():
        keys = sorted({k for r in R for k in r if k not in ('tokens', 'leak_n')})
        out[arm] = {k: float(np.mean([r[k] for r in R if k in r])) for k in keys}
        out[arm]['tokens_per_frame'] = float(np.mean([r['tokens'] for r in R]))
        out[arm]['leak'] = float(sum(r['leak_n'] for r in R) / max(sum(r['tokens'] for r in R), 1))
    ref = json.loads((REPO / f'results/vision/eci_v2_mask2/validate/{kind}_final.json').read_text())
    out['identity_N2_max_abs_diff'] = max(abs(out['N2'][k] - v) for k, v in ref['N2'].items())
    out['identity_M0_max_abs_diff'] = max(abs(out['M0'][k] - v) for k, v in ref['M0'].items())
    out['identity_h_diff'] = h - ref['h']
    (OUT / f'bench_{kind}_v2.json').write_text(json.dumps(out, indent=1))
    for im, ko, kn, nm, knb in sheet:
        save_row(im, ko, kn, G, f'random | {kind} {nm}', f'bench_{kind}_{nm}')
        RULE[0] = 'm3b'
        save_row(im, ko, knb, G, f'random | {kind} {nm}', f'bench_{kind}_{nm}')
        RULE[0] = 'm3'
    log(json.dumps(out))


def cmd_sheet(a):
    from PIL import Image, ImageDraw
    groups = {'mice': ['mice_'], 'ants': ['ants_'], 'frogs': ['frogs_'], 'benchmarks': ['bench_']}
    for dom, pre in groups.items():
        fs = sorted(f for f in (OUT / 'rows').glob('*.png') if any(f.name.startswith(p) for p in pre))
        if not fs:
            continue
        tile = 512
        sh = Image.new('RGB', (3 * tile, tile * len(fs)))
        for r, f in enumerate(fs):
            im = Image.open(f)
            w, hh = im.size
            for j in range(3):
                t = im.crop((j * w // 3, 0, (j + 1) * w // 3, hh)).resize((tile, tile), Image.BILINEAR)
                d = ImageDraw.Draw(t)
                txt = (OUT / 'rows' / f.with_suffix('.txt').name).read_text()
                d.rectangle([0, 0, tile, 16], fill=(0, 0, 0))
                d.text((4, 3), ['frame', 'old N2 (blue)', 'new mask3 (green)'][j] + ' | ' + txt[:70], fill=(255, 255, 0))
                sh.paste(t, (j * tile, r * tile))
        sh.save(OUT / f'{dom}.png')
        log(f'{dom}: {len(fs)} rows -> {OUT / f"{dom}.png"}')


def cmd_sweep(a):
    """post-hoc development sweep of (K_MAD, REACH_H) on frames outside the metric sample; frame | old | variants."""
    import torch
    from PIL import Image, ImageDraw
    from src.eci.v2 import mask3 as M3
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    dev = torch.device('cpu')
    var = [tuple(float(t) for t in v.split('/')) + ((35.0,) if v.count('/') == 1 else ()) for v in a.variants.split(',')]
    cases = [c.split('/') for c in a.cases.split(',')]
    pairs, info = {}, []
    (OUT / f'sweep{a.fill}').mkdir(parents=True, exist_ok=True)
    for dom, ser, obs, k in cases:
        if (dom, ser) not in pairs:
            pairs[(dom, ser)] = series_pair(dom, ser, dev, fill=a.fill or None)
            pairs[(dom, ser)][0].metric = set(sample_frames(pairs[(dom, ser)][0]))
        o, n = pairs[(dom, ser)]
        assert (obs, int(k)) not in o.metric, f'{obs} {k} is a metric frame'
        r = masks_for(o, n, o.V[obs], [int(k)])
        if not r:
            continue
        _, co, cn = r[0]
        P = n.pack(obs, env().window_of(dom, int(k)))
        G = o.G
        im = co['x'][0].permute(1, 2, 0).numpy()
        tiles = [(im, 'frame'), (tint(im, co['keep'][0].numpy(), G, (0, 90, 255)), f'old N2 {int(co["keep"][0].sum())}')]
        row = {'case': f'{dom}/{ser}/{obs}/{k}', 'old': int(co['keep'][0].sum())}
        for km, rh, tl in var:
            kk, _, _ = M3.keep3(cn['x'], P['bg'] if torch.is_tensor(P['bg']) else cn['bg_new'], P['mad'] if torch.is_tensor(P['mad']) else None,
                                cn['core0'], o.h, roi=cn['roi'], k_mad=km, reach_h=rh, t_lo=tl)
            row[f'km{km:g}_r{rh:g}_t{tl:g}'] = int(kk[0].sum())
            tiles.append((tint(im, kk[0].numpy(), G, (0, 220, 0)), f'K_MAD {km:g} REACH {rh:g} T_LO {tl:g}: {int(kk[0].sum())}'))
        info.append(row)
        log(json.dumps(row))
        T = 512
        sh = Image.new('RGB', (T * len(tiles), T))
        for j, (a_, t) in enumerate(tiles):
            tt = Image.fromarray(a_).resize((T, T), Image.BILINEAR)
            ImageDraw.Draw(tt).text((5, 5), t, fill=(255, 255, 0))
            sh.paste(tt, (j * T, 0))
        sh.save(OUT / f'sweep{a.fill}' / f'{obs}_{k}.png')
    (OUT / f'sweep{a.fill}' / 'sweep.json').write_text(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('dev', 'lab', 'ghost', 'bench', 'sheet', 'sweep'))
    ap.add_argument('--dom', default='mice')
    ap.add_argument('--ser', default='v1')
    ap.add_argument('--set', default='mab')
    ap.add_argument('--fill', default='')
    ap.add_argument('--rule', default='m3', choices=('m3', 'm3b'))
    ap.add_argument('--variants', default='6/6,3/6,3/10,0/10')
    ap.add_argument('--cases', default='')
    ap.add_argument('--fills', default='q98,telea')
    ap.add_argument('--bg', action='store_true')
    a = ap.parse_args()
    RULE[0] = a.rule
    {'dev': cmd_dev, 'lab': cmd_lab, 'ghost': cmd_ghost, 'bench': cmd_bench, 'sheet': cmd_sheet, 'sweep': cmd_sweep}[a.step](a)


if __name__ == '__main__':
    main()
