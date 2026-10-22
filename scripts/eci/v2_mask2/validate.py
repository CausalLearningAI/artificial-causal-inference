"""
ECI v2 THIN-PART MASK TEST, STEP 1: pixel-cue validation on HELD-OUT public benchmark data with keypoints.
The pre-registration of the whole test (arms, metrics, decision rule) is in scripts/eci/v2_mask2/mask2_test.py; this
file implements its step-1 metrics. Keypoints are used ONLY to score masks, never to build them.

Held-out sets (never looked at during development; development used CalMS21 TRAIN and MABe22 userTrain, explore.py):
    calms21   all 19 task-1 TEST videos, 24 frames per video (rng 0, uniformly over the video), 1024 x 1024 canvas
              (native 1024 x 570, 227 px pad rows), resident BLACK + intruder WHITE (pale) mouse, 7 keypoints each.
    mabe22    120 mouse_submission clips (rng 0), 8 frames per clip from the 5 fps label grid, frames where every
              mouse has all 12 keypoints; frames bicubic-upsampled x4 to 896 px (so a 16 px token is 4 native px and
              the body / token ratio is near the lab mice), keypoints scaled likewise; 3 black mice, tail_base,
              tail_mid, tail_tip.
Per video background: per-pixel median of 48 frames evenly spread (new cue), per-pixel 0.98 grey quantile of the same
48 frames (frozen dark cue). h (body half-width) = the calib rule of the previous test (median over frozen-cue
components >= 30 px of the max distance transform) on the background frames of the set.
Arms: M0 = frozen dark cue + r25 rule; N1 = new cue + r25; N2 = new cue + r25t (src/eci/v2/mask2.py).
Metrics (frame-averaged over all scored frames of the set):
    tail_seg_tok    MABe22: share of 10 tail sample points (tail_mid, tail_tip and 8 points evenly spaced on the
                    polyline tail_base -> tail_mid -> tail_tip, base excluded) whose token is kept.
    tail_seg_px     same, pixel level: an animal pixel within max(2, h / 4) px.
    tail_base_tok   share of tail_base keypoints in kept tokens (both sets; CalMS21 resident / intruder separately).
    body_tok        share of non-tail keypoints in kept tokens (MABe22 all mice; CalMS21 resident).
    pale_tok        CalMS21 intruder (white mouse): share of its 7 keypoints in kept tokens; pale_missed = share of
                    frames where no intruder keypoint is in a kept token.
    leak            share of kept tokens that contain NO pixel of the keypoint animal region: capsules of radius 1.5 h
                    along nose-neck-(center)-tail_base and to the ears / paws / hips, tail capsules of radius 0.75 h
                    along the tail polyline (MABe22), and for CalMS21 (no tail keypoints) a disk around tail_base of
                    radius = that mouse's nose - tail_base distance (tail allowance).
    tokens / frame.
    crim13    (added 2026-10-22 on the user's request, before any held-out score existed) CRIM13 released TEST videos:
              12 with a BC (pale, albino) intruder + 12 with a C57 (black) intruder (rng 0), 16 frames each (5 fps
              grid, both mice tracked), 640 canvas (align_common.canvas_gpu, the frozen CRIM13 geometry). CRIM13 has
              only one track point per mouse (no keypoints): pale_tok = BC intruder track point in a kept token
              (also with a 1-token tolerance: any of the 3 x 3 tokens around it), res_tok = resident track point,
              c57_tok = C57 intruder, leak = kept tokens with no pixel within 6 h of a track point (body + tail reach).
-> OUT/validate/<set>.json, OUT/validate/sheet_<set>.jpg (12 frames, M0 vs N2 tokens, grown pixels, keypoints)
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (N_BG, OUT, CalKP, cal_decode, dark_cue, log, mab_decode, mab_keypoints, mab_kp, q98)  # noqa
from src.eci.v2 import mask2 as M2  # noqa: E402

ARMS = ('M0', 'N1', 'N2')
CAL_SKEL = [(0, 3), (3, 6), (0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]
MAB_SKEL = [(0, 3), (3, 6), (6, 9), (0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (6, 7), (6, 8), (7, 9), (8, 9)]
MAB_TAIL = [(9, 10), (10, 11)]


def capsule(S, a, b, r):
    """(S, S) bool: pixels within r of segment a-b ((x, y) px)."""
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) + 0.5
    lo = np.floor(np.minimum(a, b) - r - 1).astype(int).clip(0, S)
    hi = np.ceil(np.maximum(a, b) + r + 1).astype(int).clip(0, S)
    m = np.zeros((S, S), bool)
    X, Y = xx[lo[1]:hi[1], lo[0]:hi[0]], yy[lo[1]:hi[1], lo[0]:hi[0]]
    ab = b - a
    t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / max(float(ab @ ab), 1e-6), 0, 1)
    m[lo[1]:hi[1], lo[0]:hi[0]] = (X - a[0] - t * ab[0]) ** 2 + (Y - a[1] - t * ab[1]) ** 2 <= r * r
    return m


def region(S, kps, skel, tail, h, tail_disk):
    m = np.zeros((S, S), bool)
    for k in kps:
        for i, j in skel:
            if np.isfinite(k[i]).all() and np.isfinite(k[j]).all():
                m |= capsule(S, k[i], k[j], 1.5 * h)
        for i, j in tail:
            if np.isfinite(k[i]).all() and np.isfinite(k[j]).all():
                m |= capsule(S, k[i], k[j], 0.75 * h)
        if tail_disk and np.isfinite(k[0]).all() and np.isfinite(k[6]).all():
            m |= capsule(S, k[6], k[6], float(np.linalg.norm(k[0] - k[6])))
    return m


def tail_points(k):
    """MABe22 (12, 2) -> (10, 2): tail_mid, tail_tip + 8 points on base -> mid -> tip (base excluded)."""
    b, m_, t = k[9], k[10], k[11]
    L1, L2 = np.linalg.norm(m_ - b), np.linalg.norm(t - m_)
    s = np.linspace(0, L1 + L2, 10)[1:-1][:8] if L1 + L2 > 0 else np.zeros(8)
    pts = [b + (m_ - b) * (u / max(L1, 1e-6)) if u <= L1 else m_ + (t - m_) * ((u - L1) / max(L2, 1e-6)) for u in s]
    return np.stack([m_, t] + pts)


def tok_of(p, G, S):
    p = np.clip(p, 0, S - 1e-3)
    return (p[..., 1] // (S / G)).astype(int) * G + (p[..., 0] // (S / G)).astype(int)


def px_near(pix, p, r):
    """pix (S, S) bool, points (n, 2) -> (n,) bool any animal pixel within r."""
    S = pix.shape[0]
    out = []
    ri = int(np.ceil(r))
    for x, y in p:
        xi, yi = int(x), int(y)
        y0, y1, x0, x1 = max(0, yi - ri), min(S, yi + ri + 1), max(0, xi - ri), min(S, xi + ri + 1)
        if y0 >= y1 or x0 >= x1:
            out.append(False)
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        out.append(bool((pix[y0:y1, x0:x1] & ((yy + 0.5 - y) ** 2 + (xx + 0.5 - x) ** 2 <= r * r)).any()))
    return np.array(out)


def body_h(cores):
    from scipy import ndimage
    hm = []
    for px in cores:
        lab, n = ndimage.label(px)
        if n == 0:
            continue
        edt = ndimage.distance_transform_edt(px)
        area = np.bincount(lab.ravel(), minlength=n + 1)
        mx = ndimage.maximum(edt, lab, np.arange(1, n + 1))
        big = area[1:].max()
        hm += [float(m) for m, a in zip(mx, area[1:]) if a >= max(30, 0.25 * big)]  # animal-sized components only
    return float(np.median(hm))


# ------------------------------------------------------------------------------------------------------- data
def cal_videos(split, n_fr, seed):
    K = CalKP(split)
    g = np.random.default_rng(seed)
    out = []
    for j, nm in enumerate(K.names):
        T = int(K.lens[j])
        ks = np.sort(g.choice(T, n_fr, replace=False))
        out.append((nm, T, ks, [K.get(nm, int(k)) for k in ks]))
    return out


def load_cal(item, split):
    nm, T, ks, kps = item
    xb = cal_decode(nm, split, np.linspace(0, T - 1, N_BG).round().astype(int), threads=2)
    x = cal_decode(nm, split, ks, threads=2)
    return nm, xb, x, ks, kps


def mab_clips(n_clip, n_fr, seed):
    seqs = mab_keypoints('mouse_submission')
    g = np.random.default_rng(seed)
    out = []
    excl = {'QO9ONQTPOYAI26MYHPHY', 'R6R4SNHGFM0ZOOH9A1TN'}
    clips = [c for c in sorted(seqs) if c not in excl]
    for c in g.choice(clips, len(clips), replace=False):
        ok = [i for i in range(30, 1800, 6) if np.isfinite(mab_kp(seqs[c], i)).all()]
        if len(ok) < n_fr:
            continue
        ks = np.sort(g.choice(ok, n_fr, replace=False))
        out.append((c, ks, [mab_kp(seqs[c], int(k), 4.0) for k in ks]))
        if len(out) == n_clip:
            break
    return out


def crim_videos(n_each, n_fr, seed):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmark/crim13'))
    from align_common import ROOT, load_seek, load_track, meta_of
    I = json.loads((Path(__file__).resolve().parents[3] / 'results/vision/benchmark_crim13/inventory/inventory.json')
                   .read_text())['videos']
    g = np.random.default_rng(seed)
    out = []
    for strain in ('BC', 'C57'):
        vs = [v for v in I if v['split'] == 'test' and meta_of(v['name'])[2] == strain]
        for v in g.choice(vs, n_each, replace=False):
            d = ROOT / v['dir']
            pos, nm = load_track(d / f'{v["name"]}_t-track.mat')
            ok = np.flatnonzero((nm == 2) & (np.arange(len(nm)) % 5 == 0) & (np.arange(len(nm)) >= 50) &
                                np.isfinite(pos).all((1, 2)))
            ks = np.sort(g.choice(ok, n_fr, replace=False))
            out.append((v['name'], strain, d, int(v['frames']), int(v['header']['width']), int(v['header']['height']),
                        ks, pos[ks]))
    return out


def load_crim(item, dev):
    from align_common import canvas_gpu, load_seek, read_frames, geom, CANVAS
    nm, strain, d, N, W, H, ks, pos = item
    offs = load_seek(d / f'{nm}_t-seek.mat')
    bgi = np.linspace(0, N - 1, N_BG).round().astype(int)
    fr, _ = read_frames(d / f'{nm}_t.seq', offs, set(bgi.tolist()) | set(ks.tolist()), threads=2)
    pt, sc = geom(W, H)
    xy = np.stack([pos[..., 0] * sc, (pos[..., 1] + pt) * sc], -1)  # (n, 2 mice, 2) canvas px
    xb = canvas_gpu([fr[i] for i in bgi], W, H, dev).to(torch.uint8)
    x = canvas_gpu([fr[i] for i in ks], W, H, dev).to(torch.uint8)
    return nm, strain, xb, x, ks, xy


def up4(a, dev):
    t = torch.from_numpy(a).to(dev).float()
    return F.interpolate(t, scale_factor=4, mode='bicubic', align_corners=False).round().clamp(0, 255).to(torch.uint8)


# ------------------------------------------------------------------------------------------------------- scoring
def masks_all(x, pack, h):
    core0 = dark_cue(x, pack['q98'])
    r = M2.cue(x, pack, core0, h)
    fr0 = F.avg_pool2d(core0.float()[:, None], 16).flatten(1)
    fr, ng = M2.token_fractions(r['pix'], r['grown'])
    return {'M0': (fr0 >= 0.25, core0), 'N1': (M2.keep(fr, ng, 'r25'), r['pix']),
            'N2': (M2.keep(fr, ng, 'r25t'), r['pix'])}, r


def score_frame(kept, pix, kps, S, G, h, kind):
    """-> metrics dict for one frame and arm."""
    o = {}
    kept = kept.cpu().numpy()
    pix = pix.cpu().numpy()
    if kind == 'mab':
        skel, tail, td = MAB_SKEL, MAB_TAIL, False
        tp = np.concatenate([tail_points(k) for k in kps])
        o['tail_seg_tok'] = float(kept[tok_of(tp, G, S)].mean())
        o['tail_seg_px'] = float(px_near(pix, tp, max(2.0, h / 4)).mean())
        o['tail_tip_tok'] = float(kept[tok_of(np.stack([k[11] for k in kps]), G, S)].mean())
        o['tail_base_tok'] = float(kept[tok_of(np.stack([k[9] for k in kps]), G, S)].mean())
        body = np.concatenate([k[:9] for k in kps])
        o['body_tok'] = float(kept[tok_of(body, G, S)].mean())
    elif kind == 'crim':
        xy, strain = kps
        cell = tok_of(xy, G, S)
        o['res_tok'] = float(kept[cell[0]])
        nb = lambda c: bool(kept.reshape(G, G)[max(0, c // G - 1):c // G + 2, max(0, c % G - 1):c % G + 2].any())  # noqa
        o['res_tok_tol1'] = float(nb(cell[0]))
        key = 'pale' if strain == 'BC' else 'c57'
        o[f'{key}_tok'] = float(kept[cell[1]])
        o[f'{key}_tok_tol1'] = float(nb(cell[1]))
        reg = np.zeros((S, S), bool)
        for p in xy:
            reg |= capsule(S, p, p, 6 * h)
        rt = F.avg_pool2d(torch.from_numpy(reg).float()[None, None], 16).flatten().numpy() > 0
        o['tokens'] = float(kept.sum())
        o['leak_n'] = float((kept & ~rt).sum())
        return o
    else:
        skel, tail, td = CAL_SKEL, [], True
        res, intr = kps
        o['body_tok'] = float(kept[tok_of(res[:6], G, S)].mean())
        o['tail_base_tok_res'] = float(kept[tok_of(res[6:7], G, S)].mean())
        h_int = kept[tok_of(intr, G, S)]
        o['pale_tok'] = float(h_int.mean())
        o['pale_missed'] = float(not h_int.any())
        o['tail_base_tok_int'] = float(h_int[6])
    reg = region(S, kps, skel, tail, h, td)
    rt = F.avg_pool2d(torch.from_numpy(reg).float()[None, None], 16).flatten().numpy() > 0
    o['tokens'] = float(kept.sum())
    o['leak_n'] = float((kept & ~rt).sum())
    return o


def run(kind, args, dev):
    if kind == 'cal':
        items = cal_videos('test', args.n_fr or 24, 0)
        with ThreadPoolExecutor(6) as ex:
            loaded = list(ex.map(lambda it: load_cal(it, 'test'), items))
        S, G = 1024, 64
        vids = [(nm, torch.from_numpy(xb).to(dev), torch.from_numpy(x).to(dev), ks, kps)
                for nm, xb, x, ks, kps in loaded]
    elif kind == 'crim':
        items = crim_videos(12, args.n_fr or 16, 0)
        with ThreadPoolExecutor(6) as ex:
            vids0 = list(ex.map(lambda it: load_crim(it, dev), items))
        vids = [(nm, xb, x, ks, [(xy[i], strain) for i in range(len(ks))]) for nm, strain, xb, x, ks, xy in vids0]
        S, G = 640, 40
    else:
        items = mab_clips(args.n_clip or 120, args.n_fr or 8, 0)
        vids = []
        for c, ks, kps in items:
            a = mab_decode('mouse_submission', c)
            ib = np.linspace(0, len(a) - 1, N_BG).round().astype(int)
            vids.append((c, up4(a[ib], dev), up4(a[ks + 1], dev), ks, kps))
        S, G = 896, 56
    log(f'{kind}: {len(vids)} videos loaded')
    packs = [M2.background_pack(xb) for _, xb, _, _, _ in vids]
    h = body_h([c.cpu().numpy() for (_, xb, _, _, _), pk in zip(vids, packs) for c in dark_cue(xb[::6], pk['q98'])])
    log(f'{kind}: h {h:.2f}')
    rows = {a: [] for a in ARMS}
    sheet = []
    for (nm, xb, x, ks, kps), pk in zip(vids, packs):
        for b0 in range(0, len(x), 8):
            ms, r = masks_all(x[b0:b0 + 8], pk, h)
            for i in range(len(ks[b0:b0 + 8])):
                for a in ARMS:
                    kept, pix = ms[a][0][i], ms[a][1][i]
                    rows[a].append({'video': nm, 'k': int(ks[b0 + i]),
                                    **score_frame(kept, pix, kps[b0 + i], S, G, h, kind)})
                if len(sheet) < 12 and (b0 + i) == 0:
                    sheet.append((x[b0 + i].cpu().numpy(), ms['M0'][0][i].cpu().numpy(), ms['N2'][0][i].cpu().numpy(),
                                  r['grown'][i].cpu().numpy(), kps[b0 + i], f'{nm} k{ks[b0 + i]}'))
    res = {'set': kind, 'S': S, 'G': G, 'h': h, 'n_videos': len(vids), 'n_frames': len(rows['M0'])}
    for a in ARMS:
        keys = [k for k in rows[a][0] if k not in ('video', 'k', 'tokens', 'leak_n')]
        keys = sorted({k for r in rows[a] for k in r if k not in ('video', 'k', 'tokens', 'leak_n')})
        res[a] = {k: float(np.mean([r[k] for r in rows[a] if k in r])) for k in keys}
        res[a]['tokens_per_frame'] = float(np.mean([r['tokens'] for r in rows[a]]))
        res[a]['leak'] = float(sum(r['leak_n'] for r in rows[a]) / max(sum(r['tokens'] for r in rows[a]), 1))
        log(f'{kind} {a}: ' + json.dumps(res[a]))
    (OUT / 'validate').mkdir(parents=True, exist_ok=True)
    (OUT / 'validate' / f'{kind}{args.tag}.json').write_text(json.dumps(res, indent=1))
    make_sheet(sheet, G, OUT / 'validate' / f'sheet_{kind}{args.tag}.jpg')


def make_sheet(items, G, path, tile=640):
    from PIL import Image, ImageDraw
    sh = Image.new('RGB', (2 * tile, tile * len(items)))
    for r, (x, k0, k2, grown, kps, title) in enumerate(items):
        S = x.shape[-1]
        base = x.transpose(1, 2, 0).astype(np.float32)
        # zoom on the keypoints' bounding box
        P = (np.asarray(kps[0]) if isinstance(kps, tuple) else np.concatenate([k for k in kps])).reshape(-1, 2)
        P = P[np.isfinite(P).all(1)]
        c = P.mean(0)
        w = int(min(S, max(256, (P.max(0) - P.min(0)).max() * 1.3)))
        x0, y0 = int(np.clip(c[0] - w / 2, 0, S - w)), int(np.clip(c[1] - w / 2, 0, S - w))
        for j, (kept, nm) in enumerate(((k0, 'M0 frozen'), (k2, 'N2 new cue r25t'))):
            im = base.copy()
            K = np.kron(kept.reshape(G, G), np.ones((S // G, S // G), bool))
            im[K] = 0.65 * im[K] + 0.35 * np.array([0, 255, 0])
            if j == 1:
                im[grown] = 0.4 * im[grown] + 0.6 * np.array([255, 0, 255])
            t = Image.fromarray(im[y0:y0 + w, x0:x0 + w].clip(0, 255).astype(np.uint8)).resize((tile, tile),
                                                                                          Image.BILINEAR)
            d = ImageDraw.Draw(t)
            for p in P:
                q = (p - [x0, y0]) * tile / w
                d.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], outline=(255, 0, 0))
            d.text((5, 5), f'{nm} {int(kept.sum())} tok | {title}', fill=(255, 255, 0))
            sh.paste(t, (j * tile, r * tile))
    sh.save(path, quality=82)


def step1(tag=''):
    """Pre-registered step-1 rule (mask2_test.py docstring) -> OUT/validate/step1.json + table.md."""
    V = {k: json.loads((OUT / 'validate' / f'{k}{tag}.json').read_text()) for k in ('mab', 'cal', 'crim')}
    m0 = {k: V[k]['M0'] for k in V}
    res = {'pass': {}, 'criteria': {}}
    for a in ('N1', 'N2'):
        c = {'1a_tail_seg_gain': V['mab'][a]['tail_seg_tok'] - m0['mab']['tail_seg_tok'],
             '1b_pale_gain_calms21': V['cal'][a]['pale_tok'] - m0['cal']['pale_tok'],
             '1b_pale_gain_crim13': V['crim'][a]['pale_tok'] - m0['crim']['pale_tok'],
             '1c_body_delta_mabe22': V['mab'][a]['body_tok'] - m0['mab']['body_tok'],
             '1c_body_delta_calms21': V['cal'][a]['body_tok'] - m0['cal']['body_tok'],
             '1d_leak': {k: V[k][a]['leak'] for k in V}}
        ok = (c['1a_tail_seg_gain'] >= 0.20 and c['1b_pale_gain_calms21'] >= 0.10 and c['1b_pale_gain_crim13'] >= 0.10
              and min(c['1c_body_delta_mabe22'], c['1c_body_delta_calms21']) >= -0.01 and max(c['1d_leak'].values()) <= 0.05)
        res['criteria'][a], res['pass'][a] = c, bool(ok)
    ps = [a for a in ('N1', 'N2') if res['pass'][a]]
    if not ps:
        res['carried_arm'] = None
    elif len(ps) == 2:
        d = V['mab']['N2']['tail_seg_tok'] - V['mab']['N1']['tail_seg_tok']
        res['carried_arm'] = 'N2' if d > 0.02 else 'N1'
    else:
        res['carried_arm'] = ps[0]
    res['pass']['M0'] = False
    (OUT / 'validate' / f'step1{tag}.json').write_text(json.dumps(res, indent=1))
    keys = sorted({k for v in V.values() for a in ARMS for k in v[a]})
    L = ['| set | metric | M0 | N1 | N2 |', '|---|---|---|---|---|']
    for k in V:
        for m in keys:
            if m in V[k]['M0']:
                L.append(f'| {k} | {m} | ' + ' | '.join(f'{V[k][a][m]:.3f}' for a in ARMS) + ' |')
    L += ['', f"step-1 pass: {res['pass']}; carried arm: {res['carried_arm']}", '', '```',
          json.dumps(res['criteria'], indent=1), '```']
    (OUT / 'validate' / f'table{tag}.md').write_text('\n'.join(L))
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='mab')
    ap.add_argument('--n-clip', type=int, default=0)
    ap.add_argument('--n-fr', type=int, default=0)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    if args.set == 'step1':
        return step1(args.tag)
    run(args.set, args, torch.device('cuda'))


if __name__ == '__main__':
    main()
