"""
Keypoint-guided mask on arm B (pre-registration scripts/eci/v2/mask5_kpg_prereg.md). CPU only (Slurm defaultp).
  score --set cal|mab|crim   -> OUT/kpg_<set>.json, kpg_<set>.md
  sheet --set ...            -> OUT/kpg_<bench>.png (frame | B | KG | zoom KG vs B)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
sys.path.insert(0, str(REPO / 'scripts/eci/v2_mask2'))
sys.path.insert(0, str(REPO / 'scripts/benchmark/crim13'))
import mask5_bench as MB  # noqa: E402

OUT = MB.OUT
BEE = MB.BEE
log = MB.log
BUILD = {'mab': [3, 6, 9], 'cal': [3, 4, 5, 6]}
HELD = {'mab': [0, 1, 2, 4, 5, 7, 8, 10, 11], 'cal': [0, 1, 2]}
SKEL = {'mab': [(3, 6), (6, 9)], 'cal': [(3, 'hips'), ('hips', 6), (4, 5)]}
CRIM_BODY = 131.0


def window_kps(kind):
    """-> fn(npz) -> (n_centre, 5) list of kps arrays on the canvas (NaN where missing)."""
    import common as CM
    from align_common import ROOT, geom, load_track
    if kind == 'cal':
        K = CM.CalKP('test')

        def f(z):
            nm = str(z['vid'])
            T = int(K.lens[K.names.index(nm)])
            w = np.clip(z['ks'][:, None] + 6 * np.arange(-2, 3)[None], 0, T - 1)
            return [[K.get(nm, int(i)) for i in row] for row in w]
        return f
    if kind == 'mab':
        seqs = CM.mab_keypoints('mouse_submission')

        def f(z):
            s = seqs[str(z['vid'])]
            w = np.clip(z['ks'][:, None] + 6 * np.arange(-2, 3)[None], 0, 1800) - 1
            out = []
            for row in w:
                r = []
                for i in row:
                    if i < 0 or i >= len(s['keypoints']):
                        r.append(np.full((3, 12, 2), np.nan, np.float32))
                        continue
                    k = CM.mab_kp(s, int(i), 512 / 224)
                    k[((k < 0) | (k > 512)).any(-1)] = np.nan
                    r.append(k)
                out.append(r)
            return out
        return f
    I = {v['name']: v for v in json.loads((REPO / 'results/vision/benchmark_crim13/inventory/inventory.json').read_text())['videos']}

    def f(z):
        v = I[str(z['vid'])]
        pos, nm = load_track(ROOT / v['dir'] / f'{v["name"]}_t-track.mat')
        W, H = int(v['header']['width']), int(v['header']['height'])
        pt, sc = geom(W, H)
        xy = np.stack([pos[..., 0] * sc, (pos[..., 1] + pt) * sc], -1)
        xy[nm < 2] = np.nan
        w = np.clip(z['ks'][:, None] + 5 * np.arange(-2, 3)[None], 0, len(nm) - 1)
        return [[xy[int(i)] for i in row] for row in w], 640 / W
    return f


def kg_mask(kind, B, kps, S, G, h, scale=1.0):
    """B (G*G,) bool, kps canvas -> (KG (G*G,) bool, fallback flag)."""
    import cv2
    cell = S // G
    gate = np.zeros((S, S), np.uint8)
    fill = np.zeros((S, S), np.uint8)
    if kind == 'crim':
        if not np.isfinite(np.asarray(kps)).all():
            return B, True
        for p in kps:
            c = (int(round(p[0])), int(round(p[1])))
            cv2.circle(gate, c, int(round(1.25 * CRIM_BODY * scale)), 1, -1)
            cv2.circle(fill, c, int(round(h)), 1, -1)
    else:
        for k in kps:
            b = k[BUILD[kind]]
            if not np.isfinite(b).all():
                return B, True
            neck, tb = k[3], k[6] if kind == 'cal' else k[9]
            L = float(np.linalg.norm(tb - neck))
            u = (tb - neck) / max(L, 1e-6)
            pts = np.vstack([b, tb + 1.0 * L * u]).round().astype(np.int32)
            hull = np.zeros((S, S), np.uint8)
            cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
            r = max(1, int(round(0.8 * L)))
            gate |= cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            P = {j: k[j] for j in BUILD[kind]}
            if kind == 'cal':
                P['hips'] = (k[4] + k[5]) / 2
            for a_, b_ in SKEL[kind]:
                cv2.line(fill, tuple(int(round(x)) for x in P[a_]), tuple(int(round(x)) for x in P[b_]), 1,
                         thickness=max(1, int(round(1.5 * h))))
    gt = gate.reshape(G, cell, G, cell).max((1, 3)).ravel() > 0
    ft = fill.reshape(G, cell, G, cell).mean((1, 3)).ravel() >= 0.25
    keep = (B & gt) | ft
    n, lab = cv2.connectedComponents(keep.reshape(G, G).astype(np.uint8), connectivity=8)
    good = np.unique(lab.ravel()[ft & (lab.ravel() > 0)])
    return np.isin(lab.ravel(), good[good > 0]), False


def held_metrics(kind, keep, kps, strain, S, G, h, reg):
    o = MB.frame_metrics(kind, keep, kps, strain, S, G, h, reg)
    if kind == 'crim':
        return {'held_kp': o['all_kp'], 'miss_held': o['miss/all'], **{k: v for k, v in o.items() if k.startswith(('kp/', 'miss/'))},
                'bg_tok': o['bg_tok'], 'iso_bg_comp': o['iso_bg_comp'], 'tok': o['tok']}
    r = {'held_kp': [], 'miss_held': [], 'bg_tok': o['bg_tok'], 'iso_bg_comp': o['iso_bg_comp'], 'tok': o['tok']}
    names = MB.PARTS[kind]
    for ai, k in enumerate(kps):
        hit = []
        for j in HELD[kind]:
            if np.isfinite(k[j]).all():
                v = float(keep[MB.tok_of(k[j], G, S)])
                hit.append(v)
                pre = ('res_' if ai == 0 else 'int_') if kind == 'cal' else ''
                r.setdefault(f'held/{pre}{names[j]}', []).append(v)
        r['held_kp'] += hit
        r['miss_held'].append(float(len(hit) > 0 and not any(hit)))
        if kind == 'cal':
            r.setdefault('miss_held/' + ('resident' if ai == 0 else 'intruder_white'), []).append(r['miss_held'][-1])
    return r


def agg(rows):
    acc = {}
    for r in rows:
        for k, v in r.items():
            acc.setdefault(k, []).extend(v)
    return {k: float(np.mean(v)) for k, v in acc.items()} | {'n_held_kp': len(acc['held_kp'])}


def cmd_score(a):
    import cv2
    import mask5_val as MV
    kind = a.set
    D = BEE / kind
    runs = [json.loads(f.read_text()) for f in sorted(D.glob('run_*.json'))]
    S, h = runs[0]['S'], runs[0]['h']
    G = S // 16
    pp = lambda p: MV.postproc(p.astype(np.float32), G, MB.CFG['t'], MB.CFG['t_core'], None, False)  # noqa: E731
    wk = window_kps(kind)
    rows = {k: [] for k in ('T', 'B', 'KG')}
    uni = {k: [] for k in ('T', 'B', 'KG')}
    nfb, nfr, per = 0, 0, []
    for f in sorted(D.glob('eval_*.npz')):
        z = np.load(f)
        W = wk(z)
        scale = 1.0
        if kind == 'crim':
            W, scale = W
        for i in range(len(z['ks'])):
            kps = z['kps'][i]
            reg = MB.region_tokens(kind, kps, S, G, h)
            wins = {'T': [pp(z['pT'][j]) for j in z['inv'][i]], 'B': [pp(z['pB'][j]) for j in z['inv'][i]]}
            kgw = []
            for t5 in range(5):
                m, fb = kg_mask(kind, wins['B'][t5], W[i][t5], S, G, h, scale)
                kgw.append(m)
                nfb += int(fb)
                nfr += 1
            wins['KG'] = kgw
            pf = {'file': f.name, 'i': i}
            for arm in rows:
                keep = wins[arm][2]
                rows[arm].append(held_metrics(kind, keep, kps, str(z['strain']), S, G, h, reg))
                uni[arm].append(np.any(wins[arm], 0).sum())
                pf[arm] = int(keep.sum())
            pf['removed'] = int((wins['B'][2] & ~wins['KG'][2]).sum())
            pf['added'] = int((wins['KG'][2] & ~wins['B'][2]).sum())
            per.append(pf)
    res = {'set': kind, 'bench': MB.NAME[kind], 'S': S, 'h': h, 'n_frames': len(per), 'circular': kind == 'crim',
           'arms': {arm: agg(R) for arm, R in rows.items()},
           'window_frames': nfr, 'window_frames_fallback_B': nfb}
    for arm in res['arms']:
        res['arms'][arm].pop('all_kp', None)
    sz = json.loads((REPO / 'results/vision/eci_v2_0B/bench_store/size.json').read_text())['bench'][MB.NAME[kind]]
    res['store'] = {arm: {'union5_tok': float(np.mean(u)), 'TB': float(np.mean(u)) * 2048 * sz['frames_5fps'] / 1e12} for arm, u in uni.items()}
    A = res['arms']
    if kind != 'crim':
        res['adopt_KG'] = bool(A['KG']['held_kp'] >= A['B']['held_kp'] - 0.005 and A['KG']['miss_held'] <= A['B']['miss_held']
                               and A['KG']['bg_tok'] < A['B']['bg_tok'])
    res['most_changed'] = sorted(per, key=lambda r: -(r['removed'] + r['added']))[:2]
    (OUT / f'kpg_{kind}.json').write_text(json.dumps(res, indent=1))
    keys = ['held_kp'] + sorted(k for k in A['B'] if k.startswith(('held/', 'kp/'))) + ['miss_held'] + \
        sorted(k for k in A['B'] if k.startswith(('miss_held/', 'miss/'))) + ['bg_tok', 'iso_bg_comp', 'tok']
    L = [f'## {MB.NAME[kind]} KG ({res["n_frames"]} held-out frames{"; CIRCULAR: built and scored on the same track point" if kind == "crim" else ""})',
         '', f'adopt KG (pre-registered): **{res.get("adopt_KG", "n/a (circular)")}**; window frames falling back to B: {nfb} / {nfr}', '',
         '| metric | T (mice probe) | B (bench probe) | KG (keypoint-guided B) |', '|---|---|---|---|']
    for k in keys:
        L.append(f'| {k} | ' + ' | '.join(f'{A[arm].get(k, float("nan")):.3f}' for arm in ('T', 'B', 'KG')) + ' |')
    L += ['', 'store (union t-2..t+2, 5 fps, fp16 1024-d): ' + '; '.join(f'{arm} {v["union5_tok"]:.1f} tok/frame = {v["TB"]:.3f} TB'
                                                                     for arm, v in res['store'].items()), '']
    (OUT / f'kpg_{kind}.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


def cmd_sheet(a):
    import cv2
    import mask5_val as MV
    from PIL import Image, ImageDraw
    kind = a.set
    D = BEE / kind
    res = json.loads((OUT / f'kpg_{kind}.json').read_text())
    S, h = res['S'], res['h']
    G = S // 16
    wk = window_kps(kind)
    files = sorted(D.glob('eval_*.npz'))
    g = np.random.default_rng(0)
    picks = [(files[j].name, int(g.integers(0, 8))) for j in g.choice(len(files), 6, replace=False)]
    picks += [(r['file'], r['i']) for r in res['most_changed']]
    T = 448
    sh = Image.new('RGB', (4 * T, T * len(picks)))
    for r, (fn, i) in enumerate(picks):
        z = np.load(D / fn)
        i = min(i, len(z['ks']) - 1)
        W = wk(z)
        scale = 1.0
        if kind == 'crim':
            W, scale = W
        kB = MV.postproc(z['pB'][z['inv'][i][2]].astype(np.float32), G, 0.9, 0.8, None, False)
        kK, _ = kg_mask(kind, kB, W[i][2], S, G, h, scale)
        im = z['img'][i].copy()
        kps = z['kps'][i]
        arr = np.asarray(kps)
        for ai, k in enumerate(arr[:, None] if kind == 'crim' else arr):
            for j, p in enumerate(k):
                if np.isfinite(p).all():
                    col = (255, 255, 0) if (kind == 'crim' or j in BUILD[kind]) else (255, 0, 255)
                    cv2.circle(im, (int(p[0]), int(p[1])), max(2, S // 256), col, -1)
        c = kps[0][11] if kind == 'mab' else (kps[1][0] if kind == 'cal' else kps[1])
        side = int(min(S, max(96, 12 * h)))
        x0, y0 = int(np.clip(c[0] - side / 2, 0, S - side)), int(np.clip(c[1] - side / 2, 0, S - side))
        both = MV.grid_tint(im, kB & kK, (255, 255, 255), 0.25)
        both = MV.grid_tint(both, kK & ~kB, (0, 220, 0), 0.5)
        both = MV.grid_tint(both, kB & ~kK, (255, 0, 0), 0.5)
        tiles = [(im, f'{"most changed" if r >= 6 else "random"} | {kind} {z["vid"]} k{z["ks"][i]} (yellow build, magenta held-out)'),
                 (MV.grid_tint(z['img'][i], kB, (0, 220, 0)), f'B bench probe {int(kB.sum())} tok'),
                 (MV.grid_tint(z['img'][i], kK, (0, 220, 0)), f'KG keypoint-guided {int(kK.sum())} tok'),
                 (both[y0:y0 + side, x0:x0 + side], 'zoom: green KG only, red B only, white both')]
        for j, (t, lab) in enumerate(tiles):
            tt = Image.fromarray(t).resize((T, T), Image.NEAREST if j == 3 else Image.BILINEAR)
            dr = ImageDraw.Draw(tt)
            dr.rectangle([0, 0, T, 15], fill=(0, 0, 0))
            dr.text((4, 2), lab[:74], fill=(255, 255, 0))
            sh.paste(tt, (j * T, r * T))
    p = OUT / f'kpg_{MB.NAME[kind]}.png'
    sh.save(p)
    log(f'{p}: {len(picks)} rows')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step')
    ap.add_argument('--set', default='cal', choices=('cal', 'mab', 'crim'))
    a = ap.parse_args()
    globals()['cmd_' + a.step](a)
