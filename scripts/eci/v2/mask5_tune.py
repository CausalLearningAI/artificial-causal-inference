"""
Per-benchmark tuned arm B (Bt) + keypoint-guided Bt (KGt) and the final pick (pre-registration
scripts/eci/v2/mask5_tune_prereg.md). CPU only (Slurm defaultp).
  run --set cal|mab|crim   -> OUT/tune_<set>.json, tune_<set>.md, tune_<bench>.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
import mask5_bench as MB  # noqa: E402
import mask5_kpg as KG  # noqa: E402

OUT, BEE, log = MB.OUT, MB.BEE, MB.log
TS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
TCS = (0.5, 0.8, 0.9, 0.95, 0.98)


def split(kind, files):
    g = np.random.default_rng(7)
    if kind == 'crim':
        st = {f: str(np.load(f)['strain']) for f in files}
        dev = []
        for s in ('BC', 'C57'):
            fs = sorted(f for f in files if st[f] == s)
            dev += [fs[i] for i in g.permutation(len(fs))[:len(fs) // 2]]
    else:
        n_dev = {'cal': 10, 'mab': 60}[kind]
        dev = [files[i] for i in g.permutation(len(files))[:n_dev]]
    dev = sorted(dev)
    return dev, sorted(f for f in files if f not in dev)


def kg_fast(kind, B, kps, S, G, h, scale=1.0):
    """mask5_kpg.kg_mask with the gate dilation as a Euclidean distance transform (prereg)."""
    import cv2
    cell = S // G
    gate = np.zeros((S, S), bool)
    fill = np.zeros((S, S), np.uint8)
    if kind == 'crim':
        if not np.isfinite(np.asarray(kps)).all():
            return B, True
        for p in kps:
            c = (int(round(p[0])), int(round(p[1])))
            g1 = np.zeros((S, S), np.uint8)
            cv2.circle(g1, c, int(round(1.25 * KG.CRIM_BODY * scale)), 1, -1)
            gate |= g1 > 0
            cv2.circle(fill, c, int(round(h)), 1, -1)
    else:
        for k in kps:
            b = k[KG.BUILD[kind]]
            if not np.isfinite(b).all():
                return B, True
            neck, tb = k[3], (k[6] if kind == 'cal' else k[9])
            L = float(np.linalg.norm(tb - neck))
            u = (tb - neck) / max(L, 1e-6)
            pts = np.vstack([b, tb + 1.0 * L * u]).round().astype(np.int32)
            hull = np.ones((S, S), np.uint8)
            cv2.fillConvexPoly(hull, cv2.convexHull(pts), 0)
            gate |= cv2.distanceTransform(hull, cv2.DIST_L2, 5) <= 0.8 * L
            P = {j: k[j] for j in KG.BUILD[kind]}
            if kind == 'cal':
                P['hips'] = (k[4] + k[5]) / 2
            for a_, b_ in KG.SKEL[kind]:
                cv2.line(fill, tuple(int(round(x)) for x in P[a_]), tuple(int(round(x)) for x in P[b_]), 1,
                         thickness=max(1, int(round(1.5 * h))))
    gt = gate.reshape(G, cell, G, cell).any((1, 3)).ravel()
    ft = fill.reshape(G, cell, G, cell).mean((1, 3)).ravel() >= 0.25
    keep = (B & gt) | ft
    n, lab = cv2.connectedComponents(keep.reshape(G, G).astype(np.uint8), connectivity=8)
    good = np.unique(lab.ravel()[ft & (lab.ravel() > 0)])
    return np.isin(lab.ravel(), good[good > 0]), False


def summarise(kind, rows):
    a = KG.agg(rows)
    full = {}
    for r in rows:
        for k in ('all_kp', 'miss_all'):
            full.setdefault(k, []).extend(r.get(k, []))
    a['all_kp'] = float(np.mean(full['all_kp']))
    a['miss_all'] = float(np.mean(full['miss_all']))
    prec = 1 - a['bg_tok'] / max(a['tok'], 1e-9)
    a['region_precision'] = prec
    a['kp_f1'] = 2 * a['all_kp'] * prec / max(a['all_kp'] + prec, 1e-9)
    return a


def metrics(kind, keep, kps, strain, S, G, h, reg):
    o = MB.frame_metrics(kind, keep, kps, strain, S, G, h, reg)
    r = KG.held_metrics(kind, keep, kps, strain, S, G, h, reg)
    r['all_kp'] = o['all_kp']
    r['miss_all'] = o['miss/all']
    return r


def cmd_run(a):
    import mask5_val as MV
    kind = a.set
    D = BEE / kind
    runs = [json.loads(f.read_text()) for f in sorted(D.glob('run_*.json'))]
    S, h = runs[0]['S'], runs[0]['h']
    G = S // 16
    files = sorted(D.glob('eval_*.npz'))
    dev, test = split(kind, files)
    log(f'{kind}: {len(dev)} dev / {len(test)} test videos')

    def pp(p, c):
        return MV.postproc(p.astype(np.float32), G, c['t'], c['t_core'], None, c['smooth'])
    # ---- dev grid
    frames = []
    for f in dev:
        z = np.load(f)
        for i in range(len(z['ks'])):
            kps = z['kps'][i]
            frames.append((z['pB'][z['inv'][i][2]], z['n2p'][i].astype(bool), kps, str(z['strain']),
                           MB.region_tokens(kind, kps, S, G, h)))
    n2dev = summarise(kind, [metrics(kind, n2, kps, st, S, G, h, reg) for _, n2, kps, st, reg in frames])
    grid = []
    for t in TS:
        for tc in TCS:
            for sm in (False, True):
                c = {'t': t, 't_core': tc, 'smooth': sm}
                m = summarise(kind, [metrics(kind, pp(p, c), kps, st, S, G, h, reg) for p, _, kps, st, reg in frames])
                grid.append((c, m))
    ok = [(c, m) for c, m in grid if m['miss_all'] <= n2dev['miss_all']]
    cfg, mdev = max(ok or grid, key=lambda cm: cm[1]['kp_f1'])
    log(f'{kind}: tuned {cfg} dev kp_f1 {mdev["kp_f1"]:.3f} (N2 dev miss {n2dev["miss_all"]:.3f}; {len(ok)} of {len(grid)} qualify)')
    # ---- test
    wk = KG.window_kps(kind)
    d0 = {'t': 0.9, 't_core': 0.8, 'smooth': False}
    rows = {k: [] for k in ('N2_pool', 'T', 'B', 'Bt', 'KG', 'KGt')}
    uni = {k: [] for k in ('Bt', 'KGt')}
    nfb = nfr = 0
    per = []
    for f in test:
        z = np.load(f)
        W = wk(z)
        scale = 1.0
        if kind == 'crim':
            W, scale = W
        for i in range(len(z['ks'])):
            kps = z['kps'][i]
            st = str(z['strain'])
            reg = MB.region_tokens(kind, kps, S, G, h)
            jj = z['inv'][i]
            bt = [pp(z['pB'][j], cfg) for j in jj]
            kgt = []
            for t5 in range(5):
                m, fb = kg_fast(kind, bt[t5], W[i][t5], S, G, h, scale)
                kgt.append(m)
                nfb += int(fb)
                nfr += 1
            b0 = pp(z['pB'][jj[2]], d0)
            ms = {'N2_pool': z['n2p'][i].astype(bool), 'T': pp(z['pT'][jj[2]], d0), 'B': b0, 'Bt': bt[2],
                  'KG': kg_fast(kind, b0, W[i][2], S, G, h, scale)[0], 'KGt': kgt[2]}
            for arm, keep in ms.items():
                rows[arm].append(metrics(kind, keep, kps, st, S, G, h, reg))
            uni['Bt'].append(np.any(bt, 0).sum())
            uni['KGt'].append(np.any(kgt, 0).sum())
            per.append({'file': f.name, 'i': i, 'changed': int((bt[2] ^ kgt[2]).sum())})
    A = {arm: summarise(kind, R) for arm, R in rows.items()}
    if kind == 'crim':
        pick = 'KGt'
    else:
        pick = 'KGt' if (A['KGt']['held_kp'] >= A['Bt']['held_kp'] - 0.005 and A['KGt']['miss_held'] <= A['Bt']['miss_held']
                         and A['KGt']['bg_tok'] < A['Bt']['bg_tok']) else 'Bt'
    sz = json.loads((REPO / 'results/vision/eci_v2_0B/bench_store/size.json').read_text())['bench'][MB.NAME[kind]]
    res = {'set': kind, 'bench': MB.NAME[kind], 'S': S, 'h': h, 'dev_files': [f.name for f in dev], 'test_files': [f.name for f in test],
           'n_test_frames': len(per), 'tuned_cfg': cfg, 'dev_tuned': mdev, 'dev_N2_pool': n2dev,
           'dev_grid': [{'cfg': c, 'kp_f1': m['kp_f1'], 'all_kp': m['all_kp'], 'miss_all': m['miss_all'], 'bg_tok': m['bg_tok']} for c, m in grid],
           'test': A, 'pick': pick, 'circular': kind == 'crim', 'window_frames': nfr, 'window_frames_fallback_Bt': nfb,
           'store': {arm: {'union5_tok': float(np.mean(u)), 'TB': float(np.mean(u)) * 2048 * sz['frames_5fps'] / 1e12} for arm, u in uni.items()},
           'most_changed': sorted(per, key=lambda r: -r['changed'])[:2]}
    (OUT / f'tune_{kind}.json').write_text(json.dumps(res, indent=1))
    keys = ['all_kp', 'held_kp'] + sorted(k for k in A['B'] if k.startswith(('held/', 'kp/'))) + ['miss_all', 'miss_held'] + \
        sorted(k for k in A['B'] if k.startswith(('miss_held/', 'miss/'))) + ['bg_tok', 'iso_bg_comp', 'tok', 'region_precision', 'kp_f1']
    arms = ('N2_pool', 'T', 'B', 'Bt', 'KG', 'KGt')
    L = [f'## {MB.NAME[kind]}: tuned B = {cfg}; FINAL PICK **{pick}**' + (' (CRIM13: pre-set, circular check only)' if kind == 'crim' else ''),
         '', f'test: {len(per)} frames from {len(test)} videos (dev {len(dev)} videos); KGt window frames falling back to Bt: {nfb} / {nfr}', '',
         '| metric | ' + ' | '.join(arms) + ' |', '|' + '---|' * (len(arms) + 1)]
    for k in keys:
        L.append(f'| {k} | ' + ' | '.join(f'{A[x].get(k, float("nan")):.3f}' for x in arms) + ' |')
    L += ['', 'store (union t-2..t+2, 5 fps, fp16 1024-d, ' + f'{sz["frames_5fps"]:,} frames): ' +
          '; '.join(f'{x} {v["union5_tok"]:.1f} tok/frame = {v["TB"]:.3f} TB' for x, v in res['store'].items()), '']
    (OUT / f'tune_{kind}.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)
    sheet(kind, res, test, cfg, S, G, h, pp, wk)


def sheet(kind, res, test, cfg, S, G, h, pp, wk):
    import cv2
    import mask5_val as MV
    from PIL import Image, ImageDraw
    g = np.random.default_rng(0)
    picks = [(test[j].name, int(g.integers(0, 8))) for j in g.choice(len(test), 6, replace=False)]
    picks += [(r['file'], r['i']) for r in res['most_changed']]
    T = 448
    sh = Image.new('RGB', (4 * T, T * len(picks)))
    for r, (fn, i) in enumerate(picks):
        z = np.load(BEE / kind / fn)
        i = min(i, len(z['ks']) - 1)
        W = wk(z)
        scale = 1.0
        if kind == 'crim':
            W, scale = W
        kb = pp(z['pB'][z['inv'][i][2]], cfg)
        kf = kg_fast(kind, kb, W[i][2], S, G, h, scale)[0] if res['pick'] == 'KGt' else kb
        im = z['img'][i].copy()
        arr = np.asarray(z['kps'][i])
        for k in (arr[:, None] if kind == 'crim' else arr):
            for j, p in enumerate(k):
                if np.isfinite(p).all():
                    col = (255, 255, 0) if (kind == 'crim' or j in KG.BUILD[kind]) else (255, 0, 255)
                    cv2.circle(im, (int(p[0]), int(p[1])), max(2, S // 256), col, -1)
        kps = z['kps'][i]
        c = kps[0][11] if kind == 'mab' else (kps[1][0] if kind == 'cal' else kps[1])
        side = int(min(S, max(96, 12 * h)))
        x0, y0 = int(np.clip(c[0] - side / 2, 0, S - side)), int(np.clip(c[1] - side / 2, 0, S - side))
        both = MV.grid_tint(im, kb & kf, (255, 255, 255), 0.25)
        both = MV.grid_tint(both, kf & ~kb, (0, 220, 0), 0.5)
        both = MV.grid_tint(both, kb & ~kf, (255, 0, 0), 0.5)
        tiles = [(im, f'{"most changed" if r >= 6 else "random"} test | {kind} {z["vid"]} k{z["ks"][i]} (yellow build, magenta held-out)'),
                 (MV.grid_tint(z['img'][i], kb, (0, 220, 0)), f'Bt t{cfg["t"]} tc{cfg["t_core"]} sm{int(cfg["smooth"])}: {int(kb.sum())} tok'),
                 (MV.grid_tint(z['img'][i], kf, (0, 220, 0)), f'final {res["pick"]}: {int(kf.sum())} tok'),
                 (both[y0:y0 + side, x0:x0 + side], 'zoom: green final only, red Bt only, white both')]
        for j, (t, lab) in enumerate(tiles):
            tt = Image.fromarray(t).resize((T, T), Image.NEAREST if j == 3 else Image.BILINEAR)
            dr = ImageDraw.Draw(tt)
            dr.rectangle([0, 0, T, 15], fill=(0, 0, 0))
            dr.text((4, 2), lab[:74], fill=(255, 255, 0))
            sh.paste(tt, (j * T, r * T))
    p = OUT / f'tune_{MB.NAME[kind]}.png'
    sh.save(p)
    log(f'{p}: {len(picks)} rows')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step')
    ap.add_argument('--set', default='cal', choices=('cal', 'mab', 'crim'))
    a = ap.parse_args()
    globals()['cmd_' + a.step](a)
