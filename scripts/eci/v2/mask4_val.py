"""
ECI animal mask v4 bake-off (src/eci/v2/mask4.py: SAM 2.1 prompted by the N2 cue) vs the frozen v2.0 N2 and mask3b.
Pre-registration of metrics and decision rule: scripts/eci/v2/mask4_prereg.md. No DINOv3 token is encoded.

Steps (wrapper scripts/eci/v2/mask4.sh):
  capture --dom D --ser S   (GPU) decode gold / sheet frames + a 10-frame clip ending at each, run N2 and mask3b via the
                            mask3_val capture path (identity-checked against v20.N2Series.n2) -> BEE/cap/*.npz
  arms --dom D --ser S      (GPU) SAM arms on the captured frames, timing -> BEE/arms/*.npz, OUT/time_<D>_<S>.json
  goldview                  (CPU) annotation views (frame + 16 px grid + N2 components) -> OUT/gold/view/
  gold                      (GPU) gold masks from the manual prompts OUT/gold/prompts.json -> OUT/gold/*.npz + PNGs
  metrics                   (CPU) gold metrics table -> OUT/metrics.json, OUT/table.md
  bench --set mab|cal|crim  (GPU) keypoint metrics on the scripts/eci/v2_mask2/validate.py frames -> OUT/bench_<set>.json
  sheet                     (CPU) approval sheets -> OUT/{mice,ants,frogs,benchmarks}.png
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
OUT = REPO / 'results/vision/eci_v2_0B/mask_approval_v2'
OLD = REPO / 'results/vision/eci_v2_0B/mask_approval'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_mask4')
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731
CLIP = 10
N_GOLD = {('mice', 'v1'): 10, ('mice', 'v2'): 10, ('ants', 'v2'): 7, ('ants', 'v3'): 7, ('ants', 'v4'): 6,
          ('frogs', 'juv'): 10, ('frogs', 'tad'): 10}
SERIES = list(N_GOLD)


def old_rows(dom, ser):
    """the existing approval-sheet rows of this series: [(role, obs, k)] (role 'rand' / 'hard')."""
    out = []
    for f in sorted((OLD / 'rows').glob(f'{dom}_{ser}_*.txt')):
        if f.stem.endswith('_m3b'):
            continue
        why, rest = f.read_text().split(' | ')[:2]
        d, s, obs, k = rest.split()
        out.append(('rand' if why == 'random' else 'hard:' + why, obs, int(k[1:])))
    return out


def frame_list(o, dom, ser):
    """gold frames = the old sheet's random frames + further metric frames (mask3_val.sample_frames, rng 0) drawn with
    rng 2 up to N_GOLD; plus the old sheet's hard cases (not gold)."""
    import mask3_val as MV
    rows = old_rows(dom, ser)
    gold = [(obs, k) for r, obs, k in rows if r == 'rand']
    pool = [f for f in MV.sample_frames(o) if f not in gold]
    g = np.random.default_rng(2)
    for i in g.choice(len(pool), N_GOLD[(dom, ser)] - len(gold), replace=False):
        gold.append(pool[i])
    out = [('gold', obs, k) for obs, k in gold]
    out += [(r, obs, k) for r, obs, k in rows if r != 'rand']
    return out


def cap_path(dom, ser, obs, k):
    return BEE / 'cap' / f'{dom}_{ser}_{obs}_{k}.npz'


def cmd_capture(a):
    import torch
    import mask3_val as MV
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    o, n = MV.series_pair(a.dom, a.ser, dev, rule='m3b')
    fl = frame_list(o, a.dom, a.ser)
    (BEE / 'cap').mkdir(parents=True, exist_ok=True)
    meta = []
    for role, obs, k in fl:
        v = o.V[obs]
        k0 = max(0, k - CLIP + 1)
        ks = list(range(k0, k + 1))
        try:
            fr = o.run_job(**o.job(v, k0, len(ks), threads=4))
        except RuntimeError as e:
            log(f'DECODE FAIL {obs} k {k}: {str(e)[:200]}')
            continue
        o.cap.clear()
        n.cap.clear()
        o.frames(v, fr, ks)
        n.frames(v, fr, ks)
        cat = lambda caps, key: torch.cat([c[key] for c in caps]).cpu().numpy()  # noqa: E731
        x = cat(o.cap, 'x')
        assert x.shape[0] == len(ks), (x.shape, len(ks))
        roi = o.cap[-1]['roi']
        rec = {'x': x.transpose(0, 2, 3, 1).astype(np.uint8), 'core0': np.packbits(cat(o.cap, 'core0'), -1),
               'n2_pix': np.packbits(cat(o.cap, 'pix'), -1), 'n2_app': np.packbits(cat(o.cap, 'app'), -1),
               'n2_keep': cat(o.cap, 'keep'), 'm3b_pix': np.packbits(cat(n.cap, 'pix')[-1:], -1),
               'm3b_keep': cat(n.cap, 'keep')[-1:],
               'roi': np.packbits(np.ones(x.shape[2:], bool) if roi is None else roi[-1].cpu().numpy().astype(bool), -1)}
        np.savez(cap_path(a.dom, a.ser, obs, k), **rec)
        meta.append({'role': role, 'obs': obs, 'k': k, 'k0': k0, 'S': int(x.shape[2]), 'G': int(o.G), 'h': float(o.h),
                     'n2_tok': int(rec['n2_keep'][-1].sum()), 'm3b_tok': int(rec['m3b_keep'][-1].sum())})
        log(json.dumps(meta[-1]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cap').mkdir(exist_ok=True)
    (OUT / 'cap' / f'{a.dom}_{a.ser}.json').write_text(json.dumps(meta, indent=1))


def load_cap(dom, ser, obs, k):
    z = np.load(cap_path(dom, ser, obs, k))
    S = z['x'].shape[1]
    up = lambda b: np.unpackbits(b, -1)[..., :S].astype(bool)  # noqa: E731
    return {'x': z['x'], 'core0': up(z['core0']), 'n2_pix': up(z['n2_pix']), 'n2_app': up(z['n2_app']),
            'n2_keep': z['n2_keep'], 'm3b_pix': up(z['m3b_pix'])[0], 'm3b_keep': z['m3b_keep'][0], 'roi': up(z['roi'])}


ARMS = ['A_s_full', 'A_l_full', 'A_s_crop', 'A_l_crop', 'A_s_video']


def cmd_arms(a):
    import torch
    from src.eci.v2 import mask4 as M4
    dev = torch.device('cuda')
    meta = json.loads((OUT / 'cap' / f'{a.dom}_{a.ser}.json').read_text())
    arms = a.arms.split(',') if a.arms else ARMS
    models = {}
    tm = {arm: [] for arm in arms}
    (BEE / 'arms').mkdir(parents=True, exist_ok=True)
    for m in meta:
        c = load_cap(a.dom, a.ser, m['obs'], m['k'])
        h = m['h']
        pix = c['n2_pix'][-1]
        objs = M4.objects(pix, core_of(c, -1), h)
        rec = {}
        info = {'n_obj': len(objs)}
        for arm in arms:
            _, size, mode = arm.split('_')
            size = {'s': 'small', 'l': 'large'}[size]
            key = (size, mode == 'video')
            if key not in models:
                models[key] = M4.load_sam(size, dev, video=mode == 'video')
            mod, proc = models[key]
            torch.cuda.synchronize()
            t0 = time.time()
            if mode == 'full':
                ch = M4.sam_full(mod, proc, c['x'][-1], objs, dev)
                ob = objs
            elif mode == 'crop':
                ch = M4.sam_crop(mod, proc, c['x'][-1], objs, dev, h)
                ob = objs
            else:
                o0 = M4.objects(c['n2_pix'][0], core_of(c, 0), h)
                ch = M4.sam_video(mod, proc, c['x'], o0, dev)
                ob = o0
            torch.cuda.synchronize()
            tm[arm].append(time.time() - t0)
            core_last = core_of(c, -1)
            for rule in ('A', 'B'):
                u, nfb = M4.assemble(ch, ob, pix, core_last, h, rule=rule, app=c['n2_app'][-1])
                nm = arm if rule == 'A' else 'B' + arm[1:]
                rec[nm + '_pix'] = np.packbits(u, -1)
                rec[nm + '_keep'] = M4.tokens(u, h)
                info[nm + '_fallback'] = nfb
                info[nm + '_tok'] = int(rec[nm + '_keep'].sum())
        np.savez(BEE / 'arms' / f'{a.dom}_{a.ser}_{m["obs"]}_{m["k"]}{a.tag}.npz', **rec)
        log(json.dumps({'obs': m['obs'], 'k': m['k'], 'n2': m['n2_tok'], **info}))
    res = {arm: {'s_per_frame_median': float(np.median(v[1:] if len(v) > 1 else v)), 'n': len(v),
                 'device': torch.cuda.get_device_name(0)} for arm, v in tm.items()}
    (OUT / 'time').mkdir(parents=True, exist_ok=True)
    (OUT / 'time' / f'{a.dom}_{a.ser}{a.tag}.json').write_text(json.dumps(res, indent=1))
    log(json.dumps(res))


def core_of(c, i):
    """prompt-object core: frozen dark cue pixels | N2 cue pixels in the opened body (the 'strong' core is not stored;
    the body of the cue stands in for it)."""
    return c['core0'][i] | c['n2_pix'][i] & ~c['n2_app'][i]


def overlay(im, pix, col, a=0.45):
    o = im.astype(np.float32).copy()
    o[pix] = (1 - a) * o[pix] + a * np.array(col)
    return o.clip(0, 255).astype(np.uint8)


def tint(im, keep, col, a=0.4):
    G = int(round(np.sqrt(keep.size)))
    S = im.shape[0]
    K = np.kron(keep.reshape(G, G), np.ones((S // G, S // G), bool))
    return overlay(im, K, col, a)


def cmd_devview(a):
    """dev panels: frame | N2 tokens | N2 pixels | per arm pixels (+ token count) for the captured frames of a series."""
    from PIL import Image, ImageDraw
    meta = json.loads((OUT / 'cap' / f'{a.dom}_{a.ser}.json').read_text())
    sel = [m for m in meta if (not a.role or m['role'].startswith(a.role))][:a.n]
    arms = a.arms.split(',')
    (OUT / 'dev').mkdir(parents=True, exist_ok=True)
    for m in sel:
        c = load_cap(a.dom, a.ser, m['obs'], m['k'])
        z = np.load(BEE / 'arms' / f'{a.dom}_{a.ser}_{m["obs"]}_{m["k"]}{a.tag}.npz')
        S = c['x'].shape[1]
        im = c['x'][-1]
        tiles = [(im, 'frame'), (tint(im, c['n2_keep'][-1], (0, 90, 255)), f'N2 tok {int(c["n2_keep"][-1].sum())}'),
                 (overlay(im, c['n2_pix'][-1], (0, 90, 255)), 'N2 px')]
        for arm in arms:
            p = np.unpackbits(z[arm + '_pix'], -1)[:, :S].astype(bool)
            tiles.append((overlay(im, p, (0, 220, 0)), f'{arm} px tok {int(z[arm + "_keep"].sum())}'))
        if a.crop:  # zoom on the union bounding box of N2 pixels
            ys, xs = np.nonzero(c['n2_pix'][-1])
            if len(ys):
                cx, cy = int(np.median(xs)), int(np.median(ys))
                s = min(S, a.crop)
                x0, y0 = int(np.clip(cx - s // 2, 0, S - s)), int(np.clip(cy - s // 2, 0, S - s))
                tiles = [(t[y0:y0 + s, x0:x0 + s], n) for t, n in tiles]
        T = 400
        sh = Image.new('RGB', (T * len(tiles), T))
        for j, (t, nm) in enumerate(tiles):
            tt = Image.fromarray(t).resize((T, T), Image.BILINEAR)
            ImageDraw.Draw(tt).text((4, 4), nm, fill=(255, 255, 0))
            sh.paste(tt, (j * T, 0))
        sh.save(OUT / 'dev' / f'{a.dom}_{a.ser}_{m["obs"]}_{m["k"]}{a.tag}{"_z" if a.crop else ""}.png')


def grid_img(im, x0, y0, side, T, step, lab_every, marks=()):
    """crop [y0:y0+side, x0:x0+side] -> T x T PIL image with grid lines every `step` source px, labels in ABSOLUTE
    source coordinates every lab_every px; marks [(x, y, colour)] drawn as small crosses."""
    from PIL import Image, ImageDraw
    c = im[y0:y0 + side, x0:x0 + side]
    t = Image.fromarray(c).resize((T, T), Image.NEAREST if T >= side * 2 else Image.BILINEAR)
    d = ImageDraw.Draw(t)
    sc = T / side
    for v in range((x0 // step + 1) * step, x0 + side, step):
        X = (v - x0) * sc
        d.line([(X, 0), (X, T)], fill=(255, 0, 255) if v % lab_every == 0 else (0, 255, 255), width=1)
        if v % lab_every == 0:
            d.text((X + 2, 2), str(v), fill=(255, 255, 0))
    for v in range((y0 // step + 1) * step, y0 + side, step):
        Y = (v - y0) * sc
        d.line([(0, Y), (T, Y)], fill=(255, 0, 255) if v % lab_every == 0 else (0, 255, 255), width=1)
        if v % lab_every == 0:
            d.text((2, Y + 2), str(v), fill=(255, 255, 0))
    d.rectangle([0, T - 14, 260, T], fill=(0, 0, 0))
    d.text((3, T - 13), f'x0={x0} y0={y0} scale={sc:.3f}', fill=(255, 255, 0))
    for x, y, col in marks:
        X, Y = (x - x0) * sc, (y - y0) * sc
        d.line([(X - 5, Y), (X + 5, Y)], fill=col, width=2)
        d.line([(X, Y - 5), (X, Y + 5)], fill=col, width=2)
    return t


def gold_frames():
    out = []
    for dom, ser in SERIES:
        f = OUT / 'cap' / f'{dom}_{ser}.json'
        if f.exists():
            out += [(dom, ser, m) for m in json.loads(f.read_text()) if m['role'] == 'gold']
    return out


def wide(cb, S, f=1.6):
    a, b, side = cb
    s2 = int(min(S, round(side * f)))
    return (int(np.clip(a + side / 2 - s2 / 2, 0, S - s2)), int(np.clip(b + side / 2 - s2 / 2, 0, S - s2)), s2)


def cmd_goldview(a):
    """annotation views: full frame with a 64 px grid + one zoom per N2 prompt object (16 px grid), and the auto body
    points (red) that seed the gold prompts."""
    from src.eci.v2 import mask4 as M4
    V = Path(a.view or OUT / 'gold' / 'view')
    V.mkdir(parents=True, exist_ok=True)
    auto = {}
    for dom, ser, m in gold_frames():
        if a.dom and dom != a.dom:
            continue
        c = load_cap(dom, ser, m['obs'], m['k'])
        im, S, h = c['x'][-1], c['x'].shape[1], m['h']
        objs = M4.objects(c['n2_pix'][-1], core_of(c, -1), h)
        key = f'{dom}_{ser}_{m["obs"]}_{m["k"]}'
        auto[key] = [{'pts': [o['pts'][0]], 'crop': list(wide(M4.crop_box(o, S, h), S))} for o in objs]
        marks = [(o['pts'][0][0], o['pts'][0][1], (255, 0, 0)) for o in objs]
        grid_img(im, 0, 0, S, 1024, 64 if S > 500 else 32, 128 if S > 500 else 64, marks).save(V / f'{key}_full.png')
        for j, o in enumerate(objs):
            ca, cb, side = wide(M4.crop_box(o, S, h), S)
            grid_img(im, ca, cb, side, 768, 8 if side < 160 else 16, 32 if side < 320 else 64,
                     [(p[0], p[1], (255, 0, 0)) for p in o['pts']]).save(V / f'{key}_obj{j}.png')
    (OUT / 'gold').mkdir(parents=True, exist_ok=True)
    (OUT / 'gold' / 'auto_seed.json').write_text(json.dumps(auto, indent=1))
    log(f'{len(auto)} frames')


def outline(m):
    import cv2
    return (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~m


def cmd_gold(a):
    """gold masks: SAM 2.1 large, one pass per hand-prompted animal on its crop (points only unless a box is given),
    mask = 'pick' (0-2) or the highest predicted IoU; union per frame. Hand prompts: OUT/gold/prompts.json
    {key: {'animals': [{'pts', 'labels', 'crop': [x0, y0, side], 'pick', 'box'}], 'thin': [[x, y], ...]}}."""
    import torch
    from PIL import Image
    from src.eci.v2 import mask4 as M4
    dev = torch.device('cuda')
    model, proc = M4.load_sam('large', dev)
    P = json.loads((OUT / 'gold' / 'prompts.json').read_text())
    G = OUT / 'gold'
    (G / 'check').mkdir(parents=True, exist_ok=True)
    keys = a.keys.split(',') if a.keys else list(P)
    for key in keys:
        e = P[key]
        dom, ser = key.split('_')[:2]
        obs, k = key[len(dom) + len(ser) + 2:].rsplit('_', 1)
        c = load_cap(dom, ser, obs, int(k))
        im, S = c['x'][-1], c['x'].shape[1]
        gold = np.zeros((S, S), bool)
        per = []
        for an in e['animals']:
            x0, y0, side = an['crop']
            side = int(min(side, S))
            x0, y0 = int(np.clip(x0, 0, S - side)), int(np.clip(y0, 0, S - side))
            kw = dict(images=np.ascontiguousarray(im[y0:y0 + side, x0:x0 + side]),
                      input_points=[[[[px - x0, py - y0] for px, py in an['pts']]]], input_labels=[[an['labels']]],
                      return_tensors='pt')
            if an.get('box'):
                bx = an['box']
                kw['input_boxes'] = [[[bx[0] - x0, bx[1] - y0, bx[2] - x0, bx[3] - y0]]]
            inp = proc(**kw).to(dev)
            with torch.no_grad(), torch.autocast('cuda', dtype=M4.adt()):
                out = model(**inp, multimask_output=True)
            ms = proc.post_process_masks(out.pred_masks.float().cpu(), inp['original_sizes'].cpu())[0][0].numpy()
            sc = out.iou_scores.float().cpu().numpy()[0, 0]
            j = an['pick'] if an.get('pick') is not None else int(np.argmax(sc))
            m = np.zeros((S, S), bool)
            m[y0:y0 + side, x0:x0 + side] = ms[j]
            if an.get('erase'):  # hand-drawn erase boxes [x0, y0, x1, y1] (background SAM includes)
                for ex in an['erase']:
                    m[ex[1]:ex[3], ex[0]:ex[2]] = False
            per.append({'scores': [float(v) for v in sc], 'pick': j, 'area': int(m.sum())})
            gold |= m
            # check view: crop | 3 multimask candidates, chosen one marked
            tiles = [im[y0:y0 + side, x0:x0 + side]]
            for t in range(3):
                mm = ms[t]
                tiles.append(overlay(tiles[0], outline(mm), (255, 0, 0) if t == j else (0, 255, 0), 0.9))
            row = np.concatenate([np.asarray(Image.fromarray(t).resize((384, 384), Image.NEAREST)) for t in tiles], 1)
            Image.fromarray(row).save(G / 'check' / f'{key}_a{len(per) - 1}.png')
        thin = np.array(e.get('thin', []), float).reshape(-1, 2)
        np.savez_compressed(G / f'{key}.npz', pix=np.packbits(gold, -1), thin=thin)
        # full check: outline of the gold mask + thin points
        ov = overlay(im, outline(gold) | outline(outline(gold) | gold), (255, 0, 0), 0.9)
        marks = [(x, y, (0, 255, 0)) for x, y in thin] + [(p[0], p[1], (255, 255, 0)) for an in e['animals']
                                                           for p, l in zip(an['pts'], an['labels']) if l == 1]
        marks += [(p[0], p[1], (0, 0, 255)) for an in e['animals'] for p, l in zip(an['pts'], an['labels']) if l == 0]
        grid_img(ov, 0, 0, S, 1024, 64 if S > 500 else 32, 128 if S > 500 else 64, marks).save(G / 'check' / f'{key}_full.png')
        for j, an in enumerate(e['animals']):
            x0, y0, side = an['crop']
            side = int(min(side, S))
            x0, y0 = int(np.clip(x0, 0, S - side)), int(np.clip(y0, 0, S - side))
            grid_img(overlay(im, outline(gold), (255, 0, 0), 0.9), x0, y0, side, 768, 8 if side < 160 else 16,
                     32 if side < 320 else 64, marks).save(G / 'check' / f'{key}_z{j}.png')
        log(json.dumps({'key': key, 'animals': per, 'gold_px': int(gold.sum()), 'thin': len(thin)}))


def arm_masks(dom, ser, m, tag=''):
    """-> {arm: (pix (S, S) bool, keep (G*G,) bool)} for N2, mask3b and every SAM arm on disk."""
    c = load_cap(dom, ser, m['obs'], m['k'])
    S = c['x'].shape[1]
    out = {'N2': (c['n2_pix'][-1], c['n2_keep'][-1].astype(bool)), 'mask3b': (c['m3b_pix'], c['m3b_keep'].astype(bool))}
    f = BEE / 'arms' / f'{dom}_{ser}_{m["obs"]}_{m["k"]}{tag}.npz'
    if f.exists():
        z = np.load(f)
        for nm in z.files:
            if nm.endswith('_pix'):
                arm = nm[:-4]
                out[arm] = (np.unpackbits(z[nm], -1)[:, :S].astype(bool), z[arm + '_keep'].astype(bool))
    return out, c


def frame_counts(pix, keep, gold, gtok, gfrac, thin, S, G):
    """per-frame sufficient statistics for the pre-registered metrics."""
    import cv2
    st = {'inter_px': int((pix & gold).sum()), 'union_px': int((pix | gold).sum()), 'tp_tok': int((keep & gtok).sum()),
          'kept': int(keep.sum()), 'gold_tok': int(gtok.sum()), 'bg_tok': int((keep & (gfrac < 0.05)).sum()),
          'thin_n': len(thin)}
    if len(thin):
        ti = np.clip(thin.round().astype(int), 0, S - 1)
        cell = (ti[:, 1] // (S // G)) * G + ti[:, 0] // (S // G)
        st['thin_tok'] = int(keep[cell].sum())
        dd = cv2.dilate(pix.astype(np.uint8), M4disk(3)) > 0
        st['thin_px'] = int(dd[ti[:, 1], ti[:, 0]].sum())
    else:
        st['thin_tok'] = st['thin_px'] = 0
    return st


def M4disk(r):
    from src.eci.v2 import mask4 as M4
    return M4.disk(r)


def summarise(rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    rec = s['tp_tok'] / max(s['gold_tok'], 1)
    pre = s['tp_tok'] / max(s['kept'], 1)
    return {'iou_px': s['inter_px'] / max(s['union_px'], 1), 'tok_recall': rec, 'tok_precision': pre,
            'tok_f1': 2 * rec * pre / max(rec + pre, 1e-9), 'thin_recall_tok': s['thin_tok'] / max(s['thin_n'], 1),
            'thin_recall_px': s['thin_px'] / max(s['thin_n'], 1), 'bg_tok_per_frame': s['bg_tok'] / len(rows),
            'bg_share': s['bg_tok'] / max(s['kept'], 1), 'tok_per_frame': s['kept'] / len(rows),
            'gold_tok_per_frame': s['gold_tok'] / len(rows), 'n_frames': len(rows), 'n_thin': s['thin_n']}


def cmd_metrics(a):
    import torch
    import torch.nn.functional as F
    from src.eci.v2 import mask4 as M4
    rows = {}  # (dom, ser) -> arm -> [frame stats]
    for dom, ser, m in gold_frames():
        key = f'{dom}_{ser}_{m["obs"]}_{m["k"]}'
        gf = OUT / 'gold' / f'{key}.npz'
        if not gf.exists():
            raise FileNotFoundError(f'no gold mask for {key}')
        z = np.load(gf)
        arms, c = arm_masks(dom, ser, m, a.tag)
        S, G = c['x'].shape[1], m['G']
        gold = np.unpackbits(z['pix'], -1)[:, :S].astype(bool)
        gtok = M4.tokens(gold, m['h'])
        gfrac = F.avg_pool2d(torch.from_numpy(gold).float()[None, None], S // G).flatten().numpy()
        for arm, (pix, keep) in arms.items():
            rows.setdefault((dom, ser), {}).setdefault(arm, []).append(
                frame_counts(pix, keep, gold, gtok, gfrac, z['thin'], S, G))
    arms_all = sorted({arm for d in rows.values() for arm in d}, key=lambda s: (s != 'N2', s != 'mask3b', s))
    res = {'series': {}, 'domain': {}, 'boot': {}}
    for (dom, ser), d in rows.items():
        res['series'][f'{dom}_{ser}'] = {arm: summarise(r) for arm, r in d.items()}
    doms = sorted({d for d, _ in rows})
    g = np.random.default_rng(0)
    for dom in doms:
        R = {arm: sum((rows[(d, s)].get(arm, []) for d, s in rows if d == dom), []) for arm in arms_all}
        res['domain'][dom] = {arm: summarise(r) for arm, r in R.items() if r}
        n = len(R['N2'])
        bs = {arm: {'tok_f1': [], 'thin_recall_tok': [], 'bg_tok_per_frame': []} for arm in arms_all if arm != 'N2'}
        for _ in range(1000):
            ix = g.integers(0, n, n)
            b0 = summarise([R['N2'][i] for i in ix])
            for arm in bs:
                if len(R[arm]) != n:
                    continue
                b1 = summarise([R[arm][i] for i in ix])
                for kk in bs[arm]:
                    bs[arm][kk].append(b1[kk] - b0[kk])
        res['boot'][dom] = {arm: {kk: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for kk, v in d_.items() if v}
                            for arm, d_ in bs.items()}
    # decision rule (pre-registered)
    cands = [arm for arm in arms_all if arm[0] in 'AB' and all(arm in res['domain'][d] for d in doms)]
    f1 = {arm: float(np.mean([res['domain'][d][arm]['tok_f1'] for d in doms])) for arm in arms_all
          if all(arm in res['domain'][d] for d in doms)}
    win = max(cands, key=lambda arm: f1[arm]) if cands else None
    ok = win is not None and all(
        res['domain'][d][win]['thin_recall_tok'] > res['domain'][d]['N2']['thin_recall_tok'] and
        res['domain'][d][win]['bg_tok_per_frame'] < res['domain'][d]['N2']['bg_tok_per_frame'] for d in doms)
    res['decision'] = {'overall_f1': f1, 'winner': win, 'adopted': bool(ok)}
    (OUT / f'metrics{a.tag}.json').write_text(json.dumps(res, indent=1))
    L = ['# Mask v4 bake-off: gold-set metrics (pre-registered in scripts/eci/v2/mask4_prereg.md)', '']
    cols = ['iou_px', 'tok_recall', 'tok_precision', 'tok_f1', 'thin_recall_tok', 'thin_recall_px', 'bg_tok_per_frame',
            'bg_share', 'tok_per_frame']
    for lvl in ('domain', 'series'):
        L += [f'## per {lvl}', '', '| ' + lvl + ' | arm | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 2)]
        for nm, d in res[lvl].items():
            for arm in arms_all:
                if arm in d:
                    L.append(f'| {nm} | {arm} | ' + ' | '.join(f'{d[arm][c_]:.3f}' for c_ in cols) + ' |')
            first = next(iter(d.values()))
            L.append(f'| {nm} | (gold: {first["n_frames"]} frames, {first["gold_tok_per_frame"]:.1f} gold tok / frame, '
                     f'{first["n_thin"]} thin points) |' + ' |' * len(cols))
        L.append('')
    L += ['## decision', '', f'overall token F1 (mean of domains): ' + ', '.join(f'{k} {v:.3f}' for k, v in f1.items()),
          '', f'winner {win}; adopted by the pre-registered rule: {ok}', '']
    if win:
        for d in doms:
            b = res['boot'][d].get(win, {})
            w, n0 = res['domain'][d][win], res['domain'][d]['N2']
            L.append(f'- {d}: token F1 {w["tok_f1"]:.3f} vs N2 {n0["tok_f1"]:.3f} (diff 95% CI {b.get("tok_f1")}); '
                     f'thin recall (token) {w["thin_recall_tok"]:.3f} vs {n0["thin_recall_tok"]:.3f} (CI {b.get("thin_recall_tok")}); '
                     f'background tokens / frame {w["bg_tok_per_frame"]:.2f} vs {n0["bg_tok_per_frame"]:.2f} (CI {b.get("bg_tok_per_frame")})')
    (OUT / f'table{a.tag}.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def cmd_bench(a):
    """benchmarks: validate.py frames / packs / h / score_frame unchanged; N2 identity vs <set>_final.json; SAM arms
    prompted by the validate N2 cue (image arms only: the frames are sparse, no clip)."""
    import torch
    sys.path.insert(0, str(REPO / 'scripts/eci/v2_mask2'))
    import validate as VA
    from src.eci.v2 import mask2 as M2
    from src.eci.v2 import mask4 as M4
    dev = torch.device('cuda')
    kind = a.set
    t0 = time.time()
    if kind == 'cal':
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(6) as ex:
            loaded = list(ex.map(lambda it: VA.load_cal(it, 'test'), VA.cal_videos('test', 24, 0)))
        S, G = 1024, 64
        vids = [(nm, torch.from_numpy(xb).to(dev), torch.from_numpy(x).to(dev), ks, kps) for nm, xb, x, ks, kps in loaded]
    elif kind == 'crim':
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(6) as ex:
            v0 = list(ex.map(lambda it: VA.load_crim(it, dev), VA.crim_videos(12, 16, 0)))
        vids = [(nm, xb, x, ks, [(xy[i], st) for i in range(len(ks))]) for nm, st, xb, x, ks, xy in v0]
        S, G = 640, 40
    else:
        from common import mab_decode
        vids = []
        for c, ks, kps in VA.mab_clips(120, 8, 0):
            arr = mab_decode('mouse_submission', c)
            ib = np.linspace(0, len(arr) - 1, VA.N_BG).round().astype(int)
            vids.append((c, VA.up4(arr[ib], dev), VA.up4(arr[ks + 1], dev), ks, kps))
        S, G = 896, 56
    packs = [M2.background_pack(xb) for _, xb, _, _, _ in vids]
    h = VA.body_h([c.cpu().numpy() for (_, xb, _, _, _), pk in zip(vids, packs) for c in VA.dark_cue(xb[::6], pk['q98'])])
    log(f'{kind}: {len(vids)} videos loaded, h {h:.2f}, {time.time() - t0:.0f}s')
    arms = a.arms.split(',') if a.arms else ['A_s_full', 'A_l_full', 'A_s_crop', 'A_l_crop']
    models = {}
    rows = {k: [] for k in ['N2'] + arms + ['B' + x[1:] for x in arms]}
    tm = {arm: [] for arm in arms}
    sheet = []
    g1 = np.random.default_rng(1)
    pickv = set(g1.choice(len(vids), 2, replace=False).tolist())
    for vi, ((nm, xb, x, ks, kps), pk) in enumerate(zip(vids, packs)):
        for b0 in range(0, len(x), 8):
            xx = x[b0:b0 + 8]
            ms, r = VA.masks_all(xx, pk, h)
            core0 = VA.dark_cue(xx, pk['q98'])
            _, app = M2.appendage(r['pix'], h)
            for i in range(len(xx)):
                kk = kps[b0 + i]
                rows['N2'].append(VA.score_frame(ms['N2'][0][i], ms['N2'][1][i], kk, S, G, h, kind))
                pix = r['pix'][i].cpu().numpy()
                ap = app[i].cpu().numpy()
                core = core0[i].cpu().numpy() | (pix & ~ap)
                img = xx[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                objs = M4.objects(pix, core, h)
                keepd = {}
                for arm in arms:
                    _, size, mode = arm.split('_')
                    size = {'s': 'small', 'l': 'large'}[size]
                    if size not in models:
                        models[size] = M4.load_sam(size, dev)
                    mod, proc = models[size]
                    torch.cuda.synchronize()
                    t1 = time.time()
                    ch = M4.sam_full(mod, proc, img, objs, dev) if mode == 'full' else M4.sam_crop(mod, proc, img, objs, dev, h)
                    torch.cuda.synchronize()
                    tm[arm].append(time.time() - t1)
                    for rule in ('A', 'B'):
                        u, _ = M4.assemble(ch, objs, pix, core, h, rule=rule, app=ap)
                        kp_ = M4.tokens(u, h)
                        nm_ = arm if rule == 'A' else 'B' + arm[1:]
                        rows[nm_].append(VA.score_frame(torch.from_numpy(kp_), torch.from_numpy(u), kk, S, G, h, kind))
                        keepd[nm_] = (kp_, u)
                if vi in pickv and b0 + i == 0:
                    sheet.append({'img': img, 'n2': ms['N2'][0][i].cpu().numpy(), 'n2pix': pix, 'arms': keepd, 'nm': nm})
        log(f'{kind} video {vi + 1}/{len(vids)} {time.time() - t0:.0f}s')
    out = {'set': kind, 'S': S, 'G': G, 'h': h, 'n_videos': len(vids), 'n_frames': len(rows['N2']),
           'device': torch.cuda.get_device_name(0), 'wall_s': time.time() - t0,
           's_per_frame_median': {arm: float(np.median(v[1:])) for arm, v in tm.items()}}
    for arm, R in rows.items():
        keys = sorted({k for r in R for k in r if k not in ('tokens', 'leak_n')})
        out[arm] = {k: float(np.mean([r[k] for r in R if k in r])) for k in keys}
        out[arm]['tokens_per_frame'] = float(np.mean([r['tokens'] for r in R]))
        out[arm]['leak'] = float(sum(r['leak_n'] for r in R) / max(sum(r['tokens'] for r in R), 1))
    ref = json.loads((REPO / f'results/vision/eci_v2_mask2/validate/{kind}_final.json').read_text())
    out['identity_N2_max_abs_diff'] = max(abs(out['N2'][k] - v) for k, v in ref['N2'].items())
    out['identity_h_diff'] = h - ref['h']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'bench_{kind}.json').write_text(json.dumps(out, indent=1))
    (BEE / 'bench').mkdir(parents=True, exist_ok=True)
    for j, sr in enumerate(sheet):
        np.savez_compressed(BEE / 'bench' / f'{kind}_{j}.npz', img=sr['img'], n2=sr['n2'], n2pix=sr['n2pix'], nm=sr['nm'],
                            **{f'{k}_keep': v[0] for k, v in sr['arms'].items()},
                            **{f'{k}_pix': np.packbits(v[1], -1) for k, v in sr['arms'].items()})
    log(json.dumps({k: v for k, v in out.items() if k in ('N2', 'identity_N2_max_abs_diff', 'identity_h_diff')}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step')
    ap.add_argument('--dom', default='mice')
    ap.add_argument('--ser', default='v1')
    ap.add_argument('--set', default='mab')
    ap.add_argument('--arms', default='')
    ap.add_argument('--tag', default='')
    ap.add_argument('--role', default='')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--crop', type=int, default=0)
    ap.add_argument('--view', default='')
    ap.add_argument('--keys', default='')
    a = ap.parse_args()
    globals()['cmd_' + a.step](a)


if __name__ == '__main__':
    main()
