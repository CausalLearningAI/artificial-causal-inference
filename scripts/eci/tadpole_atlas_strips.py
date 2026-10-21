"""
Frame strips with REAL per-token heat maps for the frog tadpole neurons of the atlas (v2 = R4, d2 fnorm SAE; the
neurons listed by the tadpole NES / NESS entries of scripts/eci/build_atlas_v37.py step_dump_tadpoles). GPU.

The tadpole codes are of a 96 px crop that follows the tadpole in the raw 60 fps video (scripts/eci/tadpole_r4.py:
locate_v2 track, dead-band boxes, crop upsampled to 384 = 24 x 24 patches). There are no dataset frames, so this
script re-cuts the crops itself with the same code (tadpole_r4.decode_iter / load_locate / crop / Enc, imported read
only) and re-encodes them: own crop of 5 fps frame k with the box of k, lag crop = frame k - 10 cut with the box of k
(d2 input [z_t, z_t - z_(t-10)]), the frogs pixel mask on the crop, the kept SAE in threshold mode; the heat of
neuron j = its code on every kept token, kept iff > threshold_per_neuron[j] (the fnorm rule of the page's codes).
Checks per (frame, neuron): max over tokens of the unthresholded code vs the stored per-video codes (videos/<obs>.npz
max_d2) and argmax token vs arg_d2 (agreement rates in the output; frames are encoded in small batches here, not in
the full pass's original batches, so bit-exactness is not expected).

Picks per neuron (fnorm codes of the layout, the values the page's searches use): top = the highest frame of each
video, the 8 videos with the highest such frame; least = 8 random videos holding a frame where the neuron is 0, one
such frame at random (fixed seed). Sprite per neuron: row 0 top frames, row 1 the same with the heat map (turbo,
scale = 99th percentile of the neuron's positive heat values over the shown frames, as v1 / v2 clips), row 2 least
frames; tile = the 96 px crop at TILE px.
Output: results/vision/eci_atlas_v37/tadpole_strips/{strips.json, tad_<neuron>.webp}
Usage: sbatch scripts/eci/tadpole_atlas_strips.sh

--v2 smooth | thr: the FROZEN v2 (scripts/eci/v2/, its frogs SAE on the same tadpole crops): neurons of the build
build/v2_frogs_tad_<variant>, picks on its codes set (<sae>_smooth or <sae>), heat = the stored per-token codes of
its full pass (scripts/eci/v2/tools.py heat_rows: exact, the max over tokens equals the frame code), no re-encoding
(CPU only). The crops are re-cut from the raw video with the same locate boxes. -> tadpole_strips_v2<variant>/
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import tadpole_r4 as T  # noqa: E402

R = T.R
SAE = 'matryoshka_btk_512_k16_r4d2_s0'
LAYOUT = REPO / 'dataset/frogs/eci_tadpole_r4'
BUILD = REPO / 'results/vision/eci_atlas_v37/build/frogs_tad_d2'
OUT = REPO / 'results/vision/eci_atlas_v37/tadpole_strips'
QC = REPO / 'results/vision/eci_rollout_r4/frogs_tadpole/qc.json'
K, TILE = 8, 160


def log(*x):
    print(time.strftime('[%H:%M:%S]'), *x, flush=True)


def v2_heat(variant, picks_rows):
    """{(layout row, j): (G, G)} from the frozen v2's stored token codes."""
    import importlib.util
    sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
    sp = importlib.util.spec_from_file_location('v2_tools', REPO / 'scripts/eci/v2/tools.py')
    TL = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(TL)
    c = TL.RUN.dcfg('frogs')
    sc = TL.RUN.series_of(c, 'tad')
    return TL.heat_rows(c, sc, picks_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2', default=None, choices=(None, 'smooth', 'thr'))
    args = ap.parse_args()
    global BUILD, OUT
    if args.v2:
        return main_v2(args.v2)
    import torch
    import matplotlib
    matplotlib.use('Agg')
    from PIL import Image
    from src.eci.viz import _overlay_rgb
    t_all = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    js = sorted({int(j) for f in BUILD.glob('data/*_n.json') for j in json.loads(f.read_text())['by']['max']})
    P = R.load_plan(T.DOM)
    excl = set(json.loads(QC.read_text())['excluded'])
    vids = [v for v in P['videos'] if v['obs'] not in excl]
    fn = np.load(LAYOUT / 'codes' / f'{SAE}_fnorm' / 'codes_max.npy', mmap_mode='r')
    thr = np.load(LAYOUT / 'codes' / f'{SAE}_fnorm' / 'threshold_per_neuron.npy').astype(np.float32)
    rng = np.random.default_rng(0)
    picks = {}
    for j in js:
        best, zero = [], []
        for v in vids:
            x = np.asarray(fn[v['lay_lo']:v['lay_lo'] + v['n'], j], np.float32)
            k = int(np.argmax(x))
            best.append((float(x[k]), v['obs'], k))
            z = np.flatnonzero(x == 0)
            if len(z):
                zero.append((v['obs'], z))
        top = [b for b in sorted(best, key=lambda b: -b[0]) if b[0] > 0][:K]
        sel = rng.choice(len(zero), size=min(K, len(zero)), replace=False) if zero else []
        least = [(0.0, zero[i][0], int(rng.choice(zero[i][1]))) for i in sel]
        picks[j] = {'top': top, 'least': least}
    frames = sorted({(o, k) for p in picks.values() for _, o, k in p['top'] + p['least']})
    log(f'{len(js)} neurons, {len(frames)} frames to re-encode')
    vb = {v['obs']: v for v in vids}
    Ls = {}

    def cut(fk):
        o, k = fk
        v = vb[o]
        if o not in Ls:
            Ls[o] = T.load_locate(o)
        L = Ls[o]
        a = T.lag_index(k)
        fr = list(T.decode_iter(v['path'], v['w'], v['h'], a, k - a + 1, threads=2))
        x0, y0 = int(L['x0'][k]), int(L['y0'][k])
        return fk, T.crop(fr[-1], x0, y0), T.crop(fr[0], x0, y0), T.crop_plane(L['pix_bg'], x0, y0, None), \
            T.crop_plane(L['roi'], x0, y0, False)
    for o in {o for o, _ in frames}:
        Ls[o] = T.load_locate(o)
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        crops = list(ex.map(cut, frames))
    t_dec = time.time() - t0
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    enc = T.Enc(dev)
    saes, best = R.load_saes(T.DOM, dev)
    sae, norm = saes['d2']
    G = T.G
    heat, chk = {}, {'pairs': 0, 'max_close': 0, 'arg_agree': 0, 'active_pairs': 0, 'max_abs_diff': 0.0}
    stored = {}
    t0 = time.time()
    BS = 32
    for b in range(0, len(crops), BS):
        ch = crops[b:b + BS]
        own = np.stack([c[1] for c in ch])
        lag = np.stack([c[2] for c in ch])
        x = enc.up(own)
        M, _ = enc.mask(x, np.stack([c[3] for c in ch]), np.stack([c[4] for c in ch]))
        Z = enc.tokens(x)
        Pz = enc.tokens(enc.up(lag))
        with torch.no_grad():
            for i, (fk, *_r) in enumerate(ch):
                pi = M[i].nonzero(as_tuple=True)[0]
                h = np.zeros((G * G, len(js)), np.float32)
                if len(pi):
                    zt = Z[i, pi]
                    xx = torch.cat([zt, (zt.float() - Pz[i, pi].float()).half()], 1)
                    zz = sae.encode(norm(xx.float()), mode='threshold')[:, js].float().cpu().numpy()
                    h[pi.cpu().numpy()] = zz
                heat[fk] = h.astype(np.float16)
    t_gpu = time.time() - t0
    # check vs the stored per-video codes of the full pass
    for fk, h in heat.items():
        o, k = fk
        if o not in stored:
            z = np.load(T.BEE / 'videos' / f'{o}.npz')
            stored[o] = (z['max_d2'], z['arg_d2'])
        mx, ag = stored[o]
        hm = h.astype(np.float32)
        for c, j in enumerate(js):
            ref, got = float(mx[k, j]), float(hm[:, c].max())
            chk['pairs'] += 1
            chk['max_abs_diff'] = max(chk['max_abs_diff'], abs(ref - got))
            chk['max_close'] += abs(ref - got) <= 0.02 * max(abs(ref), 1e-3) + 1e-3
            if ref > 0:
                chk['active_pairs'] += 1
                chk['arg_agree'] += int(int(np.argmax(hm[:, c])) == int(ag[k, j]))
    chk['max_close_frac'] = chk['max_close'] / max(chk['pairs'], 1)
    chk['arg_agree_frac'] = chk['arg_agree'] / max(chk['active_pairs'], 1)
    log(f'check vs stored codes: {json.dumps(chk)}')
    cmap = matplotlib.colormaps['turbo']
    cr = {c[0]: c[1] for c in crops}
    out = {}
    for c, j in enumerate(js):
        p = picks[j]
        H = {fk: np.where(heat[fk][:, c].astype(np.float32) > thr[j], heat[fk][:, c].astype(np.float32), 0)
             for fk in [(o, k) for _, o, k in p['top']]}
        pos = np.concatenate([h[h > 0] for h in H.values()]) if H else np.zeros(0)
        vmax = float(np.quantile(pos, 0.99)) if len(pos) else 1.0
        sp = Image.new('RGB', (K * TILE, 3 * TILE), (17, 17, 17))
        for i, (_, o, k) in enumerate(p['top']):
            im = cr[(o, k)]
            sp.paste(Image.fromarray(im).resize((TILE, TILE), Image.LANCZOS), (i * TILE, 0))
            ov = _overlay_rgb(np.asarray(im), H[(o, k)].reshape(G, G), (0, 0, im.shape[1]), vmax, cmap)
            sp.paste(Image.fromarray(ov).resize((TILE, TILE), Image.LANCZOS), (i * TILE, TILE))
        for i, (_, o, k) in enumerate(p['least']):
            sp.paste(Image.fromarray(cr[(o, k)]).resize((TILE, TILE), Image.LANCZOS), (i * TILE, 2 * TILE))
        name = f'tad_{j}.webp'
        sp.save(OUT / name, 'WEBP', quality=75, method=6)
        out[str(j)] = {'sprite': name, 'heat_vmax': round(vmax, 4),
                       'top': [[o, round(v, 4), k] for v, o, k in p['top']],
                       'least': [[o, 0.0, k] for _, o, k in p['least']]}
    (OUT / 'strips.json').write_text(json.dumps({'sae': f'{SAE}_fnorm', 'tile': TILE, 'k': K, 'check': chk,
                                                 'cost': {'decode_s': round(t_dec), 'gpu_s': round(t_gpu),
                                                          'total_s': round(time.time() - t_all),
                                                          'gpu': torch.cuda.get_device_name(0)},
                                                 'neurons': out}, separators=(',', ':')))
    log(f'{len(out)} strips -> {OUT}; decode {t_dec:.0f} s, GPU {t_gpu:.0f} s, total {time.time() - t_all:.0f} s')


def main_v2(variant):
    import matplotlib
    matplotlib.use('Agg')
    from PIL import Image
    from src.eci.viz import _overlay_rgb
    t_all = time.time()
    name = json.loads((REPO / 'results/vision/eci_v2/frogs/best.json').read_text())['name']
    sae = name + ('_smooth' if variant == 'smooth' else '')
    build = REPO / f'results/vision/eci_atlas_v37/build/v2_frogs_tad_{variant}'
    out_d = REPO / f'results/vision/eci_atlas_v37/tadpole_strips_v2{variant}'
    out_d.mkdir(parents=True, exist_ok=True)
    js = sorted({int(j) for f in build.glob('data/*_n.json') for j in json.loads(f.read_text())['by']['max']})
    P = R.load_plan(T.DOM)
    excl = set(json.loads(QC.read_text())['excluded'])
    vids = [v for v in P['videos'] if v['obs'] not in excl]
    fn = np.load(LAYOUT / 'codes' / sae / 'codes_max.npy', mmap_mode='r')
    rng = np.random.default_rng(0)
    picks = {}
    for j in js:
        best, zero = [], []
        for v in vids:
            x = np.asarray(fn[v['lay_lo']:v['lay_lo'] + v['n'], j], np.float32)
            k = int(np.argmax(x))
            best.append((float(x[k]), v['obs'], k))
            z = np.flatnonzero(x == 0)
            if len(z):
                zero.append((v['obs'], z))
        top = [b for b in sorted(best, key=lambda b: -b[0]) if b[0] > 0][:K]
        sel = rng.choice(len(zero), size=min(K, len(zero)), replace=False) if zero else []
        picks[j] = {'top': top, 'least': [(0.0, zero[i][0], int(rng.choice(zero[i][1]))) for i in sel]}
    vb = {v['obs']: v for v in vids}
    need = {}
    for j, p in picks.items():
        for _, o, k in p['top']:
            need.setdefault(vb[o]['lay_lo'] + k, set()).add(j)
    t0 = time.time()
    H = v2_heat(variant, need)
    t_heat = time.time() - t0
    frames = sorted({(o, k) for p in picks.values() for _, o, k in p['top'] + p['least']})
    Ls = {o: T.load_locate(o) for o in {o for o, _ in frames}}

    def cut(fk):
        o, k = fk
        v = vb[o]
        fr = list(T.decode_iter(v['path'], v['w'], v['h'], k, 1, threads=2))
        return fk, T.crop(fr[-1], int(Ls[o]['x0'][k]), int(Ls[o]['y0'][k]))
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        cr = dict(ex.map(cut, frames))
    t_dec = time.time() - t0
    cmap = matplotlib.colormaps['turbo']
    out = {}
    for j, p in picks.items():
        hm = {(o, k): H[(vb[o]['lay_lo'] + k, j)].astype(np.float32) for _, o, k in p['top']}
        pos = np.concatenate([h[h > 0] for h in hm.values()]) if hm else np.zeros(0)
        vmax = float(np.quantile(pos, 0.99)) if len(pos) else 1.0
        sp = Image.new('RGB', (K * TILE, 3 * TILE), (17, 17, 17))
        for i, (_, o, k) in enumerate(p['top']):
            im = cr[(o, k)]
            sp.paste(Image.fromarray(im).resize((TILE, TILE), Image.LANCZOS), (i * TILE, 0))
            ov = _overlay_rgb(np.asarray(im), hm[(o, k)], (0, 0, im.shape[1]), vmax, cmap)
            sp.paste(Image.fromarray(ov).resize((TILE, TILE), Image.LANCZOS), (i * TILE, TILE))
        for i, (_, o, k) in enumerate(p['least']):
            sp.paste(Image.fromarray(cr[(o, k)]).resize((TILE, TILE), Image.LANCZOS), (i * TILE, 2 * TILE))
        nm = f'v2{variant[0]}tad_{j}.webp'
        sp.save(out_d / nm, 'WEBP', quality=75, method=6)
        out[str(j)] = {'sprite': nm, 'heat_vmax': round(vmax, 4), 'top': [[o, round(v, 4), k] for v, o, k in p['top']],
                       'least': [[o, 0.0, k] for _, o, k in p['least']]}
    chk = {'exact': True, 'arg_agree_frac': 1.0,
           'note': 'heat = the stored per-token codes of the full pass (max over tokens == frame code, asserted)'}
    (out_d / 'strips.json').write_text(json.dumps({'sae': sae, 'tile': TILE, 'k': K, 'check': chk,
                                                   'cost': {'decode_s': round(t_dec), 'heat_s': round(t_heat),
                                                            'total_s': round(time.time() - t_all), 'gpu': None},
                                                   'neurons': out}, separators=(',', ':')))
    log(f'v2 {variant}: {len(out)} strips -> {out_d}; heat {t_heat:.0f} s, decode {t_dec:.0f} s, '
        f'total {time.time() - t_all:.0f} s')


if __name__ == '__main__':
    main()
