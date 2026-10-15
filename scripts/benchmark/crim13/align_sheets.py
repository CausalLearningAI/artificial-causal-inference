"""
CRIM13 alignment test - sheets (CPU job). Rules: eci_align.sh.
    mask    8 eval videos (4 with a BC = pale intruder, 4 with a C57 = dark intruder; rng 0) x 2 frames with both mice
            tracked: kept cells tinted red, resident track point green, intruder cyan, own-cell hit written per animal
            -> sheets/mask_BC.jpg, sheets/mask_C57.jpg
    sniff   the C1 model (fold, seed) whose AUROC-picked sniff neuron has the highest SELECTION-half AUROC; the 16 highest
            frames of that neuron on the SCORING half's eval frames; 224 x 224 native-px crops centred on the peak
            cell; green border = sniff (true positive), red = not sniff (its label written)
            -> sheets/sniff_C1_auc.jpg and the same model's honest top-1% pick -> sheets/sniff_C1_top1.jpg
            (also C4 if its eval exists -> sheets/sniff_C4_*.jpg)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from align_common import (CANVAS, CELL, GRID, OUT, PLAN, ROOT, TOK, VOCAB, geom, load_seek, load_track, log,  # noqa: E402
                          read_frames, track_cell)

SH = OUT / 'sheets'


def frame_rgb(g):
    return np.repeat(g[:, :, None], 3, 2).astype(np.float32)


def mask_sheets():
    V = pd.read_parquet(PLAN / 'videos.parquet').set_index('video')
    Fs, pos = [], {}
    for j in sorted(TOK.glob('*of*_frames.parquet')):
        if 'smoke' in j.name:
            continue
        Fm = pd.read_parquet(j)
        p = np.load(str(j).replace('_frames.parquet', '_pos40.npy'))
        st = np.r_[0, np.cumsum(Fm.n40.values)]
        for i, r in enumerate(Fm.itertuples()):
            pos[(r.video, r.v)] = p[st[i]:st[i + 1]]
        Fs.append(Fm)
    F = pd.concat(Fs)
    F = F[(F.role == 'eval') & (F.nm == 2)]
    rng = np.random.default_rng(0)
    for strain in ('BC', 'C57'):
        vids = sorted(set(F.video) & set(V.index[V.strain == strain]))
        pick = rng.choice(vids, 4, replace=False)
        tiles = []
        for q in pick:
            R = F[F.video == q]
            R = R.iloc[rng.choice(len(R), 2, replace=False)]
            W, H = int(V.loc[q, 'W']), int(V.loc[q, 'H'])
            d = ROOT / V.loc[q, 'half']
            fr, _ = read_frames(d / f'{q}_t.seq', load_seek(d / f'{q}_t-seek.mat'), R.v.tolist())
            tp, _ = load_track(d / f'{q}_t-track.mat')
            pt, s = geom(W, H)
            for r in R.itertuples():
                im = frame_rgb(fr[r.v])
                m = np.zeros(GRID * GRID, bool)
                m[pos[(q, r.v)]] = True
                M = np.kron(m.reshape(GRID, GRID), np.ones((CELL, CELL), bool))   # 640-canvas px
                yy, xx = np.mgrid[0:H, 0:W]
                cy = np.clip(((yy + pt) * s).astype(int), 0, CANVAS - 1)
                cx = np.clip((xx * s).astype(int), 0, CANVAS - 1)
                Mn = M[cy, cx]
                im[Mn, 0] = im[Mn, 0] * 0.55 + 115
                im[Mn, 1:] *= 0.65
                pil = Image.fromarray(im.clip(0, 255).astype(np.uint8))
                dr = ImageDraw.Draw(pil)
                cells = track_cell(tp[r.v], W, H)
                for a, col in ((0, (0, 255, 0)), (1, (0, 220, 255))):
                    x, y = tp[r.v, a]
                    if np.isfinite(x):
                        dr.ellipse([x - 5, y - 5, x + 5, y + 5], outline=col, width=3)
                dr.text((6, 4), f'{q} v{r.v} kept {len(pos[(q, r.v)])} res {"hit" if m[cells[0]] else "MISS"} '
                                f'int {"hit" if m[cells[1]] else "MISS"}', fill=(255, 255, 0))
                tiles.append(pil.resize((480, int(480 * H / W))))
        th = max(t.size[1] for t in tiles)
        S = Image.new('RGB', (2 * 480, 4 * th))
        for k, t in enumerate(tiles):
            S.paste(t, ((k % 2) * 480, (k // 2) * th))
        S.save(SH / f'mask_{strain}.jpg', quality=85)
        log(f'mask sheet {strain}: {list(pick)}')


def sniff_sheet(arm):
    files = sorted((OUT / 'eval').glob(f'{arm}_?_s*.json'))
    if not files:
        return
    best = max(files, key=lambda f: json.loads(f.read_text())['main']['sniff']['auc']['select_score'])
    R = json.loads(best.read_text())
    fold, seed = R['fold'], R['seed']
    Z = np.load(OUT / 'eval' / f'{arm}_picks.npz')
    P = pd.read_parquet(PLAN / 'frames.parquet')
    E = P[P.role == 'eval'].reset_index(drop=True)
    V = pd.read_parquet(PLAN / 'videos.parquet').set_index('video')
    test = E.half.values == R['test_half']
    info = {}
    for kind in ('auc', 'top1'):
        val, pk = Z[f'{fold}_s{seed}_sniff_{kind}']
        val = np.where(test, val, -np.inf)
        top = np.argsort(-val, kind='stable')[:16]
        tiles = []
        for i in top:
            q, v = E.video[i], int(E.v[i])
            W, H = int(V.loc[q, 'W']), int(V.loc[q, 'H'])
            d = ROOT / V.loc[q, 'half']
            fr, _ = read_frames(d / f'{q}_t.seq', load_seek(d / f'{q}_t-seek.mat'), [v])
            pt, s = geom(W, H)
            c = int(pk[i])
            x = ((c % GRID) * CELL + CELL / 2) / s
            y = ((c // GRID) * CELL + CELL / 2) / s - pt
            g = np.pad(fr[v], 112, constant_values=128)
            x0, y0 = int(round(x)), int(round(y))
            crop = g[y0:y0 + 224, x0:x0 + 224]
            pil = Image.fromarray(crop).convert('RGB').resize((256, 256))
            dr = ImageDraw.Draw(pil)
            lab = next(b for b in VOCAB if E[b][i])
            ok = bool(E['sniff'][i])
            dr.rectangle([0, 0, 255, 255], outline=(0, 200, 0) if ok else (220, 0, 0), width=6)
            ph = (x - x0 + 112) * 256 / 224, (y - y0 + 112) * 256 / 224
            dr.rectangle([ph[0] - 9, ph[1] - 9, ph[0] + 9, ph[1] + 9], outline=(255, 255, 0), width=2)
            dr.text((8, 8), f'{lab} {val[i]:.2f}', fill=(255, 255, 0))
            dr.text((8, 236), f'{q[:22]} v{v}', fill=(255, 255, 0))
            tiles.append(pil)
        S = Image.new('RGB', (4 * 256, 4 * 256))
        for k, t in enumerate(tiles):
            S.paste(t, ((k % 4) * 256, (k // 4) * 256))
        nm = SH / f'sniff_{arm}_{kind}.jpg'
        S.save(nm, quality=88)
        e = R['main']['sniff'][kind]
        info[kind] = {'sheet': str(nm), 'fold': fold, 'seed': seed, 'neuron': e['neuron'],
                      'select_auc': R['main']['sniff']['auc']['select_score'], 'test_auc': e['test_auc'],
                      'test_top1': e['test_top1'], 'test_rate': e['test_rate'],
                      'top16_true_sniff': int(E['sniff'].values[top].sum()),
                      'top16_labels': [next(b for b in VOCAB if E[b][i]) for i in top]}
        log(f'{arm} {kind}: ' + json.dumps(info[kind]))
    (SH / f'sniff_{arm}.json').write_text(json.dumps(info, indent=1))


def main():
    SH.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ['mask', 'C1', 'C4']
    if 'mask' in which:
        mask_sheets()
    for a in ('C1', 'C2', 'C3', 'C4'):
        if a in which:
            sniff_sheet(a)


if __name__ == '__main__':
    main()
