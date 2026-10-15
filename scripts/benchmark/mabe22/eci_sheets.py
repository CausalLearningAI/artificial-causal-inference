"""
MABe22 mouse ECI evaluation: contact sheets of the top frames of the best oral_oral / oral_genital neurons (CPU job).

For each arm given, SAE seed 0, split seed 0: the neuron selected on half 0 (by AUROC and by honest top-1%) is shown on
its 16 highest-valued frames of half 1 (frames never used to select it). Tile = the 224 frame at x2, red box = the patch
holding the frame's max for that neuron; border green = the label is positive in that frame, grey = negative; text =
clip, label index, value. Frames decoded locally (ffmpeg611), video frame = label index + offset.
Outputs: OUT/sheets/<arm>_<label>_<pick>.jpg and OUT/sheets/index.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import OUT, PLAN, TOK, decode_clip, log, offset, stage_dir  # noqa: E402
from eci_train import ARMS, halves, shards  # noqa: E402

LABS = ['oral_oral_contact', 'oral_genital_contact']


def positions(arm, lens):
    """per frame, the grid position of each stored token (list of arrays) and the grid size."""
    tok = ARMS[arm].get('eval_tok', ARMS[arm]['tok'])
    if tok == 'all448':
        return None, 32
    name, G = ('pos224', 16) if tok == 'm224' else ('pos448', 32)
    pos = np.concatenate([np.load(TOK / f'{k}_{name}.npy') for k in shards('eval')])
    st = np.r_[0, np.cumsum(lens)]
    assert st[-1] == len(pos)
    return (pos, st), G


def main():
    from PIL import Image, ImageDraw
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['A0'])
    args = ap.parse_args()
    stage = stage_dir()
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    h = halves(E, 0)
    off = offset()
    od = OUT / 'sheets'
    od.mkdir(parents=True, exist_ok=True)
    index = json.loads((od / 'index.json').read_text()) if (od / 'index.json').exists() else {}
    for arm in args.arms:
        R = json.loads((OUT / 'eval' / f'{arm}_s0.json').read_text())['m4096']['split0']
        codes = np.load(OUT / 'codes' / f'{arm}_s0_codes.npy', mmap_mode='r')
        argt = np.load(OUT / 'codes' / f'{arm}_s0_argtok.npy', mmap_mode='r')
        lens = np.load(OUT / 'codes' / f'{arm}_lens.npy')
        P, G = positions(arm, lens)
        ps = 224 // G
        for b in LABS:
            for pick in ('auc', 'top1'):
                e = R[b]['dirs'][0][pick]  # selected on half 0, tested on half 1
                j = e['neuron']
                rows = np.flatnonzero(h == 1)
                v = np.asarray(codes[rows, j], np.float32)
                top = rows[np.argsort(-v, kind='stable')[:16]]
                tiles = []
                by_clip = {}
                for f in top:
                    by_clip.setdefault(E['clip'][f], []).append(f)
                img = {}
                for c, fs in by_clip.items():
                    fr = decode_clip('mouse_submission', c, stage, 'rgb24')
                    for f in fs:
                        img[f] = fr[E.i[f] + off]
                for f in top:
                    im = Image.fromarray(img[f]).resize((448, 448), Image.BICUBIC)
                    d = ImageDraw.Draw(im)
                    a = int(argt[f, j])
                    p = a if P is None else int(P[0][P[1][f] + a]) if lens[f] > 0 else -1
                    if p >= 0 and codes[f, j] > 0:
                        r, cc = divmod(p, G)
                        d.rectangle([cc * ps * 2, r * ps * 2, (cc + 1) * ps * 2, (r + 1) * ps * 2], outline=(255, 0, 0),
                                    width=3)
                    pos_ = bool(E[b][f])
                    d.rectangle([0, 0, 447, 447], outline=(0, 220, 0) if pos_ else (128, 128, 128), width=6)
                    d.text((8, 8), f'{E["clip"][f][:10]} i{E.i[f]} v{float(codes[f, j]):.2f} {"POS" if pos_ else "neg"}',
                           fill=(255, 255, 0))
                    tiles.append(im)
                W = Image.new('RGB', (4 * 448, 4 * 448))
                for k, t in enumerate(tiles):
                    W.paste(t, ((k % 4) * 448, (k // 4) * 448))
                path = od / f'{arm}_{b.replace("_contact", "")}_{pick}.jpg'
                W.save(path, quality=88)
                npos = int(E[b].values[top].sum())
                index[path.name] = {'arm': arm, 'label': b, 'pick': pick, 'neuron': j, 'select_score_half0': e['select_score'],
                                    'test_auc': e['test_auc'], 'test_ap': e['test_ap'], 'test_top1': e['test_top1'],
                                    'test_rate': e['test_rate'], 'positives_in_16': npos}
                log(f'{path.name}: neuron {j}, {npos}/16 positive, test top1 {e["test_top1"]:.3f} rate {e["test_rate"]:.3f}')
    (od / 'index.json').write_text(json.dumps(index, indent=1))


if __name__ == '__main__':
    main()
