"""Clip-peak sheets for chosen discovered neurons + decoder change share (d2: share of the decoder row's squared norm on
the z_t - z_(t-10) half). Usage: python sheets.py <dom> <neuron> [<neuron> ...]  (dom as in diag.py DOMS)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag as D  # noqa: E402

sys.path.insert(0, str(D.REPO))
from src.eci.sae import load_sae  # noqa: E402

dom, neurons = sys.argv[1], [int(x) for x in sys.argv[2:]]
d = D.DOMS[dom]
codes = d['eci'] / 'codes'
cfg = json.loads((codes / (d['sae'] + '_fnorm') / 'config.json').read_text())
bgdir, kind = Path(cfg['backgrounds']), d['kind']
sae, _, _ = load_sae(d['eci'] / 'sae' / d['sae'] / 'sae.pt', 'cpu')
W = sae.W_dec.detach().double().numpy()
if W.shape[0] != 512:
    W = W.T
cs = (W[:, 1024:] ** 2).sum(1) / (W ** 2).sum(1)
work = D.RES / d['res'] / (d['sae'] + '_fnorm') / '_cache' / 'explorer'
picks = json.loads((work / 'picks.json').read_text())
fp, by = picks['frame_path'], picks['by_key']['max']
tiles, out = [], {'change_share_all_median': round(float(np.median(cs)), 3)}
for j in neurons:
    out[str(j)] = {'change_share': round(float(cs[j]), 3)}
    if str(j) not in by:
        continue
    pz = np.load(sorted((work / 'patch').glob(f'*{j:04d}.npz'))[0])
    m = pz['all__frame__top'].astype(np.float32)
    for k, c in enumerate(by[str(j)]['clips']['all']['frame']['top'][:6]):
        rel = fp[str(c['rows'][0])]
        G = m.shape[-1]
        _, (iy, ix) = D.peak(m[k:k + 1])
        dm = D.dark_map(rel, bgdir, c['obs'], kind)
        im = Image.open(D.jpath(rel)).convert('RGB')
        x, y = (ix + 0.5) * 512 / G, (iy + 0.5) * 512 / G
        a = np.asarray(im).astype(np.float32)
        a[dm] = a[dm] * 0.6 + np.array([0, 255, 255]) * 0.4
        im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        dr.rectangle([x - 512 / G / 2, y - 512 / G / 2, x + 512 / G / 2, y + 512 / G / 2], outline=(255, 0, 0), width=2)
        crop = im.crop((int(x) - 64, int(y) - 64, int(x) + 64, int(y) + 64)).resize((256, 256), Image.LANCZOS)
        dc = ImageDraw.Draw(crop)
        dc.rectangle([0, 0, 256, 16], fill=(0, 0, 0))
        dc.text((3, 2), f'n{j} {c["obs"]} cs={cs[j]:.2f}', fill=(255, 255, 255))
        tiles.append(crop)
sh = Image.new('RGB', (6 * 256, ((len(tiles) + 5) // 6) * 256))
for i, t in enumerate(tiles):
    sh.paste(t, ((i % 6) * 256, (i // 6) * 256))
sh.save(D.OUT / 'overlays' / f'zoom_{dom}.jpg', quality=85)
(D.OUT / f'zoom_{dom}.json').write_text(json.dumps(out, indent=1))
print(out)
