"""Demo of the single-colour 'alpha' heat style: per-clip rows (orig | alpha | alpha+outline | turbo patch) and a
top-frames grid. Input: cached heat npz; output dir. Run on Slurm."""
import io, sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.eci.viz import _overlay_rgb

f, out = Path(sys.argv[1]), Path(sys.argv[2])
z = np.load(f, allow_pickle=True)
heat, vmax, tl = z['heat'].astype(np.float32), float(z['vmax']), z['thumb_len']
th, o = z['thumbs'], np.r_[0, np.cumsum(tl)]
cmap = matplotlib.colormaps['turbo']
out.mkdir(parents=True, exist_ok=True)
tiles = []
for i in range(3):
    img = np.asarray(Image.open(io.BytesIO(th[o[i]:o[i + 1]].tobytes())).convert('RGB'))
    hm, n = heat[i * 5 + 2], img.shape[0]
    b = (0., 0., float(n))
    a0 = _overlay_rgb(img, hm, b, vmax, cmap, style='alpha', mark=False)
    a1 = _overlay_rgb(img, hm, b, vmax, cmap, style='alpha', mark=True)
    tp = _overlay_rgb(img, hm, b, vmax, cmap, style='patch')
    row = Image.fromarray(np.concatenate([img, a0, a1, tp], 1))
    row.resize((row.width * 2, row.height * 2), Image.NEAREST).save(out / f'alpha_clip{i}.webp', quality=85)
    tiles.append(a0)
g = Image.fromarray(np.concatenate(tiles, 1))
g.resize((g.width * 2, g.height * 2), Image.NEAREST).save(out / 'alpha_topframes_grid.webp', quality=85)
print('done', [t.shape for t in tiles], flush=True)
