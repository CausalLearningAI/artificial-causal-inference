"""Render example frames of one v2 mice neuron in both heat styles (patch / smooth) and check the marked patch.
Input: a cached heat file (scripts/eci/v2/run.py heat). Output: small WebP files. Run on Slurm."""
import io, sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.eci.viz import _overlay_rgb, peak_patch

f, out = Path(sys.argv[1]), Path(sys.argv[2])
z = np.load(f, allow_pickle=True)
heat, vmax, tl = z['heat'].astype(np.float32), float(z['vmax']), z['thumb_len']
th, o = z['thumbs'], np.r_[0, np.cumsum(tl)]
cmap = matplotlib.colormaps['turbo']
out.mkdir(parents=True, exist_ok=True)
for i in range(3):
    img = np.asarray(Image.open(io.BytesIO(th[o[i]:o[i + 1]].tobytes())).convert('RGB'))
    k = i * 5 + 2  # clip i, centre frame (the peak)
    hm = heat[k]
    pk = peak_patch(hm)
    assert pk is not None and hm[pk] == hm.max() and (hm.flat[np.argmax(hm)] == hm.max())
    n = img.shape[0]
    res = {s: _overlay_rgb(img, hm, (0., 0., float(n)), vmax, cmap, style=s) for s in ('patch', 'smooth')}
    # check the marked patch: white outline pixels bound exactly the argmax cell
    c = n / hm.shape[0]
    y0, x0 = int(round(pk[0] * c)), int(round(pk[1] * c))
    y1, x1 = int(round((pk[0] + 1) * c)) - 1, int(round((pk[1] + 1) * c)) - 1
    p = res['patch']
    ok = (p[y0, x0:x1 + 1] == 255).all() and (p[y1, x0:x1 + 1] == 255).all() and (p[y0 - 1, max(x0 - 1, 0):x1 + 1] == 0).all() if y0 > 0 else True
    print(f'clip {i} frame {k}: argmax patch {pk} value {hm[pk]:.4f} = map max {hm.max():.4f}; outline drawn at that cell: {ok}', flush=True)
    assert ok
    both = Image.fromarray(np.concatenate([img, res['patch'], res['smooth']], 1))
    both = both.resize((both.width * 2, both.height * 2), Image.NEAREST)
    both.save(out / f'n{int(f.stem[1:]):04d}_clip{i}_orig_patch_smooth.webp', quality=85)
