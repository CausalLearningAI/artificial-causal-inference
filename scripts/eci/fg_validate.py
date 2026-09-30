"""
Visual validation of the foreground rule (src/eci/foreground.py) on the background-sample
frames stored by scripts/eci/fg_background.py (their distances and dark cue are saved, so
no GPU is needed). Each PNG: frame | foreground overlay (feature cue red, dark-only cue
blue, dilation ring yellow) | distance map; plus per-video panel of pix_bg, n_ok and
median distance.

--domain (default mice, src/eci/domain.py) picks annotations.csv and the default dirs; --base-rule
(default the domain rule, src/eci/foreground.py RULES) is the rule that --rule JSON overrides.

Usage:
    python scripts/eci/fg_validate.py --pick obs:idx,obs:idx ... --out-dir <dir>
    python scripts/eci/fg_validate.py --stats            # foreground fraction per frame, all videos
    python scripts/eci/fg_validate.py --domain ants --stats --rule '{"dark_abs": 70}'
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.domain import DOMAINS, get_domain  # noqa: E402
from src.eci.foreground import GRID, PATCH_PX, RULES, foreground_parts, video_threshold  # noqa: E402


def masks(z, rule):
    key = rule['background']
    dist = torch.from_numpy(z[f'dist_{key}'].astype(np.float32))
    dark = torch.from_numpy(z['dark'].astype(np.float32))
    thr = video_threshold(dist.numpy(), dark.numpy(), rule)
    feat, dk, full = foreground_parts(dist, dark, thr, rule)
    return dist.numpy(), feat.numpy(), dk.numpy(), full.numpy(), thr


def render(bg_dir, ds, paths, obs, i, rule, out, title=''):
    z = np.load(bg_dir / f'{obs}.npz')
    dist, feat, dk, full, thr = masks(z, rule)
    row = int(z['rows'][i])
    img = np.asarray(Image.open(ds / paths[row]).convert('RGB'))
    up = lambda m: np.kron(m.reshape(GRID, GRID), np.ones((PATCH_PX, PATCH_PX)))  # noqa: E731
    ov = img.astype(np.float32) / 255
    col = np.zeros_like(ov)
    f, d, a = up(feat[i]) > 0, up(dk[i] & ~feat[i]) > 0, up(full[i] & ~(feat[i] | dk[i])) > 0
    col[f] = [1, 0, 0]; col[d] = [0, 0.4, 1]; col[a] = [1, 0.9, 0]
    sel = f | d | a
    ov[sel] = 0.55 * ov[sel] + 0.45 * col[sel]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.3))
    ax[0].imshow(img); ax[0].set_title(f'{obs} row {row}')
    ax[1].imshow(ov); ax[1].set_title(f'fg {full[i].mean():.1%} (red feat, blue dark-only, yellow dilation)')
    im = ax[2].imshow(dist[i].reshape(GRID, GRID), vmin=0, vmax=max(3 * thr, 0.05), cmap='magma')
    ax[2].set_title(f'cos dist to {rule["background"]}, thr {thr:.3f}')
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    for a_ in ax:
        a_.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=70)
    plt.close(fig)


def video_panel(bg_dir, obs, rule, out):
    z = np.load(bg_dir / f'{obs}.npz')
    dist, feat, dk, full, thr = masks(z, rule)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.3))
    ax[0].imshow(z['pix_bg'], cmap='gray', vmin=0, vmax=255); ax[0].set_title('pix_bg (0.98 quantile)')
    im = ax[1].imshow(z['n_ok'].reshape(GRID, GRID), cmap='viridis'); ax[1].set_title('n_ok frames for bg_masked')
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    im = ax[2].imshow(full.mean(0).reshape(GRID, GRID), cmap='viridis', vmin=0, vmax=1)
    ax[2].set_title('fraction of frames foreground'); plt.colorbar(im, ax=ax[2], fraction=0.046)
    ax[3].hist(dist.ravel(), bins=200, log=True); ax[3].axvline(thr, color='r'); ax[3].set_title('cos dist, all patches')
    for a_ in ax[:3]:
        a_.axis('off')
    fig.suptitle(obs); fig.tight_layout(); fig.savefig(out, dpi=70); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', default='mice', choices=DOMAINS)
    p.add_argument('--bg-dir', default=None, help='default <dataset dir>/<domain eci dir>/fg448/background')
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--out-dir', default=None, help='default <dataset dir>/<domain eci dir>/fg448/validation')
    p.add_argument('--pick', default='', help='obs:i[:title],...  (i = index among the n_bg sample frames)')
    p.add_argument('--panels', default='', help='comma list of obs for per-video panels')
    p.add_argument('--base-rule', default=None, choices=sorted(RULES), help='default: the domain rule (mice fg448)')
    p.add_argument('--rule', default='{}', help='JSON overrides of the base rule')
    p.add_argument('--stats', action='store_true')
    args = p.parse_args()
    dom = get_domain(args.domain)
    rule = {**RULES[args.base_rule or dom.fg_rule], **json.loads(args.rule)}
    ds = Path(args.dataset_dir)
    bg_dir = Path(args.bg_dir) if args.bg_dir else ds / dom.eci_rel / 'fg448/background'
    out = Path(args.out_dir) if args.out_dir else ds / dom.eci_rel / 'fg448/validation'
    out.mkdir(parents=True, exist_ok=True)
    paths = pd.read_csv(ds / dom.ann_rel, usecols=['frame_path'])['frame_path'].values
    for item in filter(None, args.pick.split(',')):
        obs, i, *t = item.split(':')
        render(bg_dir, ds, paths, obs, int(i), rule, out / f'mask_{obs}_{int(i):03d}.png', ' '.join(t))
    for obs in filter(None, args.panels.split(',')):
        video_panel(bg_dir, obs, rule, out / f'video_{obs}.png')
    if args.stats:
        fr, per_video = [], {}
        for f in sorted(bg_dir.glob('*.npz')):
            if f.name.endswith('.tmp.npz'):
                continue
            z = np.load(f)
            _, feat, dk, full, thr = masks(z, rule)
            ff = full.mean(1)
            fr.append(ff)
            per_video[f.stem] = {'thr': thr, 'fg_median': float(np.median(ff)), 'fg_feat_only': float(feat.mean()),
                                 'fg_dark_only': float((dk & ~feat).mean()), 'frac_frames_fg_lt_1pct': float((ff < 0.01).mean())}
        fr = np.concatenate(fr)
        res = {'rule': rule, 'n_videos': len(per_video), 'n_frames': len(fr),
               'fg_frac_per_frame': {q: float(np.percentile(fr, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
               'fg_frac_mean': float(fr.mean()), 'per_video': per_video}
        (out / 'fg_stats.json').write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != 'per_video'}, indent=1))
        pv = pd.DataFrame(per_video).T
        print(pv.describe().to_string())
        print(pv.sort_values('fg_median').head(5).to_string()); print(pv.sort_values('fg_median').tail(5).to_string())


if __name__ == '__main__':
    main()
