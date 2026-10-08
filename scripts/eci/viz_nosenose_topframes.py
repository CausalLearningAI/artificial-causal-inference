"""
Top-10 evaluation frames of the best nose-nose neuron of three mice SAEs, each frame raw and with the neuron's per-patch
activation overlaid (the ECI atlas heat: src/eci/viz.py _overlay_rgb, turbo, alpha 0.65 * w^0.7, bilinear upsampling of
the 32 x 32 patch grid onto the 512 px frame, colour scale = 99th percentile of the positive patch activations of the
shown frames, as scripts/eci/interp_sheets.py).

Models (Spatial-SAE pilot, scripts/eci/spatial_sae_pilot.py; eval set = every 5th frame of the 144 annotated videos,
172,800 frames, none used in training; neuron = best nose-nose AUROC, max pooling, codes/align_best.json):
    matryoshka  results/vision/eci_spatial_sae/mice/matryoshka/sae.pt (seed 0)
    spatial     results/vision/eci_spatial_sae/mice/spatial/sae.pt (seed 0); code = [smooth S (768), innovation R (256)]
    deployed    dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448_s0/sae.pt (reference; trained on the eval videos)

Steps:
    select  (Slurm job) per model: frame-level max activation of the neuron from codes/<name>/codes_max.npy, greedy
            top-10 with at most one frame per video in any 10 s window (a frame is skipped if an already picked frame of
            the same video is < 50 frames = 10 s at 5 fps away); then re-encode the picked frames' foreground tokens
            (fg448 token store) with the checkpoint -> per-patch 32 x 32 activation maps; checks that the max over
            patches equals the stored frame-level code. -> OUT/topframes.npz + OUT/topframes.json
    render  (login node) -> OUT/<name>_top10.png, OUT/combined_top10.png, OUT/nosenose_top10.html

Usage:
    python scripts/eci/viz_nosenose_topframes.py select --stage /localhome/$USER/$SLURM_JOB_ID
    python scripts/eci/viz_nosenose_topframes.py render
"""
import argparse
import base64
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PILOT = REPO / 'results/vision/eci_spatial_sae/mice'
CODES = PILOT / 'codes'
OUT = PILOT / 'viz_nosenose_top10'
STORE = REPO / 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1'
ANN = REPO / 'dataset/mice/v1/annotations.csv'
DATASET = REPO / 'dataset'
GRID, PATCH_PX = 32, 16
WINDOW_FRAMES = 50  # 10 s at 5 fps
N_TOP = 10
MODELS = {  # name -> (checkpoint, label)
    'matryoshka': (PILOT / 'matryoshka/sae.pt', 'Matryoshka BatchTopK (pilot retrain, seed 0)'),
    'spatial': (PILOT / 'spatial/sae.pt', 'Spatial-SAE (pilot, seed 0)'),
    'deployed': (REPO / 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448_s0/sae.pt',
                 'Deployed fg448 Matryoshka SAE (reference)'),
}
log = lambda s: print(s, flush=True)  # noqa: E731


def best_neurons():
    best = json.loads((CODES / 'align_best.json').read_text())
    return {m: best[f'{m}|max|nose_nose'] for m in MODELS}


# ---------------------------------------------------------------------------------------------- select (job)
def pick_top(act, obs, fidx, n=N_TOP, window=WINDOW_FRAMES):
    """Greedy: descending activation, skip a frame within `window` frames of an already picked frame of its video."""
    order = np.argsort(-act, kind='stable')
    picked = []
    for i in order:
        if act[i] <= 0:
            break
        if any(obs[p] == obs[i] and abs(int(fidx[p]) - int(fidx[i])) < window for p in picked):
            continue
        picked.append(int(i))
        if len(picked) == n:
            break
    return np.array(picked)


def store_layout():
    """annotations.csv row -> (shard, first token, n tokens) of the fg448 token store."""
    lay = {}
    dirs = sorted(d for d in (STORE / 'shards').iterdir() if d.is_dir() and not d.name.endswith('.tmp'))
    for s, d in enumerate(dirs):
        z = np.load(d / 'frames.npz')
        n = z['n_fg'].astype(np.int64)
        st = np.r_[0, np.cumsum(n)[:-1]]
        for r, a, b in zip(z['rows'].astype(np.int64), st, n):
            lay[int(r)] = (s, int(a), int(b))
    dim = json.loads((dirs[0] / 'shard.json').read_text())['dim']
    sizes = [json.loads((d / 'shard.json').read_text())['n_tokens'] for d in dirs]
    return lay, dirs, dim, sizes


def frame_tokens(row, lay, dirs, dim, sizes):
    s, a, n = lay[row]
    tok = np.memmap(dirs[s] / 'tokens.f16', dtype=np.float16, mode='r', shape=(sizes[s], dim))[a:a + n]
    pos = np.memmap(dirs[s] / 'pos.i16', dtype=np.int16, mode='r', shape=(sizes[s],))[a:a + n]
    return np.array(tok, dtype=np.float32), np.array(pos, dtype=np.int64)


def cmd_select(args):
    import torch
    from src.eci.sae import load_sae
    from src.eci.spatial_sae import load_spatial, neighbor_index
    torch.set_num_threads(max(1, int(args.threads)))
    stage = Path(args.stage)
    stage.mkdir(parents=True, exist_ok=True)
    # stage the inputs read in bulk (codes columns, labels, annotations, checkpoints); the token store is read for
    # the 30 picked frames only (memmap slices)
    for m in MODELS:
        (stage / m).mkdir(exist_ok=True)
        shutil.copy(CODES / m / 'codes_max.npy', stage / m / 'codes_max.npy')
        shutil.copy(MODELS[m][0], stage / m / 'sae.pt')
    for f in ('rows.npy', 'labels.parquet'):
        shutil.copy(CODES / f, stage / f)
    shutil.copy(ANN, stage / 'annotations.csv')
    log('staged')
    rows = np.load(stage / 'rows.npy')
    lab = pd.read_parquet(stage / 'labels.parquet')
    assert (lab['row'].values == rows).all()
    ann = pd.read_csv(stage / 'annotations.csv', usecols=['observation_id', 'frame_idx', 'fps', 'frame_path',
                                                          'Y_nn', 'Y_np', 'Y_nt'])
    a = ann.iloc[rows]
    assert (a['observation_id'].values == lab['obs'].values).all()
    assert set(a['fps'].unique()) == {5.0}, a['fps'].unique()
    fidx = a['frame_idx'].values
    obs = lab['obs'].values
    best = best_neurons()
    lay, dirs, dim, sizes = store_layout()
    dev = torch.device('cpu')
    out, maps_all = {}, {}
    for m in MODELS:
        j = best[m]['auroc_neuron']
        act = np.asarray(np.load(stage / m / 'codes_max.npy', mmap_mode='r')[:, j], dtype=np.float32)
        top = pick_top(act, obs, fidx)
        # top-1% precision of THIS neuron over all eval frames (k = 1% of 172,800 = 1,728 frames; frames that do not
        # fire count as ranked last, so a rarely firing neuron is not scored on its few firing frames only)
        k1 = int(round(0.01 * len(act)))
        top1 = np.argsort(-act, kind='stable')[:k1]
        prec1 = float(lab['nose_nose'].values[top1].mean())
        ck = torch.load(stage / m / 'sae.pt', map_location='cpu', weights_only=False)
        spatial = ck.get('kind') == 'spatial_btk'
        if spatial:
            sae, norm, _ = load_spatial(stage / m / 'sae.pt', dev)
        else:
            sae, norm, _ = load_sae(stage / m / 'sae.pt', dev)
        maps = np.zeros((len(top), GRID, GRID), np.float32)
        recon = []
        with torch.no_grad():
            for t, i in enumerate(top):
                tok, pos = frame_tokens(int(rows[i]), lay, dirs, dim, sizes)
                x = norm(torch.from_numpy(tok))
                p = torch.from_numpy(pos)
                if spatial:
                    nbr = neighbor_index(torch.zeros_like(p), p, GRID, GRID, sae.offsets)
                    S, R = sae.encode(x, nbr)
                    z = torch.cat([S, R], 1)[:, j]
                else:
                    z = sae.encode(x, mode='threshold')[:, j]
                z = z.numpy()
                maps[t].reshape(-1)[pos] = z
                recon.append(float(z.max()))
        diff = np.abs(np.array(recon) - act[top])
        log(f'{m} neuron {j}: firing rate {(act > 0).mean():.4f}, top-1% precision {prec1:.4f} ({k1} frames, '
            f'{int((act[top1] > 0).sum())} firing; align_best.json prec_top1pct_at_auroc_neuron '
            f'{best[m]["prec_top1pct_at_auroc_neuron"]:.4f})')
        log(f'{m} neuron {j}: top-{len(top)} frame max {act[top].round(3).tolist()}; re-encoded patch max agrees to '
            f'{diff.max():.4f} (fp16 storage)')
        if diff.max() > 0.02 * max(1.0, act[top].max()):
            raise RuntimeError(f'{m}: re-encoded patch codes do not reproduce the stored frame max ({diff.max():.4f})')
        # how many eval frames fire at all, and the activation rank of the 10th picked frame
        frames = []
        for t, i in enumerate(top):
            r = a.iloc[i]
            am = int(np.argmax(maps[t]))
            frames.append({'eval_idx': int(i), 'row': int(rows[i]), 'obs': str(obs[i]), 'frame_idx': int(fidx[i]),
                           'frame_path': str(r['frame_path']), 'act': float(act[i]), 'act_patch_max': recon[t],
                           'argmax_rc': [am // GRID, am % GRID],
                           'Y_nn': bool(r['Y_nn'] > 0), 'Y_np': bool(r['Y_np'] > 0), 'Y_nt': bool(r['Y_nt'] > 0),
                           'nose_nose': bool(lab['nose_nose'].iloc[i]), 'nose_tail': bool(lab['nose_tail'].iloc[i]),
                           'any_contact': bool(lab['any_contact'].iloc[i])})
        out[m] = {'neuron': j, 'stats': best[m], 'label': MODELS[m][1], 'frames': frames,
                  'frac_firing': float((act > 0).mean()), 'prec_top1pct': prec1, 'n_top1pct': k1,
                  'n_firing_in_top1pct': int((act[top1] > 0).sum()), 'stream': (('smooth S' if j < sae.c_s else 'innovation R')
                                                                     if spatial else 'tokenwise'),
                  'max_ap_neuron': best[m]['max_ap_neuron']}
        maps_all[m] = maps
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / 'topframes.npz', **maps_all)
    (OUT / 'topframes.json').write_text(json.dumps({'window_frames': WINDOW_FRAMES, 'models': out}, indent=1))
    log(f'-> {OUT}')


# ---------------------------------------------------------------------------------------------- render (login)
def _font(size):
    from PIL import ImageFont
    for f in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _pair(fr, hm, vmax, cmap, size, border):
    """Raw | overlay, argmax patch outlined, coloured border -> PIL image."""
    from PIL import Image, ImageDraw
    from src.eci.viz import _overlay_rgb, frame_box
    img = np.asarray(Image.open(DATASET / fr['frame_path']).convert('RGB'))
    assert img.shape[:2] == (512, 512), img.shape
    ov = _overlay_rgb(img, hm, frame_box('fg448', img.shape[1], img.shape[0]), vmax, cmap)
    r, c = fr['argmax_rc']
    ims = []
    for k, arr in enumerate((img, ov)):
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        x0, y0 = c * PATCH_PX, r * PATCH_PX
        d.rectangle([x0 - 2, y0 - 2, x0 + PATCH_PX + 1, y0 + PATCH_PX + 1], outline=(255, 255, 255), width=2)
        d.rectangle([x0 - 4, y0 - 4, x0 + PATCH_PX + 3, y0 + PATCH_PX + 3], outline=(0, 0, 0), width=2)
        ims.append(im.resize((size, size), Image.LANCZOS))
    b, gap = 7, 4
    green, grey = (34, 160, 70), (150, 150, 150)
    out = Image.new('RGB', (2 * size + gap + 2 * b, size + 2 * b), green if fr['nose_nose'] else grey)
    out.paste(ims[0], (b, b))
    out.paste(ims[1], (b + size + gap, b))
    return out


def _yn(v):
    return 'YES' if v else 'no'


def panel(name, info, maps, size=260):
    import matplotlib
    from PIL import Image, ImageDraw
    cmap = matplotlib.colormaps['turbo']
    frames = info['frames']
    hv = np.concatenate([m[m > 0] for m in maps])
    vmax = float(np.quantile(hv, 0.99)) if len(hv) else 1.0
    st = info['stats']
    n_nn = sum(f['nose_nose'] for f in frames)
    ncol = 5
    nrow = int(np.ceil(len(frames) / ncol))
    pw, ph = 2 * size + 4 + 14, size + 14
    cap_h, pad, head_h = 64, 14, 92
    W = ncol * pw + (ncol + 1) * pad
    H = head_h + nrow * (ph + cap_h + pad) + pad
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    f_title, f_sub, f_cap = _font(22), _font(15), _font(14)
    extra = ''
    if name == 'spatial':
        extra = (f'  [{info["stream"]} latent; max-AP neuron is {info["max_ap_neuron"]} '
                 f'(AP {st["max_ap"]:.3f}), not shown]')
    d.text((pad, 8), f'{info["label"]}: neuron {info["neuron"]}{extra}', font=f_title, fill=(0, 0, 0))
    d.text((pad, 38), f'nose-nose (Y_nn or Y_np), max over patches, 172,800 held-out eval frames: AUROC {st["auroc"]:.3f}'
                      f'  AP {st["ap_at_auroc_neuron"]:.3f}  base rate {st["base_rate"]:.3f}  top-1% precision '
                      f'{info["prec_top1pct"]:.3f} ({info["n_top1pct"]:,} frames)  fires (> 0) on {100 * info["frac_firing"]:.1f}% '
                      f'of frames',
           font=f_sub, fill=(40, 40, 40))
    d.text((pad, 60), f'Top {len(frames)} frames (<= 1 per video per 10 s): {n_nn}/{len(frames)} true nose-nose.  '
                      f'Left raw, right per-patch activation (turbo, full colour = {vmax:.2f}); box = argmax patch; '
                      f'border green = nose-nose, grey = not.', font=f_sub, fill=(40, 40, 40))
    for k, (fr, hm) in enumerate(zip(frames, maps)):
        rr, cc = divmod(k, ncol)
        x = pad + cc * (pw + pad)
        y = head_h + rr * (ph + cap_h + pad)
        canvas.paste(_pair(fr, hm, vmax, cmap, size, 7), (x, y))
        sec = fr['frame_idx'] / 5.0
        lines = [f'#{k + 1}  {fr["obs"]}',
                 f'frame {fr["frame_idx"]} ({int(sec // 60)}:{sec % 60:04.1f})   act {fr["act"]:.3f}',
                 f'nose-nose {_yn(fr["nose_nose"])} (nn {int(fr["Y_nn"])}, np {int(fr["Y_np"])})   '
                 f'nose-tail {_yn(fr["nose_tail"])}   contact {_yn(fr["any_contact"])}']
        for li, s in enumerate(lines):
            d.text((x + 2, y + ph + 3 + 19 * li), s, font=f_cap,
                   fill=(20, 110, 45) if (li == 2 and fr['nose_nose']) else (30, 30, 30))
    return canvas, n_nn


def cmd_render(args):
    from PIL import Image
    meta = json.loads((OUT / 'topframes.json').read_text())
    z = np.load(OUT / 'topframes.npz')
    imgs, counts = {}, {}
    for m in MODELS:
        info = meta['models'][m]
        im, n_nn = panel(m, info, z[m])
        im.save(OUT / f'{m}_top10.png', optimize=True)
        imgs[m], counts[m] = im, n_nn
        log(f'{m} neuron {info["neuron"]}: {n_nn}/{len(info["frames"])} true nose-nose -> {OUT / f"{m}_top10.png"}')
    W = max(i.width for i in imgs.values())
    comb = Image.new('RGB', (W, sum(i.height for i in imgs.values()) + 6 * (len(imgs) - 1)), (200, 200, 200))
    y = 0
    for i in imgs.values():
        comb.paste(i, (0, y))
        y += i.height + 6
    comb.save(OUT / 'combined_top10.png', optimize=True)
    secs = []
    for m, im in imgs.items():
        buf = io.BytesIO()
        im.convert('RGB').save(buf, 'JPEG', quality=88)
        info = meta['models'][m]
        secs.append(f'<section><h2>{info["label"]}: neuron {info["neuron"]} &middot; {counts[m]}/10 true nose-nose'
                    f'</h2><img alt="{m} top-10 frames" src="data:image/jpeg;base64,'
                    f'{base64.b64encode(buf.getvalue()).decode()}"></section>')
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nose-nose neurons</title><style>
:root{{--bg:#fafaf8;--fg:#1d1d1f;--muted:#555}}
@media (prefers-color-scheme: dark){{:root{{--bg:#16171a;--fg:#e6e6e6;--muted:#aaa}}}}
body{{background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif;margin:0 auto;padding:16px;max-width:1500px}}
h1{{font-size:21px}} h2{{font-size:17px;margin:22px 0 6px}} p{{color:var(--muted);max-width:75em}}
img{{width:100%;height:auto;display:block;border-radius:4px}}
</style></head><body>
<h1>Nose-nose sniffing neurons of three mice SAEs: top-10 held-out frames</h1>
<p>Evaluation set: every 5th frame of the 144 annotated videos (172,800 frames), none used to train the pilot SAEs
(the deployed fg448 SAE did see these videos in training). Neuron = best nose-nose AUROC (max over patches) in
results/vision/eci_spatial_sae/mice/codes/align_best.json. Top 10 by the neuron's frame-level max activation, at most one
frame per video in any 10-second window. Each frame: raw (left) and the neuron's per-patch activation (right, ECI atlas
heat: turbo, 32x32 patch grid bilinearly upsampled; full colour = 99th percentile of the shown positive patch
activations). The box marks the argmax patch. Green border = annotated nose-nose (Y_nn or Y_np), grey = not.</p>
{''.join(secs)}
</body></html>"""
    (OUT / 'nosenose_top10.html').write_text(html)
    log(f'-> {OUT / "combined_top10.png"}, {OUT / "nosenose_top10.html"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['select', 'render'])
    p.add_argument('--stage', default=None)
    p.add_argument('--threads', default=8)
    args = p.parse_args()
    {'select': cmd_select, 'render': cmd_render}[args.cmd](args)


if __name__ == '__main__':
    main()
