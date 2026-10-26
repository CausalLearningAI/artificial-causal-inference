"""
ECI v2.0 input layout of the MICE V2 recordings (series 'v2' of configs/eci_v2/mice_v20.json), built with the rules of
mice v1 (no new rules).

Steps (scripts/eci/v2/mice_v2_setup.sh runs each in Slurm)
    plan   (CPU) scripts/eci/rollout_r4.py cmd_plan, mice branch, on data/mice/v2: row space = dataset/mice/v2/
           annotations.csv (216 videos contiguous, frame_idx 0..n-1, the 5 fps grid of the PPCI frames, source frame
           6 k + 2 at 30 fps), source = data/mice/source/<observation_file> (start_frame 0 asserted), ffprobe checks,
           held-out roles (rng 0 permutation of the videos, n_held = round(n / 6) even: half calib, half eval), prepass
           bursts (rng [1, i]), shards balanced by frames -> results/vision/eci_rollout_r4/mice_v2/plan.json.
           'annotated' = Y_nn present (cmd_reshard's rule; v2 has no behaviour labels). Layout dir dataset/mice/v2/eci
           (codes/, sae/, annotations.csv -> ../annotations.csv, the row space, W_* columns included).
    check  (CPU) (a) source offset: decoded source frames 6 k + o (o = 0..5) at 512 vs the dataset JPG of frame k,
           3 videos x 3 frames; (b) frames k = 0 and n - 1 of 2 videos through src/eci/v2/pipeline.Series (cpu, no
           encoder) with the tight-mask pixels overlaid -> results/vision/eci_v2_0/mice/v2_setup_sheet.jpg + .json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import rollout_r4 as RR  # noqa: E402

PLAN = REPO / 'results/vision/eci_rollout_r4/mice_v2/plan.json'
LAYOUT = REPO / 'dataset/mice/v2/eci'
log = RR.log


def cfg_v2():
    return dict(RR.DOMS['mice'], ann='dataset/mice/v2/annotations.csv', exp='data/mice/v2/experiment.csv',
                bg='dataset/mice/v2/eci/fg448/background', eci='dataset/mice/v2/eci', ref_nfg=None, c1=None,
                deployed=None, n_shards=7)


def cmd_plan(a):
    """= rollout_r4.cmd_plan, dom 'mice' branch, with the v2 paths (code copied, not changed)."""
    c = cfg_v2()
    t0 = time.time()
    A, R = RR.obs_ranges_of(REPO / c['ann'])
    for o, (lo, hi) in R.items():
        fi = A['frame_idx'].values[lo:hi]
        if not (fi == np.arange(hi - lo)).all():
            raise SystemExit(f'mice v2 {o}: frame_idx is not 0..n-1')
    E = pd.read_csv(REPO / c['exp'])
    vids = list(R)
    if vids != list(E['observation_id']):
        raise SystemExit('mice v2: annotations.csv video order != experiment.csv order')
    Ei = E.set_index('observation_id')
    videos, lay = [], 0
    for o in vids:
        lo, hi = R[o]
        path = REPO / 'data/mice/source' / Ei.loc[o, 'observation_file']
        assert int(Ei.loc[o, 'start_frame']) == 0
        assert (int(Ei.loc[o, 'end_frame']) - int(Ei.loc[o, 'start_frame'])) == c['step'] * (hi - lo), o
        videos.append({'obs': o, 'src_lo': lo, 'n': hi - lo, 'lay_lo': lay, 'path': str(path), 'hflip': 0,
                       'offset': int(c['offset'])})
        lay += hi - lo
    with ThreadPoolExecutor(16) as ex_:
        probes = list(ex_.map(lambda v: RR.ffprobe(v['path']), videos))
    bad = []
    for v, p in zip(videos, probes):
        v['probe'] = p
        need = RR.voff(c, v) + c['step'] * (v['n'] - 1) + 1
        have = p['nb_frames'] if p['nb_frames'] else int(round(p['duration'] * c['fps']))
        v['src_frames'] = have
        if have < need or max(p['w'], p['h']) not in range(c['native'] - 8, c['native'] + 9):
            bad.append((v['obs'], have, need, p['w'], p['h']))
    if bad:
        raise SystemExit(f'mice v2: {len(bad)} videos too short / wrong size, e.g. {bad[:5]}')
    rng = np.random.default_rng(0)
    order = rng.permutation(len(videos))
    n_held = max(6, int(round(len(videos) / 6)))
    n_held += n_held % 2
    for j, i in enumerate(order):
        videos[i]['role'] = 'calib' if j < n_held // 2 else ('eval' if j < n_held else 'train')
    n_b = int(np.ceil(RR.FRAMES_TARGET / (4 * len(videos))))
    for i, v in enumerate(videos):
        g = np.random.default_rng([1, i])
        span = v['n'] - 26
        u = g.random(n_b)
        b = np.floor((np.arange(n_b) + u) * span / n_b).astype(int)
        assert b.min() >= 0 and b.max() + 25 < v['n'] and (np.diff(b) >= 0).all()
        v['bursts'] = b.tolist()
    lab = set(pd.read_csv(REPO / c['ann'], usecols=['observation_id', 'Y_nn']).query('Y_nn.notna()')['observation_id'])
    for v in videos:
        v['annotated'] = v['obs'] in lab
    n = np.array([v['n'] for v in videos])
    cuts = np.searchsorted(np.cumsum(n), np.linspace(0, n.sum(), c['n_shards'] + 1)[1:-1])
    shards = [s.tolist() for s in np.split(np.arange(len(videos)), cuts + 1) if len(s)]
    G = c['S'] // 16
    plan = {'domain': 'mice', 'series': 'v2', 'cfg': c, 'G': G, 'n_videos': len(videos), 'n_frames': int(n.sum()),
            'patches_per_body': c['body'] * c['S'] / c['native'] / 16, 'bursts_per_video': n_b,
            'n_prepass_frames': 4 * n_b * len(videos),
            'roles': {r: sum(v['role'] == r for v in videos) for r in ('train', 'calib', 'eval')},
            'shards': shards, 'shard_frames': [int(n[s].sum()) for s in shards], 'videos': videos,
            'note': 'built by scripts/eci/v2/mice_v2_setup.py plan = rollout_r4.cmd_plan mice branch on data/mice/v2; '
                    'patches_per_body uses the v1 body length (same rig, same 2064 px source)'}
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=1, default=str))
    for d in ('codes', 'sae'):
        (LAYOUT / d).mkdir(parents=True, exist_ok=True)
    if not (LAYOUT / 'annotations.csv').exists():
        os.symlink('../annotations.csv', LAYOUT / 'annotations.csv')
    log(f'mice v2: {len(videos)} videos, {n.sum():,} frames, {n_b} bursts/video, roles {plan["roles"]}, '
        f'shards {plan["shard_frames"]}, codecs {sorted({p["codec"] for p in probes})}, '
        f'sizes {sorted({(p["w"], p["h"]) for p in probes})}, rates {sorted({p["rate"] for p in probes})} '
        f'({time.time() - t0:.0f}s)')


def cmd_check(a):
    import torch
    from PIL import Image, ImageDraw
    from src.eci.v2.pipeline import Series
    from src.eci.v2.video import decode
    torch.set_num_threads(8)
    P = json.loads(PLAN.read_text())
    ann = pd.read_csv(REPO / 'dataset/mice/v2/annotations.csv', usecols=['frame_path'])['frame_path'].values
    out = {}
    # (a) source offset vs the dataset JPGs (512 px)
    g = np.random.default_rng(0)
    off = []
    for i in g.choice(len(P['videos']), 3, replace=False):
        v = P['videos'][int(i)]
        for k in np.sort(g.choice(np.arange(100, v['n'] - 100), 3, replace=False)):
            jpg = np.asarray(Image.open(REPO / 'dataset' / ann[v['src_lo'] + k]).convert('RGB'), np.float32)
            mae = [float(np.abs(decode(v['path'], 6 * int(k) + o, 1, 6, 512, 0, 30, threads=4)[0].astype(np.float32)
                                - jpg).mean()) for o in range(6)]
            off.append({'obs': v['obs'], 'k': int(k), 'mae_by_offset': [round(m, 3) for m in mae],
                        'best': int(np.argmin(mae))})
            log(f'offset {v["obs"]} k {k}: {off[-1]["mae_by_offset"]} best {off[-1]["best"]}')
    out['offset_check'] = off
    # (b) Series frames + tight mask, k = 0 and n - 1 of 2 videos
    c = json.loads((REPO / 'configs/eci_v2/mice_v20.json').read_text())
    sc = next(s for s in c['series'] if s['name'] == 'v2')
    se = Series(sc, torch.device('cpu'), load_encoder=False)
    tiles, rec = [], []
    for v in (se.videos[0], se.videos[-1]):
        for k in (0, v['n'] - 1):
            fr = se.run_job(**se.job(v, k, 1))
            x, fd = se.frames(v, fr, [k])
            pix = se.mask.pixels(x, v['src_lo'] + np.array([k]))[0].numpy()
            im = x[0].permute(1, 2, 0).numpy().copy()
            im[pix] = (0.4 * im[pix] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
            t = Image.fromarray(im).resize((512, 512))
            ImageDraw.Draw(t).text((6, 6), f'{v["obs"]} k {k} rot {se.rot.get(v["obs"], 0)} kept {int((fd >= 0.25).sum())}',
                                   fill=(255, 255, 0))
            tiles.append(t)
            rec.append({'obs': v['obs'], 'k': int(k), 'rot90': int(se.rot.get(v['obs'], 0)), 'shape': list(x.shape),
                        'mask_px': int(pix.sum()), 'kept_patches': int((fd >= 0.25).sum())})
            log(f'series frame {rec[-1]}')
    sheet = Image.new('RGB', (1024, 1024))
    for j, t in enumerate(tiles):
        sheet.paste(t, ((j % 2) * 512, (j // 2) * 512))
    o = REPO / 'results/vision/eci_v2_0/mice/v2_setup_sheet.jpg'
    o.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(o, quality=85)
    out['series_frames'] = rec
    o.with_suffix('.json').write_text(json.dumps(out, indent=1))
    log(f'-> {o}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('plan', 'check'))
    a = ap.parse_args()
    {'plan': cmd_plan, 'check': cmd_check}[a.step](a)
