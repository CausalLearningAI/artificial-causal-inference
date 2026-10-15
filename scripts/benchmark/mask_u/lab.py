"""
Mask-U test (rules: scripts/benchmark/eci_mask_u.sh), LAB part: DESCRIPTIVE only (test set; no SAE, no behaviour
labels, cannot change the decision). A few frames of ants v2 + v3, mice v1 and frogs v1, the current per-domain rule
vs rule U (src/eci/mask_u.py, used as implemented). Tracker points / SLEAP nodes only MEASURE the masks.

Videos (rng = np.random.default_rng(SEED), SEED = 0, each list sorted before the draw): ants 4 v2 + 4 v3 (10-min
layout dataset/ants/eci, 3000 frames per video), mice 6 (dataset/mice/v1), frogs 6 (dataset/frogs/eci, 1 h videos).
Per video (its annotations.csv rows lo..hi-1): U sample = 30 frames np.linspace(lo, hi - 1, 30).round() (U time blocks
from their frame_idx); targets = 16 frames lo + floor((j + 0.5) (hi - lo) / 16), j = 0..15. DINOv2-base 448
last-layer tokens (foreground.load_encoder_fg / FrameDatasetFG / encode_batch) feed both rules.
Current rules: foreground.RULES 'ants' / 'fg448' / 'frogs' with the existing per-video backgrounds
(FgBackgrounds(<eci>/fg448/background, obs_rows(annotations.csv), rule).mask(tokens, grey, rows)). Rule 'frogs'
uses the frame's SLEAP nodes inside the mask (near_px); U uses no tracker points.

Measures per domain and rule: kept cells per frame (mean, fraction of the 1024 grid, min / max), frames with 0 kept;
overlap = |cur & U| / |cur| and |cur & U| / |U| (pooled over frames), the per-frame Jaccard mean, and the share of
frames where > 10% / > 25% of the current mask lies outside U (per-frame counts in per_video).
    ants   tracker CSV (v2: dataset/ants/v2/tracking/<obs>.csv; v3: the ARCHIVED 10-min tracking
           archive/10min/ants/v3/dataset/ants/v3/tracking/<obs>.csv, because dataset/ants/v3 is being rebuilt to 30 min
           and its tracking/ dir is absent / being re-spliced; frames 0-2999 of dataset/ants/v3/frames/full are the
           pre-existing JPGs, extend_log frames_full_copied = 6000), rows reindexed by frame_idx, 512-px x / y, cell =
           floor(y / 16) * 32 + floor(x / 16) (as scripts/eci/mask_test.py). Per ant role (focal, yellow, blue
           centroids, gap-filled by the tracker): strict recall = centroid cell kept, lenient = some cell of its 3 x 3
           kept; frames with any ant missed (strict / lenient); colour-dot recall (mark_yellow, mark_blue);
           near2 = fraction of kept cells within 2 cells (Chebyshev) of some centroid cell
    frogs  the SLEAP nodes of the background files (22 nodes, 512 px, NaN = not predicted; x = [..., 0], y = [..., 1]
           as foreground.near_nodes): node recall = valid nodes in a kept cell / valid nodes, frames with nodes but
           none kept; hull precision = kept cells within 1 cell of the nodes' convex hull (common.hull_cells).
           CAVEAT: the current frogs rule uses these nodes (near_px 16), so its recall is not independent of them
    mice   no tracker: kept cells only
Outputs: results/vision/mask_u/maskqa/lab.json, results/vision/mask_u/sheets/lab_<domain>_<k>.jpg (current = red tint,
U = green tint, side by side; blue circles = tracker centroids / SLEAP nodes).
Run (GPU): sbatch ... --export=ALL,PY=lab,EXTRA="" scripts/benchmark/mask_u/job.sh (see eci_mask_u.sh sub()).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from common import OUT, hull_cells, log, recall_cells, sheet, stage_dir  # noqa: E402

SEED = 0
N_SAMPLE, N_TARGET = 30, 16
DATA = REPO / 'dataset'
DOM = {
    'ants': {'ann': DATA / 'ants/eci/annotations.csv', 'bg': DATA / 'ants/eci/fg448/background', 'rule': 'ants',
             'pick': {'v2': 4, 'v3': 4}},
    'mice': {'ann': DATA / 'mice/v1/annotations.csv', 'bg': DATA / 'mice/v1/eci/fg448/background', 'rule': 'fg448',
             'pick': 6},
    'frogs': {'ann': DATA / 'frogs/eci/annotations.csv', 'bg': DATA / 'frogs/eci/fg448/background', 'rule': 'frogs',
              'pick': 6},
}
TRACK = {'v2': DATA / 'ants/v2/tracking', 'v3': REPO / 'archive/10min/ants/v3/dataset/ants/v3/tracking'}
ANTS = ('focal', 'yellow', 'blue')
DOTS = ('mark_yellow', 'mark_blue')
GRID, CELL = 32, 16


def cells_of(x, y):
    """512-px x, y (any shape, NaN allowed) -> (cell index, valid)."""
    ok = np.isfinite(x) & np.isfinite(y)
    xx, yy = np.where(ok, x, 0), np.where(ok, y, 0)
    c = np.clip(np.floor(yy / CELL), 0, GRID - 1) * GRID + np.clip(np.floor(xx / CELL), 0, GRID - 1)
    return c.astype(np.int64), ok


def dilate_np(m, r=1):
    """(n, 1024) bool -> dilated by a (2r+1)^2 square on the grid."""
    from scipy import ndimage as ndi
    st = np.zeros((1, 2 * r + 1, 2 * r + 1), bool)
    st[0] = True
    return ndi.binary_dilation(m.reshape(-1, GRID, GRID), st).reshape(len(m), -1)


def pick_videos(name, A, ranges):
    rng = np.random.default_rng(SEED)
    p = DOM[name]['pick']
    if isinstance(p, dict):  # ants: per experiment
        exp = A.groupby('observation_id', sort=False)['experiment'].first()
        out = []
        for ex, n in p.items():
            out += list(rng.choice(sorted(exp.index[exp == ex]), n, replace=False))
        return out
    return list(rng.choice(sorted(ranges), p, replace=False))


def encode(paths, processor, model, dev):
    import torch
    from src.eci.foreground import FrameDatasetFG, encode_batch
    dl = torch.utils.data.DataLoader(FrameDatasetFG(paths, processor), batch_size=24, num_workers=6)
    tok, grey = [], []
    for pix, g in dl:
        tok.append(encode_batch(model, pix, dev))
        grey.append(g)
    return torch.cat(tok), torch.cat(grey).to(dev)


def ants_points(obs, ex, fidx):
    t = pd.read_csv(TRACK[ex] / f'{obs}.csv').set_index('frame_idx').reindex(fidx)
    xy = {a: (t[f'{a}_x'].values.astype(float), t[f'{a}_y'].values.astype(float)) for a in ANTS + DOTS}
    mx = max(np.nanmax(np.abs(np.r_[v[0], v[1]])) for v in xy.values())
    if not np.isfinite(mx) or mx >= 512:
        raise RuntimeError(f'{obs}: tracker coordinates outside the 512-px frame (max {mx})')
    return xy, mx


def summarise(R, extra_animals=None):
    """R: {rule: {'m': (n, 1024) bool, ...}} -> dict of per-rule numbers."""
    cur, U = R['cur']['m'], R['U']['m']
    inter = (cur & U).sum(1)
    union = (cur | U).sum(1)
    out_frac = (cur & ~U).sum(1) / np.maximum(cur.sum(1), 1)
    out = {'overlap': {'frac_cur_cells_in_U': float(inter.sum() / max(cur.sum(), 1)),
                       'frames_cur_outside_U_gt_10pct': float((out_frac > 0.10).mean()),
                       'frames_cur_outside_U_gt_25pct': float((out_frac > 0.25).mean()),
                       'max_frac_cur_outside_U': float(out_frac.max()),
                       'frac_U_cells_in_cur': float(inter.sum() / max(U.sum(), 1)),
                       'jaccard_per_frame_mean': float(np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0)))}}
    for nm, d in R.items():
        k = d['m'].sum(1)
        s = {'frames': int(len(k)), 'kept_mean': float(k.mean()), 'kept_frac_grid': float(k.mean() / 1024),
             'kept_min': int(k.min()), 'kept_max': int(k.max()), 'frames_0_kept': int((k == 0).sum())}
        for key, v in d.items():
            if key != 'm':
                s[key] = v
        out[nm] = s
    return out


def ants_measures(m, P):
    """m (n, 1024) bool, P dict of point arrays -> recall dict."""
    cc, ok = P['cent_cells'], P['cent_ok']  # (n, 3)
    rec_s, miss_s = recall_cells(m, cc[..., None], ok[..., None])
    rec_l, miss_l = recall_cells(dilate_np(m, 1), cc[..., None], ok[..., None])
    rd, _ = recall_cells(m, P['dot_cells'][..., None], P['dot_ok'][..., None])
    near = np.zeros_like(m)
    for j in range(3):
        one = np.zeros_like(m)
        one[np.arange(len(m))[ok[:, j]], cc[ok[:, j], j]] = True
        near |= dilate_np(one, 2)
    kept = m.sum(1)
    return {'centroid_recall_strict': {a: float(np.nanmean(rec_s[:, j])) for j, a in enumerate(ANTS)},
            'centroid_recall_strict_all': float(np.nanmean(rec_s)),
            'centroid_recall_lenient_3x3': {a: float(np.nanmean(rec_l[:, j])) for j, a in enumerate(ANTS)},
            'missed_frac_strict': {a: float(miss_s[:, j].mean()) for j, a in enumerate(ANTS)},
            'frames_any_ant_missed_strict': float(miss_s.any(1).mean()),
            'frames_any_ant_missed_lenient': float(miss_l.any(1).mean()),
            'dot_recall_strict': {a: float(np.nanmean(rd[:, j])) for j, a in enumerate(DOTS)},
            'near2_frac_of_kept': float(np.mean(((m & near).sum(1) / np.maximum(kept, 1))[kept > 0])) if (kept > 0).any() else None,
            'frames_with_3_centroids': float(ok.all(1).mean())}


def frogs_measures(m, nodes):
    """m (n, 1024), nodes (n, K, 2) 512 px -> recall dict."""
    cc, ok = cells_of(nodes[..., 0], nodes[..., 1])
    rec, miss = recall_cells(m, cc[:, None], ok[:, None])
    xyc = np.where(ok[..., None], nodes / CELL, 0)[:, None]
    hull = hull_cells(xyc, ok[:, None])
    kept = m.sum(1)
    has = ok.any(1)
    return {'node_recall': float(np.nanmean(rec[:, 0])),
            'node_recall_pooled': float((m[np.arange(len(m))[:, None], cc] & ok).sum() / max(ok.sum(), 1)),
            'missed_frac_frames_with_nodes': float(miss[has, 0].mean()) if has.any() else None,
            'frames_with_nodes': int(has.sum()), 'valid_nodes_per_frame': float(ok.sum(1).mean()),
            'hull_precision': float(np.mean(((m & hull).sum(1) / np.maximum(kept, 1))[(kept > 0) & has]))
            if ((kept > 0) & has).any() else None}


def run_domain(name, processor, model, dev, stage, n_sheet):
    import torch
    from src.eci.foreground import RULES, FgBackgrounds, obs_rows
    from src.eci.mask_u import UMasker
    from PIL import Image
    cfg = DOM[name]
    ann = stage / f'{name}_annotations.csv'
    shutil.copyfile(cfg['ann'], ann)  # one NFS read, then local
    cols = ['observation_id', 'frame_idx', 'frame_path'] + (['experiment'] if name == 'ants' else [])
    A = pd.read_csv(ann, usecols=cols)
    ranges = obs_rows(ann)
    bgs = FgBackgrounds(cfg['bg'], ranges, RULES[cfg['rule']], dev)
    vids = pick_videos(name, A, ranges)
    log(f'{name}: videos {vids}')
    acc = {'cur': [], 'U': [], 'U_feat': [], 'U_pix': []}
    pts, tiles, per_video = [], [], {}
    for o in vids:
        lo, hi = ranges[o]
        s_rows = np.linspace(lo, hi - 1, N_SAMPLE).round().astype(np.int64)
        t_rows = (lo + np.floor((np.arange(N_TARGET) + 0.5) * (hi - lo) / N_TARGET)).astype(np.int64)
        rows = np.r_[s_rows, t_rows]
        paths = [str(DATA / p) for p in A.frame_path.values[rows]]
        tok, grey = encode(paths, processor, model, dev)
        fidx = A.frame_idx.values
        cur, _ = bgs.mask(tok[N_SAMPLE:], grey[N_SAMPLE:], t_rows)
        um = UMasker(dev).fit(grey[:N_SAMPLE], tok[:N_SAMPLE], t_s=fidx[s_rows])
        feat, pc, U = um.parts(grey[N_SAMPLE:], tok[N_SAMPLE:], fidx[t_rows])
        cur, U, feat, pc = (x.cpu().numpy() for x in (cur, U, feat, pc))
        acc['cur'].append(cur)
        acc['U'].append(U)
        acc['U_feat'].append(feat)
        acc['U_pix'].append(pc)
        info = {'U': um.info, 'target_frame_idx': fidx[t_rows].tolist(), 'sample_frame_idx': fidx[s_rows].tolist()}
        info['kept_cur'] = cur.sum(1).tolist()
        info['kept_U'] = U.sum(1).tolist()
        info['cur_cells_outside_U'] = (cur & ~U).sum(1).tolist()
        b = bgs.get(o)
        if 'thr' in b:
            info['cur_thr'] = float(b['thr'])
        kp = [None] * N_TARGET
        if name == 'ants':
            ex = A.experiment.values[lo]
            xy, mx = ants_points(o, ex, fidx[t_rows])
            info['experiment'], info['tracker_xy_max'] = ex, float(mx)
            info['tracking_csv'] = str(TRACK[ex] / f'{o}.csv')
            c3 = [cells_of(*xy[a]) for a in ANTS]
            d2 = [cells_of(*xy[a]) for a in DOTS]
            P = {'cent_cells': np.stack([c for c, _ in c3], 1), 'cent_ok': np.stack([k for _, k in c3], 1),
                 'dot_cells': np.stack([c for c, _ in d2], 1), 'dot_ok': np.stack([k for _, k in d2], 1), 'ex': ex}
            pts.append(P)
            kp = [np.array([[xy[a][0][j], xy[a][1][j]] for a in ANTS if np.isfinite(xy[a][0][j])]) for j in range(N_TARGET)]
            info['cur_recall'] = ants_measures(cur, P)['centroid_recall_strict']
            info['U_recall'] = ants_measures(U, P)['centroid_recall_strict']
        elif name == 'frogs':
            nodes = b['nodes'][torch.from_numpy(t_rows - b['lo']).to(b['nodes'].device)].float().cpu().numpy()
            assert len(b['nodes']) == hi - lo, (o, len(b['nodes']), hi - lo)
            pts.append(nodes)
            kp = [n[np.isfinite(n).all(1)] for n in nodes]
            info['cur_recall'] = frogs_measures(cur, nodes)['node_recall']
            info['U_recall'] = frogs_measures(U, nodes)['node_recall']
        per_video[o] = info
        for j in (3, 11)[:n_sheet]:
            with Image.open(paths[N_SAMPLE + j]) as im:
                rgb = np.asarray(im.convert('RGB'))
            tiles.append((rgb, {'cur': cur[j], 'U': U[j]}, f'{o} f{fidx[t_rows[j]]}', kp[j]))
        log(f'{name} {o}: cur kept {cur.sum(1).mean():.1f} | U kept {U.sum(1).mean():.1f} (token cue '
            f'{feat.sum(1).mean():.1f}, pixel core {pc.sum(1).mean():.1f}) | recall cur {info.get("cur_recall")} '
            f'U {info.get("U_recall")} | U {json.dumps(um.info)}')
    M = {k: np.concatenate(v) for k, v in acc.items()}
    R = {'cur': {'m': M['cur']}, 'U': {'m': M['U'], 'token_cue_cells_mean': float(M['U_feat'].sum(1).mean()),
                                        'pixel_core_cells_mean': float(M['U_pix'].sum(1).mean())}}
    res = {'videos': vids, 'rule_current': cfg['rule'], 'background_dir': str(cfg['bg'])}
    if name == 'ants':
        groups = {'all': np.ones(len(M['cur']), bool)}
        ex_f = np.concatenate([[P['ex']] * N_TARGET for P in pts])
        for ex in ('v2', 'v3'):
            groups[ex] = ex_f == ex
        Pall = {k: np.concatenate([P[k] for P in pts]) for k in ('cent_cells', 'cent_ok', 'dot_cells', 'dot_ok')}
        for g, sel in groups.items():
            Rg = {nm: {'m': d['m'][sel]} for nm, d in R.items()}
            for nm in Rg:
                Rg[nm].update(ants_measures(Rg[nm]['m'], {k: v[sel] for k, v in Pall.items()}))
            if g == 'all':
                Rg['U'].update({k: v for k, v in R['U'].items() if k != 'm'})
            res[g] = summarise(Rg)
    elif name == 'frogs':
        nodes = np.concatenate(pts)
        for nm in R:
            R[nm].update(frogs_measures(R[nm]['m'], nodes))
        res['all'] = summarise(R)
    else:
        res['all'] = summarise(R)
    for k in range(0, len(tiles), 8):
        sheet(tiles[k:k + 8], OUT / 'sheets' / f'lab_{name}_{k // 8}.jpg', ncol=4)
    res['sheets'] = [str(OUT / 'sheets' / f'lab_{name}_{k // 8}.jpg') for k in range(0, len(tiles), 8)]
    return res, per_video


def main():
    import torch
    from src.eci.foreground import load_encoder_fg
    from src.eci.mask_u import U_RULE
    ap = argparse.ArgumentParser()
    ap.add_argument('--domains', nargs='+', default=['ants', 'mice', 'frogs'])
    args = ap.parse_args()
    dev = torch.device('cuda')
    stage = stage_dir()
    _, processor, model = load_encoder_fg('dinov2_base', dev)
    out = {'rule_U': U_RULE, 'seed': SEED, 'n_sample': N_SAMPLE, 'n_target': N_TARGET, 'summary': {}, 'per_video': {}}
    for d in args.domains:
        res, pv = run_domain(d, processor, model, dev, stage, 2)
        out['summary'][d], out['per_video'][d] = res, pv
        log(f'{d}: ' + json.dumps({k: v for k, v in res.items() if k not in ('videos',)}, indent=1, default=float))
    (OUT / 'maskqa').mkdir(parents=True, exist_ok=True)
    (OUT / 'maskqa' / 'lab.json').write_text(json.dumps(out, indent=1, default=float))
    log(f'wrote {OUT / "maskqa" / "lab.json"}')


if __name__ == '__main__':
    main()
