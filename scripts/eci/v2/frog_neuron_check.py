"""
Where on the frog does an ECI v2 frogs neuron fire? (stored data only: full-pass token store + SLEAP poses)

For each juvenile video (configs/eci_v2/frogs.json series 'juv') and each neuron: fraction of 5 fps frames with a
non-zero raw frame code; on those frames the PEAK token (stored argmax) -> its centre in canonical px -> the nearest
SLEAP node of that frame (dataset/frogs/v1/sleap/<obs>.npz, source px; x mirrored for hflip videos, and Left_/Right_
labels swapped there, so 'R' always means the CANONICAL right = the mutant side of FoxP1 / En1 frogs) -> body part:
head (eyes), trunk (Heart_Center, Tail_Stem), R/L forelimb (Shoulder, Elbow, Wrist, Hand), R/L hindlimb (Hip, Knee,
Ankle, Foot, toe). Also the side of the peak token relative to the body axis (Tail_Stem -> eyes midpoint): canonical
right vs left.

Contact sheet for --sheet-neuron: 3 top frames (raw frame code, >= 5 s apart) from each of the top 4 FoxP1 and top 4 WT
videos (by peak raw code) = 24 tiles; each tile = 192 px canonical crop centred on the peak token with the real
per-token activation map (stored token codes) in magenta, the peak token outlined, SLEAP forelimb nodes (R cyan,
L yellow) as dots.

Usage (Slurm CPU): sbatch scripts/eci/v2/frog_neuron_check.sh
Output: results/vision/eci_v2/fig/frog_neuron_check.json, frog_n<j>_sheet.jpg
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2')
OUT = REPO / 'results/vision/eci_v2/fig'
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731

PART = {'Right_Eye': 'head', 'Left_Eye': 'head', 'Heart_Center': 'trunk', 'Tail_Stem': 'trunk'}
for side in ('Right', 'Left'):
    s = side[0]
    for p in ('Hand', 'Wrist', 'Elbow', 'Shoulder'):
        PART[f'{side}_{p}'] = f'{s} forelimb'
    for p in ('Hip', 'Knee', 'Ankle', 'Foot', 'toe'):
        PART[f'{side}_{p}'] = f'{s} hindlimb'
PARTS = ['head', 'trunk', 'R forelimb', 'L forelimb', 'R hindlimb', 'L hindlimb']


def canon_names(names, hflip):
    """SLEAP node names -> canonical (mirrored videos: Left <-> Right)."""
    if not hflip:
        return list(names)
    return [n.replace('Left_', 'TMP_').replace('Right_', 'Left_').replace('TMP_', 'Right_') for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--neurons', default='52,368,25,61,278,1694,124,57,1968,34,74,383')
    ap.add_argument('--sheet-neuron', type=int, default=52)
    ap.add_argument('--max-dist', type=float, default=24.0, help='px: peak farther than this from every node -> "off"')
    args = ap.parse_args()
    js = [int(x) for x in args.neurons.split(',')]
    plan = json.loads((REPO / 'results/vision/eci_rollout_r4/frogs/plan.json').read_text())
    exp = pd.read_csv(REPO / 'data/frogs/v1/experiment.csv').set_index('observation_id')
    res, tops = {j: {} for j in js}, []
    W800 = 800
    for v in plan['videos']:
        obs = v['obs']
        grp = exp.loc[obs, 'group']
        hflip = int(v['hflip'])
        z = np.load(BEE / 'frogs/videos/juv' / f'{obs}.npz')
        G = int(z['G'])
        cm = z['codes_max'][:, js].astype(np.float32)
        am = z['argmax'][:, js].astype(np.int64)
        P = np.load(REPO / 'dataset/frogs/v1/sleap' / f'{obs}.npz')
        nodes = P['nodes'].astype(np.float32).copy()
        if hflip:
            nodes[..., 0] = (W800 - 1) - nodes[..., 0]
        names = canon_names([str(s) for s in P['node_names']], hflip)
        part = np.array([PART[n] for n in names])
        n = min(len(cm), len(nodes))
        ie = [names.index('Right_Eye'), names.index('Left_Eye')]
        its = names.index('Tail_Stem')
        for q, j in enumerate(js):
            act = np.flatnonzero(cm[:n, q] > 0)
            rec = dict(group=grp, hflip=hflip, n=int(n), frac_active=float(len(act) / n),
                       mean_raw=float(cm[:n, q].mean()))
            if len(act):
                p = am[act, q]
                c = np.stack([(p % G) * 16 + 8, (p // G) * 16 + 8], 1).astype(np.float32)
                nd = nodes[act]  # (a, 22, 2)
                d = np.linalg.norm(nd - c[:, None], axis=2)
                d = np.where(np.isfinite(d), d, np.inf)
                k = d.argmin(1)
                dmin = d[np.arange(len(k)), k]
                lab = np.where(dmin <= args.max_dist, part[k], 'off')
                cnt = {pp: int((lab == pp).sum()) for pp in PARTS + ['off']}
                head = nd[:, ie].mean(1)
                ax_ = head - nd[:, its]
                rel = c - nd[:, its]
                cross = ax_[:, 0] * rel[:, 1] - ax_[:, 1] * rel[:, 0]  # image coords (y down): > 0 -> frog's right
                ok = np.isfinite(cross)
                rec.update(parts=cnt, n_active=int(len(act)), median_node_dist=float(np.median(dmin[np.isfinite(dmin)])) if np.isfinite(dmin).any() else None,
                           right_share=float((cross[ok] > 0).mean()) if ok.any() else None)
            res[j][obs] = rec
            if j == args.sheet_neuron and len(act):
                order = np.argsort(-cm[:n, q])
                pick = []
                for f in order:
                    if all(abs(f - g) >= 25 for g in pick):
                        pick.append(int(f))
                    if len(pick) == 3:
                        break
                tops.append(dict(obs=obs, group=grp, peak=float(cm[order[0], q]), frames=pick))
        log(f'{obs} ({grp}) n52 frac_active {res[js[0]][obs]["frac_active"]:.3f} parts {res[js[0]][obs].get("parts")}')
    # ---- group summaries
    summ = {}
    for j in js:
        s_ = {}
        for g in ('FoxP1', 'WT', 'En1'):
            R = [r for r in res[j].values() if r['group'] == g]
            tot = {pp: sum(r.get('parts', {}).get(pp, 0) for r in R) for pp in PARTS + ['off']}
            N = max(1, sum(tot.values()))
            rs = [r['right_share'] for r in R if r.get('right_share') is not None]
            s_[g] = dict(n_videos=len(R), median_frac_active=float(np.median([r['frac_active'] for r in R])),
                         part_share={pp: round(tot[pp] / N, 3) for pp in tot},
                         median_right_share=float(np.median(rs)) if rs else None,
                         mean_raw=float(np.mean([r['mean_raw'] for r in R])))
        summ[j] = s_
        log(f'neuron {j}: ' + json.dumps(s_))
    (OUT / 'frog_neuron_check.json').write_text(json.dumps({'summary': summ, 'per_video': res,
                                                             'sheet_tops': tops}, indent=1))
    sheet(args.sheet_neuron, tops, exp, plan)


def sheet(j, tops, exp, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import torch
    from src.eci.v2.pipeline import Series
    sys.path.insert(0, str(REPO / 'scripts/eci'))
    from fig_pipeline_v2 import TokenCodes
    cfg = next(s for s in json.loads((REPO / 'configs/eci_v2/frogs.json').read_text())['series'] if s['name'] == 'juv')
    Se = Series(cfg, torch.device('cpu'), load_encoder=False)
    V = {v['obs']: v for v in plan['videos']}
    chosen = []
    for g in ('FoxP1', 'WT'):
        t = sorted([x for x in tops if x['group'] == g], key=lambda x: -x['peak'])[:4]
        chosen += t
    fig, axs = plt.subplots(4, 6, figsize=(12, 8.6))
    tile = 0
    for x in chosen:
        v = V[x['obs']]
        T = TokenCodes(BEE / 'frogs/videos/juv' / f'{x["obs"]}.npz')
        P = np.load(REPO / 'dataset/frogs/v1/sleap' / f'{x["obs"]}.npz')
        names = canon_names([str(s) for s in P['node_names']], int(v['hflip']))
        for k in x['frames']:
            fr = Se.run_job(**Se.job(v, k, 1))
            xx, _ = Se.frames(v, fr, np.array([k]))
            im = xx[0].permute(1, 2, 0).numpy()
            h = T.heat(k, j).reshape(T.G, T.G)
            pk = int(np.argmax(h)); py, px = divmod(pk, T.G)
            cy = int(np.clip(py * 16 + 8 - 96, 0, 800 - 192)); cx = int(np.clip(px * 16 + 8 - 96, 0, 800 - 192))
            ax = axs.flat[tile]
            ax.imshow(im[cy:cy + 192, cx:cx + 192])
            H = np.kron(h / max(h.max(), 1e-6), np.ones((16, 16)))[cy:cy + 192, cx:cx + 192]
            rgba = np.zeros(H.shape + (4,)); rgba[..., 0] = 1; rgba[..., 2] = 1; rgba[..., 3] = 0.55 * H
            ax.imshow(rgba)
            ax.add_patch(Rectangle((px * 16 - cx, py * 16 - cy), 16, 16, fill=False, ec='yellow', lw=1.0))
            nd = P['nodes'][k].astype(np.float32).copy()
            if int(v['hflip']):
                nd[:, 0] = 799 - nd[:, 0]
            for i, nm in enumerate(names):
                if 'Hand' in nm or 'Wrist' in nm or 'Elbow' in nm or 'Shoulder' in nm:
                    ax.plot(nd[i, 0] - cx, nd[i, 1] - cy, 'o', ms=2.5, color='cyan' if nm.startswith('Right') else 'yellow')
            ms = exp.loc[x['obs'], 'mutant_side']
            ax.set_title(f'{x["obs"]} [{x["group"]}{", mut " + str(ms) if isinstance(ms, str) else ""}]\n'
                         f'f{k}  code {float(T.cm[k, j]):.1f}', fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            tile += 1
    for a in axs.flat[tile:]:
        a.axis('off')
    fig.suptitle(f'frogs v2 neuron {j}: top frames (raw frame code), 3 per video, top 4 FoxP1 + top 4 WT videos. '
                 f'Canonical frames (mutant side mirrored to the right). Magenta = per-token activation, yellow box = '
                 f'peak token; dots = SLEAP forelimb nodes (cyan canonical R, yellow canonical L)', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f'frog_n{j}_sheet.jpg', dpi=110)
    log(f'sheet: {OUT / f"frog_n{j}_sheet.jpg"} ({tile} tiles)')


if __name__ == '__main__':
    main()
