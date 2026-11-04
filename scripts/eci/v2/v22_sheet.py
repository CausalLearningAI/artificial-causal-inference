"""Top-12 frames of the SG-selected nose-nose and nose-tail neurons of one mice model (CPU job), v21_extended_sheet style.

Neuron = the design (a) pick (selection on all SG frames, rule given) from the v22 tag pickle. Frames ranked over the
held-out NON-SG frames only (no frame used for selection appears), by the rank readout (v2: P(on), ties by the raw
code; v1: raw code), at most 2 per video, >= 25 frames apart. Heat map = the neuron's per-patch code under the model's
readout: max -> native per-token code; box3 -> native code after the 3x3 box mean; coarse -> code of the 2x2-pooled
tokens (32 x 32 grid); ms -> the scale (native or pooled) that holds the frame max. v1 = stored DINOv2 tokens (32 x 32).
Identity: map max vs the stored / recomputed frame code (in the json).
usage: v22_sheet.py --model KEY --rule bal --label NAME
"""
import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v22_core as C  # noqa: E402
import v21_extended_sheet as XS  # noqa: E402
import v22_encode as EN  # noqa: E402

T, REPO, OUT22, log = C.T, C.REPO, C.OUT22, C.log


def v2_maps(obs, k, sae, norm, j, mode):
    import torch
    from v21_freeze import feats_np
    Vd = T.Video('mice', 'v1', obs)
    i = int(np.flatnonzero(Vd.k == k)[0])
    W = Vd.window(i, i + 1)
    sel = ((W['flag'] & 1) > 0) & (W['fi'] == i)
    x, cd = feats_np(W['tok'], W['fi'], W['pos'], sel)
    G = Vd.G
    f = torch.from_numpy(x + cd).float()
    pos = W['pos'][sel].astype(np.int64)
    out = {}
    with torch.no_grad():
        if mode in ('max', 'box3', 'ms'):
            c = sae.encode(norm(f), mode='threshold')
            if mode == 'box3':
                c = EN.box3(c, np.zeros(len(pos), np.int64), pos, G)
            m = np.zeros(G * G, np.float32)
            m[pos] = c[:, j].numpy()
            out['native'] = m.reshape(G, G)
        if mode in ('coarse', 'ms'):
            fp, _, ppos, _ = EN.pool2(f, np.zeros(len(pos), np.int64), pos, G)
            cp = sae.encode(norm(fp), mode='threshold')[:, j].numpy()
            g2 = (G + 1) // 2
            m = np.zeros(g2 * g2, np.float32)
            m[ppos] = cp
            out['coarse'] = m.reshape(g2, g2)
    S = int(np.load(T.STORE / 'mice' / T.VER / 'videos' / 'v1' / obs / 'index.npz')['S'])
    best = max(out, key=lambda q: out[q].max())
    return out[best], S


def main():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    import eval as EV
    from src.eci.sae import load_sae
    from src.eci.v2 import sae as SA
    from src.eci.foreground import ODOR_ROT90
    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--rule', default='bal')
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    key, rule = a.model, a.rule
    label = a.label or f'{key}_{rule}'
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    local = Path(os.environ.get('LOCAL_DIR', '/tmp')) / f'v22_sheet_{label}'
    local.mkdir(parents=True, exist_ok=True)
    F, _ = C.mice_table()
    oc = pd.read_csv(REPO / 'dataset/mice/v1/eci/odor_corner.csv').set_index('observation_id')['odor_corner']
    rot = {o: int(ODOR_ROT90[oc[o]]) for o in F.obs.unique()}
    ann_fp = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv', usecols=['frame_path'])['frame_path'].values
    plans = XS.plans_of('mice')
    tag = pickle.load(open(OUT22 / 'tags' / f'{key}.pkl', 'rb'))
    d = tag['sets'][f'sg2non|{rule}'][0]
    X, R, Pn = C.load(key)
    is_v1 = key.startswith('v1_')
    base, mode = (key.split('@') + ['max'])[:2]
    if is_v1:
        sae = load_sae(REPO / EV.V1SAE['mice'], torch.device('cpu'))[:2] if key == 'v1_s0' else None
        assert sae is not None, 'v1 maps only for v1_s0 (the stored v1 SAE)'
        sys.path.insert(0, str(REPO / 'scripts/eci/v2_prob'))
        import study as SY
        v1_tok = SY.BH.Store('mice', 'v1', 'dense', 768)
        tix = pd.Series(np.arange(v1_tok.nf), index=pd.MultiIndex.from_arrays([v1_tok.F.obs.values, v1_tok.F.k.values.astype(int)]))
    else:
        sae = SA.load(EN.sae_path('mice', base), torch.device('cpu'))
    ev = np.flatnonzero(F.nonsg.values)
    picks = {}
    for b in C.BEH:
        j = int(d['picks'][b])
        y = F[f'y_{b}'].values[ev]
        got = XS.pick_frames(X[ev, j].astype(np.float64), R[ev, j].astype(np.float64), y, F.obs.values[ev], F.k.values[ev], gap=25)
        fr = []
        for i in got:
            r = int(ev[i])
            obs, k = str(F.obs.values[r]), int(F.k.values[r])
            if is_v1:
                fi = int(tix[(obs, k)])
                a_, b_ = int(v1_tok.starts[fi]), int(v1_tok.starts[fi + 1])
                with torch.no_grad():
                    cc = sae[0].encode(sae[1](torch.from_numpy(v1_tok.tok(a_, b_)).float()), mode='threshold')[:, j].numpy()
                m = np.zeros(32 * 32, np.float32)
                m[v1_tok.pos[a_:b_]] = cc
                m = m.reshape(32, 32)
                img = np.asarray(Image.open(REPO / 'dataset' / ann_fp[int(F.row.values[r])]).convert('RGB'))
                img = np.ascontiguousarray(np.rot90(img, rot.get(obs, 0)))
            else:
                m, S = v2_maps(obs, k, *sae, j, 'coarse' if base.startswith('VP2') else ('max' if mode == 'top10' else mode))
                img = XS.decode_v2('mice', 'v1', obs, k, S, plans, rot)
            np.save(local / f'{b}_{r}.npy', m)
            Image.fromarray(img).save(local / f'{b}_{r}.png')
            fr.append(dict(row=r, obs=obs, k=k, score=float(X[r, j]), code=float(R[r, j]), y=bool(y[i]),
                           cls=int(F.cls.values[r]), map_max=float(m.max()), identity_abs_diff=abs(float(m.max()) - float(R[r, j]))))
        picks[b] = {'neuron': j, 'frames': fr}
        log(f'{label} {b}: neuron {j}, true label in {sum(f["y"] for f in fr)}/{len(fr)} top frames')
    fig, axs = plt.subplots(4, 6, figsize=(6 * 2.0, 4 * 2.45), gridspec_kw={'hspace': 0.6, 'wspace': 0.06})
    for ax in axs.ravel():
        ax.axis('off')
    other = {'nt': 'nose-nose', 'nose_nose': 'nose-tail'}
    for bi, b in enumerate(C.BEH):
        pk = picks[b]
        for n_, f in enumerate(pk['frames']):
            ax = axs[2 * bi + n_ // 6, n_ % 6]
            img = np.asarray(Image.open(local / f'{b}_{f["row"]}.png'))
            m = np.load(local / f'{b}_{f["row"]}.npy')
            H, Wd = img.shape[:2]
            G = m.shape[0]
            q = int(np.argmax(m))
            pr, pc = q // G, q % G
            ps = H / G
            cy, cx = (pr + 0.5) * ps, (pc + 0.5) * ps
            hw = 0.36 * H / 2
            y0, x0 = int(np.clip(cy - hw, 0, H - 2 * hw)), int(np.clip(cx - hw, 0, Wd - 2 * hw))
            ax.imshow(img, extent=(0, Wd, H, 0))
            ax.imshow(np.ma.masked_less_equal(m, 0), extent=(0, Wd, H, 0), cmap='inferno', alpha=0.45,
                      interpolation='nearest', vmin=0, vmax=max(float(m.max()), 1e-6))
            ax.add_patch(Rectangle((pc * ps, pr * ps), ps, ps, fill=False, ec='cyan', lw=1.4))
            ax.set_xlim(x0, x0 + 2 * hw)
            ax.set_ylim(y0 + 2 * hw, y0)
            lab = 'yes' if f['y'] else ('NO (' + other[b] + ')' if f['cls'] in (1, 2) else 'NO')
            ax.set_title(f'{lab}\n{f["obs"]} k{f["k"]}', fontsize=6.5, color='black' if f['y'] else 'red')
        npos = sum(f['y'] for f in pk['frames'])
        axs[2 * bi, 0].text(-0.02, 1.32, f'{C.NAME[b]}: neuron #{pk["neuron"]}, true label in {npos}/{len(pk["frames"])} top frames',
                            transform=axs[2 * bi, 0].transAxes, fontsize=9, weight='bold')
    fig.suptitle(f'mice {label}: top frames of the SG-selected neurons (rule {rule}), ranked over the held-out non-SG pools\n'
                 'cyan = max patch, heat = per-patch code; caption = true label (red NO = absent; "(nose-tail)" = the sibling label)',
                 fontsize=9.5, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01)
    (OUT22 / 'sheets').mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT22 / 'sheets' / f'sheet_mice_{label}.png', dpi=100)
    ident = max(f['identity_abs_diff'] for p in picks.values() for f in p['frames'])
    (OUT22 / 'sheets' / f'sheet_mice_{label}.json').write_text(json.dumps({'model': key, 'rule': rule, 'identity_max_abs_diff': ident,
                                                                           'picks': picks}, indent=1))
    shutil.rmtree(local, ignore_errors=True)
    log(f'sheet {label}: identity (map max vs frame code) {ident:.4f}')


if __name__ == '__main__':
    main()
