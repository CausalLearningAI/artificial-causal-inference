"""Top-activating frames of the selected neuron, v1 (v1_s0) | v2 (V_s1001), one PNG per domain (CPU job).

Neuron = the modal AUROC pick over the 12 cross-fitted directions (v21_ladder.score2 'best_neuron') of that seed on the
pooled frame table (v21_extended tag pickles). Frames = all eval frames of the cell (both halves; this is an
illustration, not a metric), ranked by the ladder's readout (v1 raw frame-max code; V P(on), ties broken by V's raw
frame-max code), at most 2 per video and >= GAP frames apart. Per-patch map = the neuron's per-token SAE code:
  V  : stored DINOv3 token window (eci_tokens v2.1_m5), input x + (x_{t+1} - x_{t-1}) / 2 on the mask v5 tokens of frame t,
       image = the source frame decoded at the store's side S (ffmpeg 6.1.1, the store's geometry; mice turned by odour corner)
  v1 : mice stored DINOv2 tokens (v2_prob/mice/v1/dense); ants re-encoded from the jpg (FgEncoder + FgBackgrounds, CPU),
       image = the 512 px jpg (mice turned by odour corner)
Identity: max over tokens of the recomputed map vs the stored frame code (reported in sheet_<dom>.json).
usage: v21_extended_sheet.py --dom D      (LOCAL_DIR staging for the decoded frames)
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v21_extended as X  # noqa: E402

L, REPO, T = X.L, X.REPO, X.L.T
MODELS = {'v1': 'v1_s0', 'v2': 'V_s1001'}
NTOP, PERVID = 12, 2
GAP = {'mice': 25, 'ants': 10}
CROP = {'mice': 0.36, 'ants': 0.42}
log = L.log


def pick_frames(score, tie, y, obs, k, n=NTOP, gap=10):
    o = np.lexsort((-tie, -score))
    got, per = [], {}
    for i in o:
        v = obs[i]
        if per.get(v, 0) >= PERVID or any(obs[g] == v and abs(int(k[g]) - int(k[i])) < gap for g in got):
            continue
        got.append(i)
        per[v] = per.get(v, 0) + 1
        if len(got) == n:
            break
    return got


def v2_raw_col(dom, tag, j, F):
    od = T.BEE / dom / 'codes'
    R = pd.read_parquet(od / 'eval_rows.parquet')
    E = T.frame_sets(dom)['eval']
    rix = pd.Series(np.arange(len(R)), index=pd.MultiIndex.from_frame(R)).reindex(pd.MultiIndex.from_frame(E)).values
    rix = rix.astype(np.int64)
    Z = np.load(od / f'{tag}_eval.npy', mmap_mode='r')
    return np.asarray(Z[rix, j], np.float32), E


def v2_map(dom, ser, obs, k, sae, norm, j):
    import torch
    from v21_freeze import feats_np
    Vd = T.Video(dom, ser, obs)
    i = int(np.flatnonzero(Vd.k == k)[0])
    W = Vd.window(i, i + 1)
    sel = ((W['flag'] & 1) > 0) & (W['fi'] == i)
    x, cd = feats_np(W['tok'], W['fi'], W['pos'], sel)
    with torch.no_grad():
        c = sae.encode(norm(torch.from_numpy(x + cd).float()), mode='threshold')[:, j].numpy()
    G = Vd.G
    m = np.zeros(G * G, np.float32)
    m[W['pos'][sel]] = c
    S = int(np.load(T.STORE / dom / T.VER / 'videos' / ser / obs / 'index.npz')['S'])
    return m.reshape(G, G), S


def decode_v2(dom, ser, obs, k, S, plans, rot):
    from src.eci.v2.video import decode
    v = plans[(ser, obs)]
    img = decode(v['path'], int(v.get('offset', 2)) + 6 * int(k), 1, 6, S, v['hflip'], 30, threads=2)[0]
    return np.ascontiguousarray(np.rot90(img, rot.get(obs, 0)))


def plans_of(dom):
    import v21_store as ST
    return {(s, v['obs']): v for s, v in ST.lab_videos(dom)}


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
    ap.add_argument('--dom')
    dom = ap.parse_args().dom
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    dev = torch.device('cpu')
    local = Path(os.environ.get('LOCAL_DIR', '/tmp')) / f'ext_sheet_{dom}'
    local.mkdir(parents=True, exist_ok=True)
    F, P, SP, elig = X.table(dom)
    rot = {}
    if dom == 'mice':
        oc = pd.read_csv(REPO / 'dataset/mice/v1/eci/odor_corner.csv').set_index('observation_id')['odor_corner']
        rot = {o: int(ODOR_ROT90[oc[o]]) for o in F.obs.unique()}
        ann_fp = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv', usecols=['frame_path'])['frame_path'].values
        jpg = lambda r: REPO / 'dataset' / ann_fp[int(F.row.values[r])]  # noqa: E731
    else:
        jpg = lambda r: REPO / 'dataset' / F.frame_path.values[r]  # noqa: E731
    assert jpg(0).exists(), jpg(0)
    plans = plans_of(dom)
    cells = [f'{"v1" if dom == "mice" else "pooled"}|{b}' for b in X.CELLS[dom]]
    picks, ident = {}, {}
    sae = {'v1': load_sae(REPO / EV.V1SAE[dom], dev)[:2],
           'v2': SA.load(T.BEE / dom / 'sae' / MODELS['v2'] / 'sae_theta.pt', dev)}
    v1raw_all, _ = EV.load_dict(dom, MODELS['v1'], F)
    for g, tag in MODELS.items():
        J = pickle.load(open(X.OUT / 'tags' / dom / f'{tag}.pkl', 'rb'))
        Z = np.load(X.BEX / dom / f'{tag}.npz')
        col = {int(n): i for i, n in enumerate(Z['neurons'])}
        for c in cells:
            b = c.split('|')[1]
            j = int(J['cells'][c]['best_neuron'])
            idx = np.flatnonzero(P[f'v_{b}'].values)
            pon = Z['pon'][idx, col[j]]
            raw = Z['raw'][idx, col[j]]
            if g == 'v2':
                rawc, _ = v2_raw_col(dom, tag, j, F)
                score, tie, codes = pon, rawc[idx], rawc[idx]
            else:
                score, tie, codes = raw, pon, raw
            y = P[f'y_{b}'].values[idx].astype(bool)
            got = pick_frames(score, tie, y, F.obs.values[idx], F.k.values[idx], gap=GAP[dom])
            picks[(g, c)] = {'neuron': j, 'n_dirs_picked': sum(d['neuron'] == j for d in J['cells'][c]['dirs']),
                             'frames': [dict(row=int(idx[i]), obs=str(F.obs.values[idx[i]]), k=int(F.k.values[idx[i]]),
                                             series=str(F.series.values[idx[i]]), pon=float(pon[i]), code=float(codes[i]),
                                             y=bool(y[i])) for i in got]}
            log(f'{g} {c}: neuron {j}, top frames {sum(f["y"] for f in picks[(g, c)]["frames"])}/{len(got)} positive')
    # ---- per-patch maps + images
    v1_tok = None
    if dom == 'mice':
        sys.path.insert(0, str(REPO / 'scripts/eci/v2_prob'))
        import study as SY
        v1_tok = SY.BH.Store('mice', 'v1', 'dense', 768)
        tix = pd.Series(np.arange(v1_tok.nf), index=pd.MultiIndex.from_arrays([v1_tok.F.obs.values, v1_tok.F.k.values.astype(int)]))
    else:
        v1_ants = v1_ants_maps(F, picks, sae['v1'], dev)
    out = {}
    for (g, c), pk in picks.items():
        j = pk['neuron']
        for f in pk['frames']:
            r, obs, k = f['row'], f['obs'], f['k']
            if g == 'v2':
                ser = 'v1' if dom == 'mice' else ('v4' if f['series'] == 'v4_infested' else f['series'])
                m, S = v2_map(dom, ser, obs, k, *sae['v2'], j)
                img = decode_v2(dom, ser, obs, k, S, plans, rot)
            else:
                if dom == 'mice':
                    import torch as _t
                    fi = int(tix[(obs, k)])
                    a_, b_ = int(v1_tok.starts[fi]), int(v1_tok.starts[fi + 1])
                    s_, n_ = sae['v1']
                    with _t.no_grad():
                        cc = s_.encode(n_(_t.from_numpy(v1_tok.tok(a_, b_)).float()), mode='threshold')[:, j].numpy()
                    m = np.zeros(32 * 32, np.float32)
                    m[v1_tok.pos[a_:b_]] = cc
                    m = m.reshape(32, 32)
                else:
                    m = v1_ants[(r, j)]
                img = np.asarray(Image.open(jpg(r)).convert('RGB'))
                img = np.ascontiguousarray(np.rot90(img, rot.get(obs, 0)))
            f['map_max'] = float(m.max())
            f['identity_abs_diff'] = abs(float(m.max()) - f['code'])
            np.save(local / f'{g}_{c.replace("|", "_")}_{r}.npy', m)
            Image.fromarray(img).save(local / f'{g}_{c.replace("|", "_")}_{r}.png')
        out[f'{g}|{c}'] = pk
    ident = {g: max(f['identity_abs_diff'] for (gg, c), pk in picks.items() if gg == g for f in pk['frames']) for g in MODELS}
    log(f'identity (recomputed map max vs stored frame code), max abs diff: {ident}')
    # ---- figure: rows = behaviours x 2, blocks v1 | v2 (6 columns each)
    nb = len(cells)
    fig, axs = plt.subplots(2 * nb, 13, figsize=(13 * 1.9, 2 * nb * 2.35), gridspec_kw={'width_ratios': [1] * 6 + [0.12] + [1] * 6, 'hspace': 0.55, 'wspace': 0.08})
    for ax in axs.ravel():
        ax.axis('off')
    for bi, c in enumerate(cells):
        for g, c0 in (('v1', 0), ('v2', 7)):
            pk = picks[(g, c)]
            for n_, f in enumerate(pk['frames']):
                ax = axs[2 * bi + n_ // 6, c0 + n_ % 6]
                img = np.asarray(Image.open(local / f'{g}_{c.replace("|", "_")}_{f["row"]}.png'))
                m = np.load(local / f'{g}_{c.replace("|", "_")}_{f["row"]}.npy')
                H, Wd = img.shape[:2]
                G = m.shape[0]
                q = int(np.argmax(m))
                pr, pc = q // G, q % G
                ps = H / G
                cy, cx = (pr + 0.5) * ps, (pc + 0.5) * ps
                hw = CROP[dom] * H / 2
                y0, x0 = int(np.clip(cy - hw, 0, H - 2 * hw)), int(np.clip(cx - hw, 0, Wd - 2 * hw))
                ax.imshow(img, extent=(0, Wd, H, 0))
                mm = np.ma.masked_less_equal(m, 0)
                ax.imshow(mm, extent=(0, Wd, H, 0), cmap='inferno', alpha=0.45, interpolation='nearest',
                          vmin=0, vmax=max(float(m.max()), 1e-6))
                ax.add_patch(Rectangle((pc * ps, pr * ps), ps, ps, fill=False, ec='cyan', lw=1.6))
                ax.set_xlim(x0, x0 + 2 * hw)
                ax.set_ylim(y0 + 2 * hw, y0)
                lab = 'yes' if f['y'] else 'NO'
                ax.set_title(f'P(on) {f["pon"]:.2f} | {lab}\n{f["obs"]} k{f["k"]}', fontsize=6.5,
                             color='black' if f['y'] else 'red')
            npos = sum(f['y'] for f in pk['frames'])
            axs[2 * bi, c0].text(-0.05, 1.30, f'{X.NAME[c.split("|")[1]]} | {g} ({MODELS[g]}) neuron #{pk["neuron"]} '
                                 f'(picked in {pk["n_dirs_picked"]}/12 directions) | true label in {npos}/{len(pk["frames"])} '
                                 'top frames', transform=axs[2 * bi, c0].transAxes, fontsize=8.5, weight='bold')
    fig.suptitle(f'{dom}: top-activating frames of the selected neuron per behaviour, v1 (left) | v2 = arm V (right). '
                 '\nCyan box = max-activation patch, heat = per-patch code of the neuron; caption = P(on) (v1: HMM fitted '
                 'to its raw code), true label (red NO = label absent), video and frame', fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.925, bottom=0.01)
    X.OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(X.OUT / f'sheet_{dom}.png', dpi=100)
    (X.OUT / f'sheet_{dom}.json').write_text(json.dumps({'models': MODELS, 'identity_max_abs_diff': ident, 'cells': out},
                                                        indent=1, default=float))
    shutil.rmtree(local, ignore_errors=True)
    log(f'sheet -> {X.OUT / f"sheet_{dom}.png"}')


def v1_ants_maps(F, picks, sae_norm, dev):
    """re-encode the needed ants jpgs with v1's DINOv2 + fg mask pipeline (cmd_v1codes recipe) -> {(row, j): (32, 32)}."""
    import torch
    import eval as EV
    from src.eci.foreground import RULES, FgBackgrounds, FgEncoder, encode_batch, obs_rows
    sae, norm = sae_norm
    need = {}
    for (g, c), pk in picks.items():
        if g == 'v1':
            for f in pk['frames']:
                need.setdefault(f['row'], set()).add(pk['neuron'])
    rows_F = np.array(sorted(need))
    enc = FgEncoder('dinov2_base', dev)
    out = {}
    for src in F.src.unique():
        sel = rows_F[F.src.values[rows_F] == src]
        if not len(sel):
            continue
        ann = REPO / EV.ANN[src]
        fp = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
        fb = FgBackgrounds(REPO / 'dataset/ants/eci30/fg448/background', obs_rows(ann), RULES['ants'], dev, align='none', cache=8)
        arow = F.row.values[sel].astype(np.int64)
        ds = enc.dataset([str(REPO / 'dataset' / p) for p in fp[arow]], arow, None)
        dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
        o = 0
        for pix, grey, rr in dl:
            with torch.no_grad():
                tok = encode_batch(enc.model, pix, dev)
                mk, _ = fb.mask(tok, grey.to(dev), rr.numpy())
                for b in range(len(tok)):
                    r = int(sel[o + b])
                    pi = torch.nonzero(mk[b], as_tuple=True)[0]
                    cc = sae.encode(norm(tok[b, pi].float()), mode='threshold')
                    for j in need[r]:
                        m = np.zeros(32 * 32, np.float32)
                        m[pi.numpy()] = cc[:, j].numpy()
                        out[(r, j)] = m.reshape(32, 32)
            o += len(tok)
    return out


if __name__ == '__main__':
    main()
