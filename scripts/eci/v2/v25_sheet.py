"""
ECI v2.5 interpretation sheets (pre-registration scripts/eci/v2/v25_prereg.md, "Figures").

Per domain and behaviour, for each model block (v1 seed 0 and / or a v2.5 / V model): the 12 top held-out frames of the
selected neuron, at most 1 per video, ranked by the RAW frame-max code (ties -> earliest frame; the precision@top3/video
ranking), each with the neuron's per-patch activation map overlaid (heat, cyan box = max patch) and the ground-truth
label (caption 'yes' / red 'NO'). Held-out = mice non-SG pools (edge-excluded ext frames), ants half B. Same crop (a fixed
fraction of the frame side, centred on the max patch) and the same panel size for every block.
Maps: v2.5 arms = recomputed from the token store (v25_train.Store, edge rule, the arm's input); V = v21_extended_sheet.v2_map;
v1 = the jpg re-encoded with v1's exact pipeline (DINOv2-base 448 px + v1 foreground mask; behav.cmd_enc1 recipe, CPU).
Identity: max of the recomputed map == the stored frame code (max abs diff reported in the json).
usage: v25_sheet.py --dom mice --blocks v1_s0:611/414,W1_s2502:auc --out NAME
       block = KEY:RULE (neurons from the v2.5 tag of KEY under RULE) or KEY:j1/j2 (explicit; mice nose-nose/nose-tail)
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
import v25_eval as EV25  # noqa: E402  (installs the edge-excluded mice table + caches)
import v21_extended_sheet as XS  # noqa: E402

C, AN, TR = EV25.C, EV25.AN, EV25.TR
REPO, log = C.REPO, C.log
NTOP = 12
CROP = {'mice': 0.36, 'ants': 0.42}
BEHS = {'mice': ['nose_nose', 'nt'], 'ants': ['groom_any']}
NAME = {'nose_nose': 'nose-nose', 'nt': 'nose-tail', 'groom_any': 'grooming-to-focal (any)'}


def table(dom):
    if dom == 'mice':
        F, _ = C.mice_table()
        return F, F.nonsg.values
    F = AN.table()
    Ra = pd.read_parquet(EV25.BEE25 / 'ants' / 'codes' / 'rows_all.parquet')
    idx = pd.Series(np.arange(len(Ra)), index=pd.MultiIndex.from_arrays([Ra.series.values, Ra.obs.values, Ra.k.values]))
    r = idx.reindex(pd.MultiIndex.from_arrays([F.series.values, F.obs.astype(str).values, F.k.values])).values.astype(np.int64)
    F['fi'], F['T'] = Ra.fi.values[r], Ra['T'].values[r]
    return F, F.half.values == 'B'


def raw_codes(dom, key, F):
    if dom == 'mice':
        return C.load(key)[1]
    return AN.readout(key, F)[1]


def neurons(dom, key, spec):
    if '/' in spec or spec.isdigit():
        js = [int(x) for x in spec.split('/')]
        return dict(zip(BEHS[dom], js)), 'explicit'
    tg = pickle.load(open(EV25.tag_file(dom, key), 'rb'))
    d = tg['sets'][f'sg2non|{spec}'][0] if dom == 'mice' else tg['sets']['A2B'][0]
    return {b: int(d['picks'][b]) for b in BEHS[dom]}, (f'rule {spec}' if dom == 'mice' else 'best AUROC on half A')


def top_frames(F, ev, R, j, b):
    rows = np.flatnonzero(ev & F[f'v_{b}'].values)
    r = R[rows, j].astype(np.float64)
    o = np.lexsort((np.arange(len(rows)), -r))
    got, seen = [], set()
    for i in o:
        v = F.obs.values[rows[i]]
        if v in seen:
            continue
        seen.add(v)
        got.append(int(rows[i]))
        if len(got) == NTOP:
            break
    return got


def store_series(dom, F, r):
    return 'v1' if dom == 'mice' else ('v4' if F.series.values[r] == 'v4_infested' else F.series.values[r])


def v25_map(dom, ser, obs, fi, T_, sae, norm, j, arm, alpha):
    import torch
    x, d, _, _, (W, I) = TR.Store(dom).load_xd((ser, obs, fi, fi + 1, T_), dev=torch.device('cpu'), keep=True)
    z = np.load(TR.STORE / dom / TR.VER / 'videos' / ser / obs / 'index.npz')
    G, S = int(z['G']), int(z['S'])
    m = np.zeros(G * G, np.float32)
    with torch.no_grad():
        m[W['pos'][I['c']]] = sae.encode(norm(TR.feats_t(x, d, arm, alpha, TR.alpha1(dom) if arm == 'M2' else None).float()), mode='threshold')[:, j].numpy()
    return m.reshape(G, G), S


def v1_maps(dom, F, need, dev):
    """need {F row: set(neurons)} -> {(row, j): (32, 32)} with v1's exact token pipeline (CPU)."""
    import torch
    sys.path.insert(0, str(REPO / 'scripts/eci/v2_behav'))
    import behav as BH
    from src.eci.sae import load_sae
    from src.eci.foreground import RULES, FgBackgrounds, FgEncoder, align_rot90, encode_batch, obs_rows
    if dom == 'ants':
        picks = {('v1', 'x'): {'neuron': None, 'frames': []}}
        out = {}
        sae_norm = load_sae(REPO / BH.V1[dom]['sae'] / 'sae.pt', dev)[:2]
        for j in sorted({j for s in need.values() for j in s}):
            pk = {('v1', f'c{j}'): {'neuron': j, 'frames': [{'row': r} for r, s in need.items() if j in s]}}
            out.update(XS.v1_ants_maps(F, pk, sae_norm, dev))
        _ = picks
        return out
    c = BH.V1[dom]
    ann = REPO / c['ann']
    ri = BH.row_index(ann)
    rr = np.array(sorted(need))
    arow = ri.reindex(pd.MultiIndex.from_arrays([F.obs.values[rr].astype(str), F.k.values[rr].astype(int)])).values
    assert not np.isnan(arow).any()
    arow = arow.astype(np.int64)
    fp = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    rot = align_rot90(c['align'], ann)
    enc = FgEncoder('dinov2_base', dev)
    fb = FgBackgrounds(REPO / c['bg'], obs_rows(ann), RULES[c['rule']], dev, align=c['align'], cache=8)
    sae, norm = load_sae(REPO / c['sae'] / 'sae.pt', dev)[:2]
    ds = enc.dataset([str(REPO / 'dataset' / p) for p in fp[arow]], arow, None if rot is None else rot[arow])
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    out, o = {}, 0
    for pix, grey, rws in dl:
        with torch.no_grad():
            tok = encode_batch(enc.model, pix, dev)
            mk, _ = fb.mask(tok, grey.to(dev), rws.numpy())
            for b in range(len(tok)):
                r = int(rr[o + b])
                pi = torch.nonzero(mk[b], as_tuple=True)[0]
                cc = sae.encode(norm(tok[b, pi].float()), mode='threshold')
                for j in need[r]:
                    m = np.zeros(32 * 32, np.float32)
                    m[pi.numpy()] = cc[:, j].numpy()
                    out[(r, j)] = m.reshape(32, 32)
        o += len(tok)
    return out, fp, arow, rr


def main():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    from src.eci.v2 import sae as SA
    ap = argparse.ArgumentParser()
    ap.add_argument('--dom', required=True)
    ap.add_argument('--blocks', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dom = a.dom
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    dev = torch.device('cpu')
    local = Path(os.environ.get('LOCAL_DIR', '/tmp')) / f'v25_sheet_{a.out}'
    local.mkdir(parents=True, exist_ok=True)
    F, ev = table(dom)
    rot = {}
    if dom == 'mice':
        from src.eci.foreground import ODOR_ROT90
        oc = pd.read_csv(REPO / 'dataset/mice/v1/eci/odor_corner.csv').set_index('observation_id')['odor_corner']
        rot = {o: int(ODOR_ROT90[oc[o]]) for o in F.obs.unique()}
    plans = XS.plans_of(dom)
    blocks = []
    for bs in a.blocks.split(','):
        key, spec = bs.split(':')
        js, how = neurons(dom, key, spec)
        R = raw_codes(dom, key, F)
        blk = {'key': key, 'how': how, 'cells': {}}
        for b in BEHS[dom]:
            got = top_frames(F, ev, R, js[b], b)
            blk['cells'][b] = {'neuron': js[b], 'frames': [dict(row=r, obs=str(F.obs.values[r]), k=int(F.k.values[r]), fi=int(F.fi.values[r]),
                                                              T=int(F['T'].values[r]), code=float(R[r, js[b]]), y=bool(F[f'y_{b}'].values[r]))
                                                         for r in got]}
            log(f'{key} {b} #{js[b]}: labelled {sum(f["y"] for f in blk["cells"][b]["frames"])}/{len(got)}')
        del R
        blocks.append(blk)
    # maps + images
    for blk in blocks:
        key = blk['key']
        g = key.split('_s')[0]
        if key.startswith('v1_'):
            need = {}
            for c in blk['cells'].values():
                for f in c['frames']:
                    need.setdefault(f['row'], set()).add(c['neuron'])
            got = v1_maps(dom, F, need, dev)
            if dom == 'mice':
                maps, fp, arow, rr = got
                jpg = {int(r): REPO / 'dataset' / fp[ar] for r, ar in zip(rr, arow)}
            else:
                maps = got
                jpg = {r: REPO / 'dataset' / F.frame_path.values[r] for r in need}
        else:
            if g in TR.ARMS:
                sae, norm = SA.load(TR.BEE25 / dom / 'sae' / key / 'sae_theta.pt', dev)
                alpha = json.loads((TR.OUT25 / f'{dom}_count.json').read_text())['alpha']
            else:
                sae, norm = SA.load(XS.T.BEE / dom / 'sae' / key / 'sae_theta.pt', dev)
        for b, c in blk['cells'].items():
            j = c['neuron']
            for f in c['frames']:
                r, obs, k = f['row'], f['obs'], f['k']
                ser = store_series(dom, F, r)
                if key.startswith('v1_'):
                    m = maps[(r, j)]
                    img = np.asarray(Image.open(jpg[r]).convert('RGB'))
                    img = np.ascontiguousarray(np.rot90(img, rot.get(obs, 0)))
                else:
                    if g in TR.ARMS:
                        m, S = v25_map(dom, ser, obs, f['fi'], f['T'], sae, norm, j, g, alpha)
                    else:
                        m, S = XS.v2_map(dom, ser, obs, k, sae, norm, j)
                    img = XS.decode_v2(dom, ser, obs, k, S, plans, rot)
                f['map_max'] = float(m.max())
                f['identity_abs_diff'] = abs(float(m.max()) - f['code'])
                np.save(local / f'{key}_{b}_{r}.npy', m)
                Image.fromarray(img).save(local / f'{key}_{b}_{r}.png')
        blk['identity_max_abs_diff'] = max(f['identity_abs_diff'] for c in blk['cells'].values() for f in c['frames'])
        log(f'{key}: identity (map max vs stored frame code) max abs diff {blk["identity_max_abs_diff"]:.4f}')
    # figure: rows = 2 per behaviour, blocks side by side (6 columns each)
    nb, nk = len(BEHS[dom]), len(blocks)
    wr = []
    for i in range(nk):
        wr += [1] * 6 + ([0.12] if i < nk - 1 else [])
    fig, axs = plt.subplots(2 * nb, len(wr), figsize=(len(wr) * 1.9 * 0.97, 2 * nb * 2.4),
                            gridspec_kw={'width_ratios': wr, 'hspace': 0.55, 'wspace': 0.08})
    axs = np.atleast_2d(axs)
    for ax in axs.ravel():
        ax.axis('off')
    for bi, b in enumerate(BEHS[dom]):
        for kb, blk in enumerate(blocks):
            c0 = 7 * kb
            c = blk['cells'][b]
            for n_, f in enumerate(c['frames']):
                ax = axs[2 * bi + n_ // 6, c0 + n_ % 6]
                img = np.asarray(Image.open(local / f'{blk["key"]}_{b}_{f["row"]}.png'))
                m = np.load(local / f'{blk["key"]}_{b}_{f["row"]}.npy')
                H, Wd = img.shape[:2]
                G = m.shape[0]
                q = int(np.argmax(m))
                pr, pc = q // G, q % G
                ps = H / G
                hw = CROP[dom] * H / 2
                cy, cx = (pr + 0.5) * ps, (pc + 0.5) * ps
                y0, x0 = int(np.clip(cy - hw, 0, H - 2 * hw)), int(np.clip(cx - hw, 0, Wd - 2 * hw))
                ax.imshow(img, extent=(0, Wd, H, 0))
                ax.imshow(np.ma.masked_less_equal(m, 0), extent=(0, Wd, H, 0), cmap='inferno', alpha=0.45, interpolation='nearest',
                          vmin=0, vmax=max(float(m.max()), 1e-6))
                ax.add_patch(Rectangle((pc * ps, pr * ps), ps, ps, fill=False, ec='cyan', lw=1.4))
                ax.set_xlim(x0, x0 + 2 * hw)
                ax.set_ylim(y0 + 2 * hw, y0)
                ax.set_title(f'{"yes" if f["y"] else "NO"} | code {f["code"]:.1f}\n{f["obs"]} k{f["k"]}', fontsize=6.3,
                             color='black' if f['y'] else 'red')
            npos = sum(f['y'] for f in c['frames'])
            g = blk['key'].split('_s')
            axs[2 * bi, c0].text(-0.05, 1.32, f'{NAME[b]} | {g[0]} seed {g[1]} neuron #{c["neuron"]} ({blk["how"]}) | '
                                 f'labelled {npos}/{len(c["frames"])}', transform=axs[2 * bi, c0].transAxes, fontsize=8.5, weight='bold')
    held = 'held-out non-SG pools, edge-excluded ext frames' if dom == 'mice' else 'held-out half B (selected on A)'
    fig.suptitle(f'{dom}: top-12 frames of the selected neuron ({held}); ranked by raw frame-max code, 1 per video.\n'
                 'Heat = per-patch code, cyan = max patch;\ncaption = true label (red NO = absent), raw code, video, frame',
                 fontsize=10, y=0.997)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9 if dom == 'mice' else 0.8, bottom=0.01)
    od = TR.OUT25 / 'figures'
    od.mkdir(parents=True, exist_ok=True)
    fig.savefig(od / f'{a.out}.png', dpi=72)
    (od / f'{a.out}.json').write_text(json.dumps(blocks, indent=1, default=float))
    shutil.rmtree(local, ignore_errors=True)
    log(f'sheet -> {od / (a.out + ".png")}')


if __name__ == '__main__':
    main()
