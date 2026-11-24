"""
ECI v2.8 top-frame sheets + neuron catalogue for the v2.x Matryoshka SAE dictionaries (generalises v25_sheet / v26_sheet).

One PNG per neuron: the 12 top held-out frames by RAW frame code (frame code = max over the frame's mask-v5 tokens of the
neuron's SAE code), from distinct videos where possible (pass 1: 1 per video; pass 2: a 2nd per video), never two frames of
one video < 2 s apart (10 frames at 5 fps), code > 0 only. Held-out frames: mice = the 12 non-SG pools (edge-excluded ext
frames, annotated for nose-nose and nose-tail); ants = half B of v3 / v4 (v23_ants.table split, the A -> B direction).
Each tile: the frame (decoded from the source video at the token store's geometry; mice turned by odour corner), the
neuron's per-token code as a heat map (cyan box = max token), a crop of CROP x frame side centred on the max token, and a
caption: video, time (mm:ss of the 5 fps annotation index), raw code, ground-truth labels (mice nose-nose / nose-tail;
ants grooming-to-focal by the yellow / blue nestmate).
Header: dictionary, neuron, Matryoshka level, the roles it was picked for (rule, behaviour) with the pre-registered
held-out AUROC (HMM P(on) readout) and precision@top3/video from the version's report.json, and the held-out AUROC of the
RAW frame code for every label (computed here).
Per-token maps are recomputed from the token store (eci_tokens/v2_full, v2.1_m5) through the SAE (CPU). Identity: per
tile, max of the recomputed map == the stored frame code (fp16); --selfcheck N: N random held-out frames, max over tokens
of ALL latents == the stored code row. Max abs differences are written to the json and the log.

Catalogue mode (--catalogue N): every neuron's held-out raw-code statistics (firing rate = fraction of held-out frames with
code > 0; AUROC of the raw frame code vs every y_* label; sharpness; videos among its top-200 frames). Candidates: firing
rate in [0.2%, 20%], >= 8 distinct videos among the top-200 frames, not a picked behaviour neuron of this dictionary (any
rule). Sharpness = mean of the top-50 held-out frame codes / median of the non-zero held-out codes (how far the
strongest events stand above the neuron's typical firing). N spread over Matryoshka levels: N // n_levels best-sharpness
per level, the remainder filled by global sharpness. Writes catalogue/index.json (all candidates' stats for the chosen
ones) + one sheet per chosen neuron.

Versions: --ver 27 (v27_eval paths: PREDc / PREDd / PREDk / PREDh s2701-2703, PRED26 s2601-2603 [mice only], V s1001-1006
[mice]); --ver 26 (v26_eval paths: N0wc / PRED / SMOOTH / RW / N0x s2601-2603, both domains; ants PRED26 = PRED_s2601).
usage:
  v28_sheets.py --ver 27 --dom mice --key PREDc_s2701 --rules auc,dist [--selfcheck 200]
  v28_sheets.py --ver 26 --dom ants --key PRED_s2601 --name PRED26_s2601 --rules auc
  v28_sheets.py --ver 27 --dom mice --key PRED26_s2601 --catalogue 40
  v28_sheets.py ... --neurons 12,345   (explicit neurons, no picks)
Outputs: results/vision/eci_v2_8/sheets/<dom>/<name>/n<j>.png + sheets.json; catalogue -> .../<name>/catalogue/.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[2]
OUT28 = REPO / 'results/vision/eci_v2_8/sheets'
NTOP, PER_VID, GAP = 12, 2, 10
CROP = {'mice': 0.5, 'ants': 0.5}
NAME = {'nose_nose': 'nose-nose', 'nt': 'nose-tail', 'nn': 'nose-nose-mutual', 'np': 'nose-nose-passive',
        'groom_any': 'grooming-to-focal (any)', 'Y2F': 'yellow grooms focal', 'B2F': 'blue grooms focal',
        'single': 'grooming-to-focal (single)', 'double': 'grooming-to-focal (double)'}
G = {}  # modules, set by init()


def init(ver):
    if ver == '26':
        import v26_eval  # noqa: F401  (patches v25_train paths -> v2.6)
    elif ver == '27':
        import v27_eval  # noqa: F401  (patches v25_train paths -> v2.7)
    else:
        raise SystemExit(f'unknown --ver {ver}')
    import v25_sheet as S
    import v24_top3 as T3
    G.update(S=S, T3=T3, EV=S.EV25, C=S.C, AN=S.AN, TR=S.TR, XS=S.XS, log=S.log)


# ---------------------------------------------------------------------------------------------- tables / metadata
def table(dom):
    """-> F (frame table), held-out rows (int array), label names (y_* columns usable on the held-out rows)."""
    S = G['S']
    F, ev = S.table(dom)
    if dom == 'mice':
        ev = ev & F.v_nose_nose.values & F.v_nt.values
    else:
        ev = ev & F.v_groom_any.values
    labs = [c[2:] for c in F.columns if c.startswith('y_')]
    labs = [b for b in labs if (f'v_{b}' not in F.columns or F[f'v_{b}'].values[ev].all())]
    return F, np.flatnonzero(ev), labs


def raw_path(dom, key):
    TR = G['TR']
    if dom == 'mice':
        return G['C'].cache_paths(key)[0]
    p = TR.BEE25 / 'ants' / 'cache' / f'{key}_raw.npy'
    if not p.exists():
        raise SystemExit(f'no ants raw cache for {key} at {p} (only v2.5+ arms are supported for ants)')
    return p


def report_roles(dom, key):
    """{neuron: [{beh, rule, auc, top3}]} from the version's report.json (pre-registered metrics, held-out)."""
    TR = G['TR']
    arm = key.split('_s')[0]
    roles = {}
    if dom == 'mice':
        rep = json.loads((TR.OUT25 / 'mice' / 'report.json').read_text())
        idx = rep['models'][arm].index(key)
        for pk_key, pk in rep['picks'].items():
            a_, rule = pk_key.split('|')
            if a_ != arm or key not in pk:
                continue
            lv = rep['levels'][pk_key]
            for b, j in pk[key].items():
                roles.setdefault(int(j), []).append({'beh': b, 'rule': rule, 'auc': lv[f'{b}|auc']['seeds'][idx],
                                                     'top3': lv.get(f'{b}|top3', {}).get('seeds', [None] * (idx + 1))[idx]})
    else:
        rep = json.loads((TR.OUT25 / 'ants' / 'report.json').read_text())
        idx = rep['models'][arm].index(key)
        d = rep['dirs']['A2B']
        j = int(d['picks'][arm][key])
        lv = d['levels'][arm]
        roles[j] = [{'beh': 'groom_any', 'rule': 'auc (select half A)', 'auc': lv['groom_any|auc']['seeds'][idx],
                     'top3': lv.get('groom_any|top3', {}).get('seeds', [None] * (idx + 1))[idx]}]
    return roles


def level_of(prefixes, j):
    for i, p in enumerate(prefixes):
        if j < p:
            return i + 1, (prefixes[i - 1] if i else 0), p
    raise ValueError(j)


# ---------------------------------------------------------------------------------------------- SAE / maps
class Enc:
    """per-frame token codes (all latents) from the token store, the dictionary's own input path."""

    def __init__(self, dom, key):
        import torch
        from src.eci.v2 import sae as SA
        TR, XS = G['TR'], G['XS']
        self.dom, self.key, self.arm = dom, key, key.split('_s')[0]
        self.dev = torch.device('cpu')
        if self.arm in TR.ARMS:
            self.kind = 'v25'
            self.sae, self.norm = SA.load(TR.BEE25 / dom / 'sae' / key / 'sae_theta.pt', self.dev)
            self.alpha = json.loads((TR.OUT25 / f'{dom}_count.json').read_text())['alpha']
            self.store = TR.Store(dom)
        elif self.arm == 'V':
            self.kind = 'V'
            self.sae, self.norm = SA.load(XS.T.BEE / dom / 'sae' / key / 'sae_theta.pt', self.dev)
        else:
            raise SystemExit(f'{key}: no token path (v1 dictionaries are not supported here)')
        self.prefixes = tuple(int(p) for p in self.sae.prefixes)
        self.cache = {}

    def frame(self, F, r):
        """-> (token codes (n_tok, W) float32, grid positions (n_tok,), grid side G, store side S)."""
        if r in self.cache:
            return self.cache[r]
        import torch
        TR, XS, S = G['TR'], G['XS'], G['S']
        dom = self.dom
        ser, obs = S.store_series(dom, F, r), str(F.obs.values[r])
        if self.kind == 'v25':
            fi, T_ = int(F.fi.values[r]), int(F['T'].values[r])
            x, d, _, _, (W, I) = self.store.load_xd((ser, obs, fi, fi + 1, T_), dev=self.dev, keep=True)
            z = np.load(TR.STORE / dom / TR.VER / 'videos' / ser / obs / 'index.npz')
            Gs, Ss = int(z['G']), int(z['S'])
            with torch.no_grad():
                cc = self.sae.encode(self.norm(TR.feats_t(x, d, self.arm, self.alpha).float()), mode='threshold').numpy()
            pos = W['pos'][I['c']]
        else:
            from v21_freeze import feats_np
            T = XS.T
            k = int(F.k.values[r])
            Vd = T.Video(dom, ser, obs)
            i = int(np.flatnonzero(Vd.k == k)[0])
            W = Vd.window(i, i + 1)
            sel = ((W['flag'] & 1) > 0) & (W['fi'] == i)
            x, cd = feats_np(W['tok'], W['fi'], W['pos'], sel)
            with torch.no_grad():
                cc = self.sae.encode(self.norm(torch.from_numpy(x + cd).float()), mode='threshold').numpy()
            pos, Gs = W['pos'][sel], Vd.G
            Ss = int(np.load(T.STORE / dom / T.VER / 'videos' / ser / obs / 'index.npz')['S'])
        out = (cc.astype(np.float32), np.asarray(pos), Gs, Ss)
        if len(self.cache) < 2000:
            self.cache[r] = out
        return out

    def map(self, F, r, j):
        cc, pos, Gs, Ss = self.frame(F, r)
        m = np.zeros(Gs * Gs, np.float32)
        m[pos] = cc[:, j]
        return m.reshape(Gs, Gs), Ss


# ---------------------------------------------------------------------------------------------- frame choice / stats
def pick_top(code, rows, obs, fi):
    """top NTOP rows by code (desc; ties -> earliest row), code > 0, pass 1 one per video, pass 2 up to PER_VID per
    video, never two picks of one video < GAP frames apart."""
    c = code[rows]
    m = min(len(rows), 50000)
    top = np.argpartition(-c, m - 1)[:m] if m < len(rows) else np.arange(len(rows))
    o = rows[top[np.lexsort((rows[top], -c[top]))]]
    got, cnt = [], {}
    for per in (1, PER_VID):
        for r in o:
            if len(got) == NTOP or code[r] <= 0:
                break
            v = obs[r]
            if cnt.get(v, 0) >= per or r in got or any(obs[g] == v and abs(int(fi[g]) - int(fi[r])) < GAP for g in got):
                continue
            got.append(int(r))
            cnt[v] = cnt.get(v, 0) + 1
    return sorted(got, key=lambda r: (-code[r], r))


def label_cat(F, ev, labs):
    n = len(F)
    cat = np.zeros((n, len(labs)), np.int8)
    for q, b in enumerate(labs):
        cat[ev, q] = np.where(F[f'y_{b}'].values[ev], 2, 1)
    return cat[ev]


def read_rows(path, rows, js=None):
    """rows (sorted) [and columns js] of a (n, W) fp16 .npy, sequential row chunks -> fp16 array."""
    mm = np.load(path, mmap_mode='r')
    W = mm.shape[1] if js is None else len(js)
    out = np.empty((len(rows), W), np.float16)
    o = 0
    for c0 in range(0, mm.shape[0], 40000):
        rr = rows[(rows >= c0) & (rows < c0 + 40000)]
        if len(rr):
            blk = np.asarray(mm[c0:c0 + 40000])[rr - c0]
            out[o:o + len(rr)] = blk if js is None else blk[:, js]
            o += len(rr)
    assert o == len(rows)
    return out


def catalogue_stats(Xe, cat, obs_ev):
    """Xe (n_ev, W) fp16 held-out raw codes -> per-neuron fire, AUROC (Q, W), sharpness, videos among top-200."""
    C = G['C']
    Xt = np.ascontiguousarray(Xe.T.astype(np.float32))
    fire = (Xt > 0).mean(1)
    auc = C._auc_multi(Xt, cat)
    W = Xt.shape[0]
    sharp, nvid = np.full(W, np.nan), np.zeros(W, int)
    ocode = pd.factorize(obs_ev)[0]
    for w in range(W):
        x = Xt[w]
        nz = x[x > 0]
        if len(nz) < 50:
            continue
        sharp[w] = float(np.partition(x, len(x) - 50)[-50:].mean() / np.median(nz))
        t2 = np.argpartition(-x, 199)[:200]
        nvid[w] = len(np.unique(ocode[t2[x[t2] > 0]]))
    return fire, auc, sharp, nvid, Xt


# ---------------------------------------------------------------------------------------------- figure
def render(path, dom, F, rows, codes, j, enc, plans, rot, title, local, labs_show):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    XS, S = G['XS'], G['S']
    fig, axs = plt.subplots(3, 4, figsize=(4 * 2.5, 3 * 2.85 + 1.45))
    for ax in axs.ravel():
        ax.axis('off')
    tiles = []
    for n_, r in enumerate(rows):
        ax = axs[n_ // 4, n_ % 4]
        m, Ss = enc.map(F, r, j)
        obs, k = str(F.obs.values[r]), int(F.k.values[r])
        ip = local / f'img_{r}.npy'
        if ip.exists():
            img = np.load(ip)
        else:
            img = XS.decode_v2(dom, S.store_series(dom, F, r), obs, k, Ss, plans, rot)
            np.save(ip, img)
        H, Wd = img.shape[:2]
        Gs = m.shape[0]
        q = int(np.argmax(m))
        pr, pc = q // Gs, q % Gs
        ps = H / Gs
        hw = CROP[dom] * H / 2
        cy, cx = (pr + 0.5) * ps, (pc + 0.5) * ps
        y0, x0 = int(np.clip(cy - hw, 0, H - 2 * hw)), int(np.clip(cx - hw, 0, Wd - 2 * hw))
        ax.imshow(img, extent=(0, Wd, H, 0))
        ax.imshow(np.ma.masked_less_equal(m, 0), extent=(0, Wd, H, 0), cmap='inferno', alpha=0.45, interpolation='nearest',
                  vmin=0, vmax=max(float(m.max()), 1e-6))
        ax.add_patch(Rectangle((pc * ps, pr * ps), ps, ps, fill=False, ec='cyan', lw=1.4))
        ax.set_xlim(x0, x0 + 2 * hw)
        ax.set_ylim(y0 + 2 * hw, y0)
        t = k / 5.0
        lab = ' | '.join(f'{NAME.get(b, b)} {"YES" if F[f"y_{b}"].values[r] else "no"}' for b in labs_show)
        anyp = any(F[f'y_{b}'].values[r] for b in labs_show)
        ax.set_title(f'#{n_ + 1} {obs} {int(t // 60):02d}:{t % 60:04.1f}  code {codes[r]:.2f}\n{lab}', fontsize=6.4,
                     color='darkgreen' if anyp else 'black')
        code16 = float(np.float16(codes[r]))
        tiles.append({'row': int(r), 'obs': obs, 'series': str(F.series.values[r]), 'k': k, 'time_s': t,
                      'code': code16, 'labels': {b: bool(F[f'y_{b}'].values[r]) for b in labs_show},
                      'map_max': float(m.max()), 'identity_abs_diff': abs(float(np.float16(m.max())) - code16)})
    fig.suptitle(title, fontsize=8.2, x=0.01, ha='left', y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.5 / (3 * 2.85 + 1.45), bottom=0.01, hspace=0.32, wspace=0.04)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return tiles


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ver', required=True, choices=('26', '27'))
    ap.add_argument('--dom', required=True, choices=('mice', 'ants'))
    ap.add_argument('--key', required=True, help='dictionary key in that version, e.g. PREDc_s2701, PRED26_s2601, V_s1002')
    ap.add_argument('--name', default='', help='output folder name (default = key)')
    ap.add_argument('--rules', default='auc,dist', help='picked neurons of these rules get a sheet (picks mode)')
    ap.add_argument('--neurons', default='', help='explicit neuron list instead of the picks')
    ap.add_argument('--catalogue', type=int, default=0, help='N > 0: catalogue mode')
    ap.add_argument('--selfcheck', type=int, default=200)
    a = ap.parse_args()
    init(a.ver)
    import torch
    log, S, TR = G['log'], G['S'], G['TR']
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    t0 = time.time()
    dom, key = a.dom, a.key
    name = a.name or key
    od = OUT28 / dom / name / ('catalogue' if a.catalogue else '')
    local = Path(os.environ.get('LOCAL_DIR', '/tmp')) / f'v28_{dom}_{name}_{"cat" if a.catalogue else "pk"}'
    (local / 'png').mkdir(parents=True, exist_ok=True)
    F, ev, labs = table(dom)
    labs_show = ['nose_nose', 'nt'] if dom == 'mice' else ['Y2F', 'B2F']
    log(f'{dom} {key}: held-out rows {len(ev):,}, videos {len(set(F.obs.values[ev]))}, labels {labs}')
    roles = report_roles(dom, key)
    enc = Enc(dom, key)
    rp = raw_path(dom, key)
    plans = G['XS'].plans_of(dom)
    rot = {}
    if dom == 'mice':
        from src.eci.foreground import ODOR_ROT90
        oc = pd.read_csv(REPO / 'dataset/mice/v1/eci/odor_corner.csv').set_index('observation_id')['odor_corner']
        rot = {o: int(ODOR_ROT90[oc[o]]) for o in F.obs.unique()}
    cat = label_cat(F, ev, labs)
    res = {'dom': dom, 'ver': a.ver, 'key': key, 'name': name, 'prefixes': enc.prefixes, 'heldout_rows': int(len(ev)),
           'heldout_videos': int(len(set(F.obs.values[ev]))), 'labels': labs, 'raw_cache': str(rp),
           'ranking': 'raw frame code (max over mask tokens), held-out frames; pass 1 one per video, pass 2 up to 2 per video, '
                      f'>= {GAP} frames (2 s) apart, code > 0'}
    # ---- which neurons
    if a.catalogue:
        Xe = read_rows(rp, ev)
        fire, auc, sharp, nvid, Xt = catalogue_stats(Xe, cat, F.obs.values[ev])
        del Xe
        Wn = len(fire)
        picked = set(roles)
        cand = (fire >= 0.002) & (fire <= 0.20) & (nvid >= 8) & np.isfinite(sharp)
        cand[list(picked)] = False
        lv = np.array([level_of(enc.prefixes, j)[0] for j in range(Wn)])
        nl = len(enc.prefixes)
        quota = a.catalogue // nl
        chosen = []
        for L_ in range(1, nl + 1):
            c = np.flatnonzero(cand & (lv == L_))
            chosen += c[np.argsort(-sharp[c])][:quota].tolist()
        rest = np.flatnonzero(cand & ~np.isin(np.arange(Wn), chosen))
        chosen += rest[np.argsort(-sharp[rest])][:a.catalogue - len(chosen)].tolist()
        chosen = sorted(chosen, key=lambda j: (lv[j], -sharp[j]))
        res['catalogue'] = {'N': a.catalogue, 'n_candidates': int(cand.sum()), 'n_alive_heldout': int((fire > 0).sum()),
                            'excluded_picked': sorted(int(j) for j in picked),
                            'rule': 'firing rate on held-out frames in [0.2%, 20%], >= 8 videos among the top-200 frames, not '
                                    'a picked behaviour neuron (any rule); sharpness = mean(top-50 held-out frame codes) / '
                                    'median(non-zero held-out codes); N // n_levels best per Matryoshka level, rest by sharpness',
                            'auroc_readout': 'raw frame code, held-out frames (NOT the HMM P(on) readout of the reports)',
                            'candidates_per_level': {int(L_): int((cand & (lv == L_)).sum()) for L_ in range(1, nl + 1)}}
        js = chosen
        col = {j: Xt[j] for j in js}
        del Xt
        codes_of = {}
        for j in js:
            c = np.zeros(len(F), np.float32)
            c[ev] = col[j]
            codes_of[j] = c
        stats = {j: {'fire': float(fire[j]), 'auc': {b: float(auc[q, j]) for q, b in enumerate(labs)}, 'sharpness': float(sharp[j]),
                     'videos_top200': int(nvid[j])} for j in js}
    else:
        js = [int(x) for x in a.neurons.split(',')] if a.neurons else sorted({j for j, rr in roles.items()
                                                                              if any(r_['rule'].split(' ')[0] in a.rules.split(',') for r_ in rr)})
        R = G['T3'].cols_chunked(rp, js)
        codes_of = {j: R[:, i] for i, j in enumerate(js)}
        Xt = np.ascontiguousarray(R[ev].T)
        aucj = G['C']._auc_multi(Xt, cat)
        stats = {j: {'fire': float((R[ev, i] > 0).mean()), 'auc': {b: float(aucj[q, i]) for q, b in enumerate(labs)}}
                 for i, j in enumerate(js)}
        del Xt
    log(f'neurons {js} ({time.time() - t0:.0f}s)')
    # ---- sheets
    obs_all, fi_all = F.obs.values.astype(str), F.fi.values if 'fi' in F.columns else F.k.values
    out = []
    for j in js:
        code = codes_of[j]
        rows = pick_top(code, ev, obs_all, fi_all)
        L_, lo, hi = level_of(enc.prefixes, j)
        st = stats[j]
        rl = roles.get(j, [])
        grp = {}
        for r_ in rl:  # same neuron + behaviour -> same metrics under every rule: one entry listing the rules
            grp.setdefault((r_['beh'], r_['auc'], r_['top3']), []).append(r_['rule'])
        rtxt = '\n'.join(f'{NAME.get(b, b)} pick (rule {", ".join(rs)}): held-out AUROC {au:.3f} (P(on) readout), precision@top3/video '
                         + (f'{t3:.3f}' if t3 is not None else 'n/a') for (b, au, t3), rs in grp.items()) or 'not a picked behaviour neuron'
        main_labs = ['nose_nose', 'nt'] if dom == 'mice' else ['groom_any']
        atxt = ', '.join(f'{NAME.get(b, b)} {st["auc"][b]:.3f}' for b in main_labs if b in st['auc'])
        title = (f'{dom} | {name} | neuron #{j} | Matryoshka level {L_} (latents {lo}-{hi - 1}) | held-out firing rate {100 * st["fire"]:.2f}% '
                 + (f'| sharpness {st["sharpness"]:.2f}' if 'sharpness' in st else '') + '\n' + rtxt +
                 f'\nraw-code held-out AUROC: {atxt} | top-{len(rows)} held-out frames by raw frame code, <= 2 per video, >= 2 s apart'
                 '\nheat = per-token code, cyan box = max token; green caption = a label is YES')
        tiles = render(local / 'png' / f'n{j}.png', dom, F, rows, code, j, enc, plans, rot, title, local, labs_show)
        idm = max([t['identity_abs_diff'] for t in tiles] or [0.0])
        npos = {b: int(sum(t['labels'][b] for t in tiles)) for b in labs_show}
        if dom == 'mice':
            npos['nose_nose'] = int(sum(bool(F.y_nose_nose.values[t['row']]) for t in tiles))
        else:
            npos['groom_any'] = int(sum(bool(F.y_groom_any.values[t['row']]) for t in tiles))
        out.append({'neuron': j, 'level': L_, **st, 'roles': rl, 'n_tiles': len(tiles), 'n_videos': len({t['obs'] for t in tiles}),
                    'labelled_positive': npos, 'identity_max_abs_diff': idm, 'tiles': tiles})
        log(f'#{j} L{L_}: tiles {len(tiles)} from {out[-1]["n_videos"]} videos, labelled {npos}, identity {idm:.4g} ({time.time() - t0:.0f}s)')
    res['sheets'] = out
    res['identity_tiles'] = {'frames': int(sum(len(o['tiles']) for o in out)),
                             'max_abs_diff': float(max([o['identity_max_abs_diff'] for o in out] or [0.0]))}
    # ---- self-check: random held-out frames, all latents
    if a.selfcheck:
        rng = np.random.default_rng(28)
        rr = np.sort(rng.choice(ev, a.selfcheck, replace=False))
        Z = read_rows(rp, rr) if a.catalogue else np.load(rp, mmap_mode='r')[rr]
        mx = 0.0
        for i, r in enumerate(rr):
            cc = enc.frame(F, int(r))[0]
            ref = cc.max(0) if len(cc) else np.zeros(Z.shape[1], np.float32)
            ref = np.maximum(ref, 0)
            mx = max(mx, float(np.abs(ref.astype(np.float16).astype(np.float32) - np.asarray(Z[i], np.float32)).max()))
        res['selfcheck'] = {'frames': int(len(rr)), 'latents': int(Z.shape[1]), 'max_abs_diff_fp16': mx}
        log(f'selfcheck {len(rr)} random held-out frames x all {Z.shape[1]} latents: max abs diff {mx:.4g}')
    res['runtime_s'] = time.time() - t0
    od.mkdir(parents=True, exist_ok=True)
    for p in (local / 'png').glob('*.png'):
        shutil.copy(p, od / p.name)
    jn = 'index.json' if a.catalogue else 'sheets.json'
    (od / jn).write_text(json.dumps(res, indent=1, default=float))
    shutil.rmtree(local, ignore_errors=True)
    log(f'{len(out)} sheets -> {od} ({time.time() - t0:.0f}s); tile identity max abs diff {res["identity_tiles"]["max_abs_diff"]:.4g}')


if __name__ == '__main__':
    main()
