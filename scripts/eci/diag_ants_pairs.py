"""
Diagnostic (no new model): does the DEPLOYED ants foreground SAE (antsfg: Matryoshka BatchTopK, 1024 latents, k = 16,
DINOv2-base patch tokens at 448 px, foreground patches only), read at the right LOCATION on the 32 x 32 patch grid,
separate "yellow grooms focal" (Y_Y2F) from "blue grooms focal" (Y_B2F)?

Data
    token store   dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfg_fps1 (every foreground patch token with its
                  row-major 32 x 32 position, 1 fps = one random-offset frame in 5, all 256 videos, 153,600 frames)
    SAE           dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0/sae.pt (threshold inference, as deployed)
    stored codes  dataset/ants/eci/codes/matryoshka_btk_1024_k16_antsfg_s0/codes_max.npy (frame-level max; the
                  re-encoded patch codes must reproduce it)
    tracking      dataset/ants/{v2,v3}/tracking/<obs>.csv, 5 fps, pixel coordinates of the 512 x 512 standardized
                  video (src/tracking/get_tracking.py reads data/ants/<v>/observations/full/*.mkv, 512 px); the ECI
                  frame is the same 512 px frame resized to 448, so patch (row i, col j) covers pixels
                  [16 j, 16 j + 16) x [16 i, 16 i + 16) and a centroid (x, y) sits at grid point (x / 16, y / 16).
    labels        dataset/ants/eci/annotations.csv Y_Y2F, Y_B2F (the tracking csv's own B2F / Y2F = proximity flags,
                  unused)

Steps
    encode   (GPU or CPU)  every store token -> SAE codes, kept sparse (CSR: indptr, latent idx int16, value float16), plus
                    a check against the stored frame-level codes_max    -> <out>/patch_codes/
    analyze  (CPU)  region reads, neuron scan, baselines, tracking-quality split, contact sheet -> <out>/

Regions (per frame, grid units; patch centre = (j + .5, i + .5); G = the candidate groomer's anchor, O = the other
coloured ant's anchor, F = focal body centroid; anchor 'body' = tracked body centroid, 'mark' = colour-dot position.
When the three ants merge into one blob the tracker gives all three the SAME body centroid, so body distances tie)
    seg{r}     foreground patches within r of the segment G-F (r = 1, 2, 3)
    vor{d}     foreground patches within d of F and closer to G than to O (d = 2, 3, 4, 6)
    segvor{r}  seg{r} and closer to G than to O
    disk{r}    within r of G
    score    max of a latent over the region (0 if the region holds no foreground patch)
r (and d) are chosen on v3 (all frames, discrimination AUROC of neuron 90) before v2 is looked at.

Usage (scripts/eci/diag_ants_pairs.sh stages the inputs to /localhome):
    python scripts/eci/diag_ants_pairs.py encode  --store <S> --codes-max <C> --out <O>
    python scripts/eci/diag_ants_pairs.py analyze --codes-max <C> --out <O>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.foreground import FgTokenStore  # noqa: E402
from src.eci.sae import load_sae  # noqa: E402

SAE = REPO / 'dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0/sae.pt'
ANN = REPO / 'dataset/ants/eci/annotations.csv'
GRID, PX, M, N90 = 32, 16.0, 1024, 90
SEG_R = (1, 2, 3)
VOR_D = (2, 3, 4, 6)
log = lambda s: print(s, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------------------------- encode
@torch.no_grad()
def cmd_encode(a):
    out = Path(a.out) / 'patch_codes'
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # ~12 TFLOP in all: a CPU node is enough
    torch.backends.cuda.matmul.allow_tf32 = False
    log(f'device {dev}, {torch.get_num_threads()} threads')
    sae, norm, ck = load_sae(SAE, dev)
    meta = {k: ck.get(k) for k in ('fg_rule', 'motion_delta', 'bg_sub', 'align', 'encoder')}
    log(f'SAE {SAE.name}: {meta}, threshold {float(sae.threshold):.4f}')
    assert not meta['motion_delta'] and not meta['bg_sub'] and meta.get('align') in (None, 'none')
    store = FgTokenStore(a.store)
    stored = np.load(a.codes_max, mmap_mode='r')
    indptr, idx, val, pos, frow, fn = [np.zeros(1, np.int64)], [], [], [], [], []
    tot, chk = 0, dict(n=0, active_agree=0, max_abs=0.0, n_exact_set=0)
    t0 = time.time()
    for s, d in enumerate(store.dirs):
        z_ = np.load(d / 'frames.npz')
        rows, nfg = z_['rows'].astype(np.int64), z_['n_fg'].astype(np.int64)
        tok, ps, trow = store.tokens(s), store.pos(s), store.row(s)
        assert len(ps) == len(trow) == len(tok) == nfg.sum()
        assert (trow == np.repeat(rows, nfg)).all(), f'shard {s}: token rows do not match frames.npz'
        fid = torch.from_numpy(np.repeat(np.arange(len(rows)), nfg)).to(dev)
        fmax = torch.zeros(len(rows), M, device=dev)
        for c in range(0, len(tok), 262_144):
            x = torch.from_numpy(np.ascontiguousarray(tok[c:c + 262_144])).to(dev)
            z = sae.encode(norm(x), mode='threshold')
            fmax.index_reduce_(0, fid[c:c + len(z)], z, 'amax', include_self=True)
            nz = torch.nonzero(z)  # row-major: sorted by token, then latent
            cnt = torch.bincount(nz[:, 0], minlength=len(z))
            indptr.append((tot + torch.cumsum(cnt, 0)).cpu().numpy())
            idx.append(nz[:, 1].to(torch.int16).cpu().numpy())
            val.append(z[nz[:, 0], nz[:, 1]].half().cpu().numpy())
            tot += len(nz)
        pos.append(np.asarray(ps)); frow.append(rows); fn.append(nfg)
        ref = torch.from_numpy(np.asarray(stored[np.sort(rows)], dtype=np.float32)).to(dev)
        o = np.argsort(rows)
        mine = fmax.half().float()[torch.from_numpy(o).to(dev)]
        chk['n'] += len(rows)
        chk['active_agree'] += float(((mine > 0) == (ref > 0)).float().mean()) * len(rows)
        chk['n_exact_set'] += int(((mine > 0) == (ref > 0)).all(1).sum())
        chk['max_abs'] = max(chk['max_abs'], float((mine - ref).abs().max()))
        log(f'  shard {s}: {len(rows)} frames, {len(tok):,} tokens, nnz so far {tot:,} [{time.time() - t0:.0f}s]')
    indptr = np.concatenate(indptr)
    n_tok = len(indptr) - 1
    np.save(out / 'indptr.npy', indptr)
    np.save(out / 'idx.npy', np.concatenate(idx))
    np.save(out / 'val.npy', np.concatenate(val))
    np.save(out / 'pos.npy', np.concatenate(pos))
    np.save(out / 'frame_rows.npy', np.concatenate(frow))
    np.save(out / 'frame_nfg.npy', np.concatenate(fn))
    res = {'n_frames': chk['n'], 'n_tokens': n_tok, 'nnz': int(tot), 'l0_per_token': tot / n_tok,
           'check_vs_stored_codes_max': {
               'frac_frame_latent_firing_agrees': chk['active_agree'] / chk['n'],
               'frac_frames_identical_active_set': chk['n_exact_set'] / chk['n'],
               'max_abs_diff': chk['max_abs']},
           'sae': str(SAE), 'sae_meta': meta, 'elapsed_s': round(time.time() - t0, 1)}
    (out / 'encode.json').write_text(json.dumps(res, indent=1))
    log(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------------- analysis helpers
def auroc_cols(X, y):
    """AUROC of every column of X (n, m) for bool y, ties averaged (Mann-Whitney)."""
    from scipy.stats import rankdata
    R = rankdata(X, axis=0)
    P, N = y.sum(), (~y).sum()
    return (R[y].sum(0) - P * (P + 1) / 2) / (P * N)


def auc_ap(s, y):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return {'auroc': None, 'ap': None, 'n': int(len(y)), 'pos': int(y.sum())}
    return {'auroc': round(float(roc_auc_score(y, s)), 4), 'ap': round(float(average_precision_score(y, s)), 4),
            'n': int(len(y)), 'pos': int(y.sum())}


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.maximum(L2, 1e-9), 0.0).clip(0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def load_frames(pc, max_frames=0):
    """Per-frame table (store order): annotations row, obs, experiment, labels, tracking (grid units)."""
    rows = np.load(pc / 'frame_rows.npy')
    rows = rows[:max_frames] if max_frames else rows
    ann = pd.read_csv(ANN, usecols=['observation_id', 'frame_idx', 'frame_path', 'experiment', 'Y_Y2F', 'Y_B2F'])
    f = ann.iloc[rows].reset_index(drop=True)
    f['row'] = rows
    f['y2f'], f['b2f'] = f.pop('Y_Y2F').values > 0, f.pop('Y_B2F').values > 0
    cols = ['blue_x', 'blue_y', 'yellow_x', 'yellow_y', 'focal_x', 'focal_y', 'mark_blue_x', 'mark_blue_y',
            'mark_yellow_x', 'mark_yellow_y', 'raw_blue_x', 'raw_yellow_x', 'n_blobs']
    tr = []
    for (obs, ex), g in f.groupby(['observation_id', 'experiment'], sort=False):
        t = pd.read_csv(REPO / f'dataset/ants/{ex}/tracking/{obs}.csv').set_index('frame_idx')
        assert len(t) == 3000 and t.index.is_unique, obs
        tr.append(t.loc[g['frame_idx'].values, cols].set_index(g.index))
    tr = pd.concat(tr).loc[f.index]
    for c in cols:
        f[c] = tr[c].values
    for c in ('blue', 'yellow', 'focal'):
        f[f'{c[0]}x'], f[f'{c[0]}y'] = f[f'{c}_x'] / PX, f[f'{c}_y'] / PX
    for c in ('blue', 'yellow'):  # colour-dot (mark) positions: distinct even when the bodies merge into one blob
        f[f'm{c[0]}x'], f[f'm{c[0]}y'] = f[f'mark_{c}_x'] / PX, f[f'mark_{c}_y'] / PX
    f['good'] = f['raw_blue_x'].notna() & f['raw_yellow_x'].notna() & (f['n_blobs'] >= 2)
    f['ok'] = f[['bx', 'by', 'yx', 'yy', 'fx', 'fy', 'mbx', 'mby', 'myx', 'myy']].notna().all(1)
    f['d_yf'] = np.hypot(f.yx - f.fx, f.yy - f.fy)
    f['d_bf'] = np.hypot(f.bx - f.fx, f.by - f.fy)
    f['d_yb'] = np.hypot(f.yx - f.bx, f.yy - f.by)
    f['md_yf'] = np.hypot(f.myx - f.fx, f.myy - f.fy)
    f['md_bf'] = np.hypot(f.mbx - f.fx, f.mby - f.fy)
    f['md_yb'] = np.hypot(f.myx - f.mbx, f.myy - f.mby)
    f['tied'] = (f.d_yf - f.d_bf).abs() < 1  # body distances cannot tell the groomer (merged blob)
    return f


class Patches:
    """Sparse patch codes (CSR) + per-token frame / grid position."""

    def __init__(self, pc, nfg):
        nt = int(nfg.sum())  # nfg may be a prefix of the store's frames (--max-frames smoke test)
        self.indptr = np.load(pc / 'indptr.npy')[:nt + 1]
        self.idx = np.load(pc / 'idx.npy', mmap_mode='r')[:self.indptr[-1]].copy()
        self.val = np.load(pc / 'val.npy', mmap_mode='r')[:self.indptr[-1]].copy()
        pos = np.load(pc / 'pos.npy')[:nt].astype(np.int64)
        self.fid = np.repeat(np.arange(len(nfg)), nfg)
        self.px, self.py = pos % GRID + 0.5, pos // GRID + 0.5
        self.n_frames = len(nfg)
        self.ent_tok = np.repeat(np.arange(len(self.indptr) - 1), np.diff(self.indptr))
        self.ent_tok_t = torch.from_numpy(self.ent_tok)
        sel = self.idx == N90
        self.n90_tok, self.n90_val = self.ent_tok[sel], self.val[sel].astype(np.float32)

    def regmax_n90(self, mask):
        """(F,) max of neuron 90 over the tokens of mask (0 when none fire / none in region)."""
        out = np.zeros(self.n_frames, np.float32)
        k = mask[self.n90_tok]
        np.maximum.at(out, self.fid[self.n90_tok[k]], self.n90_val[k])
        return out

    def regmax_all(self, mask):
        """(F, 1024) max of every latent over the tokens of mask."""
        k = torch.from_numpy(mask)[self.ent_tok_t]
        tok = self.ent_tok_t[k]
        flat = torch.from_numpy(self.fid)[tok] * M + torch.from_numpy(self.idx[k.numpy()].astype(np.int64))
        out = torch.zeros(self.n_frames * M)
        out.scatter_reduce_(0, flat, torch.from_numpy(self.val[k.numpy()].astype(np.float32)), 'amax',
                            include_self=True)
        return out.view(self.n_frames, M).numpy()


def regions(P, f):
    """{name: (yellow token mask, blue token mask)} for every region definition and anchor ('body' = tracked body
    centroids of the coloured ants, 'mark' = their colour-dot positions; the focal is always its body centroid)."""
    T = lambda c: f[c].values[P.fid]  # noqa: E731
    fx, fy = T('fx'), T('fy')
    df = np.hypot(P.px - fx, P.py - fy)
    out = {}
    for anchor, pre in (('body', ''), ('mark', 'm')):
        yx, yy, bx, by = (T(pre + c) for c in ('yx', 'yy', 'bx', 'by'))
        sy, sb = seg_dist(P.px, P.py, yx, yy, fx, fy), seg_dist(P.px, P.py, bx, by, fx, fy)
        dy, db = np.hypot(P.px - yx, P.py - yy), np.hypot(P.px - bx, P.py - by)
        out.update({f'{anchor}_seg{r}': (sy <= r, sb <= r) for r in SEG_R})
        out.update({f'{anchor}_vor{d}': ((df <= d) & (dy < db), (df <= d) & (db < dy)) for d in VOR_D})
        out.update({f'{anchor}_segvor{r}': ((sy <= r) & (dy < db), (sb <= r) & (db < dy)) for r in SEG_R})
        out.update({f'{anchor}_disk{r}': (dy <= r, db <= r) for r in SEG_R})
    return out


def subsets(f):
    return {f'{e}|{q}': (f['ok'] & (f.experiment.isin(['v2', 'v3']) if e == 'pooled' else f.experiment == e)
                         & (f['good'] if q == 'good' else True)).values
            for e in ('v3', 'v2', 'pooled') for q in ('all', 'good')}


def key_numbers(ys, bs, f, m):
    """AUROC/AP of ys for Y2F, bs for B2F, and the discrimination AUROC of ys - bs among exactly-one frames."""
    y, b = f.y2f.values[m], f.b2f.values[m]
    one = y ^ b
    t = one & f.tied.values[m]
    return {'Y2F': auc_ap(ys[m], y), 'B2F': auc_ap(bs[m], b),
            'discrim_auroc': auc_ap((ys - bs)[m][one], y[one])['auroc'], 'n_exactly_one': int(one.sum()),
            'discrim_auroc_tied': auc_ap((ys - bs)[m][t], y[t])['auroc'], 'n_exactly_one_tied': int(t.sum())}


def cv_logistic(X, y, groups, kind='logit'):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups):
        mdl = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)) if kind == 'logit'
               else HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0))
        mdl.fit(X[tr], y[tr])
        p[te] = mdl.predict_proba(X[te])[:, 1]
    return p


# ---------------------------------------------------------------------------------------------- analyze
def cmd_analyze(a):
    out = Path(a.out)
    pc = out / 'patch_codes'
    t0 = time.time()
    f = load_frames(pc, a.max_frames)
    nfg = np.load(pc / 'frame_nfg.npy')[:len(f)]
    P = Patches(pc, nfg)
    log(f'{len(f):,} frames, {len(P.fid):,} tokens, {len(P.idx):,} nonzeros [{time.time() - t0:.0f}s]')
    res = {'n_frames': len(f), 'frames_by_experiment': f.experiment.value_counts().to_dict(),
           'frames_dropped_tracking_nan': int((~f.ok).sum())}
    # ---- geometry check: centroid patches should be foreground patches (and x/y not swapped)
    fgkey = np.sort(P.fid * 1024 + (P.py - .5).astype(np.int64) * GRID + (P.px - .5).astype(np.int64))
    geo = {}
    for c in ('y', 'b', 'f'):
        ok = f[f'{c}x'].notna().values
        fi = np.flatnonzero(ok)
        cx = np.clip(np.floor(f[f'{c}x'].values[ok]), 0, 31).astype(np.int64)
        cy = np.clip(np.floor(f[f'{c}y'].values[ok]), 0, 31).astype(np.int64)
        geo[c] = {'centroid_patch_is_fg': round(float(np.isin(fi * 1024 + cy * GRID + cx, fgkey).mean()), 4),
                  'xy_swapped_is_fg': round(float(np.isin(fi * 1024 + cx * GRID + cy, fgkey).mean()), 4)}
    res['geometry_check'] = geo
    log(f'geometry: {geo}')
    # ---- re-encoding check (frame-level max of neuron 90 vs stored)
    whole = P.regmax_n90(np.ones(len(P.fid), bool))
    stored = np.load(a.codes_max, mmap_mode='r')
    st90 = np.asarray(stored[:, N90], dtype=np.float32)[f.row.values]
    res['n90_frame_vs_stored'] = {'max_abs_diff': float(np.abs(whole - st90).max()),
                                  'firing_agree': float(((whole > 0) == (st90 > 0)).mean())}
    f['n90_frame'] = st90
    # ---- region reads of neuron 90
    R = regions(P, f)
    sc = {}
    for name, (my, mb) in R.items():
        sc[name] = (P.regmax_n90(my), P.regmax_n90(mb))
        f[f'n90_{name}_y'], f[f'n90_{name}_b'] = sc[name]
        f[f'nfg_{name}_y'] = np.bincount(P.fid[my], minlength=len(f))
        f[f'nfg_{name}_b'] = np.bincount(P.fid[mb], minlength=len(f))
    S = subsets(f)
    tab = {}
    for sname, m in S.items():
        tab[sname] = {name: key_numbers(ys, bs, f, m) for name, (ys, bs) in sc.items()}
        tab[sname]['baseline_distance'] = key_numbers(-f.d_yf.values, -f.d_bf.values, f, m)
        tab[sname]['baseline_distance_mark'] = key_numbers(-f.md_yf.values, -f.md_bf.values, f, m)
        tab[sname]['baseline_frame_n90'] = key_numbers(f.n90_frame.values, f.n90_frame.values, f, m)
        tab[sname]['n_frames'] = int(m.sum())
    res['region_reads_n90'] = tab
    res['empty_region_frac'] = {name: {c: round(float((f[f'nfg_{name}_{c}'][f.ok] == 0).mean()), 4) for c in 'yb'}
                                for name in R}
    crit = lambda n: tab['v3|all'][n]['discrim_auroc']  # noqa: E731
    best_seg = max((f'body_seg{r}' for r in SEG_R), key=crit)  # primary read, as specified
    fam = {k: max((n for n in R if n.startswith(k) and n[len(k):].isdigit()), key=crit)
           for k in [f'{a}_{x}' for a in ('body', 'mark') for x in ('seg', 'vor', 'segvor', 'disk')]}
    best_all = max(R, key=crit)
    for sname, m in S.items():  # geometry-only control: foreground patch count of the region, no latent at all
        for n in dict.fromkeys(fam.values()):
            tab[sname][f'baseline_regionsize_{n}'] = key_numbers(f[f'nfg_{n}_y'].values.astype(float),
                                                                 f[f'nfg_{n}_b'].values.astype(float), f, m)
    res['chosen_on_v3'] = {'primary_seg': best_seg, 'best_overall': best_all, 'best_per_family': fam}
    log(f'chosen on v3: primary {best_seg}, overall {best_all}, per family {fam} [{time.time() - t0:.0f}s]')
    for sname in S:
        log(f'  {sname}: ' + ' | '.join(
            f"{n} Y {tab[sname][n]['Y2F']['auroc']}/{tab[sname][n]['Y2F']['ap']} B {tab[sname][n]['B2F']['auroc']}/"
            f"{tab[sname][n]['B2F']['ap']} D {tab[sname][n]['discrim_auroc']}"
            + f" Dtied {tab[sname][n]['discrim_auroc_tied']}"
            for n in sorted(set(fam.values())) + ['baseline_distance', 'baseline_distance_mark', 'baseline_frame_n90',
                                                  f'baseline_regionsize_{best_all}']))
    # ---- all-neuron scan on the chosen regions (top 5 selected on v3|all, reported everywhere)
    scan = {}
    scan_sets = [best_seg] + [n for n in (best_all, fam['mark_seg']) if n != best_seg]
    scan_sets = list(dict.fromkeys(scan_sets))
    for name in scan_sets:
        Zy, Zb = P.regmax_all(R[name][0]), P.regmax_all(R[name][1])
        assert np.allclose(Zy[:, N90], sc[name][0]) and np.allclose(Zb[:, N90], sc[name][1])
        Zw = None
        aucs = {}
        for sname, m in S.items():
            y, b = f.y2f.values[m], f.b2f.values[m]
            one = y ^ b
            t = one & f.tied.values[m]
            aucs[sname] = {'Y2F': auroc_cols(Zy[m], y), 'B2F': auroc_cols(Zb[m], b),
                           'discrim': auroc_cols((Zy - Zb)[m][one], y[one]),
                           'discrim_tied': auroc_cols((Zy - Zb)[m][t], y[t])}
        if name == best_seg:  # frame-level reference scan (stored codes, same frames)
            Zw = np.asarray(stored[np.sort(f.row.values)], dtype=np.float32)[np.argsort(np.argsort(f.row.values))]
            assert np.allclose(Zw[:, N90], st90)
            m = S['v3|all']
            scan['frame_level_v3_all'] = {lab: sorted([(int(j), round(float(v), 4)) for j, v in enumerate(
                auroc_cols(Zw[m], f[col].values[m]))], key=lambda t: -t[1])[:5] for lab, col in
                (('Y2F', 'y2f'), ('B2F', 'b2f'))}
            del Zw
        sc_name = {}
        for task in ('Y2F', 'B2F', 'discrim', 'discrim_tied'):
            v = aucs['v3|all'][task]
            top = np.argsort(-v)[:5]
            sc_name[task] = {'top5_selected_on_v3_all': [
                {'neuron': int(j), **{s: round(float(aucs[s][task][j]), 4) for s in S}} for j in top],
                'neuron90': {s: round(float(aucs[s][task][N90]), 4) for s in S},
                'neuron90_rank_v3_all': int((v > v[N90]).sum()) + 1,
                'bottom1_v3_all': (int(np.argmin(v)), round(float(v.min()), 4))}
        # APs of the top neurons on each subset
        for task, (Z, col) in (('Y2F', (Zy, 'y2f')), ('B2F', (Zb, 'b2f'))):
            for e in sc_name[task]['top5_selected_on_v3_all']:
                e['ap'] = {s: auc_ap(Z[S[s], e['neuron']], f[col].values[S[s]])['ap'] for s in ('v3|all', 'v2|all')}
        scan[name] = sc_name
        del Zy, Zb
        log(f'scan {name}: ' + json.dumps({t: [(e['neuron'], e['v3|all'], e['v2|all']) for e in
                                               sc_name[t]['top5_selected_on_v3_all']] for t in sc_name})
            + f' [{time.time() - t0:.0f}s]')
    res['neuron_scan'] = scan
    # ---- baseline (c): distance features vs distance + neuron 90, grouped CV by video
    cvres = {}
    dc = ['d_yf', 'd_bf', 'd_yb', 'md_yf', 'md_bf', 'md_yb']
    dist = np.c_[f[dc].values, np.log(f[dc].values + .5), f.n_blobs.values]
    n90f = np.c_[f[f'n90_{best_seg}_y'], f[f'n90_{best_seg}_b'], f[f'n90_{best_all}_y'], f[f'n90_{best_all}_b'],
                 f.n90_frame]
    for sname in ('v3|all', 'v2|all', 'pooled|all', 'v3|good', 'v2|good', 'pooled|good'):
        m = S[sname]
        ex = (f.experiment.values == 'v3').astype(float)[:, None]
        g = f.observation_id.values[m]
        out_s = {}
        geo = np.c_[dist, f[[f'nfg_{n}_{c}' for n in dict.fromkeys([best_seg, best_all]) for c in 'yb']].values]
        for fs, X in (('distance', dist), ('distance+regionsize', geo), ('distance+regionsize+n90', np.c_[geo, n90f]),
                      ('n90_only', n90f)):
            X = np.c_[X, ex][m] if sname.startswith('pooled') else X[m]
            for kind in ('logit', 'hgb'):
                r = {}
                for task, col in (('Y2F', 'y2f'), ('B2F', 'b2f')):
                    yv = f[col].values[m]
                    r[task] = auc_ap(cv_logistic(X, yv, g, kind), yv)
                y, b = f.y2f.values[m], f.b2f.values[m]
                one = y ^ b
                pd_ = cv_logistic(X[one], y[one], g[one], kind)
                r['discrim_auroc'] = auc_ap(pd_, y[one])['auroc']
                t = f.tied.values[m][one]
                r['discrim_auroc_tied'] = auc_ap(pd_[t], y[one][t])['auroc']
                out_s[f'{fs}|{kind}'] = r
        cvres[sname] = out_s
        log(f'  CV {sname}: ' + ' | '.join(f"{k} Y {v['Y2F']['auroc']} B {v['B2F']['auroc']} D {v['discrim_auroc']} "
                                           f"Dt {v['discrim_auroc_tied']}"
                                           for k, v in out_s.items()) + f' [{time.time() - t0:.0f}s]')
    res['cv_models'] = cvres
    # ---- descriptive: groomer-focal distance when grooming (grid units)
    m = f.ok.values
    res['distance_when_grooming_median'] = {
        'd_yf|Y2F_only': float(np.median(f.d_yf[m & f.y2f & ~f.b2f])), 'd_bf|Y2F_only': float(np.median(f.d_bf[m & f.y2f & ~f.b2f])),
        'd_yf|B2F_only': float(np.median(f.d_yf[m & f.b2f & ~f.y2f])), 'd_bf|B2F_only': float(np.median(f.d_bf[m & f.b2f & ~f.y2f])),
        'd_yf|none': float(np.median(f.d_yf[m & ~f.y2f & ~f.b2f]))}
    res['tied_frac_of_exactly_one'] = {e: round(float(f.tied[(f.experiment == e) & f.ok & (f.y2f ^ f.b2f)].mean()), 4)
                                        for e in ('v2', 'v3')}
    res['good_frac'] = {e: round(float(f.good[f.experiment == e].mean()), 4) for e in ('v2', 'v3')}
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    keep = [c for c in f.columns if c.startswith(('n90_', 'nfg_', 'd_', 'md_'))] + ['tied', 'myx', 'myy', 'mbx', 'mby',
        'row', 'observation_id', 'frame_idx', 'experiment', 'y2f', 'b2f', 'good', 'ok', 'n_blobs', 'yx', 'yy', 'bx',
        'by', 'fx', 'fy']
    f[keep].to_parquet(out / 'frame_scores.parquet')
    for name in dict.fromkeys([best_seg, best_all]):
        contact_sheet(f, P, R[name], name, out)
    log(f'done [{time.time() - t0:.0f}s] -> {out}')


def contact_sheet(f, P, reg, name, out, n=8, seed=0):
    """8 v2 frames (r chosen on v3, so v2 is held out) where the region read says Y2F, 8 where it says B2F."""
    from PIL import Image, ImageDraw
    d = (f[f'n90_{name}_y'] - f[f'n90_{name}_b']).values
    ok = (f.experiment == 'v2').values & f.ok.values
    rng = np.random.default_rng(seed)
    picks = []
    for sign, lab in ((1, 'reads Y2F'), (-1, 'reads B2F')):
        s = sign * d
        cand = np.flatnonzero(ok & (s >= np.quantile(s[ok], 0.98)) & (s > 0))
        rng.shuffle(cand)
        seen, ch = set(), []
        for i in cand:
            if f.observation_id[i] not in seen:
                seen.add(f.observation_id[i]); ch.append(i)
            if len(ch) == n:
                break
        picks.append((lab, ch))
    my, mb = reg
    tile = 256
    sheet = Image.new('RGB', (n * tile, 2 * (tile + 28)), 'white')
    ann = pd.read_csv(ANN, usecols=['frame_path'])['frame_path'].values
    n90pos = {}
    k = P.idx == N90
    for t, v in zip(P.ent_tok[k], P.val[k]):
        fi = P.fid[t]
        if v > n90pos.get(fi, (0, -1))[0]:
            n90pos[fi] = (float(v), t)
    for r, (lab, ch) in enumerate(picks):
        for c, i in enumerate(ch):
            im = Image.open(REPO / 'dataset' / ann[f.row[i]]).convert('RGB').resize((512, 512))
            dr = ImageDraw.Draw(im)
            for mask, col in ((my, (255, 220, 0)), (mb, (0, 120, 255))):
                for t in np.flatnonzero(mask & (P.fid == i)):
                    x0, y0 = (P.px[t] - .5) * PX, (P.py[t] - .5) * PX
                    dr.rectangle([x0, y0, x0 + 15, y0 + 15], outline=col, width=2)
            if i in n90pos:
                t = n90pos[i][1]
                x0, y0 = (P.px[t] - .5) * PX, (P.py[t] - .5) * PX
                dr.rectangle([x0 - 2, y0 - 2, x0 + 17, y0 + 17], outline=(255, 0, 0), width=3)
            for c2, col in (('y', (255, 220, 0)), ('b', (0, 120, 255)), ('f', (255, 255, 255))):
                x, y = f[f'{c2}x'][i] * PX, f[f'{c2}y'][i] * PX  # body centroid: disc
                dr.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col, outline=(0, 0, 0))
                if c2 != 'f':  # colour-dot mark: small diamond
                    x, y = f[f'm{c2}x'][i] * PX, f[f'm{c2}y'][i] * PX
                    dr.polygon([(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)], fill=col, outline=(0, 0, 0))
            y0 = r * (tile + 28)
            sheet.paste(im.resize((tile, tile)), (c * tile, y0 + 28))
            ImageDraw.Draw(sheet).text((c * tile + 4, y0 + 2), f'{lab} | {f.observation_id[i]} f{f.frame_idx[i]} | truth '
                                       f'Y2F={int(f.y2f[i])} B2F={int(f.b2f[i])}', fill=(0, 0, 0))
            ImageDraw.Draw(sheet).text((c * tile + 4, y0 + 14), f'n90 Y {f[f"n90_{name}_y"][i]:.1f} B '
                                       f'{f[f"n90_{name}_b"][i]:.1f}', fill=(0, 0, 0))
    sheet.save(out / f'contact_sheet_{name}_v2.jpg', quality=88)
    pd.DataFrame([{'row_label': lab, 'obs': f.observation_id[i], 'frame_idx': int(f.frame_idx[i]),
                   'y2f': bool(f.y2f[i]), 'b2f': bool(f.b2f[i])} for lab, ch in picks for i in ch]).to_csv(
        out / f'contact_sheet_{name}_v2.csv', index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['encode', 'analyze'])
    p.add_argument('--store')
    p.add_argument('--codes-max')
    p.add_argument('--out', required=True)
    p.add_argument('--max-frames', type=int, default=0, help='analyze: first N store frames only (smoke test)')
    a = p.parse_args()
    {'encode': cmd_encode, 'analyze': cmd_analyze}[a.cmd](a)


if __name__ == '__main__':
    main()
