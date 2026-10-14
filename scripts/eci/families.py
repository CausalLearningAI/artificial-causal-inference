"""
NEURON FAMILIES for mice nose-nose / nose-tail (ECI, C1 SAEs): is a sniffing behaviour carried by a small FAMILY of
similar SAE neurons (feature splitting) rather than by one neuron?

SAEs (all [token_t, token_t - token_{t-5}] Matryoshka BatchTopK 4096, k 16, DINOv2-base fg448 tokens)
    C1dep   deployment SAE dataset/mice/v1/eci/sae/matryoshka_btk_4096_k16_fg448m5_s0 (token pool from ALL 432 videos,
            incl. the annotated eval videos -> not strictly held out). Frame values = the deployed codes_max.npy read at
            the eval rows; q = its readout_norm.npz (99th pct of positive read-outs on its 518,400 training frames).
    M5_s0-2 C1 validation SAEs results/vision/eci_t3_motion/mice/sae/M5_s* (trained on the 100 UNANNOTATED videos only).
            Frame values re-encoded here on the 172,800 annotated 1 fps eval frames (motion_t3.encode_codes, same
            staging as motion_t3 evaluate); q fitted here on a random 20,000 of their training frames (rng 0).

FAMILY RULE (fixed before any label was looked at; label-free)
    eligible  neuron fires (frame value > 0) on >= 1% of TRAINING frames.
    F_dec     (primary) agglomerative clustering of the eligible neurons, AVERAGE linkage on decoder cosine distance
              1 - cos(W_dec_i, W_dec_j) (full 1536-d decoder rows, unit norm), cut at distance 0.5 (mean pairwise
              cosine >= 0.5). Sensitivity cuts 0.3 / 0.7 reported, never used for the verdict.
    F_coact   (secondary) same clustering on 1 - Pearson correlation of the normalised frame values on the training
              frames, cut at 0.5.
    family    a cluster with >= 2 members; family frame value = max over members of a_norm = min(1, codes_max / q).
Scoring (labels only select and score; the levers / motion_t3 harness, sae_levers.score_model): the annotated POOLS
    are split into the two motion_t3 halves (asserted equal to results/vision/eci_t3_motion/mice/half.npy); on half A
    the best candidate is picked by AUROC (also by AP, and by top-1% precision among candidates firing on >= 1% of the
    half's frames), scored on half B, then swapped; reported = mean over the two directions. Candidate sets: 'single'
    (all 4096 neurons, raw codes = the harness numbers), 'single_norm' (a_norm), 'fam_dec', 'fam_coact'.
Conditional contact-type test (label-free contact proxies; test half only, each direction):
    sam   the neuron_map_v2 B subset (results/vision/eci_neuron_map_v2/mice: all 5,394 annotated-contact eval frames,
          weight 1, + 14,606 random others, weight 11.46): contact = SAM instances valid (4 masks) AND the largest
          touching-instance cluster holds >= 2 mice (prep.npz 'clus'). Weighted AUROC.
    blob  the neuron_map v1 subset (30,240 frames evenly spaced in the 144 annotated videos): contact = a dark blob of
          area >= 1.5 x A1 (A1 = 1985.5 px, one mouse; = the map's animal count >= 2). Unweighted AUROC.
    Within contact frames: nn (or nt) positives vs all other contact frames. Picks: (a) the all-frame AUROC picks above,
    (b) re-selected by AUROC within the contact frames of half A.
Video level: motion_t3.cmd_video measurements (diag_switch_regions.video_metrics = diag_video_level measures), raw and
    fg-count controlled, for the AUROC / top-1% picks of single and family.
Sheets: top-16 held-out frames (<= 2 per video) of the best nn / nt family (selected on half 0, frames of half 1), the
    member neurons' argmax patches outlined (one colour per member).

Steps: run (GPU: encode + analyse + sheets in one job, scripts/eci/families.sh); analyse (CPU rerun from saved codes);
    table (login safe).
Outputs: results/vision/eci_families/mice/{codes,sheets}/, results.json, table.md
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import motion_t3 as mt  # noqa: E402
import multiscale_sae as ms  # noqa: E402
import sae_levers as sl  # noqa: E402
from src.eci.readout_norm import fit_norm, load_norm  # noqa: E402
from src.eci.sae import load_sae  # noqa: E402

OUT = REPO / 'results/vision/eci_families/mice'
T3 = REPO / 'results/vision/eci_t3_motion/mice'
DEPLOY = REPO / 'dataset/mice/v1/eci/sae/matryoshka_btk_4096_k16_fg448m5_s0'
DEPLOY_CODES = REPO / 'dataset/mice/v1/eci/codes/matryoshka_btk_4096_k16_fg448m5_s0/codes_max.npy'
NMV2 = REPO / 'results/vision/eci_neuron_map_v2/mice'
NMV1 = REPO / 'results/vision/eci_neuron_map/mice'
A1_MICE = 1985.5
SAES = {'C1dep': DEPLOY / 'sae.pt', 'M5_s0': T3 / 'sae/M5_s0/sae.pt', 'M5_s1': T3 / 'sae/M5_s1/sae.pt',
        'M5_s2': T3 / 'sae/M5_s2/sae.pt'}
ELIG = 0.01
CUT_DEC, CUT_COACT = 0.5, 0.5
SENS_CUTS = (0.3, 0.7)
N_TRAIN_SUB = 20_000
LABELS = ['nose_nose', 'nose_tail']
SHORT = {'nose_nose': 'nn', 'nose_tail': 'nt'}
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------------------------- layout
def eval_layout():
    idx, tr, ev, _ = mt.layout('mice')
    lab, _, valid = ms.eval_labels('mice', idx, ev)
    unit = lab.obs.str.rsplit('_', n=2).str[0].values
    units = sorted(set(unit))
    half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
    assert np.array_equal(half, np.load(T3 / 'half.npy')), 'halves differ from motion_t3'
    return idx, tr, ev, lab, valid, half


def read_rows(path, rows):
    """codes_max rows (any order) read as contiguous runs of the sorted rows (gaps <= 64 rows read through, so the
    NFS reads are few and sequential) -> (n, m) fp16 in the order of `rows`."""
    mm = np.load(path, mmap_mode='r')
    order = np.argsort(rows, kind='stable')
    rs = rows[order]
    out = np.empty((len(rows), mm.shape[1]), np.float16)
    brk = np.flatnonzero(np.diff(rs) > 64) + 1
    for a, b in zip(np.r_[0, brk], np.r_[brk, len(rs)]):
        lo, hi = int(rs[a]), int(rs[b - 1]) + 1
        blk = np.asarray(mm[lo:hi])
        out[order[a:b]] = blk[rs[a:b] - lo]
    return out


# ---------------------------------------------------------------------------------------------- encode
@torch.no_grad()
def cmd_encode(args):
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    loc = ms.local_dir()
    cd = OUT / 'codes'
    cd.mkdir(parents=True, exist_ok=True)
    idx, tr, ev, lab, valid, half = eval_layout()
    rows = lab['row'].values
    _, pos, lens = ms.stage(idx, ev, loc / 'eval')
    assert np.array_equal(lens, np.load(T3 / 'lens.npy')), 'eval fg counts differ from motion_t3'
    tok = np.load(loc / 'eval' / 'tok.npy')
    shutil.rmtree(loc / 'eval')
    nb = {-5: mt.stage_prev5(idx, 'mice', ev, loc / 'nbr-5_eval.npy')}
    rng = np.random.default_rng(0)
    trs = np.sort(rng.choice(tr, N_TRAIN_SUB, replace=False))
    _, tpos, tlens = ms.stage(idx, trs, loc / 'train')
    ttok = np.load(loc / 'train' / 'tok.npy')
    tnb = {-5: mt.stage_prev5(idx, 'mice', trs, loc / 'nbr-5_train.npy')}
    np.save(cd / 'eval_lens.npy', lens.astype(np.int32))
    np.save(cd / 'train_frames.npy', trs)
    np.save(cd / 'train_rows.npy', idx.rows[trs])
    info = {'n_eval': len(ev), 'n_eval_tokens': int(lens.sum()), 'n_train_sub': len(trs), 'train_sub_tokens': int(tlens.sum())}
    for key, pth in SAES.items():
        t0 = time.time()
        sae, norm, _ = load_sae(pth, dev)
        codes, st = mt.encode_codes(sae, norm, 'M5', tok, nb, lens, dev)
        c_ev = codes.cpu().numpy()
        del codes
        codes, _ = mt.encode_codes(sae, norm, 'M5', ttok, tnb, tlens, dev)
        c_tr = codes.cpu().numpy()
        del codes
        torch.cuda.empty_cache()
        e = {'fve': st['fve'], 'l0': st['l0_per_token'], 'theta': float(sae.threshold)}
        if key == 'C1dep':
            # deployed read-outs at the same rows; the re-encoding is only used for the sheets' argmax patches
            dep = read_rows(DEPLOY_CODES, rows)
            a, b = dep.astype(np.float32), c_ev.astype(np.float32)
            e['reencode_vs_deployed'] = {
                'frac_entries_equal_within_1e-2': float((np.abs(a - b) <= 1e-2 * np.maximum(1, a)).mean()),
                'frac_frames_all_equal': float((np.abs(a - b) <= 1e-2 * np.maximum(1, a)).all(1).mean()),
                'fire_agreement': float(((a > 0) == (b > 0)).mean()),
                'corr_nonzero': float(np.corrcoef(a[(a > 0) | (b > 0)], b[(a > 0) | (b > 0)])[0, 1])}
            np.save(loc / 'C1dep_reenc_eval.npy', c_ev)
            c_ev = dep
            trows = idx.rows[trs]
            c_tr = read_rows(DEPLOY_CODES, trows)
            log(f'  C1dep re-encode vs deployed: {e["reencode_vs_deployed"]}')
        for nm, arr in ((f'{key}_eval.npy', c_ev), (f'{key}_train.npy', c_tr)):
            if (cd / nm).exists() and key == 'C1dep':  # written by encode_dep (same values); never rewrite under a reader
                assert np.array_equal(np.load(cd / nm, mmap_mode='r')[:500], arr[:500]), nm
                continue
            np.save(loc / nm, arr)
            shutil.move(str(loc / nm), str(cd / nm))
        info[key] = e
        log(f'  {key}: eval {c_ev.shape} train {c_tr.shape} FVE {st["fve"]:.4f} [{time.time() - t0:.0f}s]')
    (OUT / 'encode.json').write_text(json.dumps(info, indent=1))


def cmd_encode_dep(args):
    """CPU: the deployment SAE's inputs only (deployed codes_max at the eval rows and at the training-subset rows)."""
    cd = OUT / 'codes'
    cd.mkdir(parents=True, exist_ok=True)
    idx, tr, ev, lab, valid, half = eval_layout()
    rows = lab['row'].values
    lens = idx.nfg[ev].astype(np.int64)
    assert np.array_equal(lens, np.load(T3 / 'lens.npy'))
    rng = np.random.default_rng(0)
    trs = np.sort(rng.choice(tr, N_TRAIN_SUB, replace=False))
    np.save(cd / 'eval_lens.npy', lens.astype(np.int32))
    np.save(cd / 'train_frames.npy', trs)
    np.save(cd / 'train_rows.npy', idx.rows[trs])
    t0 = time.time()
    np.save(cd / 'C1dep_eval.npy', read_rows(DEPLOY_CODES, rows))
    np.save(cd / 'C1dep_train.npy', read_rows(DEPLOY_CODES, idx.rows[trs]))
    log(f'C1dep codes read in {time.time() - t0:.0f}s')


# ---------------------------------------------------------------------------------------------- families
def clusters(D, cut):
    """average-linkage agglomerative clusters on a square distance matrix, cut at `cut` -> label per row."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = np.clip((D + D.T) / 2, 0, 2)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method='average')
    return fcluster(Z, t=cut, criterion='distance')


def families_of(lbl, members_idx):
    fam = {}
    for k, j in zip(lbl, members_idx):
        fam.setdefault(int(k), []).append(int(j))
    return sorted([sorted(v) for v in fam.values() if len(v) >= 2], key=lambda v: (-len(v), v[0]))


def build_families(sae_path, q, rate, c_tr):
    sae, _, _ = load_sae(sae_path, 'cpu')
    W = sae.W_dec.detach().float().numpy()
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    el = np.flatnonzero(rate >= ELIG)
    cos = W[el] @ W[el].T
    a_tr = np.minimum(1.0, c_tr[:, el].astype(np.float32) / q[el][None])
    a_tr = a_tr - a_tr.mean(0)
    sd = np.sqrt((a_tr ** 2).sum(0))
    corr = (a_tr.T @ a_tr) / np.maximum(np.outer(sd, sd), 1e-12)
    off = ~np.eye(len(el), dtype=bool)
    F = {'fam_dec': families_of(clusters(1 - cos, CUT_DEC), el),
         'fam_coact': families_of(clusters(1 - corr, CUT_COACT), el)}
    sens = {}
    for c in SENS_CUTS:
        sens[f'fam_dec_cut{c}'] = families_of(clusters(1 - cos, c), el)
    stats = {'n_eligible': len(el),
             'decoder_cos_offdiag_quantiles': np.quantile(cos[off], [0.5, 0.9, 0.99, 0.999]).tolist(),
             'max_offdiag_cos_per_neuron_median': float(np.median(np.where(off, cos, -1).max(1))),
             'coact_corr_offdiag_quantiles': np.quantile(corr[off], [0.5, 0.9, 0.99, 0.999]).tolist()}
    for k, fl in list(F.items()) + list(sens.items()):
        sz = [len(f) for f in fl]
        stats[k] = {'n_families': len(fl), 'n_neurons_in_families': int(sum(sz)),
                    'size_hist': {str(s): int(n) for s, n in zip(*np.unique(sz, return_counts=True))} if sz else {}}
    return F, sens, stats, W


def fam_values(A, fams, dev, chunk=20_000):
    """A (n, 4096) fp16 GPU a_norm -> (n, n_fam) fp16 GPU, family value = max over members."""
    out = torch.zeros(A.shape[0], len(fams), dtype=torch.float16, device=dev)
    for i, f in enumerate(fams):
        out[:, i] = A[:, torch.tensor(f, device=dev)].max(1).values
    return out


def picks(res, b):
    """cross-fitted picks per direction -> list of dict(select_half, test_half, auc/ap/top1 -> neuron col + test)."""
    return res[b]['dirs']


def summarise(res, b):
    r = res[b]
    return {'cf_auroc': r['cf_auroc'], 'cf_ap_at_auroc': r['cf_ap_at_auroc'], 'cf_top1_at_auroc': r['cf_top1_at_auroc'],
            'cf_ap': r['cf_ap'], 'cf_top1': r['cf_top1'], 'cf_size_ctrl_auroc': r['cf_size_ctrl_auroc'],
            'insample_auroc': r['insample_auroc'], 'insample_ap': r['insample_ap'], 'base_rate': r['base_rate'],
            'cols': {sel: [d[sel]['neuron'] for d in r['dirs']] for sel in ('auc', 'ap', 'top1')}}


# ---------------------------------------------------------------------------------------------- conditional
NB = 0x7C01  # non-negative finite fp16 bit patterns (src/eci/levers.py)


@torch.no_grad()
def wauc_cols(X, y, w, block=512):
    """Weighted AUROC (ties averaged) of every column of a non-negative fp16 tensor X (n, m) vs bool y (n,), weights
    w (n,) float64; the fp16-bit-histogram method of src/eci/levers.align_gpu with weighted counts -> (m,) numpy."""
    dev = X.device
    y = torch.as_tensor(y, device=dev)
    w = torch.as_tensor(w, dtype=torch.float64, device=dev)
    P, N = float(w[y].sum()), float(w[~y].sum())
    m = X.shape[1]
    out = np.full(m, np.nan)
    if P == 0 or N == 0:
        return out
    for j0 in range(0, m, block):
        xb = X[:, j0:j0 + block]
        b = xb.shape[1]
        bits = (xb.contiguous().view(torch.int16).long() & 0x7FFF) + NB * torch.arange(b, device=dev)[None]
        wr = w[:, None].expand(-1, b)
        tot = torch.bincount(bits.flatten(), weights=wr.flatten(), minlength=b * NB).view(b, NB).flip(1)
        pos = torch.bincount(bits[y].flatten(), weights=wr[y].flatten(), minlength=b * NB).view(b, NB).flip(1)
        neg = tot - pos
        cfp = neg.cumsum(1)
        out[j0:j0 + b] = ((pos * (N - cfp) + 0.5 * pos * neg).sum(1) / (P * N)).cpu().numpy()
    return out


def wauc(y, s, w):
    from sklearn.metrics import roc_auc_score
    if y.all() or (~y).all():
        return float('nan')
    return float(roc_auc_score(y, s, sample_weight=w))


def contact_sets(lab):
    """-> {name: (eval index array, weights)} of the label-free contact frames + coverage stats."""
    rpos = pd.Series(np.arange(len(lab)), index=lab['row'].values)
    out, st = {}, {}
    F2 = pd.read_parquet(NMV2 / 'frames.parquet')
    P2 = np.load(NMV2 / 'prep.npz')
    i2 = rpos.loc[F2.row.values].values
    assert (lab.obs.values[i2] == F2.obs.values).all()
    c2 = P2['valid'] & (P2['clus'] >= 2)
    out['sam'] = (i2[c2], F2.w.values[c2].astype(np.float64), i2[P2['valid']], F2.w.values[P2['valid']].astype(np.float64))
    F1 = pd.read_parquet(NMV1 / 'frames.parquet')
    B1 = np.load(NMV1 / 'blobs.npz')
    i1 = rpos.loc[F1.row.values].values
    c1 = B1['areas'].max(1) / A1_MICE >= 1.5
    out['blob'] = (i1[c1], np.ones(c1.sum()), i1, np.ones(len(i1)))
    for k, (ic, wc, ia, wa) in out.items():
        st[k] = {'n_frames_subset': int(len(ia)), 'n_contact': int(len(ic)),
                 'weighted_contact_share': float(wc.sum() / wa.sum())}
        for b in LABELS:
            y_a, y_c = lab[b].values[ia], lab[b].values[ic]
            st[k][b] = {'pos_rate_in_contact_w': float((wc * y_c).sum() / wc.sum()),
                        'share_of_positives_in_contact_w': float(
                            (wc * y_c).sum() / max((wa * y_a).sum(), 1e-12)),
                        'n_pos_in_contact': int(y_c.sum())}
    return out, st


def conditional(V, lab, half, cset, res_b, b, elig_cols):
    """V: (n, m) fp16 numpy candidate values. -> per direction: (a) all-frame AUROC pick, (b) contact re-selection."""
    ic, wc = cset[0], cset[1]
    y = lab[b].values
    D = []
    for d in res_b['dirs']:
        s, t = d['select_half'], d['test_half']
        ms_, mt_ = half[ic] == s, half[ic] == t
        it, wt = ic[mt_], wc[mt_]
        j = d['auc']['neuron']
        e = {'test_half': t, 'allframe_pick': j,
             'allframe_pick_auc': wauc(y[it], V[it, j].astype(np.float32), wt)}
        isel, wsel = ic[ms_], wc[ms_]
        sc = np.full(V.shape[1], -np.inf)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Xs = torch.from_numpy(np.ascontiguousarray(V[isel][:, elig_cols])).to(dev)
        sc[elig_cols] = wauc_cols(Xs, y[isel], wsel)
        del Xs
        sc[~np.isfinite(sc)] = -np.inf
        jj = int(np.argmax(sc))
        e.update(contact_pick=jj, contact_pick_select_auc=float(sc[jj]),
                 contact_pick_auc=wauc(y[it], V[it, jj].astype(np.float32), wt), n_test=int(len(it)),
                 n_test_pos=int(y[it].sum()))
        D.append(e)
    return {'dirs': D, 'allframe_pick_auc': float(np.nanmean([e['allframe_pick_auc'] for e in D])),
            'contact_pick_auc': float(np.nanmean([e['contact_pick_auc'] for e in D]))}


# ---------------------------------------------------------------------------------------------- video level
def video_level(lab, half, lens):
    """-> run(x_by_dir, b): x_by_dir = {test_half: per-frame values (n,)} -> raw / fg-controlled video_metrics
    (motion_t3.cmd_video)."""
    import diag_switch_regions as dsr
    from src.eci.domain import MiceDomain
    A = dsr.design()
    vi = {o: i for i, o in enumerate(A.observation_id)}
    fv = lab.obs.map(vi).values.astype(np.int64)
    nfr = pd.read_csv(dsr.ANN, usecols=['observation_id'])['observation_id'].value_counts()
    wstart = MiceDomain().window_start(nfr.loc[A.observation_id].values, A.stage.values)
    inwin = lab.frame_idx.values >= wstart[fv]
    assert (np.bincount(fv[inwin], minlength=len(A)) == 900).all()
    w = inwin.astype(np.float64)
    vh = pd.Series(half).groupby(fv).first()
    assert (vh.values == A.half.values).all()
    vmean = lambda fr: np.bincount(fv, weights=np.asarray(fr, np.float64) * w, minlength=len(A)) / 900  # noqa: E731
    fg = vmean(lens.astype(np.float64))

    def resid(x):
        o = np.empty_like(x)
        for h in (0, 1):
            m = (A.half == h).values
            X_ = np.c_[np.ones(m.sum()), fg[m]]
            o[m] = x[m] - X_ @ np.linalg.lstsq(X_, x[m], rcond=None)[0]
        return o

    def run(x_by_dir, b):
        x = np.full(len(A), np.nan)
        for t, fr in x_by_dir.items():
            m = (A.half == t).values
            x[m] = vmean(fr)[m]
        assert np.isfinite(x).all()
        keep = lambda r: {k: r[k] for k in ('video_r_rate', 'video_r_bpm', 'delta_r_pooled', 'delta_r_pooled_bpm')}  # noqa: E731
        return {'raw': keep(dsr.video_metrics(x, A, b)[0]), 'fg_controlled': keep(dsr.video_metrics(resid(x), A, b)[0])}
    return run


# ---------------------------------------------------------------------------------------------- analyse
def cmd_analyse(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cd = OUT / 'codes'
    idx, tr, ev, lab, valid, half = eval_layout()
    lens = np.load(cd / 'eval_lens.npy').astype(np.int64)
    rank = np.argsort(np.argsort(lens + np.random.default_rng(0).random(len(lens)) * 1e-3))
    decile = np.minimum(rank * 10 // len(lens), 9)
    Yt = torch.from_numpy(lab[LABELS].values.astype(bool)).to(dev)
    cs, cstats = contact_sets(lab)
    R = {'rule': {'eligible': f'fires on >= {ELIG:.0%} of training frames', 'fam_dec': f'average linkage, 1 - decoder '
                  f'cosine, cut {CUT_DEC}', 'fam_coact': f'average linkage, 1 - Pearson(a_norm on training frames), cut '
                  f'{CUT_COACT}', 'value': 'max over members of min(1, codes_max / q99)'},
         'base_rates': {b: float(lab[b].mean()) for b in LABELS}, 'contact_sets': cstats, 'saes': {}}
    log(f'base rates {R["base_rates"]}; contact sets {json.dumps(cstats)}')
    keys = args.keys or list(SAES)
    if (OUT / 'results.json').exists() and args.keys:
        R['saes'] = {k: v for k, v in json.loads((OUT / 'results.json').read_text())['saes'].items() if k not in keys}
    vrun = video_level(lab, half, lens)
    for key in keys:
        pth = SAES[key]
        t0 = time.time()
        c_ev = np.load(cd / f'{key}_eval.npy')
        c_tr = np.load(cd / f'{key}_train.npy')
        if key == 'C1dep':
            nz = load_norm(DEPLOY)
            q, rate = nz['q'].astype(np.float32), nz['rate'].astype(np.float32)
        else:
            sae_, _, _ = load_sae(pth, 'cpu')
            q, rate, _, _ = fit_norm(c_tr, float(sae_.threshold))
        F, sens, fst, W = build_families(pth, q, rate, c_tr)
        el = np.flatnonzero(rate >= ELIG)
        X = torch.from_numpy(c_ev).to(dev)
        qt = torch.from_numpy(q).to(dev)
        A = torch.minimum(torch.ones((), device=dev, dtype=torch.float32), X.float() / qt[None]).half()
        cand = {'single': X, 'single_norm': A}
        for k, fl in F.items():
            cand[k] = fam_values(A, fl, dev)
        E = {'family_stats': fst, 'families': F, 'sensitivity_families': sens, 'q': q.tolist(), 'rate': rate.tolist(), 'q_source': 'readout_norm.npz' if key == 'C1dep' else
             f'fit_norm on {N_TRAIN_SUB} training frames', 'sets': {}}
        for cname, V in cand.items():
            res, _ = sl.score_model(V, Yt, LABELS, valid, half, decile, dev)
            Vn = V.cpu().numpy()
            members = (lambda c: [c]) if cname.startswith('single') else (lambda c, fl=F[cname]: fl[c])
            elig_cols = el if cname.startswith('single') else np.arange(V.shape[1])
            S = {}
            for b in LABELS:
                s = summarise(res, b)
                s['members'] = {sel: [members(c) for c in cols] for sel, cols in s['cols'].items()}
                s['dirs'] = res[b]['dirs']
                s['conditional'] = {k: conditional(Vn, lab, half, cs[k], res[b], b, elig_cols) for k in cs}
                vl = {}
                for sel in ('auc', 'top1'):
                    xb = {d['test_half']: Vn[:, d[sel]['neuron']] for d in res[b]['dirs']}
                    vl[sel] = vrun(xb, b)
                s['video'] = vl
                S[b] = s
            E['sets'][cname] = S
            log(f'  {key} {cname} ({V.shape[1]} cand): ' + ' | '.join(
                f"{SHORT[b]} AUROC {S[b]['cf_auroc']:.3f} AP@auc {S[b]['cf_ap_at_auroc']:.4f} top1@auc "
                f"{S[b]['cf_top1_at_auroc']:.3f} AP {S[b]['cf_ap']:.4f} top1 {S[b]['cf_top1']:.3f} | contact sam "
                f"{S[b]['conditional']['sam']['allframe_pick_auc']:.3f}/{S[b]['conditional']['sam']['contact_pick_auc']:.3f}"
                f" blob {S[b]['conditional']['blob']['allframe_pick_auc']:.3f}/"
                f"{S[b]['conditional']['blob']['contact_pick_auc']:.3f}" for b in LABELS))
        R['saes'][key] = E
        log(f'  {key}: {fst["n_eligible"]} eligible, fam_dec {fst["fam_dec"]}, fam_coact {fst["fam_coact"]} '
            f'[{time.time() - t0:.0f}s]')
        del X, A, cand
        torch.cuda.empty_cache()
        (OUT / 'results.json').write_text(json.dumps(R, indent=1, default=float))
    cmd_table(args)


# ---------------------------------------------------------------------------------------------- greedy (secondary)
K_MAX = 8


@torch.no_grad()
def cmd_greedy(args):
    """SECONDARY, added after the primary C1dep table was seen (does not change the pre-registered verdict).
    (i) ANCHOR families: the half-A best single neuron (by AP, a_norm, eligible) + its k - 1 nearest eligible neurons by
        decoder cosine (label-free given the anchor), k = 1..8; value = max of a_norm; scored on half B.
    (ii) GREEDY families (label-selected UPPER BOUND of any max-family): on half A, start from the best-AP eligible
        neuron and add, k = 2..8, the eligible neuron maximising half-A AP of the running max; scored on half B.
    Both directions; mean reported."""
    from src.eci.levers import align_gpu
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R = json.loads((OUT / 'results.json').read_text())
    idx, tr, ev, lab, valid, half = eval_layout()
    Y = torch.from_numpy(lab[LABELS].values.astype(bool)).to(dev)
    (OUT / 'greedy').mkdir(exist_ok=True)
    for key, E in R['saes'].items():
        if args.keys and key not in args.keys:
            continue
        t0 = time.time()
        q = (load_norm(DEPLOY)['q'] if key == 'C1dep' else np.array(E['q'])).astype(np.float32)
        rate = (load_norm(DEPLOY)['rate'] if key == 'C1dep' else np.array(E['rate'])).astype(np.float32)
        el = np.flatnonzero(rate >= ELIG)
        c_ev = np.load(OUT / 'codes' / f'{key}_eval.npy', mmap_mode='r')
        A = torch.from_numpy(np.minimum(1.0, np.asarray(c_ev[:, el], np.float32) / q[el][None]).astype(np.float16)).to(dev)
        sae, _, _ = load_sae(SAES[key], 'cpu')
        W = sae.W_dec.detach().float().numpy()[el]
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        cos = W @ W.T
        out = {}
        for li, b in enumerate(LABELS):
            out[b] = {'anchor': [], 'greedy': []}
            for s_, t_ in ((0, 1), (1, 0)):
                rs = torch.from_numpy(np.flatnonzero(half == s_)).to(dev)
                rt = torch.from_numpy(np.flatnonzero(half == t_)).to(dev)
                sc = align_gpu(A, Y[:, [li]], rows=rs)
                j0 = int(np.nanargmax(np.nan_to_num(sc['ap'][:, 0], nan=-1)))
                test = lambda cols: {k: float(v[0, 0]) for k, v in align_gpu(  # noqa: E731
                    A[:, cols].max(1, keepdim=True).values, Y[:, [li]], rows=rt).items() if k in ('auc', 'ap', 'prec_top')}
                nbr = np.argsort(-cos[j0])
                nbr = [j0] + [int(j) for j in nbr if j != j0][:K_MAX - 1]
                out[b]['anchor'].append([{'k': k, 'members': [int(el[j]) for j in nbr[:k]],
                                          'cos_to_anchor': [float(cos[j0, j]) for j in nbr[:k]], **test(nbr[:k])}
                                         for k in range(1, K_MAX + 1)])
                fam, cur, steps = [j0], A[:, j0].clone(), []
                steps.append({'k': 1, 'members': [int(el[j0])], 'select_ap': float(sc['ap'][j0, 0]), **test([j0])})
                for k in range(2, K_MAX + 1):
                    M = torch.maximum(A, cur[:, None])
                    r = align_gpu(M, Y[:, [li]], rows=rs)['ap'][:, 0]
                    r[fam] = -1
                    j = int(np.nanargmax(np.nan_to_num(r, nan=-1)))
                    fam.append(j)
                    cur = torch.maximum(cur, A[:, j])
                    del M
                    steps.append({'k': k, 'members': [int(el[x]) for x in fam], 'select_ap': float(r[j]), **test(fam)})
                out[b]['greedy'].append(steps)
            for kind in ('anchor', 'greedy'):
                out[b][f'{kind}_mean'] = [{m: float(np.mean([d[k][m] for d in out[b][kind]])) for m in ('auc', 'ap', 'prec_top')}
                                          for k in range(K_MAX)]
            log(f'  {key} {SHORT[b]}: anchor k=1/2/4/8 AP ' + '/'.join(f"{out[b]['anchor_mean'][k]['ap']:.4f}" for k in (0, 1, 3, 7))
                + ' top1 ' + '/'.join(f"{out[b]['anchor_mean'][k]['prec_top']:.3f}" for k in (0, 1, 3, 7))
                + ' | greedy k=1/2/4/8 AP ' + '/'.join(f"{out[b]['greedy_mean'][k]['ap']:.4f}" for k in (0, 1, 3, 7))
                + ' top1 ' + '/'.join(f"{out[b]['greedy_mean'][k]['prec_top']:.3f}" for k in (0, 1, 3, 7))
                + ' AUROC ' + '/'.join(f"{out[b]['greedy_mean'][k]['auc']:.3f}" for k in (0, 1, 3, 7)))
        (OUT / 'greedy' / f'{key}.json').write_text(json.dumps(out, indent=1))
        del A
        log(f'  {key} greedy done [{time.time() - t0:.0f}s]')


# ---------------------------------------------------------------------------------------------- sheets
COLS = [(255, 0, 0), (0, 160, 255), (255, 200, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 255, 0), (255, 255, 255), (128, 128, 128)]


@torch.no_grad()
def cmd_sheets(args):
    """Top-16 held-out frames of the best nn / nt family of every SAE x family set (from results.json + saved codes);
    tokens of the shown frames fetched one by one from the stores (base store + d5 prev), so no staging is needed."""
    from PIL import Image, ImageDraw
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R = json.loads((OUT / 'results.json').read_text())
    idx, tr, ev, lab, valid, half = eval_layout()
    d5 = mt.D5Matcher(idx, 'mice')
    sd = OUT / 'sheets'
    sd.mkdir(parents=True, exist_ok=True)
    paths_all = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv', usecols=['frame_path'])['frame_path'].values
    c = 512 // 32
    index = []
    for key, E in R['saes'].items():
        if args.keys and key not in args.keys:
            continue
        sae, norm, _ = load_sae(SAES[key], dev)
        c_ev = np.load(OUT / 'codes' / f'{key}_eval.npy', mmap_mode='r')
        q = (load_norm(DEPLOY)['q'] if key == 'C1dep' else np.array(E['q'])).astype(np.float32)
        for fk in ('fam_dec', 'fam_coact'):
            F = E['families'][fk]
            for b in LABELS:
                d = [d_ for d_ in E['sets'][fk][b]['dirs'] if d_['select_half'] == 0][0]
                mem = F[d['auc']['neuron']]
                # ranked by the UNCLIPPED max of codes / q (a_norm saturates at 1, which would leave the top frames tied)
                x = (np.asarray(c_ev[:, mem], np.float32) / q[mem][None]).max(1)
                x = np.where(half == 1, x, -1)
                pick, seen = [], {}
                for i in np.argsort(-x, kind='stable'):
                    if x[i] <= 0 or len(pick) == 16:
                        break
                    v = lab.obs.iat[i]
                    if seen.get(v, 0) >= 2:
                        continue
                    seen[v] = seen.get(v, 0) + 1
                    pick.append(int(i))
                tw, th = 256, 256 + 14
                sheet = Image.new('RGB', (4 * tw, 44 + 4 * th), 'white')
                dr = ImageDraw.Draw(sheet)
                dr.text((4, 2), f'{key} {fk}: best {SHORT[b]} family (AUROC pick on half 0), top frames of half 1 '
                                f'(<= 2 per video; ranked by unclipped max codes/q). Members: {mem[:24]}{"..." if len(mem) > 24 else ""}', fill='black')
                dr.text((4, 16), 'outline colours (first 10 members, in order): red, blue, yellow, magenta, cyan, '
                                 'orange, purple, green, white, grey', fill='black')
                dr.text((4, 30), f'half-1 test: AUROC {d["auc"]["test_auc"]:.3f} AP {d["auc"]["test_ap"]:.4f} '
                                 f'top1 {d["auc"]["test_top1"]:.3f}; caption green = {SHORT[b]} positive', fill='black')
                n_pos = {bb: 0 for bb in LABELS}
                fired = []
                for qn, i in enumerate(pick):
                    t, p_, ln = idx.load(np.array([ev[i]]))
                    pv, _, _ = d5.fetch(np.array([ev[i]]))
                    T = torch.from_numpy(t).to(dev).float()
                    T = torch.cat([T, T - torch.from_numpy(pv).to(dev).float()], 1)
                    z = sae.encode(norm(T), mode='threshold')[:, torch.tensor(mem, device=dev)].cpu().numpy()
                    im = Image.open(REPO / 'dataset' / paths_all[int(lab['row'].iat[i])]).convert('RGB')
                    d2 = ImageDraw.Draw(im)
                    fr_ = [int(j) for k, j in enumerate(mem) if z[:, k].max() > 0]
                    fired.append(fr_)
                    for k, j in enumerate(mem):
                        if z[:, k].max() <= 0:
                            continue
                        pp = int(p_[int(z[:, k].argmax())])
                        r_, c_ = divmod(pp, 32)
                        g = min(k, 9)
                        d2.rectangle([c_ * c - 2 - g, r_ * c - 2 - g, (c_ + 1) * c + 1 + g, (r_ + 1) * c + 1 + g],
                                     outline=COLS[k % len(COLS)], width=3 if k < len(COLS) else 1)
                    rr, cc = divmod(qn, 4)
                    sheet.paste(im.resize((256, 256)), (cc * tw, 44 + rr * th))
                    yy = {bb: bool(lab[bb].iat[i]) for bb in LABELS}
                    for bb in LABELS:
                        n_pos[bb] += int(yy[bb])
                    tag = (('NN ' if yy['nose_nose'] else '') + ('NT ' if yy['nose_tail'] else '')) or 'neg '
                    dr.text((cc * tw + 2, 44 + rr * th + 257), f'{tag}{lab.obs.iat[i]} f{lab.frame_idx.iat[i]} '
                            f'v={x[i]:.2f} on={fr_}'[:62], fill=(0, 140, 0) if yy[b] else (200, 0, 0))
                fn = f'{key}__{fk}__{SHORT[b]}.jpg'
                sheet.save(sd / fn, quality=85)
                index.append({'key': key, 'set': fk, 'label': b, 'members': mem, 'n_shown': len(pick),
                              'n_pos_shown': n_pos, 'frames': [{'obs': lab.obs.iat[i], 'frame_idx': int(lab.frame_idx.iat[i]),
                                                                'value': float(x[i]), 'members_firing': f_}
                                                               for i, f_ in zip(pick, fired)], 'file': fn})
                log(f'  sheet {fn}: {n_pos} of {len(pick)}')
    (sd / 'index.json').write_text(json.dumps(index, indent=1))


# ---------------------------------------------------------------------------------------------- table
def cmd_table(args):
    R = json.loads((OUT / 'results.json').read_text())
    L = ['# Neuron families, mice nn / nt (scripts/eci/families.py)', '',
         f"Base rates: nn {R['base_rates']['nose_nose']:.4f}, nt {R['base_rates']['nose_tail']:.4f}", '',
         '## Cross-fitted frame level (mean over the two half directions)', '',
         '| SAE | set | label | AUROC (auc pick) | AP @auc pick | top-1% @auc pick | AP (AP pick) | top-1% (top1 pick) | '
         'members (auc pick, per test half) |', '|---|---|---|---|---|---|---|---|---|']
    for key, E in R['saes'].items():
        for cname, S in E['sets'].items():
            for b in LABELS:
                s = S[b]
                mem = s['members']['auc']
                L.append(f"| {key} | {cname} | {SHORT[b]} | {s['cf_auroc']:.3f} | {s['cf_ap_at_auroc']:.4f} | "
                         f"{s['cf_top1_at_auroc']:.3f} | {s['cf_ap']:.4f} | {s['cf_top1']:.3f} | {mem} |")
    L += ['', '## Conditional contact type (test-half AUROC, mean of directions): all-frame pick / contact re-selected',
          '', '| SAE | set | label | SAM contact | blob contact |', '|---|---|---|---|---|']
    for key, E in R['saes'].items():
        for cname, S in E['sets'].items():
            for b in LABELS:
                c = S[b]['conditional']
                L.append(f"| {key} | {cname} | {SHORT[b]} | {c['sam']['allframe_pick_auc']:.3f} / "
                         f"{c['sam']['contact_pick_auc']:.3f} | {c['blob']['allframe_pick_auc']:.3f} / "
                         f"{c['blob']['contact_pick_auc']:.3f} |")
    L += ['', '## Video level (auc pick): r with annotated rate, raw / fg-controlled; pooled delta r raw / controlled', '',
          '| SAE | set | label | video r | video r ctl | delta r | delta r ctl |', '|---|---|---|---|---|---|---|']
    for key, E in R['saes'].items():
        for cname, S in E['sets'].items():
            for b in LABELS:
                v = S[b]['video']['auc']
                L.append(f"| {key} | {cname} | {SHORT[b]} | {v['raw']['video_r_rate']:.3f} | "
                         f"{v['fg_controlled']['video_r_rate']:.3f} | {v['raw']['delta_r_pooled']:.3f} | "
                         f"{v['fg_controlled']['delta_r_pooled']:.3f} |")
    L += ['', '## Family statistics', '']
    for key, E in R['saes'].items():
        L.append(f"- {key}: {json.dumps(E['family_stats'])}")
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def cmd_run(args):
    cmd_encode(args)
    cmd_analyse(args)
    cmd_sheets(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['run', 'encode', 'encode_dep', 'analyse', 'sheets', 'greedy', 'table'])
    p.add_argument('--keys', nargs='*', default=None)
    args = p.parse_args()
    {'run': cmd_run, 'encode': cmd_encode, 'encode_dep': cmd_encode_dep, 'analyse': cmd_analyse, 'sheets': cmd_sheets, 'greedy': cmd_greedy, 'table': cmd_table}[args.cmd](args)


if __name__ == '__main__':
    main()
