"""
NN/NT ANALYSIS (2026-10-19) of the round-1 winner A3 (DINOv3-L/16 block 20, 1024 px, odour-corner rotation, tight
pixel mask, no change channel, Matryoshka BatchTopK 512 / k 16, frame max; seed 2 = the locality seed, nn neuron n315,
nt neuron n68) on mice v1. Read-only use of scripts/eci/nnnt/ (nnnt.py, r2.py, r3.py) and their stores; outputs here:
results/vision/eci_nnnt_analysis/ (small json / jpg) and /mnt/beegfs/.../eci_codes/nnnt_analysis/ (arrays).

Labels (annotations.csv row space, per frame per observation): nose_nose = Y_nn | Y_np (directed-pair union),
nose_tail = Y_nt; nn_mutual = Y_nn, np_directional = Y_np (reference_mice_label_vocabulary).
Frames for Q1b / Q1c / Q2 / Q3 = the round-2 DIAGNOSIS set (86,400 frames = every 10th 5 fps frame of the 144
annotated videos; its A3 tokens are stored, BEE/r2/diag). Q4 uses the stored A3 frame codes of all 864,000 frames.
Cross-fitting = the round-1 pool splits (score.json 'splits'); probes use rnd0-rnd4, single neurons all 10 + r4.

Q1a  labels  (CPU) bouts from the 144 raw BORIS csvs, per-annotator / per-video frame rates, annotator share of the
     per-video rate variance (line-adjusted), bouts shorter than the 5 fps sampling step.
Q1b  audit: top 48 held-out frames (split rnd0, half 1; <= 2 per video) of the nt neuron n68, of the supervised
     probe (logistic on the 512 frame codes, fit on half 0), and the same for nose_nose (n315, probe) -> BLIND sheets
     (tiles carry only a random id; key in audit_key.json) for visual classification.
Q1c  geometry of labelled frames from the label-free mask blobs (r3 geom): largest blob size / A1, elongation,
     neuron peak on the mask edge.
Q2   position: arena box per video (label-free, token extent), 3 x 3 arena regions in odour-aligned (rotated) and
     camera coordinates; label rate given a contact blob in each region; neuron AUROC per region; probes.
     PRE-REGISTERED (before any Q2 score was computed): per label, a position-aware readout ADDS information iff its
     cross-fitted R~ minus the position-invariant one is >= +0.01 on the mean of rnd0-rnd4 AND > 0 on >= 4 of the
     5 splits. Comparisons: P1 (9 region max x 512, odour coords) vs P0 (global max 512); single neuron: its 9 region
     maxima vs its global max; rotation: P1 (odour) vs P1c (camera) with the same rule.
Q3   heat maps, v1 online (fg448al DINOv2-B 448, 32 grid, deployed 1024-wide SAE, nn n611 / nt n414) vs v2 A3
     (n315 / n68) on the same frames, rendered identically (turbo, PIL bilinear, alpha 0.65 w^0.7, vmax = per-neuron
     p99 of positive token codes over the frames shown); plus v2 2 x 2 max-pooled to 32 grid. Locality numbers on
     200 random labelled frames per label.
Q4   temporal: post-SAE aggregation of the frame codes (all 512 neurons, seeds 0-2) over centred windows of
     1 / 3 / 5 / 11 / 21 frames (0.2 / 0.6 / 1 / 2.2 / 4.2 s at 5 fps) and causal (past) windows 2 / 5 / 10 / 20
     frames (0.4 / 1 / 2 / 4 s), mean and max; round-1 scorer (cross-fitted neuron selection, R~).
     PRE-REGISTERED (before any Q4 score): Delta = R~(window) - R~(single frame), paired by seed. A window HELPS a
     label iff mean Delta >= +0.01 and Delta > 0 for all 3 seeds; best window per label = the helping window with the
     highest mean R~ (else the single frame). A MULTI-SCALE design is justified iff the best windows of nose_nose and
     nose_tail differ AND each label's best window beats the other label's best window on that label by >= 0.01
     mean R~. Probe level (seed 2): logistic on [c_t, centred means 5 / 11 / 21] vs on c_t; same >= +0.01, >= 4/5
     splits rule.


Q1-STYLE (step 'style', CPU; coordinator request 2026-10-19 ~19:00, PRE-REGISTERED before any style score):
     annotator STYLE for nose_nose (nn | np), nn_mutual, np_directional, nose_tail on the 144 annotated videos.
     Label-free CONTACT EVENTS at 5 fps on EVERY annotated frame: the 144 videos decoded at 512 px (source frame
     offset + 6 k), odour-rotated, tight mask = >= 25 % dark pixels (grey < 60 and pix_bg - grey > 40, fg448al
     512 px background) in each 8 x 8 px block -> 64 grid (= nnnt Pipe 'dark64_jpg'); components (8-conn, >= 5
     cells); A1(video) = median largest component over the video's 4-component frames; CONTACT frame = largest
     component >= 1.5 A1 (round-3 rule); contact EVENT = maximal run of contact frames; END-TO-END frame = the largest
     component's elongation sqrt(l1 / l2) >= the 75th percentile of elongation over all contact frames (label-free
     threshold). Mask check vs the 1024 px diagnosis tokens (Jaccard on the diagnosis frames).
     Measures per annotator (and per line x annotator): BORIS bout durations (median, share < 0.4 s); per 5 fps label
     run: onset lag = run start - start of the contact event it overlaps most (s), offset lag = event end - run end,
     share of runs overlapping no contact event; PASS-BY share = share of labelled frames whose contact event lasts
     < 0.5 s or that are not contact frames; MISS proxy = share of long contact events (> 1 s) with >= 50 % end-to-end
     frames that carry no nn / np / nt label at all; confusion = for frames labelled X, share also labelled Y; usage
     = share of labelled frames per code.
     H-a (some annotators MISS bouts): annotator A supports H-a iff its MISS proxy exceeds the pooled other
       annotators' by >= 0.10 (absolute) with the pool-bootstrap 95 % CI of the difference excluding 0 AND its
       labelled bouts per minute (union nn | np | nt) are lower than the others'.
     H-b (some annotators count PASS-BY contacts): annotator A supports H-b for a code iff its PASS-BY share exceeds
       the others' by >= 0.05 with the pool-bootstrap 95 % CI excluding 0 AND its BORIS share of bouts < 0.4 s is
       higher than the others'.
     Only annotators with >= 3 pools (SG, SD, MB, MF) are tested; CP / MR (1 pool each) are reported only.
     Line-matched: the same two differences are recomputed within each line where A and the others both have
       videos, and reported.
     STYLE SEPARABILITY: per-video features (frame rates, bouts / min, median bout length, share < 0.4 s for nn / np
       / nt; pass-by share; onset lag median), residualised on line x phase means; multinomial logistic, leave-one-
       POOL-out CV (annotator is nested in pool), annotators with >= 3 pools; balanced accuracy vs chance (1 / 4) and
       vs a pool-level label permutation null (200 permutations, p = share of permuted >= observed). Separable iff
       p < 0.05.

Steps (analysis.sh runs each in Slurm): labels (CPU) | diag (GPU: Q1b, Q1c, Q2, Q3) | temporal (GPU: Q4) | style (CPU)
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'nnnt'))
import nnnt as N  # noqa: E402
import r2 as R2  # noqa: E402
import r3 as R3  # noqa: E402

R4, REPO, C = N.R4, N.REPO, N.C
OUTA = Path(os.environ.get('NA_OUT') or REPO / 'results/vision/eci_nnnt_analysis')
BEEA = Path(os.environ.get('NA_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/nnnt_analysis')
G = 64
V2 = 'A3_w512_s2'
V2N = {'nose_nose': 315, 'nose_tail': 68}
V1 = 'matryoshka_btk_1024_k16_fg448al_s0'
V1N = {'nose_nose': 611, 'nose_tail': 414}
LB = ('nose_nose', 'nose_tail', 'nn_mutual', 'np_directional')
SPL5 = tuple(f'rnd{i}' for i in range(5))
REG_NAMES = {0: 'corner TL', 1: 'wall T', 2: 'corner TR (odour)', 3: 'wall L', 4: 'centre', 5: 'wall R (odour side)',
             6: 'corner BL (opposite)', 7: 'wall B', 8: 'corner BR'}
log = N.log


def jdump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))


# ---------------------------------------------------------------------------------------------- Q1a labels
def cmd_labels(args):
    e = pd.read_csv(REPO / 'data/mice/v1/experiment.csv')
    e = e[e.annotation_file.notna()].copy()
    e['annotator'] = e.annotator.fillna('?')
    assert e.observation_id.is_unique and len(e) == 144
    rows = []
    for _, r in e.iterrows():
        d = pd.read_csv(REPO / 'data/mice/v1/annotations' / r.annotation_file)
        d = d[d.behavior_type.notna()].copy()
        for c_ in ('Duration (s)', 'Start (s)', 'Stop (s)', 'Image index start', 'Image index stop'):
            d[c_] = pd.to_numeric(d[c_], errors='coerce')
        code = d.behavior_type.replace({'np?': 'np'})
        dur = d['Duration (s)'].where(d['Duration (s)'].notna(), d['Stop (s)'] - d['Start (s)'])
        for c_, du, i0, i1 in zip(code, dur, d['Image index start'], d['Image index stop']):
            rows.append({'obs': r.observation_id, 'annotator': r.annotator, 'line': r.line, 'pool': r.pool,
                         'code': c_, 'dur': du, 'i0': i0, 'i1': i1,
                         'start_frame': r.start_frame})
    B = pd.DataFrame(rows)
    a = pd.read_csv(REPO / C['ann'], usecols=['observation_id', 'frame_idx', 'W_annotator', 'W_line', 'Y_nn', 'Y_np',
                                              'Y_nt'])
    a = a[a[['Y_nn', 'Y_np', 'Y_nt']].notna().all(1)].copy()
    a['nose_nose'] = (a.Y_nn > 0) | (a.Y_np > 0)
    a['nose_tail'] = a.Y_nt > 0
    a['nn_mutual'] = a.Y_nn > 0
    a['np_directional'] = a.Y_np > 0
    nfr = a.groupby('observation_id').size()
    # bout hits on the 5 fps grid (annotation frame k = source frame 6 k + 2 + start_frame; rollout offset 2)
    B['hits'] = [max(0, int(np.floor((i1 - 2 - sf) / 6) - np.ceil((i0 - 2 - sf) / 6) + 1)) if np.isfinite(i0)
                 and np.isfinite(i1) else -1 for i0, i1, sf in zip(B.i0, B.i1, B.start_frame)]
    res = {'n_annotation_files': int(len(e)), 'n_observations': int(e.observation_id.nunique()),
           'doubly_annotated_videos': int((e.groupby('observation_id').size() > 1).sum()),
           'annotators_x_line': pd.crosstab(e.annotator, e.line).to_dict(),
           'pools_per_annotator': e.groupby('annotator').pool.nunique().to_dict(),
           'annotators_per_pool_max': int(e.groupby('pool').annotator.nunique().max())}
    bout = {}
    for c_, g in B.groupby('code'):
        du = g.dur.dropna()
        h = g.hits[g.hits >= 0]
        bout[c_] = {'n_bouts': int(len(g)), 'dur_median_s': float(du.median()), 'dur_q25_s': float(du.quantile(.25)),
                    'dur_q75_s': float(du.quantile(.75)), 'dur_mean_s': float(du.mean()),
                    'share_dur_le_0.2s': float((du <= 0.2 + 1e-9).mean()),
                    'share_dur_le_0.5s': float((du <= 0.5 + 1e-9).mean()),
                    'share_bouts_missed_by_5fps_grid': float((h == 0).mean()),
                    'grid_frames_hit_total': int(h.sum()),
                    'frames_labelled_in_annotations_csv': int(a[{'nn': 'Y_nn', 'np': 'Y_np', 'nt': 'Y_nt'}[c_]]
                                                              .gt(0).sum())}
    res['bouts'] = bout
    # per annotator
    vid = a.groupby('observation_id').agg(annotator=('W_annotator', 'first'), line=('W_line', 'first'),
                                          **{lb: (lb, 'mean') for lb in LB})
    vid['annotator'] = vid.annotator.fillna('?')
    ann = {}
    for an, g in vid.groupby('annotator'):
        bb = B[B.annotator == an]
        mins = nfr[g.index].sum() / 5 / 60
        ann[an] = {'videos': int(len(g)), 'minutes': float(mins),
                   **{f'rate_{lb}': float(a[a.observation_id.isin(g.index)][lb].mean()) for lb in LB},
                   **{f'bouts_per_min_{c_}': float((bb.code == c_).sum() / mins) for c_ in ('nn', 'np', 'nt')},
                   **{f'dur_median_{c_}': float(bb[bb.code == c_].dur.median()) for c_ in ('nn', 'np', 'nt')},
                   'nt_over_nose_nose_frames': float(a[a.observation_id.isin(g.index)].nose_tail.sum() /
                                                     max(1, a[a.observation_id.isin(g.index)].nose_nose.sum()))}
    res['per_annotator'] = ann
    # per video variability and annotator share of variance (line-adjusted)
    var = {}
    for lb in LB:
        y = vid[lb].values
        Xl = pd.get_dummies(vid.line).values.astype(float)
        Xa = pd.get_dummies(vid.annotator).values.astype(float)
        ss = ((y - y.mean()) ** 2).sum()

        def r2(X):
            bta = np.linalg.lstsq(X, y, rcond=None)[0]
            return 1 - ((y - X @ bta) ** 2).sum() / ss
        rl, rla, ra = r2(Xl), r2(np.c_[Xl, Xa]), r2(Xa)
        var[lb] = {'video_rate_mean': float(y.mean()), 'video_rate_cv': float(y.std() / y.mean()),
                   'share_videos_zero': float((y == 0).mean()), 'R2_annotator': float(ra), 'R2_line': float(rl),
                   'R2_annotator_given_line_increment': float(rla - rl)}
    res['video_variability'] = var
    # line-matched annotator comparison (SG vs the rest within each line)
    lm = {}
    for ln, g in vid.groupby('line'):
        lm[ln] = {an: {lb: float(gg[lb].mean()) for lb in ('nose_nose', 'nose_tail')} | {'videos': int(len(gg))}
                  for an, gg in g.groupby('annotator')}
    res['line_matched_rates'] = lm
    jdump(res, OUTA / 'q1a_labels.json')
    log(json.dumps(res, indent=1, default=str)[:6000])


# ---------------------------------------------------------------------------------------------- probes
def fit_logit(torch, X, y, lam, iters=150):
    """Balanced-weight L2 logistic, full-batch LBFGS. X (n, F) float32 cuda (standardised), y bool cuda."""
    n, F = X.shape
    w = torch.zeros(F, device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    sw = torch.where(y, 0.5 / y.sum(), 0.5 / (~y).sum()).float()
    yf = y.float()
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, history_size=20, line_search_fn='strong_wolfe')

    def clo():
        opt.zero_grad()
        z = X @ w + b
        loss = (sw * torch.nn.functional.binary_cross_entropy_with_logits(z, yf, reduction='none')).sum() + \
            0.5 * lam * (w * w).sum()
        loss.backward()
        return loss
    opt.step(clo)
    return w.detach(), b.detach()


def prep(torch, X, tr):
    """log1p + standardise with the training rows' mean / sd. X (n, F) float cuda."""
    Z = torch.log1p(X.float().clamp_min(0))
    mu, sd = Z[tr].mean(0), Z[tr].std(0)
    sd = torch.where(sd > 1e-6, sd, torch.ones_like(sd))
    return (Z - mu) / sd, mu, sd


def crossfit_probe(torch, X, Y, half_of, splits, dev, lams=(1e-4, 1e-3, 1e-2), keep=None):
    """X (n, F) torch cuda (raw codes >= 0), Y (n, L) bool numpy. Per label, split, direction: lambda chosen on an inner
    pool split of the training half (AP), refit, scored on the held-out half. -> {label: {R, A, per_split_R, ...}};
    keep = (split, direction) whose held-out scores are returned."""
    from sklearn.metrics import average_precision_score
    out, kept = {}, {}
    for li, lb in enumerate(('nose_nose', 'nose_tail')):
        per = []
        for sn in splits:
            half = half_of['frame'][sn]
            dirs = []
            for s_, t_ in ((0, 1), (1, 0)):
                tr, te = np.flatnonzero(half == s_), np.flatnonzero(half == t_)
                # inner split: pools of the training half in two
                pu = half_of['unit_idx'][tr]
                up = np.unique(pu)
                inner = np.isin(pu, np.random.default_rng(5).permutation(up)[:len(up) // 2])
                best = None
                for lam in lams:
                    Xs, _, _ = prep(torch, X, torch.from_numpy(tr[~inner]).to(dev))
                    yt = torch.from_numpy(Y[tr[~inner], li]).to(dev)
                    w, b = fit_logit(torch, Xs[torch.from_numpy(tr[~inner]).to(dev)], yt, lam)
                    s = (Xs[torch.from_numpy(tr[inner]).to(dev)] @ w + b).cpu().numpy()
                    ap = average_precision_score(Y[tr[inner], li], s)
                    if best is None or ap > best[0]:
                        best = (ap, lam)
                trt = torch.from_numpy(tr).to(dev)
                Xs, mu, sd = prep(torch, X, trt)
                w, b = fit_logit(torch, Xs[trt], torch.from_numpy(Y[tr, li]).to(dev), best[1])
                s = (Xs[torch.from_numpy(te).to(dev)] @ w + b).cpu().numpy().astype(np.float64)
                m = R2.metrics(s, Y[te, li])
                m['lam'] = best[1]
                dirs.append(m)
                if keep is not None and (sn, s_) == keep:
                    kept[lb] = {'te': te, 'score': s, 'w': w.cpu().numpy(), 'mu': mu.cpu().numpy(),
                                'sd': sd.cpu().numpy(), 'b': float(b)}
                del Xs
            per.append({k_: float(np.mean([d[k_] for d in dirs])) for k_ in ('R', 'A', 'AP_lift')})
        out[lb] = {'R': float(np.mean([p['R'] for p in per])), 'A': float(np.mean([p['A'] for p in per])),
                   'AP_lift': float(np.mean([p['AP_lift'] for p in per])), 'per_split_R': [p['R'] for p in per]}
    return out, kept


def compare(a, b, lb):
    """Pre-registered rule: b adds information over a iff mean dR >= 0.01 and dR > 0 on >= 4/5 splits."""
    d = np.array(b[lb]['per_split_R']) - np.array(a[lb]['per_split_R'])
    return {'dR_mean': float(d.mean()), 'dR_per_split': d.tolist(), 'n_pos': int((d > 0).sum()),
            'adds': bool(d.mean() >= 0.01 and (d > 0).sum() >= 4)}


# ---------------------------------------------------------------------------------------------- diag step
def rot_index(rot, n):
    """flat index map: at rotated position p, the original (camera) flat position (np.rot90 k = rot CCW)."""
    return np.rot90(np.arange(n * n).reshape(n, n), rot).ravel()


def cmd_diag(args):
    import torch
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    loc = N.local_dir()
    BEEA.mkdir(parents=True, exist_ok=True)
    R4.atomic_copy(R2.BEE2 / 'diag', loc / 'diag')
    root = loc / 'diag'
    F = pd.read_parquet(root / 'frames.parquet')
    tf = np.load(root / 'tok_frame.npy')
    pos = np.load(root / 'tok_pos.npy').astype(np.int64)
    fdark = np.load(root / 'tok_fdark.npy').astype(np.float32)
    tok = np.memmap(root / 'tok.f16', np.float16, 'r', shape=(len(tf), N.D))
    nf = len(F)
    starts = np.r_[0, np.cumsum(np.bincount(tf, minlength=nf))]
    geo = np.load(R3.BEE3 / 'geom_diag.npz')
    geo = {k: geo[k] for k in geo.files}
    cf, A1, _ = R3.contact_frames(F, geo)
    Y, unit = R2.labels_of(F.row.values)
    sc = json.loads((N.OUT / 'score.json').read_text())
    units = sc['units']
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    half_of = {'frame': {sn: np.array(h)[uidx] for sn, h in sc['splits'].items()}, 'unit_idx': uidx}
    P = N.load_plan()
    V = {v['obs']: v for v in P['videos']}
    rot_f = F.obs.map(lambda o: V[o]['rot']).values.astype(np.int64)
    log(f'diag: {nf} frames, {len(tf):,} tokens, contact share {cf.mean():.3f}, base rates '
        + json.dumps({lb: float(Y[:, i].mean()) for i, lb in enumerate(LB)}))
    # ---- arena box per video (label-free: 0.5 / 99.5 % of the video's kept-token rows / cols, rotated frame)
    tr_, tc_ = pos // G, pos % G
    ocode, ouniq = pd.factorize(F.obs.values)
    tob = ocode[tf]
    box = {}
    for oc, o in enumerate(ouniq):
        m = tob == oc
        box[o] = (np.quantile(tr_[m], .005), np.quantile(tr_[m], .995), np.quantile(tc_[m], .005),
                  np.quantile(tc_[m], .995))
    bx = np.array([box[o] for o in F.obs.values])  # per frame (r0, r1, c0, c1)
    bt = bx[tf]
    u_r = np.clip((tr_ + 0.5 - bt[:, 0]) / (bt[:, 1] + 1 - bt[:, 0]), 0, 0.999)
    u_c = np.clip((tc_ + 0.5 - bt[:, 2]) / (bt[:, 3] + 1 - bt[:, 2]), 0, 0.999)
    reg_od = (np.floor(u_r * 3) * 3 + np.floor(u_c * 3)).astype(np.int64)
    RM = np.stack([rot_index(r, 3) for r in range(4)])
    reg_cam = RM[rot_f[tf], reg_od]
    # ---- encode every diagnosis token with the A3 seed-2 SAE
    T = R2.tokens_to(torch, tok, dev)
    sae, norm, _ = load_sae(N.OUT / 'sae' / V2 / 'sae_theta.pt', dev)
    W = sae.n_latents
    Xg = np.zeros((nf, W), np.float16)
    Am = np.full((nf, W), -1, np.int16)
    Xod = np.zeros((nf, 9, W), np.float16)
    Xcam = np.zeros((nf, 9, W), np.float16)
    tokc = np.zeros((len(tf), 2), np.float16)  # per-token codes of n315, n68
    t0 = time.time()
    pos_t, rod_t, rcam_t = (torch.from_numpy(x).to(dev) for x in (pos, reg_od, reg_cam))
    for f0 in range(0, nf, 64):
        f1 = min(nf, f0 + 64)
        a_, b_ = int(starts[f0]), int(starts[f1])
        if a_ == b_:
            continue
        with torch.no_grad():
            c = sae.encode(norm(R2.chunk_of(T, a_, b_, dev).float()), mode='threshold')
            tokc[a_:b_] = c[:, [V2N['nose_nose'], V2N['nose_tail']]].half().cpu().numpy()
            bi = torch.from_numpy(tf[a_:b_] - f0).to(dev).long()
            Bn = f1 - f0
            cnt = torch.bincount(bi, minlength=Bn)
            off = torch.cumsum(cnt, 0) - cnt
            p = torch.arange(b_ - a_, device=dev) - off[bi]
            Tm = int(cnt.max())
            dense = torch.zeros((Bn, Tm, W), device=dev)
            dense[bi, p] = c
            mx, am = dense.max(1)
            pd_ = torch.full((Bn, Tm), -1, dtype=torch.long, device=dev)
            pd_[bi, p] = pos_t[a_:b_]
            Xg[f0:f1] = mx.half().cpu().numpy()
            Am[f0:f1] = pd_.gather(1, am).to(torch.int16).cpu().numpy()
            for rt, XX in ((rod_t, Xod), (rcam_t, Xcam)):
                rd = torch.full((Bn, Tm), -1, dtype=torch.long, device=dev)
                rd[bi, p] = rt[a_:b_]
                for r in range(9):
                    XX[f0:f1, r] = (dense * (rd == r)[..., None]).max(1).values.half().cpu().numpy()
    log(f'encoded diag tokens in {time.time() - t0:.0f}s')
    np.savez(BEEA / 'diag_A3s2.npz', Xg=Xg, Am=Am, tokc=tokc)
    res = {'n_frames': nf, 'contact_share': float(cf.mean())}
    # identity check vs the stored round-1 per-video codes
    vdir = N.BEE / 'videos'
    dif, ref = [], []
    for o, g in F.groupby('obs'):
        z = np.load(vdir / f'{o}.npz')[f'{V2}|max']
        ref.append(z[g.k.values][:, [315, 68]].astype(np.float32))
        dif.append(Xg[g.index.values][:, [315, 68]].astype(np.float32))
    ref, dif = np.concatenate(ref), np.concatenate(dif)
    res['identity_vs_stored_codes'] = {'max_abs_diff': float(np.abs(ref - dif).max()),
                                       'corr_n315': float(np.corrcoef(ref[:, 0], dif[:, 0])[0, 1]),
                                       'corr_n68': float(np.corrcoef(ref[:, 1], dif[:, 1])[0, 1]),
                                       'fire_agreement': float(((ref > 0) == (dif > 0)).mean())}
    log('identity: ' + json.dumps(res['identity_vs_stored_codes']))
    jdump(res, OUTA / 'q_diag_meta.json')
    # ---- contact location per frame (largest blob centroid), label-free
    big = geo['comp'] == geo['maxc'][tf]
    wsum = np.bincount(tf, weights=big.astype(float), minlength=nf)
    cr = np.bincount(tf, weights=big * u_r, minlength=nf) / np.maximum(wsum, 1)
    cc = np.bincount(tf, weights=big * u_c, minlength=nf) / np.maximum(wsum, 1)
    creg = (np.floor(np.clip(cr, 0, .999) * 3) * 3 + np.floor(np.clip(cc, 0, .999) * 3)).astype(np.int64)
    creg_cam = RM[rot_f, creg]
    # blob shape (Q1c): largest component, raw token coords
    s1 = np.bincount(tf, weights=big * tr_, minlength=nf)
    s2 = np.bincount(tf, weights=big * tc_, minlength=nf)
    mr, mc = s1 / np.maximum(wsum, 1), s2 / np.maximum(wsum, 1)
    vrr = np.bincount(tf, weights=big * tr_ ** 2, minlength=nf) / np.maximum(wsum, 1) - mr ** 2
    vcc = np.bincount(tf, weights=big * tc_ ** 2, minlength=nf) / np.maximum(wsum, 1) - mc ** 2
    vrc = np.bincount(tf, weights=big * tr_ * tc_, minlength=nf) / np.maximum(wsum, 1) - mr * mc
    tr2 = (vrr + vcc) / 2
    disc = np.sqrt(np.maximum(((vrr - vcc) / 2) ** 2 + vrc ** 2, 0))
    l1, l2 = tr2 + disc, np.maximum(tr2 - disc, 1e-3)
    a1f = F.obs.map(A1).values
    q2(res, torch, dev, F, Y, cf, creg, creg_cam, Xg, Xod, Xcam, half_of)
    q1c(res, F, Y, cf, geo, a1f, l1, l2, Am, tf, starts, pos)
    q1b(res, torch, dev, F, Y, Xg, Am, half_of, sae, norm, T, starts, pos, V, cf)
    q3(res, torch, dev, F, Y, tokc, Xg, starts, pos, fdark, V, half_of)
    jdump(res, OUTA / 'q_diag.json')
    log('diag done')


def boot_rate(num, den, pool_of, n=1000, seed=0):
    """Pool-bootstrap 95% CI of sum(num) / sum(den); num, den per frame; pool_of per frame."""
    pools = np.unique(pool_of)
    pi = pd.Series(range(len(pools)), index=pools).loc[pool_of].values
    Nn = np.bincount(pi, weights=np.asarray(num, float), minlength=len(pools))
    Dd = np.bincount(pi, weights=np.asarray(den, float), minlength=len(pools))
    g = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        s = g.integers(0, len(pools), len(pools))
        b.append(Nn[s].sum() / max(Dd[s].sum(), 1))
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def q2(res, torch, dev, F, Y, cf, creg, creg_cam, Xg, Xod, Xcam, half_of):
    from scipy.stats import chi2_contingency
    from sklearn.metrics import roc_auc_score
    unit = R2.labels_of(F.row.values)[1]
    out = {'pre_registered_rule': 'adds iff mean dR~ >= +0.01 over rnd0-4 and dR~ > 0 on >= 4/5 splits'}
    # label rate | contact blob in region
    for name, rg in (('odour', creg), ('camera', creg_cam)):
        tab = {}
        for r in range(9):
            m = cf & (rg == r)
            tab[r] = {'region': REG_NAMES[r] if name == 'odour' else f'camera cell {r}', 'contact_frames': int(m.sum()),
                      'share_of_contact_frames': float(m.sum() / cf.sum())}
            for li, lb in enumerate(LB[:2]):
                tab[r][f'rate_{lb}'] = float(Y[m, li].mean()) if m.any() else float('nan')
                tab[r][f'rate_{lb}_ci95'] = boot_rate(Y[:, li] & m, m.astype(float), unit)
                tab[r][f'share_of_{lb}_positives'] = float((Y[:, li] & m).sum() / max(1, (Y[:, li] & cf).sum()))
        chi = {}
        for li, lb in enumerate(LB[:2]):
            ct = np.array([[((rg == r) & cf & Y[:, li]).sum(), ((rg == r) & cf & ~Y[:, li]).sum()] for r in range(9)])
            ct = ct[ct.sum(1) > 0]
            st = chi2_contingency(ct)
            chi[lb] = {'chi2': float(st[0]), 'p_iid': float(st[1]),
                       'cramers_v': float(np.sqrt(st[0] / ct.sum())),
                       'max_over_min_rate': float((ct[:, 0] / ct.sum(1)).max() / max((ct[:, 0] / ct.sum(1)).min(),
                                                                                    1e-9))}
        out[f'rate_by_region_{name}'] = tab
        out[f'heterogeneity_{name}'] = chi
    out['positives_in_contact_frames'] = {lb: float(cf[Y[:, li]].mean()) for li, lb in enumerate(LB[:2])}
    # neuron response by region (contact frames)
    nr = {}
    for li, lb in enumerate(LB[:2]):
        j = V2N[lb]
        x = Xg[:, j].astype(np.float32)
        d = {}
        for r in range(9):
            m = cf & (creg == r)
            y = Y[m, li]
            d[REG_NAMES[r]] = {'n_pos': int(y.sum()),
                               'AUROC_within_region_contact': float(roc_auc_score(y, x[m])) if 0 < y.sum() < len(y)
                               else float('nan'),
                               'mean_code_pos': float(x[m][y].mean()) if y.any() else float('nan'),
                               'mean_code_neg': float(x[m][~y].mean()),
                               'fire_rate_neg': float((x[m][~y] > 0).mean())}
        yy = Y[cf, li]
        d['_all_contact'] = {'AUROC': float(roc_auc_score(yy, x[cf])), 'mean_code_pos': float(x[cf][yy].mean())}
        nr[f'n{j}_{lb}'] = d
    out['neuron_by_region_odour'] = nr
    # probes
    Yp = Y[:, :2]
    t0 = time.time()
    Xg_t = torch.from_numpy(Xg).to(dev)
    P0, _ = crossfit_probe(torch, Xg_t, Yp, half_of, SPL5, dev)
    log(f'P0 {json.dumps(P0)} ({time.time() - t0:.0f}s)')
    Xo = torch.from_numpy(Xod.reshape(len(F), -1)).to(dev)
    P1, _ = crossfit_probe(torch, Xo, Yp, half_of, SPL5, dev)
    del Xo
    Xc = torch.from_numpy(Xcam.reshape(len(F), -1)).to(dev)
    P1c, _ = crossfit_probe(torch, Xc, Yp, half_of, SPL5, dev)
    del Xc
    # P0 + label-free contact-location one-hot (odour coords; no-contact = its own level)
    oh = np.zeros((len(F), 10), np.float32)
    oh[np.arange(len(F)), np.where(cf, creg, 9)] = 1
    P2, _ = crossfit_probe(torch, torch.cat([Xg_t.float(), torch.from_numpy(oh * 3).to(dev)], 1), Yp, half_of, SPL5,
                           dev)
    log(f'P1 {json.dumps(P1)} | P1c {json.dumps(P1c)} | P2 {json.dumps(P2)}')
    out['probes'] = {'P0_global_max_512': P0, 'P1_region_max_odour_9x512': P1, 'P1c_region_max_camera_9x512': P1c,
                     'P2_global_max_plus_contact_region': P2}
    out['probe_rule'] = {lb: {'P1_vs_P0': compare(P0, P1, lb), 'P1c_vs_P0': compare(P0, P1c, lb),
                              'P1_odour_vs_P1c_camera': compare(P1c, P1, lb), 'P2_vs_P0': compare(P0, P2, lb)}
                         for lb in LB[:2]}
    sn_ = {}
    for li, lb in enumerate(LB[:2]):
        j = V2N[lb]
        g1, _ = crossfit_probe(torch, Xg_t[:, j:j + 1], Yp, half_of, SPL5, dev)
        r1, _ = crossfit_probe(torch, torch.from_numpy(Xod[:, :, j].astype(np.float32)).to(dev), Yp, half_of, SPL5,
                               dev)
        rc, _ = crossfit_probe(torch, torch.from_numpy(Xcam[:, :, j].astype(np.float32)).to(dev), Yp, half_of, SPL5,
                               dev)
        sn_[f'n{j}_{lb}'] = {'global_max': g1[lb], 'region_max_odour': r1[lb], 'region_max_camera': rc[lb],
                             'rule_odour_vs_global': compare(g1, r1, lb), 'rule_camera_vs_global': compare(g1, rc, lb)}
    out['single_neuron_position'] = sn_
    del Xg_t
    res['Q2'] = out
    jdump(res, OUTA / 'q_diag.json')


def q1c(res, F, Y, cf, geo, a1f, l1, l2, Am, tf, starts, pos):
    out = {}
    groups = {'nose_nose': Y[:, 0], 'nose_tail': Y[:, 1], 'nn_mutual': Y[:, 2], 'np_directional': Y[:, 3],
              'unlabelled_contact': cf & ~Y[:, :2].any(1)}
    # peak-token geometry of the selected neuron: map (frame, pos) -> dist / n_nb
    tok_of = {}
    for lb, j in V2N.items():
        pk = Am[:, j].astype(np.int64)
        dist = np.full(len(F), -1)
        nnb = np.full(len(F), -1)
        for f in np.flatnonzero(Y[:, LB.index(lb)] | (cf & ~Y[:, :2].any(1))):
            a_, b_ = starts[f], starts[f + 1]
            k = np.flatnonzero(pos[a_:b_] == pk[f])
            if len(k):
                dist[f], nnb[f] = geo['dist'][a_ + k[0]], geo['n_nb'][a_ + k[0]]
        tok_of[lb] = (dist, nnb)
    for gname, m in groups.items():
        mc = m & cf
        out[gname] = {'n': int(m.sum()), 'contact_share': float(cf[m].mean()),
                      'largest_blob_over_A1_median': float(np.median(geo['maxc'][mc] / a1f[mc])),
                      'blob_major_sd_tokens_median': float(np.median(np.sqrt(l1[mc]))),
                      'blob_elongation_median': float(np.median(np.sqrt(l1[mc] / l2[mc]))),
                      'n_components_median': float(np.median(geo['ncomp'][m]))}
        for lb in V2N:
            d, nb = tok_of[lb]
            ok = m & (d >= 0)
            if ok.any():
                out[gname][f'n{V2N[lb]}_peak_on_edge_share'] = float((d[ok] == 1).mean())
                out[gname][f'n{V2N[lb]}_peak_at_tip_share'] = float((nb[ok] <= 4).mean())
    out['_all_tokens'] = {'edge_token_share': float((geo['dist'] == 1).mean()),
                          'tip_token_share': float((geo['n_nb'] <= 4).mean())}
    res['Q1c'] = out
    jdump(res, OUTA / 'q_diag.json')


def decode_rot(v, k):
    fr = R4.decode(v['path'], v['offset'] + C['step'] * int(k), 1, C['step'], N.S, v['hflip'], C['fps'], threads=2)[0]
    return np.ascontiguousarray(np.rot90(fr, v['rot']))


def q1b(res, torch, dev, F, Y, Xg, Am, half_of, sae, norm, T, starts, pos, V, cf):
    from PIL import Image, ImageDraw
    Yp = Y[:, :2]
    Xg_t = torch.from_numpy(Xg).to(dev)
    _, kept = crossfit_probe(torch, Xg_t, Yp, half_of, ('rnd0',), dev, keep=('rnd0', 0))
    del Xg_t
    half1 = np.flatnonzero(half_of['frame']['rnd0'] == 1)
    sets = {}
    for li, lb in enumerate(LB[:2]):
        j = V2N[lb]
        x = Xg[half1, j].astype(np.float64)
        sets[f'neuron_n{j}_{lb}'] = (half1, x, 'neuron', lb, j)
        sets[f'probe_{lb}'] = (kept[lb]['te'], kept[lb]['score'], 'probe', lb, None)
    picks, summ = {}, {}
    for sname, (fr_ids, s, kind, lb, j) in sets.items():
        o = np.argsort(-s, kind='stable')
        npos = int(Y[fr_ids, LB.index(lb)].sum())
        sel, per = [], {}
        for i in o:
            ob = F.obs.values[fr_ids[i]]
            if per.get(ob, 0) >= 2:
                continue
            per[ob] = per.get(ob, 0) + 1
            sel.append(int(fr_ids[i]))
            if len(sel) == 48:
                break
        picks[sname] = sel
        top = fr_ids[o[:npos]]
        summ[sname] = {'held_out_frames': int(len(fr_ids)), 'held_out_positives': npos,
                       'R_precision_raw_top_Npos': float(Y[top, LB.index(lb)].mean()),
                       'label_share_top48_capped': float(Y[sel, LB.index(lb)].mean()),
                       'other_label_share_top48': float(Y[sel, 1 - LB.index(lb)].mean()),
                       'contact_share_top48': float(cf[sel].mean())}
    # peak location per (set, frame)
    peak = {}
    for sname, sel in picks.items():
        kind, lb, j = sets[sname][2], sets[sname][3], sets[sname][4]
        for f in sel:
            if kind == 'neuron':
                peak[(sname, f)] = int(Am[f, j])
            else:
                a_, b_ = int(starts[f]), int(starts[f + 1])
                with torch.no_grad():
                    c = sae.encode(norm(R2.chunk_of(T, a_, b_, dev).float()), mode='threshold').cpu().numpy()
                k = kept[lb]
                st = np.log1p(c) @ (k['w'] / k['sd'])
                peak[(sname, f)] = int(pos[a_ + int(np.argmax(st))])
    frames = sorted({f for sel in picks.values() for f in sel})
    g = np.random.default_rng(11)
    ids = {f: f'T{i:03d}' for i, f in zip(g.permutation(len(frames)), frames)}
    with ThreadPoolExecutor(16) as ex:
        IM = dict(zip(frames, ex.map(lambda f: decode_rot(V[F.obs.values[f]], F.k.values[f]), frames)))
    # one tile per frame: 384 px crop around the (first) peak, peak tokens of every set marked
    tiles = []
    for f in sorted(frames, key=lambda f: ids[f]):
        pk = [peak[(sn, f)] for sn in picks if f in picks[sn]]
        p0 = pk[0]
        cy, cx = (p0 // G + 0.5) * 16, (p0 % G + 0.5) * 16
        c = 384
        y0 = int(np.clip(cy - c / 2, 0, N.S - c))
        x0 = int(np.clip(cx - c / 2, 0, N.S - c))
        im = Image.fromarray(IM[f][y0:y0 + c, x0:x0 + c]).resize((320, 320), Image.BILINEAR)
        dr = ImageDraw.Draw(im)
        for p_ in set(pk):
            yy, xx = ((p_ // G + 0.5) * 16 - y0) * 320 / c, ((p_ % G + 0.5) * 16 - x0) * 320 / c
            dr.ellipse([xx - 14, yy - 14, xx + 14, yy + 14], outline=(0, 255, 255), width=2)
        tiles.append((ids[f], im))
    sdir = OUTA / 'audit'
    sdir.mkdir(parents=True, exist_ok=True)
    for s0 in range(0, len(tiles), 24):
        sh = Image.new('RGB', (6 * 324, 4 * 344), (255, 255, 255))
        for i, (tid, im) in enumerate(tiles[s0:s0 + 24]):
            x_, y_ = (i % 6) * 324, (i // 6) * 344
            sh.paste(im, (x_, y_))
            ImageDraw.Draw(sh).text((x_ + 4, y_ + 322), tid, fill=(0, 0, 0))
        sh.save(sdir / f'blind_{s0 // 24:02d}.jpg', quality=88)
    key = {ids[f]: {'frame': int(f), 'obs': F.obs.values[f], 'k': int(F.k.values[f]), 'row': int(F.row.values[f]),
                    'labels': [lb for li, lb in enumerate(LB) if Y[f, li]], 'contact': bool(cf[f]),
                    'sets': {sn: picks[sn].index(f) for sn in picks if f in picks[sn]}} for f in frames}
    jdump(key, OUTA / 'audit_key.json')
    res['Q1b'] = {'sets': summ, 'n_unique_frames': len(frames), 'sheets': sorted(str(p) for p in sdir.glob('blind_*'))}
    jdump(res, OUTA / 'q_diag.json')
    log('Q1b ' + json.dumps(res['Q1b']))


def q3(res, torch, dev, F, Y, tokc, Xg, starts, pos, fdark, V, half_of):
    from matplotlib import colormaps
    from PIL import Image, ImageDraw
    from src.eci.foreground import rotate_image
    from src.eci.viz import _overlay_rgb, make_patch_encoder
    enc = make_patch_encoder(V1, dataset_dir=str(REPO / 'dataset'))
    rows = F.row.values
    assert enc.rot is not None
    rp = np.array([V[o]['rot'] for o in F.obs.values])
    assert (enc.rot[rows] % 4 == rp % 4).all(), 'v1 / v2 rotations differ'
    cm = np.load(REPO / C['deployed'] / 'codes_max.npy', mmap_mode='r')
    oi = np.argsort(rows)
    v1max = np.empty((len(rows), 2), np.float32)
    v1max[oi] = cm[rows[oi]][:, [V1N['nose_nose'], V1N['nose_tail']]]
    fp = pd.read_csv(REPO / C['ann'], usecols=['frame_path'])['frame_path'].values
    g = np.random.default_rng(0)
    out = {'firing_rate_frames': {}}
    for li, lb in enumerate(LB[:2]):
        out['firing_rate_frames'][lb] = {'v1': float((v1max[:, li] > 0).mean()),
                                         'v2': float((Xg[:, V2N[lb]] > 0).mean())}

    def v2map(f, col):
        a_, b_ = int(starts[f]), int(starts[f + 1])
        h = np.zeros(G * G, np.float32)
        h[pos[a_:b_]] = tokc[a_:b_, col]
        am = np.zeros(G * G, np.float32)
        am[pos[a_:b_]] = fdark[a_:b_]
        return h.reshape(G, G), am.reshape(G, G)

    def v1maps(fr):
        pc, mk = enc.patch_codes([REPO / 'dataset' / fp[rows[f]] for f in fr], [V1N['nose_nose'], V1N['nose_tail']],
                                 rows[fr], return_mask=True)
        return pc, mk

    def loc_stats(h, animal):
        """h (g, g) >= 0, animal (g, g) bool at the same grid."""
        if h.max() <= 0:
            return None
        return {'mass_on_animal': float((h * animal).sum() / h.sum()),
                'half_max_area_share': float((h >= 0.5 * h.max()).mean()),
                'active_area_share': float((h > 0).mean()),
                'half_max_over_animal_area': float((h >= 0.5 * h.max()).sum() / max(1, animal.sum()))}
    # locality on 200 random labelled frames per label
    lstat = {}
    for li, lb in enumerate(LB[:2]):
        fr = g.choice(np.flatnonzero(Y[:, li]), 200, replace=False)
        pc, mk = v1maps(fr)
        S1, S2, S2p = [], [], []
        for i, f in enumerate(fr):
            h2, am = v2map(f, li)
            an64 = am >= 0.25
            an32 = an64.reshape(32, 2, 32, 2).any((1, 3))
            S2.append(loc_stats(h2, an64))
            S2p.append(loc_stats(h2.reshape(32, 2, 32, 2).max((1, 3)), an32))
            S1.append(loc_stats(pc[i, :, :, li], an32))
        lstat[lb] = {}
        for nm, S in (('v1_32grid', S1), ('v2_64grid', S2), ('v2_maxpooled_32', S2p)):
            ok = [s for s in S if s is not None]
            lstat[lb][nm] = {'frames_firing': len(ok), **{k_: float(np.mean([s[k_] for s in ok]))
                                                          for k_ in ok[0]}}
        lstat[lb]['v1_mask_tokens_mean'] = float(mk.sum((1, 2)).mean())
        lstat[lb]['v2_kept_tokens_mean'] = float(np.mean([starts[f + 1] - starts[f] for f in fr]))
    out['locality_200_labelled'] = lstat
    log('Q3 locality ' + json.dumps(lstat))
    # side-by-side sheets
    turbo = colormaps['turbo']
    sheets = []
    for li, lb in enumerate(LB[:2]):
        chosen, why = [], []
        for src, sc_ in (('v1 top', v1max[:, li]), ('v2 top', Xg[:, V2N[lb]].astype(np.float32))):
            seen = {F.obs.values[f] for f in chosen}
            for f in np.argsort(-sc_):
                if F.obs.values[f] in seen:
                    continue
                seen.add(F.obs.values[f])
                chosen.append(int(f))
                why.append(src)
                if why.count(src) == 4:
                    break
        for f in g.choice(np.setdiff1d(np.flatnonzero(Y[:, li]), chosen), 4, replace=False):
            chosen.append(int(f))
            why.append('random labelled')
        pc, _ = v1maps(np.array(chosen))
        H1 = [pc[i, :, :, li] for i in range(len(chosen))]
        H2 = [v2map(f, li)[0] for f in chosen]
        H2p = [h.reshape(32, 2, 32, 2).max((1, 3)) for h in H2]
        def p99(H):
            v = np.concatenate([h[h > 0] for h in H] + [np.zeros(0)])
            return float(np.quantile(v, 0.99)) if len(v) else 1.0
        vm1, vm2, vm2p = p99(H1), p99(H2), p99(H2p)
        tw, cw = 300, 256
        sh = Image.new('RGB', (2 * 4 * tw + 20, 30 + 6 * (tw + 18)), (255, 255, 255))
        ImageDraw.Draw(sh).text((4, 4), f'{lb}: frame | v1 fg448al n{V1N[lb]} (32 grid) | v2 A3 n{V2N[lb]} (64 grid) | '
                                         f'v2 2x2 max -> 32 grid.  turbo, bilinear, alpha 0.65 w^0.7, vmax = p99 '
                                         f'per neuron over these frames ({vm1:.2f} / {vm2:.2f} / {vm2p:.2f}); '
                                         f'512 px rotated JPG, 256 px crop', fill=(0, 0, 0))
        for i, f in enumerate(chosen):
            im = np.asarray(rotate_image(Image.open(REPO / 'dataset' / fp[rows[f]]).convert('RGB'), rp[f]))
            ph = H2[i] if H2[i].max() > 0 else H1[i]
            cy, cx = np.unravel_index(np.argmax(ph), ph.shape)
            sc = 512 / ph.shape[0]
            y0 = int(np.clip((cy + .5) * sc - cw / 2, 0, 512 - cw))
            x0 = int(np.clip((cx + .5) * sc - cw / 2, 0, 512 - cw))
            pans = [im] + [_overlay_rgb(im, h, (0, 0, 512), vm, turbo) for h, vm in
                           ((H1[i], vm1), (H2[i], vm2), (H2p[i], vm2p))]
            xb, yb = (i % 2) * (4 * tw + 20), 30 + (i // 2) * (tw + 18)
            for pi_, pan in enumerate(pans):
                sh.paste(Image.fromarray(pan[y0:y0 + cw, x0:x0 + cw]).resize((tw, tw), Image.BILINEAR),
                         (xb + pi_ * tw, yb))
            lab = '+'.join(l_ for l_i, l_ in enumerate(('nn|np', 'nt')) if Y[f, l_i]) or '-'
            ImageDraw.Draw(sh).text((xb + 2, yb + tw + 2), f'{why[i]} | {F.obs.values[f]} k{F.k.values[f]} '
                                                            f'[{lab}] v1 max {H1[i].max():.2f} v2 max {H2[i].max():.2f}',
                                    fill=(0, 0, 0))
        fn = OUTA / f'q3_heat_v1_vs_v2_{lb}.jpg'
        sh.save(fn, quality=88)
        sheets.append(str(fn))
        out[f'sheet_{lb}'] = {'path': str(fn), 'frames': [{'obs': F.obs.values[f], 'k': int(F.k.values[f]),
                                                            'why': w} for f, w in zip(chosen, why)],
                              'vmax': {'v1': vm1, 'v2': vm2, 'v2_pooled': vm2p}}
    res['Q3'] = out
    jdump(res, OUTA / 'q_diag.json')


# ---------------------------------------------------------------------------------------------- Q4 temporal
def windows(torch, X, kind, agg, w):
    """X (n, W) float cuda (one video, time-ordered) -> windowed (n, W)."""
    Fn = torch.nn.functional
    if w == 1:
        return X
    x = X.T[None]  # (1, W, n)
    if kind == 'centred':
        if agg == 'max':
            y = Fn.max_pool1d(x, w, 1, w // 2)
        else:
            y = Fn.avg_pool1d(x, w, 1, w // 2, count_include_pad=False)
    else:
        if agg == 'max':
            y = Fn.max_pool1d(Fn.pad(x, (w - 1, 0), value=0.0), w, 1)
        else:
            cs = torch.cumsum(Fn.pad(x, (1, 0)), 2)
            n = x.shape[2]
            idx = torch.arange(n, device=X.device)
            lo = (idx + 1 - w).clamp_min(0)
            y = (cs[..., idx + 1] - cs[..., lo]) / (idx + 1 - lo).float()
    return y[0].T


def cmd_temporal(args):
    import torch
    dev = torch.device('cuda')
    P = N.load_plan()
    vids = [v for v in P['videos'] if v['annotated']]
    a = pd.read_csv(REPO / C['ann'], usecols=['observation_id', 'Y_nn', 'Y_np', 'Y_nt'])
    rows = np.concatenate([np.arange(v['src_lo'], v['src_lo'] + v['n']) for v in vids])
    y = a.iloc[rows]
    Y = np.stack([(y.Y_nn.values > 0) | (y.Y_np.values > 0), y.Y_nt.values > 0], 1)
    unit = pd.Series(y.observation_id.values).str.rsplit('_', n=2).str[0].values
    sc = json.loads((N.OUT / 'score.json').read_text())
    units = sc['units']
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    half_all = {sn: np.array(h)[uidx] for sn, h in sc['splits'].items()}
    Yg = torch.from_numpy(Y).to(dev)
    lens = [v['n'] for v in vids]
    cfgs = [('centred', 'mean', 1)] + [(k_, ag, w) for k_, ws in (('centred', (3, 5, 11, 21)), ('causal', (2, 5, 10, 20)))
                                       for ag in ('mean', 'max') for w in ws]
    out = {'rule': 'helps iff mean dR~ >= 0.01 and dR~ > 0 for all 3 seeds; multi-scale iff best windows differ and '
                   'each beats the other label\'s best by >= 0.01', 'fps': 5, 'res': {}}
    prev = OUTA / 'q4_temporal.json'
    if prev.exists():
        out = json.loads(prev.read_text())
    for seed in (0, 1, 2):
        name = f'A3_w512_s{seed}'
        Xs = [torch.from_numpy(np.load(N.BEE / 'videos' / f'{v["obs"]}.npz')[f'{name}|max'].astype(np.float32))
              for v in vids]
        assert [len(x) for x in Xs] == lens
        for kind, agg, w in cfgs:
            key = f'{name}|{kind}|{agg}|{w}'
            if key in out['res']:
                continue
            t0 = time.time()
            Xw = torch.cat([windows(torch, x.to(dev), kind, agg, w).half() for x in Xs])
            o, _ = N.score_set(torch, Xw, Yg, half_all, dev)
            out['res'][key] = {lb: {k_: o['mean'][lb][k_] for k_ in ('R', 'A', 'modal_neuron', 'modal_share')}
                               for lb in LB[:2]}
            if w == 1:
                ref = sc['sets'][f'{name}|max|all']['mean']
                out['res'][key]['identity_vs_score_json'] = {lb: o['mean'][lb]['R'] - ref[lb]['R'] for lb in LB[:2]}
                assert all(abs(v_) < 1e-6 for v_ in out['res'][key]['identity_vs_score_json'].values()), \
                    out['res'][key]
            log(f'{key}: ' + json.dumps(out['res'][key]) + f' ({time.time() - t0:.0f}s)')
            del Xw
            jdump(out, prev)
    # decision
    dec = {}
    for lb in LB[:2]:
        base = np.array([out['res'][f'A3_w512_s{s}|centred|mean|1'][lb]['R'] for s in (0, 1, 2)])
        tab = {}
        for kind, agg, w in cfgs:
            r = np.array([out['res'][f'A3_w512_s{s}|{kind}|{agg}|{w}'][lb]['R'] for s in (0, 1, 2)])
            d = r - base
            tab[f'{kind}|{agg}|{w}'] = {'R_mean': float(r.mean()), 'dR_mean': float(d.mean()),
                                        'dR_per_seed': d.tolist(),
                                        'helps': bool(d.mean() >= 0.01 and (d > 0).all()) if w > 1 else False}
        helping = [k_ for k_, v_ in tab.items() if v_['helps']]
        best = max(helping, key=lambda k_: tab[k_]['R_mean']) if helping else 'centred|mean|1'
        dec[lb] = {'table': tab, 'best': best}
    bn, bt = dec['nose_nose']['best'], dec['nose_tail']['best']
    ms = (bn != bt and dec['nose_nose']['table'][bn]['R_mean'] - dec['nose_nose']['table'][bt]['R_mean'] >= 0.01
          and dec['nose_tail']['table'][bt]['R_mean'] - dec['nose_tail']['table'][bn]['R_mean'] >= 0.01)
    out['decision'] = {**dec, 'multi_scale_justified': bool(ms)}
    jdump(out, prev)
    # probe level (seed 2): c_t vs [c_t, centred means 5 / 11 / 21]
    if 'probe' not in out:
        Xs = [torch.from_numpy(np.load(N.BEE / 'videos' / f'{v["obs"]}.npz')[f'{V2}|max'].astype(np.float32))
              for v in vids]
        X0 = torch.cat([x for x in Xs]).to(dev)
        half_of = {'frame': half_all, 'unit_idx': uidx}
        p0, _ = crossfit_probe(torch, X0, Y, half_of, SPL5, dev)
        Xm = torch.cat([torch.cat([x.to(dev)] + [windows(torch, x.to(dev), 'centred', 'mean', w) for w in (5, 11, 21)],
                                  1) for x in Xs])
        del X0
        pm, _ = crossfit_probe(torch, Xm, Y, half_of, SPL5, dev)
        out['probe'] = {'c_t': p0, 'c_t_plus_means_1_2_4s': pm,
                        'rule': {lb: compare(p0, pm, lb) for lb in LB[:2]}}
        log('probe ' + json.dumps(out['probe']))
        jdump(out, prev)
    log(json.dumps(out['decision'], default=str)[:4000])


# ---------------------------------------------------------------------------------------------- Q1 style
def mask_video(v, workers=2):
    """-> (n, 64, 64) bool tight 512 px mask of every 5 fps frame of video v (rotated), via chunked decode."""
    pb = np.load(REPO / 'dataset/mice/v1/eci/fg448al/background' / f'{v["obs"]}.npz')['pix_bg'].astype(np.int16)
    out = np.zeros((v['n'], G, G), bool)
    for k0 in range(0, v['n'], 300):
        m = min(300, v['n'] - k0)
        fr = R4.decode(v['path'], v['offset'] + C['step'] * k0, m, C['step'], 512, v['hflip'], C['fps'],
                       threads=workers)
        fr = np.rot90(fr, v['rot'], axes=(1, 2)).astype(np.int32)
        gr = (fr[..., 0] * 19595 + fr[..., 1] * 38470 + fr[..., 2] * 7471 + 32768) >> 16
        dark = (gr < 60) & ((pb[None] - gr) > 40)
        out[k0:k0 + m] = dark.reshape(m, G, 8, G, 8).mean((2, 4)) >= 0.25
    return out


def geom_frames(M):
    """M (n, 64, 64) bool -> ncomp (>= 5), maxc, elongation of the largest component, centroid (r, c)."""
    from scipy import ndimage
    n = len(M)
    nc, mx, el = np.zeros(n, np.int16), np.zeros(n, np.int16), np.zeros(n, np.float32)
    st = np.ones((3, 3), bool)
    rr, cc = np.mgrid[0:G, 0:G]
    for i in range(n):
        lab, k = ndimage.label(M[i], structure=st)
        if k == 0:
            continue
        sz = np.bincount(lab.ravel())
        sz[0] = 0
        big = sz[sz >= 5]
        nc[i] = len(big)
        j = int(np.argmax(sz))
        mx[i] = sz[j]
        if sz[j] >= 3:
            m = lab == j
            r, c = rr[m].astype(float), cc[m].astype(float)
            cv = np.cov(np.stack([r, c]))
            ev = np.linalg.eigvalsh(cv)
            el[i] = np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-3))
    return nc, mx, el


def runs(x):
    """bool (n,) -> (starts, ends exclusive)."""
    d = np.diff(np.r_[0, x.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def cmd_style(args):
    P = N.load_plan()
    vids = [v for v in P['videos'] if v['annotated']]
    sdir = BEEA / 'style'
    sdir.mkdir(parents=True, exist_ok=True)
    nw = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))

    def one(v):
        f = sdir / f'{v["obs"]}.npz'
        if f.exists():
            return
        t0 = time.time()
        M = mask_video(v)
        nc, mx, el = geom_frames(M)
        loc = N.local_dir() / f'{v["obs"]}.npz'
        np.savez(loc, mask=np.packbits(M.reshape(len(M), -1), axis=1), ncomp=nc, maxc=mx, elong=el)
        R4.atomic_copy(loc, f)
        loc.unlink()
        log(f'style {v["obs"]}: {len(M)} frames in {time.time() - t0:.0f}s')
    with ThreadPoolExecutor(max(1, nw // 2)) as ex:
        list(ex.map(one, vids))
    style_analysis(vids)


def style_analysis(vids):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    sdir = BEEA / 'style'
    e = pd.read_csv(REPO / 'data/mice/v1/experiment.csv').set_index('observation_id')
    a = pd.read_csv(REPO / C['ann'], usecols=['observation_id', 'Y_nn', 'Y_np', 'Y_nt'])
    Z = {}
    for v in vids:
        z = np.load(sdir / f'{v["obs"]}.npz')
        Z[v['obs']] = {k_: z[k_] for k_ in z.files}
    # mask check vs the 1024 px diagnosis tokens
    root = R2.BEE2 / 'diag'
    Fd = pd.read_parquet(root / 'frames.parquet')
    tf = np.load(root / 'tok_frame.npy')
    pos = np.load(root / 'tok_pos.npy').astype(np.int64)
    st_ = np.r_[0, np.cumsum(np.bincount(tf, minlength=len(Fd)))]
    jac = []
    for i in range(0, len(Fd), 37):
        o, k = Fd.obs.values[i], int(Fd.k.values[i])
        m512 = np.unpackbits(Z[o]['mask'][k])[:G * G].astype(bool)
        m1024 = np.zeros(G * G, bool)
        m1024[pos[st_[i]:st_[i + 1]]] = True
        u = (m512 | m1024).sum()
        jac.append((m512 & m1024).sum() / u if u else 1.0)
    res = {'mask_check_jaccard_512_vs_1024_tight': {'n_frames': len(jac), 'mean': float(np.mean(jac)),
                                                    'median': float(np.median(jac))}}
    # A1, contact, end-to-end threshold
    per = {}
    for v in vids:
        z = Z[v['obs']]
        nc, mx, el = z['ncomp'], z['maxc'].astype(float), z['elong']
        four = mx[nc == 4]
        per[v['obs']] = {'A1': float(np.median(four)) if len(four) >= 20 else np.nan, 'mx': mx, 'el': el}
    glob = float(np.nanmedian([p['A1'] for p in per.values()]))
    for p in per.values():
        if not np.isfinite(p['A1']):
            p['A1'] = glob
        p['contact'] = p['mx'] >= 1.5 * p['A1']
    thr_el = float(np.quantile(np.concatenate([p['el'][p['contact']] for p in per.values()]), 0.75))
    res['end_to_end_elongation_threshold_p75'] = thr_el
    res['contact_frame_share'] = float(np.mean(np.concatenate([p['contact'] for p in per.values()])))
    codes = {'nn': 'Y_nn', 'np': 'Y_np', 'nt': 'Y_nt'}
    vrows = []
    lagrows = []
    for v in vids:
        o = v['obs']
        y = a.iloc[v['src_lo']:v['src_lo'] + v['n']]
        assert (y.observation_id == o).all()
        L = {c_: y[col].values > 0 for c_, col in codes.items()}
        L['nose_nose'] = L['nn'] | L['np']
        L['any'] = L['nose_nose'] | L['nt']
        p = per[o]
        ct = p['contact']
        e2e = ct & (p['el'] >= thr_el)
        cs, ce = runs(ct)
        ev_id = np.full(v['n'], -1)
        for i, (s0, s1) in enumerate(zip(cs, ce)):
            ev_id[s0:s1] = i
        ev_len = (ce - cs) / 5.0
        fr_ev_len = np.where(ev_id >= 0, ev_len[np.maximum(ev_id, 0)], 0.0)
        row = {'obs': o, 'annotator': e.loc[o, 'annotator'] if isinstance(e.loc[o, 'annotator'], str) else '?',
               'pool': e.loc[o, 'pool'], 'line': e.loc[o, 'line'], 'phase': e.loc[o, 'phase'],
               'minutes': v['n'] / 300, 'contact_events': len(cs)}
        long_e2e = [(s0, s1) for s0, s1 in zip(cs, ce) if (s1 - s0) > 5 and e2e[s0:s1].mean() >= 0.5]
        row['long_e2e_events'] = len(long_e2e)
        row['long_e2e_unlabelled'] = sum(not L['any'][s0:s1].any() for s0, s1 in long_e2e)
        for c_ in ('nn', 'np', 'nt', 'nose_nose', 'any'):
            lab = L[c_]
            row[f'rate_{c_}'] = float(lab.mean())
            rs, re_ = runs(lab)
            row[f'runs_{c_}'] = len(rs)
            row[f'passby_frames_{c_}'] = int((lab & (fr_ev_len < 0.5)).sum())
            row[f'labelled_frames_{c_}'] = int(lab.sum())
            nocon = 0
            for s0, s1 in zip(rs, re_):
                ids = ev_id[max(0, s0 - 1):min(v['n'], s1 + 1)]
                ids = ids[ids >= 0]
                if not len(ids):
                    nocon += 1
                    continue
                j = np.bincount(ids).argmax()
                lagrows.append({'obs': o, 'annotator': row['annotator'], 'line': row['line'], 'code': c_,
                                'onset_lag_s': (s0 - cs[j]) / 5, 'offset_lag_s': (ce[j] - s1) / 5,
                                'event_len_s': ev_len[j], 'run_len_s': (s1 - s0) / 5})
            row[f'runs_no_contact_{c_}'] = nocon
        for x in ('nn', 'np', 'nt'):
            for y_ in ('nn', 'np', 'nt'):
                if x != y_:
                    row[f'co_{x}_{y_}'] = int((L[x] & L[y_]).sum())
        vrows.append(row)
    Vd = pd.DataFrame(vrows)
    Lg = pd.DataFrame(lagrows)
    # BORIS bouts
    B = []
    for o, r in e[e.annotation_file.notna()].iterrows():
        d = pd.read_csv(REPO / 'data/mice/v1/annotations' / r.annotation_file)
        d = d[d.behavior_type.notna()]
        du = pd.to_numeric(d['Duration (s)'], errors='coerce')
        du = du.where(du.notna(), pd.to_numeric(d['Stop (s)'], errors='coerce') -
                      pd.to_numeric(d['Start (s)'], errors='coerce'))
        B.append(pd.DataFrame({'obs': o, 'code': d.behavior_type.replace({'np?': 'np'}).values, 'dur': du.values}))
    B = pd.concat(B)
    B['annotator'] = B.obs.map(Vd.set_index('obs').annotator)
    B['line'] = B.obs.map(Vd.set_index('obs').line)
    B['pool'] = B.obs.map(Vd.set_index('obs').pool)
    tested = [an for an, g in Vd.groupby('annotator') if g.pool.nunique() >= 3]
    res['tested_annotators'] = tested

    def ann_stats(V, Bb, Lgg):
        st = {}
        for an, g in V.groupby('annotator'):
            bb, ll = Bb[Bb.annotator == an], Lgg[Lgg.annotator == an]
            d = {'videos': int(len(g)), 'pools': int(g.pool.nunique()),
                 'bouts_per_min_any': float(g.runs_any.sum() / g.minutes.sum()),
                 'miss_proxy_long_e2e_unlabelled': float(g.long_e2e_unlabelled.sum() / max(1, g.long_e2e_events.sum())),
                 'long_e2e_events': int(g.long_e2e_events.sum())}
            for c_ in ('nn', 'np', 'nt', 'nose_nose'):
                d[f'{c_}_rate'] = float((g[f'rate_{c_}'] * g.minutes).sum() / g.minutes.sum())
                d[f'{c_}_passby_share'] = float(g[f'passby_frames_{c_}'].sum() / max(1, g[f'labelled_frames_{c_}'].sum()))
                d[f'{c_}_runs_no_contact_share'] = float(g[f'runs_no_contact_{c_}'].sum() / max(1, g[f'runs_{c_}'].sum()))
                lc = ll[ll.code == c_]
                d[f'{c_}_onset_lag_median_s'] = float(lc.onset_lag_s.median()) if len(lc) else np.nan
                d[f'{c_}_offset_lag_median_s'] = float(lc.offset_lag_s.median()) if len(lc) else np.nan
                d[f'{c_}_contact_event_len_median_s'] = float(lc.event_len_s.median()) if len(lc) else np.nan
            for c_ in ('nn', 'np', 'nt'):
                du = bb[bb.code == c_].dur.dropna()
                d[f'{c_}_boris_dur_median_s'] = float(du.median()) if len(du) else np.nan
                d[f'{c_}_boris_share_lt_0.4s'] = float((du < 0.4).mean()) if len(du) else np.nan
                d[f'{c_}_usage_share'] = float(g[f'labelled_frames_{c_}'].sum() /
                                              max(1, sum(g[f'labelled_frames_{x}'].sum() for x in ('nn', 'np', 'nt'))))
                for y_ in ('nn', 'np', 'nt'):
                    if y_ != c_:
                        d[f'P({y_}|{c_})'] = float(g[f'co_{c_}_{y_}'].sum() / max(1, g[f'labelled_frames_{c_}'].sum()))
            st[an] = d
        return st
    res['per_annotator'] = ann_stats(Vd, B, Lg)
    res['per_line_annotator'] = {ln: ann_stats(Vd[Vd.line == ln], B[B.line == ln], Lg[Lg.line == ln])
                                 for ln in sorted(Vd.line.unique())}

    # hypotheses (pool bootstrap of A - others)
    def boot_diff(Vsub, Bsub, an, fn, nb=500, seed=0):
        pools = Vsub.pool.unique()
        g = np.random.default_rng(seed)
        obs = fn(Vsub, Bsub, an)
        bs = []
        pa = Vsub[Vsub.annotator == an].pool.unique()
        po = Vsub[Vsub.annotator != an].pool.unique()
        for _ in range(nb):
            sa = g.choice(pa, len(pa))
            so = g.choice(po, len(po))
            Vb = pd.concat([Vsub[(Vsub.pool == p_) & (Vsub.annotator == an)] for p_ in sa] +
                           [Vsub[(Vsub.pool == p_) & (Vsub.annotator != an)] for p_ in so])
            bs.append(fn(Vb, Bsub, an))
        return {'diff': float(obs), 'ci95': [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))]}

    def miss_diff(Vs, Bs, an):
        A_, O_ = Vs[Vs.annotator == an], Vs[Vs.annotator != an]
        return A_.long_e2e_unlabelled.sum() / max(1, A_.long_e2e_events.sum()) - \
            O_.long_e2e_unlabelled.sum() / max(1, O_.long_e2e_events.sum())

    def passby_diff(c_):
        def f(Vs, Bs, an):
            A_, O_ = Vs[Vs.annotator == an], Vs[Vs.annotator != an]
            return A_[f'passby_frames_{c_}'].sum() / max(1, A_[f'labelled_frames_{c_}'].sum()) - \
                O_[f'passby_frames_{c_}'].sum() / max(1, O_[f'labelled_frames_{c_}'].sum())
        return f
    hyp = {}
    for an in tested:
        A_, O_ = Vd[Vd.annotator == an], Vd[Vd.annotator != an]
        bpm = (A_.runs_any.sum() / A_.minutes.sum(), O_.runs_any.sum() / O_.minutes.sum())
        md = boot_diff(Vd, B, an, miss_diff)
        h = {'H-a': {'miss_proxy_diff': md, 'bouts_per_min_A_vs_others': bpm,
                     'supported': bool(md['diff'] >= 0.10 and md['ci95'][0] > 0 and bpm[0] < bpm[1])}}
        for c_ in ('nn', 'np', 'nt'):
            pdd = boot_diff(Vd, B, an, passby_diff(c_))
            sh = (float((B[(B.annotator == an) & (B.code == c_)].dur < 0.4).mean()),
                  float((B[(B.annotator != an) & (B.code == c_)].dur < 0.4).mean()))
            h[f'H-b {c_}'] = {'passby_diff': pdd, 'boris_share_lt_0.4s_A_vs_others': sh,
                              'supported': bool(pdd['diff'] >= 0.05 and pdd['ci95'][0] > 0 and sh[0] > sh[1])}
        lm = {}
        for ln in sorted(Vd.line.unique()):
            Vl = Vd[Vd.line == ln]
            if (Vl.annotator == an).any() and (Vl.annotator != an).any():
                lm[ln] = {'miss_diff': float(miss_diff(Vl, B, an)),
                          **{f'passby_diff_{c_}': float(passby_diff(c_)(Vl, B, an)) for c_ in ('nn', 'np', 'nt')}}
        h['line_matched'] = lm
        hyp[an] = h
    res['hypotheses'] = hyp
    # style separability
    feats = []
    for c_ in ('nn', 'np', 'nt'):
        Vd[f'bpm_{c_}'] = Vd[f'runs_{c_}'] / Vd.minutes
        Vd[f'passby_{c_}'] = Vd[f'passby_frames_{c_}'] / Vd[f'labelled_frames_{c_}'].clip(lower=1)
        bb = B[B.code == c_].groupby('obs').dur
        Vd[f'durmed_{c_}'] = Vd.obs.map(bb.median()).fillna(0)
        Vd[f'short_{c_}'] = Vd.obs.map(bb.apply(lambda x: (x < 0.4).mean())).fillna(0)
        Vd[f'lrate_{c_}'] = np.log(Vd[f'rate_{c_}'] + 1e-4)
        feats += [f'lrate_{c_}', f'bpm_{c_}', f'durmed_{c_}', f'short_{c_}', f'passby_{c_}']
    Vd['onset_med'] = Vd.obs.map(Lg[Lg.code == 'any'].groupby('obs').onset_lag_s.median()).fillna(0)
    feats.append('onset_med')
    S_ = Vd[Vd.annotator.isin(tested)].copy()
    X = S_[feats].values.astype(float)
    grp = S_.line.astype(str) + '_' + S_.phase.astype(str)
    X = X - pd.DataFrame(X).groupby(grp.values).transform('mean').values
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    yA = S_.annotator.values
    pools = S_.pool.values

    def lopo(yv):
        pred = np.empty(len(yv), object)
        for p_ in np.unique(pools):
            te = pools == p_
            clf = LogisticRegression(C=0.5, max_iter=2000).fit(X[~te], yv[~te])
            pred[te] = clf.predict(X[te])
        return balanced_accuracy_score(yv, pred)
    obs_acc = lopo(yA)
    pool_ann = S_.groupby('pool').annotator.agg(lambda x: x.mode()[0])
    g = np.random.default_rng(3)
    null = []
    for _ in range(200):
        perm = pd.Series(g.permutation(pool_ann.values), index=pool_ann.index)
        null.append(lopo(S_.pool.map(perm).values))
    res['style_separability'] = {'annotators': tested, 'n_videos': int(len(S_)), 'n_pools': int(S_.pool.nunique()),
                                 'features': feats, 'balanced_accuracy_LOPO': float(obs_acc), 'chance': 1 / len(tested),
                                 'perm_null_mean': float(np.mean(null)), 'perm_null_p95': float(np.quantile(null, .95)),
                                 'p_perm': float((np.array(null) >= obs_acc).mean()),
                                 'separable': bool((np.array(null) >= obs_acc).mean() < 0.05)}
    Vd.to_csv(OUTA / 'style_per_video.csv', index=False)
    jdump(res, OUTA / 'q1_style.json')
    log(json.dumps({k: res[k] for k in ('mask_check_jaccard_512_vs_1024_tight', 'style_separability')}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('labels', 'diag', 'temporal', 'style'))
    args = p.parse_args()
    {'labels': cmd_labels, 'diag': cmd_diag, 'temporal': cmd_temporal, 'style': cmd_style}[args.step](args)


if __name__ == '__main__':
    main()
