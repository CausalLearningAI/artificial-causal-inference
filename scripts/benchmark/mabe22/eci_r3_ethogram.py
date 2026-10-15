"""
MABe22 ECI round 3, E4 ETHOGRAM step 2 (CPU Slurm job, -p defaultp): scoring of the full-clip firing bits written by
eci_r3_fullclip.py, exactly the E4 rule of eci_round3.sh (frozen; anything beyond it is labelled EXPLORATORY).

Per clip x neuron x SAE seed (seeds 100..105, firing = code > 0 under theta*), on a grid of label indices:
    time budget  fraction of grid frames firing
    bout rate    maximal runs of consecutive firing grid frames per minute (clip = 1,800 label frames = 1 min)
    latency      seconds to the first firing grid frame (label index / 30), 60 if never
Annotation per clip x behaviour (9, eci_common.BEHAVIOURS): time budget = fraction of the 1,800 label frames positive;
bout rate = maximal positive runs of the 30 fps labels per minute.
Cross-fit (clip level): halves = eci_train.halves on plan/eval_frames.parquet EXACTLY as eci_train does (split seeds 0,
1), mapped clip -> half (constant per clip, asserted). On half A the neuron with the largest Spearman rho between its
summary and the annotated summary (eligible: fires in >= 10 clips of half A); its Spearman rho on half B is reported
(NaN, i.e. constant on B, counted as 0 and counted); both directions averaged, then the 2 split seeds; mean +- sd
(ddof 1) over the 6 SAE seeds. Time budget vs time budget, bout rate vs bout rate.
Condition decoding (strain, lights: label rows, constant per clip, asserted): features (a) ethogram (time budget, bout
rate, latency of the 4,096 neurons = 12,288) at 5 fps, (b) mean frame code (4,096); StandardScaler + L2 logistic
regression, C in {1e-3, 1e-2, 1e-1, 1} by inner stratified 3-fold CV (inner score: log loss), outer stratified 5-fold
CV (shuffle, random_state 0), one row per clip; out-of-fold predictions pooled over the 1,828 clips -> macro F1
(argmax), AUROC; chance = majority-class macro F1, AUROC 0.5, and one label-permutation run (rng 0) per seed.
Frame rate: on the pre-registered subset (eci_r3_fullclip.subset15) the same cross-fitted correlations at 5 fps
(label i % 6 == 0) and 15 fps (i = 0, 2, ..., 1798); reading "5 fps is enough" for a summary iff |rho_15 - rho_5| < 0.05
(mean over seeds) on every primary behaviour (oral_oral, oral_genital, approaches, huddles); plus the 5 vs 15 fps
Spearman agreement over the subset clips of each selected neuron's summary (the 5 fps pick and the 15 fps pick).
Outputs: OUT/r3/e4/ethogram.json, OUT/r3/e4/ethogram.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eci_common import BEHAVIOURS, EXCLUDE, PLAN, RAW, SHORT, log  # noqa: E402
from eci_r3_fullclip import E4, SEEDS, all_clips, subset15  # noqa: E402
from eci_train import halves  # noqa: E402

P4 = ['oral_oral_contact', 'oral_genital_contact', 'approaches', 'huddles']
MIN_CLIPS = 10
NS = len(SEEDS)


def runs(f, axis):
    """number of maximal runs of True along axis."""
    f = np.moveaxis(f, axis, -1)
    return f[..., 0].astype(np.int32) + (f[..., 1:] & ~f[..., :-1]).sum(-1)


def summarise(f, li):
    """f (C, S, T, 4096) bool firing, li (T,) label indices -> tb, br, lat (C, S, 4096) float32."""
    tb = f.mean(2, dtype=np.float32)
    br = runs(f, 2).astype(np.float32) / 1.0  # per minute: a clip is 1,800 / 30 / 60 = 1 min
    any_ = f.any(2)
    first = f.argmax(2)
    lat = np.where(any_, li[first] / 30.0, 60.0).astype(np.float32)
    return tb, br, lat


def load_shards(fps):
    js = sorted(E4.glob(f'fps{fps}_*of*.json'))
    js = [j for j in js if 'smoke' not in j.name]
    meta = [json.loads(j.read_text()) for j in js]
    assert meta and len(meta) == meta[0]['nshard'], [j.name for j in js]
    return sorted(meta, key=lambda m: m['shard'])


def unpack_iter(meta, chunk=40):
    for m in meta:
        B = np.load(E4 / f"{m['key']}_bits.npy", mmap_mode='r')
        for a in range(0, len(m['clips']), chunk):
            yield m, a, np.unpackbits(np.asarray(B[a:a + chunk]), axis=-1).astype(bool)


def annotations(clips):
    sub = np.load(RAW / 'mouse_submission_labels.npy', allow_pickle=True).item()
    fm = np.load(RAW / 'mouse_frame_number_map.npy', allow_pickle=True).item()
    assert set(fm) - EXCLUDE == set(clips)
    vocab = list(sub['vocabulary'])
    L = sub['label_array']
    tb = np.zeros((len(clips), len(BEHAVIOURS)))
    br = np.zeros_like(tb)
    cond, nonconst = {}, {}
    for k in ('strain', 'lights'):
        cond[k] = np.zeros(len(clips), np.int64)
        nonconst[k] = 0
    for ci, c in enumerate(clips):
        a, b = fm[c]
        assert b - a == 1800
        for j, beh in enumerate(BEHAVIOURS):
            v = L[vocab.index(beh), a:b]
            assert np.isfinite(v).all()
            p = v == 1
            tb[ci, j] = p.mean()
            br[ci, j] = runs(p[None], 1)[0] / 1.0
        for k in cond:
            v = L[vocab.index(k), a:b]
            u = np.unique(v)
            nonconst[k] += int(len(u) > 1)
            cond[k][ci] = int(np.round(np.median(v)))
    return tb, br, cond, nonconst


def ranks(X):
    return rankdata(X, axis=0).astype(np.float64)


def pick_and_test(S, fire, y, A, B):
    """S (C, m) neuron summary, fire (C, m) bool, y (C,), A / B bool clip masks -> (j, rho_B, n_eligible)."""
    elig = fire[A].sum(0) >= MIN_CLIPS
    if not elig.any():
        return -1, 0.0, 0, True
    idx = np.flatnonzero(elig)
    R = ranks(S[A][:, idx])
    ry = rankdata(y[A]).astype(np.float64)
    R -= R.mean(0)
    ry -= ry.mean()
    den = np.sqrt((R ** 2).sum(0) * (ry ** 2).sum())
    rho = np.where(den > 0, (R * ry[:, None]).sum(0) / np.where(den > 0, den, 1), -np.inf)
    j = int(idx[int(np.argmax(rho))])
    r = spearmanr(S[B, j], y[B]).statistic
    nan = not np.isfinite(r)
    return j, (0.0 if nan else float(r)), int(len(idx)), nan


def crossfit(S, fire, y, H):
    """H {split: (C,) 0/1} -> mean test rho over directions then splits, details."""
    per_split, det, nans = [], [], 0
    for s, h in H.items():
        v = []
        for a in (0, 1):
            j, r, ne, nan = pick_and_test(S, fire, y, h == a, h == 1 - a)
            nans += nan
            v.append(r)
            det.append({'split': s, 'select_half': a, 'neuron': j, 'rho_test': r, 'n_eligible': ne})
        per_split.append(np.mean(v))
    return float(np.mean(per_split)), det, nans


def corr_table(SUM, ANN, H, clipmask=None):
    """SUM {'tb': (S, C, m), 'br': ...} -> {beh: {kind: {'per_seed': [...], 'mean', 'sd', 'details'}}}"""
    out = {}
    for j, b in enumerate(BEHAVIOURS):
        out[b] = {}
        for kind in ('tb', 'br'):
            ps, dets, nn = [], [], 0
            for k in range(NS):
                S = SUM[kind][k]
                fire = SUM['tb'][k] > 0
                y = ANN[kind][:, j]
                if clipmask is not None:
                    S, fire, y = S[clipmask], fire[clipmask], y[clipmask]
                    Hs = {s: h[clipmask] for s, h in H.items()}
                else:
                    Hs = H
                r, d, nans = crossfit(S, fire, y, Hs)
                ps.append(r)
                dets.append(d)
                nn += nans
            out[b][kind] = {'per_seed': ps, 'mean': float(np.mean(ps)), 'sd': float(np.std(ps, ddof=1)),
                            'details': dets, 'n_nan_test_rho': nn}
    return out


def decode(Xs, y, perm_seed=None):
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    yy = y.copy()
    if perm_seed is not None:
        yy = np.random.default_rng(perm_seed).permutation(yy)
    est = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
                       {'logisticregression__C': [1e-3, 1e-2, 1e-1, 1.0]},
                       cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring='neg_log_loss', n_jobs=1)
    pr = cross_val_predict(est, Xs, yy, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba',
                           n_jobs=1)[:, 1]
    return {'macro_f1': float(f1_score(yy, pr > 0.5, average='macro')), 'auroc': float(roc_auc_score(yy, pr))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--njobs', type=int, default=24)
    ap.add_argument('--skip-decode', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    clips, EV = all_clips()
    cidx = {c: k for k, c in enumerate(clips)}
    H = {}
    for s in (0, 1):
        h = pd.Series(halves(EV, s), index=EV.index)
        g = h.groupby(EV['clip']).agg(['min', 'max'])
        assert (g['min'] == g['max']).all()
        H[s] = g['min'].reindex(clips).values.astype(np.int64)
    log(f'halves: ' + ', '.join(f'split{s}: {int((h == 0).sum())}/{int((h == 1).sum())}' for s, h in H.items()))
    atb, abr, cond, nonconst = annotations(clips)
    ANN = {'tb': atb, 'br': abr}
    log(f'annotations done; non-constant condition clips {nonconst}; strain rate {cond["strain"].mean():.3f} '
        f'lights rate {cond["lights"].mean():.3f}')
    # ------------------------------------------------------------------ 5 fps summaries, all clips
    m5 = load_shards(5)
    li5 = np.array(m5[0]['label_i'])
    assert (li5 == np.arange(0, 1800, 6)).all()
    C, m = len(clips), 4096
    SUM5 = {k: np.zeros((NS, C, m), np.float32) for k in ('tb', 'br', 'lat')}
    MEANC = np.zeros((NS, C, m), np.float32)
    seen = np.zeros(C, bool)
    for meta, a, f in unpack_iter(m5):
        rows = np.array([cidx[c] for c in meta['clips'][a:a + len(f)]])
        tb, br, lat = summarise(f, li5)
        for k, v in (('tb', tb), ('br', br), ('lat', lat)):
            SUM5[k][:, rows] = v.transpose(1, 0, 2)
        seen[rows] = True
    for meta in m5:
        mc = np.load(E4 / f"{meta['key']}_meancode.npy")
        rows = np.array([cidx[c] for c in meta['clips']])
        MEANC[:, rows] = mc.transpose(1, 0, 2)
    assert seen.all(), f'{(~seen).sum()} clips missing at 5 fps'
    log(f'5 fps summaries: {C} clips; mean time budget {SUM5["tb"].mean():.4f}, neurons firing in >= 10 clips '
        f'(seed 100) {int(((SUM5["tb"][0] > 0).sum(0) >= 10).sum())}')
    res = {'n_clips': C, 'halves_source': 'eci_train.halves(plan/eval_frames.parquet, seed), mapped clip -> half',
           'condition_nonconstant_clips': nonconst,
           'condition_rates': {k: float(v.mean()) for k, v in cond.items()}}
    res['corr_5fps_all'] = corr_table(SUM5, ANN, H)
    log('corr 5 fps all clips: ' + ', '.join(f"{SHORT[b]} tb {r['tb']['mean']:.3f} br {r['br']['mean']:.3f}"
                                            for b, r in res['corr_5fps_all'].items()))
    # ------------------------------------------------------------------ 15 fps subset
    sub, sinfo = subset15(clips)
    m15 = load_shards(15)
    assert sum((x['clips'] for x in m15), []) == sub, '15 fps shards do not hold the pre-registered subset'
    li15o = np.array(m15[0]['label_i'])
    li15 = np.arange(0, 1800, 2)
    assert (li15o == li15[li15 % 6 != 0]).all()
    smask = np.zeros(C, bool)
    smask[[cidx[c] for c in sub]] = True
    SUM15 = {k: np.zeros((NS, C, m), np.float32) for k in ('tb', 'br', 'lat')}
    B5 = {mm['key']: np.load(E4 / f"{mm['key']}_bits.npy", mmap_mode='r') for mm in m5}
    loc5 = {c: (mm['key'], q) for mm in m5 for q, c in enumerate(mm['clips'])}
    on5 = np.flatnonzero(li15 % 6 == 0)
    off5 = np.flatnonzero(li15 % 6 != 0)
    for meta, a, f in unpack_iter(m15):
        cl = meta['clips'][a:a + len(f)]
        full = np.zeros((len(cl), NS, len(li15), m), bool)
        full[:, :, off5] = f
        for q, c in enumerate(cl):
            kk, r = loc5[c]
            full[q][:, on5] = np.unpackbits(np.asarray(B5[kk][r]), axis=-1).astype(bool)  # (S, 300, 4096)
        rows = np.array([cidx[c] for c in cl])
        tb, br, lat = summarise(full, li15)
        for k, v in (('tb', tb), ('br', br), ('lat', lat)):
            SUM15[k][:, rows] = v.transpose(1, 0, 2)
    log(f'15 fps summaries: {len(sub)} subset clips')
    res['subset15'] = sinfo
    res['corr_5fps_subset'] = corr_table(SUM5, ANN, H, smask)
    res['corr_15fps_subset'] = corr_table(SUM15, ANN, H, smask)
    # 5 vs 15 fps agreement of each selected neuron's summary (over the subset clips)
    agree = {}
    for b in BEHAVIOURS:
        agree[b] = {}
        for kind in ('tb', 'br'):
            v = {'pick5': [], 'pick15': [], 'same_neuron': []}
            for k in range(NS):
                d5 = res['corr_5fps_subset'][b][kind]['details'][k]
                d15 = res['corr_15fps_subset'][b][kind]['details'][k]
                for x5, x15 in zip(d5, d15):
                    for nm, x in (('pick5', x5), ('pick15', x15)):
                        j = x['neuron']
                        if j >= 0:
                            r = spearmanr(SUM5[kind][k][smask, j], SUM15[kind][k][smask, j]).statistic
                            v[nm].append(float(r) if np.isfinite(r) else 0.0)
                    v['same_neuron'].append(float(x5['neuron'] == x15['neuron']))
            agree[b][kind] = {q: float(np.mean(w)) for q, w in v.items()}
    res['agree_5_vs_15'] = agree
    verdict = {}
    for kind in ('tb', 'br'):
        d = {b: res['corr_15fps_subset'][b][kind]['mean'] - res['corr_5fps_subset'][b][kind]['mean'] for b in P4}
        verdict[kind] = {'delta_rho_15_minus_5': d, 'five_fps_is_enough': bool(all(abs(x) < 0.05 for x in d.values()))}
    res['frame_rate_verdict'] = verdict
    log(f'frame-rate verdict: {verdict}')
    # ------------------------------------------------------------------ condition decoding
    if not args.skip_decode:
        from joblib import Parallel, delayed
        from sklearn.metrics import f1_score
        tasks = []
        for target in ('strain', 'lights'):
            for feat in ('ethogram', 'meancode'):
                for k in range(NS):
                    for perm in (None, 0):
                        tasks.append((target, feat, k, perm))

        def feats(feat, k):
            return (np.concatenate([SUM5['tb'][k], SUM5['br'][k], SUM5['lat'][k]], 1) if feat == 'ethogram'
                    else MEANC[k]).astype(np.float64)
        log(f'condition decoding: {len(tasks)} cross-validated fits on {args.njobs} workers')
        out = Parallel(n_jobs=args.njobs, verbose=0)(delayed(decode)(feats(f_, k), cond[t], p) for t, f_, k, p in tasks)
        dec = {}
        for target in ('strain', 'lights'):
            y = cond[target]
            maj = np.full_like(y, int(y.mean() >= 0.5))
            dec[target] = {'positive_rate': float(y.mean()),
                           'chance_majority_macro_f1': float(f1_score(y, maj, average='macro')),
                           'chance_auroc': 0.5}
            for feat in ('ethogram', 'meancode'):
                rr = [o for (t, f_, k, p), o in zip(tasks, out) if (t, f_, p) == (target, feat, None)]
                pp = [o for (t, f_, k, p), o in zip(tasks, out) if (t, f_, p) == (target, feat, 0)]
                assert len(rr) == len(pp) == NS
                log(f'  {target} {feat}: ' + ', '.join(f"{r['macro_f1']:.3f}/{r['auroc']:.3f}" for r in rr)
                    + ' | perm ' + ', '.join(f"{r['macro_f1']:.3f}/{r['auroc']:.3f}" for r in pp))
                dec[target][feat] = {
                    'per_seed': rr, 'perm_per_seed': pp,
                    **{f'{q}_{st}': float(np.mean([r[q] for r in L_])) for q in ('macro_f1', 'auroc')
                       for st, L_ in (('mean', rr), ('perm_mean', pp))},
                    **{f'{q}_sd': float(np.std([r[q] for r in rr], ddof=1)) for q in ('macro_f1', 'auroc')}}
        res['decoding'] = dec
    res['wall_s'] = round(time.time() - t0, 1)
    (E4 / 'ethogram.json').write_text(json.dumps(res, indent=1, default=float))
    write_md(res)
    log(f'done in {time.time() - t0:.0f}s')


def write_md(res):
    ms = lambda d: f"{d['mean']:+.3f} ± {d['sd']:.3f}"  # noqa: E731
    L = ['# MABe22 ECI round 3 - E4 ETHOGRAM (pre-registered rule: scripts/benchmark/mabe22/eci_round3.sh)', '',
         'R codes = the 6 confirmed X_A7D2 dictionaries (seeds 100-105, own theta*), full-clip bf16-autocast encoder. '
         'Numbers: cross-fitted Spearman rho (pick on clip half A among neurons firing in >= 10 clips of A, score on '
         'half B; 2 directions x 2 split seeds averaged), mean ± sd (ddof 1) over the 6 SAE seeds.',
         f"Clips: {res['n_clips']} (5 fps all); 15 fps subset: {res['subset15']['n_subset']} clips "
         f"({res['subset15']['n_random']} random rng 0 + {res['subset15']['n_oral_oo_og']} with an oral_oral / "
         f"oral_genital positive, overlap {res['subset15']['n_overlap']}).", '',
         '## Correlation of neuron summaries with annotated summaries', '',
         '| behaviour | TB Spearman rho, 5 fps all | TB rho, 5 fps subset | TB rho, 15 fps subset | '
         'BR Spearman rho, 5 fps all | BR rho, 5 fps subset | BR rho, 15 fps subset |', '|---|---|---|---|---|---|---|']
    for b in BEHAVIOURS:
        cells = [ms(res[t][b][k]) for k in ('tb', 'br') for t in ('corr_5fps_all', 'corr_5fps_subset',
                                                                   'corr_15fps_subset')]
        L.append(f"| {SHORT[b]}{' (primary)' if b in P4 else ''} | " + ' | '.join(cells) + ' |')
    L += ['', 'TB = time budget (fraction of grid frames firing vs fraction of label frames positive); BR = bout rate '
          '(maximal runs per minute).', '', '## Frame rate (pre-registered reading: 5 fps is enough iff '
          '|rho_15 - rho_5| < 0.05 on oral_oral, oral_genital, approaches, huddles; mean over seeds)', '',
          '| summary | ' + ' | '.join(f'{SHORT[b]} rho_15 - rho_5' for b in P4) + ' | verdict |',
          '|---|' + '---|' * (len(P4) + 1)]
    for kind, nm in (('tb', 'time budget'), ('br', 'bout rate')):
        v = res['frame_rate_verdict'][kind]
        L.append(f'| {nm} | ' + ' | '.join(f"{v['delta_rho_15_minus_5'][b]:+.3f}" for b in P4)
                 + f" | {'5 fps is enough' if v['five_fps_is_enough'] else '5 fps is NOT enough'} |")
    L += ['', '5 vs 15 fps Spearman agreement (over the subset clips) of the selected neuron\'s summary, mean over '
          'seeds x splits x directions (pick5 = neuron picked at 5 fps, pick15 = at 15 fps; same = fraction of runs '
          'picking the same neuron at both rates, descriptive):', '',
          '| behaviour | TB agree pick5 | TB agree pick15 | TB same | BR agree pick5 | BR agree pick15 | BR same |',
          '|---|---|---|---|---|---|---|']
    for b in BEHAVIOURS:
        a = res['agree_5_vs_15'][b]
        L.append(f'| {SHORT[b]} | ' + ' | '.join(f"{a[k][q]:.3f}" for k in ('tb', 'br')
                                                for q in ('pick5', 'pick15', 'same_neuron')) + ' |')
    if 'decoding' in res:
        L += ['', '## Condition decoding (one row per clip, outer stratified 5-fold, out-of-fold predictions pooled; '
              'mean ± sd over the 6 SAE seeds)', '',
              '| target | features | macro F1 | AUROC | macro F1, label permutation | AUROC, label permutation | '
              'chance macro F1 (majority) | chance AUROC |', '|---|---|---|---|---|---|---|---|']
        for t, d in res['decoding'].items():
            for feat, nm in (('ethogram', 'ethogram (3 x 4096)'), ('meancode', 'mean frame code (4096)')):
                x = d[feat]
                L.append(f"| {t} (pos rate {d['positive_rate']:.3f}) | {nm} | {x['macro_f1_mean']:.3f} ± "
                         f"{x['macro_f1_sd']:.3f} | {x['auroc_mean']:.3f} ± {x['auroc_sd']:.3f} | "
                         f"{x['macro_f1_perm_mean']:.3f} | {x['auroc_perm_mean']:.3f} | "
                         f"{d['chance_majority_macro_f1']:.3f} | 0.5 |")
    (E4 / 'ethogram.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
