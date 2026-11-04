"""
ECI v2.4 metric: precision@top3/video (replaces precision@100 in the headline alignment tables).

Definition (per model seed, rule, behaviour, held-out set): for each ELIGIBLE held-out video (annotated + valid for the
behaviour, >= 1 positive frame in the scored frame set) take the 3 frames with the highest frame-max code of the
selected neuron (RAW code, before the HMM; ties -> earliest frame); precision = labelled-positive frames among them /
(3 x eligible videos). Chance = mean over eligible videos of the per-video positive-frame rate. Secondary variant
'ro': rank by the readout precision@100 actually uses (HMM P(on) for v2 models, ties by raw code, then earliest frame;
for v1 = raw, identical). Picks / selection / frame sets are those of the existing v2.3 tags (unchanged).
Bootstrap: exactly the existing level CIs (mice: v22_eval.cmd_report pool weights rng 20261104 + seed resampling
rng 20261105, the SAME index draws as precision@100 of the same arm and behaviour; ants: v23_ants.cmd_report rng
20261106, same draws as precision@100). precision@100 is recomputed with that code path (identity check).
Steps
  mice-prep --key K   (V22_FRAMES=ext: stored ext tag + caches; V22_FRAMES=full: v1 seeds on the 648,000 atlas frames,
                       selection recomputed as v22_eval.cmd_tag) -> OUT/work/mice_<frames>_<K>.pkl
  ants-prep --key K   -> OUT/work/ants_<K>.pkl
  mice-report         (V22_FRAMES=ext or full) -> OUT/work/mice_<frames>_report.json
  ants-report         -> OUT/work/ants_report.json
  summary             -> OUT/summary.md, OUT/top3.json
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/eci_v2_4/metric_top3'
WORK = OUT / 'work'
NB = 1000
K3 = 3
RULES = ('auc', 'bal', 'hyb')
NAME = {'nose_nose': 'nose-nose (nose_nose)', 'nt': 'nose-tail (nt)', 'groom_any': 'grooming-to-focal (any)'}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def cols_chunked(path, js, n_rows=None):
    """columns js of a (n, W) .npy, read sequentially in row chunks (never the full float32 matrix)."""
    mm = np.load(path, mmap_mode='r')
    out = np.empty((mm.shape[0], len(js)), np.float32)
    for c0 in range(0, mm.shape[0], 40000):
        out[c0:c0 + 40000] = np.asarray(mm[c0:c0 + 40000])[:, js]
    return out


def top3_videos(obs, y, valid, raw, ro, unit):
    """per held-out video -> dict of arrays: vid, unit, n (valid frames), pos, hits_raw, hits_ro, top frames (raw)."""
    codes, vids = pd.factorize(obs)
    st = np.r_[np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]), len(codes)]
    assert len(st) - 1 == len(vids), 'held-out rows not contiguous per video'
    out = {k: [] for k in ('vid', 'unit', 'n', 'pos', 'hits_raw', 'hits_ro', 'top_raw', 'top_ro')}
    for a, b in zip(st[:-1], st[1:]):
        m = valid[a:b]
        ii = np.flatnonzero(m)
        yy, r, x = y[a:b][ii], raw[a:b][ii].astype(np.float64), ro[a:b][ii].astype(np.float64)
        pos = np.arange(len(ii))  # earliest frame first (rows sorted by frame within a video)
        o_raw = np.lexsort((pos, -r))[:K3]
        o_ro = np.lexsort((pos, -r, -x))[:K3]
        out['vid'].append(obs[a]); out['unit'].append(unit[a]); out['n'].append(len(ii)); out['pos'].append(int(yy.sum()))
        out['hits_raw'].append(int(yy[o_raw].sum())); out['hits_ro'].append(int(yy[o_ro].sum()))
        out['top_raw'].append([(int(a + ii[o]), float(r[o]), float(x[o]), bool(yy[o])) for o in o_raw])
        out['top_ro'].append([(int(a + ii[o]), float(r[o]), float(x[o]), bool(yy[o])) for o in o_ro])
    return {k: (np.array(v) if not k.startswith('top') else v) for k, v in out.items()}


def my_top100(y, raw, ro):
    pos = np.arange(len(y))
    o = np.lexsort((pos, -raw.astype(np.float64), -ro.astype(np.float64)))[:100]
    return int(y[o].sum())


# ---------------------------------------------------------------------------------------------- mice
def cmd_mice_prep(a):
    import v22_core as C
    import v22_eval as E
    t0 = time.time()
    key, fr = a.key, C.FRAMES
    F, SP = C.mice_table()
    ev = F.nonsg.values
    sel = F.sg.values
    rows = np.flatnonzero(ev)
    res = {'key': key, 'frames': fr, 'n_frames': len(F), 'n_ev_frames': int(ev.sum()), 'cells': {}}
    if fr == 'ext':
        tag = pickle.load(open(C.OUT22 / 'tags' / f'{key}.pkl', 'rb'))
        picks = {r: tag['sets'][f'sg2non|{r}'][0]['picks'] for r in RULES}
        js = sorted({j for p in picks.values() for j in p.values()})
        pr, pp = C.cache_paths(key)
        R = cols_chunked(pr, js)
        X = R if key.startswith('v1_') else cols_chunked(pp, js)
        col = {j: i for i, j in enumerate(js)}
        res['tag_top100_label_sum'] = {r: {b: int(tag['sets'][f'sg2non|{r}'][0][b]['top'][1][:100].sum()) for b in C.BEH} for r in RULES}
    else:  # 'full': v1 seeds on the 648,000 atlas frames; selection exactly as v22_eval.cmd_tag (sg2non direction)
        import v22_full as VF
        assert key.startswith('v1_')
        pr, _ = C.cache_paths(key)
        Xf = np.load(pr).astype(np.float32) if key == 'v1_s0' else VF.assemble(key).astype(np.float32)
        assert len(Xf) == len(F) == 648000
        el = (Xf > 0).mean(0) >= C.ELIG
        st = C.neuron_stats_multi(Xf, F, [np.flatnonzero(sel)], None)[0]
        st['fire'] = np.where(el, 1.0, 0.0)
        picks = {r: C.select(st, r) for r in RULES}
        log(f'{key}: selection {picks} ({time.time() - t0:.0f}s)')
        units_ev = sorted(set(F.unit.values[ev]))
        rng = np.random.default_rng(0)
        pcol = np.zeros(len(F))
        tag = {'key': key, 'sets': {}}
        for r in RULES:
            d = {'picks': picks[r], 'units_ev': units_ev}
            for b in C.BEH:
                d[b] = E.held(F, Xf, Xf, None, picks[r][b], pcol, sel, ev, b, units_ev, rng)
            tag['sets'][f'sg2non|{r}'] = [d]
        js = sorted({j for p in picks.values() for j in p.values()})
        R = Xf[:, js].copy()
        X = R
        col = {j: i for i, j in enumerate(js)}
        del Xf
        res['tag'] = tag
        old = C.OUT22 / 'tags' / f'{key}.pkl'
        if old.exists():  # identity of the recomputed selection vs the stored v2.2 full tag
            T0 = pickle.load(open(old, 'rb'))
            res['picks_vs_stored_tag'] = {r: [T0['sets'][f'sg2non|{r}'][0]['picks'], picks[r]] for r in RULES}
    res['picks'] = picks
    obs, unit = F.obs.values[rows], F.unit.values[rows]
    for r in RULES:
        for b in C.BEH:
            j = picks[r][b]
            y = F[f'y_{b}'].values[rows]
            valid = F[f'v_{b}'].values[rows]
            raw, ro = R[rows, col[j]], X[rows, col[j]]
            c = top3_videos(obs, y, valid, raw, ro, unit)
            c['neuron'] = j
            c['my_top100_label_sum'] = my_top100(y, raw, ro)
            c['invalid_ev_frames'] = int((~valid).sum())
            c['top_raw'] = [[(t[0], int(F.k.values[rows[t[0]]]), *t[1:]) for t in v] for v in c['top_raw']]
            c['top_ro'] = [[(t[0], int(F.k.values[rows[t[0]]]), *t[1:]) for t in v] for v in c['top_ro']]
            res['cells'][f'{r}|{b}'] = c
    WORK.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(WORK / f'mice_{fr}_{key}.pkl', 'wb'))
    log(f'mice-prep {fr} {key} done ({time.time() - t0:.0f}s)')


def top3_rep(c, w_of_vid):
    """(NB+1,) precision@top3/video, chance, for one cell under replicate video weights; eligible = pos > 0."""
    e = c['pos'] > 0
    w = w_of_vid(c['vid'][e], c['unit'][e])  # (NB+1, n_elig)
    den = w.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'top3': (w @ c['hits_raw'][e]) / (K3 * den), 'top3ro': (w @ c['hits_ro'][e]) / (K3 * den),
                'chance': (w @ (c['pos'][e] / c['n'][e])) / den}


def ci_of(v, ix):
    bb = np.nanmean(v[:, 1:][ix, np.arange(NB)[:, None]], 1)
    return {'mean': float(np.nanmean(v[:, 0])), 'ci': np.nanpercentile(bb, [2.5, 97.5]).tolist(), 'seeds': v[:, 0].tolist()}


def cmd_mice_report(a):
    import v22_core as C
    import v22_eval as E
    fr = C.FRAMES
    F, _ = C.mice_table()
    units = sorted(set(F.unit.values))
    uidx = {u: i for i, u in enumerate(units)}
    non = sorted(set(F.unit.values[F.nonsg.values]))
    rng = np.random.default_rng(20261104)  # = v22_eval.cmd_report: sg2non weights are the first draws
    W = np.ones((NB + 1, len(units)))
    ix_ = [units.index(u) for u in non]
    W[1:, ix_] = np.stack([np.bincount(rng.integers(0, len(non), len(non)), minlength=len(non)) for _ in range(NB)])
    wb = lambda ue: W[:, [uidx[u] for u in ue]]  # noqa: E731
    wv = lambda vids, us: W[:, [uidx[u] for u in us]]  # noqa: E731
    if fr == 'ext':
        arms = json.loads((C.OUT22 / 'arms.json').read_text())
        arms = {k: v for k, v in arms.items()}
        load_tag = lambda k: pickle.load(open(C.OUT22 / 'tags' / f'{k}.pkl', 'rb'))  # noqa: E731
        ref = json.loads((C.OUT22 / 'report.json').read_text())['sets']['sg2non']['level_ci']
    else:
        arms = {f'v1s0-10|{r}': {'models': [f'v1_s{s}' for s in range(11)], 'rule': r} for r in RULES}
        load_tag = lambda k: pickle.load(open(WORK / f'mice_full_{k}.pkl', 'rb'))['tag']  # noqa: E731
        ref = None
    prep = {}
    per, cache = {}, {}
    for arm, spec in arms.items():
        M = []
        for k in spec['models']:
            if k not in cache:
                cache[k] = load_tag(k)
                prep[k] = pickle.load(open(WORK / f'mice_{fr}_{k}.pkl', 'rb'))
            M.append(E.model_metrics(cache[k], f'sg2non|{spec["rule"]}', wb))
        per[arm] = {mk: np.stack([m[mk] for m in M]) for mk in M[0]}
    rng2 = np.random.default_rng(20261105)
    out = {'frames': fr, 'arms': arms, 'levels': {}, 'identity_p100': {}, 'counts': {}, 'top100_check': {}}
    for arm, d in per.items():
        spec = arms[arm]
        lv = {}
        for mk, v in d.items():
            ix = rng2.integers(0, len(v), (NB, len(v)))  # identical draw sequence to v22_eval.cmd_report
            b, m_ = mk
            if m_ in ('auc', 'p100'):
                lv[f'{b}|{m_}'] = ci_of(v, ix)
            if m_ == 'p100':  # the new metrics reuse precision@100's resampling draws
                new = [top3_rep(prep[k]['cells'][f'{spec["rule"]}|{b}'], wv) for k in spec['models']]
                for nm in ('top3', 'top3ro', 'chance'):
                    lv[f'{b}|{nm}'] = ci_of(np.stack([x[nm] for x in new]), ix)
                c0 = prep[spec['models'][0]]['cells'][f'{spec["rule"]}|{b}']
                e = c0['pos'] > 0
                out['counts'][b] = {'held_out_videos': int(len(e)), 'eligible': int(e.sum()), 'excluded_no_positive': int((~e).sum()),
                                    'frames_scored': int(K3 * e.sum()), 'invalid_frames': c0['invalid_ev_frames']}
                if ref is not None:
                    r_ = ref[arm][f'{b}|p100']
                    out['identity_p100'][f'{arm}|{b}'] = {'mine': [lv[f'{b}|p100']['mean']] + lv[f'{b}|p100']['ci'],
                                                          'report': [r_['mean']] + r_['ci']}
                if fr == 'ext':
                    out['top100_check'][f'{arm}|{b}'] = [(prep[k]['tag_top100_label_sum'][spec['rule']][b],
                                                          prep[k]['cells'][f'{spec["rule"]}|{b}']['my_top100_label_sum']) for k in spec['models']]
        out['levels'][arm] = lv
        out.setdefault('picks', {})[arm] = {k: prep[k]['picks'][spec['rule']] for k in spec['models']}
    if fr == 'full':
        out['picks_vs_stored_tag'] = prep['v1_s0'].get('picks_vs_stored_tag')
        rf = json.loads((C.OUT22 / 'report.json').read_text())['sets']['sg2non']
        out['stored_full_v1s0'] = {f'{r}|{b}|{m}': rf['level_ci'][f'v1s0|{r}'][f'{b}|{m}'] for r in RULES for b in C.BEH for m in ('auc', 'p100')} \
            if 'level_ci' in rf else None
    # hand sanity check: first model of the first arm, rule auc, nose_nose, 3 eligible videos
    k0 = list(arms.values())[0]['models'][0] if fr == 'full' else 'V_s1001'
    c = prep[k0]['cells']['auc|nose_nose']
    e = np.flatnonzero(c['pos'] > 0)[:3]
    sc = {'model': k0, 'rule': 'auc', 'behaviour': 'nose_nose', 'neuron': int(c['neuron']), 'videos': []}
    hits = 0
    for i in e:
        sc['videos'].append({'video': str(c['vid'][i]), 'positive_frames': int(c['pos'][i]), 'valid_frames': int(c['n'][i]),
                             'top3 (frame k, raw code, readout, label)': [(t[1], round(t[2], 4), round(t[3], 4), t[4]) for t in c['top_raw'][i]]})
        hits += sum(t[4] for t in c['top_raw'][i])
    sc['hits'] = int(hits)
    sc['precision_on_these_3_videos'] = hits / (K3 * len(e))
    sc['stored_hits_raw'] = c['hits_raw'][e].tolist()
    out['sanity'] = sc
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f'mice_{fr}_report.json').write_text(json.dumps(out, indent=1, default=float))
    log(json.dumps({'identity_p100': out['identity_p100'], 'counts': out['counts'], 'sanity': sc}, default=float)[:3000])


# ---------------------------------------------------------------------------------------------- ants
def cmd_ants_prep(a):
    import v23_ants as AN
    t0 = time.time()
    key = a.key
    F = AN.table()
    tag = pickle.load(open(AN.OUTA / 'tags' / f'{key}.pkl', 'rb'))
    X, R, _ = AN.readout(key, F)
    res = {'key': key, 'cells': {}}
    for dn, (s_, e_) in AN.DIRS.items():
        d = tag['sets'][dn][0]
        rows = np.flatnonzero(F.half.values == e_)
        for b in ('groom_any',):
            j = d['picks'][b]
            y = F[f'y_{b}'].values[rows]
            valid = F[f'v_{b}'].values[rows]
            obs = F.obs.values[rows]
            c = top3_videos(obs, y, valid, R[rows, j], X[rows, j], obs)
            c['neuron'] = j
            c['my_top100_label_sum'] = my_top100(y, R[rows, j], X[rows, j])
            c['tag_top100_label_sum'] = int(d[b]['top'][1][:100].sum())
            c['invalid_ev_frames'] = int((~valid).sum())
            c['top_raw'] = [[(t[0], int(F.k.values[rows[t[0]]]), *t[1:]) for t in v] for v in c['top_raw']]
            res['cells'][f'{dn}|{b}'] = c
    WORK.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(WORK / f'ants_{key}.pkl', 'wb'))
    log(f'ants-prep {key} done ({time.time() - t0:.0f}s)')


def cmd_ants_report(a):
    """v23_ants.cmd_report's exact rng sequence (weights, level draws, contrast draws) + the new metrics."""
    import v22_eval as E
    import v23_ants as AN
    F = AN.table()
    vids = F.groupby('obs', sort=True).agg(stratum=('stratum', 'first'), half=('half', 'first'))
    rng = np.random.default_rng(20261106)
    tags = {k: pickle.load(open(AN.OUTA / 'tags' / f'{k}.pkl', 'rb')) for ks in AN.ARMS.values() for k in ks
            if (AN.OUTA / 'tags' / f'{k}.pkl').exists()}
    arms = {arm: ks for arm, ks in AN.ARMS.items() if all(k in tags for k in ks)}
    prep = {k: pickle.load(open(WORK / f'ants_{k}.pkl', 'rb')) for ks in arms.values() for k in ks}
    MK = ('auc', 'p100', 'F1', 'r_soft', 'bias_raw_soft_frac', 'mae_raw_soft_frac', 'bias_cal_soft_frac', 'mae_cal_soft_frac', 'mae_const_frac')
    ref = json.loads((AN.OUTA / 'report.json').read_text())['dirs']
    out = {'arms': arms, 'dirs': {}}
    for dn, (s_, e_) in AN.DIRS.items():
        ev = vids.index[vids.half == e_].tolist()
        strata = vids.loc[ev, 'stratum'].values
        W = np.ones((NB + 1, len(ev)))
        for s in np.unique(strata):
            ix = np.flatnonzero(strata == s)
            W[1:, ix] = np.stack([np.bincount(rng.integers(0, len(ix), len(ix)), minlength=len(ix)) for _ in range(NB)])
        vix = {v: i for i, v in enumerate(ev)}
        wv = lambda vs, us: W[:, [vix[v] for v in vs]]  # noqa: E731
        per = {}
        for arm, ks in arms.items():
            M = []
            for k in ks:
                d = tags[k]['sets'][dn][0]
                assert d['units_ev'] == ev
                M.append({(b, m_): v for b in AN.BEH for m_, v in E.metrics(d[b], W).items()})
            per[arm] = {mk: np.stack([m[mk] for m in M]) for mk in M[0]}
        R = {'levels': {}, 'identity_p100': {}, 'top100_check': {}}
        for arm, d in per.items():
            for (b, m_), v in d.items():
                if m_ not in MK:
                    continue
                ix = rng.integers(0, len(v), (NB, len(v)))
                if b != 'groom_any' or m_ not in ('auc', 'p100'):
                    continue
                R['levels'].setdefault(arm, {})[f'{b}|{m_}'] = ci_of(v, ix)
                if m_ == 'p100':
                    new = [top3_rep(prep[k]['cells'][f'{dn}|{b}'], wv) for k in arms[arm]]
                    for nm in ('top3', 'top3ro', 'chance'):
                        R['levels'][arm][f'{b}|{nm}'] = ci_of(np.stack([x[nm] for x in new]), ix)
                    r_ = ref[dn]['levels'][arm][f'{b}|p100']
                    R['identity_p100'][f'{arm}|{b}'] = {'mine': [R['levels'][arm][f'{b}|p100']['mean']] + R['levels'][arm][f'{b}|p100']['ci'],
                                                        'report': [r_['mean']] + r_['ci']}
                    R['top100_check'][f'{arm}|{b}'] = [(prep[k]['cells'][f'{dn}|{b}']['tag_top100_label_sum'],
                                                        prep[k]['cells'][f'{dn}|{b}']['my_top100_label_sum']) for k in arms[arm]]
                    c0 = prep[arms[arm][0]]['cells'][f'{dn}|{b}']
                    e = c0['pos'] > 0
                    R['counts'] = {'held_out_videos': int(len(e)), 'eligible': int(e.sum()), 'excluded_no_positive': int((~e).sum()),
                                   'frames_scored': int(K3 * e.sum()), 'invalid_frames': c0['invalid_ev_frames']}
            R.setdefault('picks', {})[arm] = {k: prep[k]['cells'][f'{dn}|groom_any']['neuron'] for k in arms[arm]}
        for x, y in AN.CONTRASTS:  # consume the contrast draws so the next direction's rng state matches
            if x not in per or y not in per:
                continue
            for (b, m_) in per[x]:
                if m_ not in MK:
                    continue
                rng.integers(0, len(per[x][(b, m_)]), (NB, len(per[x][(b, m_)])))
                rng.integers(0, len(per[y][(b, m_)]), (NB, len(per[y][(b, m_)])))
        out['dirs'][dn] = R
    c = prep['V_s1001']['cells']['A2B|groom_any']
    e = np.flatnonzero(c['pos'] > 0)[:3]
    sc = {'model': 'V_s1001', 'dir': 'A2B', 'behaviour': 'groom_any', 'neuron': int(c['neuron']), 'videos': []}
    hits = 0
    for i in e:
        sc['videos'].append({'video': str(c['vid'][i]), 'positive_frames': int(c['pos'][i]), 'valid_frames': int(c['n'][i]),
                             'top3 (frame k, raw code, readout, label)': [(t[1], round(t[2], 4), round(t[3], 4), t[4]) for t in c['top_raw'][i]]})
        hits += sum(t[4] for t in c['top_raw'][i])
    sc['hits'] = int(hits)
    sc['precision_on_these_3_videos'] = hits / (K3 * len(e))
    sc['stored_hits_raw'] = c['hits_raw'][e].tolist()
    out['sanity'] = sc
    (WORK / 'ants_report.json').write_text(json.dumps(out, indent=1, default=float))
    log(json.dumps({dn: {'identity': R_['identity_p100'], 'counts': R_.get('counts')} for dn, R_ in out['dirs'].items()}, default=float)[:3000])


# ---------------------------------------------------------------------------------------------- summary
def _f(x, d=3):
    return f"{x['mean']:.{d}f} [{x['ci'][0]:.{d}f}, {x['ci'][1]:.{d}f}]"


def _seeds(x):
    return ', '.join(f'{v:.3f}' for v in x['seeds'])


def cmd_summary(a):
    me = json.loads((WORK / 'mice_ext_report.json').read_text())
    mf = json.loads((WORK / 'mice_full_report.json').read_text())
    an = json.loads((WORK / 'ants_report.json').read_text())
    HDR = ('| model \\| rule | precision@top3/video [95% CI] | chance level (random 3 frames / video) [95% CI] | '
           'precision@top3/video, ranked by the precision@100 readout [95% CI] | AUROC [95% CI] | precision@100 [95% CI] | '
           'precision@top3/video per seed |')
    md = ['# precision@top3/video: ECI v2 alignment metric (replaces precision@100)', '',
          '**Definition.** For each eligible held-out video: the 3 frames with the highest frame-max code of the selected neuron '
          '(raw code, BEFORE the HMM; ties -> earliest frame). precision@top3/video = labelled-positive frames among them / '
          '(3 x eligible videos). Eligible = held-out (measurement) video, annotated and valid for the behaviour, with >= 1 '
          'positive frame in the scored frame set; videos with no positive frame are excluded (counts below). Chance level = '
          'mean over eligible videos of the per-video positive-frame rate (expected precision of 3 random frames per video).', '',
          'Neuron selection, selection / measurement sets and frame sets are UNCHANGED (stored v2.3 tags). Numbers: mean over '
          'seeds [95% CI, bootstrap over held-out pools (mice) / held-out videos within series x treatment (ants) and seeds, '
          '1,000 replicates, the same replicate weights and seed draws as precision@100 of the same arm and behaviour].', '',
          'Note: the existing precision@100 ranks v2 models (V, Vall) by the HMM P(on) readout (ties by raw code), v1 by its raw '
          'code. The headline precision@top3/video uses the raw code for every model as specified; the column "ranked by the '
          'precision@100 readout" gives the P(on)-ranked variant (identical for v1).', '',
          'Models: v1s0 = deployed v1 seed 0; V = v2 current recipe, seeds 1001-1006; Vall = v2.3 (V trained on all tokens), '
          'seeds 2001-2003; v1 = v1 seeds 0-10. Rules: auc = best AUROC on the selection set (primary); bal = best balanced AUROC; '
          'hyb = nose-nose by auc, nose-tail by bal (post hoc).', '']
    js = {'mice_ext': me, 'mice_atlas648k_v1_seeds': mf, 'ants': an}

    def mice_block(R, title, note):
        out = [f'## {title}', '', note, '']
        for b in ('nose_nose', 'nt'):
            c = R['counts'][b]
            out += [f'### {NAME[b]}', '', f"Eligible held-out videos: {c['eligible']} of {c['held_out_videos']} "
                    f"({c['excluded_no_positive']} excluded, no positive frame); frames scored per model seed: {c['frames_scored']} "
                    f"(= 3 x {c['eligible']}); invalid (unlabelled) held-out frames: {c['invalid_frames']}.", '', HDR, '|' + '---|' * 7]
            for arm, lv in R['levels'].items():
                out.append(f"| {arm} | {_f(lv[b + '|top3'])} | {_f(lv[b + '|chance'], 4)} | {_f(lv[b + '|top3ro'])} | "
                           f"{_f(lv[b + '|auc'])} | {_f(lv[b + '|p100'])} | {_seeds(lv[b + '|top3'])} |")
            out.append('')
        return out

    md += mice_block(me, 'Mice, ext frames (864,000 = 15-min atlas window + first 15 min of the habituation videos)',
                     'Select on the 11 SG pools, measure on the 12 non-SG pools (432,000 held-out frames).')
    md += mice_block(mf, 'Mice, v1 seeds 0-10 on the 648,000 atlas-window frames only',
                     'v1 seeds 1-10 have codes on the 648,000 atlas-window frames only (no ext codes), so this arm is scored on '
                     'the atlas window, NOT the ext frame set; not directly comparable with the table above. Selection '
                     'recomputed with the v2.2 rule code (v1 seed 0 picks checked against the stored v2.2 full-window tag).')
    for dn, title in (('A2B', 'select on half A, measure on half B (PRIMARY)'), ('B2A', 'select on half B, measure on half A')):
        R = an['dirs'][dn]
        c = R['counts']
        md += [f'## Ants, v3 + v4 (147 videos, first 10 min at 1 fps), {title}', '', f'### {NAME["groom_any"]}', '',
               f"Eligible held-out videos: {c['eligible']} of {c['held_out_videos']} ({c['excluded_no_positive']} excluded, no "
               f"positive frame); frames scored per model seed: {c['frames_scored']}. Rule auc.", '', HDR, '|' + '---|' * 7]
        for arm, lv in R['levels'].items():
            b = 'groom_any'
            md.append(f"| {arm} \\| auc | {_f(lv[b + '|top3'])} | {_f(lv[b + '|chance'], 4)} | {_f(lv[b + '|top3ro'])} | "
                      f"{_f(lv[b + '|auc'])} | {_f(lv[b + '|p100'])} | {_seeds(lv[b + '|top3'])} |")
        md.append('')
    # checks
    def idmax(d):
        return max(abs(x - y) for v in d.values() for x, y in zip(v['mine'], v['report']))
    md += ['## Checks', '',
           f"1. Identity: precision@100 (mean and both CI ends) recomputed through this script's bootstrap vs the stored reports: "
           f"mice ext max abs diff {idmax(me['identity_p100']):.2e} over {len(me['identity_p100'])} arm x behaviour cells; ants "
           f"max abs diff {max(idmax(an['dirs'][d]['identity_p100']) for d in an['dirs']):.2e} over "
           f"{sum(len(an['dirs'][d]['identity_p100']) for d in an['dirs'])} cells.",
           '2. Hand check (mice ext, V seed 1001, rule auc, nose-nose; top 3 frames of the first 3 eligible held-out videos by raw code):', '',
           '```', json.dumps(me['sanity'], indent=1), '```', '', 'Ants hand check (V seed 1001, A -> B, grooming-to-focal (any)):', '',
           '```', json.dumps(an['sanity'], indent=1), '```', '',
           '3. Top-100 frames: labelled positives among the stored tag top 100 vs this script\'s own ranking (same readout, '
           'ties earliest frame instead of random): see top3.json (top100_check).', '']
    if mf.get('picks_vs_stored_tag'):
        md += [f"v1 seed 0 on the 648,000 atlas frames, picks [stored v2.2 full tag, recomputed]: {json.dumps(mf['picks_vs_stored_tag'])}", '']
    (OUT / 'summary.md').write_text('\n'.join(md) + '\n')
    (OUT / 'top3.json').write_text(json.dumps(js, indent=1, default=float))
    log('summary written')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('mice-prep', 'mice-report', 'ants-prep', 'ants-report', 'summary'))
    p.add_argument('--key', default='')
    a = p.parse_args()
    {'mice-prep': cmd_mice_prep, 'mice-report': cmd_mice_report, 'ants-prep': cmd_ants_prep, 'ants-report': cmd_ants_report,
     'summary': cmd_summary}[a.step](a)
