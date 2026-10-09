"""
T4 per-video read-outs, the video-level check and the pre-registered T2 / T4 decisions (CPU, analysis only).
Inputs: results/vision/eci_t2t4/<d>/{eval,codes_best,levels}/ (scripts/eci/tight_mask_sae.py evaluate).

Per-video read-outs of one neuron, inside the NES analysis window (src/eci/domain.py window_start: mice habituation =
its last 15 min, odour / post whole; ants whole video), on the 1 fps eval frames ordered in time:
    time budget   fraction of window frames with presence > 0
    bout rate     maximal runs of consecutive frames with presence > 0 (1 fps, no gap merging) per window minute
    latency       seconds from the window start to the first frame with presence > 0 (window length if none)
    mean presence, mean extent
(no mean bout length: dropped by the user.)

Video-level check (mice, scripts/eci/diag_video_level.py measures; the neuron is chosen on the OTHER cross-fit half by
frame-level cross-fitted AUROC of the read-out's own frame values: presence pick for presence-mean / time budget /
bout rate / latency, extent pick for extent-mean):
    video r       Pearson over a half's 72 held-out videos of the read-out vs the annotated rate (5 fps, window) and
                  vs annotated bouts/min; mean over the two halves
    pooled d-r    Pearson of the per-pool switch change of the read-out vs that of the annotated rate (4 switches x 12
                  pools per half), mean over halves
    effects       per switch, dz of the change over all 24 held-out pools (each half's values divided by their sd over
                  that half's videos), p, and sign agreement with the annotated rate change
    fg control    the same after replacing the read-out by its residual on the per-video mean foreground-patch count
                  of the arm's own mask (OLS within the half)
Ants: video r only (vs the per-video annotated rate and bouts/min at 5 fps, whole video), held-out halves = the
sae_levers video halves, raw and fg-controlled.

Pre-registered decisions (stated by the user before any result)
    T2 KEPT iff in BOTH domains: the seed-mean cross-fitted best-neuron AUROC (presence) of T2 is not below the
         baseline's by more than 0.02 on any label, AND (some label improves: d AUROC >= +0.03 or AP (best-AP pick)
         ratio >= 1.3, OR plain-background neurons drop by >= 50%).
    T4 extent KEPT iff in each domain at least one label has extent beating presence (cf AUROC >= +0.03 or held-out
         video r with the annotated rate >= +0.1), OR the extent level map is the only one that separates pair from
         self cleanly: >= 60% of the contact-label AUROC picks (mice nose_nose / nose_tail, ants groom_*) land in
         'pair' and (ants) >= 60% of the on-lid picks land in 'self', and the presence-only map does not. Evaluated
         on the arm the T2 decision keeps (both arms reported).
Output: OUT/summary.json, OUT/tables.md
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from diag_video_level import SWITCHES, bouts, corr, ttest  # noqa: E402
from src.eci import contrasts as C  # noqa: E402
from src.eci.domain import AntsDomain, MiceDomain  # noqa: E402

OUT = REPO / 'results/vision/eci_t2t4'
ARMS = ('base', 't2')
SEEDS = (0, 1, 2)
LABELS = {'mice': ['nose_nose', 'nose_tail'], 'ants': ['groom_any', 'groom_yellow', 'groom_blue', 'onlid_yellow']}
CONTACT = {'mice': ['nose_nose', 'nose_tail'], 'ants': ['groom_any', 'groom_yellow', 'groom_blue']}
SELF_LABELS = {'mice': [], 'ants': ['onlid_yellow']}
METRICS = ('cf_auroc', 'cf_ap', 'cf_ap_at_auroc', 'cf_top1', 'cf_size_ctrl_auroc')
READOUTS = {'presence_mean': 'presence', 'extent_mean': 'extent', 'time_budget': 'presence', 'bout_rate': 'presence',
            'latency': 'presence'}
FPS = 5.0
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def ms_(v):
    v = np.asarray(v, float)
    return {'mean': float(np.nanmean(v)), 'sd': float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
            'per_seed': [float(x) for x in v]}


def resid(x, f):
    X = np.c_[np.ones(len(f)), f]
    b, *_ = np.linalg.lstsq(X, x, rcond=None)
    return x - X @ b


# ---------------------------------------------------------------------------------------------- truth / design
def mice_truth(stage):
    ann_src = MiceDomain.ann_path
    dst = stage / ann_src.name
    if not dst.exists():
        shutil.copyfile(ann_src, dst)
    dom = MiceDomain()
    design = C.load_design(dst, MiceDomain.experiment_csv)
    design['wstart'] = dom.window_start(design['n_frames'].values, design['stage'].values)
    ann = pd.read_csv(dst, usecols=['Y_nn', 'Y_np', 'Y_nt'])
    Y = {'nose_nose': (ann['Y_nn'].fillna(0).values > 0) | (ann['Y_np'].fillna(0).values > 0),
         'nose_tail': ann['Y_nt'].fillna(0).values > 0}
    ok = ann[['Y_nn', 'Y_np', 'Y_nt']].notna().all(axis=1).values
    gt = design[['observation_id', 'pool', 'stage', 'wstart', 'n_frames']].copy()
    vid = np.repeat(np.arange(len(design)), design['n_frames'].values)
    gt['annotated'] = np.bincount(vid, weights=ok, minlength=len(design)) == design['n_frames'].values
    for b, y in Y.items():
        r, bo = np.full(len(gt), np.nan), np.full(len(gt), np.nan)
        for i in np.flatnonzero(gt.annotated.values):
            s, e = design.row_start.iat[i] + design.wstart.iat[i], design.row_end.iat[i]
            r[i] = y[s:e].mean()
            bo[i] = bouts(y[s:e]) / ((e - s) / FPS / 60)
        gt[f'{b}_rate'], gt[f'{b}_bpm'] = r, bo
    return gt.set_index('observation_id')


def ants_truth():
    a = pd.read_csv(AntsDomain.ann_path, usecols=['observation_id', 'Y_Y2F', 'Y_B2F', 'Y_YOL'])
    y, b = a.Y_Y2F.fillna(0).values > 0, a.Y_B2F.fillna(0).values > 0
    lab = {'groom_any': y | b, 'groom_yellow': y, 'groom_blue': b}
    rows = []
    for o, g in pd.Series(np.arange(len(a))).groupby(a.observation_id.values):
        ix = g.values
        r = {'observation_id': o, 'wstart': 0}
        for k, v in lab.items():
            r[f'{k}_rate'] = v[ix].mean()
            r[f'{k}_bpm'] = bouts(v[ix]) / (len(ix) / FPS / 60)
        yo = a.Y_YOL.values[ix]
        if np.isfinite(yo).all():
            r['onlid_yellow_rate'] = (yo > 0).mean()
            r['onlid_yellow_bpm'] = bouts(yo > 0) / (len(ix) / FPS / 60)
        rows.append(r)
    return pd.DataFrame(rows).set_index('observation_id')


# ---------------------------------------------------------------------------------------------- per-video read-outs
def video_readouts(p, e, lens, lab, wstart):
    """p, e, lens (F,) per eval frame (store order); lab: obs, frame_idx -> DataFrame per video (in-window frames)."""
    df = pd.DataFrame({'obs': lab.obs.values, 'fi': lab.frame_idx.values, 'p': p, 'e': e, 'n': lens})
    df = df[df.fi.values >= wstart.reindex(df.obs.values).values].sort_values(['obs', 'fi'])
    out = []
    for o, g in df.groupby('obs', sort=True):
        on = g.p.values > 0
        w0 = wstart[o]
        out.append({'obs': o, 'presence_mean': g.p.mean(), 'extent_mean': g.e.mean(), 'time_budget': on.mean(),
                    'bout_rate': bouts(on) / (len(g) / 60.0),
                    'latency': (g.fi.values[np.argmax(on)] - w0) / FPS if on.any() else (g.fi.max() - w0 + 1) / FPS,
                    'fg_mean': g.n.mean(), 'n_frames': len(g)})
    return pd.DataFrame(out).set_index('obs')


def picks(ev, readout, b):
    """{test_half: neuron} of the AUROC pick selected on the other half."""
    return {d['test_half']: d['auc']['neuron'] for d in ev[readout][b]['dirs']}


def per_half_readouts(d, arm, seed, b, lab, half, wstart, lens):
    ev = json.loads((OUT / d / 'eval' / f'{arm}_s{seed}.json').read_text())
    z = np.load(OUT / d / 'codes_best' / f'{arm}_s{seed}.npz')
    nl = list(z['neurons'])
    res = {}
    for ro, src in (('presence', 'presence'), ('extent', 'extent')):
        for h, j in picks(ev, src, b).items():
            m = half == h
            res[(ro, h)] = video_readouts(z['presence'][m, nl.index(j)].astype(np.float32),
                                          z['extent'][m, nl.index(j)].astype(np.float32), lens[m], lab[m], wstart)
    return res


def video_check_mice(R, gt, b):
    """R[(pick readout, half)] per-video DataFrames -> per read-out dict of measures, raw and fg-controlled."""
    A = gt[gt.annotated]
    out = {}
    for name, src in READOUTS.items():
        for ctrl in (False, True):
            halves, dn_all, dr_all = {}, {sw: [] for sw in SWITCHES}, {sw: [] for sw in SWITCHES}
            for h in (0, 1):
                V = R[(src, h)]
                x = V[name].values.astype(float)
                if ctrl:
                    x = resid(x, V.fg_mean.values)
                s = pd.Series(x, index=V.index)
                vids = V.index
                e = {'video_r_rate': corr(s[vids], A.loc[vids, f'{b}_rate'])['pearson'],
                     'video_r_bpm': corr(s[vids], A.loc[vids, f'{b}_bpm'])['pearson']}
                sub = A.loc[vids]
                pools = sorted(sub.pool.unique())
                ix = {(p, st): o for o, p, st in zip(sub.index, sub.pool, sub.stage)}
                dn = {sw: np.array([s[ix[(p, bb)]] - s[ix[(p, aa)]] for p in pools]) for sw, (aa, bb) in SWITCHES.items()}
                dr = {sw: np.array([A.loc[ix[(p, bb)], f'{b}_rate'] - A.loc[ix[(p, aa)], f'{b}_rate'] for p in pools])
                      for sw, (aa, bb) in SWITCHES.items()}
                cat = lambda dd: np.concatenate([dd[sw] for sw in SWITCHES])  # noqa: E731
                e['delta_r_pooled'] = corr(cat(dn), cat(dr))['pearson']
                sd = np.std(x, ddof=1)
                for sw in SWITCHES:
                    dn_all[sw].append(dn[sw] / sd)
                    dr_all[sw].append(dr[sw])
                halves[h] = e
            eff = {}
            for sw in SWITCHES:
                n_, r_ = ttest(np.concatenate(dn_all[sw])), ttest(np.concatenate(dr_all[sw]))
                eff[sw] = {'dz': n_['dz'], 'p': n_['p'], 'truth_dz': r_['dz'], 'truth_p': r_['p'],
                           'sign_agree': bool(np.sign(n_['mean']) == np.sign(r_['mean']))}
            sig = [sw for sw in SWITCHES if eff[sw]['truth_p'] < 0.05]
            out[f'{name}{"|fgctrl" if ctrl else ""}'] = {
                'video_r_rate': float(np.mean([halves[h]['video_r_rate'] for h in (0, 1)])),
                'video_r_bpm': float(np.mean([halves[h]['video_r_bpm'] for h in (0, 1)])),
                'delta_r_pooled': float(np.mean([halves[h]['delta_r_pooled'] for h in (0, 1)])),
                'effects': eff, 'n_sig_switches': len(sig),
                'n_sig_sign_agree': int(sum(eff[sw]['sign_agree'] for sw in sig)),
                'n_sig_reproduced_p05': int(sum(eff[sw]['sign_agree'] and eff[sw]['p'] < 0.05 for sw in sig))}
    return out


def video_check_ants(R, gt, b):
    out = {}
    for name, src in READOUTS.items():
        for ctrl in (False, True):
            rr, rb = [], []
            for h in (0, 1):
                V = R[(src, h)]
                V = V[V.index.isin(gt.index[gt[f'{b}_rate'].notna()])]
                x = V[name].values.astype(float)
                if ctrl:
                    x = resid(x, V.fg_mean.values)
                rr.append(corr(x, gt.loc[V.index, f'{b}_rate'])['pearson'])
                rb.append(corr(x, gt.loc[V.index, f'{b}_bpm'])['pearson'])
            out[f'{name}{"|fgctrl" if ctrl else ""}'] = {'video_r_rate': float(np.mean(rr)), 'video_r_bpm': float(np.mean(rb))}
    return out


# ---------------------------------------------------------------------------------------------- main
def main():
    global OUT, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--domains', default='mice,ants')
    ap.add_argument('--root', default=None, help='default results/vision/eci_t2t4 (smoke tests: another root)')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = ap.parse_args()
    if args.root:
        OUT = Path(args.root)
    SEEDS = tuple(args.seeds)
    stage = Path(args.stage)
    stage.mkdir(parents=True, exist_ok=True)
    S = {}
    for d in args.domains.split(','):
        lab = pd.read_parquet(OUT / d / 'labels.parquet')
        half = np.load(OUT / d / 'half.npy')
        gt = mice_truth(stage) if d == 'mice' else ants_truth()
        wstart = gt['wstart']
        D = {'frame': {}, 'video': {}, 'levels': {}, 'arm_meta': {}, 'picks_levels': {}}
        for arm in ARMS:
            E = [json.loads((OUT / d / 'eval' / f'{arm}_s{s}.json').read_text()) for s in SEEDS]
            lens = np.load(OUT / d / 'codes_best' / f'{arm}_lens.npy').astype(np.float32)
            D['arm_meta'][arm] = {k: E[0][k] for k in ('kept_per_frame', 'frames_no_token', 'token_off_animal_share',
                                                       'token_no_dark_pixel_share', 'a1')}
            D['arm_meta'][arm].update({q: ms_([e[q] for e in E]) for q in ('fve', 'l0_per_token', 'dead_frac')})
            if arm == 'base' and all('repro_max_abs_diff_cf_auroc' in e for e in E):
                D['arm_meta'][arm]['repro_max_abs_diff_cf_auroc'] = max(e['repro_max_abs_diff_cf_auroc'] for e in E)
            for ro in ('presence', 'extent'):
                for b in LABELS[d]:
                    D['frame'][f'{arm}|{ro}|{b}'] = {q: ms_([e[ro][b][q] for e in E]) for q in METRICS}
                    D['frame'][f'{arm}|{ro}|{b}']['base_rate'] = E[0][ro][b]['base_rate']
            # level map counts (mean over seeds) and where the annotated picks land
            L = [e['levels'] for e in E]
            D['levels'][arm] = {
                'n_eligible': ms_([x['n_eligible'] for x in L]), 'plain_background': ms_([x['plain_background'] for x in L]),
                'place_bound': ms_([x['place_bound'] for x in L]),
                'social': {c: ms_([x['social'][c] for x in L]) for c in L[0]['social']},
                'social_presence': {c: ms_([x['social_presence'][c] for x in L]) for c in L[0]['social_presence']},
                'social_x_place_not_plain': {c: ms_([x['social_x_place_not_plain'][c] for x in L])
                                             for c in L[0]['social_x_place_not_plain']}}
            land = {}
            for s, e in zip(SEEDS, E):
                T = pd.read_csv(OUT / d / 'levels' / f'{arm}_s{s}.csv').set_index('neuron')
                for b in LABELS[d]:
                    for ro in ('presence', 'extent'):
                        for dd in e[ro][b]['dirs']:
                            j = dd['auc']['neuron']
                            r = T.loc[j]
                            land.setdefault(b, []).append(
                                {'seed': s, 'readout': ro, 'select_half': dd['select_half'], 'neuron': int(j),
                                 'test_auc': dd['auc']['test_auc'], 'eligible': bool(r.eligible), 'social': r.social,
                                 'social_presence': r.social_presence, 'place_bound': bool(r.place_bound),
                                 'plain_background': bool(r.plain_background), 'E_med': float(r.E_med),
                                 'NA_med': float(r.NA_med), 'conc': float(r.conc), 'active_off': float(r.active_off)})
            D['picks_levels'][arm] = land
            # video level
            V = {}
            for b in LABELS[d]:
                V[b] = []
                for s in SEEDS:
                    R = per_half_readouts(d, arm, s, b, lab, half, wstart, lens)
                    V[b].append(video_check_mice(R, gt, b) if d == 'mice' else video_check_ants(R, gt, b))
                    if s == 0:
                        pv = pd.concat({f'{ro}|h{h}': R[(ro, h)] for ro in ('presence', 'extent') for h in (0, 1)})
                        pv.to_parquet(OUT / d / f'per_video_{arm}_{b}_s0.parquet')
                agg = {}
                for k in V[b][0]:
                    agg[k] = {q: ms_([v[k][q] for v in V[b]]) for q in V[b][0][k] if q != 'effects'}
                    if 'effects' in V[b][0][k]:
                        agg[k]['effects_per_seed'] = [v[k]['effects'] for v in V[b]]
                D['video'][f'{arm}|{b}'] = agg
            log(f'{d} {arm}: done')
        S[d] = D
    # ------------------------------------------------------------------ decisions
    dec = {'T2': {}, 'T4': {}}
    for d in S:
        F_, Lv = S[d]['frame'], S[d]['levels']
        noloss = {b: F_[f't2|presence|{b}']['cf_auroc']['mean'] >= F_[f'base|presence|{b}']['cf_auroc']['mean'] - 0.02
                  for b in LABELS[d]}
        gain = {b: {'d_auroc': F_[f't2|presence|{b}']['cf_auroc']['mean'] - F_[f'base|presence|{b}']['cf_auroc']['mean'],
                    'ap_ratio': F_[f't2|presence|{b}']['cf_ap']['mean'] / F_[f'base|presence|{b}']['cf_ap']['mean']}
                for b in LABELS[d]}
        any_gain = any(g['d_auroc'] >= 0.03 or g['ap_ratio'] >= 1.3 for g in gain.values())
        pb, pt = Lv['base']['plain_background']['mean'], Lv['t2']['plain_background']['mean']
        drop = 1 - pt / pb if pb > 0 else float('nan')
        dec['T2'][d] = {'no_loss_per_label': noloss, 'gain_per_label': gain, 'any_gain': any_gain,
                        'plain_background_base': pb, 'plain_background_t2': pt, 'plain_drop': drop,
                        'plain_drop_ge_50pct': bool(drop >= 0.5),
                        'domain_ok': bool(all(noloss.values()) and (any_gain or drop >= 0.5))}
    keep_t2 = all(dec['T2'][d]['domain_ok'] for d in S)
    dec['T2']['KEPT'] = keep_t2
    for arm in ARMS:
        dec['T4'][arm] = {}
        for d in S:
            F_, V = S[d]['frame'], S[d]['video']
            per = {}
            for b in LABELS[d]:
                da = F_[f'{arm}|extent|{b}']['cf_auroc']['mean'] - F_[f'{arm}|presence|{b}']['cf_auroc']['mean']
                vr = V[f'{arm}|{b}']['extent_mean']['video_r_rate']['mean'] - V[f'{arm}|{b}']['presence_mean']['video_r_rate']['mean']
                per[b] = {'d_cf_auroc_extent_minus_presence': da, 'd_video_r_extent_minus_presence': vr,
                          'extent_beats': bool(da >= 0.03 or vr >= 0.1)}
            land = S[d]['picks_levels'][arm]

            def clean(col):
                cp = [x for b in CONTACT[d] for x in land[b]]
                ok = np.mean([x[col] == 'pair' for x in cp]) >= 0.6
                sp = [x for b in SELF_LABELS[d] for x in land[b]]
                if sp:
                    ok = ok and np.mean([x[col] == 'self' for x in sp]) >= 0.6
                share = {'contact_pair_share': float(np.mean([x[col] == 'pair' for x in cp]))}
                if sp:
                    share['self_label_self_share'] = float(np.mean([x[col] == 'self' for x in sp]))
                return bool(ok), share
            ce, se = clean('social')
            cp_, sp_ = clean('social_presence')
            dec['T4'][arm][d] = {'per_label': per, 'extent_beats_any': any(v['extent_beats'] for v in per.values()),
                                 'level_clean_extent': ce, 'level_extent_shares': se,
                                 'level_clean_presence': cp_, 'level_presence_shares': sp_,
                                 'domain_ok': bool(any(v['extent_beats'] for v in per.values()) or (ce and not cp_))}
        dec['T4'][arm]['KEPT'] = all(dec['T4'][arm][d]['domain_ok'] for d in S)
    dec['T4']['decision_arm'] = 't2' if keep_t2 else 'base'
    dec['T4']['KEPT'] = dec['T4'][dec['T4']['decision_arm']]['KEPT']
    S['decisions'] = dec
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    tables(S)


def tables(S):
    f = lambda x: f"{x['mean']:.3f} ± {x['sd']:.3f}"  # noqa: E731
    f4 = lambda x: f"{x['mean']:.4f} ± {x['sd']:.4f}"  # noqa: E731
    L = ['# T2 (tight mask) x T4 (extent read-out), frame level, mean ± sd over 3 seeds\n']
    for d in ('mice', 'ants'):
        if d not in S:
            continue
        L.append(f'\n## {d}\n')
        L.append('| label | arm | read-out | cf AUROC | cf AP (AP pick) | AP @ AUROC pick | honest top-1% | size-ctrl AUROC |')
        L.append('|---|---|---|---|---|---|---|---|')
        for b in LABELS[d]:
            for arm in ARMS:
                for ro in ('presence', 'extent'):
                    x = S[d]['frame'][f'{arm}|{ro}|{b}']
                    L.append(f'| {b} (base rate {x["base_rate"]:.4f}) | {arm} | {ro} | {f(x["cf_auroc"])} | {f4(x["cf_ap"])} | '
                             f'{f4(x["cf_ap_at_auroc"])} | {f(x["cf_top1"])} | {f(x["cf_size_ctrl_auroc"])} |')
        L.append('\nVideo level (held-out halves; read-out of the AUROC pick of its read-out), raw / fg-controlled:\n')
        if d == 'mice':
            L.append('| label | arm | read-out | video r rate | video r bouts/min | pooled d-r | sig. switches with sign agree (p<.05) |')
            L.append('|---|---|---|---|---|---|---|')
        else:
            L.append('| label | arm | read-out | video r rate | video r bouts/min |')
            L.append('|---|---|---|---|---|')
        for b in LABELS[d]:
            for arm in ARMS:
                V = S[d]['video'][f'{arm}|{b}']
                for ro in READOUTS:
                    a, c = V[ro], V[f'{ro}|fgctrl']
                    row = (f'| {b} | {arm} | {ro} | {a["video_r_rate"]["mean"]:.2f} / {c["video_r_rate"]["mean"]:.2f} | '
                           f'{a["video_r_bpm"]["mean"]:.2f} / {c["video_r_bpm"]["mean"]:.2f} |')
                    if d == 'mice':
                        row += (f' {a["delta_r_pooled"]["mean"]:.2f} / {c["delta_r_pooled"]["mean"]:.2f} | '
                                f'{a["n_sig_sign_agree"]["mean"]:.1f} ({a["n_sig_reproduced_p05"]["mean"]:.1f}) of '
                                f'{a["n_sig_switches"]["mean"]:.0f} / {c["n_sig_sign_agree"]["mean"]:.1f} '
                                f'({c["n_sig_reproduced_p05"]["mean"]:.1f}) |')
                    L.append(row)
        L.append('\nLevel map (eligible neurons, mean over seeds):\n')
        for arm in ARMS:
            Lv = S[d]['levels'][arm]
            L.append(f'- {arm}: eligible {Lv["n_eligible"]["mean"]:.0f}; plain background {Lv["plain_background"]["mean"]:.1f}; '
                     f'place-bound {Lv["place_bound"]["mean"]:.1f}; social (extent) '
                     + ', '.join(f'{c} {v["mean"]:.1f}' for c, v in Lv['social'].items())
                     + '; social (presence-only) ' + ', '.join(f'{c} {v["mean"]:.1f}' for c, v in Lv['social_presence'].items())
                     + '; not-plain social x place ' + ', '.join(f'{c} {v["mean"]:.1f}' for c, v in Lv['social_x_place_not_plain'].items()))
        L.append('\nWhere the annotated AUROC picks land (all seeds x halves x read-outs):\n')
        for arm in ARMS:
            for b, xs in S[d]['picks_levels'][arm].items():
                cnt = pd.Series([x['social'] + (',place' if x['place_bound'] else '') + (',PLAIN' if x['plain_background'] else '')
                                 for x in xs]).value_counts().to_dict()
                cntp = pd.Series([x['social_presence'] for x in xs]).value_counts().to_dict()
                L.append(f'- {arm} {b}: extent map {cnt}; presence-only map {cntp}')
    L.append('\n## Decisions\n')
    L.append('```\n' + json.dumps(S['decisions'], indent=1, default=float) + '\n```')
    (OUT / 'tables.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
