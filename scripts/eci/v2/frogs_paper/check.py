"""
ECI v2 frog discoveries vs the lab's paper (Vijatovic, Toma et al., Cell Reports 45, 117227, 2026;
doi 10.1016/j.celrep.2026.117227; preprint bioRxiv 10.1101/2024.09.20.614050), and an audit of the side alignment.

Read-only on everything outside this folder and results/vision/eci_v2/frogs_paper/. Steps (all stored data):

 A  side audit (juveniles, 35 hour-1 videos of data/frogs/v1/experiment.csv):
    - the lab's per-animal json {'mutant_side'} next to every mp4 (hour 1 AND hour 2 of the same frog, original and
      REPRED folders) vs experiment.csv mutant_side; WT: json present or not;
    - hflip rule: hflip == 1 iff (group != WT and mutant_side == left); WT hflip = seeded random; plan.json hflip ==
      experiment.csv hflip (the plan is what the v2 pipeline decodes with, src/eci/v2/pipeline.py Series.job);
    - SLEAP handedness: in SOURCE px, fraction of frames where Right_Hip / Right_Shoulder lie on the frog's right of
      the body axis (Tail_Stem -> eyes midpoint; dorsal view from a camera above the dish, image y down: cross > 0);
      and the same in CANONICAL px after the v2 mirroring (x -> 799 - x, Left_ <-> Right_ names);
    - phenotype-based side check: the FoxP1 mutant knee is held near-straight (paper Fig. 7G); per FoxP1 frog,
      which SLEAP side has the smaller knee angle (5 fps SLEAP; and the lab's own 60 fps angle tables) vs the json;
    - pixel check that the decoded canonical frame and the mirrored SLEAP nodes agree: grey level (frog = dark on the
      backlit dish) at the 4 trunk nodes mirrored (what v2 uses) vs NOT mirrored, 4 frames per video decoded with
      src/eci/v2/video.decode (the v2 decoder, hflip from plan.json).
 B  the lab's own per-video kinematics (MNV1_results/2026-03-30_finalandupdated *.tab, produced by the lab pipeline,
    60 fps; for the half crispants these tables are side-normalised, 'left' = mutant side, see paper_reproduction): knee / ankle / hip / foot / elbow / wrist / shoulder angle
    mean and std (moving), low-band power and dominant frequency, left-right correlation, locomotion. Reproduces the
    paper's Fig. 7 contrasts on exactly the ECI animals (hour 1 = the ECI videos; both hours also given).
 C  where each juvenile discovery neuron fires (copied logic of scripts/eci/v2/frog_neuron_check.py: stored argmax
    token -> nearest SLEAP node in canonical px), and for #52 / #368 the per-frame link to the posture of the limb
    under the peak (knee angle, leg extension) - label-free, from SLEAP at the same 5 fps frames.
 D  per-video neuron outcome (NES primary input, smooth set, cached full__mean) vs the lab per-video measures,
    Spearman within group (juveniles: FoxP1, WT, En1; tadpoles 44-48: En1, FoxP1, WT).
 E  every selected NES / NESS neuron for frogs (juv + tad, 3 input sets, 3 aggregations) with p.

Usage (Slurm CPU): sbatch scripts/eci/v2/frogs_paper/check.sh
Output: results/vision/eci_v2/frogs_paper/check.json (+ csv tables)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2')
LAB = Path('/nfs/scistore23/sweengrp/ftoma/Behavior_MNV1')
RAW = LAB / 'MNV1_revision_dataset/Juv'
REPRED = LAB / 'MNV1_revision_dataset_REPRED/Juv'
TAB = LAB / 'MNV1_results/2026-03-30_finalandupdated'
OUT = REPO / 'results/vision/eci_v2/frogs_paper'
NES = REPO / 'results/vision/frogs'
FOLD = {'WT': ('WT', 'WT REPRED'), 'FoxP1': ('FoxP1 gRNA1 half', 'FoxP1 gRNA1 half REPRED'),
        'En1': ('En1 gRNA4 half Genotyped', 'En1 gRNA4 half Gen REPRED')}
LABG = {'WT': 'WT', 'FoxP1': 'FoxP1 gRNA1 half', 'En1': 'En1 gRNA4 half Genotyped'}
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731

PART = {'Right_Eye': 'head', 'Left_Eye': 'head', 'Heart_Center': 'trunk', 'Tail_Stem': 'trunk'}
for _side in ('Right', 'Left'):
    for _p in ('Hand', 'Wrist', 'Elbow', 'Shoulder'):
        PART[f'{_side}_{_p}'] = f'{_side[0]} forelimb'
    for _p in ('Hip', 'Knee', 'Ankle', 'Foot', 'toe'):
        PART[f'{_side}_{_p}'] = f'{_side[0]} hindlimb'
PARTS = ['head', 'trunk', 'R forelimb', 'L forelimb', 'R hindlimb', 'L hindlimb']


def canon_names(names, hflip):
    if not hflip:
        return list(names)
    return [n.replace('Left_', 'TMP_').replace('Right_', 'Left_').replace('TMP_', 'Right_') for n in names]


def jside(p):
    return json.loads(Path(p).read_text())['mutant_side'].strip() if Path(p).exists() else None


def joint_angle(nd, names, a, b, c):
    """Unsigned angle (deg) between segments a->b and b->c; 0 = straight (parallel segments, the paper's convention)."""
    i, j, k = names.index(a), names.index(b), names.index(c)
    u, v = nd[:, j] - nd[:, i], nd[:, k] - nd[:, j]
    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def leg_ext(nd, names, s):
    """Hip -> Foot distance / summed Hip-Knee-Ankle-Foot segment length (1 = fully straight leg)."""
    P = [nd[:, names.index(f'{s}_{q}')] for q in ('Hip', 'Knee', 'Ankle', 'Foot')]
    L = sum(np.linalg.norm(P[i + 1] - P[i], axis=1) for i in range(3))
    return np.linalg.norm(P[3] - P[0], axis=1) / L


def side_of(nd, names, node):
    """cross > 0: node on the frog's right of the body axis (image y down, dorsal view)."""
    head = nd[:, [names.index('Right_Eye'), names.index('Left_Eye')]].mean(1)
    ts = nd[:, names.index('Tail_Stem')]
    ax_, rel = head - ts, nd[:, names.index(node)] - ts
    cr = ax_[:, 0] * rel[:, 1] - ax_[:, 1] * rel[:, 0]
    ok = np.isfinite(cr)
    return float((cr[ok] > 0).mean()), int(ok.sum())


def mwu(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return dict(n=[len(a), len(b)])
    u, p = stats.mannwhitneyu(a, b)
    return dict(n=[len(a), len(b)], median=[float(np.median(a)), float(np.median(b))], p_mwu=float(p),
                rank_biserial=float(2 * u / (len(a) * len(b)) - 1))


def wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return dict(n=int(len(a)))
    return dict(n=int(len(a)), median=[float(np.median(a)), float(np.median(b))],
                p_wilcoxon=float(stats.wilcoxon(a, b).pvalue), n_a_gt_b=int((a > b).sum()))


# ---------------------------------------------------------------------------------------------------------- A
def side_audit(exp, plan):
    rows = []
    P = {v['obs']: v for v in plan['videos']}
    for r in exp.itertuples():
        src = Path(r.source_file)
        g = r.group
        hour2 = sorted(str(p) for p in (RAW / FOLD[g][0]).glob(f'{r.session}_*stJuv*_2_{r.slot}.json'))
        rec = dict(obs=r.observation_id, group=g, session=r.session, slot=r.slot, csv_side=r.mutant_side
                   if isinstance(r.mutant_side, str) else '', json_h1=jside(src.with_suffix('.json')),
                   json_h1_repred=jside(REPRED / FOLD[g][1] / f'{src.stem}.json'),
                   json_h2=[jside(p) for p in hour2], hflip_csv=int(r.hflip), hflip_plan=int(P[r.observation_id]['hflip']))
        rec['hflip_rule'] = int(g != 'WT' and rec['csv_side'] == 'left')
        rec['ok'] = bool((g == 'WT' or rec['json_h1'] == rec['csv_side']) and
                         rec['hflip_csv'] == rec['hflip_plan'] and (g == 'WT' or rec['hflip_csv'] == rec['hflip_rule'])
                         and all(s in (None, rec['json_h1']) for s in rec['json_h2'])
                         and rec['json_h1_repred'] in (None, rec['json_h1']))
        # SLEAP handedness + knee posture per side (5 fps npz, source px)
        Z = np.load(REPO / 'dataset/frogs/v1/sleap' / f'{r.observation_id}.npz')
        nd = Z['nodes'].astype(np.float64)
        names = [str(s) for s in Z['node_names']]
        for node in ('Right_Hip', 'Left_Hip', 'Right_Shoulder', 'Left_Shoulder'):
            rec[f'src_rightshare_{node}'] = side_of(nd, names, node)[0]
        ndc = nd.copy()
        if rec['hflip_plan']:
            ndc[..., 0] = 799 - ndc[..., 0]
        cn = canon_names(names, rec['hflip_plan'])
        for node in ('Right_Hip', 'Left_Hip'):
            rec[f'canon_rightshare_{node}'] = side_of(ndc, cn, node)[0]
        for s in ('Right', 'Left'):
            ka = joint_angle(nd, names, f'{s}_Hip', f'{s}_Knee', f'{s}_Ankle')
            rec[f'knee_mean_{s}'] = float(np.nanmean(ka))
            rec[f'legext_{s}'] = float(np.nanmedian(leg_ext(nd, names, s)))
            ea = joint_angle(nd, names, f'{s}_Shoulder', f'{s}_Elbow', f'{s}_Wrist')
            rec[f'elbow_mean_{s}'] = float(np.nanmean(ea))
            for q in ('Hand', 'Wrist', 'Elbow', 'Knee', 'Foot'):
                rec[f'missing_{s}_{q}'] = float(np.isnan(nd[:, names.index(f'{s}_{q}'), 0]).mean())
        rec['straighter_knee_side'] = 'right' if rec['knee_mean_Right'] < rec['knee_mean_Left'] else 'left'
        rows.append(rec)
    return pd.DataFrame(rows)


def pixel_check(plan, n_frames=4):
    """Decoded canonical frame (v2 decoder) vs mirrored SLEAP trunk nodes: grey at mirrored vs unmirrored positions."""
    from src.eci.v2.video import decode
    out = []
    rng = np.random.default_rng(0)
    for v in plan['videos']:
        Z = np.load(REPO / 'dataset/frogs/v1/sleap' / f'{v["obs"]}.npz')
        nd = Z['nodes'].astype(np.float64)
        names = [str(s) for s in Z['node_names']]
        trunk = [names.index(n) for n in ('Heart_Center', 'Tail_Stem', 'Right_Eye', 'Left_Eye')]
        ok = np.flatnonzero(np.isfinite(nd[:, trunk, 0]).all(1) & (np.abs(nd[:, trunk, 0].mean(1) - 399.5) > 60))
        ks = np.sort(rng.choice(ok, n_frames, replace=False))
        g_m, g_u = [], []
        for k in ks:
            im = decode(v['path'], 5 + 12 * int(k), 1, 12, 800, int(v['hflip']), 60, threads=2)[0].mean(2)
            xy = nd[k, trunk]
            xm = (799 - xy[:, 0]) if v['hflip'] else xy[:, 0]
            xu = xy[:, 0] if v['hflip'] else (799 - xy[:, 0])

            def grey(xs):
                return float(np.mean([im[int(np.clip(y, 2, 797)) - 2:int(np.clip(y, 2, 797)) + 3,
                                         int(np.clip(x, 2, 797)) - 2:int(np.clip(x, 2, 797)) + 3].mean()
                                      for x, y in zip(xs, xy[:, 1])]))
            g_m.append(grey(xm))
            g_u.append(grey(xu))
        out.append(dict(obs=v['obs'], hflip=int(v['hflip']), frames=[int(k) for k in ks], grey_v2_nodes=g_m,
                        grey_other_mirror=g_u, agree=bool(all(a < b for a, b in zip(g_m, g_u)))))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------------------- B
def lab_tables(exp):
    """Lab per-video measures, juveniles, keyed by movie stem; returns long table (movie, group, measure, side, value)."""
    joints = ['knee', 'ankle', 'hip', 'foot', 'elbow', 'wrist', 'shoulder']
    recs = []
    for j in joints:
        for s in ('left', 'right'):
            a = pd.read_csv(TAB / f'angle_range/angle_range_Juv_{s}_{j}_res.tab', sep='\t', index_col=0)
            f = pd.read_csv(TAB / f'frequency/frequency_Juv_{s}_{j}_res.tab', sep='\t', index_col=0)
            for _, x in a.iterrows():
                recs.append((x.Movie, x.Genotype, f'{j}_angle_moving_mean', s, x.angle_moving_mean))
                recs.append((x.Movie, x.Genotype, f'{j}_angle_moving_std', s, x.angle_moving_std))
            for _, x in f.iterrows():
                recs.append((x.Movie, x.Genotype, f'{j}_dominant_freq_low', s, x['dominant_freq_4.5-']))
                recs.append((x.Movie, x.Genotype, f'{j}_sum_power_low', s, x['psf_sum_freq_0.0-4.5']))
    lr = {}
    for j in joints:
        c = pd.read_csv(TAB / f'angle_correlation/angle_correlation_Juv_LR_{j}_res.tab', sep='\t', index_col=0)
        for _, x in c.iterrows():
            lr[(x.Movie, j)] = x.corr_median
    loc = pd.read_csv(TAB / 'locomotion/locomotion_Juv_res.tab', sep='\t', index_col=0).set_index('Movie')
    L = pd.DataFrame(recs, columns=['movie', 'genotype', 'measure', 'side', 'value'])
    return L, lr, loc


def paper_reproduction(exp, L, lr, loc):
    """Per animal: mutant vs intact side (json), FoxP1 vs WT, En1 vs WT; hour 1 (the ECI videos) and both hours."""
    # animal table: (group, session, slot) -> movies of hour 1 / 2 and the json side
    an = []
    for g, lg in LABG.items():
        for p in sorted((RAW / FOLD[g][0]).glob('*.mp4')):
            parts = p.stem.split('_')
            ses, A, B = int(parts[0]), int(parts[-2]), int(parts[-1])
            an.append(dict(group=g, session=ses, slot=B, hour=A, movie=p.stem, side=jside(p.with_suffix('.json'))))
    an = pd.DataFrame(an)
    W = L.pivot_table(index=['movie', 'measure'], columns='side', values='value').reset_index()
    W = W.merge(an, on='movie', how='inner')
    # The lab's tables are side-NORMALISED for the half crispants: 'left' = the mutant (json) side, 'right' = uninjected
    # (right-mutant animals are mirrored by the lab pipeline; measured against the SLEAP node names in
    # sleap_handedness.py: the near-straight FoxP1 knee is in 'left_knee' for 24/24 right-mutant and 2/2 resolvable
    # left-mutant rows). WT rows keep the SLEAP names.
    W['mut'] = np.where(W.group == 'WT', np.nan, W['left'])
    W['int'] = np.where(W.group == 'WT', np.nan, W['right'])
    W['mean_lr'] = W[['left', 'right']].mean(1)
    res = {}
    for hours, tag in (((1,), 'hour1'), ((1, 2), 'both_hours')):
        Wh = W[W.hour.isin(hours)]
        A_ = Wh.groupby(['group', 'session', 'slot', 'measure', 'side'], dropna=False)[
            ['left', 'right', 'mut', 'int', 'mean_lr']].mean().reset_index()
        out = {}
        for m in sorted(A_.measure.unique()):
            M = A_[A_.measure == m]
            wt = M[M.group == 'WT']
            fx = M[M.group == 'FoxP1']
            en = M[M.group == 'En1']
            absf = (lambda x: np.abs(x)) if 'angle_moving_mean' in m else (lambda x: x)
            wt_sides = np.r_[absf(wt.left.values), absf(wt.right.values)]
            out[m] = dict(FoxP1_mut_vs_int=wil(absf(fx.mut.values), absf(fx.int.values)),
                          FoxP1_mut_vs_WT_sides=mwu(absf(fx.mut.values), wt_sides),
                          FoxP1_int_vs_WT_sides=mwu(absf(fx.int.values), wt_sides),
                          En1_mut_vs_int=wil(absf(en.mut.values), absf(en.int.values)),
                          En1_sides_vs_WT_sides=mwu(np.r_[absf(en.left.values), absf(en.right.values)], wt_sides),
                          En1_meanLR_vs_WT_meanLR=mwu(absf(en.mean_lr.values), absf(wt.mean_lr.values)))
        # left-right correlation and locomotion (animal level)
        for j in ('knee', 'ankle', 'elbow', 'wrist'):
            v = an[an.hour.isin(hours)].copy()
            v['c'] = [lr.get((mv, j), np.nan) for mv in v.movie]
            v = v.groupby(['group', 'session', 'slot']).c.mean().reset_index()
            out[f'LR_{j}_corr_median'] = dict(
                FoxP1_vs_WT=mwu(v[v.group == 'FoxP1'].c, v[v.group == 'WT'].c),
                En1_vs_WT=mwu(v[v.group == 'En1'].c, v[v.group == 'WT'].c))
        for c in ('time_spend_moving_ratio', 'total_distance', 'speed_moving_mean', 'acceleration_mean',
                  'directional_change_mean'):
            v = an[an.hour.isin(hours)].copy()
            v['c'] = [loc[c].get(mv, np.nan) for mv in v.movie]
            v = v.groupby(['group', 'session', 'slot']).c.mean().reset_index()
            out[f'loc_{c}'] = dict(FoxP1_vs_WT=mwu(v[v.group == 'FoxP1'].c, v[v.group == 'WT'].c),
                                   En1_vs_WT=mwu(v[v.group == 'En1'].c, v[v.group == 'WT'].c))
        res[tag] = out
    # per FoxP1 animal hour 1: which lab side has the straighter knee (smaller |mean angle|)
    k = W[(W.measure == 'knee_angle_moving_mean') & (W.hour == 1)]
    per = [dict(movie=x.movie, group=x.group, json_side=x.side, lab_abs_knee_left=abs(x.left),
                lab_abs_knee_right=abs(x.right),
                lab_straighter_knee='left' if abs(x.left) < abs(x.right) else 'right') for x in k.itertuples()]
    return res, pd.DataFrame(per), W


# ---------------------------------------------------------------------------------------------------------- C
def neuron_parts(exp, plan, js, max_dist=24.0):
    res = {j: {} for j in js}
    link = {}
    for v in plan['videos']:
        obs, hflip = v['obs'], int(v['hflip'])
        grp = exp.loc[obs, 'group']
        z = np.load(BEE / 'frogs/videos/juv' / f'{obs}.npz')
        G = int(z['G'])
        cm = z['codes_max'][:, js].astype(np.float32)
        am = z['argmax'][:, js].astype(np.int64)
        Z = np.load(REPO / 'dataset/frogs/v1/sleap' / f'{obs}.npz')
        nodes = Z['nodes'].astype(np.float32).copy()
        if hflip:
            nodes[..., 0] = 799 - nodes[..., 0]
        names = canon_names([str(s) for s in Z['node_names']], hflip)
        part = np.array([PART[n] for n in names])
        n = min(len(cm), len(nodes))
        nd = nodes[:n].astype(np.float64)
        kn = {s: joint_angle(nd, names, f'{s}_Hip', f'{s}_Knee', f'{s}_Ankle') for s in ('Right', 'Left')}
        le = {s: leg_ext(nd, names, s) for s in ('Right', 'Left')}
        # egocentric foot speed (body lengths per 0.2 s): foot relative to hip, frame to frame
        bl = np.linalg.norm(nd[:, names.index('Heart_Center')] - nd[:, names.index('Tail_Stem')], axis=1)
        fs = {}
        for s in ('Right', 'Left'):
            rel = nd[:, names.index(f'{s}_Foot')] - nd[:, names.index(f'{s}_Hip')]
            fs[s] = np.r_[np.nan, np.linalg.norm(np.diff(rel, axis=0), axis=1) / np.nanmedian(bl)]
        link[obs] = dict(group=grp, hflip=hflip,
                         knee_R=float(np.nanmean(kn['Right'])), knee_L=float(np.nanmean(kn['Left'])),
                         legext_R=float(np.nanmedian(le['Right'])), legext_L=float(np.nanmedian(le['Left'])),
                         footspeed_R=float(np.nanmean(fs['Right'])), footspeed_L=float(np.nanmean(fs['Left'])))
        for q, j in enumerate(js):
            act = np.flatnonzero(cm[:n, q] > 0)
            rec = dict(group=grp, hflip=hflip, n=int(n), frac_active=float(len(act) / n), mean_raw=float(cm[:n, q].mean()))
            if len(act):
                p = am[act, q]
                c = np.stack([(p % G) * 16 + 8, (p // G) * 16 + 8], 1).astype(np.float32)
                d = np.linalg.norm(nodes[act] - c[:, None], axis=2)
                d = np.where(np.isfinite(d), d, np.inf)
                k = d.argmin(1)
                dmin = d[np.arange(len(k)), k]
                lab = np.where(dmin <= max_dist, part[k], 'off')
                rec['parts'] = {pp: int((lab == pp).sum()) for pp in PARTS + ['off']}
                # posture of the hind limb under the peak vs the other hind limb, same frame
                for s, o in (('R', 'L'), ('L', 'R')):
                    sel = act[lab == f'{s} hindlimb']
                    S_, O_ = {'R': 'Right', 'L': 'Left'}[s], {'R': 'Right', 'L': 'Left'}[o]
                    rec[f'peak{s}_knee_peak_vs_other'] = [float(np.nanmean(kn[S_][sel])), float(np.nanmean(kn[O_][sel]))] if len(sel) else None
                    rec[f'peak{s}_legext_peak_vs_other'] = [float(np.nanmean(le[S_][sel])), float(np.nanmean(le[O_][sel]))] if len(sel) else None
                # per-frame Spearman of the code with right / left knee angle and leg extension (all frames)
                x = cm[:n, q]
                for nm, arr in (('knee_R', kn['Right']), ('knee_L', kn['Left']), ('legext_R', le['Right']),
                                ('legext_L', le['Left']), ('footspeed_R', fs['Right']), ('footspeed_L', fs['Left'])):
                    ok = np.isfinite(arr)
                    rec[f'rho_{nm}'] = float(stats.spearmanr(x[ok], arr[ok]).statistic) if ok.sum() > 100 and x[ok].std() > 0 else None
            res[j][obs] = rec
        log(f'{obs} ({grp}) parts done')
    summ = {}
    for j in js:
        s_ = {}
        for g in ('FoxP1', 'WT', 'En1'):
            R = [r for r in res[j].values() if r['group'] == g]
            tot = {pp: sum(r.get('parts', {}).get(pp, 0) for r in R) for pp in PARTS + ['off']}
            N = max(1, sum(tot.values()))
            ent = dict(n_videos=len(R), median_frac_active=float(np.median([r['frac_active'] for r in R])),
                       mean_raw=float(np.mean([r['mean_raw'] for r in R])),
                       part_share={pp: round(tot[pp] / N, 3) for pp in tot})
            for key in ('rho_knee_R', 'rho_knee_L', 'rho_legext_R', 'rho_legext_L', 'rho_footspeed_R', 'rho_footspeed_L'):
                vals = [r[key] for r in R if r.get(key) is not None]
                ent[f'median_{key}'] = float(np.median(vals)) if vals else None
            for s in ('R', 'L'):
                pv = [r[f'peak{s}_legext_peak_vs_other'] for r in R if r.get(f'peak{s}_legext_peak_vs_other')]
                if len(pv) >= 3:
                    a, b = np.array(pv).T
                    ent[f'peak{s}_legext_peak_vs_other'] = wil(a, b)
                kv = [r[f'peak{s}_knee_peak_vs_other'] for r in R if r.get(f'peak{s}_knee_peak_vs_other')]
                if len(kv) >= 3:
                    a, b = np.array(kv).T
                    ent[f'peak{s}_knee_peak_vs_other'] = wil(a, b)
            s_[g] = ent
        summ[j] = s_
    return summ, res, pd.DataFrame(link).T


# ---------------------------------------------------------------------------------------------------------- D
def outcome_vs_lab(exp, W, loc, js_juv, tad_js):
    out = {}
    z = np.load(NES / 'eci/nes/v2_frogs_c0_w2048_s6_smooth/_cache/video_summaries_max.npz')
    Y = pd.DataFrame(z['full__mean'][:, js_juv], index=z['observation_id'], columns=js_juv)
    e = exp.copy()
    e['movie'] = [Path(s).stem for s in e.source_file]
    meas = {}
    for m in ('knee_angle_moving_mean', 'knee_angle_moving_std', 'knee_sum_power_low', 'knee_dominant_freq_low',
              'elbow_angle_moving_mean', 'elbow_angle_moving_std', 'wrist_angle_moving_std', 'ankle_angle_moving_std'):
        M = W[W.measure == m].set_index('movie')
        f = (lambda x: np.abs(x)) if 'angle_moving_mean' in m else (lambda x: x)
        for col, key in (('right', 'R'), ('left', 'L')):
            meas[f'{m}_{key}'] = {mv: f(M[col].get(mv, np.nan)) for mv in e.movie}
    # canonical sides: R = the lab's right unless the video is mirrored (hflip), then the lab's left
    for g in ('FoxP1', 'WT', 'En1'):
        eg = e[e.group == g]
        res = {}
        for j in js_juv:
            y = Y.loc[eg.observation_id, j].values
            r_ = {}
            for m in sorted({k[:-2] for k in meas}):
                # ECI canonical right: mutant side for FoxP1 / En1 = lab 'left' (side-normalised tables); WT: the lab's
                # right unless the video is mirrored
                if g == 'WT':
                    cr = np.array([meas[f'{m}_{"L" if h else "R"}'][mv] for mv, h in zip(eg.movie, eg.hflip)], float)
                    cl = np.array([meas[f'{m}_{"R" if h else "L"}'][mv] for mv, h in zip(eg.movie, eg.hflip)], float)
                else:
                    cr = np.array([meas[f'{m}_L'][mv] for mv in eg.movie], float)
                    cl = np.array([meas[f'{m}_R'][mv] for mv in eg.movie], float)
                for nm, x in (('canonR', cr), ('canonL', cl), ('canonR_minus_L', cr - cl)):
                    ok = np.isfinite(x)
                    if ok.sum() >= 6 and np.std(y[ok]) > 0:
                        t = stats.spearmanr(y[ok], x[ok])
                        r_[f'{m}_{nm}'] = [round(float(t.statistic), 3), float(t.pvalue), int(ok.sum())]
            res[str(j)] = r_
        out[f'juv_{g}'] = res
    # tadpoles (lab tables 44-48; whole = bilateral)
    zt = np.load(NES / 'eci_tadpole_r4/nes/v2_frogs_c0_w2048_s6_smooth/_cache/video_summaries_max.npz')
    Yt = pd.DataFrame(zt['full__mean'][:, tad_js], index=zt['observation_id'], columns=tad_js)
    et = pd.read_csv(REPO / 'data/frogs/v1_tadpole/experiment.csv').set_index('observation_id')
    et['movie'] = [Path(s).stem for s in et.source_file]
    lt = pd.read_csv(TAB / 'locomotion/locomotion_44-48_res.tab', sep='\t', index_col=0).groupby('Movie').mean(numeric_only=True)
    ar = pd.read_csv(TAB / 'angle_range/angle_range_44-48_tail_2_res.tab', sep='\t', index_col=0).groupby('Movie').mean(numeric_only=True)
    fr = pd.read_csv(TAB / 'frequency/frequency_44-48_tail_2_res.tab', sep='\t', index_col=0).groupby('Movie').mean(numeric_only=True)
    T = pd.DataFrame(index=Yt.index)
    T['group'] = et.loc[T.index, 'group'].values
    mv = et.loc[T.index, 'movie']
    for c in ('time_spend_moving_ratio', 'total_distance', 'speed_moving_mean', 'acceleration_mean', 'directional_change_mean'):
        T[c] = lt[c].reindex(mv).values
    T['tail2_angle_moving_std'] = ar['angle_moving_std'].reindex(mv).values
    T['tail2_dominant_freq_low'] = fr['dominant_freq_4.5-'].reindex(mv).values
    T['tail2_sum_power_low'] = fr['psf_sum_freq_0.0-4.5'].reindex(mv).values
    T['tail2_sum_power_high'] = fr['psf_sum_freq_4.5-20.0'].reindex(mv).values
    T['tail2_high_over_low'] = T.tail2_sum_power_high / T.tail2_sum_power_low
    cols = [c for c in T.columns if c != 'group']
    tad = {'matched_videos': {g: int(T[T.group == g][cols[0]].notna().sum()) for g in ('WT', 'En1', 'FoxP1')},
           'paper_repro': {c: {'En1_vs_WT': mwu(T[T.group == 'En1'][c], T[T.group == 'WT'][c]),
                               'FoxP1_vs_WT': mwu(T[T.group == 'FoxP1'][c], T[T.group == 'WT'][c])} for c in cols}}
    for g in ('En1', 'WT', 'FoxP1'):
        Tg = T[T.group == g]
        r = {}
        for j in tad_js:
            y = Yt.loc[Tg.index, j].values
            rr = {}
            for c in cols:
                x = Tg[c].values.astype(float)
                ok = np.isfinite(x)
                if ok.sum() >= 8 and np.std(y[ok]) > 0:
                    t = stats.spearmanr(y[ok], x[ok])
                    rr[c] = [round(float(t.statistic), 3), float(t.pvalue), int(ok.sum())]
            r[str(j)] = rr
        tad[f'within_{g}'] = r
    out['tad'] = tad
    return out


# ---------------------------------------------------------------------------------------------------------- E
def discoveries():
    rows = []
    for series, base in (('juv', NES / 'eci/nes'), ('tad', NES / 'eci_tadpole_r4/nes')):
        for s in ('v2_frogs_c0_w2048_s6', 'v2_frogs_c0_w2048_s6_smooth', 'v2_frogs_c0_w2048_s6_prob'):
            for agg in ('', 'maxpool_bouts', 'maxpool_latency'):
                f = base / s / agg / 'selected_neurons.json'
                if not f.exists():
                    continue
                d = json.loads(f.read_text())
                for n, v in d['neurons'].items():
                    for h in v['hits']:
                        tk = [k for k in h if k.startswith('tau')][0]
                        rows.append(dict(series=series, input_set=s[21:] or 'threshold',
                                         aggregation=agg.replace('maxpool_', '') or 'mean', method='NES',
                                         contrast=h['analysis_id'], neuron=int(n), prefix=h['prefix'], round=h['round'],
                                         direction=h['direction'], effect_name=tk, effect=h[tk], p=h['p'],
                                         artefact_flag=v.get('artefact_flag', '')))
    nes_dir = REPO / 'results/vision/eci_v2/ness'
    for series, dom in (('juv', 'frogs'), ('tad', 'frogs_tadpole')):
        for s in ('v2_frogs_c0_w2048_s6', 'v2_frogs_c0_w2048_s6_smooth', 'v2_frogs_c0_w2048_s6_prob'):
            for agg in ('', 'maxpool_bouts', 'maxpool_latency'):
                f = nes_dir / dom / s / agg / 'dines/summary.csv'
                if not f.exists():
                    continue
                d = pd.read_csv(f)
                d = d[d.neuron.notna()] if 'neuron' in d else d.iloc[0:0]
                for x in d.itertuples():
                    rows.append(dict(series=series, input_set=s[21:] or 'threshold',
                                     aggregation=agg.replace('maxpool_', '') or 'mean', method='NESS',
                                     contrast=x.analysis_id, neuron=int(x.neuron), prefix=x.prefix, round=x.round,
                                     direction=getattr(x, 'direction', ''), effect_name='tau',
                                     effect=getattr(x, 'tau', np.nan), p=getattr(x, 'p', np.nan), artefact_flag=''))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exp = pd.read_csv(REPO / 'data/frogs/v1/experiment.csv')
    plan = json.loads((REPO / 'results/vision/eci_rollout_r4/frogs/plan.json').read_text())
    res = {}
    D = discoveries()
    D.to_csv(OUT / 'discoveries_frogs.csv', index=False)
    log(f'discoveries: {len(D)} rows')
    A = side_audit(exp, plan)
    A.to_csv(OUT / 'side_audit.csv', index=False)
    log(f'side audit: {int(A.ok.sum())}/{len(A)} ok')
    PX = pixel_check(plan)
    PX.to_csv(OUT / 'pixel_check.csv', index=False)
    log(f'pixel check: {int(PX.agree.sum())}/{len(PX)} agree')
    L, lr, loc = lab_tables(exp)
    rep, per, W = paper_reproduction(exp, L, lr, loc)
    per.to_csv(OUT / 'lab_knee_side.csv', index=False)
    res['paper_reproduction'] = rep
    log('lab tables done')
    js_juv = sorted({int(x) for x in D[D.series == 'juv'].neuron} | {52, 368, 25, 61, 278, 34, 383, 63, 74, 124, 1694, 1849})
    tad_js = sorted({int(x) for x in D[D.series == 'tad'].neuron})
    summ, per_video, link = neuron_parts(exp.set_index('observation_id'), plan, js_juv)
    link.to_csv(OUT / 'posture_per_video.csv')
    res['neuron_parts'] = summ
    res['neuron_parts_per_video'] = per_video
    res['outcome_vs_lab'] = outcome_vs_lab(exp, W, loc, js_juv, tad_js)
    res['side_audit_summary'] = dict(
        n=len(A), ok=int(A.ok.sum()), by_group=A.groupby('group').agg(n=('obs', 'size'), hflip=('hflip_plan', 'sum'),
                                                                      left=('csv_side', lambda s: int((s == 'left').sum())),
                                                                      right=('csv_side', lambda s: int((s == 'right').sum()))).to_dict('index'),
        pixel_agree=int(PX.agree.sum()), pixel_n=len(PX))
    (OUT / 'check.json').write_text(json.dumps(res, indent=1, default=float))
    log('done')


if __name__ == '__main__':
    main()
