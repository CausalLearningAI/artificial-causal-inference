"""
Readout normalisation of the C1 deployment SAEs (src/eci/readout_norm.py): every neuron's frame read-out codes_max
becomes an EVENT (codes_max > 0) plus an INTENSITY a_norm = min(1, codes_max / q_j) in [0, 1], q_j = the 99th percentile
of neuron j's positive read-outs over the SAE's TRAINING frames (the 1 fps token-store frames its token pool was drawn
from, scripts/eci/rollout_c1.py STORE: the 'rows' of every shard's frames.npz). Label-free; a_norm is not a calibrated
probability of any human label.

Steps (scripts/eci/readout_norm.sh runs each in Slurm, CPU only)
    run     (per domain) stage codes_max.npy + n_fg.npy of <eci>/codes/<sae>/ to $LOCAL_DIR; fit q on the training
            frames -> <eci>/sae/<sae>/readout_norm.npz; one streamed pass over every frame writes the normalised
            codes_max to $LOCAL_DIR only (31 GB for the three domains: not kept on NFS, a_norm is computed on the fly
            from the npz) and counts events / clipped frames per neuron; then
              invariance  (mice, ants) best-neuron AUROC / AP, raw vs normalised codes, on the labelled 1 fps frames of
                          the rollout's align step (rollout_c1.labelled_fps1 + sae_levers.score_model, cross-fitted
                          over the same unit halves), and NES (run_nes.py activation + run_nes_bouts.py bout rate, the
                          rollout settings: codes_max, prefixes 512 / 1024, nuisance nfg, select p, ants --day on the
                          'pairs' set, --primary-only, bouts without the sanity nulls) on the raw local copy AND on the
                          normalised codes; selections compared with the rollout's NES results
              interpretability  clipped-frame fractions per neuron; for the align.json picks (+ mice 180, the bag
                          neuron) mean a_norm / event rate on annotated-positive vs negative frames (all labelled
                          5 fps frames)
              figure data the segment of the fixed rule (FIG_RULE) -> OUT/<d>/fig_data.npz
            -> OUT/<d>/readout_norm.json, OUT/<d>/nes_{raw,norm}/ (copied back once at the end)
    figure  OUT/{mice,ants}/fig_data.npz -> OUT/fig_event_traces.{png,pdf}

Outputs: dataset/<..>/eci/sae/<sae>/readout_norm.npz, results/vision/eci_readout_norm/
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import rollout_c1 as rc  # noqa: E402
from src.eci.domain import get_domain  # noqa: E402
from src.eci.foreground import obs_rows  # noqa: E402
from src.eci.readout_norm import QUANTILE, apply_norm, fit_norm, load_norm, save_norm  # noqa: E402

OUT = REPO / 'results/vision/eci_readout_norm'
CHUNK = 50_000
EXTRA_NEURONS = {'mice': [180]}  # the bag neuron (C1 mice), reported next to the align.json picks
PRIMARY_LABEL = {'mice': 'nose_nose', 'ants': 'groom_any'}
FIG_NEURONS = {  # (label, pick kind) in align.json: the two halves must agree
    'mice': [('nose_nose', 'auroc'), ('nose_nose', 'top1')],
    'ants': [('groom_any', 'auroc'), ('groom_yellow', 'top1')]}
FIG_SECONDS = 120
FIG_RULE = ('among the annotated videos, sorted by their primary-label frame rate (ties by observation id), the one at '
            'index (n - 1) // 2 (the lower median); in it, the first non-overlapping 2-min block from the video start '
            '(frames [600 k, 600 k + 600) at 5 fps) with >= 1 annotated-positive frame')
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def train_rows(d):
    """Sorted annotations.csv rows of the SAE's training frames (every frame of the 1 fps token store)."""
    store = REPO / rc.STORE[d]
    rows, nfg = [], []
    for sd in sorted(p for p in (store / 'shards').iterdir() if p.is_dir() and not p.name.endswith('.tmp')):
        z = np.load(sd / 'frames.npz')
        rows.append(z['rows'].astype(np.int64))
        nfg.append(z['n_fg'].astype(np.int64))
    rows, nfg = np.concatenate(rows), np.concatenate(nfg)
    o = np.argsort(rows)
    rows, nfg = rows[o], nfg[o]
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError(f'{store}: duplicate frame rows')
    return rows, nfg


def theta_of(d):
    import torch
    ck = torch.load(rc.sae_dir(d) / 'sae.pt', map_location='cpu', weights_only=False)
    return float(ck['state_dict']['threshold'])


def align_picks(d):
    R = json.loads((rc.OUT / d / 'align.json').read_text())['models'][rc.sae_name(d)]
    return R


def fig_neurons(d):
    R = align_picks(d)
    out = []
    for b, kind in FIG_NEURONS[d]:
        js = R[b][f'cf_neurons_{kind}']
        if len(set(js)) != 1:
            raise SystemExit(f'{d} {b} {kind}: the two halves pick different neurons {js}')
        out.append((int(js[0]), b, kind))
    return out


def key_neurons(d):
    if d not in PRIMARY_LABEL:
        return []
    R = align_picks(d)
    js = {int(j) for r in R.values() for k in ('cf_neurons_auroc', 'cf_neurons_top1') for j in r[k]}
    return sorted(js | set(EXTRA_NEURONS.get(d, [])))


def labels_5fps(d):
    """All labelled 5 fps rows -> DataFrame(row, obs, labels...), labels, valid masks (as rollout_c1.labelled_fps1)."""
    import spatial_sae_pilot as ssp
    from sae_levers import LABELS
    dom = get_domain(d)
    lab, _ = ssp.labelled_frames(d)
    labels = list(LABELS[d])
    valid = {b: np.ones(len(lab), bool) for b in labels}
    if d == 'ants':
        v = pd.read_csv(dom.ann_path, usecols=['Y_YOL'])['Y_YOL'].values[lab['row'].values].astype(float)
        valid['onlid_yellow'] = ~np.isnan(v)
        lab['onlid_yellow'] = np.nan_to_num(v) > 0
    return lab, labels, valid


# ---------------------------------------------------------------------------------------------- invariance: AUROC
def auroc_check(d, raw, q, ncpu):
    import torch
    from sae_levers import score_model
    torch.set_num_threads(ncpu)
    dev = torch.device('cpu')
    lab, labels, valid = rc.labelled_fps1(d)
    unit = lab.obs.str.rsplit('_', n=2).str[0].values if d == 'mice' else lab.obs.values
    units = sorted(set(unit))
    half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
    rows = lab['row'].values
    nf = np.load(rc.codes_dir(d) / 'n_fg.npy', mmap_mode='r')[rows].astype(np.float64)
    rank = np.argsort(np.argsort(nf + np.random.default_rng(0).random(len(nf)) * 1e-3))
    decile = np.minimum(rank * 10 // len(nf), 9)
    Yt = torch.from_numpy(lab[labels].values.astype(bool))
    X = np.ascontiguousarray(raw[rows])
    _, A = apply_norm(X, q)
    keys = ('insample_auroc', 'insample_ap', 'insample_top1_honest', 'cf_auroc', 'cf_ap', 'cf_top1', 'cf_neurons_auroc',
            'cf_neurons_top1')
    out, arrs = {'n_frames': int(len(lab)), 'n_videos': int(lab.obs.nunique())}, {}
    for name, M in (('raw', X), ('norm', A)):
        t0 = time.time()
        res, arrays = score_model(torch.from_numpy(M), Yt, labels, valid, half, decile, dev)
        out[name] = {b: {k: ([e['auc']['neuron'] for e in r['dirs']] if k == 'cf_neurons_auroc' else
                             [e['top1']['neuron'] for e in r['dirs']] if k == 'cf_neurons_top1' else r[k])
                         for k in keys} for b, r in res.items()}
        arrs[name] = arrays
        log(f'{d} AUROC {name} ({time.time() - t0:.0f}s): ' + json.dumps(
            {b: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for b, r in out[name].items()}))
    out['per_neuron_max_abs_diff'] = {}
    for b in labels:
        dd = {}
        for qk in ('auc', 'ap'):
            a, c = arrs['raw'][f'{b}|all|{qk}'], arrs['norm'][f'{b}|all|{qk}']
            ok = np.isfinite(a) & np.isfinite(c)
            dd[qk] = float(np.abs(a[ok] - c[ok]).max())
            dd[f'{qk}_argmax_neuron_raw_norm'] = [int(np.nanargmax(a)), int(np.nanargmax(c))]
        out['per_neuron_max_abs_diff'][b] = dd
    rj = json.loads((rc.OUT / d / 'align.json').read_text())['models'][rc.sae_name(d)]
    out['rollout_align_json'] = {b: {k: rj[b][k] for k in keys} for b in labels}
    return out


# ---------------------------------------------------------------------------------------------- invariance: NES
def nes_runs(d, loc, ncpu):
    sae = rc.sae_name(d)
    common = ['--domain', d, '--sae', sae, '--prefixes', '512,1024', '--nuisance', 'nfg', '--select', 'p',
              '--primary-only']
    sset = ['--analysis-set', 'pairs', '--day'] if d == 'ants' else []
    env = dict(os.environ, OMP_NUM_THREADS=str(ncpu), OPENBLAS_NUM_THREADS=str(ncpu), MKL_NUM_THREADS=str(ncpu))
    times = {}
    for kind in ('raw', 'norm'):
        root = ['--codes-root', str(loc / kind), '--out-root', str(loc / f'nes_{kind}')]
        for script, extra in (('run_nes.py', ['--primary-pooling', 'max', '--poolings', 'max']),
                              ('run_nes_bouts.py', ['--compare-pooling', 'max', '--n-shuffles', '0'])):
            cmd = [sys.executable, '-u', str(REPO / 'scripts/eci' / script)] + common + sset + root + extra
            log(' '.join(cmd))
            t0 = time.time()
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            (loc / f'nes_{kind}_{script}.log').write_text(r.stdout + '\n' + r.stderr)
            if r.returncode:
                print(r.stdout[-3000:], r.stderr[-3000:])
                raise SystemExit(f'{script} on {kind} codes failed')
            times[f'{kind}/{script}'] = round(time.time() - t0)
            log(f'  done in {times[f"{kind}/{script}"]}s')
    return times


def compare_sel(d, loc):
    sae = rc.sae_name(d)
    sub = 'pairs' if d == 'ants' else ''
    sets = {'rollout': get_domain(d, 'pairs' if d == 'ants' else 'core').nes_root / sae / sub,
            'raw': loc / 'nes_raw' / sae / sub, 'norm': loc / 'nes_norm' / sae / sub}
    S = {k: rc.selections(v)[0] for k, v in sets.items()}
    out = {'dirs': {k: str(v) for k, v in sets.items()}, 'outcomes': {}}
    for oc in ('activation', 'bout_rate'):
        if oc not in S['norm']:
            continue
        cells, n_same, n_same_roll, dl = [], 0, 0, []
        for aid in sorted(S['norm'][oc]):
            for pf in sorted(S['norm'][oc][aid]):
                g = {k: S[k].get(oc, {}).get(aid, {}).get(pf, []) for k in S}
                nr = {k: [r['neuron'] for r in v] for k, v in g.items()}
                same = nr['raw'] == nr['norm']
                n_same += same
                n_same_roll += nr['rollout'] == nr['raw']
                for a, b in zip(g['raw'], g['norm']):
                    if a['neuron'] == b['neuron']:
                        dl.append(abs(np.log10(max(a['p'], 1e-300)) - np.log10(max(b['p'], 1e-300))))
                cells.append({'analysis': aid, 'prefix': pf, 'same_neurons_raw_norm': bool(same),
                              'same_neurons_rollout_raw': nr['rollout'] == nr['raw'],
                              **{k: [(r['neuron'], r['p'], r['tau']) for r in v] for k, v in g.items()}})
        out['outcomes'][oc] = {'n_cells': len(cells), 'n_same_neurons_raw_vs_norm': int(n_same),
                               'n_same_neurons_rollout_vs_raw': int(n_same_roll),
                               'max_abs_log10p_diff_matched_rounds': float(max(dl)) if dl else None,
                               'median_abs_log10p_diff_matched_rounds': float(np.median(dl)) if dl else None,
                               'cells': cells}
        log(f'{d} NES {oc}: {len(cells)} analysis x prefix cells, same neurons raw vs norm {n_same}, rollout vs raw '
            f'{n_same_roll}, max |dlog10 p| {out["outcomes"][oc]["max_abs_log10p_diff_matched_rounds"]}')
    return out


# ---------------------------------------------------------------------------------------------- run
def cmd_run(args):
    d = args.domain
    ncpu = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    loc = rc.local_dir()
    sae = rc.sae_name(d)
    cdir = rc.codes_dir(d)
    R = {'domain': d, 'sae': sae, 'definition': {
        'event': 'codes_max > 0 (frame read-out = max over the kept foreground patches of the threshold-mode codes)',
        'a_norm': f'min(1, codes_max / q_j), q_j = {QUANTILE} quantile of neuron j positive read-outs over the SAE '
                  'training frames (all frames of the 1 fps token store); fallback q_j = theta for neurons that never '
                  'fire there'}}
    # ---- stage
    t0 = time.time()
    for kind in ('raw', 'norm'):
        (loc / kind / sae).mkdir(parents=True, exist_ok=True)
    for f in ('codes_max.npy', 'n_fg.npy'):
        shutil.copyfile(cdir / f, loc / 'raw' / sae / f)
    shutil.copyfile(cdir / 'n_fg.npy', loc / 'norm' / sae / 'n_fg.npy')
    (loc / 'raw' / sae / 'DONE').touch()
    R['stage_s'] = round(time.time() - t0)
    log(f'staged codes_max + n_fg in {R["stage_s"]}s')
    raw = np.load(loc / 'raw' / sae / 'codes_max.npy', mmap_mode='r')
    N, m = raw.shape
    # ---- fit on the training frames
    rows, nfg_store = train_rows(d)
    ref = np.sort(np.concatenate(list(rc_fps1(d).values())))
    theta = theta_of(d)
    t0 = time.time()
    Xtr = raw[rows]
    q, rate, n_pos, fb = fit_norm(Xtr, theta)
    del Xtr
    R['fit'] = {'n_train_frames': int(len(rows)), 'n_train_frames_with_fg_tokens': int((nfg_store > 0).sum()),
                'train_rows_equal_fps1_rows_seed0': bool(np.array_equal(rows, ref)), 'theta': theta,
                'n_fallback_never_fire_on_train': int(fb.sum()), 'fit_s': round(time.time() - t0),
                'q_median': float(np.median(q[~fb])), 'q_p10_p90': [float(x) for x in np.percentile(q[~fb], [10, 90])],
                'theta_over_q_median': float(np.median(theta / q[~fb])),
                'theta_over_q_p10_p90': [float(x) for x in np.percentile(theta / q[~fb], [10, 90])],
                'train_rate_median': float(np.median(rate)),
                'train_rate_p10_p90': [float(x) for x in np.percentile(rate, [10, 90])]}
    path = save_norm(rc.sae_dir(d), q, rate, n_pos, fb, quantile=QUANTILE, sae=sae, domain=d,
                     codes=rel(cdir / 'codes_max.npy'), train_store=rc.STORE[d],
                     train_frames='rows of every shard frames.npz of the 1 fps token store (stride 5, seed 0 offsets)',
                     n_train_frames=int(len(rows)), theta=theta, created=time.strftime('%Y-%m-%d %H:%M'),
                     script='scripts/eci/readout_norm.py')
    chk = load_norm(rc.sae_dir(d))
    assert np.array_equal(chk['q'], q.astype(np.float32))
    R['npz'] = {'path': rel(path), 'bytes': path.stat().st_size}
    log(f'fit: {json.dumps(R["fit"])} -> {path}')
    # ---- one streamed pass: normalised codes (local), event / clip counts, key-neuron columns
    keys = key_neurons(d)
    is_tr = np.zeros(N, bool)
    is_tr[rows] = True
    n_ev, n_clip = np.zeros(m, np.int64), np.zeros(m, np.int64)
    n_ev_tr, n_clip_tr = np.zeros(m, np.int64), np.zeros(m, np.int64)
    n_mismatch = 0
    keycols = np.zeros((N, len(keys)), np.float32)
    out = np.lib.format.open_memmap(loc / 'norm' / sae / 'codes_max.npy', 'w+', np.float16, (N, m))
    t0 = time.time()
    for a in range(0, N, CHUNK):
        b = min(N, a + CHUNK)
        X = np.asarray(raw[a:b])
        ev, A = apply_norm(X, q)
        n_mismatch += int((ev != (A > 0)).sum())
        c1 = A == 1
        n_ev += ev.sum(0)
        n_clip += c1.sum(0)
        t = is_tr[a:b]
        n_ev_tr += ev[t].sum(0)
        n_clip_tr += c1[t].sum(0)
        if keys:
            keycols[a:b] = X[:, keys].astype(np.float32)
        out[a:b] = A
    out.flush()
    del out
    (loc / 'norm' / sae / 'DONE').touch()
    R['stream_s'] = round(time.time() - t0)
    gb = N * m * 2 / 1e9
    R['storage'] = {'codes_max_norm_float16_gb': gb, 'kept_on_nfs': False,
                    'note': 'not written next to the codes (duplicates codes_max); a_norm = apply_norm(codes_max, q) '
                            'on the fly from readout_norm.npz'}
    live = n_ev > 0
    fr_all, fr_ev = n_clip / N, np.where(live, n_clip / np.maximum(n_ev, 1), np.nan)
    ltr = n_ev_tr > 0
    R['interpretability'] = {
        'n_frames': int(N), 'n_neurons': int(m), 'n_neurons_firing_somewhere': int(live.sum()),
        'event_vs_anorm_pos_mismatches': n_mismatch,
        'clipped_frac_of_all_frames': {'median': float(np.median(fr_all[live])),
                                       'p90': float(np.percentile(fr_all[live], 90)), 'max': float(fr_all.max())},
        'clipped_frac_of_event_frames': {'median': float(np.nanmedian(fr_ev[live])),
                                         'p90': float(np.nanpercentile(fr_ev[live], 90))},
        'clipped_frac_of_event_frames_train_only': {
            'median': float(np.median(n_clip_tr[ltr] / n_ev_tr[ltr])),
            'p90': float(np.percentile(n_clip_tr[ltr] / n_ev_tr[ltr], 90))},
        'event_rate_all_frames': {'median': float(np.median(n_ev / N)), 'p90': float(np.percentile(n_ev / N, 90))}}
    thr_f = get_domain(d).nes_root / sae / ('pairs' if d == 'ants' else '') / 'maxpool_bouts' / 'thresholds.csv'
    if thr_f.exists():
        T = pd.read_csv(thr_f)
        R['interpretability']['bout_threshold_q095_zero'] = {
            'n_neurons': int((T['thr_q0.95'] == 0).sum()), 'of': int(len(T)), 'file': rel(thr_f),
            'note': 'NES bout / latency "above" = codes_max > the pooled q0.95 threshold; equals EVENT only where it is 0'}
    log(f'stream pass {R["stream_s"]}s: {json.dumps(R["interpretability"])}')
    np.savez(loc / 'per_neuron.npz', q=q, rate=rate, n_ev=n_ev, n_clip=n_clip, n_ev_tr=n_ev_tr, n_clip_tr=n_clip_tr)
    # ---- key neurons on annotated frames (all labelled 5 fps rows)
    if keys:
        lab, labels, valid = labels_5fps(d)
        kn = {}
        for i, j in enumerate(keys):
            x = keycols[lab['row'].values, i]
            ev = x > 0
            A = np.minimum(1.0, x / q[j])
            kn[j] = {'q': float(q[j]), 'train_rate': float(rate[j])}
            for b in labels:
                v = valid[b]
                y = lab[b].values.astype(bool)
                P, Nn = v & y, v & ~y
                kn[j][b] = {'n_pos': int(P.sum()), 'n_neg': int(Nn.sum()),
                            'mean_anorm_pos': float(A[P].mean()), 'mean_anorm_neg': float(A[Nn].mean()),
                            'event_rate_pos': float(ev[P].mean()), 'event_rate_neg': float(ev[Nn].mean()),
                            'mean_anorm_pos_given_event': float(A[P & ev].mean()) if (P & ev).any() else None,
                            'mean_anorm_neg_given_event': float(A[Nn & ev].mean()) if (Nn & ev).any() else None}
        R['key_neurons'] = kn
        R['key_neurons_frames'] = 'all labelled 5 fps frames (annotations.csv rows with labels)'
        log(f'key neurons: {json.dumps(kn)}')
        # ---- figure data (fixed rule)
        fn = fig_neurons(d)
        b0 = PRIMARY_LABEL[d]
        rate_v = lab.groupby('obs')[b0].mean().reset_index().sort_values([b0, 'obs']).reset_index(drop=True)
        k = (len(rate_v) - 1) // 2
        obs = rate_v.loc[k, 'obs']
        lo, hi = obs_rows(get_domain(d).ann_path)[obs]
        lv = lab[lab.obs == obs].set_index('row')
        L = FIG_SECONDS * 5
        start = None
        for s in range(0, hi - lo - L + 1, L):
            r = np.arange(lo + s, lo + s + L)
            if lv[b0].reindex(r, fill_value=False).values.any():
                start = s
                break
        if start is None:
            raise SystemExit(f'{d} {obs}: no 2-min block with a positive frame')
        r = np.arange(lo + start, lo + start + L)
        fd = {'rows': r, 'start_frame': start, 'fps': 5.0, 'obs': np.array(obs), 'rule': np.array(FIG_RULE),
              'video_rate': float(rate_v.loc[k, b0]), 'video_rank': k, 'n_videos': len(rate_v),
              'labels': np.array(labels), 'primary_label': np.array(b0),
              'neurons': np.array([j for j, _, _ in fn]), 'neuron_desc': np.array([f'{b} {kd}' for _, b, kd in fn]),
              'q': np.array([q[j] for j, _, _ in fn])}
        for b in labels:
            fd[f'y_{b}'] = lv[b].reindex(r, fill_value=False).values.astype(bool)
        for j, _, _ in fn:
            x = keycols[r, keys.index(j)]
            fd[f'raw_{j}'] = x
            fd[f'anorm_{j}'] = np.minimum(1.0, x / q[j])
        (OUT / d).mkdir(parents=True, exist_ok=True)
        np.savez(OUT / d / 'fig_data.npz', **fd)
        R['figure'] = {'obs': obs, 'start_frame': int(start), 'video_rate': fd['video_rate'], 'rank': int(k),
                       'n_videos': int(len(rate_v)), 'neurons': [[j, b, kd] for j, b, kd in fn], 'rule': FIG_RULE,
                       'segment_primary_rate': float(fd[f'y_{b0}'].mean()),
                       'segment_event_rate': {int(j): float((fd[f'raw_{j}'] > 0).mean()) for j, _, _ in fn}}
        log(f'figure: {json.dumps(R["figure"])}')
        R['auroc'] = auroc_check(d, raw, q, ncpu)
    # ---- NES raw vs norm
    R['nes_time_s'] = nes_runs(d, loc, ncpu)
    R['nes'] = compare_sel(d, loc)
    (OUT / d).mkdir(parents=True, exist_ok=True)
    for kind in ('raw', 'norm'):
        dst = OUT / d / f'nes_{kind}'
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(loc / f'nes_{kind}', dst)
        shutil.copyfile(loc / f'nes_{kind}_run_nes.py.log', OUT / d / f'nes_{kind}_run_nes.log')
        shutil.copyfile(loc / f'nes_{kind}_run_nes_bouts.py.log', OUT / d / f'nes_{kind}_run_nes_bouts.log')
    shutil.copyfile(loc / 'per_neuron.npz', OUT / d / 'per_neuron.npz')
    (OUT / d / 'readout_norm.json').write_text(json.dumps(R, indent=1, default=float))
    log(f'-> {OUT / d}')


def rc_fps1(d):
    from fg_extract_train import fps1_rows
    return fps1_rows(obs_rows(get_domain(d).ann_path), 5, 0)


# ---------------------------------------------------------------------------------------------- figure
def cmd_figure(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Nimbus Sans',
                                                                        'DejaVu Sans'],
                         'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
                         'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    HUE, GREY, INK = '#0072B2', '#D9D9D9', '#333333'  # Okabe-Ito blue; neutral grey annotation band
    NAMES = {'nose_nose': 'nose-to-nose contact (annotated)', 'groom_any': 'grooming (annotated)'}
    DOM = {'mice': 'Mice (social odour assay)', 'ants': 'Ants (social immunity)'}
    data = [(d, np.load(OUT / d / 'fig_data.npz')) for d in ('mice', 'ants')]
    n_rows = sum(len(f['neurons']) for _, f in data)
    fig = plt.figure(figsize=(7.0, 1.05 * n_rows + 0.5))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.55, left=0.075, right=0.985, top=0.93, bottom=0.07)
    i = 0
    for d, f in data:
        b0 = str(f['primary_label'])
        y = f[f'y_{b0}']
        t = np.arange(len(y)) / float(f['fps'])
        for n_i, (j, desc) in enumerate(zip(f['neurons'], f['neuron_desc'])):
            ax = fig.add_subplot(gs[i])
            # annotation band (grey) as spans
            dy = np.diff(np.r_[0, y.astype(int), 0])
            for s, e in zip(np.flatnonzero(dy == 1), np.flatnonzero(dy == -1)):
                ax.axvspan(t[s] - 0.1, t[e - 1] + 0.1, ymin=0.16, ymax=1.0, color=GREY, lw=0, zorder=0)
            a = f[f'anorm_{j}']
            ax.plot(t, a, color=HUE, lw=0.7, zorder=2)
            ev = np.flatnonzero(f[f'raw_{j}'] > 0)
            ax.vlines(t[ev], -0.17, -0.07, color=HUE, lw=0.45, zorder=2)
            ax.set_ylim(-0.2, 1.05)
            ax.set_xlim(0, t[-1] + 0.2)
            ax.set_yticks([0, 0.5, 1])
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines['left'].set_bounds(0, 1)
            ax.tick_params(colors=INK)
            ax.set_ylabel('$a_{norm}$', color=INK, labelpad=1)
            kind = 'best AUROC' if desc.endswith('auroc') else 'best top-1% precision'
            lbl = desc.rsplit(' ', 1)[0].replace('_', '-')
            ax.text(0.0, 1.10, f'{DOM[d] if n_i == 0 else ""}', transform=ax.transAxes, fontsize=7.5, fontweight='bold',
                    color=INK, va='bottom')
            ax.text(1.0, 1.10, f'neuron {j}  ({kind} for {lbl})', transform=ax.transAxes, ha='right', fontsize=6.5,
                    color=INK, va='bottom')
            if n_i == 0:
                start_s = int(f['start_frame']) / float(f['fps'])
                ax.text(0.0, 1.10, '', transform=ax.transAxes)
                ax.text(0.5, 1.10, f'video {str(f["obs"])}, {start_s / 60:.0f}-{start_s / 60 + 2:.0f} min',
                        transform=ax.transAxes, ha='center', fontsize=6.5, color='#666666', va='bottom')
            if i == n_rows - 1:
                ax.set_xlabel('time in segment (s)', color=INK)
            else:
                ax.set_xticklabels([])
            if n_i == len(f['neurons']) - 1:
                from matplotlib.patches import Patch
                from matplotlib.lines import Line2D
                h = [Line2D([], [], color=HUE, lw=0.8, label='$a_{norm}$ = min(1, read-out / $q_{99}$)'),
                     Line2D([], [], color=HUE, lw=0, marker='|', markersize=4, markeredgewidth=0.6,
                            label='event (read-out > 0)'),
                     Patch(color=GREY, lw=0, label=NAMES.get(b0, b0))]
                ax.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, -0.18 if i < n_rows - 1 else -0.45),
                          ncol=3, frameon=False, fontsize=6.3, handlelength=1.4, columnspacing=1.4, borderaxespad=0)
            i += 1
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig_event_traces.{ext}', dpi=300)
    log(f'-> {OUT}/fig_event_traces.{{png,pdf}}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('run', 'figure'))
    p.add_argument('--domain', default='mice', choices=rc.DOMAINS)
    args = p.parse_args()
    {'run': cmd_run, 'figure': cmd_figure}[args.step](args)


if __name__ == '__main__':
    main()
