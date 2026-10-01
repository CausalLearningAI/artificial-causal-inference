"""
Side-by-side comparison of foreground SAEs built on different encoders (DINOv2 vs DINOv3), ECI.

Parts (all CPU, all from files already on disk):
  health    metrics.json (held-out FVE / L0 / dead latents per Matryoshka prefix, threshold inference) and the
            per-latent frame firing rate over ALL frames (codes_max > 0): never-firing latents, rate quantiles,
            dense (> 50% of frames) and rare (< 0.1%) latents.
  latents   best single-latent AUC (direction-free: max(AUC, 1 - AUC)) on the labelled frames, codes_max and
            codes_mean (mice: social contact = any of Y_nn / Y_np / Y_nt, labelled videos only; ants: grooming =
            Y_B2F or Y_Y2F), and a cross-fitted logistic readout over all codes_max (5-fold grouped by video,
            C = 0.1, on every --stride-th labelled frame).
  poc       ants only, the v2 grooming proof of concept: grooming readout (logistic on codes_max of the v2 frames,
            5-fold grouped by video, C = 0.1) -> frame-level and video-level Welch p for treatment; the single
            latent most correlated with grooming (frame corr) and its frame / video p; NES with frames as units
            and no correction (paper protocol) and with videos as units (Bonferroni).
  nes       primary NES (codes_max, per-video mean, t, Bonferroni, full window) / bouts (bout rate, q 0.95, gap 0)
            results of the runs: selected neurons per analysis and prefix (tau, p), the
            label-shuffle null (number selected per shuffle) and, for ants pairs, the recording-day confound flag.

Output: <out-dir>/<domain>/compare.json (+ printed tables).

Usage:
    python scripts/eci/compare_encoders.py --domain mice --sae matryoshka_btk_1024_k16_fg448_s0 \
        --sae matryoshka_btk_1024_k16_fg512v3_s0 --nes-root results/vision/mice/eci/nes
    python scripts/eci/compare_encoders.py --domain ants --sae matryoshka_btk_1024_k16_antsfg_s0 \
        --sae matryoshka_btk_1024_k16_antsfgv3_s0 --nes-sub pairs
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.domain import DOMAINS, get_domain  # noqa: E402

LABELS = {'mice': ['Y_nn', 'Y_np', 'Y_nt'], 'ants': ['Y_B2F', 'Y_Y2F']}


def health(sae_dir, codes_dir, chunk=200_000):
    m = json.loads((sae_dir / 'metrics.json').read_text())
    out = {'n_train_tokens': m['n_train_tokens'], 'n_val_tokens': m['n_val_tokens'], 'encoder': m.get('encoder', 'dinov2_base'),
           'prefixes': {k: {f: v[f] for f in ('fve', 'l0_per_token', 'n_dead')} for k, v in m['val_threshold']['prefixes'].items()}}
    X = np.load(codes_dir / 'codes_max.npy', mmap_mode='r')
    fire = np.zeros(X.shape[1])
    for a in range(0, X.shape[0], chunk):
        fire += (np.asarray(X[a:a + chunk]) > 0).sum(0)
    rate = fire / X.shape[0]
    nfg = np.load(codes_dir / 'n_fg.npy', mmap_mode='r')
    out['frames'] = {'n': int(X.shape[0]), 'n_fg_median': float(np.median(nfg)), 'frames_without_fg': int((np.asarray(nfg) == 0).sum()),
                     'never_firing': int((fire == 0).sum()), 'dense_gt50pct': int((rate > 0.5).sum()),
                     'rare_lt0.1pct': int((rate < 1e-3).sum()),
                     'rate_quantiles_p10_p50_p90_max': [float(np.quantile(rate, q)) for q in (0.1, 0.5, 0.9)] + [float(rate.max())],
                     'never_firing_by_prefix': {str(p): int((fire[:p] == 0).sum()) for p in (128, 256, 512, 1024)}}
    return out


def label_frames(dom, domain):
    cols = LABELS[domain]
    a = pd.read_csv(dom.ann_path, usecols=['observation_id'] + cols + (['experiment', 'T'] if domain == 'ants' else []))
    ok = a[cols].notna().all(1).values
    y = (a[cols].fillna(0).sum(1).values > 0)
    return a, ok, y


def auc_cols(X, y):
    """AUC of every column of X for binary y (Mann-Whitney via ranks)."""
    n1, n0 = int(y.sum()), int((~y).sum())
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        r = stats.rankdata(X[:, j])
        out[j] = (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def readout(X, y, groups, C=0.1, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        pred[te] = LogisticRegression(max_iter=300, C=C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return pred, float(roc_auc_score(y, pred))


def latents(codes_dir, a, ok, y, stride):
    rows = np.nonzero(ok)[0]
    res = {'n_frames': int(len(rows)), 'n_pos': int(y[rows].sum())}
    for pool in ('max', 'mean'):
        X = np.asarray(np.load(codes_dir / f'codes_{pool}.npy', mmap_mode='r')[rows], dtype=np.float32)
        auc = auc_cols(X, y[rows])
        dfree = np.maximum(auc, 1 - auc)
        top = np.argsort(-dfree)[:5]
        res[f'best_latent_{pool}'] = [{'latent': int(j), 'auc': float(auc[j])} for j in top]
        if pool == 'max':
            sub = np.arange(0, len(rows), stride)
            _, res['readout_auc_max'] = readout(X[sub], y[rows][sub], pd.factorize(a['observation_id'].values[rows][sub])[0])
            res['readout_stride'] = stride
    return res


def welch(y, t):
    return float(stats.ttest_ind(y[t == 1], y[t == 0], equal_var=False).pvalue)


def poc(codes_dir, a, y):
    from src.eci.nes import neural_effect_search
    idx = np.nonzero(a['experiment'].values == 'v2')[0]
    assert idx[-1] - idx[0] + 1 == len(idx)
    X = np.asarray(np.load(codes_dir / 'codes_max.npy', mmap_mode='r')[idx[0]:idx[-1] + 1], dtype=np.float32)
    g = y[idx].astype(float)
    vid = a['observation_id'].values[idx]
    T = (a['T'].values[idx] == 2).astype(int)
    vpos = pd.factorize(vid)[0]
    nv = vpos.max() + 1
    Tv = np.array([T[vpos == i][0] for i in range(nv)])
    vmean = lambda s: np.bincount(vpos, weights=s, minlength=nv) / np.bincount(vpos)
    pred, auc = readout(X, g, vpos)
    res = {'n_frames': int(len(idx)), 'n_videos': int(nv), 'n_treated_videos': int(Tv.sum()),
           'annotated_grooming': {'frame_p': welch(g, T), 'video_p': welch(vmean(g), Tv),
                                  'control_rate': float(vmean(g)[Tv == 0].mean()), 'treated_rate': float(vmean(g)[Tv == 1].mean())},
           'readout': {'auc': auc, 'frame_p': welch(pred, T), 'video_p': welch(vmean(pred), Tv),
                       'control_mean': float(vmean(pred)[Tv == 0].mean()), 'treated_mean': float(vmean(pred)[Tv == 1].mean())}}
    Xc, gc = X - X.mean(0), g - g.mean()
    r = (Xc * gc[:, None]).sum(0) / (np.sqrt((Xc ** 2).sum(0) * (gc ** 2).sum()) + 1e-12)
    j = int(np.argmax(r))
    res['best_grooming_latent'] = {'latent': j, 'frame_corr': float(r[j]), 'frame_p': welch(X[:, j], T),
                                   'video_p': welch(vmean(X[:, j]), Tv)}
    for label, kw in (('frame_units_no_correction', dict(correction='none')),
                      ('video_units_bonferroni', dict(correction='bonferroni', groups=vid))):
        out = neural_effect_search(X.astype(np.float64), T, select='tau', max_rounds=3, **kw)
        res[f'nes_{label}'] = {'first_round_significant': int(out['first_round']['significant'].sum()),
                               'selected': [int(s) for s in out['selected']],
                               'selected_grooming_corr': [float(r[int(s)]) for s in out['selected']]}
    return res


def nes(run_dir, prefixes=(128, 1024)):
    """Primary selections + nulls of one NES run directory (and its maxpool_bouts/)."""
    res = {}
    for kind, sub in (('mean_activation', ''), ('bouts', 'maxpool_bouts')):
        d = run_dir / sub if sub else run_dir
        if not (d / 'summary.csv').exists():
            res[kind] = None
            continue
        s = pd.read_csv(d / 'summary.csv')
        fmt = 'p{}_max_mean_t_bonferroni_full' if kind == 'mean_activation' else 'p{}_bout_rate_q0.95_g0_t_bonferroni_full'
        prim = s[s['setting'].isin([fmt.format(p) for p in prefixes])]
        rows = prim[prim['neuron'].notna()]
        sel = {}
        for (aid, pref), grp in rows.groupby(['analysis_id', 'prefix']):
            sel.setdefault(aid, {})[int(pref)] = [{'neuron': int(r.neuron), 'tau': float(r.tau), 'p': float(r.p)}
                                                  for r in grp.sort_values('round').itertuples()]
        san = json.loads((d / 'sanity.json').read_text())
        nulls = {k: v for k, v in san.items() if k.startswith('permutation') or k == 'nulls'}
        res[kind] = {'selected': sel, 'nulls': nulls,
                     'confound': (s.drop_duplicates('analysis_id').set_index('analysis_id')['confound'].fillna('').to_dict()
                                  if 'confound' in s else {})}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', default='mice', choices=DOMAINS)
    p.add_argument('--sae', action='append', required=True, help='two or more SAE names (first = reference)')
    p.add_argument('--parts', default='health,latents,poc,nes')
    p.add_argument('--nes-root', default=None, help='default the domain NES root')
    p.add_argument('--nes-sub', default='', help="e.g. 'pairs' (ants analysis set)")
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--out-dir', default=str(REPO / 'results/vision/eci_encoders'))
    args = p.parse_args()
    dom = get_domain(args.domain)
    parts = args.parts.split(',')
    out = Path(args.out_dir) / args.domain
    out.mkdir(parents=True, exist_ok=True)
    f = out / 'compare.json'
    res = json.loads(f.read_text()) if f.exists() else {}
    a, ok, y = label_frames(dom, args.domain) if {'latents', 'poc'} & set(parts) else (None, None, None)
    nes_root = Path(args.nes_root) if args.nes_root else dom.nes_root
    for sae in args.sae:
        t0 = time.time()
        r = res.setdefault(sae, {})
        codes = dom.eci_dir / 'codes' / sae
        if 'health' in parts:
            r['health'] = health(dom.eci_dir / 'sae' / sae, codes)
        if 'latents' in parts:
            r['latents'] = latents(codes, a, ok, y, args.stride)
        if 'poc' in parts and args.domain == 'ants':
            r['poc'] = poc(codes, a, y)
        if 'nes' in parts:
            r['nes'] = nes(nes_root / sae / args.nes_sub if args.nes_sub else nes_root / sae)
        print(f'== {sae} ({time.time() - t0:.0f}s)')
        print(json.dumps({k: v for k, v in r.items() if k != 'nes'}, indent=1)[:6000], flush=True)
        f.write_text(json.dumps(res, indent=1))
    print(f'-> {f}')


if __name__ == '__main__':
    main()
