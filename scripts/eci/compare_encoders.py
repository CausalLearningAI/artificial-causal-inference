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
  bylabel   (not in the default --parts) the same as 'latents' for each label column separately (mice Y_nn / Y_np /
            Y_nt; codes_max): best single latent, latents with AUC > 0.65, cross-fitted readout.
  poc       ants only, the v2 grooming proof of concept: grooming readout (logistic on codes_max of the v2 frames,
            5-fold grouped by video, C = 0.1) -> frame-level and video-level Welch p for treatment; the single
            latent most correlated with grooming (frame corr) and its frame / video p; NES with frames as units
            and no correction (paper protocol) and with videos as units (Bonferroni).
  position  (GPU recommended; not in the default --parts) patch-position share on held-out foreground tokens with
            each SAE's own input (raw / background-subtracted / motion): share of the input-token variance and the
            variance-weighted eta^2 of the latents on the patch position (chance-corrected).
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
        res[f'n_latents_auc_gt065_{pool}'] = int((dfree > 0.65).sum())
        if pool == 'max':
            sub = np.arange(0, len(rows), stride)
            _, res['readout_auc_max'] = readout(X[sub], y[rows][sub], pd.factorize(a['observation_id'].values[rows][sub])[0])
            res['readout_stride'] = stride
    return res


def by_label(codes_dir, dom, domain, stride):
    """'latents' (codes_max) per label column: each label's own labelled frames, positive = label > 0."""
    cols = LABELS[domain]
    a = pd.read_csv(dom.ann_path, usecols=['observation_id'] + cols)
    Xall = np.load(codes_dir / 'codes_max.npy', mmap_mode='r')
    res = {}
    for c in cols:
        rows = np.nonzero(a[c].notna().values)[0]
        y = a[c].values[rows] > 0
        X = np.asarray(Xall[rows], dtype=np.float32)
        auc = auc_cols(X, y)
        dfree = np.maximum(auc, 1 - auc)
        top = np.argsort(-dfree)[:5]
        sub = np.arange(0, len(rows), stride)
        _, ro = readout(X[sub], y[sub], pd.factorize(a['observation_id'].values[rows][sub])[0])
        res[c] = {'n_frames': int(len(rows)), 'n_pos': int(y.sum()),
                  'best_latent_max': [{'latent': int(j), 'auc': float(auc[j])} for j in top],
                  'n_latents_auc_gt065_max': int((dfree > 0.65).sum()), 'readout_auc_max': ro, 'readout_stride': stride}
    return res


def position(sae_dir, n_tokens=2_000_000, seed=0, chunk=65536):
    """Patch-position share on held-out foreground tokens (the SAE's own validation units, up to n_tokens tokens,
    seeded uniform sample), with the SAE's own input (raw / background-subtracted / [token, change]):
      input_token_share   share of the normalized SAE-input variance (summed over dims) explained by the position
                          (motion SAEs: also per half)
      latent_share        per latent eta^2 of its activation on the position (threshold inference), averaged with
                          weights = the latent's activation variance (the pilot's 'eta_pos weighted'), median, and
                          the fraction of latents with eta^2 > 0.3"""
    import torch
    from src.eci.foreground import RULES, FgTokenStore, obs_rows, subtract_background_np
    from src.eci.sae import load_sae
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sae, norm, ck = load_sae(sae_dir / 'sae.pt', dev)
    m = json.loads((sae_dir / 'metrics.json').read_text())
    args = m['args']
    dom = get_domain(args.get('domain', 'mice'))
    _, is_val_row = dom.val_split(args['val_from'], args.get('split_seed', 0))
    tdir = Path(args['tokens_dir'])
    store = FgTokenStore(tdir if tdir.is_absolute() else REPO / tdir)
    idx = [np.nonzero(is_val_row[store.row(s)])[0] for s in range(len(store.dirs))]
    n_all = sum(len(i) for i in idx)
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(n_all, min(n_tokens, n_all), replace=False))
    offs = np.cumsum([0] + [len(i) for i in idx])
    D = int(ck.get('motion_delta', 0) or 0)
    xs, ps = [], []
    for s in range(len(store.dirs)):
        k = keep[(keep >= offs[s]) & (keep < offs[s + 1])] - offs[s]
        if not len(k):
            continue
        ii = idx[s][k]
        t = np.asarray(store.tokens(s)[ii])
        if ck.get('bg_sub', False):
            bg_dir = args.get('bg_dir') or str(dom.eci_dir / ('fg448al' if ck.get('align', 'none') != 'none' else 'fg448')
                                                / 'background')
            t = subtract_background_np(t, store.row(s)[ii], store.pos(s)[ii], bg_dir, obs_rows(dom.ann_path),
                                       RULES[ck.get('fg_rule', 'fg448')])
        if D > 0:
            t = np.concatenate([t, (np.asarray(store.tokens(s)[ii]).astype(np.float32) - store.prev(s)[ii]).astype(np.float16)], 1)
        xs.append(t)
        ps.append(store.pos(s)[ii].astype(np.int64))
    X, P = np.concatenate(xs), torch.from_numpy(np.concatenate(ps)).to(dev)
    N, L = len(X), sae.n_latents
    S = torch.zeros(1024, L, dtype=torch.float64, device=dev)
    c = torch.zeros(1024, dtype=torch.float64, device=dev)
    s1 = torch.zeros(L, dtype=torch.float64, device=dev)
    s2 = torch.zeros_like(s1)
    d_in = X.shape[1]
    T = torch.zeros(1024, d_in, dtype=torch.float64, device=dev)
    t1 = torch.zeros(d_in, dtype=torch.float64, device=dev)
    t2 = torch.zeros_like(t1)
    with torch.no_grad():
        for a in range(0, N, chunk):
            x = norm(torch.from_numpy(X[a:a + chunk]).to(dev))
            z = sae.encode(x, mode='threshold').double()
            lab = P[a:a + chunk]
            S.index_add_(0, lab, z)
            c.index_add_(0, lab, torch.ones_like(lab, dtype=torch.float64))
            s1 += z.sum(0); s2 += z.pow(2).sum(0)
            xd = x.double()
            T.index_add_(0, lab, xd)
            t1 += xd.sum(0); t2 += xd.pow(2).sum(0)
    G = int((c > 0).sum())

    def share(Sg, a1, a2):
        ss = a2 - a1.pow(2) / N
        e = ((Sg[c > 0].pow(2) / c[c > 0, None]).sum(0) - a1.pow(2) / N)
        return e, ss

    corr = lambda e: e - (G - 1) / (N - G) * (1 - e)
    eb, ssb = share(T, t1, t2)
    res = {'n_tokens': int(N), 'n_positions': G, 'motion_delta': D, 'bg_sub': bool(ck.get('bg_sub', False)),
           'input_token_share': float(corr(eb.sum() / ssb.sum()))}
    if D > 0:
        h = d_in // 2
        res['input_token_share_static'] = float(corr(eb[:h].sum() / ssb[:h].sum()))
        res['input_token_share_change'] = float(corr(eb[h:].sum() / ssb[h:].sum()))
    el, ssl = share(S, s1, s2)
    eta = corr(el / ssl.clamp_min(1e-12)).clamp_min(0)
    w = ssl / ssl.sum()
    eta_np = eta.cpu().numpy()
    res['latent_share'] = {'weighted': float((w * eta).sum()), 'median': float(np.median(eta_np[ssl.cpu().numpy() > 0])),
                           'frac_gt03': float((eta_np > 0.3).mean())}
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
    p.add_argument('--position-tokens', type=int, default=2_000_000, help="'position' part: held-out tokens used")
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
        if 'bylabel' in parts:
            r['bylabel'] = by_label(codes, dom, args.domain, args.stride)
        if 'position' in parts:
            r['position'] = position(dom.eci_dir / 'sae' / sae, args.position_tokens)
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
