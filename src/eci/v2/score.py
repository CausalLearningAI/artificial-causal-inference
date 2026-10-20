"""
Behaviour alignment scoring of a code matrix (ECI v2). Labels are used ONLY here (selection / scoring), never in any
representation step.

R~ (normalised R-precision) of a neuron on a frame set: P = share of positives among its top N frames (N = number of
positives; ties at the cut shared pro rata = the expected value under random tie-breaking), R~ = (P - pi) / (1 - pi),
pi = base rate. AUROC = Mann-Whitney (ties averaged). AP = average precision. Definitions and code are those of the
rounds 1-4 scorer (scripts/eci/nnnt/nnnt.py half_stats / score_set), copied here unchanged so the numbers stay
comparable.

Cross-fitting: the units (mice pools, ants videos) are split into two halves; the neuron with the highest R~ (or
AUROC) on one half is scored on the other half; both directions are averaged; the reported value is the mean over
the random splits ('rnd0'..'rnd9', rng 1000 + s).
"""
import numpy as np
import torch

NB = 0x7C01  # fp16 bit patterns of non-negative finite values (0 .. 65504) + 1


def splits_of(units, n_splits=10):
    """{split name: half (n_units,) 0/1}: 'r4' (sorted units alternating) + n_splits random halves (rng 1000 + s)."""
    u = sorted(units)
    out = {'r4': np.array([i % 2 for i in range(len(u))])}
    for s in range(n_splits):
        h = np.zeros(len(u), int)
        h[np.random.default_rng(1000 + s).permutation(len(u))[:len(u) // 2]] = 1
        out[f'rnd{s}'] = h
    return u, out


def half_stats(X, Y, rows, block=512):
    """X (n, W) fp16 >= 0 cuda, Y (n, L) bool cuda, rows LongTensor -> per neuron x label R~, AUROC, AP; rate (W,) =
    share of frames > 0; base (L,)."""
    Yr = Y[rows]
    n, L = Yr.shape
    W = X.shape[1]
    Pn = Yr.sum(0).double()
    Nn = n - Pn
    Rt = torch.full((W, L), float('nan'), dtype=torch.float64, device=X.device)
    Au, Ap = Rt.clone(), Rt.clone()
    rate = torch.zeros(W, dtype=torch.float64, device=X.device)
    Xr = X.index_select(0, rows)
    for j0 in range(0, W, block):
        xb = Xr[:, j0:j0 + block]
        b = xb.shape[1]
        bits = (xb.contiguous().view(torch.int16).long() & 0x7FFF) + NB * torch.arange(b, device=X.device)[None]
        tot = torch.bincount(bits.flatten(), minlength=b * NB).view(b, NB).double()
        rate[j0:j0 + b] = 1 - tot[:, 0] / n
        tot_d = tot.flip(1)
        ccnt = tot_d.cumsum(1)
        for li in range(L):
            if Pn[li] == 0 or Nn[li] == 0:
                continue
            pos = torch.bincount(bits[Yr[:, li]].flatten(), minlength=b * NB).view(b, NB).double().flip(1)
            neg = tot_d - pos
            ctp, cfp = pos.cumsum(1), neg.cumsum(1)
            Au[j0:j0 + b, li] = (pos * (Nn[li] - cfp) + 0.5 * pos * neg).sum(1) / (Pn[li] * Nn[li])
            Ap[j0:j0 + b, li] = (pos / Pn[li] * ctp / (ctp + cfp).clamp_min(1)).sum(1)
            k = Pn[li]
            ib = torch.searchsorted(ccnt.contiguous(), k.expand(b, 1).contiguous()).clamp_max(NB - 1)
            cb, tb, pb = ccnt.gather(1, ib)[:, 0], tot_d.gather(1, ib)[:, 0], pos.gather(1, ib)[:, 0]
            tp = ctp.gather(1, ib)[:, 0] - pb + pb * (k - (cb - tb)) / tb.clamp_min(1)
            pi = k / n
            Rt[j0:j0 + b, li] = (tp / k - pi) / (1 - pi)
        del bits
    return {'R': Rt.cpu().numpy(), 'A': Au.cpu().numpy(), 'AP': Ap.cpu().numpy(), 'rate': rate.cpu().numpy(),
            'base': (Pn / n).cpu().numpy(), 'n': n}


def crossfit(X, Y, half_of, labels):
    """X (n, W) fp16 cuda codes, Y (n, L) bool cuda, half_of {split: (n,) 0/1 per frame}, labels (L names) ->
    {'splits': {split: {label: {R, A_at_R, AP_lift_at_R, rate_at_R, A, neurons [dir0, dir1]}}}, 'mean': {label: ...}}.
    R = R~ of the R~-selected neuron on the held-out half; A = AUROC of the AUROC-selected neuron (cf AUROC);
    A_at_R = AUROC of the R~-selected neuron."""
    out = {'splits': {}}
    keys = ('R', 'A_at_R', 'AP_lift_at_R', 'rate_at_R', 'A')
    for sn, half in half_of.items():
        H = {h: half_stats(X, Y, torch.from_numpy(np.flatnonzero(half == h)).to(X.device)) for h in (0, 1)}
        rec = {}
        for li, lb in enumerate(labels):
            dirs = []
            for s_, t_ in ((0, 1), (1, 0)):
                sc = np.where(np.isfinite(H[s_]['R'][:, li]), H[s_]['R'][:, li], -np.inf)
                j = int(np.argmax(sc))
                sa = np.where(np.isfinite(H[s_]['A'][:, li]), H[s_]['A'][:, li], -np.inf)
                ja = int(np.argmax(sa))
                dirs.append({'neuron': j, 'R': float(H[t_]['R'][j, li]), 'A_at_R': float(H[t_]['A'][j, li]),
                             'AP_lift_at_R': float(H[t_]['AP'][j, li] / max(H[t_]['base'][li], 1e-12)),
                             'rate_at_R': float(H[t_]['rate'][j]), 'A': float(H[t_]['A'][ja, li])})
            rec[lb] = {k_: float(np.mean([d[k_] for d in dirs])) for k_ in keys}
            rec[lb]['neurons'] = [d['neuron'] for d in dirs]
        out['splits'][sn] = rec
    rnd = [k_ for k_ in half_of if k_.startswith('rnd')]
    out['mean'] = {}
    for lb in labels:
        m = {k_: float(np.mean([out['splits'][s][lb][k_] for s in rnd])) for k_ in keys}
        m['R_sd_splits'] = float(np.std([out['splits'][s][lb]['R'] for s in rnd]))
        sel = [j for s in rnd for j in out['splits'][s][lb]['neurons']]
        vals, cnt = np.unique(sel, return_counts=True)
        m['modal_neuron'], m['modal_share'] = int(vals[np.argmax(cnt)]), float(cnt.max() / len(sel))
        out['mean'][lb] = m
    return out


def block_of(j, blocks):
    """Union dictionary: global neuron index j, blocks [(seed, width)] -> (seed, local index)."""
    for s, w in blocks:
        if j < w:
            return s, j
        j -= w
    raise IndexError(j)


def welch(a, b):
    """mean(a) - mean(b) with the Welch 95% CI -> (diff, lo, hi)."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else 1
    h = stats.t.ppf(0.975, df) * se
    return float(d), float(d - h), float(d + h)
