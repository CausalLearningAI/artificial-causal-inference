"""
Per-neuron probability that a neuron is ON (ECI v2 outcome variant 'prob'), estimated a posteriori and LABEL-FREE from
the smoothed frame codes. No SAE is retrained.

PRE-REGISTERED 2026-10-21 (before any prob code or score existed):
  input       s_j(f) = the 0.6 s centred mean of neuron j's frame max code (the '<sae>_smooth' set, >= 0).
  fit sample  the frames of the domain's CALIBRATION videos (the same label-free held-out videos as theta* and fnorm);
              per neuron at most 20,000 of its positive values, drawn uniformly (rng 0).
  model       x = log s on the positive values; a two-component 1-D Gaussian mixture fitted by EM (init: means at the
              25th / 75th percentile of x, both variances = var(x), weights 1/2; 200 iterations; variance floor 1e-4).
              The component with the higher mean is ACTIVE ('on'), the other LOW (a weak, background-level activation).
              Silent frames (s = 0) are the point mass: P(on) = 0 there.
  P(on | s)   for s > 0: the posterior of the active component, made MONOTONE non-decreasing in s by a running maximum
              over a grid of 512 quantiles of the fitted x (a wider low component would otherwise give P a second rise
              at very low s); np.interp on log s between grid points (flat beyond the ends).
  fallback    a neuron with < 200 positive values in the sample, or with |mu_on - mu_low| < 0.5 sqrt(mean variance)
              (no two components), gets P(on | s) = 1[s > 0].
  outcome     the per-frame P(on) is stored as the codes of the set '<sae>_prob'; the NES runners then give, per video,
              the mean P(on) (expected time ON), the bout rate and the latency of P(on) above its pooled q 0.95.
  sanity      (label-free, reported) P = 0 on silent frames and monotone by construction; share of fallback neurons;
              mean P(on) per frame; Spearman correlation of P with s on the fit sample.

MAP BINARISATION TEST (PRE-REGISTERED 2026-10-21 ~19:00, before any MAP code or score existed):
  definition  per-neuron 'active' = the MAP decision of the same mixture: active iff P(on | s) > 0.5 (equivalently a
              per-neuron threshold on s, the point where the active component becomes more probable). Codes set
              '<sae>_map' = 1[P(on) > 0.5] (0 / 1). NES outcomes on it: mean = fraction of time active; bouts / latency
              from the SAME binary (the runners' pooled-quantile threshold is replaced by 0, i.e. 'above' = active;
              scripts/eci/v2/nes_v2_dispatch.py, V2_ZERO_THR=1).
  bimodality  per neuron on its fit sample: BIC gain of 2 vs 1 Gaussian component on log s (BIC_1 - BIC_2, > 10 =
              strong evidence for two components), separation |mu_on - mu_low| / sqrt(mean variance); share of neurons
              that fall back (no mixture) and share with BIC gain <= 10.
  comparison  on the same frames and splits (tools.py align): R~ (ties at the cut shared pro rata = the expectation
              under random tie-breaking) and cf AUROC of v2_map vs v2_smooth (continuous) for mice nose_nose,
              nose_tail and ants v2 groom_any, v3 groom_any, v3 on_lid_any; active fraction per neuron (sanity) and the
              active fraction of each label's selected neuron; labels grouped by base rate (< 10% rare, >= 10% common).
  rule        PROPOSE map as the interpretable default outcome ('fraction of time active') iff for EVERY listed label
              R~(map) >= R~(continuous) - 0.03 AND AUROC(map) >= AUROC(continuous) - 0.05; continuous stays the
              ranking / alignment representation either way. NES / NESS discovery counts are reported, never decide.
"""
import numpy as np


def fit_neuron(v, rng, n_max=20000, iters=200, n_grid=512):
    """v: positive smoothed values of one neuron -> dict (grid (n_grid,) log s, curve (n_grid,) P(on), params) or
    None (fallback)."""
    if len(v) < 200:
        return None
    if len(v) > n_max:
        v = v[rng.choice(len(v), n_max, replace=False)]
    x = np.log(v.astype(np.float64))
    mu = np.percentile(x, [25, 75]).astype(np.float64)
    var = np.full(2, max(x.var(), 1e-4))
    pi = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = -0.5 * (x[:, None] - mu[None]) ** 2 / var[None] - 0.5 * np.log(2 * np.pi * var[None]) + np.log(pi[None])
        m = ll.max(1, keepdims=True)
        r = np.exp(ll - m)
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-12
        pi = nk / nk.sum()
        mu = (r * x[:, None]).sum(0) / nk
        var = np.maximum((r * (x[:, None] - mu[None]) ** 2).sum(0) / nk, 1e-4)
    on = int(np.argmax(mu))
    if abs(mu[1] - mu[0]) < 0.5 * np.sqrt(var.mean()):
        return None
    grid = np.quantile(x, np.linspace(0, 1, n_grid))
    ll = -0.5 * (grid[:, None] - mu[None]) ** 2 / var[None] - 0.5 * np.log(2 * np.pi * var[None]) + np.log(pi[None])
    m = ll.max(1, keepdims=True)
    post = np.exp(ll - m)
    post = post[:, on] / post.sum(1)
    curve = np.maximum.accumulate(post)
    return {'grid': grid, 'curve': curve, 'mu': mu, 'var': var, 'pi': pi, 'on': on}


def fit(S_cal, seed=0):
    """S_cal: (n, W) smoothed codes of the calibration frames -> (grids (W, G), curves (W, G), fallback (W,) bool, info)."""
    rng = np.random.default_rng(seed)
    W = S_cal.shape[1]
    G = 512
    grids = np.zeros((W, G), np.float32)
    curves = np.zeros((W, G), np.float32)
    fb = np.zeros(W, bool)
    for j in range(W):
        col = S_cal[:, j]
        r = fit_neuron(col[col > 0], rng)
        if r is None:
            fb[j] = True
        else:
            grids[j], curves[j] = r['grid'], r['curve']
    return grids, curves, fb, {'n_fallback': int(fb.sum()), 'W': W}


def apply(S, grids, curves, fb):
    """S: (n, W) smoothed codes -> (n, W) float32 P(on)."""
    P = np.zeros(S.shape, np.float32)
    pos = S > 0
    with np.errstate(divide='ignore'):
        L = np.log(np.where(pos, S, 1).astype(np.float64))
    for j in range(S.shape[1]):
        if fb[j]:
            P[:, j] = pos[:, j]
        else:
            P[:, j] = np.where(pos[:, j], np.interp(L[:, j], grids[j], curves[j]), 0)
    return P


def bimodality(S_cal, seed=0, n_max=20000):
    """Per neuron: (BIC_1 - BIC_2 on log of the positive values (nan if < 200), separation, n_positive)."""
    rng = np.random.default_rng(seed)
    W = S_cal.shape[1]
    gain = np.full(W, np.nan)
    sep = np.full(W, np.nan)
    npos = np.zeros(W, int)
    for j in range(W):
        v = S_cal[:, j]
        v = v[v > 0]
        npos[j] = len(v)
        if len(v) < 200:
            continue
        if len(v) > n_max:
            v = v[rng.choice(len(v), n_max, replace=False)]
        x = np.log(v.astype(np.float64))
        n = len(x)
        ll1 = -0.5 * n * (np.log(2 * np.pi * max(x.var(), 1e-4)) + 1)
        mu = np.percentile(x, [25, 75]).astype(np.float64)
        var = np.full(2, max(x.var(), 1e-4))
        pi = np.array([0.5, 0.5])
        for _ in range(200):
            ll = -0.5 * (x[:, None] - mu[None]) ** 2 / var[None] - 0.5 * np.log(2 * np.pi * var[None]) + np.log(pi[None])
            m = ll.max(1, keepdims=True)
            r = np.exp(ll - m)
            r /= r.sum(1, keepdims=True)
            nk = r.sum(0) + 1e-12
            pi = nk / nk.sum()
            mu = (r * x[:, None]).sum(0) / nk
            var = np.maximum((r * (x[:, None] - mu[None]) ** 2).sum(0) / nk, 1e-4)
        ll = -0.5 * (x[:, None] - mu[None]) ** 2 / var[None] - 0.5 * np.log(2 * np.pi * var[None]) + np.log(pi[None])
        m = ll.max(1, keepdims=True)
        ll2 = float((m[:, 0] + np.log(np.exp(ll - m).sum(1))).sum())
        gain[j] = (-2 * ll1 + 2 * np.log(n)) - (-2 * ll2 + 5 * np.log(n))
        sep[j] = abs(mu[1] - mu[0]) / np.sqrt(var.mean())
    return gain, sep, npos
