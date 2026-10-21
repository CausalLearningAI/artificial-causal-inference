"""
Mixture-family selection for the per-neuron binarisation (ECI v2), label-free.

PRE-REGISTERED 2026-10-21 ~23:40 (before any family comparison was computed):
  families  (a) LOGNORMAL: 2-component Gaussian mixture on log s of the positive smoothed codes (= src/eci/v2/prob.py,
                the current default); as a density on s: sum_c pi_c N(log s; mu_c, var_c) / s.
            (b) GAMMA: 2-component gamma mixture on the positive smoothed codes s, EM with weighted maximum likelihood
                (shape by the Minka closed-form start + 5 Newton steps per M-step, scale = weighted mean / shape),
                init by the median split, 200 iterations, shape floor 0.05.
            (c) ZERO-INFLATED (exact zeros = part of the LOW state): for s > 0 its posterior P(on | s) equals that of the
                family without the zero mass (a point mass at 0 has no density at s > 0), so its decisions and active
                fractions are identical to (a) / (b); it is not fitted separately.
  data      the smoothed codes of the domain's calibration videos, split by video into two halves (sorted observation
            ids, alternating): FIT on half A, SCORE on half B; per neuron at most 20,000 positive values per half (rng 0).
  score     per neuron, mean held-out log-likelihood of the positive values under (a) vs (b) (densities on s); a neuron
            with < 200 positive values in either half is not compared.
  selection per domain the family with the higher held-out log-likelihood summed over the compared neurons becomes the
            default binarisation (MAP: active iff P(on | s) > 0.5, monotone posterior as prob.py). Reported: share of
            neurons each family wins, per-neuron decision boundaries s* (P = 0.5) of both families (median |log ratio|),
            active fractions on the calibration frames under both, and (if (b) wins) the alignment of its binary.
"""
import numpy as np
from scipy.special import digamma, gammaln, polygamma


def _gamma_logpdf(x, k, th):
    return (k - 1) * np.log(x) - x / th - gammaln(k) - k * np.log(th)


def _shape(s_):
    k = (3 - s_ + np.sqrt((s_ - 3) ** 2 + 24 * s_)) / (12 * s_)
    for _ in range(5):
        k = k - (np.log(k) - digamma(k) - s_) / (1 / k - polygamma(1, k))
        k = np.maximum(k, 0.05)
    return k


def fit_gamma(x, iters=200):
    med = np.median(x)
    r = np.stack([x <= med, x > med], 1).astype(float)
    k, th, pi = np.ones(2), np.ones(2), np.array([0.5, 0.5])
    lx = np.log(x)
    for it in range(iters):
        if it > 0:
            ll = np.stack([_gamma_logpdf(x, k[c], th[c]) + np.log(pi[c]) for c in range(2)], 1)
            m = ll.max(1, keepdims=True)
            r = np.exp(ll - m)
            r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-12
        pi = nk / nk.sum()
        mx = (r * x[:, None]).sum(0) / nk
        ml = (r * lx[:, None]).sum(0) / nk
        s_ = np.maximum(np.log(mx) - ml, 1e-6)
        k = _shape(s_)
        th = mx / k
    return {'k': k, 'th': th, 'pi': pi, 'on': int(np.argmax(k * th))}


def gamma_loglik(x, p):
    ll = np.stack([_gamma_logpdf(x, p['k'][c], p['th'][c]) + np.log(p['pi'][c]) for c in range(2)], 1)
    m = ll.max(1, keepdims=True)
    return m[:, 0] + np.log(np.exp(ll - m).sum(1))


def gamma_post(x, p):
    ll = np.stack([_gamma_logpdf(x, p['k'][c], p['th'][c]) + np.log(p['pi'][c]) for c in range(2)], 1)
    m = ll.max(1, keepdims=True)
    e = np.exp(ll - m)
    return e[:, p['on']] / e.sum(1)


def fit_lognormal(x, iters=200):
    lx = np.log(x)
    mu = np.percentile(lx, [25, 75]).astype(np.float64)
    var = np.full(2, max(lx.var(), 1e-4))
    pi = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = -0.5 * (lx[:, None] - mu[None]) ** 2 / var[None] - 0.5 * np.log(2 * np.pi * var[None]) + np.log(pi[None])
        m = ll.max(1, keepdims=True)
        r = np.exp(ll - m)
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-12
        pi = nk / nk.sum()
        mu = (r * lx[:, None]).sum(0) / nk
        var = np.maximum((r * (lx[:, None] - mu[None]) ** 2).sum(0) / nk, 1e-4)
    return {'mu': mu, 'var': var, 'pi': pi, 'on': int(np.argmax(mu))}


def lognormal_loglik(x, p):
    lx = np.log(x)
    ll = -0.5 * (lx[:, None] - p['mu'][None]) ** 2 / p['var'][None] - 0.5 * np.log(2 * np.pi * p['var'][None]) + \
        np.log(p['pi'][None])
    m = ll.max(1, keepdims=True)
    return m[:, 0] + np.log(np.exp(ll - m).sum(1)) - lx


def lognormal_post(x, p):
    lx = np.log(x)
    ll = -0.5 * (lx[:, None] - p['mu'][None]) ** 2 / p['var'][None] - 0.5 * np.log(2 * np.pi * p['var'][None]) + \
        np.log(p['pi'][None])
    m = ll.max(1, keepdims=True)
    e = np.exp(ll - m)
    return e[:, p['on']] / e.sum(1)


def boundary(post_fn, p, x_ref):
    """Monotone posterior on a grid of 512 quantiles of x_ref -> (grid log s, curve, s* where the curve first > 0.5)."""
    g = np.quantile(x_ref, np.linspace(0, 1, 512))
    curve = np.maximum.accumulate(post_fn(g, p))
    i = np.flatnonzero(curve > 0.5)
    return np.log(g), curve, (g[i[0]] if len(i) else np.inf)
