"""
Label-free per-neuron HIDDEN MARKOV readout (ECI v2): P(on) = the posterior probability that a neuron is in its 'on'
state, from its raw frame-max code sequence at 5 fps. Replaces the ad-hoc 0.6 s smoothing by a model of bouts.

Model (one per neuron, K = 2 or 3 states; definitions pre-registered in scripts/eci/v2_prob/analysis.py):
    hidden state h_t in {0..K-1}, first-order Markov chain per video (initial pi0, transition A);
    emission of the frame code c_t >= 0 given h_t = k: a point mass at 0 with probability z_k (the neuron is silent),
    else log c_t ~ N(mu_k, var_k) (density on c: N(log c; mu_k, var_k) / c);
    'on' = the state with the largest expected code (1 - z_k) exp(mu_k + var_k / 2).
    Fit: Baum-Welch (scaled forward-backward), every video one sequence, at most 200 iterations, stop when the
    log-likelihood gain per frame < 1e-7; var floor 1e-3; z clipped to [1e-6, 1 - 1e-6].
    Init (by quantiles of log c over the positive frames, q): K = 2: mu = (q25, q90), z = (0.9, 0.1);
    K = 3: mu = (q10, q50, q90), z = (0.95, 0.5, 0.05); var = var(log c) for all states; A = 0.95 on the diagonal,
    the rest spread evenly; pi0 uniform.
    Output: P(on) per frame (forward-backward posterior, float64) and the Viterbi state path (on = 1).
    Fallback: < 200 positive frames in the fit data -> P(on) = 1[c > 0], Viterbi = 1[c > 0].
"""
import numpy as np
from numba import njit

LOG2PI = np.log(2 * np.pi)


@njit(cache=True)
def _logB(x, z, mu, var):
    n, K = x.shape[0], z.shape[0]
    B = np.empty((n, K))
    for t in range(n):
        if x[t] <= 0:
            for k in range(K):
                B[t, k] = np.log(z[k])
        else:
            lx = np.log(x[t])
            for k in range(K):
                B[t, k] = np.log(1 - z[k]) - 0.5 * (LOG2PI + np.log(var[k]) + (lx - mu[k]) ** 2 / var[k]) - lx
    return B


@njit(cache=True)
def _fb(logB, starts, pi0, A):
    """scaled forward-backward over the sequences [starts[i], starts[i+1]) -> gamma (n, K), xi sum (K, K),
    log-likelihood."""
    n, K = logB.shape
    gamma = np.empty((n, K))
    xis = np.zeros((K, K))
    ll = 0.0
    al = np.empty((n, K))
    be = np.empty((n, K))
    sc = np.empty(n)
    Bs = np.empty((n, K))
    for t in range(n):
        m = logB[t].max()
        for k in range(K):
            Bs[t, k] = np.exp(logB[t, k] - m)
        ll += m
    for s in range(starts.shape[0] - 1):
        a, b = starts[s], starts[s + 1]
        if b <= a:
            continue
        tot = 0.0
        for k in range(K):
            al[a, k] = pi0[k] * Bs[a, k]
            tot += al[a, k]
        sc[a] = tot
        for k in range(K):
            al[a, k] /= tot
        for t in range(a + 1, b):
            tot = 0.0
            for k in range(K):
                v = 0.0
                for j in range(K):
                    v += al[t - 1, j] * A[j, k]
                al[t, k] = v * Bs[t, k]
                tot += al[t, k]
            sc[t] = tot
            for k in range(K):
                al[t, k] /= tot
        for k in range(K):
            be[b - 1, k] = 1.0
        for t in range(b - 2, a - 1, -1):
            for j in range(K):
                v = 0.0
                for k in range(K):
                    v += A[j, k] * Bs[t + 1, k] * be[t + 1, k]
                be[t, j] = v / sc[t + 1]
        for t in range(a, b):
            ll += np.log(sc[t])
            g = 0.0
            for k in range(K):
                gamma[t, k] = al[t, k] * be[t, k]
                g += gamma[t, k]
            for k in range(K):
                gamma[t, k] /= g
        for t in range(a, b - 1):
            tot = 0.0
            for j in range(K):
                for k in range(K):
                    tot += al[t, j] * A[j, k] * Bs[t + 1, k] * be[t + 1, k]
            for j in range(K):
                for k in range(K):
                    xis[j, k] += al[t, j] * A[j, k] * Bs[t + 1, k] * be[t + 1, k] / tot
    return gamma, xis, ll


@njit(cache=True)
def _viterbi(logB, starts, pi0, A):
    n, K = logB.shape
    path = np.zeros(n, np.int64)
    lA = np.log(A)
    d = np.empty((n, K))
    bp = np.zeros((n, K), np.int64)
    for s in range(starts.shape[0] - 1):
        a, b = starts[s], starts[s + 1]
        if b <= a:
            continue
        for k in range(K):
            d[a, k] = np.log(pi0[k]) + logB[a, k]
        for t in range(a + 1, b):
            for k in range(K):
                best, arg = -np.inf, 0
                for j in range(K):
                    v = d[t - 1, j] + lA[j, k]
                    if v > best:
                        best, arg = v, j
                d[t, k] = best + logB[t, k]
                bp[t, k] = arg
        k = 0
        for j in range(K):
            if d[b - 1, j] > d[b - 1, k]:
                k = j
        path[b - 1] = k
        for t in range(b - 1, a, -1):
            k = bp[t, k]
            path[t - 1] = k
    return path


def fit(x, starts, K=2, iters=200, tol=1e-7):
    """x (n,) raw frame codes >= 0 (videos concatenated), starts (m + 1,) sequence bounds -> params dict or None."""
    x = np.asarray(x, np.float64)
    starts = np.asarray(starts, np.int64)
    pos = x[x > 0]
    if len(pos) < 200:
        return None
    lx = np.log(pos)
    if K == 2:
        mu, z = np.percentile(lx, [25, 90]), np.array([0.9, 0.1])
    else:
        mu, z = np.percentile(lx, [10, 50, 90]), np.array([0.95, 0.5, 0.05])
    mu = mu.astype(np.float64)
    var = np.full(K, max(lx.var(), 1e-3))
    A = np.full((K, K), 0.05 / (K - 1))
    np.fill_diagonal(A, 0.95)
    pi0 = np.full(K, 1.0 / K)
    isz = x <= 0
    lxa = np.where(isz, 0.0, np.log(np.where(isz, 1.0, x)))
    prev = -np.inf
    n = len(x)
    for it in range(iters):
        g, xis, ll = _fb(_logB(x, z, mu, var), starts, pi0, A)
        st = starts[:-1][np.diff(starts) > 0]
        pi0 = np.clip(g[st].mean(0), 1e-6, None)
        pi0 /= pi0.sum()
        A = xis / np.maximum(xis.sum(1, keepdims=True), 1e-300)
        A = np.clip(A, 1e-9, None)
        A /= A.sum(1, keepdims=True)
        gs = g.sum(0)
        gp = (g * (~isz)[:, None]).sum(0)
        z = np.clip((gs - gp) / np.maximum(gs, 1e-300), 1e-6, 1 - 1e-6)
        mu = np.where(gp > 1e-9, (g * lxa[:, None] * (~isz)[:, None]).sum(0) / np.maximum(gp, 1e-300), mu)
        var = np.maximum(np.where(gp > 1e-9, (g * (~isz)[:, None] * (lxa[:, None] - mu[None]) ** 2).sum(0)
                                  / np.maximum(gp, 1e-300), var), 1e-3)
        if (ll - prev) / n < tol:
            break
        prev = ll
    ec = (1 - z) * np.exp(mu + var / 2)
    on = int(np.argmax(ec))
    return {'K': K, 'pi0': pi0, 'A': A, 'z': z, 'mu': mu, 'var': var, 'on': on, 'll': float(ll), 'iters': it + 1,
            'ec': ec}


def apply(x, starts, p):
    """-> (P(on) (n,) float64, viterbi on (n,) bool). p None = fallback 1[c > 0]."""
    x = np.asarray(x, np.float64)
    starts = np.asarray(starts, np.int64)
    if p is None:
        return (x > 0).astype(np.float64), x > 0
    lb = _logB(x, p['z'], p['mu'], p['var'])
    g, _, _ = _fb(lb, starts, p['pi0'], p['A'])
    vp = _viterbi(lb, starts, p['pi0'], p['A'])
    return g[:, p['on']], vp == p['on']


def stationary_on(p):
    """expected share of time in the on state and its mean bout length (frames) under the fitted chain."""
    if p is None:
        return float('nan'), float('nan')
    w, v = np.linalg.eig(p['A'].T)
    s = np.real(v[:, np.argmin(np.abs(w - 1))])
    s = s / s.sum()
    return float(s[p['on']]), float(1 / max(1 - p['A'][p['on'], p['on']], 1e-12))
