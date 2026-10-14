"""
Label-free EVENT CONFIDENCE for one SAE neuron: a two-state (OFF / ON) hidden Markov model on the neuron's per-frame
read-out a_t = codes_max_j(t), fitted by EM (Baum-Welch) on unlabelled videos. The event confidence of frame t is the
smoothed posterior P(ON at t | the whole video) from forward-backward.

Read-out facts it builds on (src/eci/sae.py, threshold mode): a_t is either exactly 0 or > theta (the SAE's single
global inference threshold), so the emission is ZERO-INFLATED:
    P(a = 0 | s)           = z_s                                                     (state s in {OFF, ON})
    u = log(a / theta) > 0 | a > 0, s  ~  Gamma(shape k_s, scale c_s)                (excess log-activation)
i.e. a state is "how often the neuron fires" (1 - z_s) plus "how strongly, above threshold" (the gamma on u).
Transitions: a 2 x 2 matrix A (A[OFF, ON] = bout onset, A[ON, OFF] = bout offset per frame), initial probabilities pi
per video. Two priors come out of the fit, not from labels: bouts persist (mean ON bout = 1 / A[ON, OFF] frames) and
the concept's base rate (stationary P(ON) = A[OFF, ON] / (A[OFF, ON] + A[ON, OFF])).
ON is DEFINED as the state with the higher expected read-out E[a | s] = (1 - z_s) theta E[exp(u)] (states are swapped
after the fit if needed).

The per-frame MIXTURE (ablation, no temporal coupling) is the same model with A's rows tied to (1 - w, w) and pi =
(1 - w, w): its forward-backward posterior equals the per-frame Bayes posterior, so both models share one E-step.

Inputs are (T, J) arrays for J neurons over the frames of several videos concatenated; starts (V + 1,) are the video
boundaries (video v = rows starts[v]:starts[v + 1]); every neuron is fitted independently (numba, parallel over j).

Functions
    fit(a, starts, theta, temporal=True)   -> params dict of (J, ...) arrays + 'history' (log-lik per frame per iter)
    posterior(a, starts, theta, params)    -> (T, J) float32 P(ON | video)
    summary(params, theta, fps)            -> per-neuron dict list: stationary P(ON), bout durations, emissions
"""
import math

import numpy as np
from numba import njit, prange
from scipy.special import digamma, polygamma
from scipy.stats import gamma as gamma_dist

U_MIN = 1e-3          # u = log(a / theta) floor (fp16 rounding can put a firing value at or just below theta)
Z_CLIP = (1e-8, 1 - 1e-8)
K_CLIP = (0.05, 1e4)
A_MIN = 1e-9
N_STATS = 15          # G0[2], G1[2], Su[2], Slu[2], Xi[4], init[2], loglik


def excess(a, theta):
    """a (T, J) read-outs -> fire (T, J) uint8, u (T, J) float32 (log(a / theta) floored at U_MIN where firing, else 0)."""
    a = np.asarray(a, dtype=np.float32)
    fire = a > 0
    u = np.zeros(a.shape, np.float32)
    u[fire] = np.maximum(np.log(a[fire] / np.float32(theta)), U_MIN)
    return fire.astype(np.uint8), u


@njit(parallel=True, cache=False)
def _forward_backward(fire, u, starts, z, k, sc, A, pi, post, want_post):
    """One E-step for every neuron. fire / u (T, J); starts (V + 1,); z, k, sc, pi (J, 2); A (J, 2, 2).
    -> stats (J, N_STATS); writes post[:, j] = P(ON) when want_post."""
    T, J = fire.shape
    V = starts.shape[0] - 1
    stats = np.zeros((J, N_STATS))
    Lmax = 0
    for v in range(V):
        Lmax = max(Lmax, starts[v + 1] - starts[v])
    for j in prange(J):
        lz0, lz1 = math.log(z[j, 0]), math.log(z[j, 1])
        l1z0, l1z1 = math.log(1.0 - z[j, 0]), math.log(1.0 - z[j, 1])
        c00 = -math.lgamma(k[j, 0]) - k[j, 0] * math.log(sc[j, 0])
        c01 = -math.lgamma(k[j, 1]) - k[j, 1] * math.log(sc[j, 1])
        a00, a01, a10, a11 = A[j, 0, 0], A[j, 0, 1], A[j, 1, 0], A[j, 1, 1]
        B = np.empty((Lmax, 2))
        al = np.empty((Lmax, 2))
        be = np.empty((Lmax, 2))
        c = np.empty(Lmax)
        s = np.zeros(N_STATS)
        for v in range(V):
            o = starts[v]
            L = starts[v + 1] - o
            if L == 0:
                continue
            for t in range(L):
                if fire[o + t, j]:
                    x = u[o + t, j]
                    lx = math.log(x)
                    b0 = l1z0 + c00 + (k[j, 0] - 1.0) * lx - x / sc[j, 0]
                    b1 = l1z1 + c01 + (k[j, 1] - 1.0) * lx - x / sc[j, 1]
                else:
                    b0, b1 = lz0, lz1
                m = max(b0, b1)
                B[t, 0] = math.exp(b0 - m)
                B[t, 1] = math.exp(b1 - m)
                s[14] += m
            p0 = pi[j, 0] * B[0, 0]
            p1 = pi[j, 1] * B[0, 1]
            ct = p0 + p1
            al[0, 0], al[0, 1], c[0] = p0 / ct, p1 / ct, ct
            for t in range(1, L):
                p0 = (al[t - 1, 0] * a00 + al[t - 1, 1] * a10) * B[t, 0]
                p1 = (al[t - 1, 0] * a01 + al[t - 1, 1] * a11) * B[t, 1]
                ct = p0 + p1
                al[t, 0], al[t, 1], c[t] = p0 / ct, p1 / ct, ct
            for t in range(L):
                s[14] += math.log(c[t])
            be[L - 1, 0], be[L - 1, 1] = 1.0, 1.0
            for t in range(L - 2, -1, -1):
                q0 = B[t + 1, 0] * be[t + 1, 0] / c[t + 1]
                q1 = B[t + 1, 1] * be[t + 1, 1] / c[t + 1]
                be[t, 0] = a00 * q0 + a01 * q1
                be[t, 1] = a10 * q0 + a11 * q1
                s[8] += al[t, 0] * a00 * q0
                s[9] += al[t, 0] * a01 * q1
                s[10] += al[t, 1] * a10 * q0
                s[11] += al[t, 1] * a11 * q1
            for t in range(L):
                g0 = al[t, 0] * be[t, 0]
                g1 = al[t, 1] * be[t, 1]
                n = g0 + g1
                g0, g1 = g0 / n, g1 / n
                if t == 0:
                    s[12] += g0
                    s[13] += g1
                if fire[o + t, j]:
                    x = u[o + t, j]
                    lx = math.log(x)
                    s[2] += g0
                    s[3] += g1
                    s[4] += g0 * x
                    s[5] += g1 * x
                    s[6] += g0 * lx
                    s[7] += g1 * lx
                else:
                    s[0] += g0
                    s[1] += g1
                if want_post:
                    post[o + t, j] = g1
        for i in range(N_STATS):
            stats[j, i] = s[i]
    return stats


def _gamma_mle(m, ml, k_prev):
    """Weighted gamma MLE from mean m and mean log ml (arrays): Minka's start + 4 Newton steps on log k - psi(k) = s."""
    s = np.maximum(np.log(m) - ml, 1e-8)
    k = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(4):
        f = np.log(k) - digamma(k) - s
        fp = 1 / k - polygamma(1, k)
        k = np.clip(k - f / fp, K_CLIP[0], K_CLIP[1])
    k = np.where(np.isfinite(k), k, k_prev)
    return k, m / k


def _init(fire, u):
    """Deterministic label-free start from each neuron's firing rate p: ON fires half the time, OFF at 0.1 p,
    w = P(ON) chosen so the mixture reproduces p (capped at 0.5); both states start with the gamma fitted to all u;
    mean ON bout 10 frames."""
    J = fire.shape[1]
    p = np.clip(fire.mean(0, dtype=np.float64), 1e-6, 0.99)
    f_on = np.full(J, 0.5)
    w = np.minimum(0.5, 0.9 * p / (0.5 - 0.1 * np.minimum(p, 0.99)))
    f_off = np.clip((p - w * f_on) / (1 - w), 1e-6, 0.49)
    z = np.clip(np.stack([1 - f_off, 1 - f_on], 1), *Z_CLIP)
    # OFF starts from the gamma of the lower half of the firing amplitudes, ON from the upper half (a symmetric start
    # is a fixed point of the mixture's EM: identical gammas get identical responsibilities on every firing frame)
    med = np.array([np.median(u[fire[:, j] > 0, j]) if fire[:, j].any() else 1.0 for j in range(J)], np.float32)
    ks, scs = [], []
    for sel in ((fire > 0) & (u <= med[None]), (fire > 0) & (u > med[None])):
        n1 = np.maximum(sel.sum(0, dtype=np.float64), 1)
        m = np.maximum(np.where(sel, u, 0).sum(0, dtype=np.float64) / n1, 2 * U_MIN)
        lu = np.where(sel, np.log(np.maximum(u, U_MIN)), 0)
        ml = np.minimum(lu.sum(0, dtype=np.float64) / n1, np.log(m) - 1e-6)
        k, sc = _gamma_mle(m, ml, np.ones(J))
        ks.append(k)
        scs.append(sc)
    k, sc = np.stack(ks, 1), np.stack(scs, 1)
    off = np.full(J, 0.1)
    on = w * off / (1 - w)
    A = np.stack([np.stack([1 - on, on], 1), np.stack([off, 1 - off], 1)], 1)
    pi = np.stack([1 - w, w], 1)
    return dict(z=z, k=k, sc=sc, A=A, pi=pi)


def _mstep(st, P, temporal, V):
    G0, G1, Su, Slu = st[:, 0:2], st[:, 2:4], st[:, 4:6], st[:, 6:8]
    Xi, I0 = st[:, 8:12].reshape(-1, 2, 2), st[:, 12:14]
    n = G0 + G1
    z = np.clip(G0 / np.maximum(n, 1e-300), *Z_CLIP)
    ok = G1 > 1e-6
    m = np.where(ok, Su / np.maximum(G1, 1e-300), P['sc'] * P['k'])
    ml = np.where(ok, Slu / np.maximum(G1, 1e-300), np.log(m) - 0.1)
    ml = np.minimum(ml, np.log(m) - 1e-8)
    k, sc = _gamma_mle(m, ml, P['k'])
    k, sc = np.where(ok, k, P['k']), np.where(ok, sc, P['sc'])
    if temporal:
        A = Xi / np.maximum(Xi.sum(2, keepdims=True), 1e-300)
        A = np.clip(A, A_MIN, 1)
        A = A / A.sum(2, keepdims=True)
        pi = np.clip(I0 / V, 1e-6, 1)
        pi = pi / pi.sum(1, keepdims=True)
    else:  # mixture: every row of A and pi = (1 - w, w)
        w = np.clip(n[:, 1] / n.sum(1), 1e-8, 1 - 1e-8)
        row = np.stack([1 - w, w], 1)
        A, pi = np.stack([row, row], 1), row.copy()
    return dict(z=z, k=k, sc=sc, A=A, pi=pi)


def expected_log_readout(P):
    """log E[a | s] - log theta = log(1 - z_s) - k_s log(1 - c_s)  (gamma MGF at 1; +inf when c_s >= 1)."""
    sc = P['sc']
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(sc < 1, np.log1p(-P['z']) - P['k'] * np.log1p(-np.minimum(sc, 1 - 1e-12)), np.inf)


def _swap(P):
    """Make state 1 = ON = the state with the higher expected read-out E[a | s] (firing probability x amplitude)."""
    e = expected_log_readout(P)
    s = e[:, 0] > e[:, 1]
    out = {}
    for key in ('z', 'k', 'sc', 'pi'):
        x = P[key].copy()
        x[s] = x[s][:, ::-1]
        out[key] = x
    A = P['A'].copy()
    A[s] = A[s][:, ::-1, ::-1]
    out['A'] = A
    out['swapped'] = s
    return out


def fit(a, starts, theta, temporal=True, n_iter=500, tol=1e-8, init=None):
    """EM on (T, J) read-outs of the videos given by starts. -> params dict (z, k, sc, A, pi: (J, ...)), plus
    'loglik' (J,) per frame, 'n_iter', 'converged' (J,) and 'history' (iters, J) log-lik per frame.
    temporal=False fits the per-frame mixture. Converged = per-frame log-lik gain < tol."""
    fire, u = excess(a, theta)
    starts = np.asarray(starts, np.int64)
    T, J = fire.shape
    V = len(starts) - 1
    P = init if init is not None else _init(fire, u)
    if not temporal:
        w = P['pi'][:, 1]
        row = np.stack([1 - w, w], 1)
        P = dict(P, A=np.stack([row, row], 1), pi=row.copy())
    dummy = np.zeros((1, 1), np.float32)
    hist, conv, ll_prev = [], np.zeros(J, bool), np.full(J, -np.inf)
    for it in range(n_iter):
        st = _forward_backward(fire, u, starts, P['z'], P['k'], P['sc'], P['A'], P['pi'], dummy, False)
        ll = st[:, 14] / T
        hist.append(ll)
        gain = ll - ll_prev
        if np.any(gain < -1e-6 * np.abs(ll)):
            bad = np.flatnonzero(gain < -1e-6 * np.abs(ll))
            print(f'event_hmm.fit: log-lik decreased at iter {it} for {len(bad)} neurons (max drop '
                  f'{float(-gain[bad].min()):.3g} per frame)', flush=True)
        conv = gain < tol
        ll_prev = ll
        if conv.all():
            break
        P = _mstep(st, P, temporal, V)
    out = _swap(P)
    out.update(loglik=ll, n_iter=it + 1, converged=conv, history=np.array(hist), temporal=temporal)
    return out


def posterior(a, starts, theta, params):
    """(T, J) read-outs of the videos in starts -> (T, J) float32 smoothed P(ON | video) under the fitted params."""
    fire, u = excess(a, theta)
    post = np.zeros(fire.shape, np.float32)
    _forward_backward(fire, u, np.asarray(starts, np.int64), params['z'], params['k'], params['sc'], params['A'],
                      params['pi'], post, True)
    return post


def summary(params, theta, fps):
    """Per neuron: stationary P(ON), mean bout durations (s), per-state firing probability and the median / mean
    read-out given firing (a = theta * exp(u), u ~ Gamma)."""
    A, z, k, sc = params['A'], params['z'], params['k'], params['sc']
    out = []
    for j in range(len(z)):
        on, off = A[j, 0, 1], A[j, 1, 0]
        r = {'stationary_p_on': float(on / (on + off)),
             'mean_on_bout_s': float(1 / off / fps), 'mean_off_bout_s': float(1 / on / fps),
             'p_on_to_on_per_frame': float(A[j, 1, 1]), 'p_off_to_on_per_frame': float(on)}
        for s, name in ((0, 'off'), (1, 'on')):
            g = gamma_dist(k[j, s], scale=sc[j, s])
            r[f'{name}_p_fire'] = float(1 - z[j, s])
            r[f'{name}_gamma_shape'] = float(k[j, s])
            r[f'{name}_gamma_scale'] = float(sc[j, s])
            r[f'{name}_median_readout_given_fire'] = float(theta * np.exp(g.median()))
            r[f'{name}_mean_log_excess_given_fire'] = float(k[j, s] * sc[j, s])
        out.append(r)
    return out
