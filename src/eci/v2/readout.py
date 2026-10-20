"""
Frame readout, firing rule and video summaries (ECI v2).

  frame value      c_j(f) = max over the frame's readout tokens of neuron j's code (position invariant; 0 when the
                   neuron is silent or the frame keeps no token).
  smoothing        the only temporal step (accepted prior): s_j(f) = centred mean of c_j over the frames f - h .. f + h
                   of the same video (h = 1 at 5 fps = a 0.6 s window), truncated at the video ends.
  firing rule      'fnorm' (as r4): fires iff s_j(f) / sigma_j > theta_n; sigma_j = median of neuron j's non-zero
                   smoothed values on the CALIBRATION videos (neurons never active there: the median of the sigma_j),
                   theta_n = the global value such that the mean number of firing neurons per calibration frame is
                   k (= 16, the SAE's own sparsity): the (n_frames k + 1)-th largest normalised value. Label-free, one
                   global parameter.
  summaries        per video and neuron: time spent (share of firing frames), number of bouts and bout rate per
                   minute (bout = maximal run of firing frames), latency (first firing frame / fps; censored = never
                   fires, latency = video length).
"""
import numpy as np


def smooth(x, h=1):
    """x (n, W) float -> centred mean over +-h frames (truncated at the ends), float32."""
    x = np.asarray(x, np.float32)
    if h == 0 or len(x) == 0:
        return x.copy()
    c = np.cumsum(np.concatenate([np.zeros((1, x.shape[1]), np.float64), x.astype(np.float64)]), 0)
    n = len(x)
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return ((c[hi] - c[lo]) / (hi - lo)[:, None]).astype(np.float32)


def calibrate_fnorm(S_list, k=16):
    """S_list: smoothed (n_v, W) arrays of the calibration videos -> (sigma (W,), theta_n, info)."""
    W = S_list[0].shape[1]
    nz = [[] for _ in range(W)]
    for S in S_list:
        for j in range(W):
            v = S[:, j]
            v = v[v > 0]
            if len(v):
                nz[j].append(v)
    sig = np.array([np.median(np.concatenate(v)) if v else np.nan for v in nz], np.float64)
    n_never = int(np.isnan(sig).sum())
    sig[np.isnan(sig)] = np.nanmedian(sig)
    V = np.concatenate([S / sig[None] for S in S_list]).ravel()
    nf = sum(len(S) for S in S_list)
    j = int(nf * k)
    th = float(np.partition(V, len(V) - j - 1)[len(V) - j - 1]) if j < len(V) else 0.0
    info = {'n_calib_frames': int(nf), 'n_never_active': n_never, 'mean_fires_per_frame': float((V > th).sum() / nf),
            'sigma_median': float(np.median(sig))}
    return sig.astype(np.float32), th, info


def fires(S, sig, th):
    return S / sig[None] > th


def video_summaries(F, fps=5.0):
    """F (n, W) bool firing -> dict of (W,) arrays: time_spent, n_bouts, bout_rate_per_min, latency_s, censored."""
    n = len(F)
    starts = F[0].astype(np.int64) + (F[1:] & ~F[:-1]).sum(0)
    first = np.where(F.any(0), F.argmax(0), n)
    return {'time_spent': F.mean(0), 'n_bouts': starts, 'bout_rate_per_min': starts / (n / fps / 60),
            'latency_s': first / fps, 'censored': ~F.any(0)}
