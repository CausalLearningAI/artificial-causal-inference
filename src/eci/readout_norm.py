"""
Per-neuron normalisation of the ECI frame read-out (codes_max = max over the kept animal patches of the threshold-mode
SAE codes, src/eci/fg_encode.py), so every neuron reads as a behaviour EVENT plus an INTENSITY in [0, 1].

Definitions (label-free; the SAE feature-dashboard convention "activation / max activation", Bricken et al. 2023,
with a robust max):
    EVENT      event_j(t) = codes_max_j(t) > 0. The SAE codes are z = pre * (pre > theta) with pre = relu(pre-acts)
               and one learned global inference threshold theta > 0 (src/eci/sae.py encode, mode 'threshold'), so a
               frame value is either 0 or > theta: "> 0" is exactly "at least one kept patch crossed the threshold".
    INTENSITY  a_norm_j(t) = min(1, codes_max_j(t) / q_j), 0 when not firing. q_j = 99th percentile of neuron j's
               POSITIVE frame values over the SAE's training frames (the 1 fps token-store frames the SAE pool was
               drawn from); fixed per neuron, stored with the SAE, never refitted on evaluation data. On firing frames
               a_norm >= theta / q_j > 0, so a_norm > 0 <=> EVENT.
    rate_j     fraction of the training frames with EVENT (stored next to q).
    Neurons that never fire on the training frames: q_j = theta (fallback, flagged), so any firing reads as 1.

a_norm is NOT a calibrated probability of a human-labelled behaviour (that needs labels); it is a per-neuron rescale.

Functions:
    fit_norm(codes_train, theta)   (n_frames, m) training read-outs -> q (m,), rate (m,), n_pos (m,), fallback (m,)
    apply_norm(codes, q)           (n, m) read-outs -> event (n, m) bool, a_norm (n, m) float16
    save_norm / load_norm          <sae_dir>/readout_norm.npz
"""

import json
from pathlib import Path

import numpy as np

QUANTILE = 0.99
FILENAME = 'readout_norm.npz'


def fit_norm(codes_train, theta, quantile=QUANTILE, col_chunk=256):
    """codes_train: (n_frames, m) non-negative frame read-outs (any float dtype, numpy / memmap) of the training frames.
    theta: the SAE's inference threshold (fallback scale of neurons that never fire on the training frames).
    -> q (m,) float32, rate (m,) float32, n_pos (m,) int64, fallback (m,) bool."""
    n, m = codes_train.shape
    q = np.empty(m, np.float32)
    n_pos = np.zeros(m, np.int64)
    for c in range(0, m, col_chunk):
        X = np.asarray(codes_train[:, c:c + col_chunk], dtype=np.float32)
        if (X < 0).any() or not np.isfinite(X).all():
            raise ValueError('read-outs must be finite and >= 0')
        for k in range(X.shape[1]):
            v = X[:, k]
            v = v[v > 0]
            n_pos[c + k] = len(v)
            q[c + k] = np.quantile(v, quantile) if len(v) else np.nan
    fallback = n_pos == 0
    q[fallback] = float(theta)
    return q, (n_pos / max(n, 1)).astype(np.float32), n_pos, fallback


def apply_norm(codes, q):
    """codes: (n, m) non-negative read-outs, q: (m,) -> event (n, m) bool, a_norm (n, m) float16 in [0, 1]."""
    X = np.asarray(codes, dtype=np.float32)
    event = X > 0
    a = np.minimum(1.0, X / np.asarray(q, np.float32)[None]).astype(np.float16)
    return event, a


def save_norm(sae_dir, q, rate, n_pos, fallback, **meta):
    path = Path(sae_dir) / FILENAME
    np.savez(path, q=np.asarray(q, np.float32), rate=np.asarray(rate, np.float32), n_pos=np.asarray(n_pos, np.int64),
             fallback=np.asarray(fallback, bool), quantile=np.float64(meta.pop('quantile', QUANTILE)),
             meta=np.array(json.dumps(meta)))
    return path


def load_norm(sae_dir):
    """-> dict q, rate, n_pos, fallback, quantile, meta (dict)."""
    f = np.load(Path(sae_dir) / FILENAME, allow_pickle=False)
    out = {k: f[k] for k in ('q', 'rate', 'n_pos', 'fallback')}
    out['quantile'] = float(f['quantile'])
    out['meta'] = json.loads(str(f['meta']))
    return out
