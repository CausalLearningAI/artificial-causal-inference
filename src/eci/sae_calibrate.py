"""
Post-hoc calibration of the inference threshold of a Matryoshka BatchTopK SAE (src/eci/sae.py).

Why: at inference the BatchTopK SAE uses z = pre * 1[pre > theta] with ONE global theta. During training theta is an
exponential moving average (rate threshold_lr = 0.01) of the smallest activation BatchTopK kept in each batch. The rate
is an arbitrary choice, and the EMA tracks a moving model. This module replaces it with a deterministic, rate-free
rule defined by the sparsity k the model was trained with.

Estimator (calibrate_threshold):
    Given N held-out TRAINING tokens (normalized as the SAE expects) and a dictionary of m latents, pool all N * m
    post-ReLU pre-activations a = relu((x - b_dec) @ W_enc + b_enc). With r = N * k, let v_(1) >= v_(2) >= ... be the
    pooled values sorted in decreasing order. Then

        theta* = v_(r + 1)        (the largest value that BatchTopK on the whole calibration set would DROP)

    i.e. the empirical (1 - k / m) quantile of the pooled values (lower convention). Under the inference rule
    'pre > theta*' exactly r values exceed theta* when v_(r) > v_(r + 1) (no tie at the cut), so the mean number of
    active latents per calibration token is exactly k. If fewer than r + 1 pooled values are positive, theta* = 0 and
    every positive value is kept (mean L0 < k; reported).
    This is BatchTopK applied once to the entire calibration set instead of an EMA over training batches.

Computation (exact, three streaming passes over the tokens, no N * m buffer):
    1  maximum pooled value M and number of positive values
    2  histogram of the positive values in n_bins equal bins over (0, M]
    3  the values falling in the bin that contains rank r + 1 are gathered and sorted; theta* is read off exactly.
Pre-activations are computed in fp32 with TF32 matmuls off, so the three passes see bit-identical values (asserted
via the bin counts).

Functions:
    calibrate_threshold   theta* and diagnostics (calibration L0 under theta* and under the SAE's current theta)
    sparsity_stats        mean L0 per token, FVE (sum over dims of residual SS / total SS) for a given theta
    set_threshold         write theta into sae.threshold (in place)
"""

import numpy as np
import torch


def _batches(tokens, norm, batch, device):
    """Yield normalized fp32 batches of `tokens` ((N, d) numpy or torch, any dtype/device) on `device`."""
    N = tokens.shape[0]
    for a in range(0, N, batch):
        x = tokens[a:a + batch]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(x))
        x = x.to(device, non_blocking=True)
        yield norm(x) if norm is not None else x.float()


@torch.no_grad()
def _acts(sae, x):
    return torch.relu(sae.pre_acts(x))


@torch.no_grad()
def calibrate_threshold(sae, tokens, k=None, norm=None, batch=32768, n_bins=1 << 20):
    """theta* such that the mean L0 per token on `tokens` under 'pre > theta*' is exactly k (see module docstring).

    sae     MatryoshkaBatchTopKSAE (its device is used)
    tokens  (N, d_in) held-out training tokens: RAW if `norm` is given (TokenNorm applied here), else normalized
    k       target mean active latents per token (default: sae.k, the training sparsity)
    -> (theta_star: float, info: dict)
    """
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        dev = sae.W_dec.device
        k = int(sae.k if k is None else k)
        N, m = int(tokens.shape[0]), int(sae.n_latents)
        r = N * k  # number of values to keep; theta* = the (r + 1)-th largest
        # pass 1: max and number of positives
        vmax, n_pos = torch.zeros((), device=dev), 0
        for x in _batches(tokens, norm, batch, dev):
            a = _acts(sae, x)
            vmax = torch.maximum(vmax, a.max())
            n_pos += int((a > 0).sum())
        M = float(vmax)
        info = {'estimator': 'theta* = (N*k + 1)-th largest pooled post-ReLU pre-activation over N held-out train '
                             'tokens x m latents = empirical (1 - k/m) quantile (lower)',
                'n_tokens': N, 'n_latents': m, 'k': k, 'rank_r_plus_1': r + 1, 'n_pooled': N * m,
                'n_positive': n_pos, 'max_value': M, 'quantile_level': 1 - k / m}
        if n_pos <= r:
            info.update(theta_star=0.0, note='fewer than N*k + 1 positive values: every positive value kept')
            theta = 0.0
        else:
            # pass 2: histogram of positive values over (0, M]
            scale = n_bins / M

            def bins_of(a):
                return (a * scale).long().clamp_(max=n_bins - 1)

            counts = torch.zeros(n_bins, dtype=torch.long, device=dev)
            for x in _batches(tokens, norm, batch, dev):
                a = _acts(sae, x)
                a = a[a > 0]
                counts += torch.bincount(bins_of(a), minlength=n_bins)
            assert int(counts.sum()) == n_pos, (int(counts.sum()), n_pos)
            above = torch.flip(torch.cumsum(torch.flip(counts, [0]), 0), [0])  # above[b] = # values in bins >= b
            # bin b* = the highest bin with above[b] >= r + 1 (rank r + 1 lies in it)
            b_star = int(torch.nonzero(above >= r + 1).max())
            n_higher = int(above[b_star + 1]) if b_star + 1 < n_bins else 0  # values in bins > b*
            # pass 3: gather the values of bin b*
            vals = []
            for x in _batches(tokens, norm, batch, dev):
                a = _acts(sae, x)
                a = a[a > 0]
                vals.append(a[bins_of(a) == b_star])
            vals = torch.cat(vals)
            assert vals.numel() == int(counts[b_star]), (vals.numel(), int(counts[b_star]))
            vals = vals.sort(descending=True).values
            j = r + 1 - n_higher - 1  # 0-based position of rank r + 1 inside the bin
            theta = float(vals[j])
            v_r = float(vals[j - 1]) if j >= 1 else None  # the r-th largest when it lies in the same bin
            info.update(theta_star=theta, bin=b_star, bin_width=M / n_bins, n_in_bin=int(vals.numel()),
                        n_above_bin=n_higher, v_r=v_r, tie_at_cut=bool(v_r is not None and v_r == theta))
        # check: calibration L0 under theta* (and under the SAE's current threshold)
        cur = float(sae.threshold)
        cnt_star, cnt_cur = 0, 0
        for x in _batches(tokens, norm, batch, dev):
            a = _acts(sae, x)
            cnt_star += int((a > theta).sum())
            cnt_cur += int((a > cur).sum())
        info.update(l0_calib_theta_star=cnt_star / N, l0_calib_theta_current=cnt_cur / N, theta_current=cur,
                    n_active_calib_theta_star=cnt_star)
        return theta, info
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


@torch.no_grad()
def sparsity_stats(sae, tokens, theta, norm=None, batch=32768):
    """Mean L0 per token and FVE of the full dictionary under 'pre > theta' on `tokens`. FVE = 1 - sum over dims of
    residual SS / sum over dims of total SS about the tokens' mean (as vjepa.frame_max_tokens)."""
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        dev = sae.W_dec.device
        N = int(tokens.shape[0])
        l0, s1, s2, sse = 0, 0, 0, 0
        for x in _batches(tokens, norm, batch, dev):
            a = _acts(sae, x)
            z = a * (a > theta)
            xh = sae.decode(z)
            l0 += int((z > 0).sum())
            s1 = s1 + x.double().sum(0)
            s2 = s2 + x.double().pow(2).sum(0)
            sse = sse + (x - xh).double().pow(2).sum(0)
        tss = s2 - s1.pow(2) / N
        return {'n_tokens': N, 'theta': float(theta), 'l0': l0 / N, 'fve': float(1 - sse.sum() / tss.sum())}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


@torch.no_grad()
def set_threshold(sae, theta):
    """In place: the SAE's inference threshold becomes theta (encode(mode='threshold') then uses it)."""
    sae.threshold.fill_(float(theta))
    return sae
