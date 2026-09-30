"""
SOMP spatial aggregation: one sparse vector over the SAE latents per frame (ECI).

Simultaneous Orthogonal Matching Pursuit (Tropp, Gilbert, Strauss 2006), following
ResiDual's `somp` (github.com/Flegyas/ResiDual, src/residual/sparse_decomposition.py, commit 004b0aa,
criterion 'l1'). Per frame, X (n, d) = the frame's foreground tokens; D (m, d) = dictionary.
    for i = 1..K:
        cross   = R @ D^T                          (n, m) correlation of every residual token with every atom
        score_j = sum_t |cross_tj|  (chosen atoms excluded)
        j_i     = argmax score                     one atom for ALL tokens of the frame
        C       = argmin_C ||X - C D_S||_F          least-squares refit on all chosen atoms (orthogonal projection)
        R       = X - C D_S
    importance_j = ||C[:, j]||_2 / sqrt(n)           (RMS of the atom's coefficient over the frame's tokens)
The reference returns ||C[:, j]||_2 (weights = lstsq_weights.norm(dim=1)); with a variable number of
tokens per frame we divide by sqrt(n) so a frame's value does not grow with its foreground size
(the ranking of atoms inside a frame is unchanged; the reference value = importance * sqrt(n_fg)).

Choices for the SAE setting (see the module docstring of scripts/eci/somp_encode_all.py):
    dictionary  the SAE decoder rows W_dec (unit norm): the SAE's generative model is
                x_norm = b_dec + z @ W_dec, so SOMP explains X = norm(token) - b_dec with the same atoms.
    centering   no per-frame token mean (ResiDual's centering=True); the SAE's own offset b_dec is
                removed instead (per-frame centering would delete what all foreground tokens share, and a
                frame's mean over 1 token is the token itself).
    stopping    K atoms (fixed); atoms picked after the residual energy fell below tol * ||X||^2 (frame
                already explained, e.g. fewer tokens than K) get importance 0 and index -1.

Functions:
    somp_batched    padded batch of frames (B, n, d) + mask, on the GPU
    somp_reference  one frame, numpy float64, a direct loop transcription of ResiDual's somp (tests)
    sae_dictionary  (X space, D) from an SAE and its TokenNorm
"""

import numpy as np
import torch


@torch.no_grad()
def somp_batched(X, mask, D, K, tol=1e-6, gram=None):
    """X (B, n, d) float, mask (B, n) bool (True = token of the frame), D (m, d).
    Returns dict:
        idx         (B, K) long, chosen atoms in selection order (-1 after the frame is explained)
        coef        (B, n, K) float64 least-squares coefficients of the final refit (0 on padded rows)
        importance  (B, K) float64 RMS over the frame's tokens of each chosen atom's coefficient
        energy      (B,) float64 sum_t ||x_t||^2 over the frame's tokens
        resid       (B, K) float64 residual energy ||X - C D_S||^2 after each step
    The correlations X @ D^T are computed once in X's dtype; the refits in float64."""
    B, n, d = X.shape
    m = D.shape[0]
    if K > m:
        raise ValueError(f'K={K} > dictionary size {m}')
    dev = X.device
    maskf = mask.to(X.dtype)[..., None]
    X = X * maskf
    D = D.to(X.dtype)
    XD = X @ D.T  # (B, n, m); padded rows are 0
    G64 = (D.double() @ D.double().T) if gram is None else gram.to(dev, torch.float64)
    G = G64.to(X.dtype)
    energy = (X.double() ** 2).sum((1, 2))
    idx = torch.zeros(B, K, dtype=torch.long, device=dev)
    chosen = torch.zeros(B, m, dtype=torch.bool, device=dev)
    resid = torch.zeros(B, K, dtype=torch.float64, device=dev)
    ar = torch.arange(B, device=dev)
    C = None
    for i in range(K):
        cross = XD if i == 0 else XD - torch.bmm(C.to(X.dtype), G[idx[:, :i]])  # (B, n, m)
        score = cross.abs().sum(1)
        score[chosen] = -1.0
        j = score.argmax(1)
        idx[:, i] = j
        chosen[ar, j] = True
        S = idx[:, :i + 1]
        GS = G64[S[:, :, None], S[:, None, :]]  # (B, i+1, i+1)
        BS = XD.gather(2, S[:, None, :].expand(B, n, i + 1)).double()  # (B, n, i+1) = X D_S^T
        C = torch.linalg.solve(GS, BS.transpose(1, 2)).transpose(1, 2)  # (B, n, i+1)
        resid[:, i] = energy - (C * BS).sum((1, 2))
    n_tok = mask.sum(1).double()
    importance = C.pow(2).sum(1).sqrt() / n_tok.clamp_min(1)[:, None].sqrt()
    # atoms added after the frame was already explained (residual <= tol * energy)
    prev = torch.cat([energy[:, None], resid[:, :-1]], 1)
    late = prev <= tol * energy[:, None]
    importance[late] = 0.0
    idx = idx.masked_fill(late, -1)
    return {'idx': idx, 'coef': C * mask.double()[..., None], 'importance': importance, 'energy': energy,
            'resid': resid.clamp_min(0)}


def somp_reference(X, D, K):
    """One frame, float64 numpy loop, as ResiDual's somp(criterion='l1', centering=False):
    cross = R D^T; chosen atoms scored -1 (ResiDual multiplies by 0; equal unless the residual is 0);
    lstsq refit on the chosen atoms. Returns (chosen list, W (K, n) coefficients, weights = row norms of W,
    residual energy after each step)."""
    X = np.asarray(X, np.float64)
    D = np.asarray(D, np.float64)
    chosen, notchosen = [], np.ones(D.shape[0], bool)
    R, res = X.copy(), []
    W = None
    for _ in range(K):
        cross = R @ D.T
        score = np.abs(cross).sum(0)
        score[~notchosen] = -1.0
        j = int(score.argmax())
        chosen.append(j)
        notchosen[j] = False
        A = D[chosen]  # (i, d)
        W = np.linalg.lstsq(A.T, X.T, rcond=None)[0]  # (i, n)
        R = X - (A.T @ W).T
        res.append(float((R ** 2).sum()))
    return chosen, W, np.linalg.norm(W, axis=1), np.array(res)


@torch.no_grad()
def sae_dictionary(sae, norm):
    """-> (to_x, D): to_x(tokens) = norm(tokens) - b_dec (float32), the space in which the SAE decoder
    reconstructs; D = W_dec (m, d) float32, unit-norm rows."""
    b = sae.b_dec.detach().float()

    def to_x(tok):
        return norm(tok) - b
    return to_x, sae.W_dec.detach().float()
