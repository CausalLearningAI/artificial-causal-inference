"""
Per-token heat maps of one SAE neuron (ECI v2).

Two maps per frame, both real per-token quantities (no attention, no smoothing; bilinear upsampling happens only when
the map is drawn):

  activation map   a_t = the neuron's code at token t (src/eci/sae.py encode, threshold mode). The frame value under
                   the max readout is max_t a_t. Without context tokens this is the token's own contribution.

  contribution map H_s = where the evidence of the firing tokens comes from. The neuron's pre-activation is linear in
                   the SAE input x_t = [z_t, c_t] (TokenNorm is a per-dimension affine map):
                       pre_j(t) = C_j + (z_t - mu_z) . alpha_j + (c_t - mu_c) . beta_j,
                       alpha_j = scale_z * W_enc[:d, j], beta_j = scale_c * W_enc[d:, j], C_j = b_enc_j - b_dec . W_enc[:, j]
                   and c_t = (1 / n_t) sum_{s in window(t), s != t} z_s, so it splits EXACTLY into one term per source
                   token: psi_t(t) = (z_t - mu_z) . alpha_j (own term) and psi_t(s) = (z_s - mu_c) . beta_j / n_t
                   (neighbour terms; when n_t = 0 the context is 0 and only the constant changes). Each firing token's
                   activation a_t is shared among its sources in proportion to their positive terms, and the shares
                   are summed over the firing (readout) tokens:
                       H_s = sum_{t: a_t > 0, t readout} a_t [psi_t(s)]_+ / sum_{s'} [psi_t(s')]_+ .
                   sum_s H_s = sum_t a_t (mass preserving). Without context H = a exactly (only own terms).
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.eci.v2.context import context_of, neighbour_count


def neuron_maps(sae, norm, z, pos, G, w, j, readout=None):
    """One frame. z (n, d) float kept tokens, pos (n,) long grid positions, w context window (0 = none), j neuron,
    readout (n,) bool tokens that the frame readout uses (None = all) -> (a (G*G,), H (G*G,)) float32 numpy."""
    dev = z.device
    n, d = z.shape
    fl = torch.zeros(n, dtype=torch.long, device=dev)
    x = z.float() if not w else torch.cat([z.float(), context_of(z, pos, fl, 1, G, w)], 1)
    with torch.no_grad():
        a = sae.encode(norm(x), mode='threshold')[:, j]
    rd = torch.ones(n, dtype=torch.bool, device=dev) if readout is None else readout
    A = torch.zeros(G * G, device=dev)
    A[pos] = a
    W = sae.W_enc[:, j].float()
    sc = norm.scale.float() if norm.scale.ndim else norm.scale.float().expand(x.shape[1])
    mu = norm.mean.float()
    alpha = sc[:d] * W[:d]
    own = (z.float() - mu[:d]) @ alpha  # (n,)
    H = torch.zeros(G * G, device=dev)
    fire = torch.nonzero((a > 0) & rd).flatten()
    if not w:
        H[pos[fire]] = a[fire]
        return A.cpu().numpy(), H.cpu().numpy()
    beta = sc[d:] * W[d:]
    nb = (z.float() - mu[d:]) @ beta  # (n,) neighbour term before / n_t
    nt = neighbour_count(pos, fl, 1, G, w)
    r = w // 2
    gy, gx = pos // G, pos % G
    cell = torch.full((G * G,), -1, dtype=torch.long, device=dev)
    cell[pos] = torch.arange(n, device=dev)
    for t in fire.tolist():
        terms = {t: float(own[t])}
        if nt[t] > 0:
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    yy, xx = int(gy[t]) + dy, int(gx[t]) + dx
                    if (dy == 0 and dx == 0) or not (0 <= yy < G and 0 <= xx < G):
                        continue
                    s = int(cell[yy * G + xx])
                    if s >= 0:
                        terms[s] = float(nb[s]) / float(nt[t])
        pos_sum = sum(max(v, 0.0) for v in terms.values())
        if pos_sum <= 0:
            continue
        at = float(a[t])
        for s, v in terms.items():
            if v > 0:
                H[pos[s]] += at * v / pos_sum
    return A.cpu().numpy(), H.cpu().numpy()


def locality(h, fdark, thr_animal=0.25):
    """Map h (G*G,) >= 0, animal-pixel fraction per cell fdark (G*G,) -> spread / placement numbers."""
    m = float(h.max())
    if m <= 0:
        return {'n_half_max': 0, 'peak_fdark': np.nan, 'mass_animal_px': np.nan, 'mass_on_animal_tokens': np.nan}
    hi = h >= 0.5 * m
    return {'n_half_max': int(hi.sum()), 'peak_fdark': float(fdark[int(np.argmax(h))]),
            'mass_animal_px': float((h * fdark).sum() / h.sum()),
            'mass_on_animal_tokens': float(h[fdark >= thr_animal].sum() / h.sum())}
