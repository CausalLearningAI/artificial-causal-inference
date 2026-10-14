"""
Matryoshka GATED sparse autoencoder with stochastic binary (hard-concrete / L0) gates: a threshold-free alternative to
src/eci/sae.py's BatchTopK, in which every latent outputs a BELIEF p in [0, 1] that its concept is present on a token.

References:
    L0 / hard concrete  Louizos, Welling, Kingma 2018, "Learning Sparse Neural Networks through L0 Regularization"
    Gated SAE           Rajamanoharan et al. 2024 (arXiv 2404.16014): shared encoder, separate gate / magnitude heads
    Matryoshka SAE      Bussmann et al. 2025 (arXiv 2503.17547)
    AuxK                Gao et al. 2024 (arXiv 2406.04093)

Model (x already normalized, see src.eci.sae.TokenNorm), per token x and latent j:
    proj_j = (x - b_dec) . W_enc[:, j]                     shared encoder direction (Rajamanoharan weight tying)
    g_j    = proj_j + b_gate_j                              gate logit
    p_j    = sigmoid(g_j)                                   BELIEF that latent j is on  (= P(gate_j > 0), see below)
    m_j    = relu(exp(r_mag_j) * proj_j + b_mag_j)          magnitude
    z_j    = gate_j * m_j,  x_hat = z @ W_dec + b_dec, rows of W_dec unit-norm

Training: gate_j ~ hard concrete with location log_alpha_j = g_j + beta * log(-gamma / zeta) (beta 2/3, gamma -0.1,
    zeta 1.1, Louizos et al.): u ~ U(0, 1), s = sigmoid((log u - log(1 - u) + log_alpha) / beta),
    gate = clip(s * (zeta - gamma) + gamma, 0, 1). The shift of log_alpha makes P(gate_j > 0) = sigmoid(g_j) = p_j
    EXACTLY, so the belief is the model's own probability that the latent is used, and the expected L0 is sum_j p_j.
    Loss = Matryoshka reconstruction (sum over prefixes of the MSE, as sae.py) + lambda * E[L0] (mean over tokens of
    sum_j p_j) + aux_coef * AuxK-like dead-latent loss.
lambda is NOT a tuned hyper-parameter: it is a Lagrange multiplier for the constraint E[L0] = target_l0 (label-free),
    updated by dual ascent in log space every step: log lambda += lam_lr * (E[L0]_batch - target) / target.
Inference (deterministic): p_j = sigmoid(g_j) is the belief; the code is z_j = 1[p_j > 0.5] * m_j (the gate is on iff
    it is more likely on than off). No threshold is estimated or tuned.

Dead-latent loss (AuxK-like): latents that have not been ON under the INFERENCE rule (g > 0 and m > 0) for `dead_tokens`
    training tokens reconstruct the residual with their top-k_aux gate logits; code p_j * m_j (aux_code 'belief') or
    m_j (aux_code 'magnitude' = Gao et al.'s AuxK: the dead latents' pre-activations).
L0 weighting: 'flat' (default) = lambda * sum_j p_j. 'nested' = sum over Matryoshka prefixes i of lambda * E[L0 of prefix i]
    (each nested sub-dictionary has its own rate-distortion objective MSE_i + lambda L0_i), i.e. latent j pays lambda times
    the number of reconstruction terms it enters. With 'flat', late-prefix latents pay the same price for 1 / 4 of the
    reconstruction value and die (gate16s seed 0: 93% / 56% / 20% / 2.5% alive in the four groups). The constraint is
    still on the unweighted E[L0].
Init: 'uniform' (default): W_enc = W_dec^T, b_gate = logit(target / n_latents) for every latent. init_selective(x):
    data-dependent, every latent a sharp detector of its own top target/n_latents share of tokens (see the method).
    With the uniform init, 92% of latents die in the first ~750 steps on the mice C1 tokens (gate16 seed 0: winners
    are fixed during warm-up and the 'belief' aux gradient, scaled by p ~ 0.004, cannot revive the rest).

Frame / video read-outs of the beliefs (functions below):
    noisy-OR      P_j(frame) = 1 - prod_i (1 - p_ij) over the frame's kept tokens; computed in log space as the
                  evidence E_j = -log(1 - P_j) = sum_i softplus(g_ij) (exact: -log(1 - sigmoid(g)) = softplus(g)),
                  which does not saturate and ranks frames identically to P_j.
    max           max_i p_ij  (= sigmoid(max_i g_ij)).
    temporal      per video: time budget = mean frame belief; bouts = maximal runs with frame belief > 0.5;
                  latency = index of the first frame with belief > 0.5.

Functions / classes:
    hard_concrete_sample        the relaxed Bernoulli gate (training)
    expected_gate, p_gate_one   E[gate | g] (tabulated quadrature) and P(gate = 1 | g) = sigmoid(g - 3.197)
    MatryoshkaGatedL0SAE        the model (gate_logits / beliefs / encode / decode / forward_train)
    frame_evidence_noisy_or     token logits -> per-frame noisy-OR evidence (sum of softplus), torch
    evidence_to_belief          E -> 1 - exp(-E)
    time_budget, bouts, latency temporal read-outs of one video's frame-belief series (numpy)
    save_gated, load_gated      checkpoint I/O
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src.eci.sae import TokenNorm

BETA, GAMMA, ZETA = 2.0 / 3.0, -0.1, 1.1
SHIFT = BETA * math.log(-GAMMA / ZETA)  # log_alpha = g + SHIFT  =>  P(gate > 0) = sigmoid(g)


_EG_LO, _EG_HI, _EG_STEP = -30.0, 30.0, 0.01
_EG_TABLE = None


def expected_gate(g):
    """E[gate | g] of the hard-concrete gate (no closed form): linear interpolation in a table over g in [-30, 30]
    (step 0.01) computed once by midpoint quadrature over u (8192 nodes). Diagnostic only (the 'mean' code)."""
    global _EG_TABLE
    if _EG_TABLE is None:
        u = (torch.arange(8192, dtype=torch.float64) + 0.5) / 8192
        gg = torch.arange(_EG_LO, _EG_HI + _EG_STEP / 2, _EG_STEP, dtype=torch.float64)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u))[None] / BETA + (gg[:, None] + SHIFT) / BETA)
        _EG_TABLE = (s * (ZETA - GAMMA) + GAMMA).clamp(0, 1).mean(1).float()
    t = _EG_TABLE.to(g.device)
    pos = ((g.float().clamp(_EG_LO, _EG_HI) - _EG_LO) / _EG_STEP)
    i0 = pos.floor().long().clamp_max(len(t) - 2)
    fr = pos - i0
    return t[i0] * (1 - fr) + t[i0 + 1] * fr


def p_gate_one(g):
    """P(gate == 1 | g) = sigmoid(g + SHIFT - beta * log((1 - gamma) / (zeta - 1)))."""
    return torch.sigmoid(g + SHIFT - BETA * math.log((1 - GAMMA) / (ZETA - 1)))


def hard_concrete_sample(g, beta=BETA, gamma=GAMMA, zeta=ZETA, eps=1e-6):
    """Relaxed-Bernoulli gate in [0, 1] with P(gate > 0) = sigmoid(g) (location shifted by beta * log(-gamma / zeta))."""
    u = torch.rand_like(g).clamp_(eps, 1 - eps)
    log_alpha = g + beta * math.log(-gamma / zeta)
    s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / beta)
    return (s * (zeta - gamma) + gamma).clamp(0.0, 1.0)


class MatryoshkaGatedL0SAE(nn.Module):
    sparsity = 'gated_l0'

    def __init__(self, d_in=768, n_latents=1024, prefixes=(128, 256, 512, 1024), target_l0=16.0, lam_init=1.0,
                 lam_lr=0.005, k_aux=512, aux_coef=1 / 32, dead_tokens=2_000_000, aux_code='belief', l0_weighting='flat', seed=0):
        super().__init__()
        prefixes = tuple(int(p) for p in prefixes)
        if list(prefixes) != sorted(prefixes) or prefixes[-1] != n_latents:
            raise ValueError(f'prefixes must be increasing and end at n_latents, got {prefixes}')
        self.d_in, self.n_latents, self.prefixes = d_in, n_latents, prefixes
        self.target_l0, self.lam_init, self.lam_lr = float(target_l0), float(lam_init), float(lam_lr)
        self.k_aux, self.aux_coef, self.dead_tokens = k_aux, aux_coef, dead_tokens
        self.k = int(round(target_l0))  # for callers that print sae.k
        if aux_code not in ('belief', 'magnitude'):
            raise ValueError(aux_code)
        self.aux_code = aux_code
        if l0_weighting not in ('flat', 'nested'):
            raise ValueError(l0_weighting)
        self.l0_weighting = l0_weighting
        # nested: latent j pays the L0 price once per Matryoshka reconstruction term it enters
        cw = torch.zeros(n_latents)
        for m in prefixes:
            cw[:m] += 1
        self.register_buffer('l0_cost', cw if l0_weighting == 'nested' else torch.ones(n_latents))

        g = torch.Generator().manual_seed(seed)
        W_dec = torch.randn(n_latents, d_in, generator=g)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.t().clone())  # decoder = encoder^T at init
        # gates start near the target sparsity: sigmoid(b) * n_latents = target_l0
        b0 = math.log(target_l0 / (n_latents - target_l0))
        self.b_gate = nn.Parameter(torch.full((n_latents,), b0))
        self.r_mag = nn.Parameter(torch.zeros(n_latents))
        self.b_mag = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer('log_lam', torch.tensor(math.log(lam_init)))
        self.register_buffer('tokens_since_fired', torch.zeros(n_latents, dtype=torch.long))
        self.register_buffer('threshold', torch.tensor(0.0))  # unused; kept so generic loggers / loaders do not break

    # ------------------------------------------------------------------ inference
    def _proj(self, x):
        return (x - self.b_dec) @ self.W_enc

    def _heads(self, x):
        proj = self._proj(x)
        return proj + self.b_gate, torch.relu(proj * torch.exp(self.r_mag) + self.b_mag)

    def gate_logits(self, x):
        return self._proj(x) + self.b_gate

    def beliefs(self, x):
        """Per-token belief p (n, n_latents) in [0, 1]."""
        return torch.sigmoid(self.gate_logits(x))

    def encode(self, x, mode='hard'):
        """Deterministic code. 'hard' (default; also any other mode, e.g. 'threshold' from BatchTopK callers):
        z = 1[p > 0.5] * m, no threshold to estimate. 'mean' (diagnostic): z = E[gate | g] * m (dense, not sparse)."""
        g, m = self._heads(x)
        if mode == 'mean':
            return m * expected_gate(g)
        return m * (g > 0)

    def decode(self, z, m=None):
        m = self.n_latents if m is None else m
        return z[:, :m] @ self.W_dec[:m] + self.b_dec

    @torch.no_grad()
    def init_selective(self, x, proj_std=5.0):
        """Data-dependent SELECTIVE init (call after b_dec is set; x = normalized init tokens, e.g. 65,536): every latent
        starts as a sharp detector of its own top tokens. Encoder columns are rescaled so that proj_j has std proj_std
        on x; b_gate_j = -(the 1 - target_l0 / n_latents quantile of proj_j), so the deterministic L0 on x equals
        target_l0 (each latent on for the same share of tokens); r_mag = -log(scale) keeps the magnitude head at the
        unit-norm projection (as the BatchTopK init). Returns {'det_l0', 'E_L0'} on x."""
        proj = (x - self.b_dec) @ self.W_enc
        c = proj_std / proj.std(0).clamp_min(1e-6)
        self.W_enc.mul_(c)
        self.r_mag.copy_(-torch.log(c))
        proj = proj * c
        k = max(1, int(round((1 - self.target_l0 / self.n_latents) * x.shape[0])))
        self.b_gate.copy_(-proj.kthvalue(k, dim=0).values)
        g = proj + self.b_gate
        return {'det_l0': float((g > 0).sum(1).float().mean()), 'E_L0': float(torch.sigmoid(g).sum(1).mean())}

    # ------------------------------------------------------------------ training
    @property
    def lam(self):
        return float(self.log_lam.exp())

    def forward_train(self, x):
        g, mag = self._heads(x)
        gate = hard_concrete_sample(g)
        z = gate * mag

        recon = self.b_dec.expand_as(x)
        losses, start = [], 0
        for m in self.prefixes:
            recon = recon + z[:, start:m] @ self.W_dec[start:m]
            losses.append((recon - x).pow(2).sum(1).mean())
            start = m
        rec = torch.stack(losses).sum()
        p = torch.sigmoid(g)
        el0 = p.sum(1).mean()
        lam = self.log_lam.exp()
        loss = rec + lam * (p * self.l0_cost).sum(1).mean()

        with torch.no_grad():
            fired = ((g > 0) & (mag > 0)).any(0)  # fired = ON under the deterministic inference rule
            self.tokens_since_fired += x.shape[0]
            self.tokens_since_fired[fired] = 0
            # dual ascent on the Lagrange multiplier of E[L0] = target (log space keeps lambda > 0)
            self.log_lam.add_(self.lam_lr * (float(el0) - self.target_l0) / self.target_l0).clamp_(-12.0, 12.0)

        dead = self.tokens_since_fired >= self.dead_tokens
        n_dead = int(dead.sum())
        aux = torch.zeros((), device=x.device)
        if n_dead > 0:
            k_aux = min(self.k_aux, n_dead)
            resid = (x - recon).detach()
            gd = g[:, dead]
            v, i = gd.topk(k_aux, dim=1)
            vals = mag[:, dead].gather(1, i)
            if self.aux_code == 'belief':
                vals = torch.sigmoid(v) * vals
            z_aux = torch.zeros_like(gd).scatter_(1, i, vals)
            aux = (z_aux @ self.W_dec[dead] - resid).pow(2).sum(1).mean()
            if torch.isfinite(aux):
                loss = loss + self.aux_coef * aux

        var = (x - x.mean(0)).pow(2).sum(1).mean()
        logs = {'loss': float(loss), 'rec': float(rec), 'aux': float(aux), 'n_dead': n_dead,
                'fve': float(1 - losses[-1].detach() / var), 'fve_first': float(1 - losses[0].detach() / var),
                'l0': float((z > 0).sum(1).float().mean()), 'el0': float(el0),
                'l0_det': float(((g > 0) & (mag > 0)).sum(1).float().mean()), 'lam': float(lam),
                'threshold': float(lam)}
        return loss, logs

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def remove_parallel_grad(self):
        if self.W_dec.grad is not None:
            W = self.W_dec.data
            self.W_dec.grad -= (self.W_dec.grad * W).sum(1, keepdim=True) * W


# ---------------------------------------------------------------------------------------------- read-outs
def frame_evidence_noisy_or(g, frame, n_frames, out=None):
    """g (n, m) token gate logits, frame (n,) long frame ids -> (n_frames, m) evidence sum_i softplus(g_ij), added into
    `out` if given (so chunks can be accumulated). Noisy-OR frame belief = 1 - exp(-evidence)."""
    if out is None:
        out = torch.zeros(n_frames, g.shape[1], dtype=torch.float32, device=g.device)
    out.index_add_(0, frame, Fn.softplus(g.float()))
    return out


def evidence_to_belief(e):
    return -torch.expm1(-e) if torch.is_tensor(e) else -np.expm1(-np.asarray(e))


def time_budget(belief):
    """Mean frame belief of one video (expected fraction of frames with the concept)."""
    return float(np.mean(belief))


def bouts(belief, thr=0.5):
    """Maximal runs with belief > thr -> list of (start, end_exclusive)."""
    on = np.r_[False, np.asarray(belief) > thr, False].astype(np.int8)
    d = np.diff(on)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def latency(belief, thr=0.5):
    """Index of the first frame with belief > thr, or None."""
    i = np.flatnonzero(np.asarray(belief) > thr)
    return int(i[0]) if len(i) else None


# ---------------------------------------------------------------------------------------------- checkpoint
def save_gated(path, sae, norm, extra=None):
    torch.save({'state_dict': sae.state_dict(), 'norm': norm.state_dict(), 'sparsity': 'gated_l0',
                'hparams': {'d_in': sae.d_in, 'n_latents': sae.n_latents, 'prefixes': list(sae.prefixes),
                            'target_l0': sae.target_l0, 'lam_init': sae.lam_init, 'lam_lr': sae.lam_lr,
                            'k_aux': sae.k_aux, 'aux_coef': sae.aux_coef, 'dead_tokens': sae.dead_tokens,
                            'aux_code': sae.aux_code, 'l0_weighting': sae.l0_weighting},
                **(extra or {})}, path)


def load_gated(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck.get('sparsity') == 'gated_l0', ck.get('sparsity')
    sae = MatryoshkaGatedL0SAE(**ck['hparams'])
    sae.load_state_dict(ck['state_dict'])
    norm = TokenNorm(ck['hparams']['d_in'])
    if ck['norm']['scale'].ndim == 1:
        norm.scale = torch.ones(ck['hparams']['d_in'])
    norm.load_state_dict(ck['norm'])
    return sae.to(device).eval(), norm.to(device), ck
