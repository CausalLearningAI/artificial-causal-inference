"""
Matryoshka SAE with plain per-token TopK sparsity (Gao et al. 2024, arXiv 2406.04093), the threshold-free
alternative to src/eci/sae.py's BatchTopK.

Identical to MatryoshkaBatchTopKSAE (same parameters, init, Matryoshka prefix loss, AuxK dead-latent loss,
decoder normalisation, optimiser step src.eci.sae._train_step) except the sparsifier:
    training   z = exact top-k of relu(pre) PER TOKEN (k active latents on every token, fewer only if fewer than k
               pre-activations are positive), not the n*k largest across the batch;
    inference  the same per-token top-k. There is no global threshold: encode() ignores `mode`, so callers written
               for the BatchTopK model (e.g. vjepa.frame_max_tokens, which asks for mode='threshold') get per-token
               top-k. The parent's threshold EMA is still updated during training (smallest kept activation) and
               saved, but it is never used.

Functions / classes:
    MatryoshkaTopKSAE   the model
    load_sae_any        rebuild (sae, norm, ckpt) for either sparsity, from the checkpoint's 'sparsity' field
"""

import torch

from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm


class MatryoshkaTopKSAE(MatryoshkaBatchTopKSAE):
    sparsity = 'topk'

    def batch_topk(self, pre):
        """Per-token top-k of the post-ReLU activations (name kept: forward_train calls it). -> (z, kept values)."""
        acts = torch.relu(pre)
        v, i = acts.topk(self.k, dim=1, sorted=False)
        z = torch.zeros_like(acts).scatter_(1, i, v)
        return z, v.flatten()

    def encode(self, x, mode='topk'):
        """Exact per-token top-k for every `mode` (no threshold exists for this model)."""
        pre = torch.relu(self.pre_acts(x))
        v, i = pre.topk(self.k, dim=1, sorted=False)
        return torch.zeros_like(pre).scatter_(1, i, v)


def load_sae_any(path, device='cpu'):
    """As src.eci.sae.load_sae, but builds MatryoshkaTopKSAE when the checkpoint says sparsity == 'topk'."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cls = MatryoshkaTopKSAE if ck.get('sparsity') == 'topk' else MatryoshkaBatchTopKSAE
    sae = cls(**ck['hparams'])
    sae.load_state_dict(ck['state_dict'])
    norm = TokenNorm(ck['hparams']['d_in'])
    if ck['norm']['scale'].ndim == 1:  # per-block scale (TokenNorm.fit_blocks)
        norm.scale = torch.ones(ck['hparams']['d_in'])
    norm.load_state_dict(ck['norm'])
    return sae.to(device).eval(), norm.to(device), ck
