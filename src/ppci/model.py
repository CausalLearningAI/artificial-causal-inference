"""MLP probe for PPCI.

One binary output neuron per outcome column (independent sigmoid probes).
Aggregation across outcomes (or / sum) is handled at evaluation time, not here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Cross-attention aggregator for a temporal context window.

    Takes a flat ``(B, context_size * embed_dim)`` input (as produced by
    ``apply_context_window(..., mode='concat')``), reshapes it to
    ``(B, context_size, embed_dim)``, and uses the *center frame* as a query
    over all positions.  The output is a ``(B, embed_dim)`` vector — same
    dimension as a single frame — so the downstream MLP featurizer is
    unchanged.

    Architecture
    ------------
    q  = W_q · x_center          (B, head_dim)
    k  = W_k · x_all             (B, context_size, head_dim)
    α  = softmax(q · kᵀ / √d)   (B, context_size)
    out = Σ_i α_i · x_i + x_center   residual from center frame
    return LayerNorm(out)

    Parameters: 2 × embed_dim × head_dim  (e.g. 2 × 768 × 64 = 98 K)
    Compare with naive concat first layer: (2k+1) × embed_dim × hidden_dim
    e.g. 5 × 768 × 512 = 1.97 M  →  ~20× fewer params.
    """

    def __init__(self, embed_dim: int, context_size: int, head_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.center = context_size // 2
        self.scale = head_dim ** -0.5

        self.W_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, head_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, context_size * embed_dim)`` — flat concat from dataset.
        Returns:
            ``(B, embed_dim)``
        """
        B = x.shape[0]
        x = x.view(B, self.context_size, self.embed_dim)   # (B, C, D)
        center = x[:, self.center]                          # (B, D)

        q = self.W_q(center).unsqueeze(1)                  # (B, 1, head_dim)
        k = self.W_k(x)                                    # (B, C, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale      # (B, 1, C)
        attn = attn.softmax(dim=-1)                        # (B, 1, C)

        out = (attn @ x).squeeze(1)                        # (B, D)
        return self.norm(out + center)                     # residual + norm


class MLP(nn.Module):
    """Featurizer + linear head with one sigmoid output per outcome.

    When ``context_size > 1`` a :class:`TemporalAttention` front-end is
    prepended that collapses the ``(2k+1) × embed_dim`` context window back
    to ``embed_dim`` before the standard MLP layers.

    Args:
        input_dim:     Total input dimension.  For a context window with
                       ``concat`` mode this is ``context_size × embed_dim``;
                       otherwise it equals ``embed_dim``.
        hidden_dim:    Width of each hidden layer.
        hidden_layers: Number of hidden layers (0 → linear probe).
        n_outcomes:    Number of binary output neurons.
        dropout:       Dropout probability after each hidden layer (0 = off).
        context_size:  Number of frames in the temporal window (default 1 =
                       no context).  When > 1 a TemporalAttention module is
                       added before the featurizer.
        context_head_dim: Key/query projection size for TemporalAttention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        n_outcomes: int = 1,
        dropout: float = 0.0,
        context_size: int = 1,
        context_head_dim: int = 64,
    ):
        super().__init__()
        self.n_outcomes = n_outcomes
        self.temperature: float = 1.0

        # --- optional temporal aggregation front-end ---
        if context_size > 1:
            embed_dim = input_dim // context_size
            self.temporal_agg: nn.Module = TemporalAttention(
                embed_dim, context_size, head_dim=context_head_dim
            )
            mlp_in = embed_dim
        else:
            self.temporal_agg = nn.Identity()
            mlp_in = input_dim

        # --- featurizer ---
        layers: list[nn.Module] = []
        in_dim = mlp_in
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.featurizer = nn.Sequential(*layers) if layers else nn.Identity()
        feat_dim = hidden_dim if hidden_layers > 0 else mlp_in

        # --- output head ---
        self.head = nn.Linear(feat_dim, n_outcomes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def features(self, X: torch.Tensor) -> torch.Tensor:
        """Return featurizer representation (before head)."""
        return self.featurizer(self.temporal_agg(X))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Raw logits.  Shape: (N, n_outcomes) or (N,) when n_outcomes=1."""
        logits = self.head(self.features(X))
        return logits.squeeze(-1) if self.n_outcomes == 1 else logits

    def probs(self, X: torch.Tensor) -> torch.Tensor:
        """Calibrated sigmoid probabilities. Same shape as forward()."""
        return (self.forward(X) / self.temperature).sigmoid()

    def pred(self, X: torch.Tensor) -> torch.Tensor:
        """Hard binary predictions (rounded probabilities)."""
        return self.probs(X).round()
