"""MLP probe for PPCI.

One binary output neuron per outcome column (independent sigmoid probes).
Aggregation across outcomes (or / sum) is handled at evaluation time, not here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Featurizer + linear head with one sigmoid output per outcome.

    Args:
        input_dim:     Embedding dimension.
        hidden_dim:    Width of each hidden layer.
        hidden_layers: Number of hidden layers (0 → linear probe, no featurizer).
        n_outcomes:    Number of binary output neurons.
        dropout:       Dropout probability after each hidden layer (0 = disabled).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        n_outcomes: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_outcomes = n_outcomes

        # --- featurizer ---
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.featurizer = nn.Sequential(*layers) if layers else nn.Identity()
        feat_dim = hidden_dim if hidden_layers > 0 else input_dim

        # --- output head ---
        self.head = nn.Linear(feat_dim, n_outcomes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def features(self, X: torch.Tensor) -> torch.Tensor:
        """Return featurizer representation (before head)."""
        return self.featurizer(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Raw logits.  Shape: (N, n_outcomes) or (N,) when n_outcomes=1."""
        logits = self.head(self.features(X))
        return logits.squeeze(-1) if self.n_outcomes == 1 else logits

    def probs(self, X: torch.Tensor) -> torch.Tensor:
        """Sigmoid probabilities. Same shape as forward()."""
        return self.forward(X).sigmoid()

    def pred(self, X: torch.Tensor) -> torch.Tensor:
        """Hard binary predictions (rounded probabilities)."""
        return self.probs(X).round()
