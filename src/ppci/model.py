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

    When ``n_dist > 0`` the last *n_dist* features of the (post-temporal-agg)
    input are treated as distance scalars.  They bypass the featurizer and are
    concatenated with the featurizer output before the head (**late fusion**).

    Args:
        input_dim:     Total input dimension (including distances if any).
        hidden_dim:    Width of each hidden layer.
        hidden_layers: Number of hidden layers (0 → linear probe).
        n_outcomes:    Number of binary output neurons.
        dropout:       Dropout probability after each hidden layer (0 = off).
        context_size:  Number of frames in the temporal window (default 1 =
                       no context).  When > 1 a TemporalAttention module is
                       added before the featurizer.
        context_head_dim: Key/query projection size for TemporalAttention.
        n_dist:        Number of trailing distance features for late fusion
                       (default 0 = no late fusion, all features go to featurizer).
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
        n_dist: int = 0,
    ):
        super().__init__()
        self.n_outcomes = n_outcomes
        self.temperature: float = 1.0
        self.n_dist = n_dist

        # --- optional temporal aggregation front-end ---
        if context_size > 1:
            embed_dim = input_dim // context_size
            self.temporal_agg: nn.Module = TemporalAttention(
                embed_dim, context_size, head_dim=context_head_dim
            )
            mlp_in = embed_dim - n_dist
        else:
            self.temporal_agg = nn.Identity()
            mlp_in = input_dim - n_dist

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

        # --- output head (with late-fused distances if any) ---
        self.head = nn.Linear(feat_dim + n_dist, n_outcomes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def features(self, X: torch.Tensor) -> torch.Tensor:
        """Return featurizer representation (before head, without distances)."""
        x = self.temporal_agg(X)
        x_emb = x[:, : x.shape[1] - self.n_dist] if self.n_dist > 0 else x
        return self.featurizer(x_emb)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Raw logits.  Shape: (N, n_outcomes) or (N,) when n_outcomes=1."""
        x = self.temporal_agg(X)
        if self.n_dist > 0:
            x_emb = x[:, :-self.n_dist]
            x_dist = x[:, -self.n_dist:]
            feat = self.featurizer(x_emb)
            logits = self.head(torch.cat([feat, x_dist], dim=1))
        else:
            logits = self.head(self.featurizer(x))
        return logits.squeeze(-1) if self.n_outcomes == 1 else logits

    def probs(self, X: torch.Tensor) -> torch.Tensor:
        """Calibrated sigmoid probabilities. Same shape as forward()."""
        return (self.forward(X) / self.temperature).sigmoid()

    def pred(self, X: torch.Tensor) -> torch.Tensor:
        """Hard binary predictions (rounded probabilities)."""
        return self.probs(X).round()


class SiameseMLP(nn.Module):
    """MLP for POV mode with optional Siamese architecture and late-fusion distances.

    Input layout per half: ``[embedding(D) | distances(n_dist)]``.
    Full input: ``[half_blue | half_yellow]`` = ``(N, 2 * (D + n_dist))``.

    **Late fusion**: distances are separated from embeddings before the trunk
    and concatenated with the trunk output before the head.  This prevents
    the small distance signal (2 dims) from being drowned by the embedding
    (768 dims) in the first layer.

    ``siamese=True``  (default): shared trunk processes each half's embedding
        independently (colour-invariant features).  Two separate output heads
        predict P(B2F) and P(Y2F) from ``[trunk_out | distances]``.
    ``siamese=False``: single trunk processes ``[emb_blue | emb_yellow]``
        jointly (can learn cross-ant interactions).  Single head predicts
        both outcomes from ``[trunk_out | all_distances]``.

    Interface is identical to :class:`MLP`: ``forward → (N, 2)``,
    ``probs → (N, 2)``, so the training loop needs no changes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        dropout: float = 0.0,
        context_size: int = 1,
        context_head_dim: int = 64,
        siamese: bool = True,
        n_dist: int = 2,
    ):
        super().__init__()
        self.n_outcomes = 2
        self.temperature: float = 1.0
        self.siamese = siamese

        # --- optional temporal aggregation (operates on full input) ---
        if context_size > 1:
            embed_dim = input_dim // context_size
            self.temporal_agg: nn.Module = TemporalAttention(
                embed_dim, context_size, head_dim=context_head_dim
            )
            full_dim = embed_dim
        else:
            self.temporal_agg = nn.Identity()
            full_dim = input_dim

        assert full_dim % 2 == 0, (
            f"POV input dim after temporal agg must be even, got {full_dim}"
        )
        self.half_dim = full_dim // 2
        self.n_dist = n_dist
        self.emb_dim = self.half_dim - n_dist

        # --- trunk ---
        trunk_in = self.emb_dim if siamese else 2 * self.emb_dim
        layers: list[nn.Module] = []
        in_dim = trunk_in
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        feat_dim = hidden_dim if hidden_layers > 0 else trunk_in

        # --- output heads (late fusion: distances concatenated after trunk) ---
        if siamese:
            head_in = feat_dim + n_dist
            self.head_b2f = nn.Linear(head_in, 1)
            self.head_y2f = nn.Linear(head_in, 1)
        else:
            head_in = feat_dim + 2 * n_dist
            self.head = nn.Linear(head_in, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _split(self, X: torch.Tensor):
        """Split input into per-half embeddings and distances."""
        x = self.temporal_agg(X)
        blue_half = x[:, : self.half_dim]
        yellow_half = x[:, self.half_dim :]
        blue_emb = blue_half[:, : self.emb_dim]
        blue_dist = blue_half[:, self.emb_dim :]
        yellow_emb = yellow_half[:, : self.emb_dim]
        yellow_dist = yellow_half[:, self.emb_dim :]
        return blue_emb, blue_dist, yellow_emb, yellow_dist

    def features(self, X: torch.Tensor) -> torch.Tensor:
        """Return trunk output (without late-fused distances)."""
        blue_emb, _, yellow_emb, _ = self._split(X)
        if self.siamese:
            return torch.cat([self.trunk(blue_emb), self.trunk(yellow_emb)], dim=1)
        return self.trunk(torch.cat([blue_emb, yellow_emb], dim=1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Raw logits ``(N, 2)`` — columns are ``[B2F, Y2F]``."""
        blue_emb, blue_dist, yellow_emb, yellow_dist = self._split(X)
        if self.siamese:
            h_blue = self.trunk(blue_emb)
            h_yellow = self.trunk(yellow_emb)
            logit_b2f = self.head_b2f(torch.cat([h_blue, blue_dist], dim=1))
            logit_y2f = self.head_y2f(torch.cat([h_yellow, yellow_dist], dim=1))
            return torch.cat([logit_b2f, logit_y2f], dim=1)
        h = self.trunk(torch.cat([blue_emb, yellow_emb], dim=1))
        return self.head(torch.cat([h, blue_dist, yellow_dist], dim=1))

    def probs(self, X: torch.Tensor) -> torch.Tensor:
        """Calibrated sigmoid probabilities ``(N, 2)``."""
        return (self.forward(X) / self.temperature).sigmoid()

    def pred(self, X: torch.Tensor) -> torch.Tensor:
        """Hard binary predictions ``(N, 2)``."""
        return self.probs(X).round()
