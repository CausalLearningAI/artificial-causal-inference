from typing import Optional

import torch
import torch.nn as nn


class PatchAttnPool(nn.Module):
    """Single learned query attends over a frame's spatial patch-grid tokens,
    producing one smart-weighted-average vector per frame (replaces static
    average pooling / CLS-token summarization)."""

    def __init__(self, emb_dim: int = 768, n_heads: int = 1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)

    def forward(self, patch_seq: torch.Tensor) -> torch.Tensor:
        # patch_seq: (N, P, emb_dim) -> (N, emb_dim)
        N = patch_seq.size(0)
        q = self.query.expand(N, -1, -1)
        out, _ = self.attn(q, patch_seq, patch_seq)
        return out.squeeze(1)


class MouseBehaviorClassifier(nn.Module):
    """
    Pairwise behavior classifier using cross-attention over frame embeddings.

    4 global mouse query vectors attend to the temporal sequence of frame embeddings.
    The two attended representations are concatenated and classified.

    Input:
        context_seq: (B, T, emb_dim) frame embedding sequence (CLS-token mode), or
            (B, T, P, emb_dim) coarse patch-grid tokens per frame (use_patch_grid=True) —
            pooled per-frame via PatchAttnPool before the temporal cross-attention.
        a1, a2: (B,) — mouse indices in {0, 1, 2, 3}
        offsets: (B, T) — integer frame offset of each context position relative to the
            target frame (e.g. -2..+2 for context_k=2), clamped to [-max_offset, max_offset]
        key_padding_mask: (B, T) bool, True = padding position
    Output: logits (B, n_classes) for {none=0, nt=1, nn=2}
    """

    def __init__(
        self, emb_dim: int = 768, n_heads: int = 1, hidden_dim: int = 256, n_classes: int = 3,
        max_offset: int = 8, use_patch_grid: bool = False, dropout: float = 0.0,
    ):
        super().__init__()
        self.max_offset = max_offset
        self.use_patch_grid = use_patch_grid
        self.mouse_queries = nn.Embedding(4, emb_dim)
        self.pos_emb = nn.Embedding(2 * max_offset + 1, emb_dim)
        if use_patch_grid:
            self.patch_pool = PatchAttnPool(emb_dim=emb_dim, n_heads=n_heads)
        self.cross_attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        context_seq: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_patch_grid:
            B, T, P, D = context_seq.shape
            context_seq = self.patch_pool(context_seq.reshape(B * T, P, D)).reshape(B, T, D)
        B = context_seq.size(0)
        if offsets is not None:
            idx = offsets.clamp(-self.max_offset, self.max_offset) + self.max_offset
            context_seq = context_seq + self.pos_emb(idx)  # (B, T, emb_dim)
        all_idx = torch.arange(4, device=context_seq.device)
        queries = self.mouse_queries(all_idx).unsqueeze(0).expand(B, -1, -1)  # (B, 4, emb_dim)
        # each mouse query attends to the full frame sequence
        attn_out, _ = self.cross_attn(queries, context_seq, context_seq, key_padding_mask=key_padding_mask)  # (B, 4, emb_dim)
        q1 = attn_out[torch.arange(B), a1]  # (B, emb_dim)
        q2 = attn_out[torch.arange(B), a2]  # (B, emb_dim)
        return self.head(torch.cat([q1, q2], dim=-1))


class MouseFrameClassifier(nn.Module):
    """
    Per-frame behavior detector — no mouse-identity conditioning (unlike
    MouseBehaviorClassifier, this doesn't take a1/a2). A single learned query
    attends over the temporal sequence of frame embeddings; the pooled
    representation is classified.

    Multi-label, not 3-way softmax: a frame can contain both nt and nn at once
    (via different mouse pairs), so the two behaviors are independent sigmoid
    outputs rather than mutually-exclusive classes.

    Input:
        context_seq: (B, T, emb_dim) or (B, T, P, emb_dim) — see MouseBehaviorClassifier.
        offsets: (B, T) — see MouseBehaviorClassifier.
        key_padding_mask: (B, T) bool, True = padding position.
    Output: logits (B, n_labels) for [has_nt, has_nn] — apply sigmoid, not softmax.
    """

    def __init__(
        self, emb_dim: int = 768, n_heads: int = 1, hidden_dim: int = 256, n_labels: int = 2,
        max_offset: int = 8, use_patch_grid: bool = False, dropout: float = 0.0,
    ):
        super().__init__()
        self.max_offset = max_offset
        self.use_patch_grid = use_patch_grid
        self.query = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.pos_emb = nn.Embedding(2 * max_offset + 1, emb_dim)
        if use_patch_grid:
            self.patch_pool = PatchAttnPool(emb_dim=emb_dim, n_heads=n_heads)
        self.cross_attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(
        self,
        context_seq: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_patch_grid:
            B, T, P, D = context_seq.shape
            context_seq = self.patch_pool(context_seq.reshape(B * T, P, D)).reshape(B, T, D)
        B = context_seq.size(0)
        if offsets is not None:
            idx = offsets.clamp(-self.max_offset, self.max_offset) + self.max_offset
            context_seq = context_seq + self.pos_emb(idx)  # (B, T, emb_dim)
        query = self.query.expand(B, -1, -1)  # (B, 1, emb_dim)
        attn_out, _ = self.cross_attn(query, context_seq, context_seq, key_padding_mask=key_padding_mask)  # (B, 1, emb_dim)
        return self.head(attn_out.squeeze(1))
