from typing import Optional

import torch
import torch.nn as nn


class MouseBehaviorClassifier(nn.Module):
    """
    Pairwise behavior classifier using cross-attention over frame embeddings.

    4 global mouse query vectors attend to the temporal sequence of frame embeddings.
    The two attended representations are concatenated and classified.

    Input:
        context_seq: (B, T, emb_dim) — frame embedding sequence (not mean-pooled)
        a1, a2: (B,) — mouse indices in {0, 1, 2, 3}
        key_padding_mask: (B, T) bool, True = padding position
    Output: logits (B, n_classes) for {none=0, nt=1, nn=2}
    """

    def __init__(self, emb_dim: int = 768, n_heads: int = 8, hidden_dim: int = 256, n_classes: int = 3):
        super().__init__()
        self.mouse_queries = nn.Embedding(4, emb_dim)
        self.cross_attn = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        context_seq: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = context_seq.size(0)
        all_idx = torch.arange(4, device=context_seq.device)
        queries = self.mouse_queries(all_idx).unsqueeze(0).expand(B, -1, -1)  # (B, 4, emb_dim)
        # each mouse query attends to the full frame sequence
        attn_out, _ = self.cross_attn(queries, context_seq, context_seq, key_padding_mask=key_padding_mask)  # (B, 4, emb_dim)
        q1 = attn_out[torch.arange(B), a1]  # (B, emb_dim)
        q2 = attn_out[torch.arange(B), a2]  # (B, emb_dim)
        return self.head(torch.cat([q1, q2], dim=-1))
