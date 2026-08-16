from typing import Optional

import torch
import torch.nn as nn


class PatchAttnPool(nn.Module):
    """Learned queries attend over a frame's spatial patch-grid tokens, producing one
    smart-weighted-average vector per query per frame (replaces static average pooling /
    CLS-token summarization).

    n_queries > 1 pools the same frame K independent ways and concatenates the results, so the
    output is (N, n_queries * out_dim). One query has to compress a whole frame into a single
    weighted average; with four mice in the cage and a behaviour defined by ONE pair, a single
    softmax over patches must split its mass between the pair that matters and everything else.
    K queries let different queries settle on different regions. This is the cheapest possible
    capacity increase (K extra query vectors, plus a wider projection downstream) and it is what
    MouseOPairClassifier has always done -- it carries 4 mouse queries; the frame classifier was
    written with one and never revisited.
    """

    def __init__(self, emb_dim: int = 768, out_dim: Optional[int] = None, n_heads: int = 1,
                 dropout: float = 0.0, use_layernorm: bool = False, n_queries: int = 1):
        super().__init__()
        self.out_dim = out_dim or emb_dim
        self.n_queries = n_queries
        if self.out_dim != emb_dim:
            # project raw DINO patch tokens down before pooling, so the attention itself
            # (and everything downstream) runs at the smaller dim -- a real capacity cut,
            # not just a cheaper matmul.
            self.in_proj = nn.Linear(emb_dim, self.out_dim)
        self.query = nn.Parameter(torch.randn(1, n_queries, self.out_dim) * 0.02)
        self.attn = nn.MultiheadAttention(self.out_dim, n_heads, dropout=dropout, batch_first=True)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            # pre-norm on the keys/values (raw DINO patch tokens have no normalization
            # guarantee downstream of the frozen encoder) + post-norm on the pooled output,
            # matching standard pre-LN transformer block design.
            self.norm_kv = nn.LayerNorm(self.out_dim)
            self.norm_out = nn.LayerNorm(self.out_dim)

    def forward(self, patch_seq: torch.Tensor) -> torch.Tensor:
        # patch_seq: (N, P, emb_dim) -> (N, n_queries * out_dim)
        N = patch_seq.size(0)
        if self.out_dim != patch_seq.size(-1):
            patch_seq = self.in_proj(patch_seq)
        kv = self.norm_kv(patch_seq) if self.use_layernorm else patch_seq
        q = self.query.expand(N, -1, -1)
        # need_weights=False routes this through scaled_dot_product_attention instead of the
        # math path that materializes the (N, n_heads, n_queries, P) score tensor. Numerically
        # equivalent; the weights were never read here.
        out, _ = self.attn(q, kv, kv, need_weights=False)
        if self.use_layernorm:
            out = self.norm_out(out)
        return out.reshape(N, -1)


class PatchSelfAttn(nn.Module):
    """One self-attention layer over a frame's patch tokens, as a zero-initialised residual.

    THE GAP THIS FILLS. Neither pooling module computes a function of two patch features
    jointly: attention scores are query-vs-key, never key-vs-key, so with a single learned
    query the only patch-patch coupling anywhere in the head is the softmax normaliser.
    A predicate like "nose of one mouse adjacent to the tail of another" is relational by
    definition, so under this head it can only be computed inside DINOv2's own blocks -- which
    is exactly what unfreezing them supplies. This module adds the missing pairwise term to a
    FROZEN encoder, so the two explanations can be told apart.

    Bottlenecked to `dim` because the attention is O(P^2) and P = 1024 at 448px. The output
    projection is zero-initialised, so at step 0 this module is the identity and the network is
    bit-for-bit the baseline head -- any difference in the result is attributable to the pairwise
    term itself rather than to a perturbed initialisation or a changed downstream width.
    """

    def __init__(self, emb_dim: int = 768, dim: int = 128, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_in = nn.LayerNorm(emb_dim)
        self.down = nn.Linear(emb_dim, dim)
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.up = nn.Linear(dim, emb_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, patch_seq: torch.Tensor) -> torch.Tensor:
        # patch_seq: (N, P, emb_dim) -> (N, P, emb_dim)
        h = self.down(self.norm_in(patch_seq))
        hn = self.norm_attn(h)
        h = h + self.attn(hn, hn, hn, need_weights=False)[0]
        return patch_seq + self.up(h)


class MouseOPairClassifier(nn.Module):
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
    MouseOPairClassifier, this doesn't take a1/a2). A single learned query
    attends over the temporal sequence of frame embeddings; the pooled
    representation is classified.

    Multi-label, not 3-way softmax: a frame can contain both nt and nn at once
    (via different mouse pairs), so the two behaviors are independent sigmoid
    outputs rather than mutually-exclusive classes.

    Input:
        context_seq: (B, T, emb_dim) or (B, T, P, emb_dim) — see MouseOPairClassifier.
        offsets: (B, T) — see MouseOPairClassifier.
        key_padding_mask: (B, T) bool, True = padding position.
    Output: logits (B, n_labels) for [has_nt, has_nn] — apply sigmoid, not softmax.

    use_motion (patch-grid only): alongside each position's pooled patch content, also
    pools the per-patch delta to the previous context position (zero at the first position)
    through a second, separately-learned PatchAttnPool, then projects [content; delta] back
    down to emb_dim before the existing positional-embedding + cross-attention path — so the
    temporal cross-attention gets an explicit frame-to-frame change signal instead of having
    to infer it from raw content alone. A cheap probe (mean patch-delta L2 norm, no learning)
    already separates nt/nn above a same-shape raw-content-magnitude baseline (ROC-AUC
    0.61/0.54 vs 0.53/0.46), motivating this.

    Regularization knobs (all no-ops unless set, and all inactive in eval mode):
        cross_attn_dim: bottleneck the temporal cross-attention (query/pos_emb/attn/head) down
            to this dim instead of running it at the full emb_dim. A 5-position context window
            (context_k=2) doesn't need a full 768-dim attention module (~2.4M params) — most of
            that capacity has nowhere useful to go and just memorizes train-pool idiosyncrasies.
        patch_dropout: probability of zeroing each patch token (independently, per patch per
            frame) before pooling — structured input-level augmentation, cheap to apply on
            already-encoded/cached tokens (no need to re-run the frozen encoder).
        patch_noise_std: std of Gaussian noise added to patch tokens before pooling.
        frame_dropout: probability of additionally masking a non-target context frame (offset
            != 0; the target frame itself is never dropped) during training, on top of any real
            padding — discourages the temporal cross-attention from over-relying on any single
            context position.
        use_layernorm: adds LayerNorm before/after each attention module (patch_pool's and
            cross_attn's) and inside the MLP head, matching standard pre-LN transformer block
            design. Off by default so old checkpoints (saved without these extra params) still
            load; every prior experiment in this repo trained with zero normalization anywhere
            in the trainable stack — raw DINO patch tokens went straight into an unnormalized
            MultiheadAttention, whose output fed straight into a second unnormalized
            MultiheadAttention, whose output fed an unnormalized MLP.
        patch_pool_dim: bottleneck patch_pool itself (the spatial-attention pooling over the
            256+ raw DINO patch tokens) down to this dim, on top of/independent from
            cross_attn_dim's existing bottleneck on the temporal module. patch_pool is the
            single largest capacity block (~2.4M params at full 768-dim) and was left untouched
            by every prior capacity-reduction experiment in this repo -- only the temporal
            cross-attention was ever bottlenecked.
        patch_selfattn_dim: insert a PatchSelfAttn residual at this bottleneck width before
            pooling -- the only operation in this head that computes a function of two patch
            features jointly. See PatchSelfAttn. No-op at init by construction.
        n_pool_queries: number of patch-pooling queries (see PatchAttnPool). >1 widens the
            pooled per-frame vector to n_pool_queries * patch_pool_dim, which temporal_proj
            then maps back to cross_attn_dim, so everything downstream keeps its shape.
    """

    def __init__(
        self, emb_dim: int = 768, n_heads: int = 1, hidden_dim: int = 256, n_labels: int = 2,
        max_offset: int = 8, use_patch_grid: bool = False, dropout: float = 0.0,
        n_hidden_layers: int = 1, use_motion: bool = False, cross_attn_dim: Optional[int] = None,
        patch_dropout: float = 0.0, patch_noise_std: float = 0.0, frame_dropout: float = 0.0,
        use_layernorm: bool = False, patch_pool_dim: Optional[int] = None,
        patch_selfattn_dim: Optional[int] = None, n_pool_queries: int = 1,
        pool_grid: int = 0,
    ):
        """pool_grid: pool each frame into a pool_grid x pool_grid SPATIAL GRID of region
        vectors instead of collapsing it to one vector, and let the temporal attention run over
        the T x pool_grid^2 region tokens.

        THE PROBLEM THIS FIXES. Every configuration to date pools 1024 patches down to a single
        vector per frame BEFORE the temporal stage, so all spatial structure is destroyed exactly
        where motion would have to be read. That is why --use-motion lost: it differenced
        globally-pooled vectors, which cannot tell "one mouse moved" from "the whole scene
        shifted". With regions kept, a difference is LOCAL, and a pairwise predicate has somewhere
        to live -- the two mice in an interaction are usually in the same or adjacent regions.

        Cost is negligible: the temporal attention goes from 5 tokens to T*G^2 = 80 at G=4, and
        the pooling itself is unchanged in FLOPs (the same 1024 patches are pooled, just into G^2
        groups instead of one). It composes with cross_attn_dim, which is the module that should
        shrink -- 2.36M params at 768 width to weight five positions is the single most
        over-provisioned block in the head, and the regime is overfitting, not underfitting.
        """
        super().__init__()
        if pool_grid and use_motion:
            raise ValueError('pool_grid and use_motion are not composable as written: the motion '
                             'path assumes one pooled vector per frame. Region-local differencing '
                             'is the natural version and is worth doing, but as a separate change.')
        self.pool_grid = pool_grid
        self.max_offset = max_offset
        self.use_patch_grid = use_patch_grid
        self.use_motion = use_motion
        self.patch_dropout = patch_dropout
        self.patch_noise_std = patch_noise_std
        self.frame_dropout = frame_dropout
        self.use_layernorm = use_layernorm
        self.n_pool_queries = n_pool_queries
        patch_content_dim = (patch_pool_dim or emb_dim) * n_pool_queries
        self.cross_attn_dim = cross_attn_dim or patch_content_dim
        if use_patch_grid:
            self.patch_selfattn = (PatchSelfAttn(emb_dim=emb_dim, dim=patch_selfattn_dim,
                                                 n_heads=n_heads, dropout=dropout)
                                   if patch_selfattn_dim else None)
            self.patch_pool = PatchAttnPool(emb_dim=emb_dim, out_dim=patch_pool_dim, n_heads=n_heads,
                                             dropout=dropout, use_layernorm=use_layernorm,
                                             n_queries=n_pool_queries)
            if use_motion:
                self.motion_pool = PatchAttnPool(emb_dim=emb_dim, out_dim=patch_pool_dim, n_heads=n_heads,
                                                  dropout=dropout, use_layernorm=use_layernorm,
                                                  n_queries=n_pool_queries)
                self.motion_proj = nn.Linear(2 * patch_content_dim, patch_content_dim)
        if self.cross_attn_dim != patch_content_dim:
            self.temporal_proj = nn.Linear(patch_content_dim, self.cross_attn_dim)
        self.query = nn.Parameter(torch.randn(1, 1, self.cross_attn_dim) * 0.02)
        self.pos_emb = nn.Embedding(2 * max_offset + 1, self.cross_attn_dim)
        # Separable position code: the temporal embedding above is indexed by frame offset and
        # shared across regions; this one is indexed by region and shared across time. Separable
        # rather than a joint (T x G^2) table because the two axes mean different things and the
        # temporal one must keep generalising across context windows.
        self.region_emb = (nn.Embedding(pool_grid * pool_grid, self.cross_attn_dim)
                           if pool_grid else None)
        self.cross_attn = nn.MultiheadAttention(self.cross_attn_dim, n_heads, dropout=dropout, batch_first=True)
        if use_layernorm:
            self.cross_attn_norm_in = nn.LayerNorm(self.cross_attn_dim)
            self.cross_attn_norm_out = nn.LayerNorm(self.cross_attn_dim)

        layers = []
        in_dim = self.cross_attn_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_labels))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        context_seq: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_patch_grid:
            B, T, P, D = context_seq.shape
            if self.training and self.patch_noise_std > 0:
                # scale relative to the batch's own std (detached) rather than an absolute
                # constant — DINOv2/DINOv3 raw patch-token magnitudes differ substantially,
                # so a fixed noise scale would mean something different for each encoder.
                noise_scale = context_seq.std().detach() * self.patch_noise_std
                context_seq = context_seq + torch.randn_like(context_seq) * noise_scale
            if self.training and self.patch_dropout > 0:
                keep = (torch.rand(B, T, P, 1, device=context_seq.device) > self.patch_dropout).float()
                context_seq = context_seq * keep / (1 - self.patch_dropout)
            if self.patch_selfattn is not None:
                # per frame, independently: the pairwise term is SPATIAL. Temporal mixing stays
                # the cross-attention's job, so context positions never see each other here.
                context_seq = self.patch_selfattn(
                    context_seq.reshape(B * T, P, D)).reshape(B, T, P, D)
            if self.pool_grid:
                # (B*T, P, D) -> (B*T*G^2, P/G^2, D): split the SxS patch grid into G x G
                # contiguous square regions, keeping each region's patches together so the
                # pooling attention sees one neighbourhood at a time.
                G = self.pool_grid
                S = int(P ** 0.5)
                if S * S != P or S % G:
                    raise ValueError(f'pool_grid={G} needs a square patch grid divisible by G; '
                                     f'got P={P} (S={S})')
                r = S // G
                x = context_seq.reshape(B * T, S, S, D)
                x = x.reshape(B * T, G, r, G, r, D).permute(0, 1, 3, 2, 4, 5)
                x = x.reshape(B * T * G * G, r * r, D)
                content = self.patch_pool(x).reshape(B, T * G * G, -1)
            else:
                content = self.patch_pool(context_seq.reshape(B * T, P, D)).reshape(B, T, -1)
            if self.use_motion:
                delta = context_seq[:, 1:] - context_seq[:, :-1]  # (B, T-1, P, D)
                delta = torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1)  # pad t=0 -> (B, T, P, D)
                motion = self.motion_pool(delta.reshape(B * T, P, D)).reshape(B, T, -1)
                context_seq = self.motion_proj(torch.cat([content, motion], dim=-1))
            else:
                context_seq = content
        if self.pool_grid:
            # Each context position became G^2 region tokens, so the per-position offsets and
            # padding flags have to be repeated to match -- interleaved, not tiled, because the
            # region axis is the FAST one in the reshape above.
            g2 = self.pool_grid * self.pool_grid
            if offsets is not None:
                offsets = offsets.repeat_interleave(g2, dim=1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.repeat_interleave(g2, dim=1)
        if self.cross_attn_dim != context_seq.size(-1):
            context_seq = self.temporal_proj(context_seq)
        B = context_seq.size(0)
        if offsets is not None:
            idx = offsets.clamp(-self.max_offset, self.max_offset) + self.max_offset
            context_seq = context_seq + self.pos_emb(idx)  # (B, T[*G^2], cross_attn_dim)
        if self.pool_grid:
            n_pos = context_seq.size(1) // g2
            ridx = torch.arange(g2, device=context_seq.device).repeat(n_pos)
            context_seq = context_seq + self.region_emb(ridx).unsqueeze(0)
        if self.training and self.frame_dropout > 0 and key_padding_mask is not None and offsets is not None:
            is_center = (offsets == 0)
            extra_drop = (torch.rand_like(key_padding_mask, dtype=torch.float) < self.frame_dropout) & ~is_center & ~key_padding_mask
            key_padding_mask = key_padding_mask | extra_drop
        if self.use_layernorm:
            context_seq = self.cross_attn_norm_in(context_seq)
        query = self.query.expand(B, -1, -1)  # (B, 1, cross_attn_dim)
        attn_out, _ = self.cross_attn(query, context_seq, context_seq, key_padding_mask=key_padding_mask)  # (B, 1, cross_attn_dim)
        attn_out = attn_out.squeeze(1)
        if self.use_layernorm:
            attn_out = self.cross_attn_norm_out(attn_out)
        return self.head(attn_out)
