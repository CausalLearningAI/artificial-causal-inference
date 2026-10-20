"""
Token encoder (ECI v2): DINOv3-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m, revision pinned below), blocks 1-20 of
24 (the model is truncated after block 20 and its final LayerNorm applied = 'block 20 after the final LayerNorm', as
rollout_r4 / nnnt), CLS + 4 register tokens dropped, bf16 autocast + SDPA attention, tokens rounded to fp16.
Input = the decoded frame at side S (S / 16 patches per side), ImageNet mean / std. The resolution rule (accepted
prior): S = native side rounded to a multiple of 16, never upsampled, unless the animal is so small that a crop that
follows it is used (tadpoles: 96 source px crop -> 384 px, a zoom of the crop; src/eci/v2 config 'crop').
"""
import time

import numpy as np
import torch

MODEL = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
REVISION = 'ea8dc2863c51be0a264bab82070e3e8836b02d51'  # HF snapshot used by rounds 1-4 and every v2 output
LAYER, N_PRE, D, PATCH = 20, 5, 1024, 16
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Encoder:
    def __init__(self, dev):
        from transformers import AutoModel
        m = AutoModel.from_pretrained(MODEL, revision=REVISION, attn_implementation='sdpa').to(dev).eval()
        assert len(m.layer) == 24
        m.layer = torch.nn.ModuleList(list(m.layer)[:LAYER])
        self.m = m.requires_grad_(False)
        self.dev = dev
        self.mean = torch.tensor(MEAN, device=dev)[None, :, None, None]
        self.std = torch.tensor(STD, device=dev)[None, :, None, None]
        self.gpu_s, self.n_frames = 0.0, 0

    @torch.no_grad()
    def __call__(self, x):
        """x (B, 3, S, S) uint8 cuda -> (B, (S/16)^2, 1024) fp16 tokens (row-major grid)."""
        t0 = time.time()
        xin = (x.float() / 255 - self.mean) / self.std
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = self.m(pixel_values=xin).last_hidden_state
        z = h[:, N_PRE:].float()
        G = x.shape[-1] // PATCH
        assert z.shape[1] == G * G and torch.isfinite(z).all(), z.shape
        z = z.half()
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        self.n_frames += x.shape[0]
        return z
