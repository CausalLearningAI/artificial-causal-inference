"""
Matryoshka SAE with FRAME-BUDGET BatchTopK sparsity: a threshold-free variant of src/eci/sae.py's BatchTopK.

Identical to MatryoshkaBatchTopKSAE (parameters, init, Matryoshka prefix loss, AuxK dead-latent loss, decoder
normalisation, optimiser step src.eci.sae._train_step) except the sparsifier:
    a batch is made of WHOLE frames (the tokens of a frame are contiguous rows, `seg` = frame id of every row, sorted);
    frame f with n_f tokens keeps the k * n_f largest post-ReLU activations among ITS OWN n_f x m values (ties broken
    arbitrarily by the sort). Training and inference use exactly the same rule, so there is no global threshold:
    encode(x, seg) needs the frame ids. The parent's threshold EMA is still updated during training (smallest kept
    activation) and saved, but never used.
The frame ids are passed with set_frames(seg) before forward_train (so _train_step can stay unchanged).

Functions / classes:
    frame_budget_topk     the sparsifier on (n, m) activations
    FrameBudgetSAE        the model
    self_test             random-data check that each frame keeps exactly min(k * n_f, #positive) values
    load_framebudget      rebuild (sae, norm, ckpt) from a checkpoint saved with sparsity == 'frame_budget'
"""

import torch

from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm


def frame_budget_topk(acts, seg, k):
    """acts (n, m) non-negative, seg (n,) long frame ids, sorted ascending (contiguous frames) ->
    (z (n, m) with each frame's k * n_f largest values kept, kept values (1D))."""
    n, m = acts.shape
    if n == 0:
        return acts.clone(), acts.new_zeros(0)
    flat = acts.flatten()
    _, seg_c = torch.unique_consecutive(seg, return_inverse=True)
    n_f = torch.bincount(seg_c)  # tokens per frame
    fid = seg_c.repeat_interleave(m)  # frame of every value (row-major flatten keeps frames contiguous)
    v, o = flat.sort(descending=True)
    f2, o2 = fid[o].sort(stable=True)  # group by frame, descending value inside each frame
    idx = o[o2]
    start = torch.cumsum(n_f * m, 0) - n_f * m
    rank = torch.arange(flat.numel(), device=acts.device) - start[f2]
    keep = rank < k * n_f[f2]
    sel = idx[keep]
    z = torch.zeros_like(flat)
    z[sel] = flat[sel]
    return z.view(n, m), flat[sel]


class FrameBudgetSAE(MatryoshkaBatchTopKSAE):
    sparsity = 'frame_budget'

    def set_frames(self, seg):
        self._seg = seg

    def batch_topk(self, pre):
        return frame_budget_topk(torch.relu(pre), self._seg, self.k)

    def encode(self, x, mode=None, seg=None):
        """Frame-budget codes; seg = frame id of every row (sorted). `mode` is ignored (no threshold)."""
        if seg is None:
            raise ValueError('FrameBudgetSAE.encode needs the frame ids (seg)')
        return frame_budget_topk(torch.relu(self.pre_acts(x)), seg, self.k)[0]


@torch.no_grad()
def self_test(device, k=8, m=512, seed=0):
    """Random frames of 1..300 tokens: kept count per frame == min(k * n_f, positives of the frame); every kept value
    is >= every dropped value of the same frame."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    lens = torch.randint(1, 300, (40,), generator=g)
    seg = torch.repeat_interleave(torch.arange(40), lens).to(device)
    a = torch.relu(torch.randn(int(lens.sum()), m, generator=g) - 1.5).to(device)
    z, _ = frame_budget_topk(a, seg, k)
    ok = True
    for f in range(40):
        r = seg == f
        af, zf = a[r].flatten(), z[r].flatten()
        kept = zf > 0
        want = min(k * int(r.sum()), int((af > 0).sum()))
        ok &= int(kept.sum()) == want
        if kept.any() and (~kept & (af > 0)).any():
            ok &= float(af[kept].min()) >= float(af[~kept].max())
    return {'frames': 40, 'tokens': int(lens.sum()), 'k': k, 'pass': bool(ok)}


def load_framebudget(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck.get('sparsity') == 'frame_budget', ck.get('sparsity')
    sae = FrameBudgetSAE(**ck['hparams'])
    sae.load_state_dict(ck['state_dict'])
    norm = TokenNorm(ck['hparams']['d_in'])
    if ck['norm']['scale'].ndim == 1:
        norm.scale = torch.ones(ck['hparams']['d_in'])
    norm.load_state_dict(ck['norm'])
    return sae.to(device).eval(), norm.to(device), ck
