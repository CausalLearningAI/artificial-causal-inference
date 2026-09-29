"""
Matryoshka sparse autoencoder with BatchTopK sparsity, for DINOv2 patch tokens
(ECI step 2).

References:
    BatchTopK SAE   Bussmann, Leask, Nanda 2024 (arXiv 2412.06410)
    Matryoshka SAE  Bussmann et al. 2025 (arXiv 2503.17547)
    TopK SAE / AuxK Gao et al. 2024 (arXiv 2406.04093)

Model (x already normalized, see TokenNorm):
    pre   = (x - b_dec) @ W_enc + b_enc               (n, m)
    z     = sparsify(relu(pre))                        (n, m)
    x_hat = z @ W_dec + b_dec, rows of W_dec unit-norm

Sparsity:
    training   BatchTopK: keep the n*k largest activations across the whole batch
               (so k active latents per token ON AVERAGE, variable per token).
    inference  'threshold': z = pre * (pre > theta), with one global theta estimated
               during training as an EMA of the smallest activation kept by BatchTopK.
               'topk': exact per-token top-k (option, not the default).

Matryoshka loss: the dictionary is split into nested prefixes (e.g. 128, 256, 512,
1024); for each prefix m_i the input is reconstructed with ONLY the first m_i
latents, and the loss is the SUM over prefixes of the MSE. Early latents are thus
forced to carry the coarse, general concepts.

Dead-latent auxiliary loss (AuxK): latents that have not fired for `dead_tokens`
training tokens reconstruct the residual x - x_hat with their top-k_aux
pre-activations; weight aux_coef (1/32 as in Gao et al.).

Functions / classes:
    TokenNorm                  mean-token subtraction + scalar scale to E||x|| = sqrt(d)
    MatryoshkaBatchTopKSAE     the model (encode / decode / forward_train)
    geometric_median           Weiszfeld iterations, for the b_dec init
    train_sae                  training loop over a GPU-resident fp16 token tensor
    train_sae_stream           same loop, tokens streamed from disk in random chunks
    evaluate_sae               FVE / L0 / dead latents per prefix, pooled-frame L0
    decoder_stability          max-cosine matching of decoder atoms across two seeds
    load_sae                   rebuild (sae, norm) from a checkpoint
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn


class TokenNorm(nn.Module):
    """x -> (x - mean) * scale, scale chosen so the average norm is sqrt(d).
    A scalar scale keeps FVE invariant (it is an affine map with isotropic scaling)."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(()))

    @torch.no_grad()
    def fit(self, tokens, n_sample=500_000, seed=0):
        """tokens: (N, d) tensor (any dtype/device). Stats from a random subsample."""
        g = torch.Generator(device='cpu').manual_seed(seed)
        idx = torch.randperm(tokens.shape[0], generator=g)[:n_sample].to(tokens.device)
        x = tokens[idx].float()
        mean = x.mean(0)
        avg_norm = (x - mean).norm(dim=1).mean()
        self.mean.copy_(mean.to(self.mean.device))
        self.scale.copy_((math.sqrt(x.shape[1]) / avg_norm).to(self.scale.device))
        return self

    @torch.no_grad()
    def fit_blocks(self, tokens, blocks, n_sample=500_000, seed=0):
        """As fit, but one scale per block of consecutive dims (e.g. [768, 768] for a
        [token, token change] concatenation): each block gets average norm sqrt(block size),
        so both parts weigh the same in the loss / FVE. scale becomes a (dim,) vector."""
        if sum(blocks) != self.mean.shape[0]:
            raise ValueError(f'blocks {blocks} do not sum to dim {self.mean.shape[0]}')
        g = torch.Generator(device='cpu').manual_seed(seed)
        idx = torch.randperm(tokens.shape[0], generator=g)[:n_sample].to(tokens.device)
        x = tokens[idx].float()
        mean = x.mean(0)
        scale, a = torch.empty_like(mean), 0
        for b in blocks:
            scale[a:a + b] = math.sqrt(b) / (x[:, a:a + b] - mean[a:a + b]).norm(dim=1).mean()
            a += b
        self.mean.copy_(mean.to(self.mean.device))
        self.scale = scale.to(self.mean.device)
        return self

    def forward(self, x):
        return (x.float() - self.mean) * self.scale


@torch.no_grad()
def geometric_median(x, n_iter=100, eps=1e-6):
    """Weiszfeld algorithm on the rows of x (n, d), fp32."""
    x = x.float()
    y = x.mean(0)
    for _ in range(n_iter):
        w = 1.0 / (x - y).norm(dim=1).clamp_min(eps)
        y_new = (w[:, None] * x).sum(0) / w.sum()
        if (y_new - y).norm() < 1e-5 * y.norm().clamp_min(1.0):
            y = y_new
            break
        y = y_new
    return y


class MatryoshkaBatchTopKSAE(nn.Module):
    def __init__(self, d_in=768, n_latents=1024, prefixes=(128, 256, 512, 1024), k=16,
                 k_aux=512, aux_coef=1 / 32, dead_tokens=2_000_000, threshold_lr=0.01, seed=0):
        super().__init__()
        prefixes = tuple(int(p) for p in prefixes)
        if list(prefixes) != sorted(prefixes) or prefixes[-1] != n_latents:
            raise ValueError(f'prefixes must be increasing and end at n_latents, got {prefixes}')
        self.d_in, self.n_latents, self.prefixes, self.k = d_in, n_latents, prefixes, k
        self.k_aux, self.aux_coef, self.dead_tokens, self.threshold_lr = k_aux, aux_coef, dead_tokens, threshold_lr

        g = torch.Generator().manual_seed(seed)
        W_dec = torch.randn(n_latents, d_in, generator=g)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.t().clone())  # decoder = encoder^T at init
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer('threshold', torch.tensor(-1.0))  # <0 = not estimated yet
        self.register_buffer('tokens_since_fired', torch.zeros(n_latents, dtype=torch.long))

    # ------------------------------------------------------------------ inference
    def pre_acts(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x, mode='threshold'):
        """Sparse codes (n, n_latents) for normalized x. mode: 'threshold' | 'topk'."""
        pre = torch.relu(self.pre_acts(x))
        if mode == 'threshold':
            if self.threshold < 0:
                raise RuntimeError('inference threshold not estimated (untrained SAE?)')
            return pre * (pre > self.threshold)
        if mode == 'topk':
            v, i = pre.topk(self.k, dim=1)
            return torch.zeros_like(pre).scatter_(1, i, v)
        raise ValueError(mode)

    def decode(self, z, m=None):
        """Reconstruction using only the first m latents (default: all)."""
        m = self.n_latents if m is None else m
        return z[:, :m] @ self.W_dec[:m] + self.b_dec

    # ------------------------------------------------------------------ training
    def batch_topk(self, pre):
        """Keep the n*k largest post-ReLU activations across the batch."""
        acts = torch.relu(pre)
        flat = acts.flatten()
        n_keep = min(acts.shape[0] * self.k, flat.numel())
        v, i = flat.topk(n_keep, sorted=False)
        z = torch.zeros_like(flat).scatter_(0, i, v).view_as(acts)
        return z, v

    def forward_train(self, x):
        """Returns total loss and a dict of logged scalars. x: normalized (n, d)."""
        pre = self.pre_acts(x)
        z, kept = self.batch_topk(pre)

        # Matryoshka: cumulative reconstruction per prefix, loss summed over prefixes
        recon = self.b_dec.expand_as(x)
        losses, start = [], 0
        for m in self.prefixes:
            recon = recon + z[:, start:m] @ self.W_dec[start:m]
            losses.append((recon - x).pow(2).sum(1).mean())
            start = m
        loss = torch.stack(losses).sum()
        x_hat = recon

        # bookkeeping: dead latents, running inference threshold
        with torch.no_grad():
            fired = (z > 0).any(0)
            self.tokens_since_fired += x.shape[0]
            self.tokens_since_fired[fired] = 0
            pos = kept[kept > 0]
            if pos.numel():
                mn = pos.min()
                if self.threshold < 0:
                    self.threshold.copy_(mn)
                else:
                    self.threshold.mul_(1 - self.threshold_lr).add_(self.threshold_lr * mn)

        # AuxK on dead latents
        dead = self.tokens_since_fired >= self.dead_tokens
        n_dead = int(dead.sum())
        aux = torch.zeros((), device=x.device)
        if n_dead > 0:
            k_aux = min(self.k_aux, n_dead)
            resid = (x - x_hat).detach()
            pre_dead = torch.relu(pre[:, dead])
            v, i = pre_dead.topk(k_aux, dim=1)
            z_aux = torch.zeros_like(pre_dead).scatter_(1, i, v)
            resid_hat = z_aux @ self.W_dec[dead]
            aux = (resid_hat - resid).pow(2).sum(1).mean()
            if torch.isfinite(aux):
                loss = loss + self.aux_coef * aux

        var = (x - x.mean(0)).pow(2).sum(1).mean()
        logs = {'loss': float(loss), 'aux': float(aux), 'n_dead': n_dead,
                'fve': float(1 - losses[-1].detach() / var), 'fve_first': float(1 - losses[0].detach() / var),
                'l0': float((z > 0).sum(1).float().mean()), 'threshold': float(self.threshold)}
        return loss, logs

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def remove_parallel_grad(self):
        """Project out the gradient component along each (unit) decoder row."""
        if self.W_dec.grad is not None:
            W = self.W_dec.data
            self.W_dec.grad -= (self.W_dec.grad * W).sum(1, keepdim=True) * W


def _lr_at(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))  # cosine to 10% of base


def _train_step(sae, opt, x, step, total, lr, warmup, grad_clip):
    """One optimizer step on a normalized batch x. Returns (logs, grad_norm)."""
    for pg in opt.param_groups:
        pg['lr'] = _lr_at(step, total, lr, warmup)
    loss, logs = sae.forward_train(x)
    if not math.isfinite(logs['loss']):
        raise RuntimeError(f'non-finite loss at step {step}: {logs}')
    opt.zero_grad(set_to_none=True)
    loss.backward()
    sae.remove_parallel_grad()
    gn = torch.nn.utils.clip_grad_norm_(sae.parameters(), grad_clip)
    opt.step()
    sae.normalize_decoder()
    return logs, gn


def _log_step(sae, opt, logs, gn, step, total, ep, t0, history, log_fn):
    logs.update(step=step, epoch=ep, lr=opt.param_groups[0]['lr'], grad_norm=float(gn),
                elapsed_s=round(time.time() - t0, 1))
    history.append(logs)
    log_fn(f"  step {step:6d}/{total} ep {ep} loss {logs['loss']:.2f} fve {logs['fve']:.4f} "
           f"fve@{sae.prefixes[0]} {logs['fve_first']:.4f} l0 {logs['l0']:.1f} dead {logs['n_dead']} "
           f"aux {logs['aux']:.2f} thr {logs['threshold']:.4f} gn {logs['grad_norm']:.2f} "
           f"lr {logs['lr']:.2e} {logs['elapsed_s']}s")


def train_sae(sae, norm, tokens, epochs=5, batch_size=4096, lr=5e-4, warmup=500, grad_clip=1.0,
              seed=0, log_every=200, log_fn=print):
    """tokens: (N, d) fp16 tensor on the training device (raw, un-normalized)."""
    device = tokens.device
    g = torch.Generator(device='cpu').manual_seed(seed)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    N = tokens.shape[0]
    steps_per_epoch = N // batch_size
    total = epochs * steps_per_epoch
    history, step, t0 = [], 0, time.time()
    for ep in range(epochs):
        perm = torch.randperm(N, generator=g).to(device)
        for b in range(steps_per_epoch):
            x = norm(tokens[perm[b * batch_size:(b + 1) * batch_size]])
            logs, gn = _train_step(sae, opt, x, step, total, lr, warmup, grad_clip)
            if step % log_every == 0 or step == total - 1:
                _log_step(sae, opt, logs, gn, step, total, ep, t0, history, log_fn)
            step += 1
    return history


def train_sae_stream(sae, norm, chunk_iter, chunk_tokens, epochs=2, batch_size=4096, lr=5e-4, warmup=500,
                     grad_clip=1.0, seed=0, log_every=200, log_fn=print):
    """Training on a token set too large for the GPU, streamed in chunks.

    chunk_iter(epoch) -> iterator of (chunk_id, (n, d) float16 numpy array); each chunk
    must be a uniform random sample of the training tokens (e.g. TokenStore.iter_chunks
    on a randomly ordered store, in a per-epoch random chunk order). chunk_tokens: token
    count of every chunk, to fix the schedule length up front. Within a chunk the tokens
    are shuffled on the GPU and cut into batches; the remainder (< batch_size) of each
    chunk is dropped. Same optimizer / schedule / step as train_sae.
    """
    device = sae.W_dec.device
    g = torch.Generator(device='cpu').manual_seed(seed)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    steps_per_epoch = int(sum(int(n) // batch_size for n in chunk_tokens))
    total = epochs * steps_per_epoch
    history, step, t0, wait = [], 0, time.time(), 0.0
    for ep in range(epochs):
        tw = time.time()
        for ci, block in chunk_iter(ep):
            wait += time.time() - tw
            if block.shape[0] != chunk_tokens[ci]:
                raise RuntimeError(f'chunk {ci}: {block.shape[0]} tokens, expected {chunk_tokens[ci]}')
            tok = torch.from_numpy(block).to(device)
            perm = torch.randperm(tok.shape[0], generator=g).to(device)
            for b in range(tok.shape[0] // batch_size):
                x = norm(tok[perm[b * batch_size:(b + 1) * batch_size]])
                logs, gn = _train_step(sae, opt, x, step, total, lr, warmup, grad_clip)
                if step % log_every == 0 or step == total - 1:
                    logs['io_wait_s'] = round(wait, 1)
                    _log_step(sae, opt, logs, gn, step, total, ep, t0, history, log_fn)
                step += 1
            del tok, perm
            tw = time.time()
    assert step == total, (step, total)
    return history


@torch.no_grad()
def evaluate_sae(sae, norm, tokens, n_patches=256, mode='threshold', batch_frames=64):
    """Metrics on (N_frames*n_patches, d) raw tokens, grouped by frame (contiguous rows).

    Per prefix m: FVE (1 - SSE / total SS about the eval mean), mean L0 per token
    counting only latents < m, % dead latents among the first m (never fire on the
    eval set), mean L0 of the frame vector mean-pooled over patches.
    """
    device = sae.W_dec.device
    N = tokens.shape[0]
    n_frames = N // n_patches
    P = sae.prefixes
    sse = torch.zeros(len(P), dtype=torch.float64, device=device)
    l0 = torch.zeros(len(P), dtype=torch.float64, device=device)
    pooled_l0 = torch.zeros(len(P), dtype=torch.float64, device=device)
    fire_count = torch.zeros(sae.n_latents, dtype=torch.long, device=device)
    s1 = torch.zeros(sae.d_in, dtype=torch.float64, device=device)
    s2 = torch.zeros((), dtype=torch.float64, device=device)
    for f0 in range(0, n_frames, batch_frames):
        f1 = min(n_frames, f0 + batch_frames)
        x = norm(tokens[f0 * n_patches:f1 * n_patches].to(device))
        z = sae.encode(x, mode=mode)
        active = z > 0
        fire_count += active.sum(0)
        s1 += x.double().sum(0)
        s2 += x.double().pow(2).sum()
        pooled = z.view(f1 - f0, n_patches, -1).mean(1) > 0
        for j, m in enumerate(P):
            sse[j] += (sae.decode(z, m) - x).double().pow(2).sum()
            l0[j] += active[:, :m].sum()
            pooled_l0[j] += pooled[:, :m].sum()
    n_tok = n_frames * n_patches
    total_ss = s2 - s1.pow(2).sum() / n_tok
    out = {'mode': mode, 'n_tokens': n_tok, 'n_frames': n_frames,
           'total_var_per_token': float(total_ss / n_tok), 'prefixes': {}}
    for j, m in enumerate(P):
        out['prefixes'][str(m)] = {
            'fve': float(1 - sse[j] / total_ss),
            'mse_per_token': float(sse[j] / n_tok),
            'l0_per_token': float(l0[j] / n_tok),
            'dead_frac': float((fire_count[:m] == 0).float().mean()),
            'n_dead': int((fire_count[:m] == 0).sum()),
            'pooled_frame_l0_mean': float(pooled_l0[j] / n_frames),
        }
    out['fire_rate'] = (fire_count.double() / n_tok).cpu().numpy().tolist()
    return out


@torch.no_grad()
def decoder_stability(W_a, W_b, prefixes, thresh=0.9):
    """For each atom in W_a[:m], max cosine to W_b[:m] (same prefix) and to all of W_b."""
    A = W_a / W_a.norm(dim=1, keepdim=True)
    B = W_b / W_b.norm(dim=1, keepdim=True)
    C = A @ B.t()
    out = {}
    for m in prefixes:
        same = C[:m, :m].max(1).values
        full = C[:m].max(1).values
        out[str(m)] = {'median_maxcos_same_prefix': float(same.median()),
                       f'frac_gt_{thresh}_same_prefix': float((same > thresh).float().mean()),
                       'median_maxcos_vs_full': float(full.median()),
                       f'frac_gt_{thresh}_vs_full': float((full > thresh).float().mean())}
    return out


def save_checkpoint(path, sae, norm, extra=None):
    torch.save({'state_dict': sae.state_dict(), 'norm': norm.state_dict(),
                'hparams': {'d_in': sae.d_in, 'n_latents': sae.n_latents, 'prefixes': list(sae.prefixes),
                            'k': sae.k, 'k_aux': sae.k_aux, 'aux_coef': sae.aux_coef,
                            'dead_tokens': sae.dead_tokens, 'threshold_lr': sae.threshold_lr},
                **(extra or {})}, path)


def load_sae(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sae = MatryoshkaBatchTopKSAE(**ck['hparams'])
    sae.load_state_dict(ck['state_dict'])
    norm = TokenNorm(ck['hparams']['d_in'])
    if ck['norm']['scale'].ndim == 1:  # per-block scale (TokenNorm.fit_blocks)
        norm.scale = torch.ones(ck['hparams']['d_in'])
    norm.load_state_dict(ck['norm'])
    return sae.to(device).eval(), norm.to(device), ck
