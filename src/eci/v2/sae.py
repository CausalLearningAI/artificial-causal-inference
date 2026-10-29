"""
SAE training and application (ECI v2).

Model: Matryoshka BatchTopK SAE (src/eci/sae.py MatryoshkaBatchTopKSAE), k = 16, TokenNorm with one scale per
1024-d input block (own token, context), b_dec initialised at the geometric median, Adam 5e-4 with 500 warm-up
steps and cosine decay, batches of 4096 tokens drawn uniformly without replacement (one permutation per pass), AuxK
(the src/eci/sae.py training step). Identical to the rounds 2-4 / freeze trainer (scripts/eci/nnnt/r3.py train_arm,
uniform sampler, no whitening).

Inference threshold theta* (src/eci/sae_calibrate.py): the value such that the mean number of active latents per
token is k on held-out CALIBRATION videos; inference = relu(pre-activation) where it exceeds theta*. Held-out FVE
(fraction of variance explained) on EVALUATION videos. Deployment seed (label-free): the highest held-out FVE.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch


def train(cfg, seed, Xtr, Xca, Xev, out_dir, dev):
    """cfg: W, prefixes, k, steps (+ lr, bs, warmup); X*: (n, d) fp16 SAE inputs (train on dev; calib / eval anywhere).
    Writes out_dir/sae_theta.pt and train.json; returns the record."""
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    lr, bs, warmup = cfg.get('lr', 5e-4), cfg.get('bs', 4096), cfg.get('warmup', 500)
    din = Xtr.shape[1]
    g = np.random.default_rng(seed)
    ii = torch.from_numpy(np.sort(g.choice(len(Xtr), min(250_000, len(Xtr)), replace=False))).to(dev)
    init = Xtr[ii].float()
    if cfg.get('norm') == 'global':  # one scale for all dims (inputs whose blocks were pre-scaled by the caller)
        norm = TokenNorm(din).to(dev).fit(init, seed=seed)
    else:
        norm = TokenNorm(din).to(dev).fit_blocks(init, [1024] * (din // 1024), seed=seed)
    sae = MatryoshkaBatchTopKSAE(d_in=din, n_latents=cfg['W'], prefixes=tuple(cfg['prefixes']), k=cfg['k'],
                                 seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    N, steps = len(Xtr), cfg['steps']
    perm, p, hist, t0 = torch.randperm(N, generator=gen, device=dev), 0, [], time.time()
    for step in range(steps):
        if p + bs > N:
            perm, p = torch.randperm(N, generator=gen, device=dev), 0
        b = perm[p:p + bs]
        p += bs
        logs, _ = _train_step(sae, opt, norm(Xtr[b]), step, steps, lr, warmup, 1.0)
        if step % 5000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            hist.append(logs)
    sae.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    tt = time.time() - t0
    th, _ = calibrate_threshold(sae, Xca, norm=norm)
    set_threshold(sae, th)
    st = sparsity_stats(sae, Xev, th, norm=norm)
    rec = {'cfg': {k_: (list(v) if isinstance(v, tuple) else v) for k_, v in cfg.items()}, 'seed': seed,
           'train_time_s': tt, 'steps_per_s': steps / tt, 'n_train_tokens': int(N), 'n_calib_tokens': int(len(Xca)),
           'n_eval_tokens': int(len(Xev)), 'theta_star': float(th), 'eval_fve': float(st['fve']),
           'eval_l0': float(st['l0']), 'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()),
           'history': hist}
    save_checkpoint(out_dir / 'sae_theta.pt', sae, norm, extra={'config': 'eci_v2', 'cfg': rec['cfg'], 'seed': seed,
                                                                'theta_star': float(th)})
    (out_dir / 'train.json').write_text(json.dumps(rec, indent=1, default=float))
    return rec


def load(path, dev):
    """-> (sae, norm) with theta* baked in (threshold mode)."""
    from src.eci.sae import load_sae
    sae, norm, ck = load_sae(Path(path), dev)
    assert float(sae.threshold) > 0, f'{path}: no theta*'
    return sae, norm


def deploy_seed(records):
    """{seed: train record} -> the seed with the highest held-out FVE (label-free)."""
    return int(max(records, key=lambda s: records[s]['eval_fve']))


@torch.no_grad()
def frame_codes(sae, norm, x, fl, B, pos=None):
    """Kept-token SAE inputs x (n, d) of B frames (frame index fl) -> codes_max (B, W) float32 (max over the frame's
    readout tokens; 0 for a frame without tokens), argmax token position (B, W) int16 (-1 = silent; lowest position on
    ties; pos (n,) the grid positions, None = no argmax), and the token codes (n, W)."""
    W = sae.n_latents
    c = sae.encode(norm(x), mode='threshold') if len(x) else torch.zeros((0, W), device=x.device)
    cm = torch.zeros((B, W), dtype=torch.float32, device=x.device)
    cm.index_reduce_(0, fl, c, 'amax', include_self=True)
    am = None
    if pos is not None:
        big = 1 << 30
        hit = (c == cm[fl]) & (c > 0)
        cand = torch.where(hit, pos[:, None].expand_as(c), torch.full_like(c, big, dtype=torch.int64))
        am = torch.full((B, W), big, dtype=torch.int64, device=x.device)
        am.scatter_reduce_(0, fl[:, None].expand_as(cand), cand, 'amin', include_self=True)
        am[am == big] = -1
        am = am.to(torch.int16)
    return cm, am, c
