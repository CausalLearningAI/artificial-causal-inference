"""
CPU unit tests of src/eci/sae_gated.py (tiny sizes, ~1 min). Run through Slurm (defaultp), e.g.
    srun -p defaultp -c 4 --mem 8G -t 00:15:00 python scripts/eci/gated_sae_test.py
Checks: (1) P(gate > 0) of the hard-concrete sample equals sigmoid(g) (Monte Carlo, 4 SE); (2) gradients reach every
parameter and are finite; (3) the logged expected L0 equals sum_j sigmoid(g_j) and matches the Monte-Carlo count of open
gates; (4) the dual update drives E[L0] to the target on synthetic sparse data; (5) inference is deterministic and equals
1[g > 0] * m; (6) noisy-OR evidence == 1 - prod(1 - p) in float64; (7) bouts / latency / time budget on a toy series;
(8) save / load round trip.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eci.sae import TokenNorm, _train_step  # noqa: E402
from src.eci.sae_gated import (MatryoshkaGatedL0SAE, bouts, evidence_to_belief, frame_evidence_noisy_or,  # noqa: E402
                               expected_gate, hard_concrete_sample, latency, p_gate_one, load_gated, save_gated, time_budget)

torch.set_num_threads(4)


def t1_gate_prob():
    torch.manual_seed(0)
    g = torch.linspace(-4, 4, 17)
    n = 200_000
    frac = (hard_concrete_sample(g[None].expand(n, -1).contiguous()) > 0).float().mean(0)
    p = torch.sigmoid(g)
    se = (p * (1 - p) / n).sqrt()
    err = ((frac - p).abs() / se.clamp_min(1e-6)).max()
    assert err < 4, (frac, p, err)
    # exact 0 and 1 both occur; P(gate = 1) and E[gate] match their analytic / tabulated values
    s = hard_concrete_sample(g[None].expand(n, -1).contiguous())
    assert s.min() == 0 and s.max() == 1
    p1 = p_gate_one(g)
    e1 = ((s == 1).float().mean(0) - p1).abs() / (p1 * (1 - p1) / n).sqrt().clamp_min(1e-6)
    eg_se = ((s.mean(0) - expected_gate(g)).abs() / (s.std(0) / n ** 0.5).clamp_min(1e-6)).max()
    eg = (s.mean(0) - expected_gate(g)).abs().max()
    assert e1.max() < 4 and eg_se < 4, (e1, eg, eg_se)
    print(f'1 ok: P(gate > 0) = sigmoid(g): max |MC - sigmoid| / SE = {float(err):.2f}; P(gate = 1): max err / SE '
          f'{float(e1.max()):.2f} (g = 0: {float(p1[8]):.3f}); E[gate] table max err {float(eg):.1e} = {float(eg_se):.2f} SE (g = 0: '
          f'{float(expected_gate(g)[8]):.3f})')


def t2_grads():
    torch.manual_seed(0)
    sae = MatryoshkaGatedL0SAE(d_in=16, n_latents=32, prefixes=(8, 16, 32), target_l0=4, dead_tokens=10, seed=0)
    sae.tokens_since_fired.fill_(100)  # force the aux branch: half the latents closed -> they stay dead
    with torch.no_grad():
        sae.b_gate[16:] = -30.0
    x = torch.randn(64, 16) * 4
    loss, logs = sae.forward_train(x)
    loss.backward()
    for n, p in sae.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n
    assert logs['aux'] > 0
    print(f"2 ok: grads non-zero and finite for {[n for n, _ in sae.named_parameters()]}; aux {logs['aux']:.3f}")


def t3_expected_l0():
    torch.manual_seed(1)
    sae = MatryoshkaGatedL0SAE(d_in=16, n_latents=32, prefixes=(16, 32), target_l0=4, seed=1)
    with torch.no_grad():
        sae.b_gate.normal_(0, 1.5)
    x = torch.randn(256, 16) * 4
    _, logs = sae.forward_train(x)
    el0 = float(sae.beliefs(x).sum(1).mean())
    assert abs(logs['el0'] - el0) < 1e-4, (logs['el0'], el0)
    g = sae.gate_logits(x).detach()
    mc = np.mean([float((hard_concrete_sample(g) > 0).sum(1).float().mean()) for _ in range(400)])
    assert abs(mc - el0) / el0 < 0.01, (mc, el0)
    print(f'3 ok: logged E[L0] {logs["el0"]:.4f} = sum sigmoid(g) {el0:.4f}; Monte-Carlo open gates {mc:.4f}')


def t4_dual():
    torch.manual_seed(2)
    d, m, n = 32, 64, 20_000
    D = torch.randn(48, d)
    D /= D.norm(dim=1, keepdim=True)
    codes = (torch.rand(n, 48) < 6 / 48).float() * (1 + torch.rand(n, 48) * 3)
    X = codes @ D + 0.05 * torch.randn(n, d)
    norm = TokenNorm(d).fit(X)
    for target in (3.0, 8.0):
        sae = MatryoshkaGatedL0SAE(d_in=d, n_latents=m, prefixes=(16, 64), target_l0=target, lam_lr=0.01,
                                   dead_tokens=50_000, seed=0)
        opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
        steps, hist = 3000, []
        for s in range(steps):
            b = torch.randint(0, n, (256,))
            logs, _ = _train_step(sae, opt, norm(X[b]), s, steps, 3e-3, 100, 1.0)
            hist.append((logs['el0'], logs['fve'], logs['lam']))
        sae.eval()
        with torch.no_grad():
            xx = norm(X[:5000])
            el0 = float(sae.beliefs(xx).sum(1).mean())
            z = sae.encode(xx)
            fve = float(1 - (sae.decode(z) - xx).pow(2).sum() / (xx - xx.mean(0)).pow(2).sum())
            det = float((z > 0).sum(1).float().mean())
            zm = sae.encode(xx, mode='mean')
            fve_m = float(1 - (sae.decode(zm) - xx).pow(2).sum() / (xx - xx.mean(0)).pow(2).sum())
            p = sae.beliefs(xx)
            mid = float(((p > 0.1) & (p < 0.9)).float().mean())
        assert abs(el0 - target) / target < 0.1, (target, el0, hist[-5:])
        print(f'4 ok: target {target}: final E[L0] {el0:.2f}, deterministic L0 {det:.2f}, hard-code FVE {fve:.3f}, '
              f'E[gate]-code FVE {fve_m:.3f}, lambda {sae.lam:.3f}, share of beliefs in [0.1, 0.9] {mid:.4f}')


def t5_deterministic():
    sae = MatryoshkaGatedL0SAE(d_in=16, n_latents=32, prefixes=(16, 32), target_l0=4, seed=3).eval()
    with torch.no_grad():
        sae.b_gate.normal_(0, 1.5)
        sae.b_mag.normal_(0, 1)
    x = torch.randn(128, 16) * 4
    torch.manual_seed(10)
    z1 = sae.encode(x)
    torch.manual_seed(11)
    z2 = sae.encode(x)
    g = sae.gate_logits(x)
    proj = (x - sae.b_dec) @ sae.W_enc
    ref = torch.relu(proj * sae.r_mag.exp() + sae.b_mag) * (g > 0)
    assert torch.equal(z1, z2) and torch.allclose(z1, ref)
    assert torch.equal(sae.encode(x, mode='threshold'), z1)
    print(f'5 ok: encode deterministic, = 1[g > 0] * m; L0 {float((z1 > 0).sum(1).float().mean()):.2f}')


def t6_noisy_or():
    torch.manual_seed(4)
    g = torch.randn(50, 7, dtype=torch.float64) * 2
    frame = torch.tensor(np.repeat(np.arange(5), 10))
    e = frame_evidence_noisy_or(g, frame, 5)
    P = evidence_to_belief(e.double())
    ref = 1 - torch.stack([(1 - torch.sigmoid(g[frame == f])).prod(0) for f in range(5)])
    assert torch.allclose(P, ref, atol=1e-6), (P - ref).abs().max()
    # chunked accumulation = one shot
    e2 = frame_evidence_noisy_or(g[:23], frame[:23], 5)
    e2 = frame_evidence_noisy_or(g[23:], frame[23:], 5, out=e2)
    assert torch.allclose(e, e2, atol=1e-5)
    print(f'6 ok: noisy-OR evidence matches 1 - prod(1 - p), max err {float((P - ref).abs().max()):.1e}')


def t7_temporal():
    b = np.array([0.1, 0.6, 0.7, 0.2, 0.5, 0.9, 0.9, 0.95])
    assert bouts(b) == [(1, 3), (5, 8)], bouts(b)
    assert latency(b) == 1 and latency(np.zeros(4)) is None and bouts(np.zeros(3)) == []
    assert abs(time_budget(b) - b.mean()) < 1e-12
    print('7 ok: bouts', bouts(b), 'latency', latency(b))


def t8_io():
    sae = MatryoshkaGatedL0SAE(d_in=16, n_latents=32, prefixes=(16, 32), target_l0=4, seed=5).eval()
    norm = TokenNorm(16).fit(torch.randn(1000, 16))
    x = torch.randn(64, 16)
    with tempfile.TemporaryDirectory() as d:
        save_gated(Path(d) / 's.pt', sae, norm, extra={'arm': 'test'})
        s2, n2, ck = load_gated(Path(d) / 's.pt')
    assert torch.equal(sae.encode(norm(x)), s2.encode(n2(x))) and ck['arm'] == 'test'
    print('8 ok: save / load round trip')


def t9_selective_init():
    torch.manual_seed(6)
    sae = MatryoshkaGatedL0SAE(d_in=16, n_latents=64, prefixes=(16, 64), target_l0=4, aux_code='magnitude', seed=6)
    x = torch.randn(20_000, 16) * 3
    proj0 = (x - sae.b_dec) @ sae.W_enc
    st = sae.init_selective(x, proj_std=5.0)
    g, m = sae._heads(x)
    rate = (g > 0).float().mean(0)
    assert abs(st['det_l0'] - 4) < 0.05 and st['E_L0'] < 8, st
    assert (rate - 4 / 64).abs().max() < 2e-3, rate
    # magnitude head unchanged by the rescaling: exp(r) * proj = proj0
    assert torch.allclose(m, torch.relu(proj0), atol=1e-4)
    sae.tokens_since_fired.fill_(10 ** 9)
    with torch.no_grad():
        sae.b_gate[32:] = -1e3  # keep half the latents closed -> dead
    loss, logs = sae.forward_train(x[:256])
    loss.backward()
    assert logs['aux'] > 0 and logs['n_dead'] == 32 and sae.W_enc.grad[:, 32:].abs().sum() > 0
    print(f"9 ok: selective init det L0 {st['det_l0']:.3f} (target 4), E[L0] {st['E_L0']:.2f}, per-latent on-rate "
          f"{float(rate.min()):.4f}-{float(rate.max()):.4f}; magnitude head preserved; magnitude aux {logs['aux']:.2f}")


def t10_nested():
    sae = MatryoshkaGatedL0SAE(d_in=16, n_latents=32, prefixes=(8, 16, 32), target_l0=4, l0_weighting='nested', seed=7)
    assert sae.l0_cost.tolist() == [3.0] * 8 + [2.0] * 8 + [1.0] * 16
    x = torch.randn(64, 16) * 4
    torch.manual_seed(0)
    loss, logs = sae.forward_train(x)
    p = sae.beliefs(x)
    pen = sum(float(p[:, :m].sum(1).mean()) for m in sae.prefixes)  # sum over prefixes of E[L0 of prefix]
    torch.manual_seed(0)
    flat = MatryoshkaGatedL0SAE(d_in=16, n_latents=32, prefixes=(8, 16, 32), target_l0=4, seed=7)
    loss_f, logs_f = flat.forward_train(x)
    assert abs((loss - loss_f).item() - logs['lam'] * (pen - logs['el0'])) < 1e-3 * abs(loss.item())
    print(f"10 ok: nested L0 cost = sum over prefixes of E[L0_i] ({pen:.3f} vs flat {logs['el0']:.3f})")


if __name__ == '__main__':
    for f in (t10_nested, t9_selective_init, t1_gate_prob, t2_grads, t3_expected_l0, t5_deterministic, t6_noisy_or, t7_temporal, t8_io, t4_dual):
        f()
    print('ALL TESTS PASSED')
