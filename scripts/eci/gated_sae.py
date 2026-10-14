"""
GATED_SAE: pre-registered test of a THRESHOLD-FREE sparse autoencoder for the mice ECI representation, in which each
latent outputs a BELIEF p in [0, 1] that its concept is present (src/eci/sae_gated.py: Matryoshka gated SAE with
hard-concrete / L0 stochastic binary gates; lambda = Lagrange multiplier of the label-free constraint E[L0] = target,
dual ascent; inference code z = 1[p > 0.5] * m). Question: is it an acceptable threshold-free replacement for BatchTopK
(global inference threshold theta), and does its belief read-out help an UNSUPERVISED neuron capture nose-nose (nn) /
nose-tail (nt) sniffing?

Harness = scripts/eci/sparsity_test.py / quick_repr.py (reused by import), changed ONLY in the sparsifier:
    tokens   DINOv2-base 448 px last layer + 1 s motion [x_t, x_t - x_{t-5}], fg448 mask, re-encoded once per job from the
             512 px dataset JPGs staged to local disk (quick_repr.encode_sets), checked against the store
    train    the 10,000 unlabelled res_pool train frames (1,468,712 kept tokens)
    eval     the 20,000-frame annotated B subset (weights w; nn = Y_nn OR Y_np, nt = Y_nt), usable = 16,775 frames
    SAE      4096 latents, Matryoshka prefixes (512, 1024, 2048, 4096), AuxK-like dead-latent loss (k_aux 512, 1/32, dead
             after 2M tokens), TokenNorm.fit_blocks([768, 768]), b_dec = geometric median, Adam 5e-4, 500 warm-up, cosine
             to 10%, 6000 steps x 4096 tokens (= sparsity_test.train_one)
    lambda   log lambda += 0.005 * (E[L0]_batch - target) / target every step, lambda_0 = 1 (no labels, no sweep)

Arms (3 seeds each): gate16 (target E[L0] = 16), gate8 (target 8): the model as specified (uniform init, belief AuxK).
    gate16s, gate8s (POST HOC, added after gate16 seed 0 collapsed to 310 / 4096 live latents; same rule): selective
    data-dependent init (MatryoshkaGatedL0SAE.init_selective, proj std 5) + AuxK with the magnitude code (Gao et al.).
    gate16n, gate8n (POST HOC, after gate16s seed 0 still lost 75% of latents, 93% / 56% / 20% / 2.5% alive per
    Matryoshka group): as *s + NESTED L0 (each prefix's sub-dictionary pays lambda * its own L0; same E[L0] target).
References (already measured on the same frames, results/vision/eci_sparsity/eval/): btk16, btk8, c448.

Frame read-outs of every latent (cross-fitted best neuron, vjepa.score construction, weighted, prefixes 512 / 4096):
    noisy   frame belief P = 1 - prod_i (1 - p_i) over kept tokens; scored as the evidence -log(1 - P) = sum_i softplus(g_i)
            (same ranking as P, no saturation). PRIMARY read-out of the pre-registered rule.
    max     max token belief max_i p_i, scored as softplus(max_i g_i) (same ranking).
    code    max over kept tokens of the deterministic code 1[p > 0.5] * m (comparable to the references' frame max).
Firing (honest top-1% eligibility, rate >= 1%): noisy / max = frame belief > 0.5; code = frame value > 0.

PRE-REGISTERED (fixed before any result; usable frames, prefix 4096, 3-seed means)
    acceptable  gate{k} is acceptable as the threshold-free choice iff, on the noisy read-out and on BOTH nn and nt, its AUROC
                is >= btk{k} - 0.02 and its AP >= btk{k} - 0.01 (non-inferiority at matched target sparsity). The
                two-sided reading (|delta| within the margins) is reported too.
    helps       iff on nn or nt: AP >= 1.5 x c448 (task-quoted 0.0552 / 0.0512) OR honest top-1% >= 0.25 (noisy read-out;
                the other read-outs are reported descriptively).
    calibration for the noisy-AUROC-selected nn (and nt) neuron, picked on half 0: weighted label rate vs frame belief in
                bins on half 1 (descriptive; the concept is not the label).

Steps
    run      GPU  stage JPGs -> encode once -> train all arms x seeds -> eval -> seed-0 sheets
    summary  any  OUT/table.md, OUT/summary.json (reads JSONs only)
Outputs: results/vision/eci_gated_sae/ (gitignored).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
OUT = Path(os.environ.get('GS_OUT') or REPO / 'results/vision/eci_gated_sae')
SPE = REPO / 'results/vision/eci_sparsity/eval'
D, G, WIDTH = 768, 32, 4096
LABELS = ['nose_nose', 'nose_tail']
# arm -> (target E[L0], selective init + magnitude AuxK, nested L0)
ARMCFG = {'gate16': (16.0, False, False), 'gate8': (8.0, False, False), 'gate16s': (16.0, True, False),
          'gate8s': (8.0, True, False), 'gate16n': (16.0, True, True), 'gate8n': (8.0, True, True)}
ARMS = {a: c[0] for a, c in ARMCFG.items()}
REFARM = {a: f'btk{int(c[0])}' for a, c in ARMCFG.items()}
REF = {'nose_nose': {'auroc': 0.692, 'ap': 0.0552, 'top1': 0.122},
       'nose_tail': {'auroc': 0.741, 'ap': 0.0512, 'top1': 0.099}}  # task-quoted c448 (usable, m4096)
READOUTS = ('noisy', 'max', 'code')
FIRE_THR = {'noisy': math.log(2.0), 'max': math.log(2.0), 'code': 0.0}  # belief > 0.5 <=> evidence > log 2
MIN_RATE = 0.01
COARSE = [0.0, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def sae_dir(arm, seed):
    return OUT / 'sae' / f'{arm}_s{seed}'


# ---------------------------------------------------------------------------------------------- train
def train_one(arm, seed, X, steps, out, dev, bs=4096, lr=5e-4, warmup=500, lam_lr=0.005):
    """sparsity_test.train_one with MatryoshkaGatedL0SAE (otherwise line for line the same)."""
    import torch
    from src.eci.sae import TokenNorm, _train_step, geometric_median
    from src.eci.sae_gated import MatryoshkaGatedL0SAE, save_gated
    target = ARMS[arm]
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    N, d = X.shape
    ii = torch.from_numpy(np.sort(g.choice(N, min(500_000, N), replace=False))).to(dev)
    init = X[ii].float()
    norm = TokenNorm(d).to(dev).fit_blocks(init, [D, D], seed=seed)
    _, selective, nested = ARMCFG[arm]
    sae = MatryoshkaGatedL0SAE(d_in=d, n_latents=WIDTH, prefixes=(WIDTH // 8, WIDTH // 4, WIDTH // 2, WIDTH),
                               target_l0=target, lam_lr=lam_lr, aux_code='magnitude' if selective else 'belief',
                               l0_weighting='nested' if nested else 'flat', seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    init_stats = sae.init_selective(norm(init[-65536:])) if selective else None
    if selective:
        log(f'    {arm} s{seed}: selective init {init_stats}')
    del init
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    perm, p = torch.randperm(N, generator=gen, device=dev), 0
    history, t0 = [], time.time()
    for step in range(steps):
        if p + bs > N:
            perm, p = torch.randperm(N, generator=gen, device=dev), 0
        b = perm[p:p + bs]
        p += bs
        logs, gn = _train_step(sae, opt, norm(X[b]), step, steps, lr, warmup, 1.0)
        if step % 250 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1), grad_norm=float(gn))
            history.append(logs)
            if step % 1000 == 0 or step == steps - 1:
                log(f"    {arm} s{seed} step {step}/{steps} fve {logs['fve']:.4f} E[L0] {logs['el0']:.2f} "
                    f"L0 sampled {logs['l0']:.2f} L0 det {logs['l0_det']:.2f} lambda {logs['lam']:.3f} "
                    f"dead {logs['n_dead']} {logs['elapsed_s']}s")
    sae.eval()
    # label-free sparsity check on held-out-of-batch train tokens (eval mode, deterministic): E[L0] and det L0
    with torch.no_grad():
        jj = torch.from_numpy(np.sort(np.random.default_rng(seed + 100).choice(N, min(200_000, N), replace=False))).to(dev)
        el0, dl0 = 0.0, 0.0
        for a in range(0, len(jj), 32768):
            x = norm(X[jj[a:a + 32768]])
            gl = sae.gate_logits(x)
            el0 += float(torch.sigmoid(gl).sum())
            dl0 += float((sae.encode(x) > 0).sum())
        el0, dl0 = el0 / len(jj), dl0 / len(jj)
    dead = int((sae.tokens_since_fired >= sae.dead_tokens).sum())
    extra = {'config': 'gated_sae', 'arm': arm, 'target_l0': target, 'seed': seed, 'train_frames': 'res_pool mice train',
             'init': 'selective' if selective else 'uniform', 'init_stats': init_stats, 'aux_code': sae.aux_code,
             'l0_weighting': sae.l0_weighting}
    save_gated(out / 'sae.pt', sae, norm, extra=extra)
    (out / 'train.json').write_text(json.dumps({
        **extra, 'n_pool': N, 'steps': steps, 'token_presentations': steps * bs, 'train_time_s': time.time() - t0,
        'final_lambda': sae.lam, 'train_sample_E_L0': el0, 'train_sample_det_L0': dl0, 'dead_at_end_train': dead,
        'dead_frac_train': dead / WIDTH, 'gpu': torch.cuda.get_device_name(0), 'history': history}, indent=1))
    log(f'    {arm} s{seed}: trained in {time.time() - t0:.0f}s; train-sample E[L0] {el0:.2f} (target {target}), '
        f'det L0 {dl0:.2f}, lambda {sae.lam:.3f}, dead {dead}/{WIDTH}')


# ---------------------------------------------------------------------------------------------- scoring
def cross_fit_rate(codes, Yt, wt, half, dev, thr):
    """contact_sae.cross_fit with the honest-top-1% firing rate = weighted fraction of frames with code > thr."""
    import torch
    import contact_sae as cs
    R = {}
    for h in (0, 1):
        rows = torch.from_numpy(np.flatnonzero(half == h)).to(dev)
        r = cs.align_w(codes, Yt, wt, rows)
        ww = wt[rows].double()
        r['rate'] = (((codes[rows].float() > thr).double() * ww[:, None]).sum(0) / ww.sum()).cpu().numpy()
        R[h] = r
    res = {}
    for k, b in enumerate(LABELS):
        dirs = []
        for s, t in ((0, 1), (1, 0)):
            S, T = R[s], R[t]
            e = {}
            for sel, q, honest in (('auc', 'auc', False), ('ap', 'ap', False), ('top1', 'prec_top', True)):
                sc = S[q][:, k].copy()
                if honest:
                    sc[S['rate'] < MIN_RATE] = -np.inf
                sc[~np.isfinite(sc)] = -np.inf
                j = int(np.argmax(sc))
                e[sel] = {'neuron': j, 'select_score': float(sc[j]), 'test_auc': float(T['auc'][j, k]),
                          'test_ap': float(T['ap'][j, k]), 'test_top1': float(T['prec_top'][j, k]),
                          'test_rate': float(T['rate'][j])}
            dirs.append(e)
        res[b] = {'base_half': [float(R[0]['base'][k]), float(R[1]['base'][k])],
                  'cf_auroc': float(np.mean([e['auc']['test_auc'] for e in dirs])),
                  'cf_ap_at_auroc': float(np.mean([e['auc']['test_ap'] for e in dirs])),
                  'cf_ap': float(np.mean([e['ap']['test_ap'] for e in dirs])),
                  'cf_top1': float(np.mean([e['top1']['test_top1'] for e in dirs])),
                  'cf_top1_rate': float(np.mean([e['top1']['test_rate'] for e in dirs])), 'dirs': dirs}
    return res


def score(codes, lab, rows, half_all, dev, thr):
    """vjepa.score with cross_fit_rate (size-controlled AUROC of the AUROC-picked neuron on its test half)."""
    import torch
    import diag_mice_patches as dmp
    import vjepa as vj
    L = lab.iloc[rows].reset_index(drop=True)
    half = half_all[rows]
    Y = L[LABELS].values.astype(bool)
    c = codes[torch.from_numpy(rows).to(dev)]
    r = cross_fit_rate(c, torch.from_numpy(Y).to(dev), torch.from_numpy(L.w.values).to(dev), half, dev, thr)
    dec = dmp.size_deciles(L.n_fg.values)
    for k, b in enumerate(LABELS):
        sc = []
        for di, e in enumerate(r[b]['dirs']):
            t = half == 1 - di
            col = c[:, e['auc']['neuron']].float().cpu().numpy()
            e['auc']['test_size_ctrl_auc'] = vj.size_ctrl(col[t], Y[t, k], L.w.values[t], dec[t])
            sc.append(e['auc']['test_size_ctrl_auc'])
        r[b]['cf_size_ctrl_auroc'] = float(np.nanmean(sc))
    return r


def reliability(belief, y, w, edges=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99, 1.0000001)):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (belief >= a) & (belief < b)
        ww = w[m]
        out.append({'bin': [a, min(b, 1.0)], 'weight_share': float(ww.sum() / w.sum()), 'n': int(m.sum()),
                    'mean_belief': float((belief[m] * ww).sum() / ww.sum()) if m.any() else float('nan'),
                    'label_rate': float((y[m] * ww).sum() / ww.sum()) if m.any() else float('nan')})
    return out


# ---------------------------------------------------------------------------------------------- eval
def frame_readouts(sae, norm, Xev, lens, dev, chunk=32768):
    """-> codes {noisy, max, code} (F, m) fp16 GPU + token-level statistics (FVE, L0, belief distribution)."""
    import torch
    import torch.nn.functional as Fnn
    from src.eci.sae_gated import expected_gate
    m, Fn = sae.n_latents, len(lens)
    frame = np.repeat(np.arange(Fn), lens)
    ev = torch.zeros(Fn, m, device=dev)
    gmax = torch.full((Fn, m), -1e4, device=dev)
    cmax = torch.zeros(Fn, m, device=dev)
    s1 = s2 = 0
    sse = {p: 0 for p in sae.prefixes}
    sse_mean = 0
    el0 = dl0 = ent = mid_tok = 0.0
    fire = torch.zeros(m, dtype=torch.long, device=dev)
    hist20 = torch.zeros(20, dtype=torch.float64, device=dev)
    coarse = torch.zeros(len(COARSE) - 1, dtype=torch.float64, device=dev)
    edges = torch.tensor(COARSE[1:-1], device=dev)
    with torch.no_grad():
        for a in range(0, len(Xev), chunk):
            x = norm(torch.from_numpy(np.ascontiguousarray(Xev[a:a + chunk])).to(dev))
            g, mag = sae._heads(x)
            p = torch.sigmoid(g)
            z = mag * (g > 0)
            s1 = s1 + x.double().sum(0); s2 = s2 + x.double().pow(2).sum(0)
            recon = sae.b_dec.expand_as(x)
            st = 0
            for pm in sae.prefixes:
                recon = recon + z[:, st:pm] @ sae.W_dec[st:pm]
                sse[pm] = sse[pm] + (x - recon).double().pow(2).sum(0)
                st = pm
            sse_mean = sse_mean + (x - sae.decode(mag * expected_gate(g))).double().pow(2).sum(0)
            el0 += float(p.double().sum())
            act = z > 0
            dl0 += float(act.sum())
            fire += act.sum(0)
            pc = p.clamp(1e-7, 1 - 1e-7)
            ent += float((-(pc * pc.log2() + (1 - pc) * (1 - pc).log2())).double().sum())
            mid_tok += float(((p >= 0.1) & (p <= 0.9)).sum())
            hist20 += torch.histc(p, bins=20, min=0, max=1).double()
            coarse += torch.bincount(torch.bucketize(p.flatten(), edges, right=True), minlength=len(COARSE) - 1).double()
            fr = torch.from_numpy(frame[a:a + chunk]).to(dev)
            ev.index_add_(0, fr, Fnn.softplus(g))
            gmax.index_reduce_(0, fr, g, 'amax', include_self=True)
            cmax.index_reduce_(0, fr, z, 'amax', include_self=True)
            del x, g, mag, p, z, recon, pc
    n = len(Xev)
    tss = (s2 - s1.pow(2) / n).sum()
    tot = float(hist20.sum())
    st = {'fve': float(1 - sse[sae.prefixes[-1]].sum() / tss),
          'fve_mean_gate_code': float(1 - sse_mean.sum() / tss),
          'fve_prefix': {str(p): float(1 - sse[p].sum() / tss) for p in sae.prefixes},
          'l0': dl0 / n, 'E_L0': el0 / n, 'dead_frac_tokens': float((fire == 0).float().mean()),
          'token_belief': {'hist20': (hist20 / tot).cpu().numpy().tolist(),
                           'coarse_edges': COARSE, 'coarse_share': (coarse / tot).cpu().numpy().tolist(),
                           'share_in_0.1_0.9': mid_tok / tot, 'mid_per_token': mid_tok / n,
                           'entropy_bits_per_token': ent / n}}
    codes = {'noisy': ev.clamp_max(6e4).half(), 'max': Fnn.softplus(gmax).clamp_max(6e4).half(), 'code': cmax.half()}
    del ev, gmax, cmax
    return codes, st


def frame_belief_stats(codes, w, rows):
    """frame-level belief distribution (usable rows): noisy-OR P = 1 - exp(-E) and max p."""
    import torch
    r = torch.from_numpy(rows).to(codes['noisy'].device)
    ww = torch.from_numpy(w[rows]).to(r.device).float()
    out = {}
    for ro in ('noisy', 'max'):
        P = -torch.expm1(-codes[ro][r].float())  # both stored as -log(1 - belief)
        h = torch.histc(P, bins=20, min=0, max=1).double()
        mid = ((P >= 0.1) & (P <= 0.9)).float()
        rate = (((P > 0.5).float() * ww[:, None]).sum(0) / ww.sum()).cpu().numpy()
        el = rate >= MIN_RATE
        out[ro] = {'hist20': (h / h.sum()).cpu().numpy().tolist(), 'share_in_0.1_0.9': float(mid.mean()),
                   'share_gt_0.99': float((P > 0.99).float().mean()), 'share_lt_0.01': float((P < 0.01).float().mean()),
                   'frac_latents_rate_gt_0.5': float((rate > 0.5).mean()),
                   'median_rate_eligible': float(np.median(rate[el])) if el.any() else float('nan'),
                   'n_eligible': int(el.sum()), 'dead_frac': float((rate == 0).mean())}
        del P
    return out


def eval_arm(arm, seed, Xev, tb, dev, half_all, check_cs=False):
    import torch
    import contact_sae as cs
    from src.eci.sae_gated import load_gated
    od = OUT / 'eval'
    od.mkdir(parents=True, exist_ok=True)
    key = f'{arm}_s{seed}'
    if (od / f'{key}.json').exists():
        log(f'[SKIP] eval {key}')
        return
    sae, norm, ck = load_gated(sae_dir(arm, seed) / 'sae.pt', dev)
    F, sub = tb['eval']['F'], tb['eval']['sub']
    t0 = time.time()
    codes, st = frame_readouts(sae, norm, Xev, tb['eval']['lens'], dev)
    rows = {'all': np.arange(len(F)), 'usable': np.flatnonzero(F.usable.values)}
    w = sub.w.values.astype(np.float64)
    res = {'stats': st, 'target_l0': ck['target_l0'], 'lambda': sae.lam, 'class': type(sae).__name__}
    res['frame_belief'] = frame_belief_stats(codes, w, rows['usable'])
    for ro in READOUTS:
        for fs, rr in rows.items():
            for p in (512, 4096):
                r = score(codes[ro][:, :p].contiguous(), sub, rr, half_all, dev, FIRE_THR[ro])
                res[f'{ro}/{fs}_m{p}'] = r
                if fs == 'usable':
                    log(f'  {key} {ro} {fs} m{p}: ' + ', '.join(
                        f"{b} AUROC {r[b]['cf_auroc']:.3f} AP {r[b]['cf_ap']:.4f} top1 {r[b]['cf_top1']:.3f} "
                        f"size-ctrl {r[b]['cf_size_ctrl_auroc']:.3f}" for b in LABELS))
    if check_cs:  # the code read-out with thr 0 must reproduce contact_sae.cross_fit exactly
        rr = rows['usable']
        c = codes['code'][torch.from_numpy(rr).to(dev)]
        L = sub.iloc[rr]
        ref = cs.cross_fit(c, torch.from_numpy(L[LABELS].values.astype(bool)).to(dev),
                           torch.from_numpy(L.w.values).to(dev), half_all[rr], dev)
        for b in LABELS:
            for q in ('cf_auroc', 'cf_ap', 'cf_top1'):
                assert abs(ref[b][q] - res['code/usable_m4096'][b][q]) < 1e-9, (b, q, ref[b][q])
        log('  check: code read-out == contact_sae.cross_fit (usable, m4096)')
    # reliability of the selected neurons' frame beliefs (selected on half 0, shown on half 1; usable)
    us = F.usable.values
    rel = {}
    for ro in ('noisy', 'max'):
        for b in LABELS:
            for sel in ('auc', 'top1'):
                j = res[f'{ro}/usable_m4096'][b]['dirs'][0][sel]['neuron']
                m = (half_all == 1) & us
                P = -np.expm1(-codes[ro][:, j].float().cpu().numpy()[m])
                rel[f'{ro}/{b}/{sel}'] = {'neuron': j, 'bins': reliability(P, sub[b].values[m].astype(float), w[m])}
    res['reliability'] = rel
    res['eval_s'] = time.time() - t0
    log(f"  {key}: fve {st['fve']:.4f} det L0 {st['l0']:.2f} E[L0] {st['E_L0']:.2f} token beliefs in [0.1, 0.9] "
        f"{st['token_belief']['share_in_0.1_0.9']:.5f} (per token {st['token_belief']['mid_per_token']:.2f}), entropy "
        f"{st['token_belief']['entropy_bits_per_token']:.2f} bits/token; frame {json.dumps(res['frame_belief'])[:400]}")
    (od / f'{key}.json').write_text(json.dumps(res, indent=1, default=float))
    del codes
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- sheets
def sheets(arm, seed, Xev, tb, jpg, dev, n_tiles=16):
    """Best nn / nt neurons of the noisy read-out (AUROC- and top-1%-selected on half 0, shown on half 1, ranked by frame
    belief): full frame t with the per-token belief overlay (red, alpha = p; white box = max-belief token) | crop t |
    crop t-5 (1 s). Border green = nn, blue = nt, grey = neither."""
    import torch
    from PIL import Image, ImageDraw
    import contact_sae as cs
    from src.eci.sae_gated import load_gated
    key = f'{arm}_s{seed}'
    R = json.loads((OUT / 'eval' / f'{key}.json').read_text())
    F, sub, pos, lens = tb['eval']['F'], tb['eval']['sub'], tb['eval']['pos'], tb['eval']['lens']
    half = cs.video_units(sub.obs.values)
    sae, norm, _ = load_gated(sae_dir(arm, seed) / 'sae.pt', dev)
    st = np.r_[0, np.cumsum(lens)]
    msk = (half == 1) & F.usable.values
    sd = OUT / 'sheets'
    sd.mkdir(parents=True, exist_ok=True)
    th, index = 200, []
    for b in LABELS:
        for sel in ('auc', 'top1'):
            e = R['noisy/usable_m4096'][b]['dirs'][0]
            j = e[sel]['neuron']
            fb = R['frame_belief']
            note = (f'noisy-OR {sel}-selected on half 0 for {b}; half 1: AUROC {e[sel]["test_auc"]:.3f} AP '
                    f'{e[sel]["test_ap"]:.3f} top1 {e[sel]["test_top1"]:.3f}; frame belief > 0.5 on '
                    f'{e[sel]["test_rate"]:.1%} of half-1 usable frames')
            # frame evidence of neuron j on all eval tokens (chunked)
            wj, bj = sae.W_enc[:, j], sae.b_gate[j]
            gj = np.empty(len(Xev), np.float32)
            with torch.no_grad():
                for a in range(0, len(Xev), 262144):
                    x = norm(torch.from_numpy(np.ascontiguousarray(Xev[a:a + 262144])).to(dev))
                    gj[a:a + len(x)] = ((x - sae.b_dec) @ wj + bj).cpu().numpy()
            evid = np.add.reduceat(np.logaddexp(0, gj), st[:-1])
            a_ = np.where(msk, evid, -1.0)
            chosen, per_v = [], {}
            for f in np.argsort(-a_):
                if a_[f] < 0 or len(chosen) >= n_tiles:
                    break
                v = F.obs.iat[f]
                if per_v.get(v, 0) >= 2:
                    continue
                per_v[v] = per_v.get(v, 0) + 1
                chosen.append(int(f))
            nc = 2
            nr = max(1, (len(chosen) + nc - 1) // nc)
            TW = 3 * th + 6
            sheet = Image.new('RGB', (nc * (TW + 8), nr * (th + 26) + 24), 'white')
            dr = ImageDraw.Draw(sheet)
            dr.text((4, 4), f'{key} neuron {j}: {note}', fill='black')
            dr.text((4, 13), 'tile: full t, token belief overlay (red alpha = p, white = max-p token) | crop t | crop t-5 '
                    '(1 s). border green nn, blue nt, grey none', fill='black')
            npos = {q: 0 for q in LABELS}
            for i, f in enumerate(chosen):
                pt = 1 / (1 + np.exp(-gj[st[f]:st[f + 1]]))
                pp = pos[st[f]:st[f + 1]].astype(int)
                ti = int(np.argmax(pt))
                r_, c_ = divmod(int(pp[ti]), G)
                s_ = 512 / G
                cy, cx = (r_ + 0.5) * s_, (c_ + 0.5) * s_
                side = 160
                x0 = int(np.clip(cx - side / 2, 0, 512 - side)); y0 = int(np.clip(cy - side / 2, 0, 512 - side))
                with Image.open(jpg / tb['eval']['path_t'][f]) as im:
                    full = im.convert('RGB')
                with Image.open(jpg / tb['eval']['path_p'][f]) as im:
                    prv = im.convert('RGB')
                ct = full.crop((x0, y0, x0 + side, y0 + side)).resize((th, th))
                cp = prv.crop((x0, y0, x0 + side, y0 + side)).resize((th, th))
                ov = Image.new('RGBA', full.size, (0, 0, 0, 0))
                d2 = ImageDraw.Draw(ov)
                for q, pq in zip(pp, pt):
                    rq, cq = divmod(int(q), G)
                    if pq > 0.02:
                        d2.rectangle([cq * s_, rq * s_, (cq + 1) * s_ - 1, (rq + 1) * s_ - 1],
                                     fill=(255, 0, 0, int(200 * pq)))
                d2.rectangle([c_ * s_, r_ * s_, (c_ + 1) * s_, (r_ + 1) * s_], outline=(255, 255, 255, 255), width=2)
                d2.rectangle([x0, y0, x0 + side, y0 + side], outline=(255, 220, 0, 255), width=2)
                full = Image.alpha_composite(full.convert('RGBA'), ov).convert('RGB')
                yn, yt = bool(sub.nose_nose.iat[f]), bool(sub.nose_tail.iat[f])
                npos['nose_nose'] += yn; npos['nose_tail'] += yt
                col = (0, 200, 0) if yn else (40, 110, 255) if yt else (150, 150, 150)
                tile = Image.new('RGB', (TW, th + 6), col)
                for k_, im in enumerate((full.resize((th, th)), ct, cp)):
                    tile.paste(im, (3 + k_ * th, 3))
                rr, cc = divmod(i, nc)
                X0, Y0 = cc * (TW + 8), 24 + rr * (th + 26)
                sheet.paste(tile, (X0, Y0))
                dr.text((X0 + 2, Y0 + th + 8), f'{F.obs.iat[f][:22]} P={-np.expm1(-evid[f]):.3f} maxp={pt.max():.2f} '
                        f'n(p>.5)={int((pt > 0.5).sum())}/{len(pt)}', fill='black')
            fn = f'{key}_noisy_{b}_{sel}.jpg'
            sheet.save(sd / fn, quality=85)
            index.append({'sheet': fn, 'neuron': j, 'note': note, 'n_tiles': len(chosen),
                          **{f'n_pos_{q}': npos[q] for q in LABELS}})
            log(f'sheet {fn}: {len(chosen)} tiles, positives {npos}')
    (sd / f'index_{key}.json').write_text(json.dumps(index, indent=1))


# ---------------------------------------------------------------------------------------------- run
def cmd_run(args):
    import torch
    import contact_sae as cs
    import quick_repr as qr
    import sparsity_test as spt
    dev = torch.device('cuda')
    t00 = time.time()
    stage = spt.stage_dir()
    tb = qr.tables()
    if args.smoke:
        tb = {'train': qr.take(tb['train'], np.arange(300)), 'eval': qr.take(tb['eval'], np.arange(0, 20000, 8))}
    half_all = cs.video_units(tb['eval']['sub'].obs.values)
    jpg = qr.stage_jpgs(tb, stage)
    X, einfo = qr.encode_sets('base', ['last'], tb, jpg, dev, args.bs, args.workers)
    X = X['last']
    einfo['store_check'] = qr.check_vs_store(X, tb, args.smoke)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'encode_{"_".join(args.arms)}.json').write_text(json.dumps(
        {'gpu': torch.cuda.get_device_name(0), **einfo, 'n_train_tokens': int(tb['train']['lens'].sum()),
         'n_eval_tokens': int(tb['eval']['lens'].sum()), 'wall_s_to_here': time.time() - t00}, indent=1, default=float))
    Xtr = torch.from_numpy(X['train']).to(dev)
    first = True
    for seeds in ([args.seeds[0]], args.seeds[1:]):
        if not seeds:
            continue
        for arm in args.arms:
            for seed in seeds:
                if (sae_dir(arm, seed) / 'sae.pt').exists():
                    log(f'[SKIP] train {arm}_s{seed}')
                else:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    train_one(arm, seed, Xtr, args.steps, sae_dir(arm, seed), dev)
                    torch.cuda.empty_cache()
                torch.backends.cuda.matmul.allow_tf32 = False
                eval_arm(arm, seed, X['eval'], tb, dev, half_all, check_cs=first)
                first = False
        if seeds == [args.seeds[0]]:
            for arm in args.arms:
                sheets(arm, args.seeds[0], X['eval'], tb, jpg, dev)
    del Xtr, X
    log(f'run done in {time.time() - t00:.0f}s')


# ---------------------------------------------------------------------------------------------- summary
def cmd_summary(args):
    E = OUT / 'eval'
    ms = lambda a, n=3: f'{np.mean(a):.{n}f}+-{np.std(a):.{n}f}' if len(a) > 1 else f'{np.mean(a):.{n}f}'  # noqa: E731
    G_ = {a: [json.loads(f.read_text()) for f in sorted(E.glob(f'{a}_s*.json'))] for a in ARMS}
    G_ = {a: v for a, v in G_.items() if v}
    Rf = {a: [json.loads(f.read_text()) for f in sorted(SPE.glob(f'{a}_s*.json'))] for a in ('c448', 'btk16', 'btk8')}
    rows = []  # (name, n, {key/label: {q: [..]}}, fve, l0)
    for a, rr in Rf.items():
        rows.append((a, 'frame max (theta)', rr, lambda r, key: r[key]))
    for a, rr in G_.items():
        for ro in READOUTS:
            rows.append((a, ro, rr, lambda r, key, ro=ro: r[f'{ro}/{key}']))
    S = {}
    for a, ro, rr, get in rows:
        s = {}
        for key in ('usable_m4096', 'usable_m512', 'all_m4096'):
            for b in LABELS:
                s[f'{key}/{b}'] = {q: [get(r, key)[b][q] for r in rr]
                                   for q in ('cf_auroc', 'cf_ap', 'cf_top1', 'cf_top1_rate', 'cf_size_ctrl_auroc')
                                   if q in get(rr[0], key)[b]}
        S[(a, ro)] = s
    lines = ['# gated_sae: threshold-free gated L0 SAE (beliefs) vs BatchTopK (mice B subset, weighted, cross-fitted best '
             'neuron, 3 seeds mean+-sd)', '',
             'read-outs: noisy = noisy-OR frame belief (primary); max = max token belief; code = frame max of 1[p > 0.5] m; '
             'references = frame max of the BatchTopK code (theta).', '']
    for key in ('usable_m4096', 'usable_m512', 'all_m4096'):
        lines += [f'## {key}', '', '| arm | read-out | seeds | nn AUROC | nn AP | nn top-1% | nn size-ctrl AUROC | '
                  'nt AUROC | nt AP | nt top-1% | nt size-ctrl AUROC | nn AP / c448 | nt AP / c448 |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for (a, ro), s in S.items():
            nn, nt = s[f'{key}/nose_nose'], s[f'{key}/nose_tail']
            n = len(nn['cf_auroc'])
            sc = lambda d: ms(d['cf_size_ctrl_auroc']) if 'cf_size_ctrl_auroc' in d else '-'  # noqa: E731
            lines.append(f"| {a} | {ro} | {n} | {ms(nn['cf_auroc'])} | {ms(nn['cf_ap'], 4)} | {ms(nn['cf_top1'])} | "
                         f"{sc(nn)} | {ms(nt['cf_auroc'])} | {ms(nt['cf_ap'], 4)} | {ms(nt['cf_top1'])} | {sc(nt)} | "
                         f"{np.mean(nn['cf_ap']) / REF['nose_nose']['ap']:.2f} | "
                         f"{np.mean(nt['cf_ap']) / REF['nose_tail']['ap']:.2f} |")
        lines.append('')
    # reconstruction / sparsity / beliefs
    lines += ['## reconstruction, sparsity and beliefs (eval tokens; frame stats on usable frames)', '',
              '| arm | eval FVE (hard code) | det L0 | E[L0] | train-sample E[L0] | lambda | dead frac (tokens) | '
              'token beliefs in [0.1, 0.9] | per token | entropy bits/token | share < 0.001 | share > 0.999 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for a, rr in Rf.items():
        lines.append(f"| {a} | {ms([r['stats']['fve'] for r in rr], 4)} | {ms([r['stats']['l0'] for r in rr], 2)} | - | - | "
                     f"theta {ms([r['theta'] for r in rr], 3)} | "
                     f"{ms([r['firing']['dead_frac_eval'] for r in rr], 4) if 'firing' in rr[0] else '-'} | - | - | - | - | - |")
    for a, rr in G_.items():
        tj = [json.loads((sae_dir(a, int(r_.stem.split('_s')[1])) / 'train.json').read_text())
              for r_ in sorted(E.glob(f'{a}_s*.json'))]
        tbel = [r['stats']['token_belief'] for r in rr]
        lines.append(f"| {a} | {ms([r['stats']['fve'] for r in rr], 4)} | {ms([r['stats']['l0'] for r in rr], 2)} | "
                     f"{ms([r['stats']['E_L0'] for r in rr], 2)} | {ms([t['train_sample_E_L0'] for t in tj], 2)} | "
                     f"{ms([r['lambda'] for r in rr], 3)} | {ms([r['stats']['dead_frac_tokens'] for r in rr], 4)} | "
                     f"{ms([t['share_in_0.1_0.9'] for t in tbel], 5)} | {ms([t['mid_per_token'] for t in tbel], 2)} | "
                     f"{ms([t['entropy_bits_per_token'] for t in tbel], 2)} | {ms([t['coarse_share'][0] for t in tbel], 4)} | "
                     f"{ms([t['coarse_share'][-1] for t in tbel], 6)} |")
    lines += ['', '| arm | frame read-out | frame beliefs in [0.1, 0.9] | > 0.99 | < 0.01 | frac latents with rate > 0.5 | '
              'median rate (eligible) | n eligible |', '|---|---|---|---|---|---|---|---|']
    for a, rr in G_.items():
        for ro in ('noisy', 'max'):
            f_ = [r['frame_belief'][ro] for r in rr]
            lines.append(f"| {a} | {ro} | {ms([x['share_in_0.1_0.9'] for x in f_], 4)} | "
                         f"{ms([x['share_gt_0.99'] for x in f_], 4)} | {ms([x['share_lt_0.01'] for x in f_], 4)} | "
                         f"{ms([x['frac_latents_rate_gt_0.5'] for x in f_], 4)} | "
                         f"{ms([x['median_rate_eligible'] for x in f_], 3)} | {ms([x['n_eligible'] for x in f_], 0)} |")
    # reliability (seed 0, noisy, AUROC-selected)
    lines += ['', '## reliability of the frame belief of the AUROC-selected neuron (seed 0, picked on half 0, half-1 usable '
              'frames; weighted). The concept is not the label: descriptive only.', '']
    for a, rr in G_.items():
        for ro in ('noisy', 'max'):
            for b in LABELS:
                rel = rr[0]['reliability'][f'{ro}/{b}/auc']
                lines.append(f"- {a} s0 {ro} {b} neuron {rel['neuron']}: " + '; '.join(
                    f"[{x['bin'][0]:.2f},{x['bin'][1]:.2f}) w {x['weight_share']:.3f} rate {x['label_rate']:.3f}"
                    for x in rel['bins'] if x['n'] > 0))
    # verdicts
    u = 'usable_m4096'
    V = {}
    lines += ['', '## pre-registered verdicts (usable, m4096, 3-seed means)', '']
    for a in G_:
        ref = S[(REFARM[a], 'frame max (theta)')]
        for ro in READOUTS:
            s = S[(a, ro)]
            d = {b: {'d_auroc': float(np.mean(s[f'{u}/{b}']['cf_auroc']) - np.mean(ref[f'{u}/{b}']['cf_auroc'])),
                     'd_ap': float(np.mean(s[f'{u}/{b}']['cf_ap']) - np.mean(ref[f'{u}/{b}']['cf_ap'])),
                     'd_top1': float(np.mean(s[f'{u}/{b}']['cf_top1']) - np.mean(ref[f'{u}/{b}']['cf_top1'])),
                     'ap_over_c448': float(np.mean(s[f'{u}/{b}']['cf_ap']) / REF[b]['ap']),
                     'top1': float(np.mean(s[f'{u}/{b}']['cf_top1']))} for b in LABELS}
            d['acceptable_noninferior'] = all(d[b]['d_auroc'] >= -0.02 and d[b]['d_ap'] >= -0.01 for b in LABELS)
            d['within_two_sided'] = all(abs(d[b]['d_auroc']) <= 0.02 and abs(d[b]['d_ap']) <= 0.01 for b in LABELS)
            d['helps'] = any(d[b]['ap_over_c448'] >= 1.5 or d[b]['top1'] >= 0.25 for b in LABELS)
            V[f'{a}/{ro}'] = d
            lines.append(f"- {a} [{ro}{' = PRIMARY' if ro == 'noisy' else ''}] vs {REFARM[a]}: acceptable "
                         f"(non-inferior) = {d['acceptable_noninferior']}, two-sided within = {d['within_two_sided']}, "
                         f"helps = {d['helps']}; " + '; '.join(
                             f"{b}: dAUROC {d[b]['d_auroc']:+.3f}, dAP {d[b]['d_ap']:+.4f}, dtop-1% {d[b]['d_top1']:+.3f}, "
                             f"AP x{d[b]['ap_over_c448']:.2f} c448, top-1% {d[b]['top1']:.3f}" for b in LABELS))
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'summary.json').write_text(json.dumps({'arms': {f'{a}/{ro}': s for (a, ro), s in S.items()}, 'verdict': V},
                                                 indent=1, default=float))
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['run', 'summary'])
    ap.add_argument('--arms', nargs='+', default=list(ARMS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--smoke', action='store_true', help='300 train / 2,500 eval frames; set GS_OUT and --steps')
    args = ap.parse_args()
    for a in args.arms:
        assert a in ARMS, a
    if args.smoke:
        assert os.environ.get('GS_OUT'), 'smoke: set GS_OUT to a scratch results dir'
    {'run': cmd_run, 'summary': cmd_summary}[args.step](args)


if __name__ == '__main__':
    main()
