"""
ECI v2.9: PRED-contact (= v2.7 PREDk) with the BatchTopK sparsity k varied (pre-registration scripts/eci/v2/v29_prereg.md).
The training loop is v27_train.cmd_train restricted to PREDk, with a per-SAE k: SAEs are given as ARM:SEED pairs with
ARM in PREDk8 / PREDk16 / PREDk32. Everything else (data stream, per-seed batch permutation, init, contact weights W_C from
the v2.7 prep, LR schedule, threshold calibration, checkpoints) is the v2.7 code path. All SAEs of one seed see the same
batches and start from the same weights.
usage: v29_train.py train --dom D --cfgs PREDk16:2901,PREDk8:2901,... [--max_steps N] [--hist_every N] [--tag NAME]
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v27_train as T7  # noqa: E402

TR, T6 = T7.TR, T7.T6
REPO = T7.REPO
BEE29 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_9')
OUT29 = REPO / 'results/vision/eci_v2_9'
KOF = {'PREDk8': 8, 'PREDk16': 16, 'PREDk32': 32}
CFGS = [('PREDk16', s) for s in range(2901, 2907)] + [(a, s) for a in ('PREDk8', 'PREDk32') for s in range(2901, 2904)]
log = TR.log


def cmd_train(a):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from src.eci.sae import TokenNorm, geometric_median, save_checkpoint, MatryoshkaBatchTopKSAE
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    import v20 as V
    SAE27 = T7.make_sae_class()
    dom = a.dom
    dev = torch.device('cuda')
    cfgs = [(c.split(':')[0], int(c.split(':')[1])) for c in a.cfgs.split(',')] if a.cfgs else CFGS
    seeds = list(dict.fromkeys(s for _, s in cfgs))
    steps = int(a.max_steps) if a.max_steps else T7.STEPS[dom]
    total = T7.STEPS[dom]
    ws = TR.windows(dom)
    o = np.random.default_rng(TR.WIN_RNG).permutation(len(ws))
    chunks = [[ws[i] for i in o[c0:c0 + TR.CW]] for c0 in range(0, len(o), TR.CW)]
    spec = V.R.spec()['sae']
    assert spec['k'] == 16
    lr, warmup = spec['lr'], 500
    nthr = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))
    st = T7.Store(dom)
    ex = ThreadPoolExecutor(max(4, nthr - 2))
    WC = json.loads((T7.OUT27 / f'{dom}_prep.json').read_text())['contact']['w']  # the v2.7 value (not re-measured)
    log(f'{dom}: steps {steps:,} (schedule {total:,}), {len(chunks)} chunks/pass, SAEs {cfgs}, WC {WC:.3f}')
    fcheck = {'windows': 0, 'tokens': 0}

    def gather(fs):
        xs, ps, cs = [], [], []
        for f in fs:
            x, xp, cp, _, keep = f.result()
            if len(x):
                xs.append(x)
                ps.append(xp)
                cs.append(cp)
            if keep is not None and len(x):
                rx, rp = T6.xp_np(keep[0]['tok'], keep[1])
                if not (np.array_equal(x.cpu().numpy().view(np.uint16), rx.view(np.uint16)) and np.array_equal(xp.cpu().numpy().view(np.uint16), rp.view(np.uint16))):
                    raise SystemExit('GPU FEATURE IDENTITY FAILED')
                fcheck['windows'] += 1
                fcheck['tokens'] += len(rx)
        return torch.cat(xs), torch.cat(ps), torch.cat(cs)
    pending = [ex.submit(st.load_xpc, w, dev, i < 16) for i, w in enumerate(chunks[0])]
    Xn, Pn, Cn = gather(pending)
    log(f'GPU == numpy (x_t, x_t+2), bit-exact: {fcheck}; contact-proxy fraction in chunk 0: {float(Cn.float().mean()):.4f}')
    pending = [ex.submit(st.load_xpc, w, dev) for w in chunks[1]]
    torch.backends.cuda.matmul.allow_tf32 = True

    def new_sae(s, k):  # = v27_train.new_sae for arm PREDk, with k
        torch.manual_seed(s)
        g = np.random.default_rng(s)
        ii = torch.from_numpy(np.sort(g.choice(len(Xn), min(250_000, len(Xn)), replace=False))).to(dev)
        init = Xn[ii].float()
        norm = TokenNorm(1024).to(dev).fit_blocks(init, [1024], seed=s)
        sae = SAE27(target='next', lam=1.0, d_in=1024, n_latents=4096, prefixes=T7.PREFIXES, k=k, seed=s).to(dev)
        with torch.no_grad():
            sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
            sae.b_dec1.copy_(sae.b_dec)
        return sae, norm
    M = {}
    for arm, s in cfgs:
        sae, norm = new_sae(s, KOF[arm])
        M[(arm, s)] = {'sae': sae, 'norm': norm, 'opt': torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999)), 'hist': []}
    by_seed = {s: [c for c in cfgs if c[1] == s] for s in seeds}
    gens = {s: torch.Generator(device=dev).manual_seed(s) for s in seeds}
    step, ci, cx, cp_, cc_, used = 0, 0, None, None, None, 0
    t0 = time.time()
    tlog, slog, util, wst = time.time(), 0, [], []
    he = int(a.hist_every)
    while step < steps:
        Xc = Xn if cx is None else torch.cat([cx, Xn])
        Pc = Pn if cp_ is None else torch.cat([cp_, Pn])
        Cc = Cn if cc_ is None else torch.cat([cc_, Cn])
        del Xn, Pn, Cn
        n = (len(Xc) // TR.BS) * TR.BS
        last = ci == len(chunks) - 1
        cx, cp_, cc_ = (None, None, None) if last else (Xc[n:].clone(), Pc[n:].clone(), Cc[n:].clone())
        Xg, Pg, Cg = Xc[:n], Pc[:n], Cc[:n]
        del Xc, Pc, Cc
        perms = {s: torch.randperm(n, generator=gens[s], device=dev) for s in seeds}
        for b in range(n // TR.BS):
            if step >= steps:
                break
            for s in seeds:
                ib = perms[s][b * TR.BS:(b + 1) * TR.BS]
                xb, pb, cb = Xg[ib], Pg[ib], Cg[ib]
                w = torch.where(cb, torch.tensor(WC, device=dev), torch.tensor(1.0, device=dev))
                if step % 2000 == 0 and s == seeds[0]:
                    wst.append([float(w.mean()), float((w * cb).sum() / w.sum())])
                for key in by_seed[s]:
                    m = M[key]
                    nm = m['norm']
                    lg = T7.train_step(m['sae'], m['opt'], nm(xb), nm(pb), w, step, total, lr, warmup)
                    if step % he == 0 or step == steps - 1:
                        lg.update(step=step, elapsed_s=round(time.time() - t0, 1))
                        m['hist'].append(lg)
            used += TR.BS
            step += 1
            if step % 2000 == 0:
                u, mem = TR.V23.gpu_util()
                util.append(u)
                el = time.time() - tlog
                k0 = next(iter(M))
                log(f'step {step:,}/{steps:,} chunk {ci + 1}: {(step - slog) / el:.2f} steps/s ({len(M)} SAEs), GPU util {u}% mem {mem} MiB, '
                    f'fve {k0} {M[k0]["hist"][-1]["fve"] if M[k0]["hist"] else float("nan"):.4f}, ETA {(steps - step) / ((step - slog) / el) / 60:.0f} min')
                tlog, slog = time.time(), step
        del Xg, Pg, Cg, perms
        if step >= steps or last:
            break
        ci += 1
        Xn, Pn, Cn = gather(pending)
        pending = [ex.submit(st.load_xpc, w, dev) for w in chunks[ci + 1]] if ci + 1 < len(chunks) else []
    tt = time.time() - t0
    for f in pending:
        f.cancel()
    ex.shutdown(wait=False, cancel_futures=True)
    NM = json.loads((TR.OUT25 / f'{dom}_count.json').read_text())['M_used']
    rec0 = {'node': socket.gethostname(), 'steps_run': step, 'steps_schedule': total, 'tokens_used_per_sae': used, 'passes': used / NM,
            'train_time_s': tt, 'steps_per_s': step / tt, 'n_saes_in_job': len(M), 'gpu_util_median': float(np.median(util)) if util else None,
            'gpu_identity': fcheck, 'contact_w': WC, 'batch_weight_stats_[mean, proxy share]': wst}
    log(f'train loop done: {json.dumps(rec0)}')
    if a.max_steps:  # identity / smoke run: histories only
        (OUT29 / 'train').mkdir(parents=True, exist_ok=True)
        out = {f'{arm}_s{s}': m['hist'] for (arm, s), m in M.items()}
        (OUT29 / 'train' / f'{dom}_{a.tag or "short"}.json').write_text(json.dumps({'run': rec0, 'history': out}, indent=1, default=float))
        return
    xc = torch.from_numpy(T6.sample_x(st, ws, TR.NCAL, 7))
    xe = torch.from_numpy(T6.sample_x(st, ws, TR.NCAL, 8))
    for (arm, s), m in M.items():
        sae, norm = m['sae'], m['norm']
        k = KOF[arm]
        sae.eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        th, _ = calibrate_threshold(sae, xc, norm=norm)  # mean L0 = sae.k on the calibration tokens
        set_threshold(sae, th)
        sst = sparsity_stats(sae, xe, th, norm=norm)
        fired = torch.zeros(sae.n_latents, dtype=torch.long, device=dev)
        with torch.no_grad():
            for b0 in range(0, len(xe), 65536):
                fired += (sae.encode(norm(xe[b0:b0 + 65536].to(dev).float()), mode='threshold') > 0).sum(0)
        key = f'{arm}_s{s}'
        plain = MatryoshkaBatchTopKSAE(d_in=1024, n_latents=4096, prefixes=T7.PREFIXES, k=k, seed=s)
        plain.load_state_dict({kk: v for kk, v in sae.state_dict().items() if kk not in ('W_dec1', 'b_dec1')})
        cfg = {**spec, 'k': k, 'W': 4096, 'prefixes': list(T7.PREFIXES), 'steps': total, 'arm': arm, 'target': sae.target, 'lam': sae.lam,
               'weights': f'contact proxy x {WC:.4f}', 'input': 'x_t', 'lag_frames': TR.L, 'edge_rule': f'{TR.L} <= t < T - {TR.L}'}
        rec = {'cfg': cfg, 'seed': s, 'key': key, **rec0, 'theta_star': float(th), 'eval_fve': float(sst['fve']), 'eval_l0': float(sst['l0']),
               'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()),
               'never_fire_on_eval_frac': float((fired == 0).float().mean()), 'history': m['hist']}
        out = BEE29 / dom / 'sae' / key
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(out / 'sae_theta.pt', plain, norm, extra={'config': 'eci_v2_9', 'cfg': cfg, 'seed': s, 'theta_star': float(th)})
        torch.save({'W_dec1': sae.W_dec1.detach().cpu(), 'b_dec1': sae.b_dec1.detach().cpu(), 'target': sae.target, 'lam': sae.lam}, out / 'dec1.pt')
        (out / 'train.json').write_text(json.dumps(rec, indent=1, default=float))
        (OUT29 / 'train').mkdir(parents=True, exist_ok=True)
        (OUT29 / 'train' / f'{dom}_{key}.json').write_text(json.dumps(rec, indent=1, default=float))
        log(f'{dom} {key}: k {k} theta* {th:.4f} eval FVE {sst["fve"]:.5f} L0 {sst["l0"]:.2f} dead(train) {rec["dead_at_end_train"]} '
            f'never-fire(eval) {rec["never_fire_on_eval_frac"]:.4f}')
    log('ALL CHECKPOINTS SAVED')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('train',))
    p.add_argument('--dom', default='mice', choices=('mice', 'ants'))
    p.add_argument('--cfgs', default='')
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--hist_every', type=int, default=5000)
    p.add_argument('--tag', default='')
    a = p.parse_args()
    cmd_train(a)
