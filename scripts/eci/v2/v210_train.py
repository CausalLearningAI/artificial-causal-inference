"""
ECI v2.10: PRED-contact (PREDk16) on the validation benchmarks and frogs (pre-registration scripts/eci/v2/v210_prereg.md).
cmd_train = v29_train.cmd_train with ONLY these changes: (a) the window list comes from the set's pool TRAIN role
(v210_bench.roles; mice / ants keep TR.windows), (b) more than one pass: pass p uses the window order of rng WIN_RNG + p
(pass 0 = the v2.7 order), (c) W_C and the step count come from this run's prep (mice / ants: the v2.7 values),
(d) outputs under eci_codes/v2_10 and results/vision/eci_v2_10.
Steps:
  prep  --dom D   (CPU) train-role M_t token count N (edge rule), steps rule, contact-proxy fraction -> W_C, proxy identity
  train --dom D [--cfgs PREDk16:3001,...] [--max_steps N --idcheck]   (GPU)
  --idcheck (mice, PREDk16:2701, 5,001 steps): the history at steps 0 and 5,000 must equal the stored v2.7 PREDk_s2701
            history exactly, else exit 1.
"""
import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v29_train as T9  # noqa: E402
import v210_bench as BB  # noqa: E402

T7 = T9.T7
TR, T6 = T7.TR, T7.T6
REPO = T7.REPO
BEE210 = BB.BEE
OUT210 = BB.OUT
ANTS_STEPS = 25_561
KOF = T9.KOF
log = TR.log


def videos(dom):
    if dom in ('mice', 'ants'):
        return TR.videos(dom)
    nf = {(s, o): n for s, o, n in BB.manifest(dom)}
    tr = BB.roles(dom)['train']
    keys = [tuple(x) for x in tr] if dom == 'frogs' else [(dom, o) for o in tr]
    return [(s, o, nf[(s, o)]) for s, o in keys]


def windows(dom):
    if dom in ('mice', 'ants'):
        return TR.windows(dom)
    out = []
    for ser, obs, nf in videos(dom):
        out += [(ser, obs, i0, min(nf - TR.L, i0 + TR.WF), nf) for i0 in range(TR.L, nf - TR.L, TR.WF)]
    return out


def plan(dom):
    """-> (WC, steps, N tokens per pass, passes)"""
    if dom in ('mice', 'ants'):
        WC = json.loads((T7.OUT27 / f'{dom}_prep.json').read_text())['contact']['w']
        N = json.loads((TR.OUT25 / f'{dom}_count.json').read_text())['M_used']
        return WC, T7.STEPS[dom], N, 1
    pj = json.loads((OUT210 / 'prep' / f'{dom}_prep.json').read_text())
    return pj['contact']['w'], pj['steps'], pj['M_used'], pj['passes']


# ---------------------------------------------------------------------------------------------- prep
def cmd_prep(a):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    dom = a.dom
    assert dom not in ('mice', 'ants')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 16)))
    vids = videos(dom)
    L = TR.L

    def one(v):
        z = np.load(TR.STORE / dom / TR.VER / 'videos' / v[0] / v[1] / 'index.npz')
        ptr, fl = z['ptr'], z['flag']
        assert len(z['k']) == v[2]
        if v[2] <= 2 * L:
            return 0
        return int(((fl[int(ptr[L]):int(ptr[v[2] - L])] & 1) > 0).sum())
    with ThreadPoolExecutor(16) as ex:
        cnt = list(ex.map(one, vids))
    N = int(sum(cnt))
    ws = windows(dom)
    per_pass = N // TR.BS
    passes = 1 if per_pass >= ANTS_STEPS else math.ceil(ANTS_STEPS / per_pass)
    steps = (passes * N) // TR.BS
    st = T7.Store(dom)
    cs, n = [], 0
    for i in np.random.default_rng(6).permutation(len(ws)):  # = the v2.7 prep sample order
        x, _, c, _, _ = st.load_xpc(ws[i])
        cs.append(c)
        n += len(x)
        if n >= 2_000_000:
            break
    Cp = np.concatenate(cs)
    p = float(Cp.mean())
    if p > 0:
        wc_raw = T7.SHARE_K * (1 - p) / ((1 - T7.SHARE_K) * p)
        wc = min(T7.CAP_K, wc_raw)
        share = wc * p / (wc * p + (1 - p))
    else:
        wc_raw, wc, share = None, 1.0, 0.0
    rng = np.random.default_rng(5)
    bad, tot = 0, 0
    for j, vi in enumerate(rng.choice(len(vids), min(20, len(vids)), replace=False)):
        ser, obs, nf = vids[vi]
        if nf - L - 3 <= L:
            continue
        i0 = L if j < 2 else (nf - L - 3 if j < 4 else int(rng.integers(L, nf - L - 3)))
        W = st.video(ser, obs).window(i0, i0 + 3)
        I = TR.centre_index(W, nf)
        g = T7.grid_of(dom, ser, obs)
        got = T7.proxy_of_window(W, I, g)
        ref = T7.proxy_slow(W, I, g)
        bad += int((got != ref).sum())
        tot += len(got)
    res = {'dom': dom, 'train_videos': len(vids), 'windows': len(ws), 'M_used': N, 'steps_per_pass': per_pass, 'passes': passes,
           'steps': steps, 'steps_rule': f'1 pass if >= {ANTS_STEPS} steps, else the smallest whole number of passes reaching it',
           'sample_tokens': int(len(Cp)), 'contact': {'proxy_token_frac': p, 'w_raw': wc_raw, 'w': wc,
                                                      'capped': (wc_raw is not None and wc < wc_raw), 'proxy_weight_share': share,
                                                      'no_proxy_tokens': p == 0},
           'proxy_identity': {'tokens': tot, 'mismatches': bad}, 'node': socket.gethostname()}
    (OUT210 / 'prep').mkdir(parents=True, exist_ok=True)
    (OUT210 / 'prep' / f'{dom}_prep.json').write_text(json.dumps(res, indent=1))
    log(f'prep {dom}: {json.dumps(res)}')
    if bad:
        raise SystemExit('PROXY IDENTITY FAILED')


# ---------------------------------------------------------------------------------------------- train (= v29_train.cmd_train)
def cmd_train(a):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from src.eci.sae import TokenNorm, geometric_median, save_checkpoint, MatryoshkaBatchTopKSAE
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    import v20 as V
    SAE27 = T7.make_sae_class()
    dom = a.dom
    dev = torch.device('cuda')
    cfgs = [(c.split(':')[0], int(c.split(':')[1])) for c in a.cfgs.split(',')]
    seeds = list(dict.fromkeys(s for _, s in cfgs))
    WC, total, NM, passes = plan(dom)
    steps = int(a.max_steps) if a.max_steps else total
    ws = windows(dom)
    chunks = []
    for ps in range(passes):  # (b) pass p: window order rng WIN_RNG + p (pass 0 = v2.7 / v2.9)
        o = np.random.default_rng(TR.WIN_RNG + ps).permutation(len(ws))
        chunks += [[ws[i] for i in o[c0:c0 + TR.CW]] for c0 in range(0, len(o), TR.CW)]
    spec = V.R.spec()['sae']
    assert spec['k'] == 16
    lr, warmup = spec['lr'], 500
    nthr = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))
    st = T7.Store(dom)
    ex = ThreadPoolExecutor(max(4, nthr - 2))
    log(f'{dom}: steps {steps:,} (schedule {total:,}), {passes} pass(es), {len(chunks)} chunks, {len(ws)} windows/pass, SAEs {cfgs}, WC {WC:.3f}')
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
    pending = [ex.submit(st.load_xpc, w, dev) for w in chunks[1]] if len(chunks) > 1 else []
    torch.backends.cuda.matmul.allow_tf32 = True

    def new_sae(s, k):  # = v29_train.new_sae
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
                log(f'step {step:,}/{steps:,} chunk {ci + 1}/{len(chunks)}: {(step - slog) / el:.2f} steps/s ({len(M)} SAEs), GPU util {u}% mem {mem} MiB, '
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
    rec0 = {'node': socket.gethostname(), 'steps_run': step, 'steps_schedule': total, 'tokens_used_per_sae': used, 'passes': used / NM,
            'passes_planned': passes, 'train_time_s': tt, 'steps_per_s': step / tt, 'n_saes_in_job': len(M),
            'gpu_util_median': float(np.median(util)) if util else None, 'gpu_identity': fcheck, 'contact_w': WC,
            'batch_weight_stats_[mean, proxy share]': wst}
    log(f'train loop done: {json.dumps(rec0)}')
    if a.max_steps:  # identity / smoke run: histories only
        (OUT210 / 'train').mkdir(parents=True, exist_ok=True)
        out = {f'{arm}_s{s}': m['hist'] for (arm, s), m in M.items()}
        res = {'run': rec0, 'history': out}
        if a.idcheck:
            assert dom == 'mice' and cfgs == [('PREDk16', 2701)] and steps == 5001
            ref = {h['step']: h for h in json.loads((T7.OUT27 / 'train' / 'mice_PREDk_s2701.json').read_text())['history']}
            got = {h['step']: h for h in out['PREDk16_s2701']}
            fl = ('loss', 'L0', 'L1', 'aux', 'n_dead', 'fve', 'l0')
            mis = [(s_, f_, got[s_][f_], ref[s_][f_]) for s_ in (0, 5000) for f_ in fl if got[s_][f_] != ref[s_][f_]]
            res['identity'] = {'reference': 'results/vision/eci_v2_7/train/mice_PREDk_s2701.json', 'steps': [0, 5000], 'fields': fl,
                               'mismatches': mis, 'pass': not mis}
            log(f'IDENTITY vs v2.7 PREDk_s2701: {"PASS" if not mis else "FAIL"} {mis}')
        (OUT210 / 'train' / f'{dom}_{a.tag or "short"}.json').write_text(json.dumps(res, indent=1, default=float))
        if a.idcheck and res['identity']['mismatches']:
            raise SystemExit('TRAINING IDENTITY FAILED')
        return
    xc = torch.from_numpy(T6.sample_x(st, ws, TR.NCAL, 7))
    xe = torch.from_numpy(T6.sample_x(st, ws, TR.NCAL, 8))
    for (arm, s), m in M.items():
        sae, norm = m['sae'], m['norm']
        k = KOF[arm]
        sae.eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        th, _ = calibrate_threshold(sae, xc, norm=norm)
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
               'weights': f'contact proxy x {WC:.4f}', 'input': 'x_t', 'lag_frames': TR.L, 'edge_rule': f'{TR.L} <= t < T - {TR.L}',
               'dom': dom, 'train_videos': 'v210_bench.roles(dom)["train"]' if dom not in ('mice', 'ants') else 'all store videos'}
        rec = {'cfg': cfg, 'seed': s, 'key': key, **rec0, 'theta_star': float(th), 'eval_fve': float(sst['fve']), 'eval_l0': float(sst['l0']),
               'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()),
               'never_fire_on_eval_frac': float((fired == 0).float().mean()), 'history': m['hist']}
        out = BEE210 / dom / 'sae' / key
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(out / 'sae_theta.pt', plain, norm, extra={'config': 'eci_v2_10', 'cfg': cfg, 'seed': s, 'theta_star': float(th)})
        torch.save({'W_dec1': sae.W_dec1.detach().cpu(), 'b_dec1': sae.b_dec1.detach().cpu(), 'target': sae.target, 'lam': sae.lam}, out / 'dec1.pt')
        (out / 'train.json').write_text(json.dumps(rec, indent=1, default=float))
        (OUT210 / 'train').mkdir(parents=True, exist_ok=True)
        (OUT210 / 'train' / f'{dom}_{key}.json').write_text(json.dumps(rec, indent=1, default=float))
        log(f'{dom} {key}: k {k} theta* {th:.4f} eval FVE {sst["fve"]:.5f} L0 {sst["l0"]:.2f} dead(train) {rec["dead_at_end_train"]} '
            f'never-fire(eval) {rec["never_fire_on_eval_frac"]:.4f}')
    log('ALL CHECKPOINTS SAVED')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('prep', 'train'))
    p.add_argument('--dom', default='cal', choices=('mice', 'ants', 'cal', 'mab', 'crim', 'frogs'))
    p.add_argument('--cfgs', default=','.join(f'PREDk16:{s}' for s in BB.SEEDS_P))
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--hist_every', type=int, default=5000)
    p.add_argument('--tag', default='')
    p.add_argument('--idcheck', action='store_true')
    a = p.parse_args()
    {'prep': cmd_prep, 'train': cmd_train}[a.step](a)
