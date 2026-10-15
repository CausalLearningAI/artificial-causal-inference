"""
MABe22 mouse ECI evaluation, step 3 (GPU Slurm job, one per arm): SAE training (userTrain only, no labels), theta*
calibration (held-out userTrain frames), frame read-out on the submission eval frames, cross-fitted scoring.

Arms (eci_common / the brief):   tokens   input            sparsifier                  Matryoshka
    A0     the lab recipe          m448     [z_t, z_t - z_lag]   BatchTopK k16 + theta*      512/1024/2048/4096
    A1     k = 8                   m448     [z_t, dz]            BatchTopK k8 + theta*       yes
    A2k8 / A2k16  frame budget     m448     [z_t, dz]            per-frame top (k n_f), no theta   yes
    A5     native 224 (16 x 16)    m224     [z_t, dz]            BatchTopK k16 + theta*      yes
    A6     no mask                 rnd448 (train / calib, matched count) / all448 (eval, all 1,024 tokens)
    A3     plain BatchTopK         m448     [z_t, dz]            BatchTopK k16 + theta*      no (one prefix 4096)
    A4     no temporal context     m448     z_t (768-d)          BatchTopK k16 + theta*      yes
SAE recipe (invariant_tokens.train_one / sparsity_test): 4096 latents, TokenNorm.fit_blocks per 768 block, b_dec =
geometric median, Adam 5e-4, 500 warm-up, cosine, 6000 steps x 4096 tokens (frame-budget: whole frames, >= 4096 tokens
per batch), AuxK as src/eci/sae.py, TF32 on for training (as motion_t3), off for calibration / read-out.
theta*: src/eci/sae_calibrate.calibrate_threshold on ALL tokens of the calibration frames (3 per userTrain clip).
Frame value = max over the frame's tokens of the sparse code (0 for a frame with no kept token).
Scoring (contact_sae.cross_fit, weighted by the eval inverse-probability weights): clip-level halves stratified so each
half holds positives of every behaviour (split seeds 0 and 1); best neuron on one half by AUROC / AP / honest top-1%
(eligible: weighted frame firing rate >= 1% on the selection half), scored on the other, both directions averaged.
Outputs: OUT/sae/<arm>_s<seed>/{sae.pt, train.json}; OUT/eval/<arm>_s<seed>.json; OUT/codes/<arm>_s0_{codes,argpos}.npy
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/eci'))
from eci_common import BEHAVIOURS, OUT, PLAN, TOK, log, stage_dir  # noqa: E402
import os  # noqa: E402
RUN = Path(os.environ.get('MABE_RUN') or OUT)  # where sae/ eval/ codes/ go (smoke / variant runs)

D = 768
ARMS = {
    'A0': dict(tok='m448', delta=True, sp='btk', k=16, matry=True),
    'D02': dict(tok='m448', delta=True, lag='6', sp='btk', k=16, matry=True),   # delta over 0.2 s
    'D05': dict(tok='m448', delta=True, lag='15', sp='btk', k=16, matry=True),  # 0.5 s
    'D2': dict(tok='m448', delta=True, lag='60', sp='btk', k=16, matry=True),   # 2 s (lag clipped to frame 0)
    'A1': dict(tok='m448', delta=True, sp='btk', k=8, matry=True),
    'A2k8': dict(tok='m448', delta=True, sp='frame', k=8, matry=True),
    'A2k16': dict(tok='m448', delta=True, sp='frame', k=16, matry=True),
    'A5': dict(tok='m224', delta=True, sp='btk', k=16, matry=True),
    'A6': dict(tok='rnd448', eval_tok='all448', delta=True, sp='btk', k=16, matry=True),
    'A3': dict(tok='m448', delta=True, sp='btk', k=16, matry=False),
    'A4': dict(tok='m448', delta=False, sp='btk', k=16, matry=True),
}
WIDTH = 4096
READOUTS = {'A0': (512, 1024, 4096)}


# ---------------------------------------------------------------------------------------------- data
def shards(set_):
    js = sorted(TOK.glob(f'{set_}_*of*.json'))
    js = [j for j in js if 'smoke' not in j.name]
    meta = [json.loads(j.read_text()) for j in js]
    assert meta and len(meta) == meta[0]['nshard'], [j.name for j in js]
    meta = sorted(meta, key=lambda m: m['shard'])
    return [f"{set_}_{m['shard']}of{m['nshard']}" for m in meta]


def lens_col(tok):
    return {'m448': 'n448', 'rnd448': 'n448', 'm224': 'n224', 'all448': None}[tok]


def load_shard(key, tok, delta, stage, lag=''):
    """-> X (N, 768 or 1536) fp16 numpy, lens (F,), frames DataFrame; tokens copied to local disk first."""
    Fm = pd.read_parquet(TOK / f'{key}_frames.parquet')
    arrs = {}
    for r in ('t', 'p') if delta else ('t',):
        src = TOK / f'{key}_{tok}_{r}{lag if r == "p" else ""}.npy'
        loc = stage / src.name
        if not loc.exists():
            shutil.copyfile(src, loc)
        a = np.load(loc)
        loc.unlink()
        arrs[r] = a.reshape(-1, D)
    lens = np.full(len(Fm), 1024, np.int64) if lens_col(tok) is None else Fm[lens_col(tok)].values.astype(np.int64)
    assert lens.sum() == len(arrs['t']), (key, tok, lens.sum(), arrs['t'].shape)
    if delta:
        X = np.empty((len(arrs['t']), 2 * D), np.float16)
        for c in range(0, len(X), 500_000):
            ta = arrs['t'][c:c + 500_000]
            X[c:c + len(ta), :D] = ta
            X[c:c + len(ta), D:] = (ta.astype(np.float32) - arrs['p'][c:c + 500_000].astype(np.float32)).astype(np.float16)
    else:
        X = arrs['t']
    return X, lens, Fm


def load_train(cfg, stage):
    """-> (X_train, lens_train, X_calib, lens_calib) over all train shards (frames in plan order)."""
    P = pd.read_parquet(PLAN / 'train_frames.parquet')
    out = {'train': ([], []), 'calib': ([], [])}
    for key in shards('train'):
        X, lens, Fm = load_shard(key, cfg['tok'], cfg['delta'], stage, cfg.get('lag', ''))
        role = Fm.merge(P, on=['clip', 'i'], how='left').role.values
        assert not pd.isna(role).any()
        tr = np.repeat(role, lens)
        for r in ('train', 'calib'):
            out[r][0].append(X[tr == r])
            out[r][1].append(lens[role == r])
    return tuple(np.concatenate(v) for r in ('train', 'calib') for v in out[r])


def halves(E, seed):
    """clip-level 2-fold split: behaviours from the rarest (fewest positive clips) up, the clips holding a positive of
    that behaviour (and not yet assigned) are shuffled and dealt alternately, starting with the half that holds fewer
    of that behaviour's positive clips; the remaining clips are dealt at random. -> (F,) int 0 / 1."""
    g = np.random.default_rng(seed)
    pos = {b: set(E['clip'][E[b]].unique()) for b in BEHAVIOURS}
    side = {}
    for b in sorted(BEHAVIOURS, key=lambda b: len(pos[b])):
        todo = [c for c in sorted(pos[b]) if c not in side]
        g.shuffle(todo)
        have = [sum(side.get(c) == h for c in pos[b]) for h in (0, 1)]
        h = int(have[1] < have[0])
        for c in todo:
            side[c] = h
            h = 1 - h
    rest = [c for c in sorted(E['clip'].unique()) if c not in side]
    g.shuffle(rest)
    for j, c in enumerate(rest):
        side[c] = j % 2
    return E['clip'].map(side).values.astype(np.int64)


# ---------------------------------------------------------------------------------------------- SAE
def build(cfg, din, seed):
    from src.eci.sae import MatryoshkaBatchTopKSAE
    from src.eci.sae_framebudget import FrameBudgetSAE
    pre = (WIDTH // 8, WIDTH // 4, WIDTH // 2, WIDTH) if cfg['matry'] else (WIDTH,)
    cls = FrameBudgetSAE if cfg['sp'] == 'frame' else MatryoshkaBatchTopKSAE
    return cls(d_in=din, n_latents=WIDTH, prefixes=pre, k=cfg['k'], seed=seed)


def train_one(arm, cfg, seed, X, lens, steps, out, dev, bs=4096, lr=5e-4, warmup=500):
    import torch
    from src.eci.sae import TokenNorm, _train_step, geometric_median, save_checkpoint
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    N, din = X.shape
    ii = torch.from_numpy(np.sort(g.choice(N, min(500_000, N), replace=False))).to(dev)
    init = X[ii].float()
    norm = TokenNorm(din).to(dev).fit_blocks(init, [D] * (din // D), seed=seed)
    sae = build(cfg, din, seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    frame_mode = cfg['sp'] == 'frame'
    if frame_mode:
        st = torch.from_numpy(np.r_[0, np.cumsum(lens)]).to(dev)
        nz = torch.from_numpy(np.flatnonzero(lens > 0)).to(dev)
        L_t = torch.from_numpy(lens).to(dev)
        fperm, fp = nz[torch.randperm(len(nz), generator=gen, device=dev)], 0
        mean_len = float(lens[lens > 0].mean())
        nfb = int(np.ceil(bs / mean_len)) + 64  # frames looked at per batch (enough to reach bs tokens)
    else:
        perm, p = torch.randperm(N, generator=gen, device=dev), 0
    history, t0, ntok, nfr = [], time.time(), 0, 0
    for step in range(steps):
        if frame_mode:
            if fp + nfb > len(fperm):
                fperm, fp = nz[torch.randperm(len(nz), generator=gen, device=dev)], 0
            cand = fperm[fp:fp + nfb]
            cl = torch.cumsum(L_t[cand], 0)
            nf = int(torch.searchsorted(cl, torch.tensor([bs], device=dev)).item()) + 1  # frames to reach >= bs
            nf = min(nf, len(cand))
            fr = cand[:nf]
            fp += nf
            ln = L_t[fr]
            seg = torch.repeat_interleave(torch.arange(nf, device=dev), ln)
            off = torch.arange(int(ln.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(ln, 0) - ln, ln)
            b = st[fr][seg] + off
            sae.set_frames(seg)
            nfr += nf
        else:
            if p + bs > N:
                perm, p = torch.randperm(N, generator=gen, device=dev), 0
            b = perm[p:p + bs]
            p += bs
        ntok += len(b)
        logs, gn = _train_step(sae, opt, norm(X[b]), step, steps, lr, warmup, 1.0)
        if step % 1000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1), batch_tokens=int(len(b)))
            history.append(logs)
            log(f"    {arm} s{seed} step {step}/{steps} fve {logs['fve']:.4f} l0 {logs['l0']:.2f} dead {logs['n_dead']} "
                f"thr {logs['threshold']:.3f} batch {len(b)} {logs['elapsed_s']}s")
    sae.eval()
    dead = int((sae.tokens_since_fired >= sae.dead_tokens).sum())
    extra = {'config': 'mabe22_eci', 'arm': arm, 'cfg': cfg, 'seed': seed}
    if frame_mode:
        extra['sparsity'] = 'frame_budget'
    save_checkpoint(out / 'sae.pt', sae, norm, extra=extra)
    info = {**extra, 'n_pool_tokens': N, 'n_pool_frames': int(len(lens)), 'steps': steps, 'token_presentations': ntok,
            'frames_per_batch_mean': (nfr / steps) if frame_mode else None, 'train_time_s': time.time() - t0,
            'dead_at_end_train': dead, 'threshold_ema': float(sae.threshold), 'gpu': torch.cuda.get_device_name(0),
            'history': history}
    (out / 'train.json').write_text(json.dumps(info, indent=1, default=float))
    log(f'    {arm} s{seed}: trained in {time.time() - t0:.0f}s, final fve {history[-1]["fve"]:.4f}, dead {dead}')


def load_any(path, dev):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('sparsity') == 'frame_budget':
        from src.eci.sae_framebudget import load_framebudget
        return load_framebudget(path, dev)
    from src.eci.sae import load_sae
    return load_sae(path, dev)


def frame_max(sae, norm, X, lens, dev, frame_mode, max_tok=16384):
    """-> codes (F, m) fp16 GPU (max over each frame's tokens), argmax local token index (F, m) int16 GPU, stats.
    Chunks of whole frames (padded to the chunk's longest frame)."""
    import torch
    Fn, m = len(lens), sae.n_latents
    st = np.r_[0, np.cumsum(lens)]
    codes = torch.zeros(Fn, m, dtype=torch.float16, device=dev)
    arg = torch.zeros(Fn, m, dtype=torch.int16, device=dev)
    s1 = s2 = sse = 0.0
    l0, f0 = 0, 0
    with torch.no_grad():
        while f0 < Fn:
            f1, Lm = f0 + 1, int(lens[f0])
            while (f1 < Fn and st[f1 + 1] - st[f0] <= max_tok
                   and (f1 + 1 - f0) * max(Lm, int(lens[f1]), 1) * m <= (1 << 27)):
                Lm = max(Lm, int(lens[f1]))
                f1 += 1
            a, b = st[f0], st[f1]
            if b > a:
                x = norm(torch.from_numpy(np.ascontiguousarray(X[a:b])).to(dev))
                ln = torch.from_numpy(lens[f0:f1]).to(dev)
                seg = torch.repeat_interleave(torch.arange(f1 - f0, device=dev), ln)
                z = sae.encode(x, seg=seg) if frame_mode else sae.encode(x, mode='threshold')
                xh = sae.decode(z)
                s1 = s1 + x.double().sum(0)
                s2 = s2 + x.double().pow(2).sum(0)
                sse = sse + (x - xh).double().pow(2).sum(0)
                l0 += int((z > 0).sum())
                Lm = int(ln.max())
                off = torch.arange(len(z), device=dev) - torch.repeat_interleave(torch.cumsum(ln, 0) - ln, ln)
                pad = torch.zeros(f1 - f0, Lm, m, device=dev)
                pad[seg, off] = z
                v, j = pad.max(1)
                codes[f0:f1] = v.half()
                arg[f0:f1] = j.to(torch.int16)
                del pad, z, xh, x
            f0 = f1
    N = int(st[-1])
    tss = s2 - s1.pow(2) / N
    return codes, arg, {'fve': float(1 - sse.sum() / tss.sum()), 'l0': l0 / N, 'n_tokens': N}


def score(codes, E, Ht, dev):
    """contact_sae.cross_fit for every split -> {split: per-behaviour result}."""
    import torch
    import contact_sae as cs
    cs.LABELS = list(BEHAVIOURS)
    Y = torch.from_numpy(E[BEHAVIOURS].values.astype(bool)).to(dev)
    w = torch.from_numpy(E.w.values.astype(np.float64)).to(dev)
    out = {}
    for s, half in Ht.items():
        r = cs.cross_fit(codes, Y, w, half, dev)
        for b in BEHAVIOURS:
            r[b]['cf_rate_auc_pick'] = float(np.mean([d['auc']['test_rate'] for d in r[b]['dirs']]))
            r[b]['cf_rate_ap_pick'] = float(np.mean([d['ap']['test_rate'] for d in r[b]['dirs']]))
        out[f'split{s}'] = r
    return out


def cmd_run(args):
    import torch
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    from src.eci.sae_framebudget import self_test
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    cfg = ARMS[args.arm]
    frame_mode = cfg['sp'] == 'frame'
    (RUN / 'sae').mkdir(parents=True, exist_ok=True)
    (RUN / 'eval').mkdir(parents=True, exist_ok=True)
    (RUN / 'codes').mkdir(parents=True, exist_ok=True)
    if frame_mode:
        for k in (8, 16):
            r = self_test(dev, k=k)
            log(f'frame-budget self-test: {r}')
            assert r['pass'], r
    todo = [s for s in args.seeds if not (RUN / 'sae' / f'{args.arm}_s{s}' / 'sae.pt').exists()]
    Xtr, ltr, Xca, lca = load_train(cfg, stage)
    log(f'{args.arm} {cfg}: train {Xtr.shape} on {len(ltr)} frames (mean {ltr.mean():.1f} tokens), calib {Xca.shape} '
        f'on {len(lca)} frames')
    if todo:
        Xg = torch.from_numpy(Xtr).to(dev)
        for seed in todo:
            train_one(args.arm, cfg, seed, Xg, ltr, args.steps, RUN / 'sae' / f'{args.arm}_s{seed}', dev)
            torch.cuda.empty_cache()
        del Xg
        torch.cuda.empty_cache()
    del Xtr
    torch.backends.cuda.matmul.allow_tf32 = False
    # theta* (BatchTopK arms) on the held-out calibration frames
    models, calib = {}, {}
    Xca_g = torch.from_numpy(Xca).to(dev)
    for seed in args.seeds:
        sae, norm, ck = load_any(RUN / 'sae' / f'{args.arm}_s{seed}' / 'sae.pt', dev)
        if frame_mode:
            calib[seed] = {'theta': None, 'note': 'frame budget: no threshold'}
            cs_ = frame_max(sae, norm, Xca, lca, dev, True)[2]
            calib[seed]['calib_fve_l0'] = cs_
        else:
            ema = float(sae.threshold)
            th, info = calibrate_threshold(sae, Xca_g, norm=norm)
            set_threshold(sae, th)
            info['theta_ema'] = ema
            info['calib_stats'] = sparsity_stats(sae, Xca_g, th, norm=norm)
            calib[seed] = info
            log(f'  s{seed} theta* {th:.4f} (EMA {ema:.4f}) calib L0 {info["l0_calib_theta_star"]:.3f}')
        models[seed] = (sae, norm)
    del Xca_g
    # eval read-out, streamed over the eval shards
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    etok = cfg.get('eval_tok', cfg['tok'])
    parts = {s: ([], [], []) for s in args.seeds}
    Fall, lens_all = [], []
    for key in shards('eval'):
        X, lens, Fm = load_shard(key, etok, cfg['delta'], stage, cfg.get('lag', ''))
        Fall.append(Fm)
        lens_all.append(lens)
        for seed, (sae, norm) in models.items():
            c, a, stt = frame_max(sae, norm, X, lens, dev, frame_mode)
            parts[seed][0].append(c)
            parts[seed][1].append(a)
            parts[seed][2].append(stt)
        del X
        log(f'  read-out {key} done')
    Fm = pd.concat(Fall).reset_index(drop=True)
    assert (Fm['clip'].values == E['clip'].values).all() and (Fm.i.values == E.i.values).all()
    lens = np.concatenate(lens_all)
    Ht = {s: halves(E, s) for s in (0, 1)}
    for s, h in Ht.items():
        miss = [b for b in BEHAVIOURS if min(E[b][h == 0].sum(), E[b][h == 1].sum()) == 0]
        assert not miss, (s, miss)
    for seed in args.seeds:
        codes = torch.cat(parts[seed][0])
        arg = torch.cat(parts[seed][1])
        sts = parts[seed][2]
        ntot = sum(x['n_tokens'] for x in sts)
        stats = {'l0': sum(x['l0'] * x['n_tokens'] for x in sts) / ntot, 'n_tokens': ntot,
                 'fve_shard_mean': float(np.mean([x['fve'] for x in sts]))}
        w = torch.from_numpy(E.w.values).to(dev).double()
        fr = (((codes > 0).double() * w[:, None]).sum(0) / w.sum()).cpu().numpy()
        res = {'arm': args.arm, 'cfg': cfg, 'seed': seed, 'calib': calib[seed], 'eval_stats': stats,
               'eval_frames': len(E), 'eval_frames_0_tokens': int((lens == 0).sum()),
               'firing': {'dead_frac_eval': float((fr == 0).mean()), 'n_eligible_1pct': int((fr >= 0.01).sum()),
                          'median_rate_alive': float(np.median(fr[fr > 0])) if (fr > 0).any() else None,
                          'frac_rate_gt_0.5': float((fr > 0.5).mean())},
               'halves': {f'split{s}': {'n_clips_half0': int(len(set(E['clip'][h == 0]))),
                                        'n_clips_half1': int(len(set(E['clip'][h == 1])))} for s, h in Ht.items()}}
        for m in READOUTS.get(args.arm, (4096,)):
            res[f'm{m}'] = score(codes[:, :m].contiguous(), E, Ht, dev)
            r0 = res[f'm{m}']['split0']
            log(f'  {args.arm} s{seed} m{m}: ' + ', '.join(
                f"{b[:10]} {r0[b]['cf_auroc']:.3f}/{r0[b]['cf_ap']:.4f}/{r0[b]['cf_top1']:.3f}" for b in BEHAVIOURS))
        res['fire_rate_w'] = fr.round(6).tolist()
        (RUN / 'eval' / f'{args.arm}_s{seed}.json').write_text(json.dumps(res, indent=1, default=float))
        if seed == args.seeds[0]:
            np.save(RUN / 'codes' / f'{args.arm}_s{seed}_codes.npy', codes.cpu().numpy())
            np.save(RUN / 'codes' / f'{args.arm}_s{seed}_argtok.npy', arg.cpu().numpy())
            np.save(RUN / 'codes' / f'{args.arm}_lens.npy', lens)
        del codes, arg
        torch.cuda.empty_cache()
    log(f'{args.arm} done in {time.time() - t00:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=list(ARMS), required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=6000)
    cmd_run(ap.parse_args())


if __name__ == '__main__':
    main()
