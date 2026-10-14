"""
SPARSITY_TEST: fast, pre-registered test of the SAE sparsity mechanism for the mice ECI representation. Question: does
the sparsifier (BatchTopK + one global inference threshold theta vs plain per-token TopK) or the number of active
latents k help an UNSUPERVISED neuron capture nose-nose (nn) / nose-tail (nt) sniffing?

Everything except the sparsifier / k is the quick_repr C1 harness (scripts/eci/quick_repr.py, reused by import):
    tokens   DINOv2-base 448 px last layer + 1 s motion [x_t, x_t - x_{t-5}], fg448 mask, RE-ENCODED once per job from
             the 512 px dataset JPGs staged to local disk (quick_repr.encode_sets), checked against the store
    train    the 10,000 unlabelled res_pool train frames (1,468,712 kept tokens)
    eval     the 20,000-frame annotated B subset (weights w; nn = Y_nn OR Y_np, nt = Y_nt), usable = 16,775 frames
    SAE      4096 latents, Matryoshka prefixes (512, 1024, 2048, 4096), AuxK dead-latent loss (k_aux 512, 1/32,
             dead after 2M tokens), TokenNorm.fit_blocks([768, 768]), b_dec = geometric median, Adam 5e-4, 500 warm-up,
             cosine to 10%, 6000 steps x 4096 tokens (= motion_t3.train_one, the levers recipe)
    read-out frame value = MAX over kept tokens; cross-fitted best neuron (vjepa.score, weighted), prefixes 512 / 4096

Arms (3 seeds each)
    btk{4,8,16,32}  MatryoshkaBatchTopKSAE (src/eci/sae.py): train keeps the n*k largest activations of the batch;
                    inference z = pre * (pre > theta), theta = EMA of the smallest kept activation (as deployed)
    tk{4,8,16,32}   MatryoshkaTopKSAE (src/eci/sae_topk.py): exact per-token top-k in training AND inference, no theta
    c448            reference: the res_pool C1 SAEs (BatchTopK k = 16, store tokens, same 10,000 frames), evaluated
                    here (no training) = quick_repr's c448. btk16 is its retrain on re-encoded tokens (sanity check).
Firing: a neuron fires on a frame iff its frame value > 0 (BatchTopK: above theta on >= 1 kept token; TopK: in the
    top-k of >= 1 kept token). Per-neuron firing rate = weighted fraction of usable frames. 'Eligible' = rate >= 1%
    (contact_sae.MIN_RATE, the honest top-1% eligibility rule).

PRE-REGISTERED (fixed before any result; usable frames, prefix 4096, 3-seed means, reference = c448 as quoted in the
task: nn AUROC 0.692 / AP 0.0552 / top-1% 0.122, nt 0.741 / 0.0512 / 0.099; the c448 re-evaluated here is reported too)
    helps        an arm helps iff on nn or nt: AP >= 1.5 x c448 OR honest top-1% >= 0.25, AND the OTHER label's AUROC
                 is not more than 0.02 below c448's.
    TopK ok      tk{k} is 'within noise' of btk{k} iff on BOTH labels |delta AP| <= 0.01 (the stated seed noise) and
                 |delta AUROC| <= 0.02. If so at k = 16, TopK is acceptable as the threshold-free choice.
    best arm     (for the contact sheets) = the arm with the highest mean over nn / nt of AP / c448 AP; sheets are its
                 seed-0 best nn and nt neurons (AUROC- and top-1%-selected on half 0, shown on half 1).

RESULTS (jobs 808392 / 808393, L40S, 2026-10-14; smoke 808381; 3 SAE seeds; usable frames; mean +- sd)
    store check: re-encoded eval t vs BDIR tok448 cos min 0.9999998; train t vs store 0.9999998; t-5 vs d5 0.99996.
    sparsifier self-test (65,536 eval tokens): tk arms exactly k actives on every token; btk arms mean L0 ~ k, max 14-128.
    prefix 4096        nose_nose AUROC / AP / top-1%                  nose_tail AUROC / AP / top-1%
    c448 (reference)   0.692+-.023 / 0.0552+-.0105 / 0.122+-.012      0.741+-.053 / 0.0512+-.0111 / 0.099+-.010
    btk4               0.674+-.013 / 0.0512+-.0049 / 0.093+-.028      0.688+-.044 / 0.0355+-.0087 / 0.073+-.013
    btk8               0.673+-.011 / 0.0613+-.0142 / 0.141+-.027      0.697+-.046 / 0.0516+-.0096 / 0.121+-.009
    btk16              0.690+-.021 / 0.0599+-.0045 / 0.141+-.002      0.719+-.063 / 0.0398+-.0098 / 0.089+-.021
    btk32              0.674+-.014 / 0.0517+-.0045 / 0.110+-.012      0.704+-.031 / 0.0389+-.0026 / 0.082+-.008
    tk4                0.636+-.006 / 0.0463+-.0091 / 0.098+-.031      0.632+-.007 / 0.0299+-.0065 / 0.067+-.009
    tk8                0.635+-.023 / 0.0577+-.0041 / 0.146+-.017      0.721+-.030 / 0.0533+-.0114 / 0.105+-.016
    tk16               0.662+-.039 / 0.0423+-.0018 / 0.099+-.018      0.690+-.039 / 0.0323+-.0014 / 0.083+-.009
    tk32               0.658+-.019 / 0.0516+-.0067 / 0.113+-.010      0.703+-.046 / 0.0496+-.0186 / 0.099+-.030
    eval FVE c448 0.780 | btk4 0.641, 8 0.723, 16 0.781, 32 0.830 | tk4 0.628, 8 0.707, 16 0.766, 32 0.817.
    dead (never fires on eval): k = 4 arms 75-78% (AuxK does not rescue them); all other arms <= 2.5%.
    median frame firing rate of eligible (>= 1%) neurons: c448 0.091, btk4/8/16/32 0.188/0.065/0.094/0.123,
    tk4/8/16/32 0.149/0.085/0.105/0.157. AUROC-picked nn neuron fires on 0.74 (c448) / 0.29-0.71 (other arms) of frames.
    Verdicts: NO arm helps (best AP ratio 1.11x, btk8 nn; best top-1% 0.146, tk8 nn; rule needs 1.5x or 0.25).
    TopK is NOT within noise of BatchTopK at any k (AUROC gaps 0.016-0.056, TopK mostly lower; at k = 16 TopK is lower
    on every metric: nn AP -0.018). Best arm btk8 (1.06x mean AP ratio): its best neurons respond to two mice in close,
    aligned contact (head-to-head or head-to-rear) and do not separate nn from nt.

Steps
    run      GPU  stage JPGs -> encode once -> train all arms x seeds on the same tokens -> eval -> seed-0 sheets
    summary  any  OUT/table.md, OUT/summary.json (reads JSONs only)
Outputs: results/vision/eci_sparsity/ (gitignored).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
OUT = Path(os.environ.get('SP_OUT') or REPO / 'results/vision/eci_sparsity')
RPS = REPO / 'results/vision/eci_res_pool/mice/sae'
D, G, WIDTH = 768, 32, 4096
LABELS = ['nose_nose', 'nose_tail']
ARMS = {f'{s}{k}': (s, k) for s in ('btk', 'tk') for k in (4, 8, 16, 32)}
REF = {'nose_nose': {'auroc': 0.692, 'ap': 0.0552, 'top1': 0.122},
       'nose_tail': {'auroc': 0.741, 'ap': 0.0512, 'top1': 0.099}}  # task-quoted c448 (usable, m4096)
MIN_RATE = 0.01
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def stage_dir():
    """LOCAL_DIR (must be under /localhome, or under $TMPDIR when the node has no /localhome)."""
    d = Path(os.environ.get('LOCAL_DIR') or '')
    tmp = os.environ.get('TMPDIR') or '/nonexistent'
    if not (str(d).startswith('/localhome') or str(d).startswith(tmp)):
        raise RuntimeError(f'LOCAL_DIR must be under /localhome or $TMPDIR, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


def sae_dir(arm, seed):
    return (RPS / f'c448_s{seed}') if arm == 'c448' else (OUT / 'sae' / f'{arm}_s{seed}')


# ---------------------------------------------------------------------------------------------- train
def train_one(arm, seed, X, steps, out, dev, bs=4096, lr=5e-4, warmup=500):
    """motion_t3.train_one with the sparsifier class and k as arguments (otherwise line for line the same)."""
    import torch
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    from src.eci.sae_topk import MatryoshkaTopKSAE
    kind, k = ARMS[arm]
    cls = MatryoshkaTopKSAE if kind == 'tk' else MatryoshkaBatchTopKSAE
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    N, d = X.shape
    ii = torch.from_numpy(np.sort(g.choice(N, min(500_000, N), replace=False))).to(dev)
    init = X[ii].float()
    norm = TokenNorm(d).to(dev).fit_blocks(init, [D, D], seed=seed)
    sae = cls(d_in=d, n_latents=WIDTH, prefixes=(WIDTH // 8, WIDTH // 4, WIDTH // 2, WIDTH), k=k, seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
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
        if step % 1000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            history.append(logs)
            log(f"    {arm} s{seed} step {step}/{steps} fve {logs['fve']:.4f} l0 {logs['l0']:.2f} dead {logs['n_dead']} "
                f"thr {logs['threshold']:.3f} {logs['elapsed_s']}s")
    sae.eval()
    dead = int((sae.tokens_since_fired >= sae.dead_tokens).sum())
    extra = {'config': 'sparsity_test', 'arm': arm, 'sparsity': 'topk' if kind == 'tk' else 'batchtopk', 'k': k,
             'train_frames': 'res_pool mice train', 'seed': seed}
    save_checkpoint(out / 'sae.pt', sae, norm, extra=extra)
    (out / 'train.json').write_text(json.dumps({**extra, 'n_pool': N, 'steps': steps, 'token_presentations': steps * bs,
                                                'train_time_s': time.time() - t0, 'dead_at_end_train': dead,
                                                'dead_frac_train': dead / WIDTH, 'threshold': float(sae.threshold),
                                                'gpu': torch.cuda.get_device_name(0), 'history': history}, indent=1))
    log(f'    {arm} s{seed}: trained in {time.time() - t0:.0f}s, final fve {history[-1]["fve"]:.4f}, '
        f'l0 {history[-1]["l0"]:.2f}, dead {dead}/{WIDTH}, theta {float(sae.threshold):.3f}')


# ---------------------------------------------------------------------------------------------- eval
def firing(codes, w, rows):
    """codes (F, m) fp16 GPU -> per-neuron weighted firing rate (frame value > 0) on `rows` (numpy)."""
    import torch
    r = torch.from_numpy(rows).to(codes.device)
    ww = torch.from_numpy(w[rows]).to(codes.device).float()
    return (((codes[r] > 0).float() * ww[:, None]).sum(0) / ww.sum()).cpu().numpy()


def eval_arm(arm, seed, Xev, tb, dev, half_all):
    import torch
    import vjepa as vj
    from src.eci.sae_topk import load_sae_any
    od = OUT / 'eval'
    od.mkdir(parents=True, exist_ok=True)
    key = f'{arm}_s{seed}'
    if (od / f'{key}.json').exists():
        log(f'[SKIP] eval {key}')
        return
    sae, norm, ck = load_sae_any(sae_dir(arm, seed) / 'sae.pt', dev)
    log(f'  eval {key}: {type(sae).__name__} k {sae.k} theta {float(sae.threshold):.3f}')
    F, sub = tb['eval']['F'], tb['eval']['sub']
    # self-test of the sparsifier on 65,536 eval tokens: TopK must give <= k actives per token (== k where >= k
    # pre-activations are positive); BatchTopK uses the threshold
    with torch.no_grad():
        x = norm(torch.from_numpy(np.ascontiguousarray(Xev[:65536])).to(dev))
        z = sae.encode(x, mode='threshold')
        nact = (z > 0).sum(1)
        npos = (torch.relu(sae.pre_acts(x)) > 0).sum(1)
        test = {'tokens': 65536, 'l0_mean': float(nact.float().mean()), 'l0_max': int(nact.max()),
                'frac_l0_eq_k': float((nact == sae.k).float().mean()),
                'frac_eq_min_k_npos': float((nact == torch.clamp(npos, max=sae.k)).float().mean())}
        if ARMS.get(arm, ('btk',))[0] == 'tk':
            assert test['l0_max'] <= sae.k and test['frac_eq_min_k_npos'] == 1.0, test
    codes, st, _ = vj.frame_max_tokens(sae, norm, Xev, tb['eval']['lens'], dev)
    rows = {'all': np.arange(len(F)), 'usable': np.flatnonzero(F.usable.values)}
    res = {'stats': st, 'sparsifier_test': test, 'k': sae.k, 'theta': float(sae.threshold),
           'class': type(sae).__name__}
    for fs, rr in rows.items():
        for p in (512, 4096):
            r = vj.score(codes[:, :p].contiguous(), sub, rr, half_all, dev)
            res[f'{fs}_m{p}'] = r
            log(f'  {key} {fs} m{p}: ' + ', '.join(
                f"{b} AUROC {r[b]['cf_auroc']:.3f} AP {r[b]['cf_ap']:.4f} top1 {r[b]['cf_top1']:.3f}" for b in LABELS))
    w = sub.w.values.astype(np.float64)
    fr = firing(codes, w, rows['usable'])
    elig = fr >= MIN_RATE
    res['fire_rate_usable_w'] = fr.tolist()
    sel = {b: {s: [d[s]['test_rate'] for d in res['usable_m4096'][b]['dirs']] for s in ('auc', 'ap', 'top1')}
           for b in LABELS}
    res['firing'] = {'dead_frac_eval': float((fr == 0).mean()), 'n_eligible': int(elig.sum()),
                     'median_rate_eligible': float(np.median(fr[elig])) if elig.any() else float('nan'),
                     'median_rate_alive': float(np.median(fr[fr > 0])) if (fr > 0).any() else float('nan'),
                     'frac_rate_gt_0.5': float((fr > 0.5).mean()), 'selected_neuron_rates': sel}
    log(f"  {key}: fve {st['fve']:.4f} l0 {st['l0']:.2f} sparsifier test {test} firing "
        f"{ {k: v for k, v in res['firing'].items() if k != 'selected_neuron_rates'} } selected {sel}")
    (od / f'{key}.json').write_text(json.dumps(res, indent=1, default=float))
    del codes
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- sheets
def sheets(arm, seed, Xev, tb, jpg, dev, n_tiles=16):
    """quick_repr.sheets with the sparsity-aware loader: best nn / nt neurons (AUROC- and top-1%-selected on half 0,
    shown on half 1): full frame t (red = argmax patch, yellow = crop) | crop t | crop t-5. Border green = nn, blue = nt,
    grey = neither."""
    from PIL import Image, ImageDraw
    import contact_sae as cs
    import vjepa as vj
    from src.eci.sae_topk import load_sae_any
    key = f'{arm}_s{seed}'
    R = json.loads((OUT / 'eval' / f'{key}.json').read_text())
    F, sub, pos, lens = tb['eval']['F'], tb['eval']['sub'], tb['eval']['pos'], tb['eval']['lens']
    half = cs.video_units(sub.obs.values)
    fr = R['fire_rate_usable_w']
    picks = []
    for b in LABELS:
        for sel in ('auc', 'top1'):
            e = R['usable_m4096'][b]['dirs'][0]
            j = e[sel]['neuron']
            picks.append((f'best_{b}_{sel}', j, (half == 1) & F.usable.values,
                          f'{sel}-selected on half 0 for {b}; half 1: AUROC {e[sel]["test_auc"]:.3f} AP '
                          f'{e[sel]["test_ap"]:.3f} top1 {e[sel]["test_top1"]:.3f}; fires on {fr[j]:.1%} of usable frames'))
    sae, norm, _ = load_sae_any(sae_dir(arm, seed) / 'sae.pt', dev)
    J = sorted({p[1] for p in picks})
    codes, _, arg = vj.frame_max_tokens(sae, norm, Xev, lens, dev, want_arg=J)
    codes = codes[:, J].float().cpu().numpy()
    arg = arg.cpu().numpy()
    st = np.r_[0, np.cumsum(lens)]
    sd = OUT / 'sheets'
    sd.mkdir(parents=True, exist_ok=True)
    th, index = 200, []
    for name, j, msk, note in picks:
        q = J.index(j)
        a = codes[:, q].copy()
        a[~msk] = -1
        chosen, per_v = [], {}
        for f in np.argsort(-a):
            if a[f] <= 0 or len(chosen) >= n_tiles:
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
        dr.text((4, 13), 'tile: full t (red = argmax patch) | crop t | crop t-5 (1 s). border green nn, blue nt, grey none',
                fill='black')
        npos = {b: 0 for b in LABELS}
        for i, f in enumerate(chosen):
            ti = int(arg[f, q])
            assert st[f] <= ti < st[f + 1]
            p = int(pos[ti])
            r_, c_ = divmod(p, G)
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
            d2 = ImageDraw.Draw(full)
            d2.rectangle([c_ * s_, r_ * s_, (c_ + 1) * s_, (r_ + 1) * s_], outline=(255, 40, 40), width=3)
            d2.rectangle([x0, y0, x0 + side, y0 + side], outline=(255, 220, 0), width=2)
            yn, yt = bool(sub.nose_nose.iat[f]), bool(sub.nose_tail.iat[f])
            npos['nose_nose'] += yn; npos['nose_tail'] += yt
            col = (0, 200, 0) if yn else (40, 110, 255) if yt else (150, 150, 150)
            tile = Image.new('RGB', (TW, th + 6), col)
            for k_, im in enumerate((full.resize((th, th)), ct, cp)):
                tile.paste(im, (3 + k_ * th, 3))
            rr, cc = divmod(i, nc)
            X0, Y0 = cc * (TW + 8), 24 + rr * (th + 26)
            sheet.paste(tile, (X0, Y0))
            dr.text((X0 + 2, Y0 + th + 8), f'{F.obs.iat[f][:22]} k{int(F.k.iat[f])} a={a[f]:.2f}', fill='black')
        fn = f'{key}_{name}.jpg'
        sheet.save(sd / fn, quality=85)
        index.append({'sheet': fn, 'neuron': j, 'note': note, 'n_tiles': len(chosen),
                      **{f'n_pos_{b}': npos[b] for b in LABELS}})
        log(f'sheet {fn}: {len(chosen)} tiles, positives {npos}')
    (sd / f'index_{key}.json').write_text(json.dumps(index, indent=1))


# ---------------------------------------------------------------------------------------------- run
def cmd_run(args):
    import torch
    import contact_sae as cs
    import quick_repr as qr
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    tb = qr.tables()
    if args.smoke:
        tb = {'train': qr.take(tb['train'], np.arange(300)), 'eval': qr.take(tb['eval'], np.arange(0, 20000, 8))}
        args.steps = 100
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
    for seeds in ([args.seeds[0]], args.seeds[1:]):  # seed 0 of every arm first (interim read-out + sheets)
        if not seeds:
            continue
        for arm in args.arms:
            for seed in seeds:
                if arm != 'c448':
                    if (sae_dir(arm, seed) / 'sae.pt').exists():
                        log(f'[SKIP] train {arm}_s{seed}')
                    else:
                        torch.backends.cuda.matmul.allow_tf32 = True  # as motion_t3 train
                        train_one(arm, seed, Xtr, args.steps, sae_dir(arm, seed), dev)
                        torch.cuda.empty_cache()
                torch.backends.cuda.matmul.allow_tf32 = False
                eval_arm(arm, seed, X['eval'], tb, dev, half_all)
        if seeds == [args.seeds[0]]:
            for arm in args.arms:
                sheets(arm, args.seeds[0], X['eval'], tb, jpg, dev)
    log(f'run done in {time.time() - t00:.0f}s')


# ---------------------------------------------------------------------------------------------- summary
def cmd_summary(args):
    E = OUT / 'eval'
    arms = ['c448'] + list(ARMS)
    R = {a: [json.loads(f.read_text()) for f in sorted(E.glob(f'{a}_s*.json'))] for a in arms}
    R = {a: v for a, v in R.items() if v}
    ms = lambda a, n=3: f'{np.mean(a):.{n}f}+-{np.std(a):.{n}f}' if len(a) > 1 else f'{np.mean(a):.{n}f}'  # noqa: E731
    S = {}
    for a, rr in R.items():
        s = {'n_seeds': len(rr)}
        for key in ('usable_m4096', 'usable_m512', 'all_m4096'):
            for b in LABELS:
                s[f'{key}/{b}'] = {q: [r[key][b][q] for r in rr] for q in ('cf_auroc', 'cf_ap', 'cf_top1', 'cf_top1_rate')}
                s[f'{key}/{b}']['auc_sel_rate'] = [float(np.mean([d['auc']['test_rate'] for d in r[key][b]['dirs']]))
                                                   for r in rr]
        s['fve'] = [r['stats']['fve'] for r in rr]
        s['l0'] = [r['stats']['l0'] for r in rr]
        s['theta'] = [r.get('theta', float('nan')) for r in rr]
        if 'firing' in rr[0]:
            for q in ('dead_frac_eval', 'median_rate_eligible', 'median_rate_alive', 'frac_rate_gt_0.5', 'n_eligible'):
                s[q] = [r['firing'][q] for r in rr]
        else:  # old-format eval json
            fr = [np.array(r['fire_rate_w']) for r in rr]
            s['dead_frac_eval'] = [float((f == 0).mean()) for f in fr]
        S[a] = s
    lines = ['# sparsity_test: BatchTopK (theta) vs per-token TopK, k in {4, 8, 16, 32} (mice B subset, weighted, '
             'cross-fitted best neuron, 3 seeds mean+-sd)', '']
    for key in ('usable_m4096', 'usable_m512', 'all_m4096'):
        lines += [f'## {key}', '', '| arm | seeds | nn AUROC | nn AP | nn top-1% | nt AUROC | nt AP | nt top-1% | '
                  'nn AP / c448 | nt AP / c448 |', '|---|---|---|---|---|---|---|---|---|---|']
        for a, s in S.items():
            nn, nt = s[f'{key}/nose_nose'], s[f'{key}/nose_tail']
            rn = np.mean(nn['cf_ap']) / REF['nose_nose']['ap'] if key == 'usable_m4096' else float('nan')
            rt = np.mean(nt['cf_ap']) / REF['nose_tail']['ap'] if key == 'usable_m4096' else float('nan')
            lines.append(f"| {a} | {s['n_seeds']} | {ms(nn['cf_auroc'])} | {ms(nn['cf_ap'], 4)} | {ms(nn['cf_top1'])} | "
                         f"{ms(nt['cf_auroc'])} | {ms(nt['cf_ap'], 4)} | {ms(nt['cf_top1'])} | {rn:.2f} | {rt:.2f} |")
        lines.append('')
    lines += ['## reconstruction and firing (eval, usable frames for rates; frame fires iff max over kept tokens > 0)', '',
              '| arm | eval FVE | token L0 | theta | dead frac | median rate (eligible >= 1%) | median rate (alive) | '
              'frac neurons rate > 50% | rate of AUROC-picked nn / nt neuron | rate of top-1%-picked nn / nt neuron |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for a, s in S.items():
        g = lambda q, n=3: ms(s[q], n) if q in s else '-'  # noqa: E731
        u = 'usable_m4096'
        lines.append(f"| {a} | {g('fve', 4)} | {g('l0', 2)} | {g('theta', 3)} | {g('dead_frac_eval', 4)} | "
                     f"{g('median_rate_eligible')} | {g('median_rate_alive')} | {g('frac_rate_gt_0.5')} | "
                     f"{ms(s[u + '/nose_nose']['auc_sel_rate'], 2)} / {ms(s[u + '/nose_tail']['auc_sel_rate'], 2)} | "
                     f"{ms(s[u + '/nose_nose']['cf_top1_rate'], 2)} / {ms(s[u + '/nose_tail']['cf_top1_rate'], 2)} |")
    # verdicts
    u = 'usable_m4096'
    V = {}
    lines += ['', '## pre-registered verdicts (usable, m4096, 3-seed means; reference = task-quoted c448)', '']
    for a, s in S.items():
        if a == 'c448':
            continue
        v = {}
        for b, ob in (('nose_nose', 'nose_tail'), ('nose_tail', 'nose_nose')):
            apr = np.mean(s[f'{u}/{b}']['cf_ap']) / REF[b]['ap']
            t1 = np.mean(s[f'{u}/{b}']['cf_top1'])
            drop = REF[ob]['auroc'] - np.mean(s[f'{u}/{ob}']['cf_auroc'])
            v[b] = {'ap_over_c448': float(apr), 'top1': float(t1), 'pass': bool(apr >= 1.5 or t1 >= 0.25),
                    'other_label_auroc_drop': float(drop)}
        v['helps'] = any(v[b]['pass'] and v[b]['other_label_auroc_drop'] <= 0.02 for b in LABELS)
        v['mean_ap_ratio'] = float(np.mean([v[b]['ap_over_c448'] for b in LABELS]))
        V[a] = v
        lines.append(f"- {a}: helps = {v['helps']}; " + '; '.join(
            f"{b}: AP x{v[b]['ap_over_c448']:.2f}, top-1% {v[b]['top1']:.3f}, other-label AUROC drop "
            f"{v[b]['other_label_auroc_drop']:+.3f}" for b in LABELS))
    lines += ['', 'TopK vs BatchTopK at matched k (within noise iff on both labels |dAP| <= 0.01 and |dAUROC| <= 0.02):', '']
    for k in (4, 8, 16, 32):
        a, b_ = f'tk{k}', f'btk{k}'
        if a not in S or b_ not in S:
            continue
        d = {b: {'d_ap': float(np.mean(S[a][f'{u}/{b}']['cf_ap']) - np.mean(S[b_][f'{u}/{b}']['cf_ap'])),
                 'd_auroc': float(np.mean(S[a][f'{u}/{b}']['cf_auroc']) - np.mean(S[b_][f'{u}/{b}']['cf_auroc'])),
                 'd_top1': float(np.mean(S[a][f'{u}/{b}']['cf_top1']) - np.mean(S[b_][f'{u}/{b}']['cf_top1']))}
             for b in LABELS}
        d['within_noise'] = all(abs(d[b]['d_ap']) <= 0.01 and abs(d[b]['d_auroc']) <= 0.02 for b in LABELS)
        V[f'tk_vs_btk_k{k}'] = d
        lines.append(f"- k = {k}: within noise = {d['within_noise']}; " + '; '.join(
            f"{b}: dAP {d[b]['d_ap']:+.4f}, dAUROC {d[b]['d_auroc']:+.3f}, dtop-1% {d[b]['d_top1']:+.3f}" for b in LABELS))
    cand = {a: v['mean_ap_ratio'] for a, v in V.items() if a in ARMS}
    if cand:
        best = max(cand, key=cand.get)
        V['best_arm'] = best
        lines += ['', f'Best arm (highest mean AP / c448 over nn, nt): {best} (x{cand[best]:.2f}); sheets: '
                  f'sheets/{best}_s0_best_*.jpg']
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'summary.json').write_text(json.dumps({'arms': S, 'verdict': V}, indent=1, default=float))
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['run', 'summary'])
    ap.add_argument('--arms', nargs='+', default=['c448'] + list(ARMS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--smoke', action='store_true', help='300 train / 2,500 eval frames, 100 steps; set SP_OUT')
    args = ap.parse_args()
    for a in args.arms:
        assert a == 'c448' or a in ARMS, a
    if args.smoke:
        assert os.environ.get('SP_OUT'), 'smoke: set SP_OUT to a scratch results dir'
    {'run': cmd_run, 'summary': cmd_summary}[args.step](args)


if __name__ == '__main__':
    main()
