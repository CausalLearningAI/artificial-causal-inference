"""
QUICK_REPR: fast, pre-registered single-change tests of the mice ECI representation (can an UNSUPERVISED SAE neuron
capture nose-nose (nn) / nose-tail (nt) sniffing?). Each arm changes ONE thing vs C1.

C1 = DINOv2-base 448 px last-layer patch tokens + 1 s motion [x_t, x_t - x_{t-5}] (TokenNorm.fit_blocks([768, 768])),
the fg448 mask (store positions, 32 x 32 grid), Matryoshka BatchTopK SAE (4096 latents, k = 16, motion_t3.train_one =
levers recipe, 6000 steps x 4096), frame read-out = MAX over kept patches.

Frames (the V-JEPA / res_pool harness, reused exactly)
    train  the 10,000 unlabelled frames of results/vision/eci_res_pool/mice/train/frames.parquet (+ their t-5 frames,
           prow / ppath; pos448.i16 = the store mask)
    eval   the 20,000-frame annotated B subset (results/vision/eci_repr_diag/mice/b/subset.parquet, inverse-sampling
           weights w, nose_nose = Y_nn OR Y_np, nose_tail = Y_nt), mask = BDIR pos448.i16; t-5 row = row - 5 clipped to
           the video's first row (motion_t3.nbr_rows). Scored on all 20,000 and the 16,775 'usable' frames
           (vjepa plan), cross-fitted best neuron (vjepa.score = contact_sae.cross_fit, weighted), prefixes 512 / 4096.
Tokens are RE-ENCODED in the job from the 512 px dataset JPGs (HF dinov2-base processor, whole frame bicubic -> 448, no
crop; fp32 forward, TF32 off, tokens rounded to fp16 = the store's rule). Checked against the store (cosine).

Arms
    c448   control: the res_pool C1 SAEs RETRAINED on these 10,000 train frames (store tokens), evaluated here (no
           training). The deployed C1 (eci_t3_motion M5_s*) is read from results/vision/eci_vjepa/eval/C1_M5_s*.json.
    W01    T-W: shrinkage whitening before the SAE. Input x = [x_t, x_t - x_{t-5}] (1536-d) -> C1's TokenNorm.fit_blocks
           ([768, 768], fitted on the train tokens) -> PCA of the JOINT 1536-d vector over ALL train tokens -> x_w = V
           diag((lam + eps)^(-1/2)) V^T (x - mu) (ZCA, coordinates kept), eps = 0.1 x mean(lam). Then the identical
           train_one (whose own TokenNorm refits per-block scales on x_w).
    W10    T-W, milder: eps = 1.0 x mean(lam).
    reg    T-R: facebook/dinov2-with-registers-base last layer (patch tokens after CLS + 4 registers), same processor /
           448 input / 32 x 32 grid / mask / motion / SAE.
    L8     T-L: DINOv2-base block 8 output (of 12), with the final LayerNorm applied (DINOv2 get_intermediate_layers
           norm=True convention); same motion (the t-5 token from the same layer), SAE, mask.
    L10    T-L: block 10.
Supervised ceiling (reg, L8, L10): diag_mice_patches mean_lin + mil_ctx on the eval t tokens alone (pool-grouped 5-fold
    CV, 3 seeds), exactly the DINOv2 '448_subset' reference in results/vision/eci_repr_diag/mice/b/results.json.

PRE-REGISTERED RULE (fixed before any result; frames = usable, prefix 4096, C1 = deployed M5 reference)
    an arm 'helps' iff on nose_nose or nose_tail: best-neuron cf AP >= 1.5 x C1 OR honest top-1% >= 0.25, AND the
    OTHER label's cf AUROC is not more than 0.02 below C1's. Dream target: top-1% >= 0.40.

Steps
    run      GPU  one job: stage JPGs -> encode the token sets the arms need -> train SAE seeds -> eval -> probe ->
                  sheets (all in /localhome; only JSONs / sheets / SAE checkpoints are written to NFS)
    summary  any  OUT/table.md, OUT/summary.json (reads JSONs only)
Outputs: results/vision/eci_quick_repr/ (gitignored).

RESULTS (jobs 797079 / 797080 / 797081, RTX 3090, 2026-10-14; 3 SAE seeds; usable frames, prefix 4096; mean +- sd)
    store check: re-encoded eval t tokens vs BDIR tok448 cos min 0.9999997; train t vs store 0.9999998; t-5 vs d5 0.99996
    PCA spectrum (T-W): top 10 PCs 47% of variance, 329 PCs for 90%; eps = 0.122 (W01) / 1.22 (W10)
                       nose_nose AUROC / AP / top-1%                  nose_tail AUROC / AP / top-1%
    C1 deployed        0.679+-.011 / 0.0517+-.0024 / 0.114+-.007      0.714+-.029 / 0.0485+-.0095 / 0.102+-.017
    c448 (retrained)   0.692+-.023 / 0.0552+-.0105 / 0.122+-.012      0.741+-.053 / 0.0512+-.0111 / 0.099+-.010
    W01                0.705+-.016 / 0.0558+-.0087 / 0.099+-.034      0.708+-.045 / 0.0404+-.0136 / 0.103+-.006
    W10                0.669+-.027 / 0.0543+-.0095 / 0.131+-.021      0.693+-.025 / 0.0335+-.0009 / 0.083+-.005
    reg                0.681+-.035 / 0.0446+-.0058 / 0.093+-.024      0.730+-.051 / 0.0399+-.0132 / 0.093+-.024
    L8                 0.657+-.017 / 0.0396+-.0029 / 0.112+-.009      0.677+-.039 / 0.0200+-.0038 / 0.080+-.021
    L10                0.673+-.006 / 0.0557+-.0020 / 0.135+-.011      0.727+-.010 / 0.0498+-.0052 / 0.107+-.026
    Verdict: NO arm helps (best AP ratio vs C1 1.08x, best top-1% 0.135; dream 0.40 not approached). W10 seed 0 alone
    read nn AP 1.25x / top-1% 0.155: a lucky seed, gone at 3 seeds.
    Supervised ceiling, all 20,000 frames, AUROC / AP (mil_ctx 3 seeds x 5 folds; seed sd of AP <= 0.035):
                       nn mean_lin     nn mil_ctx      nt mean_lin     nt mil_ctx
    DINOv2 last (ref)  0.815 / 0.140   0.871 / 0.244   0.817 / 0.075   0.912 / 0.236
    reg                0.811 / 0.126   0.885 / 0.305   0.817 / 0.078   0.909 / 0.221
    L8                 0.744 / 0.073   0.828 / 0.172   0.780 / 0.052   0.859 / 0.149
    L10                0.808 / 0.117   0.860 / 0.215   0.821 / 0.069   0.915 / 0.221
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
OUT = Path(os.environ.get('QR_OUT') or REPO / 'results/vision/eci_quick_repr')
RPT = REPO / 'results/vision/eci_res_pool/mice/train'
RPS = REPO / 'results/vision/eci_res_pool/mice/sae'
VJ = REPO / 'results/vision/eci_vjepa'
BDIR = REPO / 'results/vision/eci_repr_diag/mice/b'
ANN = REPO / 'dataset/mice/v1/annotations.csv'
D, G = 768, 32
LABELS = ['nose_nose', 'nose_tail']
MODELS = {'base': 'facebook/dinov2-base', 'reg': 'facebook/dinov2-with-registers-base'}
# arm -> token set
TOKSET = {'c448': 'last', 'W01': 'last', 'W10': 'last', 'reg': 'reg', 'L8': 'L8', 'L10': 'L10'}
EPS = {'W01': 0.1, 'W10': 1.0}
REF_C1 = {'nose_nose': (0.679, 0.0517, 0.114), 'nose_tail': (0.714, 0.0485, 0.102)}  # task reference, usable m4096
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def stage_dir():
    d = Path(os.environ.get('LOCAL_DIR') or '')
    if not str(d).startswith('/localhome'):
        raise RuntimeError(f'LOCAL_DIR must be under /localhome, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------- frames
def tables():
    """-> {split: dict(F, pos, lens, path_t, path_p)} in the harness frame order."""
    from src.eci.foreground import obs_rows
    T = pd.read_parquet(RPT / 'frames.parquet')
    tpos = np.fromfile(RPT / 'pos448.i16', np.int16)
    assert len(tpos) == T.n_fg.sum()
    B = pd.read_parquet(VJ / 'plan' / 'eval_frames.parquet')
    sub = pd.read_parquet(BDIR / 'subset.parquet')
    assert (sub.row.values == B.row.values).all() and (sub.n_fg.values == B.n_fg.values).all()
    bpos = np.fromfile(BDIR / 'pos448.i16', np.int16)
    assert len(bpos) == B.n_fg.sum()
    fp_all = pd.read_csv(ANN, usecols=['frame_path'])['frame_path'].values
    rng = obs_rows(ANN)
    lo = np.array([rng[o][0] for o in B.obs])
    hi = np.array([rng[o][1] for o in B.obs])
    assert ((B.row.values >= lo) & (B.row.values < hi)).all()
    prow = np.maximum(B.row.values - 5, lo)
    assert (fp_all[B.row.values] == B.frame_path.values).all()
    assert (fp_all[T.row.values] == T.frame_path.values).all() and (fp_all[T.prow.values] == T.ppath.values).all()
    bp = fp_all[prow]
    assert all(Path(a).parent == Path(b).parent for a, b in zip(B.frame_path.values, bp))
    assert all(Path(a).parent == Path(b).parent for a, b in zip(T.frame_path.values, T.ppath.values))
    return {'train': dict(F=T, pos=tpos, lens=T.n_fg.values.astype(np.int64), path_t=T.frame_path.values,
                          path_p=T.ppath.values),
            'eval': dict(F=B, sub=sub, pos=bpos, lens=B.n_fg.values.astype(np.int64), path_t=B.frame_path.values,
                         path_p=bp)}


def take(t, ix):
    """frame subset of one split dict (smoke test)."""
    st = np.r_[0, np.cumsum(t['lens'])]
    o = {'F': t['F'].iloc[ix].reset_index(drop=True), 'pos': np.concatenate([t['pos'][st[i]:st[i + 1]] for i in ix]),
         'lens': t['lens'][ix], 'path_t': t['path_t'][ix], 'path_p': t['path_p'][ix]}
    if 'sub' in t:
        o['sub'] = t['sub'].iloc[ix].reset_index(drop=True)
    return o


def stage_jpgs(tb, stage, workers=64):
    """parallel NFS -> /localhome copy of every needed JPG (tar was ~18 files/s; small-file latency bound)."""
    from concurrent.futures import ThreadPoolExecutor
    paths = sorted(set(np.concatenate([tb[s][k] for s in tb for k in ('path_t', 'path_p')]).tolist()))
    dst = stage / 'jpg'
    for d in sorted({str(Path(p).parent) for p in paths}):
        (dst / d).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda p: shutil.copyfile(REPO / 'dataset' / p, dst / p), paths, chunksize=32))
    log(f'staged {len(paths):,} JPGs in {time.time() - t0:.0f}s ({len(paths) / (time.time() - t0):.0f} files/s)')
    return dst


# ---------------------------------------------------------------------------------------------- encode
def encode_sets(kind, sets, tb, jpg, dev, bs, workers):
    """-> {set: {split: X (N, 1536) fp16 numpy = [x_t, x_t - x_{t-5}]}} for the token sets of one model."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    proc = AutoImageProcessor.from_pretrained(MODELS['base'], use_fast=True, size={'height': 448, 'width': 448},
                                              do_center_crop=False)  # the same pixels for both models
    model = AutoModel.from_pretrained(MODELS[kind]).to(dev).eval()
    model.requires_grad_(False)
    npre = 1 + (getattr(model.config, 'num_register_tokens', 0) or 0)
    need_hs = any(s.startswith('L') for s in sets)
    layers = {s: int(s[1:]) for s in sets if s.startswith('L')}
    log(f'model {MODELS[kind]}: prefix tokens {npre}, sets {sets}, hidden states {need_hs}')

    class Imgs(torch.utils.data.Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            with Image.open(jpg / self.paths[i]) as im:
                im = im.convert('RGB')
            return i, proc(images=im, return_tensors='pt')['pixel_values'][0]

    out = {s: {} for s in sets}
    info = {'checked_final_ln': None}
    for split in ('train', 'eval'):
        t = tb[split]
        st = np.r_[0, np.cumsum(t['lens'])]
        N = int(st[-1])
        for s in sets:
            out[s][split] = np.empty((N, 2 * D), np.float16)
        for which in ('t', 'p'):
            paths = t['path_' + which]
            dl = torch.utils.data.DataLoader(Imgs(paths), batch_size=bs, num_workers=workers, shuffle=False,
                                             pin_memory=True, prefetch_factor=4)
            t0, gpu, n = time.time(), 0.0, 0
            for b, (ii, pix) in enumerate(dl):
                torch.cuda.synchronize(); g0 = time.time()
                with torch.inference_mode():
                    o = model(pixel_values=pix.to(dev, non_blocking=True), output_hidden_states=need_hs)
                    hs = {}
                    for s in sets:
                        if s in ('last', 'reg'):
                            h = o.last_hidden_state
                        else:
                            h = model.layernorm(o.hidden_states[layers[s]])
                        assert h.shape[1] == npre + G * G and torch.isfinite(h).all()
                        hs[s] = h[:, npre:].float().half()
                    if need_hs and info['checked_final_ln'] is None:
                        dd = (model.layernorm(o.hidden_states[-1]) - o.last_hidden_state).abs().max().item()
                        info['checked_final_ln'] = dd
                        log(f'  check: layernorm(hidden_states[-1]) vs last_hidden_state max abs {dd:.2e} '
                            f'(n hidden states {len(o.hidden_states)})')
                torch.cuda.synchronize(); gpu += time.time() - g0
                ii = ii.numpy()
                for s in sets:
                    H = hs[s].cpu().numpy()
                    X = out[s][split]
                    for k, f in enumerate(ii):
                        p = t['pos'][st[f]:st[f + 1]].astype(np.int64)
                        if which == 't':
                            X[st[f]:st[f + 1], :D] = H[k, p]
                        else:
                            X[st[f]:st[f + 1], D:] = (X[st[f]:st[f + 1], :D].astype(np.float32)
                                                      - H[k, p].astype(np.float32)).astype(np.float16)
                n += len(ii)
                if b % 200 == 0:
                    log(f'  {split} {which} batch {b}: {n:,}/{len(paths):,} frames, {n / (time.time() - t0):.0f} f/s')
            info[f'{split}_{which}'] = {'frames': len(paths), 'wall_s': round(time.time() - t0, 1), 'gpu_s': round(gpu, 1)}
            log(f'  {split} {which}: {len(paths):,} frames in {time.time() - t0:.0f}s (gpu {gpu:.0f}s)')
    del model
    torch.cuda.empty_cache()
    return out, info


def check_vs_store(X, tb, smoke=False):
    """re-encoded last-layer tokens vs the store: eval t vs BDIR tok448 (first 300k tokens); train t / t-5 vs store
    + d5 store on the first 64 train frames."""
    import motion_t3 as mt
    import spatial_sae_pilot as ssp
    cos = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)  # noqa: E731
    r = {}
    if not smoke:
        ref = np.load(BDIR / 'tok448.npy', mmap_mode='r')
        n = 300_000
        c = cos(np.asarray(X['eval'][:n, :D], np.float32), np.asarray(ref[:n], np.float32))
        r['eval_t_vs_bdir_tok448'] = {'n': n, 'cos_min': float(c.min()), 'cos_mean': float(c.mean())}
    idx = ssp.StoreIndex('mice')
    F = tb['train']['F']
    fr = F.store_frame.values[:64]
    assert (np.diff(fr) > 0).all()
    tok, pos, lens = idx.load(fr)
    m = int(lens.sum())
    assert (pos == tb['train']['pos'][:m]).all()
    prev, _, ok = mt.D5Matcher(idx, 'mice').fetch(fr)
    a = np.asarray(X['train'][:m, :D], np.float32)
    c = cos(a, tok.astype(np.float32))
    r['train_t_vs_store'] = {'n': m, 'cos_min': float(c.min()), 'cos_mean': float(c.mean())}
    pv = a - np.asarray(X['train'][:m, D:], np.float32)
    c = cos(pv[ok], prev[ok].astype(np.float32))
    r['train_prev_vs_d5'] = {'n': int(ok.sum()), 'cos_min': float(c.min()), 'cos_mean': float(c.mean())}
    log(f'store check: {json.dumps(r)}')
    return r


# ---------------------------------------------------------------------------------------------- whitening
class Whiten:
    """x (raw [x_t, dx] fp16) -> C1 TokenNorm (blocks) -> ZCA shrinkage whitening; callable on GPU tensors."""

    def __init__(self, pre, mu, W):
        self.pre, self.mu, self.W = pre, mu, W

    def __call__(self, x):
        return (self.pre(x) - self.mu) @ self.W


def fit_whiten(Xg, frac, dev, out):
    import torch
    from src.eci.sae import TokenNorm
    N = len(Xg)
    g = np.random.default_rng(0)
    ii = torch.from_numpy(np.sort(g.choice(N, min(500_000, N), replace=False))).to(dev)
    pre = TokenNorm(2 * D).to(dev).fit_blocks(Xg[ii].float(), [D, D], seed=0)
    s1 = torch.zeros(2 * D, dtype=torch.float64, device=dev)
    C = torch.zeros(2 * D, 2 * D, dtype=torch.float64, device=dev)
    for a in range(0, N, 200_000):
        x = pre(Xg[a:a + 200_000]).double()
        s1 += x.sum(0)
        C += x.T @ x
    mu = s1 / N
    C = C / N - torch.outer(mu, mu)
    lam, V = torch.linalg.eigh(C)
    lam = lam.clamp_min(0)
    eps = frac * lam.mean()
    W = (V * (lam + eps).rsqrt()) @ V.T
    ls = lam.flip(0).cpu().numpy()
    cum = np.cumsum(ls) / ls.sum()
    info = {'eps_frac_of_mean_lambda': frac, 'eps': float(eps), 'mean_lambda': float(lam.mean()),
            'lambda_max': float(ls[0]), 'lambda_min': float(ls[-1]), 'var_top10': float(cum[9]),
            'var_top100': float(cum[99]), 'n_pc_90': int(np.searchsorted(cum, 0.9) + 1),
            'max_gain_over_min_gain': float(((ls[-1] + float(eps)) ** -0.5) / ((ls[0] + float(eps)) ** -0.5)),
            'pca_on': 'joint 1536-d [x_t, x_t - x_{t-5}] after C1 TokenNorm.fit_blocks([768, 768]), all train tokens',
            'n_tokens': N}
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'whiten.npz', pre_mean=pre.mean.cpu().numpy(), pre_scale=pre.scale.cpu().numpy(),
             mu=mu.float().cpu().numpy(), W=W.float().cpu().numpy(), lam=ls)
    (out / 'whiten.json').write_text(json.dumps(info, indent=1))
    log(f'whitening: {json.dumps(info)}')
    return Whiten(pre, mu.float(), W.float())


def load_whiten(path, dev):
    import torch
    from src.eci.sae import TokenNorm
    z = np.load(path)
    pre = TokenNorm(2 * D).to(dev)
    pre.mean.copy_(torch.from_numpy(z['pre_mean']))
    pre.scale = torch.from_numpy(z['pre_scale']).to(dev)
    return Whiten(pre, torch.from_numpy(z['mu']).to(dev), torch.from_numpy(z['W']).to(dev))


# ---------------------------------------------------------------------------------------------- run
def sae_dir(arm, seed):
    return (RPS / f'c448_s{seed}') if arm == 'c448' else (OUT / 'sae' / f'{arm}_s{seed}')


def train_arm(arm, seed, Xtr, dev, steps):
    """Xtr: GPU fp16 (N, 1536) raw. Whitened arms: the transform is fitted once per arm (seed independent)."""
    import torch
    import motion_t3 as mt
    o = sae_dir(arm, seed)
    if (o / 'sae.pt').exists():
        log(f'[SKIP] {o}')
        return
    X = Xtr
    if arm in EPS:
        wp = OUT / 'sae' / f'{arm}_whiten'
        wt = load_whiten(wp / 'whiten.npz', dev) if (wp / 'whiten.npz').exists() else fit_whiten(Xtr, EPS[arm], dev, wp)
        X = torch.empty_like(Xtr)
        for a in range(0, len(X), 200_000):
            X[a:a + 200_000] = wt(Xtr[a:a + 200_000]).half()
    mt.train_one(arm, seed, X, steps, o, {'config': 'quick_repr', 'arm': arm, 'tokset': TOKSET[arm],
                                          'whiten_eps_frac': EPS.get(arm), 'train_frames': 'res_pool mice train'}, dev)
    del X
    torch.cuda.empty_cache()


def eval_arm(arm, seed, Xev, tb, dev, half_all):
    import torch
    import vjepa as vj
    from src.eci.sae import load_sae
    od = OUT / 'eval'
    od.mkdir(parents=True, exist_ok=True)
    key = f'{arm}_s{seed}'
    if (od / f'{key}.json').exists():
        log(f'[SKIP] eval {key}')
        return json.loads((od / f'{key}.json').read_text())
    sae, norm, _ = load_sae(sae_dir(arm, seed) / 'sae.pt', dev)
    tf = norm
    if arm in EPS:
        wt = load_whiten(OUT / 'sae' / f'{arm}_whiten' / 'whiten.npz', dev)
        tf = lambda x: norm(wt(x))  # noqa: E731
    F, sub = tb['eval']['F'], tb['eval']['sub']
    codes, st, _ = vj.frame_max_tokens(sae, tf, Xev, tb['eval']['lens'], dev)
    rows = {'all': np.arange(len(F)), 'usable': np.flatnonzero(F.usable.values)}
    res = {'stats': st}
    for fs, rr in rows.items():
        for p in (512, 4096):
            r = vj.score(codes[:, :p].contiguous(), sub, rr, half_all, dev)
            res[f'{fs}_m{p}'] = r
            log(f'  {key} {fs} m{p}: ' + ', '.join(
                f"{b} AUROC {r[b]['cf_auroc']:.3f} AP {r[b]['cf_ap']:.4f} top1 {r[b]['cf_top1']:.3f} "
                f"size-ctrl {r[b]['cf_size_ctrl_auroc']:.3f}" for b in LABELS))
    w = torch.from_numpy(sub.w.values).to(dev).float()
    res['fire_rate_w'] = (((codes > 0).float() * w[:, None]).sum(0) / w.sum()).cpu().numpy().tolist()
    (od / f'{key}.json').write_text(json.dumps(res, indent=1, default=float))
    del codes
    torch.cuda.empty_cache()
    return res


def probe(tokset, Xev, tb, dev, seeds, steps):
    import torch
    import diag_mice_patches as dmp
    o = OUT / f'probe_{tokset}.json'
    if o.exists():
        log(f'[SKIP] {o}')
        return
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    sub = tb['eval']['sub']
    tok = np.ascontiguousarray(Xev[:, :D])
    bag = dmp.Bag(tok, tb['eval']['pos'], tb['eval']['lens'], G, dev)
    res = dmp.cross_validate(bag, sub, sub.w.values, dev, ['mean_lin', 'mil_ctx'], seeds, None, tokset, steps)
    summ = dmp.summarise(res)
    o.write_text(json.dumps({'summary': summ, 'runs': res, 'wall_s': time.time() - t0,
                             'gpu': torch.cuda.get_device_name(0)}, indent=1, default=float))
    for s in summ:
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
    del bag, tok
    torch.cuda.empty_cache()


def sheets(arm, seed, Xev, tb, jpg, dev, n_tiles=16):
    """best nn / nt neurons (auc- and top1-selected on half 0, shown on half 1): full frame t (red = argmax patch,
    yellow = crop) | crop t | crop t-5. Border green = nose_nose, blue = nose_tail, grey = neither."""
    from PIL import Image, ImageDraw
    import contact_sae as cs
    import vjepa as vj
    from src.eci.sae import load_sae
    key = f'{arm}_s{seed}'
    R = json.loads((OUT / 'eval' / f'{key}.json').read_text())
    F, sub, pos, lens = tb['eval']['F'], tb['eval']['sub'], tb['eval']['pos'], tb['eval']['lens']
    half = cs.video_units(sub.obs.values)
    picks = []
    for b in LABELS:
        for sel in ('auc', 'top1'):
            e = R['usable_m4096'][b]['dirs'][0]
            picks.append((f'best_{b}_{sel}', e[sel]['neuron'], (half == 1) & F.usable.values,
                          f'{sel}-selected on half 0 for {b} (usable frames, prefix 4096); half 1: AUROC '
                          f'{e[sel]["test_auc"]:.3f} AP {e[sel]["test_ap"]:.3f} top1 {e[sel]["test_top1"]:.3f}'))
    sae, norm, _ = load_sae(sae_dir(arm, seed) / 'sae.pt', dev)
    tf = norm
    if arm in EPS:
        wt = load_whiten(OUT / 'sae' / f'{arm}_whiten' / 'whiten.npz', dev)
        tf = lambda x: norm(wt(x))  # noqa: E731
    J = sorted({p[1] for p in picks})
    codes, _, arg = vj.frame_max_tokens(sae, tf, Xev, lens, dev, want_arg=J)
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
        nr = (len(chosen) + nc - 1) // nc
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


def cmd_run(args):
    import torch
    import contact_sae as cs
    dev = torch.device('cuda')
    t00 = time.time()
    stage = stage_dir()
    tb = tables()
    if args.smoke:
        tb = {'train': take(tb['train'], np.arange(300)), 'eval': take(tb['eval'], np.arange(0, 20000, 8))}
        args.steps, args.probe_seeds = 100, [0]
    half_all = cs.video_units(tb['eval']['sub'].obs.values)
    jpg = stage_jpgs(tb, stage)
    sets = sorted({TOKSET[a] for a in args.arms})
    kinds = {'reg': ['reg']} if sets == ['reg'] else {'base': [s for s in sets if s != 'reg']}
    if 'reg' in sets and len(sets) > 1:
        kinds['reg'] = ['reg']
    X, einfo = {}, {}
    for kind, ss in kinds.items():
        o, inf = encode_sets(kind, ss, tb, jpg, dev, args.bs, args.workers)
        X.update(o)
        einfo[kind] = inf
    if 'last' in X:
        einfo['store_check'] = check_vs_store(X['last'], tb, args.smoke)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'encode_{"_".join(args.arms)}.json').write_text(json.dumps(
        {'gpu': torch.cuda.get_device_name(0), **einfo, 'n_train_tokens': int(tb['train']['lens'].sum()),
         'n_eval_tokens': int(tb['eval']['lens'].sum())}, indent=1, default=float))
    torch.backends.cuda.matmul.allow_tf32 = True  # as motion_t3 train
    for seeds in ([args.seeds[0]], args.seeds[1:]):  # seed 0 of every arm first (interim read-out), then the rest
        if not seeds:
            continue
        for arm in args.arms:
            for seed in seeds:
                if arm != 'c448':
                    Xtr = torch.from_numpy(X[TOKSET[arm]]['train']).to(dev)
                    train_arm(arm, seed, Xtr, dev, args.steps)
                    del Xtr
                    torch.cuda.empty_cache()
                torch.backends.cuda.matmul.allow_tf32 = False
                eval_arm(arm, seed, X[TOKSET[arm]]['eval'], tb, dev, half_all)
                torch.backends.cuda.matmul.allow_tf32 = True
        if seeds == [args.seeds[0]]:
            for arm in args.arms:
                sheets(arm, args.seeds[0], X[TOKSET[arm]]['eval'], tb, jpg, dev)
    for s in args.probe_sets:
        probe(s, X[s]['eval'], tb, dev, args.probe_seeds, 50 if args.smoke else 3000)
    log(f'run done in {time.time() - t00:.0f}s')


# ---------------------------------------------------------------------------------------------- summary
def agg(files, key, b):
    rr = [json.loads(f.read_text())[key][b] for f in files]
    return {q: [x[q] for x in rr] for q in ('cf_auroc', 'cf_ap', 'cf_top1', 'cf_size_ctrl_auroc')}


def cmd_summary(args):
    E = OUT / 'eval'
    cfgs = [('C1dep', sorted((VJ / 'eval').glob('C1_M5_s*.json')))] + [
        (a, sorted(E.glob(f'{a}_s*.json'))) for a in ('c448', 'W01', 'W10', 'reg', 'L8', 'L10')]
    cfgs = [(a, f) for a, f in cfgs if f]
    ms = lambda a, n=3: f'{np.mean(a):.{n}f}' + (f' +- {np.std(a):.{n}f}' if len(a) > 1 else '')  # noqa: E731
    lines = ['# quick_repr: single-change tests vs C1 (mice B subset, weighted, cross-fitted best neuron)', '']
    summ = {}
    for key in ('usable_m4096', 'usable_m512', 'all_m4096'):
        lines += ['', f'## {key}', '', '| label | config | seeds | cf AUROC | cf AP | honest top-1% | size-ctrl AUROC | '
                  'AP / C1dep | AP / c448 |', '|---|---|---|---|---|---|---|---|---|']
        for b in LABELS:
            base = {}
            for a, fs in cfgs:
                v = agg(fs, key, b)
                base.setdefault(a, np.mean(v['cf_ap']))
                r1 = np.mean(v['cf_ap']) / base['C1dep'] if 'C1dep' in base else float('nan')
                r2 = np.mean(v['cf_ap']) / base['c448'] if 'c448' in base else float('nan')
                lines.append(f"| {b} | {a} | {len(fs)} | {ms(v['cf_auroc'])} | {ms(v['cf_ap'], 4)} | {ms(v['cf_top1'])} | "
                             f"{ms(v['cf_size_ctrl_auroc'])} | {r1:.2f} | {r2:.2f} |")
                summ.setdefault(key, {}).setdefault(b, {})[a] = {**v, 'n_seeds': len(fs)}
    # verdict (usable_m4096 vs the C1 reference of the task; also vs the retrained c448 control)
    S = summ.get('usable_m4096', {})
    verdict = {}
    for a, _ in cfgs:
        if a in ('C1dep', 'c448'):
            continue
        for refname in ('C1dep', 'c448'):
            if refname not in S.get('nose_nose', {}):
                continue
            v = {}
            for b, ob in (('nose_nose', 'nose_tail'), ('nose_tail', 'nose_nose')):
                x, r = S[b][a], S[b][refname]
                apr = np.mean(x['cf_ap']) / np.mean(r['cf_ap'])
                drop_other = np.mean(S[ob][refname]['cf_auroc']) - np.mean(S[ob][a]['cf_auroc'])
                v[b] = {'ap_over_ref': float(apr), 'top1': float(np.mean(x['cf_top1'])),
                        'pass': bool(apr >= 1.5 or np.mean(x['cf_top1']) >= 0.25),
                        'other_label_auroc_drop': float(drop_other), 'dream_top1_ge_0.40': bool(np.mean(x['cf_top1']) >= 0.40)}
            v['helps'] = any(v[b]['pass'] and v[b]['other_label_auroc_drop'] <= 0.02 for b in LABELS)
            verdict.setdefault(a, {})[f'vs_{refname}'] = v
    summ['verdict'] = verdict
    lines += ['', 'Pre-registered (usable, m4096): helps iff on nn or nt, AP >= 1.5x C1 OR top-1% >= 0.25, AND the other '
              "label's AUROC drops <= 0.02. Dream: top-1% >= 0.40.", '']
    for a, d in verdict.items():
        for ref, v in d.items():
            lines.append(f"- {a} {ref}: helps={v['helps']}; " + '; '.join(
                f"{b}: AP x{v[b]['ap_over_ref']:.2f}, top1 {v[b]['top1']:.3f}, other-AUROC drop "
                f"{v[b]['other_label_auroc_drop']:+.3f}" for b in LABELS))
    ref = json.loads((BDIR / 'results.json').read_text())['summary']
    prs = {t: json.loads((OUT / f'probe_{t}.json').read_text())['summary'] for t in ('reg', 'L8', 'L10')
           if (OUT / f'probe_{t}.json').exists()}
    if prs:
        lines += ['', '## supervised ceiling (pool-grouped 5-fold CV, weighted, all 20,000 B frames): AUROC / AP', '',
                  '| label | probe | DINOv2 last (ref) | ' + ' | '.join(prs) + ' |', '|---|---|---|' + '---|' * len(prs)]
        f = lambda s: f"{s[0]['auroc_mean']:.3f} / {s[0]['ap_mean']:.3f}" if s else '-'  # noqa: E731
        for b in LABELS:
            for p_ in ('mean_lin', 'mil_ctx'):
                a = [x for x in ref if x['res'] == '448_subset' and x['probe'] == p_ and x['behaviour'] == b]
                cs_ = {t: [x for x in pr if x['probe'] == p_ and x['behaviour'] == b] for t, pr in prs.items()}
                lines.append(f'| {b} | {p_} | {f(a)} | ' + ' | '.join(f(c) for c in cs_.values()) + ' |')
                summ.setdefault('ceiling', {}).setdefault(b, {})[p_] = {'dinov2_last': a[0] if a else None,
                                                                        **{t: (c[0] if c else None) for t, c in cs_.items()}}
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['run', 'summary'])
    ap.add_argument('--arms', nargs='+', default=['c448', 'W01', 'W10'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--probe-sets', nargs='*', default=[])
    ap.add_argument('--probe-seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--smoke', action='store_true', help='300 train / 2,500 eval frames, 100 steps; set QR_OUT')
    args = ap.parse_args()
    if args.smoke:
        assert os.environ.get('QR_OUT'), 'smoke: set QR_OUT to a scratch results dir'
    {'run': cmd_run, 'summary': cmd_summary}[args.step](args)


if __name__ == '__main__':
    main()
