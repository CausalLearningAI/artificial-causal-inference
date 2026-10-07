"""
Is mouse nose-nose / nose-tail contact present in DINOv2 PATCH tokens at all? (ECI representation diagnostic,
mice v1.) Supervised probes are an upper bound on what an unsupervised SAE on the same tokens could expose.

Labels (same definitions as the alignment audit, copied from scripts/eci/spatial_sae_pilot.py labelled_frames):
    nose_nose = Y_nn (mutual) OR Y_np (directional), nose_tail = Y_nt; frames with all three labels only.
Split: 5-fold cross-validation over the 24 annotated POOLS (data/mice/v1/experiment.csv 'pool'; seeded
permutation, fold sizes 5,5,5,5,4). Inside each training fold 4 pools are held out as an inner validation set that
picks the linear probe's L2 strength and the MIL checkpoint (by validation AP); the test fold is touched once.
Training oversamples positives (MIL: 25% of every batch; linear: class-balanced loss weights); evaluation is on every
test frame at the natural base rate (Diagnostic B: inverse-sampling weights restore it).

Probes (per behaviour):
    mean_lin   logistic regression on the foreground-MEAN token (768-d, standardised)
    max_lin    logistic regression on the per-dimension MAX over foreground tokens
    mil        gated-attention multiple-instance pooling (Ilse et al. 2018) over the frame's foreground tokens:
               h_i = GELU(W LN(x_i) + row_emb(r_i) + col_emb(c_i)), a_i = softmax_i(w^T tanh(V h_i) * sigm(U h_i)),
               score = MLP(sum_i a_i h_i); learned 2D position embedding of the grid coordinate
    mil_ctx    the same with each token concatenated with the mean of the LN'd FOREGROUND tokens of its 3x3
               neighbourhood (self included; background patches are not stored). Implemented as
               W LN(x_i) + W' mean_3x3(LN(x)) = a linear layer on the concatenation.
Metrics per test fold: AUROC, AP, size-controlled AUROC (AUROC inside each decile of the foreground-patch count,
deciles fixed over all eval frames, averaged with weights = positives per decile), base rate; also the AUROC of the
foreground-patch count alone. Linear probes are convex and deterministic (seed-free); MIL is run with 3 seeds.

Steps:
    a        Diagnostic A, 448 px, the fg448 token store (every annotated store frame, 1 fps)
             -> OUT/a/{results.json, oof_scores.parquet, sheet_<beh>_pos.jpg, sheet_<beh>_neg.jpg}
    b_extract  Diagnostic B: a ~20k-frame subset (all positives of either behaviour, capped, plus random negatives;
             sampling weights recorded), JPGs staged to $STAGE, DINOv2-base re-encoded at 448 (32x32) and 896 (64x64);
             the foreground set is the store's 448 mask, each 448 patch -> its 2x2 children at 896 (same frame area).
             -> OUT/b/{frames.npz, tok448.f16, pos448.i16, tok896.f16, pos896.i16, extract.json}
    b_probe  the probes on both resolutions of the subset, same pool folds, weighted metrics -> OUT/b/results.json

Frames: dataset/mice/v1/frames/full/*.jpg are 512 x 512 (the standardised 5 fps video); 896 px is therefore an
UPSAMPLED 512 frame: a finer patch grid (8 frame px per patch instead of 16), no new pixel information. The raw
sources (data/mice/source, ~2060 x 2062 HEVC, 30 fps, keyframe every 250 frames) are not decoded here.

Usage (inside Slurm, scripts/eci/diag_mice_patches.sh):
    python scripts/eci/diag_mice_patches.py a --out results/vision/eci_repr_diag/mice
    python scripts/eci/diag_mice_patches.py b_extract --out ... --stage /localhome/$USER/$SLURM_JOB_ID
    python scripts/eci/diag_mice_patches.py b_probe --out ... --stage /localhome/$USER/$SLURM_JOB_ID
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.foreground import FgTokenStore  # noqa: E402

STORE = REPO / 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1'
ANN = REPO / 'dataset/mice/v1/annotations.csv'
EXP = REPO / 'data/mice/v1/experiment.csv'
BEH = ['nose_nose', 'nose_tail']
log = lambda s: print(s, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------------------------- data
def labelled_frames():
    """annotations.csv rows with all three labels -> DataFrame(row, obs, nose_nose, nose_tail)."""
    a = pd.read_csv(ANN, usecols=['observation_id', 'Y_nn', 'Y_np', 'Y_nt'])
    ok = a[['Y_nn', 'Y_np', 'Y_nt']].notna().all(1).values
    sub = a[ok]
    nn_, np_, nt = (sub[c].values > 0 for c in ('Y_nn', 'Y_np', 'Y_nt'))
    return pd.DataFrame({'row': np.flatnonzero(ok), 'obs': sub['observation_id'].values,
                         'nose_nose': nn_ | np_, 'nose_tail': nt})


class StoreIndex:
    """Per-frame layout of the token store (copy of scripts/eci/spatial_sae_pilot.py StoreIndex, mice only)."""

    def __init__(self):
        self.store = FgTokenStore(STORE)
        sh, start, nfg, rows = [], [], [], []
        for s, d in enumerate(self.store.dirs):
            z = np.load(d / 'frames.npz')
            n = z['n_fg'].astype(np.int64)
            if n.sum() != self.store.sizes[s]:
                raise RuntimeError(f'{d}: n_fg sums to {n.sum()}, shard has {self.store.sizes[s]} tokens')
            sh.append(np.full(len(n), s)); start.append(np.r_[0, np.cumsum(n)[:-1]]); nfg.append(n); rows.append(z['rows'])
        self.shard, self.start, self.nfg = np.concatenate(sh), np.concatenate(start), np.concatenate(nfg)
        self.rows = np.concatenate(rows).astype(np.int64)

    def load(self, frames):
        """Tokens of the given frames (must be sorted) -> (tokens (N, d) fp16, pos (N,) int16, lens (F,))."""
        assert (np.diff(frames) > 0).all()
        lens = self.nfg[frames]
        out = np.empty((int(lens.sum()), self.store.dim), dtype=np.float16)
        pos = np.empty(int(lens.sum()), dtype=np.int16)
        o = 0
        for s in np.unique(self.shard[frames]):
            fs = frames[self.shard[frames] == s]
            tok, ps = self.store.tokens(s), self.store.pos(s)
            st, n = self.start[fs], self.nfg[fs]
            brk = np.flatnonzero(st[1:] != st[:-1] + n[:-1]) + 1
            for a, b in zip(np.r_[0, brk], np.r_[brk, len(fs)]):
                lo, hi = st[a], st[b - 1] + n[b - 1]
                out[o:o + hi - lo] = tok[lo:hi]
                pos[o:o + hi - lo] = ps[lo:hi]
                o += hi - lo
        assert o == len(out)
        return out, pos, lens


def eval_frames():
    """Store frames of annotated videos with all three labels -> (StoreIndex, sorted frame idx, labels DataFrame)."""
    idx = StoreIndex()
    lab = labelled_frames().set_index('row')
    ev = np.flatnonzero(np.isin(idx.rows, lab.index.values))
    lab = lab.loc[idx.rows[ev]].reset_index()
    pool_of = pd.read_csv(EXP).set_index('observation_id')['pool']
    lab['pool'] = pool_of.loc[lab['obs'].values].values
    lab['n_fg'] = idx.nfg[ev]
    return idx, ev, lab


def pool_folds(pools, k=5, seed=0):
    u = np.array(sorted(set(pools)))
    perm = np.random.default_rng(seed).permutation(u)
    return [sorted(f.tolist()) for f in np.array_split(perm, k)]


def inner_val(train_pools, fold, n_val=4):
    return set(np.random.default_rng(100 + fold).choice(sorted(train_pools), n_val, replace=False).tolist())


class Bag:
    """Frames as bags of tokens. tok (N, d) fp16 torch (CPU or GPU), pos (N,), lens (F,)."""

    def __init__(self, tok, pos, lens, grid, dev):
        self.grid, self.dev = grid, dev
        need = tok.nbytes + (6 << 30)
        free = torch.cuda.mem_get_info()[0] if dev.type == 'cuda' else 0
        self.tok = torch.from_numpy(tok)
        if free > need:
            self.tok = self.tok.to(dev)
        self.on_gpu = self.tok.device.type == 'cuda'
        self.pos = torch.from_numpy(pos.astype(np.int64)).to(self.tok.device)
        self.lens = torch.from_numpy(lens.astype(np.int64))
        self.start = torch.cumsum(self.lens, 0) - self.lens
        log(f'  bag: {len(lens):,} frames, {len(tok):,} tokens, grid {grid}, tokens on {self.tok.device}')

    def gather(self, fidx):
        fidx = torch.as_tensor(np.asarray(fidx), dtype=torch.long)
        ln, st = self.lens[fidx], self.start[fidx]
        rep = torch.repeat_interleave(torch.arange(len(fidx)), ln)
        off = torch.arange(int(ln.sum())) - torch.repeat_interleave(torch.cumsum(ln, 0) - ln, ln)
        ti = (st[rep] + off).to(self.tok.device)
        x = self.tok[ti].to(self.dev, non_blocking=True)
        return x, self.pos[ti].to(self.dev), rep.to(self.dev), len(fidx)

    @torch.no_grad()
    def pooled(self, chunk=2048):
        """-> (mean (F, d), max (F, d)) float32 numpy over each frame's tokens (zeros for empty frames)."""
        Fn, d = len(self.lens), self.tok.shape[1]
        mean, mx = np.zeros((Fn, d), np.float32), np.zeros((Fn, d), np.float32)
        for c in range(0, Fn, chunk):
            fr = np.arange(c, min(c + chunk, Fn))
            x, _, f, B = self.gather(fr)
            x = x.float()
            s = torch.zeros(B, d, device=self.dev).index_add_(0, f, x)
            n = torch.bincount(f, minlength=B).clamp_min(1)[:, None].float()
            m = torch.full((B, d), -torch.inf, device=self.dev).index_reduce_(0, f, x, 'amax', include_self=True)
            m[torch.isinf(m)] = 0
            mean[fr], mx[fr] = (s / n).cpu().numpy(), m.cpu().numpy()
        return mean, mx


# ---------------------------------------------------------------------------------------------- metrics
def size_deciles(nfg):
    edges = np.unique(np.quantile(nfg, np.linspace(0, 1, 11)))
    return np.clip(np.searchsorted(edges, nfg, side='right') - 1, 0, len(edges) - 2)


def metrics(y, s, w, dec):
    y = y.astype(bool)
    out = {'auroc': float(roc_auc_score(y, s, sample_weight=w)),
           'ap': float(average_precision_score(y, s, sample_weight=w)),
           'base_rate': float(np.average(y, weights=w)), 'n_frames': int(len(y)), 'n_pos': int(y.sum())}
    num = den = 0.0
    for q in np.unique(dec):
        m = dec == q
        if 0 < y[m].sum() < m.sum():
            npos = float(y[m].sum())
            num += npos * roc_auc_score(y[m], s[m], sample_weight=w[m])
            den += npos
    out['auroc_size_ctrl'] = float(num / den) if den else float('nan')
    return out


def wap(y, s, w):
    return float(average_precision_score(y.astype(bool), s, sample_weight=w))


# ---------------------------------------------------------------------------------------------- linear probe
def fit_logreg(X, y, lam, dev, iters=300):
    """X (n, d) standardised torch on dev, y bool np -> (w, b); class-balanced logistic loss + lam ||w||^2."""
    yt = torch.from_numpy(y.astype(np.float32)).to(dev)
    p = yt.mean()
    wt = torch.where(yt > 0, 0.5 / p, 0.5 / (1 - p))
    w = torch.zeros(X.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn='strong_wolfe', tolerance_grad=1e-6)

    def closure():
        opt.zero_grad()
        loss = (wt * F.binary_cross_entropy_with_logits(X @ w + b, yt, reduction='none')).mean() + lam * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def run_linear(feat, y, w_eval, tr, va, te, dev, lams=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1)):
    mu, sd = feat[tr].mean(0), feat[tr].std(0) + 1e-6
    Z = lambda i: torch.from_numpy((feat[i] - mu) / sd).to(dev)  # noqa: E731
    Xtr, Xva, Xte = Z(tr), Z(va), Z(te)
    best = None
    for lam in lams:
        wv, bv = fit_logreg(Xtr, y[tr], lam, dev)
        ap = wap(y[va], (Xva @ wv + bv).cpu().numpy(), w_eval[va])
        if best is None or ap > best[0]:
            best = (ap, lam, wv, bv)
    ap, lam, wv, bv = best
    return (Xte @ wv + bv).cpu().numpy(), {'lam': lam, 'val_ap': ap}


# ---------------------------------------------------------------------------------------------- MIL
def seg_softmax(a, frame, B):
    mx = torch.full((B,), -torch.inf, device=a.device).index_reduce_(0, frame, a.detach(), 'amax', include_self=True)
    e = torch.exp(a - mx[frame])
    s = torch.zeros(B, device=a.device).index_add_(0, frame, e)
    return e / s[frame]


def nbr_mean(c, pos, frame, B, G):
    """Mean of per-token features c (n, h) over each token's 3x3 neighbourhood of present tokens (self included)."""
    h = c.shape[1]
    grid = c.new_zeros(B, G * G, h)
    grid[frame, pos] = c
    cnt = c.new_zeros(B, G * G)
    cnt[frame, pos] = 1
    gs = F.avg_pool2d(grid.view(B, G, G, h).permute(0, 3, 1, 2), 3, 1, 1, count_include_pad=True)
    cs = F.avg_pool2d(cnt.view(B, 1, G, G), 3, 1, 1, count_include_pad=True)
    m = (gs / cs.clamp_min(1e-6)).permute(0, 2, 3, 1).reshape(B, G * G, h)
    return m[frame, pos]


class MIL(nn.Module):
    def __init__(self, d_in=768, h=256, att=128, grid=32, ctx=False, drop=0.1):
        super().__init__()
        self.G = grid
        self.ln = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, h)
        self.ctx = nn.Linear(d_in, h, bias=False) if ctx else None
        self.row, self.col = nn.Embedding(grid, h), nn.Embedding(grid, h)
        self.V, self.U, self.w = nn.Linear(h, att), nn.Linear(h, att), nn.Linear(att, 1)
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
        nn.init.normal_(self.row.weight, std=0.02); nn.init.normal_(self.col.weight, std=0.02)

    def forward(self, x, pos, frame, B):
        xn = self.ln(x.float())
        hd = self.proj(xn) + self.row(pos // self.G) + self.col(pos % self.G)
        if self.ctx is not None:
            hd = hd + nbr_mean(self.ctx(xn), pos, frame, B, self.G)
        hd = self.drop(F.gelu(hd))
        a = self.w(torch.tanh(self.V(hd)) * torch.sigmoid(self.U(hd))).squeeze(1)
        att = seg_softmax(a, frame, B)
        z = torch.zeros(B, hd.shape[1], device=hd.device).index_add_(0, frame, att[:, None] * hd)
        return self.head(z).squeeze(1), att


@torch.no_grad()
def mil_predict(model, bag, fidx, chunk=512, want_att=False):
    model.eval()
    s, top = np.zeros(len(fidx), np.float32), []
    for c in range(0, len(fidx), chunk):
        fr = fidx[c:c + chunk]
        x, p, f, B = bag.gather(fr)
        logit, att = model(x, p, f, B)
        s[c:c + len(fr)] = logit.float().cpu().numpy()
        if want_att:  # top-3 attended positions and attention mass of the top patch, per frame
            f_np, a_np, p_np = f.cpu().numpy(), att.float().cpu().numpy(), p.cpu().numpy()
            bounds = np.r_[0, np.cumsum(np.bincount(f_np, minlength=B))]
            for j in range(B):
                aa, pp = a_np[bounds[j]:bounds[j + 1]], p_np[bounds[j]:bounds[j + 1]]
                o = np.argsort(-aa)[:3]
                n50 = int(np.searchsorted(np.cumsum(np.sort(aa)[::-1]), 0.5) + 1) if len(aa) else 0
                top.append((pp[o].tolist() + [-1] * (3 - len(o)), float(aa[o[0]]) if len(o) else 0.0, n50))
    model.train()
    return s, top


def run_mil(bag, y, w_eval, tr, va, te, ctx, seed, dev, steps=3000, bs=128, pos_frac=0.25, lr=1e-3, wd=0.05,
            eval_every=300):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ptr, ntr = tr[y[tr]], tr[~y[tr]]
    model = MIL(d_in=bag.tok.shape[1], grid=bag.grid, ctx=ctx).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n_pos = int(round(pos_frac * bs))
    yb = torch.cat([torch.ones(n_pos), torch.zeros(bs - n_pos)]).to(dev)
    best, hist, t0 = (-1.0, None, -1), [], time.time()
    for step in range(steps):
        g = lr * min(1.0, (step + 1) / 100) * 0.5 * (1 + np.cos(np.pi * step / steps))
        for pg in opt.param_groups:
            pg['lr'] = g
        fb = np.r_[rng.choice(ptr, n_pos), rng.choice(ntr, bs - n_pos)]
        x, p, f, B = bag.gather(fb)
        logit, _ = model(x, p, f, B)
        loss = F.binary_cross_entropy_with_logits(logit, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % eval_every == 0:
            sv, _ = mil_predict(model, bag, va)
            ap = wap(y[va], sv, w_eval[va])
            hist.append((step + 1, round(float(loss), 4), round(ap, 4)))
            if ap > best[0]:
                best = (ap, {k: v.detach().clone() for k, v in model.state_dict().items()}, step + 1)
    model.load_state_dict(best[1])
    st, top = mil_predict(model, bag, te, want_att=True)
    return st, top, {'val_ap': best[0], 'best_step': best[2], 'hist': hist, 'train_s': round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------------------------- CV driver
def cross_validate(bag, lab, w_eval, dev, probes, seeds, out_rows, tag, steps):
    """All probes x behaviours x folds (x seeds for MIL). Returns list of result dicts; appends OOF scores."""
    pools = lab['pool'].values
    folds = pool_folds(pools)
    dec = size_deciles(lab['n_fg'].values)
    mean_f = max_f = None
    if {'mean_lin', 'max_lin'} & set(probes):
        t0 = time.time()
        mean_f, max_f = bag.pooled()
        log(f'  pooled features in {time.time() - t0:.0f}s')
    res = []
    for b in BEH:
        y = lab[b].values.astype(bool)
        for k, test_pools in enumerate(folds):
            te = np.flatnonzero(np.isin(pools, test_pools))
            trp = sorted(set(pools) - set(test_pools))
            vp = inner_val(trp, k)
            va = np.flatnonzero(np.isin(pools, list(vp)))
            tr = np.flatnonzero(np.isin(pools, [p for p in trp if p not in vp]))
            base = metrics(y[te], lab['n_fg'].values[te].astype(float), w_eval[te], dec[te])
            res.append({'res': tag, 'probe': 'n_fg_only', 'behaviour': b, 'fold': k, 'seed': -1, **base})
            for pr in probes:
                for sd in ([-1] if pr.endswith('_lin') else seeds):
                    t0 = time.time()
                    top = None
                    if pr.endswith('_lin'):
                        s, info = run_linear(mean_f if pr == 'mean_lin' else max_f, y, w_eval, tr, va, te, dev)
                    else:
                        s, top, info = run_mil(bag, y, w_eval, tr, va, te, pr == 'mil_ctx', sd, dev, steps=steps)
                    m = metrics(y[te], s, w_eval[te], dec[te])
                    r = {'res': tag, 'probe': pr, 'behaviour': b, 'fold': k, 'seed': sd, **m,
                         **info, 'wall_s': round(time.time() - t0, 1)}
                    res.append(r)
                    log(f"  [{tag}] {b} fold {k} {pr} seed {sd}: AUROC {m['auroc']:.3f} AP {m['ap']:.3f} "
                        f"size-ctrl {m['auroc_size_ctrl']:.3f} (base {m['base_rate']:.4f}, n {m['n_frames']}) "
                        f"val AP {info['val_ap']:.3f} {r['wall_s']}s" + (f" best step {info['best_step']}" if top else ''))
                    if out_rows is not None:
                        d = {'i': te, 'res': tag, 'probe': pr, 'behaviour': b, 'fold': k, 'seed': sd, 'score': s,
                             'y': y[te]}
                        if top is not None:
                            d['top_pos'] = [t[0] for t in top]
                            d['top_att'] = [t[1] for t in top]
                            d['n50'] = [t[2] for t in top]
                        out_rows.append(pd.DataFrame(d))
    return res


def summarise(res):
    df = pd.DataFrame(res)
    g = df.groupby(['res', 'probe', 'behaviour'])
    out = []
    for (r, p, b), d in g:
        row = {'res': r, 'probe': p, 'behaviour': b, 'n_runs': len(d), 'n_frames': int(d.n_frames.sum() / max(1, d.seed.nunique())),
               'base_rate_mean': float(d.base_rate.mean())}
        for m in ('auroc', 'ap', 'auroc_size_ctrl'):
            row[m + '_mean'], row[m + '_sd'] = float(d[m].mean()), float(d[m].std(ddof=0))
        if p.startswith('mil'):  # sd over seeds of the fold-mean, and over folds of the seed-mean
            row['auroc_sd_seeds'] = float(d.groupby('seed').auroc.mean().std(ddof=0))
            row['ap_sd_seeds'] = float(d.groupby('seed').ap.mean().std(ddof=0))
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------- sheets
def contact_sheet(oof, lab, beh, path, positives=True, n=16, tile=320, grid=32):
    from PIL import Image, ImageDraw
    d = oof[(oof.probe == 'mil') & (oof.seed == 0) & (oof.behaviour == beh)].copy()
    d['pct'] = d.groupby('fold').score.rank(pct=True)  # scores of different folds are not on one scale
    d = d[d.y == positives].sort_values('pct', ascending=False).head(n)
    fp = pd.read_csv(ANN, usecols=['frame_path', 'frame_idx'])
    rows = lab['row'].values[d.i.values]
    cols = int(np.ceil(np.sqrt(n)))
    sheet = Image.new('RGB', (cols * tile, int(np.ceil(len(d) / cols)) * (tile + 18)), 'white')
    px = 512 / grid
    for j, (r, (_, e)) in enumerate(zip(rows, d.iterrows())):
        im = Image.open(REPO / 'dataset' / fp.frame_path.values[r]).convert('RGB')
        dr = ImageDraw.Draw(im)
        for rank, p in enumerate(e.top_pos):
            if p < 0:
                continue
            y0, x0 = (p // grid) * px, (p % grid) * px
            dr.rectangle([x0, y0, x0 + px, y0 + px], outline=(255, 0, 0) if rank == 0 else (255, 220, 0),
                         width=3 if rank == 0 else 1)
        im = im.resize((tile, tile))
        x, y = (j % cols) * tile, (j // cols) * (tile + 18)
        sheet.paste(im, (x, y))
        ImageDraw.Draw(sheet).text((x + 3, y + tile + 2), f"{lab.obs.values[e.i]} f{fp.frame_idx.values[r]} "
                                   f"p{e.pct:.3f} a{e.top_att:.2f} n50={e.n50}", fill=(0, 0, 0))
    sheet.save(path, quality=80)


# ---------------------------------------------------------------------------------------------- steps
def cmd_a(args):
    out = Path(args.out) / 'a'
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    idx, ev, lab = eval_frames()
    if args.max_videos:  # smoke test: a few videos per pool
        keep = lab.groupby('pool').obs.transform(lambda o: o.isin(sorted(o.unique())[:args.max_videos])).values
        ev, lab = ev[keep], lab[keep].reset_index(drop=True)
    tok, pos, lens = idx.load(ev)
    log(f'A: {len(ev):,} frames, {lab.obs.nunique()} videos, {lab.pool.nunique()} pools, {len(tok):,} tokens, '
        f'loaded in {time.time() - t0:.0f}s; base rates {lab[BEH].mean().round(4).to_dict()}; '
        f'frames without foreground {(lens == 0).sum()}')
    bag = Bag(tok, pos, lens, 32, dev)
    oof = []
    res = cross_validate(bag, lab, np.ones(len(lab)), dev, args.probes, args.seeds, oof, '448_store', args.steps)
    oof = pd.concat(oof, ignore_index=True)
    oof['row'] = lab['row'].values[oof.i.values]
    oof.to_parquet(out / 'oof_scores.parquet')
    summ = summarise(res)
    (out / 'results.json').write_text(json.dumps({'summary': summ, 'runs': res, 'folds': pool_folds(lab.pool.values),
                                                  'n_frames': len(lab), 'n_videos': int(lab.obs.nunique()),
                                                  'n_pools': int(lab.pool.nunique()),
                                                  'gpu': torch.cuda.get_device_name(0),
                                                  'wall_s': round(time.time() - t0, 1), 'args': vars(args)}, indent=1))
    for s in summ:
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
    if 'mil' in args.probes:
        att = oof[(oof.probe == 'mil') & (oof.seed == 0)]
        for b in BEH:
            for yv in (True, False):
                a = att[(att.behaviour == b) & (att.y == yv)]
                log(f'  attention mil seed 0 {b} {"pos" if yv else "neg"}: top-patch mass median {a.top_att.median():.3f}, '
                    f'patches for 50% mass median {a.n50.median():.0f}')
            contact_sheet(oof, lab, b, out / f'sheet_{b}_pos.jpg', True)
            contact_sheet(oof, lab, b, out / f'sheet_{b}_neg.jpg', False)
    log(f'A done in {time.time() - t0:.0f}s -> {out}')


def cmd_b_extract(args):
    from src.eci.extract import load_encoder
    out = Path(args.out) / 'b'
    out.mkdir(parents=True, exist_ok=True)
    stage = Path(args.stage)
    stage.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32, as the store
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    idx, ev, lab = eval_frames()
    rng = np.random.default_rng(0)
    anyp = (lab.nose_nose | lab.nose_tail).values
    P, N = np.flatnonzero(anyp), np.flatnonzero(~anyp)
    ps = np.sort(rng.choice(P, min(len(P), args.pos_cap), replace=False))
    ns = np.sort(rng.choice(N, args.n_subset - len(ps), replace=False))
    sel = np.sort(np.r_[ps, ns])
    w = np.where(anyp[sel], len(P) / len(ps), len(N) / len(ns))
    sub = lab.iloc[sel].reset_index(drop=True)
    sub['w'] = w
    log(f'B subset: {len(sel):,} frames ({len(ps):,}/{len(P):,} positives, {len(ns):,}/{len(N):,} negatives), '
        f'{sub.pool.nunique()} pools; weighted base rates '
        + str({b: round(float(np.average(sub[b], weights=w)), 4) for b in BEH}))
    tok_s, pos_s, lens_s = idx.load(ev[sel])
    fp = pd.read_csv(ANN, usecols=['frame_path'])['frame_path'].values[sub.row.values]
    lst = stage / 'frames.txt'
    lst.write_text('\n'.join(fp) + '\n')
    t1 = time.time()
    subprocess.run(f'tar -C {REPO / "dataset"} -cf - -T {lst} | tar -C {stage} -xf -', shell=True, check=True)
    log(f'staged {len(fp):,} JPGs in {time.time() - t1:.0f}s')

    _, proc448, model = load_encoder('dinov2_base', 448, dev, center_crop=False)
    _, proc896, _ = load_encoder('dinov2_base', 896, dev, center_crop=False)
    from PIL import Image

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(fp)

        def __getitem__(self, i):
            with Image.open(stage / fp[i]) as im:
                im = im.convert('RGB')
            return (proc448(images=im, return_tensors='pt')['pixel_values'][0],
                    proc896(images=im, return_tensors='pt')['pixel_values'][0])

    starts = np.r_[0, np.cumsum(lens_s)]
    n896 = 4 * lens_s
    tok448 = np.lib.format.open_memmap(stage / 'tok448.npy', 'w+', np.float16, (int(lens_s.sum()), 768))
    tok896 = np.lib.format.open_memmap(stage / 'tok896.npy', 'w+', np.float16, (int(n896.sum()), 768))
    pos896 = np.empty(int(n896.sum()), np.int16)
    s896 = np.r_[0, np.cumsum(n896)]
    gpu_s = {448: 0.0, 896: 0.0}
    t2 = time.time()
    dl = torch.utils.data.DataLoader(DS(), batch_size=args.bs896, num_workers=8, shuffle=False, pin_memory=True)
    i0 = 0
    for bi, (p448, p896) in enumerate(dl):
        for res_, pix in ((448, p448), (896, p896)):
            torch.cuda.synchronize(); tt = time.time()
            with torch.inference_mode():
                hs = model(pixel_values=pix.to(dev, non_blocking=True)).last_hidden_state.float()[:, 1:].half()
            torch.cuda.synchronize(); gpu_s[res_] += time.time() - tt
            G = res_ // 14
            assert hs.shape[1] == G * G, hs.shape
            hs = hs.cpu().numpy()
            for j in range(len(pix)):
                f = i0 + j
                p = pos_s[starts[f]:starts[f + 1]].astype(np.int64)
                if res_ == 448:
                    tok448[starts[f]:starts[f + 1]] = hs[j, p]
                else:
                    r, c = p // 32, p % 32
                    q = np.sort(np.concatenate([(2 * r + dr) * 64 + 2 * c + dc for dr in (0, 1) for dc in (0, 1)]))
                    tok896[s896[f]:s896[f + 1]] = hs[j, q]
                    pos896[s896[f]:s896[f + 1]] = q
        i0 += len(p448)
        if bi % 100 == 0:
            log(f'  {i0:,}/{len(fp):,} frames [{time.time() - t2:.0f}s]')
    wall = time.time() - t2
    tok448.flush(); tok896.flush()
    # consistency: re-encoded 448 tokens vs the store
    a = np.asarray(tok448[:2_000_000], np.float32); b = tok_s[:2_000_000].astype(np.float32)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    chk = {'max_abs_diff': float(np.abs(a - b).max()), 'cos_min': float(cos.min()), 'cos_mean': float(cos.mean()),
           'n_tokens_checked': len(a)}
    log(f'448 re-encode vs store: {chk}')
    pos_s.astype(np.int16).tofile(stage / 'pos448.i16')
    pos896.tofile(stage / 'pos896.i16')
    np.savez(stage / 'frames.npz', lens448=lens_s, lens896=n896)
    sub.to_parquet(stage / 'subset.parquet')
    info = {'n_frames': len(fp), 'n_pos_any': int(len(ps)), 'n_pos_total': int(len(P)), 'n_neg': int(len(ns)),
            'n_neg_total': int(len(N)), 'tokens448': int(lens_s.sum()), 'tokens896': int(n896.sum()),
            'mask_896': 'store 448 foreground mask, each 448 patch -> its 2x2 children on the 64x64 grid',
            'frames_source': 'dataset/mice/v1/frames/full JPG 512x512, bicubic-resized to 448 / 896 by the HF processor',
            'gpu': torch.cuda.get_device_name(0), 'batch_size': args.bs896, 'wall_s_both_res': round(wall, 1),
            'gpu_s_448': round(gpu_s[448], 1), 'gpu_s_896': round(gpu_s[896], 1),
            'gpu_s_per_1k_frames_448': round(1000 * gpu_s[448] / len(fp), 1),
            'gpu_s_per_1k_frames_896': round(1000 * gpu_s[896] / len(fp), 1),
            'store_check_448': chk, 'stage_s': round(time.time() - t0, 1)}
    (stage / 'extract.json').write_text(json.dumps(info, indent=1))
    log(json.dumps(info))
    t3 = time.time()
    for f in ('tok448.npy', 'tok896.npy', 'pos448.i16', 'pos896.i16', 'frames.npz', 'subset.parquet', 'extract.json'):
        subprocess.run(['cp', str(stage / f), str(out / f)], check=True)
    log(f'copied to {out} in {time.time() - t3:.0f}s')


def cmd_b_probe(args):
    out = Path(args.out) / 'b'
    src = Path(args.stage) if (Path(args.stage) / 'tok896.npy').exists() else out
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    sub = pd.read_parquet(src / 'subset.parquet')
    z = np.load(src / 'frames.npz')
    res = []
    for r in args.resolutions:
        tok = np.load(src / f'tok{r}.npy', mmap_mode='r')
        tok = np.ascontiguousarray(tok)
        pos = np.fromfile(src / f'pos{r}.i16', dtype=np.int16)
        bag = Bag(tok, pos, z[f'lens{r}'], r // 14, dev)
        res += cross_validate(bag, sub, sub.w.values, dev, args.probes, args.seeds, None, f'{r}_subset', args.steps)
        del bag, tok
        torch.cuda.empty_cache()
    summ = summarise(res)
    (out / 'results.json').write_text(json.dumps({'summary': summ, 'runs': res, 'gpu': torch.cuda.get_device_name(0),
                                                  'wall_s': round(time.time() - t0, 1), 'args': vars(args)}, indent=1))
    for s in summ:
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
    log(f'B probes done in {time.time() - t0:.0f}s -> {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['a', 'b_extract', 'b_probe'])
    p.add_argument('--out', default=str(REPO / 'results/vision/eci_repr_diag/mice'))
    p.add_argument('--stage', default='/tmp')
    p.add_argument('--probes', nargs='+', default=['mean_lin', 'max_lin', 'mil', 'mil_ctx'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--max-videos', type=int, default=0, help='a: first N videos per pool only (smoke test)')
    p.add_argument('--n-subset', type=int, default=20000)
    p.add_argument('--pos-cap', type=int, default=6000)
    p.add_argument('--bs896', type=int, default=16)
    p.add_argument('--resolutions', nargs='+', type=int, default=[448, 896])
    args = p.parse_args()
    {'a': cmd_a, 'b_extract': cmd_b_extract, 'b_probe': cmd_b_probe}[args.cmd](args)


if __name__ == '__main__':
    main()
