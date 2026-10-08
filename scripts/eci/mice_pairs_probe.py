"""
Per-mouse / per-pair token checks, step 4 (GPU), Check 2: do ordered-pair tokens carry mouse nose contact better than
patch tokens, on the SAME frames? Supervised probes = an upper bound on what an SAE on the same tokens could expose.

Frames: the B subset of the patch diagnostic (results/vision/eci_repr_diag/mice/b: DINOv2-base 448 px foreground patch
tokens tok448 / pos448, 32 x 32 grid; sampling weights w restore the natural base rate) restricted to the frames whose
combined-splitter masks pass the AUTOMATIC rule (results/vision/eci_mice_pairs/bsub/geom.npz; no label used).

Patch -> mouse (pre-registered): patch (r, c) belongs to the mouse holding the most of its 16 x 16 frame pixels if that
is >= 64 px (25 % of the patch), else to none. Mouse m's token set = its STORED foreground patches (the store keeps
only foreground patches); if that is empty, the stored patch with the most pixels of m; if m overlaps no stored patch
its token blocks are zeros (counted).
Tokens:
    indiv_m   [mean of m's patch tokens (768) ; max (768) ; area / A1_m, centroid y, x (/ 512), cos 2 theta,
              sin 2 theta (principal axis, mod pi)]  = 1541
    pair i->j [indiv_i ; indiv_j ; CONTACT = (mean of i's tokens on patches within 1 patch (3 x 3) of j's patch mask ;
              mean of j's within 1 of i's), zeros when not adjacent (1536) ; geometry = centroid distance (patches),
              minimum mask-to-mask pixel distance / 16, cos 2 (theta_j - theta_i), sin 2 (theta_j - theta_i),
              cos 2 phi, sin 2 phi (phi = angle of j's centroid in i's axis frame), adjacent flag (7)]   = 4625
    Head and tail cannot be told apart from a mask without keypoints: every angle is encoded mod pi (2x angle), so a
    pair token is the same whichever end of either mouse is its nose. 12 ordered pairs per frame.
Every feature is standardised with the training fold's mean / sd (over the training frames' pair tokens).
Probes (each behaviour, the patch diagnostic's protocol: pool-grouped 5-fold CV with the same fold and inner-validation
pools, 3000 AdamW steps, batches of 128 with 25 % positives, checkpoint by inner-validation weighted AP, 3 seeds; the
test fold is evaluated once at the natural base rate with the inverse-sampling weights):
    P1        gated-attention MIL over the 12 pair tokens (Linear 256 -> GELU -> gated attention 128 -> MLP head)
    P1_nocontact  P1 without the CONTACT block (3089-d)
    P1_geom   P1 on geometry only: [area, centroid, cos / sin 2 theta of i and j ; pair geometry] (17-d)
    P2        a linear pair score w . z_ij + b, frame score = max over the 12 pairs (one direction on the pair token)
    P3        the patch baseline mil_ctx (scripts/eci/diag_mice_patches.py MIL, ctx=True, unchanged) trained and
              tested on the usable frames only
    P3_all    mil_ctx trained on all 20,000 subset frames (the original b_probe protocol), tested on the usable frames
              of the test fold (b_probe saved no out-of-fold scores, so it is re-run)
    mindist   no training: frame score = - min over pairs of the mask-to-mask distance (is contact just distance?)
    n_fg_only frame score = number of foreground patches
Metrics: AUROC, AP, size-controlled AUROC (inside foreground-count deciles fixed over the usable frames).
Output results/vision/eci_mice_pairs/bsub/probe/{results.json, oof.parquet}.
Usage: python scripts/eci/mice_pairs_probe.py [--probes P1 P2 ...] [--seeds 0 1 2] [--steps 3000]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from diag_mice_patches import (BEH, Bag, inner_val, metrics, pool_folds, run_mil, seg_softmax,  # noqa: E402
                               size_deciles, summarise, wap)

OUT = REPO / 'results/vision/eci_mice_pairs/bsub'
PAIRS = [(i, j) for i in range(4) for j in range(4) if i != j]
log = lambda s: print(s, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------------------------- tokens
def build_tokens(tok, pos, starts, sub_idx, cnt, stats, mind):
    """-> indiv (U, 4, 1541) f32, contact (U, 12, 1536) f16, pgeom (U, 12, 7) f32, info."""
    U = len(sub_idx)
    indiv = np.zeros((U, 4, 1541), np.float32)
    contact = np.zeros((U, 12, 1536), np.float16)
    pgeom = np.zeros((U, 12, 7), np.float32)
    n_fallback = n_empty = 0
    n_tok = np.zeros((U, 4), np.int32)
    k3 = np.ones((3, 3), bool)
    for u, i in enumerate(sub_idx):
        T = np.asarray(tok[starts[i]:starts[i + 1]], np.float32)
        P = pos[starts[i]:starts[i + 1]].astype(np.int64)
        c = cnt[u].reshape(4, 1024).astype(np.int32)
        lab = np.where(c.max(0) >= 64, c.argmax(0) + 1, 0)  # (1024,)
        own = []
        for m in range(4):
            sel = lab[P] == m + 1
            if not sel.any():
                cm = c[m, P]
                if cm.max() > 0:
                    sel = np.zeros(len(P), bool)
                    sel[int(np.argmax(cm))] = True
                    n_fallback += 1
                else:
                    n_empty += 1
            own.append(sel)
            n_tok[u, m] = int(sel.sum())
            if sel.any():
                indiv[u, m, :768] = T[sel].mean(0)
                indiv[u, m, 768:1536] = T[sel].max(0)
            a, cy, cx, th = stats[u, m, :4]
            indiv[u, m, 1536:] = (a, cy, cx, np.cos(2 * th), np.sin(2 * th))
        near = [ndi.binary_dilation((lab == m + 1).reshape(32, 32), k3).ravel()[P] for m in range(4)]
        for k, (a_, b_) in enumerate(PAIRS):
            ia, jb = own[a_] & near[b_], own[b_] & near[a_]
            if ia.any():
                contact[u, k, :768] = T[ia].mean(0)
            if jb.any():
                contact[u, k, 768:] = T[jb].mean(0)
            ca, cb = stats[u, a_, 1:3] * 512, stats[u, b_, 1:3] * 512
            rel = cb - ca
            th = stats[u, a_, 3]
            ax = np.array([np.sin(th), np.cos(th)])          # (y, x) of i's major axis
            pe = np.array([np.cos(th), -np.sin(th)])
            phi = np.arctan2(rel @ pe, rel @ ax)
            dth = stats[u, b_, 3] - th
            pgeom[u, k] = (np.hypot(*rel) / 16, mind[u, a_, b_], np.cos(2 * dth), np.sin(2 * dth),
                           np.cos(2 * phi), np.sin(2 * phi), float(ia.any() or jb.any()))
    info = {'n_mouse_fallback_patch': int(n_fallback), 'n_mouse_no_stored_patch': int(n_empty),
            'tokens_per_mouse_median': float(np.median(n_tok)), 'frac_pairs_adjacent': float(pgeom[..., 6].mean())}
    return indiv, contact, pgeom, info


class PairData:
    """Pair tokens composed on the GPU per batch; feature mode full | nocontact | geom."""

    def __init__(self, indiv, contact, pgeom, mode, dev):
        self.mode, self.dev = mode, dev
        I = torch.tensor([p[0] for p in PAIRS]); J = torch.tensor([p[1] for p in PAIRS])
        self.I, self.J = I.to(dev), J.to(dev)
        if mode == 'geom':
            self.indiv = torch.from_numpy(indiv[..., 1536:]).to(dev)
        else:
            self.indiv = torch.from_numpy(indiv).to(dev)
        self.contact = torch.from_numpy(contact).to(dev) if mode == 'full' else None
        self.pgeom = torch.from_numpy(pgeom).to(dev)
        self.mu = self.sd = None

    def raw(self, f):
        f = torch.as_tensor(np.asarray(f), device=self.dev)
        ind = self.indiv[f]
        parts = [ind[:, self.I], ind[:, self.J]]
        if self.contact is not None:
            parts.append(self.contact[f].float())
        parts.append(self.pgeom[f])
        return torch.cat(parts, -1)

    def fit_norm(self, tr, chunk=2048):
        s = s2 = 0
        n = 0
        for c in range(0, len(tr), chunk):
            x = self.raw(tr[c:c + chunk]).double()
            x = x.reshape(-1, x.shape[-1])
            s = s + x.sum(0); s2 = s2 + (x * x).sum(0); n += len(x)
        mu = s / n
        self.mu = mu.float()
        self.sd = ((s2 / n - mu ** 2).clamp_min(0).sqrt() + 1e-4).float()

    @property
    def dim(self):
        return int(self.raw(np.zeros(1, int)).shape[-1])

    def get(self, f):
        return (self.raw(f) - self.mu) / self.sd


class PairMIL(nn.Module):
    def __init__(self, d, h=256, att=128, drop=0.1, linear=False):
        super().__init__()
        self.linear = linear
        if linear:
            self.w = nn.Linear(d, 1)
            return
        self.proj = nn.Linear(d, h)
        self.drop = nn.Dropout(drop)
        self.V, self.U, self.a = nn.Linear(h, att), nn.Linear(h, att), nn.Linear(att, 1)
        self.head = nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))

    def forward(self, x):  # x (B, 12, d)
        if self.linear:
            s = self.w(x).squeeze(-1)
            return s.max(1).values, s
        h = self.drop(F.gelu(self.proj(x)))
        att = torch.softmax(self.a(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1), 1)
        return self.head((att[..., None] * h).sum(1)).squeeze(-1), att


@torch.no_grad()
def pair_predict(model, data, fidx, chunk=1024):
    model.eval()
    out = np.zeros(len(fidx), np.float32)
    for c in range(0, len(fidx), chunk):
        out[c:c + chunk] = model(data.get(fidx[c:c + chunk]))[0].float().cpu().numpy()
    model.train()
    return out


def run_pair(data, y, w_eval, tr, va, te, seed, dev, linear, steps=3000, bs=128, pos_frac=0.25, lr=1e-3, wd=0.05,
             eval_every=300):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ptr, ntr = tr[y[tr]], tr[~y[tr]]
    model = PairMIL(data.dim, linear=linear).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n_pos = int(round(pos_frac * bs))
    yb = torch.cat([torch.ones(n_pos), torch.zeros(bs - n_pos)]).to(dev)
    best, hist, t0 = (-1.0, None, -1), [], time.time()
    for step in range(steps):
        g = lr * min(1.0, (step + 1) / 100) * 0.5 * (1 + np.cos(np.pi * step / steps))
        for pg in opt.param_groups:
            pg['lr'] = g
        fb = np.r_[rng.choice(ptr, n_pos), rng.choice(ntr, bs - n_pos)]
        logit, _ = model(data.get(fb))
        loss = F.binary_cross_entropy_with_logits(logit, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % eval_every == 0:
            ap = wap(y[va], pair_predict(model, data, va), w_eval[va])
            hist.append((step + 1, round(float(loss), 4), round(ap, 4)))
            if ap > best[0]:
                best = (ap, {k: v.detach().clone() for k, v in model.state_dict().items()}, step + 1)
    model.load_state_dict(best[1])
    return pair_predict(model, data, te), {'val_ap': best[0], 'best_step': best[2], 'hist': hist,
                                           'train_s': round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', nargs='+', default=['mindist', 'P1', 'P1_nocontact', 'P1_geom', 'P2', 'P3', 'P3_all'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--max-frames', type=int, default=0, help='smoke test: N usable frames spread over the subset')
    args = ap.parse_args()
    stage = Path(os.environ.get('STAGE', REPO / 'results/vision/eci_repr_diag/mice/b'))
    out = OUT / 'probe'
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    sub = pd.read_parquet(stage / 'subset.parquet')
    tgt = pd.read_parquet(OUT / 'targets.parquet')
    assert (tgt.row.values == sub.row.values).all(), 'targets.parquet is not in subset order'
    lens = np.load(stage / 'frames.npz')['lens448'].astype(np.int64)
    starts = np.r_[0, np.cumsum(lens)]
    tok = np.load(stage / 'tok448.npy', mmap_mode='r')
    pos = np.fromfile(stage / 'pos448.i16', dtype=np.int16)
    gz = np.load(OUT / 'geom.npz')
    uidx = gz['idx']
    cnt, stats, mind = gz['cnt'], gz['stats'], gz['mind']
    if args.max_frames:
        k = np.linspace(0, len(uidx) - 1, args.max_frames).astype(int)  # spread over all pools
        uidx, cnt, stats, mind = uidx[k], cnt[k], stats[k], mind[k]
    log(f'{len(uidx):,} usable of {len(sub):,} subset frames [{time.time() - t0:.0f}s]')
    t1 = time.time()
    tok_ram = np.ascontiguousarray(tok)
    indiv, contact, pgeom, tinfo = build_tokens(tok_ram, pos, starts, uidx, cnt, stats, mind)
    log(f'pair tokens built in {time.time() - t1:.0f}s: {json.dumps(tinfo)}')

    lab = sub.iloc[uidx].reset_index(drop=True)
    w_eval = lab.w.values
    pools = lab.pool.values
    folds = pool_folds(sub.pool.values)  # the b_probe folds (all 24 pools)
    dec = size_deciles(lab.n_fg.values)
    res, oof = [], []

    def record(pr, b, k, sd, y, te, s, info):
        m = metrics(y[te], s, w_eval[te], dec[te])
        res.append({'res': 'usable', 'probe': pr, 'behaviour': b, 'fold': k, 'seed': sd, **m, **info})
        oof.append(pd.DataFrame({'u': te, 'row': lab.row.values[te], 'probe': pr, 'behaviour': b, 'fold': k,
                                 'seed': sd, 'score': s, 'y': y[te]}))
        log(f"  {b} fold {k} {pr} seed {sd}: AUROC {m['auroc']:.3f} AP {m['ap']:.3f} size-ctrl "
            f"{m['auroc_size_ctrl']:.3f} (base {m['base_rate']:.4f}, n {m['n_frames']}, pos {m['n_pos']})"
            + (f" val AP {info['val_ap']:.3f} step {info['best_step']} {info['train_s']}s" if 'val_ap' in info else ''))

    def split(k):
        test_pools = folds[k]
        trp = sorted(set(sub.pool.values) - set(test_pools))
        vp = inner_val(trp, k)
        te = np.flatnonzero(np.isin(pools, test_pools))
        va = np.flatnonzero(np.isin(pools, list(vp)))
        tr = np.flatnonzero(np.isin(pools, [p for p in trp if p not in vp]))
        return tr, va, te, test_pools, vp

    # training-free references
    for pr in ('n_fg_only', 'mindist'):
        if pr != 'n_fg_only' and pr not in args.probes:
            continue
        for b in BEH:
            y = lab[b].values.astype(bool)
            for k in range(5):
                te = split(k)[2]
                if pr == 'n_fg_only':
                    s = lab.n_fg.values[te].astype(float)
                else:
                    s = -np.where(np.eye(4, dtype=bool)[None], np.inf, mind[te]).reshape(len(te), -1).min(1)
                record(pr, b, k, -1, y, te, s, {})

    modes = {'P1': ('full', False), 'P1_nocontact': ('nocontact', False), 'P1_geom': ('geom', False), 'P2': ('full', True)}
    for pr in [p for p in args.probes if p in modes]:
        mode, linear = modes[pr]
        data = PairData(indiv, contact, pgeom, mode, dev)
        log(f'{pr}: pair-token dim {data.dim}')
        for b in BEH:
            y = lab[b].values.astype(bool)
            for k in range(5):
                tr, va, te, _, _ = split(k)
                data.fit_norm(tr)
                for sd in args.seeds:
                    s, info = run_pair(data, y, w_eval, tr, va, te, sd, dev, linear, steps=args.steps)
                    record(pr, b, k, sd, y, te, s, info)
        del data
        torch.cuda.empty_cache()

    if 'P3' in args.probes or 'P3_all' in args.probes:
        if 'P3_all' in args.probes:  # mil_ctx on every subset frame; test = usable frames of the test fold
            bag = Bag(tok_ram, pos, lens, 32, dev)
            sp = sub.pool.values
            umap = np.full(len(sub), -1)
            umap[uidx] = np.arange(len(uidx))
            for b in BEH:
                yall = sub[b].values.astype(bool)
                y = lab[b].values.astype(bool)
                for k in range(5):
                    _, _, te_u, test_pools, vp = split(k)
                    trp = sorted(set(sp) - set(test_pools))
                    va_a = np.flatnonzero(np.isin(sp, list(vp)))
                    tr_a = np.flatnonzero(np.isin(sp, [p for p in trp if p not in vp]))
                    for sd in args.seeds:
                        s, _, info = run_mil(bag, yall, sub.w.values, tr_a, va_a, uidx[te_u], True, sd, dev,
                                             steps=args.steps)
                        info.pop('hist')
                        record('P3_all', b, k, sd, y, te_u, s, info)
            del bag
            torch.cuda.empty_cache()
        if 'P3' in args.probes:
            sel = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in uidx])
            bag = Bag(tok_ram[sel], pos[sel], lens[uidx], 32, dev)
            for b in BEH:
                y = lab[b].values.astype(bool)
                for k in range(5):
                    tr, va, te, _, _ = split(k)
                    for sd in args.seeds:
                        s, _, info = run_mil(bag, y, w_eval, tr, va, te, True, sd, dev, steps=args.steps)
                        info.pop('hist')
                        record('P3', b, k, sd, y, te, s, info)
            del bag
            torch.cuda.empty_cache()
    summ = summarise(res)
    for s in summ:  # summarise() adds seed sds only for names starting with 'mil'
        d = pd.DataFrame([r for r in res if r['probe'] == s['probe'] and r['behaviour'] == s['behaviour']])
        if d.seed.nunique() > 1:
            s['auroc_sd_seeds'] = float(d.groupby('seed').auroc.mean().std(ddof=0))
            s['ap_sd_seeds'] = float(d.groupby('seed').ap.mean().std(ddof=0))
    pd.concat(oof, ignore_index=True).to_parquet(out / 'oof.parquet')
    (out / 'results.json').write_text(json.dumps({'summary': summ, 'runs': [{k: v for k, v in r.items() if k != 'hist'}
                                                                            for r in res],
                                                  'folds': folds, 'n_usable': int(len(uidx)), 'token_info': tinfo,
                                                  'gpu': torch.cuda.get_device_name(0),
                                                  'wall_s': round(time.time() - t0, 1), 'args': vars(args)}, indent=1))
    for s in summ:
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
    log(f'done in {time.time() - t0:.0f}s -> {out}')


if __name__ == '__main__':
    main()
