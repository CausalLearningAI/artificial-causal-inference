"""
T2 (tight animal mask) SAE arm and the T4 read-outs / level map, for mice and ants. GPU.

Arms (identical recipe, splits and evaluation; only the mask differs)
    base  the levers 'w4096' SAEs (results/vision/eci_sae_levers/<d>/sae/w4096_s{0,1,2}, Matryoshka BatchTopK 4096, k 16),
          today's mask (the store's tokens).
    t2    the same recipe (scripts/eci/sae_levers.py train_one, cfg 'w4096', cap mice 4M / ants 2M tokens, pool seed 0,
          6000 steps, seeds 0-2) trained on the TIGHT-mask tokens of the SAME training frames: store tokens whose patch is
          kept by the tight rule (results/vision/eci_t2t4/<d>/mask/masks.npz, scripts/eci/tight_mask.py). Patches kept
          by the tight rule but absent from the store: mice 7.1% of the new patches, encoded with the store's encoder
          (scripts/eci/tight_mask_encode.py) and merged in; ants 1.0%, skipped (apply.json).
Read-outs per frame and neuron (threshold inference, the frame's kept tokens)
    presence  max over the kept patches (= codes_max / codes_best)
    extent    number of kept patches where the neuron is active (> 0); in animal units = extent / a1, a1 = the arm's
              median single-animal component area on its train frames (sae_levers.merged_meta: frames whose mask
              splits into exactly one component per animal)
Frame-level scores: sae_levers.score_model for each read-out (cross-fitted best-neuron AUROC with AP at that neuron,
best AP, honest top-1% precision (neurons firing on >= 1% of frames), size-controlled AUROC within the arm's
foreground-count deciles).

LEVEL MAP: two axes and a flag, per eligible neuron (fires on >= 1% of eval frames), over its K = round(1% of eval
frames) highest-presence frames (top frames with presence > 0). RULES FIXED HERE BEFORE ANY LABEL IS LOOKED AT.
Analysis tools: the neuron map's dark blobs and site maps (scripts/eci/neuron_map.py site_frame, single-animal dark
area A1; a patch's site blob = the blob holding most of its blob pixels, else the nearest blob within 16 px, else none
= off-animal; a blob's animal count = clip(round(area / A1), 1, 4)).
    per top frame   E  = extent / a1 (animal units); NB = distinct site blobs among the active patches;
                    NA = sum of the animal counts of those blobs (a merged blob of 2 counts 2)
    social axis     from the medians over the top frames (E~, NA~):
                        off-animal  NA~ = 0
                        self        NA~ <= 1 and E~ <= 1.5
                        collective  NA~ >= 3 or E~ >= 2.5
                        pair        otherwise (2 animals' worth, or a merged blob of 2)
    place-bound     conc > 0.3 AND argmax-off <= 0.5: the activation sits at a fixed arena location while on animals.
                    conc = 1 - H / H_ref(K) of the argmax patches of the top frames on an 8 x 8 grid of 4 x 4-patch cells
                    (neuron_map (a); H_ref = mean entropy of K kept patches drawn uniformly, 200 draws); argmax-off =
                    share of those argmax patches with no site blob
    plain background flag   active-off > 0.5: more than half of the neuron's active patches over its top frames have no
                    site blob (not on or next to an animal)
    presence-only social level (comparison for the T4 rule): the animal count of the site blob at the argmax patch,
                    median over the top frames' on-animal argmax patches: <= 1 self, 2 pair, >= 3 collective, none on
                    an animal -> off-animal (neuron_map's count statistic)

Steps
    train     t2 arm only: stage the tight train tokens -> OUT/<d>/sae/t2_w4096_s{seed}/, OUT/<d>/train_meta_t2.json
    evaluate  per arm (base, t2) and seed: presence + extent codes, scores, level map, picks' codes, sheets (seed 0)
              -> OUT/<d>/eval/{arm}_s{seed}.json, levels/{arm}_s{seed}.csv, codes_best/{arm}_s{seed}.npz, sheets/
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import multiscale_sae as ms  # noqa: E402
import sae_levers as sl  # noqa: E402
import spatial_sae_pilot as ssp  # noqa: E402
from src.eci.sae import load_sae  # noqa: E402

OUT = REPO / 'results/vision/eci_t2t4'
LEV = REPO / 'results/vision/eci_sae_levers'
D = 768
GRID, PX = 32, 16
SEEDS = (0, 1, 2)
MIN_RATE, TOP_FRAC = 0.01, 0.01
SELF_E, COLL_E, COLL_NA = 1.5, 2.5, 3
PLACE_CONC, PLACE_OFF, PLAIN_OFF = 0.3, 0.5, 0.5
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------------------------- data
def mask_bits(domain, frames):
    z = np.load(OUT / domain / 'mask' / 'masks.npz')
    k = np.searchsorted(z['frames'], frames)
    assert (z['frames'][k] == frames).all(), 'frames missing from masks.npz'
    return z['bits'][k], (z['n_new'][k] - z['new_only'][k]).astype(np.int64), k, z['new_only'][k].astype(np.int64)


def load_extra(domain, k, n_frames):
    """Encoded tight-mask patches absent from the store (scripts/eci/tight_mask_encode.py), restricted to the requested
    frames (k = their indices into masks.npz) -> (tok (n, 768) fp16, frame position (n,), pos (n,)) sorted by (frame,
    pos), or None when the domain has none (ants: 1.0% of the new patches, skipped)."""
    parts = sorted((OUT / domain / 'extra').glob('part_*'))
    if not parts:
        return None
    inv = np.full(int(k.max()) + 1, -1, np.int64)
    inv[k] = np.arange(n_frames)
    T, Fp, Pp = [], [], []
    for d in parts:
        fidx = np.load(d / 'fidx.npy').astype(np.int64)
        ok = fidx <= k.max()
        ok[ok] = inv[fidx[ok]] >= 0
        T.append(np.asarray(np.load(d / 'tok.npy', mmap_mode='r')[np.flatnonzero(ok)]))
        Fp.append(inv[fidx[ok]])
        Pp.append(np.load(d / 'pos.npy')[ok])
    T, Fp, Pp = np.concatenate(T), np.concatenate(Fp), np.concatenate(Pp)
    o = np.lexsort((Pp, Fp))
    return T[o], Fp[o], Pp[o]


def stage(idx, frames, dst, arm, domain, chunk=8000):
    """Tokens of the store frames (sorted), base = all store tokens, t2 = those kept by the tight mask
    -> (tok memmap (N, 768) fp16, pos (N,) int16, lens (F,) int64)."""
    if arm == 'base':
        return ms.stage(idx, frames, dst)
    bits, n_keep, k, new_only = mask_bits(domain, frames)
    ext = load_extra(domain, k, len(frames))
    if ext is not None:
        n_ext = np.bincount(ext[1], minlength=len(frames))
        assert (n_ext == new_only).all(), 'encoded extra patches do not cover the absent patches'
        n_keep = n_keep + n_ext
        ecut = np.searchsorted(ext[1], np.arange(0, len(frames) + chunk, chunk))
    dst.mkdir(parents=True, exist_ok=True)
    n = int(n_keep.sum())
    t0 = time.time()
    tok = np.lib.format.open_memmap(dst / 'tok.npy', 'w+', np.float16, (n, D))
    pos = np.empty(n, np.int16)
    lens = np.zeros(len(frames), np.int64)
    o = 0
    for c0 in range(0, len(frames), chunk):
        t, p, ln = idx.load(frames[c0:c0 + chunk])
        new = np.unpackbits(bits[c0:c0 + chunk], axis=1).astype(bool)
        fi = np.repeat(np.arange(len(ln)), ln)
        keep = new[fi, p.astype(np.int64)]
        t, p, fi = t[keep], p[keep], fi[keep]
        if ext is not None:
            a, b = ecut[c0 // chunk], ecut[c0 // chunk + 1]
            t = np.concatenate([t, ext[0][a:b]])
            p = np.concatenate([p, ext[2][a:b].astype(p.dtype)])
            fi = np.concatenate([fi, ext[1][a:b] - c0])
            srt = np.lexsort((p, fi))
            t, p, fi = t[srt], p[srt], fi[srt]
        lens[c0:c0 + chunk] = np.bincount(fi, minlength=len(ln))
        k = len(p)
        tok[o:o + k] = t
        pos[o:o + k] = p
        o += k
    assert o == n and (lens == n_keep).all(), (o, n)
    tok.flush()
    del tok
    log(f'  staged {len(frames):,} frames, {n:,} tight-mask tokens ({n * D * 2 / 1e9:.1f} GB) in {time.time() - t0:.0f}s')
    return np.load(dst / 'tok.npy', mmap_mode='r'), pos, lens


# ---------------------------------------------------------------------------------------------- train
def cmd_train(args):
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out_dir) if args.out_dir else OUT / args.domain
    loc = ms.local_dir()
    idx = ssp.StoreIndex(args.domain)
    tr, _, _, _, train_v, _ = idx.split(args.domain, ms.N_TRAIN[args.domain], 0)
    tr = np.sort(tr)
    if args.max_frames:
        tr = np.sort(np.random.default_rng(0).choice(tr, args.max_frames, replace=False))
    vids, fvid = sl.video_index(idx.obs[tr])
    tok, pos, lens = stage(idx, tr, loc / 'train', 't2', args.domain)
    n = len(pos)
    tvid = np.repeat(fvid, lens)
    nc, largest, meta = sl.merged_meta(args.domain, pos, lens)
    merged_t = np.repeat(largest > meta['threshold'], lens)
    cap = min(sl.CAP[args.domain], n)
    sel = np.sort(np.random.default_rng(0).choice(n, cap, replace=False))
    pools = {'uniform': (torch.from_numpy(sl.gather(tok, sel)).to(dev), torch.from_numpy(tvid[sel]).to(dev),
                         torch.from_numpy(pos[sel].astype(np.int64)).to(dev), torch.from_numpy(merged_t[sel]).to(dev))}
    del tok
    shutil.rmtree(loc / 'train')
    base_meta = json.loads((LEV / args.domain / 'train_meta_w4096_w16384.json').read_text())
    meta.update(domain=args.domain, arm='t2', n_train_frames=int(len(tr)), n_train_tokens=int(n), cap=int(cap),
                pool_seed=0, steps=args.steps, train_videos=train_v,
                tokens_per_frame=float(n / len(tr)), base_tokens_per_frame=base_meta['n_train_tokens'] / base_meta['n_train_frames'],
                base_median_single_animal_area=base_meta['median_single_animal_area'],
                frames_with_no_token=float((lens == 0).mean()))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'train_meta_t2.json').write_text(json.dumps(meta, indent=1))
    log(f'  t2 train: {n:,} tokens ({n / len(tr):.1f}/frame vs base {meta["base_tokens_per_frame"]:.1f}), pool {cap:,}; '
        f'single-animal area {meta["median_single_animal_area"]} patches (base {base_meta["median_single_animal_area"]}), '
        f'n_comp hist {meta["n_comp_hist"]}')
    cfg = sl.parse_cfg('w4096')
    for seed in args.seeds:
        sl.train_one('t2_w4096', cfg, seed, pools, None, args.steps, out / 'sae' / f't2_w4096_s{seed}',
                     {'config': 't2_w4096', 'lever': cfg, 'domain': args.domain, 'cap': cap, 'pool_seed': 0,
                      'mask': 'tight', 'rule': json.loads((OUT / 'rule.json').read_text())}, dev)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- evaluate
@torch.no_grad()
def encode_pe(sae, norm, tok, lens, dev):
    """-> presence (F, m) fp16, extent (F, m) fp16, stats (FVE, L0, dead fraction)."""
    m, F_ = sae.n_latents, len(lens)
    st = np.r_[0, np.cumsum(lens)]
    pres = torch.zeros(F_, m, dtype=torch.float16, device=dev)
    ext = torch.zeros(F_, m, dtype=torch.float16, device=dev)
    s1 = torch.zeros(D, dtype=torch.float64, device=dev)
    s2 = sse = l0 = 0.0
    fire = torch.zeros(m, dtype=torch.float64, device=dev)
    ntok = 0
    for f0, f1 in sl.frame_chunks(lens, max(4096, 2 ** 27 // m)):
        lo, hi = st[f0], st[f1]
        if hi == lo:
            continue
        x = norm(torch.from_numpy(np.ascontiguousarray(tok[lo:hi])).to(dev))
        z = sae.encode(x, mode='threshold')
        xh = sae.decode(z)
        s1 += x.double().sum(0)
        s2 += float(x.double().pow(2).sum())
        sse += float((x - xh).double().pow(2).sum())
        act = z > 0
        fire += act.sum(0).double()
        l0 += float(act.sum())
        ntok += len(x)
        fr = torch.repeat_interleave(torch.arange(f1 - f0, device=dev), torch.from_numpy(lens[f0:f1]).to(dev))
        mx = torch.zeros(f1 - f0, m, device=dev)
        mx.index_reduce_(0, fr, z, 'amax', include_self=True)
        pres[f0:f1] = mx.half()
        cnt = torch.zeros(f1 - f0, m, device=dev)
        cnt.index_add_(0, fr, act.float())
        ext[f0:f1] = cnt.half()
    tss = s2 - float(s1.pow(2).sum()) / ntok
    return pres, ext, {'fve': 1 - sse / tss, 'l0_per_token': l0 / ntok, 'dead_frac': float(((fire == 0).double().mean())),
                       'n_tokens': ntok}


def entropy64(h):
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


@torch.no_grad()
def level_map(sae, norm, tok, pos, lens, pres, ext, site, bcount, a1, dev, KB):
    """Per-neuron level-map statistics (module docstring) -> DataFrame over all neurons (eligible flag) + checks."""
    m, F_ = sae.n_latents, len(lens)
    st = np.r_[0, np.cumsum(lens)]
    fire = (pres > 0).float().mean(0)
    K = int(round(TOP_FRAC * F_))
    topi = torch.empty(K, m, dtype=torch.long, device=dev)
    topv = torch.empty(K, m, device=dev)
    for j0 in range(0, m, 512):
        v, i = torch.topk(pres[:, j0:j0 + 512].float(), K, dim=0)
        topi[:, j0:j0 + 512], topv[:, j0:j0 + 512] = i, v
    istop = torch.zeros(F_, m, dtype=torch.bool, device=dev)
    istop.scatter_(0, topi, topv > 0)
    nb = torch.zeros(F_, m, dtype=torch.int8, device=dev)
    na = torch.zeros(F_, m, dtype=torch.int8, device=dev)
    act_n = torch.zeros(m, dtype=torch.float64, device=dev)
    act_off = torch.zeros(m, dtype=torch.float64, device=dev)
    max_n = torch.zeros(m, dtype=torch.float64, device=dev)
    max_off = torch.zeros(m, dtype=torch.float64, device=dev)
    hist_cell = torch.zeros(m, 64, device=dev)
    hist_cnt = torch.zeros(m, 5, device=dev)
    found = torch.zeros(m, dtype=torch.float64, device=dev)
    site_t = torch.from_numpy(site).to(dev)
    bc_t = torch.from_numpy(bcount).to(dev).float()
    for f0, f1 in sl.frame_chunks(lens, max(4096, 2 ** 27 // m)):  # same chunks as encode_pe (bit-equal codes)
        lo, hi = st[f0], st[f1]
        if hi == lo:
            continue
        nf = f1 - f0
        z = sae.encode(norm(torch.from_numpy(np.ascontiguousarray(tok[lo:hi])).to(dev)), mode='threshold')
        P = torch.from_numpy(pos[lo:hi].astype(np.int64)).to(dev)
        fr = torch.repeat_interleave(torch.arange(nf, device=dev), torch.from_numpy(lens[f0:f1]).to(dev))
        a = (z > 0) & istop[f0:f1][fr]
        s = site_t[f0 + fr, P].long()
        off = s == 0
        act_n += a.sum(0).double()
        act_off += (a & off[:, None]).sum(0).double()
        on = ~off
        key = fr[on] * KB + s[on] - 1
        touch = torch.zeros(nf * KB, m, device=dev)
        touch.index_reduce_(0, key, a[on].float(), 'amax', include_self=True)
        touch = touch.view(nf, KB, m)
        nb[f0:f1] = touch.sum(1).clamp_max(127).to(torch.int8)
        na[f0:f1] = (touch * bc_t[f0:f1, :, None]).sum(1).clamp_max(127).to(torch.int8)
        del touch
        ismax = a & (z.half() == pres[f0:f1][fr])
        cell = (P // GRID) // 4 * 8 + (P % GRID) // 4
        hist_cell += ismax.T.float() @ torch.nn.functional.one_hot(cell, 64).float()
        cnt_at = torch.where(on, bc_t[f0 + fr, (s - 1).clamp_min(0)], torch.zeros_like(s, dtype=torch.float32)).long()
        hist_cnt += ismax.T.float() @ torch.nn.functional.one_hot(cnt_at, 5).float()
        max_n += ismax.sum(0).double()
        max_off += (ismax & off[:, None]).sum(0).double()
        fm = torch.zeros(nf, m, device=dev)
        fm.index_reduce_(0, fr, ismax.float(), 'amax', include_self=True)
        found += (fm > 0).sum(0).double()
    ok = topv > 0
    gather = lambda X: torch.where(ok, X.gather(0, topi).float(), torch.full_like(topv, float('nan')))  # noqa: E731
    E = gather(ext) / a1
    med = lambda X: torch.nanmedian(X, 0).values.cpu().numpy()  # noqa: E731
    Em, NAm, NBm = med(E), med(gather(na)), med(gather(nb))
    n_top = ok.sum(0).cpu().numpy()
    # reference entropy of K kept patches (uniform over the eval tokens)
    rng = np.random.default_rng(0)
    cells = (pos.astype(np.int64) // GRID) // 4 * 8 + (pos.astype(np.int64) % GRID) // 4
    Href = float(np.mean([entropy64(np.bincount(cells[rng.choice(len(cells), K, replace=False)], minlength=64))
                          for _ in range(200)]))
    hc = hist_cell.cpu().numpy()
    hn = hist_cnt.cpu().numpy()
    T = pd.DataFrame({'neuron': np.arange(m), 'fire_frac': fire.cpu().numpy(), 'n_top': n_top,
                      'E_med': Em, 'NA_med': NAm, 'NB_med': NBm,
                      'active_off': (act_off / act_n.clamp_min(1)).cpu().numpy(),
                      'argmax_off': (max_off / max_n.clamp_min(1)).cpu().numpy(),
                      'conc': [1 - entropy64(h) / Href for h in hc],
                      'argmax_found': (found / torch.from_numpy(n_top).to(dev).clamp_min(1)).cpu().numpy()})
    T['eligible'] = T.fire_frac >= MIN_RATE
    T['plain_background'] = T.active_off > PLAIN_OFF
    T['place_bound'] = (T.conc > PLACE_CONC) & (T.argmax_off <= PLACE_OFF)
    T['social'] = np.where(T.n_top == 0, 'none', np.where(T.NA_med == 0, 'off-animal', np.where((T.NA_med <= 1) & (T.E_med <= SELF_E), 'self',
                           np.where((T.NA_med >= COLL_NA) | (T.E_med >= COLL_E), 'collective', 'pair'))))
    # presence-only social level: animal count at the argmax patch (on-animal argmax patches)
    on_tot = hn[:, 1:].sum(1)
    cum = np.cumsum(hn[:, 1:], 1) / np.maximum(on_tot, 1)[:, None]
    medc = 1 + np.argmax(cum >= 0.5, 1)
    T['argmax_count_med'] = np.where(on_tot > 0, medc, 0)
    T['social_presence'] = np.where(on_tot == 0, 'off-animal', np.where(medc <= 1, 'self', np.where(medc == 2, 'pair',
                                                                                                         'collective')))
    for c in range(5):
        T[f'argmax_count_{c}'] = hn[:, c] / np.maximum(hn.sum(1), 1)
    checks = {'K': K, 'H_ref': Href, 'argmax_found_min_eligible': float(T.argmax_found[T.eligible].min()),
              'argmax_found_mean_eligible': float(T.argmax_found[T.eligible].mean())}
    return T, topi, topv, checks


def level_counts(T):
    e = T[T.eligible]
    out = {'n_latents': int(len(T)), 'n_eligible': int(len(e)),
           'n_rare': int(((T.fire_frac > 0) & ~T.eligible).sum()), 'n_dead': int((T.fire_frac == 0).sum()),
           'plain_background': int(e.plain_background.sum()), 'place_bound': int(e.place_bound.sum()),
           'social': {c: int((e.social == c).sum()) for c in ('self', 'pair', 'collective', 'off-animal')},
           'social_presence': {c: int((e.social_presence == c).sum()) for c in ('self', 'pair', 'collective', 'off-animal')}}
    ok = e[~e.plain_background]
    out['social_x_place_not_plain'] = {f'{c}|{"place" if p else "free"}': int(((ok.social == c) & (ok.place_bound == p)).sum())
                                       for c in ('self', 'pair', 'collective', 'off-animal') for p in (False, True)}
    out['median_E_by_social'] = {c: float(e.E_med[e.social == c].median()) for c in ('self', 'pair', 'collective')
                                 if (e.social == c).any()}
    return out


@torch.no_grad()
def active_patches(sae, norm, tok, pos, st, i, j, dev):
    T = torch.from_numpy(np.ascontiguousarray(tok[st[i]:st[i + 1]])).to(dev)
    if len(T) == 0:
        return np.zeros(0, int), -1
    pre = torch.relu((norm(T) - sae.b_dec) @ sae.W_enc[:, j] + sae.b_enc[j])
    z = pre * (pre > sae.threshold)
    P = pos[st[i]:st[i + 1]].astype(int)
    return P[(z > 0).cpu().numpy()], int(P[int(z.argmax())])


def draw_frame(lab, i, act, amax, th):
    from PIL import Image, ImageDraw
    im = Image.open(REPO / 'dataset' / lab.frame_path.iat[i]).convert('RGB')
    if im.size != (512, 512):
        im = im.resize((512, 512))
    d = ImageDraw.Draw(im)
    for p in act:
        r, c = divmod(int(p), GRID)
        d.rectangle([c * PX, r * PX, c * PX + PX - 1, r * PX + PX - 1], outline=(255, 220, 0), width=2)
    if amax >= 0:
        r, c = divmod(amax, GRID)
        d.rectangle([c * PX - 2, r * PX - 2, c * PX + PX + 1, r * PX + PX + 1], outline=(255, 0, 0), width=2)
    return im.resize((th, th))


def sheet(path, rows, lab, sae, norm, tok, pos, st, dev, labels_col=None, n_tiles=8, th=256):
    """rows: [(title, neuron, frames)] -> one sheet, every active patch outlined yellow, the argmax patch red."""
    from PIL import Image, ImageDraw
    h = 28
    S = Image.new('RGB', (n_tiles * th, len(rows) * (th + h)), 'white')
    dr = ImageDraw.Draw(S)
    man = []
    for r_, (title, j, frames) in enumerate(rows):
        y0 = r_ * (th + h)
        dr.text((4, y0 + 2), title, fill=(0, 0, 0))
        for c_, i in enumerate(frames[:n_tiles]):
            act, amax = active_patches(sae, norm, tok, pos, st, i, j, dev)
            S.paste(draw_frame(lab, i, act, amax, th), (c_ * th, y0 + h))
            y = bool(lab[labels_col].iat[i]) if labels_col else None
            dr.text((c_ * th + 2, y0 + 15), f'{lab.obs.iat[i]} f{lab.frame_idx.iat[i]} n_act {len(act)}'
                    + ('' if y is None else (' POS' if y else ' neg')), fill=(0, 120, 0) if y else (90, 90, 90))
            man.append({'title': title, 'neuron': int(j), 'obs': lab.obs.iat[i], 'frame_idx': int(lab.frame_idx.iat[i]),
                        'n_active': int(len(act)), 'label': y})
    S.save(path, quality=85)
    return man


def top_frames(x, rows_ok, obs, n=8, per_video=2):
    x = np.where(rows_ok, x, 0)
    pick, seen = [], {}
    for i in np.argsort(-x, kind='stable'):
        if x[i] <= 0 or len(pick) == n:
            break
        if seen.get(obs[i], 0) >= per_video:
            continue
        seen[obs[i]] = seen.get(obs[i], 0) + 1
        pick.append(int(i))
    return pick


def cmd_evaluate(args):
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out_dir) if args.out_dir else OUT / args.domain
    loc = ms.local_dir()
    lout = loc / 'out'
    for d in ('eval', 'levels', 'codes_best', 'sheets'):
        (lout / d).mkdir(parents=True, exist_ok=True)
    idx = ssp.StoreIndex(args.domain)
    _, ev, _, _, _, _ = idx.split(args.domain, ms.N_TRAIN[args.domain], 0)
    ev = np.sort(ev)
    S = np.load(OUT / args.domain / 'mask' / 'sites.npz')
    assert (S['frames'] == ev).all()
    sub = np.arange(len(ev))
    if args.max_frames:
        sub = np.linspace(0, len(ev) - 1, args.max_frames).astype(int)
        ev = ev[sub]
    site, bcount, sdark = S['site'][sub], S['bcount'][sub], S['dark'][sub]
    lab, _, valid = ms.eval_labels(args.domain, idx, ev)
    labels = sl.LABELS[args.domain]
    unit = lab.obs.str.rsplit('_', n=2).str[0].values if args.domain == 'mice' else lab.obs.values
    units = sorted(set(unit))
    half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
    assert (half == np.load(LEV / args.domain / 'half.npy')[sub]).all(), 'halves differ from sae_levers'
    KB = bcount.shape[1]
    Yt = torch.from_numpy(lab[labels].values.astype(bool)).to(dev)
    a1 = {'base': json.loads((LEV / args.domain / 'train_meta_w4096_w16384.json').read_text())['median_single_animal_area'],
          't2': json.loads((out / 'train_meta_t2.json').read_text())['median_single_animal_area']}
    log(f'{args.domain}: {len(ev):,} eval frames; a1 (patches per animal) {a1}')
    for arm in args.arms:
        tok_m, pos, lens = stage(idx, ev, loc / 'eval', arm, args.domain)
        tok = np.load(loc / 'eval' / 'tok.npy')
        del tok_m
        shutil.rmtree(loc / 'eval')
        st = np.r_[0, np.cumsum(lens)]
        rank = np.argsort(np.argsort(lens + np.random.default_rng(0).random(len(lens)) * 1e-3))
        decile = np.minimum(rank * 10 // len(lens), 9)
        # on-animal shares of the kept tokens (neuron-map tools)
        fid = np.repeat(np.arange(len(lens)), lens)
        on = site[fid, pos.astype(np.int64)] > 0
        dk = sdark[fid, pos.astype(np.int64)] > 0
        arm_meta = {'arm': arm, 'kept_per_frame': float(lens.mean()), 'frames_no_token': float((lens == 0).mean()),
                    'token_off_animal_share': float(1 - on.mean()), 'token_no_dark_pixel_share': float(1 - dk.mean()),
                    'a1': a1[arm]}
        log(f'  [{arm}] {json.dumps(arm_meta)}')
        np.save(lout / 'codes_best' / f'{arm}_lens.npy', lens.astype(np.int16))
        for seed in args.seeds:
            t0 = time.time()
            key = f'{arm}_s{seed}'
            pth = (LEV / args.domain / 'sae' / f'w4096_s{seed}' / 'sae.pt' if arm == 'base'
                   else out / 'sae' / f't2_w4096_s{seed}' / 'sae.pt')
            sae, norm, ck = load_sae(pth, dev)
            pres, ext, stt = encode_pe(sae, norm, tok, lens, dev)
            resP, _ = sl.score_model(pres, Yt, labels, valid, half, decile, dev)
            resE, _ = sl.score_model(ext, Yt, labels, valid, half, decile, dev)
            T, topi, topv, chk = level_map(sae, norm, tok, pos, lens, pres, ext, site, bcount, a1[arm], dev, KB)
            T.to_csv(lout / 'levels' / f'{key}.csv', index=False)
            e = {'arm': arm, 'seed': seed, 'ckpt': str(pth), **stt, **arm_meta, 'presence': resP, 'extent': resE,
                 'levels': level_counts(T), 'level_checks': chk}
            if arm == 'base' and not args.max_frames:  # reproduction of the levers numbers
                ref = json.loads((LEV / args.domain / 'eval' / f'w4096_s{seed}.json').read_text())['labels']
                e['repro_max_abs_diff_cf_auroc'] = max(abs(ref[b]['cf_auroc'] - resP[b]['cf_auroc']) for b in labels)
            js = sorted({d[s]['neuron'] for R in (resP, resE) for b in labels for d in R[b]['dirs'] for s in ('auc', 'ap', 'top1')})
            jt = torch.tensor(js, device=dev)
            np.savez_compressed(lout / 'codes_best' / f'{key}.npz', neurons=np.array(js),
                                presence=pres[:, jt].cpu().numpy(), extent=ext[:, jt].cpu().numpy())
            (lout / 'eval' / f'{key}.json').write_text(json.dumps(e, indent=1, default=float))
            f = lambda R, b: f"{R[b]['cf_auroc']:.3f}/{R[b]['cf_ap']:.4f}/{R[b]['cf_top1']:.3f}/{R[b]['cf_size_ctrl_auroc']:.3f}"  # noqa: E731
            log(f'  {key}: FVE {stt["fve"]:.4f} L0 {stt["l0_per_token"]:.2f} | cf AUROC/AP/top1/size-ctrl: '
                + ' | '.join(f'{b} P {f(resP, b)} E {f(resE, b)}' for b in labels)
                + (f' | repro diff {e["repro_max_abs_diff_cf_auroc"]:.2e}' if 'repro_max_abs_diff_cf_auroc' in e else '')
                + f' | levels {json.dumps(e["levels"])} | checks {json.dumps(chk)} ({time.time() - t0:.0f}s)')
            if seed == args.seeds[0]:
                man = []
                rows = []
                for b in labels:
                    for R, rn in ((resP, 'presence'), (resE, 'extent')):
                        dd = R[b]['dirs'][0]
                        j = dd['auc']['neuron']
                        x = (pres if rn == 'presence' else ext)[:, j].float().cpu().numpy()
                        fr_ = top_frames(x, valid[b] & (half == dd['test_half']), lab.obs.values)
                        rows.append((f'{key} {b}: {rn}-AUROC pick n{j} (select half {dd["select_half"]}, test AUROC '
                                     f'{dd["auc"]["test_auc"]:.3f}); top {rn} frames of the test half', j, fr_, b))
                for b in labels:
                    rr = [r for r in rows if r[3] == b]
                    man += sheet(lout / 'sheets' / f'{key}__{b}.jpg', [r[:3] for r in rr], lab, sae, norm, tok, pos, st,
                                 dev, labels_col=b)
                rng = np.random.default_rng(0)
                E_ = T[T.eligible]
                groups = [(f'social={c}', E_[(E_.social == c) & ~E_.plain_background], 'fire_frac') for c in ('self', 'pair', 'collective')]
                groups += [('place-bound', E_[E_.place_bound & ~E_.plain_background], 'conc'),
                           ('plain-background', E_[E_.plain_background], 'active_off')]
                lv = []
                for gname, G, sc in groups:
                    if not len(G):
                        continue
                    picks = [int(G.neuron.values[np.argmax(G[sc].values)])]
                    rest = [n_ for n_ in G.neuron.values if n_ != picks[0]]
                    if rest:
                        picks.append(int(rng.choice(rest)))
                    for j in picks:
                        r = T.iloc[j]
                        tf = topi[:, j][topv[:, j] > 0].cpu().numpy()
                        tf = tf[np.linspace(0, len(tf) - 1, min(8, len(tf))).round().astype(int)] if len(tf) else tf
                        lv.append((f'{key} {gname}: n{j} fire {r.fire_frac:.3f} E~ {r.E_med:.2f} NA~ {r.NA_med:.1f} NB~ '
                                   f'{r.NB_med:.1f} active-off {r.active_off:.2f} conc {r.conc:.2f} argmax-off '
                                   f'{r.argmax_off:.2f} [{r.social}{", place" if r.place_bound else ""}'
                                   f'{", PLAIN" if r.plain_background else ""}] (8 of its top-1% frames, spread by rank)',
                                   j, list(tf)))
                man += sheet(lout / 'sheets' / f'{key}__levels.jpg', lv, lab, sae, norm, tok, pos, st, dev)
                pd.DataFrame(man).to_csv(lout / 'sheets' / f'{key}_manifest.csv', index=False)
            del pres, ext, topi, topv
            torch.cuda.empty_cache()
        del tok
    lab.drop(columns=['frame_path']).to_parquet(lout / 'labels.parquet')
    np.save(lout / 'half.npy', half)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(lout, out, dirs_exist_ok=True)
    log(f'results copied to {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train', 'evaluate'])
    p.add_argument('--domain', default='mice', choices=['mice', 'ants'])
    p.add_argument('--arms', nargs='+', default=['base', 't2'])
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--max-frames', type=int, default=0, help='smoke test: train / eval frame subset')
    p.add_argument('--out-dir', default=None, help='default results/vision/eci_t2t4/<domain>')
    args = p.parse_args()
    {'train': cmd_train, 'evaluate': cmd_evaluate}[args.cmd](args)


if __name__ == '__main__':
    main()
