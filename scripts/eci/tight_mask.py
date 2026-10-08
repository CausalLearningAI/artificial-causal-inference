"""
T2 tight animal mask (src/eci/tight_mask.py): per-video pixel backgrounds, label-free parameter choice, application to
the existing foreground token stores (mice fg448 1 fps, ants antsfg 1 fps). CPU only. No behaviour label is read.

Videos: the store videos of the sae_levers split (spatial_sae_pilot.StoreIndex.split, seed 0): mice 100 train videos
(unannotated) + 144 eval videos (annotated); ants 128 train + 128 eval videos.

Steps
    background  per video: the 200 background sample frames (rows of dataset/<d>/eci/fg448/background/<obs>.npz) ->
                per-block masked per-pixel median background, noise sigma -> OUT/<d>/pixbg/<obs>.npz.
                TRAIN videos only, calibration on every 10th store frame: for every delta in 0..63 grey levels and
                X in XS, the share of kept patches among
                    far   patches > 2 patches from any dark-cue core patch (dark fraction > 0.15): surely not a dark
                          animal = the set the existing token threshold calibrates on (video_threshold far_q)
                    core  dark-cue core patches (dark fraction > 0.15): surely a dark animal
                and the kept patches per frame -> OUT/<d>/calib_parts.npz
    choose      PARAMETER RULE, fixed before the rule is applied to any eval frame (train videos only, label-free):
                    sigma_d  = median over the domain's train videos of the video noise sigma, floored at 1 grey
                               level (the quantisation step: on ants > 60% of empty-arena pixels equal their background
                               exactly, so the MAD is 0)
                    delta    = the smallest integer >= MAD_K x max(sigma_ants, sigma_mice), MAD_K = 6 (the 'mad_k' of
                               the existing foreground rule), for which some X of XS_CHOICE reaches a far-patch keep
                               rate <= FAR_RATE = 0.005 (1 - thr_q of the existing token threshold) in BOTH domains:
                               ONE value for both domains, set by the noisier one
                    X        = the smallest such value of XS_CHOICE at that delta
                -> OUT/rule.json (also hard-coded into src/eci/tight_mask.py TIGHT_RULE afterwards)
    apply       every train / eval store frame: the tight mask (packed bits), compared with the store's mask; on eval
                frames also the dark-blob site maps of scripts/eci/neuron_map.py (analysis tool for the level map)
                -> OUT/<d>/mask/{masks.npz, sites.npz, apply.json}
    sheet       visual check: old vs new mask on a few eval frames per domain -> OUT/<d>/mask/mask_sheet.jpg

Usage (scripts/eci/tight_mask.sh): python scripts/eci/tight_mask.py <step> --domains mice,ants --workers 32
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from src.eci.domain import get_domain  # noqa: E402
from src.eci.foreground import segment_of  # noqa: E402
from src.eci.tight_mask import (DARK_FRAC, GRID, PX, diff_fraction, dilate_np, pixel_background_blocks,  # noqa: E402
                                robust_sigma)

OUT = REPO / 'results/vision/eci_t2t4'
BG_OLD = {'mice': REPO / 'dataset/mice/v1/eci/fg448/background', 'ants': REPO / 'dataset/ants/eci/fg448/background'}
N_TRAIN = {'mice': 100, 'ants': 128}
XS = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
XS_CHOICE = (0.05, 0.10, 0.15, 0.20, 0.25)
NT = 64  # deltas 0..63
MAD_K, FAR_RATE = 6.0, 0.005
CALIB_EVERY = 10
KB = 32  # blob ids kept per frame in sites.npz (largest first beyond this; counted)
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def grey_of(path):
    from PIL import Image
    with Image.open(REPO / 'dataset' / path) as im:
        g = np.asarray(im.convert('RGB').convert('L'), dtype=np.uint8)
    if g.shape != (512, 512):
        raise ValueError(f'{path}: shape {g.shape}')
    return g


def split_frames(domain):
    import spatial_sae_pilot as ssp
    idx = ssp.StoreIndex(domain)
    tr, ev, _, _, train_v, eval_v = idx.split(domain, N_TRAIN[domain], 0)
    return idx, np.sort(tr), np.sort(ev), train_v, eval_v


def frame_paths(domain):
    return pd.read_csv(get_domain(domain).ann_path, usecols=['frame_path'])['frame_path'].values


def patch_counts_gt(grey, bg):
    """(B, 1024, NT) int16: pixels of each patch with |grey - bg| > t, t = 0..NT-1."""
    d = np.minimum(np.abs(grey.astype(np.int16) - bg.astype(np.int16)), NT).astype(np.int64)
    B = len(d)
    d = d.reshape(B, GRID, PX, GRID, PX).transpose(0, 1, 3, 2, 4).reshape(B * GRID * GRID, PX * PX)
    key = np.arange(B * GRID * GRID)[:, None] * (NT + 1) + d
    h = np.bincount(key.ravel(), minlength=B * GRID * GRID * (NT + 1)).reshape(B, GRID * GRID, NT + 1)
    ge = np.cumsum(h[..., ::-1], axis=-1)[..., ::-1]  # ge[..., t] = pixels with d >= t
    return ge[..., 1:NT + 1].astype(np.int16)        # [..., t] = d >= t + 1 = d > t


def dark_frac(grey, pix_bg, dark_abs=60, dark_rel=40):
    g = grey.astype(np.int16)
    d = (g < dark_abs) & ((pix_bg.astype(np.int16)[None] - g) > dark_rel)
    return d.reshape(len(g), GRID, PX, GRID, PX).mean((2, 4)).reshape(len(g), -1)


# ---------------------------------------------------------------------------------------------- background
def _bg_video(job):
    domain, obs, sample_paths, calib_rows, calib_paths = job
    z = np.load(BG_OLD[domain] / f'{obs}.npz')
    grey = np.stack([grey_of(p) for p in sample_paths])
    bg, src, n_clean, excl = pixel_background_blocks(grey, z['dark'], z['seg_bounds'], z['pix_bg'])
    sigma, _ = robust_sigma(grey, bg, excl, z['seg_bounds'])
    od = OUT / domain / 'pixbg'
    np.savez_compressed(od / f'{obs}.npz', bg=bg, src=src, n_clean=n_clean, sigma=sigma, rows=z['rows'],
                        seg_bounds=z['seg_bounds'])
    res = {'obs': obs, 'sigma': sigma, 'src_frac': [float((src == k).mean()) for k in range(3)]}
    if len(calib_rows):
        seg_all = segment_of(calib_rows, z['rows'], z['seg_bounds'])
        acc = {'far_n': 0, 'core_n': 0, 'far_kept': 0, 'core_kept': 0, 'kept': 0}
        for a in range(0, len(calib_rows), 32):
            g = np.stack([grey_of(p) for p in calib_paths[a:a + 32]])
            fr = patch_counts_gt(g, bg[seg_all[a:a + 32]]).astype(np.float32) / (PX * PX)  # (B, 1024, NT)
            core = dark_frac(g, z['pix_bg']) > DARK_FRAC
            far = ~dilate_np(core, 2)
            keep = fr[..., None] >= XS[None, None, None, :]  # (B, 1024, NT, nX)
            acc['far_n'] += int(far.sum())
            acc['core_n'] += int(core.sum())
            acc['far_kept'] = acc['far_kept'] + keep[far].sum(0)
            acc['core_kept'] = acc['core_kept'] + keep[core].sum(0)
            acc['kept'] = acc['kept'] + keep.sum((0, 1))
            acc['core_total'] = acc.get('core_total', 0) + int(core.sum())
        res.update(n_frames=len(calib_rows), old_dark_core_per_frame=acc.pop('core_total') / len(calib_rows), **acc)
    return res


def cmd_background(args):
    from multiprocessing import get_context
    t0 = time.time()
    for domain in args.domains.split(','):
        (OUT / domain / 'pixbg').mkdir(parents=True, exist_ok=True)
        idx, tr, ev, train_v, eval_v = split_frames(domain)
        paths = frame_paths(domain)
        jobs = []
        for obs in sorted(set(train_v) | set(eval_v)):
            z = np.load(BG_OLD[domain] / f'{obs}.npz')
            crow = np.zeros(0, np.int64)
            if obs in set(train_v):
                f = tr[idx.obs[tr] == obs]
                crow = np.sort(idx.rows[f])[::CALIB_EVERY]
            jobs.append((domain, obs, paths[z['rows']], crow, paths[crow]))
        if args.max_videos:  # smoke test: a few train + eval videos
            jobs = [j for j in jobs if len(j[3])][:args.max_videos] + [j for j in jobs if not len(j[3])][:args.max_videos]
        log(f'{domain}: {len(jobs)} videos ({len(train_v)} train with calibration frames) ({time.time() - t0:.0f}s)')
        R = []
        with get_context('fork').Pool(args.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(_bg_video, jobs)):
                R.append(r)
                if k % 25 == 0:
                    log(f'  {domain} {k + 1}/{len(jobs)} videos ({time.time() - t0:.0f}s)')
        cal = [r for r in R if 'far_kept' in r]
        sig = pd.DataFrame([{'obs': r['obs'], 'sigma': r['sigma'], 'train': r['obs'] in set(train_v),
                             'src_block': r['src_frac'][0], 'src_video': r['src_frac'][1], 'src_pixbg': r['src_frac'][2]}
                            for r in R])
        sig.to_csv(OUT / domain / 'sigma.csv', index=False)
        np.savez_compressed(OUT / domain / 'calib_parts.npz', obs=np.array([r['obs'] for r in cal]),
                            sigma=np.array([r['sigma'] for r in cal]), n_frames=np.array([r['n_frames'] for r in cal]),
                            far_n=np.array([r['far_n'] for r in cal]), core_n=np.array([r['core_n'] for r in cal]),
                            far_kept=np.stack([r['far_kept'] for r in cal]),
                            core_kept=np.stack([r['core_kept'] for r in cal]), kept=np.stack([r['kept'] for r in cal]),
                            old_dark_core_per_frame=np.array([r['old_dark_core_per_frame'] for r in cal]), XS=XS)
        log(f'{domain}: sigma (train videos) median {sig[sig.train].sigma.median():.2f} '
            f'[{sig[sig.train].sigma.quantile(0.05):.2f}, {sig[sig.train].sigma.quantile(0.95):.2f}] grey levels; '
            f'background source block/video/pix_bg {sig[["src_block", "src_video", "src_pixbg"]].mean().round(4).to_dict()}'
            f' ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- choose
def calib_table(domain):
    z = np.load(OUT / domain / 'calib_parts.npz')
    far = z['far_kept'].sum(0) / z['far_n'].sum()      # (NT, nX)
    core = z['core_kept'].sum(0) / z['core_n'].sum()
    per_frame = z['kept'].sum(0) / z['n_frames'].sum()
    return {'sigma_median': float(np.median(z['sigma'])), 'far': far, 'core': core, 'kept_per_frame': per_frame,
            'n_videos': int(len(z['obs'])), 'n_frames': int(z['n_frames'].sum()),
            'old_dark_core_per_frame': float((z['old_dark_core_per_frame'] * z['n_frames']).sum() / z['n_frames'].sum())}


def cmd_choose(args):
    T = {d: calib_table(d) for d in ('mice', 'ants')}
    smax = max(max(T[d]['sigma_median'], 1.0) for d in T)
    d0 = int(np.ceil(MAD_K * smax - 1e-9))
    xi = [i for i, x in enumerate(XS) if x in XS_CHOICE]
    delta = i = None
    for t in range(d0, NT):
        ok = [k for k in xi if all(T[d]['far'][t, k] <= FAR_RATE for d in T)]
        if ok:
            delta, i = t, ok[0]
            break
    if delta is None:
        raise SystemExit(f'no (delta >= {d0}, X in {XS_CHOICE}) reaches far rate <= {FAR_RATE}')
    rule = {'delta': delta, 'frac': float(XS[i]), 'mad_k': MAD_K, 'far_rate_max': FAR_RATE, 'delta_min': d0,
            'sigma_median': {d: T[d]['sigma_median'] for d in T},
            'at_choice': {d: {'far_keep_rate': float(T[d]['far'][delta, i]), 'dark_core_keep_rate': float(T[d]['core'][delta, i]),
                              'kept_per_frame': float(T[d]['kept_per_frame'][delta, i]),
                              'old_dark_core_patches_per_frame': T[d]['old_dark_core_per_frame'],
                              'n_train_videos': T[d]['n_videos'], 'n_calib_frames': T[d]['n_frames']} for d in T},
            'grid_at_delta': {d: {f'X={x:.2f}': {'far': round(float(T[d]['far'][delta, k]), 5),
                                                 'core': round(float(T[d]['core'][delta, k]), 4),
                                                 'kept_per_frame': round(float(T[d]['kept_per_frame'][delta, k]), 1)}
                                  for k, x in enumerate(XS)} for d in T},
            'grid_far_X': {d: {f'delta={t}': [round(float(v), 5) for v in T[d]['far'][t]] for t in range(4, NT, 2)}
                           for d in T},
            'XS': XS.tolist()}
    (OUT / 'rule.json').write_text(json.dumps(rule, indent=1))
    log(json.dumps({k: rule[k] for k in ('delta', 'frac', 'sigma_median', 'at_choice')}, indent=1))
    for d in T:
        log(f'{d} at delta {delta}: ' + json.dumps(rule['grid_at_delta'][d]))


# ---------------------------------------------------------------------------------------------- apply
def _apply_video(job):
    import neuron_map as nm
    domain, obs, rows, paths, is_eval, delta, frac = job
    pb = np.load(OUT / domain / 'pixbg' / f'{obs}.npz')
    pix_bg = np.load(BG_OLD[domain] / f'{obs}.npz')['pix_bg']
    seg = segment_of(rows, pb['rows'], pb['seg_bounds'])
    bits = np.zeros((len(rows), GRID * GRID // 8), np.uint8)
    site = np.zeros((int(is_eval.sum()), GRID * GRID), np.int8)
    bcount = np.zeros((int(is_eval.sum()), KB), np.int8)
    nblob = np.zeros(int(is_eval.sum()), np.int16)
    dcnt = np.zeros((int(is_eval.sum()), GRID * GRID), np.uint8)
    e = 0
    for a in range(0, len(rows), 64):
        g = np.stack([grey_of(p) for p in paths[a:a + 64]])
        m = diff_fraction(g, pb['bg'][seg[a:a + 64]], delta) >= frac
        bits[a:a + 64] = np.packbits(m, axis=1)
        for k in np.flatnonzero(is_eval[a:a + 64]):
            dc, si, ar = nm.site_frame(g[k], pix_bg, nm.BLOB[domain])
            n = len(ar)
            nblob[e] = n
            if n > KB:
                si = np.where(si > KB, 0, si)
            site[e], dcnt[e] = si, dc
            c = np.clip(np.round(ar[:KB] / nm.A1[domain]), 1, 4).astype(np.int8)
            bcount[e, :len(c)] = c
            e += 1
    return obs, bits, site, bcount, nblob, dcnt


def old_masks(idx, frames):
    """(F, 1024) bool store mask of store frames (sorted)."""
    out = np.zeros((len(frames), GRID * GRID), bool)
    for s in np.unique(idx.shard[frames]):
        k = np.flatnonzero(idx.shard[frames] == s)
        pos = idx.store.pos(s)
        st, n = idx.start[frames[k]], idx.nfg[frames[k]]
        for i, a, b in zip(k, st, n):
            out[i, pos[a:a + b]] = True
    return out


def cmd_apply(args):
    from multiprocessing import get_context
    rule = json.loads((OUT / 'rule.json').read_text())
    delta, frac = rule['delta'], rule['frac']
    t0 = time.time()
    for domain in args.domains.split(','):
        od = OUT / domain / 'mask'
        od.mkdir(parents=True, exist_ok=True)
        idx, tr, ev, train_v, eval_v = split_frames(domain)
        frames = np.union1d(tr, ev)
        is_ev = np.isin(frames, ev)
        paths = frame_paths(domain)
        obs_f = idx.obs[frames]
        groups = pd.Series(np.arange(len(frames))).groupby(obs_f).indices
        jobs = [(domain, o, idx.rows[frames[ix]], paths[idx.rows[frames[ix]]], is_ev[ix], delta, frac)
                for o, ix in groups.items()]
        log(f'{domain}: {len(frames):,} frames ({len(tr):,} train, {len(ev):,} eval), {len(jobs)} videos, '
            f'delta {delta} X {frac}')
        bits = np.zeros((len(frames), 128), np.uint8)
        ev_pos = np.full(len(frames), -1, np.int64)
        ev_pos[is_ev] = np.arange(is_ev.sum())
        site = np.zeros((len(ev), GRID * GRID), np.int8)
        bcount = np.zeros((len(ev), KB), np.int8)
        nblob = np.zeros(len(ev), np.int16)
        dcnt = np.zeros((len(ev), GRID * GRID), np.uint8)
        with get_context('fork').Pool(args.workers) as pool:
            for k, (o, b, si, bc, nb, dc) in enumerate(pool.imap_unordered(_apply_video, jobs)):
                ix = groups[o]
                bits[ix] = b
                ie = ev_pos[ix[is_ev[ix]]]
                site[ie], bcount[ie], nblob[ie], dcnt[ie] = si, bc, nb, dc
                if k % 25 == 0:
                    log(f'  {domain} {k + 1}/{len(jobs)} videos ({time.time() - t0:.0f}s)')
        new = np.unpackbits(bits, axis=1).astype(bool)
        old = old_masks(idx, frames)
        n_old, n_new = old.sum(1), new.sum(1)
        both = (old & new).sum(1)
        new_only = (new & ~old).sum(1)
        np.savez_compressed(od / 'masks.npz', frames=frames, bits=bits, rows=idx.rows[frames], is_eval=is_ev,
                            n_old=n_old.astype(np.int16), n_new=n_new.astype(np.int16), new_only=new_only.astype(np.int16))
        np.savez_compressed(od / 'sites.npz', frames=ev, site=site, bcount=bcount, n_blobs=nblob, dark=dcnt)
        # on-animal shares of the kept patches (eval frames, analysis tools of the neuron map)
        oe, ne = old[is_ev], new[is_ev]
        on = site > 0
        hasdark = dcnt > 0

        def share(m, q):
            return float(q[m].mean()) if m.any() else float('nan')
        st = {'domain': domain, 'rule': {'delta': delta, 'frac': frac}, 'n_frames': int(len(frames)),
              'n_train_frames': int(len(tr)), 'n_eval_frames': int(len(ev)),
              'kept_per_frame_old': {'mean': float(n_old.mean()), 'median': float(np.median(n_old))},
              'kept_per_frame_new': {'mean': float(n_new.mean()), 'median': float(np.median(n_new))},
              'kept_per_frame_old_train': float(n_old[~is_ev].mean()), 'kept_per_frame_new_train': float(n_new[~is_ev].mean()),
              'kept_per_frame_old_eval': float(n_old[is_ev].mean()), 'kept_per_frame_new_eval': float(n_new[is_ev].mean()),
              'frac_old_dropped': float(1 - both.sum() / n_old.sum()),
              'frac_new_absent_from_store': float(new_only.sum() / max(n_new.sum(), 1)),
              'frames_with_no_new_patch': float((n_new == 0).mean()),
              'frames_with_no_old_patch': float((n_old == 0).mean()),
              'eval_off_animal_share': {'old': share(oe, ~on), 'new': share(ne, ~on), 'new_only': share(ne & ~oe, ~on),
                                        'old_dropped': share(oe & ~ne, ~on)},
              'eval_no_dark_pixel_share': {'old': share(oe, ~hasdark), 'new': share(ne, ~hasdark),
                                           'old_dropped': share(oe & ~ne, ~hasdark)},
              'eval_on_animal_patches_kept': {'old': share(on, oe), 'new': share(on, ne)},
              'eval_frames_more_than_KB_blobs': int((nblob > KB).sum()),
              'elapsed_s': round(time.time() - t0, 1)}
        (od / 'apply.json').write_text(json.dumps(st, indent=1))
        log(f'{domain}: ' + json.dumps(st))


# ---------------------------------------------------------------------------------------------- sheet
def cmd_sheet(args):
    from PIL import Image, ImageDraw
    for domain in args.domains.split(','):
        od = OUT / domain / 'mask'
        z = np.load(od / 'masks.npz')
        paths = frame_paths(domain)
        idx, _, _, _, _ = split_frames(domain)
        ev = np.flatnonzero(z['is_eval'])
        pick = ev[np.linspace(0, len(ev) - 1, 12).round().astype(int)]
        old = old_masks(idx, z['frames'][pick])
        new = np.unpackbits(z['bits'][pick], axis=1).astype(bool)
        th = 384
        sheet = Image.new('RGB', (4 * th, 3 * (th + 16)), 'white')
        dr = ImageDraw.Draw(sheet)
        for q, i in enumerate(pick):
            im = Image.open(REPO / 'dataset' / paths[z['rows'][i]]).convert('RGB')
            d2 = ImageDraw.Draw(im)
            for p in range(GRID * GRID):
                r, c = divmod(p, GRID)
                box = [c * PX, r * PX, c * PX + PX - 1, r * PX + PX - 1]
                if old[q, p] and not new[q, p]:
                    d2.rectangle(box, outline=(230, 30, 30), width=1)
                elif new[q, p] and not old[q, p]:
                    d2.rectangle(box, outline=(30, 120, 255), width=1)
                elif new[q, p]:
                    d2.rectangle([box[0] + 3, box[1] + 3, box[2] - 3, box[3] - 3], outline=(0, 200, 0), width=1)
            rr, cc = divmod(q, 4)
            sheet.paste(im.resize((th, th)), (cc * th, rr * (th + 16)))
            dr.text((cc * th + 2, rr * (th + 16) + th + 2),
                    f'{idx.obs[z["frames"][i]]} old {old[q].sum()} new {new[q].sum()} (red: dropped, blue: added, green: both)',
                    fill='black')
        sheet.save(od / 'mask_sheet.jpg', quality=88)
        log(f'{domain}: {od / "mask_sheet.jpg"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['background', 'choose', 'apply', 'sheet'])
    p.add_argument('--domains', default='mice,ants')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--max-videos', type=int, default=0, help='background smoke test')
    a = p.parse_args()
    {'background': cmd_background, 'choose': cmd_choose, 'apply': cmd_apply, 'sheet': cmd_sheet}[a.cmd](a)


if __name__ == '__main__':
    main()
