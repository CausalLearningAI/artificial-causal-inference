"""
COMBINED test: the final ECI behaviour-representation pipeline (ants + mice).

Pipeline: frames -> frozen DINOv2-base patch tokens (448 px) -> animal (foreground) mask -> Matryoshka BatchTopK SAE per
patch (4096 latents, k = 16) -> frame read-out = max over kept patches -> video summaries -> NES.

Arms (identical frames, splits, recipe as levers w4096 / motion_t3; 3 seeds each)
    C0  BASE: levers w4096 on today's mask, static tokens. Checkpoints results/vision/eci_sae_levers/<d>/sae/w4096_s*,
        evaluation REUSED from results/vision/eci_t2t4/<d> (arm 'base': presence / extent / level map, = levers numbers).
    C1  today's mask + M5 motion ([token_t, token_t - token_{t-5}], TokenNorm.fit_blocks([768, 768])). Checkpoints
        results/vision/eci_t3_motion/<d>/sae/M5_s*, RE-EVALUATED here with the presence / extent / level-map harness
        (its cf AUROC is checked against the T3 evaluation).
    C2  the T2b mask + M5 motion. Trained here with motion_t3.train_one (same recipe, cap mice 4M / ants 2M, pool seed 0).

T2b mask = T2 (src/eci/tight_mask.py: keep patch iff >= X of its 16 x 16 pixels differ from the per-time-block median
    background image by more than delta grey levels, no dilation) with a NOISE-SCALED threshold PER DOMAIN, one rule:
    sigma_d  = median over the domain's TRAIN videos of 1.4826 x MAD of the empty-arena pixels around their background
               (results/vision/eci_t2t4/<d>/calib_parts.npz, scripts/eci/tight_mask.py background), floored at 1 grey level
    delta_d  = ceil(6 x sigma_d); X_d = the smallest of {5, 10, 15, 20, 25}% keeping <= 0.5% of empty-floor patches
               (patches > 2 patches from any dark-cue core patch, train calibration frames); if none qualifies, delta_d
               += 2 until one does. Fixed on TRAIN videos before any evaluation -> OUT/rule.json.
    Token_t of kept patches absent from the base store and token_{t-5} of kept patches without a d5-store counterpart
    are encoded with the store's exact encoder path (FgEncoder('dinov2_base'), FrameDatasetFG, encode_batch; A40 for
    bit identity, checked against the store / the d5 prev tokens). Mice: the T2 'extra' tokens (scripts/eci/
    tight_mask_encode.py) are reused for token_t when the mice T2b rule equals the T2 rule.

Evaluation (same harness as levers / T2 / T3): sae_levers.score_model on the presence (max) and extent read-outs, the
    level map of scripts/eci/tight_mask_sae.py (T2's site maps, mask independent), video level with
    scripts/eci/readouts.py (mice: diag_video_level measures incl. pooled d-r and per-switch dz / signs; ants: video r),
    raw and with the foreground-count control (each arm's own mask).

PRE-REGISTERED DECISION (fixed before any C2 result; interpretation choices marked *)
    An arm is a WIN over C0 iff in BOTH domains
      (i)  no label's frame-level cross-fitted AUROC (presence, 3-seed mean) is below C0's by more than 0.02, AND
      (ii) at least one label improves: frame level (cf AUROC 3-seed mean >= C0 + 0.03 and above C0's best seed, OR
           cf AP (best-AP pick) 3-seed mean >= 1.3 x C0 and above C0's best seed) OR video level (count-controlled
           presence-mean r, 3-seed mean, >= C0 + 0.10; *with the annotated rate OR with the annotated bouts/min).
    If both C1 and C2 win: choose the one with the larger *mean count-controlled presence-mean video r (over both
    domains' labels and both truths, rate and bouts/min), provided its *frame-level mean cf AUROC (over both domains'
    labels) is within 0.01 of the other's; otherwise the better frame-level one. Neither wins -> C0 stays.

Steps (scripts/eci/combined.sh)
    choose    (login safe, reads a small npz) -> OUT/rule.json
    apply     (CPU) T2b masks of all train + eval store frames + the encode plan + mask statistics, on-lid coverage,
              visual check sheets -> OUT/<d>/mask/
    encode    (GPU, A40) tokens to encode -> OUT/<d>/extra/part_<task>/
    train     (GPU) C2 x seeds -> OUT/<d>/sae/C2_s<seed>/, OUT/<d>/train_meta_C2.json
    evaluate  (GPU) C1 / C2 x seeds -> OUT/<d>/{eval,levels,codes_best,sheets}/
    readouts  (CPU) video level, tables, decision -> OUT/summary.json, OUT/tables.md
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import tight_mask as tm  # noqa: E402  (scripts/eci/tight_mask.py)
from src.eci.foreground import segment_of  # noqa: E402
from src.eci.tight_mask import GRID, PX, diff_fraction  # noqa: E402

OUT = REPO / 'results/vision/eci_combined'
T2 = REPO / 'results/vision/eci_t2t4'
T3 = REPO / 'results/vision/eci_t3_motion'
LEV = REPO / 'results/vision/eci_sae_levers'
D = 768
DOMAINS = ('mice', 'ants')
SEEDS = (0, 1, 2)
MAD_K, FAR_RATE, XS_CHOICE, SIGMA_FLOOR, STEP = 6.0, 0.005, (0.05, 0.10, 0.15, 0.20, 0.25), 1.0, 2
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def popcount(bits):
    return np.unpackbits(bits, axis=1).sum(1).astype(np.int64)


# ---------------------------------------------------------------------------------------------- choose
def cmd_choose(args):
    t2rule = json.loads((T2 / 'rule.json').read_text())
    R = {'mad_k': MAD_K, 'far_rate_max': FAR_RATE, 'xs_choice': list(XS_CHOICE), 'sigma_floor': SIGMA_FLOOR,
         'delta_step': STEP, 'source': str((T2 / '<d>/calib_parts.npz').relative_to(REPO)), 't2_rule': t2rule}
    for d in DOMAINS:
        T = tm.calib_table(d)
        XS = list(np.load(T2 / d / 'calib_parts.npz')['XS'])
        xi = [XS.index(x) for x in XS_CHOICE]
        sig = T['sigma_median']
        s_used = max(sig, SIGMA_FLOOR)
        d0 = int(math.ceil(MAD_K * s_used - 1e-9))
        delta = i = None
        tried = []
        for t in range(d0, tm.NT, STEP):
            ok = [k for k in xi if T['far'][t, k] <= FAR_RATE]
            tried.append({'delta': t, 'far_at_X': {f'{XS[k]:.2f}': round(float(T['far'][t, k]), 5) for k in xi}})
            if ok:
                delta, i = t, ok[0]
                break
        if delta is None:
            raise SystemExit(f'{d}: no delta in [{d0}, {tm.NT}) reaches far rate <= {FAR_RATE}')
        R[d] = {'sigma_median': sig, 'sigma_used': s_used, 'delta_min': d0, 'delta': int(delta), 'frac': float(XS[i]),
                'far_keep_rate': float(T['far'][delta, i]), 'dark_core_keep_rate': float(T['core'][delta, i]),
                'kept_per_frame_train_calib': float(T['kept_per_frame'][delta, i]),
                'old_dark_core_patches_per_frame': T['old_dark_core_per_frame'],
                'n_train_videos': T['n_videos'], 'n_calib_frames': T['n_frames'], 'search': tried,
                'equals_t2_rule': bool(delta == t2rule['delta'] and abs(XS[i] - t2rule['frac']) < 1e-9),
                't2_at_its_rule': {'far_keep_rate': t2rule['at_choice'][d]['far_keep_rate'],
                                   'dark_core_keep_rate': t2rule['at_choice'][d]['dark_core_keep_rate'],
                                   'kept_per_frame': t2rule['at_choice'][d]['kept_per_frame']}}
        log(f'{d}: sigma {sig:.3f} (used {s_used:.2f}) -> delta {delta}, X {XS[i]:.2f}; far {T["far"][delta, i]:.5f}, '
            f'core {T["core"][delta, i]:.4f}, kept/frame {T["kept_per_frame"][delta, i]:.1f}; '
            f'same as T2: {R[d]["equals_t2_rule"]}')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'rule.json').write_text(json.dumps(R, indent=1))


def rule_of(d):
    R = json.loads((OUT / 'rule.json').read_text())
    return R[d]


# ---------------------------------------------------------------------------------------------- store masks
def store_masks(store, starts, nfg, f0, frames, shard_of):
    """(F, 1024) bool kept positions of `frames` (sorted store frame indices) in a store layout
    (per shard: start / n per frame within the shard, f0 = first store frame of every shard)."""
    out = np.zeros((len(frames), GRID * GRID), bool)
    for s in np.unique(shard_of[frames]):
        k = np.flatnonzero(shard_of[frames] == s)
        pos = store.pos(s)
        lf = frames[k] - f0[s]
        for i, a, b in zip(k, starts[s][lf], nfg[s][lf]):
            out[i, pos[a:a + b]] = True
    return out


def d5_layout(idx, domain):
    import motion_t3 as mt
    m = mt.D5Matcher(idx, domain)
    return m


def base_masks(idx, frames):
    return tm.old_masks(idx, frames)


def d5_masks(m, idx, frames):
    return store_masks(m.d5, m.d5_start, m.d5_n, m.f0, frames, idx.shard)


# ---------------------------------------------------------------------------------------------- apply
def _mask_video(job):
    domain, obs, rows, paths, delta, frac = job
    pb = np.load(T2 / domain / 'pixbg' / f'{obs}.npz')
    seg = segment_of(rows, pb['rows'], pb['seg_bounds'])
    bits = np.zeros((len(rows), GRID * GRID // 8), np.uint8)
    for a in range(0, len(rows), 64):
        g = np.stack([tm.grey_of(p) for p in paths[a:a + 64]])
        m = diff_fraction(g, pb['bg'][seg[a:a + 64]], delta) >= frac
        bits[a:a + 64] = np.packbits(m, axis=1)
    return obs, bits


def cmd_apply(args):
    from multiprocessing import get_context
    for domain in args.domains.split(','):
        t0 = time.time()
        od = OUT / domain / 'mask'
        od.mkdir(parents=True, exist_ok=True)
        r = rule_of(domain)
        idx, tr, ev, train_v, eval_v = tm.split_frames(domain)
        frames = np.union1d(tr, ev)
        is_ev = np.isin(frames, ev)
        Z2 = np.load(T2 / domain / 'mask' / 'masks.npz')
        assert np.array_equal(Z2['frames'], frames), 'T2 masks.npz frames differ'
        if r['equals_t2_rule']:
            bits = Z2['bits'].copy()
            src = 'T2 masks.npz (identical rule)'
        else:
            paths = tm.frame_paths(domain)
            groups = pd.Series(np.arange(len(frames))).groupby(idx.obs[frames]).indices
            jobs = [(domain, o, idx.rows[frames[ix]], paths[idx.rows[frames[ix]]], r['delta'], r['frac'])
                    for o, ix in groups.items()]
            log(f'{domain}: applying delta {r["delta"]} X {r["frac"]} to {len(frames):,} frames of {len(jobs)} videos')
            bits = np.zeros((len(frames), GRID * GRID // 8), np.uint8)
            with get_context('fork').Pool(args.workers) as pool:
                for k, (o, b) in enumerate(pool.imap_unordered(_mask_video, jobs)):
                    bits[groups[o]] = b
                    if k % 25 == 0:
                        log(f'  {domain} {k + 1}/{len(jobs)} videos ({time.time() - t0:.0f}s)')
            src = 'computed'
        new = np.unpackbits(bits, axis=1).astype(bool)
        old = base_masks(idx, frames)
        t2m = np.unpackbits(Z2['bits'], axis=1).astype(bool)
        m5 = d5_layout(idx, domain)
        d5 = d5_masks(m5, idx, frames)
        need_t = new & ~old
        need_5 = new & ~d5
        n_new, n_old, n_t2 = new.sum(1), old.sum(1), t2m.sum(1)
        np.savez_compressed(od / 'masks.npz', frames=frames, bits=bits, rows=idx.rows[frames], is_eval=is_ev,
                            n_old=n_old.astype(np.int16), n_new=n_new.astype(np.int16),
                            new_only=need_t.sum(1).astype(np.int16), no_d5=need_5.sum(1).astype(np.int16),
                            needt_bits=np.packbits(need_t, axis=1), need5_bits=np.packbits(need_5, axis=1))
        # on-animal shares on eval frames (T2's dark-blob site maps, mask independent)
        S = np.load(T2 / domain / 'mask' / 'sites.npz')
        assert np.array_equal(S['frames'], frames[is_ev])
        on = S['site'] > 0
        hasdark = S['dark'] > 0
        sh = lambda m, q: float(q[m].mean()) if m.any() else float('nan')  # noqa: E731
        oe, te, ne = old[is_ev], t2m[is_ev], new[is_ev]
        st = {'domain': domain, 'rule': {'delta': r['delta'], 'frac': r['frac']}, 'mask_source': src,
              'n_frames': int(len(frames)), 'n_train_frames': int(len(tr)), 'n_eval_frames': int(len(ev)),
              'kept_per_frame': {k: {'all': float(v.mean()), 'train': float(v[~is_ev].mean()), 'eval': float(v[is_ev].mean()),
                                     'median': float(np.median(v))} for k, v in (('today', n_old), ('t2', n_t2), ('t2b', n_new))},
              'frames_with_no_t2b_patch': float((n_new == 0).mean()),
              'share_t2b_absent_from_store': float(need_t.sum() / max(n_new.sum(), 1)),
              'share_t2b_without_d5': float(need_5.sum() / max(n_new.sum(), 1)),
              'frames_needing_token_t': int((need_t.sum(1) > 0).sum()), 'tokens_token_t': int(need_t.sum()),
              'frames_needing_token_t5': int((need_5.sum(1) > 0).sum()), 'tokens_token_t5': int(need_5.sum()),
              'share_today_dropped_by_t2b': float(1 - (old & new).sum() / old.sum()),
              'jaccard_t2b_vs_t2': float((new & t2m).sum() / max((new | t2m).sum(), 1)),
              'eval_off_animal_share_of_kept': {'today': sh(oe, ~on), 't2': sh(te, ~on), 't2b': sh(ne, ~on)},
              'eval_no_dark_pixel_share_of_kept': {'today': sh(oe, ~hasdark), 't2': sh(te, ~hasdark), 't2b': sh(ne, ~hasdark)},
              'eval_on_animal_patches_kept': {'today': sh(on, oe), 't2': sh(on, te), 't2b': sh(on, ne)},
              'eval_dark_pixel_patches_kept': {'today': sh(hasdark, oe), 't2': sh(hasdark, te), 't2b': sh(hasdark, ne)},
              'elapsed_s': round(time.time() - t0, 1)}
        if domain == 'ants':
            st['onlid'] = onlid_coverage(idx, ev, frames, old, t2m, new)
        (od / 'apply.json').write_text(json.dumps(st, indent=1))
        log(f'{domain}: ' + json.dumps({k: v for k, v in st.items() if k != 'onlid'}))
        if 'onlid' in st:
            log(f'{domain} on-lid: ' + json.dumps({k: v for k, v in st['onlid'].items() if k != 'onlid_frames_eval_idx'}))
        mask_sheets(domain, idx, frames, is_ev, old, t2m, new)


def onlid_coverage(idx, ev, frames, old, t2m, new):
    """Ants v3: is the yellow-marked ant covered (any kept patch within 1 patch of its tracked body centroid), on-lid vs
    not (as scripts/eci/tight_mask.py onlid), for today / T2 / T2b; plus kept patches per frame."""
    import multiscale_sae as ms
    lab = pd.read_parquet(T2 / 'ants' / 'labels.parquet')
    k = np.searchsorted(frames, ev)
    A, _ = ms.ants_anchors(lab)
    yx, yy = A[:, 2], A[:, 3]
    has = np.isfinite(yx) & np.isfinite(yy) & (lab.experiment == 'v3').values
    on = lab.onlid_yellow.values.astype(bool)
    r = np.clip(np.nan_to_num(np.floor(yy), nan=0).astype(int), 0, GRID - 1)
    c = np.clip(np.nan_to_num(np.floor(yx), nan=0).astype(int), 0, GRID - 1)

    def cover(m):
        m = m.reshape(-1, GRID, GRID)
        o = np.zeros(len(m), bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                o |= m[np.arange(len(m)), np.clip(r + dr, 0, GRID - 1), np.clip(c + dc, 0, GRID - 1)]
        return o
    res = {'n_v3_frames_with_yellow_anchor': int(has.sum()), 'n_onlid_yellow': int((has & on).sum())}
    cv = {n: cover(m[k]) for n, m in (('today', old), ('t2', t2m), ('t2b', new))}
    for nm_, sel in (('onlid', has & on), ('not_onlid', has & ~on)):
        res[nm_] = {f'yellow_covered_{n}': float(v[sel].mean()) for n, v in cv.items()}
        res[nm_].update({f'kept_per_frame_{n}': float(m[k][sel].sum(1).mean()) for n, m in
                         (('today', old), ('t2', t2m), ('t2b', new))})
    res['onlid_frames_eval_idx'] = np.flatnonzero(has & on).tolist()[:4000]
    return res


def draw_masks(im, a, b, colours=((230, 30, 30), (30, 120, 255), (0, 200, 0))):
    """a = reference mask, b = new mask: red a-only, blue b-only, green both (inner box)."""
    from PIL import ImageDraw
    d2 = ImageDraw.Draw(im)
    for p in range(GRID * GRID):
        r, c = divmod(p, GRID)
        box = [c * PX, r * PX, c * PX + PX - 1, r * PX + PX - 1]
        if a[p] and not b[p]:
            d2.rectangle(box, outline=colours[0], width=1)
        elif b[p] and not a[p]:
            d2.rectangle(box, outline=colours[1], width=1)
        elif b[p]:
            d2.rectangle([box[0] + 3, box[1] + 3, box[2] - 3, box[3] - 3], outline=colours[2], width=1)
    return im


def mask_sheets(domain, idx, frames, is_ev, old, t2m, new):
    """Visual check at full 512 px: (1) today (ref) vs T2b on 6 spread eval frames, (2) T2 (ref) vs T2b on the same
    frames, (3) ants: 6 v3 on-lid frames, today vs T2b and T2 vs T2b side by side."""
    from PIL import Image, ImageDraw
    od = OUT / domain / 'mask'
    paths = tm.frame_paths(domain)
    ev = np.flatnonzero(is_ev)
    pick = ev[np.linspace(0, len(ev) - 1, 6).round().astype(int)]
    sets = [('spread', pick)]
    if domain == 'ants':
        js = json.loads((od / 'apply.json').read_text())['onlid']['onlid_frames_eval_idx']
        if js:
            sel = np.asarray(js)[np.linspace(0, len(js) - 1, 6).round().astype(int)]
            sets.append(('onlid', ev[sel]))
    for name, fr in sets:
        S = Image.new('RGB', (2 * 512, len(fr) * (512 + 16)), 'white')
        dr = ImageDraw.Draw(S)
        for q, i in enumerate(fr):
            for col, (ref, lab_) in enumerate(((old, 'today'), (t2m, 'T2'))):
                im = Image.open(REPO / 'dataset' / paths[idx.rows[frames[i]]]).convert('RGB')
                draw_masks(im, ref[i], new[i])
                S.paste(im, (col * 512, q * (512 + 16)))
                dr.text((col * 512 + 2, q * (512 + 16) + 514),
                        f'{idx.obs[frames[i]]} r{idx.rows[frames[i]]}: {lab_} {ref[i].sum()} vs T2b {new[i].sum()} '
                        f'(red {lab_} only, blue T2b only, green both)', fill='black')
        S.save(od / f'mask_sheet_{name}.jpg', quality=90)
        log(f'{domain}: {od / f"mask_sheet_{name}.jpg"}')


# ---------------------------------------------------------------------------------------------- encode
def cmd_encode(args):
    import torch
    import motion_t3 as mt
    import spatial_sae_pilot as ssp
    from src.eci.foreground import FgEncoder, FrameDatasetFG, encode_batch
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    loc = Path(os.environ['LOCAL_DIR']) / f'enc_{args.task}'
    loc.mkdir(parents=True, exist_ok=True)
    domain = args.domain
    Z = np.load(OUT / domain / 'mask' / 'masks.npz')
    r = rule_of(domain)
    idx = ssp.StoreIndex(domain)
    frames = Z['frames']
    reuse_t = r['equals_t2_rule'] and (T2 / domain / 'extra').exists()
    f_t = np.flatnonzero(Z['new_only'] > 0) if not reuse_t else np.zeros(0, np.int64)
    f_5 = np.flatnonzero(Z['no_d5'] > 0)
    row_t = idx.rows[frames[f_t]]
    row_5, paths_all = mt.nbr_rows(domain, idx, frames[f_5], -5)
    it_f = np.r_[f_t, f_5].astype(np.int64)
    it_k = np.r_[np.zeros(len(f_t), np.int8), np.ones(len(f_5), np.int8)]
    it_r = np.r_[row_t, row_5].astype(np.int64)
    urows = np.unique(it_r)
    mine_rows = np.array_split(urows, args.n_tasks)[args.task]
    sel = np.flatnonzero(np.isin(it_r, mine_rows))
    sel = sel[np.lexsort((it_k[sel], it_f[sel], it_r[sel]))]  # by row
    it_f, it_k, it_r = it_f[sel], it_k[sel], it_r[sel]
    bitsT, bits5 = Z['needt_bits'], Z['need5_bits']
    posl = [np.flatnonzero(np.unpackbits((bitsT if k == 0 else bits5)[f])) for f, k in zip(it_f, it_k)]
    n = int(sum(len(p) for p in posl))
    rr = np.searchsorted(mine_rows, it_r)  # local row index of every item
    log(f'{domain} task {args.task}/{args.n_tasks}: {len(urows):,} distinct rows overall, {len(mine_rows):,} here; '
        f'{len(sel):,} items ({int((it_k == 0).sum()):,} token_t, {int((it_k == 1).sum()):,} token_t-5; token_t reused '
        f'from T2: {reuse_t}), {n:,} tokens')
    lp = mt.stage_files([str(REPO / 'dataset' / paths_all[q]) for q in mine_rows], loc / 'jpg')
    enc = FgEncoder('dinov2_base', dev)
    ds = FrameDatasetFG(lp, enc.processor)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                     pin_memory=True, prefetch_factor=4)
    tok = np.lib.format.open_memmap(loc / 'tok.npy', 'w+', np.float16, (n, D))
    fidx = np.empty(n, np.int32)
    pos = np.empty(n, np.int16)
    kind = np.empty(n, np.int8)
    d5m = mt.D5Matcher(idx, domain)
    starts = np.searchsorted(rr, np.arange(len(mine_rows) + 1))
    o, i0, t0, chk = 0, 0, time.time(), {'t_vs_store_max_abs': 0.0, 't5_vs_d5prev_max_abs': 0.0, 'n_t': 0, 'n_5': 0}
    for b, (pix, _) in enumerate(dl):
        T = encode_batch(enc.model, pix, dev).cpu().numpy()
        for kk in range(len(pix)):
            for it in range(starts[i0 + kk], starts[i0 + kk + 1]):
                p = posl[it]
                tok[o:o + len(p)] = T[kk, p]
                fidx[o:o + len(p)], pos[o:o + len(p)], kind[o:o + len(p)] = it_f[it], p, it_k[it]
                o += len(p)
                if b < 3:  # identity checks on the same frames' stored tokens
                    fs = np.array([frames[it_f[it]]])
                    if it_k[it] == 0:
                        st, ps, _ = idx.load(fs)
                        if len(ps):
                            chk['t_vs_store_max_abs'] = max(chk['t_vs_store_max_abs'], float(np.abs(
                                st.astype(np.float32) - T[kk, ps.astype(np.int64)].astype(np.float32)).max()))
                            chk['n_t'] += len(ps)
                    else:
                        _, ps, _ = idx.load(fs)
                        pv, _, ok = d5m.fetch(fs)
                        if ok.any():
                            chk['t5_vs_d5prev_max_abs'] = max(chk['t5_vs_d5prev_max_abs'], float(np.abs(
                                pv[ok].astype(np.float32) - T[kk, ps[ok].astype(np.int64)].astype(np.float32)).max()))
                            chk['n_5'] += int(ok.sum())
        i0 += len(pix)
        if b == 2:
            log(f'  identity checks (first 3 batches): {json.dumps(chk)}')
        if b % 200 == 0:
            log(f'  batch {b}/{len(dl)} {i0:,}/{len(mine_rows):,} rows {i0 / (time.time() - t0):.1f} f/s')
    assert o == n and i0 == len(mine_rows)
    tok.flush()
    del tok
    shutil.rmtree(loc / 'jpg')
    np.save(loc / 'fidx.npy', fidx)
    np.save(loc / 'pos.npy', pos)
    np.save(loc / 'kind.npy', kind)
    info = {'domain': domain, 'task': args.task, 'n_tasks': args.n_tasks, 'n_rows': int(len(mine_rows)),
            'n_items': int(len(sel)), 'n_tokens': n, 'n_token_t': int((kind == 0).sum()), 'n_token_t5': int((kind == 1).sum()),
            'reuse_t2_extra_for_token_t': bool(reuse_t), 'checks': chk, 'elapsed_s': round(time.time() - t0, 1),
            'frames_per_s': i0 / max(time.time() - t0, 1e-9), 'gpu': torch.cuda.get_device_name(0)}
    (loc / 'info.json').write_text(json.dumps(info, indent=1))
    dst = OUT / domain / 'extra' / f'part_{args.task}'
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(loc, dst, dirs_exist_ok=True)
    shutil.rmtree(loc)
    log(f'done -> {dst}: {json.dumps(info)}')


# ---------------------------------------------------------------------------------------------- C2 tokens
class Extra:
    """Encoded tokens of one kind (0 token_t, 1 token_t-5), keyed by store frame * 1024 + pos, sorted."""

    def __init__(self, keys, tok):
        o = np.argsort(keys, kind='stable')
        self.keys, self.tok = keys[o], tok[o]
        assert (np.diff(self.keys) > 0).all(), 'duplicate extra keys'

    def take(self, keys):
        j = np.searchsorted(self.keys, keys)
        ok = (j < len(self.keys)) & (self.keys[np.minimum(j, len(self.keys) - 1)] == keys)
        assert ok.all(), f'{int((~ok).sum())} kept patches have no encoded token'
        return self.tok[j]


def load_extras(domain, frames_all):
    """-> Extra token_t, Extra token_t-5 (frames_all = masks.npz frames, fidx indexes it)."""
    K = {0: [], 1: []}
    T = {0: [], 1: []}
    r = rule_of(domain)
    if r['equals_t2_rule']:
        Z2 = np.load(T2 / domain / 'mask' / 'masks.npz')
        assert np.array_equal(Z2['frames'], frames_all)
        for d in sorted((T2 / domain / 'extra').glob('part_*')):
            f = np.load(d / 'fidx.npy').astype(np.int64)
            K[0].append(frames_all[f] * 1024 + np.load(d / 'pos.npy').astype(np.int64))
            T[0].append(np.load(d / 'tok.npy'))
    for d in sorted((OUT / domain / 'extra').glob('part_*')):
        info = json.loads((d / 'info.json').read_text())
        kind = np.load(d / 'kind.npy')
        f = np.load(d / 'fidx.npy').astype(np.int64)
        p = np.load(d / 'pos.npy').astype(np.int64)
        t = np.load(d / 'tok.npy')
        for k in (0, 1):
            m = kind == k
            if m.any():
                K[k].append(frames_all[f[m]] * 1024 + p[m])
                T[k].append(t[m])
        n_tasks = info['n_tasks']
    parts = sorted((OUT / domain / 'extra').glob('part_*'))
    if parts and len(parts) != n_tasks:
        raise RuntimeError(f'{domain}: {len(parts)}/{n_tasks} encode tasks present')
    ex = {k: Extra(np.concatenate(K[k]) if K[k] else np.zeros(0, np.int64),
                   np.concatenate(T[k]) if T[k] else np.zeros((0, D), np.float16)) for k in (0, 1)}
    log(f'  extras: token_t {len(ex[0].keys):,}, token_t-5 {len(ex[1].keys):,}')
    return ex


def d5_chunk(m, idx, fc):
    """d5 prev tokens of the store frames fc (sorted) -> keys (frame * 1024 + pos) sorted, prev (n, 768) fp16."""
    K, P = [], []
    for s in np.unique(idx.shard[fc]):
        fs = fc[idx.shard[fc] == s]
        lf = fs - m.f0[s]
        brk = np.flatnonzero(np.diff(lf) != 1) + 1
        pos_s = m.pos('d5', s)
        pv = np.memmap(m.d5.dirs[s] / 'prev.f16', np.float16, 'r', shape=(int(m.d5.sizes[s]), D))
        for a, b in zip(np.r_[0, brk], np.r_[brk, len(lf)]):
            l0, l1 = lf[a], lf[b - 1]
            lo, hi = m.d5_start[s][l0], m.d5_start[s][l1] + m.d5_n[s][l1]
            fr = np.repeat(np.arange(l0, l1 + 1), m.d5_n[s][l0:l1 + 1]) + m.f0[s]
            K.append(fr.astype(np.int64) * 1024 + pos_s[lo:hi])
            P.append(np.asarray(pv[lo:hi]))
    K = np.concatenate(K) if K else np.zeros(0, np.int64)
    P = np.concatenate(P) if P else np.zeros((0, D), np.float16)
    assert (np.diff(K) > 0).all()
    return K, P


def stage_c2(domain, idx, frames, dst, chunk=4000):
    """C2 tokens of the store frames (sorted): T2b-kept patches, token_t (base store, else encoded) and token_t-5 (d5
    prev, else encoded) -> (tok memmap, prev memmap, pos (N,) int16, lens (F,) int64, info)."""
    Z = np.load(OUT / domain / 'mask' / 'masks.npz')
    k = np.searchsorted(Z['frames'], frames)
    assert (Z['frames'][k] == frames).all()
    bits = Z['bits'][k]
    n_keep = popcount(bits)
    ex = load_extras(domain, Z['frames'])
    m = d5_layout(idx, domain)
    dst.mkdir(parents=True, exist_ok=True)
    n = int(n_keep.sum())
    tok = np.lib.format.open_memmap(dst / 'tok.npy', 'w+', np.float16, (n, D))
    prev = np.lib.format.open_memmap(dst / 'prev.npy', 'w+', np.float16, (n, D))
    pos = np.empty(n, np.int16)
    lens = np.zeros(len(frames), np.int64)
    o, t0, n_t_enc, n_5_enc = 0, time.time(), 0, 0
    for c0 in range(0, len(frames), chunk):
        fc = frames[c0:c0 + chunk]
        t, p, ln = idx.load(fc)
        new = np.unpackbits(bits[c0:c0 + chunk], axis=1).astype(bool)
        fi = np.repeat(np.arange(len(fc)), ln)
        keep = new[fi, p.astype(np.int64)]
        kb = fc[fi[keep]].astype(np.int64) * 1024 + p[keep]
        # kept patches absent from the base store
        allk = (fc[:, None].astype(np.int64) * 1024 + np.arange(1024)[None])[new]  # sorted (frame, pos)
        miss = allk[~np.isin(allk, kb)]
        keys = np.concatenate([kb, miss])
        tt = np.concatenate([t[keep], ex[0].take(miss)])
        so = np.argsort(keys, kind='stable')
        keys, tt = keys[so], tt[so]
        assert np.array_equal(keys, allk)
        kd, pv = d5_chunk(m, idx, fc)
        j = np.searchsorted(kd, keys)
        ok = (j < len(kd)) & (kd[np.minimum(j, len(kd) - 1)] == keys)
        pp = np.empty_like(tt)
        pp[ok] = pv[j[ok]]
        pp[~ok] = ex[1].take(keys[~ok])
        n_t_enc += len(miss)
        n_5_enc += int((~ok).sum())
        nn = len(keys)
        tok[o:o + nn], prev[o:o + nn], pos[o:o + nn] = tt, pp, (keys % 1024).astype(np.int16)
        lens[c0:c0 + chunk] = new.sum(1)
        o += nn
    assert o == n and (lens == n_keep).all()
    tok.flush()
    prev.flush()
    del tok, prev
    info = {'n_frames': int(len(frames)), 'n_tokens': n, 'kept_per_frame': n / len(frames),
            'token_t_encoded': n_t_enc, 'token_t5_encoded': n_5_enc, 'stage_s': round(time.time() - t0, 1)}
    log(f'  staged C2 tokens: {json.dumps(info)} ({n * D * 4 / 1e9:.1f} GB token + prev)')
    return np.load(dst / 'tok.npy', mmap_mode='r'), np.load(dst / 'prev.npy', mmap_mode='r'), pos, lens, info


# ---------------------------------------------------------------------------------------------- train
def cmd_train(args):
    import torch
    import motion_t3 as mt
    import multiscale_sae as ms
    import sae_levers as sl
    import spatial_sae_pilot as ssp
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    out = OUT / args.domain
    loc = ms.local_dir()
    idx = ssp.StoreIndex(args.domain)
    tr, _, _, _, train_v, _ = idx.split(args.domain, ms.N_TRAIN[args.domain], 0)
    tr = np.sort(tr)
    tok, prev, pos, lens, info = stage_c2(args.domain, idx, tr, loc / 'train')
    n = len(pos)
    nc, largest, meta = sl.merged_meta(args.domain, pos, lens)
    cap = min(sl.CAP[args.domain], n)
    sel = np.sort(np.random.default_rng(args.pool_seed).choice(n, cap, replace=False))
    static = torch.from_numpy(sl.gather(tok, sel))
    ch = torch.empty((cap, D), dtype=torch.float16)
    for c0 in range(0, cap, 500_000):
        s = sel[c0:c0 + 500_000]
        ch[c0:c0 + len(s)] = (static[c0:c0 + len(s)].float() - torch.from_numpy(sl.gather(prev, s)).float()).half()
    X = torch.cat([static, ch], 1).to(dev)
    del static, ch, tok, prev
    shutil.rmtree(loc / 'train')
    zero = float(sum(int((X[a:a + 500_000, D:].float().abs().sum(1) == 0).sum()) for a in range(0, len(X), 500_000))
                 / len(X))
    rn = float((X[:200_000, D:].float().norm(dim=1) / X[:200_000, :D].float().norm(dim=1)).median())
    t2meta = T2 / args.domain / 'train_meta_t2.json'
    meta.update(domain=args.domain, arm='C2', rule=rule_of(args.domain), n_train_frames=int(len(tr)), n_train_tokens=int(n),
                cap=int(cap), pool_seed=args.pool_seed, steps=args.steps, stage=info, zero_change_frac=zero,
                median_change_over_token_norm=rn, tokens_per_frame=n / len(tr),
                t2_n_train_tokens=json.loads(t2meta.read_text())['n_train_tokens'] if t2meta.exists() else None,
                frames_with_no_token=float((lens == 0).mean()), pool_first_idx=sel[:5].tolist())
    out.mkdir(parents=True, exist_ok=True)
    (out / 'train_meta_C2.json').write_text(json.dumps(meta, indent=1))
    log(f'C2 train: {n:,} tokens ({n / len(tr):.1f}/frame), pool {cap:,}, zero-change {100 * zero:.2f}%, median '
        f'|change|/|token| {rn:.3f}; a1 {meta["median_single_animal_area"]}')
    for seed in args.seeds:
        mt.train_one('C2', seed, X, args.steps, out / 'sae' / f'C2_s{seed}',
                     {'config': 'C2', 'arm': {'delta': 5, 'kind': 'back'}, 'motion_delta': 5, 'domain': args.domain,
                      'cap': cap, 'pool_seed': args.pool_seed, 'blocks': [D, D], 'mask': 't2b',
                      'rule': rule_of(args.domain)}, dev)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- evaluate
class MotionTokens:
    """Row slices -> float32 (n, 1536) [token_t, token_t - token_t-5] (motion_t3 evaluation input, fp32 change)."""

    def __init__(self, tok, prev):
        self.tok, self.prev = tok, prev

    def __len__(self):
        return len(self.tok)

    def __getitem__(self, s):
        a = np.asarray(self.tok[s], np.float32)
        return np.concatenate([a, a - np.asarray(self.prev[s], np.float32)], 1)


def encode_pe(sae, norm, tok, lens, dev):
    """tight_mask_sae.encode_pe for any input width (static block 0:768 also scored separately)."""
    import torch
    import sae_levers as sl
    m, F_ = sae.n_latents, len(lens)
    st = np.r_[0, np.cumsum(lens)]
    pres = torch.zeros(F_, m, dtype=torch.float16, device=dev)
    ext = torch.zeros(F_, m, dtype=torch.float16, device=dev)
    dd = norm.mean.shape[0]
    s1 = torch.zeros(dd, dtype=torch.float64, device=dev)
    s2 = torch.zeros(dd, dtype=torch.float64, device=dev)
    sse = torch.zeros(dd, dtype=torch.float64, device=dev)
    fire = torch.zeros(m, dtype=torch.float64, device=dev)
    l0 = ntok = 0
    with torch.no_grad():
        for f0, f1 in sl.frame_chunks(lens, max(4096, 2 ** 27 // m)):
            lo, hi = st[f0], st[f1]
            if hi == lo:
                continue
            x = norm(torch.from_numpy(np.ascontiguousarray(tok[lo:hi])).to(dev))
            z = sae.encode(x, mode='threshold')
            xh = sae.decode(z)
            s1 += x.double().sum(0)
            s2 += x.double().pow(2).sum(0)
            sse += (x - xh).double().pow(2).sum(0)
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
    tss = s2 - s1.pow(2) / ntok
    fve = lambda a, b: float(1 - sse[a:b].sum() / tss[a:b].sum())  # noqa: E731
    stt = {'fve': fve(0, dd), 'l0_per_token': l0 / ntok, 'dead_frac': float((fire == 0).double().mean()), 'n_tokens': ntok}
    if dd > D:
        stt.update(fve_static=fve(0, D), fve_change=fve(D, dd))
    return pres, ext, stt


def cmd_evaluate(args):
    import torch
    import motion_t3 as mt
    import multiscale_sae as ms
    import sae_levers as sl
    import spatial_sae_pilot as ssp
    import tight_mask_sae as tms
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    domain = args.domain
    out = OUT / domain
    loc = ms.local_dir()
    lout = loc / 'out'
    for d in ('eval', 'levels', 'codes_best', 'sheets'):
        (lout / d).mkdir(parents=True, exist_ok=True)
    idx = ssp.StoreIndex(domain)
    _, ev, _, _, _, _ = idx.split(domain, ms.N_TRAIN[domain], 0)
    ev = np.sort(ev)
    S = np.load(T2 / domain / 'mask' / 'sites.npz')
    assert (S['frames'] == ev).all()
    site, bcount, sdark = S['site'], S['bcount'], S['dark']
    lab, _, valid = ms.eval_labels(domain, idx, ev)
    labels = sl.LABELS[domain]
    unit = lab.obs.str.rsplit('_', n=2).str[0].values if domain == 'mice' else lab.obs.values
    units = sorted(set(unit))
    half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
    assert (half == np.load(LEV / domain / 'half.npy')).all(), 'halves differ from sae_levers'
    KB = bcount.shape[1]
    Yt = torch.from_numpy(lab[labels].values.astype(bool)).to(dev)
    a1 = {'C1': json.loads((LEV / domain / 'train_meta_w4096_w16384.json').read_text())['median_single_animal_area']}
    if (out / 'train_meta_C2.json').exists():
        a1['C2'] = json.loads((out / 'train_meta_C2.json').read_text())['median_single_animal_area']
    log(f'{domain}: {len(ev):,} eval frames; a1 {a1}')
    for arm in args.arms:
        if arm == 'C1':
            _, pos, lens = ms.stage(idx, ev, loc / 'eval')
            tok = np.load(loc / 'eval' / 'tok.npy')
            shutil.rmtree(loc / 'eval')
            prev = mt.stage_prev5(idx, domain, ev, loc / 'prev_eval.npy')
            st_info = dict(mt.STAGE_INFO)
        else:
            tok_m, prev, pos, lens, st_info = stage_c2(domain, idx, ev, loc / 'eval')
            tok = np.load(loc / 'eval' / 'tok.npy')
            del tok_m
            os.remove(loc / 'eval' / 'tok.npy')
        X = MotionTokens(tok, prev)
        st = np.r_[0, np.cumsum(lens)]
        rank = np.argsort(np.argsort(lens + np.random.default_rng(0).random(len(lens)) * 1e-3))
        decile = np.minimum(rank * 10 // len(lens), 9)
        fid = np.repeat(np.arange(len(lens)), lens)
        on = site[fid, pos.astype(np.int64)] > 0
        dk = sdark[fid, pos.astype(np.int64)] > 0
        arm_meta = {'arm': arm, 'kept_per_frame': float(lens.mean()), 'frames_no_token': float((lens == 0).mean()),
                    'token_off_animal_share': float(1 - on.mean()), 'token_no_dark_pixel_share': float(1 - dk.mean()),
                    'a1': a1[arm], 'stage': st_info}
        log(f'  [{arm}] {json.dumps(arm_meta)}')
        np.save(lout / 'codes_best' / f'{arm}_lens.npy', lens.astype(np.int16))
        for seed in args.seeds:
            t0 = time.time()
            key = f'{arm}_s{seed}'
            pth = T3 / domain / 'sae' / f'M5_s{seed}' / 'sae.pt' if arm == 'C1' else out / 'sae' / f'C2_s{seed}' / 'sae.pt'
            sae, norm, ck = load_sae(pth, dev)
            assert sae.d_in == 2 * D
            pres, ext, stt = encode_pe(sae, norm, X, lens, dev)
            resP, _ = sl.score_model(pres, Yt, labels, valid, half, decile, dev)
            resE, _ = sl.score_model(ext, Yt, labels, valid, half, decile, dev)
            with torch.no_grad():
                T, topi, topv, chk = tms.level_map(sae, norm, X, pos, lens, pres, ext, site, bcount, a1[arm], dev, KB)
            T.to_csv(lout / 'levels' / f'{key}.csv', index=False)
            e = {'arm': arm, 'seed': seed, 'ckpt': str(pth.relative_to(REPO)), **stt, **arm_meta, 'presence': resP,
                 'extent': resE, 'levels': tms.level_counts(T), 'level_checks': chk}
            if arm == 'C1':  # reproduction of the T3 M5 evaluation (same codes, same scoring)
                ref = json.loads((T3 / domain / 'eval' / f'M5_s{seed}.json').read_text())['labels']
                e['repro_max_abs_diff_cf_auroc_vs_t3'] = max(abs(ref[b]['cf_auroc'] - resP[b]['cf_auroc']) for b in labels)
            js = sorted({d[s]['neuron'] for R in (resP, resE) for b in labels for d in R[b]['dirs'] for s in ('auc', 'ap', 'top1')})
            jt = torch.tensor(js, device=dev)
            np.savez_compressed(lout / 'codes_best' / f'{key}.npz', neurons=np.array(js),
                                presence=pres[:, jt].cpu().numpy(), extent=ext[:, jt].cpu().numpy())
            (lout / 'eval' / f'{key}.json').write_text(json.dumps(e, indent=1, default=float))
            f = lambda R, b: f"{R[b]['cf_auroc']:.3f}/{R[b]['cf_ap']:.4f}/{R[b]['cf_top1']:.3f}/{R[b]['cf_size_ctrl_auroc']:.3f}"  # noqa: E731
            log(f'  {key}: FVE {stt["fve"]:.4f} (static {stt.get("fve_static", float("nan")):.4f} change '
                f'{stt.get("fve_change", float("nan")):.4f}) L0 {stt["l0_per_token"]:.2f} | cf AUROC/AP/top1/size-ctrl: '
                + ' | '.join(f'{b} P {f(resP, b)}' for b in labels)
                + (f' | repro diff vs T3 {e["repro_max_abs_diff_cf_auroc_vs_t3"]:.2e}' if arm == 'C1' else '')
                + f' | plain {e["levels"]["plain_background"]} eligible {e["levels"]["n_eligible"]} ({time.time() - t0:.0f}s)')
            if seed == args.seeds[0]:
                man = []
                for b in labels:
                    rows = []
                    for R, rn in ((resP, 'presence'),):
                        dd = R[b]['dirs'][0]
                        j = dd['auc']['neuron']
                        x = pres[:, j].float().cpu().numpy()
                        fr_ = tms.top_frames(x, valid[b] & (half == dd['test_half']), lab.obs.values)
                        rows.append((f'{key} {b}: {rn}-AUROC pick n{j} (select half {dd["select_half"]}, test AUROC '
                                     f'{dd["auc"]["test_auc"]:.3f}); top {rn} frames of the test half', j, fr_))
                    man += tms.sheet(lout / 'sheets' / f'{key}__{b}.jpg', rows, lab, sae, norm, X, pos, st, dev,
                                     labels_col=b)
                pd.DataFrame(man).to_csv(lout / 'sheets' / f'{key}_manifest.csv', index=False)
            del pres, ext, topi, topv
            torch.cuda.empty_cache()
        del tok, prev, X
        for f_ in ('prev_eval.npy',):
            if (loc / f_).exists():
                os.remove(loc / f_)
        shutil.rmtree(loc / 'eval', ignore_errors=True)
    lab.drop(columns=['frame_path']).to_parquet(lout / 'labels.parquet')
    np.save(lout / 'half.npy', half)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(lout, out, dirs_exist_ok=True)
    log(f'results copied to {out}')


# ---------------------------------------------------------------------------------------------- readouts / decision
ARMDIR = {'C0': (T2, 'base'), 'C1': (OUT, 'C1'), 'C2': (OUT, 'C2')}
METRICS = ('cf_auroc', 'cf_ap', 'cf_ap_at_auroc', 'cf_top1', 'cf_size_ctrl_auroc')


def ms_(v):
    v = np.asarray(v, float)
    return {'mean': float(np.nanmean(v)), 'sd': float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
            'per_seed': [float(x) for x in v]}


def per_half(root, d, prefix, seed, b, lab, half, wstart, lens):
    import readouts as rd
    ev = json.loads((root / d / 'eval' / f'{prefix}_s{seed}.json').read_text())
    z = np.load(root / d / 'codes_best' / f'{prefix}_s{seed}.npz')
    nl = list(z['neurons'])
    res = {}
    for ro in ('presence', 'extent'):
        for h, j in rd.picks(ev, ro, b).items():
            m = half == h
            res[(ro, h)] = rd.video_readouts(z['presence'][m, nl.index(j)].astype(np.float32),
                                             z['extent'][m, nl.index(j)].astype(np.float32), lens[m], lab[m], wstart)
    return res


def cmd_readouts(args):
    import readouts as rd
    import sae_levers as sl
    stage = Path(os.environ.get('LOCAL_DIR', '/tmp')) / 'readouts'
    stage.mkdir(parents=True, exist_ok=True)
    S = {}
    for d in DOMAINS:
        labels = sl.LABELS[d]
        lab = pd.read_parquet(OUT / d / 'labels.parquet')
        lab2 = pd.read_parquet(T2 / d / 'labels.parquet')
        assert (lab.obs.values == lab2.obs.values).all() and (lab.frame_idx.values == lab2.frame_idx.values).all()
        half = np.load(OUT / d / 'half.npy')
        assert np.array_equal(half, np.load(T2 / d / 'half.npy'))
        gt = rd.mice_truth(stage) if d == 'mice' else rd.ants_truth()
        wstart = gt['wstart']
        Dd = {'frame': {}, 'video': {}, 'levels': {}, 'arm_meta': {}, 'effects': {}}
        for arm, (root, prefix) in ARMDIR.items():
            E = [json.loads((root / d / 'eval' / f'{prefix}_s{s}.json').read_text()) for s in SEEDS]
            lens = np.load(root / d / 'codes_best' / f'{prefix}_lens.npy').astype(np.float32)
            Dd['arm_meta'][arm] = {k: E[0].get(k) for k in ('kept_per_frame', 'frames_no_token', 'token_off_animal_share',
                                                            'token_no_dark_pixel_share', 'a1')}
            Dd['arm_meta'][arm].update({q: ms_([e[q] for e in E]) for q in ('fve', 'l0_per_token', 'dead_frac')})
            for q in ('repro_max_abs_diff_cf_auroc', 'repro_max_abs_diff_cf_auroc_vs_t3'):
                if all(q in e for e in E):
                    Dd['arm_meta'][arm][q] = max(e[q] for e in E)
            for b in labels:
                Dd['frame'][f'{arm}|{b}'] = {q: ms_([e['presence'][b][q] for e in E]) for q in METRICS}
                Dd['frame'][f'{arm}|{b}']['base_rate'] = E[0]['presence'][b]['base_rate']
            L = [e['levels'] for e in E]
            Dd['levels'][arm] = {'n_eligible': ms_([x['n_eligible'] for x in L]),
                                 'plain_background': ms_([x['plain_background'] for x in L]),
                                 'place_bound': ms_([x['place_bound'] for x in L]),
                                 'social': {c: ms_([x['social'][c] for x in L]) for c in L[0]['social']},
                                 'social_presence': {c: ms_([x['social_presence'][c] for x in L]) for c in L[0]['social_presence']},
                                 'social_x_place_not_plain': {c: ms_([x['social_x_place_not_plain'][c] for x in L])
                                                              for c in L[0]['social_x_place_not_plain']}}
            for b in labels:
                V = []
                for s in SEEDS:
                    R = per_half(root, d, prefix, s, b, lab, half, wstart, lens)
                    V.append(rd.video_check_mice(R, gt, b) if d == 'mice' else rd.video_check_ants(R, gt, b))
                agg = {}
                for k in V[0]:
                    agg[k] = {q: ms_([v[k][q] for v in V]) for q in V[0][k] if q != 'effects'}
                    if 'effects' in V[0][k]:
                        eff = {}
                        for sw in V[0][k]['effects']:
                            eff[sw] = {'dz': ms_([v[k]['effects'][sw]['dz'] for v in V]),
                                       'truth_dz': V[0][k]['effects'][sw]['truth_dz'],
                                       'truth_p': V[0][k]['effects'][sw]['truth_p'],
                                       'sign_agree_seeds': int(sum(v[k]['effects'][sw]['sign_agree'] for v in V)),
                                       'p_per_seed': [v[k]['effects'][sw]['p'] for v in V]}
                        agg[k]['effects'] = eff
                Dd['video'][f'{arm}|{b}'] = agg
            log(f'{d} {arm}: done')
        S[d] = Dd
    S['decision'] = decide(S)
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    tables(S)


def decide(S):
    import sae_levers as sl
    R = {}
    for arm in ('C1', 'C2'):
        R[arm] = {}
        for d in DOMAINS:
            F_, V = S[d]['frame'], S[d]['video']
            per = {}
            for b in sl.LABELS[d]:
                A, B = F_[f'{arm}|{b}'], F_[f'C0|{b}']
                da = A['cf_auroc']['mean'] - B['cf_auroc']['mean']
                ra = A['cf_ap']['mean'] / B['cf_ap']['mean']
                g_auc = da >= 0.03 and A['cf_auroc']['mean'] > max(B['cf_auroc']['per_seed'])
                g_ap = ra >= 1.3 and A['cf_ap']['mean'] > max(B['cf_ap']['per_seed'])
                va, vb = V[f'{arm}|{b}']['presence_mean|fgctrl'], V[f'C0|{b}']['presence_mean|fgctrl']
                dr = va['video_r_rate']['mean'] - vb['video_r_rate']['mean']
                dbp = va['video_r_bpm']['mean'] - vb['video_r_bpm']['mean']
                per[b] = {'d_cf_auroc': da, 'ratio_cf_ap': ra, 'drop_gt_0.02': bool(da < -0.02),
                          'frame_gain_auroc': bool(g_auc), 'frame_gain_ap': bool(g_ap),
                          'd_ctl_video_r_rate': dr, 'd_ctl_video_r_bpm': dbp,
                          'video_gain': bool(dr >= 0.10 or dbp >= 0.10)}
                per[b]['gain'] = bool(g_auc or g_ap or per[b]['video_gain'])
            R[arm][d] = {'per_label': per, 'no_drop': not any(v['drop_gt_0.02'] for v in per.values()),
                         'any_gain': any(v['gain'] for v in per.values())}
            R[arm][d]['domain_ok'] = bool(R[arm][d]['no_drop'] and R[arm][d]['any_gain'])
        R[arm]['WIN'] = all(R[arm][d]['domain_ok'] for d in DOMAINS)
        R[arm]['mean_cf_auroc'] = float(np.mean([S[d]['frame'][f'{arm}|{b}']['cf_auroc']['mean']
                                                 for d in DOMAINS for b in sl.LABELS[d]]))
        R[arm]['mean_ctl_video_r'] = float(np.mean([S[d]['video'][f'{arm}|{b}']['presence_mean|fgctrl'][q]['mean']
                                                    for d in DOMAINS for b in sl.LABELS[d]
                                                    for q in ('video_r_rate', 'video_r_bpm')]))
    winners = [a for a in ('C1', 'C2') if R[a]['WIN']]
    if len(winners) == 2:
        v = max(winners, key=lambda a: R[a]['mean_ctl_video_r'])
        other = [a for a in winners if a != v][0]
        if abs(R[v]['mean_cf_auroc'] - R[other]['mean_cf_auroc']) <= 0.01 or R[v]['mean_cf_auroc'] > R[other]['mean_cf_auroc']:
            final, why = v, 'both win; larger mean count-controlled video r, frame-level mean AUROC within 0.01 (or higher)'
        else:
            final = max(winners, key=lambda a: R[a]['mean_cf_auroc'])
            why = 'both win; video-r leader is > 0.01 below on frame-level mean AUROC -> better frame-level arm'
    elif len(winners) == 1:
        final, why = winners[0], 'only winning arm'
    else:
        final, why = 'C0', 'no arm wins'
    R['winners'], R['FINAL'], R['why'] = winners, final, why
    return R


def tables(S):
    import sae_levers as sl
    f = lambda x: f"{x['mean']:.3f} ± {x['sd']:.3f}"  # noqa: E731
    f4 = lambda x: f"{x['mean']:.4f} ± {x['sd']:.4f}"  # noqa: E731
    L = ['# COMBINED test (C0 base / C1 today mask + M5 / C2 T2b mask + M5), mean ± sd over 3 seeds\n']
    R = json.loads((OUT / 'rule.json').read_text())
    for d in DOMAINS:
        r = R[d]
        L.append(f'\n## {d}\n')
        L.append(f"T2b rule: sigma {r['sigma_median']:.3f} (used {r['sigma_used']:.2f}), delta {r['delta']}, X {r['frac']:.2f}; "
                 f"empty-floor keep {100 * r['far_keep_rate']:.3f}%, dark-core keep {100 * r['dark_core_keep_rate']:.2f}% "
                 f"(train calibration); same as T2: {r['equals_t2_rule']}\n")
        ap = OUT / d / 'mask' / 'apply.json'
        if ap.exists():
            a = json.loads(ap.read_text())
            L.append('kept patches / frame (all frames): ' + ', '.join(f"{k} {v['all']:.1f}" for k, v in a['kept_per_frame'].items())
                     + f"; off-animal share of kept (eval): {json.dumps({k: round(v, 3) for k, v in a['eval_off_animal_share_of_kept'].items()})}"
                     + f"; on-animal patches kept: {json.dumps({k: round(v, 3) for k, v in a['eval_on_animal_patches_kept'].items()})}\n")
            if 'onlid' in a:
                o = a['onlid']
                L.append('on-lid yellow ant covered (v3): ' + json.dumps({k: round(v, 3) for k, v in o['onlid'].items()})
                         + '; not on lid: ' + json.dumps({k: round(v, 3) for k, v in o['not_onlid'].items()}) + '\n')
        L.append('| label | arm | cf AUROC | cf AP (AP pick) | honest top-1% | size-ctrl AUROC | video r rate raw / ctl | '
                 'video r bouts raw / ctl |' + (' pooled d-r raw / ctl |' if d == 'mice' else ''))
        L.append('|---|---|---|---|---|---|---|---|' + ('---|' if d == 'mice' else ''))
        for b in sl.LABELS[d]:
            for arm in ARMDIR:
                x = S[d]['frame'][f'{arm}|{b}']
                v, c = S[d]['video'][f'{arm}|{b}']['presence_mean'], S[d]['video'][f'{arm}|{b}']['presence_mean|fgctrl']
                row = (f"| {b} ({x['base_rate']:.4f}) | {arm} | {f(x['cf_auroc'])} | {f4(x['cf_ap'])} | {f(x['cf_top1'])} | "
                       f"{f(x['cf_size_ctrl_auroc'])} | {v['video_r_rate']['mean']:.2f} / {c['video_r_rate']['mean']:.2f} | "
                       f"{v['video_r_bpm']['mean']:.2f} / {c['video_r_bpm']['mean']:.2f} |")
                if d == 'mice':
                    row += f" {v['delta_r_pooled']['mean']:.2f} / {c['delta_r_pooled']['mean']:.2f} |"
                L.append(row)
        if d == 'mice':
            L.append('\nper-switch dz of the held-out presence-mean (seed mean; annotated-rate dz, p in brackets; sign '
                     'agreement over 3 seeds), raw | fg-controlled:\n')
            for b in sl.LABELS[d]:
                for arm in ARMDIR:
                    E0 = S[d]['video'][f'{arm}|{b}']['presence_mean']['effects']
                    E1 = S[d]['video'][f'{arm}|{b}']['presence_mean|fgctrl']['effects']
                    L.append(f'- {b} {arm}: ' + '; '.join(
                        f"{sw} {E0[sw]['dz']['mean']:+.2f} ({E0[sw]['truth_dz']:+.2f}, p {E0[sw]['truth_p']:.3f}) "
                        f"{E0[sw]['sign_agree_seeds']}/3 | {E1[sw]['dz']['mean']:+.2f} {E1[sw]['sign_agree_seeds']}/3"
                        for sw in E0))
        L.append('\nComposition (eligible neurons, mean over seeds):\n')
        for arm in ARMDIR:
            Lv = S[d]['levels'][arm]
            am = S[d]['arm_meta'][arm]
            L.append(f"- {arm}: kept/frame {am['kept_per_frame']:.1f}, FVE {am['fve']['mean']:.4f}; eligible "
                     f"{Lv['n_eligible']['mean']:.0f}; plain background {Lv['plain_background']['mean']:.1f}; place-bound "
                     f"{Lv['place_bound']['mean']:.1f}; social (extent) "
                     + ', '.join(f"{c} {v['mean']:.1f}" for c, v in Lv['social'].items())
                     + '; not-plain social x place ' + ', '.join(f"{c} {v['mean']:.1f}" for c, v in Lv['social_x_place_not_plain'].items()))
    L.append('\n## Pre-registered decision\n')
    L.append('```\n' + json.dumps(S['decision'], indent=1, default=float) + '\n```')
    (OUT / 'tables.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['choose', 'apply', 'encode', 'train', 'evaluate', 'readouts'])
    p.add_argument('--domain', default='mice', choices=list(DOMAINS))
    p.add_argument('--domains', default='mice,ants')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--n-tasks', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--num-workers', type=int, default=12)
    p.add_argument('--arms', nargs='+', default=['C1', 'C2'])
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--pool-seed', type=int, default=0)
    a = p.parse_args()
    {'choose': cmd_choose, 'apply': cmd_apply, 'encode': cmd_encode, 'train': cmd_train, 'evaluate': cmd_evaluate,
     'readouts': cmd_readouts}[a.cmd](a)


if __name__ == '__main__':
    main()
