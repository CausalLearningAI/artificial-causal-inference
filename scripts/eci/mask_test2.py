"""
MASK test 2: a DINOv2-only animal mask (the token cue alone, no dark-pixel cue) vs today's mask (ants v2/v3 + mice v1).

Follow-up of scripts/eci/mask_test.py (M_today / M_pix / M_pix_nodil). Same frames, splits, SAE recipe (COMBINED C1:
Matryoshka BatchTopK 4096 latents / k 16 on [token_t, token_t - token_{t-5}], DINOv2-base 448, TokenNorm per 768 block,
motion_t3.train_one, 6000 steps, pool cap mice 4M / ants 2M tokens), evaluation and accuracy code (imported from
mask_test.py, with its module paths redirected to OUT here).

Today's mask (src/eci/foreground.py foreground_parts): core = TOKEN cue OR DARK cue, then a 1-patch dilation
    token cue  feat = drop_isolated(dist > thr_video), dist = 1 - cos(DINOv2 token, the video's per-position median
               background token of the time block, bg_seg), thr_video = the 0.995 quantile of dist on patches > 2
               patches from any dark-cue patch (load_background / video_threshold; the threshold sample uses the dark
               cue, the per-frame rule does not)
    dark cue   dark_fraction > 0.15 (grey < 60 and > 40 below the per-pixel 0.98-quantile background)
    dilation   mice 'fg448': every 8-neighbour of a core patch; ants 'ants': only 8-neighbours holding a dark pixel
Arms (new here)
    M_dino      (D1)  feat | 3x3-dilate(feat)  - plain 1-patch dilation onto every neighbour, both domains: no pixel
                      quantity in the per-frame rule at all
    M_dino_cur  (D2)  feat with each domain's CURRENT dilation rule: mice = D1 (fg448 dilates onto every neighbour, so
                      the two arms are the same mask and only D1 is run); ants = feat | (3x3-dilate(feat) & dark > 0)
Reused: M_today = C1 (results/vision/eci_combined, pool 0) + T0p1 / T0p2 (results/vision/eci_mask_test, pools 1, 2);
    M_pix = P1 / P1p1 / P1p2 (eci_mask_test; mice has pools 0, 1 only).
Runs: 3 SAE seeds (0 1 2) x 3 training-pool draws (pool seed 0 1 2) = 9 runs per arm, tags D1 / D1p1 / D1p2 (D2 ...).

Token cue reconstruction (step 'cue'): the token stores keep only today's patches, and the token cue is a subset of
    today's mask, so dist is recomputed for every stored patch (store fp16 token vs the video's bg_seg, on the GPU, as
    the extraction did) and feat = drop_isolated(dist > thr) on stored patches. This is exact: a patch with dist > thr
    outside the store must be isolated (a non-isolated one is kept), and isolation of a stored patch depends only on
    stored neighbours. Check: on every 8th frame the dark cue is recomputed from the JPGs and today's rule applied to
    feat | dark core must give the store mask back (mismatches reported in masks.json).
Tokens: D1 / D2 patches inside today's mask come from the base store (token_t) and the d5 store (token_{t-5}); a
    today patch without a d5 counterpart gets zero change, exactly as M_today. Ants D1 patches OUTSIDE today's mask
    (non-dark neighbours of token-cue patches) have token_t and token_{t-5} encoded with the store's encoder path
    (mask_test.cmd_encode) -> OUT/ants/extra/ (deleted by 'clean').

PRE-REGISTERED DECISION (fixed before any D1 / D2 number was seen)
    M_dino is preferred over M_today iff in BOTH domains
      (i)  no label's mean frame-level cross-fitted AUROC (presence read-out, mean over the 9 pool x seed runs) is
           below M_today's 9-run mean by more than 0.02 (mice nose_nose, nose_tail; ants groom_any, groom_yellow (Y2F),
           groom_blue (B2F), onlid_yellow), AND
      (ii) mask recall is not worse by more than 2 points (mice patch recall vs SAM 2.1 + tails, all valid frames; ants
           strict tracker-centroid recall AND patch recall vs the pixel + tracker mask, as mask_test).
    If it passes, also reported: whether it beats M_pix on ants (9-run mean cf AUROC per label). M_dino_cur is scored
    with the same rule and reported alongside.

Steps (scripts/eci/mask_test2.sh)
    cue       (GPU) token cue + D1 / D2 masks of all train + eval store frames, reconstruction check, encode plan
              -> OUT/<d>/mask/{masks.npz, masks.json}
    encode    (GPU, array; ants only) D1 tokens outside today's mask -> OUT/ants/extra/part_<task>/
    accuracy  (CPU) mask accuracy of M_today / M_pix / M_dino (/ M_dino_cur) + sheets -> OUT/<d>/accuracy.json, sheets/
    train     (GPU) one arm x 3 pools x 3 seeds -> OUT/<d>/sae/<tag>_s<seed>/
    evaluate  (GPU) one arm x 3 pools x 3 seeds -> OUT/<d>/{eval, codes_best}/
    readouts  (CPU) video level, 9-run tables, decision -> OUT/summary.json, OUT/tables.md
    clean     delete the encoded token parts
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import combined as cb  # noqa: E402  (scripts/eci/combined.py)
import mask_test as mt1  # noqa: E402  (scripts/eci/mask_test.py)
import tight_mask as tm  # noqa: E402
from src.eci.tight_mask import GRID, dilate_np  # noqa: E402

OUT = REPO / 'results/vision/eci_mask_test2'
MT = mt1.OUT          # results/vision/eci_mask_test (M_pix, M_today pool replicates)
CB = cb.OUT           # results/vision/eci_combined (C1 = M_today pool 0)
D = cb.D
DOMAINS = cb.DOMAINS
SEEDS = cb.SEEDS
POOLS = (0, 1, 2)
MASKS = {'dino': 'D1', 'dinocur': 'D2'}
NAMES = {'C1': 'M_today', 'P1': 'M_pix', 'D1': 'M_dino', 'D2': 'M_dino_cur'}
log = cb.log

# redirect mask_test's module paths and hooks to this test (functions read these globals at call time)
mt1.OUT = OUT
mt1.ARM_NAMES.update({'D1': 'M_dino', 'D2': 'M_dino_cur', **{f'D{a}p{p}': f'{NAMES[f"D{a}"]} pool {p}'
                                                            for a in (1, 2) for p in (1, 2)}})
unpack = mt1.unpack
local = mt1.local


def tag_of(mask, pool):
    return MASKS[mask] + ('' if pool == 0 else f'p{pool}')


# ---------------------------------------------------------------------------------------------- cue
def _dark_video(job):
    domain, obs, lpaths = job
    from PIL import Image
    pb = np.load(tm.BG_OLD[domain] / f'{obs}.npz')['pix_bg']
    out = np.zeros((len(lpaths), GRID * GRID), np.float32)
    for a in range(0, len(lpaths), 64):
        g = np.stack([np.asarray(Image.open(p).convert('RGB').convert('L'), np.uint8) for p in lpaths[a:a + 64]])
        out[a:a + 64] = tm.dark_frac(g, pb)
    return obs, out


def dark_of(domain, idx, frames, workers, tag):
    """(F, 1024) float32 dark fraction (foreground.dark_fraction semantics) of store frames, JPGs staged locally."""
    import motion_t3 as mt
    from multiprocessing import get_context
    paths = tm.frame_paths(domain)
    lp = mt.stage_files([str(REPO / 'dataset' / paths[r]) for r in idx.rows[frames]], local() / f'jpg_{tag}')
    groups = pd.Series(np.arange(len(frames))).groupby(idx.obs[frames]).indices
    jobs = [(domain, o, [lp[i] for i in ix]) for o, ix in groups.items()]
    out = np.zeros((len(frames), GRID * GRID), np.float32)
    with get_context('fork').Pool(workers) as pool:
        for o, a in pool.imap_unordered(_dark_video, jobs):
            out[groups[o]] = a
    shutil.rmtree(local() / f'jpg_{tag}')
    return out


def apply_rule(core, dark, rule):
    """foreground_parts dilation step on numpy masks."""
    grown = dilate_np(core, rule.get('dilate', 1))
    if rule.get('dilate_requires_dark', False):
        grown = core | (grown & (dark > 0))
    return grown


def cmd_cue(args):
    import spatial_sae_pilot as ssp
    for domain in args.domains.split(','):
        t0 = time.time()
        od = OUT / domain / 'mask'
        od.mkdir(parents=True, exist_ok=True)
        Zc = np.load(CB / domain / 'mask' / 'masks.npz')
        frames, is_ev = Zc['frames'], Zc['is_eval']
        idx = ssp.StoreIndex(domain)
        assert np.array_equal(idx.rows[frames], Zc['rows'])
        from src.eci.foreground import RULES
        rule = idx.store.info[0]['rule']
        assert all(i['rule'] == rule for i in idx.store.info), 'shards record different rules'
        names = [k for k, v in RULES.items() if v == rule]  # older shards record the rule dict, not its name
        assert names, f'store rule {rule} is none of foreground.RULES'
        rule_name = idx.store.info[0].get('rule_name', names[0])
        assert RULES[rule_name] == rule
        # reconstruction-check subset: every 8th frame; dark cue from the JPGs (CPU pool BEFORE any CUDA use)
        chk = np.arange(0, len(frames), 8)
        dark_chk = dark_of(domain, idx, frames[chk], args.workers, 'chk')
        log(f'{domain}: dark cue of {len(chk):,} check frames ({time.time() - t0:.0f}s)')
        raw, dist_stats = token_cue_raw(domain, idx, frames, rule)
        import torch
        from src.eci.foreground import drop_isolated
        feat = np.zeros_like(raw)
        for a in range(0, len(raw), 50_000):
            feat[a:a + 50_000] = drop_isolated(torch.from_numpy(raw[a:a + 50_000])).numpy()
        today = cb.base_masks(idx, frames)
        assert not (feat & ~today).any(), 'token cue outside the store mask'
        # check: today's rule on feat | dark core must give the store mask back
        rec = apply_rule(feat[chk] | (dark_chk > rule['dark_frac']), dark_chk, rule)
        mism = rec ^ today[chk]
        st = {'domain': domain, 'rule_name': rule_name, 'rule': rule, 'n_frames': int(len(frames)),
              'n_eval_frames': int(is_ev.sum()), 'dist': dist_stats,
              'check': {'n_frames': int(len(chk)), 'mismatch_patches': int(mism.sum()),
                        'frames_with_mismatch': int(mism.any(1).sum()),
                        'store_patches_checked': int(today[chk].sum()),
                        'rec_not_in_store': int((rec & ~today[chk]).sum()),
                        'store_not_in_rec': int((today[chk] & ~rec).sum())}}
        d1 = dilate_np(feat, 1)
        arms = {'feat': feat, 'dino': d1}
        extra = {}
        if rule.get('dilate_requires_dark', False):  # ants: D2 needs dark > 0 of every frame (mask_test's darkany)
            Zm = np.load(MT / domain / 'mask' / 'masks.npz')
            assert np.array_equal(Zm['frames'], frames)
            dany = unpack(Zm['darkany_bits'])
            st['check']['darkany_vs_recomputed_mismatch'] = int((dany[chk] ^ (dark_chk > 0)).sum())
            arms['dinocur'] = feat | (d1 & dany)
            assert not (arms['dinocur'] & ~today).any()
            extra['dinocur_bits'] = np.packbits(arms['dinocur'], axis=1)
        else:
            st['dinocur'] = 'identical to dino: the current dilation rule (fg448) adds every neighbour'
        d5 = cb.d5_masks(cb.d5_layout(idx, domain), idx, frames)
        need_t = d1 & ~today
        need_5 = d1 & ~today & ~d5
        n = {k: v.sum(1) for k, v in dict(today=today, raw_token_cue=raw, **arms).items()}
        st.update({
            'kept_per_frame': {k: {'all': float(v.mean()), 'train': float(v[~is_ev].mean()),
                                   'eval': float(v[is_ev].mean())} for k, v in n.items()},
            'frames_no_patch': {k: {'train': float((v[~is_ev] == 0).mean()), 'eval': float((v[is_ev] == 0).mean())}
                                for k, v in n.items()},
            'dino_inside_today_share': float((d1 & today).sum() / max(d1.sum(), 1)),
            'today_inside_dino_share': float((d1 & today).sum() / today.sum()),
            'tokens_token_t_to_encode': int(need_t.sum()), 'frames_token_t': int((need_t.sum(1) > 0).sum()),
            'tokens_token_t5_to_encode': int(need_5.sum()), 'frames_token_t5': int((need_5.sum(1) > 0).sum()),
            'today_without_d5_tokens': int((today & ~d5).sum())})
        if 'dinocur' in arms:
            st['today_inside_dinocur_share'] = float(arms['dinocur'].sum() / today.sum())
        st['elapsed_s'] = round(time.time() - t0, 1)
        np.savez_compressed(od / 'masks.npz', frames=frames, rows=Zc['rows'], is_eval=is_ev,
                            feat_bits=np.packbits(feat, axis=1), dino_bits=np.packbits(d1, axis=1),
                            new_only=need_t.sum(1).astype(np.int16), no_d5=need_5.sum(1).astype(np.int16),
                            needt_bits=np.packbits(need_t, axis=1), need5_bits=np.packbits(need_5, axis=1), **extra)
        (od / 'masks.json').write_text(json.dumps(st, indent=1))
        log(f'{domain}: {json.dumps(st)}')


def token_cue_raw(domain, idx, frames, rule):
    """(F, 1024) bool dist > thr_video on the stored patches of `frames` (sorted store frames), computed as the
    extraction did (FgBackgrounds.mask: 1 - cos(fp16 token as float32, bg_seg of the time block), GPU)."""
    import torch
    import torch.nn.functional as F
    from src.eci.foreground import load_background, segment_of
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    raw = np.zeros((len(frames), GRID * GRID), bool)
    groups = pd.Series(np.arange(len(frames))).groupby(idx.obs[frames]).indices
    order = sorted(groups, key=lambda o: groups[o][0])
    t0, n_tok, near = time.time(), 0, 0
    thrs = []
    for k, o in enumerate(order):
        ix = groups[o]
        fr = frames[ix]
        b = load_background(tm.BG_OLD[domain], o, rule, dev)
        assert b['align'] == 'none'
        thrs.append(b['thr'])
        tok, pos, ln = idx.load(fr)
        seg = segment_of(idx.rows[fr], b['rows'], b['seg_bounds'])
        fi = np.repeat(np.arange(len(fr)), ln)
        for a in range(0, len(tok), 200_000):
            tt = torch.from_numpy(tok[a:a + 200_000]).to(dev)
            sg = torch.from_numpy(seg[fi[a:a + 200_000]]).to(dev)
            pp = torch.from_numpy(pos[a:a + 200_000].astype(np.int64)).to(dev)
            d = 1 - F.cosine_similarity(tt.float(), b['bg'][sg, pp], dim=-1)
            hit = (d > b['thr']).cpu().numpy()
            near += int((torch.abs(d - b['thr']) < 1e-5).sum())
            raw[ix[fi[a:a + 200_000][hit]], pos[a:a + 200_000][hit]] = True
        n_tok += len(tok)
        if k % 50 == 0:
            log(f'  token cue {k + 1}/{len(order)} videos, {n_tok:,} tokens ({time.time() - t0:.0f}s)')
    return raw, {'n_tokens': n_tok, 'n_videos': len(order), 'within_1e-5_of_thr': near,
                 'thr_median': float(np.median(thrs)), 'thr_min': float(np.min(thrs)), 'thr_max': float(np.max(thrs))}


# ---------------------------------------------------------------------------------------------- tokens
def load_extras2(domain, frames_all, group):
    """This test's encoded tokens of `group` (copied to local disk, memory-mapped) -> {0: token_t, 1: token_t-5}
    PartExtra, keys = store frame * 1024 + pos."""
    E = {0: [], 1: []}
    mine = sorted((OUT / domain / 'extra').glob('part_*'))
    infos = [json.loads((d / 'info.json').read_text()) for d in mine]
    if not mine or len(mine) != infos[0]['n_tasks']:
        raise RuntimeError(f'{domain}: {len(mine)} encode parts present')
    for d, i in zip(mine, infos):
        if i['group'] != group:
            continue
        dst = local() / 'extras2' / d.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(d / 'tok.npy', dst / 'tok.npy')
        t = np.load(dst / 'tok.npy', mmap_mode='r')
        f = np.load(d / 'fidx.npy').astype(np.int64)
        p = np.load(d / 'pos.npy').astype(np.int64)
        kind = np.load(d / 'kind.npy')
        for k in (0, 1):
            m = np.flatnonzero(kind == k)
            if len(m):
                E[k].append((frames_all[f[m]] * 1024 + p[m], t, m.astype(np.int64)))
    ex = {k: mt1.PartExtra(E[k]) for k in (0, 1)}
    log(f'  extras2 (group {group}): token_t {len(ex[0].keys):,}, token_t-5 {len(ex[1].keys):,}')
    return ex


def stage_arm(domain, idx, frames, dst, group, mask, chunk=4000):
    """mask_test.stage_p1 for the D1 / D2 masks (replaces it via mt1.stage_p1): token_t from the base store, else an
    encoded extra; token_t-5 from the d5 store, else an encoded extra (patch outside today's mask), else zero change
    (today patch without d5 counterpart, as M_today). Staged once per job into a cache reused by every pool / call;
    `dst` is created empty (the callers remove it)."""
    dst.mkdir(parents=True, exist_ok=True)
    cache = local() / f'stage_{mask}_{group}'
    if (cache / 'info.json').exists():
        info = json.loads((cache / 'info.json').read_text())
        assert np.array_equal(np.load(cache / 'frames.npy'), frames)
        log(f'  staged {mask} tokens reused from {cache}')
        return (np.load(cache / 'tok.npy', mmap_mode='r'), np.load(cache / 'prev.npy', mmap_mode='r'),
                np.load(cache / 'pos.npy'), np.load(cache / 'lens.npy'), info)
    cache.mkdir(parents=True, exist_ok=True)
    Z = np.load(OUT / domain / 'mask' / 'masks.npz')
    k = np.searchsorted(Z['frames'], frames)
    assert (Z['frames'][k] == frames).all()
    bits = Z[f'{mask}_bits'][k]
    n_keep = cb.popcount(bits)
    n = int(n_keep.sum())
    need = 2 * n * D * 2
    free = shutil.disk_usage(cache).free
    if free < need + 20e9:
        raise RuntimeError(f'{cache}: {free / 1e9:.0f} GB free, staging needs {need / 1e9:.0f} GB + 20 GB')
    ex = load_extras2(domain, Z['frames'], group) if int(Z['new_only'].sum()) and mask == 'dino' else None
    m = cb.d5_layout(idx, domain)
    tok = np.lib.format.open_memmap(cache / 'tok.npy', 'w+', np.float16, (n, D))
    prev = np.lib.format.open_memmap(cache / 'prev.npy', 'w+', np.float16, (n, D))
    pos = np.empty(n, np.int16)
    lens = np.zeros(len(frames), np.int64)
    o, t0 = 0, time.time()
    cnt = {'token_t_encoded': 0, 'token_t5_encoded': 0, 'token_t5_zero_change': 0}
    for c0 in range(0, len(frames), chunk):
        fc = frames[c0:c0 + chunk]
        t, p, ln = idx.load(fc)
        new = np.unpackbits(bits[c0:c0 + chunk], axis=1).astype(bool)
        fi = np.repeat(np.arange(len(fc)), ln)
        keep = new[fi, p.astype(np.int64)]
        kb = fc[fi[keep]].astype(np.int64) * 1024 + p[keep]
        allk = (fc[:, None].astype(np.int64) * 1024 + np.arange(1024)[None])[new]
        miss = allk[~np.isin(allk, kb)]
        if len(miss) and ex is None:
            raise RuntimeError(f'{len(miss)} kept patches outside the store and no encoded extras')
        keys = np.concatenate([kb, miss])
        tt = np.concatenate([t[keep], ex[0].take(miss) if len(miss) else np.zeros((0, D), np.float16)])
        so = np.argsort(keys, kind='stable')
        keys, tt = keys[so], tt[so]
        assert np.array_equal(keys, allk)
        in_store = np.isin(keys, kb)
        kd, pv = cb.d5_chunk(m, idx, fc)
        j = np.searchsorted(kd, keys)
        ok = (j < len(kd)) & (kd[np.minimum(j, len(kd) - 1)] == keys)
        pp = np.empty_like(tt)
        pp[ok] = pv[j[ok]]
        enc5 = ~ok & ~in_store
        zero = ~ok & in_store
        if enc5.any():
            pp[enc5] = ex[1].take(keys[enc5])
        pp[zero] = tt[zero]
        cnt['token_t_encoded'] += len(miss)
        cnt['token_t5_encoded'] += int(enc5.sum())
        cnt['token_t5_zero_change'] += int(zero.sum())
        nn = len(keys)
        tok[o:o + nn], prev[o:o + nn], pos[o:o + nn] = tt, pp, (keys % 1024).astype(np.int16)
        lens[c0:c0 + chunk] = new.sum(1)
        o += nn
    assert o == n and (lens == n_keep).all()
    tok.flush()
    prev.flush()
    del tok, prev
    shutil.rmtree(local() / 'extras2', ignore_errors=True)
    info = {'mask': mask, 'n_frames': int(len(frames)), 'n_tokens': n, 'kept_per_frame': n / len(frames),
            'frames_no_token': float((lens == 0).mean()), **cnt, 'stage_s': round(time.time() - t0, 1)}
    np.save(cache / 'frames.npy', frames)
    np.save(cache / 'pos.npy', pos)
    np.save(cache / 'lens.npy', lens)
    (cache / 'info.json').write_text(json.dumps(info))
    log(f'  staged {mask} tokens: {json.dumps(info)} ({need / 1e9:.1f} GB token + prev)')
    return np.load(cache / 'tok.npy', mmap_mode='r'), np.load(cache / 'prev.npy', mmap_mode='r'), pos, lens, info


mt1.stage_p1 = stage_arm


# ---------------------------------------------------------------------------------------------- accuracy
def arm_masks2(domain, idx, store_frames):
    """{arm: (F, 1024) bool} of M_today (C1) / M_pix (P1) / M_dino (D1) / M_dino_cur (D2, ants) at store frames."""
    Zc = np.load(CB / domain / 'mask' / 'masks.npz')
    Zp = np.load(MT / domain / 'mask' / 'masks.npz')
    Z2 = np.load(OUT / domain / 'mask' / 'masks.npz')
    k = np.searchsorted(Zc['frames'], store_frames)
    assert (Zc['frames'][k] == store_frames).all(), 'frames outside the store frame set'
    assert np.array_equal(Zp['frames'], Zc['frames']) and np.array_equal(Z2['frames'], Zc['frames'])
    order = np.argsort(store_frames, kind='stable')
    today = np.empty((len(store_frames), GRID * GRID), bool)
    today[order] = cb.base_masks(idx, store_frames[order])
    M = {'C1': today, 'P1': unpack(Zp['bits'][k]), 'D1': unpack(Z2['dino_bits'][k])}
    if 'dinocur_bits' in Z2.files:
        M['D2'] = unpack(Z2['dinocur_bits'][k])
    return M


_sheets1 = mt1.sheets


def sheets2(od, domain, fpaths, obs, pick, tags, M, ref, cells, side=400, per=6):
    """mask_test.sheets with the panels of this test (M_today / M_pix / M_dino [/ M_dino_cur])."""
    from PIL import Image, ImageDraw
    from src.eci.tight_mask import PX
    arms = tuple(M)
    for q0 in range(0, len(pick), per):
        rows = pick[q0:q0 + per]
        S = Image.new('RGB', (len(arms) * side, len(rows) * (side + 18)), 'white')
        dr = ImageDraw.Draw(S)
        for q, i in enumerate(rows):
            base = Image.open(REPO / 'dataset' / fpaths[i]).convert('RGB')
            for col, arm in enumerate(arms):
                im = base.copy()
                ov = Image.new('RGBA', im.size, (0, 0, 0, 0))
                d2 = ImageDraw.Draw(ov)
                k, r_ = M[arm][i], ref[i]
                for p in range(GRID * GRID):
                    rr, cc = divmod(p, GRID)
                    box = [cc * PX, rr * PX, cc * PX + PX - 1, rr * PX + PX - 1]
                    if r_[p] and not k[p]:
                        d2.rectangle(box, fill=(255, 0, 0, 110), outline=(255, 0, 0, 255))
                    elif r_[p] and k[p]:
                        d2.rectangle([box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2], outline=(0, 220, 0, 255))
                    elif k[p]:
                        d2.rectangle(box, outline=(40, 120, 255, 255))
                if cells is not None:
                    for c_ in cells[i]:
                        if c_ >= 0:
                            y, x = divmod(int(c_), GRID)
                            d2.ellipse([x * PX + 5, y * PX + 5, x * PX + 11, y * PX + 11], fill=(255, 230, 0, 255))
                im = Image.alpha_composite(im.convert('RGBA'), ov).convert('RGB').resize((side, side))
                S.paste(im, (col * side, q * (side + 18)))
                dr.text((col * side + 2, q * (side + 18) + side + 2),
                        f'{NAMES[arm]} kept {int(k.sum())}, ref {int(r_.sum())}, missed {int((r_ & ~k).sum())} | '
                        f'{obs[i]} {tags[q0 + q]}', fill='black')
        p = od / f'mask_sheet_{q0 // per}.jpg'
        S.save(p, quality=88)
        log(f'  sheet {p}')


def cmd_accuracy(args):
    mt1.arm_masks = arm_masks2
    mt1.sheets = sheets2
    mt1.cmd_accuracy(args)


# ---------------------------------------------------------------------------------------------- train / evaluate
def runs_of(args):
    for pool in args.pools:
        yield argparse.Namespace(domain=args.domain, mask=args.mask, tag=tag_of(args.mask, pool), pool_seed=pool,
                                 seeds=args.seeds, steps=args.steps)


def cmd_train(args):
    for ns in runs_of(args):
        log(f'== train {ns.tag} (mask {ns.mask}, pool seed {ns.pool_seed})')
        mt1.cmd_train(ns)


def cmd_evaluate(args):
    for ns in runs_of(args):
        log(f'== evaluate {ns.tag} (mask {ns.mask}, pool seed {ns.pool_seed})')
        mt1.cmd_evaluate(ns)


# ---------------------------------------------------------------------------------------------- readouts / decision
METRICS = mt1.METRICS
FAMILY = {'M_today': [('C1', CB), ('T0p1', MT), ('T0p2', MT)],
          'M_pix': [('P1', MT), ('P1p1', MT), ('P1p2', MT)],
          'M_dino': [('D1', OUT), ('D1p1', OUT), ('D1p2', OUT)],
          'M_dino_cur': [('D2', OUT), ('D2p1', OUT), ('D2p2', OUT)]}
VKEYS = (('presence_mean', 'video_r_rate'), ('presence_mean', 'video_r_bpm'),
         ('presence_mean|fgctrl', 'video_r_rate'), ('presence_mean|fgctrl', 'video_r_bpm'))


def cmd_readouts(args):
    import readouts as rd
    import sae_levers as sl
    stage = local() / 'readouts'
    stage.mkdir(parents=True, exist_ok=True)
    S = {}
    for d in DOMAINS:
        labels = sl.LABELS[d]
        lab = pd.read_parquet(CB / d / 'labels.parquet')
        half = np.load(CB / d / 'half.npy')
        labo = pd.read_parquet(OUT / d / 'labels.parquet')
        assert (lab.obs.values == labo.obs.values).all() and (lab.frame_idx.values == labo.frame_idx.values).all()
        assert np.array_equal(half, np.load(OUT / d / 'half.npy'))
        gt = rd.mice_truth(stage) if d == 'mice' else rd.ants_truth()
        Dd = {'runs': {}, 'family': {}, 'arm_meta': {}}
        for fam, arms in FAMILY.items():
            arms = [(a, r) for a, r in arms if (r / d / 'eval' / f'{a}_s2.json').exists()]
            if not arms:
                continue
            R = {b: {q: [] for q in METRICS} for b in labels}
            V = {b: {f'{k}|{q}': [] for k, q in VKEYS} for b in labels}
            meta = {q: [] for q in ('fve', 'l0_per_token', 'dead_frac')}
            kept, fnt = [], []
            for a, root in arms:
                lens = np.load(root / d / 'codes_best' / f'{a}_lens.npy').astype(np.float32)
                for s in SEEDS:
                    e = json.loads((root / d / 'eval' / f'{a}_s{s}.json').read_text())
                    for q in meta:
                        meta[q].append(e[q])
                    kept.append(e.get('kept_per_frame', np.nan))
                    fnt.append(e.get('frames_no_token', np.nan))
                    for b in labels:
                        for q in METRICS:
                            R[b][q].append(e['presence'][b][q])
                        Rh = cb.per_half(root, d, a, s, b, lab, half, gt['wstart'], lens)
                        v = rd.video_check_mice(Rh, gt, b) if d == 'mice' else rd.video_check_ants(Rh, gt, b)
                        for k, q in VKEYS:
                            V[b][f'{k}|{q}'].append(v[k][q])
                        Dd['runs'][f'{a}|s{s}|{b}'] = {'cf_auroc': e['presence'][b]['cf_auroc'],
                                                       'cf_ap': e['presence'][b]['cf_ap']}
                log(f'{d} {fam} {a}: done')
            Dd['family'][fam] = {b: {**{q: cb.ms_(R[b][q]) for q in METRICS},
                                     **{k: cb.ms_(x) for k, x in V[b].items()},
                                     'pool_means': [float(np.mean(R[b]['cf_auroc'][3 * i:3 * i + 3]))
                                                    for i in range(len(arms))]}
                                 for b in labels}
            Dd['arm_meta'][fam] = {'arms': [a for a, _ in arms], 'n_runs': 3 * len(arms),
                                   'kept_per_frame': float(np.nanmean(kept)), 'frames_no_token': float(np.nanmean(fnt)),
                                   **{q: cb.ms_(x) for q, x in meta.items()}}
        if d == 'mice':  # M_dino_cur is the same mask as M_dino in mice
            Dd['family']['M_dino_cur'] = Dd['family']['M_dino']
            Dd['arm_meta']['M_dino_cur'] = dict(Dd['arm_meta']['M_dino'], note='same mask as M_dino (fg448 dilation)')
        S[d] = Dd
    S['accuracy'] = {d: json.loads((OUT / d / 'accuracy.json').read_text()) for d in DOMAINS}
    S['masks'] = {d: json.loads((OUT / d / 'mask' / 'masks.json').read_text()) for d in DOMAINS}
    S['decision'] = decide(S)
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    tables(S)


def decide(S):
    import sae_levers as sl
    arm_of = {'M_dino': 'D1', 'M_dino_cur': 'D2'}
    R = {}
    for fam in ('M_dino', 'M_dino_cur'):
        R[fam] = {}
        for d in DOMAINS:
            F = S[d]['family']
            per = {}
            for b in sl.LABELS[d]:
                da = F[fam][b]['cf_auroc']['mean'] - F['M_today'][b]['cf_auroc']['mean']
                dp = F[fam][b]['cf_auroc']['mean'] - F['M_pix'][b]['cf_auroc']['mean']
                per[b] = {'d_cf_auroc_vs_today': da, 'drop_gt_0.02': bool(da < -0.02), 'd_cf_auroc_vs_pix': dp}
            arm = arm_of[fam] if (d == 'ants' or fam == 'M_dino') else 'D1'
            ra = mt1.recall_measures(S['accuracy'][d], d, arm)
            r0 = mt1.recall_measures(S['accuracy'][d], d, 'C1')
            rec = {k: {'arm': ra[k], 'today': r0[k], 'd_points': 100 * (ra[k] - r0[k]),
                       'worse_gt_2pt': bool(100 * (ra[k] - r0[k]) < -2)} for k in ra}
            ok = not any(v['drop_gt_0.02'] for v in per.values()) and not any(v['worse_gt_2pt'] for v in rec.values())
            R[fam][d] = {'per_label': per, 'recall': rec, 'passes': bool(ok)}
        R[fam]['PASSES'] = all(R[fam][d]['passes'] for d in DOMAINS)
        R[fam]['beats_M_pix_on_ants_all_labels'] = all(v['d_cf_auroc_vs_pix'] > 0
                                                       for v in R[fam]['ants']['per_label'].values())
    R['FINAL'] = 'M_dino' if R['M_dino']['PASSES'] else 'M_today'
    return R


def tables(S):
    import sae_levers as sl
    f = lambda x: f"{x['mean']:.3f} ± {x['sd']:.3f}"  # noqa: E731
    f4 = lambda x: f"{x['mean']:.4f} ± {x['sd']:.4f}"  # noqa: E731
    L = ['# MASK test 2: M_dino (token cue + plain 1-patch dilation) / M_dino_cur (token cue + current dilation) vs '
         'M_today / M_pix; mean ± sd over pool x seed runs (9 per family; mice M_pix 6)\n']
    for d in DOMAINS:
        A, mj = S['accuracy'][d], S['masks'][d]
        L.append(f'\n## {d}\n')
        kp = mj['kept_per_frame']
        L.append('Kept patches / frame (eval): ' + ', '.join(f"{k} {v['eval']:.1f}" for k, v in kp.items())
                 + '; eval frames with no patch: ' + ', '.join(f"{k} {v['eval']:.4f}" for k, v in mj['frames_no_patch'].items())
                 + f"; M_dino inside M_today {mj['dino_inside_today_share']:.3f}, M_today inside M_dino "
                 f"{mj['today_inside_dino_share']:.3f}\n")
        L.append(f"Token-cue reconstruction check ({mj['check']['n_frames']} frames): {json.dumps(mj['check'])}; "
                 f"dist {json.dumps(mj['dist'])}\n")
        arms = [a for a in ('C1', 'P1', 'D1', 'D2') if a in A[('sam_tails' if d == 'mice' else 'anchors')]]
        L.append('### Mask accuracy\n')
        if d == 'mice':
            L.append('SAM 2.1 + tails\n')
            L.append('| stratum | arm | kept/frame | patch recall | precision | far share | frames animal < 50% | '
                     'frames animal missed | tail-only recall |')
            L.append('|---|---|---|---|---|---|---|---|---|')
            for s in A['sam_tails']['C1']:
                for a in arms:
                    m = A['sam_tails'][a][s]
                    L.append(f"| {s} | {NAMES[a]} | {m['kept_per_frame']:.1f} | {m['patch_recall']:.4f} | "
                             f"{m['patch_precision']:.4f} | {m['far_share_of_kept']:.4f} | "
                             f"{m['frames_any_animal_under_half']:.4f} | {m['frames_any_animal_missed']:.4f} | "
                             f"{m['tail_only_patch_recall']:.4f} |")
        else:
            L.append('| stratum | arm | centroid recall strict | lenient | frames any ant missed strict | '
                     'pixel+tracker recall | precision | far share | SAM B1nz_zoom recall | SAM B1n recall |')
            L.append('|---|---|---|---|---|---|---|---|---|---|')
            for s in A['anchors']['C1']:
                for a in arms:
                    m, q = A['anchors'][a][s], A['pixel_tracker'][a][s]
                    sz = A['sam']['B1nz_zoom'][a].get(s, {}).get('patch_recall', float('nan'))
                    sn = A['sam']['B1n'][a].get(s, {}).get('patch_recall', float('nan'))
                    L.append(f"| {s} | {NAMES[a]} | {m['anchor_recall_strict']:.4f} | {m['anchor_recall_lenient']:.4f} | "
                             f"{m['frames_any_ant_missed_strict']:.4f} | {q['patch_recall']:.4f} | "
                             f"{q['patch_precision']:.4f} | {q['far_share_of_kept']:.4f} | {sz:.4f} | {sn:.4f} |")
        L.append('\n### Downstream (C1 recipe, presence read-out; all pool x seed runs)\n')
        L.append('| label | family | runs | cf AUROC | pool means | cf AP | honest top-1% | size-ctrl AUROC | '
                 'video r rate raw / ctl | video r bouts raw / ctl |')
        L.append('|---|---|---|---|---|---|---|---|---|---|')
        for b in sl.LABELS[d]:
            for fam, x in S[d]['family'].items():
                x = x[b]
                n = len(x['cf_auroc']['per_seed'])
                L.append(f"| {b} | {fam} | {n} | {f(x['cf_auroc'])} | " + ', '.join(f'{v:.3f}' for v in x['pool_means'])
                         + f" | {f4(x['cf_ap'])} | {f(x['cf_top1'])} | {f(x['cf_size_ctrl_auroc'])} | "
                         f"{x['presence_mean|video_r_rate']['mean']:.2f} / {x['presence_mean|fgctrl|video_r_rate']['mean']:.2f} | "
                         f"{x['presence_mean|video_r_bpm']['mean']:.2f} / {x['presence_mean|fgctrl|video_r_bpm']['mean']:.2f} |")
        L.append('\n' + '; '.join(f"{fam}: kept/frame {m['kept_per_frame']:.1f}, frames no token "
                                  f"{m['frames_no_token']:.4f}, FVE {m['fve']['mean']:.4f}, dead {m['dead_frac']['mean']:.3f}"
                                  for fam, m in S[d]['arm_meta'].items()) + '\n')
    L.append('\n## Pre-registered decision\n')
    L.append('```\n' + json.dumps(S['decision'], indent=1, default=float) + '\n```')
    (OUT / 'tables.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def cmd_clean(args):
    for d in DOMAINS:
        for p in sorted((OUT / d / 'extra').glob('part_*')):
            for f in ('tok.npy', 'fidx.npy', 'pos.npy', 'kind.npy'):
                if (p / f).exists():
                    (p / f).unlink()
            log(f'cleaned {p}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['cue', 'encode', 'accuracy', 'train', 'evaluate', 'readouts', 'clean'])
    p.add_argument('--domain', default='mice', choices=list(DOMAINS))
    p.add_argument('--domains', default='mice,ants')
    p.add_argument('--mask', default='dino', choices=list(MASKS))
    p.add_argument('--pools', type=int, nargs='+', default=list(POOLS))
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--n-tasks', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--num-workers', type=int, default=12)
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--steps', type=int, default=6000)
    a = p.parse_args()
    if a.cmd == 'encode':
        mt1.cmd_encode(a)  # reads OUT/<d>/mask/masks.npz (new_only / no_d5 / needt_bits / need5_bits), writes OUT/<d>/extra
        return
    {'cue': cmd_cue, 'accuracy': cmd_accuracy, 'train': cmd_train, 'evaluate': cmd_evaluate,
     'readouts': cmd_readouts, 'clean': cmd_clean}[a.cmd](a)


if __name__ == '__main__':
    main()
