"""
MASK test: does the ECI animal mask need the DINOv2 token cue, or does a pixel-only rule do as well?
(ants v2/v3 + mice v1, the COMBINED C1 representation; scripts/eci/combined.py is the harness reused here)

Arms (identical frames, splits and SAE recipe = COMBINED C1: Matryoshka BatchTopK 4096 latents / k 16 on
[token_t, token_t - token_{t-5}] (DINOv2-base 448 tokens, 1 s motion at 5 fps), TokenNorm per 768 block,
motion_t3.train_one, 6000 steps, pool cap mice 4M / ants 2M tokens, pool seed 0, seeds 0 1 2)
    M_today      today's rule (src/eci/foreground.py: token cue OR dark cue, then the 1-patch dilation; mice 'fg448' onto
                 every neighbour, ants 'ants' only onto neighbours holding a dark pixel) = COMBINED arm C1. Reused:
                 checkpoints results/vision/eci_t3_motion/<d>/sae/M5_s*, evaluation results/vision/eci_combined/<d>/.
    M_pix_nodil  the T2b pixel core alone, no dilation = COMBINED arm C2 (reused, same place).
    M_pix        arm 'P1', trained and evaluated here: the T2b pixel core + EXACTLY today's dilation:
                     mice   core | 3x3-dilate(core)                       (fg448: every 8-neighbour)
                     ants   core | (3x3-dilate(core) & dark > 0)          (ants: only neighbours with >= 1 dark pixel)
                 dark = foreground.dark_fraction (grey < 60 AND > 40 below the video's 0.98-quantile pixel background
                 dataset/<d>/eci/fg448/background/<obs>.npz pix_bg) - a pixel quantity; no DINOv2 anywhere in the mask.
T2b core (scripts/eci/combined.py choose; fixed on TRAIN videos before any evaluation, results/vision/eci_combined/
    rule.json): keep a patch iff >= X of its 16 x 16 pixels differ from the per-time-block median background image by
    more than delta grey levels in ABSOLUTE value (darker or lighter); delta = ceil(6 x sigma), sigma = median over the
    train videos of the 1.4826 x MAD pixel noise, floored at 1 grey level, raised in steps of 2 until some X in
    {5, 10, 15, 20, 25}% keeps <= 0.5% of empty-floor patches; X = the smallest such. Mice: sigma 2.97 -> delta 30,
    X 25%; ants: sigma 0 (floor 1) -> delta 6, X 5%. The dilation has no parameter (copied from today's rules).

Mask accuracy (step 'accuracy', no behaviour label chooses anything; contact labels only split the report)
    mice  reference = SAM 2.1 instance masks of the 20,000-frame B subset (results/vision/eci_mice_pairs/bsub, commit
          ec82118) extended by the neuron_map_v3 'line' tail cue: results/vision/eci_neuron_map_v3/mice/prep_line.npz
          'bits' (mouse k overlaps the patch iff >= 26 px of its extended mask or >= 10 of its tail pixels fall in it),
          'tonly' (tail-only patches). Frames: the 16,775 'valid' frames of neuron_map_v2 (splitter correct, 4 masks).
    ants  (a) TRACKER ANCHORS (dataset/ants/<exp>/tracking/<obs>.csv focal / yellow / blue body centroids,
              gap-filled): an ant is covered iff the patch holding its centroid is kept (lenient: any of its 3 x 3
              patches). Independent of every mask cue. All 76,800 eval frames with the 3 centroids.
          (b) PIXEL + TRACKER animal mask of neuron_map_v2 (results/vision/eci_neuron_map_v2/ants/prep.npz 'bits'):
              patches with >= 26 px differing by > 30 grey levels from the block median background, in 8-connected
              components holding a tracked ant. It SHARES the pixel-difference cue with the pixel rules (at delta 30,
              not 6) and is therefore biased toward them; reported with that caveat.
          (c) SAM 2.1 masks of the instance-split test (results/vision/eci_split_test/ants/labels_B1nz_zoom.npz, per-ant
              zoom crops, union of the 3 instances; patch = >= 26 SAM px) on its eval frames that are store frames.
              SAM often returns gaster-only ant masks there (split-test sheets): recall is meaningful, precision is not.
    Measures: patch recall = kept reference patches / reference patches; precision = kept patches that are reference
    patches / kept patches; far share = kept patches > 1 patch away from every reference patch; per animal the share
    of its reference patches kept -> frames with any animal < 50% covered, and with any animal entirely missed.
    Strata: contact (mice nose_nose | nose_tail; ants groom_any) / non-contact; ants also yellow on lid (v3).
    Visual sheets: 12 frames per domain (seed 0), one row per frame, panels M_today / M_pix / M_pix_nodil.

PRE-REGISTERED DECISION (written before any P1 or accuracy number was seen)
    A pixel-only arm A in {M_pix_nodil, M_pix} is NOT WORSE than M_today iff in BOTH domains
      (i)  no label's frame-level cross-fitted AUROC (presence read-out, 3-seed mean) is below M_today's by > 0.02
           (mice nose_nose, nose_tail; ants groom_any, groom_yellow (Y2F), groom_blue (B2F), onlid_yellow), AND
      (ii) mask recall is not worse by more than 2 points: mice patch recall vs SAM + tails (all valid frames); ants
           strict anchor recall (all 3 ants, all eval frames) AND patch recall vs the pixel + tracker mask.
    The simplest not-worse arm wins: M_pix_nodil, else M_pix; if neither, M_today stays.

RESULTS (2026-10-15; results/vision/eci_mask_test/tables.md, summary.json): M_today STAYS (no pixel-only arm is not worse
    in both domains).
    Masks (eval kept / frame): mice today 149, M_pix 194, M_pix_nodil 90; ants today 50.0, M_pix 47.1, M_pix_nodil 47.1 -
    on ants the T2b core (delta 6, X 5%) already holds every dark neighbour, so today's dilation adds 0.001 patches /
    frame (M_pix = M_pix_nodil to 22 of 3.6M eval tokens).
    Accuracy, mice vs SAM + tails (16,775 frames): patch recall today 0.966 / M_pix 0.985 / nodil 0.897; tail-only
    recall 0.839 / 0.931 / 0.551; precision 0.428 / 0.330 / 0.656; kept patches > 1 patch from any mouse 11.5% / 30.1% /
    10.1% (M_pix admits the white odour object and disturbed bedding); frames with a mouse < 50% covered 0.05% / 0.22% /
    0.57%. Same on contact and non-contact frames. Ants: strict centroid recall 0.992 / 1.000 / 1.000 (lenient 1.000 all),
    pixel + tracker recall 0.976 / 1.000 / 1.000 (shared cue), SAM B1nz union recall 0.986 / 0.999 / 0.999; on-lid
    frames the same. The pixel masks cover the animals at least as well as today's in both domains.
    Downstream cf AUROC (3 seeds, pool 0) today / M_pix / nodil: mice nn 0.688 / 0.674 / 0.671, nt 0.731 / 0.721 /
    0.720; ants groom_any 0.902 / 0.810 / 0.809, Y2F 0.853 / 0.781 / 0.781, B2F 0.861 / 0.789 / 0.787, on-lid 0.965 /
    0.738 / 0.903. Decision: M_pix passes mice (d AUROC -0.015 / -0.010, recall +1.9 pt) and fails ants (all 4 labels
    > 0.02 below); M_pix_nodil fails both (mice recall -6.8 pt; ants all labels).
    Pool-seed sensitivity (added after the first ants P1 run showed on-lid 0.738 for a mask identical to C2's 0.903;
    pools re-drawn, same seeds): ants 9 runs each, today / M_pix: groom_any 0.866 / 0.829, Y2F 0.825 / 0.798, B2F 0.832 /
    0.792, on-lid 0.960 ± 0.011 / 0.815 ± 0.104 (pool means today 0.965 0.961 0.953, pixel 0.738 0.910 0.797) - the
    ants loss is real, pixel-mask on-lid is unstable across pools; the 3-seed gap on grooming (0.07-0.09) is about half
    that over pools (0.027-0.040). Mice 6 runs each: nn 0.678 / 0.671, nt 0.733 / 0.724 (within noise).

Steps (scripts/eci/mask_test.sh)
    masks     (CPU) P1 masks of all train + eval store frames, the encode plan -> OUT/<d>/mask/{masks.npz, masks.json}
    accuracy  (CPU) reference metrics + sheets -> OUT/<d>/accuracy.json, OUT/<d>/sheets/
    encode    (GPU, array) tokens no existing store / extra holds -> OUT/<d>/extra/part_<task>/ (deleted at the end)
    train     (GPU) P1 x seeds -> OUT/<d>/sae/P1_s<seed>/
    evaluate  (GPU) P1 x seeds -> OUT/<d>/{eval, codes_best}/
    readouts  (CPU) video level, tables, decision -> OUT/summary.json, OUT/tables.md
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import combined as cb  # noqa: E402  (scripts/eci/combined.py)
import tight_mask as tm  # noqa: E402  (scripts/eci/tight_mask.py)
from src.eci.tight_mask import GRID, PX, dilate_np  # noqa: E402

OUT = REPO / 'results/vision/eci_mask_test'
CB = cb.OUT
T2 = cb.T2
T3 = cb.T3
V2 = REPO / 'results/vision/eci_neuron_map_v2'
V3 = REPO / 'results/vision/eci_neuron_map_v3'
SPLIT = REPO / 'results/vision/eci_split_test'
BG_OLD = tm.BG_OLD
D = cb.D
DOMAINS = cb.DOMAINS
SEEDS = cb.SEEDS
ARM_NAMES = {'C1': 'M_today', 'C2': 'M_pix_nodil', 'P1': 'M_pix',
             'T0p1': 'M_today pool 1', 'T0p2': 'M_today pool 2', 'P1p1': 'M_pix pool 1', 'P1p2': 'M_pix pool 2'}
# pool-seed replicates (sensitivity check added after the first ants P1 result; not part of the decision):
# the same mask, recipe and SAE seeds, only the 2M / 4M-token training pool re-drawn (pool seed 1, 2)
REPLICATES = ('T0p1', 'T0p2', 'P1p1', 'P1p2')
FAMILY = {'M_today': ('C1', 'T0p1', 'T0p2'), 'M_pix': ('P1', 'P1p1', 'P1p2')}
log = cb.log


def unpack(bits):
    return np.unpackbits(bits, axis=1).astype(bool)


def local():
    d = os.environ.get('LOCAL_DIR')
    if not d:
        raise RuntimeError('LOCAL_DIR not set (job staging dir)')
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------- masks
def _dark_video(job):
    obs, lpaths = job
    pb = np.load(BG_OLD['ants'] / f'{obs}.npz')['pix_bg']
    anyb = np.zeros((len(lpaths), GRID * GRID // 8), np.uint8)
    core = np.zeros_like(anyb)
    for a in range(0, len(lpaths), 64):
        from PIL import Image
        g = np.stack([np.asarray(Image.open(p).convert('RGB').convert('L'), np.uint8) for p in lpaths[a:a + 64]])
        dk = tm.dark_frac(g, pb)
        anyb[a:a + 64] = np.packbits(dk > 0, axis=1)
        core[a:a + 64] = np.packbits(dk > 0.15, axis=1)
    return obs, anyb, core


def ants_dark(idx, frames, workers):
    """(F, 1024) bool dark > 0 and dark > 0.15 of the store frames (JPGs staged to local disk first)."""
    import motion_t3 as mt
    from multiprocessing import get_context
    paths = tm.frame_paths('ants')
    rows = idx.rows[frames]
    lp = mt.stage_files([str(REPO / 'dataset' / paths[r]) for r in rows], local() / 'jpg')
    groups = pd.Series(np.arange(len(frames))).groupby(idx.obs[frames]).indices
    jobs = [(o, [lp[i] for i in ix]) for o, ix in groups.items()]
    anyb = np.zeros((len(frames), GRID * GRID // 8), np.uint8)
    core = np.zeros_like(anyb)
    t0 = time.time()
    with get_context('fork').Pool(workers) as pool:
        for k, (o, a, c) in enumerate(pool.imap_unordered(_dark_video, jobs)):
            anyb[groups[o]], core[groups[o]] = a, c
            if k % 32 == 0:
                log(f'  dark {k + 1}/{len(jobs)} videos ({time.time() - t0:.0f}s)')
    shutil.rmtree(local() / 'jpg')
    return unpack(anyb), unpack(core)


def cmd_masks(args):
    import spatial_sae_pilot as ssp
    for domain in args.domains.split(','):
        t0 = time.time()
        od = OUT / domain / 'mask'
        od.mkdir(parents=True, exist_ok=True)
        Zc = np.load(CB / domain / 'mask' / 'masks.npz')
        frames, is_ev = Zc['frames'], Zc['is_eval']
        idx = ssp.StoreIndex(domain)
        assert np.array_equal(idx.rows[frames], Zc['rows'])
        core = unpack(Zc['bits'])                       # T2b (= C2's mask)
        today = cb.base_masks(idx, frames)              # store mask (= C1's mask)
        d5 = cb.d5_masks(cb.d5_layout(idx, domain), idx, frames)
        grown = dilate_np(core, 1)
        st = {'domain': domain, 'rule_core': cb.rule_of(domain), 'n_frames': int(len(frames)),
              'n_eval_frames': int(is_ev.sum())}
        extra = {}
        if domain == 'mice':
            p1 = grown
            st['dilation'] = 'fg448: every 8-neighbour of a core patch'
        else:
            dany, dcore = ants_dark(idx, frames, args.workers)
            p1 = core | (grown & dany)
            st['dilation'] = "ants: 8-neighbours of a core patch holding >= 1 dark pixel (dark_fraction > 0)"
            # sanity: today's dark core (dark > 0.15) must sit inside today's store mask
            st['check_dark_core_inside_today'] = float((dcore & today).sum() / max(dcore.sum(), 1))
            st['check_dark_any_patches_per_frame'] = float(dany.sum(1).mean())
            extra = {'darkany_bits': np.packbits(dany, axis=1)}
        need_t = p1 & ~today & ~core   # token_t: not in the base store, not among the T2b extras
        need_5 = p1 & ~d5 & ~core      # token_t-5: not in the d5 store, not among the T2b extras
        n = {k: v.sum(1) for k, v in (('today', today), ('pix_nodil', core), ('pix', p1))}
        np.savez_compressed(od / 'masks.npz', frames=frames, bits=np.packbits(p1, axis=1), rows=Zc['rows'],
                            is_eval=is_ev, n_new=n['pix'].astype(np.int16), new_only=need_t.sum(1).astype(np.int16),
                            no_d5=need_5.sum(1).astype(np.int16), needt_bits=np.packbits(need_t, axis=1),
                            need5_bits=np.packbits(need_5, axis=1), **extra)
        st.update({
            'kept_per_frame': {k: {'all': float(v.mean()), 'train': float(v[~is_ev].mean()), 'eval': float(v[is_ev].mean())}
                               for k, v in n.items()},
            'pix_inside_today_share': float((p1 & today).sum() / p1.sum()),
            'today_inside_pix_share': float((p1 & today).sum() / today.sum()),
            'jaccard_pix_vs_today': float((p1 & today).sum() / (p1 | today).sum()),
            'tokens_token_t_to_encode': int(need_t.sum()), 'frames_token_t': int((need_t.sum(1) > 0).sum()),
            'tokens_token_t5_to_encode': int(need_5.sum()), 'frames_token_t5': int((need_5.sum(1) > 0).sum()),
            'elapsed_s': round(time.time() - t0, 1)})
        (od / 'masks.json').write_text(json.dumps(st, indent=1))
        log(f'{domain}: {json.dumps(st)}')


# ---------------------------------------------------------------------------------------------- accuracy
def ref_metrics(kept, ref_bits, ks, sel, tonly=None):
    """kept (F, 1024) bool, ref_bits (F, 1024) int instance bit field, ks = instance bits, sel (F,) bool frames."""
    kept, rb = kept[sel], ref_bits[sel]
    ref = rb > 0
    near = dilate_np(ref, 1)
    out = {'n_frames': int(sel.sum()), 'kept_per_frame': float(kept.sum(1).mean()),
           'ref_per_frame': float(ref.sum(1).mean()),
           'patch_recall': float((kept & ref).sum() / max(ref.sum(), 1)),
           'patch_precision': float((kept & ref).sum() / max(kept.sum(), 1)),
           'far_share_of_kept': float((kept & ~near).sum() / max(kept.sum(), 1))}
    cov = np.full((len(kept), len(ks)), np.nan)
    for j, k in enumerate(ks):
        rk = (rb & k) > 0
        nk = rk.sum(1)
        cov[nk > 0, j] = (kept & rk).sum(1)[nk > 0] / nk[nk > 0]
    has = np.isfinite(cov)
    out['frames_any_animal_under_half'] = float((np.where(has, cov, 1) < 0.5).any(1).mean())
    out['frames_any_animal_missed'] = float((np.where(has, cov, 1) == 0).any(1).mean())
    out['mean_animal_coverage'] = float(np.nanmean(cov))
    if tonly is not None:
        t = tonly[sel] > 0
        out['tail_only_patch_recall'] = float((kept & t).sum() / max(t.sum(), 1))
        out['tail_only_patches_per_frame'] = float(t.sum(1).mean())
    return out


def anchor_metrics(kept, cells, sel):
    """cells (F, 3) patch index of each ant's centroid (-1 missing); strict = own patch kept, lenient = 3 x 3."""
    kept, cells = kept[sel], cells[sel]
    ok = (cells >= 0).all(1)
    kept, cells = kept[ok], cells[ok]
    K = kept.reshape(-1, GRID, GRID)
    r, c = np.divmod(cells, GRID)
    fi = np.arange(len(K))[:, None]
    strict = K[fi, r, c]
    len_ = np.zeros_like(strict)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            len_ |= K[fi, np.clip(r + dr, 0, GRID - 1), np.clip(c + dc, 0, GRID - 1)]
    return {'n_frames': int(len(K)), 'anchor_recall_strict': float(strict.mean()),
            'anchor_recall_lenient': float(len_.mean()),
            'per_ant_strict': {a: float(strict[:, i].mean()) for i, a in enumerate(('focal', 'yellow', 'blue'))},
            'frames_any_ant_missed_strict': float((~strict).any(1).mean()),
            'frames_any_ant_missed_lenient': float((~len_).any(1).mean())}


def arm_masks(domain, idx, store_frames):
    """{arm: (F, 1024) bool} of today / P1 / C2 at the given store frames."""
    Zc = np.load(CB / domain / 'mask' / 'masks.npz')
    Zp = np.load(OUT / domain / 'mask' / 'masks.npz')
    k = np.searchsorted(Zc['frames'], store_frames)
    assert (Zc['frames'][k] == store_frames).all(), 'frames outside the store frame set'
    assert np.array_equal(Zp['frames'], Zc['frames'])
    order = np.argsort(store_frames, kind='stable')
    today = np.empty((len(store_frames), GRID * GRID), bool)
    today[order] = cb.base_masks(idx, store_frames[order])
    return {'C1': today, 'P1': unpack(Zp['bits'][k]), 'C2': unpack(Zc['bits'][k])}


def strata(contact, extra=None):
    s = {'all': np.ones(len(contact), bool), 'contact': contact, 'noncontact': ~contact}
    s.update(extra or {})
    return s


def cmd_accuracy(args):
    import spatial_sae_pilot as ssp
    for domain in args.domains.split(','):
        t0 = time.time()
        idx = ssp.StoreIndex(domain)
        R = {'domain': domain, 'arms': ARM_NAMES}
        od = OUT / domain
        (od / 'sheets').mkdir(parents=True, exist_ok=True)
        if domain == 'mice':
            F = pd.read_parquet(V2 / 'mice' / 'frames.parquet')
            P = np.load(V3 / 'mice' / 'prep_line.npz')
            valid = np.load(V2 / 'mice' / 'prep.npz')['valid']
            M = arm_masks(domain, idx, F.store_frame.values)
            contact = (F.nose_nose.values | F.nose_tail.values)
            R['contact_agrees_with_bsub_set'] = float((contact == (F.set.values == 'contact')).mean())
            S = {k: v & valid for k, v in strata(contact).items()}
            ref = P['bits'].astype(np.int64)
            R['reference'] = 'SAM 2.1 + line tail cue (neuron_map_v3 prep_line bits), valid frames'
            R['sam_tails'] = {arm: {s: ref_metrics(M[arm], ref, (1, 2, 4, 8), m, P['tonly']) for s, m in S.items()}
                              for arm in M}
            R['n_valid'] = int(valid.sum())
            rng = np.random.default_rng(0)
            pick = np.r_[np.sort(rng.choice(np.flatnonzero(S['contact']), 6, replace=False)),
                         np.sort(rng.choice(np.flatnonzero(S['noncontact']), 6, replace=False))]
            tags = ['contact'] * 6 + ['non-contact'] * 6
            sheets(od / 'sheets', domain, F.frame_path.values, F.obs.values, pick, tags, M, ref > 0, None)
        else:
            F = pd.read_parquet(V2 / 'ants' / 'frames.parquet')
            P = np.load(V2 / 'ants' / 'prep.npz')
            M = arm_masks(domain, idx, F.store_frame.values)
            contact = F.groom_any.values.astype(bool)
            onlid = F.onlid_yellow.values.astype(bool) & F.valid_onlid_yellow.values.astype(bool)
            not_onlid = ~F.onlid_yellow.values.astype(bool) & F.valid_onlid_yellow.values.astype(bool)
            S = strata(contact, {'onlid_yellow': onlid, 'v3_not_onlid': not_onlid})
            # (a) tracker anchors
            cells = np.full((len(F), 3), -1, np.int64)
            for (ex, o), g in F.groupby(['experiment', 'obs']):
                t = pd.read_csv(REPO / f'dataset/ants/{ex}/tracking/{o}.csv').set_index('frame_idx').reindex(g.frame_idx.values)
                for j, a in enumerate(('focal', 'yellow', 'blue')):
                    x, y = t[f'{a}_x'].values, t[f'{a}_y'].values
                    okp = np.isfinite(x) & np.isfinite(y)
                    cc = np.full(len(g), -1, np.int64)
                    cc[okp] = (np.clip(np.floor(y[okp] / PX), 0, GRID - 1) * GRID
                               + np.clip(np.floor(x[okp] / PX), 0, GRID - 1)).astype(np.int64)
                    cells[g.index.values, j] = cc
            R['anchors'] = {arm: {s: anchor_metrics(M[arm], cells, m) for s, m in S.items()} for arm in M}
            R['anchor_frames_with_3_centroids'] = float((cells >= 0).all(1).mean())
            # (b) pixel + tracker reference (neuron_map_v2)
            ref = P['bits'].astype(np.int64)
            R['pixel_tracker'] = {arm: {s: ref_metrics(M[arm], ref, (1, 2, 4), m) for s, m in S.items()} for arm in M}
            # (c) SAM 2.1 split-test masks (union of instances), store frames only
            E = pd.read_parquet(SPLIT / 'ants' / 'eval.parquet')
            row2f = pd.Series(np.arange(len(idx.rows)), index=idx.rows)
            sf = row2f.reindex(E.row.values).values
            Zc = np.load(CB / domain / 'mask' / 'masks.npz')
            ins = np.isfinite(sf)
            ins[ins] = np.isin(sf[ins].astype(np.int64), Zc['frames'])
            R['sam_frames_in_store'] = int(ins.sum())
            R['sam'] = {}
            for meth in ('B1nz_zoom', 'B1n'):
                L = np.load(SPLIT / 'ants' / f'labels_{meth}.npz')['labels']
                cnt = np.stack([(L == k).reshape(len(L), GRID, PX, GRID, PX).sum((2, 4)).reshape(len(L), -1)
                                for k in (1, 2, 3)], -1)[ins]
                sref = ((cnt >= 26) * np.array([1, 2, 4])).sum(-1)
                Ms = arm_masks(domain, idx, sf[ins].astype(np.int64))
                c_ = E.groom_any.values[ins].astype(bool)
                R['sam'][meth] = {arm: {s: ref_metrics(Ms[arm], sref, (1, 2, 4), m) for s, m in strata(c_).items()}
                                  for arm in Ms}
            rng = np.random.default_rng(0)
            pick = np.r_[np.sort(rng.choice(np.flatnonzero(contact), 4, replace=False)),
                         np.sort(rng.choice(np.flatnonzero(~contact), 4, replace=False)),
                         np.sort(rng.choice(np.flatnonzero(onlid), 4, replace=False))]
            tags = ['groom contact'] * 4 + ['non-contact'] * 4 + ['yellow on lid'] * 4
            sheets(od / 'sheets', domain, F.frame_path.values, F.obs.values, pick, tags, M, ref > 0, cells)
        R['elapsed_s'] = round(time.time() - t0, 1)
        (od / 'accuracy.json').write_text(json.dumps(R, indent=1))
        log(f'{domain}: accuracy -> {od / "accuracy.json"} ({R["elapsed_s"]}s)')
        print(json.dumps(acc_summary(R), indent=1))


def acc_summary(R):
    keys = ('patch_recall', 'patch_precision', 'far_share_of_kept', 'frames_any_animal_under_half',
            'frames_any_animal_missed', 'tail_only_patch_recall', 'anchor_recall_strict', 'anchor_recall_lenient',
            'frames_any_ant_missed_strict', 'frames_any_ant_missed_lenient', 'kept_per_frame')
    out = {}
    for refn in ('sam_tails', 'anchors', 'pixel_tracker'):
        if refn in R:
            out[refn] = {f'{arm}|{s}': {k: round(v, 4) for k, v in m.items() if k in keys}
                         for arm, d in R[refn].items() for s, m in d.items()}
    return out


def sheets(od, domain, fpaths, obs, pick, tags, M, ref, cells, side=400, per=6):
    """Rows = frames, panels M_today / M_pix / M_pix_nodil. Reference animal patch kept: green box; missed: red
    filled; kept but not reference: blue box. Ants: tracker centroids as yellow dots."""
    from PIL import Image, ImageDraw
    arms = ('C1', 'P1', 'C2')
    s = side / 512
    for q0 in range(0, len(pick), per):
        rows = pick[q0:q0 + per]
        S = Image.new('RGB', (3 * side, len(rows) * (side + 18)), 'white')
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
                        f'{ARM_NAMES[arm]} kept {int(k.sum())}, ref {int(r_.sum())}, missed {int((r_ & ~k).sum())} | '
                        f'{obs[i]} {tags[q0 + q]}', fill='black')
        p = od / f'mask_sheet_{q0 // per}.jpg'
        S.save(p, quality=88)
        log(f'  sheet {p}')


# ---------------------------------------------------------------------------------------------- encode
def cmd_encode(args):
    import torch
    import motion_t3 as mt
    import spatial_sae_pilot as ssp
    from src.eci.foreground import FgEncoder, FrameDatasetFG, encode_batch
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    loc = local() / f'enc_{args.task}'
    loc.mkdir(parents=True, exist_ok=True)
    domain = args.domain
    Z = np.load(OUT / domain / 'mask' / 'masks.npz')
    idx = ssp.StoreIndex(domain)
    frames = Z['frames']
    f_t = np.flatnonzero(Z['new_only'] > 0)
    f_5 = np.flatnonzero(Z['no_d5'] > 0)
    row_t = idx.rows[frames[f_t]]
    row_5, paths_all = mt.nbr_rows(domain, idx, frames[f_5], -5)
    it_f = np.r_[f_t, f_5].astype(np.int64)
    it_k = np.r_[np.zeros(len(f_t), np.int8), np.ones(len(f_5), np.int8)]
    it_r = np.r_[row_t, row_5].astype(np.int64)
    # tasks 0 .. n/2-1 encode the TRAIN frames' items, n/2 .. n-1 the EVAL frames' (a t-5 row is in its frame's video)
    assert args.n_tasks % 2 == 0, 'n_tasks must be even (train / eval halves)'
    half_n = args.n_tasks // 2
    group = 'train' if args.task < half_n else 'eval'
    in_g = Z['is_eval'][it_f] == (group == 'eval')
    urows = np.unique(it_r[in_g])
    mine_rows = np.array_split(urows, half_n)[args.task % half_n]
    sel = np.flatnonzero(np.isin(it_r, mine_rows) & in_g)
    sel = sel[np.lexsort((it_k[sel], it_f[sel], it_r[sel]))]
    it_f, it_k, it_r = it_f[sel], it_k[sel], it_r[sel]
    bitsT, bits5 = Z['needt_bits'], Z['need5_bits']
    posl = [np.flatnonzero(np.unpackbits((bitsT if k == 0 else bits5)[f])) for f, k in zip(it_f, it_k)]
    n = int(sum(len(p) for p in posl))
    rr = np.searchsorted(mine_rows, it_r)
    log(f'{domain} task {args.task}/{args.n_tasks}: {len(urows):,} rows overall, {len(mine_rows):,} here; {len(sel):,} '
        f'items ({int((it_k == 0).sum()):,} token_t, {int((it_k == 1).sum()):,} token_t-5), {n:,} tokens')
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
    o, i0, t0 = 0, 0, time.time()
    chk = {'t_vs_store_max_abs': 0.0, 't5_vs_d5prev_max_abs': 0.0, 'n_t': 0, 'n_5': 0}
    for b, (pix, _) in enumerate(dl):
        T = encode_batch(enc.model, pix, dev).cpu().numpy()
        for kk in range(len(pix)):
            for it in range(starts[i0 + kk], starts[i0 + kk + 1]):
                p = posl[it]
                tok[o:o + len(p)] = T[kk, p]
                fidx[o:o + len(p)], pos[o:o + len(p)], kind[o:o + len(p)] = it_f[it], p, it_k[it]
                o += len(p)
                if b < 3:  # bit identity against the stores on the same frames
                    fs = np.array([frames[it_f[it]]])
                    st_, ps, _ = idx.load(fs)
                    if it_k[it] == 0 and len(ps):
                        chk['t_vs_store_max_abs'] = max(chk['t_vs_store_max_abs'], float(np.abs(
                            st_.astype(np.float32) - T[kk, ps.astype(np.int64)].astype(np.float32)).max()))
                        chk['n_t'] += len(ps)
                    elif it_k[it] == 1:
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
    info = {'domain': domain, 'task': args.task, 'n_tasks': args.n_tasks, 'group': group, 'n_rows': int(len(mine_rows)), 'n_tokens': n,
            'n_token_t': int((kind == 0).sum()), 'n_token_t5': int((kind == 1).sum()), 'checks': chk,
            'elapsed_s': round(time.time() - t0, 1), 'gpu': torch.cuda.get_device_name(0)}
    (loc / 'info.json').write_text(json.dumps(info, indent=1))
    dst = OUT / domain / 'extra' / f'part_{args.task}'
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(loc, dst, dirs_exist_ok=True)
    shutil.rmtree(loc)
    log(f'done -> {dst}: {json.dumps(info)}')


# ---------------------------------------------------------------------------------------------- P1 tokens
class PartExtra:
    """combined.Extra over several token arrays (in RAM or local memmaps): keys sorted, (array, row) per key."""

    def __init__(self, entries):
        self.toks = [t for _, t, _ in entries]
        keys = np.concatenate([k for k, _, _ in entries]) if entries else np.zeros(0, np.int64)
        src = np.concatenate([np.full(len(k), i, np.int32) for i, (k, _, _) in enumerate(entries)]) if entries else np.zeros(0, np.int32)
        off = np.concatenate([o for _, _, o in entries]) if entries else np.zeros(0, np.int64)
        o = np.argsort(keys, kind='stable')
        self.keys, self.src, self.off = keys[o], src[o], off[o]
        assert (np.diff(self.keys) > 0).all(), 'duplicate extra keys'

    def take(self, q):
        j = np.searchsorted(self.keys, q)
        ok = (j < len(self.keys)) & (self.keys[np.minimum(j, len(self.keys) - 1)] == q)
        assert ok.all(), f'{int((~ok).sum())} kept patches have no encoded token'
        out = np.empty((len(q), D), np.float16)
        s_, o_ = self.src[j], self.off[j]
        for s in np.unique(s_):
            m = np.flatnonzero(s_ == s)
            r = np.argsort(o_[m], kind='stable')
            out[m[r]] = self.toks[s][o_[m][r]]
        return out


def load_extras(domain, frames_all, group):
    """Every encoded token outside the stores: T2 extras (mice token_t) and COMBINED extras (C2) in RAM, this test's
    extras of `group` ('train' / 'eval') copied to local disk and memory-mapped -> {kind: PartExtra} (0 token_t,
    1 token_t-5), keys = store frame * 1024 + pos."""
    E = {0: [], 1: []}
    srcs = []
    if cb.rule_of(domain)['equals_t2_rule']:
        Z2 = np.load(T2 / domain / 'mask' / 'masks.npz')
        assert np.array_equal(Z2['frames'], frames_all)
        srcs += [(d, False) for d in sorted((T2 / domain / 'extra').glob('part_*'))]
    srcs += [(d, False) for d in sorted((CB / domain / 'extra').glob('part_*'))]
    mine = sorted((OUT / domain / 'extra').glob('part_*'))
    if mine:
        infos = [json.loads((d / 'info.json').read_text()) for d in mine]
        if len(mine) != infos[0]['n_tasks']:
            raise RuntimeError(f'{domain}: {len(mine)}/{infos[0]["n_tasks"]} encode tasks present')
        srcs += [(d, True) for d, i in zip(mine, infos) if i['group'] == group]
    n_tok = 0
    for d, mm in srcs:
        f = np.load(d / 'fidx.npy').astype(np.int64)
        p = np.load(d / 'pos.npy').astype(np.int64)
        if mm:
            dst = local() / 'extras' / d.name
            dst.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(dst).free
            need = (d / 'tok.npy').stat().st_size
            if free < need + 20e9:
                raise RuntimeError(f'{dst}: {free / 1e9:.0f} GB free, extras need {need / 1e9:.0f} GB + 20 GB')
            shutil.copyfile(d / 'tok.npy', dst / 'tok.npy')
            t = np.load(dst / 'tok.npy', mmap_mode='r')
        else:
            t = np.load(d / 'tok.npy')
        kind = np.load(d / 'kind.npy') if (d / 'kind.npy').exists() else np.zeros(len(f), np.int8)
        for k in (0, 1):
            m = np.flatnonzero(kind == k)
            if len(m):
                E[k].append((frames_all[f[m]] * 1024 + p[m], t, m.astype(np.int64)))
        n_tok += len(f)
    ex = {k: PartExtra(E[k]) for k in (0, 1)}
    log(f'  extras ({len(srcs)} parts, group {group}, {n_tok:,} tokens): token_t {len(ex[0].keys):,}, '
        f'token_t-5 {len(ex[1].keys):,}')
    return ex


def stage_p1(domain, idx, frames, dst, group, mask='pix', chunk=4000):
    """combined.stage_c2 for the P1 mask: token_t from the base store else an extra, token_t-5 from the d5 store else
    an extra -> (tok memmap, prev memmap, pos, lens, info)."""
    Z = np.load(OUT / domain / 'mask' / 'masks.npz')
    k = np.searchsorted(Z['frames'], frames)
    assert (Z['frames'][k] == frames).all()
    bits = Z['bits'][k] if mask == 'pix' else np.packbits(cb.base_masks(idx, frames), axis=1)
    n_keep = cb.popcount(bits)
    n = int(n_keep.sum())
    dst.mkdir(parents=True, exist_ok=True)
    need = 2 * n * D * 2
    free = shutil.disk_usage(dst).free
    if free < need + 20e9:
        raise RuntimeError(f'{dst}: {free / 1e9:.0f} GB free, staging needs {need / 1e9:.0f} GB + 20 GB')
    ex = load_extras(domain, Z['frames'], group) if mask == 'pix' else None
    m = cb.d5_layout(idx, domain)
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
        allk = (fc[:, None].astype(np.int64) * 1024 + np.arange(1024)[None])[new]
        miss = allk[~np.isin(allk, kb)]
        keys = np.concatenate([kb, miss])
        tt = np.concatenate([t[keep], ex[0].take(miss) if len(miss) else np.zeros((0, D), np.float16)])
        so = np.argsort(keys, kind='stable')
        keys, tt = keys[so], tt[so]
        assert np.array_equal(keys, allk)
        kd, pv = cb.d5_chunk(m, idx, fc)
        j = np.searchsorted(kd, keys)
        ok = (j < len(kd)) & (kd[np.minimum(j, len(kd) - 1)] == keys)
        pp = np.empty_like(tt)
        pp[ok] = pv[j[ok]]
        if mask == 'pix':
            pp[~ok] = ex[1].take(keys[~ok])
        else:  # today's mask: base patches without a d5 counterpart get zero change, as COMBINED C1 / motion_t3
            pp[~ok] = tt[~ok]
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
            'token_t_not_in_store': n_t_enc, 'token_t5_not_in_d5': n_5_enc, 'stage_s': round(time.time() - t0, 1)}
    log(f'  staged P1 tokens: {json.dumps(info)} ({need / 1e9:.1f} GB token + prev)')
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
    loc = local()
    idx = ssp.StoreIndex(args.domain)
    tr, _, _, _, train_v, _ = idx.split(args.domain, ms.N_TRAIN[args.domain], 0)
    tr = np.sort(tr)
    # same train frames as C1 / C2 (frames with >= 1 kept patch under today's mask, sae_levers split)
    c2meta = json.loads((CB / args.domain / 'train_meta_C2.json').read_text())
    assert c2meta['n_train_frames'] == len(tr)
    tok, prev, pos, lens, info = stage_p1(args.domain, idx, tr, loc / 'train', 'train', args.mask)
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
    shutil.rmtree(loc / 'extras', ignore_errors=True)
    zero = float(sum(int((X[a:a + 500_000, D:].float().abs().sum(1) == 0).sum()) for a in range(0, len(X), 500_000))
                 / len(X))
    rn = float((X[:200_000, D:].float().norm(dim=1) / X[:200_000, :D].float().norm(dim=1)).median())
    meta.update(domain=args.domain, arm=args.tag, mask=args.mask, rule=cb.rule_of(args.domain),
                n_train_frames=int(len(tr)), n_train_tokens=int(n), cap=int(cap), pool_seed=args.pool_seed,
                steps=args.steps, stage=info, zero_change_frac=zero, median_change_over_token_norm=rn,
                tokens_per_frame=n / len(tr), c2_n_train_tokens=c2meta['n_train_tokens'],
                frames_with_no_token=float((lens == 0).mean()), pool_first_idx=sel[:5].tolist())
    out.mkdir(parents=True, exist_ok=True)
    (out / f'train_meta_{args.tag}.json').write_text(json.dumps(meta, indent=1))
    log(f'P1 train: {n:,} tokens ({n / len(tr):.1f}/frame), pool {cap:,}, zero-change {100 * zero:.2f}%, median '
        f'|change|/|token| {rn:.3f}')
    for seed in args.seeds:
        mt.train_one(args.tag, seed, X, args.steps, out / 'sae' / f'{args.tag}_s{seed}',
                     {'config': args.tag, 'arm': {'delta': 5, 'kind': 'back'}, 'motion_delta': 5, 'domain': args.domain,
                      'cap': cap, 'pool_seed': args.pool_seed, 'blocks': [D, D], 'mask': args.mask,
                      'rule': cb.rule_of(args.domain)}, dev)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- evaluate
def cmd_evaluate(args):
    import torch
    import multiscale_sae as ms
    import sae_levers as sl
    import spatial_sae_pilot as ssp
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    domain = args.domain
    out = OUT / domain
    loc = local()
    lout = loc / 'out'
    for d in ('eval', 'codes_best'):
        (lout / d).mkdir(parents=True, exist_ok=True)
    idx = ssp.StoreIndex(domain)
    _, ev, _, _, _, _ = idx.split(domain, ms.N_TRAIN[domain], 0)
    ev = np.sort(ev)
    S = np.load(T2 / domain / 'mask' / 'sites.npz')
    assert (S['frames'] == ev).all()
    lab, _, valid = ms.eval_labels(domain, idx, ev)
    labels = sl.LABELS[domain]
    unit = lab.obs.str.rsplit('_', n=2).str[0].values if domain == 'mice' else lab.obs.values
    units = sorted(set(unit))
    half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
    assert (half == np.load(CB / domain / 'half.npy')).all(), 'halves differ from COMBINED'
    Yt = torch.from_numpy(lab[labels].values.astype(bool)).to(dev)
    tok, prev, pos, lens, st_info = stage_p1(domain, idx, ev, loc / 'eval', 'eval', args.mask)  # local memmaps
    shutil.rmtree(loc / 'extras', ignore_errors=True)
    X = cb.MotionTokens(tok, prev)
    rank = np.argsort(np.argsort(lens + np.random.default_rng(0).random(len(lens)) * 1e-3))
    decile = np.minimum(rank * 10 // len(lens), 9)
    fid = np.repeat(np.arange(len(lens)), lens)
    on = S['site'][fid, pos.astype(np.int64)] > 0
    dk = S['dark'][fid, pos.astype(np.int64)] > 0
    arm_meta = {'arm': args.tag, 'kept_per_frame': float(lens.mean()), 'frames_no_token': float((lens == 0).mean()),
                'token_off_animal_share': float(1 - on.mean()), 'token_no_dark_pixel_share': float(1 - dk.mean()),
                'stage': st_info}
    log(f'  [{args.tag}] {json.dumps(arm_meta)}')
    np.save(lout / 'codes_best' / f'{args.tag}_lens.npy', lens.astype(np.int16))
    for seed in args.seeds:
        t0 = time.time()
        key = f'{args.tag}_s{seed}'
        pth = out / 'sae' / key / 'sae.pt'
        sae, norm, ck = load_sae(pth, dev)
        assert sae.d_in == 2 * D
        pres, ext, stt = cb.encode_pe(sae, norm, X, lens, dev)
        resP, _ = sl.score_model(pres, Yt, labels, valid, half, decile, dev)
        resE, _ = sl.score_model(ext, Yt, labels, valid, half, decile, dev)
        e = {'arm': args.tag, 'pool_seed': args.pool_seed, 'seed': seed, 'ckpt': str(pth.relative_to(REPO)), **stt, **arm_meta, 'presence': resP,
             'extent': resE}
        js = sorted({d[s]['neuron'] for R in (resP, resE) for b in labels for d in R[b]['dirs'] for s in ('auc', 'ap', 'top1')})
        jt = torch.tensor(js, device=dev)
        np.savez_compressed(lout / 'codes_best' / f'{key}.npz', neurons=np.array(js),
                            presence=pres[:, jt].cpu().numpy(), extent=ext[:, jt].cpu().numpy())
        (lout / 'eval' / f'{key}.json').write_text(json.dumps(e, indent=1, default=float))
        f = lambda R, b: f"{R[b]['cf_auroc']:.3f}/{R[b]['cf_ap']:.4f}/{R[b]['cf_top1']:.3f}/{R[b]['cf_size_ctrl_auroc']:.3f}"  # noqa: E731
        log(f'  {key}: FVE {stt["fve"]:.4f} L0 {stt["l0_per_token"]:.2f} dead {stt["dead_frac"]:.3f} | cf AUROC/AP/top1/'
            f'size-ctrl: ' + ' | '.join(f'{b} {f(resP, b)}' for b in labels) + f' ({time.time() - t0:.0f}s)')
        del pres, ext
        torch.cuda.empty_cache()
    del tok, prev, X
    shutil.rmtree(loc / 'eval', ignore_errors=True)
    lab.drop(columns=['frame_path']).to_parquet(lout / 'labels.parquet')
    np.save(lout / 'half.npy', half)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(lout, out, dirs_exist_ok=True)
    log(f'results copied to {out}')


# ---------------------------------------------------------------------------------------------- readouts / decision
ARMDIR = {'C1': (CB, 'C1'), 'P1': (OUT, 'P1'), 'C2': (CB, 'C2')}
METRICS = ('cf_auroc', 'cf_ap', 'cf_ap_at_auroc', 'cf_top1', 'cf_size_ctrl_auroc')


def cmd_readouts(args):
    import readouts as rd
    import sae_levers as sl
    stage = local() / 'readouts'
    stage.mkdir(parents=True, exist_ok=True)
    S = {}
    for d in DOMAINS:
        labels = sl.LABELS[d]
        lab = pd.read_parquet(CB / d / 'labels.parquet')
        labp = pd.read_parquet(OUT / d / 'labels.parquet')
        assert (lab.obs.values == labp.obs.values).all() and (lab.frame_idx.values == labp.frame_idx.values).all()
        half = np.load(CB / d / 'half.npy')
        assert np.array_equal(half, np.load(OUT / d / 'half.npy'))
        gt = rd.mice_truth(stage) if d == 'mice' else rd.ants_truth()
        Dd = {'frame': {}, 'video': {}, 'arm_meta': {}}
        armdir = dict(ARMDIR, **{t: (OUT, t) for t in REPLICATES if (OUT / d / 'eval' / f'{t}_s2.json').exists()})
        Dd['arms'] = list(armdir)
        for arm, (root, prefix) in armdir.items():
            E = [json.loads((root / d / 'eval' / f'{prefix}_s{s}.json').read_text()) for s in SEEDS]
            lens = np.load(root / d / 'codes_best' / f'{prefix}_lens.npy').astype(np.float32)
            Dd['arm_meta'][arm] = {k: E[0].get(k) for k in ('kept_per_frame', 'frames_no_token', 'token_off_animal_share',
                                                            'token_no_dark_pixel_share')}
            Dd['arm_meta'][arm].update({q: cb.ms_([e[q] for e in E]) for q in ('fve', 'l0_per_token', 'dead_frac')})
            for b in labels:
                Dd['frame'][f'{arm}|{b}'] = {q: cb.ms_([e['presence'][b][q] for e in E]) for q in METRICS}
                Dd['frame'][f'{arm}|{b}']['base_rate'] = E[0]['presence'][b]['base_rate']
                V = []
                for s in SEEDS:
                    R = cb.per_half(root, d, prefix, s, b, lab, half, gt['wstart'], lens)
                    V.append(rd.video_check_mice(R, gt, b) if d == 'mice' else rd.video_check_ants(R, gt, b))
                Dd['video'][f'{arm}|{b}'] = {k: {q: cb.ms_([v[k][q] for v in V]) for q in V[0][k] if q != 'effects'}
                                             for k in V[0]}
            log(f'{d} {arm}: done')
        # pooled over pool replicates: every run of a mask family (3 pools x 3 seeds when all present)
        Dd['pooled'] = {}
        for fam, arms in FAMILY.items():
            arms = [a for a in arms if a in armdir]
            for b in labels:
                v = [x for a in arms for x in Dd['frame'][f'{a}|{b}']['cf_auroc']['per_seed']]
                Dd['pooled'][f'{fam}|{b}'] = {'arms': arms, 'cf_auroc': cb.ms_(v),
                                             'cf_ap': cb.ms_([x for a in arms for x in Dd['frame'][f'{a}|{b}']['cf_ap']['per_seed']]),
                                             'pool_means': [Dd['frame'][f'{a}|{b}']['cf_auroc']['mean'] for a in arms]}
        S[d] = Dd
    S['accuracy'] = {d: json.loads((OUT / d / 'accuracy.json').read_text()) for d in DOMAINS}
    S['decision'] = decide(S)
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    tables(S)


def recall_measures(A, d, arm):
    if d == 'mice':
        return {'sam_tails_patch_recall': A['sam_tails'][arm]['all']['patch_recall']}
    return {'anchor_recall_strict': A['anchors'][arm]['all']['anchor_recall_strict'],
            'pixel_tracker_patch_recall': A['pixel_tracker'][arm]['all']['patch_recall']}


def decide(S):
    import sae_levers as sl
    R = {}
    for arm in ('C2', 'P1'):
        R[arm] = {}
        for d in DOMAINS:
            per = {}
            for b in sl.LABELS[d]:
                da = S[d]['frame'][f'{arm}|{b}']['cf_auroc']['mean'] - S[d]['frame'][f'C1|{b}']['cf_auroc']['mean']
                per[b] = {'d_cf_auroc': da, 'drop_gt_0.02': bool(da < -0.02)}
            ra, r0 = recall_measures(S['accuracy'][d], d, arm), recall_measures(S['accuracy'][d], d, 'C1')
            rec = {k: {'arm': ra[k], 'today': r0[k], 'd_points': 100 * (ra[k] - r0[k]),
                       'worse_gt_2pt': bool(100 * (ra[k] - r0[k]) < -2)} for k in ra}
            ok = not any(v['drop_gt_0.02'] for v in per.values()) and not any(v['worse_gt_2pt'] for v in rec.values())
            R[arm][d] = {'per_label': per, 'recall': rec, 'not_worse': bool(ok)}
        R[arm]['NOT_WORSE'] = all(R[arm][d]['not_worse'] for d in DOMAINS)
    final = 'C2' if R['C2']['NOT_WORSE'] else ('P1' if R['P1']['NOT_WORSE'] else 'C1')
    R['FINAL'] = final
    R['FINAL_name'] = ARM_NAMES[final]
    return R


def tables(S):
    import sae_levers as sl
    f = lambda x: f"{x['mean']:.3f} ± {x['sd']:.3f}"  # noqa: E731
    f4 = lambda x: f"{x['mean']:.4f} ± {x['sd']:.4f}"  # noqa: E731
    L = ['# MASK test: M_today (C1) / M_pix (P1: T2b core + today dilation) / M_pix_nodil (C2), mean ± sd over 3 seeds\n']
    for d in DOMAINS:
        A = S['accuracy'][d]
        mj = json.loads((OUT / d / 'mask' / 'masks.json').read_text())
        L.append(f'\n## {d}\n')
        L.append(f"T2b core delta {mj['rule_core']['delta']}, X {mj['rule_core']['frac']:.2f}; dilation {mj['dilation']}; "
                 'kept patches / frame (eval): ' + ', '.join(f"{k} {v['eval']:.1f}" for k, v in mj['kept_per_frame'].items())
                 + f"; M_pix inside M_today {mj['pix_inside_today_share']:.3f}, M_today inside M_pix "
                 f"{mj['today_inside_pix_share']:.3f}\n")
        L.append('### Mask accuracy\n')
        refs = ('sam_tails',) if d == 'mice' else ('anchors', 'pixel_tracker')
        for refn in refs:
            sts = list(A[refn]['C1'])
            if refn == 'anchors':
                L.append(f'\n{refn} (tracker centroids; strict = own patch kept, lenient = 3 x 3)\n')
                L.append('| stratum | arm | n frames | kept/frame | anchor recall strict | lenient | frames any ant missed strict | lenient |')
                L.append('|---|---|---|---|---|---|---|---|')
                for s in sts:
                    for arm in ARMDIR:
                        m = A[refn][arm][s]
                        L.append(f"| {s} | {ARM_NAMES[arm]} | {m['n_frames']} | - | {m['anchor_recall_strict']:.4f} | "
                                 f"{m['anchor_recall_lenient']:.4f} | {m['frames_any_ant_missed_strict']:.4f} | "
                                 f"{m['frames_any_ant_missed_lenient']:.4f} |")
            else:
                L.append(f'\n{refn}\n')
                L.append('| stratum | arm | n frames | kept/frame | ref/frame | patch recall | precision | far share | '
                         'frames animal < 50% | frames animal missed |' + (' tail-only recall |' if d == 'mice' else ''))
                L.append('|---|---|---|---|---|---|---|---|---|---|' + ('---|' if d == 'mice' else ''))
                for s in sts:
                    for arm in ARMDIR:
                        m = A[refn][arm][s]
                        row = (f"| {s} | {ARM_NAMES[arm]} | {m['n_frames']} | {m['kept_per_frame']:.1f} | "
                               f"{m['ref_per_frame']:.1f} | {m['patch_recall']:.4f} | {m['patch_precision']:.4f} | "
                               f"{m['far_share_of_kept']:.4f} | {m['frames_any_animal_under_half']:.4f} | "
                               f"{m['frames_any_animal_missed']:.4f} |")
                        if d == 'mice':
                            row += f" {m['tail_only_patch_recall']:.4f} |"
                        L.append(row)
        if d == 'ants':
            for meth, X in A['sam'].items():
                L.append(f'\nSAM 2.1 split-test {meth} union ({A["sam_frames_in_store"]} store frames; recall only meaningful)\n')
                L.append('| stratum | arm | patch recall | frames animal < 50% |')
                L.append('|---|---|---|---|')
                for s in X['C1']:
                    for arm in ARMDIR:
                        m = X[arm][s]
                        L.append(f"| {s} | {ARM_NAMES[arm]} | {m['patch_recall']:.4f} | {m['frames_any_animal_under_half']:.4f} |")
        L.append('\n### Downstream (C1 recipe, presence read-out)\n')
        L.append('| label | arm | cf AUROC | cf AP (AP pick) | honest top-1% | size-ctrl AUROC | video r rate raw / ctl | '
                 'video r bouts raw / ctl |')
        L.append('|---|---|---|---|---|---|---|---|')
        for b in sl.LABELS[d]:
            for arm in S[d]['arms']:
                x = S[d]['frame'][f'{arm}|{b}']
                v, c = S[d]['video'][f'{arm}|{b}']['presence_mean'], S[d]['video'][f'{arm}|{b}']['presence_mean|fgctrl']
                L.append(f"| {b} ({x['base_rate']:.4f}) | {ARM_NAMES[arm]} | {f(x['cf_auroc'])} | {f4(x['cf_ap'])} | "
                         f"{f(x['cf_top1'])} | {f(x['cf_size_ctrl_auroc'])} | {v['video_r_rate']['mean']:.2f} / "
                         f"{c['video_r_rate']['mean']:.2f} | {v['video_r_bpm']['mean']:.2f} / {c['video_r_bpm']['mean']:.2f} |")
        L.append('\n' + '; '.join(f"{ARM_NAMES[a]}: kept/frame {m['kept_per_frame']:.1f}, FVE {m['fve']['mean']:.4f}, "
                                  f"dead {m['dead_frac']['mean']:.3f}" for a, m in S[d]['arm_meta'].items()) + '\n')
        if any(len(v['arms']) > 1 for v in S[d]['pooled'].values()):
            L.append('\nPool-seed sensitivity (frame cf AUROC; all runs of a mask family = pools x 3 seeds; pool means listed):\n')
            L.append('| label | family | runs | cf AUROC mean ± sd | pool means | cf AP mean |')
            L.append('|---|---|---|---|---|---|')
            for b in sl.LABELS[d]:
                for fam in FAMILY:
                    x = S[d]['pooled'][f'{fam}|{b}']
                    L.append(f"| {b} | {fam} | {len(x['cf_auroc']['per_seed'])} | {f(x['cf_auroc'])} | "
                             + ', '.join(f'{v:.3f}' for v in x['pool_means']) + f" | {x['cf_ap']['mean']:.4f} |")
    L.append('\n## Pre-registered decision\n')
    L.append('```\n' + json.dumps(S['decision'], indent=1, default=float) + '\n```')
    (OUT / 'tables.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def cmd_clean(args):
    """Delete this test's encoded token parts (large); keep their info.json."""
    for d in DOMAINS:
        for p in sorted((OUT / d / 'extra').glob('part_*')):
            for f in ('tok.npy', 'fidx.npy', 'pos.npy', 'kind.npy'):
                if (p / f).exists():
                    (p / f).unlink()
            log(f'cleaned {p}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['masks', 'accuracy', 'encode', 'train', 'evaluate', 'readouts', 'clean'])
    p.add_argument('--domain', default='mice', choices=list(DOMAINS))
    p.add_argument('--domains', default='mice,ants')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--n-tasks', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--num-workers', type=int, default=12)
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--pool-seed', type=int, default=0)
    p.add_argument('--tag', default='P1', help="arm name of train / evaluate outputs (replicates: e.g. P1p1, T0p1)")
    p.add_argument('--mask', default='pix', choices=['pix', 'today'], help="token mask of train / evaluate")
    a = p.parse_args()
    {'masks': cmd_masks, 'accuracy': cmd_accuracy, 'encode': cmd_encode, 'train': cmd_train, 'evaluate': cmd_evaluate,
     'readouts': cmd_readouts, 'clean': cmd_clean}[a.cmd](a)


if __name__ == '__main__':
    main()
