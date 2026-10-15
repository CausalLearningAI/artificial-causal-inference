"""
MABe22 mouse ECI - FALSE-POSITIVE FIGURE of the best unsupervised oral_oral neuron (arm D2, SAE seed 0, neuron 11).

Question: why does no single neuron isolate oral-oral (nose-to-nose) sniffing? Are its false positives (a) sniffing-like
contacts the annotators did not label, or (b) other contacts (head-to-body, body-to-body, huddles) it cannot separate?

Neuron and frames. The neuron is the one the harness picked for oral_oral by AUROC on clip half 0 of split seed 0
(eval/D2_s0.json m4096 split0 dirs[0]['auc'], asserted = 11). EVERY frame shown or counted here is from the held-out
half 1 of split seed 0 (never used to pick the neuron), as in eci_sheets.py. Activation = the frame code of the neuron
(max over the frame's kept 448 tokens; codes/D2_s0_codes.npy); peak patch = its argmax token (codes/D2_s0_argtok.npy),
mapped to the 32 x 32 grid through the frame's mask positions. The original token store (with pos448) was deleted, so
the 'arena' mask is RECOMPUTED for the clips of rows 1-2 exactly as eci_encode.py did it (same plan frames, lags
1 s + extra lags 6 / 15 / 60, DINOv2-base fp32 TF32 off, RTX 3090 like the original run); a frame is used only if its
recomputed kept-token count equals the stored D2_lens (checked and reported for every frame of those clips).

PRE-REGISTERED RULES (fixed before any number was looked at):
  Panel a, three rows of 6 crops (at most 1 frame per clip in a row; ties broken by eval-table order):
    row 1  "Labelled sniffing, neuron fires":  the 6 highest-activation half-1 frames with oral_oral = 1.
    row 2  "Not labelled, neuron fires":       the 6 highest-activation half-1 frames with oral_oral = 0.
    row 3  "Labelled sniffing, neuron silent": half-1 frames with oral_oral = 1 and activation 0, in the order of
                                               numpy default_rng(0).permutation, first 6 distinct clips.
    Crop = 56 x 56 px of the 224 video frame (pad white outside the frame), centred on the peak patch (rows 1-2) or on
    the midpoint of the closest nose-nose pair (row 3). Rings = noses of all mice (keypoints); box = peak patch.
  Keypoints (analysis and display only, never a feature): mouse_submission_keypoints.npy, (1800, 3 mice, 12 points,
    yx) on a 512 px scale, index = label index; video frame = label index + 1 (plan/offset.json); a point at (0, 0) is
    missing. Point identities (label-free, from the intra-mouse mean distance matrix, pose/pose.json): 0 and 11 are the
    farthest pair (the two ends of the chain); 0 is the end whose two nearest chain points 1, 2 are mutually symmetric
    (d(0,1) 15.6, d(0,2) 15.7, d(1,2) 15.4 px) = nose + ears; 9 = tail base (step 9-10 / 10-11 = 25.7 / 27.3 px, the
    long tail segments); 10, 11 = tail. Body points = 1-8 (ears, neck, paws, centre back). A visual check with every
    index drawn is written to fig_fp/qa/qa_keypoints.png.
  Contact type per frame (label-free, first match wins), distances = min over mouse pairs (ordered where it matters):
    1 nose-nose        nose(a) - nose(b)                < T
    2 nose-tail base   nose(a) - point 9 (b)            < T
    3 nose-other body  nose(a) - points 1-8 (b)         < T
    4 body-body only   points 1-9 (a) - points 1-9 (b) < T
    5 no contact       otherwise (tail points 10, 11 never count)
    T = the median min nose-nose distance over ALL oral_oral-positive eval frames (both halves; chosen once).
  Panel b: composition of the neuron's honest top 1% = the highest-activation half-1 frames holding 1% of the half-1
    inverse-probability weight mass (eval_frames.parquet w; ties pro rata, as contact_sae.wmetrics), by contact type,
    weighted; vs the same composition over all half-1 frames; and the weighted labelled oral_oral rate inside each
    type. Check: the overall top-1% oral_oral rate must equal the harness's test_top1 (sheets/index.json 0.0301).
  mm scale: the arena floor is fitted label-free per clip (eci_encode.arena_quad lines: per-pixel 0.98 quantile of 30
    evenly spaced grey frames > 100, one robust line per side), side = mean of the 4 corner-to-corner edges; arena side =
    520 mm (MABe22 paper, 52 cm arena); the median over the figure clips + 100 random eval clips (rng 0) is used.

Stages: --stage data (GPU job: selection, decoding, mask, scale, numbers -> OUT/fig_fp/facts.json, crops.npz, qa/);
        --stage plot (CPU: reads those -> OUT/fig_fp/fig_fp.png (300 dpi), fig_fp.pdf, caption.md).
"""
import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import BEHAVIOURS, LAG, OUT, PLAN, RAW, decode_clip, log, offset, stage_dir  # noqa: E402

ARM, NEURON, LABEL = 'D2', 11, 'oral_oral_contact'
FD = OUT / 'fig_fp'
CROP = 56
NOSE, TAILBASE, BODY = 0, 9, list(range(1, 9))
CATS = ['nose–nose', 'nose–tail base', 'nose–other body point', 'body–body only', 'no contact']
SHORTLAB = {'oral_genital_contact': 'oral–genital', 'oral_ear_contact': 'oral–ear', 'approaches': 'approach',
            'huddles': 'huddle', 'contact': 'contact', 'close': 'close', 'watching': 'watching', 'chases': 'chase',
            'oral_oral_contact': 'oral–oral'}
ROWS = ['Labelled sniffing,\nneuron fires', 'Not labelled,\nneuron fires', 'Labelled sniffing,\nneuron silent']
ARENA_MM = 520.0


# ------------------------------------------------------------------------------------------------ geometry
def pair_min(K, A, B, ordered):
    """K (F, 3, 12, 2) float, NaN = missing -> (F,) min distance between points A of mouse p and B of mouse q != p."""
    best = np.full(len(K), np.inf)
    for p in range(3):
        for q in range(3):
            if p == q or (not ordered and q < p):
                continue
            d = np.linalg.norm(K[:, p, A][:, :, None] - K[:, q, B][:, None, :], axis=-1)
            d = np.where(np.isnan(d), np.inf, d).reshape(len(K), -1).min(1)
            best = np.minimum(best, d)
    return best


def contact_type(K, T):
    nn = pair_min(K, [NOSE], [NOSE], False)
    ntb = pair_min(K, [NOSE], [TAILBASE], True)
    nbd = pair_min(K, [NOSE], BODY, True)
    bb = pair_min(K, BODY + [TAILBASE], BODY + [TAILBASE], False)
    cat = np.full(len(K), 4)
    cat[bb < T] = 3
    cat[nbd < T] = 2
    cat[ntb < T] = 1
    cat[nn < T] = 0
    return cat, {'nose_nose': nn, 'nose_tailbase': ntb, 'nose_body': nbd, 'body_body': bb}


def closest_nose_mid(k):
    """k (3, 12, 2) yx 512, NaN missing -> midpoint (y, x) of the closest nose-nose pair (512 scale) or None."""
    best, mid = np.inf, None
    for p in range(3):
        for q in range(p + 1, 3):
            d = np.linalg.norm(k[p, NOSE] - k[q, NOSE])
            if np.isfinite(d) and d < best:
                best, mid = d, (k[p, NOSE] + k[q, NOSE]) / 2
    return mid


def quad_side(fl):
    """fl (224, 224) bool floor -> mean side length (224 px) of the fitted floor quadrilateral (arena_quad's lines)."""
    from eci_encode import _robust_line
    rows = np.flatnonzero(fl.mean(1) > 0.3)
    cols = np.flatnonzero(fl.mean(0) > 0.3)
    if len(rows) < 40 or len(cols) < 40:
        return None, None
    rr = np.arange(rows[0] + 10, rows[-1] - 9)
    cc = np.arange(cols[0] + 10, cols[-1] - 9)
    fx = lambda v, a: np.array([np.flatnonzero(v[r])[a] if v[r].any() else -1 for r in range(len(v))])  # noqa: E731
    left, right = fx(fl[rr], 0), fx(fl[rr], -1)
    top, bot = fx(fl[:, cc].T, 0), fx(fl[:, cc].T, -1)
    L = _robust_line(rr[left >= 0], left[left >= 0])   # x = L(y)
    R = _robust_line(rr[right >= 0], right[right >= 0])
    Tl = _robust_line(cc[top >= 0], top[top >= 0])     # y = T(x)
    B = _robust_line(cc[bot >= 0], bot[bot >= 0])

    def corner(V, H):  # x = V0 y + V1, y = H0 x + H1
        x = (V[0] * H[1] + V[1]) / (1 - V[0] * H[0])
        return np.array([H[0] * x + H[1], x])
    C = [corner(L, Tl), corner(R, Tl), corner(R, B), corner(L, B)]  # (y, x): TL, TR, BR, BL
    sides = [float(np.linalg.norm(C[a] - C[(a + 1) % 4])) for a in range(4)]
    return float(np.mean(sides)), {'corners_yx': [c.round(2).tolist() for c in C], 'sides': sides}


# ------------------------------------------------------------------------------------------------ data stage
def stage_data():
    import torch
    from eci_encode import Enc, clip_mask
    from eci_train import halves
    stage = stage_dir()
    FD.mkdir(parents=True, exist_ok=True)
    (FD / 'qa').mkdir(exist_ok=True)
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    h = halves(E, 0)
    off = offset()
    sel = json.loads((OUT / 'eval' / f'{ARM}_s0.json').read_text())['m4096']['split0'][LABEL]['dirs'][0]['auc']
    assert sel['neuron'] == NEURON, sel
    loc = {}
    for n in (f'{ARM}_s0_codes.npy', f'{ARM}_s0_argtok.npy', f'{ARM}_lens.npy'):
        loc[n] = stage / n
        shutil.copyfile(OUT / 'codes' / n, loc[n])
    v = np.asarray(np.load(loc[f'{ARM}_s0_codes.npy'], mmap_mode='r')[:, NEURON], np.float32)
    argt = np.asarray(np.load(loc[f'{ARM}_s0_argtok.npy'], mmap_mode='r')[:, NEURON]).astype(np.int64)
    lens = np.load(loc[f'{ARM}_lens.npy'])
    for p in loc.values():
        p.unlink()
    assert len(v) == len(E) == len(lens)
    log(f'neuron {NEURON}: weighted firing rate {np.average(v > 0, weights=E.w):.4f}; half-1 frames {(h == 1).sum()}')
    # keypoints of every eval frame
    kl = stage / 'kp.npy'
    shutil.copyfile(RAW / 'mouse_submission_keypoints.npy', kl)
    sk = np.load(kl, allow_pickle=True).item()['sequences']
    kl.unlink()
    K = np.stack([sk[c]['keypoints'][i] for c, i in zip(E['clip'], E.i)]).astype(np.float64)  # (F, 3, 12, 2) yx 512
    del sk
    K[(K == 0).all(-1)] = np.nan
    assert K.shape[1:] == (3, 12, 2)
    y = E[LABEL].values.astype(bool)
    nn_all = pair_min(K, [NOSE], [NOSE], False)
    T = float(np.median(nn_all[y & np.isfinite(nn_all)]))
    cat, D = contact_type(K, T)
    log(f'T = {T:.2f} px (512 scale; median min nose-nose over {int((y & np.isfinite(nn_all)).sum())} oral_oral+ frames)')
    # ---- panel b: honest top 1% of half 1
    r1 = np.flatnonzero(h == 1)
    s, w = v[r1], E.w.values[r1].astype(np.float64)
    o = np.argsort(-s, kind='stable')
    so, wo = s[o], w[o]
    _, start = np.unique(-so, return_index=True)
    gt = np.add.reduceat(wo, start)
    cc = np.cumsum(gt)
    k = 0.01 * w.sum()
    ib = int(np.searchsorted(cc, k - 1e-9))
    gfrac = np.zeros(len(gt))
    gfrac[:ib] = 1
    gfrac[ib] = (k - (cc[ib] - gt[ib])) / gt[ib]
    frac = np.zeros(len(s))
    frac[o] = np.repeat(gfrac, np.diff(np.r_[start, len(so)]))
    wt = frac * w  # weight of each half-1 frame inside the top 1%
    yt, ct = y[r1], cat[r1]
    prec = float((wt * yt).sum() / wt.sum())
    log(f'top 1%: weight {wt.sum():.1f}, frames with weight > 0: {(wt > 0).sum()}, oral_oral rate {prec:.4f} '
        f'(harness test_top1 {sel["test_top1"]:.4f})')
    assert abs(prec - sel['test_top1']) < 1e-6, (prec, sel['test_top1'])
    comp = []
    for c, name in enumerate(CATS):
        m = ct == c
        comp.append({'type': name, 'top1_weight': float(wt[m].sum()), 'top1_share': float(wt[m].sum() / wt.sum()),
                     'top1_frames': int(((wt > 0) & m).sum()),
                     'top1_oral_oral_rate': float((wt[m] * yt[m]).sum() / wt[m].sum()) if wt[m].sum() > 0 else None,
                     'top1_contact_label_rate': (float((wt[m] * E['contact'].values[r1][m]).sum() / wt[m].sum())
                                                 if wt[m].sum() > 0 else None),
                     'half1_weight': float(w[m].sum()), 'half1_share': float(w[m].sum() / w.sum()),
                     'half1_frames': int(m.sum()),
                     'half1_oral_oral_rate': float((w[m] * yt[m]).sum() / w[m].sum()) if m.any() else None,
                     'half1_contact_label_rate': (float((w[m] * E['contact'].values[r1][m]).sum() / w[m].sum())
                                                  if m.any() else None)})
        log(f"  {name:24s} top1 {comp[-1]['top1_share']:.3f} (oo {comp[-1]['top1_oral_oral_rate']}) "
            f"half1 {comp[-1]['half1_share']:.4f}")
    # oral_oral+ frames by contact type (all eval frames, for the reading)
    oo_by_type = {CATS[c]: int((y & (cat == c)).sum()) for c in range(5)}
    # ---- panel a selection
    def top_rows(mask):
        cand = r1[mask[r1] & (v[r1] > 0)]
        cand = cand[np.argsort(-v[cand], kind='stable')]
        out, seen = [], set()
        for f in cand:
            if E['clip'][f] not in seen:
                seen.add(E['clip'][f])
                out.append(int(f))
            if len(out) == 6:
                break
        return out, int(len(cand))
    row1, n1 = top_rows(y)
    row2, n2 = top_rows(~y)
    cand3 = r1[y[r1] & (v[r1] == 0)]
    row3, seen = [], set()
    for f in cand3[np.random.default_rng(0).permutation(len(cand3))]:
        if E['clip'][f] not in seen:
            seen.add(E['clip'][f])
            row3.append(int(f))
        if len(row3) == 6:
            break
    log(f'candidates: row1 {n1}, row2 {n2}, row3 {len(cand3)}')
    # ---- decode + mask
    dev = torch.device('cuda')
    enc = Enc(dev)
    glob_bg = np.load(PLAN / 'pixbg_global.npy')
    need = sorted({E['clip'][f] for f in row1 + row2 + row3})
    crops, items, mask_check, sides = [], [], {}, {}
    rowof = {f: 0 for f in row1} | {f: 1 for f in row2} | {f: 2 for f in row3}
    frames_of = {}
    for c in need:
        fr = decode_clip('mouse_submission', c, stage, 'rgb24')
        assert len(fr) == 1801
        grey = (0.299 * fr[..., 0] + 0.587 * fr[..., 1] + 0.114 * fr[..., 2]).round().clip(0, 255)
        bgi = np.linspace(0, len(fr) - 1, 30).round().astype(int)
        pbg = np.sort(grey[bgi], 0)[28]
        sides[c] = quad_side(pbg > 100)
        if c == need[0] and sides[c][0] is not None:  # QA: the fitted floor quadrilateral on the clip background
            from PIL import Image, ImageDraw
            qa = Image.fromarray(pbg.astype(np.uint8)).convert('RGB').resize((672, 672), Image.NEAREST)
            cs_ = [(x * 3, y * 3) for y, x in sides[c][1]['corners_yx']]
            ImageDraw.Draw(qa).polygon(cs_, outline=(255, 0, 0))
            qa.save(FD / 'qa' / 'qa_arena.png')
        rows_c = np.flatnonzero(E['clip'].values == c)
        pos = None
        if any(rowof[f] < 2 for f in rows_c if f in rowof):
            ts = E.i.values[rows_c] + off
            M = clip_mask(enc, fr, ts, ts - LAG, 'arena', glob_bg, dev, (6, 15, 60))
            mk = M['mask'].cpu().numpy()
            nk = mk.sum(1)
            mask_check[c] = {'frames': int(len(rows_c)), 'n448_equal': int((nk == lens[rows_c]).sum()),
                             'kept_mean': float(nk.mean())}
            pos = {int(f): np.flatnonzero(mk[q]) for q, f in enumerate(rows_c)}
            del M
        for f in rows_c:
            if f not in rowof:
                continue
            img = fr[E.i[f] + off]
            k512 = K[f]
            k224 = k512 * 224 / 512
            it = {'row': rowof[f], 'clip': c, 'label_index': int(E.i[f]), 'video_frame': int(E.i[f] + off),
                  'eval_row': int(f), 'stratum': E['stratum'][f], 'weight': float(E.w[f]), 'activation': float(v[f]),
                  'labels': [b for b in BEHAVIOURS if bool(E[b][f])], 'contact_type': CATS[int(cat[f])],
                  'min_nose_nose_px512': float(D['nose_nose'][f]) if np.isfinite(D['nose_nose'][f]) else None,
                  'n_tokens': int(lens[f])}
            if rowof[f] < 2:
                p_ = pos[int(f)]
                assert len(p_) == lens[f], (c, f, len(p_), lens[f])
                g = int(p_[argt[f]])
                rr_, cc_ = divmod(g, 32)
                cy, cx = rr_ * 7 + 3.5, cc_ * 7 + 3.5
                it['peak_patch_grid'] = [rr_, cc_]
            else:
                mid = closest_nose_mid(k512)
                cy, cx = mid * 224 / 512 + 0.5
            y0, x0 = int(round(cy - CROP / 2)), int(round(cx - CROP / 2))
            pad = np.pad(img, ((CROP, CROP), (CROP, CROP), (0, 0)), constant_values=255)
            crops.append(pad[y0 + CROP:y0 + 2 * CROP, x0 + CROP:x0 + 2 * CROP])
            it['crop_origin_yx224'] = [y0, x0]
            it['noses_in_crop_xy'] = [[float(k224[m, NOSE, 1] + 0.5 - x0), float(k224[m, NOSE, 0] + 0.5 - y0)]
                                      for m in range(3) if np.isfinite(k224[m, NOSE]).all()]
            if rowof[f] < 2:
                it['patch_in_crop_xywh'] = [cc_ * 7 - x0, rr_ * 7 - y0, 7, 7]
            items.append(it)
            frames_of[int(f)] = (img, k224)
        del fr
    # QA image: two frames of the figure with every keypoint index drawn (nose identification check)
    from PIL import Image, ImageDraw
    qa = Image.new('RGB', (2 * 672, 672), 'white')
    for q, f in enumerate([row1[0], row3[0]]):
        img, k224 = frames_of[f]
        im = Image.fromarray(img).resize((672, 672), Image.BICUBIC)
        d = ImageDraw.Draw(im)
        for m in range(3):
            for pt in range(12):
                if np.isfinite(k224[m, pt]).all():
                    yy, xx = (k224[m, pt] + 0.5) * 3
                    col = (255, 60, 60) if pt == NOSE else ((60, 160, 255) if pt == TAILBASE else (255, 220, 0))
                    d.ellipse([xx - 3, yy - 3, xx + 3, yy + 3], outline=col)
                    d.text((xx + 4, yy - 6), str(pt), fill=col)
        qa.paste(im, (q * 672, 0))
    qa.save(FD / 'qa' / 'qa_keypoints.png')
    # mm scale: figure clips + 100 random eval clips
    allc = np.array(sorted(set(E['clip'])))
    extra = [c for c in np.random.default_rng(0).choice(allc, 100, replace=False) if c not in sides]
    for c in extra:
        g_ = decode_clip('mouse_submission', c, stage, 'gray')
        bgi = np.linspace(0, len(g_) - 1, 30).round().astype(int)
        sides[c] = quad_side(np.sort(g_[bgi], 0)[28] > 100)
    sv = np.array([x[0] for x in sides.values() if x[0] is not None])
    side224 = float(np.median(sv))
    mm_per_px512 = ARENA_MM / (side224 * 512 / 224)
    log(f'arena side {side224:.2f} px224 (median of {len(sv)} clips, IQR {np.percentile(sv, 25):.2f}-'
        f'{np.percentile(sv, 75):.2f}); {mm_per_px512:.4f} mm per px512')
    for it in items:
        it['min_nose_nose_mm'] = None if it['min_nose_nose_px512'] is None else it['min_nose_nose_px512'] * mm_per_px512
    facts = {
        'neuron': {'arm': ARM, 'sae_seed': 0, 'neuron': NEURON, 'picked_for': LABEL, 'pick': 'AUROC on half 0, split 0',
                   'test_auroc_half1': sel['test_auc'], 'test_ap_half1': sel['test_ap'],
                   'test_top1_half1': sel['test_top1'], 'test_rate_half1': sel['test_rate']},
        'frames': 'held-out half 1 of split seed 0 (halves(E, 0) == 1)', 'n_half1_frames': int(len(r1)),
        'T_px512': T, 'T_mm': T * mm_per_px512,
        'T_rule': 'median min nose-nose distance over all oral_oral-positive eval frames (both halves)',
        'scale': {'arena_side_px224_median': side224, 'n_clips': int(len(sv)),
                  'arena_side_px224_iqr': [float(np.percentile(sv, 25)), float(np.percentile(sv, 75))],
                  'arena_mm': ARENA_MM, 'mm_per_px512': mm_per_px512, 'px512_per_mm': 1 / mm_per_px512},
        'panel_b': {'top1_weight_mass': float(wt.sum()), 'half1_weight_mass': float(w.sum()),
                    'top1_frames_with_weight': int((wt > 0).sum()), 'top1_oral_oral_rate': prec,
                    'harness_test_top1': sel['test_top1'], 'top1_activation_threshold': float(so[start[ib]]),
                    'composition': comp},
        'oral_oral_positive_eval_frames_by_type': oo_by_type,
        'selection': {'row1_candidates': n1, 'row2_candidates': n2, 'row3_candidates': int(len(cand3))},
        'mask_recompute_check': mask_check,
    }
    # keep panel-a order: row 1/2 by activation, row 3 in permutation order
    order = {f: q for q, f in enumerate(row1 + row2 + row3)}
    idx = sorted(range(len(items)), key=lambda q: order[items[q]['eval_row']])
    facts['crops'] = [items[q] for q in idx]
    np.savez_compressed(FD / 'crops.npz', crops=np.stack([crops[q] for q in idx]))
    (FD / 'facts.json').write_text(json.dumps(facts, indent=1, default=float))
    log(f'mask check: {mask_check}')
    log('data stage done')


# ------------------------------------------------------------------------------------------------ plot stage
def stage_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    F = json.loads((FD / 'facts.json').read_text())
    C = np.load(FD / 'crops.npz')['crops']
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                         'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.5,
                         'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2,
                         'ytick.major.size': 2, 'savefig.facecolor': 'white'})
    INK, GREY = '#1a1a1a', '#8c8c8c'
    NOSEC, PATCHC, BARC = '#56B4E9', '#D55E00', '#0072B2'  # Okabe-Ito sky blue / vermillion / blue
    mm = 1 / 25.4
    W, H = 180, 156
    fig = plt.figure(figsize=(W * mm, H * mm))
    ax_ = lambda x, y, w, h: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])  # noqa: E731  (mm, from top-left)
    x0, sz, gap, rowh, top = 25.0, 23.5, 2.6, 33.0, 4.0
    fig.text(1 / W, 1 - 2.5 / H, 'a', fontsize=9, fontweight='bold', va='top')
    up = 6
    for q, it in enumerate(F['crops']):
        r, k = it['row'], [i for i, x in enumerate(F['crops']) if x['row'] == it['row']].index(q)
        a = ax_(x0 + k * (sz + gap), top + r * rowh, sz, sz)
        im = Image.fromarray(C[q]).resize((CROP * up, CROP * up), Image.BICUBIC)
        a.imshow(np.asarray(im), extent=[0, CROP, CROP, 0], interpolation='none')
        for (nx, ny) in it['noses_in_crop_xy']:
            if 0 <= nx <= CROP and 0 <= ny <= CROP:
                a.scatter([nx], [ny], s=22, facecolors='none', edgecolors=NOSEC, linewidths=0.8)
        if 'patch_in_crop_xywh' in it:
            px, py, pw, ph = it['patch_in_crop_xywh']
            a.add_patch(Rectangle((px, py), pw, ph, fill=False, ec=PATCHC, lw=0.8))
        a.set_xlim(0, CROP)
        a.set_ylim(CROP, 0)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_linewidth(0.4)
            sp.set_color(GREY)
        oth = [SHORTLAB[b] for b in it['labels'] if b != LABEL]
        lab = textwrap.wrap(', '.join(oth) if oth else 'no other label', 22)[:2]
        d = it['min_nose_nose_mm']
        lab.append('nose–nose ' + (f'{d:.1f} mm' if d is not None else 'n/a'))
        fig.text((x0 + k * (sz + gap) + sz / 2) / W, 1 - (top + r * rowh + sz + 0.8) / H, '\n'.join(lab),
                 ha='center', va='top', fontsize=6, color=INK, linespacing=1.15)
        if k == 0:
            fig.text(1 / W, 1 - (top + r * rowh + sz / 2) / H, ROWS[r], ha='left', va='center', fontsize=7,
                     color=INK, linespacing=1.2)
    # legend for the markers (one line, under panel a)
    ly = top + 3 * rowh + 0.5
    a = ax_(x0, ly, 60, 3)
    a.axis('off')
    a.set_xlim(0, 60)
    a.set_ylim(0, 3)
    a.scatter([1.5], [1.5], s=22, facecolors='none', edgecolors=NOSEC, linewidths=0.8)
    a.text(3.5, 1.5, 'nose (keypoints)', va='center', fontsize=6)
    a.add_patch(Rectangle((22, 0.4), 2.2, 2.2, fill=False, ec=PATCHC, lw=0.8))
    a.text(25.5, 1.5, 'neuron peak patch', va='center', fontsize=6)
    # panel b
    by = top + 3 * rowh + 9
    fig.text(1 / W, 1 - (by - 1.5) / H, 'b', fontsize=9, fontweight='bold', va='top')
    P = F['panel_b']['composition']
    a = ax_(x0 + 12, by + 2, 82, 30)
    yy = np.arange(len(P))
    t1 = np.array([p['top1_share'] for p in P]) * 100
    hp = np.array([p['half1_share'] for p in P]) * 100
    a.barh(yy - 0.19, t1, 0.36, color=BARC, label="neuron's top 1%")
    a.barh(yy + 0.19, hp, 0.36, color='#cfcfcf', label='all held-out eval frames')
    a.set_yticks(yy)
    a.set_yticklabels([p['type'] for p in P], fontsize=7)
    a.invert_yaxis()
    a.set_xlabel('share of inverse-probability-weighted frames (%)', fontsize=7)
    a.set_xlim(0, max(t1.max(), hp.max()) * 1.18)
    a.tick_params(labelsize=6)
    for s_ in ('top', 'right'):
        a.spines[s_].set_visible(False)
    for i_, (v1, v2) in enumerate(zip(t1, hp)):
        a.text(v1 + 0.6, i_ - 0.19, f'{v1:.1f}', va='center', fontsize=6, color=BARC)
        a.text(v2 + 0.6, i_ + 0.19, f'{v2:.1f}', va='center', fontsize=6, color='#6b6b6b')
    a.legend(frameon=False, fontsize=6, loc='upper right', handlelength=1.2)
    # right-hand column: labelled oral_oral rate inside each type (top 1%)
    xr = x0 + 12 + 82 + 6
    fig.text(xr / W, 1 - (by + 0.2) / H, 'labelled oral–oral\nin top 1% / in all frames', fontsize=6, va='bottom',
             color=INK, linespacing=1.15)
    for i_, p in enumerate(P):
        yc = by + 2 + 30 * (i_ + 0.5) / len(P)
        tr = p['top1_oral_oral_rate']
        hr = p['half1_oral_oral_rate']
        fig.text(xr / W, 1 - yc / H, (f'{100 * tr:.1f}%' if tr is not None else '–') + ' / ' +
                 (f'{100 * hr:.2f}%' if hr is not None else '–'), fontsize=6, va='center', color=INK)
    fig.text(xr / W, 1 - (by + 2 + 30 + 2.5) / H,
             f"contact = keypoints closer than T = {F['T_mm']:.1f} mm\n(median nose–nose on labelled oral–oral)",
             fontsize=6, va='top', color='#4d4d4d', linespacing=1.15)
    fig.savefig(FD / 'fig_fp.png', dpi=300)
    fig.savefig(FD / 'fig_fp.pdf')
    log('plot stage done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['data', 'plot', 'all'], default='all')
    a_ = ap.parse_args()
    if a_.stage in ('data', 'all'):
        stage_data()
    if a_.stage in ('plot', 'all'):
        stage_plot()
