"""
Animal mask v4 CANDIDATE: SAM 2.1 (Segment Anything 2.1, Meta) prompted label-free by the frozen N2 thin-part cue
(src/eci/v2/mask2.py). NOT part of the frozen v2.0 (mask2.py / v20.py N2Series unchanged). Tested in
scripts/eci/v2/mask4_val.py (pre-registration: scripts/eci/v2/mask4_prereg.md).

One algorithm for every domain; the only per-domain number is the existing body half-width h (calibration).

Per frame (all sizes in pixels at the encoder input side S):
  1. PROMPTS (objects): connected components (8-conn) of the N2 cue pixels that contain a body part (opening of the
     cue by a disk of radius round(h/2)) of area >= MIN_BODY_H2 * h^2 and at least one core pixel (frozen dark cue or
     strong change). Per object: box = bounding box of the whole cue component padded by PAD_H * h; positive points =
     farthest-point samples of the body (start at the deepest body pixel, distance transform), n = clip(round(body
     area / (PTS_AREA_H2 * h^2)), 1, MAX_PTS). A huddle / touching group is one object (the mask is a union anyway).
  2. SAM: per object, the 3 multimask outputs; a mask is VALID iff area <= MAX_GROW * cue-component area and it covers
     >= MIN_COVER of the object's body pixels; the valid mask with the highest predicted IoU is used, none valid ->
     the cue component itself (fallback, counted).
     mode 'full': one SAM pass on the whole frame (all objects). mode 'crop': one SAM pass per object on a square crop
     around its box (side = max(box side, CROP_MIN_H * h), padded by CROP_PAD_H * h, clipped to the frame), so the
     1024 px SAM input sees small animals (ants) zoomed; masks pasted back.
     mode 'video' (SAM 2 video predictor): prompts on the first frame of a clip, propagated to the last frame.
  3. CLEAN: connected components of the union that contain no core pixel are dropped (no isolated background).
  4. HYBRID (rule 'B'): union | the N2 appendage pixels (mask2.appendage of the cue) whose connected component touches
     the SAM union (dilated by 1 px) -- thin attached parts SAM misses (ant legs / antennae at 2 px width).
Token rule (unchanged from N2): token kept iff >= 25% of its 16 x 16 patch is mask OR >= MIN_THIN_PX (16) appendage
pixels of the mask (mask2.appendage on the final mask).
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.eci.v2 import mask2 as M2

HF = '/mnt/beegfs/locatgrp/shared/rcadei/hf_cache'
SAM_DIR = {'small': f'{HF}/models--facebook--sam2.1-hiera-small/snapshots/ee5bba1d82bb8749febdf90f45e84b687142ba03',
           'large': f'{HF}/models--facebook--sam2.1-hiera-large/snapshots/665f8e2ad61cf5f53d65644ff27c8ee525124610'}
MIN_BODY_H2, PAD_H, PTS_AREA_H2, MAX_PTS = 0.5, 0.5, 8.0, 4
MAX_GROW, MIN_COVER = 3.0, 0.5
CROP_MIN_H, CROP_PAD_H = 4.0, 1.0
CROP_BATCH = 4  # crops per SAM pass (memory only; results do not depend on it beyond float rounding)


def adt():
    """autocast dtype: bf16 where supported (L40S / A100 / A40), else fp16."""
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def load_sam(size, dev, video=False):
    if video:
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        return Sam2VideoModel.from_pretrained(SAM_DIR[size]).to(dev).eval(), Sam2VideoProcessor.from_pretrained(SAM_DIR[size])
    from transformers import Sam2Model, Sam2Processor
    return Sam2Model.from_pretrained(SAM_DIR[size]).to(dev).eval(), Sam2Processor.from_pretrained(SAM_DIR[size])


def fps_points(b, n):
    """farthest-point samples of the bool mask b: start at the max of the distance transform."""
    import cv2
    dt = cv2.distanceTransform(b.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(b)
    i0 = int(np.argmax(dt[ys, xs]))
    pts = [i0]
    d = (xs - xs[i0]) ** 2 + (ys - ys[i0]) ** 2
    for _ in range(n - 1):
        i = int(np.argmax(d * (dt[ys, xs] >= 0.5 * dt[ys, xs].max())))  # deep interior only (not on a leg)
        if d[i] == 0:
            break
        pts.append(i)
        d = np.minimum(d, (xs - xs[i]) ** 2 + (ys - ys[i]) ** 2)
    return [[float(xs[i]), float(ys[i])] for i in pts]


def objects(pix, core, h):
    """pix (S, S) bool cue pixels, core (S, S) bool -> list of dict box [x0, y0, x1, y1], pts [[x, y], ...], comp
    (S, S) bool, body (S, S) bool."""
    import cv2
    S = pix.shape[0]
    ro = max(1, int(round(h / 2)))
    body = cv2.morphologyEx(pix.astype(np.uint8), cv2.MORPH_OPEN, disk(ro)) > 0
    n, lab = cv2.connectedComponents(pix.astype(np.uint8), connectivity=8)
    out = []
    for c in range(1, n):
        comp = lab == c
        b = body & comp
        if b.sum() < MIN_BODY_H2 * h * h or not (core & comp).any():
            continue
        ys, xs = np.nonzero(comp)
        p = PAD_H * h
        box = [max(0., xs.min() - p), max(0., ys.min() - p), min(S - 1., xs.max() + p), min(S - 1., ys.max() + p)]
        npt = int(np.clip(round(b.sum() / (PTS_AREA_H2 * h * h)), 1, MAX_PTS))
        out.append({'box': box, 'pts': fps_points(b, npt), 'comp': comp, 'body': b})
    return out


def _select(masks, scores, ob):
    """masks (3, H, W) bool np, scores (3,) -> chosen mask or None."""
    best, bs = None, -1e9
    ca, bb = ob['comp'].sum(), ob['body']
    for m, s in zip(masks, scores):
        if m.sum() > MAX_GROW * ca or (m & bb).sum() < MIN_COVER * bb.sum():
            continue
        if s > bs:
            best, bs = m, s
    return best


@torch.no_grad()
def sam_full(model, proc, img, objs, dev):
    """img (S, S, 3) uint8, objs -> list of chosen masks (or None) per object."""
    if not objs:
        return []
    inp = proc(images=img, input_boxes=[[o['box'] for o in objs]], input_points=[[o['pts'] for o in objs]],
               input_labels=[[[1] * len(o['pts']) for o in objs]], return_tensors='pt').to(dev)
    with torch.autocast('cuda', dtype=adt()):
        out = model(**inp, multimask_output=True)
    ms = proc.post_process_masks(out.pred_masks.float().cpu(), inp['original_sizes'].cpu())[0].numpy()
    sc = out.iou_scores.float().cpu().numpy()[0]
    return [_select(ms[i], sc[i], o) for i, o in enumerate(objs)]


def crop_box(o, S, h):
    x0, y0, x1, y1 = o['box']
    side = max(x1 - x0, y1 - y0, CROP_MIN_H * h) + 2 * CROP_PAD_H * h
    side = int(min(S, round(side)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a = int(np.clip(round(cx - side / 2), 0, S - side))
    b = int(np.clip(round(cy - side / 2), 0, S - side))
    return a, b, side


@torch.no_grad()
def sam_crop(model, proc, img, objs, dev, h, prompts=None):
    """one SAM pass per object on its crop. prompts: optional per-object override dict pts / labels (gold building)."""
    if not objs:
        return []
    S = img.shape[0]
    ims, boxes, pts, labs, geo = [], [], [], [], []
    for i, o in enumerate(objs):
        a, b, side = crop_box(o, S, h)
        geo.append((a, b, side))
        ims.append(np.ascontiguousarray(img[b:b + side, a:a + side]))
        x0, y0, x1, y1 = o['box']
        boxes.append([[x0 - a, y0 - b, x1 - a, y1 - b]])
        P = o['pts'] if prompts is None else prompts[i]['pts']
        L = [1] * len(P) if prompts is None else prompts[i]['labels']
        pts.append([[[x - a, y - b] for x, y in P]])
        labs.append([L])
    res = []
    for j in range(0, len(ims), CROP_BATCH):
        sl = slice(j, j + CROP_BATCH)
        kw = dict(images=ims[sl], input_points=pts[sl], input_labels=labs[sl], return_tensors='pt')
        if prompts is None or all(p.get('box', True) for p in prompts[sl]):
            kw['input_boxes'] = boxes[sl]
        inp = proc(**kw).to(dev)
        with torch.autocast('cuda', dtype=adt()):
            out = model(**inp, multimask_output=True)
        ms = proc.post_process_masks(out.pred_masks.float().cpu(), inp['original_sizes'].cpu())
        sc = out.iou_scores.float().cpu().numpy()
        for t in range(len(ms)):
            i = j + t
            a, b, side = geo[i]
            full = np.zeros((3, S, S), bool)
            full[:, b:b + side, a:a + side] = ms[t][0].numpy()
            res.append(full if prompts is not None else _select(full, sc[t][0], objs[i]))
            if prompts is not None:
                res[-1] = (full, sc[t][0])
    return res


@torch.no_grad()
def sam_video(model, proc, clip, objs, dev):
    """clip (L, S, S, 3) uint8, objs prompted on frame 0 -> per-object bool mask (S, S) at the last frame."""
    if not objs:
        return []
    L, S = clip.shape[0], clip.shape[1]
    ses = proc.init_video_session(video=list(clip), inference_device=dev, dtype=torch.float32)
    ids = list(range(1, len(objs) + 1))
    proc.add_inputs_to_inference_session(ses, frame_idx=0, obj_ids=ids, input_boxes=[[o['box'] for o in objs]],
                                         input_points=[[o['pts'] for o in objs]],
                                         input_labels=[[[1] * len(o['pts']) for o in objs]])
    last = None
    with torch.autocast('cuda', dtype=adt()):
        for out in model.propagate_in_video_iterator(ses, start_frame_idx=0):
            if out.frame_idx == L - 1:
                last = proc.post_process_masks([out.pred_masks.float()], original_sizes=[[S, S]], binarize=True)[0]
    m = last[:, 0].cpu().numpy()
    return [m[i] for i in range(len(objs))]


def assemble(chosen, objs, pix, core, h, rule='A', app=None):
    """chosen per-object masks (None -> cue component), -> final (S, S) bool mask and number of fallbacks."""
    import cv2
    S = pix.shape[0]
    u = np.zeros((S, S), bool)
    nfb = 0
    for m, o in zip(chosen, objs):
        if m is None or (m.sum() > MAX_GROW * o['comp'].sum()):
            m = o['comp']
            nfb += 1
        u |= m
    if rule == 'B':
        if app is None:
            _, a = M2.appendage(torch.from_numpy(pix)[None], h)
            app = a[0].numpy()
        n, lab = cv2.connectedComponents(app.astype(np.uint8), connectivity=8)
        if n > 1:
            ud = cv2.dilate(u.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            touch = np.unique(lab[ud & (lab > 0)])
            u |= np.isin(lab, touch[touch > 0])
    n, lab = cv2.connectedComponents(u.astype(np.uint8), connectivity=8)
    if n > 1:
        good = np.unique(lab[core & (lab > 0)])
        u = np.isin(lab, good[good > 0])
    return u, nfb


def tokens(pix, h, patch=16, dev=None):
    """(S, S) bool np -> (G*G,) bool r25t token rule (as N2); dev: run the morphology there (same result)."""
    p = torch.from_numpy(pix)[None]
    if dev is not None:
        p = p.to(dev)
    _, app = M2.appendage(p, h)
    fr = F.avg_pool2d(p.float()[:, None], patch).flatten(1)
    na = F.avg_pool2d(app.float()[:, None], patch).flatten(1) * patch * patch
    return ((fr >= 0.25) | (na >= M2.MIN_THIN_PX))[0].cpu().numpy()
