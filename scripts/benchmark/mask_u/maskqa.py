"""
Mask-U test (rules: scripts/benchmark/eci_mask_u.sh) - mask QA on sample frames, current per-domain rule vs rule U
(src/eci/mask_u.py), GPU job. Keypoints are read ONLY to measure the masks.

Domains / current rules:
    mabe     MABe22 mouse triplets (mouse_submission clips, plan eval frames): current = 'arena' (mabe22 eci_encode
             clip_mask: FG_RULE token cue + dark cue with the drop condition waived inside the fitted arena quadrilateral,
             dilation onto all neighbours); keypoints 512-px scale, (y, x), video frame = label + 1
    beetle   MABe22 beetles (beetle_userTrain): current = 'b15hc' (eci_r3_beetle.beetle_mask: bright cue, waived in
             animal-sized floor holes); one keypoint per animal, convention a0yx_a1xy; recall = keypoint in a kept patch
    calms    CalMS21 task-1 videos: current = 'dark' (plan/mask_rule.txt; calms21 eci_encode.Masker), canvas 1024 x 1024
             (padded), 32 x 32 cells of 32 px; 7 MARS keypoints per mouse (resident = black = mouse 0, intruder = white =
             mouse 1), score > 0.5; video frame = label + 0
Per frame: kept cells, keypoint recall per animal, animal missed (no valid keypoint in a kept cell), and (mabe / calms)
hull precision = fraction of kept cells within 1 cell of some animal's keypoint convex hull; (beetle) far-40 = fraction
of kept cells whose centre is > 40 px (224 scale) from both keypoints (eci_r3_beetle.kp_far).
--pick dev|final: dev = the clips used to set U's parameters (rng 7); final = beetle 24 clips x 16 frames of the round-3
beetle mask check (rng 1) - MABe mouse / CalMS21 final numbers come from the full encode stores (encode.py).
Outputs: OUT/maskqa/<domain>_<pick>.json, OUT/sheets/<domain>_<pick>_<k>.jpg
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from common import OUT, hull_cells, log, recall_cells, sheet, stage_dir  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------- MABe
def mabe_setup():
    sys.path.insert(0, str(REPO / 'scripts/benchmark/mabe22'))


def mabe_grey(frames):
    return (0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]).round().clip(0, 255).astype(np.uint8)


def mabe_current(tok_bg, tok_t, grey_all, bgi, ts, glob_bg, dev):
    """mabe22 eci_encode.clip_mask with pixbg 'arena' (the round-3 / encoder-study mask), on precomputed DINOv2-448
    tokens of the 30 background frames bgi and the target frames ts -> (n, 1024) bool, info."""
    import torch
    import torch.nn.functional as F
    from eci_encode import arena_quad
    from src.eci.foreground import FG_RULE, cosine_distance, foreground_parts, patch_background, video_threshold
    g = torch.from_numpy(grey_all).to(dev)
    pix = g[torch.as_tensor(bgi, device=dev)].float().kthvalue(int(round(0.98 * 30)), dim=0).values.round()
    quad = arena_quad(pix.cpu().numpy() > 100)
    hole = torch.from_numpy(quad['inside']).to(dev)

    def dark_of(idx):
        gg = g[torch.as_tensor(idx, device=dev)].float()
        dk = (gg < FG_RULE['dark_abs']) & (((pix[None] - gg) > FG_RULE['dark_rel']) | hole[None])
        return F.avg_pool2d(dk.float()[:, None], 7).flatten(1)
    dark_bg = dark_of(bgi)
    bg, n_ok = patch_background(tok_bg, exclude=dark_bg > FG_RULE['dark_frac'], min_frames=8)
    thr = video_threshold(cosine_distance(tok_bg, bg).cpu().numpy(), dark_bg.cpu().numpy(), FG_RULE)
    feat, dk, mask = foreground_parts(cosine_distance(tok_t, bg), dark_of(ts), thr, FG_RULE)
    return mask, {'thr': thr, 'arena_px': int(hole.sum())}


def mabe_clip_masks(enc, fr, ts, dev, glob_bg=None):
    """one decoded clip (1801, 224, 224, 3), target video frames ts -> (cur (n,1024), U (n,1024), U parts, infos)."""
    import torch
    from src.eci.mask_u import UMasker
    T = len(fr)
    bgi = np.linspace(0, T - 1, 30).round().astype(int)
    grey = mabe_grey(fr)
    need = np.unique(np.concatenate([bgi, ts]))
    pos_of = {v: j for j, v in enumerate(need)}
    tok = enc(fr[need], 448)
    tb = tok[torch.tensor([pos_of[v] for v in bgi], device=dev)]
    tt = tok[torch.tensor([pos_of[v] for v in ts], device=dev)]
    cur, ci = mabe_current(tb, tt, grey, bgi, ts, glob_bg, dev)
    um = UMasker(dev).fit(torch.from_numpy(grey[bgi]).to(dev), tb, t_s=bgi)
    parts = um.parts(torch.from_numpy(grey[ts]).to(dev), tt, ts)
    return cur, parts[2], parts, {'cur': ci, 'U': um.info}, tt


def mabe_kp(clips_i, set_='submission'):
    """[(clip, label i)] -> cells (n, 3, 12), valid (n, 3, 12), xy cells (n, 3, 12, 2), xy px at 224 (n, 3, 12, 2)."""
    from eci_common import RAW
    f = RAW / ('mouse_submission_keypoints.npy' if set_ == 'submission' else 'mouse_user_train.npy')
    sk = np.load(f, allow_pickle=True).item()['sequences']
    kp = np.stack([np.asarray(sk[c]['keypoints'][i], np.float32) for c, i in clips_i]) * 224 / 512  # (n, 3, 12, 2) (y, x)
    valid = ~(kp == 0).all(-1)
    y, x = np.clip(kp[..., 0], 0, 223.9), np.clip(kp[..., 1], 0, 223.9)
    cells = (y // 7).astype(int) * 32 + (x // 7).astype(int)
    return cells, valid, np.stack([x / 7, y / 7], -1), np.stack([x, y], -1)


def run_mabe(args, enc, dev):
    from eci_common import PLAN, decode_clip, offset
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    clips = sorted(E['clip'].unique())
    rng = np.random.default_rng(7)
    pick = list(rng.choice(clips, args.n, replace=False))
    off = offset()
    stage = stage_dir()
    acc = {'cur': [], 'U': []}
    rows, tiles, infos = [], [], {}
    for k, c in enumerate(pick):
        R = E[E['clip'] == c]
        R = R.iloc[np.linspace(0, len(R) - 1, min(args.frames, len(R))).round().astype(int)]
        fr = decode_clip('mouse_submission', c, stage, 'rgb24')
        ts = R.i.values + off
        cur, U, parts, info, _ = mabe_clip_masks(enc, fr, ts, dev)
        infos[c] = info
        cells, valid, xyc, xypx = mabe_kp([(c, i) for i in R.i.values])
        hull = hull_cells(xyc, valid)
        for nm, m in (('cur', cur.cpu().numpy()), ('U', U.cpu().numpy())):
            rec, miss = recall_cells(m, cells, valid)
            acc[nm].append({'rec': rec, 'miss': miss, 'kept': m.sum(1), 'prec': (m & hull).sum(1) / np.maximum(m.sum(1), 1),
                            'w': R.w.values})
        for j in range(0, len(ts), max(1, len(ts) // 2)):
            if len(tiles) < 16:
                kpj = xypx[j][valid[j]]
                tiles.append((fr[ts[j]], {'cur': cur[j].cpu().numpy(), 'U': U[j].cpu().numpy()}, f'{c[:6]} i{R.i.values[j]}', kpj))
        log(f'{c}: cur kept {acc["cur"][-1]["kept"].mean():.1f} rec {np.nanmean(acc["cur"][-1]["rec"]):.3f} | U kept '
            f'{acc["U"][-1]["kept"].mean():.1f} rec {np.nanmean(acc["U"][-1]["rec"]):.3f} | U info {json.dumps(info["U"])}')
    return summarise(acc, ['mouse0', 'mouse1', 'mouse2']), tiles, infos


# -------------------------------------------------------------------------------------------------------------- beetle
def run_beetle(args, enc, dev):
    import torch
    import eci_r3_beetle as BT
    from eci_common import decode_clip
    from src.eci.mask_u import UMasker
    clips = BT.clips_all()
    if args.pick == 'final':
        rng = np.random.default_rng(1)  # the round-3 beetle mask check clips and frames (mask_mode)
        pick = list(rng.choice(clips, 24, replace=False))
    else:
        rng = np.random.default_rng(7)
        pick = list(rng.choice(clips, args.n, replace=False))
    kps = BT.keypoints(pick)
    stage = stage_dir()
    acc = {'cur': [], 'U': []}
    tiles, infos = [], {}
    for ci, c in enumerate(pick):
        fr = decode_clip(BT.VSET, c, stage, 'rgb24')
        assert len(fr) == BT.T_CLIP
        ts = np.sort(np.random.default_rng([2, ci]).choice(BT.GRID_B, 16, replace=False))
        cur, cinfo, _ = BT.beetle_mask(enc, fr, ts, 'b15hc', dev)
        grey = BT.grey_of(fr)
        bgi = np.linspace(0, BT.T_CLIP - 1, 30).round().astype(int)
        tb, tt = enc(fr[bgi], 448), enc(fr[ts], 448)
        um = UMasker(dev).fit(torch.from_numpy(grey[bgi]).to(dev), tb, t_s=bgi)
        U = um.mask(torch.from_numpy(grey[ts]).to(dev), tt, ts)
        infos[c] = {'cur': cinfo, 'U': um.info}
        kp_t = kps[c][ts]
        for nm, m in (('cur', cur.cpu().numpy()), ('U', U.cpu().numpy())):
            hit = BT.kp_recall(m, kp_t, 'a0yx_a1xy')  # (n, 2)
            far = BT.kp_far(m, kp_t, 'a0yx_a1xy')[1]
            acc[nm].append({'rec': hit.astype(float), 'miss': ~hit, 'kept': m.sum(1),
                            'prec': 1 - far / np.maximum(m.sum(1), 1), 'w': np.ones(len(ts))})
        if len(tiles) < 16:
            xy = BT.kp_xy(kp_t[:1], 'a0yx_a1xy')[0] * 224
            tiles.append((fr[ts[0]], {'cur': cur[0].cpu().numpy(), 'U': U[0].cpu().numpy()}, f'{c[:6]} t{ts[0]}', xy))
        log(f'{c}: cur kept {acc["cur"][-1]["kept"].mean():.1f} rec {acc["cur"][-1]["rec"].mean():.3f} | U kept '
            f'{acc["U"][-1]["kept"].mean():.1f} rec {acc["U"][-1]["rec"].mean():.3f} | U {json.dumps(um.info)}')
    return summarise(acc, ['beetle0', 'beetle1'], prec_name='near40_frac_of_kept'), tiles, infos


# ------------------------------------------------------------------------------------------------------------- CalMS21
def calms_setup():
    sys.path.insert(0, str(REPO / 'scripts/benchmark/calms21'))


def calms_canvas_grey(frames, dev):
    from eci_encode import canvas_gpu
    c = canvas_gpu(frames, dev)
    return (0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]).round()


def calms_valid(dev):
    import torch
    from calms_common import CANVAS, PADY, H
    v = torch.zeros((CANVAS, CANVAS), dtype=torch.bool, device=dev)
    v[PADY:PADY + H] = True
    return v


def calms_video_masks(enc, masker, frames_bg, frames_t, dev, bs=16, t_bg=None, t_t=None):
    """-> (cur (n,1024), U (n,1024), U parts, info, DINOv2 448 t tokens)."""
    import torch
    from src.eci.mask_u import UMasker
    masker.video(frames_bg, ['dark'])
    tb = enc.run(frames_bg, bs)
    tt = enc.run(frames_t, bs)
    cur = torch.cat([masker.masks(frames_t[a:a + 64], tt[a:a + 64], ['dark'])['dark'] for a in range(0, len(frames_t), 64)])
    um = UMasker(dev).fit(calms_canvas_grey(frames_bg, dev), tb, calms_valid(dev), t_bg)
    parts = [um.parts(calms_canvas_grey(frames_t[a:a + 32], dev), tt[a:a + 32], None if t_t is None else t_t[a:a + 32]) for a in range(0, len(frames_t), 32)]
    parts = tuple(torch.cat([p[j] for p in parts]) for j in range(3))
    return cur, parts[2], parts, {'U': um.info, 'cur_thr': float(masker.state['dark'][1])}, tt


def calms_kp(split, video, idx):
    """-> cells (n, 2, 7), valid (n, 2, 7), xy cells (n, 2, 7, 2), xy px on the canvas (n, 2, 7, 2)."""
    from calms_common import DER, PADY
    Z = np.load(DER / f'task1_{split}.npz')
    names = list(Z['names'])
    st = np.r_[0, np.cumsum(Z['lens'])][names.index(video)]
    kp, sc = Z['keypoints'][st + idx], Z['scores'][st + idx]  # (n, 2, 2, 7) x / y; (n, 2, 7)
    x, y = kp[:, :, 0, :], kp[:, :, 1, :] + PADY
    cells = np.clip((y // 32).astype(int), 0, 31) * 32 + np.clip((x // 32).astype(int), 0, 31)
    return cells, sc > 0.5, np.stack([x / 32, y / 32], -1), np.stack([x, y], -1)


def run_calms(args, enc, dev):
    from calms_common import PLAN, decode_frames, offset, video_path
    from eci_encode import Masker, NBG
    stage = stage_dir()
    off = offset()
    P = pd.read_parquet(PLAN / 'train_frames.parquet')
    inv = json.loads((REPO / 'results/vision/benchmark_calms21/inventory/inventory.json').read_text())['videos']['train']['per_video']
    vids = sorted(P.video.unique())
    pick = list(np.random.default_rng(7).choice(vids, args.n, replace=False))
    masker = Masker(enc, dev)
    acc = {'cur': [], 'U': []}
    tiles, infos = [], {}
    for q in pick:
        R = P[P.video == q]
        R = R.iloc[np.linspace(0, len(R) - 1, min(args.frames, len(R))).round().astype(int)]
        N = inv[q]['frames']
        bgi = np.linspace(0, N - 1, NBG).round().astype(int)
        v = R.i.values + off
        loc = stage / f'{q}.mp4'
        shutil.copyfile(video_path('train', q), loc)
        fr, _ = decode_frames(loc, set(bgi.tolist()) | set(v.tolist()), threads=8)
        loc.unlink()
        cur, U, parts, info, _ = calms_video_masks(enc, masker, [fr[x] for x in bgi], [fr[x] for x in v], dev, t_bg=bgi, t_t=v)
        infos[q] = info
        cells, valid, xyc, xypx = calms_kp('train', q, R.i.values)
        hull = hull_cells(xyc, valid)
        for nm, m in (('cur', cur.cpu().numpy()), ('U', U.cpu().numpy())):
            rec, miss = recall_cells(m, cells, valid)
            acc[nm].append({'rec': rec, 'miss': miss, 'kept': m.sum(1), 'prec': (m & hull).sum(1) / np.maximum(m.sum(1), 1),
                            'w': np.ones(len(v))})
        from calms_common import PADY, H
        for j in np.linspace(0, len(v) - 1, 4).round().astype(int):
            canvas = np.full((1024, 1024, 3), 116, np.uint8)
            canvas[PADY:PADY + H] = fr[v[j]]
            tiles.append((canvas, {'cur': cur[j].cpu().numpy(), 'U': U[j].cpu().numpy()}, f'{q[:9]} v{v[j]}', xypx[j][valid[j]]))
        log(f'{q}: cur kept {acc["cur"][-1]["kept"].mean():.1f} rec {np.nanmean(acc["cur"][-1]["rec"], 0)} | U kept '
            f'{acc["U"][-1]["kept"].mean():.1f} rec {np.nanmean(acc["U"][-1]["rec"], 0)} | U {json.dumps(info["U"])}')
    return summarise(acc, ['resident_black', 'intruder_white']), tiles, infos


def summarise(acc, animals, prec_name='hull_precision'):
    out = {}
    for nm, L in acc.items():
        rec = np.concatenate([x['rec'] for x in L])
        miss = np.concatenate([x['miss'] for x in L])
        kept = np.concatenate([x['kept'] for x in L])
        prec = np.concatenate([x['prec'] for x in L])
        w = np.concatenate([x['w'] for x in L])
        ok = kept > 0
        out[nm] = {'frames': int(len(kept)), 'kept_mean': float(kept.mean()), 'kept_frac_grid': float(kept.mean() / 1024),
                   'frames_0_kept': int((~ok).sum()), prec_name: float(np.mean(prec[ok])) if ok.any() else None,
                   'recall_per_animal': {a: float(np.nanmean(rec[:, j])) for j, a in enumerate(animals)},
                   'missed_frac_per_animal': {a: float(miss[:, j].mean()) for j, a in enumerate(animals)},
                   'recall_all': float(np.nanmean(rec))}
    return out


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', choices=['mabe', 'beetle', 'calms'], required=True)
    ap.add_argument('--pick', choices=['dev', 'final'], default='dev')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--frames', type=int, default=12)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    dev = torch.device('cuda')
    if args.domain in ('mabe', 'beetle'):
        mabe_setup()
        from eci_encode import Enc
        enc = Enc(dev)
        res, tiles, infos = (run_mabe if args.domain == 'mabe' else run_beetle)(args, enc, dev)
    else:
        calms_setup()
        from eci_encode import Enc
        enc = Enc('d2_448', dev, True)
        res, tiles, infos = run_calms(args, enc, dev)
    from src.eci.mask_u import U_RULE
    key = f'{args.domain}_{args.pick}{args.tag}'
    (OUT / 'maskqa').mkdir(parents=True, exist_ok=True)
    (OUT / 'maskqa' / f'{key}.json').write_text(json.dumps({'rule_U': U_RULE, 'summary': res, 'per_video': infos},
                                                           indent=1, default=float))
    for k in range(0, len(tiles), 8):
        sheet(tiles[k:k + 8], OUT / 'sheets' / f'{key}_{k // 8}.jpg', ncol=4)
    log(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
