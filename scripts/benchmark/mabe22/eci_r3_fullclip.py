"""
MABe22 ECI round 3, E4 ETHOGRAM step 1 (GPU Slurm job, sharded over clips): full-clip firing bits of the 6 CONFIRMED R
dictionaries on the submission clips (rule: eci_round3.sh, E4 paragraph; this script adds no rule).

Recipe R per grid frame (label index i, video frame t = i + offset, offset in plan/offset.json):
    mask      eci_encode.clip_mask(..., pixbg_mode='arena', glob_bg=plan/pixbg_global.npy): from the PLAIN 448 tokens of
              the frame and of the clip's 30 evenly spaced background frames (the 224 tokens that clip_mask also
              returns are not part of E4 and are replaced by zeros, never computed)
    tokens    D4 orbit-averaged 448 tokens (eci_encode.Enc.orbit logic) of t and of the lag frame max(t - 60, 0), the
              lag read at t's mask positions; stored as fp16 then x = [z_t, z_t - z_lag] built exactly as
              eci_train.load_shard (fp16 t, fp16 of the fp32 difference of the fp16 tokens)
    SAE       confirm/sae/X_A7D2_s{100..105}/sae.pt (eci_train.load_any, TokenNorm included), theta* =
              confirm/eval/X_A7D2_s{seed}.json['calib']['theta_star'] via sae_calibrate.set_threshold,
              sae.encode(norm(x), mode='threshold'); frame code = max over the frame's kept tokens (0 if none)
    precision PRE-REGISTERED by cost (L40S fp32 119 img/s projects E4 above 14 GPU-h): every DINOv2 forward (mask
              tokens and orbit tokens) under torch.autocast('cuda', bfloat16); interpolation, normalisation, mask
              logic, SAE and everything else fp32 with TF32 off.
Grids:  --fps 5   label i = 0, 6, ..., 1794 (300 frames), all 1,828 submission clips (eval_frames.parquet clips)
        --fps 15  label i = 0, 2, ..., 1798 restricted to i % 6 != 0 (600 frames; the i % 6 == 0 frames are the 5 fps
                  run's: the lag t - 60 keeps the residue mod 6 and the 30 background frames are the same), on the
                  pre-registered SUBSET = rng 0 choice of 300 of the sorted clip ids + every clip with an
                  oral_oral_contact or oral_genital_contact positive label frame (labels read inside the job)
fp32 reference (--ref N, the first N clips of the shard, 5 fps only): the eval-plan frames of the clip
    (plan/eval_frames.parquet, label i = 30..1794 step 6) re-encoded in fp32 TF32 off exactly as R (plain-token mask,
    orbit tokens of t and max(t - 60, 0)); per seed the bf16 and fp32 frame codes at those frames are stored and
    compared (firing-bit Jaccard per frame, Pearson of the code values, mask Jaccard).
Outputs (OUT/r3/e4/, written once per shard): fps{F}_{k}of{n}_bits.npy (C, 6, T, 512) uint8 = np.packbits of the
    firing bits over the 4,096 neurons (rows = the 6 seeds 100..105, frames in grid order); fps{F}_{k}of{n}_kept.npy
    (C, T) int16 kept tokens per frame; fps5_..._meancode.npy (C, 6, 4096) float32 mean over the 300 grid frames of the
    frame code; fps{F}_{k}of{n}.json (clips in row order, label indices, cost, sanity stats); fps5_..._ref.npz.
"""
import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/eci'))
from eci_common import EXCLUDE, OUT, PLAN, RAW, decode_clip, log, offset, stage_dir  # noqa: E402

SEEDS = [100, 101, 102, 103, 104, 105]
CONF = OUT / 'confirm'
E4 = OUT / 'r3' / 'e4'
LAG60 = 60
VSET = 'mouse_submission'


def grid(fps):
    if fps == 5:
        return np.arange(0, 1800, 6)
    i = np.arange(0, 1800, 2)
    return i[i % 6 != 0]


def all_clips():
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    clips = sorted(E['clip'].unique())
    assert len(clips) == 1828 and not (set(clips) & EXCLUDE)
    return clips, E


def subset15(clips):
    """pre-registered 15 fps subset (labels loaded here; job only)."""
    sub = np.load(RAW / 'mouse_submission_labels.npy', allow_pickle=True).item()
    fm = np.load(RAW / 'mouse_frame_number_map.npy', allow_pickle=True).item()
    assert set(fm) - EXCLUDE == set(clips), 'eval clips != label clips minus EXCLUDE'
    vocab = list(sub['vocabulary'])
    L = sub['label_array']
    ro, rg = vocab.index('oral_oral_contact'), vocab.index('oral_genital_contact')
    rnd = np.random.default_rng(0).choice(np.array(clips), 300, replace=False)
    oral = [c for c in clips if ((L[ro, fm[c][0]:fm[c][1]] == 1) | (L[rg, fm[c][0]:fm[c][1]] == 1)).any()]
    S = sorted(set(rnd.tolist()) | set(oral))
    return S, {'n_random': 300, 'n_oral_oo_og': len(oral), 'n_overlap': len(set(rnd.tolist()) & set(oral)),
               'n_subset': len(S)}


class Encoders:
    """plain 448 / orbit 448 DINOv2-base tokens, bf16 autocast optional (forward only), TF32 off."""

    def __init__(self, dev):
        import torch
        from eci_encode import Enc
        import invariant_tokens as it
        from src.eci.priors import d4_pos_map
        self.torch, self.it = torch, it
        self.enc = Enc(dev)  # loads the model, sets TF32 off, ImageNet mean / std
        assert not torch.backends.cuda.matmul.allow_tf32
        assert it.check_d4_torch()
        self.maps = torch.stack([torch.from_numpy(d4_pos_map(g)) for g in range(8)]).to(dev)
        self.dev = dev
        self.gpu_s = {'bf16': 0.0, 'fp32': 0.0}
        self.n_img = {'bf16': 0, 'fp32': 0}

    def _prep(self, fr):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(fr)).to(self.dev).permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(x, size=(448, 448), mode='bicubic', align_corners=False).round().clamp(0, 255)
        return (x / 255 - self.enc.mean) / self.enc.std

    def _fwd(self, x, bf16):
        torch = self.torch
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            h = self.enc.model(pixel_values=x).last_hidden_state[:, 1:]
        h = h.float()
        assert h.shape[1] == 1024 and torch.isfinite(h).all()
        return h

    def plain(self, frames, bf16, bs=64):
        torch = self.torch
        torch.cuda.synchronize()
        t0 = time.time()
        out = []
        with torch.inference_mode():
            for a in range(0, len(frames), bs):
                out.append(self._fwd(self._prep(frames[a:a + bs]), bf16))
        torch.cuda.synchronize()
        k = 'bf16' if bf16 else 'fp32'
        self.gpu_s[k] += time.time() - t0
        self.n_img[k] += len(frames)
        return torch.cat(out)

    def orbit(self, frames, bf16, bs=32):
        torch = self.torch
        torch.cuda.synchronize()
        t0 = time.time()
        out = []
        with torch.inference_mode():
            for a in range(0, len(frames), bs):
                x = self._prep(frames[a:a + bs])
                acc = 0
                for g in range(8):
                    h = self._fwd(self.it.d4_torch(x, g), bf16)
                    acc = acc + h[:, self.maps[g]]
                out.append(acc / 8)
        torch.cuda.synchronize()
        k = 'bf16' if bf16 else 'fp32'
        self.gpu_s[k] += time.time() - t0
        self.n_img[k] += 8 * len(frames)
        return torch.cat(out)


class MaskEnc:
    """the enc argument of eci_encode.clip_mask: plain 448 tokens at the chosen precision; 224 -> zeros (unused)."""

    def __init__(self, E, bf16):
        self.E, self.bf16 = E, bf16

    def __call__(self, frames, size, bs=64):
        if size == 224:
            return self.E.torch.zeros(len(frames), 256, 768, device=self.E.dev)
        assert size == 448
        return self.E.plain(frames, self.bf16, bs)


def clip_codes(E, saes, fr, ts, glob_bg, bf16):
    """video frames ts of one clip -> per seed (T, 4096) fp32 GPU frame codes, mask (T, 1024) bool, stats."""
    import torch
    from eci_encode import clip_mask
    dev = E.dev
    M = clip_mask(MaskEnc(E, bf16), fr, ts, ts, 'arena', glob_bg, dev)
    mask = M['mask'].clone()
    info = M['info']
    del M
    lag = np.maximum(ts - LAG60, 0)
    need = np.unique(np.concatenate([ts, lag]))
    O = E.orbit(fr[need], bf16)
    pos = {v: j for j, v in enumerate(need)}
    jt = torch.tensor([pos[v] for v in ts], device=dev)
    jl = torch.tensor([pos[v] for v in lag], device=dev)
    qi, pi = torch.nonzero(mask, as_tuple=True)  # row-major: grouped by frame
    t16 = O[jt[qi], pi].half()
    l16 = O[jl[qi], pi].half()
    del O
    X = torch.cat([t16, (t16.float() - l16.float()).half()], 1)  # as eci_train.load_shard
    T = len(ts)
    out, l0 = {}, {}
    with torch.no_grad():
        for s, (sae, norm) in saes.items():
            codes = torch.zeros(T, sae.n_latents, device=dev)
            nact = 0
            for a in range(0, len(X), 32768):
                z = sae.encode(norm(X[a:a + 32768]), mode='threshold')
                nact += int((z > 0).sum())
                seg = qi[a:a + 32768]
                codes.scatter_reduce_(0, seg[:, None].expand_as(z), z, reduce='amax', include_self=True)
                del z
            out[s] = codes
            l0[s] = nact / max(len(X), 1)
    return out, mask, {'mask_info': info, 'l0_per_token': l0, 'n_tokens': int(len(X))}


def main():
    import torch
    from eci_train import load_any
    from src.eci.sae_calibrate import set_threshold
    ap = argparse.ArgumentParser()
    ap.add_argument('--fps', type=int, choices=[5, 15], required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--smoke', type=int, default=0, help='only the first N clips of the shard')
    ap.add_argument('--ref', type=int, default=0, help='fp32 reference at the eval-plan frames of the first N clips')
    args = ap.parse_args()
    t00 = time.time()
    dev = torch.device('cuda')
    stage = stage_dir()
    E4.mkdir(parents=True, exist_ok=True)
    key = f'fps{args.fps}_{args.shard}of{args.nshard}' + (f'_smoke{args.smoke}' if args.smoke else '')
    if (E4 / f'{key}.json').exists():
        log(f'[SKIP] {key} exists')
        return
    clips, EV = all_clips()
    sub_info = None
    if args.fps == 15:
        clips, sub_info = subset15(clips)
        log(f'15 fps subset: {sub_info}')
        if args.shard == 0 and not args.smoke:
            (E4 / 'subset15.json').write_text(json.dumps({**sub_info, 'clips': clips}, indent=1))
    mine = list(np.array_split(np.array(clips), args.nshard)[args.shard])
    if args.smoke:
        mine = mine[:args.smoke]
    off = offset()
    gi = grid(args.fps)
    ts = gi + off
    glob_bg = np.load(PLAN / 'pixbg_global.npy')
    E = Encoders(dev)
    saes = {}
    for s in SEEDS:
        sae, norm, _ = load_any(CONF / 'sae' / f'X_A7D2_s{s}' / 'sae.pt', dev)
        th = json.loads((CONF / 'eval' / f'X_A7D2_s{s}.json').read_text())['calib']['theta_star']
        set_threshold(sae, th)
        saes[s] = (sae.eval(), norm)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    log(f'{key}: {len(mine)} clips, {len(gi)} grid frames (label {gi[0]}..{gi[-1]}), offset {off}, '
        f'{torch.cuda.get_device_name(0)}')
    C, T = len(mine), len(gi)
    bits = np.zeros((C, len(SEEDS), T, 512), np.uint8)
    kept = np.zeros((C, T), np.int16)
    meanc = np.zeros((C, len(SEEDS), 4096), np.float32) if args.fps == 5 else None
    l0s, fired_frames, act_per_frame, minfo = [], [], [], {}
    ref = {'clip': [], 'i': [], 'kept_bf16': [], 'kept_fp32': [], 'mask_jacc': [],
           **{f'bf16_s{s}': [] for s in SEEDS}, **{f'fp32_s{s}': [] for s in SEEDS}}
    tdec = 0.0
    pool = ThreadPoolExecutor(1)
    fut = pool.submit(decode_clip, VSET, mine[0], stage, 'rgb24')
    for ci, c in enumerate(mine):
        t0 = time.time()
        fr = fut.result()
        tdec += time.time() - t0
        if ci + 1 < C:
            fut = pool.submit(decode_clip, VSET, mine[ci + 1], stage, 'rgb24')
        if len(fr) != 1801:
            raise RuntimeError(f'{c}: {len(fr)} frames')
        assert ts.max() < len(fr)
        codes, mask, st = clip_codes(E, saes, fr, ts, glob_bg, True)
        kept[ci] = mask.sum(1).cpu().numpy()
        minfo[c] = {'thr': st['mask_info']['thr'], 'kept_per_frame': st['mask_info']['kept_per_frame']}
        l0s.append([st['l0_per_token'][s] for s in SEEDS])
        for k, s in enumerate(SEEDS):
            fb = (codes[s] > 0).cpu().numpy()
            bits[ci, k] = np.packbits(fb, axis=1)
            if meanc is not None:
                meanc[ci, k] = codes[s].mean(0).cpu().numpy()
            fired_frames.append(float(fb.any(1).mean()))
            act_per_frame.append(float(fb.sum(1).mean()))
        if ci < args.ref and args.fps == 5:
            ie = EV.loc[EV['clip'] == c, 'i'].values
            assert np.isin(ie, gi).all()
            rows = np.searchsorted(gi, ie)
            rc, rmask, _ = clip_codes(E, saes, fr, ie + off, glob_bg, False)
            bm = mask[torch.as_tensor(rows, device=dev)]
            inter = (bm & rmask).sum(1).float()
            uni = (bm | rmask).sum(1).float().clamp_min(1)
            ref['clip'] += [c] * len(ie)
            ref['i'] += ie.tolist()
            ref['kept_bf16'] += bm.sum(1).cpu().numpy().tolist()
            ref['kept_fp32'] += rmask.sum(1).cpu().numpy().tolist()
            ref['mask_jacc'] += (inter / uni).cpu().numpy().tolist()
            for s in SEEDS:
                ref[f'bf16_s{s}'].append(codes[s][torch.as_tensor(rows, device=dev)].half().cpu().numpy())
                ref[f'fp32_s{s}'].append(rc[s].half().cpu().numpy())
            del rc, rmask
        del codes, mask, fr
        if ci % 25 == 0 or ci == C - 1:
            log(f'  clip {ci + 1}/{C} {c}: kept/frame {kept[ci].mean():.1f}, L0/token {np.mean(l0s[-1]):.2f}, '
                f'frames firing {fired_frames[-1]:.3f}, active/frame {act_per_frame[-1]:.1f}; images {E.n_img}, '
                f'gpu {E.gpu_s["bf16"]:.0f}+{E.gpu_s["fp32"]:.0f}s, decode wait {tdec:.0f}s, wall {time.time() - t00:.0f}s')
    pool.shutdown()
    arrays = {'bits': bits, 'kept': kept}
    if meanc is not None:
        arrays['meancode'] = meanc
    for k, a in arrays.items():
        np.save(stage / f'{key}_{k}.npy', a)
        shutil.copyfile(stage / f'{key}_{k}.npy', E4 / f'{key}_{k}.npy')
        (stage / f'{key}_{k}.npy').unlink()
        log(f'  saved {k} {a.shape} {a.dtype}')
    agree = None
    if ref['clip']:
        R = {k: (np.concatenate(v) if k.startswith(('bf16', 'fp32')) else np.array(v)) for k, v in ref.items()}
        np.savez(stage / f'{key}_ref.npz', **R)
        shutil.copyfile(stage / f'{key}_ref.npz', E4 / f'{key}_ref.npz')
        (stage / f'{key}_ref.npz').unlink()
        agree = {'n_frames': int(len(R['clip'])), 'n_clips': int(len(set(R['clip']))),
                 'mask_jaccard_mean': float(R['mask_jacc'].mean()), 'kept_bf16_mean': float(R['kept_bf16'].mean()),
                 'kept_fp32_mean': float(R['kept_fp32'].mean()), 'per_seed': {}}
        for s in SEEDS:
            b, f = R[f'bf16_s{s}'].astype(np.float32), R[f'fp32_s{s}'].astype(np.float32)
            A, B = b > 0, f > 0
            uni = (A | B).sum(1)
            jac = np.where(uni > 0, (A & B).sum(1) / np.maximum(uni, 1), 1.0)
            agree['per_seed'][s] = {'firing_jaccard_mean': float(jac.mean()),
                                    'pearson_all': float(np.corrcoef(b.ravel(), f.ravel())[0, 1]),
                                    'active_bf16_per_frame': float(A.sum(1).mean()),
                                    'active_fp32_per_frame': float(B.sum(1).mean())}
        log(f'  bf16 vs fp32 at {agree["n_frames"]} eval-plan frames: ' + json.dumps(agree))
    l0s = np.array(l0s)
    info = {'key': key, 'fps': args.fps, 'shard': args.shard, 'nshard': args.nshard, 'clips': mine,
            'label_i': gi.tolist(), 'offset': off, 'seeds': SEEDS, 'precision': 'bf16 autocast (DINOv2 forward only)',
            'subset15': sub_info, 'n_clips': C, 'frames': int(C * T),
            'images_encoded': E.n_img, 'gpu_s': {k: round(v, 1) for k, v in E.gpu_s.items()},
            'decode_wait_s': round(tdec, 1), 'wall_s': round(time.time() - t00, 1),
            'img_per_gpu_s_bf16': round(E.n_img['bf16'] / max(E.gpu_s['bf16'], 1e-9), 1),
            'kept_per_frame_mean': float(kept.mean()), 'frames_with_0_kept': int((kept == 0).sum()),
            'l0_per_token_mean_by_seed': dict(zip(map(str, SEEDS), l0s.mean(0).round(3).tolist())),
            'frac_frames_with_firing_mean': float(np.mean(fired_frames)),
            'active_neurons_per_frame_mean': float(np.mean(act_per_frame)),
            'thr_quantiles': np.quantile([v['thr'] for v in minfo.values()], [0, .5, 1]).round(4).tolist(),
            'bf16_vs_fp32': agree, 'gpu': torch.cuda.get_device_name(0)}
    (E4 / f'{key}.json').write_text(json.dumps(info, indent=1))
    log('done: ' + json.dumps({k: v for k, v in info.items() if k not in ('clips', 'label_i')}))


if __name__ == '__main__':
    main()
