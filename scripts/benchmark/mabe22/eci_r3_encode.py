"""
MABe22 mouse ECI round 3 (rules: eci_round3.sh): the shared round-3 token store (GPU Slurm job, sharded over clips).

Store dir: MABE_TOK (set by eci_round3.sh to results/vision/benchmark_mabe22/eci/tok_r3/). Frames: plan/train_frames
.parquet (train + calib roles) and plan/eval_frames.parquet, exactly as eci_encode.py (video frame = label i + offset).
Modes:
    store   mask recomputed with eci_encode.clip_mask (pixbg 'arena', bit-identical rule to the confirmation store);
            D4-orbit 448 tokens (eci_encode.Enc.orbit) of t and of t-L for L in --lags (frames, clipped to 0), plain 448
            tokens of t and t-60; all at t's mask positions. Kinds: m448o_t, m448o_p<L> (L = 30 saved as 'm448o_p', the
            name eci_train reads for the 1 s lag), m448_t, m448_p60, pos448; <key>_frames.parquet (clip, i, n448).
    scale   E3 token density: S-px input (S in --sizes, bicubic upscale of the 224 frame), D4-orbit tokens of t and t-60 on
            the (S/14)^2 grid. Mask = the store's 448 mask transferred by patch centre: patch (r, c) of the G = S/14 grid is
            kept iff the 448 patch containing its centre, (floor((r + .5) 32 / G), floor((c + .5) 32 / G)), is kept.
            Kinds m<S>o_t, m<S>o_p60, pos<S>; frames parquet gets n<S> (key <set>_<i>of<n>_s<S>).
    adapt   E6: as scale at 448 with the encoder weights from --encoder (state_dict for facebook/dinov2-base); masks
            = the store's pos448 (unchanged). Kinds ma448o_t, ma448o_p60 (key <set>_<i>of<n>_adapt).
Precision: fp32, TF32 off (R's). Outputs fp16 tokens, copied to the store once per shard.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/eci'))
from eci_common import LAG, OUT, PLAN, TOK, decode_clip, log, offset, stage_dir  # noqa: E402
from eci_encode import SETS, Enc, clip_mask, plan, shard_clips  # noqa: E402


def orbit_S(enc, frames, S, bs):
    """frames (n, 224, 224, 3) uint8 -> (n, G*G, 768) fp32 GPU: tokens of the S-px input averaged over the 8 D4
    transforms, each read back at the original patch positions (Enc.orbit generalised to grid G = S / 14)."""
    import torch
    import torch.nn.functional as F
    import invariant_tokens as it
    from src.eci.priors import d4_pos_map
    G = S // 14
    maps = torch.stack([torch.from_numpy(d4_pos_map(g, G)) for g in range(8)]).to(enc.dev)
    out = []
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        for a in range(0, len(frames), bs):
            x = torch.from_numpy(np.ascontiguousarray(frames[a:a + bs])).to(enc.dev).permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(S, S), mode='bicubic', align_corners=False).round().clamp(0, 255)
            x = (x / 255 - enc.mean) / enc.std
            acc = 0
            for g in range(8):
                h = enc.model(pixel_values=it.d4_torch(x, g)).last_hidden_state[:, 1:].float()
                assert h.shape[1] == G * G
                acc = acc + h[:, maps[g]]
            out.append(acc / 8)
    torch.cuda.synchronize()
    enc.gpu_s += time.time() - t0
    enc.n_img += 8 * len(frames)
    return torch.cat(out)


def transfer_mask(pos448, G):
    """kept 448 positions (flat, 32 x 32 grid) -> kept flat positions on the G x G grid (patch-centre rule)."""
    m = np.zeros((32, 32), bool)
    m.flat[pos448.astype(np.int64)] = True
    idx = np.floor((np.arange(G) + 0.5) * 32 / G).astype(int)
    return np.flatnonzero(m[np.ix_(idx, idx)]).astype(np.int16)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=list(SETS), required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--mode', choices=['store', 'scale', 'adapt'], default='store')
    ap.add_argument('--lags', nargs='*', type=int, default=[15, 30, 45, 60, 120])
    ap.add_argument('--sizes', nargs='*', type=int, default=[336, 672])
    ap.add_argument('--encoder', default='')
    ap.add_argument('--smoke', type=int, default=0)
    args = ap.parse_args()
    t00 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device('cuda')
    stage = stage_dir()
    TOK.mkdir(parents=True, exist_ok=True)
    base = f'{args.set}_{args.shard}of{args.nshard}'
    suf = {'store': '', 'scale': '_s' + '_'.join(map(str, args.sizes)), 'adapt': '_adapt'}[args.mode]
    key = base + suf + (f'_smoke{args.smoke}' if args.smoke else '')
    if (TOK / f'{key}.json').exists():
        log(f'[SKIP] {key}')
        return
    P = plan(args.set)
    clips = shard_clips(P, args.shard, args.nshard)
    if args.smoke:
        clips = clips[:args.smoke]
    off = offset()
    vset = SETS[args.set][0]
    enc = Enc(dev)
    if args.mode == 'adapt':
        sd = torch.load(args.encoder, map_location='cpu', weights_only=False)
        sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
        missing, unexpected = enc.model.load_state_dict(sd, strict=True), None
        enc.model.eval().requires_grad_(False)
        log(f'adapted encoder loaded from {args.encoder}')
    REF = None
    if args.mode != 'store':
        RF = pd.read_parquet(TOK / f'{base}_frames.parquet')
        REF = {'F': RF, 'pos': np.load(TOK / f'{base}_pos448.npy'), 'st': np.r_[0, np.cumsum(RF.n448.values)]}
    glob_bg = np.load(PLAN / 'pixbg_global.npy')
    log(f'{key}: {len(clips)} clips, mode {args.mode}, offset {off}, {torch.cuda.get_device_name(0)}')
    A, meta, cinfo, tdec = {}, [], {}, 0.0
    add = lambda k, v: A.setdefault(k, []).append(v)  # noqa: E731
    for ci, c in enumerate(clips):
        R = P[P['clip'] == c]
        t0 = time.time()
        fr = decode_clip(vset, c, stage, 'rgb24')
        tdec += time.time() - t0
        if len(fr) != 1801:
            raise RuntimeError(f'{c}: {len(fr)} frames')
        ts = R.i.values + off
        row = {'clip': c}
        if args.mode == 'store':
            M = clip_mask(enc, fr, ts, ts - LAG, 'arena', glob_bg, dev, (60,))
            cinfo[c] = M['info']
            mask = M['mask']
            orb = {0: enc.orbit(fr[ts])}
            for L_ in args.lags:
                orb[L_] = enc.orbit(fr[np.maximum(ts - L_, 0)])
            for q in range(len(R)):
                p = torch.nonzero(mask[q]).flatten()
                add('pos448', p.cpu().numpy().astype(np.int16))
                add('m448_t', M['tok_t'][q, p].half().cpu().numpy())
                add('m448_p60', M['tok_xl'][60][q, p].half().cpu().numpy())
                for L_, o in orb.items():
                    nm = 'm448o_t' if L_ == 0 else ('m448o_p' if L_ == 30 else f'm448o_p{L_}')
                    add(nm, o[q, p].half().cpu().numpy())
                meta.append({'clip': c, 'i': int(R.i.values[q]), 'n448': int(len(p))})
            del M, orb
        else:
            rr = np.flatnonzero(REF['F']['clip'].values == c)
            assert (REF['F'].i.values[rr] == R.i.values).all()
            p448 = [REF['pos'][REF['st'][r_]:REF['st'][r_ + 1]] for r_ in rr]
            sizes = [448] if args.mode == 'adapt' else args.sizes
            mrow = [{'clip': c, 'i': int(R.i.values[q]), 'n448': int(len(p448[q]))} for q in range(len(R))]
            for S in sizes:
                G = S // 14
                bs = 8 if S > 448 else 16
                ot = orbit_S(enc, fr[ts], S, bs)
                op = orbit_S(enc, fr[np.maximum(ts - 60, 0)], S, bs)
                pre = 'ma448o' if args.mode == 'adapt' else f'm{S}o'
                for q in range(len(R)):
                    pq = p448[q] if S == 448 else transfer_mask(p448[q], G)
                    pt = torch.from_numpy(pq.astype(np.int64)).to(dev)
                    add(f'{pre}_t', ot[q, pt].half().cpu().numpy())
                    add(f'{pre}_p60', op[q, pt].half().cpu().numpy())
                    if args.mode == 'scale':
                        add(f'pos{S}', pq)
                        mrow[q][f'n{S}'] = int(len(pq))
                del ot, op
            meta += mrow
        del fr
        if ci % 25 == 0:
            log(f'  clip {ci}/{len(clips)}: {enc.n_img} images, gpu {enc.gpu_s:.0f}s, decode {tdec:.0f}s')
    Fm = pd.DataFrame(meta)
    assert len(Fm) == int(P['clip'].isin(clips).sum())
    for k, v in A.items():
        a = np.concatenate(v)
        np.save(stage / f'{key}_{k}.npy', a)
        shutil.copyfile(stage / f'{key}_{k}.npy', TOK / f'{key}_{k}.npy')
        (stage / f'{key}_{k}.npy').unlink()
        log(f'  saved {k} {a.shape}')
    Fm.to_parquet(TOK / f'{key}_frames.parquet')
    info = {'set': args.set, 'shard': args.shard, 'nshard': args.nshard, 'mode': args.mode, 'lags': args.lags,
            'sizes': args.sizes if args.mode == 'scale' else None, 'encoder': args.encoder or 'facebook/dinov2-base',
            'clips': len(clips), 'frames': len(Fm), 'offset': off, 'images_encoded': enc.n_img,
            'gpu_s': round(enc.gpu_s, 1), 'decode_s': round(tdec, 1),
            'img_per_gpu_s': round(enc.n_img / max(enc.gpu_s, 1e-9), 1), 'gpu': torch.cuda.get_device_name(0),
            'wall_s': round(time.time() - t00, 1),
            'kept_mean': {c_: float(Fm[c_].mean()) for c_ in Fm.columns if c_.startswith('n')},
            'frames_with_0_kept': int((Fm.n448 == 0).sum()), 'per_clip': cinfo}
    (TOK / f'{key}.json').write_text(json.dumps(info, indent=1, default=float))
    log('done: ' + json.dumps({k: v for k, v in info.items() if k != 'per_clip'}, default=float))


if __name__ == '__main__':
    main()
