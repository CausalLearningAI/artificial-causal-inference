"""
Mask-U test (rules: scripts/benchmark/eci_mask_u.sh) - MABe22 mouse token store for the two SAE arms (GPU job).

Frames: the round-3 plans (results/vision/benchmark_mabe22/eci/plan/train_frames.parquet: 13 train + 3 calib frames per
userTrain clip; eval_frames.parquet: 18,496 submission frames with IPW weights); video frame = label i + offset (1).
Per clip (decoded once, parallel ffmpeg workers, bounded look-ahead):
    masks   DINOv2-base 448 (fp32, TF32 off; mabe22 eci_encode.Enc) tokens of the 30 evenly spaced background frames and
            of the plan frames t ->
              cur  the current MABe mask 'arena' (maskqa.mabe_current = eci_encode.clip_mask --pixbg arena)
              U    rule U (src/eci/mask_u.py)
    tokens  DINOv3-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m) at 512 px (224 bicubic-upscaled, 32 x 32 grid = the
            mask grid), block 20 after the final LayerNorm, bf16 autocast + SDPA, of t and of t - 60 (clipped to 0), at the
            UNION of the two masks' cells (t's cells for both); per token flags in_cur / in_u
Outputs (BeeGFS TOK/mabe/): <set>_<s>of<n>_{t,p60}.npy (fp16, N x 1024), _pos.npy (int16), _incur.npy / _inu.npy (bool),
_frames.parquet (clip, i, n, n_cur, n_u), _masks.npz (packed 32 x 32 bits per frame: cur, U), <set>_<s>of<n>.json
(timing, frames / s, GPU s, agreement of 'cur' with the encoder-study r3mask positions). Mask sheet of shard 0.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'scripts/benchmark/mabe22'))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from common import OUT, TOK, log, sheet, stage_dir  # noqa: E402
from eci_common import PLAN, decode_clip, offset  # noqa: E402
from maskqa import mabe_clip_masks  # noqa: E402

SETS = {'train': ('mouse_userTrain', 'train_frames.parquet'), 'eval': ('mouse_submission', 'eval_frames.parquet')}
R3MASK = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/encoders/r3mask')
LAG = 60
LAYER = 20


def r3_positions(set_):
    """{(clip, i): positions} of the encoder-study 'arena' masks (read-only), or {} if absent."""
    out = {}
    n = {'train': 8, 'eval': 12}[set_]
    for s in range(n):
        f = R3MASK / f'{set_}_{s}of{n}_frames.parquet'
        if not f.exists():
            return {}
        Fm = pd.read_parquet(f)
        pos = np.load(R3MASK / f'{set_}_{s}of{n}_pos448.npy')
        st = np.r_[0, np.cumsum(Fm.n448.values)]
        for j, (c, i) in enumerate(zip(Fm['clip'].values, Fm.i.values)):
            out[(c, int(i))] = pos[st[j]:st[j + 1]]
    return out


def main():
    import torch
    from eci_encode import Enc
    from eci_enc_encode import Encoder
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=list(SETS), required=True)
    ap.add_argument('--shards', type=int, nargs='+', required=True)
    ap.add_argument('--nshard', type=int, required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--smoke', type=int, default=0)
    args = ap.parse_args()
    dev = torch.device('cuda')
    stage = stage_dir()
    od = TOK / 'mabe'
    od.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(PLAN / SETS[args.set][1])
    clips_all = np.array(sorted(P['clip'].unique()))
    off = offset()
    m448 = Enc(dev)
    enc = Encoder('dinov3l', 512, [LAYER], dev, bf16=True)
    r3 = r3_positions(args.set)
    log(f'{args.set}: {len(clips_all)} clips, r3mask frames {len(r3)}, {torch.cuda.get_device_name(0)}')
    for sh in args.shards:
        t00 = time.time()
        base = f'{args.set}_{sh}of{args.nshard}' + (f'_smoke{args.smoke}' if args.smoke else '')
        if (od / f'{base}.json').exists():
            log(f'[SKIP] {base}')
            continue
        clips = list(np.array_split(clips_all, args.nshard)[sh])
        if args.smoke:
            clips = clips[:args.smoke]
        pool = ThreadPoolExecutor(args.workers)

        def dec(c_):
            t_ = time.time()
            return decode_clip(SETS[args.set][0], c_, stage, 'rgb24'), time.time() - t_
        fut = {j: pool.submit(dec, clips[j]) for j in range(min(len(clips), 2 * args.workers))}
        A = {k: [] for k in ('t', 'p60', 'pos', 'incur', 'inu')}
        meta, mcur, mu, tiles, agree, infos = [], [], [], [], [], {}
        tdec = twait = 0.0
        gpu_mask = 0.0
        for ci, c in enumerate(clips):
            t0 = time.time()
            fr, dt = fut.pop(ci).result()
            twait += time.time() - t0
            tdec += dt
            if ci + 2 * args.workers < len(clips):
                fut[ci + 2 * args.workers] = pool.submit(dec, clips[ci + 2 * args.workers])
            assert len(fr) == 1801, (c, len(fr))
            R = P[P['clip'] == c]
            ts = R.i.values + off
            torch.cuda.synchronize()
            t1 = time.time()
            cur, U, parts, info, _ = mabe_clip_masks(m448, fr, ts, dev)
            torch.cuda.synchronize()
            gpu_mask += time.time() - t1
            infos[c] = info
            uni = cur | U
            for kd, v in (('t', ts), ('p60', np.maximum(ts - LAG, 0))):
                h = enc.encode(fr, v, args.bs)[LAYER]  # (n, 1024, 1024) fp32
                A[kd] += [h[q, uni[q]].half().cpu().numpy() for q in range(len(ts))]
                del h
            cn, un, nn = cur.cpu().numpy(), U.cpu().numpy(), uni.cpu().numpy()
            for q, i in enumerate(R.i.values):
                p = np.flatnonzero(nn[q])
                A['pos'].append(p.astype(np.int16))
                A['incur'].append(cn[q][p])
                A['inu'].append(un[q][p])
                meta.append({'clip': c, 'i': int(i), 'n': int(len(p)), 'n_cur': int(cn[q].sum()), 'n_u': int(un[q].sum())})
                if (c, int(i)) in r3:
                    ref = np.zeros(1024, bool)
                    ref[r3[(c, int(i))].astype(np.int64)] = True
                    agree.append((float((ref == cn[q]).mean()), bool((ref == cn[q]).all())))
            mcur.append(np.packbits(cn, axis=1))
            mu.append(np.packbits(un, axis=1))
            if sh == args.shards[0] and len(tiles) < 16:
                tiles.append((fr[ts[0]], {'cur': cn[0], 'U': un[0]}, f'{c[:6]} i{R.i.values[0]}', None))
            del fr
            if ci % 50 == 0:
                el = time.time() - t00
                log(f'  {base} clip {ci}/{len(clips)}: {len(meta)} frames ({len(meta) / max(el, 1e-9):.1f} frames/s wall), '
                    f'gpu mask {gpu_mask:.0f}s + D3L {enc.gpu_s:.0f}s, decode {tdec:.0f}s summed, waited {twait:.0f}s; '
                    f'kept cur {np.mean([m["n_cur"] for m in meta]):.1f} U {np.mean([m["n_u"] for m in meta]):.1f} '
                    f'union {np.mean([m["n"] for m in meta]):.1f}')
        pool.shutdown()
        Fm = pd.DataFrame(meta)
        saved = {}
        for k, v in A.items():
            a = np.concatenate(v)
            nm = f'{base}_{k}.npy'
            np.save(stage / nm, a)
            shutil.copyfile(stage / nm, od / nm)
            (stage / nm).unlink()
            saved[nm] = list(a.shape)
        Fm.to_parquet(od / f'{base}_frames.parquet')
        np.savez(od / f'{base}_masks.npz', cur=np.concatenate(mcur), U=np.concatenate(mu))
        if tiles:
            sheet(tiles, OUT / 'sheets' / f'mabe_encode_{base}.jpg')
        wall = time.time() - t00
        ag = np.array(agree) if agree else np.zeros((0, 2))
        res = {'set': args.set, 'shard': sh, 'nshard': args.nshard, 'clips': len(clips), 'frames': len(Fm), 'offset': off,
               'encoder': 'facebook/dinov3-vitl16-pretrain-lvd1689m', 'size': 512, 'layer': LAYER, 'bf16': True,
               'lag': LAG, 'gpu': torch.cuda.get_device_name(0), 'wall_s': round(wall, 1),
               'frames_per_wall_s': round(len(Fm) / wall, 2), 'gpu_s_mask': round(gpu_mask, 1),
               'gpu_s_d3l': round(enc.gpu_s, 1), 'd3l_inputs_per_gpu_s': round(enc.n_img / max(enc.gpu_s, 1e-9), 1),
               'decode_s_summed': round(tdec, 1), 'decode_wait_s': round(twait, 1), 'workers': args.workers,
               'kept_cur_mean': float(Fm.n_cur.mean()), 'kept_u_mean': float(Fm.n_u.mean()),
               'kept_union_mean': float(Fm.n.mean()), 'frames_0_cur': int((Fm.n_cur == 0).sum()),
               'frames_0_u': int((Fm.n_u == 0).sum()),
               'cur_vs_r3mask': {'frames_compared': int(len(ag)), 'cell_agreement': float(ag[:, 0].mean()) if len(ag) else None,
                                 'frames_identical': float(ag[:, 1].mean()) if len(ag) else None},
               'U_thr_pix_q': np.quantile([v['U']['thr_pix'] for v in infos.values()], [0, .5, 1]).tolist(),
               'U_still_px_mean': float(np.mean([v['U']['still_px_blocks'] for v in infos.values()])),
               'saved': saved}
        enc.gpu_s, enc.n_img = 0.0, 0
        (od / f'{base}.json').write_text(json.dumps(res, indent=1, default=float))
        (OUT / 'encode_json').mkdir(parents=True, exist_ok=True)
        (OUT / 'encode_json' / f'mabe_{base}.json').write_text(json.dumps(res, indent=1, default=float))
        log('done: ' + json.dumps(res, default=float))


if __name__ == '__main__':
    main()
