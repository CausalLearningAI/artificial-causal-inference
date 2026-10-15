"""
Mask-U test (rules: scripts/benchmark/eci_mask_u.sh) - CalMS21 token store for the two SAE arms (GPU job). OWN store:
nothing of the CalMS21 phase-2 stores is read or written.

Frames: results/vision/benchmark_calms21/plan/train_frames.parquet (10,000 train + 2,000 calib frames of the task-1
train videos, labels never read) and eval_frames.parquet (10,000 test frames, IPW weights); video frame = label i + 0.
Per video (copied to local disk, stream-decoded once by ffmpeg 6.1.1, --workers videos in parallel threads):
    canvas  1024 x 570 padded to 1024 x 1024 (ImageNet-mean rows), as the CalMS21 grid
    masks   on the 32 x 32 cell grid, DINOv2-base 448 tokens (bf16 autocast, as the grid) of the 32 evenly spaced
            background frames and of the plan frames t:
              cur  the current CalMS21 rule 'dark' (plan/mask_rule.txt; calms21 eci_encode.Masker)
              U    rule U (src/eci/mask_u.py; valid = the video rows of the canvas)
    tokens  the phase-2 base D3L-64: DINOv3-L/16 at 1024 px (64 x 64 grid), block 20 after the final LayerNorm, bf16 + SDPA
            (calms21 eci_encode.Enc 'd3l_1024_L20'), of t and t - 60, at the 64-grid children of the UNION of the two
            masks' cells; per token flags in_cur / in_u (inherited from the parent cell)
Outputs (BeeGFS TOK/calms/): <set>_<s>of<n>_{t,p60}.npy (fp16, N x 1024), _pos.npy (int16, 64 grid), _incur / _inu.npy,
_frames.parquet (video, i, v, n, n_cur, n_u in 64-grid tokens; n32_cur / n32_u in cells), _masks.npz (packed 32 x 32
cell bits: cur, U), <set>_<s>of<n>.json (timing, frames / s, GPU s). Mask sheet of the job's first shard.
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
sys.path.insert(0, str(REPO / 'scripts/benchmark/mabe22'))
sys.path.insert(0, str(REPO / 'scripts/benchmark/calms21'))  # before mabe22: calms21's eci_encode
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from common import OUT, TOK, log, sheet, stage_dir  # noqa: E402
from calms_common import OUT as COUT, PADY, PLAN, H, decode_frames, offset, video_path  # noqa: E402
from maskqa import calms_video_masks  # noqa: E402

LAG = 60
NBG = 32


def main():
    import torch
    from eci_encode import Enc, Masker, to_grid
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['train', 'eval'], required=True)
    ap.add_argument('--shards', type=int, nargs='+', required=True)
    ap.add_argument('--nshard', type=int, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--smoke', type=int, default=0)
    args = ap.parse_args()
    assert (PLAN / 'mask_rule.txt').read_text().split()[0] == 'dark'
    dev = torch.device('cuda')
    stage = stage_dir()
    od = TOK / 'calms'
    od.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(PLAN / f'{args.set}_frames.parquet')
    split = 'train' if args.set == 'train' else 'test'
    nfr = json.loads((COUT / 'inventory' / 'inventory.json').read_text())['videos'][split]['per_video']
    off = offset()
    m448 = Enc('d2_448', dev, True)
    masker = Masker(m448, dev)
    enc = Enc('d3l_1024_L20', dev, True)
    assert enc.G == 64
    cnt = P.groupby('video').size().reset_index(name='n').sort_values(['n', 'video'], ascending=[False, True], kind='stable')
    allv = list(cnt.video)
    for sh in args.shards:
        t00 = time.time()
        base = f'{args.set}_{sh}of{args.nshard}' + (f'_smoke{args.smoke}' if args.smoke else '')
        if (od / f'{base}.json').exists():
            log(f'[SKIP] {base}')
            continue
        vids = [v for j, v in enumerate(allv) if j % args.nshard == sh]
        if args.smoke:
            vids = vids[:args.smoke]
        pool = ThreadPoolExecutor(args.workers)

        def dec(q):
            t_ = time.time()
            loc = stage / f'{q}.mp4'
            shutil.copyfile(video_path(split, q), loc)
            v = P[P.video == q].i.values + off
            N = nfr[q]['frames']
            want = set(np.linspace(0, N - 1, NBG).round().astype(int).tolist()) | set(v.tolist()) | set((v - LAG).tolist())
            fr, _ = decode_frames(loc, want, threads=args.threads)
            loc.unlink()
            return fr, time.time() - t_
        fut = {j: pool.submit(dec, vids[j]) for j in range(min(len(vids), args.workers + 1))}
        A = {k: [] for k in ('t', 'p60', 'pos', 'incur', 'inu')}
        meta, mcur, mu, tiles = [], [], [], []
        tdec = twait = gpu_mask = 0.0
        g0 = enc.gpu_s
        for k, q in enumerate(vids):
            t0 = time.time()
            fr, dt = fut.pop(k).result()
            twait += time.time() - t0
            tdec += dt
            if k + args.workers + 1 < len(vids):
                fut[k + args.workers + 1] = pool.submit(dec, vids[k + args.workers + 1])
            R = P[P.video == q]
            v = R.i.values + off
            N = nfr[q]['frames']
            assert v.min() - LAG >= 0 and v.max() <= N - 1, q
            bgi = np.linspace(0, N - 1, NBG).round().astype(int)
            torch.cuda.synchronize()
            t1 = time.time()
            cur, U, parts, info, _ = calms_video_masks(m448, masker, [fr[x] for x in bgi], [fr[x] for x in v], dev,
                                                       t_bg=bgi, t_t=v)
            torch.cuda.synchronize()
            gpu_mask += time.time() - t1
            cn, un = cur.cpu().numpy(), U.cpu().numpy()
            uni = cn | un
            p32 = [np.flatnonzero(m) for m in uni]
            pg = [torch.from_numpy(to_grid(p, 64).astype(np.int64)).to(dev) for p in p32]
            for kd, vv in (('t', v), ('p60', v - LAG)):
                A[kd] += enc.run([fr[x] for x in vv], args.bs, pg)
            for j, r in enumerate(R.itertuples()):
                p64 = to_grid(p32[j], 64).astype(np.int64)
                par = (p64 // 64 // 2) * 32 + (p64 % 64) // 2
                A['pos'].append(p64.astype(np.int16))
                A['incur'].append(cn[j][par])
                A['inu'].append(un[j][par])
                meta.append({'video': q, 'i': int(r.i), 'v': int(r.i + off), 'n': int(len(p64)), 'n_cur': int(4 * cn[j].sum()),
                             'n_u': int(4 * un[j].sum()), 'n32_cur': int(cn[j].sum()), 'n32_u': int(un[j].sum())})
            mcur.append(np.packbits(cn, axis=1))
            mu.append(np.packbits(un, axis=1))
            if sh == args.shards[0] and len(tiles) < 8:
                canvas = np.full((1024, 1024, 3), 116, np.uint8)
                canvas[PADY:PADY + H] = fr[v[0]]
                tiles.append((canvas, {'cur': cn[0], 'U': un[0]}, f'{q[:9]} v{v[0]}', None))
            del fr
            el = time.time() - t00
            log(f'  {base} video {k + 1}/{len(vids)} {q}: {len(v)} frames; {len(meta) / el:.2f} frames/s wall; gpu mask '
                f'{gpu_mask:.0f}s D3L {enc.gpu_s - g0:.0f}s; decode {tdec:.0f}s summed, waited {twait:.0f}s; cells cur '
                f'{cn.sum(1).mean():.1f} U {un.sum(1).mean():.1f}; U thr_pix {info["U"]["thr_pix"]:.1f}')
        pool.shutdown()
        Fm = pd.DataFrame(meta)
        saved = {}
        for kk, lst in A.items():
            a = np.concatenate(lst)
            nm = f'{base}_{kk}.npy'
            np.save(stage / nm, a)
            shutil.copyfile(stage / nm, od / nm)
            (stage / nm).unlink()
            saved[nm] = list(a.shape)
        assert saved[f'{base}_t.npy'][0] == Fm.n.sum()
        Fm.to_parquet(od / f'{base}_frames.parquet')
        np.savez(od / f'{base}_masks.npz', cur=np.concatenate(mcur), U=np.concatenate(mu))
        if tiles:
            sheet(tiles, OUT / 'sheets' / f'calms_encode_{base}.jpg')
        wall = time.time() - t00
        res = {'set': args.set, 'shard': sh, 'nshard': args.nshard, 'videos': len(vids), 'frames': len(Fm),
               'encoder': 'DINOv3-L/16 1024 px block 20 (d3l_1024_L20), bf16', 'lag': LAG, 'gpu': torch.cuda.get_device_name(0),
               'wall_s': round(wall, 1), 'frames_per_wall_s': round(len(Fm) / wall, 3), 'gpu_s_mask': round(gpu_mask, 1),
               'gpu_s_d3l': round(enc.gpu_s - g0, 1), 'decode_s_summed': round(tdec, 1), 'decode_wait_s': round(twait, 1),
               'cells_cur_mean': float(Fm.n32_cur.mean()), 'cells_u_mean': float(Fm.n32_u.mean()),
               'tokens_union_mean': float(Fm.n.mean()), 'frames_0_cur': int((Fm.n32_cur == 0).sum()),
               'frames_0_u': int((Fm.n32_u == 0).sum()), 'saved': saved,
               'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        (od / f'{base}.json').write_text(json.dumps(res, indent=1, default=float))
        (OUT / 'encode_json').mkdir(parents=True, exist_ok=True)
        (OUT / 'encode_json' / f'calms_{base}.json').write_text(json.dumps(res, indent=1, default=float))
        log('done: ' + json.dumps(res, default=float))


if __name__ == '__main__':
    main()
