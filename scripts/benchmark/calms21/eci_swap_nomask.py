"""
CalMS21 ECI - ROLE SWAP, EXPLORATORY arm S5 (eci_swap.sh addendum 2): no-mask tokens for D3L-64, the construction of
the grid's phase-2 P_NOMASK (eci_p2_encode.py --parts nomask) with the set roles swapped:
    SAE pool (store set 'eval' = TEST videos): t and t - 60 tokens at a uniformly random subset of the frame's VIDEO
        cells (padding excluded) of the same size as its masked count (rng seeded by crc32 of video_i, as P_NOMASK)
        -> <base>_<C>_rnd_{t,p60}.npy, _rnd_pos.npy
    eval frames (store set 'train' = TRAIN videos): t and t - 60 tokens of ALL video cells, row-major
        -> <base>_<C>_all_{t,p60}.npy
Reuses the base pass's frames parquet and mask counts. GPU job; CALMS_OUT / CALMS_TOK as eci_swap_job.sh.
"""
import argparse
import json
import shutil
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mabe22'))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
from calms_common import OUT, TOK, decode_frames, log, stage_dir, video_path  # noqa: E402
from eci_encode import Enc, device, gpu_name, to_grid  # noqa: E402
from eci_p2_encode import video_grid_cells  # noqa: E402


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['train', 'eval'], required=True)
    ap.add_argument('--shards', type=int, nargs='+', required=True)
    ap.add_argument('--nshard', type=int, required=True)
    ap.add_argument('--config', default='d3l_1024_L20')
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    assert str(OUT).endswith('benchmark_calms21/swap') and 'pairpush' in str(TOK)
    dev = device()
    stage = stage_dir()
    split = 'train' if args.set == 'train' else 'test'
    pool_set = args.set == 'eval'          # swapped roles: 'eval' = SAE pool (random cells), 'train' = eval (all cells)
    e = Enc(args.config, dev, True)
    G, C = e.G, args.config
    vgc = video_grid_cells(G)
    for sh in args.shards:
        t00 = time.time()
        base = f'{args.set}_{sh}of{args.nshard}'
        jkey = f'nomask_{base}_{C}'
        if (TOK / f'{jkey}.json').exists():
            log(f'[SKIP] {jkey}')
            continue
        RF = pd.read_parquet(TOK / f'{base}_frames.parquet')
        vids = list(dict.fromkeys(RF.video.values))
        e.gpu_s, e.n_in, e.n_tok = 0.0, 0, 0
        pool = ThreadPoolExecutor(args.workers)

        def dec(q):
            loc = stage / f'{q}.mp4'
            shutil.copyfile(video_path(split, q), loc)
            v = RF.v.values[RF.video.values == q]
            fr, _ = decode_frames(loc, set(v.tolist()) | set((v - 60).tolist()), threads=3)
            loc.unlink()
            return fr
        fut = {j: pool.submit(dec, vids[j]) for j in range(min(len(vids), args.workers + 1))}
        A = {}
        for k, q in enumerate(vids):
            fr = fut.pop(k).result()
            if k + args.workers + 1 < len(vids):
                fut[k + args.workers + 1] = pool.submit(dec, vids[k + args.workers + 1])
            rr = np.flatnonzero(RF.video.values == q)
            v = RF.v.values[rr]
            if pool_set:
                rp = []
                for r in rr:
                    n = int(RF.n32.values[r]) * (4 if G == 64 else 1)
                    g_ = np.random.default_rng(zlib.crc32(f'{q}_{RF.i.values[r]}'.encode()))
                    rp.append(np.sort(g_.choice(vgc, n, replace=False)).astype(np.int64))
                pt = [torch.from_numpy(p).to(dev) for p in rp]
                A.setdefault('rnd_t', []).extend(e.run([fr[x] for x in v], args.bs, pt))
                A.setdefault('rnd_p60', []).extend(e.run([fr[x - 60] for x in v], args.bs, pt))
                A.setdefault('rnd_pos', []).extend([p.astype(np.int16) for p in rp])
            else:
                pt = [torch.from_numpy(vgc.astype(np.int64)).to(dev)] * len(v)
                A.setdefault('all_t', []).extend(e.run([fr[x] for x in v], args.bs, pt))
                A.setdefault('all_p60', []).extend(e.run([fr[x - 60] for x in v], args.bs, pt))
            del fr
            log(f'  {base} video {k + 1}/{len(vids)} {q}: {len(v)} frames; gpu {e.gpu_s:.0f}s; '
                f'{sum(len(RF.v.values[RF.video.values == x]) for x in vids[:k + 1]) / (time.time() - t00):.2f} frames/s wall')
        pool.shutdown()
        saved = {}
        for kd, lst in A.items():
            a = np.concatenate(lst)
            nm = f'{base}_{C}_{kd}.npy'
            np.save(stage / nm, a)
            shutil.copyfile(stage / nm, TOK / nm)
            (stage / nm).unlink()
            saved[nm] = list(a.shape)
            log(f'  saved {nm} {a.shape}')
        wall = time.time() - t00
        info = {'set': args.set, 'shard': sh, 'config': C, 'frames': len(RF), 'gpu': gpu_name(dev), 'wall_s': round(wall, 1),
                'frames_per_wall_s': round(len(RF) / wall, 3), 'gpu_s': round(e.gpu_s, 1), 'inputs': e.n_in,
                'inputs_per_gpu_s': round(e.n_in / max(e.gpu_s, 1e-9), 2), 'video_cells_on_grid': int(len(vgc)),
                'saved': saved}
        (TOK / f'{jkey}.json').write_text(json.dumps(info, indent=1))
        (OUT / 'encode_json').mkdir(parents=True, exist_ok=True)
        (OUT / 'encode_json' / f'{jkey}.json').write_text(json.dumps(info, indent=1))
        log('done: ' + json.dumps(info))


if __name__ == '__main__':
    main()
