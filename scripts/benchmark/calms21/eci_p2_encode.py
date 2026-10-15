"""
CalMS21 ECI phase 2 (rules: eci_recheck.sh) - extra token stores for the BASE config (GPU job; several pass-A shards
per job, each video decoded once for every part). Reuses pass A's mask cells and video list (TOK/<set>_<i>of6_frames
.parquet, _pos32.npy). Same canvas / resize / encoder / layer / bf16 as the base arm (eci_encode.Enc).
Parts (--parts):
    lags     masked tokens of the lag frames t - 30 (1 s) and t - 120 (4 s) at t's cells (lag frame clamped to 0)
             -> <base>_<C>_p30.npy, _p120.npy
    nomask   train: t and t - 60 tokens at a uniformly random subset of the VIDEO cells (padding excluded) of the same
             size as the frame's masked count (rng seeded by video and i)  -> _<C>_rnd_{t,p60,pos}.npy;
             eval: t and t - 60 tokens of ALL video cells  -> _<C>_all_{t,p60}.npy (positions implicit, row-major)
    orbit    t and t - 60 tokens averaged over the 8 D4 transforms of the (square) canvas, each transformed grid read
             back at the original patch positions (priors.d4_pos_map on the G grid; mabe22 eci_encode.Enc.orbit
             convention), at the masked cells  -> _<C>_orb_{t,p60}.npy
Outputs on BeeGFS (TOK); JSON with timing / throughput -> TOK and OUT/encode_json/p2_*.json
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'mabe22'))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2] / 'scripts/eci'))
from calms_common import OUT, PLAN, TOK, decode_frames, log, offset, stage_dir, video_path  # noqa: E402
from eci_encode import Enc, canvas_gpu, device, gpu_name, sync, to_grid, video_cells  # noqa: E402

LAGS = (30, 120)


def video_grid_cells(G):
    vc = video_cells().reshape(32, 32)
    if G == 64:
        vc = np.kron(vc, np.ones((2, 2), bool))
    return np.flatnonzero(vc.ravel())


def orbit_run(e, frames, bs, pos, maps):
    """D4-averaged tokens at pos (list of index tensors) -> list of (n_i, d) fp16 numpy."""
    import torch
    import invariant_tokens as it
    out = []
    sync(e.dev)
    t0 = time.time()
    with torch.inference_mode():
        for a in range(0, len(frames), bs):
            part = frames[a:a + bs]
            x = e.prep(canvas_gpu(part, e.dev))
            acc = 0
            for g in range(8):
                h = e.e.forward(it.d4_torch(x, g))[e.last]
                acc = acc + h[:, maps[g]]
            acc = acc / 8
            out += [acc[j, pos[a + j]].half().cpu().numpy() for j in range(len(part))]
            e.n_in += 8 * len(part)
            e.n_tok += 8 * len(part) * (e.e.npre + e.G ** 2)
    sync(e.dev)
    e.gpu_s += time.time() - t0
    return out


def main():
    import torch
    from src.eci.priors import d4_pos_map
    import invariant_tokens as it
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['train', 'eval'], required=True)
    ap.add_argument('--shards', type=int, nargs='+', required=True)
    ap.add_argument('--nshard', type=int, default=6)
    ap.add_argument('--config', required=True)
    ap.add_argument('--parts', nargs='+', default=['lags', 'nomask', 'orbit'])
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads', type=int, default=3)
    args = ap.parse_args()
    dev = device()
    stage = stage_dir()
    off = offset()
    split = 'train' if args.set == 'train' else 'test'
    nfr = json.loads((OUT / 'inventory' / 'inventory.json').read_text())['videos'][split]['per_video']
    P = pd.read_parquet(PLAN / f'{args.set}_frames.parquet')
    e = Enc(args.config, dev, True)
    assert not e.video, 'phase-2 parts are defined for an image encoder base'
    G = e.G
    vgc = video_grid_cells(G)
    maps = [torch.from_numpy(d4_pos_map(g, G)).to(dev) for g in range(8)] if 'orbit' in args.parts else None
    if maps is not None:
        assert it.check_d4_torch()
    C = args.config
    for sh in args.shards:
        t00 = time.time()
        base = f'{args.set}_{sh}of{args.nshard}'
        jkey = f'p2_{base}_{C}_' + '_'.join(args.parts)
        if (TOK / f'{jkey}.json').exists():
            log(f'[SKIP] {jkey}')
            continue
        RF = pd.read_parquet(TOK / f'{base}_frames.parquet')
        pos_all = np.load(TOK / f'{base}_pos32.npy')
        pst = np.r_[0, np.cumsum(RF.n32.values)]
        vids = list(dict.fromkeys(RF.video.values))
        e.gpu_s, e.n_in, e.n_tok = 0.0, 0, 0
        gs = {}
        pool = ThreadPoolExecutor(args.workers)

        def dec(q):
            t_ = time.time()
            loc = stage / f'{q}.mp4'
            shutil.copyfile(video_path(split, q), loc)
            v = RF.v.values[RF.video.values == q]
            want = set(v.tolist()) | set(np.maximum(v - 60, 0).tolist())
            for L_ in LAGS:
                want |= set(np.maximum(v - L_, 0).tolist())
            fr, _ = decode_frames(loc, want, threads=args.threads)
            loc.unlink()
            return fr, time.time() - t_
        fut = {j: pool.submit(dec, vids[j]) for j in range(min(len(vids), args.workers + 1))}
        A = {}
        tdec = twait = 0.0
        for k, q in enumerate(vids):
            t0 = time.time()
            fr, dt = fut.pop(k).result()
            twait += time.time() - t0
            tdec += dt
            if k + args.workers + 1 < len(vids):
                fut[k + args.workers + 1] = pool.submit(dec, vids[k + args.workers + 1])
            rr = np.flatnonzero(RF.video.values == q)
            v = RF.v.values[rr]
            assert (RF.i.values[rr] + off == v).all()
            p32 = [pos_all[pst[r]:pst[r + 1]] for r in rr]
            pg = [torch.from_numpy(to_grid(p, G).astype(np.int64)).to(dev) for p in p32]
            bsz = args.bs

            def put(key, lst):
                A.setdefault(key, []).extend(lst)
            if 'lags' in args.parts:
                for L_ in LAGS:
                    t_ = time.time()
                    put(f'p{L_}', e.run([fr[max(x - L_, 0)] for x in v], bsz, pg))
                    gs[f'lag{L_}'] = gs.get(f'lag{L_}', 0) + time.time() - t_
            if 'nomask' in args.parts:
                t_ = time.time()
                if args.set == 'train':
                    rp = []
                    for j, r in enumerate(rr):
                        n = len(pg[j])
                        g_ = np.random.default_rng(zlib.crc32(f'{q}_{RF.i.values[r]}'.encode()))
                        rp.append(np.sort(g_.choice(vgc, n, replace=False)).astype(np.int64))
                    rpt = [torch.from_numpy(p).to(dev) for p in rp]
                    put('rnd_t', e.run([fr[x] for x in v], bsz, rpt))
                    put('rnd_p60', e.run([fr[x - 60] for x in v], bsz, rpt))
                    put('rnd_pos', [p.astype(np.int16) for p in rp])
                else:
                    allp = [torch.from_numpy(vgc.astype(np.int64)).to(dev)] * len(v)
                    put('all_t', e.run([fr[x] for x in v], bsz, allp))
                    put('all_p60', e.run([fr[x - 60] for x in v], bsz, allp))
                gs['nomask'] = gs.get('nomask', 0) + time.time() - t_
            if 'orbit' in args.parts:
                t_ = time.time()
                put('orb_t', orbit_run(e, [fr[x] for x in v], max(1, bsz // 2), pg, maps))
                put('orb_p60', orbit_run(e, [fr[x - 60] for x in v], max(1, bsz // 2), pg, maps))
                gs['orbit'] = gs.get('orbit', 0) + time.time() - t_
            del fr
            log(f'  {base} video {k + 1}/{len(vids)} {q}: {len(v)} frames; '
                f'gpu {e.gpu_s:.0f}s ({json.dumps({a: round(b) for a, b in gs.items()})}); waited {twait:.0f}s')
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
        info = {'set': args.set, 'shard': sh, 'config': C, 'parts': args.parts, 'frames': len(RF), 'gpu': gpu_name(dev),
                'wall_s': round(wall, 1), 'frames_per_wall_s': round(len(RF) / wall, 3), 'gpu_s': round(e.gpu_s, 1),
                'gpu_s_by_part': {a: round(b, 1) for a, b in gs.items()}, 'inputs': e.n_in,
                'inputs_per_gpu_s': round(e.n_in / max(e.gpu_s, 1e-9), 2), 'decode_s_summed': round(tdec, 1),
                'decode_wait_s': round(twait, 1), 'video_cells_on_grid': int(len(vgc)), 'saved': saved,
                'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == 'cuda' else None}
        (TOK / f'{jkey}.json').write_text(json.dumps(info, indent=1))
        (OUT / 'encode_json').mkdir(parents=True, exist_ok=True)
        (OUT / 'encode_json' / f'{jkey}.json').write_text(json.dumps(info, indent=1))
        log('done: ' + json.dumps(info))


if __name__ == '__main__':
    main()
