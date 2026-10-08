"""
T2: encode the tight-mask patches that the existing foreground token stores do not hold (GPU).

The tight rule (src/eci/tight_mask.py) keeps some patches today's mask did not (mostly lighter-than-background or
changed-floor pixels); their share of the new kept patches is in results/vision/eci_t2t4/<d>/mask/apply.json
('frac_new_absent_from_store': mice 7.1%, above the 2% skip limit). They are encoded here with exactly the store's
encoder path (src/eci/foreground.py FgEncoder('dinov2_base'), FrameDatasetFG, encode_batch: whole 512 px frame resized
to 448, fp32 forward, fp16 rounding; no alignment), and the same frames' STORE patches are compared with the stored
tokens as a check of identical encoding.

Output: results/vision/eci_t2t4/<d>/extra/part_<task>/{tok.npy (N, 768) fp16, fidx.npy (N,) int32 = index into
masks.npz frames, pos.npy (N,) int16, check.json}. Written on /localhome, copied back once.

Usage: python scripts/eci/tight_mask_encode.py --domain mice --task 0 --n-tasks 2
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import spatial_sae_pilot as ssp  # noqa: E402
import tight_mask as tm  # noqa: E402
from src.eci.foreground import FgEncoder, encode_batch  # noqa: E402

OUT = REPO / 'results/vision/eci_t2t4'
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True)
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--n-tasks', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=48)
    ap.add_argument('--num-workers', type=int, default=8)
    args = ap.parse_args()
    import os
    loc = Path(os.environ['LOCAL_DIR']) / f'extra_{args.task}'
    loc.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    z = np.load(OUT / args.domain / 'mask' / 'masks.npz')
    sel = np.flatnonzero(z['new_only'] > 0)
    sel = sel[args.task::args.n_tasks]
    idx = ssp.StoreIndex(args.domain)
    frames = z['frames'][sel]
    new = np.unpackbits(z['bits'][sel], axis=1).astype(bool)
    old = tm.old_masks(idx, frames)
    extra = new & ~old
    assert (extra.sum(1) == z['new_only'][sel]).all()
    n = int(extra.sum())
    log(f'{args.domain} task {args.task}/{args.n_tasks}: {len(sel):,} frames with absent patches, {n:,} tokens to encode')
    paths = tm.frame_paths(args.domain)[z['rows'][sel]]
    enc = FgEncoder('dinov2_base', dev)
    dset = enc.dataset([str(REPO / 'dataset' / q) for q in paths], rows=np.arange(len(sel)))
    loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    tok = np.lib.format.open_memmap(loc / 'tok.npy', 'w+', np.float16, (n, 768))
    fidx = np.empty(n, np.int32)
    pos = np.empty(n, np.int16)
    o, t0, chk = 0, time.time(), []
    for bi, (pix, grey, r) in enumerate(loader):
        T = encode_batch(enc.model, pix, dev).cpu().numpy()  # (B, 1024, 768) fp16
        r = r.numpy()
        for k, i in enumerate(r):
            p = np.flatnonzero(extra[i])
            tok[o:o + len(p)] = T[k, p]
            fidx[o:o + len(p)] = sel[i]
            pos[o:o + len(p)] = p
            o += len(p)
        if bi < 3:  # identical-encoding check on the same frames' store patches
            st, ps, ln = idx.load(frames[r])
            s0 = np.r_[0, np.cumsum(ln)]
            d = max(float(np.abs(st[s0[k]:s0[k + 1]].astype(np.float32) - T[k, ps[s0[k]:s0[k + 1]]].astype(np.float32)).max())
                    for k in range(len(r)) if ln[k])
            ref = float(np.abs(st.astype(np.float32)).mean())
            chk.append({'batch': bi, 'max_abs_diff_vs_store': d, 'mean_abs_store_token': ref})
            log(f'  check batch {bi}: max |re-encoded - stored| {d:.4g} (mean |token| {ref:.3f})')
        if bi % 200 == 0:
            log(f'  batch {bi}/{len(loader)} ({time.time() - t0:.0f}s)')
    assert o == n
    tok.flush()
    del tok
    np.save(loc / 'fidx.npy', fidx)
    np.save(loc / 'pos.npy', pos)
    (loc / 'check.json').write_text(json.dumps({'n_frames': int(len(sel)), 'n_tokens': n, 'checks': chk,
                                                'elapsed_s': round(time.time() - t0, 1)}, indent=1))
    dst = OUT / args.domain / 'extra' / f'part_{args.task}'
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(loc, dst, dirs_exist_ok=True)
    shutil.rmtree(loc)
    log(f'done -> {dst} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
