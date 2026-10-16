"""
Follow-up checks for the ECI v0 run on ants v2 (scripts/eci/v0/ants_v2_v0.py), CPU only:
  1. Reconstruction of the MEAN-POOLED tokens (the SAE input the paper actually uses), on batch a
     (9 videos, 10,800 frames): FVE with the across-frame mean (as in the main run), and the error
     relative to the vector norm (1 - SSE / sum ||x||^2), next to the token-level numbers.
  2. Background / position checks for the found codes: per-position mean of the video-level code,
     frame-level F1 (code > 0) against each position / batch one-hot (the paper's 550 had F1 0.52 for
     position 1), and permutation p-values for the share of video-level variance explained by
     position / batch / day (10,000 label permutations).
  3. The trivial F1 baseline for grooming: a code that is always on.
Writes results/vision/ants/eci_v0/checks.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ants_v2_v0 import DIM, N_CODES, N_FRAMES, FVE, eta2, f1_rule  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
BIG = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_v0/ants_v2')
OUT = REPO / 'results/vision/ants/eci_v0'


def main():
    torch.set_num_threads(8)
    res = json.loads((OUT / 'results.json').read_text())
    videos = pd.read_csv(OUT / 'videos.csv')
    found = [f['index0'] for f in res['found']]

    # 1. reconstruction on batch a
    sae = torch.load(BIG / 'sae_topk_4608_k20_s0.pt', map_location='cpu', weights_only=False).eval()
    tok = np.load(BIG / 'tokens_batch_a.npy', mmap_mode='r')
    f_tok, f_pool = FVE(), FVE()
    sse_pool = norm_pool = 0.0
    with torch.inference_mode():
        for s in range(0, len(tok), 64):
            x = torch.from_numpy(np.ascontiguousarray(tok[s:s + 64]))
            _, _, xh = sae(x.reshape(-1, DIM))
            f_tok.update(x.reshape(-1, DIM), xh)
            p = x.mean(1)
            _, _, ph = sae(p)
            f_pool.update(p, ph)
            sse_pool += ((p - ph).double() ** 2).sum().item()
            norm_pool += (p.double() ** 2).sum().item()
    res_chk = dict(batch_a_frames=len(tok), fve_tokens=f_tok.value(), fve_meanpooled=f_pool.value(),
                   meanpooled_rel_error_vs_norm=sse_pool / norm_pool)
    print('reconstruction (batch a):', res_chk, flush=True)

    # 2. background / position checks
    codes = np.load(BIG / 'frame_codes_meanpool.npy', mmap_mode='r')
    vcodes = np.load(OUT / 'video_codes.npy')
    rng = np.random.default_rng(0)
    per = {}
    for j in found:
        c = np.asarray(codes[:, j])
        act = c > 0
        v = vcodes[:, j]
        row = dict(position_means={int(k): float(v[videos.position == k].mean()) for k in sorted(videos.position.unique())},
                   batch_means={k: float(v[videos.batch == k].mean()) for k in sorted(videos.batch.unique())})
        row['f1_by_position'] = {int(k): f1_rule(act, np.repeat((videos.position == k).to_numpy(), N_FRAMES).astype(np.int8))['f1']
                                 for k in sorted(videos.position.unique())}
        row['f1_by_batch'] = {k: f1_rule(act, np.repeat((videos.batch == k).to_numpy(), N_FRAMES).astype(np.int8))['f1']
                              for k in sorted(videos.batch.unique())}
        for name, g in [('position', videos.position), ('batch', videos.batch), ('day', videos.recording_date)]:
            obs = eta2(v, g)
            null = np.array([eta2(v, rng.permutation(g.to_numpy())) for _ in range(10000)])
            row[f'eta2_{name}'] = obs
            row[f'eta2_{name}_null_mean'] = float(null.mean())
            row[f'eta2_{name}_perm_p'] = float((1 + (null >= obs).sum()) / (1 + len(null)))
        per[f'N_{j + 1}'] = row
        print(f'N_{j + 1}:', json.dumps(row), flush=True)
    res_chk['found_background'] = per

    # 3. trivial grooming F1 baseline (always on)
    fr = pd.read_csv(BIG / 'frames.csv')
    prev = float(((fr.groom_yellow + fr.groom_blue) > 0).mean())
    res_chk['always_on_f1_groom_any'] = 2 * prev / (1 + prev)
    print('always-on F1 (groom any):', res_chk['always_on_f1_groom_any'])
    (OUT / 'checks.json').write_text(json.dumps(res_chk, indent=1))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
