"""
v0 audit round 2, OUR re-implementation path (pre-registration: see align.py docstring, 2026-10-20).

Same 8 SAEs as round 1 (overcomplete TopKSAE 384 -> 4608, k 20, Adam 5e-4, MSE, clip 1.0, 10 epochs, no
shuffle, trained on ALL 52,800 frames x 256 patch tokens of the 44 valid videos; no split):
  cached = the GPU seed-0 SAE of scripts/eci/v0 (batch 128); b128 seeds 1-4 and b20 seeds 0-2 retrained on CPU
  with the round-1 seeds (also a reproducibility check against round 1). Steps: b128 = 413 / epoch (4130),
  b20 = 2640 / epoch (26,400).
Encodings: mean (paper text), tok0 (top-left patch = the old s4 'CLS' default), patchavg (diagnostic).
Writes audit2_<key>.json (+ per-code arrays audit2_<key>_<enc>.npz) and saves the SAE to BeeGFS.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v0'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ants_v2_v0 import train_sae, N_FRAMES, DIM  # noqa: E402
from nes_original import neural_effect_search  # noqa: E402
from audit import encode  # noqa: E402
from align import analyse  # noqa: E402

BIG = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_v0_audit/ants_v2')


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--token-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--key', required=True, help='cached | b128_s<seed> | b20_s<seed>')
    ap.add_argument('--threads', type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    BIG.mkdir(parents=True, exist_ok=True)
    exp = pd.read_csv(REPO / 'data/ants/v2/experiment.csv')
    videos = exp[exp.valid == 1].reset_index(drop=True)
    assert len(videos) == 44
    fr = pd.read_csv(Path(a.token_dir) / 'frames.csv')
    assert (fr.observation_id.to_numpy() == np.repeat(videos.observation_id, N_FRAMES).to_numpy()).all()
    y = ((fr.groom_yellow + fr.groom_blue) > 0).to_numpy().astype(np.int8)
    pos1 = np.repeat(((videos.position == 1) & (videos.batch != 'e')).to_numpy(), N_FRAMES).astype(np.int8)
    tokens = np.concatenate([np.load(Path(a.token_dir) / f'tokens_batch_{b}.npy') for b in 'abcde'])
    assert tokens.shape == (52800, 256, DIM)
    log(f'tokens {tokens.shape}: SAE trains on ALL of them (44 videos x 1200 frames x 256 patches)')
    if a.key == 'cached':
        sae = torch.load('/mnt/beegfs/locatgrp/shared/rcadei/eci_v0/ants_v2/sae_topk_4608_k20_s0.pt', map_location='cpu', weights_only=False)
        hist = [dict(note='GPU seed-0 SAE of scripts/eci/v0, batch 128')]
    else:
        bf, seed = int(a.key.split('_')[0][1:]), int(a.key.split('_s')[1])
        sae, hist = train_sae(tokens, 'cpu', 10, bf, seed)
        torch.save(sae, BIG / f'sae_{a.key}.pt')
    sae.eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(key=a.key, train=hist, n_train_frames=int(len(tokens)), n_train_videos=44,
               steps_per_epoch=None if a.key == 'cached' else int(np.ceil(len(tokens) / bf)))
    rng = np.random.default_rng(0)
    for mode in ['mean', 'tok0', 'patchavg']:
        codes, fve = encode(sae, tokens, mode, 'cpu')
        r, per = analyse(codes, videos, y, pos1, neural_effect_search, rng)
        r['fve_on_input'] = fve
        res[mode] = r
        np.savez_compressed(out / f'audit2_{a.key}_{mode}.npz', **per)
        log(f"{a.key}/{mode}: alive {r['n_alive']} mean-agg hits {r['mean']['n_hits']} rec {r['mean']['recursive_hits']} | "
            f"rate-agg hits {r['rate']['n_hits']} | best AUROC {r['best_auroc']:.3f}")
        (out / f'audit2_{a.key}.json').write_text(json.dumps(res, indent=1))
    log('DONE')


if __name__ == '__main__':
    main()
