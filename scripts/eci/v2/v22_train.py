"""
ECI v2.2 SAE variants (pre-registration scripts/eci/v2/v22_prereg.md). Same SAE trainer as V (src/eci/v2/sae.train:
Matryoshka BatchTopK, k16, AuxK, TokenNorm, 30k steps x 4096, lr 5e-4, theta* on calibration tokens), same V input
x + (x_{t+1} - x_{t-1}) / 2 on mask-v5 tokens. Only what is named changes:
  VP2  trained on the 2x2-pooled tokens only (v22_encode prep, <role>_P2.npy); read out on pooled tokens
  VMS  trained on native tokens + 2x2-pooled tokens together (all of both, mixed uniformly); read out on both (ms)
  VW8  8192 latents, prefixes 128/256/512/2048/8192 (V's four levels plus a deeper 6144-latent level); native tokens
Step: train --dom D --variant X --seed s  (GPU) -> BEE22/<dom>/sae/<X>_s<seed>, OUT22/train/<dom>_<X>_s<seed>.json
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v21_ladder as L  # noqa: E402

T = L.T
BEE22 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2')
OUT22 = L.REPO / 'results/vision/eci_v2_2'
VARIANTS = {'VP2': {'data': ('P2',)}, 'VMS': {'data': ('V', 'P2')},
            'VW8': {'data': ('V',), 'W': 8192, 'prefixes': [128, 256, 512, 2048, 8192]}}


def main():
    import torch
    import v20 as V
    from src.eci.v2 import sae as SA
    p = argparse.ArgumentParser()
    p.add_argument('--dom', required=True)
    p.add_argument('--variant', required=True, choices=tuple(VARIANTS))
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    spec = VARIANTS[a.variant]
    dev = torch.device('cuda')
    od, od2 = T.BEE / a.dom / 'data', BEE22 / a.dom / 'data'
    X, info = {}, {}
    for r in T.CAP:
        parts = []
        for d in spec['data']:
            if d == 'V':  # exactly v21_train.cmd_train arm V
                B_ = np.load(od / f'{r}_B.npy', mmap_mode='r')
                parts.append(torch.from_numpy(np.asarray(B_[:, :1024], np.float32) + np.load(od / f'{r}_C.npy').astype(np.float32)).half())
            else:
                parts.append(torch.from_numpy(np.load(od2 / f'{r}_{d}.npy')))
        info[r] = [len(x) for x in parts]
        X[r] = torch.cat(parts) if len(parts) > 1 else parts[0]
    cfg = {**V.R.spec()['sae'], 'context_window': 0, 'arm': a.variant, 'store': T.VER, 'data': list(spec['data'])}
    for k in ('W', 'prefixes'):
        if k in spec:
            cfg[k] = spec[k]
    key = f'{a.variant}_s{a.seed}'
    t0 = time.time()
    rec = SA.train(cfg, a.seed, V.HostTok(X['train'], dev), X['calib'], X['eval'], BEE22 / a.dom / 'sae' / key, dev)
    rec.update(node=socket.gethostname(), gpu=torch.cuda.get_device_name(0), tokens_per_part=info,
               wall_s=time.time() - t0, tokens_per_s=cfg['steps'] * cfg.get('bs', 4096) / rec['train_time_s'])
    (OUT22 / 'train').mkdir(parents=True, exist_ok=True)
    (OUT22 / 'train' / f'{a.dom}_{key}.json').write_text(json.dumps(rec, indent=1, default=float))
    L.log(f'{a.dom} {key}: eval FVE {rec["eval_fve"]:.5f} L0 {rec["eval_l0"]:.2f} dead {rec["dead_at_end_train"]} '
          f'({rec["train_time_s"]:.0f}s, {rec["gpu"]}, {rec["tokens_per_s"]:.0f} tokens/s)')


if __name__ == '__main__':
    main()
