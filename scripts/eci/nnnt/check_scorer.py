"""Check nnnt.half_stats against the existing scorers on synthetic fp16 codes with many ties (zeros): AUROC / AP vs
src/eci/levers.align_gpu, normalised R-precision vs scripts/eci/rollout_r4.rprec_halves. GPU, a few seconds."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
sys.path.insert(0, str(REPO / 'scripts/eci/nnnt'))
import nnnt  # noqa: E402
import rollout_r4 as R4  # noqa: E402
from src.eci.levers import align_gpu  # noqa: E402

dev = torch.device('cuda')
g = np.random.default_rng(0)
n, W = 50_000, 64
X = np.where(g.random((n, W)) < 0.8, 0, g.gamma(2, 1, (n, W))).astype(np.float16)
X[:, :8] = np.round(X[:, :8])  # heavy ties
y = g.random(n) < 0.02
X[y, 3] += 2  # one informative neuron
Xg, Yg = torch.from_numpy(X).to(dev), torch.from_numpy(np.stack([y, ~y], 1)).to(dev)
rows = torch.from_numpy(np.sort(g.choice(n, n // 2, replace=False))).to(dev)
H = nnnt.half_stats(torch, Xg, Yg, rows, block=16)
A = align_gpu(Xg, Yg, rows=rows)
r, pi = R4.rprec_halves(Xg, Yg[:, 0], rows, dev)
print('max |AUROC diff|', np.nanmax(np.abs(H['A'] - A['auc'])), '| max |AP diff|', np.nanmax(np.abs(H['AP'] - A['ap'])),
      '| max |R~ diff|', np.nanmax(np.abs(H['R'][:, 0] - r)), '| informative neuron R~', H['R'][3, 0], 'AUROC',
      H['A'][3, 0])
assert np.nanmax(np.abs(H['A'] - A['auc'])) < 1e-9 and np.nanmax(np.abs(H['R'][:, 0] - r)) < 1e-6
print('SCORER OK')
