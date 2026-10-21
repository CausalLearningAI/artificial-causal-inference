"""Bimodality arm on the RAW (unsmoothed) frame max codes of the calibration videos (same statistic as prob.bimodality),
to separate real two-level activations from the partial-window levels the 0.6 s mean creates. -> OUT/<domain>/bimodality_raw.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run as RUN  # noqa: E402
from src.eci.v2 import prob as PR  # noqa: E402

dom = sys.argv[1]
c = RUN.dcfg(dom)
name = RUN.best_name(c)
cal = []
for sc in c['series']:
    P = json.loads((RUN.REPO / sc['plan']).read_text())
    mx = np.load(RUN.codes_dir(c, sc, f'{name}_raw') / 'codes_max.npy', mmap_mode='r')
    cal += [np.asarray(mx[v['lay_lo']:v['lay_lo'] + v['n']]).astype(np.float32) for v in P['videos'] if v['role'] == 'calib']
gain, sep, npos = PR.bimodality(np.concatenate(cal))
ok = np.isfinite(gain)
sm = np.load(RUN.OUT / dom / 'bimodality.npz')
out = {'raw_bic_gain_median': float(np.nanmedian(gain)), 'raw_share_bic_gain_gt10': float((gain[ok] > 10).mean()),
       'raw_separation_median': float(np.nanmedian(sep)),
       'raw_separation_p10_p90': [float(np.nanquantile(sep, .1)), float(np.nanquantile(sep, .9))],
       'smoothed_separation_median': float(np.nanmedian(sm['separation'])),
       'share_neurons_raw_separation_ge_2': float((sep[ok] >= 2).mean())}
(RUN.OUT / dom / 'bimodality_raw.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out))
