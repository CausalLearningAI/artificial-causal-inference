"""Reconstruct config.json for the odour_tr{F,S}_derm_last_popw runs.

Those two runs trained to completion (30 epochs, ~4 GPU-hours each) and saved best_model.pt
and val_probs.npz, then crashed at the very last statement: the final json.dump referenced
`args.derm_env_prior`, an argument the 12:01-12:09 refactor renamed to `--derm-prevalence`
without updating the dump line. Everything the summary would have contained is recoverable:

  - the launch arguments are fixed by xfit_odour.sh (WEIGHTS=corrected SELECT=last), so they
    are copied from the sibling `_last` DERM config and the two popw overrides re-applied
    (derm_prevalence=population, derm_floor=1e-4 -- the trainer's mode-dependent default);
  - ap_report / rate_report / best_ap are pure functions of val_probs.npz, recomputed here
    with the same src.mice_behavior.metrics code the trainer calls;
  - history exists only in wandb (runs zbp54spt / the trS sibling) and is left empty.

Without config.json the run is invisible downstream: xfit_odour.sh's predict stage gates on
it and predict_dense.py reads input_size/context_k/train_odour and the build_model keys.

Usage:  python scripts/mice_behavior/recover_popw_config.py odour_trF_derm_last_popw
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.metrics import ap_report, rate_report

FRAME = Path(__file__).parent.parent.parent / 'results/vision/mice/frame'


def main(tag: str) -> None:
    assert tag.endswith('_popw'), tag
    out = FRAME / tag
    dst = out / 'config.json'
    if dst.exists():
        raise SystemExit(f'{dst} already exists -- refusing to overwrite a real summary')
    for f in ('best_model.pt', 'val_probs.npz'):
        if not (out / f).exists():
            raise SystemExit(f'{out / f} missing -- the run did not get far enough to recover')

    template = FRAME / tag.removesuffix('_popw') / 'config.json'
    cfg = json.load(open(template))

    d = np.load(out / 'val_probs.npz', allow_pickle=True)
    probs, labs, obs = d['probs'], d['labels'], d['obs'].astype(str)
    apr = ap_report(probs, labs, obs, tolerances=(0, 1, 2))
    rr = rate_report(probs, labs, obs)
    auc = {n: float(roc_auc_score(labs[:, i], probs[:, i])) for i, n in enumerate(('nt', 'nn'))}

    cfg.update({
        'derm_prevalence': 'population',
        'derm_floor': 1e-4,
        'ap_report': apr, 'rate_report': rr, 'roc_auc': auc,
        'best_ap': apr['macro/tol0']['ap'],
        'history': [],
        'reconstructed': ('summary crashed on the stale derm_env_prior key after training '
                          'completed; args copied from the _last sibling + the popw overrides, '
                          'metrics recomputed from val_probs.npz. History is in wandb only.'),
    })
    for k in ('jpeg_cache_gib', 'jpeg_cache_frames'):
        cfg.pop(k, None)
    json.dump(cfg, open(dst, 'w'), indent=2)
    print(f'wrote {dst}  (best_ap {cfg["best_ap"]:.4f}, auc nt {auc["nt"]:.4f} nn {auc["nn"]:.4f})')


if __name__ == '__main__':
    main(sys.argv[1])
