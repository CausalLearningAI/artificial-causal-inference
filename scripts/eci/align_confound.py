"""
Camera-period confound check for the mice genotype (family B, het vs wt within a stage) NES selections (ECI).

The on-screen odor corner (dataset/mice/v1/eci/odor_corner.csv) marks the recording period, and the camera zoom
differs by period (arena side ~327 px in the BL period, ~357 BR, ~340 TR). The corner is borderline linked to
genotype at pool level, so an image feature that tracks the period could pass for a genotype effect. For each
primary family-B selection of each SAE (codes_max, per-video mean, t, Bonferroni, full window, prefixes 128 / 1024;
mean-activation and bout runs), the outcome is the per-video mean of codes_max of that neuron over the stage's videos
(the mean-activation outcome; for bout selections it is a proxy of the bout rate), and an OLS fit gives the genotype
p value:
    plain       y ~ het
    + arena     y ~ het + arena side (px, mean of the arena box height and width)
    + corner    y ~ het + corner group dummies (TR / BL / BR)
Also the het / wt counts per corner group of each stage.

Output: <out-dir>/confound.json

Usage:
    python scripts/eci/align_confound.py --sae matryoshka_btk_1024_k16_fg448_s0 --sae matryoshka_btk_1024_k16_fg448al_s0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from src.eci.domain import get_domain  # noqa: E402
from align_arena import selections  # noqa: E402


def ols_p(y, X):
    """p value of the first column of X (with intercept) in OLS y ~ 1 + X."""
    import statsmodels.api as sm
    return float(sm.OLS(y, sm.add_constant(X, has_constant='add')).fit().pvalues[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sae', action='append', required=True)
    p.add_argument('--out-dir', default=str(REPO / 'results/vision/eci_align/mice'))
    args = p.parse_args()
    D = get_domain('mice')
    design = D.load_design()
    oc = pd.read_csv(D.eci_dir / 'odor_corner.csv').set_index('observation_id')
    design['corner'] = oc.loc[design['observation_id'], 'odor_corner'].values
    design['arena_side'] = ((oc['arena_r1'] - oc['arena_r0'] + oc['arena_c1'] - oc['arena_c0']) / 2).loc[
        design['observation_id']].values
    res = {'counts': {}, 'arena_side_by_corner': design.groupby('corner')['arena_side'].describe()[['mean', 'min', 'max']]
           .round(1).to_dict('index')}
    for st, g in design.groupby('stage'):
        res['counts'][f'stage{st}'] = pd.crosstab(g['corner'], g['genotype']).to_dict('index')
    for sae in args.sae:
        X = np.load(D.eci_dir / 'codes' / sae / 'codes_max.npy', mmap_mode='r')
        sel = [s for s in selections(D.nes_root / sae) if s['analysis_id'].startswith('B_')]
        out = []
        for s in sel:
            st = int(s['analysis_id'].split('stage')[1])
            g = design[design['stage'] == st]
            y = np.array([float(np.asarray(X[a:b, s['neuron']], dtype=np.float32).mean())
                          for a, b in zip(g['row_start'], g['row_end'])])
            het = (g['genotype'].values == 'het').astype(float)
            cd = pd.get_dummies(g['corner']).astype(float).values[:, 1:]
            out.append({**s, 'p_plain': ols_p(y, het[:, None]),
                        'p_arena': ols_p(y, np.c_[het, g['arena_side'].values]),
                        'p_corner': ols_p(y, np.c_[het, cd]) if cd.shape[1] else None,
                        'mean_by_corner': pd.Series(y).groupby(g['corner'].values).mean().round(4).to_dict()})
        res[sae] = out
        print(sae)
        for o in out:
            print(f"  {o['kind']:16s} {o['analysis_id']:9s} p{o['prefix']:<5d} n{o['neuron']:<5d} NES p {o['p']:.2e}  OLS p "
                  f"plain {o['p_plain']:.2e}  +arena {o['p_arena']:.2e}  +corner {o['p_corner']:.2e}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'confound.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('counts', 'arena_side_by_corner')}, indent=1))


if __name__ == '__main__':
    main()
