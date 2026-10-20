"""
Analyse codes written by the ORIGINAL repo's s4-sae_encoding.py, following notebooks/istant/istant-experiment.ipynb
(per-video mean of every manifest column -> agg_df, treatment -> T; their neural_effect_search with method
Bonferroni / FDR / t-test and their neural_effect_search_base test 'AD'), then the pre-registered alignment
analysis of align.py (identical to the one on our re-implementation). --orig-root = the stage copy of the
original repo (its src/ is imported, unmodified apart from output paths in s3).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig-root', required=True)
    ap.add_argument('--manifest-dir', required=True)
    ap.add_argument('--codes-dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--save-codes', default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.orig_root)
    from src.neuralscience.features_store import FeatureStore, FeatureSpecAlternative
    from src.neuralscience.nes import neural_effect_search, neural_effect_search_base
    sys.path.insert(0, str(HERE))
    from align import analyse

    md = Path(a.manifest_dir)
    labels = pd.read_parquet(md / 'labels.parquet')
    store = FeatureStore(FeatureSpecAlternative(str(md), a.codes_dir))
    codes = store.get_batch(np.arange(len(labels))).astype(np.float32)
    assert codes.shape == (52800, 4608), codes.shape
    if a.save_codes:
        np.save(a.save_codes, codes)

    # --- notebook path --------------------------------------------------------------------------------
    feat = pd.DataFrame(codes, columns=[f'N_{j}' for j in range(1, 4609)])
    df = pd.concat([labels.drop(columns=['path']).reset_index(drop=True), feat], axis=1)
    df.insert(0, 'main_index', np.arange(len(df)))
    df.insert(1, 'iloc_index', np.arange(len(df)))
    agg = df.groupby('video_id').mean(numeric_only=True).sort_index().rename(columns={'treatment': 'T'})
    agg = agg[['main_index', 'iloc_index', 'T'] + [c for c in agg.columns if c not in ('main_index', 'iloc_index', 'T')]]
    nb = {}
    for m in ['Bonferroni', 'FDR', 't-test']:
        nb[m + 'NES'] = neural_effect_search(agg, method=m, alpha=0.05, k=2000, neuron_prefix='N_', fallback_start_neur_index=4)
        nb[m + 'AD'] = neural_effect_search_base(agg, test='AD', method=m, alpha=0.05, k=2000, neuron_prefix='N_',
                                                 fallback_start_neur_index=4, sn=[])
    print({k: v['neurons'] for k, v in nb.items()}, flush=True)

    # --- pre-registered alignment analysis -------------------------------------------------------------
    exp = pd.read_csv(REPO / 'data/ants/v2/experiment.csv')
    videos = exp[exp.valid == 1].reset_index(drop=True)
    assert (labels.video_id.to_numpy() == np.repeat(exp.index[exp.valid == 1], 1200)).all()
    y = labels.grooming_2focal.to_numpy().astype(np.int8)
    pos1 = ((labels.exp_pos_1 == 1) & (labels.exp_batch_4 == 0)).to_numpy().astype(np.int8)
    res, per = analyse(codes, videos, y, pos1, neural_effect_search, np.random.default_rng(0))
    res['notebook_nes'] = nb
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'orig_{a.tag}.npz', **per)
    (out / f'orig_{a.tag}.json').write_text(json.dumps(res, indent=1, default=float))
    print(f"{a.tag}: alive {res['n_alive']} single-pass hits {res['mean']['n_hits']} recursive {res['mean']['recursive_hits']} "
          f"best AUROC {res['best_auroc']:.3f}", flush=True)


if __name__ == '__main__':
    main()
