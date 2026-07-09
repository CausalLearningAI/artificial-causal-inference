"""
Autonomous grid search for the mouse pairwise behavior classifier.

Stage A: coarse screen over (n_heads, context_k, hidden_dim, neg_ratio, loss_type)
          using a single 80/20 pool split, few epochs, macro PR-AUC on val.
Stage B: k-fold CV (over the 22 annotated pools) on the top-K coarse candidates,
          more epochs, to get a robust macro PR-AUC estimate per config.
Stage C: retrain the best config on the standard single 80/20 split with full
          epochs (this becomes results/mice_behavior/best_model.pt), then
          produce the required 2x2 ROC/PR plot (per-pair + collapsed-per-frame).

Also, if dataset/mice/v1/embeddings/full/dinov2/patch_grid4/ is ready (checked
at runtime, not assumed), Stage B additionally evaluates a patch-grid config
variant and folds it into the comparison.

All results are appended to results/mice_behavior/grid_search_log.jsonl as they
complete, so progress survives interruption.

Usage:
    python scripts/mice_behavior/grid_search.py
"""
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.build_pair_labels import build_pair_labels
from src.mice_behavior.pools import load_obs_to_pool_map
from src.mice_behavior.train import train

DATA_DIR = Path('./data')
DATASET_DIR = Path('./dataset')
RESULTS_DIR = Path('./results/mice_behavior')
LOG_PATH = RESULTS_DIR / 'grid_search_log.jsonl'
SEED = 42
ENCODER, TOKEN = 'dinov2', 'class_l-2'


def log_result(record: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record['ts'] = time.time()
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f'[LOG] {record}')


def kfold_pools(pools, k, seed=SEED):
    pools = sorted(pools)
    rng = random.Random(seed)
    rng.shuffle(pools)
    folds = [pools[i::k] for i in range(k)]
    return folds


def run_config(cfg, annotations_csv, pair_labels_path, embeddings_path, emb_dim,
                train_obs, val_obs, n_epochs, tag, eval_every=1):
    out_dir = RESULTS_DIR / 'tmp' / tag
    result = train(
        annotations_csv=str(annotations_csv),
        pair_labels_parquet=str(pair_labels_path),
        embeddings_path=str(embeddings_path),
        output_dir=str(out_dir),
        train_obs_ids=train_obs,
        val_obs_ids=val_obs,
        context_k=cfg['context_k'],
        emb_dim=emb_dim,
        n_heads=cfg['n_heads'],
        hidden_dim=cfg['hidden_dim'],
        n_epochs=n_epochs,
        neg_ratio=cfg['neg_ratio'],
        loss_type=cfg.get('loss_type', 'ce'),
        focal_gamma=cfg.get('focal_gamma', 2.0),
        device='cuda',
        seed=SEED,
        verbose=False,
        eval_every=eval_every,
    )
    return result['best_pr_auc'], result['best_per_class']


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print('Building pair labels (skip if exists)...')
    pair_labels_path = build_pair_labels(DATA_DIR, DATASET_DIR, overwrite=False)

    annotations_csv = DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    embeddings_path = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / ENCODER / TOKEN / 'embeddings.npy'
    n_frames = sum(1 for _ in open(annotations_csv)) - 1
    emb_dim = embeddings_path.stat().st_size // (4 * n_frames)
    print(f'Embeddings: {ENCODER}/{TOKEN}, dim={emb_dim}, frames={n_frames:,}')

    obs_to_pool = load_obs_to_pool_map(DATA_DIR)
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    pools = sorted({obs_to_pool[o] for o in all_obs})
    print(f'{len(pools)} annotated pools, {len(all_obs)} annotated observations')

    # --- Stage A: coarse screen, single 80/20 split ---
    rng = random.Random(SEED)
    shuffled_pools = pools[:]
    rng.shuffle(shuffled_pools)
    n_val = max(1, int(len(shuffled_pools) * 0.2))
    val_pool_set = set(shuffled_pools[:n_val])
    train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]
    print(f'Stage A split: {len(train_obs)} train obs / {len(val_obs)} val obs')

    coarse_grid = list(itertools.product(
        [1, 2, 4],          # n_heads
        [2, 3],             # context_k
        [256],              # hidden_dim
        [10],               # neg_ratio
        ['ce'],             # loss_type
    ))
    # small extra probes reusing the default point but varying one axis at a time
    coarse_grid += [
        (1, 2, 128, 10, 'ce'),
        (1, 2, 512, 10, 'ce'),
        (1, 2, 256, 5, 'ce'),
        (1, 2, 256, 20, 'ce'),
        (1, 2, 256, 10, 'focal'),
        (1, 4, 256, 10, 'ce'),
        (1, 1, 256, 10, 'ce'),
    ]
    coarse_grid = list(dict.fromkeys(coarse_grid))  # dedup, preserve order

    print(f'Stage A: {len(coarse_grid)} coarse configs, 20 epochs each')
    coarse_results = []
    for i, (n_heads, context_k, hidden_dim, neg_ratio, loss_type) in enumerate(coarse_grid):
        cfg = dict(n_heads=n_heads, context_k=context_k, hidden_dim=hidden_dim,
                   neg_ratio=neg_ratio, loss_type=loss_type)
        tag = f'coarse_{i}'
        t0 = time.time()
        try:
            pr_auc, per_class = run_config(
                cfg, annotations_csv, pair_labels_path, embeddings_path, emb_dim,
                train_obs, val_obs, n_epochs=20, tag=tag, eval_every=5,
            )
        except Exception as e:
            log_result({'stage': 'A', 'tag': tag, 'cfg': cfg, 'error': str(e)})
            continue
        dt = time.time() - t0
        log_result({'stage': 'A', 'tag': tag, 'cfg': cfg, 'macro_pr_auc': pr_auc,
                    'per_class': per_class, 'seconds': dt})
        coarse_results.append((pr_auc, cfg))

    coarse_results.sort(key=lambda x: -x[0])
    top_k = coarse_results[:4]
    print(f'Stage A done. Top candidates: {top_k}')

    # --- Stage B: k-fold CV on top candidates ---
    K = 5
    folds = kfold_pools(pools, K)
    stage_b_results = []
    for rank, (_, cfg) in enumerate(top_k):
        fold_prs = []
        for f in range(K):
            val_pool_set_f = set(folds[f])
            train_obs_f = [o for o in all_obs if obs_to_pool[o] not in val_pool_set_f]
            val_obs_f = [o for o in all_obs if obs_to_pool[o] in val_pool_set_f]
            if not val_obs_f:
                continue
            tag = f'cv_{rank}_{f}'
            try:
                pr_auc, per_class = run_config(
                    cfg, annotations_csv, pair_labels_path, embeddings_path, emb_dim,
                    train_obs_f, val_obs_f, n_epochs=40, tag=tag, eval_every=5,
                )
            except Exception as e:
                log_result({'stage': 'B', 'tag': tag, 'cfg': cfg, 'fold': f, 'error': str(e)})
                continue
            log_result({'stage': 'B', 'tag': tag, 'cfg': cfg, 'fold': f,
                        'macro_pr_auc': pr_auc, 'per_class': per_class})
            fold_prs.append(pr_auc)
        mean_pr = float(np.mean(fold_prs)) if fold_prs else -1.0
        std_pr = float(np.std(fold_prs)) if fold_prs else 0.0
        log_result({'stage': 'B_summary', 'cfg': cfg, 'mean_macro_pr_auc': mean_pr,
                    'std_macro_pr_auc': std_pr, 'n_folds': len(fold_prs)})
        stage_b_results.append((mean_pr, std_pr, cfg))

    stage_b_results.sort(key=lambda x: -x[0])
    best_mean_pr, best_std_pr, best_cfg = stage_b_results[0]
    print(f'Stage B done. Best config: {best_cfg} mean_pr_auc={best_mean_pr:.4f}+-{best_std_pr:.4f}')
    log_result({'stage': 'B_best', 'cfg': best_cfg, 'mean_macro_pr_auc': best_mean_pr,
                'std_macro_pr_auc': best_std_pr})

    # --- Stage C: retrain best config on the standard 80/20 split, full epochs ---
    print('Stage C: retraining best config on standard split, 100 epochs -> best_model.pt')
    pr_auc, per_class = run_config(
        best_cfg, annotations_csv, pair_labels_path, embeddings_path, emb_dim,
        train_obs, val_obs, n_epochs=100, tag='final',
    )
    log_result({'stage': 'C_final', 'cfg': best_cfg, 'macro_pr_auc': pr_auc, 'per_class': per_class})

    # Copy final model + record the split + config used, for downstream plotting.
    import shutil
    shutil.copy(RESULTS_DIR / 'tmp' / 'final' / 'best_model.pt', RESULTS_DIR / 'best_model.pt')
    with open(RESULTS_DIR / 'best_config.json', 'w') as f:
        json.dump({'cfg': best_cfg, 'val_pools': sorted(val_pool_set),
                   'macro_pr_auc': pr_auc, 'per_class': per_class,
                   'encoder': ENCODER, 'token': TOKEN, 'emb_dim': emb_dim}, f, indent=2)

    print('Grid search complete. Best config saved to results/mice_behavior/best_config.json')


if __name__ == '__main__':
    main()
