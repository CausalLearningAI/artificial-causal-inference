"""
Autonomous overnight hyperparameter search for the mouse pairwise behavior
classifier, run entirely in-process (one long SLURM allocation, not many
short jobs) via train_fast().

Optimizes pair_macro_pr_auc (nt/nn only, 'none' excluded — see report.py),
train_fast()'s early-stopping / model-selection metric, the same corrected
definition now used everywhere else in the pipeline.

Stage A: random search over CLS hyperparameters (cheap, ~1-2 min/trial with
    early stopping), single 80/20 pool split (same split as train_cls_final.py),
    time-boxed.
Stage B: k-fold CV over pools on the top Stage-A candidates, for a variance-
    aware ranking (a single 80/20 split is noisy — 4 pools out of ~22).
Stage C: the best CLS config (by CV mean) is evaluated on the patch-grid
    architecture too, since patch-grid's extra PatchAttnPool stage may
    prefer different hyperparameters (fewer folds — patch-grid is ~20x
    slower per epoch than CLS).
Stage D: final retrain of the winning config for each variant on the
    standard 80/20 split. Only PROMOTED to results/mice_behavior/{cls,
    patchgrid}/ (overwriting the current baseline) if it beats the existing
    config.json's best_pr_auc on the same metric — a worse search result
    never silently clobbers a better already-committed one.

Every trial is logged to results/mice_behavior/search/log.jsonl as it
completes (survives interruption/resume). A human-readable
results/mice_behavior/search/SUMMARY.md is written at the end.

Usage:
    python scripts/mice_behavior/grid_search.py
"""
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.build_pair_labels import build_pair_labels
from src.mice_behavior.fast_data import FastBatchData, load_cls_embeddings, load_patchgrid_embeddings
from src.mice_behavior.model import MouseBehaviorClassifier
from src.mice_behavior.pools import load_obs_to_pool_map
from src.mice_behavior.report import collect_val_predictions_fast, generate_report
from src.mice_behavior.train import train_fast

DATA_DIR = Path('./data')
DATASET_DIR = Path('./dataset')
RESULTS_DIR = Path('./results/mice_behavior')
SEARCH_DIR = RESULTS_DIR / 'search'
LOG_PATH = SEARCH_DIR / 'log.jsonl'
TMP_DIR = SEARCH_DIR / 'tmp'
SEED = 42
ENCODER, TOKEN = 'dinov2', 'class_l-2'
PATCH_GRID_DIR = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / 'dinov2' / 'patch_grid4'

# Wall-clock budgets, not trial counts — actual per-trial cost varies with config
# (context_k, hidden_dim) so a time budget adapts better than a fixed trial count.
STAGE_A_BUDGET_SEC = 4 * 3600
STAGE_B_BUDGET_SEC = 2.5 * 3600
STAGE_C_BUDGET_SEC = 1.5 * 3600
STAGE_A_TOPK = 6
STAGE_B_K = 4
STAGE_C_K = 2
STAGE_A_EPOCHS = 80
STAGE_B_EPOCHS = 80
STAGE_C_EPOCHS = 80
STAGE_D_EPOCHS = 100


def log_result(record: dict):
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    record['ts'] = time.time()
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f'[LOG] {record}', flush=True)


def kfold_pools(pools, k, seed=SEED):
    pools = sorted(pools)
    rng = random.Random(seed)
    rng.shuffle(pools)
    return [pools[i::k] for i in range(k)]


def sample_cfg(rng: random.Random) -> dict:
    loss_type = rng.choice(['ce', 'focal'])
    return dict(
        n_heads=rng.choice([1, 2, 4, 8]),
        context_k=rng.choice([1, 2, 3, 4, 6]),
        hidden_dim=rng.choice([128, 256, 384, 512]),
        neg_ratio=rng.choice([5, 10, 15, 20]),
        loss_type=loss_type,
        focal_gamma=rng.choice([1.0, 2.0, 3.0]) if loss_type == 'focal' else 2.0,
        dropout=rng.choice([0.0, 0.05, 0.1, 0.2, 0.3]),
        weight_decay=rng.choice([0.0, 1e-5, 1e-4, 1e-3]),
        lr=rng.choice([3e-4, 1e-3, 3e-3]),
    )


def run_trial(cfg, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim,
              train_obs, val_obs, n_epochs, tag, use_patch_grid=False):
    out_dir = TMP_DIR / tag
    # Patch-grid's extra PatchAttnPool stage was found (this session, manually) to collapse
    # into a dead uniform-softmax state at lr>=1e-3 even with grad clipping — clamp lr for
    # patch-grid trials specifically rather than let the search rediscover this by wasting
    # trials on collapsed runs.
    lr = min(cfg['lr'], 3e-4) if use_patch_grid else cfg['lr']
    kwargs = dict(
        annotations_csv=str(annotations_csv), pair_labels_parquet=str(pair_labels_path),
        embeddings_path=str(cls_embeddings_path), output_dir=str(out_dir),
        train_obs_ids=train_obs, val_obs_ids=val_obs, context_k=cfg['context_k'], emb_dim=emb_dim,
        n_heads=cfg['n_heads'], hidden_dim=cfg['hidden_dim'], n_epochs=n_epochs,
        neg_ratio=cfg['neg_ratio'], loss_type=cfg['loss_type'], focal_gamma=cfg['focal_gamma'],
        lr=lr, dropout=cfg['dropout'], weight_decay=cfg['weight_decay'],
        device='cuda', seed=SEED, verbose=False, use_patch_grid=use_patch_grid, eval_every=1,
    )
    if use_patch_grid:
        kwargs.update(
            patch_embeddings_path=str(PATCH_GRID_DIR / 'embeddings.npy'),
            patch_global_idx_path=str(PATCH_GRID_DIR / 'global_idx.npy'),
            n_patches=16,
            # 2080ti has 11GB VRAM; patch-grid's per-batch context (16x more tokens than CLS)
            # OOM'd during backward() at the default batch_size=4096 (found earlier this session).
            batch_size=1024,
        )
    result = train_fast(**kwargs)
    del result['model']
    torch.cuda.empty_cache()
    return result['best_pr_auc'], result['best_per_class']


def main():
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    pair_labels_path = build_pair_labels(DATA_DIR, DATASET_DIR, overwrite=False)
    annotations_csv = DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    cls_embeddings_path = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / ENCODER / TOKEN / 'embeddings.npy'
    n_frames = sum(1 for _ in open(annotations_csv)) - 1
    emb_dim = cls_embeddings_path.stat().st_size // (4 * n_frames)

    obs_to_pool = load_obs_to_pool_map(DATA_DIR)
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    pools = sorted({obs_to_pool[o] for o in all_obs})
    print(f'{len(pools)} pools, {len(all_obs)} annotated observations, emb_dim={emb_dim}', flush=True)

    # Standard split — same seed/logic as train_cls_final.py / train_patchgrid_final.py.
    rng_split = random.Random(SEED)
    shuffled = pools[:]
    rng_split.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.2))
    val_pool_set = set(shuffled[:n_val])
    train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]

    # ---------------- Stage A: random search on CLS, single split ----------------
    print(f'=== Stage A: random search (budget {STAGE_A_BUDGET_SEC/3600:.1f}h) ===', flush=True)
    sample_rng = random.Random(SEED + 1)
    stage_a_results = []
    trial_i = 0
    t_stage = time.time()
    while time.time() - t_stage < STAGE_A_BUDGET_SEC:
        cfg = sample_cfg(sample_rng)
        tag = f'A_{trial_i}'
        t0 = time.time()
        try:
            pr_auc, per_class = run_trial(
                cfg, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim,
                train_obs, val_obs, STAGE_A_EPOCHS, tag,
            )
        except Exception as e:
            log_result({'stage': 'A', 'tag': tag, 'cfg': cfg, 'error': str(e), 'traceback': traceback.format_exc()})
            trial_i += 1
            continue
        dt = time.time() - t0
        log_result({'stage': 'A', 'tag': tag, 'cfg': cfg, 'pair_macro_pr_auc': pr_auc,
                    'per_class': per_class, 'seconds': dt})
        stage_a_results.append((pr_auc, cfg))
        trial_i += 1

    stage_a_results.sort(key=lambda x: -x[0])
    print(f'Stage A done: {trial_i} trials in {(time.time()-t_stage)/60:.1f} min', flush=True)
    if not stage_a_results:
        print('Stage A found nothing usable — aborting search.', flush=True)
        return
    top_candidates = stage_a_results[:STAGE_A_TOPK]
    log_result({'stage': 'A_summary', 'n_trials': trial_i, 'top_candidates': top_candidates})

    # ---------------- Stage B: k-fold CV on top CLS candidates ----------------
    print(f'=== Stage B: {STAGE_B_K}-fold CV on top {len(top_candidates)} CLS candidates ===', flush=True)
    folds = kfold_pools(pools, STAGE_B_K)
    t_stage = time.time()
    stage_b_results = []
    for rank, (_, cfg) in enumerate(top_candidates):
        if time.time() - t_stage > STAGE_B_BUDGET_SEC:
            log_result({'stage': 'B', 'note': 'budget exhausted, skipping remaining candidates', 'rank': rank})
            break
        fold_prs = []
        for f in range(STAGE_B_K):
            val_pool_set_f = set(folds[f])
            train_obs_f = [o for o in all_obs if obs_to_pool[o] not in val_pool_set_f]
            val_obs_f = [o for o in all_obs if obs_to_pool[o] in val_pool_set_f]
            if not val_obs_f:
                continue
            tag = f'B_{rank}_{f}'
            try:
                pr_auc, per_class = run_trial(
                    cfg, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim,
                    train_obs_f, val_obs_f, STAGE_B_EPOCHS, tag,
                )
            except Exception as e:
                log_result({'stage': 'B', 'tag': tag, 'cfg': cfg, 'fold': f, 'error': str(e)})
                continue
            log_result({'stage': 'B', 'tag': tag, 'cfg': cfg, 'fold': f, 'pair_macro_pr_auc': pr_auc})
            fold_prs.append(pr_auc)
        mean_pr = float(np.mean(fold_prs)) if fold_prs else -1.0
        std_pr = float(np.std(fold_prs)) if fold_prs else 0.0
        log_result({'stage': 'B_summary', 'cfg': cfg, 'mean_pair_macro_pr_auc': mean_pr,
                    'std_pair_macro_pr_auc': std_pr, 'n_folds': len(fold_prs)})
        stage_b_results.append((mean_pr, std_pr, cfg))

    stage_b_results.sort(key=lambda x: -x[0])
    if not stage_b_results:
        print('Stage B produced nothing — falling back to Stage A best.', flush=True)
        best_cls_cfg = top_candidates[0][1]
    else:
        best_mean, best_std, best_cls_cfg = stage_b_results[0]
        print(f'Stage B done: best CLS cfg mean_pr_auc={best_mean:.4f}+-{best_std:.4f}  {best_cls_cfg}', flush=True)
        log_result({'stage': 'B_best', 'cfg': best_cls_cfg, 'mean_pair_macro_pr_auc': best_mean, 'std': best_std})

    # ---------------- Stage C: does it transfer to patch-grid? ----------------
    print(f'=== Stage C: {STAGE_C_K}-fold CV, top CLS configs on patch-grid ===', flush=True)
    pg_candidates = [c for _, c in top_candidates[:3]]
    if best_cls_cfg not in pg_candidates:
        pg_candidates = [best_cls_cfg] + pg_candidates
    folds_c = kfold_pools(pools, STAGE_C_K, seed=SEED + 2)
    t_stage = time.time()
    stage_c_results = []
    for rank, cfg in enumerate(pg_candidates):
        if time.time() - t_stage > STAGE_C_BUDGET_SEC:
            log_result({'stage': 'C', 'note': 'budget exhausted, skipping remaining candidates', 'rank': rank})
            break
        fold_prs = []
        for f in range(STAGE_C_K):
            val_pool_set_f = set(folds_c[f])
            train_obs_f = [o for o in all_obs if obs_to_pool[o] not in val_pool_set_f]
            val_obs_f = [o for o in all_obs if obs_to_pool[o] in val_pool_set_f]
            if not val_obs_f:
                continue
            tag = f'C_{rank}_{f}'
            try:
                pr_auc, per_class = run_trial(
                    cfg, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim,
                    train_obs_f, val_obs_f, STAGE_C_EPOCHS, tag, use_patch_grid=True,
                )
            except Exception as e:
                log_result({'stage': 'C', 'tag': tag, 'cfg': cfg, 'fold': f, 'error': str(e)})
                continue
            log_result({'stage': 'C', 'tag': tag, 'cfg': cfg, 'fold': f, 'pair_macro_pr_auc': pr_auc})
            fold_prs.append(pr_auc)
        mean_pr = float(np.mean(fold_prs)) if fold_prs else -1.0
        log_result({'stage': 'C_summary', 'cfg': cfg, 'mean_pair_macro_pr_auc': mean_pr, 'n_folds': len(fold_prs)})
        stage_c_results.append((mean_pr, cfg))

    stage_c_results.sort(key=lambda x: -x[0])
    best_pg_cfg = stage_c_results[0][1] if stage_c_results else best_cls_cfg
    if stage_c_results:
        print(f'Stage C done: best patch-grid cfg mean_pr_auc={stage_c_results[0][0]:.4f}  {best_pg_cfg}', flush=True)
        log_result({'stage': 'C_best', 'cfg': best_pg_cfg, 'mean_pair_macro_pr_auc': stage_c_results[0][0]})

    # ---------------- Stage D: final retrain, promote only if it wins ----------------
    print('=== Stage D: final retrain on standard split, promote if better than baseline ===', flush=True)
    summary_lines = [
        '# Overnight hyperparameter search summary\n',
        f'Started: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start))}\n',
        f'Stage A: {trial_i} random-search trials on CLS ({STAGE_A_BUDGET_SEC/3600:.1f}h budget)\n',
        f'Stage B: {STAGE_B_K}-fold CV on top {len(top_candidates)} CLS candidates\n',
        f'Stage C: {STAGE_C_K}-fold CV, transferring top CLS candidates to patch-grid\n\n',
    ]

    for variant, use_patch_grid, cfg, name in [
        ('cls', False, best_cls_cfg, 'CLS-token (baseline) mouse behavior classifier'),
        ('patchgrid', True, best_pg_cfg, 'Patch-grid (attention-pooled) mouse behavior classifier'),
    ]:
        out_dir = RESULTS_DIR / variant
        baseline_pr_auc = -1.0
        if (out_dir / 'config.json').exists():
            baseline_pr_auc = json.load(open(out_dir / 'config.json')).get('best_pr_auc', -1.0)

        tag = f'D_{variant}'
        lr_used = min(cfg['lr'], 3e-4) if use_patch_grid else cfg['lr']
        try:
            kwargs = dict(
                annotations_csv=str(annotations_csv), pair_labels_parquet=str(pair_labels_path),
                embeddings_path=str(cls_embeddings_path), output_dir=str(TMP_DIR / tag),
                train_obs_ids=train_obs, val_obs_ids=val_obs, context_k=cfg['context_k'], emb_dim=emb_dim,
                n_heads=cfg['n_heads'], hidden_dim=cfg['hidden_dim'], n_epochs=STAGE_D_EPOCHS,
                neg_ratio=cfg['neg_ratio'], loss_type=cfg['loss_type'], focal_gamma=cfg['focal_gamma'],
                lr=lr_used, dropout=cfg['dropout'], weight_decay=cfg['weight_decay'],
                device='cuda', seed=SEED, verbose=True, use_patch_grid=use_patch_grid, eval_every=1,
            )
            if use_patch_grid:
                kwargs.update(
                    patch_embeddings_path=str(PATCH_GRID_DIR / 'embeddings.npy'),
                    patch_global_idx_path=str(PATCH_GRID_DIR / 'global_idx.npy'),
                    n_patches=16,
                    batch_size=1024,
                )
            result = train_fast(**kwargs)
        except Exception as e:
            log_result({'stage': 'D', 'variant': variant, 'cfg': cfg, 'error': str(e), 'traceback': traceback.format_exc()})
            summary_lines.append(f'## {variant}\nFinal retrain FAILED: {e}\n\n')
            continue

        new_pr_auc = result['best_pr_auc']
        log_result({'stage': 'D', 'variant': variant, 'cfg': cfg, 'pair_macro_pr_auc': new_pr_auc,
                    'baseline_pr_auc': baseline_pr_auc})

        cfg_final = {**cfg, 'lr': lr_used}
        summary_lines.append(f'## {variant}\n')
        summary_lines.append(f'- Baseline (current results/mice_behavior/{variant}/config.json): {baseline_pr_auc:.4f}\n')
        summary_lines.append(f'- Search result: {new_pr_auc:.4f}  cfg={cfg_final}\n')

        if new_pr_auc > baseline_pr_auc:
            print(f'  {variant}: NEW BEST {new_pr_auc:.4f} > baseline {baseline_pr_auc:.4f} — promoting', flush=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(result['model'].state_dict(), out_dir / 'best_model.pt')
            with open(out_dir / 'history.json', 'w') as f:
                json.dump(result['history'], f, indent=2)
            with open(out_dir / 'config.json', 'w') as f:
                json.dump({'cfg': cfg_final, 'val_pools': sorted(val_pool_set), 'n_epochs': STAGE_D_EPOCHS,
                           'best_pr_auc': new_pr_auc, 'best_per_class': result['best_per_class'],
                           'emb_dim': emb_dim, 'promoted_by_search': True}, f, indent=2)

            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = MouseBehaviorClassifier(
                emb_dim=emb_dim, n_heads=cfg['n_heads'], hidden_dim=cfg['hidden_dim'],
                use_patch_grid=use_patch_grid, dropout=cfg['dropout'],
            ).to(dev)
            model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=dev, weights_only=True))
            model.eval()
            load_fn = (
                load_patchgrid_embeddings(str(PATCH_GRID_DIR / 'embeddings.npy'), str(PATCH_GRID_DIR / 'global_idx.npy'), 16, emb_dim)
                if use_patch_grid else load_cls_embeddings(str(cls_embeddings_path), emb_dim)
            )
            val_data = FastBatchData(
                str(annotations_csv), str(pair_labels_path), val_obs, cfg['context_k'], emb_dim, load_fn,
                n_patches=16 if use_patch_grid else None,
            )
            probs, labels = collect_val_predictions_fast(model, val_data, dev)
            generate_report(probs, labels, result['history'], name, cfg_final, out_dir)
            summary_lines.append(f'- **PROMOTED** to results/mice_behavior/{variant}/ (report.png regenerated)\n\n')
        else:
            print(f'  {variant}: search result {new_pr_auc:.4f} did not beat baseline {baseline_pr_auc:.4f} — keeping baseline', flush=True)
            summary_lines.append(f'- NOT promoted (baseline was already better)\n\n')

    with open(SEARCH_DIR / 'SUMMARY.md', 'w') as f:
        f.writelines(summary_lines)
    print(f'Search complete in {(time.time()-t_start)/3600:.2f}h. See {SEARCH_DIR / "SUMMARY.md"}', flush=True)


if __name__ == '__main__':
    main()
