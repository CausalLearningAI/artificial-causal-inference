"""
Autonomous hyperparameter search for the mouse pairwise behavior classifier,
run entirely in-process (one long SLURM allocation) via train().

Every candidate is ranked and promoted using the SAME full, non-subsampled
validation-set pair-level macro PR-AUC (nt/nn only, 'none' excluded) —
computed once, right after each trial's training finishes, via
collect_val_predictions_fast(). This is deliberately NOT the fast internal
per-epoch metric train() tracks for early stopping (that one runs on a
small rebalanced subsample for speed and is NOT comparable across runs with
different subsample sizes — a previous version of this script compared the
two inconsistently and reported a false improvement).

Runs a direct random search per variant (CLS, patch-grid) on the standard
80/20 pool split (same split as train_{cls,patchgrid}_final.py) — no k-fold
CV (dropped: not worth the added complexity/cost for this problem size).
Search space includes a `stride` parameter: context position i is at true
frame offset i*stride, so context_k=2/stride=3 covers +-6 frames at the same
attention cost as context_k=2/stride=1 (+-2 frames) — a wide dense window
(e.g. context_k=6) was the direct cause of a CUDA OOM in an earlier version
of this search (patch-grid's per-frame tensor is 16x CLS's; a 13-frame dense
window exceeded available GPU memory). context_k*stride is capped at 8 (the
model's max_offset) so the position-embedding table stays well-defined.

Only PROMOTED to results/vision/mice/opair/{cls,patchgrid}/ if the winning
config's full-val score beats a freshly-recomputed full-val score for the
CURRENT best_model.pt (not whatever number happens to be sitting in
config.json, which may have been computed differently) — a worse search
result never silently clobbers a better already-committed one.

Every trial is logged to results/vision/mice/opair/search/log.jsonl as it
completes. A human-readable results/vision/mice/opair/search/SUMMARY.md is
written at the end.

Usage:
    python scripts/mice_behavior/grid_search.py
"""
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.build_pair_labels import build_pair_labels
from src.mice_behavior.fast_data import FastBatchData, load_cls_embeddings, load_patchgrid_embeddings
from src.mice_behavior.model import MouseBehaviorClassifier
from src.mice_behavior.pools import load_obs_to_pool_map
from src.mice_behavior.report import collect_val_predictions_fast, generate_report
from src.mice_behavior.train import train

DATA_DIR = Path('./data')
DATASET_DIR = Path('./dataset')
RESULTS_DIR = Path('./results/vision/mice/opair')
SEARCH_DIR = RESULTS_DIR / 'search'
LOG_PATH = SEARCH_DIR / 'log.jsonl'
TMP_DIR = SEARCH_DIR / 'tmp'
SEED = 42
ENCODER, TOKEN = 'dinov2', 'class_l-2'
PATCH_GRID_DIR = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / 'dinov2' / 'patch_grid4'

# Wall-clock budgets, not trial counts — actual per-trial cost varies with config
# (context_k*stride reach, hidden_dim) so a time budget adapts better than a fixed count.
CLS_BUDGET_SEC = 3.5 * 3600
PATCHGRID_BUDGET_SEC = 3.5 * 3600
SEARCH_EPOCHS = 80
FINAL_EPOCHS = 100
MAX_OFFSET = 8  # must match MouseBehaviorClassifier's max_offset
MAX_TRAIN_FRAMES_PATCHGRID = 200_000  # ~4.9GB GPU-resident at 16 patches x 768dim x fp16


def log_result(record: dict):
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    record['ts'] = time.time()
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f'[LOG] {record}', flush=True)


def sample_cfg(rng: random.Random) -> dict:
    loss_type = rng.choice(['ce', 'focal'])
    # context_k fixed at 2 (T=5 dense positions) — larger dense windows (context_k=3,4) still
    # cost more attention compute per step for little benefit; reach further in time via
    # stride instead (e.g. stride=4 covers +-8 frames at the exact same T=5 cost as stride=1's
    # +-2 frames).
    context_k = 2
    max_stride = max(1, MAX_OFFSET // context_k)
    stride = rng.choice(list(range(1, max_stride + 1)))
    return dict(
        n_heads=rng.choice([1, 2, 4, 8]),
        context_k=context_k,
        stride=stride,
        hidden_dim=rng.choice([128, 256, 384, 512]),
        neg_ratio=rng.choice([5, 10, 15, 20]),
        loss_type=loss_type,
        focal_gamma=rng.choice([1.0, 2.0, 3.0]) if loss_type == 'focal' else 2.0,
        dropout=rng.choice([0.0, 0.05, 0.1, 0.2, 0.3]),
        weight_decay=rng.choice([0.0, 1e-5, 1e-4, 1e-3]),
        lr=rng.choice([3e-4, 1e-3, 3e-3]),
    )


def full_val_pair_macro_pr_auc(model, dev, annotations_csv, pair_labels_path, val_obs,
                                context_k, stride, emb_dim, use_patch_grid, cls_embeddings_path):
    """The one true metric — always the full, non-subsampled validation set, so every
    comparison in this script (trial vs trial, trial vs existing baseline) is apples-to-apples."""
    load_fn = (
        load_patchgrid_embeddings(str(PATCH_GRID_DIR / 'embeddings.npy'), str(PATCH_GRID_DIR / 'global_idx.npy'), 16, emb_dim)
        if use_patch_grid else load_cls_embeddings(str(cls_embeddings_path), emb_dim)
    )
    val_data = FastBatchData(
        str(annotations_csv), str(pair_labels_path), val_obs, context_k, emb_dim, load_fn,
        n_patches=16 if use_patch_grid else None, stride=stride,
    )
    probs, labels = collect_val_predictions_fast(model, val_data, dev)
    per_class = {c: average_precision_score((labels == c).astype(int), probs[:, c]) for c in (1, 2)}
    macro = float(np.mean(list(per_class.values())))
    return macro, per_class


def run_trial(cfg, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim,
              train_obs, val_obs, n_epochs, tag, use_patch_grid=False):
    out_dir = TMP_DIR / tag
    # Patch-grid's extra PatchAttnPool stage was found (earlier this session, manually) to collapse
    # into a dead uniform-softmax state at lr>=1e-3 even with grad clipping — clamp lr for
    # patch-grid trials specifically rather than let the search rediscover this by wasting trials.
    lr = min(cfg['lr'], 3e-4) if use_patch_grid else cfg['lr']
    kwargs = dict(
        annotations_csv=str(annotations_csv), pair_labels_parquet=str(pair_labels_path),
        embeddings_path=str(cls_embeddings_path), output_dir=str(out_dir),
        train_obs_ids=train_obs, val_obs_ids=val_obs, context_k=cfg['context_k'], stride=cfg['stride'],
        emb_dim=emb_dim, n_heads=cfg['n_heads'], hidden_dim=cfg['hidden_dim'], n_epochs=n_epochs,
        neg_ratio=cfg['neg_ratio'], loss_type=cfg['loss_type'], focal_gamma=cfg['focal_gamma'],
        lr=lr, dropout=cfg['dropout'], weight_decay=cfg['weight_decay'],
        device='cuda', seed=SEED, verbose=False, use_patch_grid=use_patch_grid, eval_every=1,
    )
    if use_patch_grid:
        kwargs.update(
            patch_embeddings_path=str(PATCH_GRID_DIR / 'embeddings.npy'),
            patch_global_idx_path=str(PATCH_GRID_DIR / 'global_idx.npy'),
            n_patches=16,
            # The available GPU has limited VRAM; patch-grid's per-batch context (16x more tokens
            # than CLS) OOM'd during backward() at the default batch_size=4096 (found earlier
            # this session).
            batch_size=1024,
            # Patch-grid's full train split is ~15GB GPU-resident (16x CLS's footprint) — never
            # fits on a GPU with limited VRAM, silently falling back to slow CPU-gather every
            # trial. Bounding to ~200k frames (~4.9GB) keeps every positive-containing observation
            # plus a big, still-diverse pool of negative-only ones, and fits comfortably alongside
            # the ~3.1GB GPU-resident val set.
            max_train_frames=MAX_TRAIN_FRAMES_PATCHGRID,
        )
    result = train(**kwargs)
    model = result['model']
    dev = next(model.parameters()).device
    full_pr_auc, full_per_class = full_val_pair_macro_pr_auc(
        model, dev, annotations_csv, pair_labels_path, val_obs, cfg['context_k'], cfg['stride'],
        emb_dim, use_patch_grid, cls_embeddings_path,
    )
    del result['model']
    torch.cuda.empty_cache()
    shutil.rmtree(out_dir, ignore_errors=True)  # only the final winner's checkpoint matters, not every trial's
    return full_pr_auc, full_per_class, result['best_pr_auc']


def search_variant(variant_name, use_patch_grid, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim,
                    train_obs, val_obs, budget_sec):
    print(f'=== Searching {variant_name} (budget {budget_sec/3600:.1f}h) ===', flush=True)
    sample_rng = random.Random(SEED + (2 if use_patch_grid else 1))
    results = []
    trial_i = 0
    t_stage = time.time()
    while time.time() - t_stage < budget_sec:
        cfg = sample_cfg(sample_rng)
        tag = f'{variant_name}_{trial_i}'
        t0 = time.time()
        try:
            full_pr_auc, full_per_class, internal_pr_auc = run_trial(
                cfg, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim, train_obs, val_obs,
                SEARCH_EPOCHS, tag, use_patch_grid=use_patch_grid,
            )
        except Exception as e:
            log_result({'variant': variant_name, 'tag': tag, 'cfg': cfg, 'error': str(e), 'traceback': traceback.format_exc()})
            trial_i += 1
            continue
        dt = time.time() - t0
        log_result({'variant': variant_name, 'tag': tag, 'cfg': cfg, 'full_val_pair_macro_pr_auc': full_pr_auc,
                    'full_val_per_class': full_per_class, 'internal_subsampled_pr_auc': internal_pr_auc, 'seconds': dt})
        results.append((full_pr_auc, cfg))
        trial_i += 1

    results.sort(key=lambda x: -x[0])
    print(f'{variant_name} search done: {trial_i} trials in {(time.time()-t_stage)/60:.1f} min', flush=True)
    return results, trial_i


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['cls', 'patchgrid', 'both'], default='both',
                         help="Run only one variant's search. Submit 'cls' and 'patchgrid' as two "
                              "separate SLURM jobs (recommended) so patch-grid always starts with a "
                              "clean GPU — running both in one process risked the second variant's "
                              "training data falling back to slow CPU-gather when the first variant's "
                              "GPU memory wasn't fully reclaimed in between (observed in testing).")
    args = parser.parse_args()

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

    summary_lines = [
        '# Hyperparameter search summary\n',
        f'Started: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start))}\n',
        'All scores below are full, non-subsampled validation-set pair-level macro PR-AUC '
        '(nt/nn only, computed identically for the search, the baseline, and the final report).\n\n',
    ]

    all_variants = [
        ('cls', False, CLS_BUDGET_SEC, 'CLS-token (baseline) mouse behavior classifier'),
        ('patchgrid', True, PATCHGRID_BUDGET_SEC, 'Patch-grid (attention-pooled) mouse behavior classifier'),
    ]
    variants = [v for v in all_variants if args.variant == 'both' or v[0] == args.variant]

    for variant, use_patch_grid, budget, name in variants:
        results, n_trials = search_variant(
            variant, use_patch_grid, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim,
            train_obs, val_obs, budget,
        )
        out_dir = RESULTS_DIR / variant
        summary_lines.append(f'## {variant} ({n_trials} trials)\n')

        # Recompute the CURRENT champion's full-val score fresh — never trust whatever
        # number happens to be sitting in config.json, which may have been computed under
        # a different (e.g. subsampled) methodology.
        baseline_pr_auc = -1.0
        if (out_dir / 'best_model.pt').exists() and (out_dir / 'config.json').exists():
            base_cfg = json.load(open(out_dir / 'config.json'))['cfg']
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            base_model = MouseBehaviorClassifier(
                emb_dim=emb_dim, n_heads=base_cfg['n_heads'], hidden_dim=base_cfg['hidden_dim'],
                use_patch_grid=use_patch_grid, dropout=base_cfg.get('dropout', 0.1),
            ).to(dev)
            base_model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=dev, weights_only=True))
            base_model.eval()
            baseline_pr_auc, _ = full_val_pair_macro_pr_auc(
                base_model, dev, annotations_csv, pair_labels_path, val_obs,
                base_cfg['context_k'], base_cfg.get('stride', 1), emb_dim, use_patch_grid, cls_embeddings_path,
            )
            del base_model
            torch.cuda.empty_cache()
        log_result({'variant': variant, 'stage': 'baseline_recomputed', 'full_val_pair_macro_pr_auc': baseline_pr_auc})
        summary_lines.append(f'- Current baseline (recomputed, full val): {baseline_pr_auc:.4f}\n')

        if not results:
            summary_lines.append('- Search found nothing usable (all trials errored).\n\n')
            continue
        best_score, best_cfg = results[0]
        summary_lines.append(f'- Best search result (full val): {best_score:.4f}  cfg={best_cfg}\n')

        if best_score <= baseline_pr_auc:
            print(f'  {variant}: search result {best_score:.4f} did not beat baseline {baseline_pr_auc:.4f} — keeping baseline', flush=True)
            summary_lines.append('- NOT promoted (baseline was already better)\n\n')
            continue

        # Retrain the winner once more at full epoch budget (search trials all ran at
        # SEARCH_EPOCHS) so the promoted checkpoint gets its full early-stopping runway.
        print(f'  {variant}: NEW BEST {best_score:.4f} > baseline {baseline_pr_auc:.4f} — final retrain + promoting', flush=True)
        lr_used = min(best_cfg['lr'], 3e-4) if use_patch_grid else best_cfg['lr']
        kwargs = dict(
            annotations_csv=str(annotations_csv), pair_labels_parquet=str(pair_labels_path),
            embeddings_path=str(cls_embeddings_path), output_dir=str(TMP_DIR / f'final_{variant}'),
            train_obs_ids=train_obs, val_obs_ids=val_obs, context_k=best_cfg['context_k'], stride=best_cfg['stride'],
            emb_dim=emb_dim, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'], n_epochs=FINAL_EPOCHS,
            neg_ratio=best_cfg['neg_ratio'], loss_type=best_cfg['loss_type'], focal_gamma=best_cfg['focal_gamma'],
            lr=lr_used, dropout=best_cfg['dropout'], weight_decay=best_cfg['weight_decay'],
            device='cuda', seed=SEED, verbose=True, use_patch_grid=use_patch_grid, eval_every=1,
        )
        if use_patch_grid:
            kwargs.update(
                patch_embeddings_path=str(PATCH_GRID_DIR / 'embeddings.npy'),
                patch_global_idx_path=str(PATCH_GRID_DIR / 'global_idx.npy'),
                n_patches=16, batch_size=1024, max_train_frames=MAX_TRAIN_FRAMES_PATCHGRID,
            )
        try:
            result = train(**kwargs)
        except Exception as e:
            log_result({'variant': variant, 'stage': 'final_retrain', 'cfg': best_cfg, 'error': str(e), 'traceback': traceback.format_exc()})
            summary_lines.append(f'- Final retrain FAILED: {e}\n\n')
            continue

        model = result['model']
        dev = next(model.parameters()).device
        final_score, final_per_class = full_val_pair_macro_pr_auc(
            model, dev, annotations_csv, pair_labels_path, val_obs,
            best_cfg['context_k'], best_cfg['stride'], emb_dim, use_patch_grid, cls_embeddings_path,
        )
        log_result({'variant': variant, 'stage': 'final_retrain', 'cfg': best_cfg, 'full_val_pair_macro_pr_auc': final_score})

        if final_score <= baseline_pr_auc:
            # Rare (search trial ran fewer epochs and got lucky, or run-to-run noise) but
            # possible — re-check before promoting the fully-retrained model too.
            print(f'  {variant}: full retrain {final_score:.4f} did not beat baseline {baseline_pr_auc:.4f} after all — NOT promoting', flush=True)
            summary_lines.append(f'- Full retrain scored {final_score:.4f}, did not beat baseline — NOT promoted\n\n')
            continue

        cfg_final = {**best_cfg, 'lr': lr_used}
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / 'best_model.pt')
        with open(out_dir / 'history.json', 'w') as f:
            json.dump(result['history'], f, indent=2)
        with open(out_dir / 'config.json', 'w') as f:
            json.dump({'cfg': cfg_final, 'val_pools': sorted(val_pool_set), 'n_epochs': FINAL_EPOCHS,
                       'best_pr_auc': final_score, 'best_per_class': final_per_class,
                       'emb_dim': emb_dim, 'promoted_by_search': True}, f, indent=2)

        load_fn = (
            load_patchgrid_embeddings(str(PATCH_GRID_DIR / 'embeddings.npy'), str(PATCH_GRID_DIR / 'global_idx.npy'), 16, emb_dim)
            if use_patch_grid else load_cls_embeddings(str(cls_embeddings_path), emb_dim)
        )
        val_data = FastBatchData(
            str(annotations_csv), str(pair_labels_path), val_obs, best_cfg['context_k'], emb_dim, load_fn,
            n_patches=16 if use_patch_grid else None, stride=best_cfg['stride'],
        )
        probs, labels = collect_val_predictions_fast(model, val_data, dev)
        generate_report(probs, labels, result['history'], name, cfg_final, out_dir)
        summary_lines.append(f'- **PROMOTED** to results/vision/mice/opair/{variant}/ (full-val score {final_score:.4f}, report.png regenerated)\n\n')
        del result['model']
        torch.cuda.empty_cache()

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    # Append when only one variant ran (e.g. two separate SLURM jobs for cls/patchgrid) so
    # both jobs' results accumulate into a single summary rather than overwriting each other.
    mode = 'a' if args.variant != 'both' and (SEARCH_DIR / 'SUMMARY.md').exists() else 'w'
    with open(SEARCH_DIR / 'SUMMARY.md', mode) as f:
        f.writelines(summary_lines)
    print(f'Search complete in {(time.time()-t_start)/3600:.2f}h. See {SEARCH_DIR / "SUMMARY.md"}', flush=True)


if __name__ == '__main__':
    main()
