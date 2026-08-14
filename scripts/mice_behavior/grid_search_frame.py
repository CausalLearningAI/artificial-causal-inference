"""
Autonomous hyperparameter search for the per-frame behavior classifier, run
entirely in-process (one long SLURM allocation) via train_frame().

Objective: full, non-subsampled validation-set frame-level macro AP (mean of
AP(nt), AP(nn)) — computed once, right after each trial's training finishes,
via collect_frame_val_predictions(). Deliberately NOT the internal per-epoch
smoothed_ap train_frame() tracks for early stopping (that one runs on a
rebalanced val subsample for per-epoch speed and isn't comparable across
trials with different subsample sizes) — see grid_search.py's docstring for
the pairwise model, which follows the exact same discipline for the same reason.

Two variants (CLS, patch-grid), same idea as grid_search.py:
  - CLS: pooled DINOv2 CLS-token embedding. A prior full search (260 trials)
    plateaued at full-val macro AP 0.158, config n_heads=8/hidden=512/dropout=0.2.
  - patch-grid: 4x4 coarse spatial tokens instead of one pooled vector. A single
    diagnostic run (same hyperparams as the CLS winner) already beat it by +34%
    (0.211) but showed much worse loss-curve overfitting (val_loss 2->11 across
    training) than CLS ever did at the SAME dropout/weight_decay — so patch-grid's
    regularization search range is deliberately wider (this diagnostic run's
    settings clearly weren't enough), and it uses max_train_frames bounding +
    smaller batch_size + capped lr, mirroring grid_search.py's patch-grid handling
    (its per-frame footprint is 16x CLS's, doesn't fit GPU-resident otherwise).

Only PROMOTED to results/vision/mice/frame/{cls,patchgrid}/ if the winning
config's full-val score beats a freshly-recomputed full-val score for the
CURRENT best_model.pt.

Every trial is logged to results/vision/mice/frame/search/log.jsonl (shared
across both variants, distinguished by a 'variant' field — same convention as
grid_search.py's pairwise search). A human-readable
results/vision/mice/frame/search/SUMMARY.md is written at the end (appended
to when only one variant is run, so separate SLURM jobs for cls/patchgrid
accumulate into one summary rather than overwriting each other).

Usage:
    python scripts/mice_behavior/grid_search_frame.py --variant patchgrid
    python scripts/mice_behavior/grid_search_frame.py --variant cls
    python scripts/mice_behavior/grid_search_frame.py --variant both
"""
import argparse
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
from src.mice_behavior.batch_data import (
    FrameBatchData, load_cls_embeddings, load_patchgrid_embeddings, load_patchgrid_concat_embeddings, cached_loader,
)
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import load_obs_to_pool_map, get_val_pools
from src.mice_behavior.report import collect_frame_val_predictions, generate_frame_report
from src.mice_behavior.train import train_frame

DATA_DIR = Path('./data')
DATASET_DIR = Path('./dataset')
RESULTS_DIR = Path('./results/vision/mice')
FRAME_DIR = RESULTS_DIR / 'frame'
SEARCH_DIR = FRAME_DIR / 'search'
LOG_PATH = SEARCH_DIR / 'log.jsonl'
TMP_DIR = SEARCH_DIR / 'tmp'
SEED = 42
ENCODER, TOKEN = 'dinov2', 'class_l-2'
PATCH_GRID_DIR = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / 'dinov2' / 'patch_grid4'
PATCH_GRID_DIR_DINOV3 = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / 'dinov3' / 'patch_grid4'

CLS_BUDGET_SEC = 3.5 * 3600
PATCHGRID_BUDGET_SEC = 3.5 * 3600
SEARCH_EPOCHS = 80
FINAL_EPOCHS = 100
MAX_OFFSET = 8  # must match MouseFrameClassifier's max_offset
MAX_TRAIN_FRAMES_PATCHGRID = 200_000  # ~4.9GB GPU-resident at 16 patches x 768dim x fp16 — same budget as grid_search.py's pairwise patch-grid (same per-frame footprint; frame-level doesn't multiply by 12 pairs, but the underlying frame count is unchanged, so the same bound applies)
MAX_TRAIN_FRAMES_PATCHGRID_CONCAT = 60_000  # concat is 1536-dim (2x768); val alone (126,000 frames, always loaded unbounded) already costs ~6.2GB GPU-resident at this width, so train is cut further than a simple halving to leave headroom for batch activations on an 11GB 2080ti (same GPU class as the single-encoder patch-grid searches)


def log_result(record: dict):
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    record['ts'] = time.time()
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f'[LOG] {record}', flush=True)


def sample_cfg(rng: random.Random, use_patch_grid: bool) -> dict:
    # context_k fixed at 2 (T=5 dense positions), reach further via stride instead —
    # same rationale as grid_search.py's sample_cfg (attention cost stays fixed).
    context_k = 2
    max_stride = max(1, MAX_OFFSET // context_k)
    stride = rng.choice(list(range(1, max_stride + 1)))
    if use_patch_grid:
        # dropout kept wider than CLS's range — the patch-grid diagnostic run overfit much
        # harder (val_loss 2->11 across training) at dropout=0.2, the setting sufficient for
        # CLS, and this range showed no clear good/bad split across an initial 32-trial run.
        # weight_decay, by contrast, was ALSO widened initially but a first 32-trial run
        # showed a stark, monotonic collapse as it increases (mean AP 0.151 @ 1e-4 -> 0.037 @
        # 1e-1) — every value beyond CLS's own range actively hurt, so reverted to match.
        # Also lr is capped separately below (PatchAttnPool collapse precedent from the
        # pairwise search).
        dropout = rng.choice([0.1, 0.2, 0.3, 0.4, 0.5])
        weight_decay = rng.choice([0.0, 1e-5, 1e-4, 1e-3])
    else:
        dropout = rng.choice([0.0, 0.05, 0.1, 0.2, 0.3])
        weight_decay = rng.choice([0.0, 1e-5, 1e-4, 1e-3])
    return dict(
        n_heads=rng.choice([1, 2, 4, 8]),
        context_k=context_k,
        stride=stride,
        hidden_dim=rng.choice([128, 256, 384, 512]),
        neg_ratio=rng.choice([5, 10, 15, 20]),
        dropout=dropout,
        weight_decay=weight_decay,
        lr=rng.choice([3e-4, 1e-3, 3e-3]),
    )


def full_val_frame_macro_ap(model, dev, annotations_csv, pair_labels_path, val_obs,
                             context_k, stride, emb_dim, load_fn, n_patches=None):
    """The one true metric — always the full, non-subsampled validation set.
    load_fn: a load_embeddings_fn (typically a shared cached_loader(...) instance — see
    run_trial/main — so repeated calls across trials reuse the same in-RAM embeddings
    instead of re-reading them from NFS every time)."""
    val_data = FrameBatchData(
        str(annotations_csv), str(pair_labels_path), val_obs, context_k, emb_dim, load_fn,
        n_patches=n_patches, stride=stride,
    )
    probs, labels = collect_frame_val_predictions(model, val_data, dev)
    per_label = {name: average_precision_score(labels[:, i], probs[:, i]) for i, name in enumerate(['nt', 'nn'])}
    macro = float(np.mean(list(per_label.values())))
    return macro, per_label


def run_trial(cfg, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim,
              train_obs, val_obs, n_epochs, tag, load_fn, use_patch_grid=False,
              max_train_frames=MAX_TRAIN_FRAMES_PATCHGRID, batch_size=1024):
    out_dir = TMP_DIR / tag
    # Patch-grid's extra PatchAttnPool stage was found (pairwise search, earlier session) to
    # collapse into a dead uniform-softmax state at lr>=1e-3 even with grad clipping — clamp
    # lr for patch-grid trials specifically rather than let the search rediscover this.
    lr = min(cfg['lr'], 3e-4) if use_patch_grid else cfg['lr']
    kwargs = dict(
        annotations_csv=str(annotations_csv), pair_labels_parquet=str(pair_labels_path),
        embeddings_path=str(cls_embeddings_path), output_dir=str(out_dir),
        train_obs_ids=train_obs, val_obs_ids=val_obs, context_k=cfg['context_k'], stride=cfg['stride'],
        emb_dim=emb_dim, n_heads=cfg['n_heads'], hidden_dim=cfg['hidden_dim'], n_epochs=n_epochs,
        neg_ratio=cfg['neg_ratio'], lr=lr, dropout=cfg['dropout'], weight_decay=cfg['weight_decay'],
        device='cuda', seed=SEED, verbose=False, use_patch_grid=use_patch_grid, eval_every=1,
        embeddings_loader=load_fn,
    )
    n_patches = None
    if use_patch_grid:
        n_patches = 16
        kwargs.update(
            n_patches=n_patches,
            # The available GPU has limited VRAM; patch-grid's per-batch context (16x more
            # tokens than CLS) needs a smaller batch — same precedent as grid_search.py.
            batch_size=batch_size,
            max_train_frames=max_train_frames,
        )
    result = train_frame(**kwargs)
    model = result['model']
    dev = next(model.parameters()).device
    full_ap, full_per_label = full_val_frame_macro_ap(
        model, dev, annotations_csv, pair_labels_path, val_obs, cfg['context_k'], cfg['stride'],
        emb_dim, load_fn, n_patches=n_patches,
    )
    del result['model']
    torch.cuda.empty_cache()
    shutil.rmtree(out_dir, ignore_errors=True)
    return full_ap, full_per_label, result['best_ap']


def search_variant(variant_name, use_patch_grid, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim,
                    train_obs, val_obs, budget_sec, load_fn,
                    max_train_frames=MAX_TRAIN_FRAMES_PATCHGRID, batch_size=1024):
    print(f'=== Searching {variant_name} (budget {budget_sec/3600:.1f}h) ===', flush=True)
    sample_rng = random.Random(SEED + (2 if use_patch_grid else 1))
    results = []
    trial_i = 0
    t_stage = time.time()
    while time.time() - t_stage < budget_sec:
        cfg = sample_cfg(sample_rng, use_patch_grid)
        tag = f'{variant_name}_{trial_i}'
        t0 = time.time()
        try:
            full_ap, full_per_label, internal_ap = run_trial(
                cfg, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim, train_obs, val_obs,
                SEARCH_EPOCHS, tag, load_fn, use_patch_grid=use_patch_grid,
                max_train_frames=max_train_frames, batch_size=batch_size,
            )
        except Exception as e:
            log_result({'variant': variant_name, 'tag': tag, 'cfg': cfg, 'error': str(e), 'traceback': traceback.format_exc()})
            trial_i += 1
            continue
        dt = time.time() - t0
        log_result({'variant': variant_name, 'tag': tag, 'cfg': cfg, 'full_val_frame_macro_ap': full_ap,
                    'full_val_per_label': full_per_label, 'internal_subsampled_ap': internal_ap, 'seconds': dt})
        results.append((full_ap, cfg))
        trial_i += 1

    results.sort(key=lambda x: -x[0])
    print(f'{variant_name} search done: {trial_i} trials in {(time.time()-t_stage)/60:.1f} min', flush=True)
    return results, trial_i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--variant', default='both',
        choices=['cls', 'patchgrid', 'patchgrid_dinov3', 'patchgrid_concat', 'both'],
        help="Run only one variant's search. Submit each as its own separate SLURM job "
             "(recommended) — same rationale as grid_search.py. 'both' means cls+patchgrid "
             "only (original two); the DINOv3/concat variants are always run explicitly.")
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

    # Stable, insertion-robust split (see pools.get_val_pools docstring) — a plain
    # shuffle-the-pool-list split silently reshuffles which pools land in val whenever
    # the pool count changes (e.g. a new annotated pool added), making results across
    # dataset versions incomparable by luck of the draw, not real model quality.
    val_pool_set = get_val_pools(pools, seed=SEED)
    train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]

    summary_lines = [
        '# Per-frame classifier hyperparameter search summary\n',
        f'Started: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start))}\n',
        'All scores below are full, non-subsampled validation-set frame-level macro AP '
        '(nt/nn), computed identically for the search, the baseline, and the final report.\n\n',
    ]

    # Each entry: (use_patch_grid, budget, title, emb_dim, raw_load_fn, max_train_frames, batch_size).
    # emb_dim/raw_load_fn are per-variant now (not a single module-wide value) since the DINOv3
    # and concat variants read different embedding sources with different dimensionality —
    # concat is 1536-dim (768+768, both sources L2-normalized before concatenation: the 20x
    # norm mismatch between DINOv2 and DINOv3 CLS/patch tokens collapsed a prior unnormalized
    # comparison to exact-chance predictions) and gets a halved max_train_frames to keep the
    # same GPU-resident footprint as the single-encoder variants on the same GPU class.
    all_variants = {
        'cls': dict(
            use_patch_grid=False, budget=CLS_BUDGET_SEC, title='CLS-token per-frame mouse behavior classifier',
            emb_dim=emb_dim, raw_load_fn=load_cls_embeddings(str(cls_embeddings_path), emb_dim),
            max_train_frames=None, batch_size=4096,
        ),
        'patchgrid': dict(
            use_patch_grid=True, budget=PATCHGRID_BUDGET_SEC,
            title='Patch-grid (attention-pooled) per-frame mouse behavior classifier — DINOv2',
            emb_dim=emb_dim,
            raw_load_fn=load_patchgrid_embeddings(str(PATCH_GRID_DIR / 'embeddings.npy'), str(PATCH_GRID_DIR / 'global_idx.npy'), 16, emb_dim),
            max_train_frames=MAX_TRAIN_FRAMES_PATCHGRID, batch_size=1024,
        ),
        'patchgrid_dinov3': dict(
            use_patch_grid=True, budget=PATCHGRID_BUDGET_SEC,
            title='Patch-grid (attention-pooled) per-frame mouse behavior classifier — DINOv3',
            emb_dim=emb_dim,
            raw_load_fn=load_patchgrid_embeddings(str(PATCH_GRID_DIR_DINOV3 / 'embeddings.npy'), str(PATCH_GRID_DIR_DINOV3 / 'global_idx.npy'), 16, emb_dim),
            max_train_frames=MAX_TRAIN_FRAMES_PATCHGRID, batch_size=1024,
        ),
        'patchgrid_concat': dict(
            use_patch_grid=True, budget=PATCHGRID_BUDGET_SEC,
            title='Patch-grid (attention-pooled) per-frame mouse behavior classifier — DINOv2+DINOv3 concat (L2-normalized)',
            emb_dim=2 * emb_dim,
            raw_load_fn=load_patchgrid_concat_embeddings(
                str(PATCH_GRID_DIR / 'embeddings.npy'), str(PATCH_GRID_DIR_DINOV3 / 'embeddings.npy'),
                str(PATCH_GRID_DIR / 'global_idx.npy'), 16, emb_dim, emb_dim,
            ),
            max_train_frames=MAX_TRAIN_FRAMES_PATCHGRID_CONCAT, batch_size=128,
        ),
    }
    # Output folder names are distinct from the CLI --variant identifiers (kept stable so
    # existing sbatch scripts/search log entries don't need updating) — this is where the
    # "4x4 pooled" vs "256 raw tokens" distinction (see train_patchgrid_online.py, a
    # separate script — not one of this search's own variants) and encoder choice are made
    # explicit in results/vision/mice/frame/, since 'patchgrid'/'patchgrid_dinov3'/etc. alone
    # don't say which resolution or encoder without opening the config.
    # LEGACY. These four directories were deleted on 2026-08-14 -- they held the 4x4-era runs,
    # scored on the retired rd14/rd19/rd29/rd35_3 split. Re-running this script recreates them
    # from scratch; it does not resurrect the old results, and nothing it writes here is
    # comparable to a current run. See results/vision/mice/frame/RETIRED.md for what they
    # established, and rename_runs.py for the naming scheme current runs follow instead.
    OUTPUT_DIR_NAME = {
        'cls': 'cls',
        'patchgrid': 'patchgrid4x4_dinov2',
        'patchgrid_dinov3': 'patchgrid4x4_dinov3',
        'patchgrid_concat': 'patchgrid4x4_concat',
    }
    variant_names = ['cls', 'patchgrid'] if args.variant == 'both' else [args.variant]

    for variant in variant_names:
        spec = all_variants[variant]
        use_patch_grid, budget, name = spec['use_patch_grid'], spec['budget'], spec['title']
        variant_emb_dim = spec['emb_dim']
        max_train_frames, batch_size = spec['max_train_frames'], spec['batch_size']
        # Separate cached loader per variant — different variants read different embedding
        # files, so they can't share one cache (see cached_loader's docstring: it's keyed on
        # obs_boundary, not on which underlying file was loaded).
        load_fn = cached_loader(spec['raw_load_fn'])
        n_patches = 16 if use_patch_grid else None

        results, n_trials = search_variant(
            variant, use_patch_grid, annotations_csv, pair_labels_path, cls_embeddings_path, variant_emb_dim,
            train_obs, val_obs, budget, load_fn, max_train_frames=max_train_frames, batch_size=batch_size,
        )
        out_dir = FRAME_DIR / OUTPUT_DIR_NAME[variant]
        summary_lines.append(f'## {variant} ({n_trials} trials)\n')

        # Recompute the CURRENT champion's full-val score fresh — never trust whatever
        # number happens to be sitting in config.json.
        baseline_ap = -1.0
        if (out_dir / 'best_model.pt').exists() and (out_dir / 'config.json').exists():
            base_cfg = json.load(open(out_dir / 'config.json'))['cfg']
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            base_model = MouseFrameClassifier(
                emb_dim=variant_emb_dim, n_heads=base_cfg['n_heads'], hidden_dim=base_cfg['hidden_dim'],
                use_patch_grid=use_patch_grid, dropout=base_cfg.get('dropout', 0.1),
            ).to(dev)
            base_model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=dev, weights_only=True))
            base_model.eval()
            baseline_ap, _ = full_val_frame_macro_ap(
                base_model, dev, annotations_csv, pair_labels_path, val_obs,
                base_cfg['context_k'], base_cfg.get('stride', 1), variant_emb_dim, load_fn, n_patches=n_patches,
            )
            del base_model
            torch.cuda.empty_cache()
        log_result({'variant': variant, 'stage': 'baseline_recomputed', 'full_val_frame_macro_ap': baseline_ap})
        summary_lines.append(f'- Current baseline (recomputed, full val): {baseline_ap:.4f}\n')

        if not results:
            summary_lines.append('- Search found nothing usable (all trials errored).\n\n')
            continue
        best_score, best_cfg = results[0]
        summary_lines.append(f'- Best search result (full val): {best_score:.4f}  cfg={best_cfg}\n')

        if best_score <= baseline_ap:
            print(f'  {variant}: search result {best_score:.4f} did not beat baseline {baseline_ap:.4f} — keeping baseline', flush=True)
            summary_lines.append('- NOT promoted (baseline was already better)\n\n')
            continue

        print(f'  {variant}: NEW BEST {best_score:.4f} > baseline {baseline_ap:.4f} — final retrain + promoting', flush=True)
        lr_used = min(best_cfg['lr'], 3e-4) if use_patch_grid else best_cfg['lr']
        kwargs = dict(
            annotations_csv=str(annotations_csv), pair_labels_parquet=str(pair_labels_path),
            embeddings_path=str(cls_embeddings_path), output_dir=str(TMP_DIR / f'final_{variant}'),
            train_obs_ids=train_obs, val_obs_ids=val_obs, context_k=best_cfg['context_k'], stride=best_cfg['stride'],
            emb_dim=variant_emb_dim, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'], n_epochs=FINAL_EPOCHS,
            neg_ratio=best_cfg['neg_ratio'], lr=lr_used, dropout=best_cfg['dropout'], weight_decay=best_cfg['weight_decay'],
            device='cuda', seed=SEED, verbose=True, use_patch_grid=use_patch_grid, eval_every=1,
            embeddings_loader=load_fn,
        )
        if use_patch_grid:
            kwargs.update(n_patches=16, batch_size=batch_size, max_train_frames=max_train_frames)
        try:
            result = train_frame(**kwargs)
        except Exception as e:
            log_result({'variant': variant, 'stage': 'final_retrain', 'cfg': best_cfg, 'error': str(e), 'traceback': traceback.format_exc()})
            summary_lines.append(f'- Final retrain FAILED: {e}\n\n')
            continue

        model = result['model']
        dev = next(model.parameters()).device
        final_score, final_per_label = full_val_frame_macro_ap(
            model, dev, annotations_csv, pair_labels_path, val_obs,
            best_cfg['context_k'], best_cfg['stride'], variant_emb_dim, load_fn, n_patches=n_patches,
        )
        log_result({'variant': variant, 'stage': 'final_retrain', 'cfg': best_cfg, 'full_val_frame_macro_ap': final_score})

        if final_score <= baseline_ap:
            print(f'  {variant}: full retrain {final_score:.4f} did not beat baseline {baseline_ap:.4f} after all — NOT promoting', flush=True)
            summary_lines.append(f'- Full retrain scored {final_score:.4f}, did not beat baseline — NOT promoted\n\n')
            del result['model']
            torch.cuda.empty_cache()
            continue

        cfg_final = {**best_cfg, 'lr': lr_used}
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / 'best_model.pt')
        with open(out_dir / 'history.json', 'w') as f:
            json.dump(result['history'], f, indent=2)
        with open(out_dir / 'config.json', 'w') as f:
            json.dump({'cfg': cfg_final, 'val_pools': sorted(val_pool_set), 'n_epochs': FINAL_EPOCHS,
                       'best_ap': final_score, 'best_per_label': final_per_label,
                       'emb_dim': variant_emb_dim, 'promoted_by_search': True}, f, indent=2)

        val_data = FrameBatchData(
            str(annotations_csv), str(pair_labels_path), val_obs, best_cfg['context_k'], variant_emb_dim, load_fn,
            n_patches=n_patches, stride=best_cfg['stride'],
        )
        probs, labels = collect_frame_val_predictions(model, val_data, dev)
        generate_frame_report(probs, labels, result['history'], name, cfg_final, out_dir)
        summary_lines.append(f'- **PROMOTED** to results/vision/mice/frame/{OUTPUT_DIR_NAME[variant]}/ (full-val score {final_score:.4f}, report.png regenerated)\n\n')
        del result['model']
        torch.cuda.empty_cache()

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    mode = 'a' if args.variant != 'both' and (SEARCH_DIR / 'SUMMARY.md').exists() else 'w'
    with open(SEARCH_DIR / 'SUMMARY.md', mode) as f:
        f.writelines(summary_lines)
    print(f'Search complete in {(time.time()-t_start)/3600:.2f}h. See {SEARCH_DIR / "SUMMARY.md"}', flush=True)


if __name__ == '__main__':
    main()
