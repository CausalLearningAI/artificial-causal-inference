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

Direct random search on the standard 80/20 pool split (same split as
train_frame_final.py). Only PROMOTED to results/mice_behavior/frame/ if the
winning config's full-val score beats a freshly-recomputed full-val score for
the CURRENT best_model.pt.

Every trial is logged to results/mice_behavior/frame/search/log.jsonl. A
human-readable results/mice_behavior/frame/search/SUMMARY.md is written at the
end. Nested under frame/ (not the shared results/mice_behavior/search/ the
pairwise cls+patchgrid search uses) since this search only ever concerns the
one frame variant — unlike grid_search.py, which searches two variants
sharing one script/log.

Usage:
    python scripts/mice_behavior/grid_search_frame.py
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
from src.mice_behavior.batch_data import FrameBatchData, load_cls_embeddings
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import load_obs_to_pool_map
from src.mice_behavior.report import collect_frame_val_predictions, generate_frame_report
from src.mice_behavior.train import train_frame

DATA_DIR = Path('./data')
DATASET_DIR = Path('./dataset')
RESULTS_DIR = Path('./results/mice_behavior')
FRAME_DIR = RESULTS_DIR / 'frame'
SEARCH_DIR = FRAME_DIR / 'search'
LOG_PATH = SEARCH_DIR / 'log.jsonl'
TMP_DIR = SEARCH_DIR / 'tmp'
SEED = 42
ENCODER, TOKEN = 'dinov2', 'class_l-2'

BUDGET_SEC = 3.5 * 3600
SEARCH_EPOCHS = 80
FINAL_EPOCHS = 100
MAX_OFFSET = 8  # must match MouseFrameClassifier's max_offset


def log_result(record: dict):
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    record['ts'] = time.time()
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f'[LOG] {record}', flush=True)


def sample_cfg(rng: random.Random) -> dict:
    # context_k fixed at 2 (T=5 dense positions), reach further via stride instead —
    # same rationale as grid_search.py's sample_cfg (attention cost stays fixed).
    context_k = 2
    max_stride = max(1, MAX_OFFSET // context_k)
    stride = rng.choice(list(range(1, max_stride + 1)))
    return dict(
        n_heads=rng.choice([1, 2, 4, 8]),
        context_k=context_k,
        stride=stride,
        hidden_dim=rng.choice([128, 256, 384, 512]),
        neg_ratio=rng.choice([5, 10, 15, 20]),
        dropout=rng.choice([0.0, 0.05, 0.1, 0.2, 0.3]),
        weight_decay=rng.choice([0.0, 1e-5, 1e-4, 1e-3]),
        lr=rng.choice([3e-4, 1e-3, 3e-3]),
    )


def full_val_frame_macro_ap(model, dev, annotations_csv, pair_labels_path, val_obs,
                             context_k, stride, emb_dim, cls_embeddings_path):
    """The one true metric — always the full, non-subsampled validation set."""
    load_fn = load_cls_embeddings(str(cls_embeddings_path), emb_dim)
    val_data = FrameBatchData(
        str(annotations_csv), str(pair_labels_path), val_obs, context_k, emb_dim, load_fn, stride=stride,
    )
    probs, labels = collect_frame_val_predictions(model, val_data, dev)
    per_label = {name: average_precision_score(labels[:, i], probs[:, i]) for i, name in enumerate(['nt', 'nn'])}
    macro = float(np.mean(list(per_label.values())))
    return macro, per_label


def run_trial(cfg, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim,
              train_obs, val_obs, n_epochs, tag):
    out_dir = TMP_DIR / tag
    kwargs = dict(
        annotations_csv=str(annotations_csv), pair_labels_parquet=str(pair_labels_path),
        embeddings_path=str(cls_embeddings_path), output_dir=str(out_dir),
        train_obs_ids=train_obs, val_obs_ids=val_obs, context_k=cfg['context_k'], stride=cfg['stride'],
        emb_dim=emb_dim, n_heads=cfg['n_heads'], hidden_dim=cfg['hidden_dim'], n_epochs=n_epochs,
        neg_ratio=cfg['neg_ratio'], lr=cfg['lr'], dropout=cfg['dropout'], weight_decay=cfg['weight_decay'],
        device='cuda', seed=SEED, verbose=False, eval_every=1,
    )
    result = train_frame(**kwargs)
    model = result['model']
    dev = next(model.parameters()).device
    full_ap, full_per_label = full_val_frame_macro_ap(
        model, dev, annotations_csv, pair_labels_path, val_obs, cfg['context_k'], cfg['stride'],
        emb_dim, cls_embeddings_path,
    )
    del result['model']
    torch.cuda.empty_cache()
    shutil.rmtree(out_dir, ignore_errors=True)
    return full_ap, full_per_label, result['best_ap']


def search(annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim, train_obs, val_obs, budget_sec):
    print(f'=== Searching frame classifier (budget {budget_sec/3600:.1f}h) ===', flush=True)
    sample_rng = random.Random(SEED)
    results = []
    trial_i = 0
    t_stage = time.time()
    while time.time() - t_stage < budget_sec:
        cfg = sample_cfg(sample_rng)
        tag = f'frame_{trial_i}'
        t0 = time.time()
        try:
            full_ap, full_per_label, internal_ap = run_trial(
                cfg, annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim, train_obs, val_obs,
                SEARCH_EPOCHS, tag,
            )
        except Exception as e:
            log_result({'tag': tag, 'cfg': cfg, 'error': str(e), 'traceback': traceback.format_exc()})
            trial_i += 1
            continue
        dt = time.time() - t0
        log_result({'tag': tag, 'cfg': cfg, 'full_val_frame_macro_ap': full_ap,
                    'full_val_per_label': full_per_label, 'internal_subsampled_ap': internal_ap, 'seconds': dt})
        results.append((full_ap, cfg))
        trial_i += 1

    results.sort(key=lambda x: -x[0])
    print(f'frame search done: {trial_i} trials in {(time.time()-t_stage)/60:.1f} min', flush=True)
    return results, trial_i


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

    # Standard split — same seed/logic as train_frame_final.py / train_cls_final.py.
    rng_split = random.Random(SEED)
    shuffled = pools[:]
    rng_split.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.2))
    val_pool_set = set(shuffled[:n_val])
    train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]

    summary_lines = [
        '# Per-frame classifier hyperparameter search summary\n',
        f'Started: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start))}\n',
        'All scores below are full, non-subsampled validation-set frame-level macro AP '
        '(nt/nn), computed identically for the search, the baseline, and the final report.\n\n',
    ]

    results, n_trials = search(annotations_csv, pair_labels_path, cls_embeddings_path, emb_dim, train_obs, val_obs, BUDGET_SEC)
    out_dir = FRAME_DIR
    summary_lines.append(f'## frame ({n_trials} trials)\n')

    # Recompute the CURRENT champion's full-val score fresh — never trust whatever
    # number happens to be sitting in config.json.
    baseline_ap = -1.0
    if (out_dir / 'best_model.pt').exists() and (out_dir / 'config.json').exists():
        base_cfg = json.load(open(out_dir / 'config.json'))['cfg']
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_model = MouseFrameClassifier(
            emb_dim=emb_dim, n_heads=base_cfg['n_heads'], hidden_dim=base_cfg['hidden_dim'],
            dropout=base_cfg.get('dropout', 0.1),
        ).to(dev)
        base_model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=dev, weights_only=True))
        base_model.eval()
        baseline_ap, _ = full_val_frame_macro_ap(
            base_model, dev, annotations_csv, pair_labels_path, val_obs,
            base_cfg['context_k'], base_cfg.get('stride', 1), emb_dim, cls_embeddings_path,
        )
        del base_model
        torch.cuda.empty_cache()
    log_result({'stage': 'baseline_recomputed', 'full_val_frame_macro_ap': baseline_ap})
    summary_lines.append(f'- Current baseline (recomputed, full val): {baseline_ap:.4f}\n')

    if not results:
        summary_lines.append('- Search found nothing usable (all trials errored).\n\n')
    else:
        best_score, best_cfg = results[0]
        summary_lines.append(f'- Best search result (full val): {best_score:.4f}  cfg={best_cfg}\n')

        if best_score <= baseline_ap:
            print(f'  frame: search result {best_score:.4f} did not beat baseline {baseline_ap:.4f} — keeping baseline', flush=True)
            summary_lines.append('- NOT promoted (baseline was already better)\n\n')
        else:
            print(f'  frame: NEW BEST {best_score:.4f} > baseline {baseline_ap:.4f} — final retrain + promoting', flush=True)
            kwargs = dict(
                annotations_csv=str(annotations_csv), pair_labels_parquet=str(pair_labels_path),
                embeddings_path=str(cls_embeddings_path), output_dir=str(TMP_DIR / 'final'),
                train_obs_ids=train_obs, val_obs_ids=val_obs, context_k=best_cfg['context_k'], stride=best_cfg['stride'],
                emb_dim=emb_dim, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'], n_epochs=FINAL_EPOCHS,
                neg_ratio=best_cfg['neg_ratio'], lr=best_cfg['lr'], dropout=best_cfg['dropout'],
                weight_decay=best_cfg['weight_decay'], device='cuda', seed=SEED, verbose=True, eval_every=1,
            )
            try:
                result = train_frame(**kwargs)
            except Exception as e:
                log_result({'stage': 'final_retrain', 'cfg': best_cfg, 'error': str(e), 'traceback': traceback.format_exc()})
                summary_lines.append(f'- Final retrain FAILED: {e}\n\n')
                result = None

            if result is not None:
                model = result['model']
                dev = next(model.parameters()).device
                final_score, final_per_label = full_val_frame_macro_ap(
                    model, dev, annotations_csv, pair_labels_path, val_obs,
                    best_cfg['context_k'], best_cfg['stride'], emb_dim, cls_embeddings_path,
                )
                log_result({'stage': 'final_retrain', 'cfg': best_cfg, 'full_val_frame_macro_ap': final_score})

                if final_score <= baseline_ap:
                    print(f'  frame: full retrain {final_score:.4f} did not beat baseline {baseline_ap:.4f} after all — NOT promoting', flush=True)
                    summary_lines.append(f'- Full retrain scored {final_score:.4f}, did not beat baseline — NOT promoted\n\n')
                else:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), out_dir / 'best_model.pt')
                    with open(out_dir / 'history.json', 'w') as f:
                        json.dump(result['history'], f, indent=2)
                    with open(out_dir / 'config.json', 'w') as f:
                        json.dump({'cfg': best_cfg, 'val_pools': sorted(val_pool_set), 'n_epochs': FINAL_EPOCHS,
                                   'best_ap': final_score, 'best_per_label': final_per_label,
                                   'emb_dim': emb_dim, 'promoted_by_search': True}, f, indent=2)

                    load_fn = load_cls_embeddings(str(cls_embeddings_path), emb_dim)
                    val_data = FrameBatchData(
                        str(annotations_csv), str(pair_labels_path), val_obs, best_cfg['context_k'], emb_dim, load_fn,
                        stride=best_cfg['stride'],
                    )
                    probs, labels = collect_frame_val_predictions(model, val_data, dev)
                    generate_frame_report(probs, labels, result['history'], 'Per-frame (no identity) mouse behavior classifier', best_cfg, out_dir)
                    summary_lines.append(f'- **PROMOTED** to results/mice_behavior/frame/ (full-val score {final_score:.4f}, report.png regenerated)\n\n')
                del result['model']
                torch.cuda.empty_cache()

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    with open(SEARCH_DIR / 'SUMMARY.md', 'w') as f:
        f.writelines(summary_lines)
    print(f'Search complete in {(time.time()-t_start)/3600:.2f}h. See {SEARCH_DIR / "SUMMARY.md"}', flush=True)


if __name__ == '__main__':
    main()
