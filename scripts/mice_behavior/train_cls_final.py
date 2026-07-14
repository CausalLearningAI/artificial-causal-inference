"""
Full training run of the CLS-token (baseline) variant of the mouse behavior
classifier, on the standard 80/20 pool split, with per-epoch validation
(eval_every=1) so a full loss curve is available. Mirrors
train_patchgrid_final.py exactly (same CFG, same seed/split) for a direct
comparison against the patch-grid attention-pooling variant.

Writes results/mice_behavior/cls/{best_model.pt, history.json, config.json,
report.png, roc_pr_data.npz}.

Usage:
    python scripts/mice_behavior/train_cls_final.py
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.build_pair_labels import build_pair_labels
from src.mice_behavior.fast_data import FastBatchData, load_cls_embeddings
from src.mice_behavior.model import MouseBehaviorClassifier
from src.mice_behavior.pools import load_obs_to_pool_map
from src.mice_behavior.report import collect_val_predictions_fast, generate_report
from src.mice_behavior.train import train

DATA_DIR = Path('./data')
DATASET_DIR = Path('./dataset')
RESULTS_DIR = Path('./results/mice_behavior')
SEED = 42
ENCODER, TOKEN = 'dinov2', 'class_l-2'

# Must match train_patchgrid_final.py's CFG exactly for a fair comparison.
# n_heads raised 1->8 (more attention subspaces, same total width) after the first full runs
# showed severe overfitting (best macro PR-AUC hit within ~15 epochs, then val_loss climbed
# for the remaining 85). Dropout/weight_decay/early_stop_patience are train() defaults now.
CFG = dict(n_heads=8, context_k=2, hidden_dim=256, neg_ratio=10, loss_type='ce')
N_EPOCHS = 100


def main():
    pair_labels_path = build_pair_labels(DATA_DIR, DATASET_DIR, overwrite=False)
    annotations_csv = DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    cls_embeddings_path = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / ENCODER / TOKEN / 'embeddings.npy'
    n_frames = sum(1 for _ in open(annotations_csv)) - 1
    emb_dim = cls_embeddings_path.stat().st_size // (4 * n_frames)

    obs_to_pool = load_obs_to_pool_map(DATA_DIR)
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    pools = sorted({obs_to_pool[o] for o in all_obs})

    rng = random.Random(SEED)
    shuffled = pools[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.2))
    val_pool_set = set(shuffled[:n_val])
    train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]
    print(f'Split: {len(train_obs)} train obs / {len(val_obs)} val obs ({len(pools)-n_val}/{n_val} pools)')

    out_dir = RESULTS_DIR / 'cls'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Training CLS-only model: {CFG}, {N_EPOCHS} epochs, eval every epoch...')
    result = train(
        annotations_csv=str(annotations_csv),
        pair_labels_parquet=str(pair_labels_path),
        embeddings_path=str(cls_embeddings_path),
        output_dir=str(out_dir),
        train_obs_ids=train_obs,
        val_obs_ids=val_obs,
        context_k=CFG['context_k'],
        emb_dim=emb_dim,
        n_heads=CFG['n_heads'],
        hidden_dim=CFG['hidden_dim'],
        n_epochs=N_EPOCHS,
        neg_ratio=CFG['neg_ratio'],
        loss_type=CFG['loss_type'],
        device='cuda',
        seed=SEED,
        verbose=True,
        use_patch_grid=False,
        eval_every=1,
    )
    history = result['history']
    print(f"Best macro PR-AUC: {result['best_pr_auc']:.4f}  {result['best_per_class']}")

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({'cfg': CFG, 'val_pools': sorted(val_pool_set), 'n_epochs': N_EPOCHS,
                   'best_pr_auc': result['best_pr_auc'], 'best_per_class': result['best_per_class'],
                   'emb_dim': emb_dim}, f, indent=2)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MouseBehaviorClassifier(
        emb_dim=emb_dim, n_heads=CFG['n_heads'], hidden_dim=CFG['hidden_dim'], use_patch_grid=False,
    ).to(dev)
    model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=dev, weights_only=True))
    model.eval()

    val_data = FastBatchData(
        str(annotations_csv), str(pair_labels_path), val_obs, CFG['context_k'], emb_dim,
        load_cls_embeddings(str(cls_embeddings_path), emb_dim),
    )
    probs, labels = collect_val_predictions_fast(model, val_data, dev)
    generate_report(probs, labels, history, 'CLS-token (baseline) mouse behavior classifier', CFG, out_dir)
    print(f'Saved {out_dir / "report.png"}')


if __name__ == '__main__':
    main()
