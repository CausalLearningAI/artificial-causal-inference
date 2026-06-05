"""Train mouse pairwise behavior classifier (baseline)."""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.build_pair_labels import build_pair_labels
from src.mice_behavior.train import train


def obs_to_pool(obs_id: str) -> str:
    # observation_id format: {genotype}_{line}_{sex}_{seed}_{odor}_{phase}
    # Pool key = line_sex_seed (same group of 4 mice across sessions)
    parts = obs_id.split('_')
    return '_'.join(parts[1:4])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--dataset-dir', default='./dataset')
    p.add_argument('--results-dir', default='./results/mice_behavior')
    p.add_argument('--encoder', default='dinov2', choices=['dinov2', 'dinov3', 'siglip', 'siglip2'])
    p.add_argument('--token', default='class', choices=['class', 'mean'])
    p.add_argument('--context-k', type=int, default=2)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val-frac', type=float, default=0.2, help='Fraction of pools held out for validation')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite-labels', action='store_true')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    dataset_dir = Path(args.dataset_dir)

    # Step 1: build sparse pair labels from BORIS CSVs
    pair_labels_path = build_pair_labels(data_dir, dataset_dir, overwrite=args.overwrite_labels)

    # Step 2: pool-level train/val split (split by pool, not by observation)
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    pools = list({obs_to_pool(o) for o in all_obs})
    rng = random.Random(args.seed)
    rng.shuffle(pools)
    n_val = max(1, int(len(pools) * args.val_frac))
    val_pool_set = set(pools[:n_val])

    train_obs = [o for o in all_obs if obs_to_pool(o) not in val_pool_set]
    val_obs = [o for o in all_obs if obs_to_pool(o) in val_pool_set]
    print(
        f'Split: {len(train_obs)} train obs / {len(val_obs)} val obs  '
        f'({len(pools) - n_val}/{n_val} pools)'
    )

    # Step 3: infer embedding dim from file size
    annotations_csv = dataset_dir / 'mice' / 'v1' / 'annotations.csv'
    embeddings_path = (
        dataset_dir / 'mice' / 'v1' / 'embeddings' / 'full'
        / args.encoder / args.token / 'embeddings.npy'
    )
    n_frames = sum(1 for _ in open(annotations_csv)) - 1  # fast line count
    emb_dim = embeddings_path.stat().st_size // (4 * n_frames)
    print(f'Embeddings: {args.encoder}/{args.token}, dim={emb_dim}, frames={n_frames:,}')

    # Step 4: train
    train(
        annotations_csv=str(annotations_csv),
        pair_labels_parquet=str(pair_labels_path),
        embeddings_path=str(embeddings_path),
        output_dir=args.results_dir,
        train_obs_ids=train_obs,
        val_obs_ids=val_obs,
        context_k=args.context_k,
        emb_dim=emb_dim,
        n_heads=args.n_heads,
        hidden_dim=args.hidden_dim,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
