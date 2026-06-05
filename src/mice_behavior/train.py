import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from .dataset import MousePairDataset, collate_fn
from .model import MouseBehaviorClassifier

LABEL_NAMES = ['none', 'nt', 'nn']


class DynamicNegativeSampler(Sampler):
    """Each epoch: all positive samples + a fresh random draw of neg_ratio×n_pos negatives."""

    def __init__(self, labels: np.ndarray, neg_ratio: int = 1, seed: int = 42):
        self.pos_idx = np.where(labels > 0)[0]
        self.neg_idx = np.where(labels == 0)[0]
        self.neg_ratio = neg_ratio
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n_neg = min(len(self.neg_idx), self.neg_ratio * len(self.pos_idx))
        neg_sample = self.rng.choice(self.neg_idx, size=n_neg, replace=False)
        idx = np.concatenate([self.pos_idx, neg_sample])
        self.rng.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self):
        return len(self.pos_idx) * (1 + self.neg_ratio)


def train(
    annotations_csv: str,
    pair_labels_parquet: str,
    embeddings_path: str,
    output_dir: str = './results/mice_behavior',
    train_obs_ids=None,
    val_obs_ids=None,
    context_k: int = 2,
    emb_dim: int = 768,
    n_heads: int = 8,
    hidden_dim: int = 256,
    n_epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    neg_ratio: int = 1,
    device: str = 'cuda',
    seed: int = 42,
):
    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Building train dataset...')
    train_ds = MousePairDataset(
        annotations_csv, pair_labels_parquet, embeddings_path,
        obs_ids=train_obs_ids, context_k=context_k, emb_dim=emb_dim,
    )
    labels = train_ds.samples[:, 3]
    sampler = DynamicNegativeSampler(labels, neg_ratio=neg_ratio, seed=seed)
    n_pos = (labels > 0).sum()
    print(f'  DynamicNegativeSampler: {n_pos:,} pos + {neg_ratio}×{n_pos:,} neg per epoch ({len(sampler):,} samples/epoch)')
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    val_loader = None
    if val_obs_ids:
        print('Building val dataset...')
        val_ds = MousePairDataset(
            annotations_csv, pair_labels_parquet, embeddings_path,
            obs_ids=val_obs_ids, context_k=context_k, emb_dim=emb_dim,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    model = MouseBehaviorClassifier(emb_dim=emb_dim, n_heads=n_heads, hidden_dim=hidden_dim).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # class weights from the sampled distribution (not the full dataset)
    # neg_ratio negatives are drawn per positive, so none count ≈ n_pos * neg_ratio
    n_pos_total = int((labels > 0).sum())
    sampled_counts = np.bincount(labels, minlength=3).clip(1).astype(np.float32)
    sampled_counts[0] = n_pos_total * neg_ratio
    class_weights = torch.tensor(sampled_counts.sum() / (3 * sampled_counts), dtype=torch.float32).to(dev)
    print(f'  class weights: none={class_weights[0]:.2f}  nt={class_weights[1]:.2f}  nn={class_weights[2]:.2f}')
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_bal_acc = -1.0
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        t0 = time.time()
        for ctx, a1, a2, labels_b, mask in train_loader:
            ctx, a1, a2, labels_b, mask = ctx.to(dev), a1.to(dev), a2.to(dev), labels_b.to(dev), mask.to(dev)
            optimizer.zero_grad()
            logits = model(ctx, a1, a2, key_padding_mask=mask)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels_b)
            correct += (logits.argmax(1) == labels_b).sum().item()
            n += len(labels_b)

        msg = f'epoch {epoch:3d}/{n_epochs}  loss={total_loss/n:.4f}  train_acc={correct/n:.4f}  ({time.time()-t0:.1f}s)'

        if val_loader is not None:
            val_acc, bal_acc, per_class = _evaluate(model, val_loader, dev)
            msg += f'  val_acc={val_acc:.4f}  bal_acc={bal_acc:.4f}  ' + '  '.join(f'{k}={v}' for k, v in per_class.items())
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                torch.save(model.state_dict(), output_dir / 'best_model.pt')

        print(msg)

    if val_loader is None:
        torch.save(model.state_dict(), output_dir / 'model.pt')

    return model


def _evaluate(model, loader, dev):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ctx, a1, a2, labels, mask in loader:
            logits = model(ctx.to(dev), a1.to(dev), a2.to(dev), key_padding_mask=mask.to(dev))
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(labels)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = (preds == labels).float().mean().item()
    per_class = {}
    recalls = []
    for c in range(3):
        tp = ((preds == c) & (labels == c)).sum().item()
        recall = tp / max((labels == c).sum().item(), 1)
        precision = tp / max((preds == c).sum().item(), 1)
        per_class[LABEL_NAMES[c]] = f'R={recall:.3f}/P={precision:.3f}'
        recalls.append(recall)
    bal_acc = float(np.mean(recalls))
    return acc, bal_acc, per_class
