import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import MousePairDataset, collate_fn
from .model import MouseBehaviorClassifier

LABEL_NAMES = ['none', 'nt', 'nn']


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
    n_epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
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
    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
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
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        t0 = time.time()
        for ctx, a1, a2, labels, mask in train_loader:
            ctx, a1, a2, labels, mask = ctx.to(dev), a1.to(dev), a2.to(dev), labels.to(dev), mask.to(dev)
            optimizer.zero_grad()
            logits = model(ctx, a1, a2, key_padding_mask=mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)

        msg = f'epoch {epoch:3d}/{n_epochs}  loss={total_loss/n:.4f}  train_acc={correct/n:.4f}  ({time.time()-t0:.1f}s)'

        if val_loader is not None:
            val_acc, per_class = _evaluate(model, val_loader, dev)
            msg += f'  val_acc={val_acc:.4f}  ' + '  '.join(f'{k}={v:.3f}' for k, v in per_class.items())
            if val_acc > best_val_acc:
                best_val_acc = val_acc
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
    per_class = {
        LABEL_NAMES[c]: (
            ((preds == c) & (labels == c)).sum().item()
            / max((labels == c).sum().item(), 1)
        )
        for c in range(3)
    }
    return acc, per_class
