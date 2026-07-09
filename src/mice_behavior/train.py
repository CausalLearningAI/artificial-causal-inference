import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Sampler

from .dataset import MousePairDataset, MousePairDatasetPatchGrid, collate_fn
from .model import MouseBehaviorClassifier

LABEL_NAMES = ['none', 'nt', 'nn']


class FocalLoss(nn.Module):
    """CE with a (1-p_t)^gamma modulating factor that downweights easy examples,
    on top of the same static class weights used for plain CrossEntropyLoss."""

    def __init__(self, weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        w_t = self.weight[target]
        return (-w_t * (1 - p_t).pow(self.gamma) * logp_t).mean()


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
    n_heads: int = 1,
    hidden_dim: int = 256,
    n_epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    neg_ratio: int = 1,
    device: str = 'cuda',
    seed: int = 42,
    loss_type: str = 'ce',
    focal_gamma: float = 2.0,
    verbose: bool = True,
    use_patch_grid: bool = False,
    patch_embeddings_path: str = None,
    patch_global_idx_path: str = None,
    n_patches: int = 16,
    eval_every: int = 1,
):
    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Building train dataset...')
    if use_patch_grid:
        train_ds = MousePairDatasetPatchGrid(
            annotations_csv, pair_labels_parquet, cls_embeddings_path=embeddings_path,
            embeddings_path=patch_embeddings_path, global_idx_path=patch_global_idx_path,
            obs_ids=train_obs_ids, context_k=context_k, emb_dim=emb_dim, n_patches=n_patches,
        )
    else:
        train_ds = MousePairDataset(
            annotations_csv, pair_labels_parquet, embeddings_path,
            obs_ids=train_obs_ids, context_k=context_k, emb_dim=emb_dim,
        )
    labels = train_ds.samples[:, 3]
    sampler = DynamicNegativeSampler(labels, neg_ratio=neg_ratio, seed=seed)
    n_pos = (labels > 0).sum()
    print(f'  DynamicNegativeSampler: {n_pos:,} pos + {neg_ratio}×{n_pos:,} neg per epoch ({len(sampler):,} samples/epoch)')
    # num_workers=0: __getitem__ only slices already-RAM-preloaded numpy arrays (no disk I/O
    # to overlap with GPU compute), so multi-process workers add nothing — and each worker's
    # copy-on-write fork of the preloaded dataset (up to ~19GB for the patch-grid variant) risks
    # ballooning past the job's memory allocation as pages get touched, which is what caused
    # multi-minute inter-epoch stalls even with persistent_workers=True.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_obs_ids:
        print('Building val dataset...')
        if use_patch_grid:
            val_ds = MousePairDatasetPatchGrid(
                annotations_csv, pair_labels_parquet, cls_embeddings_path=embeddings_path,
                embeddings_path=patch_embeddings_path, global_idx_path=patch_global_idx_path,
                obs_ids=val_obs_ids, context_k=context_k, emb_dim=emb_dim, n_patches=n_patches,
            )
        else:
            val_ds = MousePairDataset(
                annotations_csv, pair_labels_parquet, embeddings_path,
                obs_ids=val_obs_ids, context_k=context_k, emb_dim=emb_dim,
            )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
            collate_fn=collate_fn,
        )

    model = MouseBehaviorClassifier(
        emb_dim=emb_dim, n_heads=n_heads, hidden_dim=hidden_dim, use_patch_grid=use_patch_grid,
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # class weights from the sampled distribution (not the full dataset)
    # neg_ratio negatives are drawn per positive, so none count ≈ n_pos * neg_ratio
    n_pos_total = int((labels > 0).sum())
    sampled_counts = np.bincount(labels, minlength=3).clip(1).astype(np.float32)
    sampled_counts[0] = n_pos_total * neg_ratio
    class_weights = torch.tensor(sampled_counts.sum() / (3 * sampled_counts), dtype=torch.float32).to(dev)
    if verbose:
        print(f'  class weights: none={class_weights[0]:.2f}  nt={class_weights[1]:.2f}  nn={class_weights[2]:.2f}')
    if loss_type == 'focal':
        criterion = FocalLoss(class_weights, gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_pr_auc = -1.0
    best_per_class = {}
    history = {'epoch': [], 'train_loss': [], 'eval_epoch': [], 'val_loss': [], 'val_acc': [], 'macro_pr_auc': []}
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        t0 = time.time()
        for ctx, offsets, a1, a2, labels_b, mask in train_loader:
            ctx, offsets, a1, a2, labels_b, mask = (
                ctx.to(dev), offsets.to(dev), a1.to(dev), a2.to(dev), labels_b.to(dev), mask.to(dev)
            )
            optimizer.zero_grad()
            logits = model(ctx, a1, a2, offsets=offsets, key_padding_mask=mask)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels_b)
            correct += (logits.argmax(1) == labels_b).sum().item()
            n += len(labels_b)

        train_loss = total_loss / n
        msg = f'epoch {epoch:3d}/{n_epochs}  loss={train_loss:.4f}  train_acc={correct/n:.4f}  ({time.time()-t0:.1f}s)'

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)

        do_eval = val_loader is not None and (epoch % eval_every == 0 or epoch == n_epochs)
        if do_eval:
            val_acc, pr_auc, per_class, val_loss = _evaluate(model, val_loader, dev, criterion)
            msg += f'  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  macro_pr_auc={pr_auc:.4f}  ' + '  '.join(f'{k}={v}' for k, v in per_class.items())
            history['eval_epoch'].append(epoch)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['macro_pr_auc'].append(pr_auc)
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_per_class = per_class
                torch.save(model.state_dict(), output_dir / 'best_model.pt')

        if verbose:
            print(msg)

    if val_loader is None:
        torch.save(model.state_dict(), output_dir / 'model.pt')

    import json
    with open(output_dir / 'loss_history.json', 'w') as f:
        json.dump(history, f)

    return {'model': model, 'best_pr_auc': best_pr_auc, 'best_per_class': best_per_class, 'history': history}


def _evaluate(model, loader, dev, criterion=None):
    """Threshold-free model selection: macro PR-AUC (average precision) per class,
    since argmax-based accuracy depends on the arbitrary decision boundary induced
    by the reweighted loss rather than the model's actual ranking quality."""
    model.eval()
    all_probs, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for ctx, offsets, a1, a2, labels, mask in loader:
            labels_dev = labels.to(dev)
            logits = model(ctx.to(dev), a1.to(dev), a2.to(dev), offsets=offsets.to(dev), key_padding_mask=mask.to(dev))
            all_probs.append(torch.softmax(logits, dim=1).cpu())
            all_labels.append(labels)
            if criterion is not None:
                all_logits.append((logits, labels_dev))
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = (probs.argmax(1) == labels).mean()
    per_class, pr_aucs = {}, []
    for c in range(3):
        y_true = (labels == c).astype(int)
        pr = average_precision_score(y_true, probs[:, c])
        per_class[LABEL_NAMES[c]] = f'PR-AUC={pr:.3f}'
        pr_aucs.append(pr)
    macro_pr_auc = float(np.mean(pr_aucs))
    if criterion is not None:
        n_total = sum(l.size(0) for _, l in all_logits)
        val_loss = sum(criterion(lg, lb).item() * lg.size(0) for lg, lb in all_logits) / n_total
        return acc, macro_pr_auc, per_class, val_loss
    return acc, macro_pr_auc, per_class
