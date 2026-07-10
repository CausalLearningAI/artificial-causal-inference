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


def _subsample_val(val_ds, neg_ratio, seed=42):
    """Fixed (seeded, one-time) downsample of the validation set's 'none' negatives.
    Unlike DynamicNegativeSampler this is NOT redrawn every epoch — the same subsample
    is reused across all epochs so the val metric stays comparable epoch-to-epoch. Val
    doesn't need the full unbalanced population to give a representative macro PR-AUC,
    and with eval_every=1 the full val set (6x more batches than the sampled train set)
    was the dominant cost of every epoch."""
    labels = val_ds.samples[:, 3]
    pos_idx = np.where(labels > 0)[0]
    neg_idx = np.where(labels == 0)[0]
    rng = np.random.default_rng(seed)
    n_neg = min(len(neg_idx), neg_ratio * max(len(pos_idx), 1))
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.sort(np.concatenate([pos_idx, neg_sample]))
    val_ds.samples = val_ds.samples[keep]
    return val_ds


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
    batch_size: int = 4096,
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
    val_neg_ratio: int = 20,
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
    # Worker processes parallelize __getitem__'s Python-level work across separate
    # interpreters. Each worker forks a copy-on-write view of the preloaded dataset, so
    # keep the count modest and the job's --mem generous relative to dataset size.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=6, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=True, prefetch_factor=2,
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
        if val_neg_ratio is not None:
            n_before = len(val_ds)
            val_ds = _subsample_val(val_ds, neg_ratio=val_neg_ratio, seed=seed)
            print(f'  fixed val subsample: {n_before:,} -> {len(val_ds):,} samples '
                  f'(all positives + {val_neg_ratio}x negatives, seeded — same subset every epoch)')
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True,
            collate_fn=collate_fn, persistent_workers=True, prefetch_factor=2,
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
        # Accumulate on-GPU, sync via .item() once per epoch — a per-batch .item()/.cpu()
        # call forces a GPU<->CPU sync, serializing otherwise-async GPU work.
        total_loss = torch.zeros((), device=dev)
        correct = torch.zeros((), device=dev)
        n = 0
        t0 = time.time()
        for ctx, offsets, a1, a2, labels_b, mask in train_loader:
            ctx, offsets, a1, a2, labels_b, mask = (
                ctx.to(dev, non_blocking=True), offsets.to(dev, non_blocking=True),
                a1.to(dev, non_blocking=True), a2.to(dev, non_blocking=True),
                labels_b.to(dev, non_blocking=True), mask.to(dev, non_blocking=True),
            )
            optimizer.zero_grad()
            logits = model(ctx, a1, a2, offsets=offsets, key_padding_mask=mask)
            loss = criterion(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            with torch.no_grad():
                total_loss += loss.detach() * labels_b.size(0)
                correct += (logits.argmax(1) == labels_b).sum()
            n += labels_b.size(0)

        train_loss = (total_loss / n).item()
        train_acc = (correct / n).item()
        msg = f'epoch {epoch:3d}/{n_epochs}  loss={train_loss:.4f}  train_acc={train_acc:.4f}  ({time.time()-t0:.1f}s)'

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


def train_fast(
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
    batch_size: int = 4096,
    lr: float = 1e-3,
    neg_ratio: int = 10,
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
    # Applies per FRAME now (all 12 pairs kept together), not per individual pair-sample as
    # before — the same numeric ratio would now pull in ~93% of the full val set every epoch
    # (most val frames have no positive pair, so 20x-the-positive-frame-count exceeds the
    # negative-frame pool almost entirely). 2 keeps per-epoch eval fast while still giving a
    # frame-complete sample for the aggregated-per-frame metric.
    val_neg_ratio: int = 2,
    grad_clip: float = 0.5,
    use_amp: bool = True,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 15,
):
    """Same training logic as train(), but uses FastBatchData's vectorized batch
    construction instead of Dataset/DataLoader — see fast_data.py."""
    import json
    from .fast_data import FastBatchData, load_cls_embeddings, load_patchgrid_embeddings

    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_fn = (
        load_patchgrid_embeddings(patch_embeddings_path, patch_global_idx_path, n_patches, emb_dim)
        if use_patch_grid else load_cls_embeddings(embeddings_path, emb_dim)
    )

    print('Building train dataset (vectorized)...')
    train_data = FastBatchData(annotations_csv, pair_labels_parquet, train_obs_ids, context_k, emb_dim, load_fn, n_patches)
    labels = train_data.labels
    pos_idx = np.where(labels > 0)[0]
    neg_idx = np.where(labels == 0)[0]
    rng = np.random.default_rng(seed)

    # Move the whole flat embedding array onto GPU once, eliminating repeated host->device
    # transfers during training. Falls back to CPU/numpy gather if it doesn't fit in VRAM.
    if dev.type == 'cuda':
        try:
            train_data.to_device(dev)
            print(f'  train data resident on GPU ({train_data.flat.nbytes/1e9:.1f} GB)')
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print('  train data too large for GPU memory, falling back to CPU gather')

    val_data, val_keep = None, None
    if val_obs_ids:
        print('Building val dataset (vectorized)...')
        val_data = FastBatchData(annotations_csv, pair_labels_parquet, val_obs_ids, context_k, emb_dim, load_fn, n_patches)
        if dev.type == 'cuda':
            try:
                val_data.to_device(dev)
                print(f'  val data resident on GPU ({val_data.flat.nbytes/1e9:.1f} GB)')
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print('  val data too large for GPU memory, falling back to CPU gather')
        n_kept_frames = None
        if val_neg_ratio is not None:
            # Subsample whole frames (all 12 ordered pairs kept together), not individual
            # pair-samples — this keeps every kept frame's 12-pair block complete, which the
            # per-epoch "aggregated per frame" PR-AUC below needs (it reshapes into
            # (n_frames, 12, 3), same as report.py's final evaluation).
            v_labels = val_data.labels
            n_frames_total = len(v_labels) // 12
            frame_labels = v_labels[:n_frames_total * 12].reshape(n_frames_total, 12)
            frame_has_pos = (frame_labels > 0).any(axis=1)
            pos_frames = np.where(frame_has_pos)[0]
            neg_frames = np.where(~frame_has_pos)[0]
            v_rng = np.random.default_rng(seed)
            n_neg_frames = min(len(neg_frames), val_neg_ratio * max(len(pos_frames), 1))
            neg_frame_sample = v_rng.choice(neg_frames, size=n_neg_frames, replace=False)
            keep_frames = np.sort(np.concatenate([pos_frames, neg_frame_sample]))
            val_keep = (keep_frames[:, None] * 12 + np.arange(12)[None, :]).reshape(-1)
            n_kept_frames = len(keep_frames)
            print(f'  fixed val subsample: {n_frames_total:,} -> {n_kept_frames:,} frames ({len(val_keep):,} samples)')
        else:
            val_keep = np.arange(len(val_data))
            n_kept_frames = len(val_data) // 12

    model = MouseBehaviorClassifier(
        emb_dim=emb_dim, n_heads=n_heads, hidden_dim=hidden_dim, use_patch_grid=use_patch_grid, dropout=dropout,
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_pos_total = max(len(pos_idx), 1)
    pos_labels = labels[pos_idx]
    n1 = max((pos_labels == 1).sum(), 1)
    n2 = max((pos_labels == 2).sum(), 1)
    n0 = max(neg_ratio * n_pos_total, 1)
    sampled_counts = np.array([n0, n1, n2], dtype=np.float32)
    class_weights = torch.tensor(sampled_counts.sum() / (3 * sampled_counts), dtype=torch.float32).to(dev)
    if verbose:
        print(f'  class weights: none={class_weights[0]:.2f}  nt={class_weights[1]:.2f}  nn={class_weights[2]:.2f}')
    criterion = FocalLoss(class_weights, gamma=focal_gamma) if loss_type == 'focal' else nn.CrossEntropyLoss(weight=class_weights)

    best_pr_auc = -1.0
    best_per_class = {}
    best_epoch = 0
    epochs_since_best = 0
    history = {'epoch': [], 'train_loss': [], 'eval_epoch': [], 'val_loss': [], 'val_acc': [],
               'pair_macro_pr_auc': [], 'frame_macro_pr_auc': []}

    # Mixed precision: this workload was confirmed GPU-compute-bound (83% utilization on a
    # 2080ti), so using tensor cores via autocast is the next lever, not more data-pipeline
    # changes. GradScaler keeps backward numerically stable under fp16; gradients are
    # unscaled before clipping so max_norm means the same thing as without AMP.
    amp_enabled = use_amp and dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    for epoch in range(1, n_epochs + 1):
        model.train()
        n_neg_draw = min(len(neg_idx), neg_ratio * len(pos_idx))
        neg_sample = rng.choice(neg_idx, size=n_neg_draw, replace=False)
        epoch_idx = np.concatenate([pos_idx, neg_sample])
        rng.shuffle(epoch_idx)

        total_loss = torch.zeros((), device=dev)
        correct = torch.zeros((), device=dev)
        n = 0
        t0 = time.time()
        for b0 in range(0, len(epoch_idx), batch_size):
            batch_idx = epoch_idx[b0:b0 + batch_size]
            ctx, offs, a1, a2, lbl, mask = train_data.get_batch(batch_idx)
            # .float() after the transfer, not before — upcasting pre-transfer would double
            # the host->device payload (GPU-resident path already returns fp32, so this is
            # then a no-op there).
            ctx, offs, a1, a2, lbl, mask = (
                ctx.to(dev, non_blocking=True).float(), offs.to(dev, non_blocking=True),
                a1.to(dev, non_blocking=True), a2.to(dev, non_blocking=True),
                lbl.to(dev, non_blocking=True), mask.to(dev, non_blocking=True),
            )
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                logits = model(ctx, a1, a2, offsets=offs, key_padding_mask=mask)
                loss = criterion(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                total_loss += loss.detach() * lbl.size(0)
                correct += (logits.argmax(1) == lbl).sum()
            n += lbl.size(0)

        train_loss = (total_loss / n).item()
        train_acc = (correct / n).item()
        msg = f'epoch {epoch:3d}/{n_epochs}  loss={train_loss:.4f}  train_acc={train_acc:.4f}  ({time.time()-t0:.1f}s)'
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)

        do_eval = val_data is not None and (epoch % eval_every == 0 or epoch == n_epochs)
        if do_eval:
            model.eval()
            all_probs, all_labels = [], []
            loss_sum = torch.zeros((), device=dev)
            n_total = 0
            with torch.no_grad():
                for b0 in range(0, len(val_keep), batch_size):
                    batch_idx = val_keep[b0:b0 + batch_size]
                    ctx, offs, a1, a2, lbl, mask = val_data.get_batch(batch_idx)
                    ctx, offs, a1, a2, lbl, mask = (
                        ctx.to(dev, non_blocking=True).float(), offs.to(dev, non_blocking=True),
                        a1.to(dev, non_blocking=True), a2.to(dev, non_blocking=True),
                        lbl.to(dev, non_blocking=True), mask.to(dev, non_blocking=True),
                    )
                    with torch.amp.autocast('cuda', enabled=amp_enabled):
                        logits = model(ctx, a1, a2, offsets=offs, key_padding_mask=mask)
                        batch_loss = criterion(logits, lbl)
                    all_probs.append(torch.softmax(logits, dim=1).float())
                    all_labels.append(lbl)
                    loss_sum += batch_loss.float() * logits.size(0)
                    n_total += logits.size(0)
            probs = torch.cat(all_probs).cpu().numpy()
            labels_np = torch.cat(all_labels).cpu().numpy()
            acc = float((probs.argmax(1) == labels_np).mean())

            # 'none' isn't a behavior — it's the negative default for a pair, and its own
            # per-pair AP is trivially near-1.0 given it's ~99.7% of samples. Macro PR-AUC
            # (the model-selection / early-stopping criterion) is the mean over the two real
            # behaviors, nt and nn, only.
            per_class = {}
            for c in range(3):
                y_true = (labels_np == c).astype(int)
                pr = average_precision_score(y_true, probs[:, c])
                per_class[LABEL_NAMES[c]] = f'PR-AUC={pr:.3f}'
            pair_pr_auc = {c: average_precision_score((labels_np == c).astype(int), probs[:, c]) for c in (1, 2)}
            pair_macro_pr_auc = float(np.mean(list(pair_pr_auc.values())))

            probs_r = probs.reshape(n_kept_frames, 12, 3)
            labels_r = labels_np.reshape(n_kept_frames, 12)
            frame_pr_auc = {
                c: average_precision_score((labels_r == c).any(axis=1).astype(int), probs_r[:, :, c].max(axis=1))
                for c in (1, 2)
            }
            frame_macro_pr_auc = float(np.mean(list(frame_pr_auc.values())))

            val_loss = (loss_sum / n_total).item()
            msg += (f'  val_loss={val_loss:.4f}  val_acc={acc:.4f}  pair_macro_pr_auc={pair_macro_pr_auc:.4f}'
                    f'  frame_macro_pr_auc={frame_macro_pr_auc:.4f}  ' + '  '.join(f'{k}={v}' for k, v in per_class.items()))
            history['eval_epoch'].append(epoch)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(acc)
            history['pair_macro_pr_auc'].append(pair_macro_pr_auc)
            history['frame_macro_pr_auc'].append(frame_macro_pr_auc)
            if pair_macro_pr_auc > best_pr_auc:
                best_pr_auc = pair_macro_pr_auc
                best_per_class = per_class
                best_epoch = epoch
                epochs_since_best = 0
                torch.save(model.state_dict(), output_dir / 'best_model.pt')
            else:
                epochs_since_best += eval_every

        if verbose:
            print(msg)

        # This model overfits fast (best macro PR-AUC is typically reached within the first
        # ~15-20 epochs, then val_loss climbs monotonically while train_loss keeps dropping —
        # confirmed on both CLS and patch-grid full 100-epoch runs). Stopping once the best
        # checkpoint hasn't improved for early_stop_patience epochs avoids wasting the rest of
        # the budget purely overfitting, and is a straightforward speed win on top of it.
        if early_stop_patience is not None and do_eval and epochs_since_best >= early_stop_patience:
            if verbose:
                print(f'  early stopping: no improvement for {epochs_since_best} epochs (best was epoch {best_epoch})')
            break

    if val_data is None:
        torch.save(model.state_dict(), output_dir / 'model.pt')

    return {'model': model, 'best_pr_auc': best_pr_auc, 'best_per_class': best_per_class, 'history': history}


def _evaluate(model, loader, dev, criterion=None):
    """Threshold-free model selection: macro PR-AUC (average precision) per class,
    since argmax-based accuracy depends on the arbitrary decision boundary induced
    by the reweighted loss rather than the model's actual ranking quality."""
    model.eval()
    # Keep everything on-GPU through the loop — one .cpu() transfer at the end instead of
    # one per batch, avoiding a per-batch GPU<->CPU sync.
    all_probs, all_labels = [], []
    loss_sum = torch.zeros((), device=dev)
    n_total = 0
    with torch.no_grad():
        for ctx, offsets, a1, a2, labels, mask in loader:
            labels_dev = labels.to(dev, non_blocking=True)
            logits = model(
                ctx.to(dev, non_blocking=True), a1.to(dev, non_blocking=True), a2.to(dev, non_blocking=True),
                offsets=offsets.to(dev, non_blocking=True), key_padding_mask=mask.to(dev, non_blocking=True),
            )
            all_probs.append(torch.softmax(logits, dim=1))
            all_labels.append(labels_dev)
            if criterion is not None:
                loss_sum += criterion(logits, labels_dev) * logits.size(0)
            n_total += logits.size(0)
    probs = torch.cat(all_probs).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    acc = (probs.argmax(1) == labels).mean()
    per_class, pr_aucs = {}, []
    for c in range(3):
        y_true = (labels == c).astype(int)
        pr = average_precision_score(y_true, probs[:, c])
        per_class[LABEL_NAMES[c]] = f'PR-AUC={pr:.3f}'
        pr_aucs.append(pr)
    macro_pr_auc = float(np.mean(pr_aucs))
    if criterion is not None:
        val_loss = (loss_sum / n_total).item()
        return acc, macro_pr_auc, per_class, val_loss
    return acc, macro_pr_auc, per_class
