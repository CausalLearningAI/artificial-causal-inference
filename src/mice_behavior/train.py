import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

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


def train(
    annotations_csv: str,
    pair_labels_parquet: str,
    embeddings_path: str,
    output_dir: str = './results/mice_behavior',
    train_obs_ids=None,
    val_obs_ids=None,
    context_k: int = 2,
    stride: int = 1,
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
    # Checkpoint selection / early stopping compare a trailing moving average of
    # pair_macro_pr_auc, not the raw per-eval value — with only ~2.5-4k positive pair-samples,
    # the subsampled per-epoch metric is noisy enough that the single highest of ~80 evals is
    # usually just a lucky fluctuation (confirmed: a promoted model's "best epoch" turned out
    # to be an isolated spike in an otherwise-flat/noisy trajectory, not a real improvement).
    smooth_window: int = 5,
    # Bounds how many distinct observations' frames ever get loaded/GPU-resident for training
    # — patch-grid's full train split is ~15GB (16x CLS's per-frame footprint), which never
    # fits on a GPU with limited VRAM, and every epoch only samples a small fraction of frames
    # anyway via neg_ratio subsampling. None = unrestricted (CLS's default; it already fits in full).
    max_train_frames: int = None,
):
    """Vectorized training loop — builds batches directly from FastBatchData
    instead of going through Dataset/DataLoader — see fast_data.py."""
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
    train_data = FastBatchData(
        annotations_csv, pair_labels_parquet, train_obs_ids, context_k, emb_dim, load_fn, n_patches,
        stride=stride, max_frames=max_train_frames, seed=seed,
    )
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
        val_data = FastBatchData(annotations_csv, pair_labels_parquet, val_obs_ids, context_k, emb_dim, load_fn, n_patches, stride=stride)
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

    # Random (class-prior) baselines for train_loss/val_loss, computed on the SAME populations
    # and weights the criterion actually sees each epoch — NOT the true full-dataset
    # proportions, since both train and val are evaluated on reweighted subsamples (train:
    # neg_ratio, val: val_neg_ratio, a different ratio), and class_weights is calibrated only
    # once from train's own composition then reused as-is for val. Train's population exactly
    # matches the calibration population, so weighted-mean reduces to a clean mean-of-logs;
    # val's doesn't (different ratio), so it needs the general weighted formula.
    train_prior = sampled_counts / sampled_counts.sum()
    random_train_loss = float(-np.mean(np.log(np.clip(train_prior, 1e-12, None))))
    random_val_loss = pair_random_val = frame_random_val = None
    if val_data is not None:
        val_labels_kept = val_data.labels[val_keep]
        n_val_counts = np.array([(val_labels_kept == c).sum() for c in range(3)], dtype=np.float64)
        p_val = n_val_counts / n_val_counts.sum()
        w = class_weights.detach().cpu().numpy().astype(np.float64)
        random_val_loss = float(
            np.sum(n_val_counts * w * -np.log(np.clip(p_val, 1e-12, None))) / np.sum(n_val_counts * w)
        )
        # Random PR-AUC baselines for the SAME val_keep subsample the per-epoch curve below is
        # computed on (AP of an uninformative classifier converges to the positive prevalence
        # of whatever set it's evaluated on — val_keep's prevalence is much higher than the
        # true full-val prevalence report.py uses for its own, separately-computed baselines).
        pair_random_val = float(np.mean([(val_labels_kept == c).mean() for c in (1, 2)]))
        labels_r_kept = val_labels_kept.reshape(n_kept_frames, 12)
        frame_random_val = float(np.mean([(labels_r_kept == c).any(axis=1).mean() for c in (1, 2)]))

    best_pr_auc = -1.0
    best_per_class = {}
    best_epoch = 0
    epochs_since_best = 0
    history = {'epoch': [], 'train_loss': [], 'eval_epoch': [], 'val_loss': [], 'val_acc': [],
               'pair_macro_pr_auc': [], 'frame_macro_pr_auc': [],
               'random_train_loss': random_train_loss, 'random_val_loss': random_val_loss,
               'pair_random_val': pair_random_val, 'frame_random_val': frame_random_val}

    # Mixed precision: this workload was confirmed GPU-compute-bound (83% utilization on a
    # that GPU), so using tensor cores via autocast is the next lever, not more data-pipeline
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

            # Trailing moving average over the last `smooth_window` evals, not the raw value —
            # see smooth_window's docstring above for why (noisy single-epoch spikes otherwise
            # win "best checkpoint" purely by chance).
            recent = history['pair_macro_pr_auc'][-smooth_window:]
            smoothed_pr_auc = float(np.mean(recent))
            msg += f'  smoothed_pr_auc={smoothed_pr_auc:.4f}'

            if smoothed_pr_auc > best_pr_auc:
                best_pr_auc = smoothed_pr_auc
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
