"""Qualitative error figures for the per-frame behavior classifier.

Why this is not error_overview_frame.py
---------------------------------------
That script does the same 4-bucket layout but is bound to the old CLS-embedding pipeline
and, more importantly, thresholds at p>0.5. That rule is wrong for these models: training
undersamples negatives to 1:1, so predictions carry a large prior shift (measured
calibration slopes of 0.06-0.12, i.e. mean predicted rate overstates the true rate by
8-17x). A 0.5 cut therefore does not correspond to any sensible operating point, and the
resulting FN row is dominated by frames the model actually ranked highly.

Here the threshold is chosen per behavior as the one maximising F1 on the evaluated split,
and it is printed on the figure along with the precision/recall it implies -- so the picture
is tied to a stated operating point rather than an arbitrary one. This is also the only
place in the pipeline that surfaces precision/recall at all; every other metric we track
(AP, ROC-AUC, rate correlation) is threshold-free.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve

BUCKETS = ('TP', 'FP', 'FN', 'TN')
BUCKET_COLOR = {'TP': '#2e7d32', 'FP': '#c62828', 'FN': '#ef6c00', 'TN': '#455a64'}


def best_f1_threshold(y, p):
    """Threshold maximising F1. Returns (threshold, precision, recall, f1)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    i = int(np.argmax(f1[:-1])) if len(thr) else 0
    if not len(thr):
        return 0.5, 0.0, 0.0, 0.0
    return float(thr[i]), float(prec[i]), float(rec[i]), float(f1[i])


def _load_frame(gi, frame_paths, jpeg_cache, dataset_root):
    """Prefer the in-RAM JPEG cache the caller already populated; fall back to disk."""
    import io
    if jpeg_cache is not None and int(gi) in jpeg_cache:
        buf = jpeg_cache[int(gi)]
        return Image.open(io.BytesIO(buf.tobytes() if hasattr(buf, 'tobytes') else bytes(buf))).convert('L')
    return Image.open(Path(dataset_root) / frame_paths[int(gi)]).convert('L')


def plot_confusion_examples(probs, labels, sample_obs, frame_gi, frame_paths, out_dir,
                            label_names=('nt', 'nn'), n_examples=8, seed=42,
                            jpeg_cache=None, dataset_root='dataset', title_prefix=''):
    """One figure per behavior: rows TP/FP/FN/TN, columns are random examples from that bucket.

    Examples are drawn at RANDOM within a bucket (not by extreme confidence) so each row is
    representative of that error mode rather than of its tail, then sorted by probability so
    the row reads left-to-right. Each panel is captioned with the model's probability, and
    the row label carries the bucket's absolute count and share of the split.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = {}

    for li, name in enumerate(label_names):
        y, p = labels[:, li].astype(int), probs[:, li]
        thr, prec, rec, f1 = best_f1_threshold(y, p)
        pred = p >= thr
        idx = {'TP': np.where((y == 1) & pred)[0], 'FP': np.where((y == 0) & pred)[0],
               'FN': np.where((y == 1) & ~pred)[0], 'TN': np.where((y == 0) & ~pred)[0]}

        fig, axes = plt.subplots(len(BUCKETS), n_examples,
                                 figsize=(1.65 * n_examples, 1.95 * len(BUCKETS)))
        axes = np.atleast_2d(axes)
        for r, b in enumerate(BUCKETS):
            pool = idx[b]
            pick = rng.choice(pool, size=min(n_examples, len(pool)), replace=False) if len(pool) else []
            pick = sorted(pick, key=lambda i: -p[i])
            for c in range(n_examples):
                ax = axes[r, c]
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(BUCKET_COLOR[b]); sp.set_linewidth(2)
                if c >= len(pick):
                    ax.set_facecolor('#f5f5f5')
                    continue
                i = pick[c]
                try:
                    ax.imshow(_load_frame(frame_gi[i], frame_paths, jpeg_cache, dataset_root),
                              cmap='gray', vmin=0, vmax=255)
                except Exception as e:                       # a figure must never kill a run
                    ax.text(.5, .5, f'{e.__class__.__name__}', ha='center', va='center', fontsize=6)
                ax.set_title(f'p={p[i]:.3f}', fontsize=7, color=BUCKET_COLOR[b], pad=2)
                ax.set_xlabel(f'{str(sample_obs[i])[:18]}', fontsize=5, labelpad=1)
            share = 100 * len(pool) / max(len(y), 1)
            axes[r, 0].set_ylabel(f'{b}\nn={len(pool):,}\n{share:.2f}%', fontsize=8,
                                  color=BUCKET_COLOR[b], rotation=0, ha='right', va='center',
                                  labelpad=26)
        fig.suptitle(
            f'{title_prefix}{name}  —  operating point: threshold {thr:.3f} (max-F1)  '
            f'precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}\n'
            f'prevalence {100*y.mean():.2f}%  ({int(y.sum()):,} positives of {len(y):,} val frames)',
            fontsize=10)
        fig.tight_layout(rect=[0.02, 0, 1, 0.93])
        f = out_dir / f'confusion_examples_{name}.png'
        fig.savefig(f, dpi=130)
        plt.close(fig)
        written[name] = f
        print(f'  [viz] {name}: thr={thr:.3f} P={prec:.3f} R={rec:.3f} F1={f1:.3f} -> {f}',
              flush=True)
    return written
