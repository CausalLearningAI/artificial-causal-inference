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


def _pick_confident(pool, p, sample_obs, n, most_confident_high):
    """The n most confident frames in a bucket, at most one per observation.

    Confidence means distance from the decision, not distance from 0.5: for TP and FP the
    model's conviction rises with p, for FN and TN it rises as p falls. So each bucket is
    sorted in the direction that puts its own strongest case first -- the TPs it was surest
    about, and the FPs it was surest about and still got wrong.

    The one-per-observation rule is what makes the row informative. Positives arrive in
    contiguous bouts (22% of nt bouts and 38% of nn bouts are a single frame, but the rest run
    consecutively), and adjacent frames of one bout score almost identically, so an unfiltered
    top-n is typically eight views of the same two seconds of one video. Capping at one frame
    per observation turns the row into eight independent cases and makes a failure mode that
    recurs across videos visually distinguishable from one video going wrong.
    """
    if not len(pool):
        return []
    order = pool[np.argsort(-p[pool] if most_confident_high else p[pool])]
    seen, out = set(), []
    for i in order:
        o = str(sample_obs[i])
        if o in seen:
            continue
        seen.add(o); out.append(i)
        if len(out) == n:
            break
    # fewer distinct observations than columns: top up with the next most confident frames
    # rather than leaving the row short, since a short row reads as "no such errors exist".
    if len(out) < n:
        out += [i for i in order if i not in set(out)][:n - len(out)]
    return out


def plot_confusion_examples(probs, labels, sample_obs, frame_gi, frame_paths, out_dir,
                            label_names=('nt', 'nn'), n_examples=8, seed=42,
                            jpeg_cache=None, dataset_root='dataset', title_prefix=''):
    """One figure per behavior: rows TP/FP/FN/TN, columns the most confident cases in each.

    Each row shows where the model was most sure, in the direction that bucket is sure about
    (see _pick_confident), with at most one frame per observation so the eight columns are
    eight independent cases rather than eight frames of one bout. Panels are captioned with
    the probability and the source observation; the row label carries the bucket's absolute
    count and share of the split.

    Confident errors are the diagnostic ones. A high-probability FP is a frame the model
    argues hard for and is wrong about, and a low-probability FN is a real bout it saw no
    evidence in at all -- those two rows say what the representation is actually missing,
    which a random draw from the same buckets mostly buries in borderline cases.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
            # TP/FP are most convincing at high p, FN/TN at low p
            pick = _pick_confident(pool, p, sample_obs, n_examples, b in ('TP', 'FP'))
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
