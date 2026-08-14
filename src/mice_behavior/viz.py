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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.metrics import precision_recall_curve

BUCKETS = ('TP', 'FP', 'FN', 'TN')
BUCKET_COLOR = {'TP': '#2e7d32', 'FP': '#c62828', 'FN': '#ef6c00', 'TN': '#455a64'}
BUCKET_LONG = {'TP': 'true positives', 'FP': 'false positives',
               'FN': 'false negatives', 'TN': 'true negatives'}

# Human annotation, drawn on every panel including context frames. Deliberately a different
# green from BUCKET_COLOR['TP'] -- in a strip figure the border means "the annotator marked
# this frame", which is a statement about the DATA and must not be confused with a statement
# about the model's prediction.
LABEL_POS = '#00a000'
LABEL_NEG = '#cfcfcf'


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


def _pick_confident(pool, p, sample_obs, n, most_confident_high, top_up=True):
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

    top_up=False drops the one-per-observation rule's fallback and returns a SHORT list when
    the bucket spans fewer than n observations. The grid figure tops up because a short row
    reads as "no such errors exist"; the strip figure does not, because there every row is
    captioned with its own video and a repeated video would silently break the "n independent
    cases" claim the figure is making.
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
    if top_up and len(out) < n:
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


def plot_error_strips(probs, labels, sample_obs, frame_gi, frame_paths, out_dir,
                      label_names=('nt', 'nn'), buckets=('FP', 'FN'), n_rows=10, context=3,
                      jpeg_cache=None, dataset_root='dataset', title_prefix='',
                      frame_idx=None):
    """One figure per (behavior, error type): each row is one error shown as a time strip.

    Why a strip and not a single frame
    ----------------------------------
    plot_confusion_examples answers "which frames does it get wrong"; it cannot answer "what
    was happening". At 5 fps a nose-to-nose contact lasts a handful of frames, and a single
    still of two mice near each other is compatible with contact, with approach, and with
    passing by -- the three cases the classifier has to separate. Showing t-{context}..t+{context}
    around the error makes the event legible, and it makes the model's own trajectory legible
    too: an FP that ramps 0.02 -> 0.99 -> 0.03 is a momentary pose confusion, whereas one
    that sits at 0.99 across the whole window is a sustained misreading of the configuration.

    Why the human label is drawn on every frame
    -------------------------------------------
    The context frames carry annotations of their own, and those annotations are the point.
    A "false positive" whose neighbours are annotated positive is an off-by-one against a bout
    the model did find -- a boundary disagreement, not a hallucination -- and the tolerant-AP
    columns in metrics.py exist precisely because that case is common here (22%/38% of nt/nn
    bouts are a single frame, so boundaries are a large share of the label mass). A green
    border marks a frame the annotator marked positive for THIS behavior; the center frame
    additionally carries an inner rectangle in the bucket color, so the two claims -- what the
    human said, what the model got wrong -- never collapse into one mark.

    A frame annotated positive for the OTHER behavior gets a small corner tag. An nn false
    positive on a frame the human called nt is a confusion BETWEEN behaviors and has a
    different fix from an nn false positive on an empty cage.

    Rows are the n_rows most confident errors, one per video, sorted in the direction that
    bucket is confident about (see _pick_confident): highest p for FP, lowest for FN.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probs, labels = np.asarray(probs), np.asarray(labels)
    frame_gi = np.asarray(frame_gi, dtype=np.int64)
    sample_of_gi = {int(g): i for i, g in enumerate(frame_gi)}
    offsets = list(range(-context, context + 1))
    written = {}

    for li, name in enumerate(label_names):
        y, p = labels[:, li].astype(int), probs[:, li]
        other = [j for j in range(labels.shape[1]) if j != li]
        thr, prec, rec, f1 = best_f1_threshold(y, p)
        pred = p >= thr
        pools = {'TP': np.where((y == 1) & pred)[0], 'FP': np.where((y == 0) & pred)[0],
                 'FN': np.where((y == 1) & ~pred)[0], 'TN': np.where((y == 0) & ~pred)[0]}

        for b in buckets:
            rows = _pick_confident(pools[b], p, sample_obs, n_rows, b in ('TP', 'FP'),
                                   top_up=False)
            if not rows:
                print(f'  [viz] {name}/{b}: bucket empty, no strip figure', flush=True)
                continue
            if len(rows) < n_rows:
                # never silently render a shorter figure than asked for: the cap is the number
                # of distinct val videos containing this error, which is itself a finding.
                print(f'  [viz] {name}/{b}: only {len(rows)} distinct videos have this error '
                      f'(asked {n_rows})', flush=True)

            fig, axes = plt.subplots(len(rows), len(offsets),
                                     figsize=(1.55 * len(offsets), 1.72 * len(rows) + 1.1))
            axes = np.atleast_2d(axes)
            for r, i in enumerate(rows):
                g0, obs = int(frame_gi[i]), str(sample_obs[i])
                for c, d in enumerate(offsets):
                    ax = axes[r, c]
                    ax.set_xticks([]); ax.set_yticks([])
                    j = sample_of_gi.get(g0 + d)
                    # a context frame belongs to this strip only if it is the same video --
                    # global frame indices run straight across the observation boundary, so
                    # without this an error near the start of a video would show the end of
                    # the previous one.
                    if j is None or str(sample_obs[j]) != obs:
                        for sp in ax.spines.values():
                            sp.set_color('#eeeeee')
                        ax.set_facecolor('#fafafa')
                        ax.text(.5, .5, '—', ha='center', va='center', color='#bbbbbb', fontsize=9)
                        continue
                    pos_here = labels[j, li] > 0
                    for sp in ax.spines.values():
                        sp.set_color(LABEL_POS if pos_here else LABEL_NEG)
                        sp.set_linewidth(3.0 if pos_here else 1.0)
                    try:
                        ax.imshow(_load_frame(frame_gi[j], frame_paths, jpeg_cache, dataset_root),
                                  cmap='gray', vmin=0, vmax=255)
                    except Exception as e:           # a figure must never kill a run
                        ax.text(.5, .5, f'{e.__class__.__name__}', ha='center', va='center',
                                fontsize=6)
                    if d == 0:
                        ax.add_patch(mpatches.Rectangle(
                            (0.035, 0.035), 0.93, 0.93, transform=ax.transAxes, fill=False,
                            edgecolor=BUCKET_COLOR[b], linewidth=2.6, zorder=5))
                    tags = [label_names[o] for o in other if labels[j, o] > 0]
                    if tags:
                        ax.text(0.04, 0.96, '+'.join(tags), transform=ax.transAxes,
                                ha='left', va='top', fontsize=6.5, color='#1565c0',
                                bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='#1565c0',
                                          lw=0.6, alpha=0.85))
                    ax.set_title(f'{d:+d}   p={p[j]:.3f}' if d else f'0   p={p[j]:.3f}',
                                 fontsize=7, pad=2,
                                 color=BUCKET_COLOR[b] if d == 0 else '#444444',
                                 fontweight='bold' if d == 0 else 'normal')
                fnum = f'\nframe {int(frame_idx[i]):,}' if frame_idx is not None else ''
                axes[r, 0].set_ylabel(f'{obs}{fnum}\np={p[i]:.3f}', fontsize=7, rotation=0,
                                      ha='right', va='center', labelpad=8, color='#333333')

            handles = [
                Line2D([], [], color=LABEL_POS, lw=3, label=f'human-annotated {name} on that frame'),
                Line2D([], [], color=LABEL_NEG, lw=1.6, label=f'human-annotated not-{name}'),
                Line2D([], [], color=BUCKET_COLOR[b], lw=2.6, label=f'the {b} frame itself (offset 0)'),
                mpatches.Patch(fc='white', ec='#1565c0', label='corner tag: other behavior annotated here'),
            ]
            fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=7.5,
                       frameon=False, bbox_to_anchor=(0.5, 0.004))
            fig.suptitle(
                f'{title_prefix}{name}  —  {len(rows)} most confident {BUCKET_LONG[b]}, one per video\n'
                f'operating point: p ≥ {thr:.3f} (max-F1)  precision {prec:.3f}  recall {rec:.3f}  '
                f'F1 {f1:.3f}   |   bucket n={len(pools[b]):,} '
                f'({100*len(pools[b])/max(len(y), 1):.2f}% of {len(y):,} val frames)\n'
                f'columns are frame offsets at 5 fps (±{context} frames = ±{context/5:.1f} s)',
                fontsize=10)
            fig.tight_layout(rect=[0.015, 0.022, 1, 0.955])
            f = out_dir / f'error_strip_{name}_{b}.png'
            fig.savefig(f, dpi=125)
            plt.close(fig)
            written[f'{name}_{b}'] = f
            print(f'  [viz] {name}/{b}: {len(rows)} strips ±{context} -> {f}', flush=True)
    return written
