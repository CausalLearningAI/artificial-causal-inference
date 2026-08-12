"""Evaluation metrics for the mice per-frame behavior classifier.

Why tolerant AP exists
----------------------
Annotated bouts are extremely short: measured over all 144 annotated v1 observations at
5 fps, 22% of nt bouts and 38% of nn bouts are a SINGLE frame (0.2 s), and 41% / 60% are
<= 2 frames. Bout-boundary annotation is not reliable to that precision -- a +-1 frame
disagreement on a 2-frame event is a 50% error -- so plain frame-exact AP spends much of
its dynamic range penalising boundary jitter rather than genuine detection failure.

Tolerant AP scores a prediction as correct if it lands within `tolerance` frames of an
annotated positive, by dilating the label (never the prediction) before computing AP. It
stays threshold-free and calibration-free, exactly like plain AP, so both can be reported
side by side; the gap between them quantifies how much of the frame-level error is pure
boundary jitter.

Dilation raises prevalence, which mechanically raises AP -- so ALWAYS report tolerant AP
next to its own dilated prevalence (or as enrichment = AP / prevalence), never as if it
were comparable to plain AP on an absolute scale.
"""
import numpy as np
from sklearn.metrics import average_precision_score


def dilate_labels(labels: np.ndarray, sample_obs: np.ndarray, tolerance: int = 1) -> np.ndarray:
    """Widen each positive label by +-`tolerance` frames, without crossing observation
    boundaries.

    Args:
        labels:     (N, C) binary array, rows ordered by (observation, frame_idx).
        sample_obs: (N,) observation identifier per row -- used so dilation never leaks
                    from the end of one video into the start of the next.
        tolerance:  frames of slack on each side. 0 returns `labels` unchanged.

    Returns:
        (N, C) dilated binary array.
    """
    if tolerance <= 0:
        return labels.copy()
    out = labels.copy()
    for shift in range(1, tolerance + 1):
        for direction in (1, -1):
            shifted = np.roll(labels, direction * shift, axis=0)
            same_obs = np.roll(sample_obs, direction * shift) == sample_obs
            out = np.maximum(out, shifted * same_obs[:, None])
    return out


def ap_report(probs: np.ndarray, labels: np.ndarray, sample_obs: np.ndarray,
              label_names=('nt', 'nn'), tolerances=(0, 1, 2)) -> dict:
    """Plain and tolerant AP, each with its own prevalence and enrichment.

    Enrichment (AP / prevalence) is the interpretable figure under this much imbalance:
    an uninformative ranker scores AP == prevalence, so enrichment is "how many times
    better than chance", and unlike raw AP it is comparable across behaviors whose
    prevalence differs (nt ~1.1%, nn ~3.2%) and across dilation levels.
    """
    out = {}
    for tol in tolerances:
        dil = dilate_labels(labels, sample_obs, tol)
        for i, name in enumerate(label_names):
            prev = float(dil[:, i].mean())
            ap = float(average_precision_score(dil[:, i], probs[:, i]))
            out[f'{name}/tol{tol}'] = {
                'ap': ap,
                'prevalence': prev,
                'enrichment': ap / prev if prev > 0 else float('nan'),
            }
        aps = [out[f'{n}/tol{tol}']['ap'] for n in label_names]
        out[f'macro/tol{tol}'] = {'ap': float(np.mean(aps))}
    return out


def format_ap_report(rep: dict, label_names=('nt', 'nn'), tolerances=(0, 1, 2)) -> str:
    lines = [f'{"":<6} {"tol":>4} {"AP":>8} {"prev":>8} {"enrich":>8}']
    for name in label_names:
        for tol in tolerances:
            r = rep[f'{name}/tol{tol}']
            lines.append(f'{name:<6} {tol:>4} {r["ap"]:>8.4f} {100*r["prevalence"]:>7.2f}% {r["enrichment"]:>7.1f}x')
    for tol in tolerances:
        lines.append(f'{"macro":<6} {tol:>4} {rep[f"macro/tol{tol}"]["ap"]:>8.4f}')
    return '\n'.join(lines)
