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


def rate_report(probs: np.ndarray, labels: np.ndarray, sample_obs: np.ndarray,
                label_names=('nt', 'nn'), calib: dict = None) -> dict:
    """Per-observation behaviour-rate agreement: correlation, MAE, and MAE's trivial baseline.

    ALWAYS read MAE against `mae_baseline_pp` -- the MAE you get by predicting the dataset
    mean rate for every observation. Unlike correlation, MAE has a trivial baseline that is
    hard to beat under this much imbalance, so a small absolute MAE means little on its own.
    Measured on the res448 checkpoint (frame AP 0.338): even after an *oracle* affine
    calibration fitted on the evaluation set itself, MAE beat the predict-the-mean baseline
    by only 27% (nt: 0.56 vs 0.77 pp) and 14% (nn: 1.48 vs 1.73 pp).

    `mae_raw_pp` uses the mean predicted probability directly and will be enormous whenever
    the model was trained with undersampling and/or pos_weight (prior shift): the same
    checkpoint scored 14.8 pp against a 1.23% true rate, i.e. 12x too high. `mae_calibrated_pp`
    applies the best affine rescale of the prediction; fit it on TRAINING folds only, never
    on the evaluation fold, or the number is meaningless.

    `calib` is how you actually honour that last sentence. Pass {label: (slope, intercept)}
    obtained from a fit fold and the affine rescale is APPLIED, not refitted; the report then
    marks `calibration_source='held-out'`. Leaving it None refits on the evaluation fold's own
    labels, which is an ORACLE calibration -- an upper bound, not an achievable number -- and
    is flagged as `calibration_source='oracle (fitted on this fold)'` so it can never again be
    read as if it were held out. Every mice-behaviour number reported before 2026-08-14 used
    the oracle path, so `mae_calibrated_pp` and `mae_vs_baseline` in those runs are optimistic.
    `calibration_fit` always carries the (slope, intercept) actually used, so a caller can fit
    on train observations and feed them straight back in here for the val pass.

    Correlation is the metric to select on -- it has no trivial baseline (a constant predictor
    scores 0), it is unaffected by the affine bias, and for PPI++ the variance reduction is
    ~(1 - r^2), so r is literally what determines how much the model helps downstream.
    """
    from scipy import stats

    out = {}
    df_obs = np.asarray(sample_obs)
    uniq = np.unique(df_obs)
    for i, name in enumerate(label_names):
        true = np.array([labels[df_obs == o, i].mean() for o in uniq])
        pred = np.array([probs[df_obs == o, i].mean() for o in uniq])
        if len(uniq) < 2 or np.ptp(true) == 0 or np.ptp(pred) == 0:
            # correlation is undefined with <2 observations or a constant series -- return NaNs
            # rather than raising, so smoke runs and degenerate splits still produce a report.
            out[name] = {'n_observations': int(len(uniq)), 'pearson_r': float('nan'),
                         'pearson_p': float('nan'), 'spearman_rho': float('nan'),
                         'r2': float('nan'), 'ppi_variance_reduction': float('nan'),
                         'true_rate_mean_pp': float(true.mean() * 100),
                         'mae_raw_pp': float(np.abs(pred - true).mean() * 100),
                         'mae_calibrated_pp': float('nan'),
                         'mae_baseline_pp': float(np.abs(true.mean() - true).mean() * 100),
                         'calibration_slope': float('nan'), 'mae_vs_baseline': 'n/a (degenerate)',
                         'calibration_source': 'n/a (degenerate)',
                         'calibration_fit': None}
            continue
        r = stats.pearsonr(true, pred)
        rho = stats.spearmanr(true, pred)
        if calib is not None and name in calib:
            slope, icept = calib[name]
            calib_src = 'held-out'
        else:
            # NOTE: this fits on the evaluation fold's own labels -- an oracle upper bound.
            slope, icept = np.polyfit(pred, true, 1)
            calib_src = 'oracle (fitted on this fold)'
        out[name] = {
            'calibration_source': calib_src,
            'calibration_fit': (float(slope), float(icept)),
            'n_observations': int(len(uniq)),
            'pearson_r': float(r[0]), 'pearson_p': float(r[1]), 'spearman_rho': float(rho[0]),
            'r2': float(r[0] ** 2),
            'ppi_variance_reduction': float(1 - r[0] ** 2),
            'true_rate_mean_pp': float(true.mean() * 100),
            'mae_raw_pp': float(np.abs(pred - true).mean() * 100),
            'mae_calibrated_pp': float(np.abs((slope * pred + icept) - true).mean() * 100),
            'mae_baseline_pp': float(np.abs(true.mean() - true).mean() * 100),
            'calibration_slope': float(slope),
        }
        o = out[name]
        o['mae_vs_baseline'] = (f'{100*(1 - o["mae_calibrated_pp"]/o["mae_baseline_pp"]):.0f}% better'
                                if o['mae_baseline_pp'] > 0 else 'n/a')
    return out


def format_rate_report(rep: dict, label_names=('nt', 'nn')) -> str:
    lines = ['per-observation behaviour-rate agreement (the downstream-relevant quantity):']
    for name in label_names:
        r = rep[name]
        lines.append(
            f'  [{name}] n={r["n_observations"]}  true mean {r["true_rate_mean_pp"]:.2f}pp\n'
            f'      SELECT ON -> Pearson r={r["pearson_r"]:+.3f} (p={r["pearson_p"]:.2g})  '
            f'Spearman={r["spearman_rho"]:+.3f}  R2={r["r2"]:.3f}  '
            f'=> PPI var x{r["ppi_variance_reduction"]:.2f} (eff. N x{1/max(r["ppi_variance_reduction"],1e-9):.1f})\n'
            f'      MAE raw {r["mae_raw_pp"]:.2f}pp (calib slope {r["calibration_slope"]:.3f}) | '
            f'calibrated {r["mae_calibrated_pp"]:.2f}pp | predict-the-mean baseline '
            f'{r["mae_baseline_pp"]:.2f}pp -> {r["mae_vs_baseline"]}\n'
            f'      calibration: {r.get("calibration_source", "unknown")}')
    return '\n'.join(lines)


def format_ap_report(rep: dict, label_names=('nt', 'nn'), tolerances=(0, 1, 2)) -> str:
    lines = [f'{"":<6} {"tol":>4} {"AP":>8} {"prev":>8} {"enrich":>8}']
    for name in label_names:
        for tol in tolerances:
            r = rep[f'{name}/tol{tol}']
            lines.append(f'{name:<6} {tol:>4} {r["ap"]:>8.4f} {100*r["prevalence"]:>7.2f}% {r["enrichment"]:>7.1f}x')
    for tol in tolerances:
        lines.append(f'{"macro":<6} {tol:>4} {rep[f"macro/tol{tol}"]["ap"]:>8.4f}')
    return '\n'.join(lines)
