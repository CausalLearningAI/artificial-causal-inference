import numpy as np


def noise_bound(p):
    """BA upper bound from pairwise disagreement rate on positive frames.

    Assumes symmetric noise: p_disagree = 2ε(1-ε)  →  ε = (1 − √(1−2p)) / 2.
    Since annotators agree almost perfectly on negatives (ε₀ ≈ 0), the bound
    simplifies to  1 − ε₁/2.
    """
    if np.isnan(p) or p >= 0.5:
        return np.nan
    return 1 - (1 - np.sqrt(1 - 2 * p)) / 2


def mixed_ba_bound(fracs, mat):
    """Weighted BA bound for a mixed-annotator dataset.

    BA*_mix = Σ_i f_i² · 1  +  Σ_{i≠j} f_i · f_j · IA-BA(i, j)

    Same-annotator train/test pairs (weight f_i²) are unconstrained (ceiling=1);
    cross-annotator pairs are capped by the measured pairwise IA-BA.

    Parameters
    ----------
    fracs : pd.Series  annotator → fraction of videos
    mat   : pd.DataFrame  symmetric IA-BA matrix (diagonal = 1)
    """
    bound, total_w = 0.0, 0.0
    for i, fi in fracs.items():
        for j, fj in fracs.items():
            if i not in mat.index or j not in mat.columns:
                continue
            w = fi * fj
            bound += w * (1.0 if i == j else mat.loc[i, j])
            total_w += w
    return bound / total_w if total_w > 0 else float("nan")
