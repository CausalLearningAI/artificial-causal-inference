# Verbatim copy of the original ECI-paper NES code (read-only reference):
#   /nfs/scistore19/locatgrp/tmencatt/projects/neuralscience/src/neuralscience/nes.py  (Dec 18 2025)
# Vendored here so the "v0" replication (scripts/eci/v0/ants_v2_v0.py) runs the exact same search code
# without importing from another user's checkout. Do NOT edit below this header: the point is fidelity,
# bugs included (e.g. zero-SE strata give p = 0 in the stratified 'AF' step).
# nes.py
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
from statsmodels.stats.multitest import multipletests
from typing import List, Optional, Dict, Any, Literal
from typing import Any, Literal, List, Optional
from scipy.stats import ttest_ind, ttest_1samp, t as student_t
from statsmodels.stats.multitest import multipletests


SUPPORTED_INFERENCE = ["top-k", "Bonferroni", "FDR", "t-test"]


def _stratified_ttests_lotp(
    df: pd.DataFrame,
    sn: List[str],
    *,
    treatment: str = "T",
    neuron_prefix: str = "N_",
    bin_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Stratified test via Law of Total Probability (binary strata from binarized `sn`).
    Returns per-neuron DataFrame with columns: ['neuron','ate','se','tstat','pval','kept_strata','N'].
    """
    if not sn:
        raise ValueError("sn must be a non-empty list of stratifying neuron names.")
    if treatment not in df.columns:
        raise ValueError(f"Treatment column '{treatment}' not found.")
    neuron_cols = [c for c in df.columns if c.startswith(neuron_prefix)]
    if not neuron_cols:
        raise ValueError(f"No columns start with prefix '{neuron_prefix}'.")

    # Build strata ids from binarized sn
    Sbin = (df[sn].to_numpy() > bin_threshold).astype(np.int8)
    powers = (1 << np.arange(Sbin.shape[1], dtype=np.int64))
    # gid = pd.Series((Sbin * powers).sum(axis=1), name="_gid") FIXME: check if this works pay attention 
    gid = pd.Series((Sbin * powers).sum(axis=1), index=df.index, name="_gid")


    N = len(df)
    T = df[treatment].astype(int)
    neurons = df[neuron_cols]

    # Split by treatment
    idx1 = T == 1
    idx0 = ~idx1
    gid1 = gid[idx1]
    gid0 = gid[idx0]

    # Per-stratum counts
    n1 = gid1.value_counts().rename("n1")
    n0 = gid0.value_counts().rename("n0")
    n_all = gid.value_counts().rename("n")

    # Per-stratum means/vars for all neurons (vectorized)
    mean1 = neurons[idx1].groupby(gid1).mean()
    mean0 = neurons[idx0].groupby(gid0).mean()
    var1  = neurons[idx1].groupby(gid1).var(ddof=1)
    var0  = neurons[idx0].groupby(gid0).var(ddof=1)

    # Align on same strata index
    strata = n_all.index.union(n1.index).union(n0.index)
    n1 = n1.reindex(strata, fill_value=0)
    n0 = n0.reindex(strata, fill_value=0)
    n_all = n_all.reindex(strata, fill_value=0)
    mean1 = mean1.reindex(strata).astype(float)
    mean0 = mean0.reindex(strata).astype(float)
    var1  = var1.reindex(strata).astype(float)
    var0  = var0.reindex(strata).astype(float)

    # Keep only strata with both arms
    ok = (n1 > 0) & (n0 > 0)
    if ok.sum() == 0:
        return pd.DataFrame({
            "neuron": neuron_cols,
            "ate": np.nan,
            "se": np.nan,
            "tstat": np.nan,
            "pval": np.nan,
            "kept_strata": 0,
            "N": N,
        })

    strata_kept = strata[ok.values]
    n1, n0, n_all = n1.loc[strata_kept], n0.loc[strata_kept], n_all.loc[strata_kept]
    mean1 = mean1.loc[strata_kept]
    mean0 = mean0.loc[strata_kept]
    var1  = var1.loc[strata_kept]
    var0  = var0.loc[strata_kept]

    # Weights w_s ∝ prevalence of each stratum (renormalized over kept strata)
    w = (n_all / N).to_numpy(dtype=float)
    w = w / w.sum()

    # ATE per neuron
    diff_means = (mean1 - mean0).to_numpy()             # (S, P)
    ate = (w[:, None] * diff_means).sum(axis=0)         # (P,)

    # SE^2: sum_s w_s^2 ( var1_s/n1_s + var0_s/n0_s )
    n1_arr = n1.to_numpy(dtype=float)[:, None]
    n0_arr = n0.to_numpy(dtype=float)[:, None]
    contrib_var = (var1.to_numpy() / n1_arr) + (var0.to_numpy() / n0_arr)  # (S,P)
    se2 = ((w[:, None] ** 2) * contrib_var).sum(axis=0)
    se = np.sqrt(se2)

    # Satterthwaite df (Welch-style)
    v_s = (w[:, None] ** 2) * contrib_var
    d1 = np.maximum(n1.to_numpy(dtype=float) - 1, 1.0)[:, None]
    d0 = np.maximum(n0.to_numpy(dtype=float) - 1, 1.0)[:, None]
    a = (w[:, None] ** 2) * (var1.to_numpy() / n1_arr)
    b = (w[:, None] ** 2) * (var0.to_numpy() / n0_arr)
    satter_denom = (a * a) / d1 + (b * b) / d0
    num = (v_s.sum(axis=0)) ** 2
    den = satter_denom.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df_eff = num / den
    df_eff = np.clip(df_eff, 1.0, 1e9)

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = ate / se
    pval = 2.0 * student_t.sf(np.abs(tstat), df=df_eff)

    return pd.DataFrame({
        "neuron": neuron_cols,
        "ate": ate,
        "se": se,
        "tstat": tstat,
        "pval": pval,
        "kept_strata": int(ok.sum()),
        "N": int(N),
    })

def _to_full_neuron_name(suffix: str, neuron_prefix: str) -> str:
    """Turn '38' -> 'N_38' (or whatever prefix you use)."""
    return f"{neuron_prefix}{suffix}"

def get_significant_neuron(
    data: pd.DataFrame,
    sn: List[str],
    *,
    base_fn,                              # usually neural_effect_search_base
    neuron_prefix: str = "N_",
    alpha: float = 0.05,
    method: str = "Bonferroni",
    **base_kwargs: Any,
) -> Optional[str]:
    """
    Call base_fn with AF+current sn. If there are significant neurons,
    return the TOP-1 (as full column name, e.g. 'N_38'). Otherwise None.
    """
    res = base_fn(
        data=data,
        method=method,
        alpha=alpha,
        neuron_prefix=neuron_prefix,
        sn=sn,
        **base_kwargs,   # e.g. fallback_start_neur_index, covariate_col, etc.
    )
    # base_fn already returns only significant ones (after correction) and
    # sorted by |effect| desc. If none, return None.
    if not res["neurons"]:
        return None
    # Take the top-1 and convert to full column name
    top_suffix = str(res["neurons"][0])
    return _to_full_neuron_name(top_suffix, neuron_prefix)

def neural_effect_search_base(
    data: pd.DataFrame,
    test: Literal["AD", "AIPW", "AF"] = "AD",
    method: Literal["top-k", "Bonferroni", "FDR", "t-test"] = "Bonferroni",
    alpha: float = 0.05,
    k: int = 10,
    neuron_prefix: str = "N_",
    fallback_start_neur_index: int = 4,
    covariate_col: str = "W",
    sn: Optional[List[str]] = None,          # <-- NEW: stratification neurons for AF
    residualization: bool = True
):
    """
    Args:
        data: dataframe with columns T, neurons (N_*), and covariate W
        test:   "AD" (associational difference), "AIPW", or "AF" (stratified LOTP t-test)
        method: multiple testing correction or "top-k"
        sn:     list of neuron columns to define strata (required for test="AF")
    Returns:
        dict with keys:
            - "neurons": list[str]
            - "LATE":   list[float]    effect estimate
            - "STD":    list[float]    variability proxy (SE for AF/AIPW; sum of SDs for AD/top-k)
            - "p_vals": list[float]    per selected neuron
    """

    # Identify neuron columns
    cols = [c for c in data.columns if neuron_prefix in c]
    if not cols:
        cols = list(data.columns)[fallback_start_neur_index:]

    T = data["T"].to_numpy()
    Y_mat = data[cols].to_numpy()  # (n, d)
    W = data[[covariate_col]].to_numpy() if covariate_col in data.columns else None  # (n, 1) or None
    if len(sn) == 0 and test == "AF":
        test = "AD"  # fallback to AIPW if empty sn
    if len(sn)>0 and residualization:
        # replace Y_mat with the linear residul predicting from sn
        sn_cols = [c for c in sn if c in data.columns]
        if len(sn_cols) < len(sn):
            missing = set(sn) - set(sn_cols)
            raise ValueError(f"Some stratifying neurons in sn are not in data columns: {missing}")
        if len(sn_cols) == 0:
            raise ValueError("sn must contain at least one valid column name.")

        # linear regression of each neuron on sn, take residuals
        for t in [0,1]:
            idx = (T==t)
            S = data.loc[idx, sn_cols].to_numpy()  # (n_t, s)
            Y_t = data.loc[idx, cols].to_numpy()   # (n_t, d)
            # add intercept
            S = np.hstack([np.ones((S.shape[0],1)), S])  # (n_t, s+1)
            # fit linear model: beta = (S^T S)^(-1) S^T Y
            beta = np.linalg.pinv(S.T @ S) @ (S.T @ Y_t)  # (s+1, d)
            Y_pred = S @ beta                            # (n_t, d)
            Y_mat[idx,:] = Y_t - Y_pred                  # residuals

    # -------------------------
    # Case 1: Top-k selection (no tests)
    # -------------------------
    if method == "top-k":
        treated = Y_mat[T == 1]
        control = Y_mat[T == 0]
        effects = treated.mean(axis=0) - control.mean(axis=0)
        stds = treated.std(axis=0, ddof=1) + control.std(axis=0, ddof=1)
        order = np.argsort(np.abs(effects))[::-1][:k]
        ranked_cols = [cols[i] for i in order]
        p_vals_out = [float("nan")] * len(ranked_cols)

    else:
        # -------------------------
        # Case 2: Run AD, AIPW, or AF
        # -------------------------
        if test == "AD":
            treated = Y_mat[T == 1]
            control = Y_mat[T == 0]
            effects = treated.mean(axis=0) - control.mean(axis=0)
            stds = treated.std(axis=0, ddof=1) + control.std(axis=0, ddof=1)
            _, pvals_raw = ttest_ind(treated, control, axis=0, equal_var=False)

        elif test == "AIPW":
            if W is None:
                raise ValueError(f"Covariate column '{covariate_col}' missing for AIPW.")
            n, d = Y_mat.shape
            p_hat = T.mean()  # RCT propensity score

            # add intercept
            X = np.hstack([np.ones((n, 1)), W])  # shape (n, 2)

            # outcome models: linear regression separately in T=0 and T=1
            def fit_linear(Y, X):
                return np.linalg.pinv(X.T @ X) @ (X.T @ Y)

            beta0 = fit_linear(Y_mat[T == 0], X[T == 0])
            beta1 = fit_linear(Y_mat[T == 1], X[T == 1])

            mu0 = X @ beta0
            mu1 = X @ beta1

            # AIPW pseudo-outcome
            aipw = (
                (T[:, None] / p_hat) * (Y_mat - mu1) +
                ((1 - T)[:, None] / (1 - p_hat)) * (Y_mat - mu0) +
                (mu1 - mu0)
            )

            effects = aipw.mean(axis=0)
            stds = aipw.std(axis=0, ddof=1)  # variability proxy
            _, pvals_raw = ttest_1samp(aipw, popmean=0, axis=0)

        elif test == "AF":
            if not sn:
                raise ValueError("For test='AF', you must provide a non-empty 'sn' list of stratifying neurons.")
            # compute stratified results
            af_df = _stratified_ttests_lotp(data, sn, treatment="T", neuron_prefix=neuron_prefix)
            af_df = af_df.set_index("neuron").reindex(cols)  # align to 'cols'
            effects = af_df["ate"].to_numpy()
            stds = af_df["se"].to_numpy()                    # use SE as variability proxy
            pvals_raw = af_df["pval"].to_numpy()

        else:
            raise ValueError("test must be 'AD', 'AIPW', or 'AF'")

        # Guard NaNs before corrections
        valid = np.isfinite(pvals_raw)
        if not valid.any():
            return {"neurons": [], "LATE": [], "STD": [], "p_vals": []}

        # -------------------------
        # Multiple-testing correction
        # -------------------------
        if method == "Bonferroni":
            reject, pvals_adj, _, _ = multipletests(pvals_raw[valid], alpha=alpha, method="bonferroni")
            sig_idx_global = np.where(valid)[0][np.where(reject)[0]]
            order = np.argsort(np.abs(effects[sig_idx_global]))[::-1]
            ranked_cols = [cols[i] for i in sig_idx_global[order]]
            p_vals_out = [float(pvals_adj[j]) for j in np.where(reject)[0][order]]

        elif method == "FDR":
            reject, pvals_adj, _, _ = multipletests(pvals_raw[valid], alpha=alpha, method="fdr_bh")
            sig_idx_global = np.where(valid)[0][np.where(reject)[0]]
            order = np.argsort(np.abs(effects[sig_idx_global]))[::-1]
            ranked_cols = [cols[i] for i in sig_idx_global[order]]
            p_vals_out = [float(pvals_adj[j]) for j in np.where(reject)[0][order]]

        elif method == "t-test":
            rej = (pvals_raw[valid] < alpha)
            sig_idx_global = np.where(valid)[0][np.where(rej)[0]]
            order = np.argsort(np.abs(effects[sig_idx_global]))[::-1]
            ranked_cols = [cols[i] for i in sig_idx_global[order]]
            p_vals_out = [float(pvals_raw[i]) for i in sig_idx_global[order]]

        else:
            raise ValueError("Unknown method")

        if not ranked_cols:
            return {"neurons": [], "LATE": [], "STD": [], "p_vals": []}

    # -------------------------
    # Format output
    # -------------------------
    neurons = [c.split("_")[-1] if neuron_prefix in c else c for c in ranked_cols]
    col_idx = {c: i for i, c in enumerate(cols)}
    late_vals = [float(effects[col_idx[c]]) for c in ranked_cols]
    std_vals = [float(stds[col_idx[c]]) for c in ranked_cols]

    return {"neurons": neurons, "LATE": late_vals, "STD": std_vals, "p_vals": p_vals_out}

from typing import List, Dict, Any

def neural_effect_search(
    data,
    *,
    base_fn = neural_effect_search_base,
    neuron_prefix: str = "N_",
    alpha: float = 0.05,
    method: str = "Bonferroni",
    max_steps: int = 10000,
    **base_kwargs: Any,
) -> Dict[str, List[float]]:
    """
    Helo.
    Discovery-only:
      - Start with sn = [].
      - Repeatedly call base_fn(..., test='AF', sn=sn).
      - Take top-1 significant neuron not yet in sn; record its stats from THIS step.
      - Append it to sn and repeat until none left.
    Returns a dict in the SAME schema as base_fn:
        {"neurons": <suffixes>, "LATE": <effects>, "STD": <std/se>, "p_vals": <pvals>}
    The stats correspond to the step the neuron was discovered (not re-evaluated later).
    """
    sn_full: List[str] = []              # internal: full column names (e.g., "N_38")
    seen = set()

    # what we will return (same format as base)
    out_neurons_suffix: List[str] = []   # e.g., ["38","251",...]
    out_effects: List[float] = []
    out_stds: List[float] = []
    out_pvals: List[float] = []

    for _ in range(max_steps):
        res = base_fn(
            data=data,
            test="AF",
            method=method,
            alpha=alpha,
            neuron_prefix=neuron_prefix,
            sn=sn_full,
            **base_kwargs,
        )
        # base_fn already returns significant + ranked by |effect|
        if not res["neurons"]:
            break

        # pick first candidate not yet in sn
        picked_idx = None
        for j, suf in enumerate(res["neurons"]):
            full = f"{neuron_prefix}{suf}"
            if full not in seen:
                picked_idx = j
                picked_full = full
                picked_suf = suf
                break

        if picked_idx is None:
            # no new info → stop
            break

        # record stats *from this discovery step*
        out_neurons_suffix.append(str(picked_suf))
        out_effects.append(float(res["LATE"][picked_idx]))
        out_stds.append(float(res["STD"][picked_idx]))
        out_pvals.append(float(res["p_vals"][picked_idx]))

        # update state
        sn_full.append(picked_full)
        seen.add(picked_full)

    return {
        "neurons": out_neurons_suffix,   # suffixes, like base_fn
        "LATE": out_effects,
        "STD": out_stds,
        "p_vals": out_pvals,
    }

