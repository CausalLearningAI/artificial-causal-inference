"""Visualisation helpers for the inter-annotator agreement analysis."""

import random
import warnings
from collections import defaultdict

import matplotlib.cm as mcm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


# ── constants ─────────────────────────────────────────────────────────────────

_OUTCOME_LABEL = {
    "Y2F": "yellow ant grooming the focal ant",
    "B2F": "blue ant grooming the focal ant",
    "YOL": "yellow ant self-grooming",
    "BOL": "blue ant self-grooming",
    "FOL": "focal ant self-grooming",
}

# Positive-annotation colours per outcome (shown in the frame label bar)
_OUTCOME_COLOR = {
    "Y2F": "#ccaa00",   # yellow
    "B2F": "#1f77b4",   # blue
    "YOL": "#ccaa00",
    "BOL": "#1f77b4",
    "FOL": "#888888",
}
_NEG_COLOR  = "#cccccc"   # grey  → not annotated
_DIAG_COLOR = "#e8e8e8"   # light grey → diagonal (self, undefined)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_ba_matrix(pairs_df, outcome, annotators):
    """Full n×n BA matrix.  mat[b, a] = BA(a→b): a=ground-truth, b=predictor."""
    mat = pd.DataFrame(np.nan, index=annotators, columns=annotators)
    for (a1, a2), grp in pairs_df[pairs_df.outcome == outcome].groupby(["ann_a", "ann_b"]):
        if a1 in annotators and a2 in annotators:
            mat.loc[a2, a1] = grp["ba_12"].mean()   # a1 truth, a2 predictor
            mat.loc[a1, a2] = grp["ba_21"].mean()   # a2 truth, a1 predictor
    return mat


def _build_sym_matrix(pairs_df, outcome, annotators, metric):
    """Full n×n matrix for a symmetric metric (kappa, ia_ba)."""
    mat = pd.DataFrame(np.nan, index=annotators, columns=annotators)
    for (a1, a2), v in pairs_df[pairs_df.outcome == outcome].groupby(["ann_a", "ann_b"])[metric].mean().items():
        if a1 in annotators and a2 in annotators:
            mat.loc[a1, a2] = v
            mat.loc[a2, a1] = v
    return mat


def _draw_heatmap(ax, mat, annotators, cmap_name, vmin, vmax, fmt=".2f"):
    """Draw one heatmap, suppressing seaborn warnings.

    Diagonal (NaN) is rendered in light grey via cmap.set_bad().
    Off-diagonal NaN (missing pairs) is also light grey.
    """
    mat_np = mat.to_numpy().astype(float)
    ann    = np.where(
        np.isnan(mat_np),
        "",
        np.vectorize(lambda v: format(v, fmt))(np.where(np.isnan(mat_np), 0, mat_np)),
    )

    cmap = mcm.get_cmap(cmap_name).copy()
    cmap.set_bad(color=_DIAG_COLOR)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.heatmap(
            mat_np, annot=ann, fmt="",
            xticklabels=annotators, yticklabels=annotators,
            cmap=cmap, vmin=vmin, vmax=vmax,
            linewidths=0.6, linecolor="#dddddd",
            ax=ax, cbar=False, annot_kws={"size": 9},
        )
    ax.tick_params(left=False, bottom=False, labelsize=9)


# ── plot_iaba_heatmap ─────────────────────────────────────────────────────────

def plot_iaba_heatmap(pairs_df, outcomes, save_path=None):
    """Two rows of full BA + κ heatmaps (one column per outcome).

    x-axis = ground-truth annotator, y-axis = predictive annotator.
    Diagonal = light grey (undefined).  Note: BA is asymmetric — BA(A→B) can
    differ substantially from BA(B→A) when annotators have different positive
    rates; this is expected behaviour, not a bug.
    """
    annotators = sorted({a for p in pairs_df["pair"].unique() for a in p.split("-")})
    n_out = len(outcomes)

    rows_spec = [
        ("ba_dir", "Balanced Accuracy", "RdYlGn", 0.5, 1.0),
        ("kappa",  "Cohen's κ",         "RdYlGn", 0.0, 1.0),
    ]

    fig, axes = plt.subplots(
        2, n_out,
        figsize=(n_out * 4.5, 8.0),
        gridspec_kw={"hspace": 0.45, "wspace": 0.28},
    )
    if n_out == 1:
        axes = axes.reshape(2, 1)

    for row_i, (metric, metric_label, cmap, vmin, vmax) in enumerate(rows_spec):
        for col_i, out in enumerate(outcomes):
            ax  = axes[row_i, col_i]
            mat = (_build_ba_matrix(pairs_df, out, annotators) if metric == "ba_dir"
                   else _build_sym_matrix(pairs_df, out, annotators, metric))
            _draw_heatmap(ax, mat, annotators, cmap, vmin, vmax)

            ax.set_title(f"{out} — {metric_label}", fontsize=10, pad=6)
            ax.set_xlabel("Ground-truth annotator" if row_i == 1 else "", fontsize=8)
            ax.set_ylabel("Predictive annotator" if col_i == 0 else "", fontsize=8)

    # One colorbar per row on the right
    for row_i, (_, _, cmap, vmin, vmax) in enumerate(rows_spec):
        cbar_ax = fig.add_axes([0.93, 0.55 - row_i * 0.50, 0.015, 0.38])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

    fig.subplots_adjust(left=0.10, right=0.91, top=0.96, bottom=0.07)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── plot_disagreement_temporal ────────────────────────────────────────────────

# Outcomes shown in the per-frame annotation bar (always both grooming outcomes)
_BAR_OUTCOMES = ["Y2F", "B2F"]


def _classify_error(df1, df2, fi, outcome_col, bar_cols, context):
    """Classify disagreement type at frame fi from the context strip.

    Returns a list (possibly multi-tagged) from:
      'start'    – both annotators see the event in the window; positives skewed
                   after fi (disagreement is about when the event begins)
      'end'      – both see the event; positives skewed before fi (event ending)
      'boundary' – both see the event; positives centred on fi (ambiguous edge)
      'event'    – one annotator sees no positive in the window at all
      'identity' – at the focal frame Y2F/B2F labels are swapped between
                   annotators (each thinks the other ant is the active one)
    """
    tags = []
    window = list(range(fi - context, fi + context + 1))

    def col_vals(df, c):
        return [int(df.at[f, c]) if (f in df.index and c in df.columns) else -1
                for f in window]

    l1 = col_vals(df1, outcome_col)
    l2 = col_vals(df2, outcome_col)

    a1_has = any(v == 1 for v in l1)
    a2_has = any(v == 1 for v in l2)

    if not (a1_has and a2_has):
        tags.append("event")
    else:
        # Determine start vs end from where positives fall relative to fi
        pos_frames = ([window[i] for i, v in enumerate(l1) if v == 1] +
                      [window[i] for i, v in enumerate(l2) if v == 1])
        mean_rel = sum(f - fi for f in pos_frames) / len(pos_frames)
        if mean_rel > 0.4:
            tags.append("start")
        elif mean_rel < -0.4:
            tags.append("end")
        else:
            tags.append("boundary")

    # identity confusion: outcomes swap at the focal frame
    if len(bar_cols) >= 2:
        bc0, bc1 = bar_cols[0], bar_cols[1]
        v00 = int(df1.at[fi, bc0]) if (fi in df1.index and bc0 in df1.columns) else -1
        v01 = int(df2.at[fi, bc0]) if (fi in df2.index and bc0 in df2.columns) else -1
        v10 = int(df1.at[fi, bc1]) if (fi in df1.index and bc1 in df1.columns) else -1
        v11 = int(df2.at[fi, bc1]) if (fi in df2.index and bc1 in df2.columns) else -1
        if (v00 == 1 and v01 == 0 and v10 == 0 and v11 == 1) or \
           (v00 == 0 and v01 == 1 and v10 == 1 and v11 == 0):
            tags.append("identity")

    return tags


def plot_disagreement_temporal(
    labels, pairs_df, frame_root, outcome,
    k=10, context=2, min_gap=5, seed=0,
    exp_df=None, target_fps=5.0,
    save_path=None,
):
    """Temporal strips [t-context … t … t+context] for k disagreement events.

    Annotation bar overlaid at the bottom of each frame:
      left half  = annotator 1 → [Y2F block | B2F block]
      right half = annotator 2 → [Y2F block | B2F block]
    Block colour = outcome colour if annotated, grey if not.
    """
    col       = f"Y_{outcome}"
    bar_cols  = [f"Y_{o}" for o in _BAR_OUTCOMES]

    # ── Worst pair ────────────────────────────────────────────────────────────
    pair_ba    = pairs_df[pairs_df.outcome == outcome].groupby(["ann_a", "ann_b"])["ia_ba"].mean()
    pair_kappa = pairs_df[pairs_df.outcome == outcome].groupby(["ann_a", "ann_b"])["kappa"].mean()
    if pair_ba.empty:
        print(f"{outcome}: no pairs found"); return None

    worst_a1, worst_a2 = pair_ba.idxmin()
    worst_ba    = pair_ba.min()
    worst_kappa = pair_kappa.get((worst_a1, worst_a2), float("nan"))

    # ── Collect events, stratified across videos ──────────────────────────────
    events_by_video = defaultdict(list)
    for stem, adict in labels.items():
        if worst_a1 not in adict or worst_a2 not in adict:
            continue
        if not (frame_root / stem).exists():
            continue
        df1, df2 = adict[worst_a1], adict[worst_a2]
        if col not in df1.columns or col not in df2.columns:
            continue
        idx    = df1.index.intersection(df2.index)
        y1, y2 = df1.loc[idx, col], df2.loc[idx, col]
        mask   = (y1 != y2) & ((y1 == 1) | (y2 == 1))

        last = -min_gap - 1
        for fi in sorted(idx[mask].tolist()):
            if fi < context or fi > idx.max() - context:
                continue
            if fi - last >= min_gap:
                events_by_video[stem].append(fi)
                last = fi

    all_stems = list(events_by_video.keys())
    if not all_stems:
        print(f"{outcome}: no disagreement events found"); return None

    rng   = random.Random(seed)
    pools = {s: rng.sample(evs, len(evs)) for s, evs in events_by_video.items()}
    rng.shuffle(all_stems)
    sample, i = [], 0
    while len(sample) < min(k, sum(len(v) for v in pools.values())):
        s = all_stems[i % len(all_stems)]
        if pools[s]:
            sample.append({"stem": s, "frame": pools[s].pop()})
        i += 1
    sample.sort(key=lambda d: (d["stem"], d["frame"]))

    # ── Layout ────────────────────────────────────────────────────────────────
    n_rows    = len(sample)
    strip_len = 2 * context + 1
    fig, axes = plt.subplots(
        n_rows, strip_len,
        figsize=(strip_len * 2.6, n_rows * 2.8),
        gridspec_kw={"hspace": 0.25, "wspace": 0.04},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, info in enumerate(sample):
        stem, fi = info["stem"], info["frame"]
        df1 = labels[stem][worst_a1]
        df2 = labels[stem][worst_a2]

        for col_off, offset in enumerate(range(-context, context + 1)):
            ax       = axes[row, col_off]
            frame_fi = fi + offset
            img_path = frame_root / stem / f"frame_{frame_fi:06d}.jpg"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                    W, H = img.size
                else:
                    ax.set_facecolor("#eeeeee")
                    W, H = 512, 512

            # ── Annotation bar: 2 rows (Y2F top, B2F bottom) × 2 cols (ann1|ann2)
            #    placed BELOW the frame (axes extended downward)
            bar_h  = H * 0.22   # total bar height (two stacked rows)
            sub_h  = bar_h / 2  # height per outcome row
            anns   = [worst_a1, worst_a2]
            dfs    = [df1, df2]

            ax.set_xlim(0, W)
            ax.set_ylim(H + bar_h, 0)   # extend axes below frame

            for ai, (ann, df) in enumerate(zip(anns, dfs)):
                for oi, out_bar in enumerate(_BAR_OUTCOMES):  # oi=0: Y2F (top), oi=1: B2F (bottom)
                    bc    = f"Y_{out_bar}"
                    lv    = int(df.at[frame_fi, bc]) if (frame_fi in df.index and bc in df.columns) else -1
                    color = _OUTCOME_COLOR.get(out_bar, "#888") if lv == 1 else _NEG_COLOR
                    ax.add_patch(patches.Rectangle(
                        (ai * W / 2, H + oi * sub_h), W / 2, sub_h,
                        linewidth=0, color=color, zorder=2,
                    ))

                # Annotator label centred in the column (across both rows)
                ax.text(ai * W / 2 + W / 4, H + bar_h / 2, ann,
                        ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold", zorder=3)

            # Thin white divider between the two annotators
            ax.add_patch(patches.Rectangle(
                (W / 2 - 0.5, H), 1, bar_h,
                linewidth=0, color="white", zorder=4,
            ))

            # ── Border ────────────────────────────────────────────────────────
            is_center = (offset == 0)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor("#cc2222" if is_center else "#cccccc")
                sp.set_linewidth(2.5 if is_center else 0.5)
            ax.set_xticks([]); ax.set_yticks([])

            if row == 0:
                ax.set_title(
                    "t = 0" if is_center else f"t {offset:+d}",
                    fontsize=8,
                    color="#cc2222" if is_center else "#555",
                    fontweight="bold" if is_center else "normal",
                    pad=4,
                )

        # ── Row label ─────────────────────────────────────────────────────────
        parts = stem.split("_")
        batch = parts[1] if len(parts) > 1 else "?"
        pos   = parts[2] if len(parts) > 2 else "?"
        nestbox = (exp_df.loc[stem, "nestbox"]
                   if exp_df is not None and stem in exp_df.index else "?")
        total_s = int(fi / target_fps)
        mins, secs = divmod(total_s, 60)

        error_tags = _classify_error(df1, df2, fi, col, bar_cols, context)
        tag_str    = ", ".join(error_tags)

        axes[row, 0].set_ylabel(
            f"Experiment: v{parts[0]}, Nestbox: {nestbox}\nBatch: {batch}, Position: {pos}\n{mins}min {secs:02d}s\n\nMisalignment: [{tag_str}]",
            fontsize=6.5, rotation=0, labelpad=70, va="center",
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    beh = _OUTCOME_LABEL.get(outcome, outcome)
    fig.suptitle(
        f"Examples of annotation misalignment\n"
        f"Annotators: {worst_a1}, {worst_a2}  "
        f"(Balanced Acc. = {worst_ba:.3f},  Cohen's κ = {worst_kappa:.3f})\n"
        f"Behaviour: {beh}",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.01)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
