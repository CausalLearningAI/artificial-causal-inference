#!/usr/bin/env python3
"""Classical vs PPI++ genotype contrast for mice v1, and the confidence-interval shrinkage.

Two modes:

  REAL   --predictions <csv>   A pool-level table with columns
                                 pool, genotype, f_nt, f_nn
                               giving the model's mean predicted rate per pool, for ALL 72
                               pools. Predictions on the 24 annotated pools MUST be
                               out-of-fold (see `dump_pool_predictions.py --kfold`), which
                               this script enforces via `assert_crossfitted` when a
                               `train_pools` column or --train-pools is supplied.

  PROJECT (default)            No predictions yet. The CLASSICAL side is still real -- it
                               needs only annotations -- and the PPI++ side is simulated at
                               the true v1 design (het 18/18, wt 6/30) using the real
                               per-stratum pool-rate means and SDs, swept over pool-level
                               correlation r. Everything projected is labelled as such in
                               the figure. This is a power analysis, not a result.

Usage:
    python scripts/mice_behavior/ppi_report.py
    python scripts/mice_behavior/ppi_report.py --predictions results/.../pool_preds.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mice_behavior.ppi import (  # noqa: E402
    StratumData, classical_contrast, ppi_contrast, projected_variance_factor,
    assert_crossfitted,
)

LABELS = ("nt", "nn")
STRATA = ("het", "wt")
OUT_DIR = ROOT / "results" / "vision" / "mice" / "frame" / "_figures"

# Observation-level Pearson r measured on the 24-observation val split for the best
# checkpoint to date (res448_k2_ft2_d4photo). Shown on the figure as a reference marker
# ONLY -- pool-level r is a different quantity (averaging 6 observations damps noise in
# both y and f, so it is usually higher), and n=24 gives it an SE of roughly 0.2.
MEASURED_OBS_R = {"nt": 0.542, "nn": 0.352}

# ── dataviz palette (validated: CVD dE 24.7, normal-vision dE 33.6, both >= 3:1) ──────
C_CLASSICAL = "#2a78d6"
C_PPI       = "#eb6834"
INK         = "#0b0b0b"
INK_2       = "#52514e"
MUTED       = "#898781"
GRID        = "#e1e0d9"
SURFACE     = "#fcfcfb"


# ── data loading ─────────────────────────────────────────────────────────────

def load_pool_rates() -> pd.DataFrame:
    """True per-pool behaviour rate for the 24 annotated pools.

    The 288 unannotated observations carry NaN in annotations.csv (NOT 0 -- the zero-fill
    happens downstream in batch_data), so a plain groupby would silently average them away
    and report 72 'annotated' pools. Filter on experiment.csv's `annotator` instead.
    """
    ann = pd.read_csv(ROOT / "dataset" / "mice" / "v1" / "annotations.csv",
                      usecols=["observation_id", "Y_nt", "Y_nn"])
    exp = pd.read_csv(ROOT / "data" / "mice" / "v1" / "experiment.csv")
    obs = ann.groupby("observation_id")[["Y_nt", "Y_nn"]].mean().reset_index()
    obs = obs.rename(columns={"Y_nt": "nt", "Y_nn": "nn"})
    obs = obs.merge(exp[["observation_id", "pool", "genotype", "annotator"]],
                    on="observation_id")
    lab = obs[obs.annotator.notna()]
    pool = lab.groupby(["pool", "genotype"])[list(LABELS)].mean().reset_index()
    if len(pool) != 24:
        raise RuntimeError(f"expected 24 annotated pools, got {len(pool)}")
    return pool


def real_classical(pool: pd.DataFrame) -> dict:
    """Classical (labeled-only) contrast -- fully real, no model involved."""
    out = {}
    for y in LABELS:
        het = pool.loc[pool.genotype == "het", y].to_numpy()
        wt  = pool.loc[pool.genotype == "wt",  y].to_numpy()
        out[y] = classical_contrast(StratumData("het", het, het, np.zeros(0)),
                                    StratumData("wt",  wt,  wt,  np.zeros(0)))
    return out


# ── projection ───────────────────────────────────────────────────────────────

DESIGN = {"het": (18, 18), "wt": (6, 30)}     # (labeled pools, unlabeled pools)


def project_ppi_width(pool: pd.DataFrame, label: str, r: float,
                      reps: int = 600, seed: int = 0) -> tuple[float, float]:
    """Expected PPI++ CI width at pool-level correlation r, and the classical width.

    Simulates at the real design using each stratum's REAL mean and SD, so the only invented
    quantity is r. f is generated as a deliberately miscalibrated affine function of y
    (13x slope, large offset) to confirm the estimator is indifferent to that -- lambda
    absorbs it. Averaging over reps folds in the lambda-estimation penalty, which the
    closed-form `projected_variance_factor` ignores and which is NOT negligible at n=6.
    """
    rng = np.random.default_rng(seed)
    stats_by_stratum = {a: (pool.loc[pool.genotype == a, label].mean(),
                           pool.loc[pool.genotype == a, label].std(ddof=1)) for a in STRATA}
    w_ppi, w_cls = [], []
    for _ in range(reps):
        strata = {}
        for a in STRATA:
            n_l, n_u = DESIGN[a]
            mu, sd = stats_by_stratum[a]
            y = rng.normal(mu, sd, n_l + n_u)
            noise = rng.normal(0, sd * np.sqrt(max(1 / r ** 2 - 1, 0.0)), n_l + n_u)
            f = 13.0 * (y + noise) + 0.05
            strata[a] = StratumData(a, y[:n_l], f[:n_l], f[n_l:])
        w_ppi.append(ppi_contrast(strata["het"], strata["wt"]).ci_width)
        w_cls.append(classical_contrast(strata["het"], strata["wt"]).ci_width)
    return float(np.mean(w_ppi)), float(np.mean(w_cls))


def equivalent_extra_pools(w_classical: float, w_ppi: float, n_labeled: int = 24) -> float:
    """How many MORE annotated pools the classical estimator would need to match PPI's CI.

    Classical width scales ~ 1/sqrt(n), so n_eff = n * (w_cls / w_ppi)^2. This is the
    framing that means something to whoever decides annotation budget: the model is worth
    N pools of somebody's labelling time.
    """
    if w_ppi <= 0:
        return 0.0
    return n_labeled * ((w_classical / w_ppi) ** 2 - 1.0)


# ── real PPI (when predictions exist) ────────────────────────────────────────

def real_ppi(pool: pd.DataFrame, preds: pd.DataFrame, train_pools=None) -> dict:
    """PPI++ contrast from actual model predictions on all 72 pools."""
    merged = preds.merge(pool[["pool"] + list(LABELS)], on="pool", how="left")
    out = {}
    for y in LABELS:
        strata = {}
        for a in STRATA:
            sub = merged[merged.genotype == a]
            lab = sub[sub[y].notna()]
            unl = sub[sub[y].isna()]
            if train_pools is not None:
                assert_crossfitted(lab.pool.tolist(), train_pools, stratum_name=f"{a}/{y}")
            strata[a] = StratumData(a, lab[y].to_numpy(), lab[f"f_{y}"].to_numpy(),
                                    unl[f"f_{y}"].to_numpy())
        out[y] = {"ppi": ppi_contrast(strata["het"], strata["wt"]),
                  "r": {a: strata[a].r for a in STRATA}}
    return out


# ── figure ───────────────────────────────────────────────────────────────────

def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=1.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def make_figure(pool, classical, projected, out_path: Path, r_used: dict,
                real=None) -> Path:
    r_grid = projected["r_grid"]
    fig = plt.figure(figsize=(13.4, 5.8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.05, 0.7], wspace=0.36,
                          left=0.052, right=0.985, top=0.735, bottom=0.135)

    mode = "measured from predictions" if real else "projected (no predictions yet)"

    def _panel_title(ax, title, note):
        ax.set_title(title, fontsize=11.5, color=INK, fontweight="bold", loc="left",
                     pad=24)
        ax.text(0, 1.025, note, transform=ax.transAxes, fontsize=8.4, color=MUTED,
                va="bottom", ha="left")

    # ── Panel A: the forest plot -- the shrinkage itself ──────────────────────
    axA = fig.add_subplot(gs[0, 0]); _style(axA)
    rows, ytick, ylab = [], [], []
    for i, y in enumerate(reversed(LABELS)):        # nt on top, matching B and C
        base = i * 2.4
        c = classical[y]
        p = real[y]["ppi"] if real else projected["est"][y]
        rows.append((base + 0.40, c, C_CLASSICAL))
        rows.append((base - 0.40, p, C_PPI))
        ytick.append(base); ylab.append(y)

    for ypos, est, color in rows:
        lo, hi = est.ci_low * 100, est.ci_high * 100
        axA.plot([lo, hi], [ypos, ypos], color=color, lw=2.4, solid_capstyle="butt",
                 zorder=3)
        for x in (lo, hi):
            axA.plot([x, x], [ypos - 0.14, ypos + 0.14], color=color, lw=2.0, zorder=3)
        axA.plot([est.theta * 100], [ypos], "o", ms=8, color=color,
                 markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4)
        axA.text(hi + 0.05, ypos, f"{est.ci_width*100:.2f}pp", va="center", ha="left",
                 fontsize=8.6, color=color, fontweight="bold")

    axA.axvline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    axA.set_yticks(ytick); axA.set_yticklabels(ylab, fontsize=11.5, color=INK)
    axA.set_ylim(-1.35, 3.5)
    axA.set_xlim(right=axA.get_xlim()[1] + 0.30)
    axA.set_xlabel("genotype contrast   het − wt   (percentage points)", fontsize=9.5,
                   color=INK_2)
    axA.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    axA.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=C_CLASSICAL, lw=2.4, marker="o", ms=7,
                          markeredgecolor=SURFACE, markeredgewidth=1.6,
                          label="classical · 24 annotated pools"),
               plt.Line2D([], [], color=C_PPI, lw=2.4, marker="o", ms=7,
                          markeredgecolor=SURFACE, markeredgewidth=1.6,
                          label="PPI++ · all 72 pools")]
    leg = axA.legend(handles=handles, fontsize=8.4, frameon=False, loc="lower center",
                     bbox_to_anchor=(0.5, -0.015), ncol=1, handlelength=1.8)
    for t in leg.get_texts():
        t.set_color(INK_2)
    _panel_title(axA, "A · 95% interval on the genotype contrast",
                 f"dashed line = no effect · PPI++ side {mode}")

    # ── Panel B: width vs classifier quality ─────────────────────────────────
    axB = fig.add_subplot(gs[0, 1]); _style(axB)
    for y, ls in zip(LABELS, ("-", (0, (5, 2)))):
        axB.axhline(classical[y].ci_width * 100, ls=ls, color=C_CLASSICAL, lw=2.0,
                    zorder=2)
        axB.plot(r_grid, np.array(projected["width_ppi"][y]) * 100, ls=ls, color=C_PPI,
                 lw=2.2, zorder=3)
        rr = r_used[y]
        w = float(np.interp(rr, r_grid, np.array(projected["width_ppi"][y]) * 100))
        axB.plot([rr], [w], "o", ms=8, color=C_PPI, markeredgecolor=SURFACE,
                 markeredgewidth=2.0, zorder=5)
        axB.text(r_grid[-1], classical[y].ci_width * 100 + 0.022, y, fontsize=9,
                 color=C_CLASSICAL, ha="right", va="bottom", fontweight="bold")
        axB.text(r_grid[-1], np.array(projected["width_ppi"][y])[-1] * 100 - 0.025, y,
                 fontsize=9, color=C_PPI, ha="right", va="top", fontweight="bold")
    axB.set_xlabel("pool-level correlation  r  (predicted vs true rate)",
                   fontsize=9.5, color=INK_2)
    axB.set_ylabel("95% CI width (pp)", fontsize=9.5, color=INK_2)
    axB.grid(color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)
    axB.set_ylim(0, 1.52)
    handles = [plt.Line2D([], [], color=C_CLASSICAL, lw=2.0, label="classical"),
               plt.Line2D([], [], color=C_PPI, lw=2.2, label="PPI++"),
               plt.Line2D([], [], color=C_PPI, lw=0, marker="o", ms=7,
                          markeredgecolor=SURFACE, markeredgewidth=1.6,
                          label="r measured at observation\nlevel (n=24) — see caveat")]
    leg = axB.legend(handles=handles, fontsize=8.2, frameon=False, loc="lower left",
                     handlelength=1.8, borderpad=0.2, labelspacing=0.55)
    for t in leg.get_texts():
        t.set_color(INK_2)
    _panel_title(axB, "B · what a better classifier buys",
                 "solid = nt · dashed = nn")

    # ── Panel C: stat tiles, not a two-bar chart ─────────────────────────────
    axC = fig.add_subplot(gs[0, 2]); axC.set_facecolor(SURFACE); axC.axis("off")
    for i, y in enumerate(LABELS):
        top = 0.86 - i * 0.46
        v = projected["extra_pools"][y]
        w = float(np.interp(r_used[y], r_grid, projected["width_ppi"][y]))
        shrink = 100 * (1 - w / classical[y].ci_width)
        axC.add_patch(plt.Rectangle((0.02, top - 0.36), 0.96, 0.36, transform=axC.transAxes,
                                    facecolor="#f4f3ef", edgecolor=GRID, lw=1.0, zorder=1))
        axC.text(0.08, top - 0.075, y, transform=axC.transAxes, fontsize=10.5,
                 color=INK_2, va="top", fontweight="bold")
        axC.text(0.08, top - 0.145, f"+{v:.1f}", transform=axC.transAxes, fontsize=27,
                 color=C_PPI, va="top", fontweight="bold")
        axC.text(0.08, top - 0.305, "equivalent extra\nannotated pools",
                 transform=axC.transAxes, fontsize=8.4, color=INK_2, va="top",
                 linespacing=1.35)
        axC.text(0.95, top - 0.145, f"−{shrink:.1f}%", transform=axC.transAxes,
                 fontsize=12, color=INK_2, va="top", ha="right", fontweight="bold")
        axC.text(0.95, top - 0.215, "CI width", transform=axC.transAxes, fontsize=8.2,
                 color=MUTED, va="top", ha="right")
    _panel_title(axC, "C · what the model is worth",
                 "on top of the 24 pools already annotated")

    fig.suptitle("Prediction-powered inference on the mice v1 genotype contrast",
                 fontsize=13.5, color=INK, fontweight="bold", x=0.052, ha="left", y=0.965)
    fig.text(0.052, 0.905,
             "Genotype is constant within a pool, so pools are the unit: 24 annotated "
             "(18 het + 6 wt) and 48 unannotated (18 het + 30 wt).",
             fontsize=9.2, color=INK_2, ha="left")
    fig.text(0.052, 0.858,
             "The classical interval uses the 24 annotated pools only. PPI++ additionally "
             "rectifies model predictions across all 72 — unbiased for any classifier, "
             "however miscalibrated.",
             fontsize=9.2, color=INK_2, ha="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=None,
                    help="pool-level prediction csv (pool, genotype, f_nt, f_nn)")
    ap.add_argument("--train-pools", nargs="*", default=None,
                    help="pools the predictor was trained on; triggers the cross-fit guard")
    ap.add_argument("--reps", type=int, default=600, help="simulation reps per r")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "ppi_ci_shrinkage.png")
    args = ap.parse_args()

    pool = load_pool_rates()
    classical = real_classical(pool)

    print("=" * 74)
    print("CLASSICAL (labeled pools only, 18 het + 6 wt) — real data, no model")
    for y in LABELS:
        c = classical[y]
        print(f"  [{y}] ATE {c.theta*100:+.3f}pp  95% CI [{c.ci_low*100:+.3f}, "
              f"{c.ci_high*100:+.3f}]  width {c.ci_width*100:.3f}pp  "
              f"{'SIGNIFICANT' if c.significant else 'not significant'}")

    r_grid = np.round(np.arange(0.10, 0.951, 0.05), 3)
    projected = {"r_grid": r_grid, "width_ppi": {}, "extra_pools": {}, "est": {}}
    for y in LABELS:
        widths = [project_ppi_width(pool, y, float(r), reps=args.reps)[0] for r in r_grid]
        projected["width_ppi"][y] = widths
        rr = MEASURED_OBS_R[y]
        w_at = float(np.interp(rr, r_grid, widths))
        projected["extra_pools"][y] = equivalent_extra_pools(classical[y].ci_width, w_at)
        # a projected Estimate for panel A: real point estimate, projected width
        c = classical[y]
        half = w_at / 2
        projected["est"][y] = type(c)(theta=c.theta, se=half / 1.96, ci_low=c.theta - half,
                                      ci_high=c.theta + half, alpha=0.05, method="ppi++ (projected)",
                                      n_labeled=24, n_unlabeled=48)

    print("\nPPI++ (PROJECTED at the r measured on 24 val observations)")
    for y in LABELS:
        rr = MEASURED_OBS_R[y]
        w = float(np.interp(rr, r_grid, projected["width_ppi"][y]))
        c = classical[y].ci_width
        print(f"  [{y}] r={rr:.3f} -> width {w*100:.3f}pp vs classical {c*100:.3f}pp  "
              f"({100*(1-w/c):+.1f}%)  = +{projected['extra_pools'][y]:.1f} equivalent pools")
        print(f"        closed-form ceiling ignoring lambda noise: "
              f"het {projected_variance_factor(rr,18,18):.3f}  wt {projected_variance_factor(rr,6,30):.3f}")

    real = None
    if args.predictions:
        preds = pd.read_csv(args.predictions)
        real = real_ppi(pool, preds, args.train_pools)
        print("\nPPI++ (MEASURED from predictions)")
        for y in LABELS:
            e = real[y]["ppi"]; c = classical[y]
            print(f"  [{y}] ATE {e.theta*100:+.3f}pp  95% CI [{e.ci_low*100:+.3f}, "
                  f"{e.ci_high*100:+.3f}]  width {e.ci_width*100:.3f}pp "
                  f"({100*(1-e.ci_width/c.ci_width):+.1f}% vs classical)  "
                  f"lambda={e.lam}  pool-r={real[y]['r']}")

    path = make_figure(pool, classical, projected, args.out, MEASURED_OBS_R, real)
    print(f"\n  figure: {path}")

    summary = {
        "classical": {y: classical[y].as_dict() for y in LABELS},
        "projected_extra_pools": projected["extra_pools"],
        "measured_obs_r": MEASURED_OBS_R,
        "design": {a: {"labeled_pools": DESIGN[a][0], "unlabeled_pools": DESIGN[a][1]}
                   for a in STRATA},
    }
    if real:
        summary["ppi_measured"] = {y: real[y]["ppi"].as_dict() for y in LABELS}
    js = args.out.with_suffix(".json")
    with open(js, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  summary: {js}")


if __name__ == "__main__":
    main()
