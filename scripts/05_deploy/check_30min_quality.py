#!/usr/bin/env python3
"""
Post-rebuild quality checks of a 30-min extension (after generate_annotations.py).

  annotations  deployed-model CSVs: rows frame_id < --identical-below must equal the archived
               10-min CSVs byte for byte; diff counts on the rows up to the old end
  predictions  per video positive rate / mean probability, minutes 10-30 vs 0-10, and the
               pooled minute-to-minute steps (the 9->10 step vs all others), for the version
               and the uniformly processed references (v6, vA)
  tracking     tracking-quality metrics (check_30min_prefix.quality) per segment, median over
               videos, version vs references; crop padding rate per segment
  embeddings   NaN rows of the live POV caches per segment
  sheet        contact sheet of the live POV crops around the seam

Usage: python scripts/05_deploy/check_30min_quality.py --version v5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "05_deploy"))

from check_30min_prefix import quality, contact_sheet, _median_dict  # noqa: E402
from src.tracking.tracker import AntTracker                           # noqa: E402

WORK = ROOT / "results" / "ppci" / "ants" / "regress_30min"
ANN = ROOT / "results" / "ppci" / "ants" / "annotations"


def prediction_stats(version: str, split: int) -> dict:
    per, minute = [], []
    for p in sorted((ANN / version).glob("*.csv")):
        d = pd.read_csv(p, usecols=["frame_id", "Y2F_prob", "B2F_prob", "Y2F", "B2F"])
        e, l = d[d.frame_id < split], d[d.frame_id >= split]
        r = {"obs": p.stem, "n_rows": len(d)}
        for o in ("Y2F", "B2F"):
            r[f"{o}_pos_early"], r[f"{o}_pos_late"] = e[o].mean(), l[o].mean()
            r[f"{o}_prob_early"], r[f"{o}_prob_late"] = e[f"{o}_prob"].mean(), l[f"{o}_prob"].mean()
        per.append(r)
        g = d.groupby(d.frame_id // 1800)[["Y2F_prob", "B2F_prob"]].mean()
        minute.append(g)
    df = pd.DataFrame(per)
    prof = pd.concat(minute).groupby(level=0).mean()
    out = {"n_videos": len(df), "n_rows_unique": sorted(df.n_rows.unique().tolist())}
    for o in ("Y2F", "B2F"):
        for m in ("pos", "prob"):
            e, l = df[f"{o}_{m}_early"], df[f"{o}_{m}_late"]
            out[f"{o}_{m}_early_mean"], out[f"{o}_{m}_late_mean"] = float(e.mean()), float(l.mean())
            out[f"{o}_{m}_late_over_early_pooled"] = float(l.mean() / e.mean())
            out[f"{o}_{m}_late_minus_early_median"] = float((l - e).median())
        steps = prof[f"{o}_prob"].diff().dropna()
        s10 = float(steps.loc[split // 1800])
        others = steps.drop(index=split // 1800).abs()
        out[f"{o}_pooled_step_at_minute_{split // 1800}"] = s10
        out[f"{o}_pooled_abs_step_other_minutes_max"] = float(others.max())
        out[f"{o}_pooled_abs_step_other_minutes_median"] = float(others.median())
        out[f"{o}_pooled_minute_profile"] = [round(float(x), 4) for x in prof[f"{o}_prob"]]
    return out


def tracking_stats(version: str, n_old: int, radius: int) -> dict:
    seg = {"0-10": [], "10-30": []}
    for p in sorted((ROOT / "dataset" / "ants" / version / "tracking").glob("*.csv")):
        df = pd.read_csv(p)
        if len(df) <= n_old:
            continue
        seg["0-10"].append(quality(df, 0, n_old, radius))
        seg["10-30"].append(quality(df, n_old, len(df), radius))
    return {k: _median_dict(v) | {"n_videos": len(v)} for k, v in seg.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", default="v5")
    ap.add_argument("--refs", nargs="*", default=["v6", "vA"])
    ap.add_argument("--n-old", type=int, default=3000)
    ap.add_argument("--identical-below", type=int, default=17976)
    ap.add_argument("--sheet-obs", nargs="*", default=["5_1_5", "5_12_7", "5_24_3", "5_8_3", "5_19_5"])
    a = ap.parse_args()
    v, n_old, split = a.version, a.n_old, a.n_old * 6
    rep = {}

    # annotations vs archived 10-min CSVs
    arch = ROOT / "archive" / "10min" / "ants" / v / "results" / "ppci" / "ants" / "annotations" / v
    n_files, n_rows, below, tail, tail_rows, missing = 0, set(), 0, 0, 0, []
    for p in sorted((ANN / v).glob("*.csv")):
        new = p.read_text().split("\n")
        n_files += 1
        n_rows.add(len(new) - 2)
        q = arch / p.name
        if not q.exists():
            missing.append(p.stem)
            continue
        old = q.read_text().split("\n")
        hdr_ok = new[0] == old[0]
        for i in range(1, len(old) - 1):
            fid = i - 1
            if fid < a.identical_below:
                below += (old[i] != new[i]) or not hdr_ok
            else:
                tail_rows += 1
                tail += old[i] != new[i]
    rep["annotations"] = {"n_csv": n_files, "rows_per_csv": sorted(n_rows),
                          f"rows_differing_frame_id_below_{a.identical_below}": below,
                          f"rows_differing_frame_id_{a.identical_below}_to_old_end": tail,
                          "rows_compared_tail": tail_rows, "csv_without_archived_counterpart": missing}
    print(json.dumps(rep["annotations"]), flush=True)

    rep["predictions"] = {r: prediction_stats(r, split) for r in [v] + a.refs}
    radius = 80
    rep["tracking"] = {r: tracking_stats(r, n_old, radius) for r in [v] + a.refs}

    # NaN embedding rows per segment (live caches)
    emb = {}
    for c in ("blue", "yellow"):
        d = ROOT / "dataset" / "ants" / v / "embeddings" / "pov" / c / "dinov2" / "class"
        e = torch.load(d / "embeddings.pt", weights_only=True)
        fi = pd.read_parquet(d / "row_keys.parquet")["frame_idx"].to_numpy()
        nan = torch.isnan(e).any(dim=1).numpy()
        zero = (e.abs().sum(dim=1) == 0).numpy()
        emb[c] = {"rows": len(e), "nan_0_10": int(nan[fi < n_old].sum()), "nan_10_30": int(nan[fi >= n_old].sum()),
                  "zero_0_10": int(zero[fi < n_old].sum()), "zero_10_30": int(zero[fi >= n_old].sum())}
        del e
    rep["embeddings"] = emb

    # contact sheet from the live crops around the seam
    frames = list(range(n_old - 4, n_old + 4))
    rows = []
    for o in a.sheet_obs:
        for c in ("blue", "yellow"):
            d = ROOT / "dataset" / "ants" / v / "frames" / "pov" / c / o
            rows.append((f"{o} {c} (live crops)", [cv2.imread(str(d / f"frame_{f:06d}.jpg")) for f in frames]))
    contact_sheet(rows, WORK / "seam_contact_sheet_live.png", frames)

    (WORK / "quality_30min.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("predictions", "embeddings")}, indent=1, default=float))


if __name__ == "__main__":
    main()
