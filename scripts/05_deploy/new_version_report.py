#!/usr/bin/env python3
"""
Deployment of the final PPCI model on a NEW ants version (inference only) and its checks.

  annotate    generate_annotations.py's CSV writer (30 fps rows, Y2F/B2F prob + 0/1), with the
              model's shape reference taken from the target version itself instead of the last
              finetune version (v4), so it does not depend on v4's dataset being on disk while
              another job rebuilds it. load_state_dict is strict, so a shape mismatch fails loud.
  tracking    tracking quality (check_30min_prefix.quality: dot detection rate, n_blobs, frozen
              centroids, >50 px jumps per 1k frames = identity-swap proxy, crop padding) per video
              and segment (minutes 0-10, 10-30), median over videos; references for comparison.
  diagnostic  deployed-model metrics on the annotated frames (compute_metrics as deploy_model's
              eval: acc, bacc, recall, precision, averaged over Y2F/B2F) plus average precision,
              overall and per treatment; same numbers for the reference versions.
  rates       predicted grooming rate per treatment from the annotation CSVs, minutes 0-10 vs
              10-30; on annotated videos also the true 0-10 rate.

Outputs: results/ppci/ants/<version>/report/<name>.json (+ per-video CSVs).

Usage:
  python scripts/05_deploy/new_version_report.py --version v4_infested annotate
  python scripts/05_deploy/new_version_report.py --version v4_infested tracking --refs v4 v5 v6 [--obs ...]
  python scripts/05_deploy/new_version_report.py --version v4_infested diagnostic --refs v4 v5
  python scripts/05_deploy/new_version_report.py --version v4_infested rates
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "05_deploy"))

DEPLOY = ROOT / "results" / "ppci" / "ants" / "performances" / "final"
ANN = ROOT / "results" / "ppci" / "ants" / "annotations"
N_SPLIT_ML = 3000          # 10 min at 5 fps
N_SPLIT_30 = 18000         # 10 min at 30 fps


def out_dir(v: str) -> Path:
    d = ROOT / "results" / "ppci" / "ants" / v / "report"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dump(v: str, name: str, obj) -> None:
    p = out_dir(v) / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=float))
    print(f"-> {p.relative_to(ROOT)}", flush=True)


# ── model ────────────────────────────────────────────────────────────────────

def load_model(ref_version: str, device):
    """generate_annotations.load_model with the shape reference taken from ref_version."""
    import torch
    from ppci.dataset import PPCIDataset
    from ppci.hparam_search import DS_KWARGS, BASE_CFG, _build_finetune_cfg
    from ppci.train import build_model
    cfg = json.loads((DEPLOY / "config.json").read_text())
    k = int(cfg.get("context_window", BASE_CFG.training.context_window))
    mode = cfg.get("context_mode", BASE_CFG.training.context_mode)
    ds_ref = PPCIDataset.from_disk("ants", ref_version, cfg["encoder"], cfg.get("token", "class"),
                                   frame_type=cfg.get("frame_type", "full"), dist_mode=cfg.get("dist_mode", "none"),
                                   n_val_videos=0, **DS_KWARGS)
    if k > 0:
        ds_ref.apply_context_window(k, mode=mode)
    model = build_model(ds_ref, _build_finetune_cfg(cfg))
    del ds_ref
    model.load_state_dict(torch.load(DEPLOY / "model.pt", map_location=device, weights_only=True))
    return model.to(device).eval(), cfg


def cmd_annotate(a) -> None:
    import torch
    import ppci.generate_annotations as ga
    ga.load_model = lambda deploy_dir, device: load_model(a.version, device)
    ga.generate_annotations_for_version(version=a.version, deploy_dir=DEPLOY,
                                        out_dir=ANN / a.version, threshold=0.5)


# ── tracking quality ─────────────────────────────────────────────────────────

def cmd_tracking(a) -> None:
    from check_30min_prefix import quality, _median_dict
    rep, per = {}, []
    for v in [a.version] + a.refs:
        seg = {"0-10": [], "10-30": []}
        files = sorted((ROOT / "dataset" / "ants" / v / "tracking").glob("*.csv"))
        if v == a.version and a.obs:
            files = [f for f in files if f.stem in set(a.obs)]
        for p in files:
            df = pd.read_csv(p)
            for name, lo, hi in (("0-10", 0, N_SPLIT_ML), ("10-30", N_SPLIT_ML, len(df))):
                if hi - lo < 2:
                    continue
                q = quality(df, lo, hi, a.radius)
                seg[name].append(q)
                per.append({"version": v, "obs": p.stem, "segment": name} | q)
        rep[v] = {k: (_median_dict(x) | {"n_videos": len(x)}) if x else {"n_videos": 0} for k, x in seg.items()}
        print(v, json.dumps({k: {m: round(rep[v][k].get(m, float("nan")), 4) for m in
                                 ("n_videos", "dot_blue_rate", "dot_yellow_rate", "n_blobs_3_rate",
                                  "mark_blue_jump_gt50_per1k", "mark_yellow_jump_gt50_per1k")}
                             for k in rep[v]}), flush=True)
    name = a.name or "tracking_quality"
    _dump(a.version, name, rep)
    pd.DataFrame(per).to_csv(out_dir(a.version) / f"{name}_per_video.csv", index=False)


# ── diagnostic metrics on annotated frames ───────────────────────────────────

def _metrics_one(v: str, model, cfg: dict, device) -> dict:
    import torch
    from sklearn.metrics import average_precision_score, balanced_accuracy_score
    from ppci.dataset import PPCIDataset
    from ppci.hparam_search import DS_KWARGS
    from ppci.train import compute_metrics
    ds = PPCIDataset.from_disk("ants", v, cfg["encoder"], cfg.get("token", "class"),
                               frame_type=cfg["frame_type"], dist_mode=cfg["dist_mode"],
                               n_val_videos=0, **DS_KWARGS)          # eval_window_only=True (as deploy eval)
    ds.apply_context_window(int(cfg["context_window"]), mode=cfg["context_mode"])
    m = compute_metrics(model, ds.X, ds.Y, device)
    ann = ds.annotated_mask.numpy()
    with torch.no_grad():
        prob = torch.cat([model.probs(ds.X[i:i + 4096].to(device)).cpu()
                          for i in range(0, len(ds.X), 4096)]).numpy()
    Y = ds.Y.numpy()
    out = {"n_frames_annotated": int(ann.sum()), "n_videos_annotated": int(len(np.unique(ds.obs_ids[ann]))),
           "compute_metrics": {k: float(x) for k, x in m.items()}}
    outcomes = ("Y2F", "B2F")

    def block(mask) -> dict:
        r = {"n_frames": int(mask.sum())}
        for j, o in enumerate(outcomes):
            y, p = Y[mask, j], prob[mask, j]
            r[f"{o}_true_rate"], r[f"{o}_pred_rate"] = float(y.mean()), float((p >= 0.5).mean())
            r[f"{o}_bacc"] = float(balanced_accuracy_score(y, p >= 0.5)) if len(np.unique(y)) > 1 else float("nan")
            r[f"{o}_ap"] = float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")
        r["bacc_mean"] = float(np.nanmean([r[f"{o}_bacc"] for o in outcomes]))
        r["ap_mean"] = float(np.nanmean([r[f"{o}_ap"] for o in outcomes]))
        return r

    out["all"] = block(ann)
    out["by_treatment"] = {str(t): block(ann & (ds.T == t)) for t in sorted(np.unique(ds.T[ann]))}
    # per-video true vs predicted positive rate (annotated window)
    vids = []
    for o_id in np.unique(ds.obs_ids[ann]):
        mk = ann & (ds.obs_ids == o_id)
        vids.append({"obs": o_id, "T": str(ds.T[mk][0]), "n": int(mk.sum())} |
                    {f"{o}_{w}": float(x) for j, o in enumerate(outcomes)
                     for w, x in (("true", Y[mk, j].mean()), ("pred", (prob[mk, j] >= 0.5).mean()))})
    del ds
    return out, vids


def cmd_diagnostic(a) -> None:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(a.version, device)
    rep = {}
    for v in [a.version] + a.refs:
        try:
            rep[v], vids = _metrics_one(v, model, cfg, device)
            pd.DataFrame(vids).to_csv(out_dir(a.version) / f"diagnostic_per_video_{v}.csv", index=False)
            print(v, json.dumps(rep[v]["compute_metrics"]), json.dumps(rep[v]["all"]), flush=True)
        except Exception as e:                       # a reference being rebuilt must not sink the run
            rep[v] = {"error": repr(e)}
            print(f"[{v}] FAILED: {e!r}", flush=True)
    rep["deployed_metrics_json"] = json.loads((DEPLOY / "metrics.json").read_text())
    _dump(a.version, "diagnostic", rep)


# ── predicted rates per treatment ────────────────────────────────────────────

def cmd_rates(a) -> None:
    exp = pd.read_csv(ROOT / "data" / "ants" / a.version / "experiment.csv").set_index("observation_id")
    rows = []
    for p in sorted((ANN / a.version).glob("*.csv")):
        d = pd.read_csv(p, usecols=["frame_id", "Y2F", "B2F", "Y2F_prob", "B2F_prob"])
        d["any"] = d["Y2F"] | d["B2F"]
        e, l = d[d.frame_id < N_SPLIT_30], d[d.frame_id >= N_SPLIT_30]
        r = {"obs": p.stem, "T": str(exp.loc[p.stem, "treatment"]), "n_rows": len(d),
             "annotated": bool(isinstance(exp.loc[p.stem, "annotation_file"], str))}
        for o in ("Y2F", "B2F", "any"):
            r[f"{o}_0_10"], r[f"{o}_10_30"] = float(e[o].mean()), float(l[o].mean())
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir(a.version) / "rates_per_video.csv", index=False)
    cols = [f"{o}_{w}" for o in ("Y2F", "B2F", "any") for w in ("0_10", "10_30")]
    by_t = df.groupby("T")[cols].mean()
    by_t["n_videos"] = df.groupby("T").size()
    for o in ("Y2F", "B2F", "any"):
        by_t[f"{o}_late_over_early"] = by_t[f"{o}_10_30"] / by_t[f"{o}_0_10"]
    rep = {"n_csv": len(df), "rows_per_csv": sorted(df.n_rows.unique().tolist()),
           "pooled": df[cols].mean().to_dict(),
           "by_treatment": by_t.round(5).to_dict(orient="index")}
    print(by_t.round(4).to_string(), flush=True)
    _dump(a.version, "rates", rep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("annotate")
    t = sub.add_parser("tracking")
    t.add_argument("--refs", nargs="*", default=["v4", "v5", "v6"])
    t.add_argument("--obs", nargs="*", default=None)
    t.add_argument("--radius", type=int, default=80)
    t.add_argument("--name", default=None)
    d = sub.add_parser("diagnostic")
    d.add_argument("--refs", nargs="*", default=["v4", "v5"])
    sub.add_parser("rates")
    a = ap.parse_args()
    {"annotate": cmd_annotate, "tracking": cmd_tracking, "diagnostic": cmd_diagnostic, "rates": cmd_rates}[a.cmd](a)


if __name__ == "__main__":
    main()
