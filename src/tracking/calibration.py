"""Colour-bound calibration for ant tracking.

Provides grid-search optimisation of HSV colour bounds and
path/config helpers shared by scripts and notebooks.
"""

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from omegaconf import OmegaConf

from .detection import compute_ant_mask, _find_best_color_blob

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fmt_hms(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ── Paths ────────────────────────────────────────────────────────────────────

def frames_dir(version: str, obs_id: str) -> Path:
    return PROJECT_ROOT / f"dataset/ants/{version}/frames/full/{obs_id}"


def config_path(version: str) -> Path:
    return PROJECT_ROOT / f"configs/tracking/ants/{version}.yaml"


def list_obs_ids(version: str) -> List[str]:
    d = PROJECT_ROOT / f"dataset/ants/{version}/frames/full"
    return sorted(p.name for p in d.iterdir() if p.is_dir())


# ── Config I/O ───────────────────────────────────────────────────────────────

def load_config(version: str) -> dict:
    return OmegaConf.to_container(
        OmegaConf.load(config_path(version)), resolve=True,
    )


def save_config(version: str, updates: dict) -> None:
    path = config_path(version)
    lines = path.read_text().splitlines()
    written: set = set()
    new_lines: List[str] = []
    for line in lines:
        m = re.match(r'^(\s*)(\w+)\s*:', line)
        if m and m.group(2) in updates:
            key = m.group(2)
            new_lines.append(f"{m.group(1)}{key}: {_yaml_value(updates[key])}")
            written.add(key)
        else:
            new_lines.append(line)
    for key, val in updates.items():
        if key not in written:
            new_lines.append(f"{key}: {_yaml_value(val)}")
    path.write_text("\n".join(new_lines) + "\n")
    print(f"  saved  {path.relative_to(PROJECT_ROOT)}")


def _yaml_value(val) -> str:
    if isinstance(val, (list, np.ndarray)):
        return "[" + ", ".join(str(int(x)) for x in val) + "]"
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def get_bounds_from_config(version: str) -> dict:
    """Read HSV colour bounds for blue/yellow markings from the YAML config."""
    cfg = load_config(version)
    result: dict = {}
    _DEFAULTS = {
        "blue":   {"lower": [90, 60, 50], "upper": [130, 255, 255]},
        "yellow": {"lower": [0, 80, 80],  "upper": [25, 255, 255]},
    }
    for color in ("blue", "yellow"):
        lb = cfg.get(f"{color}_marking_lb", _DEFAULTS[color]["lower"])
        ub = cfg.get(f"{color}_marking_ub", _DEFAULTS[color]["upper"])
        result[f"{color}_lower"] = np.array(lb, dtype=np.uint8)
        result[f"{color}_upper"] = np.array(ub, dtype=np.uint8)
    return result


# ── Frames & background ─────────────────────────────────────────────────────

def load_frame(version: str, obs_id: str, frame_idx: int = 100) -> np.ndarray:
    files = sorted((frames_dir(version, obs_id)).glob("frame_*.jpg"))
    if not files:
        raise FileNotFoundError(f"No frames for {version}/{obs_id}")
    path = files[min(frame_idx, len(files) - 1)]
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read {path}")
    return img


_BG_N_FRAMES = 100


def background_path(version: str, obs_id: str, quantile: float) -> Path:
    q_str = f"q{int(quantile * 100)}"
    return PROJECT_ROOT / f"dataset/ants/{version}/backgrounds/{q_str}/{obs_id}.npy"


def get_background(version: str, obs_id: str) -> np.ndarray:
    """Load cached background or compute+save using config quantile."""
    cfg = load_config(version)
    quantile = cfg.get("quantile", 0.85)
    cached = background_path(version, obs_id, quantile)
    if cached.exists():
        return np.load(cached)
    files = sorted((frames_dir(version, obs_id)).glob("frame_*.jpg"))
    indices = np.linspace(0, len(files) - 1, min(_BG_N_FRAMES, len(files)), dtype=int)
    stack = np.array([cv2.imread(str(files[i])) for i in indices])
    bg = np.quantile(stack, quantile, axis=0).astype(np.uint8)
    cached.parent.mkdir(parents=True, exist_ok=True)
    np.save(cached, bg)
    return bg


# ── Colour-bound optimisation ────────────────────────────────────────────────

def _sv_survival(
    hsv: np.ndarray, h_lo: int, h_hi: int,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """2D survival function over S and V for pixels with H in [h_lo, h_hi].

    Returns surv[256][256] where surv[s][v] = count of pixels with
    H in range, S >= s, V >= v.  O(pixels + 256^2).
    """
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sel = (H >= h_lo) & (H <= h_hi)
    if mask is not None:
        sel = sel & (mask > 0)
    hist = np.zeros((256, 256), dtype=np.int64)
    np.add.at(hist, (S[sel].ravel(), V[sel].ravel()), 1)
    surv = np.cumsum(hist[::-1], axis=0)[::-1]
    surv = np.cumsum(surv[:, ::-1], axis=1)[:, ::-1]
    return surv


def optimize_color_bounds(
    version: str,
    n_frames_per_obs: int = 10,
    max_obs: int = 15,
    verbose: bool = True,
    w_precision: float = 1.0,
    w_no_detection: float = 2.0,
    w_over_detection: float = 2.0,
) -> Dict[str, dict]:
    """Grid-search HSV lower bounds to maximise detection quality.

    Uses 2D survival functions (histogram trick) so that evaluating each
    (S_lo, V_lo) candidate is an O(1) lookup.  Proxy metric:
    n_body / n_all (fraction of matching pixels on ant bodies).

    Combined score (Stage 2):
        score = w_precision * segmentation_precision
              - w_no_detection * no_detection_rate
              - w_over_detection * over_detection_rate

    Saves best bounds to config and returns them.
    """
    cfg = load_config(version)
    min_area_body = cfg["min_area_body"]
    top_k = 500

    # ── Sample frames and precompute ────────────────────────────────────
    obs_ids = list_obs_ids(version)
    if max_obs is not None and len(obs_ids) > max_obs:
        rng = np.random.default_rng(42)
        obs_ids = sorted(rng.choice(obs_ids, max_obs, replace=False))

    frame_data: List[Tuple[np.ndarray, np.ndarray]] = []
    n_obs = len(obs_ids)
    for oi, obs_id in enumerate(obs_ids):
        if verbose:
            print(f"  {version}: loading obs {oi+1}/{n_obs}", flush=True)
            print(f"[progress] {version} loading {oi+1}/{n_obs}", flush=True)
        files = sorted(frames_dir(version, obs_id).glob("frame_*.jpg"))
        if not files:
            continue
        bg = get_background(version, obs_id)
        indices = np.linspace(
            0, len(files) - 1,
            min(n_frames_per_obs, len(files)), dtype=int,
        )
        for idx in indices:
            frame = cv2.imread(str(files[idx]))
            if frame is None:
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_nb = cv2.absdiff(frame, bg)
            ant_mask = compute_ant_mask(frame_nb, min_area=min_area_body)
            frame_data.append((hsv, ant_mask))

    n_frames = len(frame_data)
    if verbose:
        print(f"  {version}: {n_frames} frames from {n_obs} obs", flush=True)
    if n_frames == 0:
        print(f"  WARNING: no frames loaded for {version}")
        return {}

    # ── Grid definition ─────────────────────────────────────────────────
    grids = {
        "blue": {
            "h_candidates": [
                (h_lo, h_hi)
                for h_lo in range(80, 105, 5)
                for h_hi in range(115, 145, 5)
            ],
            "s_range": np.arange(20, 140, 10),
            "v_range": np.arange(20, 130, 10),
        },
        "yellow": {
            "h_candidates": [(0, h_hi) for h_hi in range(8, 28, 3)],
            "s_range": np.arange(20, 160, 10),
            "v_range": np.arange(20, 140, 10),
        },
    }

    # ── Stage 1: proxy search via survival functions ────────────────────
    min_area_color = cfg.get("min_area_color", 10)
    max_area_color = cfg.get("max_area_color", 100)

    results: Dict[str, dict] = {}
    for color, grid in grids.items():
        s_vals = grid["s_range"]
        v_vals = grid["v_range"]

        # Collect (proxy_score, h_lo, h_hi, s_lo, v_lo) across all H ranges
        all_candidates: List[Tuple[float, int, int, int, int]] = []

        h_cands = grid["h_candidates"]
        stage1_start = time.time()
        for hi, (h_lo, h_hi) in enumerate(h_cands):
            if verbose and (hi % 10 == 0 or hi == len(h_cands) - 1):
                done = hi + 1
                elapsed = time.time() - stage1_start
                eta = (elapsed / done) * (len(h_cands) - done) if done else 0.0
                print(
                    f"  {version} {color} proxy: H {hi+1}/{len(h_cands)}",
                    flush=True,
                )
                print(
                    f"    [stage1 eta={_fmt_hms(eta)} elapsed={_fmt_hms(elapsed)}]",
                    flush=True,
                )
                print(
                    f"[progress] {version} {color} proxy {hi+1}/{len(h_cands)}",
                    flush=True,
                )
            score_grid = np.zeros(
                (len(s_vals), len(v_vals)), dtype=np.float64,
            )

            for hsv_img, search_mask in frame_data:
                surv_all = _sv_survival(hsv_img, h_lo, h_hi)
                surv_body = _sv_survival(hsv_img, h_lo, h_hi, mask=search_mask)
                n_all = surv_all[s_vals[:, None], v_vals[None, :]]
                n_body = surv_body[s_vals[:, None], v_vals[None, :]]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(n_all > 0, n_body / n_all, 0.0)
                score_grid += ratio

            score_grid /= n_frames
            for si in range(len(s_vals)):
                for vi in range(len(v_vals)):
                    all_candidates.append((
                        score_grid[si, vi],
                        int(h_lo), int(h_hi), int(s_vals[si]), int(v_vals[vi]),
                    ))

        # Sort descending by proxy score (best first → early stopping works)
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        # Filter: skip candidates whose proxy score is below 20% of best
        best_proxy = all_candidates[0][0] if all_candidates else 0.0
        proxy_threshold = best_proxy * 0.2
        candidates = [c for c in all_candidates if c[0] >= proxy_threshold]
        n_after_proxy = len(candidates)
        n_skipped_proxy = len(all_candidates) - n_after_proxy

        if len(candidates) > top_k:
            candidates = candidates[:top_k]
        n_total = len(candidates)
        n_skipped_topk = n_after_proxy - n_total

        if verbose:
            print(
                f"  {version} {color}: verifying {n_total} candidates"
                f" ({n_skipped_proxy} pruned by proxy threshold"
                f", {n_skipped_topk} pruned by top-k={top_k})...",
                flush=True,
            )

        # ── Stage 2: exact verification with early stopping ─────────────
        # All detection restricted to ant_mask (foreground), matching
        # the tracker's runtime scope.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        h_img, w_img = frame_data[0][0].shape[:2]
        blob_buf = np.empty((h_img, w_img), dtype=np.uint8)

        best_score = -np.inf
        best_metrics = None
        best_lb = None
        best_ub = None
        n_early_stopped = 0

        stage2_start = time.time()
        # Rolling estimate of per-candidate duration for stabler ETA
        avg_candidate_sec: Optional[float] = None
        eta_ema_alpha = 0.2
        for ci, (_, h_lo, h_hi, s_lo, v_lo) in enumerate(candidates):
            cand_start = time.time()
            if verbose and (ci % max(n_total // 10, 1) == 0):
                done = ci + 1
                elapsed = time.time() - stage2_start
                if avg_candidate_sec is not None:
                    eta = avg_candidate_sec * (n_total - done)
                else:
                    eta = (elapsed / done) * (n_total - done) if done else 0.0
                print(
                    f"  {version} {color} verify: {ci+1}/{n_total}"
                    f"  (best={best_score:.3f}"
                    f"  early_stopped={n_early_stopped})",
                    flush=True,
                )
                print(
                    f"    [stage2 eta={_fmt_hms(eta)} elapsed={_fmt_hms(elapsed)}]",
                    flush=True,
                )
                print(
                    f"[progress] {version} {color} verify {ci+1}/{n_total}",
                    flush=True,
                )
            ratio_sum = 0.0
            n_no_detection = 0
            n_over_detection = 0
            aborted = False

            for fi, (hsv_img, ant_mask) in enumerate(frame_data):
                H, S, V = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

                # Color thresholding restricted to ant foreground
                full_px = (
                    (H >= h_lo) & (H <= h_hi) & (S >= s_lo) & (V >= v_lo)
                    & (ant_mask > 0)
                ).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    full_px, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                n_all_frame = 0
                n_valid_blobs = 0
                for cnt in contours:
                    blob_buf[:] = 0
                    cv2.drawContours(blob_buf, [cnt], -1, 255, -1)
                    cv2.morphologyEx(blob_buf, cv2.MORPH_CLOSE, kernel,
                                     dst=blob_buf)
                    a = cv2.countNonZero(blob_buf)
                    if min_area_color <= a <= max_area_color:
                        n_all_frame += a
                        n_valid_blobs += 1

                if n_valid_blobs == 0:
                    n_no_detection += 1
                elif n_valid_blobs >= 2:
                    n_over_detection += 1

                # Segmentation precision: fraction of color pixels in
                # the best blob vs all valid color pixels on ant bodies.
                if n_all_frame > 0:
                    det = _find_best_color_blob(
                        hsv_img, ant_mask, h_lo, h_hi,
                        min_s=s_lo, min_v=v_lo,
                        min_blob_area=min_area_color,
                        max_blob_area=max_area_color,
                    )
                    n_det_frame = int((det > 0).sum()) if det is not None else 0
                    ratio_sum += n_det_frame / n_all_frame

                # Early stopping: upper-bound on final score
                # Best case: remaining frames all have perfect precision,
                # no new no_det or over_det
                k = fi + 1
                if k >= 20 and best_score > -np.inf:
                    remaining = n_frames - k
                    upper_bound = (
                        w_precision * (ratio_sum + remaining) / n_frames
                        - w_no_detection * n_no_detection / n_frames
                        - w_over_detection * n_over_detection / n_frames
                    )
                    if upper_bound < best_score:
                        n_early_stopped += 1
                        aborted = True
                        break

            cand_elapsed = time.time() - cand_start
            if avg_candidate_sec is None:
                avg_candidate_sec = cand_elapsed
            else:
                avg_candidate_sec = (
                    eta_ema_alpha * cand_elapsed
                    + (1.0 - eta_ema_alpha) * avg_candidate_sec
                )

            if aborted:
                continue

            seg_precision = ratio_sum / n_frames
            no_det_rate = n_no_detection / n_frames
            over_det_rate = n_over_detection / n_frames

            score = (
                w_precision * seg_precision
                - w_no_detection * no_det_rate
                - w_over_detection * over_det_rate
            )
            if score > best_score:
                best_score = score
                best_metrics = {
                    "segmentation_precision": float(seg_precision),
                    "no_detection_rate": float(no_det_rate),
                    "over_detection_rate": float(over_det_rate),
                }
                best_lb = [h_lo, s_lo, v_lo]
                best_ub = [h_hi, 255, 255]

        if verbose:
            print(
                f"  {version} {color}: {n_early_stopped}/{n_total}"
                f" candidates early-stopped",
                flush=True,
            )
            print(f"[progress] {version} {color} done", flush=True)

        results[color] = {
            "lb": best_lb, "ub": best_ub,
            "score": float(best_score),
            **best_metrics,
        }
        if verbose:
            m = best_metrics
            print(
                f"  {version} {color}: lb={best_lb} ub={best_ub}"
                f"  score={best_score:.3f}"
                f"  (prec={m['segmentation_precision']:.3f}"
                f"  no_det={m['no_detection_rate']:.3f}"
                f"  over_det={m['over_detection_rate']:.3f})",
                flush=True,
            )

    # ── Save ────────────────────────────────────────────────────────────
    save_config(version, {
        "blue_marking_lb":   results["blue"]["lb"],
        "blue_marking_ub":   results["blue"]["ub"],
        "yellow_marking_lb": results["yellow"]["lb"],
        "yellow_marking_ub": results["yellow"]["ub"],
    })
    return results
