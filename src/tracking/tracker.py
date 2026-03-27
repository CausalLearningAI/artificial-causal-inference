"""Core ant tracking logic.

Tracks three ants (blue-marked, yellow/orange-marked, focal/unlabeled) in a
video using background subtraction + HSV color detection.

Algorithm (per frame)
---------------------
  b <- background(v, q=0.85)

  FOR f in v:
      1. blobs         <- detect body blobs (0-3, each with n_agents_in_blob)
      2. blue/yellow   <- detect colour marks (0-2 raw positions)
      3. centroids     <- assign 3 identities to blob slots
                          (colour marks first, then nearest prev_mark)
      4. marks         <- update persistent mark positions (leash rule)
      5. emit row

Steps 1-2 are stateless (detection.py).  Steps 3-4 use temporal state.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from typing import Dict, Any, List, Optional, Tuple

from .detection import (
    compute_ant_mask,
    find_body_blobs,
    filter_color_pixels,
    find_mark_blob,
)

TrackingRow = Dict[str, Any]

_IDENTITIES = ("blue", "yellow", "focal")


class AntTracker:
    """
    Tracks three ants in a video: blue-marked nestmate, yellow/orange-marked
    nestmate, and the focal (unmarked) ant.

    Parameters
    ----------
    blue_lower, blue_upper : np.ndarray
        HSV bounds (OpenCV scale: H in [0,179], S/V in [0,255]) for the blue
        marker.
    yellow_lower, yellow_upper : np.ndarray
        HSV bounds for the yellow/orange marker.
    quantile : float
        Temporal quantile for background estimation (default 0.85).
    n_background_frames : int
        Number of evenly-spaced frames used for background estimation.
    max_dist_activity : int
        Pixel distance below which B2F / Y2F flags are set to 1.
    max_step : int
        Maximum search radius for colour-dot tracking (pixels).
        Also used as the leash distance for persistent mark positions.
    min_area_color : int
        Minimum contour area (px^2) for a colour detection to be accepted.
    max_area_color : int
        Maximum contour area (px^2) for a colour detection to be accepted.
    min_area_body : int
        Minimum contour area (px^2) for a body blob to be accepted.
    max_area_body : int
        Maximum single-ant body area (px^2).  Blobs above this are treated
        as merged (multiple ants).
    """

    def __init__(
        self,
        blue_lower: np.ndarray,
        blue_upper: np.ndarray,
        yellow_lower: np.ndarray,
        yellow_upper: np.ndarray,
        quantile: float = 0.85,
        n_background_frames: int = 40,
        max_dist_activity: int = 80,
        max_step: int = 150,
        min_area_color: int = 10,
        max_area_color: int = 400,
        min_area_body: int = 100,
        max_area_body: int = 8000,
    ):
        self.blue_lower = blue_lower
        self.blue_upper = blue_upper
        self.yellow_lower = yellow_lower
        self.yellow_upper = yellow_upper
        self.quantile = quantile
        self.n_background_frames = n_background_frames
        self.max_dist_activity = max_dist_activity
        self.max_step = max_step
        self.min_area_color = min_area_color
        self.max_area_color = max_area_color
        self.min_area_body = min_area_body
        self.max_area_body = max_area_body

        # Decompose HSV bounds for the detection API
        self._blue_h = (int(blue_lower[0]), int(blue_upper[0]))
        self._yellow_h = (int(yellow_lower[0]), int(yellow_upper[0]))
        self._blue_min_s = int(blue_lower[1])
        self._blue_min_v = int(blue_lower[2])
        self._yellow_min_s = int(yellow_lower[1])
        self._yellow_min_v = int(yellow_lower[2])

    # ------------------------------------------------------------------
    # Background estimation
    # ------------------------------------------------------------------

    def get_background(self, video_path: Path) -> np.ndarray:
        """Estimate background via temporal quantile of sampled frames."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(self.n_background_frames, frame_count)
        indices = np.linspace(0, frame_count - 1, n, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"No frames read from {video_path}")
        return np.quantile(np.array(frames), self.quantile, axis=0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Steps 1-2: per-frame detection (stateless, delegates to detection.py)
    # ------------------------------------------------------------------

    def _detect(
        self,
        frame: np.ndarray,
        background: np.ndarray,
    ) -> Tuple[
        List[Dict],
        Optional[int], Optional[Tuple[int, int]],
        Optional[int], Optional[Tuple[int, int]],
    ]:
        """Run stateless detection: body blobs + colour marks.

        Returns
        -------
        blobs      : body blob dicts from find_body_blobs
        blue_bi    : index of blob with blue mark, or None
        blue_pos   : (x, y) of blue mark, or None
        yellow_bi  : index of blob with yellow mark, or None
        yellow_pos : (x, y) of yellow mark, or None
        """
        frame_nb = cv2.absdiff(frame, background)

        # Step 1: body blobs
        ant_mask = compute_ant_mask(frame_nb, min_area=self.min_area_body)
        blobs = find_body_blobs(ant_mask, self.min_area_body, self.max_area_body)

        if not blobs:
            return blobs, None, None, None, None

        # Step 2: colour marks
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blue_mask = filter_color_pixels(
            hsv, ant_mask,
            h_range=self._blue_h,
            min_s=self._blue_min_s, min_v=self._blue_min_v,
            min_blob_area=self.min_area_color,
            max_blob_area=self.max_area_color,
        )
        yellow_mask = filter_color_pixels(
            hsv, ant_mask,
            h_range=self._yellow_h,
            min_s=self._yellow_min_s, min_v=self._yellow_min_v,
            min_blob_area=self.min_area_color,
            max_blob_area=self.max_area_color,
        )

        blue_bi, blue_pos = find_mark_blob(blue_mask, blobs)
        yellow_bi, yellow_pos = find_mark_blob(yellow_mask, blobs)

        # If both marks land on the same blob, yellow_bi == blue_bi.
        # _assign_identities handles this naturally via slots:
        #   - capacity >= 2: blue claims slot 0, yellow claims slot 1 of same blob
        #   - capacity == 1: blue claims the slot, yellow falls to temporal matching
        return blobs, blue_bi, blue_pos, yellow_bi, yellow_pos

    # ------------------------------------------------------------------
    # Step 3: assign 3 identities to blob slots
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_identities(
        blobs: List[Dict],
        blue_bi: Optional[int],
        yellow_bi: Optional[int],
        prev_marks: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Assign blue/yellow/focal to blob slots using marks + temporal.

        Each blob has ``n_agents_in_blob`` slots (a merged blob holds
        multiple ants).  Colour-marked blobs claim their slot first.
        Remaining identities are matched to remaining slots by nearest
        distance to their previous mark position.

        Returns centroid dict: ``{identity: np.ndarray([x, y])}``.
        """
        # ── Build flat slot list: (blob_index, centroid) ─────────────────
        slots: List[Tuple[int, np.ndarray]] = []
        for bi, blob in enumerate(blobs):
            centroid = np.array(blob["centroid"], dtype=float)
            for _ in range(blob["n_agents_in_blob"]):
                slots.append((bi, centroid))

        assigned: Dict[str, np.ndarray] = {}
        used_slots: set = set()

        # ── Colour marks claim their slots ───────────────────────────────
        for identity, mark_bi in [("blue", blue_bi), ("yellow", yellow_bi)]:
            if mark_bi is not None:
                for si, (bi, centroid) in enumerate(slots):
                    if bi == mark_bi and si not in used_slots:
                        assigned[identity] = centroid.copy()
                        used_slots.add(si)
                        break

        # ── Remaining identities → nearest prev_mark to remaining slots ──
        remaining_ids = [k for k in _IDENTITIES if k not in assigned]
        remaining_slots = [
            (si, centroid) for si, (_, centroid) in enumerate(slots)
            if si not in used_slots
        ]

        if remaining_ids and remaining_slots:
            slot_pts = np.array([c for _, c in remaining_slots], dtype=float)
            prev_pts = np.array(
                [prev_marks[k] for k in remaining_ids], dtype=float,
            )
            dists = cdist(slot_pts, prev_pts)  # (n_slots, n_ids)

            used_s: set = set()
            used_i: set = set()
            for flat_idx in np.argsort(dists, axis=None):
                si_local = int(flat_idx // dists.shape[1])
                ii = int(flat_idx % dists.shape[1])
                if si_local in used_s or ii in used_i:
                    continue
                assigned[remaining_ids[ii]] = remaining_slots[si_local][1].copy()
                used_s.add(si_local)
                used_i.add(ii)
                if len(used_i) == len(remaining_ids):
                    break

        # ── Edge: fewer slots than identities → keep previous ────────────
        for k in _IDENTITIES:
            if k not in assigned:
                assigned[k] = prev_marks[k].copy()

        return assigned

    # ------------------------------------------------------------------
    # Step 4: update persistent mark positions (leash rule)
    # ------------------------------------------------------------------

    @staticmethod
    def _update_mark(
        raw_pos: Optional[Tuple[int, int]],
        prev_mark: np.ndarray,
        centroid: np.ndarray,
        max_step: int,
    ) -> np.ndarray:
        """Update one mark position.

        - If detected this frame → raw position (ground truth).
        - If not detected but previous mark is within max_step of the
          current centroid → keep previous (still plausible).
        - Otherwise → snap to centroid (leash broke, reset).
        """
        if raw_pos is not None:
            return np.array(raw_pos, dtype=float)
        if float(np.linalg.norm(prev_mark - centroid)) <= max_step:
            return prev_mark.copy()
        return centroid.copy()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _find_initial_state(
        self,
        cap: cv2.VideoCapture,
        background: np.ndarray,
        h: int,
        w: int,
        max_scan: int = 300,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Scan forward to find first frame where both marks are visible.

        Returns (centroids, marks) dicts, each mapping identity → (x, y).
        """
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        default_centroids = {
            "blue":   np.array([w * 0.25, h * 0.25], dtype=float),
            "yellow": np.array([w * 0.50, h * 0.50], dtype=float),
            "focal":  np.array([w * 0.75, h * 0.75], dtype=float),
        }
        default_marks = {
            "blue":   default_centroids["blue"].copy(),
            "yellow": default_centroids["yellow"].copy(),
            "focal":  default_centroids["focal"].copy(),
        }

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(min(max_scan, frame_count)):
            ret, frame = cap.read()
            if not ret:
                break

            blobs, blue_bi, blue_pos, yellow_bi, yellow_pos = (
                self._detect(frame, background)
            )

            if blue_pos is not None and yellow_pos is not None:
                centroids = self._assign_identities(
                    blobs, blue_bi, yellow_bi, default_marks,
                )
                marks = {
                    "blue":   np.array(blue_pos, dtype=float),
                    "yellow": np.array(yellow_pos, dtype=float),
                    "focal":  centroids["focal"].copy(),
                }
                return centroids, marks

        return default_centroids, default_marks

    # ------------------------------------------------------------------
    # Main tracking loop
    # ------------------------------------------------------------------

    def track_video(
        self,
        video_path: Path,
        background: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Track all frames of a video, returning per-frame positions.

        Parameters
        ----------
        video_path : Path
            Path to the video file.
        background : np.ndarray, optional
            Pre-computed background image. If None, estimated from video.

        Returns
        -------
        pd.DataFrame with columns:
            frame_idx,
            blue_x, blue_y, yellow_x, yellow_y, focal_x, focal_y,
            mark_blue_x, mark_blue_y, mark_yellow_x, mark_yellow_y,
            raw_blue_x, raw_blue_y, raw_yellow_x, raw_yellow_y,
            B2F, Y2F, n_blobs
        """
        if background is None:
            background = self.get_background(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Scan ahead for initial identity
        prev_centroids, prev_marks = self._find_initial_state(
            cap, background, h, w,
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        rows: List[TrackingRow] = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── Steps 1-2: detect blobs + colour marks ───────────────────
            blobs, blue_bi, blue_pos, yellow_bi, yellow_pos = (
                self._detect(frame, background)
            )
            n_blobs = len(blobs)

            # ── Step 3: assign identities to blob slots ──────────────────
            centroids = self._assign_identities(
                blobs, blue_bi, yellow_bi, prev_marks,
            )

            # ── Step 4: update persistent mark positions ─────────────────
            marks = {
                "blue": self._update_mark(
                    blue_pos, prev_marks["blue"],
                    centroids["blue"], self.max_step,
                ),
                "yellow": self._update_mark(
                    yellow_pos, prev_marks["yellow"],
                    centroids["yellow"], self.max_step,
                ),
                "focal": centroids["focal"].copy(),
            }

            # ── Proximity flags ──────────────────────────────────────────
            B2F, Y2F = 0, 0
            if n_blobs == 1:
                B2F, Y2F = 1, 1
            elif n_blobs >= 2:
                d_bf = float(np.linalg.norm(
                    centroids["blue"] - centroids["focal"],
                ))
                d_yf = float(np.linalg.norm(
                    centroids["yellow"] - centroids["focal"],
                ))
                if d_bf < self.max_dist_activity:
                    B2F = 1
                if d_yf < self.max_dist_activity:
                    Y2F = 1

            rows.append({
                "frame_idx":        frame_num,
                "blue_x":           float(centroids["blue"][0]),
                "blue_y":           float(centroids["blue"][1]),
                "yellow_x":         float(centroids["yellow"][0]),
                "yellow_y":         float(centroids["yellow"][1]),
                "focal_x":          float(centroids["focal"][0]),
                "focal_y":          float(centroids["focal"][1]),
                "mark_blue_x":      float(marks["blue"][0]),
                "mark_blue_y":      float(marks["blue"][1]),
                "mark_yellow_x":    float(marks["yellow"][0]),
                "mark_yellow_y":    float(marks["yellow"][1]),
                "raw_blue_x":       float(blue_pos[0])   if blue_pos   is not None else float("nan"),
                "raw_blue_y":       float(blue_pos[1])   if blue_pos   is not None else float("nan"),
                "raw_yellow_x":     float(yellow_pos[0]) if yellow_pos is not None else float("nan"),
                "raw_yellow_y":     float(yellow_pos[1]) if yellow_pos is not None else float("nan"),
                "B2F":              B2F,
                "Y2F":              Y2F,
                "n_blobs":          n_blobs,
            })

            prev_centroids = centroids
            prev_marks = marks
            frame_num += 1

        cap.release()
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Visualization / POV utilities
    # ------------------------------------------------------------------

    @staticmethod
    def draw_tracking(
        frame: np.ndarray,
        blue_pos: Tuple[float, float],
        yellow_pos: Tuple[float, float],
        focal_pos: Tuple[float, float],
        pov_radius: int = 80,
        frame_idx: int = 0,
        n_detected: int = 3,
        raw_blue_pos: Optional[Tuple[float, float]] = None,
        raw_yellow_pos: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Draw tracking overlay on a copy of frame.

        Filled circle + POV ring: matched body position for each ant.
        Small cross: raw dot detection (only when detected this frame).
        """
        vis = frame.copy()
        C_BLUE   = (200,  80,  80)
        C_YELLOW = ( 50, 210, 210)
        C_FOCAL  = (160, 160, 160)

        for pos, label, color in [
            (blue_pos,   "blue",   C_BLUE),
            (yellow_pos, "yellow", C_YELLOW),
            (focal_pos,  "focal",  C_FOCAL),
        ]:
            cx, cy = int(pos[0]), int(pos[1])
            cv2.circle(vis, (cx, cy), 7, color, -1)
            cv2.circle(vis, (cx, cy), pov_radius, color, 2)
            cv2.putText(vis, label, (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        for raw_pos, color in [
            (raw_blue_pos,   C_BLUE),
            (raw_yellow_pos, C_YELLOW),
        ]:
            if raw_pos is not None:
                rx, ry = int(raw_pos[0]), int(raw_pos[1])
                r = 5
                cv2.line(vis, (rx - r, ry), (rx + r, ry), color, 2)
                cv2.line(vis, (rx, ry - r), (rx, ry + r), color, 2)

        cv2.putText(
            vis,
            f"frame {frame_idx}  n_ants={n_detected}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA,
        )
        return vis

    @staticmethod
    def crop_pov(frame: np.ndarray, cx: float, cy: float, radius: int) -> np.ndarray:
        """Crop a 2*radius x 2*radius square centered at (cx, cy)."""
        x, y = int(round(cx)), int(round(cy))
        h, w = frame.shape[:2]
        patch = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)

        src_top    = max(0, y - radius)
        src_bottom = min(h, y + radius)
        src_left   = max(0, x - radius)
        src_right  = min(w, x + radius)

        dst_top  = max(0, radius - y)
        dst_left = max(0, radius - x)

        patch[
            dst_top  : dst_top  + (src_bottom - src_top),
            dst_left : dst_left + (src_right  - src_left),
        ] = frame[src_top:src_bottom, src_left:src_right]
        return patch


# ---------------------------------------------------------------------------
# Auto color-bound estimation
# ---------------------------------------------------------------------------

_SEED_RANGES = {
    "blue":   {"lower": np.array([ 85,  70,  50]), "upper": np.array([135, 255, 255])},
    "yellow": {"lower": np.array([  0,  50,  40]), "upper": np.array([ 45, 255, 255])},
}

_MAX_DOT_AREA_FRACTION = 0.01


def _sample_color_pixels(
    frame_bgr: np.ndarray,
    seed_lower: np.ndarray,
    seed_upper: np.ndarray,
    min_area: int = 5,
    foreground_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Return HSV pixels of the best (most-saturated small) blob within the seed
    range, or None.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    seed_mask = cv2.inRange(hsv, seed_lower, seed_upper)
    if foreground_mask is not None:
        seed_mask = cv2.bitwise_and(seed_mask, foreground_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    max_area = frame_area * _MAX_DOT_AREA_FRACTION

    best_px, best_sat = None, -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        blob_mask = np.zeros(closed_mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
        pixels = hsv[(blob_mask == 255) & (seed_mask == 255)]
        if len(pixels) == 0:
            continue
        mean_sat = float(np.mean(pixels[:, 1]))
        if mean_sat > best_sat:
            best_sat = mean_sat
            best_px  = pixels

    return best_px


def estimate_color_bounds(
    video_path: Path,
    n_frames: int = 30,
    margin_h: int = 8,
    margin_sv: int = 30,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
) -> Dict[str, np.ndarray]:
    """
    Automatically estimate HSV bounds for blue and yellow/orange markers.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frame_count - 1, min(n_frames, frame_count), dtype=int)

    samples: Dict[str, List[np.ndarray]] = {"blue": [], "yellow": []}
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        for color, seed in _SEED_RANGES.items():
            px = _sample_color_pixels(frame, seed["lower"], seed["upper"])
            if px is not None:
                samples[color].append(px)
    cap.release()

    result: Dict[str, np.ndarray] = {}
    margins    = np.array([margin_h, margin_sv, margin_sv])
    clip_upper = np.array([179, 255, 255])

    for color in ("blue", "yellow"):
        lo_key = f"{color}_lower"
        hi_key = f"{color}_upper"
        if not samples[color]:
            result[lo_key] = _SEED_RANGES[color]["lower"].copy()
            result[hi_key] = _SEED_RANGES[color]["upper"].copy()
            continue
        all_px = np.concatenate(samples[color], axis=0)
        lo = np.percentile(all_px, percentile_low,  axis=0).astype(int)
        hi = np.percentile(all_px, percentile_high, axis=0).astype(int)
        result[lo_key] = np.clip(lo - margins, 0, clip_upper).astype(np.uint8)
        result[hi_key] = np.clip(hi + margins, 0, clip_upper).astype(np.uint8)

    return result
