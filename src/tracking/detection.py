"""Pure per-frame detection functions for ant tracking.

Stateless building blocks — no temporal context, no I/O.  Used by both
the temporal tracker (src/tracking/tracker.py) and the notebook
visualisation utilities (notebooks/tracking/utils.py).

Pipeline
--------
  1. compute_ant_mask    — binary mask of moving objects
  2. find_body_blobs     — contours + centroids + estimated ant count
  3. filter_color_pixels — color marking mask (blue or yellow)
  4. match_identities    — assign blue / yellow / focal to bodies
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── Step 1: Ant mask ─────────────────────────────────────────────────────────

def compute_ant_mask(
    frame_nb: np.ndarray,
    min_area: int = 80,
) -> np.ndarray:
    """Binary mask of moving objects from a background-subtracted frame.

    Uses the 99th-percentile threshold (more generous than a 99.5th-percentile
    body detector) to capture full bodies including legs.  Morphological close
    (7×7 ellipse) reconnects body parts.  Blobs smaller than *min_area* are
    removed as noise.
    """
    gray = cv2.cvtColor(frame_nb, cv2.COLOR_BGR2GRAY)
    tv = max(float(np.percentile(gray, 99)), 12)
    _, mask = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Remove small noise blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean, [cnt], -1, 255, -1)
    return clean


# ── Step 2: Body blobs ───────────────────────────────────────────────────────

def find_body_blobs(
    ant_mask: np.ndarray,
    min_area: int,
    max_area: int,
    n_agents: int = 3,
) -> List[Dict]:
    """Find body blobs from the moving-object mask.

    If any blob exceeds ``max_area * n_agents`` the detection is
    considered unreliable and an empty list is returned.
    If more blobs than *n_agents*, the smallest are dropped (noise).
    Each blob gets ``n_agents_in_blob`` — an estimated agent count
    distributed proportionally by area so the total sums to *n_agents*.

    Returns a list of dicts sorted by area descending::

        centroid         : (x, y)
        area             : float
        contour          : np.ndarray
        n_agents_in_blob : int  (1 … n_agents)
    """
    contours, _ = cv2.findContours(
        ant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    blobs: List[Dict] = []
    absurd_area = max_area * n_agents
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > absurd_area:
            return []  # detection too messy, treat as no ants found
        if area < min_area:
            continue  # noise blob
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        blobs.append({
            "centroid": (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])),
            "area": float(area),
            "contour": cnt,
            "n_agents_in_blob": 1,
        })
    blobs.sort(key=lambda b: b["area"], reverse=True)
    if len(blobs) > n_agents:
        blobs = blobs[:n_agents]

    # Distribute agent count proportionally by area
    if blobs:
        total_area = sum(b["area"] for b in blobs)
        for b in blobs:
            b["n_agents_in_blob"] = max(1, round(n_agents * b["area"] / total_area))

    return blobs


# ── Step 3: Color pixel filter ───────────────────────────────────────────────

def _find_best_color_blob(
    hsv: np.ndarray,
    search_mask: np.ndarray,
    h_lo: int,
    h_hi: int,
    min_s: int,
    min_v: int,
    min_blob_area: int = 10,
    max_blob_area: int = 400,
    wrap_hue: bool = False,
) -> Optional[np.ndarray]:
    """Find the largest blob matching a colour range within *search_mask*.

    Pipeline: raw HSV filter → contours → pick largest → morph-close fill →
    area-bounds check.  Returns a filled mask of the detected blob, or None.
    """
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h_img, w_img = hsv.shape[:2]

    if wrap_hue:
        hue_ok = (H <= h_hi) | (H >= h_lo)
    else:
        hue_ok = (H >= h_lo) & (H <= h_hi)

    color_u8 = (
        hue_ok & (S >= min_s) & (V >= min_v) & (search_mask > 0)
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        color_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    best_filled = None
    best_area = 0
    for cnt in contours:
        filled = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(filled, [cnt], -1, 255, -1)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
        filled_area = cv2.countNonZero(filled)
        if min_blob_area <= filled_area <= max_blob_area and filled_area > best_area:
            best_area = filled_area
            best_filled = filled

    return best_filled


def filter_color_pixels(
    hsv: np.ndarray,
    ant_mask: np.ndarray,
    h_range: Tuple[int, int],
    min_s: int,
    min_v: int,
    min_blob_area: int = 10,
    max_blob_area: int = 400,
    wrap_hue: bool = False,
) -> np.ndarray:
    """Detect the colour marking (blue or yellow) on ant foreground.

    Searches within *ant_mask* for the largest colour-matching blob.
    Returns a filled mask of the best detection, or a zero mask if
    nothing found.
    """
    h_img, w_img = hsv.shape[:2]
    m = _find_best_color_blob(
        hsv, ant_mask, h_range[0], h_range[1],
        min_s=min_s, min_v=min_v,
        min_blob_area=min_blob_area, max_blob_area=max_blob_area,
        wrap_hue=wrap_hue,
    )
    if m is not None:
        return m
    return np.zeros((h_img, w_img), dtype=np.uint8)


# ── Step 4: Identity matching ────────────────────────────────────────────────

def _largest_blob_centroid(
    mask: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    min_area: int = 3,
) -> Optional[Tuple[int, int, float]]:
    """Centroid of the largest blob in *mask* within *radius* of *center*.

    Returns ``(x, y, area)`` or None.
    """
    h, w = mask.shape[:2]
    region = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(region, center, radius, 255, -1)
    local = cv2.bitwise_and(mask, region)
    contours, _ = cv2.findContours(
        local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), float(area))


def find_mark_blob(
    color_mask: np.ndarray,
    body_blobs: List[Dict],
) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    """Find which blob owns the colour mark.

    Computes the centroid of the detected colour mask and assigns it
    to the nearest body blob.  Returns ``(blob_index, (x, y))`` or
    ``(None, None)`` if no mark detected.
    """
    if color_mask.max() == 0:
        return None, None

    contours, _ = cv2.findContours(
        color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None
    mx, my = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    # Assign to nearest body blob
    best_bi = 0
    best_dist = float("inf")
    for bi, blob in enumerate(body_blobs):
        cx, cy = blob["centroid"]
        d = (mx - cx) ** 2 + (my - cy) ** 2
        if d < best_dist:
            best_dist = d
            best_bi = bi

    return best_bi, (mx, my)


def match_identities(
    body_blobs: List[Dict],
    blue_mask: np.ndarray,
    yellow_mask: np.ndarray,
) -> Tuple[List[str], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Assign blue / yellow / focal identities to body blobs.

    Uses ``find_mark_blob`` to get the centroid of each colour mask and
    assigns it to the nearest body blob.  Blue first (most reliable hue),
    then yellow (preferring a different body).  Focal by exclusion only
    when **both** marks are found.

    Returns
    -------
    labels     : list parallel to *body_blobs*
                 (``"blue"`` / ``"yellow"`` / ``"focal"`` / ``"unknown"``)
    blue_pos   : ``(x, y)`` of blue mark centroid, or None
    yellow_pos : ``(x, y)`` of yellow mark centroid, or None
    """
    n = len(body_blobs)
    if n == 0:
        return [], None, None

    labels: List[str] = ["unknown"] * n
    blue_pos: Optional[Tuple[int, int]] = None
    yellow_pos: Optional[Tuple[int, int]] = None

    # ── Detect mark positions ────────────────────────────────────────────
    blue_bi, blue_pos   = find_mark_blob(blue_mask,   body_blobs)
    yellow_bi, yellow_pos = find_mark_blob(yellow_mask, body_blobs)

    # ── Joint assignment (min total distance, capacity-aware) ────────────
    if blue_bi is not None and yellow_bi is not None:
        if blue_bi != yellow_bi:
            # Different blobs: trivially optimal
            labels[blue_bi]   = "blue"
            labels[yellow_bi] = "yellow"
        elif body_blobs[blue_bi]["n_agents_in_blob"] >= 2:
            # Same merged blob with room for both — valid, blue gets the label
            labels[blue_bi] = "blue"
        else:
            # Conflict: both marks nearest to a single-capacity blob.
            # Find the (bi, bj) pair minimising total squared distance that
            # is valid (bi != bj, OR the blob has capacity >= 2).
            bx, by = blue_pos
            yx, yy = yellow_pos
            best_cost = float("inf")
            best_bi, best_yj = blue_bi, yellow_bi  # fallback: same blob
            for i in range(n):
                for j in range(n):
                    if i == j and body_blobs[i]["n_agents_in_blob"] < 2:
                        continue
                    cx_i, cy_i = body_blobs[i]["centroid"]
                    cx_j, cy_j = body_blobs[j]["centroid"]
                    cost = (bx-cx_i)**2 + (by-cy_i)**2 + (yx-cx_j)**2 + (yy-cy_j)**2
                    if cost < best_cost:
                        best_cost = cost
                        best_bi, best_yj = i, j
            labels[best_bi] = "blue"
            if best_yj != best_bi:
                labels[best_yj] = "yellow"
    elif blue_bi is not None:
        labels[blue_bi] = "blue"
    elif yellow_bi is not None:
        labels[yellow_bi] = "yellow"

    # ── Focal by exclusion (only when both marks detected) ───────────────
    if blue_pos is not None and yellow_pos is not None:
        labels = ["focal" if l == "unknown" else l for l in labels]

    return labels, blue_pos, yellow_pos
