"""Ant tracking module.

Provides AntTracker for HSV-based color tracking of nestmate ants,
and stateless detection building blocks in detection.py.
"""

from .tracker import AntTracker
from .detection import (
    compute_ant_mask,
    find_body_blobs,
    filter_color_pixels,
    find_mark_blob,
    match_identities,
)

__all__ = [
    "AntTracker",
    "compute_ant_mask",
    "find_body_blobs",
    "filter_color_pixels",
    "find_mark_blob",
    "match_identities",
]
