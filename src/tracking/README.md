# src/tracking module map

This folder contains the tracking core, calibration utilities, and stage-specific entrypoints.

## Core runtime

- `tracker.py`: main ant-tracking algorithm and drawing/crop helpers.
- `detection.py`: color/blob detection helpers used by tracker/calibration.

## Bounds calibration

- `calibration.py`: shared calibration logic (grid/proxy/verify search + config writes).
- `optimize_bounds.py`: CLI entrypoint that runs calibration for one or more versions.
- `diagnose_colors.py`: debugging helper for color-detection diagnostics.

## Tracking + outputs

- `get_tracking.py`: creates per-observation tracking CSVs.
- `visualize_tracking.py`: generates annotated demo videos from tracking CSVs.
- `get_pov_frames.py`: extracts POV crops around tracked ants.

## Evaluation + reporting

- `evaluate_tracking.py`: compares tracking outcomes against annotations (when available).
- `tracking_summary.py`: computes aggregate tracking metrics for summary output.

## Typical execution order

1. `optimize_bounds.py` (optional if bounds already in config)
2. `get_tracking.py`
3. `visualize_tracking.py`
4. `get_pov_frames.py`
5. `src/embedding/get_embeddings.py` with `+frame_type=pov`

For orchestration, prefer scripts under `scripts/03_tracking/`.
