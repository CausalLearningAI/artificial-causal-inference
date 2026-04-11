# Stage 3: Tracking Pipeline

HSV color-based ant tracking pipeline with 4 explicit steps:

0. bounds
1. tracking (+demo)
2. pov crops
3. embeddings

## Canonical launcher

Use:

```bash
bash scripts/03_tracking/launch_full.sh v2 v3 v4 v5
```

Default behavior is resume-friendly and matches the current recommended setup:
- versions: `v2 v3 v4 v5`
- bounds: `RUN_BOUNDS=auto` (skip when configs already contain bounds)
- encoders: `dinov2 dinov3`
- token: `class`
- overwrite flags for steps 1-3: `false`

Common variants:

```bash
# from config only (skip step 0)
RUN_BOUNDS=false RUN_TRACK=true RUN_POV=true RUN_EMBED=true \
	bash scripts/03_tracking/launch_full.sh v2 v3 v4 v5

# only steps 2-3 (POV + embeddings)
RUN_BOUNDS=false RUN_TRACK=false RUN_POV=true RUN_EMBED=true \
	bash scripts/03_tracking/launch_full.sh v2 v3 v4 v5

# only step 1 (tracking + demos)
RUN_BOUNDS=false RUN_TRACK=true RUN_POV=false RUN_EMBED=false \
	bash scripts/03_tracking/launch_full.sh v2 v3 v4 v5

# summary
bash scripts/03_tracking/report_summary.sh v2 v3 v4 v5
```

## Step toggles and overwrite controls

`launch_full.sh` supports independent toggles:

- `RUN_BOUNDS=true|false|auto`
- `RUN_TRACK=true|false`
- `RUN_POV=true|false`
- `RUN_EMBED=true|false`

Per-step overwrite controls:

- `OVERWRITE_BOUNDS=true|false`
- `OVERWRITE_TRACKING=true|false`
- `OVERWRITE_POV=true|false`
- `OVERWRITE_EMBEDDINGS=true|false`

Encoder selection:

- `ENCODERS="dinov2 dinov3"`
- `TOKEN=class`

## Script map

### User-facing scripts
- `launch_full.sh`: single canonical launcher for all step combinations.
- `report_summary.sh`: status + outputs + key metrics summary.

### Internal job scripts (normally not run directly)
- `job_bounds.sh`: optimize bounds for one submission.
- `job_track.sh`: per-version tracking + sampled demos.
- `job_pov.sh`: per-version POV extraction.
- `job_embed.sh`: per-version embedding extraction for one encoder/token.

## Re-run semantics (important)

- `OVERWRITE_TRACKING=false` prevents recomputing tracking CSVs.
- `OVERWRITE_POV=false` prevents recomputing existing POV crops.
- `OVERWRITE_EMBEDDINGS=false` skips existing embedding outputs.
- Demo generation is sampled (`n_sample=10` in `job_track.sh`).
- With `OVERWRITE_TRACKING=false`, existing sampled demo files are skipped (no rewrite).

## Outputs

- Tracking CSVs: `dataset/ants/{version}/tracking/*.csv`
- Demo videos: `results/tracking/ants/{version}/tracking_viz/*.mp4`
- POV frames: `dataset/ants/{version}/frames/pov/`
- POV embeddings: `dataset/ants/{version}/embeddings/pov/{encoder}/{token}/`
