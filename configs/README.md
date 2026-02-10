# Config Management Guide

## Overview

The project uses Hydra for configuration management with a streamlined approach:
- **One config = all settings** for that experiment
- Simply specify the experiment path (e.g., `ants/v1`), no need to repeat subject/version

## Structure

```
configs/
├── config.yaml              # Main config with global paths
├── experiment/              # Experiment control configs
│   ├── ants/
│   │   ├── v1.yaml         # Subject, version, and overwrite flags
│   │   └── v2.yaml
│   └── mice/
│       ├── v1.yaml
│       └── v2.yaml
├── data/                    # Video processing configs
│   ├── ants/
│   │   ├── v1.yaml         # FPS, resolution, codecs, paths
│   │   └── v2.yaml
│   └── mice/
│       ├── v1.yaml
│       └── v2.yaml
└── dataset/                # Dataset extraction configs
    ├── ants/
    │   ├── v1.yaml         # Covariates, outcomes, treatment definitions
    │   └── v2.yaml
    └── mice/
        ├── v1.yaml
        └── v2.yaml
```

## Config Purposes

- **`experiment/`**: Experiment specifications and processing control (overwrite flags, subject/version)
- **`data/`**: Video processing settings (FPS, resolution, paths, codecs)
- **`dataset/`**: Dataset structure (covariates, outcomes, treatment, annotation format)

## Usage

### Running Scripts

Simply specify the experiment path:

```bash
# Process data for ants/v1
python src/data/standardize.py experiment=ants/v1

# Extract frames for mice/v2
python src/dataset/get_frames.py experiment=mice/v2

# Override specific parameters if needed
python src/data/standardize.py experiment=ants/v1 data.target_fps=10
```

### In Bash Scripts

```bash
# Clean and simple
process_experiment "ants" "v1"
process_experiment "mice" "v2"
```

### Adding a New Experiment

To add a new experiment (e.g., `frogs/v1`), create three config files:

#### 1. Experiment Config: `configs/experiment/frogs/v1.yaml`

This controls subject/version identifiers and overwrite behavior:

```yaml
# @package _global_
defaults:
  - /data: frogs/v1
  - /dataset: frogs/v1

subject: frogs
version: v1

# Processing overwrite flags (set to true to regenerate)
overwrite:
  videos: false        # Re-process source videos to standardized format
  frames: false        # Re-extract frames from videos
  annotations: false   # Re-generate annotation CSV from raw annotations
  hf: false            # Re-generate HuggingFace dataset
```

#### 2. Data Config: `configs/data/frogs/v1.yaml`

This defines video processing settings:

```yaml
# Data configuration for frogs/v1

# Where source videos are located (before processing)
source_path: ./data/frogs/v1/observations/source

# Video processing parameters
frame_format: rgb24              # Color format (rgb24, bgr24, rgba, etc.)
target_fps: 5                    # Target frames per second for standardized videos
target_resolution:
  width: 512                     # Target video width in pixels
  height: 512                    # Target video height in pixels
video_codec: libopenh264         # Codec for output videos
bitrate: 2M                      # Video bitrate (affects quality/size)
remove_audio: true               # Whether to strip audio tracks
```

**Required fields:**
- `source_path`: Path to raw video files
- `frame_format`: Output color format
- `target_fps`: Frames per second for processing
- `target_resolution`: Output video dimensions
- `video_codec`: Video encoding codec
- `bitrate`: Video quality
- `remove_audio`: Audio handling

#### 3. Dataset Config: `configs/dataset/frogs/v1.yaml`

This defines the causal structure and data extraction:

```yaml
# Dataset configuration for frogs/v1

description: "Brief description of the experiment"

# Treatment definition (intervention variable)
treatment:
  column: treatment              # Column name in experiment.csv
  type: categorical              # categorical or continuous
  values: [0, 1, 2]             # Allowed values (for categorical only)

# Covariates (features to extract from metadata)
covariates:
  batch:
    type: string                 # Data type: string, int, float
  temperature:
    type: float
  tank_id:
    type: int

# Outcomes (behaviors to extract from annotations)
outcomes:
  - jumping                      # List of outcome variable names
  - swimming

# Map annotation labels to binary outcome vectors
outcome_mapping:
  "jump": [1, 0]                 # Maps "jump" → jumping=1, swimming=0
  "swim": [0, 1]                 # Maps "swim" → jumping=0, swimming=1
  "both": [1, 1]                 # Maps "both" → jumping=1, swimming=1

# Annotation file format (CSV parsing settings)
annotation_format:
  skiprows: 3                    # Skip first N rows (headers, metadata)
  skipfooter: 1                  # Skip last N rows (footers)
  columns:
    start_frame: "Start Frame"   # Column name for behavior start frame
    end_frame: "End Frame"       # Column name for behavior end frame
    outcome: "Behavior Type"     # Column name for behavior label
```

**Required fields:**

- **treatment**: Defines the causal intervention
  - `column`: Column name from experiment.csv
  - `type`: `categorical` or `continuous`
  - `values`: List of valid values (categorical only)

- **covariates**: Pre-treatment features from experiment.csv
  - Each covariate needs: `type` (string, int, float)

- **outcomes**: Behavior outcomes from annotation files
  - List of outcome variable names (must match `outcome_mapping`)

- **outcome_mapping**: Maps annotation labels to binary outcome vectors
  - Format: `"label": [outcome1_value, outcome2_value, ...]`
  - Vector length must match number of outcomes

- **annotation_format**: How to parse annotation CSV files
  - `skiprows`: How many header rows to skip
  - `skipfooter`: How many footer rows to skip
  - `columns`: Map standard names to actual CSV column names
    - `start_frame`: Column with behavior start frame
    - `end_frame`: Column with behavior end frame
    - `outcome`: Column with behavior label

#### 4. Use it in scripts:

```bash
# Process videos
python src/data/standardize.py experiment=frogs/v1

# Extract frames
python src/dataset/get_frames.py experiment=frogs/v1

# Generate annotations
python src/dataset/get_annotations.py experiment=frogs/v1

# Or run the full pipeline
bash scripts/01_dataset/run.sh  # (after updating the script)
```

## Benefits

✅ **No redundancy**: Specify experiment once, not multiple times  
✅ **Clear organization**: Subfolder structure mirrors data structure  
✅ **Easy to extend**: Just copy an existing config and modify  
✅ **Override flexibility**: Can still override individual parameters  
✅ **Type safety**: Hydra validates configs at runtime

## What Changed

### Before (redundant):
```bash
python src/data/standardize.py experiment=ants/v1 subject=ants version=v1
```

### After (clean):
```bash
python src/data/standardize.py experiment=ants/v1
```

## Config Contents

### Data configs (`data/`)
- **experiment**: subject and version identifiers
- **data**: all data processing settings (fps, resolution, paths, etc.)

### Dataset configs (`dataset/`)
- **treatment**: how to determine treatment assignment
- **covariates**: what features to extract from metadata
- **outcomes**: what labels to extract from annotations

Global settings (paths.data_dir, paths.cache_dir, etc.) are in [config.yaml](config.yaml).
