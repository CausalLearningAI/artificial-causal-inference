# Data Ingestion Guide

This document outlines how to prepare and structure data for a new experiment. Follow these steps to ensure your data is compatible with the processing pipeline.

## Quick Start Checklist

For a new experiment `{subject}/{version}` (e.g., `frogs/v1`), you need:

- ✅ Source video files in `data/{subject}/{version}/observations/source/`
- ✅ Annotation CSV files in `data/{subject}/{version}/annotations/`
- ✅ Experiment metadata in `data/{subject}/{version}/experiment.csv`
- ✅ Config files in `configs/` (see [configs/README.md](../configs/README.md))

## Step-by-Step: Adding a New Experiment

### 1. Create Directory Structure

```bash
mkdir -p data/{subject}/{version}/observations/source
mkdir -p data/{subject}/{version}/annotations
```

Example for `frogs/v1`:
```bash
mkdir -p data/frogs/v1/observations/source
mkdir -p data/frogs/v1/annotations
```

### 2. Place Source Videos

Put all raw video files in `data/{subject}/{version}/observations/source/`:

```
data/frogs/v1/observations/source/
├── frog_001.mp4
├── frog_002.mp4
├── frog_003.mp4
└── ...
```

**Supported formats:** `.mp4`, `.mkv`, `.avi`, or other FFmpeg-compatible formats

**Requirements:**
- One video file per observation/recording
- Unique filename for each video (used as `observation_id`)
- Videos can have different FPS/resolution (will be standardized)

### 3. Create Annotation Files

Create one CSV file per video in `data/{subject}/{version}/annotations/`:

```
data/frogs/v1/annotations/
├── frog_001.csv
├── frog_002.csv
├── frog_003.csv
└── ...
```

**Annotation CSV Format:**

Your annotation files should contain frame-level behavior labels. The exact format is flexible (configured in `configs/dataset/{subject}/{version}.yaml`), but must include:

- **Start frame**: Frame number where behavior begins
- **End frame**: Frame number where behavior ends  
- **Behavior label**: String identifier for the behavior

**Example annotation file** (`frog_001.csv`):
```csv
# Header rows can appear here
# Metadata, etc.
-----
Start Frame, End Frame, Behavior Type
0, 150, jumping
151, 300, swimming
301, 450, jumping
...
```

**Configuration:** Specify how to parse your CSV format in `configs/dataset/{subject}/{version}.yaml`:

```yaml
annotation_format:
  skiprows: 3                    # Number of header rows to skip
  skipfooter: 1                  # Number of footer rows to skip
  columns:
    start_frame: "Start Frame"   # Your column name for start frame
    end_frame: "End Frame"       # Your column name for end frame
    outcome: "Behavior Type"     # Your column name for behavior label
```

### 4. Create Experiment Metadata CSV

Create `data/{subject}/{version}/experiment.csv` with observation-level metadata:

**Required columns:**
- `observation_id`: Unique identifier for each observation (should match video filename without extension)
- `observation_file`: Video filename (e.g., `frog_001.mp4`)
- `annotation_file`: Annotation filename (e.g., `frog_001.csv`)
- `treatment`: Treatment/intervention value
- `start_frame`: First valid frame index in the video
- `end_frame`: Last valid frame index in the video

**Additional columns (covariates):**
- Add any pre-treatment covariates as columns
- Examples: `batch`, `date`, `temperature`, `tank_id`, `sex`, `age`, etc.
- Types supported: string, int, float (specify in `configs/dataset/{subject}/{version}.yaml`)

**Example** (`experiment.csv`):
```csv
observation_id,observation_file,annotation_file,treatment,batch,tank_id,temperature,start_frame,end_frame
frog_001,frog_001.mp4,frog_001.csv,0,A,1,22.5,0,9000
frog_002,frog_002.mp4,frog_002.csv,1,A,2,22.5,0,9000
frog_003,frog_003.mp4,frog_003.csv,0,B,1,23.0,100,9100
frog_004,frog_004.mp4,frog_004.csv,1,B,2,23.0,100,9100
...
```

**Notes:**
- File paths in `observation_file` and `annotation_file` are relative to the source/annotations directories
- `start_frame` and `end_frame` define the valid time window for analysis (useful for trimming intro/outro)
- All covariate columns must be declared in `configs/dataset/{subject}/{version}.yaml`

### 5. Create Config Files

See [configs/README.md](../configs/README.md) for detailed instructions on creating:
- `configs/experiment/{subject}/{version}.yaml` - Experiment control
- `configs/data/{subject}/{version}.yaml` - Video processing settings  
- `configs/dataset/{subject}/{version}.yaml` - Treatment, covariates, outcomes

### 6. Run Processing Pipeline

Once data and configs are in place:

```bash
# Full pipeline (in scripts/01_dataset/run.sh, update to include your experiment)
bash scripts/01_dataset/run.sh

# Or run steps individually:
python src/data/standardize.py experiment={subject}/{version}     # Standardize videos
python src/dataset/get_frames.py experiment={subject}/{version}   # Extract frames
python src/dataset/get_annotations.py experiment={subject}/{version}  # Generate annotations
python src/dataset/get_dataset.py experiment={subject}/{version}  # Create HF dataset
```

## Complete Example

Here's a complete example for `frogs/v1`:

**Directory structure:**
```
data/frogs/v1/
├── observations/
│   └── source/
│       ├── frog_001.mp4
│       ├── frog_002.mp4
│       └── frog_003.mp4
├── annotations/
│   ├── frog_001.csv
│   ├── frog_002.csv
│   └── frog_003.csv
└── experiment.csv
```

**experiment.csv:**
```csv
observation_id,observation_file,annotation_file,treatment,batch,temperature,start_frame,end_frame
frog_001,frog_001.mp4,frog_001.csv,0,A,22.5,0,9000
frog_002,frog_002.mp4,frog_002.csv,1,A,22.5,0,9000
frog_003,frog_003.mp4,frog_003.csv,1,B,23.0,100,9100
```

**frog_001.csv:**
```csv
Timestamp, 2024_01_15_10_30_00
Video file, frog_001.mp4
-----
Start Frame, End Frame, Behavior Type
0, 150, jumping
151, 300, swimming
301, 450, jumping
```

After processing, the pipeline will generate:
- `data/frogs/v1/observations/full/` - Standardized videos (same FPS/resolution)
- `dataset/frogs/v1/frames/full/` - Extracted frames as JPG images
- `dataset/frogs/v1/annotations.csv` - Frame-level dataset with treatments/outcomes
- HuggingFace dataset with embeddings and features

## Directory Structure After Processing

```
data/frogs/v1/
├── observations/
│   ├── source/              # [YOU CREATE] Raw videos
│   │   ├── frog_001.mp4
│   │   └── ...
│   ├── full/                # [AUTO-GENERATED] Standardized videos
│   │   ├── frog_001.mp4
│   │   └── ...
│   └── metadata.json        # [AUTO-GENERATED] Video statistics
├── annotations/             # [YOU CREATE] Raw annotation CSVs
│   ├── frog_001.csv
│   └── ...
└── experiment.csv           # [YOU CREATE] Observation metadata
```

## Troubleshooting

**"No video files found in data/{subject}/{version}/observations/source"**
- Check that video files are in the correct directory
- Verify files have supported extension (`.mp4`, `.mkv`)
- Check `source_path` in `configs/data/{subject}/{version}.yaml`

**"Annotation file not found for {observation_id}"**
- Ensure annotation filename matches `annotation_file` column in `experiment.csv`
- Check files are in `data/{subject}/{version}/annotations/`

**"Column not found in experiment.csv"**
- Verify all covariates/treatment columns exist in `experiment.csv`
- Check column names match exactly (case-sensitive)

**"Error parsing annotation CSV"**
- Verify `annotation_format` settings in `configs/dataset/{subject}/{version}.yaml`
- Check `skiprows`/`skipfooter` values match your CSV format
- Ensure column names in `annotation_format.columns` match your CSV