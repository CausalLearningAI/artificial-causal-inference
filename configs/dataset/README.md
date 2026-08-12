# Dataset Configurations

This directory contains configuration files for different subject/version combinations.

## Directory Structure

```
dataset/
├── ants/
│   ├── v1.yaml
│   └── v2.yaml
├── frogs/
│   └── v1.yaml
└── mice/
    └── v1.yaml
```

Each subject has its own directory with version-specific YAML files.

## Configuration Structure

Each config file should contain:

### 1. **Metadata**
- `subject`: Subject type (ants, frogs, mice)
- `version`: Version identifier (v1, v2, etc.)
- `description`: Human-readable description

### 2. **Frame Processing**
- `target_fps`: Target frames per second for extraction
- `output_format`: Color format for extracted frames
- `overwrite_frames`: If true, re-extract frames even if they exist
- `overwrite_annotations`: If true, regenerate annotations even if they exist

### 3. **Covariates**
List of covariate names from `experiment.csv`:
- These are features that describe the observation context
- Named as `W_{name}` in the final dataset
- May differ between subjects/versions

### 4. **Outcomes**
List of outcome variable names:
- These are labels extracted from annotation files
- Named as `Y_{name}` in the final dataset
- May differ between subjects/versions

### 5. **Annotation Format**
Specifies how to read annotation CSV files:
- `skiprows`: Number of header rows to skip
- `skipfooter`: Number of footer rows to skip
- `engine`: Parser engine (usually "python")
- `columns`: Mapping of standard names to actual column names

### 6. **Behavior Mapping** (subject-specific)
Maps behavior strings to outcome values:
- Key: behavior string from annotation file
- Value: list of outcome values (one per outcome in order)

### 7. **Treatment**
Treatment variable specification:
- `column`: Column name in experiment.csv
- `type`: categorical or continuous
- `values`: List of possible values (for categorical)

## Example

See [`ants/v1.yaml`](ants/v1.yaml) for a complete example.

## Usage

```python
from pathlib import Path
import yaml

# Load config for specific subject/version
subject = "ants"
version = "v1"
config_path = Path(f"configs/dataset/{subject}/{version}.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Access configuration
covariates = config['covariates']
outcome_map = config['outcome_mapping']
```
