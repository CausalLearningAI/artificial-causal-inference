#!/usr/bin/env python3
"""
Hugging Face Dataset Generator

Loads a Hugging Face Dataset from processed frames and annotations.
Generates dataset on-the-fly from:
- frames: dataset/{subject}/{version}/frames/full/{observation_id}/frame_*.jpg
- annotations: dataset/{subject}/{version}/annotations.csv

Usage:
    from src.dataset.get_dataset import load_dataset
    
    # Load dataset for ants v1
    dataset = load_dataset(subject="ants", version="v1")
    
    # Access samples
    sample = dataset[0]
    # {'image': PIL.Image, 'observation_id': 'a1', 'frame_idx': 0, 'T': 2, 
    #  'W_batch': 'a', 'W_position': 1, ..., 'Y_Y2F': 0, 'Y_B2F': 1}
    
    # Save to disk (optional)
    dataset.save_to_disk("dataset/subject/version/hf")
    
    # Load from disk
    dataset = load_dataset(subject="ants", version="v1", from_disk=True)
"""

import pandas as pd
import yaml
import time
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import datasets
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Image as HFImage


def load_dataset(
    subject: str,
    version: str,
    split: Optional[str] = None,
    dataset_root: Optional[Path] = None,
    config_root: Optional[Path] = None,
    from_disk: bool = False,
    cache_dir: Optional[Path] = None,
    frame_type: str = "full",
    pov_identity: str = "blue",
) -> Dataset | DatasetDict:
    """
    Load a Hugging Face Dataset for a subject/version combination.
    
    Args:
        subject: Subject type (ants, frogs, mice)
        version: Version identifier (v1, v2, etc.)
        split: Optional split name (e.g., 'train', 'test') for DatasetDict
        dataset_root: Root directory for datasets (default: ./dataset)
        config_root: Root directory for configs (default: ./configs)
        from_disk: If True, load pre-generated HF dataset from disk
        cache_dir: Cache directory for HF datasets (default: None)
        
    Returns:
        Dataset or DatasetDict with frames and annotations
    """
    # Set default paths
    if dataset_root is None:
        workspace_root = Path(__file__).parent.parent.parent
        dataset_root = workspace_root / "dataset"
    else:
        dataset_root = Path(dataset_root)
    
    if config_root is None:
        workspace_root = Path(__file__).parent.parent.parent
        config_root = workspace_root / "configs"
    else:
        config_root = Path(config_root)
    
    subject_dir = dataset_root / subject / version

    # HF metadata is view-independent — canonical path is hf/full/.
    hf_dir = subject_dir / "hf" / "full"

    annotations_csv = subject_dir / "annotations.csv"
    frames_dir = subject_dir / "frames" / frame_type

    # Try to load from disk first if requested
    if from_disk:
        if (hf_dir / "dataset_info.json").exists():
            try:
                print(f"Loading pre-generated HF dataset from {hf_dir}")
                dataset = datasets.load_from_disk(str(hf_dir))
                # Validate image decoding path early to avoid worker-time crashes.
                if len(dataset) > 0:
                    _ = dataset[0]["image"]
                dataset.subject = subject
                # Note: dataset.version is a read-only HF property; skip setting it
                return dataset
            except Exception as e:
                print(f"HF dataset at {hf_dir} is invalid ({type(e).__name__}: {e}), regenerating from annotations...")
        else:
            print(f"HF dataset not found at {hf_dir}, generating from annotations...")
    
    # Load config
    config_path = config_root / "dataset" / subject / f"{version}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"[ERROR] Config not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check if frames exist
    if not frames_dir.exists():
        raise FileNotFoundError(
            f"[ERROR] Frames directory not found: {frames_dir}\\n"
            f"Run: python src/dataset/get_frames.py experiment={subject}/{version}"
        )

    # Check if annotations exist
    if not annotations_csv.exists():
        raise FileNotFoundError(
            f"[ERROR] Annotations not found: {annotations_csv}\n"
            f"Run: python -m src.dataset.get_annotations experiment={subject}/{version}"
        )

    # Load annotations
    df = pd.read_csv(annotations_csv)

    # Remap frame_path to the requested frame_type (annotations always store "full" paths)
    if frame_type == "pov":
        # New POV layout is color-first: frames/pov/{blue|yellow}/{obs_id}/frame_*.jpg
        # Start from annotations full path and inject both frame_type and identity.
        df['frame_path'] = df['frame_path'].str.replace(
            "frames/full/", f"frames/pov/{pov_identity}/", regex=False
        )
    elif frame_type != "full":
        df['frame_path'] = df['frame_path'].str.replace(
            "frames/full/", f"frames/{frame_type}/", regex=False
        )
    df['image_path'] = df['frame_path'].apply(lambda p: str(dataset_root / p))

    # Define dataset features using config
    features = _create_features(df, config)

    # Keep only columns that are in features schema + image_path
    feature_columns = list(features.keys())
    feature_columns.remove('image')  # Will store path instead, decode on-the-fly
    columns_to_keep = feature_columns + ['image_path']
    df = df[columns_to_keep]

    # Rename image_path to image for HF Image feature (it will decode lazily)
    df = df.rename(columns={'image_path': 'image'})
    
    # Create dataset with features - HF will handle lazy image decoding automatically
    dataset = Dataset.from_pandas(df, features=features, preserve_index=False)
    # Store subject/version in metadata so _infer_subject_version can find them
    if dataset.info is not None:
        metadata = {"subject": subject, "version": version}
        if frame_type == "pov":
            metadata["pov_identity"] = pov_identity
        dataset.info.metadata = metadata
    # Apply split if requested
    if split:
        dataset_dict = dataset.train_test_split(test_size=0.2, seed=42)
        return dataset_dict

    return dataset



def _create_features(df: pd.DataFrame, config: Dict[str, Any]) -> Features:
    """
    Create HF Features schema from dataframe columns using config specifications.
    
    Args:
        df: Annotations dataframe
        config: Configuration dictionary with covariate/outcome type definitions
        
    Returns:
        Features object defining dataset schema
    """
    feature_dict = {
        'image': HFImage(),
        'observation_id': Value('string'),
        'frame_idx': Value('int64'),
    }

    # Infer treatment type from config or data
    if 'T' in df.columns:
        treatment_type = config.get('treatment', {}).get('type', None)
        if treatment_type == 'categorical':
            # Use Value('int64') for numeric codes, Value('string') for string labels.
            # ClassLabel would use the value as a direct index, which breaks
            # for non-0-indexed treatment codes (e.g. [1,2] or [2,4,6,7,8,9]).
            feature_dict['T'] = Value('string') if df['T'].dtype == 'object' else Value('int64')
        elif df['T'].dtype == 'object':
            feature_dict['T'] = Value('string')
        else:
            feature_dict['T'] = Value('int64')
    
    # Add covariates (W_*) using config datatypes
    w_cols = [c for c in df.columns if c.startswith('W_')]
    covariate_config = config.get('covariates', {})
    
    for col in w_cols:
        covariate_name = col.replace('W_', '')
        
        # Get type from config, fallback to inferring from data
        if isinstance(covariate_config, dict) and covariate_name in covariate_config:
            hf_type = covariate_config[covariate_name].get('type', 'string')
        else:
            # Fallback: infer from data
            dtype = df[col].dtype
            if dtype == 'object':
                hf_type = 'string'
            elif dtype in ['int64', 'int32']:
                hf_type = 'int64'
            elif dtype in ['float64', 'float32']:
                hf_type = 'float32'
            else:
                hf_type = 'string'
        
        # Convert config type string to HF type
        if hf_type in ['int', 'int64']:
            feature_dict[col] = Value('int64')
        elif hf_type in ['float', 'float32', 'float64']:
            feature_dict[col] = Value('float32')
        elif hf_type == 'categorical':
            unique_vals = sorted([str(x) for x in df[col].dropna().unique()])
            feature_dict[col] = ClassLabel(names=unique_vals)
        else:  # string or default
            feature_dict[col] = Value('string')
    
    # Add outcomes (Y_*)
    y_cols = [c for c in df.columns if c.startswith('Y_')]
    for col in y_cols:
        # Use Value('int64') for integer-valued outcomes.
        # ClassLabel would treat values as direct indices, breaking for any
        # non-0-indexed labels (same issue as treatment above).
        unique_vals = df[col].dropna().unique()
        if df[col].dtype in ['float64', 'float32'] and len(unique_vals) > 10:
            feature_dict[col] = Value('float32')
        else:
            feature_dict[col] = Value('int64')
    
    return Features(feature_dict)


def save_dataset(
    dataset: Dataset | DatasetDict,
    output_dir: Path
) -> Path:
    """
    Save dataset to disk.
    
    Args:
        dataset: HF Dataset or DatasetDict to save
        output_dir: Directory to save to
        
    Returns:
        Path to saved dataset
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Strip metadata field to avoid DatasetInfo compatibility issues with older datasets versions
    if hasattr(dataset, "info") and hasattr(dataset.info, "__dict__"):
        dataset.info.__dict__.pop("metadata", None)
    dataset.save_to_disk(str(output_dir), num_proc=4)
    return output_dir


def generate_and_save_dataset(
    subject: str,
    version: str,
    overwrite: bool = False,
    dataset_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    Load dataset, and save to disk in dataset/{subject}/{version}/hf format.
    
    Args:
        subject: Subject type
        version: Version identifier
        overwrite: If True, regenerate even if exists
        dataset_root: Root directory for datasets (default: ./dataset)
        
    Returns:
        Path to saved dataset, or None if skipped
    """
    if dataset_root is None:
        workspace_root = Path(__file__).parent.parent.parent
        dataset_root = workspace_root / "dataset"
    else:
        dataset_root = Path(dataset_root)
    
    output_dir = dataset_root / subject / version / "hf" / "full"
    
    # Check if already exists and skip if not overwriting
    if output_dir.exists() and not overwrite:
        print(f"[SKIP] HF dataset already generated at {output_dir}")
        return None
    
    # Load dataset
    dataset = load_dataset(
        subject=subject,
        version=version,
        dataset_root=dataset_root,
        from_disk=False
    )
    
    # Save to disk
    output_dir = save_dataset(dataset, output_dir)
    
    return output_dir


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    """Generate and save HF dataset"""
    import argparse
    
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def hydra_main(cfg: DictConfig):
        subject = cfg.subject
        version = cfg.version
        
        # Get overwrite flag from config
        overwrite = cfg.overwrite.hf
        
        # Generate and save
        output_dir = generate_and_save_dataset(
            subject=subject,
            version=version,
            overwrite=overwrite
        )
        
        if output_dir:
            print(f"Saved to {output_dir}")
    
    hydra_main()


if __name__ == "__main__":
    main()
