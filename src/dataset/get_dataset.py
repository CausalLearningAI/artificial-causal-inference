#!/usr/bin/env python3
"""
Hugging Face Dataset Generator

Loads a Hugging Face Dataset from processed frames and annotations.
Generates dataset on-the-fly from:
- frames: datasets/{subject}/{version}/frames/full/{observation_id}/frame_*.jpg
- annotations: datasets/{subject}/{version}/annotations.csv

Usage:
    from src.dataset.get_dataset import load_dataset
    
    # Load dataset for ants v1
    dataset = load_dataset(subject="ants", version="v1")
    
    # Access samples
    sample = dataset[0]
    # {'image': PIL.Image, 'observation_id': 'a1', 'frame_idx': 0, 'T': 2, 
    #  'W_batch': 'a', 'W_position': 1, ..., 'Y_Y2F': 0, 'Y_B2F': 1}
    
    # Save to disk (optional)
    dataset.save_to_disk("datasets/subject/version/hf")
    
    # Load from disk
    dataset = load_dataset(subject="ants", version="v1", from_disk=True)
"""

import pandas as pd
import yaml
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
    cache_dir: Optional[Path] = None
) -> Dataset | DatasetDict:
    """
    Load a Hugging Face Dataset for a subject/version combination.
    
    Args:
        subject: Subject type (ants, frogs, mice)
        version: Version identifier (v1, v2, etc.)
        split: Optional split name. If None, returns all data as single Dataset.
               If provided (e.g., 'train', 'test'), returns DatasetDict with splits.
        dataset_root: Root directory for datasets (default: ./datasets)
        config_root: Root directory for configs (default: ./configs)
        from_disk: If True, load pre-generated HF dataset from disk
        cache_dir: Cache directory for HF datasets (default: None)
        
    Returns:
        Dataset or DatasetDict with frames and annotations
    """
    # Set default paths
    if dataset_root is None:
        workspace_root = Path(__file__).parent.parent.parent
        dataset_root = workspace_root / "datasets"
    else:
        dataset_root = Path(dataset_root)
    
    if config_root is None:
        workspace_root = Path(__file__).parent.parent.parent
        config_root = workspace_root / "configs"
    else:
        config_root = Path(config_root)
    
    subject_dir = dataset_root / subject / version
    hf_dir = subject_dir / "hf"
    annotations_csv = subject_dir / "annotations.csv"
    frames_dir = subject_dir / "frames" / "full"
    
    # Try to load from disk first if requested
    if from_disk:
        if hf_dir.exists():
            print(f"Loading pre-generated HF dataset from {hf_dir}")
            return datasets.load_from_disk(str(hf_dir))
        else:
            print(f"HF dataset not found at {hf_dir}, generating from annotations...")
    
    # Load config
    config_path = config_root / "datasets" / subject / f"{version}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {config_path}")
    
    # Check if annotations exist
    if not annotations_csv.exists():
        raise FileNotFoundError(
            f"Annotations not found: {annotations_csv}\n"
            f"Run: python -m src.dataset.get_annotations --subject {subject} --version {version}"
        )
    
    # Check if frames exist
    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Frames directory not found: {frames_dir}\n"
            f"Run: python src/dataset/get_frames.py experiment.subject={subject} experiment.version={version}"
        )
    
    # Load annotations
    print(f"Loading annotations from {annotations_csv}")
    df = pd.read_csv(annotations_csv)
    
    # Add full image paths
    df['image_path'] = df['frame_path'].apply(lambda p: str(dataset_root / p))
    
    # Verify images exist (sample check)
    sample_size = min(10, len(df))
    missing = [p for p in df['image_path'].head(sample_size) if not Path(p).exists()]
    if missing:
        print(f"Warning: Some frames are missing (checked {sample_size} samples, found {len(missing)} missing)")
        print(f"Example missing: {missing[0]}")
    
    # Define dataset features using config
    features = _create_features(df, config)
    
    # Keep only columns that are in features schema + image_path
    feature_columns = list(features.keys())
    feature_columns.remove('image')  # Will store path instead, decode on-the-fly
    columns_to_keep = feature_columns + ['image_path']
    df = df[columns_to_keep]
    
    # Convert to HF Dataset - MUCH faster: store paths, decode images lazily on access
    print(f"Creating Hugging Face Dataset ({len(df):,} samples)...")
    
    # Rename image_path to image for HF Image feature (it will decode lazily)
    df = df.rename(columns={'image_path': 'image'})
    
    # Create dataset with features - HF will handle lazy image decoding automatically
    dataset = Dataset.from_pandas(df, features=features, preserve_index=False)
    
    # Apply split if requested
    if split:
        # Create train/test split (80/20 by default)
        # You can customize split logic here
        dataset_dict = dataset.train_test_split(test_size=0.2, seed=42)
        print(f"Created splits: train={len(dataset_dict['train'])}, test={len(dataset_dict['test'])}")
        return dataset_dict
    
    print(f"✓ Dataset created successfully")
    print(f"  Columns: {dataset.column_names}")
    
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
        'T': Value('int64'),  # Treatment as integer (can have any discrete values)
    }
    
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
        # Outcomes are typically binary or categorical
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 10:  # Categorical
            feature_dict[col] = ClassLabel(names=sorted([str(x) for x in unique_vals]))
        else:  # Continuous
            feature_dict[col] = Value('float32')
    
    return Features(feature_dict)


def save_dataset(
    dataset: Dataset | DatasetDict,
    subject: str,
    version: str,
    dataset_root: Optional[Path] = None,
    format: str = "arrow"
) -> Path:
    """
    Save dataset to disk in datasets/{subject}/{version}/hf format.
    
    Args:
        dataset: HF Dataset or DatasetDict to save
        subject: Subject type
        version: Version identifier
        dataset_root: Root directory for datasets (default: ./datasets)
        format: Format to save in ('arrow', 'parquet') - arrow is default for HF
        
    Returns:
        Path to saved dataset
    """
    if dataset_root is None:
        workspace_root = Path(__file__).parent.parent.parent
        dataset_root = workspace_root / "datasets"
    else:
        dataset_root = Path(dataset_root)
    
    output_dir = dataset_root / subject / version / "hf"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving HF dataset to {output_dir}...")
    dataset.save_to_disk(str(output_dir))
    print(f"✓ HF dataset saved to {output_dir}")
    
    return output_dir


def generate_and_save_dataset(
    subject: str,
    version: str,
    dataset_root: Optional[Path] = None,
    config_root: Optional[Path] = None
) -> Optional[Path]:
    """
    Generate and save HF dataset for a subject/version combination.
    
    Args:
        subject: Subject type (ants, frogs, mice)
        version: Version identifier (v1, v2, etc.)
        dataset_root: Root directory for datasets
        config_root: Root directory for configs
        
    Returns:
        Path to saved dataset directory, or None if skipped
    """
    # Load dataset
    dataset = load_dataset(
        subject=subject,
        version=version,
        dataset_root=dataset_root,
        config_root=config_root,
        from_disk=False
    )
    
    # Save to disk
    output_dir = save_dataset(
        dataset=dataset,
        subject=subject,
        version=version,
        dataset_root=dataset_root
    )
    
    return output_dir


def main():
    """Generate and save HF dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Hugging Face Dataset')
    parser.add_argument('--subject', type=str, default='ants', help='Subject type')
    parser.add_argument('--version', type=str, default='v1', help='Version identifier')
    parser.add_argument('--load-only', action='store_true', help='Only load, do not save')
    
    args = parser.parse_args()
    
    if args.load_only:
        # Load and display info
        dataset = load_dataset(subject=args.subject, version=args.version)
        print("\nDataset Info:")
        print(dataset)
        if isinstance(dataset, DatasetDict):
            print("\nSample from first split:")
            first_split = list(dataset.keys())[0]
            print(dataset[first_split][0])
        else:
            print("\nFirst sample:")
            print(dataset[0])
    else:
        # Generate and save
        output_dir = generate_and_save_dataset(
            subject=args.subject,
            version=args.version
        )
        print(f"\n✓ Dataset ready at: {output_dir}")


if __name__ == "__main__":
    main()
