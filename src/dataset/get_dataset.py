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
    dataset.save_to_disk("path/to/save")
    
    # Push to HF Hub (optional)
    dataset.push_to_hub("username/dataset-name")
"""

import pandas as pd
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
    
    subject_dir = dataset_root / subject / version
    annotations_csv = subject_dir / "annotations.csv"
    frames_dir = subject_dir / "frames" / "full"
    
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
    
    # Define dataset features
    features = _create_features(df)
    
    # Convert to HF Dataset
    print(f"Creating Hugging Face Dataset ({len(df)} samples)...")
    
    def load_image(example):
        """Load image from path."""
        example['image'] = Image.open(example['image_path']).convert('RGB')
        return example
    
    # Create dataset from dataframe
    dataset = Dataset.from_pandas(df, features=features)
    
    # Load images (lazy loading via map)
    dataset = dataset.map(load_image, remove_columns=['image_path'])
    
    # Apply split if requested
    if split:
        # Create train/test split (80/20 by default)
        # You can customize split logic here
        dataset_dict = dataset.train_test_split(test_size=0.2, seed=42)
        print(f"Created splits: train={len(dataset_dict['train'])}, test={len(dataset_dict['test'])}")
        return dataset_dict
    
    print(f"Dataset created successfully!")
    print(f"Columns: {dataset.column_names}")
    
    return dataset


def _create_features(df: pd.DataFrame) -> Features:
    """
    Create HF Features schema from dataframe columns.
    
    Args:
        df: Annotations dataframe
        
    Returns:
        Features object defining dataset schema
    """
    feature_dict = {
        'image': HFImage(),
        'observation_id': Value('string'),
        'frame_idx': Value('int64'),
        'T': ClassLabel(names=sorted([str(x) for x in df['T'].unique()])),  # Treatment as categorical
    }
    
    # Add covariates (W_*)
    w_cols = [c for c in df.columns if c.startswith('W_')]
    for col in w_cols:
        dtype = df[col].dtype
        if dtype == 'object':
            feature_dict[col] = Value('string')
        elif dtype in ['int64', 'int32']:
            feature_dict[col] = Value('int64')
        elif dtype in ['float64', 'float32']:
            feature_dict[col] = Value('float32')
        else:
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
    output_dir: Path,
    format: str = "parquet"
) -> None:
    """
    Save dataset to disk in specified format.
    
    Args:
        dataset: HF Dataset or DatasetDict to save
        output_dir: Directory to save dataset
        format: Format to save in ('parquet', 'arrow', 'csv')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "parquet":
        dataset.save_to_disk(str(output_dir))
        print(f"Dataset saved to {output_dir}")
    elif format == "arrow":
        dataset.save_to_disk(str(output_dir), max_shard_size="500MB")
        print(f"Dataset saved to {output_dir}")
    elif format == "csv":
        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                split_dataset.to_csv(str(output_dir / f"{split_name}.csv"))
        else:
            dataset.to_csv(str(output_dir / "dataset.csv"))
        print(f"Dataset saved as CSV to {output_dir}")
    else:
        raise ValueError(f"Unsupported format: {format}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load Hugging Face Dataset')
    parser.add_argument('--subject', type=str, default='ants', help='Subject type')
    parser.add_argument('--version', type=str, default='v1', help='Version identifier')
    parser.add_argument('--split', action='store_true', help='Create train/test split')
    parser.add_argument('--save', type=str, help='Save dataset to directory')
    parser.add_argument('--format', type=str, default='parquet', choices=['parquet', 'arrow', 'csv'])
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(
        subject=args.subject,
        version=args.version,
        split='train' if args.split else None
    )
    
    # Print info
    print("\nDataset Info:")
    print(dataset)
    
    if isinstance(dataset, DatasetDict):
        print("\nSample from train split:")
        print(dataset['train'][0])
    else:
        print("\nFirst sample:")
        print(dataset[0])
    
    # Save if requested
    if args.save:
        save_dataset(dataset, args.save, format=args.format)


if __name__ == "__main__":
    main()
