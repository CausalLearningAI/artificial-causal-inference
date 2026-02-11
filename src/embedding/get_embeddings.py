#!/usr/bin/env python3
"""
Extract embeddings from frames and save to: dataset/{subject}/{version}/embeddings/full/{encoder}/{token}/

Usage:
    from src.embedding import extract_embeddings_to_disk
    from src.dataset.get_dataset import load_dataset
    
    dataset = load_dataset(subject='ants', version='v1')
    embeddings_path = extract_embeddings_to_disk(
        dataset,
        encoder='dinov2',
        token='class'
    )
    
    # Load later
    from src.embedding import load_embeddings_from_disk
    embeddings = load_embeddings_from_disk(
        subject='ants', version='v1',
        encoder='dinov2', token='class'
    )
"""

import torch
import numpy as np
import os
import time
from pathlib import Path
import sys
from typing import Optional, Dict, Any, Tuple, List
from tqdm import tqdm
import logging

from datasets import Dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoImageProcessor, AutoModel, 
    SiglipImageProcessor, SiglipVisionModel,
    AutoProcessor, CLIPVisionModel,
    ViTMAEModel, ResNetForImageClassification
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extract embeddings from images using pretrained ViT models."""
    
    AVAILABLE_ENCODERS = {
        'dinov2': ('facebook/dinov2-base', 'AutoModel'),
        'dinov2_large': ('facebook/dinov2-large', 'AutoModel'),
        'siglip': ('google/siglip-base-patch16-512', 'SiglipVisionModel'),
        'vit': ('google/vit-base-patch16-224', 'ViTForImageClassification'),
        'clip': ('openai/clip-vit-base-patch32', 'CLIPVisionModel'),
        'clip_large': ('openai/clip-vit-large-patch14-336', 'CLIPVisionModel'),
        'mae': ('facebook/vit-mae-large', 'ViTMAEModel'),
        'resnet': ('microsoft/resnet-50', 'ResNetForImageClassification'),
    }
    
    def __init__(
        self,
        encoder: str = 'dinov2',
        device: str = 'cuda',
        batch_size: int = 32,
        num_workers: int = 4,
        token: str = 'class',
        verbose: bool = True
    ):
        """
        Initialize embedding extractor.
        
        Args:
            encoder: Model identifier (see AVAILABLE_ENCODERS)
            device: 'cuda', 'cpu', or specific GPU device
            batch_size: Batch size for processing
            num_workers: Number of workers for data loading
            token: Token type ('class' or 'mean')
            verbose: Print progress information
        """
        if encoder not in self.AVAILABLE_ENCODERS:
            raise ValueError(
                f"Encoder '{encoder}' not supported. "
                f"Choose from: {list(self.AVAILABLE_ENCODERS.keys())}"
            )
        
        self.encoder_name = encoder
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.token = token
        self.verbose = verbose
        
        # Load model and processor
        self._load_model()
    
    def _load_model(self):
        """Load pretrained model and processor."""
        model_id, model_class = self.AVAILABLE_ENCODERS[self.encoder_name]
        
        if self.verbose:
            print(f"  Loading model: {model_id}")
        
        # Load processor
        if 'siglip' in self.encoder_name:
            self.processor = SiglipImageProcessor.from_pretrained(model_id)
        elif 'clip' in self.encoder_name:
            self.processor = AutoProcessor.from_pretrained(model_id)
        else:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
        
        # Load model
        if model_class == 'AutoModel':
            self.model = AutoModel.from_pretrained(model_id).to(self.device)
        elif model_class == 'SiglipVisionModel':
            self.model = SiglipVisionModel.from_pretrained(model_id).to(self.device)
        elif model_class == 'CLIPVisionModel':
            self.model = CLIPVisionModel.from_pretrained(model_id).to(self.device)
        elif model_class == 'ViTMAEModel':
            self.model = ViTMAEModel.from_pretrained(model_id).to(self.device)
        else:
            self.model = ResNetForImageClassification.from_pretrained(model_id).to(self.device)
        
        self.model.eval()
        self.model.requires_grad_(False)
        
        if self.verbose:
            print(f"  Device: {self.device}")
            print(f"  Batch size: {self.batch_size}, Workers: {self.num_workers}")
    
    def extract(self, images: List) -> torch.Tensor:
        """
        Extract embeddings from PIL images.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        # Preprocess
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract token
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        if self.token == 'class':
            # CLS token (first token)
            embeddings = hidden_states[:, 0]
        elif self.token == 'mean':
            # Mean pooling over all tokens except CLS
            embeddings = hidden_states[:, 1:].mean(dim=1)
        else:
            raise ValueError(f"Token '{self.token}' not supported. Use 'class' or 'mean'.")
        
        # Handle ResNet special case (no transformer tokens)
        if 'resnet' in self.encoder_name:
            embeddings = hidden_states.mean(dim=[2, 3])
        
        return embeddings.cpu()
    
    def extract_batch(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract embeddings from a DataLoader.
        
        Args:
            dataloader: DataLoader yielding batches with 'image' key
            num_samples: Total samples (for tqdm, optional)
            
        Returns:
            Tensor of shape (num_samples, embedding_dim)
        """
        all_embeddings = []
        
        with tqdm(
            total=num_samples,
            desc=f"  Extracting {self.encoder_name} ({self.token})",
            disable=not self.verbose,
            unit="frame"
        ) as pbar:
            for batch in dataloader:
                images = batch['image']
                
                embeddings = self.extract(images)
                all_embeddings.append(embeddings)
                
                pbar.update(len(images))
        
        return torch.cat(all_embeddings, dim=0)


def _infer_subject_version(dataset: Dataset, subject: Optional[str], version: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    inferred_subject = subject
    inferred_version = version

    # Try to get from attributes (for backwards compatibility)
    if inferred_subject is None and hasattr(dataset, "subject"):
        inferred_subject = getattr(dataset, "subject")
    if inferred_version is None and hasattr(dataset, "version"):
        inferred_version = getattr(dataset, "version")
    
    # Try to get from metadata (preferred method)
    if hasattr(dataset, "info") and dataset.info is not None and hasattr(dataset.info, "metadata"):
        metadata = dataset.info.metadata
        if metadata is not None:
            if inferred_subject is None and "subject" in metadata:
                inferred_subject = metadata["subject"]
            if inferred_version is None and "version" in metadata:
                inferred_version = metadata["version"]

    return inferred_subject, inferred_version


def extract_embeddings_to_disk(
    dataset: Dataset,
    encoder: str = 'dinov2',
    token: str = 'class',
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'cuda',
    output_dir: str = './dataset',
    force: bool = False,
    verbose: bool = True,
) -> str:
    """
    Extract embeddings and save to dataset/{subject}/{version}/embeddings/full/{encoder}/{token}/
    
    Args:
        dataset: HF Dataset with 'image' column
        encoder: Model to use (dinov2, siglip, clip, etc.)
        token: 'class' or 'mean'
        batch_size: Batch size
        num_workers: Number of workers
        device: Device to use
        output_dir: Root dataset directory (default: './dataset')
        force: Recompute if exists
        verbose: Print progress
        
    Returns:
        Path to saved embeddings dataset
        
    Example:
        >>> path = extract_embeddings_to_disk(
        ...     dataset, encoder='dinov2'
        ... )
    """
    subject, version = _infer_subject_version(dataset, None, None)
    if subject is None or version is None:
        raise ValueError("subject and version are required (or must be present in dataset metadata).")

    # Build output path: dataset/{subject}/{version}/embeddings/full/{encoder}/{token}/
    output_dir = Path(output_dir) / subject / version / 'embeddings' / 'full' / encoder / token
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if exists
    if output_dir.exists() and list(output_dir.glob('*')) and not force:
        if verbose:
            print(f"[SKIP] Embeddings already exist at {output_dir}")
            print(f"       Use overwrite.embeddings=true to recompute")
        return str(output_dir)
    
    if verbose:
        print(f"\nExtracting embeddings:")
        print(f"  Output: {output_dir}")
    
    # Extract embeddings
    extractor = EmbeddingExtractor(
        encoder=encoder,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        token=token,
        verbose=verbose
    )
    
    # Custom collate function to handle PIL images
    def collate_pil_images(batch):
        """Collate function that keeps PIL images as a list."""
        # batch is a list of dicts, each with keys like 'image', etc.
        # We need to keep images as a list of PIL images
        result = {}
        for key in batch[0].keys():
            if key == 'image':
                # Keep images as list of PIL images
                result[key] = [item[key] for item in batch]
            else:
                # For other keys, try default collation
                result[key] = [item[key] for item in batch]
        return result
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_pil_images,
    )
    
    extract_start = time.time()
    embeddings = extractor.extract_batch(dataloader, num_samples=len(dataset))
    extract_time = time.time() - extract_start
    
    # Create embedding dataset with metadata
    if verbose:
        print(f"  Saving embeddings...")
    
    save_start = time.time()
    col_name = f'embedding_{encoder}_{token}'
    emb_dataset = Dataset.from_dict({col_name: embeddings.tolist()})
    emb_dataset.set_format(type='torch', columns=[col_name])
    
    # Save
    emb_dataset.save_to_disk(str(output_dir))
    save_time = time.time() - save_start
    
    if verbose:
        print(f"  ✓ Embeddings saved")
        print(f"  Dimensions: {embeddings.shape}")
        print(f"  Extract time: {extract_time:.1f}s ({len(dataset)/extract_time:.1f} frames/s)")
        print(f"  Save time: {save_time:.1f}s")
    
    return str(output_dir)


def load_embeddings_from_disk(
    subject: str,
    version: str,
    encoder: str = 'dinov2',
    token: str = 'class',
    dataset_root: str = './dataset'
) -> torch.Tensor:
    """
    Load embeddings from: dataset/{subject}/{version}/embeddings/full/{encoder}/{token}/
    
    Args:
        subject: Dataset subject (e.g., 'ants', 'mice')
        version: Dataset version (e.g., 'v1')
        encoder: Model used
        token: Token type used
        dataset_root: Root dataset directory (default 'dataset')
        
    Returns:
        torch.Tensor of shape (num_samples, embedding_dim)
    """
    path = Path(dataset_root) / subject / version / 'embeddings' / 'full' / encoder / token
    
    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found at {path}")
    
    dataset = Dataset.load_from_disk(str(path))
    col_name = f'embedding_{encoder}_{token}'
    
    return torch.stack([torch.tensor(emb) for emb in dataset[col_name]])


def add_embeddings_from_disk(
    dataset: Dataset,
    subject: Optional[str] = None,
    version: Optional[str] = None,
    encoder: str = 'dinov2',
    token: str = 'class',
    dataset_root: str = './dataset',
    column_name: Optional[str] = None,
    overwrite: bool = False
) -> Dataset:
    """Attach embeddings from disk as a new column in the dataset."""
    subject, version = _infer_subject_version(dataset, subject, version)
    if subject is None or version is None:
        raise ValueError("subject and version are required (or must be present in dataset metadata).")

    col_name = column_name or f"embedding_{encoder}_{token}"
    if col_name in dataset.column_names and not overwrite:
        raise ValueError(f"Column '{col_name}' already exists. Set overwrite=True to replace it.")

    if col_name in dataset.column_names:
        dataset = dataset.remove_columns(col_name)

    embeddings = load_embeddings_from_disk(subject, version, encoder, token, dataset_root)
    dataset = dataset.add_column(col_name, embeddings.tolist())
    dataset.set_format(type='torch', columns=[col_name])
    return dataset


def load_dataset_with_embeddings(
    subject: str,
    version: str,
    encoder: str = 'dinov2',
    token: str = 'class',
    dataset_root: str = './dataset',
    from_disk: bool = False,
    cache_dir: Optional[str] = None,
    overwrite: bool = False
) -> Dataset:
    """Load dataset and attach embeddings as a new column."""
    from src.dataset.get_dataset import load_dataset

    dataset = load_dataset(
        subject=subject,
        version=version,
        from_disk=from_disk,
        dataset_root=dataset_root,
        cache_dir=cache_dir,
    )
    return add_embeddings_from_disk(
        dataset=dataset,
        subject=subject,
        version=version,
        encoder=encoder,
        token=token,
        dataset_root=dataset_root,
        overwrite=overwrite,
    )


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _load_dataset_root(cfg: "DictConfig") -> str:
    if hasattr(cfg, "paths") and hasattr(cfg.paths, "datasets_dir"):
        return cfg.paths.datasets_dir
    return "./dataset"


def _get_cfg_value(cfg: "DictConfig", name: str, default):
    return cfg.get(name, default) if hasattr(cfg, "get") else default


def _run_from_hydra(cfg: "DictConfig") -> int:
    """Main entry point when called via Hydra."""
    start_time = time.time()
    
    try:
        from src.dataset.get_dataset import load_dataset
    except ImportError as exc:
        print(f"[ERROR] Import error: {exc}")
        return 1

    subject = cfg.subject
    version = cfg.version
    encoder = _get_cfg_value(cfg, "encoder", "dinov2")
    token = _get_cfg_value(cfg, "token", "class")
    batch_size = _get_cfg_value(cfg, "batch_size", 32)
    num_workers = _get_cfg_value(cfg, "num_workers", 4)
    device = _get_cfg_value(cfg, "device", "cuda")
    overwrite = False
    if hasattr(cfg, "overwrite"):
        overwrite = getattr(cfg.overwrite, "embeddings", False)
    dataset_root = _load_dataset_root(cfg)

    # Print header
    print(f"\n{'='*70}")
    print(f"GET EMBEDDINGS: {subject}/{version}")
    print(f"{'='*70}\n")
    
    # Load dataset
    print(f"Loading dataset {subject}/{version}...")
    load_start = time.time()
    try:
        dataset = load_dataset(subject=subject, version=version, dataset_root=dataset_root)
        load_time = time.time() - load_start
        print(f"  ✓ Loaded {len(dataset):,} frames in {load_time:.1f}s")
    except Exception as exc:
        print(f"[ERROR] Failed to load dataset: {exc}")
        return 1

    # Extract embeddings
    try:
        path = extract_embeddings_to_disk(
            dataset=dataset,
            encoder=encoder,
            token=token,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            output_dir=dataset_root,
            force=overwrite,
            verbose=True,
        )
    except Exception as exc:
        print(f"\n[ERROR] Failed to extract embeddings: {exc}")
        return 1

    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ Embeddings saved to: {path}")
    print(f"Total time: {_format_time(total_time)}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    @hydra.main(version_base=None, config_path="../../configs", config_name="embedding/config")
    def _main(cfg: DictConfig) -> int:
        return _run_from_hydra(cfg)

    raise SystemExit(_main())
