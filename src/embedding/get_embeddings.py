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
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor, AutoModel,
    SiglipVisionModel,
    CLIPVisionModel,
    ViTMAEModel, ResNetForImageClassification
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class _PreprocessedDataset(torch.utils.data.Dataset):
    """Wraps a HF Dataset and applies the image processor in __getitem__.

    This allows DataLoader workers to run preprocessing in parallel with
    GPU inference, eliminating the main-thread preprocessing bottleneck.
    """

    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        processed = self.processor(images=image, return_tensors='pt')
        # Remove the batch dim added by the processor (B=1 → drop it)
        return {k: v.squeeze(0) for k, v in processed.items()}


class EmbeddingExtractor:
    """Extract embeddings from images using pretrained ViT models."""

    AVAILABLE_ENCODERS = {
        # --- DINOv2 family (Meta, 2023) ---
        'dinov2':            ('facebook/dinov2-base',                          'AutoModel'),
        'dinov2_large':      ('facebook/dinov2-large',                         'AutoModel'),
        'dinov2_giant':      ('facebook/dinov2-giant',                         'AutoModel'),
        'dinov2_reg':        ('facebook/dinov2-with-registers-base',            'AutoModel'),
        'dinov2_reg_large':  ('facebook/dinov2-with-registers-large',           'AutoModel'),
        # --- DINOv3 family (Meta, Aug 2025) — trained on LVD-1689M (1.7B images) ---
        'dinov3':            ('facebook/dinov3-vitb16-pretrain-lvd1689m',       'AutoModel'),
        'dinov3_large':      ('facebook/dinov3-vitl16-pretrain-lvd1689m',       'AutoModel'),
        # --- SigLIP family (Google) ---
        'siglip':            ('google/siglip-base-patch16-512',                 'SiglipVisionModel'),
        'siglip_large':      ('google/siglip-so400m-patch14-384',               'SiglipVisionModel'),
        'siglip2':           ('google/siglip2-so400m-patch14-384',              'AutoModel'),  # Feb 2025
        # --- CLIP family (OpenAI) ---
        'clip':              ('openai/clip-vit-base-patch32',                   'CLIPVisionModel'),
        'clip_large':        ('openai/clip-vit-large-patch14-336',              'CLIPVisionModel'),
        # --- Other ---
        'vit':               ('google/vit-base-patch16-224',                    'AutoModel'),
        'mae':               ('facebook/vit-mae-large',                         'ViTMAEModel'),
        'resnet':            ('microsoft/resnet-50',                             'ResNetForImageClassification'),
        'aimv2':             ('apple/aimv2-large-patch14-224',                  'AutoModel'),  # Apple, 2025
    }

    def __init__(
        self,
        encoder: str = 'dinov2',
        device: str = 'cuda',
        batch_size: int = 32,
        num_workers: int = 4,
        token: str = 'class',
        layer: int = -1,
        verbose: bool = True
    ):
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
        self.layer = layer
        self.verbose = verbose

        self._load_model()

    def _load_model(self):
        """Load pretrained model and processor."""
        model_id, model_class = self.AVAILABLE_ENCODERS[self.encoder_name]

        if self.verbose:
            print(f"  Loading model: {model_id}")

        # Processor: use AutoImageProcessor with fast=True for all models
        # (fixes the "slow image processor" warning and is faster)
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)

        # Model
        if model_class == 'AutoModel':
            full_model = AutoModel.from_pretrained(model_id)
            # SigLIP2 (and other vision-language models) load as a full V+L model;
            # extract the vision tower to avoid "input_ids required" errors.
            if hasattr(full_model, 'vision_model'):
                self.model = full_model.vision_model.to(self.device)
            else:
                self.model = full_model.to(self.device)
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

    def _forward_tensors(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run model forward pass on already-preprocessed tensors.

        Uses fp16 autocast (GPU only) and inference_mode for maximum speed.
        Output is always float32.
        """
        pixel_values = pixel_values.to(self.device)

        use_fp16 = self.device.type == 'cuda'
        autocast_ctx = torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_fp16)

        with torch.inference_mode(), autocast_ctx:
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)

        # Extract token representation from the selected hidden layer
        if 'resnet' in self.encoder_name:
            hidden = outputs.hidden_states[self.layer]  # (B, C, H, W)
            embeddings = hidden.float().mean(dim=[2, 3])
        else:
            # hidden_states is a tuple of (B, seq_len, dim) tensors
            # Fall back to last_hidden_state if hidden_states unavailable
            if outputs.hidden_states is not None:
                hidden = outputs.hidden_states[self.layer].float()
            else:
                hidden = outputs.last_hidden_state.float()

            if self.token == 'class':
                embeddings = hidden[:, 0]           # CLS token
            elif self.token == 'mean':
                embeddings = hidden[:, 1:].mean(dim=1)  # mean over patch tokens
            else:
                raise ValueError(f"Token '{self.token}' not supported. Use 'class' or 'mean'.")

        return embeddings.cpu()

    def extract(self, images: List) -> torch.Tensor:
        """
        Extract embeddings from PIL images (convenience method).

        Args:
            images: List of PIL Image objects

        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        inputs = self.processor(images=images, return_tensors='pt')
        pixel_values = inputs['pixel_values']
        return self._forward_tensors(pixel_values)

    def extract_batch(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract embeddings from a DataLoader yielding PIL image batches.

        Args:
            dataloader: DataLoader yielding batches with 'image' key (PIL images)
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

    def extract_batch_to_file(
        self,
        hf_dataset,
        output_file: Path,
        num_samples: int,
        embedding_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract embeddings directly to a memory-mapped .npy file.

        Preprocessing runs in DataLoader workers (parallel to GPU inference)
        for maximum throughput. Only the current batch is kept in memory.

        Uses an atomic write pattern: data is written to ``output_file.tmp``
        and only renamed to ``output_file`` after the full extraction succeeds.
        A killed job therefore leaves a ``.tmp`` file (which is cleaned up on
        the next run) rather than a silently-corrupt final file.

        Args:
            hf_dataset: HuggingFace Dataset with 'image' column
            output_file: Path to save embeddings (.npy file)
            num_samples: Total number of samples
            embedding_dim: Embedding dimension (inferred from first batch if None)

        Returns:
            Memory-mapped array of shape (num_samples, embedding_dim)
        """
        # Write to a temp file so a killed job never leaves a silently-corrupt
        # final file.  The .tmp is cleaned up at the start of the next run.
        tmp_file = Path(str(output_file) + ".tmp")
        tmp_file.unlink(missing_ok=True)  # remove any leftover from a prior killed job

        pre_ds = _PreprocessedDataset(hf_dataset, self.processor)

        dataloader = DataLoader(
            pre_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            shuffle=False,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

        embeddings_mmap = None
        current_idx = 0

        with tqdm(
            total=num_samples,
            desc=f"  Extracting {self.encoder_name} ({self.token})",
            disable=not self.verbose,
            unit="frame"
        ) as pbar:
            for batch in dataloader:
                pixel_values = batch['pixel_values']
                embeddings = self._forward_tensors(pixel_values)

                if embeddings_mmap is None:
                    if embedding_dim is None:
                        embedding_dim = embeddings.shape[1]
                    embeddings_mmap = np.memmap(
                        tmp_file,           # <-- write to .tmp, not final path
                        dtype='float32',
                        mode='w+',
                        shape=(num_samples, embedding_dim)
                    )

                batch_len = len(embeddings)
                embeddings_mmap[current_idx:current_idx + batch_len] = embeddings.numpy()
                current_idx += batch_len
                pbar.update(batch_len)

                if current_idx % (self.batch_size * 10) == 0:
                    embeddings_mmap.flush()

        if embeddings_mmap is not None:
            embeddings_mmap.flush()

        # Atomically promote the temp file to the final path now that extraction
        # is complete.  On POSIX this rename is atomic.
        tmp_file.rename(output_file)

        # Re-open final file as read-only memmap and return it.
        return np.memmap(output_file, dtype='float32', mode='r',
                         shape=(num_samples, embedding_dim))


def _infer_subject_version(dataset: Dataset, subject: Optional[str], version: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    inferred_subject = subject
    inferred_version = version

    if inferred_subject is None and hasattr(dataset, "subject"):
        inferred_subject = getattr(dataset, "subject")
    if inferred_version is None and hasattr(dataset, "version"):
        inferred_version = getattr(dataset, "version")

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
    layer: int = -1,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'cuda',
    output_dir: str = './dataset',
    force: bool = False,
    verbose: bool = True,
    frame_type: str = 'full',
    pov_identity: str = 'blue',
) -> str:
    """
    Extract embeddings and save to dataset/{subject}/{version}/embeddings/full/{encoder}/{token}/

    Args:
        dataset: HF Dataset with 'image' column
        encoder: Model to use (see EmbeddingExtractor.AVAILABLE_ENCODERS)
        token: 'class' or 'mean'
        batch_size: Batch size
        num_workers: Number of DataLoader workers (preprocessing runs in these)
        device: Device to use
        output_dir: Root dataset directory (default: './dataset')
        force: Recompute if exists
        verbose: Print progress

    Returns:
        Path to saved embeddings directory
    """
    subject, version = _infer_subject_version(dataset, None, None)
    if subject is None or version is None:
        raise ValueError("subject and version are required (or must be present in dataset metadata).")

    token_dir = token if layer == -1 else f"{token}_l{layer}"
    output_dir = Path(output_dir) / subject / version / 'embeddings' / frame_type
    if frame_type == 'pov':
        if pov_identity not in {'blue', 'yellow'}:
            raise ValueError(f"Invalid pov_identity='{pov_identity}'. Use 'blue' or 'yellow'.")
        output_dir = output_dir / pov_identity
    output_dir = output_dir / encoder / token_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_file_check = output_dir / "embeddings.npy"
    pt_file_check  = output_dir / "embeddings.pt"
    tmp_file_check = output_dir / "embeddings.npy.tmp"

    # Remove any leftover temp file from a previously killed job.
    # Its presence means extraction never completed, so we re-extract.
    if tmp_file_check.exists():
        tmp_file_check.unlink()
        if verbose:
            print(f"[CLEANUP] Removed leftover embeddings.npy.tmp at {output_dir} "
                  f"(previous job was killed) — re-extracting.")

    if output_dir.exists() and list(output_dir.glob('*')) and not force:
        # Fast-path: .npy exists but .pt is missing — convert without re-extracting
        if npy_file_check.exists() and not pt_file_check.exists():
            if verbose:
                print(f"[CONVERT] embeddings.npy found but embeddings.pt missing — converting...")
            try:
                raw = np.load(npy_file_check, allow_pickle=True)
                if raw.dtype == object:
                    arr = np.array(raw.tolist(), dtype=np.float32)
                else:
                    arr = raw.astype(np.float32)
            except Exception:
                file_size = npy_file_check.stat().st_size
                num_samples = len(dataset)
                embedding_dim = file_size // 4 // num_samples
                arr = np.array(
                    np.memmap(npy_file_check, dtype='float32', mode='r', shape=(num_samples, embedding_dim)),
                    dtype=np.float32
                )
            emb = torch.from_numpy(arr)
            n_zero = int((emb.abs().sum(dim=1) == 0).sum())
            if n_zero > 0:
                zero_pct = 100.0 * n_zero / len(emb)
                npy_file_check.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Conversion aborted: {n_zero:,}/{len(emb):,} rows ({zero_pct:.1f}%) "
                    f"are all-zeros in {npy_file_check} — the original extraction was "
                    f"incomplete. Corrupt .npy deleted; re-run with overwrite.embeddings=true."
                )
            torch.save(emb, pt_file_check)
            if verbose:
                print(f"  ✓ embeddings.pt saved ({pt_file_check})")
            return str(output_dir)
        # Validate existing .pt for corruption (zero rows from a previous incomplete run).
        if pt_file_check.exists():
            emb_check = torch.load(pt_file_check, weights_only=True)
            n_zero = int((emb_check.float().abs().sum(dim=1) == 0).sum())
            if n_zero > 0:
                zero_pct = 100.0 * n_zero / len(emb_check)
                pt_file_check.unlink(missing_ok=True)
                npy_file_check.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Existing embeddings are corrupt: {n_zero:,}/{len(emb_check):,} rows "
                    f"({zero_pct:.1f}%) are all-zeros — a previous extraction job was "
                    f"killed before completion. Corrupt files deleted; re-run to extract."
                )
            del emb_check
        if verbose:
            print(f"[SKIP] Embeddings already exist at {output_dir}")
            print(f"       Use overwrite.embeddings=true to recompute")
        return str(output_dir)

    if verbose:
        print(f"\nExtracting embeddings:")
        print(f"  Output: {output_dir}")

    extractor = EmbeddingExtractor(
        encoder=encoder,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        token=token,
        layer=layer,
        verbose=verbose
    )

    # Extract embeddings directly to disk
    extract_start = time.time()
    npy_file = output_dir / "embeddings.npy"
    embeddings_mmap = extractor.extract_batch_to_file(
        dataset,
        npy_file,
        num_samples=len(dataset)
    )
    extract_time = time.time() - extract_start

    if verbose:
        print(f"  ✓ Embeddings extracted to disk")
        print(f"  Dimensions: {embeddings_mmap.shape}")
        print(f"  Extract time: {extract_time:.1f}s ({len(dataset)/extract_time:.1f} frames/s)")

    # Save .pt first — embeddings.npy is raw memmap binary (no numpy header)
    n_samples, emb_dim = embeddings_mmap.shape
    pt_file = output_dir / "embeddings.pt"
    if verbose:
        print(f"  Saving embeddings.pt...")
    arr = np.array(
        np.memmap(npy_file, dtype='float32', mode='r', shape=(n_samples, emb_dim)),
        dtype=np.float32,
    )
    emb_tensor = torch.from_numpy(arr)
    torch.save(emb_tensor, pt_file)

    # Validate: any all-zero rows indicate the job was killed before completion.
    n_zero = int((emb_tensor.abs().sum(dim=1) == 0).sum())
    if n_zero > 0:
        zero_pct = 100.0 * n_zero / n_samples
        # Delete the corrupt files so a re-run will redo extraction (not skip).
        pt_file.unlink(missing_ok=True)
        npy_file.unlink(missing_ok=True)
        raise RuntimeError(
            f"Embedding extraction incomplete: {n_zero:,}/{n_samples:,} rows "
            f"({zero_pct:.1f}%) are all-zeros — the job was likely killed before "
            f"completion. Corrupt files deleted; re-run to extract again."
        )
    del arr, emb_tensor

    if verbose:
        print(f"  ✓ embeddings.pt saved")
        print(f"  Creating HuggingFace Dataset format...")

    save_start = time.time()
    col_name = f'embedding_{encoder}_{token}'

    # Read via memmap to avoid materialising Python lists (avoids ~14 GB RAM spike)
    emb_np = np.array(
        np.memmap(npy_file, dtype='float32', mode='r', shape=(n_samples, emb_dim)),
        dtype=np.float32,
    )
    emb_dataset = Dataset.from_dict({col_name: emb_np})
    emb_dataset.set_format(type='torch', columns=[col_name])
    emb_dataset.save_to_disk(str(output_dir / "dataset"))
    del emb_np
    del embeddings_mmap
    save_time = time.time() - save_start

    if verbose:
        print(f"  ✓ HuggingFace Dataset saved")
        print(f"  Save time: {save_time:.1f}s")

    return str(output_dir)


def load_embeddings_from_disk(
    subject: str,
    version: str,
    encoder: str = 'dinov2',
    token: str = 'class',
    layer: int = -1,
    dataset_root: str = './dataset',
    frame_type: str = 'full',
    pov_identity: str = 'blue',
) -> torch.Tensor:
    """
    Load embeddings from: dataset/{subject}/{version}/embeddings/{frame_type}/{encoder}/{token}/

    Args:
        subject: Dataset subject (e.g., 'ants', 'mice')
        version: Dataset version (e.g., 'v1')
        encoder: Model used
        token: Token type used
        dataset_root: Root dataset directory (default 'dataset')
        frame_type: 'full' or 'pov' (default 'full')

    Returns:
        torch.Tensor of shape (num_samples, embedding_dim)
    """
    token_dir = token if layer == -1 else f"{token}_l{layer}"
    path = Path(dataset_root) / subject / version / 'embeddings' / frame_type
    if frame_type == 'pov':
        path = path / pov_identity
    path = path / encoder / token_dir

    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found at {path}")

    pt_file = path / "embeddings.pt"
    if pt_file.exists():
        return torch.load(pt_file, weights_only=True)

    npy_file = path / "embeddings.npy"
    if npy_file.exists():
        try:
            embeddings_np = np.load(npy_file, mmap_mode='r')
            return torch.from_numpy(np.array(embeddings_np, dtype=np.float32))
        except Exception:
            pass
        try:
            embeddings_np = np.load(npy_file, allow_pickle=True)
            return torch.tensor(np.array(embeddings_np.tolist(), dtype=np.float32))
        except Exception:
            pass

    dataset_path = path / "dataset"
    if dataset_path.exists():
        dataset = Dataset.load_from_disk(str(dataset_path))
    else:
        dataset = Dataset.load_from_disk(str(path))

    col_name = f'embedding_{encoder}_{token}'
    col_data = dataset[col_name]
    if isinstance(col_data, torch.Tensor):
        return col_data.clone().detach().float()
    return torch.from_numpy(np.array(col_data, dtype=np.float32))


def add_embeddings_from_disk(
    dataset: Dataset,
    subject: Optional[str] = None,
    version: Optional[str] = None,
    encoder: str = 'dinov2',
    token: str = 'class',
    layer: int = -1,
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

    embeddings = load_embeddings_from_disk(subject, version, encoder, token, layer, dataset_root)
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
    encoder     = _get_cfg_value(cfg, "encoder",     "dinov2")
    token       = _get_cfg_value(cfg, "token",       "class")
    layer       = _get_cfg_value(cfg, "layer",       -1)
    batch_size  = _get_cfg_value(cfg, "batch_size",  32)
    num_workers = _get_cfg_value(cfg, "num_workers", 4)
    device      = _get_cfg_value(cfg, "device",      "cuda")
    frame_type  = _get_cfg_value(cfg, "frame_type",  "full")
    pov_identity = _get_cfg_value(cfg, "pov_identity", "blue")
    overwrite = False
    if hasattr(cfg, "overwrite"):
        overwrite = getattr(cfg.overwrite, "embeddings", False)
    dataset_root = _load_dataset_root(cfg)

    print(f"\n{'='*70}")
    if frame_type == 'pov':
        print(f"GET EMBEDDINGS: {subject}/{version}  frame_type={frame_type} ({pov_identity})")
    else:
        print(f"GET EMBEDDINGS: {subject}/{version}  frame_type={frame_type}")
    print(f"{'='*70}\n")

    print(f"Loading dataset {subject}/{version} (frame_type={frame_type})...")
    load_start = time.time()
    try:
        dataset = load_dataset(
            subject=subject, version=version,
            dataset_root=dataset_root, frame_type=frame_type, pov_identity=pov_identity,
        )
        load_time = time.time() - load_start
        print(f"  ✓ Loaded {len(dataset):,} frames in {load_time:.1f}s")
    except Exception as exc:
        print(f"[ERROR] Failed to load dataset: {exc}")
        return 1

    try:
        path = extract_embeddings_to_disk(
            dataset=dataset,
            encoder=encoder,
            token=token,
            layer=layer,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            output_dir=dataset_root,
            force=overwrite,
            verbose=True,
            frame_type=frame_type,
            pov_identity=pov_identity,
        )
    except Exception as exc:
        print(f"\n[ERROR] Failed to extract embeddings: {exc}")
        return 1

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
        code = _run_from_hydra(cfg)
        if code != 0:
            raise SystemExit(code)
        return code

    _main()
