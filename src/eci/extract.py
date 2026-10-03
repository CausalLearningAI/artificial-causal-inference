"""
Frame sampling and DINOv2 token extraction for the ECI (Exploratory Causal
Inference) pipeline on mice v1.

Step 1 of the pipeline trains a top-k sparse autoencoder (SAE) on individual
final-layer DINOv2 patch tokens (arXiv 2510.14073). Storing patch tokens for all
2.6M frames is ~1 TB, so we only extract them for a stratified training subset:
every observation contributes the same number of frames, spread evenly in time.

Layer used: `last_hidden_state` of HF `Dinov2Model`, i.e. the output of the last
transformer block AFTER the final LayerNorm (equivalent to DINOv2's
`x_norm_patchtokens` / `x_norm_clstoken`). This is NOT the `class_l-2`
embeddings (which are hidden_states[-2], pre-norm).

Functions:
    build_frame_table   one row per frame on disk, with pool/stage/genotype metadata
    sample_frames       stratified-in-time sampling (n per observation, seeded jitter)
    sample_frames_stride  every stride-th frame per observation (1 fps = stride 5), seeded offset
    load_encoder        DINOv2 (or DINOv3: 'dinov3_base' ViT-B/16, 'dinov3_small' ViT-S/16) model + its processor
    extract_tokens      writes patch tokens / CLS / metadata to an output directory
"""

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

MODEL_IDS = {'dinov2_base': 'facebook/dinov2-base', 'dinov3_base': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
             'dinov3_small': 'facebook/dinov3-vits16-pretrain-lvd1689m'}
PATCH_SIZES = {'dinov2_base': 14, 'dinov3_base': 16, 'dinov3_small': 16}
STAGES = {('H', 'S'): 1, ('O', 'S'): 2, ('P', 'S'): 3, ('H', 'F'): 4, ('O', 'F'): 5, ('P', 'F'): 6}


def build_frame_table(dataset_dir='dataset', data_dir='data', subject='mice', version='v1'):
    """One row per JPG frame found on disk, for every observation in experiment.csv.

    Columns: row_idx (global row in annotations.csv, -1 if the frame is not there),
    observation_id, frame_idx, frame_path (relative to dataset_dir), pool, stage (1-6),
    phase, odor, genotype ('wt'/'het'), annotated (behavior annotation file exists).
    """
    dataset_dir, data_dir = Path(dataset_dir), Path(data_dir)
    exp = pd.read_csv(data_dir / subject / version / 'experiment.csv')
    frames_root = dataset_dir / subject / version / 'frames' / 'full'

    ann = pd.read_csv(dataset_dir / subject / version / 'annotations.csv', usecols=['frame_path'])
    ann_row = pd.Series(np.arange(len(ann), dtype=np.int64), index=ann['frame_path'].values)

    rows = []
    for rec in exp.itertuples(index=False):
        folder = Path(rec.observation_file).stem
        obs_dir = frames_root / folder
        if not obs_dir.is_dir():
            continue
        names = sorted(n for n in os.listdir(obs_dir) if n.startswith('frame_') and n.endswith('.jpg'))
        frame_idx = np.array([int(n[6:-4]) for n in names], dtype=np.int64)
        rel = [f'{subject}/{version}/frames/full/{folder}/{n}' for n in names]
        rows.append(pd.DataFrame({
            'observation_id': rec.observation_id,
            'frame_idx': frame_idx,
            'frame_path': rel,
            'pool': rec.pool,
            'stage': STAGES[(rec.phase, rec.odor)],
            'phase': rec.phase,
            'odor': rec.odor,
            'genotype': rec.genotype,
            'annotated': isinstance(rec.annotation_file, str) and len(rec.annotation_file) > 0,
        }))
    table = pd.concat(rows, ignore_index=True)
    table.insert(0, 'row_idx', ann_row.reindex(table['frame_path'].values).fillna(-1).astype(np.int64).values)
    return table


def sample_frames(table, n_per_obs=64, seed=0):
    """Pick n_per_obs frames per observation: split its timeline into n_per_obs equal
    bins and draw one frame uniformly inside each bin (seeded, so reproducible)."""
    rng = np.random.default_rng(seed)
    picks = []
    for obs_id in sorted(table['observation_id'].unique()):
        obs = table[table['observation_id'] == obs_id].sort_values('frame_idx')
        n = len(obs)
        if n < n_per_obs:
            raise ValueError(f'{obs_id} has only {n} frames (< {n_per_obs})')
        edges = np.round(np.linspace(0, n, n_per_obs + 1)).astype(np.int64)  # integer bin edges
        pos = edges[:-1] + rng.integers(0, edges[1:] - edges[:-1])  # one frame per bin [lo, hi)
        assert len(np.unique(pos)) == n_per_obs
        picks.append(obs.iloc[pos])
    return pd.concat(picks, ignore_index=True)


def sample_frames_stride(table, stride=5, seed=0):
    """Every `stride`-th frame of each observation (stride 5 at 5 fps = 1 frame per second),
    starting at a seeded random offset in [0, stride) drawn per observation."""
    rng = np.random.default_rng(seed)
    picks = []
    for obs_id in sorted(table['observation_id'].unique()):
        obs = table[table['observation_id'] == obs_id].sort_values('frame_idx')
        offset = int(rng.integers(0, stride))
        picks.append(obs.iloc[offset::stride])
    return pd.concat(picks, ignore_index=True)


def load_encoder(name='dinov2_base', resolution=224, device='cuda', center_crop=True):
    """DINOv2 with its standard HF preprocessing (bicubic resize of the shorter side to
    resolution*256/224, center crop to resolution, ImageNet normalization).
    center_crop=False: the WHOLE frame is resized (bicubic) to resolution x resolution,
    no crop (used by the foreground pipeline, src/eci/foreground.py, at 448)."""
    from transformers import AutoImageProcessor, AutoModel
    model_id = MODEL_IDS[name]
    patch = PATCH_SIZES[name]
    if resolution % patch != 0:
        raise ValueError(f'resolution must be a multiple of the patch size {patch}, got {resolution}')
    if name != 'dinov2_base':
        # DINOv3: its own processor (ImageNet mean / std, bilinear resize) on the whole frame, no crop
        if center_crop:
            raise ValueError(f'{name}: only center_crop=False is supported')
        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True,
                                                       size={'height': resolution, 'width': resolution})
    elif center_crop:
        processor = AutoImageProcessor.from_pretrained(
            model_id, use_fast=True,
            size={'shortest_edge': int(round(resolution * 256 / 224))},
            crop_size={'height': resolution, 'width': resolution},
        )
    else:
        processor = AutoImageProcessor.from_pretrained(
            model_id, use_fast=True, size={'height': resolution, 'width': resolution}, do_center_crop=False,
        )
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    model.requires_grad_(False)
    return model_id, processor, model


class _FrameDataset(torch.utils.data.Dataset):
    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            image = im.convert('RGB')
        return self.processor(images=image, return_tensors='pt')['pixel_values'][0]


def extract_tokens(frames, out_dir, dataset_dir='dataset', encoder='dinov2_base', resolution=224,
                   batch_size=128, num_workers=8, device='cuda', extra_config=None):
    """Encode the frames in `frames` (a sample_frames() table) and write to out_dir:
        patch_tokens.npy  float16 (N, n_patches, D)  final layer, after final LayerNorm
        cls.npy           float16 (N, D)             same layer, CLS token
        metadata.parquet  one row per frame, same order
        config.json       model id, resolution, layer, timings
    Writes into out_dir.tmp and renames at the end, so a crash never leaves a partial out_dir.
    """
    out_dir = Path(out_dir)
    tmp_dir = Path(str(out_dir) + '.tmp')
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    device = torch.device(device)
    model_id, processor, model = load_encoder(encoder, resolution, device)
    n_reg = getattr(model.config, 'num_register_tokens', 0) or 0
    n_prefix = 1 + n_reg
    n_patches = (resolution // model.config.patch_size) ** 2
    dim = model.config.hidden_size
    N = len(frames)

    paths = [str(Path(dataset_dir) / p) for p in frames['frame_path']]
    loader = torch.utils.data.DataLoader(
        _FrameDataset(paths, processor), batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=device.type == 'cuda',
        prefetch_factor=4 if num_workers > 0 else None)

    patch_mm = np.lib.format.open_memmap(tmp_dir / 'patch_tokens.npy', mode='w+', dtype=np.float16,
                                         shape=(N, n_patches, dim))
    cls_mm = np.lib.format.open_memmap(tmp_dir / 'cls.npy', mode='w+', dtype=np.float16, shape=(N, dim))

    t0, cur, batch_times = time.time(), 0, []
    for b, pix in enumerate(loader):
        tb = time.time()
        with torch.inference_mode():
            hs = model(pixel_values=pix.to(device, non_blocking=True)).last_hidden_state.float()
        if hs.shape[1] != n_prefix + n_patches:
            raise RuntimeError(f'unexpected token count {hs.shape[1]} (expected {n_prefix + n_patches})')
        if not torch.isfinite(hs).all():
            raise RuntimeError(f'non-finite values in batch {b}')
        if hs.abs().max() > 6e4:
            raise RuntimeError(f'values overflow float16 in batch {b}: max |x| = {hs.abs().max():.1f}')
        B = hs.shape[0]
        cls_mm[cur:cur + B] = hs[:, 0].half().cpu().numpy()
        patch_mm[cur:cur + B] = hs[:, n_prefix:].half().cpu().numpy()
        cur += B
        if device.type == 'cuda':
            torch.cuda.synchronize()
        batch_times.append(time.time() - tb)
        if b % 20 == 0:
            el = time.time() - t0
            print(f'  batch {b:4d}  {cur:6d}/{N}  {cur / el:6.1f} frames/s  gpu {batch_times[-1]:.3f}s/batch', flush=True)
    assert cur == N, f'wrote {cur} rows, expected {N}'
    patch_mm.flush(); cls_mm.flush()
    del patch_mm, cls_mm
    elapsed = time.time() - t0

    frames.reset_index(drop=True).to_parquet(tmp_dir / 'metadata.parquet', index=False)
    config = {
        'model_id': model_id, 'resolution': resolution, 'patch_size': model.config.patch_size,
        'n_patches': n_patches, 'dim': dim, 'n_frames': N,
        'layer': 'last_hidden_state (final block output after final LayerNorm = x_norm_patchtokens)',
        'preprocessing': {'resize_shortest_edge': processor.size['shortest_edge'],
                          'center_crop': resolution, 'resample': 'bicubic',
                          'mean': processor.image_mean, 'std': processor.image_std},
        'precision': 'fp32 forward, stored as float16',
        'elapsed_s': round(elapsed, 1), 'mean_gpu_batch_s': round(float(np.mean(batch_times[1:] or batch_times)), 4),
        'batch_size': batch_size,
    }
    config.update(extra_config or {})
    (tmp_dir / 'config.json').write_text(json.dumps(config, indent=2))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    print(f'Done: {N} frames in {elapsed:.0f}s -> {out_dir}')
    return out_dir
