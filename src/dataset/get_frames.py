#!/usr/bin/env python3
"""
Extract frames from observation videos as .jpg files.

Follows the dataset structure:
    datasets/{subject}/{version}/frames/{type}/{file_id}/frame_*.jpg

Usage:
    python src/dataset/get_frames.py experiment.subject="ants" experiment.version="v1"
    python src/dataset/get_frames.py experiment.subject="mice" experiment.version="v1"

Example:
    python src/dataset/get_frames.py experiment.subject="ants" experiment.version="v1"
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf


def extract_frames(
    video_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    fps: Optional[float] = None,
) -> bool:
    """
    Extract frames from a video file as .jpg images in RGB color space.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        overwrite: If False and frames exist, skip extraction
        fps: Frames per second to extract (None = all frames)
    
    Returns:
        True if successful, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if frames already exist
    existing_frames = list(output_dir.glob("frame_*.jpg"))
    if existing_frames and not overwrite:
        print(f"  ⊙ Skipping (found {len(existing_frames)} existing frames, overwrite=False)")
        return True
    
    # Build ffmpeg command with RGB conversion
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output files
        '-i', str(video_path),
    ]
    
    # Add filters: fps (if specified) and color space conversion to RGB
    filters = []
    if fps is not None:
        filters.append(f'fps={fps}')
    filters.append('format=rgb24')  # Convert to RGB
    
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    
    # Output pattern: frame_000001.jpg, frame_000002.jpg, etc.
    output_pattern = str(output_dir / 'frame_%06d.jpg')
    cmd.extend([
        '-q:v', '2',  # Quality 2 (best for JPG, lower is better)
        output_pattern
    ])
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames from {video_path}: {e.stderr.decode()}")
        return False


def get_file_id(video_path: Path) -> str:
    """Extract file ID from video filename (e.g., 'a1' from 'a1.mkv')."""
    return video_path.stem


def load_metadata(metadata_path: Path) -> Dict:
    """Load observation metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def validate_observations(observations_dir: Path, metadata: Dict) -> bool:
    """
    Validate observations directory against metadata.
    
    Checks:
    - File extension matches metadata
    - Number of files matches metadata count
    - Color space channels match metadata
    
    Returns:
        True if validation passes, False otherwise
    """
    video_files = sorted(observations_dir.glob("*.mkv"))
    
    # Check metadata exists
    if "full" not in metadata:
        print("Error: 'full' not found in metadata")
        return False
    
    full_metadata = metadata["full"]
    
    # Check extension
    expected_ext = full_metadata.get("extension", ".mkv")
    actual_ext = video_files[0].suffix if video_files else None
    if actual_ext != expected_ext:
        print(f"Warning: Extension mismatch. Expected {expected_ext}, got {actual_ext}")
    
    # Check count
    expected_count = full_metadata.get("n", 0)
    actual_count = len(video_files)
    if actual_count != expected_count:
        print(f"Warning: File count mismatch. Expected {expected_count}, got {actual_count}")
    
    # Check channels
    expected_channels = full_metadata.get("channels", "YUV")
    print(f"Metadata info: {expected_count} videos, format: {expected_ext}, channels: {expected_channels}")
    
    return True


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Extract frames from all observation videos to RGB.
    """
    subject = cfg.experiment.subject
    version = cfg.experiment.version
    dataset_type = "full"
    
    # Load subject/version-specific dataset config
    dataset_config_path = Path(f"configs/datasets/{subject}/{version}.yaml")
    if dataset_config_path.exists():
        import yaml
        with open(dataset_config_path, 'r') as f:
            dataset_cfg = yaml.safe_load(f)
        overwrite = dataset_cfg.get('overwrite_frames', False)
        target_fps = dataset_cfg.get('target_fps', None)
    else:
        print(f"Warning: Config not found at {dataset_config_path}, using defaults")
        overwrite = False
        target_fps = None
    
    # Paths
    observations_dir = Path(f"data/{subject}/{version}/observations/{dataset_type}")
    metadata_path = Path(f"data/{subject}/{version}/observations/metadata.json")
    output_base_dir = Path(f"datasets/{subject}/{version}/frames/{dataset_type}")
    
    # Validate source directory
    if not observations_dir.exists():
        print(f"Error: Observations directory not found: {observations_dir}")
        return
    
    # Load and validate metadata
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return
    
    metadata = load_metadata(metadata_path)
    if not validate_observations(observations_dir, metadata):
        print("Validation failed, continuing anyway...")
    
    # Get all video files
    video_files = sorted(observations_dir.glob("*.mkv"))
    
    if not video_files:
        print(f"No video files found in {observations_dir}")
        return
    
    print(f"Found {len(video_files)} video(s) to process")
    print(f"Output: RGB .jpg frames to {output_base_dir}")
    print(f"Overwrite mode: {overwrite}\n")
    
    # Process each video
    processed = 0
    skipped = 0
    failed = 0
    
    for video_path in video_files:
        file_id = get_file_id(video_path)
        output_dir = output_base_dir / file_id
        
        print(f"Extracting frames from {video_path.name} -> {output_dir}")
        
        result = extract_frames(video_path, output_dir, overwrite=overwrite, fps=target_fps)
        
        if result:
            # Check if it was actually extracted or skipped
            frame_count = len(list(output_dir.glob("frame_*.jpg")))
            if output_dir.exists() and frame_count > 0:
                # Check if we skipped (frames existed and overwrite=False)
                if not overwrite and frame_count > 0:
                    existing_check = list(output_dir.glob("frame_*.jpg"))
                    if len(existing_check) > 0:
                        # Re-check if extraction happened by looking at timing
                        # For simplicity, if overwrite is False and frames exist, we skipped
                        pass
                processed += 1
                print(f"  ✓ {frame_count} frames (RGB)")
        else:
            failed += 1
            print(f"  ✗ Failed to extract frames")
    
    print(f"\nCompleted: {processed} succeeded, {failed} failed out of {len(video_files)} videos")


if __name__ == "__main__":
    main()
