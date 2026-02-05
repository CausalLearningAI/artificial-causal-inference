#!/usr/bin/env python3
"""
Extract frames from observation videos as .jpg files.

Follows the dataset structure:
    dataset/{subject}/{version}/frames/{type}/{file_id}/frame_*.jpg

Usage:
    python src/dataset/get_frames.py experiment.subject="ants" experiment.version="v1"
    python src/dataset/get_frames.py experiment.subject="mice" experiment.version="v1"

Example:
    python src/dataset/get_frames.py experiment.subject="ants" experiment.version="v1"
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import hydra
from omegaconf import DictConfig


def extract_frames(
    video_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    fps: Optional[float] = None,
    frame_format: str = "rgb24",
) -> bool:
    """
    Extract frames from a video file as .jpg images in RGB color space.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        overwrite: If False and frames exist, skip extraction
        fps: Target frames per second (None = all frames)
        frame_format: Output format (e.g., 'rgb24', 'bgr24', 'rgba')
    
    Returns:
        True if successful, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if frames already exist
    existing_frames = list(output_dir.glob("frame_*.jpg"))
    if existing_frames and not overwrite:
        return True
    
    # Build ffmpeg command with RGB conversion
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output files
        '-i', str(video_path),
    ]
    
    # Add filters: fps (if specified) and color space conversion
    filters = []
    if fps is not None:
        filters.append(f'fps={fps}')
    filters.append(f'format={frame_format}')  # Convert to specified format
    
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    
    # Output pattern: frame_000000.jpg, frame_000001.jpg, etc. (0-indexed)
    output_pattern = str(output_dir / 'frame_%06d.jpg')
    cmd.extend([
        '-start_number', '0',  # Start frame numbering from 0
        '-q:v', '2',  # Quality 2 (best for JPG, lower is better)
        output_pattern
    ])
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames from {video_path}: {e.stderr.decode()}")
        return False


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
    """Extract frames from all observation videos to RGB."""
    subject = cfg.subject
    version = cfg.version
    
    # Get configuration from Hydra
    overwrite = cfg.overwrite.frames
    target_fps = cfg.data.target_fps
    frame_format = cfg.data.frame_format
    
    # Paths
    observations_dir = Path(f"data/{subject}/{version}/observations/full")
    metadata_path = Path(f"data/{subject}/{version}/observations/metadata.json")
    output_base_dir = Path(f"dataset/{subject}/{version}/frames/full")
    
    # Validate source directory
    if not observations_dir.exists():
        print(f"[ERROR] Observations directory not found: {observations_dir}")
        return
    
    # Load and validate metadata
    if not metadata_path.exists():
        print(f"[ERROR] Metadata file not found: {metadata_path}")
        return
    
    metadata = load_metadata(metadata_path)
    
    # Get all video files
    video_files = sorted(observations_dir.glob("*.mkv"))
    
    if not video_files:
        print(f"[ERROR] No video files found in {observations_dir}")
        return
    
    # If overwrite is False, check if all frames already exist - skip entire process
    if not overwrite:
        all_exist = True
        for video_path in video_files:
            file_id = get_file_id(video_path)
            output_dir = output_base_dir / file_id
            existing_frames = list(output_dir.glob("frame_*.jpg")) if output_dir.exists() else []
            if not existing_frames:
                all_exist = False
                break
        
        if all_exist:
            print(f"[SKIP] Frames already extracted for all {len(video_files)} observations")
            return
    
    print(f"Extracting frames from {len(video_files)} observations...")
    
    # Process each video with progress tracking
    processed = 0
    failed = 0
    total = len(video_files)
    start_time = time.time()
    
    for idx, video_path in enumerate(video_files, 1):
        file_id = get_file_id(video_path)
        output_dir = output_base_dir / file_id
        
        result = extract_frames(video_path, output_dir, overwrite=overwrite, fps=target_fps, frame_format=frame_format)
        
        if result:
            processed += 1
        else:
            failed += 1
        
        # Calculate progress
        elapsed = time.time() - start_time
        if idx > 1:
            avg_time = elapsed / (idx - 1)
            remaining = avg_time * (total - idx)
        else:
            remaining = 0
        
        percent = (idx / total) * 100
        progress_str = f"[{idx:3d}/{total}] {percent:5.1f}% | {_format_time(elapsed)} elapsed"
        if idx > 1:
            progress_str += f" | ~{_format_time(remaining)} remaining"
        
        # Print progress every N videos or at the end
        if idx % 20 == 0 or idx == total:
            print(progress_str)
    
    # Summary
    elapsed = time.time() - start_time
    print(f"Completed: {processed} successful, {failed} errors ({_format_time(elapsed)})")


if __name__ == "__main__":
    main()
