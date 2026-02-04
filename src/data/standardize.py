#!/usr/bin/env python3
"""
Process videos from experiment.csv by filtering valid entries,
trimming frames, and reducing fps and resolution.

Usage:
    python standardize.py experiment=ants/v1
    python standardize.py experiment=ants/v2 data.target_fps=10
    
Example:
    python standardize.py experiment=ants/v1
"""

import csv
import subprocess
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf


def read_experiment_csv(csv_path: Path) -> List[Dict]:
    """Read experiment.csv and return list of valid entries."""
    valid_entries = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['valid'] == '1':
                valid_entries.append(row)
    
    return valid_entries


def process_video(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    cfg: DictConfig
) -> bool:
    """
    Process a single video: trim frames, reduce fps and resolution.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        start_frame: First frame to include
        end_frame: Last frame to include
        cfg: Hydra configuration
    
    Returns:
        True if successful, False otherwise
    """
    # Get original fps first
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(input_path)
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        fps_str = result.stdout.strip()
        # Parse fraction if needed (e.g., "30000/1001")
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            original_fps = num / den
        else:
            original_fps = float(fps_str)
    except subprocess.CalledProcessError as e:
        print(f"Error probing video {input_path}: {e}")
        return False
    
    # Calculate time stamps
    start_time = start_frame / original_fps
    duration = (end_frame - start_frame + 1) / original_fps
    
    # Get settings from config
    target_fps = cfg.data.target_fps
    width = cfg.data.target_resolution.width
    height = cfg.data.target_resolution.height
    target_resolution = f"{width}x{height}"
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-ss', str(start_time),
        '-t', str(duration),
        '-vf', f'fps={target_fps},scale={target_resolution}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', cfg.data.video_codec,
        '-b:v', cfg.data.bitrate,
    ]
    
    # Add audio removal if configured
    if cfg.data.remove_audio:
        cmd.append('-an')
    
    # Add overwrite flag if configured
    if cfg.data.overwrite:
        cmd.append('-y')
    
    cmd.append(str(output_path))
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"✓ Processed: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error processing {input_path.name}: {e}")
        if e.stderr:
            print(f"  stderr: {e.stderr.decode()}")
        return False


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to process all valid videos from an experiment.
    
    Args:
        cfg: Hydra configuration
    """
    subject = cfg.subject
    version = cfg.version
    
    # Print configuration
    if cfg.data.get('show_progress', True):
        print("Configuration:")
        print(OmegaConf.to_yaml(cfg))
        print()
    
    # Setup paths
    data_dir = Path(cfg.paths.data_dir)
    # Build experiment path from subject + version
    exp_path = f"{subject}/{version}"
    exp_dir = data_dir / exp_path
    
    csv_path = exp_dir / 'experiment.csv'
    output_dir = exp_dir / 'observations' / cfg.data.output_folder
    source_dir = Path(cfg.data.source_path)
    
    # Validate paths
    if not csv_path.exists():
        print(f"Error: experiment.csv not found at {csv_path}")
        return
    
    import os
    if not os.path.isdir(str(source_dir)):
        raise FileNotFoundError(
            f"Source directory not found at {source_dir}\n"
            f"Config source_path: {cfg.data.source_path}\n"
            f"(Check if external volumes are mounted)"
        )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Read valid entries
    valid_entries = read_experiment_csv(csv_path)
    print(f"Found {len(valid_entries)} valid entries to process")
    
    # Process each valid video
    success_count = 0
    error_count = 0
    
    for entry in valid_entries:
        observation_file = entry['observation_file']
        start_frame = int(entry['start_frame'])
        end_frame = int(entry['end_frame'])
        
        input_path = source_dir / observation_file
        output_path = output_dir / observation_file
        
        if not input_path.exists():
            print(f"✗ Warning: Source file not found: {input_path}")
            error_count += 1
            continue
        
        if process_video(input_path, output_path, start_frame, end_frame, cfg):
            success_count += 1
        else:
            error_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Experiment: {exp_path}")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(valid_entries)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
