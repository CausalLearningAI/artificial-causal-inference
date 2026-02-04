#!/usr/bin/env python3
"""
Analyze video files in data/{subject}/{version}/observations/* folders
and create JSON metadata files for source, full, and povs subdirectories.

Usage:
    python get_metadata.py                  # defaults: ants/v1
    python get_metadata.py experiment.subject=mice  # mice/v1
    python get_metadata.py experiment.subject=ants experiment.version=v2
"""

import json
import os
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def get_video_info(video_path):
    """Extract video information using ffprobe."""
    try:
        # Get comprehensive video information
        cmd = [
            'ffprobe', 
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration,r_frame_rate,width,height,pix_fmt',
            '-show_entries', 'format=duration',
            '-of', 'json',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Warning: Could not analyze {video_path}")
            return None
            
        data = json.loads(result.stdout)
        
        # Extract duration - try stream first, then format
        duration = None
        if 'streams' in data and len(data['streams']) > 0:
            if 'duration' in data['streams'][0]:
                duration = float(data['streams'][0]['duration'])
        
        # If duration not in stream, try format
        if duration is None and 'format' in data and 'duration' in data['format']:
            duration = float(data['format']['duration'])
        
        # Extract FPS
        fps = None
        if 'streams' in data and len(data['streams']) > 0:
            fps_str = data['streams'][0].get('r_frame_rate', '0/1')
            if '/' in fps_str:
                num, den = fps_str.split('/')
                if int(den) != 0:
                    fps = float(num) / float(den)
        
        # Extract resolution (width and height)
        width = None
        height = None
        if 'streams' in data and len(data['streams']) > 0:
            width = data['streams'][0].get('width')
            height = data['streams'][0].get('height')
        
        # Extract pixel format (channels)
        channels = None
        if 'streams' in data and len(data['streams']) > 0:
            pix_fmt = data['streams'][0].get('pix_fmt')
            if pix_fmt:
                # Map common pixel formats to channel descriptions
                if 'rgb' in pix_fmt.lower():
                    channels = 'RGB'
                elif 'bgr' in pix_fmt.lower():
                    channels = 'BGR'
                elif 'yuv' in pix_fmt.lower():
                    channels = 'YUV'
                elif 'gray' in pix_fmt.lower():
                    channels = 'GRAY'
                else:
                    channels = pix_fmt.upper()
        
        # Get file size in MB
        size_bytes = os.path.getsize(video_path)
        size_mb = size_bytes / (1024 * 1024)
        
        return {
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'channels': channels,
            'size_mb': size_mb
        }
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None


def analyze_video_folder(video_folder):
    """Analyze all videos in a folder and create metadata JSON."""
    video_folder = Path(video_folder)
    
    if not video_folder.exists():
        print(f"  Folder does not exist: {video_folder}")
        return None
    
    # Find all video files
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm'}
    video_files = []
    
    for file in video_folder.iterdir():
        if file.is_file() and file.suffix.lower() in video_extensions:
            video_files.append(file)
    
    if not video_files:
        print(f"  No video files found in {video_folder}")
        return None
    
    # Get extension (assuming all videos have the same extension)
    extensions = set(f.suffix.lower() for f in video_files)
    extension = extensions.pop() if len(extensions) == 1 else list(extensions)
    
    # Analyze each video
    video_data = []
    for video_file in video_files:
        info = get_video_info(video_file)
        if info:
            video_data.append(info)
    
    if not video_data:
        print(f"  Could not analyze any videos in {video_folder}")
        return None
    
    # Calculate statistics
    durations = [v['duration'] for v in video_data if v['duration'] is not None]
    fps_values = [v['fps'] for v in video_data if v['fps'] is not None]
    widths = [v['width'] for v in video_data if v['width'] is not None]
    heights = [v['height'] for v in video_data if v['height'] is not None]
    channels_values = [v['channels'] for v in video_data if v['channels'] is not None]
    sizes = [v['size_mb'] for v in video_data]
    
    # Check consistency for fps, resolution_h, resolution_w
    fps_consistent = len(set(fps_values)) <= 1 if fps_values else True
    width_consistent = len(set(widths)) <= 1 if widths else True
    height_consistent = len(set(heights)) <= 1 if heights else True
    
    # Create metadata (durations reported in minutes)
    metadata = {
        'extension': extension if isinstance(extension, str) else list(extension),
        'n': len(video_files),
        'fps': round(fps_values[0], 2) if fps_consistent and fps_values else "INCONSISTENT",
        'resolution_w': widths[0] if width_consistent and widths else "INCONSISTENT",
        'resolution_h': heights[0] if height_consistent and heights else "INCONSISTENT",
        'channels': channels_values[0] if channels_values else None,
        'duration_min_min': round(min(durations) / 60.0, 2) if durations else None,
        'duration_min_max': round(max(durations) / 60.0, 2) if durations else None,
        'duration_min_total': round(sum(durations) / 60.0, 2) if durations else None,
        'size_MB_min': round(min(sizes), 2) if sizes else None,
        'size_MB_max': round(max(sizes), 2) if sizes else None,
        'size_MB_total': round(sum(sizes), 2) if sizes else None
    }
    
    return metadata


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Main function to process video folders for given experiment."""
    subject = cfg.subject
    version = cfg.version
    
    # Setup paths
    data_dir = Path(cfg.paths.data_dir)
    exp_path = f"{subject}/{version}"
    exp_dir = data_dir / exp_path
    obs_dir = exp_dir / 'observations'
    source_folder = Path(cfg.data.source_path)
    
    print(f"Processing: {exp_path}")
    print(f"Source path: {source_folder}")
    print()
    
    # Create overall metadata dict
    all_metadata = {}
    
    if source_folder.exists():
        print("✓ Analyzing 'source' folder...")
        metadata = analyze_video_folder(source_folder)
        if metadata:
            all_metadata['source'] = metadata
            print(f"  Videos: {metadata['n']}")
            print(f"  FPS: {metadata['fps']}, Resolution: {metadata['resolution_w']}x{metadata['resolution_h']}")
            print(f"  Total duration: {metadata['duration_min_total']} min")
            print()
    else:
        print("✗ 'source' folder not found")
        print()
    
    # 2. Analyze 'full' folder
    full_folder = obs_dir / 'full'
    if full_folder.exists():
        print("✓ Analyzing 'full' folder...")
        metadata = analyze_video_folder(full_folder)
        if metadata:
            all_metadata['full'] = metadata
            print(f"  Videos: {metadata['n']}")
            print(f"  FPS: {metadata['fps']}, Resolution: {metadata['resolution_w']}x{metadata['resolution_h']}")
            print(f"  Total duration: {metadata['duration_min_total']} min")
            print()
    else:
        print("⊘ 'full' folder not found (will be created by standardize.py)")
        print()
    
    # 3. Analyze 'povs' subfolder structure
    povs_folder = obs_dir / 'povs'
    if povs_folder.exists():
        print("✓ Analyzing 'povs' subfolders...")
        pov_metadata = {}
        
        for pov_subdir in sorted(povs_folder.iterdir()):
            if pov_subdir.is_dir():
                pov_name = pov_subdir.name
                print(f"  {pov_name}...")
                metadata = analyze_video_folder(pov_subdir)
                if metadata:
                    pov_metadata[pov_name] = metadata
                    print(f"    Videos: {metadata['n']}")
                    print(f"    FPS: {metadata['fps']}, Resolution: {metadata['resolution_w']}x{metadata['resolution_h']}")
                    print(f"    Total duration: {metadata['duration_min_total']} min")
        
        if pov_metadata:
            all_metadata['povs'] = pov_metadata
        print()
    else:
        print("⊘ 'povs' folder not found")
        print()
    
    # Save combined metadata to observations/metadata.json
    if all_metadata:
        metadata_path = obs_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        print("="*60)
        print(f"✓ Metadata saved to: {metadata_path}")
        print("="*60)
    else:
        print("No metadata to save")


if __name__ == '__main__':
    main()
