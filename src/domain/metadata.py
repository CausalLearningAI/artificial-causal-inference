"""Domain-specific metadata extraction utilities"""
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def get_video_info(video_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract video information using ffprobe
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video metadata (duration, fps, width, height, etc.)
        or None if extraction fails
    """
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration,r_frame_rate,width,height,pix_fmt',
            '-show_entries', 'format=duration,size',
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
        
        # Extract resolution
        width = height = None
        if 'streams' in data and len(data['streams']) > 0:
            width = data['streams'][0].get('width')
            height = data['streams'][0].get('height')
        
        # Extract file size
        size_bytes = None
        if 'format' in data and 'size' in data['format']:
            size_bytes = int(data['format']['size'])
        
        # Extract pixel format
        pix_fmt = None
        if 'streams' in data and len(data['streams']) > 0:
            pix_fmt = data['streams'][0].get('pix_fmt')
        
        return {
            'duration_s': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024 * 1024) if size_bytes else None,
            'pixel_format': pix_fmt,
        }
        
    except Exception as e:
        print(f"Error analyzing {video_path}: {e}")
        return None


def create_experiment_metadata(
    experiment_path: Path,
    include_videos: bool = True
) -> Dict[str, Any]:
    """
    Create comprehensive metadata for an experiment
    
    Args:
        experiment_path: Path to experiment (e.g., data/ants/v1)
        include_videos: Whether to analyze video files
        
    Returns:
        Dictionary with experiment metadata
    """
    metadata = {
        'experiment_path': str(experiment_path),
        'domain': experiment_path.parent.name,
        'version': experiment_path.name,
    }
    
    # Check for experiment settings
    settings_file = experiment_path / 'experiment_settings.csv'
    if settings_file.exists():
        settings = pd.read_csv(settings_file)
        metadata['num_experiments'] = len(settings)
        metadata['settings_columns'] = list(settings.columns)
    
    # Analyze videos if requested
    if include_videos:
        video_dir = experiment_path / 'original' / 'video'
        if video_dir.exists():
            video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.mkv'))
            
            if video_files:
                video_metadata = []
                for vf in video_files:
                    info = get_video_info(vf)
                    if info:
                        info['filename'] = vf.name
                        video_metadata.append(info)
                
                if video_metadata:
                    # Aggregate statistics
                    metadata['num_videos'] = len(video_metadata)
                    metadata['video_extension'] = video_files[0].suffix
                    
                    # Calculate aggregates
                    durations = [v['duration_s'] for v in video_metadata if v['duration_s']]
                    if durations:
                        metadata['duration_s_min'] = min(durations)
                        metadata['duration_s_max'] = max(durations)
                        metadata['duration_s_mean'] = np.mean(durations)
                    
                    fps_values = [v['fps'] for v in video_metadata if v['fps']]
                    if fps_values:
                        metadata['fps'] = fps_values[0]  # Assume consistent
                    
                    widths = [v['width'] for v in video_metadata if v['width']]
                    heights = [v['height'] for v in video_metadata if v['height']]
                    if widths and heights:
                        metadata['resolution_w'] = widths[0]
                        metadata['resolution_h'] = heights[0]
                    
                    sizes = [v['size_mb'] for v in video_metadata if v['size_mb']]
                    if sizes:
                        metadata['size_mb_min'] = min(sizes)
                        metadata['size_mb_max'] = max(sizes)
                        metadata['size_mb_total'] = sum(sizes)
                    
                    pix_fmts = [v['pixel_format'] for v in video_metadata if v['pixel_format']]
                    if pix_fmts:
                        metadata['pixel_format'] = pix_fmts[0]
    
    # Check for behavior data
    behavior_dir = experiment_path / 'behavior'
    if behavior_dir.exists():
        behavior_files = list(behavior_dir.glob('*.csv'))
        metadata['num_behavior_files'] = len(behavior_files)
    
    return metadata


def save_metadata_json(experiment_path: Path, metadata: Dict[str, Any]) -> Path:
    """
    Save metadata to JSON file
    
    Args:
        experiment_path: Path to experiment
        metadata: Metadata dictionary
        
    Returns:
        Path to saved JSON file
    """
    output_dir = experiment_path / 'original'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'metadata.json'
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return output_file
