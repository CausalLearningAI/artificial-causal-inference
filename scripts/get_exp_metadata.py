#!/usr/bin/env python3
"""
Script to analyze video files in data/*/v*/original/video folders and create JSON metadata files.
"""

import os
import json
from pathlib import Path
import subprocess


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
        print(f"Folder does not exist: {video_folder}")
        return
    
    # Find all video files
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm'}
    video_files = []
    
    for file in video_folder.iterdir():
        if file.is_file() and file.suffix.lower() in video_extensions:
            video_files.append(file)
    
    if not video_files:
        print(f"No video files found in {video_folder}")
        return
    
    # Get extension (assuming all videos have the same extension)
    extensions = set(f.suffix.lower() for f in video_files)
    extension = extensions.pop() if len(extensions) == 1 else list(extensions)
    
    # Analyze each video
    video_data = []
    for video_file in video_files:
        print(f"Analyzing {video_file.name}...")
        info = get_video_info(video_file)
        if info:
            video_data.append(info)
    
    if not video_data:
        print(f"Could not analyze any videos in {video_folder}")
        return
    
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
    
    # Create metadata
    metadata = {
        'extension': extension if isinstance(extension, str) else list(extension),
        'n': len(video_files),
        'fps': round(fps_values[0], 2) if fps_consistent and fps_values else "INCONSISTENT",
        'resolution_h': widths[0] if width_consistent and widths else "INCONSISTENT",
        'resolution_w': heights[0] if height_consistent and heights else "INCONSISTENT",
        'channels': channels_values[0] if channels_values else None,
        'duration_s_min': round(min(durations), 2) if durations else None,
        'duration_s_max': round(max(durations), 2) if durations else None,
        'size_MB_min': round(min(sizes), 2) if sizes else None,
        'size_MB_max': round(max(sizes), 2) if sizes else None,
        'size_MB': round(sum(sizes), 2) if sizes else None
    }
    
    # Save JSON file
    json_path = video_folder.parent / 'metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created {json_path}")
    print(f"  Videos: {metadata['n']}")
    print(f"  Extension: {metadata['extension']}")
    print(f"  FPS: {metadata['fps']}")
    print(f"  Resolution: {metadata['resolution_h']}x{metadata['resolution_w']}")
    print(f"  Channels: {metadata['channels']}")
    print(f"  Duration: {metadata['duration_s_min']}s - {metadata['duration_s_max']}s")
    print(f"  Size: {metadata['size_MB_min']}MB - {metadata['size_MB_max']}MB (Total: {metadata['size_MB']}MB)")
    print()


def main():
    """Main function to process all video folders."""
    base_dir = Path(__file__).parent.parent / 'data'
    
    # Find all video folders matching pattern data/*/v*/original/video
    video_folders = []
    
    for species_folder in base_dir.iterdir():
        if species_folder.is_dir():
            for version_folder in species_folder.iterdir():
                if version_folder.is_dir():
                    video_folder = version_folder / 'original' / 'video'
                    if video_folder.exists():
                        video_folders.append(video_folder)
    
    print(f"Found {len(video_folders)} video folders")
    print()
    
    for folder in video_folders:
        print(f"Processing {folder.relative_to(base_dir.parent)}...")
        analyze_video_folder(folder)


if __name__ == '__main__':
    main()
