"""Video cropping utilities for animal-centric POV videos"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class CropConfig:
    """Configuration for video cropping"""
    window_size: int = 224  # Size of cropped region (window_size x window_size)
    follow_method: str = "centroid"  # "centroid", "nose", "body_center"
    padding: int = 20  # Extra padding around detected position
    interpolation: str = "linear"  # "linear", "nearest"
    background_color: Tuple[int, int, int] = (128, 128, 128)  # RGB fill color


def extract_animal_pov_video(
    video_path: Path,
    tracking_data: pd.DataFrame,
    animal_id: str,
    crop_config: CropConfig,
    output_path: Path,
    fps: Optional[float] = None,
) -> bool:
    """
    Extract cropped video following a specific animal's position
    
    Args:
        video_path: Path to original video file
        tracking_data: DataFrame with columns: frame, animal_id, x, y
        animal_id: Identifier of animal to follow (e.g., "blue", "yellow", "mouse_1")
        crop_config: Cropping configuration
        output_path: Where to save cropped video
        fps: Frames per second (if None, use source video fps)
        
    Returns:
        True if successful, False otherwise
        
    Raises:
        FileNotFoundError: If video_path doesn't exist
        ValueError: If tracking_data is invalid
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if tracking_data.empty:
        raise ValueError("Tracking data is empty")
    
    required_cols = {'frame', 'animal_id', 'x', 'y'}
    if not required_cols.issubset(tracking_data.columns):
        raise ValueError(
            f"Tracking data must have columns: {required_cols}. "
            f"Found: {set(tracking_data.columns)}"
        )
    
    # Filter for this animal
    animal_data = tracking_data[tracking_data['animal_id'] == animal_id].copy()
    if animal_data.empty:
        log.warning(f"No tracking data found for animal: {animal_id}")
        return False
    
    # Open source video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Failed to open video: {video_path}")
        return False
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) if fps is None else fps
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    log.info(f"Processing video: {video_path.name}")
    log.info(f"  Frames: {frame_count}, FPS: {src_fps}, Size: {src_width}x{src_height}")
    log.info(f"  Animal: {animal_id}, Output size: {crop_config.window_size}x{crop_config.window_size}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        src_fps,
        (crop_config.window_size, crop_config.window_size)
    )
    
    if not out.isOpened():
        log.error(f"Failed to create output video writer: {output_path}")
        cap.release()
        return False
    
    # Process frames
    frame_idx = 0
    successful_frames = 0
    
    with tqdm(total=frame_count, desc="Cropping video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Look up position for this frame
            pos_data = animal_data[animal_data['frame'] == frame_idx]
            
            if not pos_data.empty:
                x = int(pos_data['x'].iloc[0])
                y = int(pos_data['y'].iloc[0])
                
                # Crop around animal
                crop = _crop_frame(
                    frame,
                    x, y,
                    crop_config.window_size,
                    crop_config.background_color
                )
                successful_frames += 1
            else:
                # Use previous frame or blank
                crop = np.full(
                    (crop_config.window_size, crop_config.window_size, 3),
                    crop_config.background_color,
                    dtype=np.uint8
                )
            
            out.write(crop)
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    
    log.info(f"✓ Saved cropped video: {output_path}")
    log.info(f"  Successful frames: {successful_frames}/{frame_count}")
    
    return True


def _crop_frame(
    frame: np.ndarray,
    center_x: int,
    center_y: int,
    window_size: int,
    background_color: Tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """
    Crop a region from frame centered at (center_x, center_y)
    
    Args:
        frame: Input frame (H, W, C)
        center_x: X coordinate of center
        center_y: Y coordinate of center
        window_size: Size of output crop (window_size x window_size)
        background_color: Color to fill padding (BGR)
        
    Returns:
        Cropped frame (window_size, window_size, 3)
    """
    h, w = frame.shape[:2]
    half_w = window_size // 2
    
    # Calculate crop bounds
    x_min = max(0, center_x - half_w)
    x_max = min(w, center_x + half_w)
    y_min = max(0, center_y - half_w)
    y_max = min(h, center_y + half_w)
    
    # Extract crop region
    crop = frame[y_min:y_max, x_min:x_max]
    
    # Pad if necessary
    if crop.shape[0] < window_size or crop.shape[1] < window_size:
        # Calculate padding
        pad_top = (center_y - half_w) - y_min if y_min > 0 else max(0, half_w - center_y)
        pad_bottom = window_size - crop.shape[0] - pad_top
        pad_left = (center_x - half_w) - x_min if x_min > 0 else max(0, half_w - center_x)
        pad_right = window_size - crop.shape[1] - pad_left
        
        crop = cv2.copyMakeBorder(
            crop,
            max(0, pad_top), max(0, pad_bottom),
            max(0, pad_left), max(0, pad_right),
            cv2.BORDER_CONSTANT,
            value=background_color
        )
    
    return crop[:window_size, :window_size]


def run_cropping(
    settings: pd.DataFrame,
    tracking_results: pd.DataFrame,
    crop_config: CropConfig,
    domain: str,
    version: str,
    data_dir: Path,
) -> pd.DataFrame:
    """
    Run cropping pipeline for all experiments
    
    Args:
        settings: Experiment settings DataFrame
        tracking_results: Tracking results from all experiments
        crop_config: Crop configuration
        domain: Domain name (ants, mice)
        version: Version (v1, v2)
        data_dir: Base data directory
        
    Returns:
        DataFrame with cropping results
    """
    results = []
    
    # Get animal IDs for this domain
    if domain == 'ants':
        animal_ids = ['blue', 'yellow']
    elif domain == 'mice':
        # For mice, use individual IDs from tracking data
        animal_ids = tracking_results['animal_id'].unique().tolist()
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    output_base = data_dir / domain / version / "cropped" / "video"
    
    # Process each experiment
    for _, exp in settings.iterrows():
        exp_id = exp['experiment_id']
        video_file = exp.get('video_file', '')
        
        if not video_file:
            log.warning(f"No video file for experiment: {exp_id}")
            continue
        
        video_path = data_dir / domain / version / "original" / "video" / video_file
        
        # Get tracking for this experiment
        exp_tracking = tracking_results[tracking_results['experiment_id'] == exp_id]
        
        # Crop for each animal
        for animal_id in animal_ids:
            output_path = output_base / f"{exp_id}_{animal_id}.mp4"
            
            try:
                success = extract_animal_pov_video(
                    video_path,
                    exp_tracking,
                    animal_id,
                    crop_config,
                    output_path,
                )
                
                results.append({
                    'experiment_id': exp_id,
                    'animal_id': animal_id,
                    'output_path': str(output_path),
                    'success': success,
                })
            except Exception as e:
                log.error(f"Error processing {exp_id}/{animal_id}: {e}")
                results.append({
                    'experiment_id': exp_id,
                    'animal_id': animal_id,
                    'output_path': str(output_path),
                    'success': False,
                })
    
    return pd.DataFrame(results)
