"""Ant-specific experiment handler"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from .base import DomainHandler


class AntsHandler(DomainHandler):
    """Handler for ant experiments (v1, v2, etc.)"""
    
    def load_experiment_settings(self) -> pd.DataFrame:
        """
        Load and parse ant experiment_settings.csv
        Ants have 3x3 grid positions (a-e rows, 1-9 positions)
        """
        settings_path = self.config.experiment_path / "experiment_settings.csv"
        df = pd.read_csv(settings_path)
        
        # Standardize column names
        df = df.rename(columns={
            'Experiment': 'experiment_id_raw',
            'Valid': 'valid',
            'FPS': 'fps'
        })
        
        # Parse experiment IDs: format is like "a1", "a2", ..., "e9"
        # First character is letter (a-e), second is position (1-9)
        df['experiment_id'] = df['experiment_id_raw']
        df['row'] = df['experiment_id'].str[0].apply(lambda x: ord(x) - ord('a'))
        df['position'] = df['experiment_id'].str[1].astype(int) - 1
        
        # Add standardized columns for general use
        df['video_file'] = df['Video'] if 'Video' in df.columns else ""
        df['behavior_file'] = df['experiment_id'].apply(lambda x: f"{x}.csv")
        
        # Convert valid to boolean
        df['valid'] = df['valid'].astype(bool)
        
        return df
    
    def get_tracking_params(self) -> Dict[str, Any]:
        """
        Return tracking method descriptor for ant experiments.

        HSV color bounds and algorithm hyperparameters live in
        configs/tracking/params/ants/{version}.yaml and are consumed by
        src/tracking/get_tracking.py.  Call estimate_color_bounds() from
        src/tracking/tracker.py to auto-derive bounds from a video sample.
        """
        return {
            'method': 'color_hsv',
            'colors': ['blue', 'yellow'],
        }
    
    def get_grid_layout(self) -> Dict[str, Any]:
        """Get ant grid layout info"""
        return {
            'rows': 5,      # a-e
            'cols': 9,      # 1-9
            'total': 45,
        }
