"""Mice-specific experiment handler"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import DomainHandler


class MiceHandler(DomainHandler):
    """Handler for mouse experiments (v1, v2, etc.)"""
    
    def load_experiment_settings(self) -> pd.DataFrame:
        """
        Load and parse mice experiment_settings.csv
        Mice experiments have different naming convention
        """
        settings_path = self.config.experiment_path / "experiment_settings.csv"
        df = pd.read_csv(settings_path)
        
        # Standardize column names
        if 'observation' in df.columns:
            df = df.rename(columns={
                'observation': 'experiment_id',
                'video': 'video_file',
                'annotation': 'behavior_file',
            })
        
        # Add valid column if not present (assume all are valid)
        if 'valid' not in df.columns:
            df['valid'] = True
        
        # Standardize valid column to boolean
        df['valid'] = df['valid'].astype(bool)
        
        return df
    
    def get_tracking_params(self) -> Dict[str, Any]:
        """
        Return tracking params for mice
        Can use pose estimation (DeepLabCut, SLEAP) or other methods
        """
        return {
            'method': 'pose_estimation',
            'model': 'deeplabcut',  # or 'sleap', 'custom'
            'bodyparts': [
                'nose',
                'left_ear',
                'right_ear',
                'base_of_tail',
                'center_of_mass',
            ],
            'min_confidence': 0.8,
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get mice experiment metadata"""
        metadata = super().get_metadata()
        
        # Add mouse-specific metadata
        settings = self.load_experiment_settings()
        if 'gene' in settings.columns:
            metadata['genes'] = settings['gene'].unique().tolist()
        if 'genotype' in settings.columns:
            metadata['genotypes'] = settings['genotype'].unique().tolist()
        if 'sex' in settings.columns:
            metadata['sexes'] = settings['sex'].unique().tolist()
        
        return metadata
