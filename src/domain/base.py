"""Base classes for domain-specific experiment handling"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class ExperimentConfig:
    """Base configuration for any experiment"""
    domain: str
    version: str
    data_dir: Path
    fps: float = 30.0
    resolution: tuple = (700, 700)
    
    @property
    def experiment_path(self) -> Path:
        """Construct path to experiment folder: data/{domain}/{version}"""
        return self.data_dir / self.domain / self.version
    
    @property
    def behavior_dir(self) -> Path:
        """Path to behavior CSV files"""
        return self.experiment_path / "behavior"
    
    @property
    def video_dir(self) -> Path:
        """Path to original videos"""
        return self.experiment_path / "original" / "video"
    
    @property
    def cropped_video_dir(self) -> Path:
        """Path for cropped animal POV videos"""
        return self.experiment_path / "cropped" / "video"
    
    @property
    def tracking_dir(self) -> Path:
        """Path for tracking data"""
        return self.experiment_path / "tracking"


class DomainHandler(ABC):
    """Abstract base for domain-specific experiment processing"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    @abstractmethod
    def load_experiment_settings(self) -> pd.DataFrame:
        """
        Load and parse experiment_settings.csv
        
        Returns:
            DataFrame with standardized columns:
            - experiment_id: str (unique ID for each experiment)
            - video_file: str (filename of video)
            - behavior_file: str (filename of behavior CSV)
            - valid: bool (whether to include in analysis)
        """
        pass
    
    @abstractmethod
    def get_tracking_params(self) -> Dict[str, Any]:
        """Return domain-specific tracking parameters"""
        pass
    
    def get_video_path(self, video_file: str, original: bool = True) -> Path:
        """Get full path to video file"""
        if original:
            return self.config.video_dir / video_file
        else:
            return self.config.cropped_video_dir / video_file
    
    def get_behavior_path(self, behavior_file: str) -> Path:
        """Get full path to behavior CSV file"""
        return self.config.behavior_dir / behavior_file
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata"""
        metadata_path = self.config.experiment_path / "original" / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def validate_experiment(self, experiment_id: str) -> bool:
        """Check if experiment has required files"""
        settings = self.load_experiment_settings()
        return experiment_id in settings['experiment_id'].values
