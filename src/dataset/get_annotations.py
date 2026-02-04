"""
Annotation Extractor for Causal Inference Datasets

This module provides an abstract framework for extracting annotations from
different subjects (ants, frogs, mice) and versions. Each row represents a
single frame with:
- frame_path: path to the frame image
- observation_id: unique identifier for the observation
- T: treatment assignment
- W_{name}: covariates (from configs)
- Y_{name}: outcomes (from configs)

The extraction logic is delegated to domain-specific handlers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm


class AnnotationExtractor(ABC):
    """
    Abstract base class for extracting annotations from experiment data.
    
    Each subject/version combination should implement its own extractor
    with specific logic for parsing annotations and determining treatments.
    """
    
    @abstractmethod
    def extract_labels(
        self,
        annotation_file: Path,
        observation_id: str,
        config: Dict
    ) -> pd.DataFrame:
        """
        Extract labels from a single annotation file.
        
        Args:
            annotation_file: Path to annotation CSV
            observation_id: Identifier for this observation
            config: Configuration dictionary with covariate/outcome definitions
            
        Returns:
            DataFrame with columns: observation_id, frame_idx, T, W_{name}, Y_{name}
        """
        pass
    
    @abstractmethod
    def get_treatment(self, observation_id: str, metadata: Dict) -> int:
        """
        Determine treatment assignment for an observation.
        
        Args:
            observation_id: Identifier for the observation
            metadata: Metadata from experiment.csv or other sources
            
        Returns:
            Treatment assignment (0 or 1, or other discrete values)
        """
        pass


class AntsV1Extractor(AnnotationExtractor):
    """Annotation extractor for ants version 1."""
    
    def __init__(self, outcome_mapping: Dict[str, List[int]]):
        """Initialize with outcome mapping from config."""
        self.outcome_mapping = outcome_mapping
    
    def map_behaviour_to_label(self, behaviour: str) -> Tuple[int, ...]:
        """
        Map behavior string to label tuple using config.
        
        Args:
            behaviour: Behavior string from annotation file
            
        Returns:
            Tuple of outcome values
        """
        behaviour = behaviour.strip()
        
        if behaviour in self.outcome_mapping:
            return tuple(self.outcome_mapping[behaviour])
        else:
            raise ValueError(f'Unknown behaviour: {behaviour}. '
                           f'Known behaviors: {list(self.outcome_mapping.keys())}')
    
    def label_frame(self, frame_id: int, behaviors: pd.DataFrame, 
                    col_map: Dict = None) -> Tuple[int, ...]:
        """
        Get label for a specific frame by checking which behaviors span it.
        
        Args:
            frame_id: Frame number (0-indexed)
            behaviors: DataFrame with behavior annotations
            col_map: Column name mappings from config
            
        Returns:
            Tuple of outcome labels for this frame
        """
        if col_map is None:
            col_map = {
                'start_frame': 'Beginning-frame',
                'end_frame': 'End-frame',
                'behavior': 'Behavior'
            }
        
        start_col = col_map.get('start_frame', 'Beginning-frame')
        end_col = col_map.get('end_frame', 'End-frame')
        behavior_col = col_map.get('behavior', 'Behavior')
        
        # Initialize all outcomes to 0
        label_values = None
        
        for _, row in behaviors.iterrows():
            start_frame = int(row[start_col])
            end_frame = int(row[end_col])
            
            if frame_id >= start_frame and frame_id < end_frame:
                current_label = self.map_behaviour_to_label(row[behavior_col])
                
                if label_values is None:
                    label_values = list(current_label)
                else:
                    # Use max to handle overlapping behaviors
                    label_values = [max(a, b) for a, b in zip(label_values, current_label)]
        
        # Return zeros if no behavior found for this frame
        if label_values is None:
            # Infer number of outcomes from first outcome mapping
            n_outcomes = len(next(iter(self.outcome_mapping.values())))
            return tuple([0] * n_outcomes)
        
        return tuple(label_values)
    
    def extract_labels(
        self,
        annotation_file: Path,
        observation_id: str,
        config: Dict
    ) -> pd.DataFrame:
        """
        Extract labels from ants v1 annotation file.
        Returns one row per frame with labels for all outcomes.
        """
        # Read annotation file using format from config
        ann_format = config.get('annotation_format', {})

        def find_column(df: pd.DataFrame, candidates) -> str:
            """Resolve column name with tolerant matching (hyphens/spaces/underscores)."""
            cols = list(df.columns)
            # exact match
            for c in candidates:
                if c in cols:
                    return c
            # normalized match
            norm = {col.lower().replace('-', '').replace('_', '').replace(' ', ''): col for col in cols}
            for c in candidates:
                key = c.lower().replace('-', '').replace('_', '').replace(' ', '')
                if key in norm:
                    return norm[key]
            raise KeyError(candidates[0])
        
        try:
            read_kwargs = {
                'skiprows': ann_format.get('skiprows', 3),
                'skipfooter': ann_format.get('skipfooter', 1),
                'engine': ann_format.get('engine', 'python')
            }
            behaviors = pd.read_csv(annotation_file, **read_kwargs)
        except Exception as e:
            raise ValueError(f"Failed to read annotation file {annotation_file}: {e}")
        
        # Resolve column names with tolerant matching
        col_map_cfg = ann_format.get('columns', {})
        start_col = find_column(behaviors, [col_map_cfg.get('start_frame', 'Beginning-frame'), 'Beginning-frame', 'Beginning frame', 'Start frame'])
        end_col = find_column(behaviors, [col_map_cfg.get('end_frame', 'End-frame'), 'End-frame', 'End frame', 'Stop frame'])
        behavior_col = find_column(behaviors, [col_map_cfg.get('outcome', 'Behavior'), 'Behavior'])

        # Store resolved columns
        col_map = {
            'start_frame': start_col,
            'end_frame': end_col,
            'behavior': behavior_col
        }
        
        # Validate behaviors against outcome_mapping
        if behaviors.shape[0] > 0:
            unique_behaviors = behaviors[behavior_col].unique()
            unknown_behaviors = [b for b in unique_behaviors if b.strip() not in self.outcome_mapping]
            if unknown_behaviors:
                raise ValueError(
                    f"Unknown behaviors in {annotation_file}: {unknown_behaviors}\n"
                    f"Known behaviors from config: {list(self.outcome_mapping.keys())}"
                )
        
        # Get total number of frames in this observation
        n_frames = int(config.get('frame_count', 0) or 0)
        if n_frames <= 0:
            raise ValueError(f"frame_count not provided or invalid for {observation_id}")
        
        # FPS handling: annotations are in source-video frame indices
        source_fps = config.get('source_fps', None)
        target_fps = config.get('target_fps', 30)
        
        if source_fps is None:
            fps_ratio = 1.0
        else:
            fps_ratio = source_fps / target_fps
        
        # offset: subtract from annotation frame indices to map to clipped-video coordinates
        offset = int(config.get('start_frame_offset', 0) or 0)
        
        # Get outcome names from config
        outcome_names = config.get('outcomes', ['Y2F', 'B2F'])
        n_outcomes = len(outcome_names)
        
        # Vectorized labeling: create arrays and fill via masks
        outcome_arrays = {f'Y_{name}': np.zeros(n_frames, dtype=int) for name in outcome_names}

        # Precompute mapping from clipped frames to annotation frames (handles FPS conversion)
        # For clipped frame i (0-indexed), source frame = offset + i * fps_ratio (in source_fps coordinates)
        annotation_frame_ids = (offset + np.arange(n_frames) * fps_ratio).astype(int)

        if behaviors.shape[0] > 0:
            for _, row in behaviors.iterrows():
                source_start = int(row[col_map['start_frame']])
                source_end = int(row[col_map['end_frame']])
                behavior_str = str(row[col_map['behavior']]).strip()

                # Boolean mask where the annotation frame falls inside the behavior interval
                mask = (annotation_frame_ids >= source_start) & (annotation_frame_ids < source_end)
                if not mask.any():
                    continue

                # Get label tuple and apply with np.maximum
                label_tuple = self.map_behaviour_to_label(behavior_str)
                for idx, outcome_name in enumerate(outcome_names):
                    arr = outcome_arrays[f'Y_{outcome_name}']
                    arr[mask] = np.maximum(arr[mask], label_tuple[idx])
        
        # Build result DataFrame (frame_idx 0-indexed)
        result_data = {
            'frame_idx': np.arange(n_frames),
            'fps': np.full(n_frames, target_fps, dtype=float)
        }
        result_data.update(outcome_arrays)
        result = pd.DataFrame(result_data)
        result['observation_id'] = observation_id
        result['T'] = None  # Will be filled from experiment metadata
        
        return result
    
    def get_treatment(self, observation_id: str, metadata: Dict) -> int:
        """
        Determine treatment for ants v1.
        Treatment is typically in the experiment.csv metadata.
        """
        return metadata.get('treatment', 0)


class AntsV2Extractor(AnnotationExtractor):
    """Annotation extractor for ants version 2 (same structure as v1, different behaviors)."""
    
    def __init__(self, outcome_mapping: Dict[str, List[int]]):
        """Initialize with outcome mapping from config."""
        self.outcome_mapping = outcome_mapping
    
    def map_behaviour_to_label(self, behaviour: str) -> Tuple[int, ...]:
        """
        Map outcome string to label tuple using config.
        
        Args:
            behaviour: Outcome string from annotation file
            
        Returns:
            Tuple of outcome values
        """
        behaviour = behaviour.strip()
        
        if behaviour in self.outcome_mapping:
            return tuple(self.outcome_mapping[behaviour])
        else:
            raise ValueError(f'Unknown behaviour: {behaviour}. '
                           f'Known behaviors: {list(self.outcome_mapping.keys())}')
    
    def extract_labels(
        self,
        annotation_file: Path,
        observation_id: str,
        config: Dict
    ) -> pd.DataFrame:
        """Extract labels from ants v2 annotation file (same format as v1)."""
        # Read annotation file using format from config
        ann_format = config.get('annotation_format', {})

        def find_column(df: pd.DataFrame, candidates) -> str:
            """Resolve column name with tolerant matching (hyphens/spaces/underscores)."""
            cols = list(df.columns)
            # exact match
            for c in candidates:
                if c in cols:
                    return c
            # normalized match
            norm = {col.lower().replace('-', '').replace('_', '').replace(' ', ''): col for col in cols}
            for c in candidates:
                key = c.lower().replace('-', '').replace('_', '').replace(' ', '')
                if key in norm:
                    return norm[key]
            raise KeyError(candidates[0])
        
        try:
            read_kwargs = {
                'skiprows': ann_format.get('skiprows', 3),
                'skipfooter': ann_format.get('skipfooter', 1),
                'engine': ann_format.get('engine', 'python')
            }
            behaviors = pd.read_csv(annotation_file, **read_kwargs)
        except Exception as e:
            raise ValueError(f"Failed to read annotation file {annotation_file}: {e}")
        
        # Resolve column names with tolerant matching
        col_map_cfg = ann_format.get('columns', {})
        start_col = find_column(behaviors, [col_map_cfg.get('start_frame', 'Beginning-frame'), 'Beginning-frame', 'Beginning frame', 'Start frame'])
        end_col = find_column(behaviors, [col_map_cfg.get('end_frame', 'End-frame'), 'End-frame', 'End frame', 'Stop frame'])
        behavior_col = find_column(behaviors, [col_map_cfg.get('outcome', 'Behavior'), 'Behavior'])

        # Store resolved columns
        col_map = {
            'start_frame': start_col,
            'end_frame': end_col,
            'behavior': behavior_col
        }
        
        # Validate behaviors against outcome_mapping
        if behaviors.shape[0] > 0:
            unique_behaviors = behaviors[behavior_col].unique()
            unknown_behaviors = [b for b in unique_behaviors if b.strip() not in self.outcome_mapping]
            if unknown_behaviors:
                raise ValueError(
                    f"Unknown behaviors in {annotation_file}: {unknown_behaviors}\n"
                    f"Known behaviors from config: {list(self.outcome_mapping.keys())}"
                )
        
        # Get total number of frames in this observation
        n_frames = int(config.get('frame_count', 0) or 0)
        if n_frames <= 0:
            raise ValueError(f"frame_count not provided or invalid for {observation_id}")
        
        # FPS handling: annotations are in source-video frame indices
        source_fps = config.get('source_fps', None)
        target_fps = config.get('target_fps', 30)
        
        if source_fps is None:
            fps_ratio = 1.0
        else:
            fps_ratio = source_fps / target_fps
        
        # offset: subtract from annotation frame indices to map to clipped-video coordinates
        offset = int(config.get('start_frame_offset', 0) or 0)
        
        # Get outcome names from config
        outcome_names = config.get('outcomes', ['Y2F', 'B2F'])
        n_outcomes = len(outcome_names)
        
        # Vectorized labeling: create arrays and fill via masks
        outcome_arrays = {f'Y_{name}': np.zeros(n_frames, dtype=int) for name in outcome_names}

        # Precompute mapping from clipped frames to annotation frames (handles FPS conversion)
        # For clipped frame i (0-indexed), source frame = offset + i * fps_ratio (in source_fps coordinates)
        annotation_frame_ids = (offset + np.arange(n_frames) * fps_ratio).astype(int)

        if behaviors.shape[0] > 0:
            for _, row in behaviors.iterrows():
                source_start = int(row[col_map['start_frame']])
                source_end = int(row[col_map['end_frame']])
                behavior_str = str(row[col_map['behavior']]).strip()

                # Boolean mask where the annotation frame falls inside the behavior interval
                mask = (annotation_frame_ids >= source_start) & (annotation_frame_ids < source_end)
                if not mask.any():
                    continue

                # Get label tuple and apply with np.maximum
                label_tuple = self.map_behaviour_to_label(behavior_str)
                for idx, outcome_name in enumerate(outcome_names):
                    arr = outcome_arrays[f'Y_{outcome_name}']
                    arr[mask] = np.maximum(arr[mask], label_tuple[idx])
        
        # Build result DataFrame (frame_idx 0-indexed)
        result_data = {
            'frame_idx': np.arange(n_frames),
            'fps': np.full(n_frames, target_fps, dtype=float)
        }
        result_data.update(outcome_arrays)
        result = pd.DataFrame(result_data)
        result['observation_id'] = observation_id
        result['T'] = None  # Will be filled from experiment metadata
        
        return result
    
    def get_treatment(self, observation_id: str, metadata: Dict) -> int:
        """Treatment logic for ants v2."""
        return metadata.get('treatment', 0)


class DatasetGenerator:
    """
    Generate dataset tables using subject/version-specific extractors.
    
    Coordinates the extraction process and combines annotations with frame paths.
    """
    
    def __init__(
        self,
        data_root: Path,
        dataset_root: Path,
        config_dir: Path,
        subject: str,
        version: str,
        extractor: Optional[AnnotationExtractor] = None
    ):
        """
        Initialize the dataset generator.
        
        Args:
            data_root: Root directory containing raw data
            dataset_root: Root directory for processed datasets
            config_dir: Directory containing dataset configs
            subject: Subject type (ants, frogs, mice)
            version: Version of the dataset
            extractor: Domain-specific annotation extractor (optional, will auto-create)
        """
        self.data_root = Path(data_root)
        self.dataset_root = Path(dataset_root)
        self.subject = subject
        self.version = version
        
        # Load subject/version-specific configuration
        config_path = Path(config_dir) / subject / f"{version}.yaml"
        if not config_path.exists():
            raise ValueError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create extractor if not provided
        if extractor is None:
            extractor = get_extractor(subject, version, self.config)
        self.extractor = extractor
        
        # Define paths
        self.data_subject_dir = self.data_root / subject / version
        self.annotations_dir = self.data_subject_dir / "annotations"
        self.frames_dir = self.dataset_root / subject / version / "frames" / "full"
        self.experiment_csv = self.data_subject_dir / "experiment.csv"
        self.observations_metadata = self.data_subject_dir / "observations" / "metadata.json"
    
    def load_observations_metadata(self) -> Dict:
        """Load observations metadata to get source FPS."""
        if not self.observations_metadata.exists():
            print(f"Warning: Observations metadata not found: {self.observations_metadata}")
            return {}
        
        import json
        with open(self.observations_metadata, 'r') as f:
            return json.load(f)
    
    def load_experiment_metadata(self) -> pd.DataFrame:
        """Load experiment metadata CSV."""
        if not self.experiment_csv.exists():
            raise ValueError(f"Experiment CSV not found: {self.experiment_csv}")
        
        return pd.read_csv(self.experiment_csv)
    
    def generate_dataset_table(self) -> pd.DataFrame:
        """
        Generate complete dataset table for all annotations.
        
        Returns:
            DataFrame with one row per frame containing:
            - frame_path: relative path to frame image
            - observation_id: identifier for the observation
            - T: treatment assignment
            - W_{name}: covariates
            - Y_{name}: outcomes
        """
        all_rows = []
        
        # Load experiment metadata
        experiments = self.load_experiment_metadata()
        
        # Load observations metadata to get source FPS
        obs_metadata = self.load_observations_metadata()
        source_fps = obs_metadata.get('source', {}).get('fps', None)
        
        if source_fps is None:
            print("Warning: Could not determine source FPS from observations metadata")
            print("Assuming 1:1 frame mapping (no FPS conversion)")
        else:
            target_fps = self.config.get('target_fps', 30)
            print(f"FPS conversion: source={source_fps} fps -> target={target_fps} fps")
        
        # Process each observation
        for _, exp in tqdm(experiments.iterrows(), total=len(experiments), desc="Processing observations"):
            observation_id = exp['observation_id']
            
            # Skip invalid observations
            if exp.get('valid', 1) == 0:
                continue
            
            # Get annotation file
            annotation_file = self.annotations_dir / f"{observation_id}.csv"
            if not annotation_file.exists():
                print(f"Warning: Annotation file not found for {observation_id}, skipping")
                continue

            # Locate frames for this observation
            frame_dir = self.frames_dir / observation_id
            if not frame_dir.exists():
                print(f"Warning: Frame directory not found for {observation_id}, skipping")
                continue
            frame_files = sorted(frame_dir.glob("frame_*.jpg"))
            frame_count = len(frame_files)
            if frame_count == 0:
                print(f"Warning: No frames found for {observation_id}, skipping")
                continue
            
            # Prepare config for this observation
            # For annotation extraction: we iterate over all clipped frames (0 to frame_count)
            # For each frame, check if it falls within any behavior event in the annotation file
            obs_config = {
                **self.config,
                'start_frame_offset': exp.get('start_frame', 0),  # offset for mapping back to source
                'source_fps': source_fps,
                'target_fps': self.config.get('target_fps', 30),
                'frame_count': frame_count  # total frames in clipped video
            }
            
            # Extract labels using domain-specific logic
            try:
                labels_df = self.extractor.extract_labels(
                    annotation_file,
                    observation_id,
                    obs_config
                )
            except Exception as e:
                print(f"Warning: Failed to extract labels for {observation_id}: {e}")
                continue
            
            # Get treatment
            metadata = exp.to_dict()
            treatment = self.extractor.get_treatment(observation_id, metadata)
            labels_df['T'] = treatment
            
            # Add covariates from experiment metadata
            if 'covariates' in self.config:
                for cov_name in self.config['covariates']:
                    if cov_name in exp:
                        labels_df[f'W_{cov_name}'] = exp[cov_name]
                        
            # Match frames with labels
            for _, row in labels_df.iterrows():
                frame_idx = int(row['frame_idx'])
                
                # Construct frame path (both frame_idx and filenames are 0-indexed)
                frame_name = f"frame_{frame_idx:06d}.jpg"
                frame_abs_path = frame_dir / frame_name
                
                # Only include if frame exists
                if frame_abs_path.exists():
                    frame_rel_path = str(frame_abs_path.relative_to(self.dataset_root))
                    
                    row_dict = {
                        'frame_path': frame_rel_path,
                        **row.to_dict()
                    }
                    all_rows.append(row_dict)
        
        # Create final dataframe
        dataset_df = pd.DataFrame(all_rows)
        
        if len(dataset_df) == 0:
            print(f"Warning: Generated empty dataset for {self.subject}/{self.version}")
            return dataset_df
        
        # Reorder columns: observation_id, frame_idx, fps, frame_path, T, W_*, Y_*
        base_cols = ['observation_id', 'frame_idx', 'fps', 'frame_path', 'T']
        covariate_cols = sorted([c for c in dataset_df.columns if c.startswith('W_')])
        outcome_cols = sorted([c for c in dataset_df.columns if c.startswith('Y_')])
        
        ordered_cols = base_cols + covariate_cols + outcome_cols
        dataset_df = dataset_df[ordered_cols]
        
        return dataset_df
    
    def save_dataset(self, output_path: Optional[Path] = None) -> Path:
        """
        Generate and save the dataset table to CSV.
        
        Args:
            output_path: Where to save the CSV
            
        Returns:
            Path to saved CSV file
        """
        if output_path is None:
            output_path = self.dataset_root / self.subject / self.version / "annotations.csv"
        
        output_path = Path(output_path)
        
        # Check if file exists and overwrite flag
        overwrite = self.config.get('overwrite_annotations', False)
        if output_path.exists() and not overwrite:
            print(f"⊙ Skipping annotation generation: dataset already exists (overwrite_annotations=False)")
            print(f"  Existing dataset found at: {output_path}")
            # Load and return existing dataset
            df = pd.read_csv(output_path)
            print(f"  Loaded: {len(df):,} frames")
            return output_path
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate dataset
        dataset_df = self.generate_dataset_table()
        
        # Save to CSV
        dataset_df.to_csv(output_path, index=False)
        
        print(f"✓ Annotations saved to {output_path}")
        print(f"  Total frames: {len(dataset_df):,}")
        print(f"  Columns: {list(dataset_df.columns)}")
        
        return output_path


def get_extractor(subject: str, version: str, config: Dict) -> AnnotationExtractor:
    """
    Factory function to get appropriate extractor for subject/version.
    
    Args:
        subject: Subject type (ants, frogs, mice)
        version: Version identifier (v1, v2, etc.)
        config: Configuration dictionary for this subject/version
        
    Returns:
        Appropriate AnnotationExtractor instance
    """
    outcome_mapping = config.get('outcome_mapping', {})
    
    if subject == "ants":
        if version == "v1":
            return AntsV1Extractor(outcome_mapping)
        elif version == "v2":
            return AntsV2Extractor(outcome_mapping)
    elif subject == "frogs":
        # Implement FrogsV1Extractor when ready
        raise NotImplementedError(f"Extractor for frogs {version} not yet implemented")
    elif subject == "mice":
        # Implement MiceV1Extractor when ready
        raise NotImplementedError(f"Extractor for mice {version} not yet implemented")
    
    raise ValueError(f"Unknown subject/version combination: {subject}/{version}")


import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Generate annotations using Hydra configuration"""
    subject = cfg.subject
    version = cfg.version
    
    workspace_root = Path(__file__).parent.parent.parent
    data_root = workspace_root / "data"
    dataset_root = workspace_root / "datasets"
    config_dir = workspace_root / "configs" / "datasets"
    
    generator = DatasetGenerator(
        data_root=data_root,
        dataset_root=dataset_root,
        config_dir=config_dir,
        subject=subject,
        version=version
    )
    
    output_path = generator.save_dataset()


if __name__ == "__main__":
    main()
