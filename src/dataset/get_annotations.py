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

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import hydra
from omegaconf import DictConfig


class AnnotationExtractor:
    """Annotation extractor that works for all subjects/versions."""
    
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
    
    def extract_labels(
        self,
        annotation_file: Path,
        observation_id: str,
        config: Dict
    ) -> pd.DataFrame:
        """
        Extract labels from annotation file.
        
        Returns:
            DataFrame with one row per frame containing frame_idx, fps, and Y_* outcomes
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
                'engine': 'python'
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
        Determine treatment assignment from experiment metadata.
        """
        return metadata.get('treatment', 0)


class DatasetGenerator:
    """
    Generate dataset tables from extracted frames and annotations.
    
    Coordinates the extraction process and combines annotations with frame paths.
    """
    
    def __init__(
        self,
        data_root: Path,
        dataset_root: Path,
        config_dir: Path,
        subject: str,
        version: str,
        extractor: Optional[AnnotationExtractor] = None,
        overwrite: bool = False,
        annotations: str = 'yes',
    ):
        """
        Initialize the dataset generator.

        Args:
            data_root: Root directory containing raw data
            dataset_root: Root directory for processed datasets
            config_dir: Directory containing dataset configs
            subject: Subject type (ants, frogs, mice)
            version: Version of the dataset
            extractor: Annotation extractor (optional, will auto-create)
            overwrite: If True, regenerate annotations even if they exist
            annotations: Annotation availability — 'yes' | 'no' | 'partial'.
                         Comes from configs/experiment/{subject}/{version}.yaml.
        """
        self.data_root = Path(data_root)
        self.dataset_root = Path(dataset_root)
        self.subject = subject
        self.version = version
        self.overwrite = overwrite
        self.annotations = annotations

        # Load subject/version-specific dataset configuration
        config_path = Path(config_dir) / subject / f"{version}.yaml"
        if not config_path.exists():
            raise ValueError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create extractor if not provided
        if extractor is None:
            outcome_mapping = self.config.get('outcome_mapping', {})
            extractor = AnnotationExtractor(outcome_mapping)
        self.extractor = extractor
        
        # Define paths
        self.data_subject_dir = self.data_root / subject / version
        self.annotations_dir = self.data_subject_dir / "annotations"
        self.frames_dir = self.dataset_root / subject / version / "frames" / "full"
        self.experiment_csv = self.data_subject_dir / "experiment.csv"
        self.observations_metadata = self.data_subject_dir / "observations" / "metadata.json"
    
    def load_observations_metadata(self) -> Dict:
        """Load observations metadata with video and extracted frame FPS info."""
        if not self.observations_metadata.exists():
            raise FileNotFoundError(
                f"Observations metadata not found: {self.observations_metadata}\n"
                f"Run get_metadata.py first: src/data/get_metadata.py"
            )
        
        import json
        with open(self.observations_metadata, 'r') as f:
            return json.load(f)
    
    def load_experiment_metadata(self) -> pd.DataFrame:
        """Load experiment metadata CSV."""
        if not self.experiment_csv.exists():
            raise ValueError(f"Experiment CSV not found: {self.experiment_csv}")
        
        df = pd.read_csv(self.experiment_csv)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        return df
    
    def generate_dataset_table(self) -> pd.DataFrame:
        """
        Generate complete dataset table for all annotations.
        
        Returns:
            DataFrame with columns: frame_path, observation_id, frame_idx, fps, T, W_*, Y_*
        """
        all_rows = []
        
        # Load experiment metadata
        experiments = self.load_experiment_metadata()
        
        # Load observations metadata to get source FPS and actual target FPS
        obs_metadata = self.load_observations_metadata()
        source_fps = obs_metadata.get('source', {}).get('fps', None)
        
        # Get actual target FPS from the extracted frames (the ground truth)
        target_fps = obs_metadata.get('full', {}).get('fps', None)
        if target_fps is None:
            raise ValueError(
                f"target_fps not found in metadata for {self.subject}/{self.version}\n"
                f"Metadata path: {self.observations_metadata}\n"
                f"Run get_metadata.py first: src/data/get_metadata.py"
            )
        
        if source_fps is None:
            raise ValueError(
                f"source_fps not found in metadata for {self.subject}/{self.version}\n"
                f"Metadata path: {self.observations_metadata}\n"
                f"Run get_metadata.py first: src/data/get_metadata.py"
            )
        
        print(f"FPS conversion: source={source_fps} fps -> target={target_fps} fps")
        
        # Check global annotation availability (from experiment config)
        if self.annotations == 'no':
            print(f"Experiment config indicates no annotations available (annotations: {self.annotations})")
            expect_annotations = False
        elif self.annotations == 'partial':
            print(f"Experiment config indicates partial annotations (annotations: {self.annotations})")
            expect_annotations = True  # Will check per observation
        else:  # 'yes' or any other value
            expect_annotations = True
        
        # Process each observation
        for _, exp in tqdm(experiments.iterrows(), total=len(experiments), desc="Processing observations"):
            observation_id = exp['observation_id']
            
            # Skip invalid observations
            if exp.get('valid', 1) == 0:
                continue
            
            # Determine if this observation has annotations
            if not expect_annotations:
                # Config says no annotations for this dataset
                has_annotations = False
            else:
                # Check annotation file from experiment.csv
                annotation_filename = exp.get('annotation_file', '')
                
                # Check if annotation_filename is valid (not NaN, not empty)
                if pd.isna(annotation_filename) or not annotation_filename:
                    has_annotations = False
                    if self.annotations != 'partial':
                        print(f"Warning: No annotation file specified for {observation_id}, will assign NA to outcomes")
                else:
                    annotation_file = self.annotations_dir / annotation_filename
                    if not annotation_file.exists():
                        if self.annotations == 'partial':
                            print(f"Warning: Annotation file {annotation_filename} not found for {observation_id}, will assign NA to outcomes")
                            has_annotations = False
                        else:
                            raise FileNotFoundError(f"Annotation file {annotation_filename} not found for {observation_id}")
                    else:
                        has_annotations = True

            # Get observation file (video filename) for frame directory lookup
            observation_file = exp.get('observation_file', '')
            if observation_file:
                file_stem = Path(observation_file).stem
            else:
                file_stem = observation_id
            
            # Locate frames for this observation
            frame_dir = self.frames_dir / file_stem
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
                'source_fps': source_fps,  # FPS of original video (where annotations are recorded)
                'target_fps': target_fps,  # FPS of extracted frames (from metadata.json['full']['fps'])
                'frame_count': frame_count  # total frames in clipped video
            }
            
            # Extract labels or create NA labels if no annotation file
            if has_annotations:
                try:
                    labels_df = self.extractor.extract_labels(
                        annotation_file,
                        observation_id,
                        obs_config
                    )
                except Exception as e:
                    print(f"Warning: Failed to extract labels for {observation_id}: {e}")
                    continue
            else:
                # Create labels DataFrame with NA for outcomes
                outcome_names = self.config.get('outcomes', [])
                
                # If outcomes is None or empty, don't create any outcome columns
                if outcome_names is None or len(outcome_names) == 0:
                    outcome_names = []
                
                labels_data = {
                    'frame_idx': np.arange(frame_count),
                    'fps': np.full(frame_count, target_fps, dtype=float),
                    'observation_id': observation_id,
                    'T': None
                }
                # Add outcome columns with NA values (if outcomes are defined)
                for outcome_name in outcome_names:
                    labels_data[f'Y_{outcome_name}'] = np.nan
                labels_df = pd.DataFrame(labels_data)
            
            # Get treatment
            metadata = exp.to_dict()
            treatment = self.extractor.get_treatment(observation_id, metadata)
            labels_df['T'] = treatment
            
            # Add covariates from experiment metadata
            if 'covariates' in self.config:
                for cov_name in self.config['covariates']:
                    if cov_name in exp:
                        labels_df[f'W_{cov_name}'] = exp[cov_name]
                        
            # Build set of existing frame indices with a single glob (avoids per-frame NFS stat)
            existing_indices = {
                int(p.stem.split("_")[1])
                for p in frame_files
            }

            # Match frames with labels — vectorized path construction, no per-frame stat
            labels_df = labels_df[labels_df['frame_idx'].isin(existing_indices)].copy()
            labels_df['frame_path'] = labels_df['frame_idx'].apply(
                lambda i: str((frame_dir / f"frame_{i:06d}.jpg").relative_to(self.dataset_root))
            )
            all_rows.extend(labels_df.to_dict('records'))
        
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
        if output_path.exists() and not self.overwrite:
            try:
                # Check if file is not empty before trying to read
                if output_path.stat().st_size > 0:
                    df = pd.read_csv(output_path)
                    print(f"[SKIP] Annotations already generated ({len(df):,} frames)")
                    return output_path
                else:
                    print(f"[WARNING] Existing file is empty, regenerating: {output_path}")
            except pd.errors.EmptyDataError:
                print(f"[WARNING] Existing file has no data, regenerating: {output_path}")
            except Exception as e:
                print(f"[WARNING] Error reading existing file, regenerating: {e}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate dataset
        dataset_df = self.generate_dataset_table()
        
        # Save to CSV
        dataset_df.to_csv(output_path, index=False)
        
        print(f"Saved {len(dataset_df):,} frames with {len(dataset_df.columns)} columns")
        
        return output_path


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Generate annotations using Hydra configuration."""
    subject = cfg.subject
    version = cfg.version
    
    workspace_root = Path(__file__).parent.parent.parent
    data_root = workspace_root / "data"
    dataset_root = workspace_root / "dataset"
    config_dir = workspace_root / "configs" / "dataset"
    
    generator = DatasetGenerator(
        data_root=data_root,
        dataset_root=dataset_root,
        config_dir=config_dir,
        subject=subject,
        version=version,
        overwrite=cfg.overwrite.annotations,
        annotations=cfg.get('annotations', 'yes'),
    )
    
    output_path = generator.save_dataset()


if __name__ == "__main__":
    main()
