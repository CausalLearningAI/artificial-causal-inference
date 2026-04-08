"""PPCIDataset: wraps an HF Dataset + pre-extracted embeddings for PPCI training.

Responsibilities:
- Build Y tensor based on task (binary / multilabel / or / sum)
- Build environment IDs E from covariate combinations (warn & skip missing cols)
- Split train / val by full observation videos (by count or explicit list)
- Expose DataLoaders for flat training (ERM / DERM) and per-env training (vREx / IRM)
- Compute DERM sample weights
- Support concatenation of multiple PPCIDatasets (multi-dataset training)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .model import MLP

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tracking_distances(
    subject: str,
    version: str,
    obs_ids: np.ndarray,
    frame_idxs: np.ndarray,
    dataset_root: str = "./dataset",
    scale: float = 512.0,
) -> dict[str, np.ndarray]:
    """Load tracking CSVs and return per-pair normalised Euclidean distances.

    Returns a dict with keys ``"B2F"``, ``"Y2F"``, ``"B2Y"`` — each an
    ``(N,)`` float32 array of distances / *scale*.  Rows with missing tracking
    data (merge mismatches) are filled with 0.
    """
    import pandas as pd
    from pathlib import Path

    N = len(obs_ids)
    tracking_dir = Path(dataset_root) / subject / version / "tracking"
    parts = []
    for obs_id in np.unique(obs_ids):
        csv_path = tracking_dir / f"{obs_id}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(
            csv_path,
            usecols=["frame_idx", "blue_x", "blue_y", "yellow_x", "yellow_y",
                      "focal_x", "focal_y"],
        )
        df["observation_id"] = obs_id
        parts.append(df)

    if not parts:
        return {"B2F": np.zeros(N, dtype=np.float32),
                "Y2F": np.zeros(N, dtype=np.float32),
                "B2Y": np.zeros(N, dtype=np.float32)}

    tracking = pd.concat(parts, ignore_index=True)
    lookup = pd.DataFrame({
        "observation_id": obs_ids,
        "frame_idx":      frame_idxs,
        "_order":         np.arange(N),
    })
    merged = lookup.merge(tracking, on=["observation_id", "frame_idx"], how="left")
    merged = merged.sort_values("_order")

    bx = merged["blue_x"].to_numpy(dtype=np.float64)
    by = merged["blue_y"].to_numpy(dtype=np.float64)
    yx = merged["yellow_x"].to_numpy(dtype=np.float64)
    yy = merged["yellow_y"].to_numpy(dtype=np.float64)
    fx = merged["focal_x"].to_numpy(dtype=np.float64)
    fy = merged["focal_y"].to_numpy(dtype=np.float64)

    def _dist(ax, ay, bx_, by_):
        d = np.sqrt((ax - bx_) ** 2 + (ay - by_) ** 2) / scale
        return np.nan_to_num(d, nan=0.0).astype(np.float32)

    return {
        "B2F": _dist(bx, by, fx, fy),
        "Y2F": _dist(yx, yy, fx, fy),
        "B2Y": _dist(bx, by, yx, yy),
    }


def _label_encode(values: list) -> torch.Tensor:
    """Map arbitrary values (strings, ints, bools) to consecutive ints."""
    unique = sorted(set(str(v) for v in values))
    mapping = {v: i for i, v in enumerate(unique)}
    return torch.tensor([mapping[str(v)] for v in values], dtype=torch.long)


def compute_Y(dataset, outcome_cols: List[str]) -> torch.Tensor:
    """Build the training Y tensor from HF Dataset columns or a pandas DataFrame.

    Always returns per-column binary values (independent probes).
    Aggregation across outcomes (or / sum) is for evaluation only — see
    ``aggregate_probs`` in train.py.

    Returns:
        float (N,)    when len(outcome_cols) == 1
        float (N, k)  when len(outcome_cols) > 1
    """
    import pandas as pd
    if isinstance(dataset, pd.DataFrame):
        cols = [torch.from_numpy(dataset[c].to_numpy(dtype=np.float32)) for c in outcome_cols]
    else:
        cols = [torch.tensor(dataset[c], dtype=torch.float32) for c in outcome_cols]
    if len(cols) == 1:
        return cols[0]
    return torch.stack(cols, dim=1)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PPCIDataset:
    """Prediction-Powered Causal Inference dataset.

    Wraps a HuggingFace Dataset and pre-extracted embeddings for PPCI training.

    Args:
        dataset:               HF Dataset (columns: observation_id, T, W_*, Y_*).
        embeddings:            Embedding tensor (N, D) aligned with dataset rows.
        outcome_cols:          Which Y_* columns to use as outcome (one binary probe
                               per column; aggregation for eval is done separately).
        task:                  Aggregation used at evaluation time: "or" | "sum" | None.
                               Has no effect on training targets or the MLP.
        treatment_col:         Name of the treatment column (default "T").
        env_cols:              Columns defining environments (e.g. ["W_batch"]).
                               Missing columns are warned about and skipped.
        env_include_treatment: Also include T in the environment definition.
        n_val_videos:          Number of observation_ids held out for validation (0 = none).
        val_videos:            Explicit list of observation_ids for validation
                               (overrides n_val_videos when provided).
        seed:                  RNG seed for val split selection.
    """

    def __init__(
        self,
        dataset: HFDataset,
        embeddings: torch.Tensor,
        outcome_cols: List[str],
        task: str = "binary",
        treatment_col: str = "T",
        env_cols: Optional[List[str]] = None,
        env_include_treatment: bool = False,
        n_val_videos: int = 5,
        val_videos: Optional[List[str]] = None,
        seed: int = 0,
        name: Optional[str] = None,
    ):
        if len(embeddings) != len(dataset):
            raise ValueError(
                f"embeddings length {len(embeddings)} != dataset length {len(dataset)}"
            )

        self.name = name
        self.task = task
        self.outcome_cols = outcome_cols
        self.treatment_col = treatment_col
        self.env_cols_requested = list(env_cols or [])
        self.env_include_treatment = env_include_treatment
        self.seed = seed

        # Select only needed columns before to_pandas() — avoids deserializing
        # large columns (e.g. 'image') that can cost 70+s on 636k-row datasets.
        needed = (
            {"observation_id", "frame_idx", treatment_col}
            | set(outcome_cols)
            | set(env_cols or [])
            | {c for c in dataset.column_names if c.startswith("W_")}
        )
        df = dataset.select_columns([c for c in dataset.column_names if c in needed]).to_pandas()

        # core tensors
        self.X = embeddings.float()
        missing_cols = [c for c in outcome_cols if c not in df.columns]
        if missing_cols:
            # No outcome columns at all → all-NaN Y, nothing is annotated
            n = len(embeddings)
            self.Y = (
                torch.full((n, len(outcome_cols)), float("nan"))
                if len(outcome_cols) > 1
                else torch.full((n,), float("nan"))
            )
            self.annotated_mask = torch.zeros(n, dtype=torch.bool)
            self.has_annotations = False
        else:
            self.Y = compute_Y(df, outcome_cols)  # always per-column binary
            # Detect per-frame NaN (partial annotations: some obs annotated, others NaN)
            nan_mask = torch.isnan(self.Y)
            if nan_mask.any():
                # Per-frame: annotated iff no NaN across all outcome columns
                if self.Y.dim() == 1:
                    self.annotated_mask = ~nan_mask
                else:
                    self.annotated_mask = ~nan_mask.any(dim=1)
                n_ann = int(self.annotated_mask.sum())
                n_tot = len(self.Y)
                self.has_annotations = "partial"
                warnings.warn(
                    f"[PPCIDataset] Partial annotations: {n_ann}/{n_tot} frames "
                    f"({n_ann * 100 / n_tot:.1f}%) have ground-truth labels. "
                    f"Unannotated frames have NaN in Y.",
                    stacklevel=2,
                )
            else:
                self.annotated_mask = torch.ones(len(self.Y), dtype=torch.bool)
                self.has_annotations = True
        self.Yhat: Optional[torch.Tensor] = None   # set by add_predictions()
        self.T = np.array([str(v) for v in df[treatment_col].to_numpy()])
        self.obs_ids = df["observation_id"].to_numpy()
        self.frame_idx = torch.from_numpy(df["frame_idx"].to_numpy(dtype=np.int64))

        # covariates W
        self.W, self.W_cols = self._build_W(df)

        # environments E
        self.E, self.env_cols_used = self._build_E(df)

        # train/val split
        self._build_split(n_val_videos, val_videos)

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_W(self, dataset) -> Tuple[torch.Tensor, List[str]]:
        import pandas as pd
        if isinstance(dataset, pd.DataFrame):
            col_names = list(dataset.columns)
        else:
            col_names = dataset.column_names
        w_cols = sorted(c for c in col_names if c.startswith("W_"))
        if not w_cols:
            return torch.zeros(len(self.X), 0), []
        tensors = []
        for col in w_cols:
            vals = dataset[col].tolist() if isinstance(dataset, pd.DataFrame) else dataset[col]
            if isinstance(vals[0], (bool, np.bool_)):
                tensors.append(torch.from_numpy(np.array(vals, dtype=np.float32)))
            elif isinstance(vals[0], str):
                tensors.append(_label_encode(vals).float())
            else:
                tensors.append(torch.tensor(vals, dtype=torch.float32))
        return torch.stack(tensors, dim=1), w_cols

    def _build_E(self, dataset) -> Tuple[torch.Tensor, List[str]]:
        """Compute integer environment IDs from covariate combinations."""
        import pandas as pd
        if isinstance(dataset, pd.DataFrame):
            all_cols = list(dataset.columns)
        else:
            all_cols = dataset.column_names
        cols = list(self.env_cols_requested)

        missing = [c for c in cols if c not in all_cols]
        if missing:
            warnings.warn(
                f"[PPCIDataset] Environment columns not found in dataset (skipped): {missing}. "
                f"Available W_* columns: {[c for c in all_cols if c.startswith('W_')]}",
                stacklevel=3,
            )
            cols = [c for c in cols if c in all_cols]

        if self.env_include_treatment and self.treatment_col in all_cols:
            cols.append(self.treatment_col)

        if not cols:
            self.n_envs = 1
            return torch.zeros(len(self.X), dtype=torch.long), []

        if isinstance(dataset, pd.DataFrame):
            col_arrays = [dataset[c].astype(str).to_numpy() for c in cols]
        else:
            col_arrays = [np.asarray(dataset[c]).astype(str) for c in cols]
        if len(col_arrays) == 1:
            keys = col_arrays[0]
        else:
            keys = col_arrays[0].copy()
            for arr in col_arrays[1:]:
                keys = np.char.add(np.char.add(keys, "_"), arr)
        _, inverse = np.unique(keys, return_inverse=True)
        self.n_envs = int(inverse.max()) + 1
        return torch.from_numpy(inverse.astype(np.int64)), cols

    def _build_split(self, n_val_videos: int, val_videos: Optional[List[str]]):
        unique_vids = np.unique(self.obs_ids)
        if n_val_videos == 0 and val_videos is None:
            self.val_videos: List[str] = []
            self.train_mask = torch.ones(len(self.X), dtype=torch.bool)
            self.val_mask = torch.zeros(len(self.X), dtype=torch.bool)
            return
        if val_videos is not None:
            chosen = list(val_videos)
        else:
            rng = np.random.RandomState(self.seed)
            n = min(n_val_videos, len(unique_vids))
            chosen = [str(v) for v in rng.choice(unique_vids, size=n, replace=False)]
        self.val_videos = chosen
        val_flag = np.isin(self.obs_ids, chosen)
        self.val_mask = torch.tensor(val_flag, dtype=torch.bool)
        self.train_mask = ~self.val_mask

    # ------------------------------------------------------------------
    # Subset properties
    # ------------------------------------------------------------------

    @property
    def X_train(self) -> torch.Tensor:
        return self.X[self.train_mask]

    @property
    def Y_train(self) -> torch.Tensor:
        return self.Y[self.train_mask]

    @property
    def T_train(self) -> np.ndarray:
        return self.T[self.train_mask.numpy()]

    @property
    def E_train(self) -> torch.Tensor:
        return self.E[self.train_mask]

    @property
    def X_val(self) -> torch.Tensor:
        return self.X[self.val_mask]

    @property
    def Y_val(self) -> torch.Tensor:
        return self.Y[self.val_mask]

    @property
    def T_val(self) -> np.ndarray:
        return self.T[self.val_mask.numpy()]

    # ------------------------------------------------------------------
    # Temporal context
    # ------------------------------------------------------------------

    def apply_context_window(self, context_window: int, mode: str = "concat") -> "PPCIDataset":
        """Replace each frame embedding with a window of temporally-adjacent embeddings.

        For frame at temporal rank *r* within its observation, gathers
        embeddings at ranks ``r-k … r+k`` (clamped to observation boundaries)
        and either concatenates or mean-pools them.

        Args:
            context_window: Number of frames to include on each side (k).
                            0 → no-op (returns self unchanged).
            mode:           "concat" → output dim = (2k+1) × D  (default)
                            "mean"   → output dim = D (original)

        Returns:
            self, for method chaining.  Mutates ``self.X`` in-place.
        """
        if context_window <= 0:
            return self
        if getattr(self, "_context_window_applied", 0) > 0:
            return self

        k = context_window
        N, D = self.X.shape
        obs_ids = self.obs_ids
        frame_idx_np = self.frame_idx.numpy()

        # Build (N, 2k+1) matrix of flat-position neighbours for each frame.
        neighbor_matrix = np.empty((N, 2 * k + 1), dtype=np.int64)

        for obs in np.unique(obs_ids):
            local_pos = np.where(obs_ids == obs)[0]          # flat positions in this obs
            local_fi  = frame_idx_np[local_pos]              # their frame indices
            sort_ord  = np.argsort(local_fi)                 # sort by frame_idx
            sorted_flat = local_pos[sort_ord]                # flat positions in temporal order
            n_obs = len(sorted_flat)

            # inv_sort[i] = temporal rank of local_pos[i]
            inv_sort = np.empty(n_obs, dtype=np.int64)
            inv_sort[sort_ord] = np.arange(n_obs)

            for j, offset in enumerate(range(-k, k + 1)):
                nb_ranks = np.clip(inv_sort + offset, 0, n_obs - 1)
                neighbor_matrix[local_pos, j] = sorted_flat[nb_ranks]

        # Process in row-chunks so the per-step temporary is tiny.
        # Without chunking, mean mode peaks at 3×(N,D) ≈ 5.4 GB for N=636k,
        # D=768; chunked, peak ≈ 2×(N,D) + chunk_size×D ≈ 3.6 GB + ~150 MB.
        _CHUNK = 50_000
        self._embed_dim: int = D
        if mode == "concat":
            self._context_size: int = 2 * k + 1
            out = torch.empty(N, (2 * k + 1) * D, dtype=self.X.dtype)
            for start in range(0, N, _CHUNK):
                rows = slice(start, min(start + _CHUNK, N))
                parts = [self.X[neighbor_matrix[rows, j]] for j in range(2 * k + 1)]
                out[rows] = torch.cat(parts, dim=1)
        else:  # mean
            self._context_size = 1
            out = torch.zeros(N, D, dtype=self.X.dtype)
            for start in range(0, N, _CHUNK):
                rows = slice(start, min(start + _CHUNK, N))
                chunk = torch.zeros(rows.stop - rows.start, D, dtype=self.X.dtype)
                for j in range(2 * k + 1):
                    chunk.add_(self.X[neighbor_matrix[rows, j]])
                out[rows] = chunk.div_(2 * k + 1)

        del neighbor_matrix
        self.X = out

        self._context_window_applied: int = k
        return self

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def _train_stride_indices(self, frame_stride: int = 1) -> torch.Tensor:
        """Indices into X_train / Y_train / E_train for temporal subsampling.

        Keeps frames where frame_idx % frame_stride == 0, i.e. one frame per
        ``frame_stride`` consecutive frames.  With frame_stride=1 (default)
        all training frames are kept.

        Unannotated frames (NaN in Y) are always excluded so that NaN never
        enters loss computation.
        """
        ann = self.annotated_mask[self.train_mask]
        if frame_stride <= 1:
            return ann.nonzero(as_tuple=True)[0]
        fi = self.frame_idx[self.train_mask]
        return (ann & (fi % frame_stride == 0)).nonzero(as_tuple=True)[0]

    def get_train_loader(
        self,
        batch_size: int,
        weights: Optional[torch.Tensor] = None,
        shuffle: bool = True,
        frame_stride: int = 1,
    ) -> DataLoader:
        """Flat DataLoader over training samples.

        Args:
            weights:      Optional sample weights (N_train,) for DERM.
                          When provided, each batch yields (X, Y, E, w).
            frame_stride: Keep 1 frame per ``frame_stride`` frames
                          (reduces temporal correlation; default 1 = all frames).

        Unannotated frames (NaN in Y) are automatically excluded.
        """
        idx = self._train_stride_indices(frame_stride)
        tensors: list = [self.X_train[idx], self.Y_train[idx], self.E_train[idx]]
        if weights is not None:
            tensors.append(weights[idx])
        ds = TensorDataset(*tensors)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def get_env_train_loaders(self, batch_size: int, frame_stride: int = 1) -> List[DataLoader]:
        """Per-environment DataLoaders for vREx / IRM.

        Args:
            frame_stride: Keep 1 frame per ``frame_stride`` frames
                          (reduces temporal correlation; default 1 = all frames).

        Returns one DataLoader per training environment.
        Unannotated frames (NaN in Y) are automatically excluded.
        """
        idx = self._train_stride_indices(frame_stride)
        X = self.X_train[idx]
        Y = self.Y_train[idx]
        E = self.E_train[idx]

        loaders = []
        for e in E.unique().tolist():
            mask = E == e
            loaders.append(
                DataLoader(
                    TensorDataset(X[mask], Y[mask]),
                    batch_size=batch_size, shuffle=True, drop_last=False,
                )
            )
        return loaders

    # ------------------------------------------------------------------
    # DERM weights
    # ------------------------------------------------------------------

    def compute_derm_weights(self) -> torch.Tensor:
        """Compute per-sample DERM weights for training samples.

        weight_i  =  Var(Y | E=e_i)  /  P(Y=y_i, E=e_i)

        Multilabel task falls back to uniform weights with a warning.
        """
        # For multi-output, use the mean across outcome columns
        Y_tr_raw = self.Y_train
        Y_tr = (Y_tr_raw.float().mean(dim=1) if Y_tr_raw.dim() == 2 else Y_tr_raw.float())
        E_tr = self.E_train
        N = len(Y_tr)

        var_map: Dict[int, float] = {}
        count_map: Dict[Tuple[int, int], int] = {}

        for e in E_tr.unique().tolist():
            mask = E_tr == e
            y_e = Y_tr[mask]
            n_e = int(mask.sum())
            var_map[e] = float(y_e.var().item()) if n_e > 1 else 0.0

            n_classes = int(Y_tr.max().item()) + 1
            for cls in range(n_classes):
                count_map[(e, cls)] = int((y_e.round().long() == cls).sum().item())

        weights = torch.zeros(N, dtype=torch.float32)
        for i, (e_i, y_i) in enumerate(zip(E_tr.tolist(), Y_tr.tolist())):
            v = var_map.get(e_i, 0.0)
            cnt = count_map.get((e_i, int(round(y_i))), 0)
            prob = max(cnt / N, 1e-6)
            weights[i] = v / prob

        mean_w = weights.mean()
        if mean_w > 0:
            weights = weights / mean_w
        return weights

    # ------------------------------------------------------------------
    # Convenience constructor from disk
    # ------------------------------------------------------------------

    @classmethod
    def from_disk(
        cls,
        subject: str,
        version: str,
        encoder: str,
        token: str,
        frame_type: str = "full",
        dist_mode: str = "none",
        dataset_root: str = "./dataset",
        **kwargs,
    ) -> "PPCIDataset":
        """Load HF dataset + embeddings from disk and construct a PPCIDataset.

        Args:
            subject:      Domain name, e.g. "ants".
            version:      Dataset version, e.g. "v3".
            encoder:      Embedding encoder name, e.g. "dinov2".
            token:        Token type, e.g. "class".
            frame_type:   "full" (default) or "pov".
            dist_mode:    How to handle tracking distances:
                          "none"  — no distances (default).
                          "early" — distances concatenated with embeddings
                                    (model treats them as part of the embedding).
                          "late"  — distances concatenated with embeddings but
                                    the model separates them for late fusion.
                          For "full": appends [dist_B2F, dist_Y2F] (2 features).
                          For "pov":  each half gets [dist_to_focal, dist_to_other]
                                      from that ant's perspective (2 features per half).
            dataset_root: Root of the dataset directory (default "./dataset").
            **kwargs:     Forwarded to PPCIDataset.__init__ (env_cols, n_val_videos, …).

        Returns:
            A fully initialised PPCIDataset.
        """
        from src.dataset.get_dataset import load_dataset
        from src.embedding.get_embeddings import load_embeddings_from_disk

        hf = load_dataset(subject, version, from_disk=True)
        use_dist = dist_mode in ("early", "late")

        if frame_type == "full":
            emb = load_embeddings_from_disk(subject, version, encoder, token,
                                            dataset_root=dataset_root)
            if use_dist:
                meta = hf.select_columns(["observation_id", "frame_idx"]).to_pandas()
                dists = _load_tracking_distances(
                    subject, version,
                    meta["observation_id"].to_numpy(),
                    meta["frame_idx"].to_numpy(dtype=np.int64),
                    dataset_root=dataset_root,
                )
                # [dist_B2F, dist_Y2F] — both pairwise distances to focal
                dist_tensor = torch.from_numpy(
                    np.stack([dists["B2F"], dists["Y2F"]], axis=1)
                )
                emb = torch.cat([emb, dist_tensor], dim=1)  # (N, D+2)

            kwargs.setdefault("name", f"{subject} {version}")
            ds = cls(hf, emb, **kwargs)
            ds.frame_type = "full"
            ds.n_dist = 2 if dist_mode == "late" else 0
            return ds

        # ── POV branch ────────────────────────────────────────────────────────
        meta = hf.select_columns(["observation_id", "frame_idx"]).to_pandas()
        obs_ids_arr   = meta["observation_id"].to_numpy()
        frame_idx_arr = meta["frame_idx"].to_numpy(dtype=np.int64)

        emb_blue = load_embeddings_from_disk(subject, version, encoder, token,
                                             dataset_root=dataset_root,
                                             frame_type="pov", pov_identity="blue")
        emb_yellow = load_embeddings_from_disk(subject, version, encoder, token,
                                               dataset_root=dataset_root,
                                               frame_type="pov", pov_identity="yellow")

        # Replace NaN embeddings with zeros (some POV crops fail extraction)
        for name, emb_t in [("blue", emb_blue), ("yellow", emb_yellow)]:
            nan_mask = torch.isnan(emb_t).any(dim=1)
            n_nan = int(nan_mask.sum())
            if n_nan > 0:
                import warnings
                warnings.warn(
                    f"[PPCIDataset] {n_nan}/{len(emb_t)} NaN rows in {name} "
                    f"POV embeddings ({subject}/{version}/{encoder}/{token}) "
                    f"— replaced with zeros"
                )
                emb_t[nan_mask] = 0.0

        if use_dist:
            dists = _load_tracking_distances(
                subject, version, obs_ids_arr, frame_idx_arr,
                dataset_root=dataset_root,
            )
            # Blue half: [emb_blue | dist_B2F | dist_B2Y]
            # Yellow half: [emb_yellow | dist_Y2F | dist_Y2B]
            blue_dists = torch.from_numpy(
                np.stack([dists["B2F"], dists["B2Y"]], axis=1)
            )
            yellow_dists = torch.from_numpy(
                np.stack([dists["Y2F"], dists["B2Y"]], axis=1)
            )
            parts = [emb_blue, blue_dists, emb_yellow, yellow_dists]
            n_dist = 2
        else:
            parts = [emb_blue, emb_yellow]
            n_dist = 0

        emb_pov = torch.cat(parts, dim=1)
        kwargs.setdefault("name", f"{subject} {version} pov")
        ds = cls(hf, emb_pov, **kwargs)
        ds.frame_type = "pov"
        ds.n_dist = n_dist if dist_mode == "late" else 0
        return ds

    # ------------------------------------------------------------------
    # Multi-dataset concatenation
    # ------------------------------------------------------------------

    @classmethod
    def concat(cls, datasets: List["PPCIDataset"]) -> "PPCIDataset":
        """Merge multiple PPCIDatasets for multi-dataset training.

        Environments are kept isolated per source dataset: environment e from
        dataset i is mapped to a unique global ID, so environments with the
        same covariate values in different source datasets are treated as
        *different* environments (safe default when data distributions differ).

        The train/val masks from each source dataset are preserved.

        All datasets must share the same task, outcome_cols, and treatment_col.
        """
        if not datasets:
            raise ValueError("datasets list is empty")
        if len(datasets) == 1:
            return datasets[0]

        ref = datasets[0]
        for i, d in enumerate(datasets[1:], 1):
            if d.task != ref.task:
                raise ValueError(f"datasets[{i}].task={d.task} != datasets[0].task={ref.task}")
            if d.outcome_cols != ref.outcome_cols:
                raise ValueError(
                    f"datasets[{i}].outcome_cols={d.outcome_cols} != {ref.outcome_cols}"
                )
            if d.treatment_col != ref.treatment_col:
                raise ValueError(
                    f"datasets[{i}].treatment_col={d.treatment_col} != {ref.treatment_col}"
                )

        # --- Re-map environment IDs to avoid collisions ---
        env_offset = 0
        E_list = []
        for d in datasets:
            E_list.append(d.E + env_offset)
            env_offset += int(d.E.max().item()) + 1

        # --- Prefix observation IDs to avoid collision ---
        obs_ids = np.concatenate(
            [np.array([f"ds{i}_{oid}" for oid in d.obs_ids]) for i, d in enumerate(datasets)]
        )

        # --- Concatenate tensors ---
        X = torch.cat([d.X for d in datasets], dim=0)
        Y = torch.cat([d.Y for d in datasets], dim=0)
        T = np.concatenate([d.T for d in datasets], axis=0)
        E = torch.cat(E_list, dim=0)
        frame_idx = torch.cat([d.frame_idx for d in datasets], dim=0)

        # W: align columns across datasets
        all_W_cols = ref.W_cols
        W_parts = []
        for d in datasets:
            if d.W_cols == all_W_cols:
                W_parts.append(d.W)
            else:
                # Pad missing columns with zeros, reorder to match ref
                part = torch.zeros(len(d.X), len(all_W_cols))
                for j, col in enumerate(all_W_cols):
                    if col in d.W_cols:
                        src_j = d.W_cols.index(col)
                        part[:, j] = d.W[:, src_j]
                W_parts.append(part)
        W = torch.cat(W_parts, dim=0)

        # Train/val masks
        train_mask = torch.cat([d.train_mask for d in datasets], dim=0)
        val_mask = torch.cat([d.val_mask for d in datasets], dim=0)

        # Build the merged instance without going through the normal constructor
        obj = cls.__new__(cls)
        obj.name = " + ".join(d.name for d in datasets if d.name) or None
        obj.task = ref.task
        obj.outcome_cols = ref.outcome_cols
        obj.treatment_col = ref.treatment_col
        obj.env_cols_requested = ref.env_cols_requested
        obj.env_include_treatment = ref.env_include_treatment
        obj.env_cols_used = ref.env_cols_used
        obj.seed = ref.seed
        obj.X = X
        obj.Y = Y
        obj.T = T
        obj.E = E
        obj.W = W
        obj.W_cols = all_W_cols
        obj.obs_ids = obs_ids
        obj.frame_idx = frame_idx
        obj.train_mask = train_mask
        obj.val_mask = val_mask
        obj.annotated_mask = torch.cat([d.annotated_mask for d in datasets], dim=0)
        # has_annotations: True if all fully annotated, "partial" if any partial/missing
        ann_states = [d.has_annotations for d in datasets]
        if all(a is True for a in ann_states):
            obj.has_annotations = True
        elif all(a is False for a in ann_states):
            obj.has_annotations = False
        else:
            obj.has_annotations = "partial"
        obj.n_envs = int(E.max().item()) + 1
        obj.val_videos = [vid for d in datasets for vid in d.val_videos]
        return obj

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n_train = int(self.train_mask.sum())
        n_val = int(self.val_mask.sum())
        n_envs_tr = int(self.E_train.unique().numel())
        lines = ["PPCIDataset summary:"]
        if self.name:
            lines.append(f"  name            : {self.name}")
        n_annotated = int(self.annotated_mask.sum())
        n_ann_obs = len(np.unique(self.obs_ids[self.annotated_mask.numpy()]))
        n_tot_obs = len(np.unique(self.obs_ids))
        lines += [
            f"  total frames    : {len(self.X):,}  (train={n_train:,}, val={n_val:,})",
            f"  annotated frames: {n_annotated:,}/{len(self.X):,}  ({n_ann_obs} / {n_tot_obs} observations)",
            f"  embedding dim   : {self.X.shape[1]}",
            f"  task            : {self.task}",
            f"  outcome columns : {self.outcome_cols}",
            f"  treatment values: {sorted(np.unique(self.T).tolist())}",
            f"  obs per treatment: {[int(np.unique(self.obs_ids[self.T == t]).size) for t in sorted(np.unique(self.T).tolist())]}",
            f"  env columns     : {self.env_cols_used}",
            f"  n_envs (train)  : {n_envs_tr}",
            f"  val videos      : {self.val_videos}",
        ]
        return "\n".join(lines)

    def add_predictions(self, model: MLP, device: torch.device, batch_size: int = 4096) -> PPCIDataset:
        """Run model inference and attach frame-level predictions to the dataset.

        Stores predictions as ``self.Yhat`` (shape matching ``self.Y``).
        Subsequent calls to ``obs_level()`` will include ``Yhat_*`` columns.
        Returns ``self`` for method chaining::

            obs_df = dataset.add_predictions(model, device).obs_level()

        Args:
            batch_size: Number of frames per forward pass.  Reduce if OOM
                        (relevant when context_window > 0 inflates embedding dim).
        """
        import torch as _torch
        model.eval()
        chunks = []
        with _torch.no_grad():
            for start in range(0, len(self.X), batch_size):
                x_batch = self.X[start : start + batch_size].to(device)
                chunks.append(model.probs(x_batch).cpu())
        self.Yhat = _torch.cat(chunks, dim=0)
        return self

    def obs_level(self, eval_task: Optional[str] = None) -> pd.DataFrame:
        """Aggregate frame-level data to observation level (one row per video).

        Frames are averaged (mean) within each observation.
        Includes ground-truth Y columns always, and Yhat columns if
        ``add_predictions()`` has been called first.

        Args:
            eval_task: Optional cross-outcome aggregation applied to both Y and
                       Yhat (when predictions are present):
                       "or"  → P(any positive) = 1 − ∏(1 − p_k)
                       "sum" → expected count  = Σ p_k

        Returns:
            DataFrame with one row per observation_id and columns:
              observation_id   — unique video identifier
              T                — treatment (constant within observation)
              Y_{label}        — true outcome, aggregated over frames
              Y_avg            — mean across outcomes (multilabel only)
              Yhat_{label}     — predicted outcome (if add_predictions() called)
              Yhat_avg         — mean across predicted outcomes (multilabel only)
              Y_{task}         — cross-outcome true outcome  (if eval_task set)
              Yhat_{task}      — cross-outcome prediction    (if eval_task set)
              W_*              — covariates (first value within each observation)
              _has_annotations — per-observation bool: True if ground-truth Y is
                                 real (not NaN), False for unannotated observations
        """
        Y = self.Y.float()
        rows: dict = {
            "observation_id": self.obs_ids,
            "T": self.T,
        }

        # Ground-truth Y columns
        if Y.dim() == 1:
            rows[f"Y_{self.outcome_cols[0].replace('Y_', '')}"] = Y.numpy()
        else:
            for k, col in enumerate(self.outcome_cols):
                rows[f"Y_{col.replace('Y_', '')}"] = Y[:, k].numpy()
            if Y.shape[1] > 1:
                rows["Y_avg"] = Y.mean(dim=-1).numpy()

        # Predicted Yhat columns (only if add_predictions() was called)
        if self.Yhat is not None:
            Yhat = self.Yhat
            if Yhat.dim() == 1:
                rows[f"Yhat_{self.outcome_cols[0].replace('Y_', '')}"] = Yhat.numpy()
            else:
                for k, col in enumerate(self.outcome_cols):
                    rows[f"Yhat_{col.replace('Y_', '')}"] = Yhat[:, k].numpy()
                if Yhat.shape[1] > 1:
                    rows["Yhat_avg"] = Yhat.mean(dim=-1).numpy()

            # Optional cross-outcome aggregation
            if eval_task in ("or", "sum") and Yhat.dim() == 2:
                if eval_task == "or":
                    rows["Yhat_or"] = (1.0 - (1.0 - Yhat).prod(dim=-1)).numpy()
                    rows["Y_or"]    = (1.0 - (1.0 - Y).prod(dim=-1)).numpy()
                else:  # sum
                    rows["Yhat_sum"] = Yhat.sum(dim=-1).numpy()
                    rows["Y_sum"]    = Y.sum(dim=-1).numpy()

        # Covariates
        if self.W.shape[1] > 0:
            for i, col in enumerate(self.W_cols):
                rows[col] = self.W[:, i].numpy()

        # Per-frame annotation flag (for per-observation aggregation)
        rows["_annotated"] = self.annotated_mask.numpy().astype(float)

        df_frames = pd.DataFrame(rows)
        y_cols = [c for c in df_frames.columns if c.startswith(("Y_", "Yhat_"))]
        w_cols = [c for c in df_frames.columns if c.startswith("W_")]

        # pandas .mean() skips NaN by default: annotated obs get real Y values,
        # unannotated obs get NaN — exactly the right behaviour.
        obs_y    = df_frames.groupby("observation_id")[y_cols].mean().reset_index()
        obs_meta = df_frames.groupby("observation_id")[["T"] + w_cols].first().reset_index()
        # Per-observation: annotated if ALL frames in that observation are annotated
        obs_ann  = df_frames.groupby("observation_id")["_annotated"].min().reset_index()
        obs_ann["_has_annotations"] = obs_ann["_annotated"] > 0.5
        obs_ann = obs_ann.drop(columns=["_annotated"])

        result = obs_y.merge(obs_meta, on="observation_id").merge(obs_ann, on="observation_id")
        return result

    def __repr__(self) -> str:
        return self.summary()

    def __len__(self) -> int:
        return len(self.X)
