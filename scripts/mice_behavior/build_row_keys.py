"""
Write row_keys.parquet sidecars for the LEGACY mice v1 embedding caches, and optionally
verify them by recomputing features from the frames the keys name.

Background: the caches under dataset/mice/v1/embeddings/full/{dinov2,dinov3}/{class_l-2,
patch_grid4}/ were extracted from the old 2,556,000-frame HF shards (data-*-of-00234).
annotations.csv was rebuilt on 2026-07-20 (2,592,000 rows, other row order), so embedding
row != annotations row. The old-row -> current-annotations-row permutation is
dataset/mice/v1/eci/diagnostics/cls_l2_ann_row.npy (see README.txt next to it):
    class_l-2  : embedding row i  -> annotations row mapping[i]
    patch_grid4: embedding row i  -> annotations row mapping[global_idx[i]]
                 (global_idx.npy holds OLD frame-table rows)
The embedding arrays themselves are never modified by this script. Since then
scripts/mice_behavior/complete_emb_caches.py has APPENDED the missing frames after these
legacy rows (and extended row_keys.parquet / global_idx.npy); this script only derives and
checks the legacy prefix (first 2,556,000 class_l-2 rows, first 774,000 patch_grid4 rows).

Usage:
    python scripts/mice_behavior/build_row_keys.py            # write the 4 sidecars
    python scripts/mice_behavior/build_row_keys.py --verify   # + recompute >=8 frames per cache
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mice_behavior.emb_index import load_row_keys, write_row_keys  # noqa: E402

V1 = ROOT / 'dataset' / 'mice' / 'v1'
EMB = V1 / 'embeddings' / 'full'
MAPPING = V1 / 'eci' / 'diagnostics' / 'cls_l2_ann_row.npy'
N_OLD = 2_556_000
N_OLD_PATCH = 774_000  # legacy patch_grid4 rows (129 observations annotated at extraction time)
CACHES = {
    ('dinov2', 'class_l-2'): 'cls',
    ('dinov3', 'class_l-2'): 'cls',
    ('dinov2', 'patch_grid4'): 'patch',
    ('dinov3', 'patch_grid4'): 'patch',
}
MODEL_IDS = {'dinov2': 'facebook/dinov2-base', 'dinov3': 'facebook/dinov3-vitb16-pretrain-lvd1689m'}


def cache_ann_rows(emb_dir: Path, kind: str, mapping: np.ndarray) -> np.ndarray:
    """Current annotations row for each embedding row of this legacy cache."""
    if kind == 'cls':
        n = (emb_dir / 'embeddings.npy').stat().st_size // (4 * 768)
        assert n >= N_OLD, f'{emb_dir}: expected >= {N_OLD} rows, found {n}'
        return mapping
    g = np.load(emb_dir / 'global_idx.npy').astype(np.int64)[:N_OLD_PATCH]
    return mapping[g]


def open_cache(emb_dir: Path, kind: str, n_rows: int):
    if kind == 'cls':
        return np.memmap(emb_dir / 'embeddings.npy', dtype='float32', mode='r', shape=(n_rows, 768))
    return np.memmap(emb_dir / 'embeddings.npy', dtype='float16', mode='r', shape=(n_rows, 16, 768))


class Recomputer:
    """Re-extracts features with the same encoder/layer/preprocessing as the original runs:
    class_l-2 = hidden_states[-2][:, 0] (src/embedding/get_embeddings.py, layer=-2, token=class);
    patch_grid4 = last_hidden_state minus CLS/register tokens, adaptive-avg-pooled to 4x4
    (scripts/mice_behavior/extract_patch_embeddings.py). Processor: AutoImageProcessor(use_fast=True).
    Runs fp32 (the originals used fp16 autocast on GPU, so cosines are ~0.9999, not exactly 1)."""

    def __init__(self, device='cuda'):
        import torch
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}

    def _get(self, encoder):
        if encoder not in self.models:
            from transformers import AutoImageProcessor, AutoModel
            proc = AutoImageProcessor.from_pretrained(MODEL_IDS[encoder], use_fast=True)
            model = AutoModel.from_pretrained(MODEL_IDS[encoder]).to(self.device).eval()
            self.models[encoder] = (proc, model)
        return self.models[encoder]

    def __call__(self, encoder, kind, frame_paths):
        from PIL import Image
        torch = self.torch
        proc, model = self._get(encoder)
        images = [Image.open(ROOT / 'dataset' / p).convert('RGB') for p in frame_paths]
        px = proc(images=images, return_tensors='pt')['pixel_values'].to(self.device)
        with torch.inference_mode():
            out = model(pixel_values=px, output_hidden_states=True)
        if kind == 'cls':
            return out.hidden_states[-2][:, 0].float().cpu().numpy()
        n_prefix = 1 + getattr(model.config, 'num_register_tokens', 0)
        tok = out.last_hidden_state[:, n_prefix:].float()
        B, N, D = tok.shape
        side = int(round(N ** 0.5))
        grid = tok.transpose(1, 2).reshape(B, D, side, side)
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (4, 4)).reshape(B, D, 16).transpose(1, 2)
        return pooled.cpu().numpy()


def cosine(a, b):
    a, b = a.reshape(len(a), -1).astype(np.float64), b.reshape(len(b), -1).astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true', help='recompute features for sample rows and report cosines')
    p.add_argument('--n-verify', type=int, default=8)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    ann = pd.read_csv(V1 / 'annotations.csv', usecols=['observation_id', 'frame_idx', 'frame_path'])
    mapping = np.load(MAPPING).astype(np.int64)
    assert len(mapping) == N_OLD and len(np.unique(mapping)) == N_OLD and mapping.max() < len(ann)

    recompute = Recomputer(args.device) if args.verify else None
    all_ok = True
    for (enc, tok), kind in CACHES.items():
        emb_dir = EMB / enc / tok
        ann_rows = cache_ann_rows(emb_dir, kind, mapping)
        out = emb_dir / 'row_keys.parquet'
        if out.exists() and not args.overwrite:
            keys = load_row_keys(emb_dir)
            n = len(ann_rows)
            same = (len(keys) >= n
                    and (keys['observation_id'].values[:n] == ann['observation_id'].values[ann_rows]).all()
                    and (keys['frame_idx'].values[:n] == ann['frame_idx'].values[ann_rows]).all())
            print(f'[EXISTS] {out} ({len(keys):,} rows; legacy prefix of {n:,} matches derivation: {same})')
        else:
            if (emb_dir / 'embeddings.npy').stat().st_size > len(ann_rows) * (4 * 768 if kind == 'cls' else 2 * 16 * 768):
                raise SystemExit(f'{emb_dir} has appended rows beyond the legacy prefix; refusing to '
                                 f'overwrite its row_keys.parquet with legacy-only keys')
            write_row_keys(emb_dir, ann['observation_id'].values[ann_rows], ann['frame_idx'].values[ann_rows])
            print(f'[WROTE] {out} ({len(ann_rows):,} rows, '
                  f'{pd.unique(ann["observation_id"].values[ann_rows]).size} observations)')

        if recompute is None:
            continue
        n = len(ann_rows)
        # spread across the file, forcing several rows past the 612,000 divergence point
        rows = np.unique(np.concatenate([
            np.linspace(0, n - 1, args.n_verify // 2).astype(np.int64),
            np.linspace(612_000 + 1_234, n - 1, args.n_verify - args.n_verify // 2).astype(np.int64),
        ]))
        arr = open_cache(emb_dir, kind, n)
        cached = np.asarray(arr[rows], dtype=np.float32)
        keys = load_row_keys(emb_dir)
        # the frame named by the sidecar key -> its current annotations row -> its JPG
        key_df = keys.iloc[rows].reset_index(drop=True)
        ann_idx = ann.reset_index().merge(key_df, on=['observation_id', 'frame_idx'], how='right')
        fresh = recompute(enc, kind, ann_idx['frame_path'].tolist())
        cos_key = cosine(cached, fresh)
        # what positional indexing (embedding row == annotations row) would have compared against
        fresh_naive = recompute(enc, kind, ann['frame_path'].values[rows].tolist())
        cos_naive = cosine(cached, fresh_naive)
        ok = bool((cos_key > 0.999).all())
        all_ok &= ok
        print(f'  verify {enc}/{tok}: ' + ('OK' if ok else 'FAILED'))
        for r, k, ck, cn in zip(rows, key_df.itertuples(index=False), cos_key, cos_naive):
            print(f'    emb row {r:>9,} -> {k.observation_id} frame {k.frame_idx:>5}: '
                  f'cos {ck:.5f}   (positional row {r:,}: {cn:.4f})')
    if recompute is not None:
        print('ALL SIDECARS VERIFIED' if all_ok else 'VERIFICATION FAILED')
        sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
