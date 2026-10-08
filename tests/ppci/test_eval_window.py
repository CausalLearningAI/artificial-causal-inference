"""Tests for the annotated-window mechanism (experiment.csv `annotation_end_frame`).

get_annotations.py writes a per-row `eval_window` flag and NaN outcomes past the window when
experiment.csv has the column; PPCIDataset.from_disk drops rows outside the window before any
context window (eval_window_only=True, the default) and windows the embedding cache to match.

Synthetic world: subject 's', version 'v', source 30 fps -> 5 fps (ratio 6), 12 ML frames per
observation, annotation_end_frame = 36 source frames = ML frames 0-5.
  obsA annotated: groom-orange [0, 12) -> ML 0-1 Y2F; groom-blue [24, 36) -> ML 4-5 B2F;
                  groom-orange [48, 54) -> ML 8, past the window (must end up NaN, not 1)
  obsB no annotation file (all NaN)

(a) labels past the window are NaN (never 0), eval_window flags, HF schema keeps the flag
(b) no column -> no eval_window, old labels (0 past the annotations); empty cell == no column
(c) from_disk filters BEFORE the context window
(d) legacy cache (window rows only) accepted with a warning; any other length raises
(e) eval_window_only=False keeps every row; an HF dataset without the column is unchanged
(f) POV: windowed first, then the NaN -> 0 replacement (count covers in-window rows only)

Runs standalone: `python tests/ppci/test_eval_window.py`, or under pytest.
"""
import json
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import src.dataset.get_dataset as gd  # noqa: E402
from src.dataset.get_annotations import DatasetGenerator  # noqa: E402
from ppci.dataset import PPCIDataset  # noqa: E402

N_ML = 12          # ML frames per observation
WIN = 6            # ML frames inside the window (36 source frames / ratio 6)
OBS = ['obsA', 'obsB']
D = 6
DS_KW = dict(outcome_cols=['Y_Y2F', 'Y_B2F'], task='multilabel', n_val_videos=0)


def _world(tmp: Path, ann_end=36, with_col=True) -> Path:
    data = tmp / 'data' / 's' / 'v'
    (data / 'annotations').mkdir(parents=True)
    (data / 'observations').mkdir()
    exp = pd.DataFrame({'observation_id': OBS, 'observation_file': [f'{o}.mkv' for o in OBS],
                        'annotation_file': ['obsA.csv', np.nan], 'treatment': ['A', 'B'],
                        'batch': [1, 2], 'valid': [1, 1], 'start_frame': [0, 0],
                        'end_frame': [N_ML * 6, N_ML * 6]})
    if with_col:
        exp['annotation_end_frame'] = ann_end
    exp.to_csv(data / 'experiment.csv', index=False)
    rows = ['groom-orange,0,12', 'groom-blue,24,36', 'groom-orange,48,54']
    (data / 'annotations' / 'obsA.csv').write_text(
        'x\nx\nx\nBehavior,Beginning-frame,End-frame\n' + '\n'.join(rows) + '\nfooter\n')
    (data / 'observations' / 'metadata.json').write_text(
        json.dumps({'source': {'fps': 30.0}, 'full': {'fps': 5.0}}))
    for o in OBS:
        fdir = tmp / 'dataset' / 's' / 'v' / 'frames' / 'full' / o
        fdir.mkdir(parents=True)
        for i in range(N_ML):
            Image.new('RGB', (4, 4)).save(fdir / f'frame_{i:06d}.jpg')
    cfg = tmp / 'configs' / 'dataset' / 's'
    cfg.mkdir(parents=True)
    cfg.joinpath('v.yaml').write_text(yaml.safe_dump({
        'treatment': {'column': 'treatment', 'type': 'categorical'},
        'covariates': {'batch': {'type': 'string'}},
        'outcomes': ['Y2F', 'B2F'],
        'outcome_mapping': {'groom-orange': [1, 0], 'groom-blue': [0, 1]},
        'annotation_format': {'skiprows': 3, 'skipfooter': 1},
    }))
    return tmp


def _table(tmp: Path) -> pd.DataFrame:
    gen = DatasetGenerator(tmp / 'data', tmp / 'dataset', tmp / 'configs' / 'dataset', 's', 'v',
                           annotations='partial')
    return gen.generate_dataset_table()


def _hf(tmp: Path):
    """annotations.csv -> HF dataset saved to hf/full and loaded back, as the pipeline does."""
    df = _table(tmp)
    df.to_csv(tmp / 'dataset' / 's' / 'v' / 'annotations.csv', index=False)
    kw = dict(dataset_root=tmp / 'dataset', config_root=tmp / 'configs')
    ds = gd.load_dataset('s', 'v', **kw)
    ds.info.__dict__.pop('metadata', None)      # as gd.save_dataset, without its num_proc=8
    ds.save_to_disk(str(tmp / 'dataset' / 's' / 'v' / 'hf' / 'full'))
    return gd.load_dataset('s', 'v', from_disk=True, **kw)


def _cache(n_rows: int) -> torch.Tensor:
    """Row r of a full-length cache = [obs code, frame_idx, noise...]."""
    obs = np.repeat([1, 2], N_ML)[:n_rows]
    fi = np.tile(np.arange(N_ML), 2)[:n_rows]
    v = np.random.default_rng(0).normal(size=(n_rows, D)).astype(np.float32)
    v[:, 0], v[:, 1] = obs, fi
    return torch.from_numpy(v)


def _write_cache(tmp: Path, emb: torch.Tensor, frame_type='full', identity='blue'):
    p = tmp / 'dataset' / 's' / 'v' / 'embeddings' / frame_type
    p = (p / identity if frame_type == 'pov' else p) / 'enc' / 'class'
    p.mkdir(parents=True, exist_ok=True)
    torch.save(emb, p / 'embeddings.pt')


def _from_disk(tmp: Path, hf, **kw):
    orig = gd.load_dataset
    gd.load_dataset = lambda *a, **k: hf
    try:
        return PPCIDataset.from_disk('s', 'v', 'enc', 'class', dataset_root=str(tmp / 'dataset'),
                                     **DS_KW, **kw)
    finally:
        gd.load_dataset = orig


def _keep() -> np.ndarray:
    return np.tile(np.arange(N_ML) < WIN, 2)


def test_labels_past_window_are_nan():
    with tempfile.TemporaryDirectory() as d:
        tmp = _world(Path(d))
        df = _table(tmp)
        a = df[df.observation_id == 'obsA'].sort_values('frame_idx')
        assert a['eval_window'].tolist() == [True] * WIN + [False] * (N_ML - WIN)
        assert a['Y_Y2F'].iloc[:WIN].tolist() == [1, 1, 0, 0, 0, 0]
        assert a['Y_B2F'].iloc[:WIN].tolist() == [0, 0, 0, 0, 1, 1]
        assert a[['Y_Y2F', 'Y_B2F']].iloc[WIN:].isna().all().all()   # frame 8 annotated, still NaN
        b = df[df.observation_id == 'obsB']
        assert b['eval_window'].tolist() == [True] * WIN + [False] * (N_ML - WIN)
        assert b[['Y_Y2F', 'Y_B2F']].isna().all().all()
        hf = _hf(tmp)
        assert str(hf.features['eval_window'].dtype) == 'bool'
        hp = hf.to_pandas()
        assert hp['eval_window'].to_numpy().tolist() == _keep().tolist()
        assert hp.loc[~_keep(), 'Y_Y2F'].isna().all() and (hp.loc[_keep() & (hp.observation_id == 'obsA'), 'Y_Y2F'].notna()).all()


def test_no_column_no_change():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        nocol = _table(_world(Path(d1), with_col=False))
        assert 'eval_window' not in nocol.columns
        a = nocol[nocol.observation_id == 'obsA'].sort_values('frame_idx')
        # old behaviour: zeros past the annotations, the frame-8 bout labelled
        assert a['Y_Y2F'].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        empty = _table(_world(Path(d2), ann_end=np.nan))
        assert empty['eval_window'].all()
        pd.testing.assert_frame_equal(empty.drop(columns='eval_window'), nocol)


def test_filter_before_context_window():
    with tempfile.TemporaryDirectory() as d:
        tmp = _world(Path(d))
        hf = _hf(tmp)
        full = _cache(2 * N_ML)
        _write_cache(tmp, full)
        ds = _from_disk(tmp, hf)
        assert len(ds) == 2 * WIN
        assert torch.equal(ds.X, full[torch.from_numpy(_keep())])
        assert ds.frame_idx.max().item() == WIN - 1
        assert ds.has_annotations == 'partial' and int(ds.annotated_mask.sum()) == WIN
        ds.apply_context_window(2, mode='mean')
        # last in-window frame of obsA: neighbours clamp at frame 5, never reach frames 6-7
        expect = full[[3, 4, 5, 5, 5]].mean(dim=0)
        got = ds.X[(ds.obs_ids == 'obsA') & (ds.frame_idx.numpy() == WIN - 1)][0]
        assert torch.allclose(got, expect, atol=1e-6), (got, expect)
        leaky = full[[3, 4, 5, 6, 7]].mean(dim=0)
        assert not torch.allclose(got, leaky, atol=1e-3)


def test_legacy_cache_and_wrong_length():
    with tempfile.TemporaryDirectory() as d:
        tmp = _world(Path(d))
        hf = _hf(tmp)
        full = _cache(2 * N_ML)
        legacy = full[torch.from_numpy(_keep())].clone()
        _write_cache(tmp, legacy)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ds = _from_disk(tmp, hf)
        assert any('legacy cache' in str(x.message) for x in w)
        assert torch.equal(ds.X, legacy)
        _write_cache(tmp, _cache(2 * N_ML - 1))
        try:
            _from_disk(tmp, hf)
            raise AssertionError('wrong-length cache did not raise')
        except ValueError as e:
            assert 'expected 24' in str(e) and '12' in str(e), e


def test_eval_window_only_false_and_no_column_hf():
    with tempfile.TemporaryDirectory() as d:
        tmp = _world(Path(d))
        hf = _hf(tmp)
        full = _cache(2 * N_ML)
        _write_cache(tmp, full)
        ds = _from_disk(tmp, hf, eval_window_only=False)
        assert len(ds) == 2 * N_ML and torch.equal(ds.X, full)
        assert torch.isnan(ds.Y[torch.from_numpy(~_keep())]).all()
        assert not ds.annotated_mask[torch.from_numpy(~_keep())].any()
        # dataset without the column: every row, X untouched, same labels as the HF table
        hf_nocol = hf.remove_columns('eval_window')
        ds0 = _from_disk(tmp, hf_nocol)
        assert len(ds0) == 2 * N_ML and torch.equal(ds0.X, full)
        assert torch.equal(torch.isnan(ds0.Y), torch.isnan(ds.Y))


def test_pov_window_then_nan_to_zero():
    with tempfile.TemporaryDirectory() as d:
        tmp = _world(Path(d))
        hf = _hf(tmp)
        blue, yellow = _cache(2 * N_ML), _cache(2 * N_ML) + 100
        blue[2] = float('nan')      # obsA frame 2, inside the window
        blue[8] = float('nan')      # obsA frame 8, outside
        _write_cache(tmp, blue, 'pov', 'blue')
        _write_cache(tmp, yellow, 'pov', 'yellow')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ds = _from_disk(tmp, hf, frame_type='pov')
        msgs = [str(x.message) for x in w if 'NaN rows in blue' in str(x.message)]
        assert msgs and msgs[0].split(' NaN')[0].endswith('1/12'), msgs
        assert ds.X.shape == (2 * WIN, 2 * D) and not torch.isnan(ds.X).any()
        assert (ds.X[2, :D] == 0).all()
        k = torch.from_numpy(_keep())
        assert torch.equal(ds.X[:, D:], yellow[k])


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    if len(sys.argv) > 1:
        tests = [t for t in tests if any(a in t.__name__ for a in sys.argv[1:])]
    failed = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f'  PASS  {t.__name__}  ({time.time() - t0:.1f}s)')
        except Exception:
            failed += 1
            print(f'  FAIL  {t.__name__}')
            traceback.print_exc()
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
