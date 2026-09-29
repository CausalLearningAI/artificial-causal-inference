"""
Evaluation-only encode for the foreground SAEs: every frame (5 fps) of the ANNOTATED videos of
the held-out validation pools -> DINOv2 (448, no crop) + foreground mask + each SAE -> per-frame
codes_max / codes_mean over foreground patches. Behaviour labels are not read here; they are
only joined in scripts/eci/eval_sae_fg.py.

Output: dataset/mice/v1/eci/fg448/eval_codes/task_XX.npz  (rows, n_fg, codes_max_<sae>, codes_mean_<sae>)

Usage:
    python scripts/eci/fg_eval_encode.py --task 0 --n-tasks 6 --saes matryoshka_btk_1024_k16_fg448_s0,matryoshka_btk_1024_k16_fg448_s1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.fg_encode import encode_rows  # noqa: E402
from src.eci.foreground import obs_rows  # noqa: E402


def eval_videos(data_dir, val_pools):
    exp = pd.read_csv(Path(data_dir) / 'mice/v1/experiment.csv')
    sel = exp.pool.isin(val_pools) & exp.annotation_file.notna()
    return sorted(exp[sel].observation_id)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--data-dir', default=str(REPO / 'data'))
    p.add_argument('--bg-dir', default=str(REPO / 'dataset/mice/v1/eci/fg448/background'))
    p.add_argument('--val-from', default=str(REPO / 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_ep20_s0/metrics.json'))
    p.add_argument('--saes', default='matryoshka_btk_1024_k16_fg448_s0,matryoshka_btk_1024_k16_fg448_s1')
    p.add_argument('--out-dir', default=str(REPO / 'dataset/mice/v1/eci/fg448/eval_codes'))
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--n-tasks', type=int, default=6)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f'task_{args.task:02d}.npz'
    if f.exists():
        print(f'[SKIP] {f}')
        return
    ds = Path(args.dataset_dir)
    ann = ds / 'mice/v1/annotations.csv'
    vids = eval_videos(args.data_dir, json.loads(Path(args.val_from).read_text())['val_pools'])[args.task::args.n_tasks]
    ranges = obs_rows(ann)
    rows = np.concatenate([np.arange(*ranges[o]) for o in vids])
    print(f'{len(vids)} videos {vids}, {len(rows)} frames', flush=True)
    saes = args.saes.split(',')
    frame_paths = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    res = encode_rows(rows, frame_paths, [ds / 'mice/v1/eci/sae' / s / 'sae.pt' for s in saes], args.bg_dir, ann,
                      ds, batch_size=128, num_workers=22)
    arrs = {'rows': rows, 'n_fg': res[0]['n_fg']}
    for s, r in zip(saes, res):
        arrs[f'codes_max_{s}'] = r['codes_max']
        arrs[f'codes_mean_{s}'] = r['codes_mean']
    np.savez(str(f).replace('.npz', '.tmp.npz'), **arrs)
    Path(str(f).replace('.npz', '.tmp.npz')).rename(f)
    print(f'Done -> {f}')


if __name__ == '__main__':
    main()
