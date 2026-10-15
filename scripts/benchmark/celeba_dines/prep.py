"""
CelebA DINES benchmark, step 0 (CPU job): consolidate the inputs and check them.

Source (READ-ONLY, Tommaso Mencattini's NES project): /nfs/scistore19/locatgrp/tmencatt/projects/neuralscience
  data/manifests/test_flwrlabs/celeba/labels.parquet                        40 binary attributes, 19,962 test images
  data/manifests/test_flwrlabs/celeba/SAE/TOPK/MEAN/SIGLIP/somixevi/codes   SAE codes (19,962 x 9,216), .npy shards
  results/experiments-semi-synthetic/simulation_M3_ATE:{tau}_size:{N}_seed:{s}.csv   (index, T, W, Y, Z) per trial
  results/experiments-semi-synthetic/experiment_M3_...csv                   the same trial with N_1..N_9216 (large)

Writes
  $BEEGFS/codes_f32.npy            all codes, one float32 matrix (row = manifest row), ~0.7 GB
  $BEEGFS/labels.npz               attributes (int8, 19,962 x 40), names, celeb_id
  $BEEGFS/original_trials.npz      per original trial: index, T, W, Y, Z (from the small simulation CSVs)
  results/eci/dines_celeba/best_f1_atoms.csv   best-F1 atom (firing Z > 0 vs label) per attribute, on all test codes
  results/eci/dines_celeba/prep_checks.json    the experiment-CSV vs codes[main_index] check on a sample of trials

Why codes[main_index] instead of the 720 large experiment CSVs (~30 GB of text): the trial matrix is by construction
the code row of each sampled image (WithFeaturesRCT.to_csv_df writes features of sample['index']); reading the codes
once avoids streaming 30 GB from NFS. The check below verifies that equality on a sample of trials of every size.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path('/nfs/scistore19/locatgrp/tmencatt/projects/neuralscience')
MANIFEST = SRC / 'data/manifests/test_flwrlabs/celeba'
CODES = MANIFEST / 'SAE/TOPK/MEAN/SIGLIP/somixevi/codes'
TRIALS = SRC / 'results/experiments-semi-synthetic'
BEEGFS = Path('/mnt/beegfs/locatgrp/shared/rcadei/dines_celeba')
OUT = Path('results/eci/dines_celeba')
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SIZES = [30, 50, 100, 250, 500, 1000, 2500, 5000]
SEEDS = list(range(420, 430))
CHECK = [(0.7, 30, 420), (0.0, 100, 421), (0.4, 250, 420), (0.8, 500, 423), (0.7, 1000, 420), (0.1, 2500, 425),
         (0.7, 5000, 420), (0.3, 5000, 429)]


def trial_name(kind, tau, N, seed):
    return f'{kind}_M3_ATE:{tau}_size:{N}_seed:{seed}.csv'


def load_codes(local):
    meta = json.loads((CODES / 'metadata.json').read_text())
    index = pd.read_parquet(CODES / 'index.parquet').sort_values('row_index').reset_index(drop=True)
    assert (index['row_index'].values == np.arange(len(index))).all()
    shards = local / 'shards'
    shutil.copytree(CODES / 'shards', shards)
    out = np.empty((len(index), meta['feature_dim']), dtype=np.float32)
    for sid, grp in index.groupby('shard_id'):
        arr = np.load(shards / f'shard_{int(sid):05d}.npy')
        arr = arr.reshape(arr.shape[0], -1)
        out[grp['row_index'].values] = arr[grp['offset'].values]
    return out, meta


def best_f1_table(codes, labels, names):
    """F1 of the firing indicator (Z > 0) of every atom against every attribute; best and runner-up atom."""
    fire = (codes > 0).astype(np.float32)
    n_fire = fire.sum(0)
    rows = []
    for a, name in enumerate(names):
        y = labels[:, a].astype(np.float32)
        tp = fire.T @ y
        f1 = 2 * tp / (n_fire + y.sum())
        order = np.argsort(-f1)
        j, j2 = order[0], order[1]
        rows.append({'attribute': name, 'prevalence': float(y.mean()), 'best_atom': int(j + 1), 'best_f1': float(f1[j]),
                     'precision': float(tp[j] / max(n_fire[j], 1)), 'recall': float(tp[j] / y.sum()),
                     'fire_rate': float(n_fire[j] / len(y)), 'second_atom': int(j2 + 1), 'second_f1': float(f1[j2])})
    return pd.DataFrame(rows).sort_values('best_f1', ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-check', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    local = Path(os.environ.get('LOCAL_DIR', '/tmp/dines_prep'))
    local.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    BEEGFS.mkdir(parents=True, exist_ok=True)

    codes, meta = load_codes(local)
    print(f'codes {codes.shape} {codes.dtype}, min {codes.min():.3g}, mean active atoms/image '
          f'{(codes > 0).sum(1).mean():.1f} ({time.time() - t0:.0f}s)', flush=True)
    np.save(local / 'codes_f32.npy', codes)
    shutil.copy(local / 'codes_f32.npy', BEEGFS / 'codes_f32.npy')

    lab = pd.read_parquet(MANIFEST / 'labels.parquet')
    names = [c for c in lab.columns if c not in ('path', 'celeb_id')]
    labels = lab[names].astype(int).clip(0, 1).values.astype(np.int8)
    np.savez(BEEGFS / 'labels.npz', labels=labels, names=np.array(names), celeb_id=lab['celeb_id'].values)
    tab = best_f1_table(codes, labels, names)
    tab.to_csv(OUT / 'best_f1_atoms.csv', index=False)
    print(tab.to_string(index=False, float_format=lambda v: f'{v:.3f}'), flush=True)
    for nm, atom in (('Wearing_Hat', 38), ('Eyeglasses', 6051), ('Smiling', 38), ('Wearing_Hat', 408), ('Eyeglasses', 408)):
        a = names.index(nm)
        y, f = labels[:, a].astype(bool), codes[:, atom - 1] > 0
        tp = (y & f).sum()
        print(f'F1(atom {atom} vs {nm}) = {2 * tp / (y.sum() + f.sum()):.3f}')

    # the 720 original trials: (index, T, W, Y, Z) from the small simulation CSVs
    sim_dir = local / 'sim'
    sim_dir.mkdir()
    trials = {}
    for seed in SEEDS:
        for N in SIZES:
            for tau in TAUS:
                name = trial_name('simulation', tau, N, seed)
                shutil.copy(TRIALS / name, sim_dir / name)
                d = pd.read_csv(sim_dir / name)
                key = f'{tau}_{N}_{seed}'
                trials[key] = d[['index', 'T', 'W', 'Y', 'Z']].values.astype(np.int32)
                # the trial's labels must be the sampled images' attributes (bucket sampling, no fallback used)
                for col, att in (('W', 'Smiling'), ('Y', 'Eyeglasses'), ('Z', 'Wearing_Hat')):
                    lab_col = labels[d['index'].values, names.index(att)]
                    assert (lab_col == d[col].values).all(), (key, col)
    np.savez(BEEGFS / 'original_trials.npz', **trials)
    print(f'{len(trials)} original trials saved; labels match the sampled images in all of them '
          f'({time.time() - t0:.0f}s)', flush=True)

    checks = []
    if not args.skip_check:
        for tau, N, seed in CHECK:
            t1 = time.time()
            name = trial_name('experiment', tau, N, seed)
            shutil.copy(TRIALS / name, local / name)
            df = pd.read_csv(local / name)
            sim = trials[f'{tau}_{N}_{seed}']
            ncols = [f'N_{i + 1}' for i in range(codes.shape[1])]
            X = df[ncols].values
            ref = codes[df['main_index'].values]
            rec = {'trial': name, 'rows': len(df), 'main_index_eq_sim_index': bool((df['main_index'].values == sim[:, 0]).all()),
                   'T_eq': bool((df['T'].values == sim[:, 1]).all()),
                   'WYZ_eq': bool((df[['W', 'Y', 'Z']].values == sim[:, 2:]).all()),
                   'max_abs_diff_codes': float(np.abs(X - ref).max()),
                   'max_rel_diff_codes': float((np.abs(X - ref) / np.maximum(np.abs(ref), 1e-6)).max()),
                   'seconds': time.time() - t1}
            checks.append(rec)
            print(rec, flush=True)
            os.remove(local / name)
        (OUT / 'prep_checks.json').write_text(json.dumps(checks, indent=1))
    print(f'done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
