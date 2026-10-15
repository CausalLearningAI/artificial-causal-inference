"""
CelebA DINES benchmark, step 2 (CPU job array): run DINES (delta = 0, 0.02, 0.05; linear forward phase, cross-fitted
ridge/logistic terminal phase) and this repo's NES (src/eci/nes.py, default settings) on one block of trials.

Trial matrix = codes[index] (float64), with codes = $BEEGFS/codes_f32.npy (written by prep.py; prep_checks.json
verifies that this equals the saved experiment CSVs). The task reads only the rows its trials need, in one sorted
read from beegfs (not NFS); the trial list (.npz) is copied to the job's local disk.

  --design original|M|S|H|I --task i   task i of the design's trial list (seeds x N x tau), chunked by N so a task
                                       stays well under an hour: CHUNK[N] trials per task. --count prints the number
                                       of tasks. Original trials from $BEEGFS/original_trials.npz (= Tommaso's saved
                                       simulation CSVs), new designs from $BEEGFS/designs.npz (gen_designs.py).

DINES compute settings (fixed before any metric was seen, for run time): 2-fold cross-fitting (as the theory's toy,
Appendix B.1) and at most 1024 conditioning sets in the terminal phase (terminal_complete = False beyond).

Writes one small JSON per trial: results/eci/dines_celeba/trials/<design>/<tau>_<N>_<seed>.json with every method's
selection (atoms as 1-indexed strings, the naming of the saved runs), DINES's forward set, terminal statuses and
timings. Existing files are skipped (--overwrite to redo).
"""
import argparse
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from eci.dines import dines  # noqa: E402
from eci.nes import neural_effect_search  # noqa: E402

BEEGFS = Path('/mnt/beegfs/locatgrp/shared/rcadei/dines_celeba')
OUT = Path('results/eci/dines_celeba')
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SIZES = [30, 50, 100, 250, 500, 1000, 2500, 5000]
NEW_SIZES = [250, 1000, 2500, 5000]
DELTAS = [0.0, 0.02, 0.05]
N_FOLDS, MAX_SUBSETS = 2, 1024
CHUNK = {30: 9, 50: 9, 100: 9, 250: 9, 500: 9, 1000: 3, 2500: 1, 5000: 1}


def lab(j):
    return str(int(j) + 1)


def run_dines(Z, T, delta, seed):
    t0 = time.time()
    try:
        r = dines(Z, T, alpha=0.05, delta=delta, seed=seed, n_folds=N_FOLDS, max_subsets=MAX_SUBSETS)
    except Exception as err:  # recorded, never silently dropped: the summary counts errors
        return {'error': repr(err), 'elapsed_s': time.time() - t0, 'selected': []}
    return {'selected': [lab(j) for j in r['selected']], 'kept': [lab(j) for j in r['kept']],
            'undetermined': [lab(j) for j in r['undetermined']], 'set_aside': [lab(j) for j in r['set_aside']],
            'forward_set': [lab(j) for j in r['forward_set']], 'terminal_complete': r['terminal_complete'],
            'forward_capped': r['forward_capped'], 'm': r['m'],
            'forward': [{'atom': lab(f['atom']), 'p': f['p'], 'rho': f['rho'], 'added': f['added'],
                         'set_aside': f['set_aside'], 'n_significant': f['n_significant']} for f in r['forward']],
            'terminal': [{'atom': lab(a['atom']), 'status': a['status'], 'max_p': a['max_p'],
                          'sepset': None if a['sepset'] is None else [lab(s) for s in a['sepset']],
                          'p_empty': a['p_empty'], 'rho_empty': a['rho_empty'], 'p_rest': a['p_rest'],
                          'rho_rest': a['rho_rest'], 'n_sets': a['n_sets_tested'], 'min_ess': a['min_ess'],
                          'fail_skipped': [[lab(s) for s in A] for A in a['fail_skipped']]} for a in r['terminal']],
            'n_sets': len(r['sets']), 'n_sets_skipped': int(sum(s['skipped'] for s in r['sets'])),
            'elapsed_s': r['elapsed_s'], 'elapsed_forward_s': r['elapsed_forward_s'],
            'elapsed_terminal_s': r['elapsed_terminal_s']}


def run_nes(Z, T):
    t0 = time.time()
    try:
        r = neural_effect_search(Z, T, alpha=0.05)
    except Exception as err:
        return {'error': repr(err), 'elapsed_s': time.time() - t0, 'selected': []}
    return {'selected': [lab(j) for j in r['selected']], 'stopped': r.get('stopped'), 'n_tested': r['n_tested'],
            'elapsed_s': time.time() - t0}


def task_list(design):
    """[(N, [(tau, seed), ...]), ...]: one entry per task."""
    sizes, seeds = (SIZES, range(420, 430)) if design == 'original' else (NEW_SIZES, range(420, 430))
    tasks = []
    for N in sizes:
        trials = [(tau, seed) for seed in seeds for tau in TAUS]
        tasks += [(N, trials[i:i + CHUNK[N]]) for i in range(0, len(trials), CHUNK[N])]
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', required=True)
    ap.add_argument('--task', type=int, default=None)
    ap.add_argument('--count', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    tasks = task_list(args.design)
    if args.count:
        print(len(tasks))
        return
    t_job = time.time()
    N, block = tasks[args.task]
    src, prefix = (BEEGFS / 'original_trials.npz', '') if args.design == 'original' else (BEEGFS / 'designs.npz', f'{args.design}_')
    out_dir = OUT / 'trials' / args.design
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(t, sd) for t, sd in block if args.overwrite or not (out_dir / f'{t}_{N}_{sd}.json').exists()]
    print(f'design {args.design} task {args.task} N {N}: {len(todo)} of {len(block)} trials to run', flush=True)
    if not todo:
        return
    local = Path(os.environ.get('LOCAL_DIR', '/tmp/dines_run'))
    local.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shutil.copy(src, local / src.name)
    trials = np.load(local / src.name)
    arrs = {(t, sd): trials[f'{prefix}{t}_{N}_{sd}'] for t, sd in todo}
    rows = np.unique(np.concatenate([a[:, 0] for a in arrs.values()]))
    block_codes = np.asarray(np.load(BEEGFS / 'codes_f32.npy', mmap_mode='r')[rows])  # one sorted read
    meta = json.loads((OUT / 'designs_meta.json').read_text())[args.design] if prefix else None
    print(f'staged {len(rows)} code rows in {time.time() - t0:.0f}s', flush=True)
    for (tau, seed), arr in arrs.items():
        t1 = time.time()
        idx, T = np.searchsorted(rows, arr[:, 0]), arr[:, 1].astype(np.int64)
        Z = block_codes[idx].astype(np.float64)
        rec = {'design': args.design, 'tau': tau, 'N': N, 'seed': seed, 'n_treated': int(T.sum()),
               'host': socket.gethostname(), 'dines_version': 'v2 (forward set-aside, reported = kept + undetermined)',
               'n_folds': N_FOLDS, 'max_subsets': MAX_SUBSETS, 'methods': {}}
        if meta:
            rec['roles'] = meta['roles']
        for d in DELTAS:
            rec['methods'][f'DINES_d{d}'] = run_dines(Z, T, d, seed)
        rec['methods']['NES_repo'] = run_nes(Z, T)
        rec['elapsed_trial_s'] = time.time() - t1
        (out_dir / f'{tau}_{N}_{seed}.json').write_text(json.dumps(rec))
        summary = '  '.join(f"{k}: {v['selected']} ({v['elapsed_s']:.1f}s{', ERR' if 'error' in v else ''})"
                            for k, v in rec['methods'].items())
        print(f'tau {tau} N {N} seed {seed} [{rec["elapsed_trial_s"]:.0f}s]  {summary}', flush=True)
    print(f'block done in {time.time() - t_job:.0f}s', flush=True)


if __name__ == '__main__':
    main()
