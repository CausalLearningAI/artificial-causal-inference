"""
CelebA DINES benchmark, step 1 (CPU job): generate the NEW semi-synthetic designs with the NES sampling machinery
(Tommaso's src/neuralscience/runnable/s6-experiment.py:99-138 and rct.py Model4RCT / RCTSampler): sample binary
attribute labels from a structural model with T ~ Bern(1/2), then draw one CelebA test image per unit uniformly WITH
replacement from the bucket of images whose design attributes equal the sampled labels exactly. Every other
attribute of the image follows its CelebA distribution given the design attributes (so it is a descendant of them).

Effect sizes follow the original design: tau in {0, 0.1, ..., 0.8}, p(child = 1 | T = t) = 0.5 + (t - 1/2) tau, and
for a child with a co-parent W the Model4 table p_11 = 0.5 + tau/2, p_01 = 0.5 - tau/2, p_10 = 0.2 + tau, p_00 = 0.2
(p_tw; ATE = tau). N in {250, 1000, 2500, 5000}, seeds 420-429.

Designs (attributes chosen from best_f1_atoms.csv: only Wearing_Hat has best F1 >= 0.8 on the test codes; Eyeglasses
0.748 and Blond_Hair 0.744 are next; Smiling 0.492 has no well-aligned atom):
  M  mediator             T -> A -> B,  p(B = 1 | A) = 0.2 + 0.6 A                  A = Wearing_Hat, B = Eyeglasses
  S  separator spouse     T -> A <- W, W -> B, A -> B; A ~ Model4(T, W);             A = Wearing_Hat, W = Eyeglasses,
                          p(B = 1 | A, W) = (1 - A)(0.2 + 0.4 W)  (a hat hides the hair) B = Blond_Hair
  H  as S, W unaligned    W = Smiling (expected failure: Tier 2 needs an aligned spouse)
  Why B = Blond_Hair and the (1 - A) factor: CelebA's test split has 0 images that are blond AND wear a hat AND
  eyeglasses and only 13 blond images with a hat, so with three free binary labels a bucket is empty. Making the hat
  hide the blond hair gives those cells probability 0; every bucket that can be drawn has >= 62 images. B still needs
  {A, W} as separator (given A = 0, B depends on W, and W depends on T given A: the collider T -> A <- W).
  Buckets that can be drawn must be non-empty (checked per trial).
  I  original design      T -> Hat, T -> Eyeglasses <- Smiling (Model4, s6-experiment.py), regenerated with this code
Truth (direct) = the best-F1 atom of each direct attribute; B's atom = 'indirect', W's atom = 'spouse'.

Writes $BEEGFS/designs.npz (key '<design>_<tau>_<N>_<seed>' -> int32 (N, 2 + k): image index, T, design labels) and
results/eci/dines_celeba/designs_meta.json (attributes, roles as 1-indexed atom labels, bucket sizes, label checks).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BEEGFS = Path('/mnt/beegfs/locatgrp/shared/rcadei/dines_celeba')
OUT = Path('results/eci/dines_celeba')
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SIZES = [250, 1000, 2500, 5000]
SEEDS = list(range(420, 430))

DESIGNS = {
    'M': {'attrs': {'A': 'Wearing_Hat', 'B': 'Eyeglasses'}, 'direct': ['A'], 'indirect': ['B'], 'spouse': []},
    'S': {'attrs': {'W': 'Eyeglasses', 'A': 'Wearing_Hat', 'B': 'Blond_Hair'}, 'direct': ['A'], 'indirect': ['B'],
          'spouse': ['W']},
    'H': {'attrs': {'W': 'Smiling', 'A': 'Wearing_Hat', 'B': 'Blond_Hair'}, 'direct': ['A'], 'indirect': ['B'],
          'spouse': ['W']},
    'I': {'attrs': {'W': 'Smiling', 'Y': 'Eyeglasses', 'Zh': 'Wearing_Hat'}, 'direct': ['Y', 'Zh'], 'indirect': [],
          'spouse': ['W']},
}


def model4(rng, T, W, tau):
    """Child with co-parent W: p_11 = .5 + tau/2, p_01 = .5 - tau/2, p_10 = .2 + tau, p_00 = .2 (rct.py Model4RCT)."""
    p = np.where(W == 1, np.where(T == 1, 0.5 + tau / 2, 0.5 - tau / 2), np.where(T == 1, 0.2 + tau, 0.2))
    return (rng.random(len(T)) < p).astype(np.int32)


def sample_labels(design, rng, N, tau):
    T = (rng.random(N) < 0.5).astype(np.int32)
    if design == 'M':
        A = (rng.random(N) < 0.5 + (T - 0.5) * tau).astype(np.int32)
        B = (rng.random(N) < 0.2 + 0.6 * A).astype(np.int32)
        return T, {'A': A, 'B': B}
    if design in ('S', 'H'):
        W = (rng.random(N) < 0.5).astype(np.int32)
        A = model4(rng, T, W, tau)
        B = (rng.random(N) < (1 - A) * (0.2 + 0.4 * W)).astype(np.int32)
        return T, {'W': W, 'A': A, 'B': B}
    if design == 'I':
        W = (rng.random(N) < 0.5).astype(np.int32)
        Y = model4(rng, T, W, tau)
        Zh = (rng.random(N) < 0.5 + (T - 0.5) * tau).astype(np.int32)
        return T, {'W': W, 'Y': Y, 'Zh': Zh}
    raise ValueError(design)


def main():
    lab = np.load(BEEGFS / 'labels.npz', allow_pickle=False)
    labels, names = lab['labels'], list(lab['names'])
    best = pd.read_csv(OUT / 'best_f1_atoms.csv').set_index('attribute')
    trials, meta = {}, {}
    for d_id, (design, spec) in enumerate(DESIGNS.items()):
        roles_keys = list(spec['attrs'])
        cols = [names.index(spec['attrs'][k]) for k in roles_keys]
        key_img = labels[:, cols].astype(np.int64) @ (2 ** np.arange(len(cols)))
        buckets = {int(b): np.flatnonzero(key_img == b) for b in range(2 ** len(cols))}
        sizes = {format(b, f'0{len(cols)}b')[::-1]: len(v) for b, v in buckets.items()}
        print(design, spec['attrs'], 'bucket sizes (bits in order', roles_keys, '):', sizes, flush=True)
        atom = {k: str(int(best.loc[spec['attrs'][k], 'best_atom'])) for k in roles_keys}
        meta[design] = {'attrs': spec['attrs'], 'atoms': atom, 'bucket_sizes': sizes,
                        'best_f1': {k: float(best.loc[spec['attrs'][k], 'best_f1']) for k in roles_keys},
                        'roles': {r: [atom[k] for k in spec[r]] for r in ('direct', 'indirect', 'spouse')},
                        'label_checks': {}}
        for t_id, tau in enumerate(TAUS):
            for N in SIZES:
                for seed in SEEDS:
                    rng = np.random.default_rng([seed, d_id, t_id, N])
                    T, L = sample_labels(design, rng, N, tau)
                    key = np.stack([L[k] for k in roles_keys], 1).astype(np.int64) @ (2 ** np.arange(len(cols)))
                    idx = np.empty(N, dtype=np.int64)
                    for b in np.unique(key):
                        rows = np.flatnonzero(key == b)
                        if len(buckets[int(b)]) == 0:
                            raise RuntimeError(f'design {design}: drew labels {b} whose image bucket is empty')
                        idx[rows] = buckets[int(b)][rng.integers(0, len(buckets[int(b)]), len(rows))]
                    assert (labels[idx][:, cols] == np.stack([L[k] for k in roles_keys], 1)).all()
                    trials[f'{design}_{tau}_{N}_{seed}'] = np.column_stack([idx, T] + [L[k] for k in roles_keys]).astype(np.int32)
            # empirical label effects at N = 5000, pooled over seeds (a check of the generator)
            big = [trials[f'{design}_{tau}_5000_{s}'] for s in SEEDS]
            arr = np.concatenate(big)
            eff = {k: float(arr[arr[:, 1] == 1, 2 + i].mean() - arr[arr[:, 1] == 0, 2 + i].mean())
                   for i, k in enumerate(roles_keys)}
            meta[design]['label_checks'][str(tau)] = eff
        print(design, 'label mean differences (T=1 minus T=0) at N=5000:', meta[design]['label_checks'], flush=True)
    np.savez(BEEGFS / 'designs.npz', **trials)
    (OUT / 'designs_meta.json').write_text(json.dumps(meta, indent=1))
    print(f'{len(trials)} trials written')


if __name__ == '__main__':
    main()
