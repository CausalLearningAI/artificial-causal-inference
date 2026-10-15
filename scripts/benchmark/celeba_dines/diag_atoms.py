"""
CelebA DINES benchmark, diagnostic (CPU job): what do the atoms that DINES / NES select wrongly encode?

For each atom (1-indexed labels): its three best attributes by F1 (firing Z > 0 vs label, all 19,962 test images),
and its firing rate within the cells of (attribute = 0/1) x (that attribute's principal atom fires / silent) for
Eyeglasses (principal 6051) and Wearing_Hat (principal 38). An atom that fires mostly on, e.g., eyeglasses images that
6051 misses carries eyeglasses information that 6051 does not (feature splitting, theory O6): its conditional effect
given 6051 is then real, not noise.

  python diag_atoms.py 408 8747 5398     -> results/eci/dines_celeba/diag_atoms.json and stdout
"""
import json
import sys
from pathlib import Path

import numpy as np

BEEGFS = Path('/mnt/beegfs/locatgrp/shared/rcadei/dines_celeba')
OUT = Path('results/eci/dines_celeba')


def main():
    atoms = [int(a) for a in sys.argv[1:]]
    codes = np.load(BEEGFS / 'codes_f32.npy', mmap_mode='r')
    lab = np.load(BEEGFS / 'labels.npz')
    labels, names = lab['labels'].astype(bool), list(lab['names'])
    cols = sorted(set(atoms) | {38, 6051})
    fire = {a: np.asarray(codes[:, a - 1]) > 0 for a in cols}
    out = {}
    for a in atoms:
        f = fire[a]
        f1 = 2 * (labels & f[:, None]).sum(0) / (labels.sum(0) + f.sum())
        top = np.argsort(-f1)[:3]
        rec = {'fire_rate': float(f.mean()), 'top_attributes': [(names[i], round(float(f1[i]), 3)) for i in top], 'cells': {}}
        for att, prin in (('Eyeglasses', 6051), ('Wearing_Hat', 38)):
            y = labels[:, names.index(att)]
            for yv in (0, 1):
                for pv in (0, 1):
                    m = (y == yv) & (fire[prin] == pv)
                    rec['cells'][f'{att}={yv},{prin} fires={pv}'] = (int(m.sum()), round(float(f[m].mean()), 3) if m.any() else None)
        out[str(a)] = rec
        print(a, json.dumps(rec))
    (OUT / 'diag_atoms.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
