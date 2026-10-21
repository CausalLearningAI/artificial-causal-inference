"""
Left/right convention of the two SLEAP prediction sets of the 35 juvenile hour-1 videos (and hour 2 when present):
ORIGINAL (MNV1_revision_dataset/Juv/<group>/*.predictions.analysis.h5; what the lab's kinematic tables in
MNV1_results/2026-03-30_finalandupdated were computed from, used_settings.yaml ROOT_DIR) vs REPRED (what ECI uses,
dataset/frogs/v1/sleap). For every 12th frame (5 fps): share of frames where Right_Hip / Right_Shoulder lie on the
frog's right of the body axis (Tail_Stem -> eyes midpoint; camera above the dish, image y down: cross > 0), and the mean
unsigned knee angle (Hip-Knee-Ankle, 0 = straight) per label, plus the SLEAP model (provenance).
Needs h5py + numpy (/usr/bin/python3 on the login node; the .sh installs h5py into the job dir otherwise). Usage: sbatch scripts/eci/v2/frogs_paper/sleap_handedness.sh
Output: results/vision/eci_v2/frogs_paper/sleap_handedness.csv
"""
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[4]
RAW = Path('/nfs/scistore23/sweengrp/ftoma/Behavior_MNV1/MNV1_revision_dataset/Juv')
REPRED = Path('/nfs/scistore23/sweengrp/ftoma/Behavior_MNV1/MNV1_revision_dataset_REPRED/Juv')
FOLD = {'WT': ('WT', 'WT REPRED'), 'FoxP1': ('FoxP1 gRNA1 half', 'FoxP1 gRNA1 half REPRED'),
        'En1': ('En1 gRNA4 half Genotyped', 'En1 gRNA4 half Gen REPRED')}


def stats_of(h5):
    with h5py.File(h5, 'r') as h:
        names = [x.decode() for x in h['node_names'][:]]
        tr = h['tracks'][0, :, :, ::12]  # (2, K, n)
        prov = h['provenance'][()]
    prov = prov.decode() if isinstance(prov, bytes) else str(prov)
    try:
        model = json.loads(prov).get('model_paths', prov)
    except Exception:
        model = prov[:200]
    nd = np.transpose(tr, (2, 1, 0)).astype(np.float64)  # (n, K, 2)
    head = nd[:, [names.index('Right_Eye'), names.index('Left_Eye')]].mean(1)
    ts = nd[:, names.index('Tail_Stem')]
    ax = head - ts
    out = {'model': str(model)[-160:]}
    for node in ('Right_Hip', 'Left_Hip', 'Right_Shoulder', 'Left_Shoulder'):
        rel = nd[:, names.index(node)] - ts
        cr = ax[:, 0] * rel[:, 1] - ax[:, 1] * rel[:, 0]
        ok = np.isfinite(cr)
        out[f'rightshare_{node}'] = round(float((cr[ok] > 0).mean()), 4) if ok.any() else ''
    for s in ('Right', 'Left'):
        i, j, k = (names.index(f'{s}_{q}') for q in ('Hip', 'Knee', 'Ankle'))
        u, v = nd[:, j] - nd[:, i], nd[:, k] - nd[:, j]
        c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        out[f'knee_{s}'] = round(float(np.nanmean(np.degrees(np.arccos(np.clip(c, -1, 1))))), 2)
    return out


def main():
    rows = []
    with open(REPO / 'data/frogs/v1/experiment.csv') as f:
        exp = list(csv.DictReader(f))
    for r in exp:
        g = r['group']
        src = Path(r['source_file'])
        files = [src] + sorted(p for p in (RAW / FOLD[g][0]).glob(f"{r['session']}_*stJuv*_2_{r['slot']}.mp4"))
        for mp4 in files:
            hour = 1 if mp4 == src else 2
            for which, h5 in (('original', mp4.parent / f'{mp4.name}.predictions.analysis.h5'),
                              ('repred', REPRED / FOLD[g][1] / f'{mp4.name}.predictions.analysis.h5')):
                if not h5.exists():
                    rows.append(dict(obs=r['observation_id'], group=g, hour=hour, mutant_side=r['mutant_side'],
                                     set=which, movie=mp4.stem, missing=1))
                    continue
                rows.append(dict(obs=r['observation_id'], group=g, hour=hour, mutant_side=r['mutant_side'], set=which,
                                 movie=mp4.stem, missing=0, **stats_of(h5)))
                print(rows[-1], flush=True)
    keys = sorted({k for x in rows for k in x}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    out = REPO / 'results/vision/eci_v2/frogs_paper/sleap_handedness.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print('wrote', out, len(rows), file=sys.stderr)


if __name__ == '__main__':
    main()
