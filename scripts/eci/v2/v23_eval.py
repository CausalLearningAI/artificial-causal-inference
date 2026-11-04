"""
ECI v2.3 evaluation helpers (pre-registration scripts/eci/v2/v23_prereg.md). Run with V22_FRAMES=ext: the v2.2 full
pipeline (v22_full.py codes2 / cache, v22_eval.py tag / report) on the 'ext' frame set = every labelled mice frame =
the 15-min atlas window (648,000 frames, = v2.2 FULL WINDOW) + the first 15 min of the 48 habituation (_H) videos
(frames 0-4499, 216,000 frames) = 864,000 frames at 5 fps.
Steps
  frames      (CPU) -> BEE23/ext/frames.parquet (v22_full frames.parquet columns + full_row, part), OUT/ext/frames.{json,md}
  idcodes     (CPU) ext codes on the 648,000 atlas frames vs the v2.2 FULL WINDOW codes of the same SAEs (V s1001-1006)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
assert os.environ.get('V22_FRAMES') == 'ext', 'run with V22_FRAMES=ext'
import v22_core as C  # noqa: E402
import v22_full as VF  # noqa: E402

REPO, log = C.REPO, C.log
FULL = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2/full')


def cmd_frames(a):
    t0 = time.time()
    ann = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv', usecols=['observation_id', 'frame_idx', 'Y_nn', 'Y_np', 'Y_nt'])
    ann['row'] = np.arange(len(ann))
    A = ann[ann.Y_nn.notna().values].rename(columns={'observation_id': 'obs', 'frame_idx': 'k'})
    Ff = pd.read_parquet(FULL / 'frames.parquet')  # v2.2 full window (648,000), its video order
    order = pd.unique(Ff.obs.values)
    assert set(order) == set(A.obs.unique()) and len(order) == 144, 'labelled videos != the 144 atlas videos'
    A['vo'] = A.obs.map({o: i for i, o in enumerate(order)})
    A = A.sort_values(['vo', 'k']).reset_index(drop=True).drop(columns='vo')
    g = A.groupby('obs', sort=False).k.agg(['min', 'max', 'size'])
    assert (g['max'] - g['min'] + 1 == g['size']).all(), 'labelled frames not contiguous per video'
    isH = A.obs.str.endswith('_H').values
    A['part'] = np.where(isH & (A.k.values < 4500), 'first15_H', 'atlas')
    key = pd.MultiIndex.from_arrays([A.obs.values, A.k.values])
    fr = pd.Series(np.arange(len(Ff)), index=pd.MultiIndex.from_arrays([Ff.obs.values, Ff.k.values])).reindex(key)
    A['full_row'] = fr.fillna(-1).astype(np.int64).values
    sub = A[A.full_row >= 0]
    assert len(sub) == len(Ff) == 648000 and (sub.full_row.values == np.arange(len(Ff))).all(), 'atlas frames not an ordered subset'
    assert (sub.part == 'atlas').all() and (A.part[A.full_row < 0] == 'first15_H').all()
    for c in ('row', 'Y_nn', 'Y_np', 'Y_nt', 'code_row'):
        if c == 'code_row':
            continue
        assert np.array_equal(sub[c].fillna(-9).values, Ff[c].fillna(-9).values), c
    cr = np.full(len(A), -1, np.int64)
    cr[A.full_row.values >= 0] = Ff.code_row.values
    A['code_row'] = cr
    e = A[A.part == 'first15_H']
    gh = e.groupby('obs').k.agg(['min', 'max', 'size'])
    res = {'frames': int(len(A)), 'atlas': int(len(sub)), 'first15_H': int(len(e)), 'H_videos': int(len(gh)),
           'first15_H_k_min_max_per_video': [int(gh['min'].min()), int(gh['max'].max())],
           'first15_H_frames_per_video': sorted(set(gh['size'].tolist()))}
    assert res['frames'] == 864000 and res['first15_H'] == 216000 and res['H_videos'] == 48, res
    C.BEEF.mkdir(parents=True, exist_ok=True)
    A[['obs', 'k', 'row', 'Y_nn', 'Y_np', 'Y_nt', 'code_row', 'full_row', 'part']].to_parquet(C.BEEF / 'frames.parquet')
    F = C.mice_table_full(A)
    res['counts'] = {}
    for fs, m0 in (('ext', np.ones(len(F), bool)), ('atlas', F.part.values == 'atlas'), ('first15_H', F.part.values == 'first15_H')):
        for part, m in (('all', np.ones(len(F), bool)), ('sg', F.sg.values), ('nonsg', F.nonsg.values), ('unknown', F.annot.values == 'NA')):
            res['counts'][f'{fs}|{part}'] = VF.counts(F[m0 & m])
    C.OUT22.mkdir(parents=True, exist_ok=True)
    (C.OUT22 / 'frames.json').write_text(json.dumps(res, indent=1))
    md = ['Bouts are counted within each frame set separately (a bout crossing frame 4500 of an H video counts once in each part).', '',
          '| frame set | pools | videos | frames | ' + ' | '.join(f'{C.NAME[b]}: positive frames / bouts / videos with >= 1 bout' for b in VF.LABS) + ' |',
          '|---|---|---|---|' + '---|' * len(VF.LABS)]
    for k, d in res['counts'].items():
        md.append(f"| {k} | {d['pools']} | {d['videos']} | {d['frames']:,} | " +
                  ' | '.join(f"{d[b]['frames_on']:,} / {d[b]['bouts']:,} / {d[b]['videos_with_bout']}" for b in VF.LABS) + ' |')
    (C.OUT22 / 'frames.md').write_text('\n'.join(md) + '\n')
    log(f'ext frames done ({time.time() - t0:.0f}s): {json.dumps({k: v for k, v in res.items() if k != "counts"})}')


def cmd_idcodes(a):
    """ext codes (parts of this frame set) restricted to the atlas frames vs the v2.2 FULL WINDOW parts."""
    Fe = pd.read_parquet(C.BEEF / 'frames.parquet', columns=['full_row'])
    m = Fe.full_row.values >= 0
    res = {}
    VF2 = {'BEEF': FULL}
    for s in range(1001, 1007):
        for ro in ('max', 'top10'):
            key = f'V_s{s}'
            new = VF.assemble(key, ro)[m]
            # old: assemble from the v2.2 full parts (same function, other directory)
            old_dir = VF.BEEF
            VF.BEEF = VF2['BEEF']
            try:
                old = VF.assemble(key, ro)
            finally:
                VF.BEEF = old_dir
            d = np.abs(new.astype(np.float32) - old.astype(np.float32))
            res[f'{key}|{ro}'] = {'max_abs_diff': float(d.max()), 'share_entries_diff_gt_0.01': float((d > 0.01).mean()),
                                  'share_entries_diff_gt_0.5': float((d > 0.5).mean()),
                                  'fire_agree': float(((new > 0) == (old > 0)).mean())}
            log(f'{key}|{ro}: {res[f"{key}|{ro}"]}')
    (C.OUT22 / 'identity_codes.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('frames', 'idcodes'))
    a = p.parse_args()
    {'frames': cmd_frames, 'idcodes': cmd_idcodes}[a.step](a)
