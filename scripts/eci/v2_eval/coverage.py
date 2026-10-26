"""
ECI v0 / v1 / v2 evaluation audit, step 1: LABEL COVERAGE. CPU only, no code / score is read.

For every annotated behaviour column (ants: Y2F, B2F, FOL, YOL, BOL and the unions groom_any = Y2F|B2F,
onlid_any = YOL|BOL, lid_any3 = FOL|YOL|BOL; mice: nn, np, nt and nose_nose = nn|np) and every experiment, count on
three frame sets:
    all    every labelled 5 fps frame of the annotation file (mice: atlas window, habituation '_H' frames >= 4500);
    probe  the v2_prob Tier B probe windows (results/vision/eci_v2_prob/<dom>/probe.parquet; contiguous 5 fps);
    f1     the V2-full 1 fps token store 'full1' (eci_codes/v2_diag/<dom>/full1/s*/frames.parquet; ants only, see below).
Counts: labelled frames, positive frames, bouts (runs of consecutive positive 5 fps frames inside one video; on a
subset, a bout counts if >= 1 of its frames is in the subset), videos with a label, videos with >= 1 positive, and the
same bouts / positive videos per half of the atlas split (units sorted, alternating; mice unit = pool).
Output: results/vision/eci_v2_eval/coverage_counts.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/eci_v2_eval'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_diag')
BEH = {'ants': {'Y2F': ['Y_Y2F'], 'B2F': ['Y_B2F'], 'FOL': ['Y_FOL'], 'YOL': ['Y_YOL'], 'BOL': ['Y_BOL'],
                'groom_any': ['Y_Y2F', 'Y_B2F'], 'onlid_any': ['Y_YOL', 'Y_BOL'], 'lid_any3': ['Y_FOL', 'Y_YOL', 'Y_BOL']},
       'mice': {'nn': ['Y_nn'], 'np': ['Y_np'], 'nt': ['Y_nt'], 'nose_nose': ['Y_nn', 'Y_np']}}
ANN = {'ants': ['dataset/ants/eci30/annotations.csv', 'dataset/ants/eci_c1v4_noday/annotations.csv'],
       'mice': ['dataset/mice/v1/annotations.csv']}


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def load(dom):
    cols = sorted({c for v in BEH[dom].values() for c in v})
    parts = []
    for f in ANN[dom]:
        a = pd.read_csv(REPO / f, dtype={'observation_id': str},
                        usecols=['observation_id', 'frame_idx'] + cols + (['experiment'] if dom == 'ants' else []))
        a['src'] = f
        parts.append(a)
    a = pd.concat(parts, ignore_index=True)
    if dom == 'mice':
        a['experiment'] = 'v1'
        a = a[~a.observation_id.str.endswith('_H') | (a.frame_idx >= 4500)]
    # keep the first file's copy of a duplicated (obs, k) (eci30 holds v2 / v3; c1v4 holds v4 / v4_infested)
    dup = a.duplicated(['experiment', 'observation_id', 'frame_idx'])
    log(f'{dom}: {len(a):,} rows, {int(dup.sum()):,} duplicate (experiment, obs, k) dropped; experiments '
        f'{a.experiment.value_counts().to_dict()}; files per experiment '
        f'{a.groupby("experiment").src.unique().apply(list).to_dict()}')
    a = a[~dup].sort_values(['experiment', 'observation_id', 'frame_idx']).reset_index(drop=True)
    a['unit'] = a.observation_id.str.rsplit('_', n=2).str[0] if dom == 'mice' else a.observation_id
    return a


def bout_ids(a, y, V):
    """bout id per row (-1 = not positive): runs of positive labelled rows, consecutive frame_idx, same video."""
    o, k = a.observation_id.to_numpy(), a.frame_idx.to_numpy()
    p = y & V
    new = p & np.r_[True, (o[1:] != o[:-1]) | (k[1:] != k[:-1] + 1) | ~p[:-1]]
    bid = np.cumsum(new) - 1
    return np.where(p, bid, -1)


def counts(a, sub, y, V, bid, half):
    m = sub & V
    if not m.any():
        return None
    o = a.observation_id.to_numpy()
    b = np.unique(bid[m & (bid >= 0)])
    r = {'labelled': int(m.sum()), 'pos': int((y & m).sum()), 'base_rate': float((y & m).sum() / m.sum()),
         'bouts': int(len(b)), 'videos': int(len(np.unique(o[m]))), 'videos_pos': int(len(np.unique(o[m & y])))}
    for h in (0, 1):
        mh = m & (half == h)
        r[f'h{h}_bouts'] = int(len(np.unique(bid[mh & (bid >= 0)])))
        r[f'h{h}_videos_pos'] = int(len(np.unique(o[mh & y])))
        r[f'h{h}_pos'] = int((y & mh).sum())
    r['rule_20b_3v_per_half'] = bool(min(r['h0_bouts'], r['h1_bouts']) >= 20 and
                                     min(r['h0_videos_pos'], r['h1_videos_pos']) >= 3)
    r['rule_20b_total_3v_per_half'] = bool(r['bouts'] >= 20 and min(r['h0_videos_pos'], r['h1_videos_pos']) >= 3)
    return r


def subset_keys(dom, a):
    out = {}
    key = pd.MultiIndex.from_arrays([a.observation_id.to_numpy(), a.frame_idx.to_numpy()])
    pr = REPO / f'results/vision/eci_v2_prob/{dom}/probe.parquet'
    P = pd.read_parquet(pr)
    out['probe'] = key.isin(pd.MultiIndex.from_arrays([P.obs.astype(str).to_numpy(), P.k.astype(int).to_numpy()]))
    if dom == 'ants':
        fs = [pd.read_parquet(p / 'frames.parquet', columns=['obs', 'k', 'series'])
              for p in sorted((BEE / dom / 'full1').glob('s*'))]
        F1 = pd.concat(fs, ignore_index=True)
        log(f'full1 {dom}: {len(F1):,} frames, series {F1.series.value_counts().to_dict()}, k % 5 '
            f'{(F1.k % 5).value_counts().to_dict()}')
        out['f1'] = key.isin(pd.MultiIndex.from_arrays([F1.obs.astype(str).to_numpy(), F1.k.astype(int).to_numpy()]))
    return out


def main():
    res = {}
    for dom in ('ants', 'mice'):
        a = load(dom)
        subs = {'all': np.ones(len(a), bool)}
        subs.update(subset_keys(dom, a))
        res[dom] = {}
        for ex, g in a.groupby('experiment', sort=True):
            gi = g.index.to_numpy()
            for sname, sm in subs.items():
                s = sm[gi]
                if not s.any():
                    continue
                gg = a.loc[gi]
                units = sorted(set(gg.unit[s]))
                half = gg.unit.map({u: i % 2 for i, u in enumerate(units)}).fillna(-1).to_numpy()
                for b, cols in BEH[dom].items():
                    V = gg[cols].notna().all(1).to_numpy()
                    y = (gg[cols].fillna(0).to_numpy() > 0).any(1)
                    r = counts(gg, s, y, V, bout_ids(gg, y, V), half)
                    if r is not None:
                        res[dom].setdefault(ex, {}).setdefault(b, {})[sname] = r
                        log(f'{dom} {ex} {b} {sname}: ' + json.dumps(r))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'coverage_counts.json').write_text(json.dumps(res, indent=1))
    log('done')


if __name__ == '__main__':
    sys.exit(main())
