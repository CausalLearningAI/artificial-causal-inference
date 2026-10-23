"""
Train/test overlap of the v1 SAEs (deployed fg448al_s0 mice, antsfg_s0 ants) with the release-check scored frames
(scripts/eci/v2_mask2/release_ab.py frames: mice diag set restricted to the atlas window; ants scored_v2 / scored_v3).
Exact frame = same video and 5 fps frame; same video = the scored frame's video is among v1's TRAINING videos (v1's
held-out val pools / videos excluded, as train_sae_fg.py did). Also reported for the v2 pools (M0 bursts, P1 1 fps).
CPU only.  -> results/vision/eci_v2_diag/overlap.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
BEE2 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_mask2')
OUT = REPO / 'results/vision/eci_v2_diag/overlap.json'


def v1_rows(store):
    return np.concatenate([np.load(f)['rows'] for f in sorted(glob.glob(f'{store}/shards/shard_*/frames.npz'))])


def frames(dom, kind, root=BEE2):
    return pd.concat([pd.read_parquet(p / 'frames.parquet') for p in sorted((root / dom / kind).glob('s*'),
                                                                               key=lambda p: int(p.name[1:]))],
                     ignore_index=True)


def stats(sc_key, sc_vid, tr_key, tr_vid, ymask=None):
    m = np.ones(len(sc_key), bool) if ymask is None else ymask
    ex = pd.Index(tr_key)
    vid = set(tr_vid)
    k = pd.Index(sc_key[m])
    return {'n_scored': int(m.sum()), 'exact_share': float(k.isin(ex).mean()) if m.sum() else None,
            'same_video_share': float(np.isin(sc_vid[m], list(vid)).mean()) if m.sum() else None}


def main():
    res = {}
    # ------------------------------------------------------------ mice
    ann = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv', usecols=['observation_id', 'Y_nn', 'Y_np', 'Y_nt'])
    exp = pd.read_csv(REPO / 'data/mice/v1/experiment.csv').set_index('observation_id')
    mt = json.loads((REPO / 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448al_s0/metrics.json').read_text())
    val_pools = set(mt['val_pools'])
    r1 = v1_rows(REPO / 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448al_fps1')
    pool1 = exp.loc[ann.observation_id.values[r1], 'pool'].values
    r1t = r1[~np.isin(pool1, list(val_pools))]
    F = frames('mice', 'diag')
    obs, k = F.obs.values, F.k.values.astype(int)
    keep = ~pd.Series(obs).str.endswith('_H').values | (k >= 4500)  # atlas window (release_ab.mice_window)
    rows = F.row.values.astype(np.int64)[keep]
    a = ann.iloc[rows]
    y_nn = ((a.Y_nn.fillna(0) > 0) | (a.Y_np.fillna(0) > 0)).values
    y_nt = (a.Y_nt.fillna(0) > 0).values
    sc_vid = ann.observation_id.values[rows]
    o_all = ann.observation_id.values
    start = pd.Series(np.arange(len(o_all))).groupby(o_all).min()
    key = lambda o, kk: np.char.add(np.char.add(np.asarray(o, str), '|'), np.asarray(kk).astype(str))  # noqa: E731
    sc_key = key(obs[keep], k[keep])
    tr_vid = o_all[r1t]
    tr_key = key(tr_vid, r1t - start.loc[tr_vid].values)
    res['mice'] = {'v1_train_frames': int(len(r1t)), 'v1_val_pools': sorted(val_pools),
                   'scored_window_all': stats(sc_key, sc_vid, tr_key, tr_vid),
                   'scored_nose_nose_pos': stats(sc_key, sc_vid, tr_key, tr_vid, y_nn),
                   'scored_nose_tail_pos': stats(sc_key, sc_vid, tr_key, tr_vid, y_nt)}
    P = frames('mice', 'pool')
    res['mice']['v2_M0_pool'] = stats(sc_key, sc_vid, key(P.obs.values, P.k.values), P.obs.values)
    P1 = frames('mice', 'pool1', Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_diag'))
    res['mice']['v2_P1_pool'] = stats(sc_key, sc_vid, key(P1.obs.values, P1.k.values), P1.obs.values)
    # ------------------------------------------------------------ ants
    a10 = pd.read_csv(REPO / 'dataset/ants/eci/annotations.csv', usecols=['observation_id', 'frame_idx', 'frame_path'],
                      dtype={'observation_id': str})
    a30 = pd.read_csv(REPO / 'dataset/ants/eci30/annotations.csv',
                      usecols=['observation_id', 'frame_idx', 'frame_path', 'Y_Y2F', 'Y_B2F', 'Y_YOL', 'Y_BOL', 'Y_FOL'],
                      dtype={'observation_id': str})
    ma = json.loads((REPO / 'dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0/metrics.json').read_text())
    val_a = set(ma['val_pools'])
    r1a = v1_rows(REPO / 'dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfg_fps1')
    r1a = r1a[~np.isin(a10.observation_id.values[r1a], list(val_a))]
    key_obs = (a10.observation_id.values[r1a] + '|' + a10.frame_idx.values[r1a].astype(str))
    key_path = a10.frame_path.values[r1a]
    res['ants'] = {'v1_train_frames': int(len(r1a)), 'v1_val_videos': sorted(val_a),
                   'eci_obs_examples': sorted(set(a10.observation_id))[:3] + sorted(set(a10.observation_id))[-3:],
                   'eci30_obs_examples': sorted(set(a30.observation_id))[:3] + sorted(set(a30.observation_id))[-3:]}
    for ser in ('v2', 'v3'):
        Fa = frames('ants', f'scored_{ser}')
        rw = Fa.row.values.astype(np.int64)
        b = a30.iloc[rw]
        s_obs = (b.observation_id.values + '|' + b.frame_idx.values.astype(str))
        s_path = b.frame_path.values
        groom = ((b.Y_Y2F.fillna(0) > 0) | (b.Y_B2F.fillna(0) > 0)).values
        out = {'by_obs_frame': stats(s_obs, b.observation_id.values, key_obs, a10.observation_id.values[r1a]),
               'by_frame_path': stats(s_path, np.array([p.rsplit('/', 1)[0] for p in s_path]), key_path,
                                      np.array([p.rsplit('/', 1)[0] for p in key_path])),
               'groom_pos_by_frame_path': stats(s_path, np.array([p.rsplit('/', 1)[0] for p in s_path]), key_path,
                                                np.array([p.rsplit('/', 1)[0] for p in key_path]), groom)}
        if ser == 'v3':
            lid = ((b.Y_YOL.fillna(0) > 0) | (b.Y_BOL.fillna(0) > 0) | (b.Y_FOL.fillna(0) > 0)).values
            out['lid_pos_by_frame_path'] = stats(s_path, np.array([p.rsplit('/', 1)[0] for p in s_path]), key_path,
                                                 np.array([p.rsplit('/', 1)[0] for p in key_path]), lid)
        P = pd.concat([frames('ants', 'pool_v2'), frames('ants', 'pool_v3')])
        out['v2_M0_pool_by_obs_frame'] = stats(s_obs, b.observation_id.values,
                                               (P.obs.astype(str) + '|' + P.k.astype(str)).values, P.obs.astype(str).values)
        res['ants'][ser] = out
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
