"""Hand-labelled mark set + scorer training.

  python label.py montage --domain mice --tag standin [--per_video 30]
      Picks long ISOLATED tracklets of the stand-in tracks (variant standin_blob), 4 dumped crops
      spread over each, and writes montages (one tracklet per row) to
      results/tracking_pilot/{domain}/labelling/ plus labels/{domain}_candidates.tsv.
      Labels are given per row (= per tracklet) by LOOKING at the 4 crops; all 4 viewed crops get
      that label. Rows judged unclear are labelled 'skip'.
  python label.py train --domain mice
      Split by time block (2-min blocks: even -> train, odd -> held-out test; whole tracklets on
      one side), choose the feature set (mice: hand vs DINOv2) by 5-fold grouped CV INSIDE the
      training half only, refit on the training half, report held-out accuracy + confusion matrix,
      save results/tracking_pilot/{domain}/scorer/scorer.pkl (+ report json).
"""
import argparse
import json
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from assign import DEFAULTS, frame_geometry, make_tracklets
from common import IDENTITIES, N_ANIMALS, RESULTS, ROOT, VIDEOS, load_tracks, out_dir
from score import design_matrix

LAB_DIR = ROOT / 'scripts/tracking_pilot/identity/labels'
BLOCK_S = 120


def montage(domain, tag, per_video, batch=1, seed=0):
    rng = np.random.default_rng(seed)
    md = RESULTS / domain / 'labelling'
    md.mkdir(parents=True, exist_ok=True)
    cands = []
    lp = LAB_DIR / f'{domain}_labels.tsv'
    done = pd.read_csv(lp, sep='\t') if lp.exists() else pd.DataFrame(columns=['video', 'frames', 'crops'])
    done_keys = set()
    for r in done.itertuples():
        for fn, f in zip(r.crops.split(';'), str(r.frames).split(';')):
            done_keys.add((r.video, int(fn.rsplit('.', 1)[0].rsplit('_', 1)[1]), int(f)))
    for vid in VIDEOS[domain]:
        tr0 = load_tracks(RESULTS / domain / vid / 'standin' / 'tracks.parquet')
        tr0 = tr0[tr0.variant == 'standin_blob'].reset_index(drop=True)
        tr, tab = make_tracklets(frame_geometry(tr0, DEFAULTS, N_ANIMALS[domain]), N_ANIMALS[domain], DEFAULTS)
        feats = pd.read_parquet(out_dir(domain, vid) / tag / 'feats.parquet')
        feats = feats[(feats.variant == 'standin_blob') & (feats.crop_file != '') & feats.mask_ok]
        feats = feats.merge(tr[['frame_src', 'track_id', 'tracklet']], on=['frame_src', 'track_id'])
        g = feats.groupby('tracklet')
        ok = g.size()
        dur = g.frame_src.max() - g.frame_src.min()
        good = ok.index[(ok >= 6) & (dur >= 3 * 30)]
        # skip tracklets whose span contains an already-labelled crop of the same track
        tid, f0, f1 = g.track_id.first(), g.frame_src.min(), g.frame_src.max()
        lab_tf = [(t, f) for v, t, f in done_keys if v == vid]
        good = [k for k in good if not any(t == tid.loc[k] and f0.loc[k] <= f <= f1.loc[k]
                                           for t, f in lab_tf)]
        # spread over time: sort by start, take evenly spaced
        starts = g.frame_src.min().loc[good].sort_values()
        pick = starts.index[np.linspace(0, len(starts) - 1, min(per_video, len(starts))).astype(int)] \
            if len(starts) else []
        for k in pd.unique(np.asarray(pick)):
            d = feats[feats.tracklet == k].sort_values('frame_src')
            sel = d.iloc[np.linspace(0, len(d) - 1, 4).astype(int)]
            cands.append({'video': vid, 'tracklet': int(k), 'f_start': int(d.frame_src.min()),
                          'f_end': int(d.frame_src.max()), 'track_id': int(d.track_id.iloc[0]),
                          'crops': ';'.join(sel.crop_file), 'frames': ';'.join(map(str, sel.frame_src)),
                          'label': ''})
    cands = pd.DataFrame(cands)
    cands.insert(0, 'row', [f'{domain[0]}{batch}{i:03d}' for i in range(len(cands))])
    cands['batch'] = batch
    LAB_DIR.mkdir(exist_ok=True)
    p = LAB_DIR / f'{domain}_candidates_b{batch}.tsv'
    cands.to_csv(p, sep='\t', index=False)
    # montages: 8 rows per image (mice) / 10 (ants)
    per = 8 if domain == 'mice' else 10
    tile = 224 if domain == 'mice' else 165
    for m0 in range(0, len(cands), per):
        rows = []
        for r in cands.iloc[m0:m0 + per].itertuples():
            ims = []
            for fn in r.crops.split(';'):
                im = cv2.imread(str(out_dir(domain, r.video) / tag / 'crops' / fn))
                ims.append(cv2.resize(im, (tile, tile), interpolation=cv2.INTER_LINEAR))
            lab = np.full((tile, 110, 3), 255, np.uint8)
            cv2.putText(lab, r.row, (5, 40), 0, 0.8, (0, 0, 0), 2)
            cv2.putText(lab, r.video, (5, 75), 0, 0.5, (0, 0, 0), 1)
            cv2.putText(lab, f'{r.f_start / 30:.0f}s', (5, 100), 0, 0.5, (0, 0, 0), 1)
            rows.append(np.hstack([lab] + ims))
        cv2.imwrite(str(md / f'montage_b{batch}_{m0 // per:02d}.png'), np.vstack(rows))
    print(f'{len(cands)} candidate tracklets -> {p}; montages in {md}')


def load_labelled(domain, tag='standin'):
    c = pd.read_csv(LAB_DIR / f'{domain}_labels.tsv', sep='\t', dtype={'label': str})
    c = c[c.label.isin(IDENTITIES[domain])]
    rows = []
    for r in c.itertuples():
        for fn, f in zip(r.crops.split(';'), r.frames.split(';')):
            # crop file = <variant>_<frame:06d>_<track_id>.jpg ; variant may contain '_'
            tid = int(fn.rsplit('.', 1)[0].rsplit('_', 1)[1])
            rows.append({'row': r.row, 'video': r.video, 'tracklet': f'{r.batch}_{r.tracklet}',
                         'crop_viewed': fn, 'frame_src': int(f), 'track_id': tid, 'label': r.label})
    lab = pd.DataFrame(rows)
    keep = []
    for vid, g in lab.groupby('video'):
        od = out_dir(domain, vid) / tag
        feats = pd.read_parquet(od / 'feats.parquet')
        feats = feats[feats.variant == 'standin_blob'].copy()
        feats['_i'] = feats.index.values  # row position in feats.parquet / dino.npy
        m = g.merge(feats, on=['frame_src', 'track_id'], how='left', suffixes=('', '_f'))
        miss = m._i.isna()
        if miss.any():
            print(f'  {vid}: {int(miss.sum())} labelled crops no longer read (not isolated '
                  f'under the current geometry) -> dropped')
        keep.append(m[~miss])
    lab = pd.concat(keep, ignore_index=True)
    lab['_i'] = lab._i.astype(int)
    return lab


def train(domain, tag='standin', read_p=0.6):
    lab = load_labelled(domain, tag)
    ids = IDENTITIES[domain]
    # design matrices
    mats = {}
    kinds = ['hand', 'dino'] if domain == 'mice' else ['colour']
    for kind in kinds:
        parts = []
        for vid, g in lab.groupby('video', sort=False):
            od = out_dir(domain, vid) / tag
            feats = pd.read_parquet(od / 'feats.parquet')
            dino = np.load(od / 'dino.npy').astype(np.float32) if domain == 'mice' else None
            idx = g._i.astype(int).values
            parts.append((g.index.values, design_matrix(domain, feats.iloc[idx],
                                                        None if dino is None else dino[idx], kind)))
        X = np.zeros((len(lab), parts[0][1].shape[1]), np.float32)
        for ix, x in parts:
            X[ix] = x
        mats[kind] = X
    y = lab.label.values
    block = (lab.frame_src // (BLOCK_S * 30)).values
    # whole tracklets on one side: use the tracklet's first labelled frame
    first = lab.groupby(['video', 'tracklet']).frame_src.transform('min') // (BLOCK_S * 30)
    is_test = (first % 2 == 1).values
    groups = (lab.video + '_' + lab.tracklet.astype(str)).values

    def model(C):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))

    cv = {}
    trn = ~is_test
    for kind in kinds:
        for C in [0.01, 0.1, 1.0]:
            accs = []
            for a, b in GroupKFold(5).split(mats[kind][trn], y[trn], groups[trn]):
                m = model(C).fit(mats[kind][trn][a], y[trn][a])
                accs.append((m.predict(mats[kind][trn][b]) == y[trn][b]).mean())
            cv[(kind, C)] = float(np.mean(accs))
    best = max(cv, key=cv.get)
    kind, C = best
    m = model(C).fit(mats[kind][trn], y[trn])
    P = m.predict_proba(mats[kind][is_test])
    pred = m.classes_[P.argmax(1)]
    yt = y[is_test]
    cm = confusion_matrix(yt, pred, labels=ids)
    readable = P.max(1) >= read_p
    rep = {
        'n_labelled_crops': int(len(lab)), 'n_tracklets': int(lab.groupby(['video', 'tracklet']).ngroups),
        'per_class_crops': lab.label.value_counts().to_dict(),
        'n_train_crops': int(trn.sum()), 'n_test_crops': int(is_test.sum()),
        'n_test_tracklets': int(len(set(groups[is_test]))),
        'test_per_class': pd.Series(yt).value_counts().to_dict(),
        'cv_in_train': {f'{k}_C{c}': v for (k, c), v in cv.items()},
        'chosen': {'kind': kind, 'C': C},
        'heldout_accuracy': float((pred == yt).mean()),
        'heldout_confusion_rows_true_cols_pred': {'labels': ids, 'matrix': cm.tolist()},
        'heldout_read_p': read_p,
        'heldout_frac_readable': float(readable.mean()),
        'heldout_accuracy_on_readable': float((pred[readable] == yt[readable]).mean()) if readable.any() else None,
        'heldout_per_video_accuracy': {v: float((pred[lab.video.values[is_test] == v] ==
                                                yt[lab.video.values[is_test] == v]).mean())
                                       for v in np.unique(lab.video.values[is_test])},
    }
    # for reference only (selection above used CV inside the training half): held-out accuracy of
    # the best setting of every OTHER feature kind
    rep['heldout_accuracy_other_kinds'] = {}
    for k2 in kinds:
        if k2 == kind:
            continue
        c2 = max([c for (kk, c) in cv if kk == k2], key=lambda c: cv[(k2, c)])
        m2 = model(c2).fit(mats[k2][trn], y[trn])
        p2 = m2.predict(mats[k2][is_test])
        rep['heldout_accuracy_other_kinds'][f'{k2}_C{c2}'] = {
            'accuracy': float((p2 == yt).mean()),
            'confusion': confusion_matrix(yt, p2, labels=ids).tolist()}
    # tracklet-level held-out accuracy (sum of log-probs over the tracklet's 4 crops)
    lp = pd.DataFrame(np.log(np.clip(P, 0.02, 1)), columns=m.classes_)
    lp['g'] = groups[is_test]
    lp['y'] = yt
    tl = lp.groupby('g').agg({**{c: 'sum' for c in m.classes_}, 'y': 'first'})
    rep['heldout_tracklet_accuracy'] = float((tl[list(m.classes_)].idxmax(1) == tl.y).mean())
    # tracklet confirmation at margin tau (same vote rule as assign.py, 4 crops per tracklet)
    Lv = tl[list(m.classes_)].values
    srt = np.sort(Lv, 1)
    margin = srt[:, -1] - srt[:, -2]
    win = np.array(m.classes_)[Lv.argmax(1)]
    rep['heldout_tracklet_confirm_by_tau'] = {
        str(t): {'confirmed': int((margin >= t).sum()), 'confirmed_wrong': int(((margin >= t) & (win != tl.y.values)).sum()),
                 'of': int(len(tl))} for t in [2, 3, 5, 8]}
    # final scorer: refit on ALL labelled data with the chosen setting (test numbers above are
    # from the training-half fit)
    final = model(C).fit(mats[kind], y)
    sd = RESULTS / domain / 'scorer'
    sd.mkdir(parents=True, exist_ok=True)
    with open(sd / 'scorer.pkl', 'wb') as fh:
        pickle.dump({'model': final, 'kind': kind, 'read_p': read_p, 'C': C}, fh)
    with open(sd / 'scorer_report.json', 'w') as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['montage', 'train'])
    ap.add_argument('--domain', required=True)
    ap.add_argument('--tag', default='standin')
    ap.add_argument('--per_video', type=int, default=30)
    ap.add_argument('--batch', type=int, default=1)
    a = ap.parse_args()
    if a.cmd == 'montage':
        montage(a.domain, a.tag, a.per_video, a.batch)
    else:
        train(a.domain, a.tag)
