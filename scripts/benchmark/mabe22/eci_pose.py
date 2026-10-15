"""
MABe22 mouse ECI evaluation: POSE REFERENCE (CPU job; analysis only, keypoints never enter any representation).

What tracking gives for free: two label-free per-frame scalars from mouse_submission_keypoints.npy (3 mice x 12 points,
512 px scale; the label-frame index is the keypoint index):
    -min nose-to-nose distance over the 3 mouse pairs           (score: closer = higher)
    -min nose-to-tail-base distance over the 6 ordered pairs
each scored as a single "neuron" with the SAME weighted metrics (contact_sae.wmetrics: AUROC / AP / top-1%) on the same
eval frames and halves as the SAE arms (no selection needed: one fixed scalar; mean over the 2 halves x 2 split seeds).
Keypoint indices, label-free: the intra-mouse mean distance matrix over all frames (index 0 and 11 = the two ends of the
body chain = the farthest pair; 0 = the end whose two nearest points (1, 2) are mutually symmetric, i.e. nose + ears;
the tail chain = the points nearest 11 in sequence, tail base = the chain point where the inter-point step changes
from body to tail); reported in pose.json together with the semantic check of eci_prep (labels used only to validate).
Output: OUT/pose/pose.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/eci'))
from eci_common import BEHAVIOURS, OUT, PLAN, RAW, log  # noqa: E402
from eci_train import halves  # noqa: E402


def main():
    import contact_sae as cs
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    sk = np.load(RAW / 'mouse_submission_keypoints.npy', allow_pickle=True).item()['sequences']
    # label-free index identification: intra-mouse mean distance matrix on a sample of clips
    rng = np.random.default_rng(0)
    cl = rng.choice(sorted(set(E['clip'])), 200, replace=False)
    K = np.concatenate([sk[c]['keypoints'][::10] for c in cl]).astype(np.float32)  # (n, 3, 12, 2)
    Dm = np.linalg.norm(K[:, :, :, None] - K[:, :, None, :], axis=-1).mean((0, 1))  # (12, 12)
    a, b = np.unravel_index(np.argmax(Dm), Dm.shape)
    seq = np.argsort(Dm[0])  # points by distance from point 0
    ident = {'intra_mouse_mean_dist_px512': Dm.round(1).tolist(), 'farthest_pair': [int(a), int(b)],
             'points_by_distance_from_0': seq.tolist(), 'd(0,1)': float(Dm[0, 1]), 'd(0,2)': float(Dm[0, 2]),
             'd(1,2)': float(Dm[1, 2]), 'step_along_index': [float(Dm[i, i + 1]) for i in range(11)],
             'used': {'nose': 0, 'tail_base': 9}}
    log(f'identification: farthest pair {a},{b}; from 0: {seq.tolist()}; steps {np.round(ident["step_along_index"], 1)}')
    nn = np.zeros(len(E), np.float32)
    nt = np.zeros(len(E), np.float32)
    n_missing = n_zero_dist = 0
    for f in range(len(E)):
        k = sk[E['clip'][f]]['keypoints'][E.i[f]].astype(np.float32)  # (3, 12, 2)
        nose, tb = k[:, 0], k[:, 9]
        ok = ~(k == 0).all((1, 2))  # a mouse whose keypoints are all (0, 0) is missing
        bad_n = (nose == 0).all(1)
        bad_t = (tb == 0).all(1)
        dnn = [np.linalg.norm(nose[p] - nose[q]) for p in range(3) for q in range(p + 1, 3)
               if ok[p] and ok[q] and not bad_n[p] and not bad_n[q]]
        dnt = [np.linalg.norm(nose[p] - tb[q]) for p in range(3) for q in range(3)
               if p != q and ok[p] and ok[q] and not bad_n[p] and not bad_t[q]]
        nn[f] = min(dnn) if dnn else 1e4
        nt[f] = min(dnt) if dnt else 1e4
        n_missing += int((~ok).sum())
        n_zero_dist += int((min(dnn) if dnn else 1) == 0)
    w = E.w.values.astype(np.float64)
    res = {'identification': ident, 'scores': {}, 'missing_mice_frames': n_missing, 'zero_nn_dist_frames': n_zero_dist,
           'note': 'a mouse with all-(0,0) keypoints is treated as missing; a frame without a valid pair scores -1e4'}
    log(f'missing mice {n_missing}, zero nose-nose distance frames {n_zero_dist}')
    for name, s in (('neg_min_nose_nose', -nn), ('neg_min_nose_tailbase', -nt)):
        out = {}
        for b in BEHAVIOURS:
            y = E[b].values.astype(bool)
            m = []
            for sd in (0, 1):
                h = halves(E, sd)
                for hh in (0, 1):
                    r = h == hh
                    m.append(cs.wmetrics(s[r].astype(np.float64), y[r], w[r]))
            m = np.array(m)
            out[b] = {'auroc': float(m[:, 0].mean()), 'ap': float(m[:, 1].mean()), 'top1': float(m[:, 2].mean()),
                      'auroc_sd': float(m[:, 0].std()), 'ap_sd': float(m[:, 1].std())}
        res['scores'][name] = out
        log(f'{name}: ' + ', '.join(f"{b[:10]} {v['auroc']:.3f}/{v['ap']:.4f}/{v['top1']:.3f}" for b, v in out.items()))
    (OUT / 'pose').mkdir(parents=True, exist_ok=True)
    (OUT / 'pose' / 'pose.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
