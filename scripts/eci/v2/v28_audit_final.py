"""
ECI v2.8 audit, last step (POST HOC): detailed held-out numbers for a short list of (dictionary, nose-nose neuron, nose-tail
neuron) candidates chosen by the SG-only rule "sgap" (v28_audit_combine / audit report):
  per behaviour, the candidate with the best SG average precision, nose-nose candidates restricted to SG discrimination
  AUROC <= 0.4 (prefers nose-nose over nose-tail), nose-tail candidates != the nose-nose pick with SG discrimination >= 0.6.
Outputs per neuron: firing rate (all frames), held-out AUROC / AP / precision@top-k/video (k = 3, 10, 30) with video- and
pool-bootstrap 95% CIs, AUROC vs nose-nose-mutual and nose-nose-passive, and the overlap of the two neurons' top frames.
usage: v28_audit_final.py --items KEY:NN:NT,KEY:NN:NT  -> results/vision/eci_v2_8/audit/final.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v28_audit_neurons as A  # noqa: E402

C = A.C


def main(a):
    F, _ = C.mice_table()
    non = F.nonsg.values
    rr = np.flatnonzero(non)
    obs = F.obs.values[rr]
    vcode, vids = pd.factorize(obs)
    vunit = pd.Series(F.unit.values[rr]).groupby(vcode).first().values
    gcode = pd.factorize(vunit)[0]
    rng = np.random.default_rng(20261126)
    Wv, Wp = A.video_weights(len(vids), rng), A.video_weights(len(vids), rng, gcode)
    ysub = {}
    for nm, col in (('nose-nose-mutual', 'Y_nn'), ('nose-nose-passive', 'Y_np')):
        ysub[nm] = (F[col].fillna(0).values > 0) if col in F.columns else None
    out = {}
    for it in a.items.split(','):
        key, nn, nt = it.split(':')
        nn, nt = int(nn), int(nt)
        pr, pp = A.mice_paths(key)
        import v24_top3 as T3
        Xc, Rc = T3.cols_chunked(pp, [nn, nt]), T3.cols_chunked(pr, [nn, nt])  # one sequential read per file
        cols = {nn: (Xc[:, 0], Rc[:, 0]), nt: (Xc[:, 1], Rc[:, 1])}
        res = {}
        for b, j in (('nose_nose', nn), ('nt', nt)):
            X, R = cols[j]
            sc = X.astype(np.float64) + 1e-6 * R / max(R.max(), 1e-12)
            y, v = F[f'y_{b}'].values[rr], F[f'v_{b}'].values[rr]
            d = {'neuron': j, 'fire_rate_all_frames': float((X > 0).mean()), 'fire_rate_raw_all_frames': float((R > 0).mean())}
            for kind, W in (('video', Wv), ('pool', Wp)):
                bt = A.boot_metrics(sc[rr], R[rr], y, v, obs, vcode, W, A.KS)
                for m, arr in bt.items():
                    d[f'{m}|{kind}_ci'] = A.np.percentile(arr, [2.5, 97.5]).tolist()
            pt = A.boot_metrics(sc[rr], R[rr], y, v, obs, vcode, np.ones((1, len(vids))), A.KS)
            d.update({m: float(arr[0]) for m, arr in pt.items()})
            if b == 'nose_nose':
                for nm, ys in ysub.items():
                    if ys is not None:
                        d[f'auc_vs_{nm}'] = float(C.auroc_cols(sc[rr][:, None], np.arange(len(rr)), ys[rr])[0])
            res[b] = d
        # overlap of the two neurons' held-out top-3 frames per video (raw ranking)
        tops = {}
        for j in (nn, nt):
            R = cols[j][1][rr]
            s = []
            st = np.r_[np.flatnonzero(np.r_[True, vcode[1:] != vcode[:-1]]), len(vcode)]
            for a0, b0 in zip(st[:-1], st[1:]):
                o = np.lexsort((np.arange(b0 - a0), -R[a0:b0].astype(np.float64)))[:3]
                s += (a0 + o).tolist()
            tops[j] = set(s)
        res['top3_frame_overlap'] = len(tops[nn] & tops[nt]) / len(tops[nn])
        res['activation_corr_heldout_P(on)'] = float(np.corrcoef(cols[nn][0][rr], cols[nt][0][rr])[0, 1])
        out[f'{key}:{nn}:{nt}'] = res
        print(key, nn, nt, json.dumps(res), flush=True)
    (A.OUT / 'final.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--items', required=True)
    main(p.parse_args())
