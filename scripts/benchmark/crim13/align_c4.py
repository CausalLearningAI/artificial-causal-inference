"""
CRIM13 alignment test - EXPLORATORY arm C4 read-out (addendum 1 of eci_align.sh; outside the pre-registered verdicts).
C4 = C1 (d3l_640_L20, x = [z_t, z_t - z_{t-50}]) with NO mask: SAEs trained on random video cells (align_train.py --arm
C4, theta* on the random-cell calib tokens, OUT/sae/C4_calib.json); here every eval frame of both halves is re-read
and re-encoded (same canvas, same encoder, same lag) and the frame value = max over ALL video cells of the 40 grid.
Scoring = align_train.score (same picks, folds, strain split, leave-one-resident-out, secondary metrics).
Outputs: OUT/eval/C4_<fold>_s<seed>.json, OUT/eval/C4_picks.npz, OUT/encode_json/c4_eval.json
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2] / 'scripts/eci'))
sys.path.insert(0, str(HERE.parent / 'mabe22'))
from align_common import LAG, OUT, PLAN, ROOT, SCORED, canvas_gpu, load_seek, log, read_frames, video_cells  # noqa: E402
from align_encode import Enc  # noqa: E402
from align_train import FOLDS, score  # noqa: E402


def main():
    import torch
    import eci_train as ET
    from src.eci.sae_calibrate import set_threshold
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    t00 = time.time()
    cal = json.loads((OUT / 'sae' / 'C4_calib.json').read_text())
    models = {}
    for k, v in cal.items():
        fold, s = k.split('_s')
        sae, norm, _ = ET.load_any(OUT / 'sae' / f'C4_{fold}_s{s}' / 'sae.pt', dev)
        set_threshold(sae, v['theta'])
        models[(fold, int(s))] = (sae, norm)
    P = pd.read_parquet(PLAN / 'frames.parquet')
    E = P[P.role == 'eval'].reset_index(drop=True)
    V = pd.read_parquet(PLAN / 'videos.parquet').set_index('video')
    enc = Enc('d3l_640_L20', dev)
    vids = list(dict.fromkeys(E.video))
    pool = ThreadPoolExecutor(4)

    def dec(q):
        R = E[E.video == q]
        d = ROOT / V.loc[q, 'half']
        offs = load_seek(d / f'{q}_t-seek.mat')
        fr, _ = read_frames(d / f'{q}_t.seq', offs, set(R.v.tolist()) | set((R.v - LAG).tolist()), threads=3)
        return fr
    fut = {j: pool.submit(dec, vids[j]) for j in range(min(len(vids), 5))}
    m = 4096
    codes = {k: torch.zeros(len(E), m, dtype=torch.float16, device=dev) for k in models}
    arg = {k: torch.zeros(len(E), m, dtype=torch.int16, device=dev) for k in models}
    cells = np.zeros(len(E), np.int64)
    pos_all = [None] * len(E)
    for k_, q in enumerate(vids):
        fr = fut.pop(k_).result()
        if k_ + 5 < len(vids):
            fut[k_ + 5] = pool.submit(dec, vids[k_ + 5])
        rows = np.flatnonzero(E.video.values == q)
        v = E.v.values[rows]
        W, H = int(V.loc[q, 'W']), int(V.loc[q, 'H'])
        vc = np.flatnonzero(video_cells(W, H))
        vct = torch.from_numpy(vc).to(dev)
        with torch.inference_mode():
            for a in range(0, len(v), 32):
                zt = enc.run(canvas_gpu([fr[x] for x in v[a:a + 32]], W, H, dev))[:, vct].float()
                zl = enc.run(canvas_gpu([fr[x - LAG] for x in v[a:a + 32]], W, H, dev))[:, vct].float()
                B, T = zt.shape[:2]
                x = torch.cat([zt, zt - zl], -1).reshape(B * T, -1)
                rr = torch.from_numpy(rows[a:a + B]).to(dev)
                for kk, (sae, norm) in models.items():
                    z = sae.encode(norm(x), mode='threshold').view(B, T, m)
                    mx, am = z.max(1)
                    codes[kk][rr] = mx.half()
                    arg[kk][rr] = am.to(torch.int16)
                    del z
                del zt, zl, x
        cells[rows] = len(vc)
        for r in rows:
            pos_all[r] = vc.astype(np.int16)
        if k_ % 20 == 0:
            log(f'  video {k_ + 1}/{len(vids)}: {(rows[-1] + 1)} frames, encoder {enc.gpu_s:.0f}s, wall {time.time() - t00:.0f}s')
    pool.shutdown()
    Y = torch.from_numpy(E[SCORED].values.astype(bool)).to(dev)
    w = torch.from_numpy(E.w.values.astype(np.float64)).to(dev)
    picks = {}
    for (fold, seed) in models:
        c = codes[(fold, seed)]
        sel, tst = FOLDS[fold]
        res = {'arm': 'C4', 'config': 'd3l_640_L20', 'lag': LAG, 'fold': fold, 'sel_half': sel, 'test_half': tst,
               'seed': seed, 'exploratory': True, 'readout': 'all video cells', 'calib': cal[f'{fold}_s{seed}'],
               'eval_frames': len(E), 'cells_per_frame_mean': float(cells.mean()), 'labels': SCORED,
               **score(c, E, Y, w, sel, tst, dev)}
        mm = res['main']
        log(f'  C4 {fold} s{seed}: ' + ', '.join(f"{b} {mm[b]['auc']['test_auc']:.3f}" for b in SCORED) + ' (test AUROC)')
        (OUT / 'eval' / f'C4_{fold}_s{seed}.json').write_text(json.dumps(res, indent=1, default=float))
        for b in SCORED:
            for kind in ('auc', 'top1'):
                j = mm[b][kind]['neuron']
                av = arg[(fold, seed)][:, j].long().cpu().numpy()
                pk = np.array([pos_all[i][av[i]] for i in range(len(E))], np.int16)
                picks[f'{fold}_s{seed}_{b}_{kind}'] = np.stack([c[:, j].float().cpu().numpy(), pk.astype(np.float32)])
    np.savez_compressed(OUT / 'eval' / 'C4_picks.npz', **picks)
    info = {'frames': len(E), 'images': enc.n_in, 'encoder_gpu_s': enc.gpu_s, 'wall_s': time.time() - t00,
            'gpu': torch.cuda.get_device_name(0), 'cells_per_frame_mean': float(cells.mean())}
    (OUT / 'encode_json' / 'c4_eval.json').write_text(json.dumps(info, indent=1))
    log('done ' + json.dumps(info))


if __name__ == '__main__':
    main()
