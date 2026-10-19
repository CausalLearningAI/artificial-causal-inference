"""
Frame strips of the v0 neurons (the original ECI paper recipe, scripts/eci/v0/) for the ECI atlas page
(scripts/eci/build_atlas_v37.py, v0 entries).

v0 encodes each frame as ONE token (the mean of the 256 DINOv2-S/14 patch tokens), so it has no per-patch codes: no
heat maps, no arena maps. What its outputs do support are single frames ranked by a neuron's code (frame codes at 2 fps:
the top-20 codes of every frame, or the dense matrix for ants v2). Per neuron shown on the page (every raw-p < 0.05 hit
of any comparison + the 3 smallest-p codes of every comparison, the page's candidates rule):
  top    the highest frame of each video, the 8 videos with the highest such frame (1 per video)
  least  8 random videos that hold a frame where the code is exactly 0, one such frame picked at random in each
         (fixed seed); fewer if fewer videos hold one
  firing rate = share of the analysed 2 fps frames where the code is > 0
Tiles: the 512 px dataset JPG of that frame on the 5 fps grid (dataset/<subject>/<series>/frames/full/<video>/), resized
to TILE px; one WebP sprite per neuron, row 0 = top, row 1 = least. ants v2: its 2 fps frames are source frames
s = 15 i of the 30 fps raw video; the 5 fps JPG nearest to s is k = round((s - 2) / 6) (the R4 plan of ants v2: source
frame = 2 + 6 k), at most 0.1 s away.

Output: results/vision/eci_atlas_v37/v0/<key>/{frames.json, v0_<key>_<neuron>.webp}
Usage (Slurm, CPU): sbatch scripts/eci/v0/v0_atlas_frames.sh
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

REPO = Path(__file__).resolve().parents[3]
BIG = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_v0')
OUT = REPO / 'results/vision/eci_atlas_v37/v0'
# key -> (results dir, big dir, subject)
RUNS = {'ants_v2': ('results/vision/ants/eci_v0', 'ants_v2', 'ants'),
        'ants_v3': ('results/vision/ants/eci_v0_ants_v3', 'ants_v3', 'ants'),
        'ants_v4': ('results/vision/ants/eci_v0_ants_v4', 'ants_v4', 'ants'),
        'mice_v1': ('results/vision/mice/eci_v0_mice_v1', 'mice_v1', 'mice'),
        'frogs_v1': ('results/vision/frogs/eci_v0_frogs_v1', 'frogs_v1', 'frogs')}
N_CODES, K, TILE, N_CAND = 4608, 8, 160, 3


def log(*x):
    print(time.strftime('[%H:%M:%S]'), *x, flush=True)


def neuron_set(res):
    """0-based codes shown on the page: every raw-p < 0.05 hit + the N_CAND smallest p of every comparison."""
    P = pd.read_csv(res / 'pvalues_all_codes.csv')
    pc = [c for c in P.columns if c == 'p' or c.startswith('p_')]
    out = set()
    for c in pc:
        p = P[c].to_numpy(float)
        ok = np.isfinite(p)
        out |= set(np.where(ok & (p < 0.05))[0].tolist())
        idx = np.where(ok)[0]
        out |= set(idx[np.argsort(p[idx], kind='stable')[:N_CAND]].tolist())
    return sorted(out)


def jpg_dir(key, vids):
    """observation id -> its JPG folder relative to dataset/."""
    if key == 'mice_v1':
        return {o: f'mice/v1/frames/full/{Path(f).stem}' for o, f in zip(vids.observation_id, vids.observation_file)}
    if key == 'frogs_v1':
        return {o: f'frogs/v1/frames/full/{o}' for o in vids.observation_id}
    if key == 'ants_v2':
        return {o: f'ants/v2/frames/full/{o}' for o in vids.observation_id}
    return {o: f'ants/{e}/frames/full/{o}' for o, e in zip(vids.observation_id, vids.experiment)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', default=','.join(RUNS))
    ap.add_argument('--local', required=True, help='local scratch dir (JPGs are staged there)')
    a = ap.parse_args()
    loc = Path(a.local)
    for key in a.keys.split(','):
        t0 = time.time()
        rel, big, subj = RUNS[key]
        res, bd = REPO / rel, BIG / big
        vids = pd.read_csv(res / 'videos.csv', dtype={'observation_id': str})
        fr = pd.read_csv(bd / 'frames.csv', dtype={'observation_id': str})
        if key == 'ants_v2':
            fr['frame_idx'] = np.maximum(0, np.round((fr['source_frame'].to_numpy() - 2) / 6)).astype(int)
            X = np.load(bd / 'frame_codes_meanpool.npy', mmap_mode='r')
            assert X.shape == (len(fr), N_CODES), X.shape
            col = lambda j: np.asarray(X[:, j], dtype=np.float32)  # noqa: E731
        else:
            ti = np.load(bd / 'frame_codes_top20_idx.npy').astype(np.int64)
            tv = np.load(bd / 'frame_codes_top20_val.npy').astype(np.float32)
            assert ti.shape == tv.shape == (len(fr), 20), (ti.shape, tv.shape, len(fr))
            m = tv > 0
            S = sparse.csc_matrix((tv[m], (np.nonzero(m)[0], ti[m])), shape=(len(fr), N_CODES))
            col = lambda j: S[:, j].toarray().ravel()  # noqa: E731
        obs = fr['observation_id'].to_numpy()
        assert set(obs) == set(vids.observation_id), key
        codes = neuron_set(res)
        vcodes = np.load(res / 'video_codes.npy')
        rng = np.random.default_rng(0)
        picks = {}
        for j in codes:
            v = col(j)
            # cross-check: the video means of these frame codes are the tested per-video values
            vm = pd.Series(v).groupby(obs).mean().reindex(vids.observation_id).to_numpy()
            assert np.allclose(vm, vcodes[:, j], rtol=1e-4, atol=1e-6), (key, j, np.abs(vm - vcodes[:, j]).max())
            df = pd.DataFrame({'obs': obs, 'v': v, 'k': fr['frame_idx'].to_numpy()})
            top = df.loc[df.groupby('obs')['v'].idxmax()].sort_values('v', ascending=False)
            top = top[top.v > 0].head(K)
            z = df[df.v == 0]
            zv = z['obs'].unique()
            sel = rng.choice(zv, size=min(K, len(zv)), replace=False) if len(zv) else []
            least = [z[z.obs == o].sample(1, random_state=int(rng.integers(1 << 31))).iloc[0] for o in sel]
            picks[j] = {'firing_rate': float((v > 0).mean()),
                        'top': [[r.obs, float(r.v), int(r.k)] for r in top.itertuples()],
                        'least': [[r['obs'], 0.0, int(r['k'])] for r in least]}
        jd = jpg_dir(key, vids)
        need = sorted({f'{jd[o]}/frame_{k:06d}.jpg' for p in picks.values() for o, _, k in p['top'] + p['least']})
        lst = loc / f'{key}_frames.txt'
        lst.write_text('\n'.join(need) + '\n')
        st = loc / key
        st.mkdir(parents=True, exist_ok=True)
        subprocess.run(f'tar -C {REPO / "dataset"} -cf - -T {lst} | tar -C {st} -xf -', shell=True, check=True)
        od = OUT / key
        if od.exists():
            shutil.rmtree(od)
        od.mkdir(parents=True)
        tmp = loc / f'{key}_out'
        tmp.mkdir(exist_ok=True)
        for j, p in picks.items():
            sp = Image.new('RGB', (K * TILE, 2 * TILE), (17, 17, 17))
            for r, kind in enumerate(('top', 'least')):
                for i, (o, _, k) in enumerate(p[kind]):
                    im = Image.open(st / f'{jd[o]}/frame_{k:06d}.jpg').convert('RGB')
                    w, h = im.size
                    s = TILE / max(w, h)
                    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
                    sp.paste(im, (i * TILE + (TILE - im.size[0]) // 2, r * TILE + (TILE - im.size[1]) // 2))
            name = f'v0_{key}_{j + 1}.webp'
            sp.save(tmp / name, 'WEBP', quality=72, method=6)
            p['sprite'] = name
        for f in tmp.iterdir():
            shutil.copy(f, od / f.name)
        (od / 'frames.json').write_text(json.dumps({'key': key, 'tile': TILE, 'k': K, 'neurons': {
            str(j + 1): p for j, p in picks.items()}}, separators=(',', ':')))
        mb = sum(f.stat().st_size for f in od.iterdir()) / 1e6
        log(f'{key}: {len(codes)} neurons, {len(need)} JPGs, {mb:.1f} MB -> {od} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
