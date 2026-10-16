"""
Exact frame of every published clip of the online model (atlas v37 builder, clip times in the raw source video).

The online model's mice and frogs clips were picked for the published page (v36, 2026-10-04); their picks cache was
re-run later (2026-10-15), so many published clips are no longer in it. A published clip is stored as [video, mean
activation (4 significant digits)] per neuron, length (single frame / 1 s = 5 frames) and row. This script finds,
in the SAE's own codes (codes_max, every 5 fps frame), the window(s) of that video whose mean activation of that
neuron rounds to the published mean: exactly one window -> its first frame_idx; none or several (a silent clip, mean 0,
or ties) -> null, and the page shows 'time unknown'. No guessing.

Output: results/vision/eci_atlas_v37/clip_frames/<tag>.json {"<neuron>|<length>|<video>|<mean>": frame_idx or null}
Usage (CPU job, codes staged to $LOCAL_DIR): sbatch scripts/eci/atlas_clip_frames.sh
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/vision/eci_atlas_v37'
SITE = OUT / 'site'
TAGS = {'fg448al': ('dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0', 'dataset/mice/v1/annotations.csv'),
        'frogsfg': ('dataset/frogs/eci/codes/matryoshka_btk_1024_k16_frogsfg_s0', 'dataset/frogs/eci/annotations.csv')}
LEN = {'frame': 1, '1s': 5}


def sig4_tol(v):
    """Half a unit in the 4th significant digit of v (v != 0)."""
    return 0.5 * 10.0 ** (np.floor(np.log10(abs(v))) - 3)


def main():
    s = (SITE / 'index.html').read_text()
    i = s.find('const ALL = ')
    A = json.loads(s[i + len('const ALL = '):s.find('\n', i)].rstrip(';'))
    loc = Path(os.environ.get('LOCAL_DIR') or OUT / 'tmp') / 'clip_frames'
    loc.mkdir(parents=True, exist_ok=True)
    (OUT / 'clip_frames').mkdir(exist_ok=True)
    for tag, (cdir, ann) in TAGS.items():
        es = [e for e in A['saes'] if e['tag'] == tag]
        obs = es[0].get('obs') or A['obs']
        want = {}  # neuron -> {(L, video, mean)}
        for f in sorted({f for e in es for f in e['nfile']['max']['files']}):
            D = json.loads((SITE / f).read_text())
            N = D['by']['max'] if 'by' in D else D['neurons']
            for j, n in N.items():
                for Ls in (n.get('clips') or {}).values():
                    for L, c in Ls.items():
                        for lst in c['info'].values():
                            for it in lst:
                                want.setdefault(int(j), set()).add((L, int(it[0]), float(it[1])))
        print(f'[{tag}] {sum(len(x) for x in want.values())} distinct clips of {len(want)} neurons', flush=True)
        src = ROOT / cdir / 'codes_max.npy'
        dst = loc / f'{tag}_codes_max.npy'
        shutil.copy(src, dst)
        Z = np.load(dst)
        M = pd.read_csv(ROOT / ann, usecols=['observation_id', 'frame_idx'])
        assert len(M) == len(Z), (len(M), Z.shape)
        o = M['observation_id'].astype(str).values
        cut = np.flatnonzero(o[1:] != o[:-1]) + 1
        lo = np.r_[0, cut]
        hi = np.r_[cut, len(o)]
        rng = {o[a]: (a, b) for a, b in zip(lo, hi)}
        assert len(rng) == len(lo), 'observation rows not contiguous'
        fidx = M['frame_idx'].values
        out, cnt = {}, {'unique': 0, 'silent': 0, 'none': 0, 'ties': 0}
        for j, clips in sorted(want.items()):
            x = Z[:, j].astype(np.float32)
            for L, v, m in sorted(clips):
                key = f'{j}|{L}|{v}|{m:g}'
                if m == 0:
                    out[key] = None
                    cnt['silent'] += 1
                    continue
                a, b = rng[obs[v][0]]
                w = LEN[L]
                seg = x[a:b]
                mm = seg if w == 1 else np.lib.stride_tricks.sliding_window_view(seg, w).mean(axis=1)
                hit = np.flatnonzero(np.abs(mm - m) <= sig4_tol(m) * (1 + 1e-6))
                if len(hit) == 1:
                    out[key] = int(fidx[a + hit[0]])
                    cnt['unique'] += 1
                else:
                    out[key] = None
                    cnt['none' if not len(hit) else 'ties'] += 1
        (OUT / 'clip_frames' / f'{tag}.json').write_text(json.dumps(out))
        print(f'[{tag}] {json.dumps(cnt)}', flush=True)
        del Z
        dst.unlink()


if __name__ == '__main__':
    sys.exit(main())
