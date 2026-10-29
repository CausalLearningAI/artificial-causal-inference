"""
ECI v2.0 FULL token store for the three design benchmarks (CRIM13, CalMS21, MABe22): ALL patches of every 5 fps frame
of every video / clip, fp16, with per-token mask flags (tight = the temporal study's B.fd >= 0.25; N2 = the thin-part
rule of src/eci/v2/mask2.py; both with the per-clip background and with a POOLED per-setup background).

Encoder (frozen v2.0, unchanged): src/eci/v2/encode.Encoder (DINOv3-L/16 block 20, rev ea8dc28, bf16, fp16 tokens) at
each benchmark's v2 side S (scripts/eci/v2_temporal/temporal.py BENCHES: MABe22 512, CalMS21 1024, CRIM13 640).
5 fps grid = source frames k with k % B.step == 0, 0 <= k < N (B.step: MABe22 / CalMS21 6 at 30 fps, CRIM13 5 at 25 fps).
The benchmark classes of temporal.py are imported read-only (that file belongs to another study).

Steps:
    size    (CPU) frames per benchmark x (S/16)^2 x 2048 bytes -> results/vision/eci_v2_0B/bench_store/size.json.
            Rule (caller): total above 5 TB (decimal, 1e12 bytes, the unit of every manifest here) -> STOP, no encode.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci/v2_temporal'))
OUT = REPO / 'results/vision/eci_v2_0B/bench_store'
TOKROOT = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_tokens/v2_full')
BYTES_PER_TOKEN = 1024 * 2  # fp16 x 1024
LIMIT_TB = 5.0
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def video_universe(name):
    """-> list of (video, set / split, N source frames) for ALL videos / clips of the benchmark (not only the plans)."""
    import numpy as np
    import temporal as T
    B = T.BENCHES[name]()
    if name == 'mabe22':
        EC = B.EC
        out = []
        for st in ('mouse_userTrain', 'mouse_submission'):
            clips = sorted(p.stem for p in (EC.VID / st).iterdir() if p.suffix == '.avi')
            out += [(c, st, 1801) for c in clips if c not in EC.EXCLUDE]
        return B, out
    if name == 'calms21':
        return B, [(q, sp, n) for q, (sp, n) in sorted(B.lens.items())]
    inv = json.loads(B.AC.INV.read_text())['videos']
    return B, [(v['name'], v['split'], int(v['frames'])) for v in sorted(inv, key=lambda v: v['name'])]


def cmd_size(a):
    import numpy as np
    res = {'unit': 'bytes; TB = 1e12 bytes, TiB = 2^40', 'bytes_per_token': BYTES_PER_TOKEN, 'limit_TB': LIMIT_TB,
           'rule_5fps': 'k % step == 0, 0 <= k < N', 'bench': {}}
    tot = 0
    for name in ('crim13', 'calms21', 'mabe22'):
        B, U = video_universe(name)
        G = B.S // 16
        fr = int(sum(math.ceil(n / B.step) for _, _, n in U))
        by = fr * G * G * BYTES_PER_TOKEN
        sets = {}
        for _, s, n in U:
            sets.setdefault(s, [0, 0])
            sets[s][0] += 1
            sets[s][1] += math.ceil(n / B.step)
        res['bench'][name] = {'S': B.S, 'G': G, 'step': B.step, 'videos': len(U), 'source_frames': int(sum(n for *_, n in U)),
                              'frames_5fps': fr, 'tokens': fr * G * G, 'bytes': by, 'TB': by / 1e12, 'TiB': by / 2**40,
                              'per_set': {k: {'videos': v[0], 'frames_5fps': v[1]} for k, v in sets.items()},
                              'flag_bytes_per_flag_file': fr * G * G // 8}
        tot += by
        log(f'{name}: {len(U)} videos, {fr:,} 5 fps frames, G {G}, {by / 1e12:.3f} TB')
    res['total_bytes'] = tot
    res['total_TB'] = tot / 1e12
    res['total_TiB'] = tot / 2**40
    res['above_limit'] = bool(tot / 1e12 > LIMIT_TB)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'size.json').write_text(json.dumps(res, indent=1))
    log(f'TOTAL {tot / 1e12:.3f} TB ({tot / 2**40:.3f} TiB); above {LIMIT_TB} TB: {res["above_limit"]}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('size',))
    a = p.parse_args()
    {'size': cmd_size}[a.step](a)


if __name__ == '__main__':
    main()
