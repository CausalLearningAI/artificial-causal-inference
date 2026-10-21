"""Preview clip of CRIM13 in the style of scripts/benchmark/make_clips.sh (3 s, 15 fps, long side 360, WebP).
cands: contact sheet of mid-bout frames; clip: write 45 PNG frames (every 25/15 -> source frames start + round(i*5/3))."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import seqio  # noqa: E402

mode, seq, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
r = seqio.Reader(seq)
print('frames', len(r), seqio.header(seq)['width'], seqio.header(seq)['height'])
if mode == 'cands':
    fr = [int(x) for x in sys.argv[4:]]
    ims = []
    for f in fr:
        g = cv2.resize(r.gray(f), (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(g, str(f), (5, 20), 0, 0.6, 255, 2)
        ims.append(g)
    while len(ims) % 3: ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(str(out / 'sheet.jpg'), np.vstack([np.hstack(ims[i:i + 3]) for i in range(0, len(ims), 3)]))
else:
    s = int(sys.argv[4])
    for i in range(45):
        g = r.gray(s + round(i * 25 / 15))
        h, w = g.shape
        k = 360 / max(h, w)
        g = cv2.resize(g, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f'f{i + 1:02d}.png'), g)
