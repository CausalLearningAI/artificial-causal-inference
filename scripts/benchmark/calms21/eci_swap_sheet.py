"""
CalMS21 ECI - ROLE SWAP (rules: eci_swap.sh): contact sheet of a sniff_face neuron (CPU job). Input: OUT/sheets/<arm>_s
<seed>_sniff_face.json written by eci_swap_train.py --sheet (top-16 frames among the sniff_face-sequence eval frames,
peak patch on the 64-grid canvas). Each tile = a 320 x 320 native-pixel crop of the video frame centred on the peak
patch (yellow box = the 16 x 16 canvas patch = 16 x 16 native px), border GREEN = sniff_face positive, RED = negative.
-> OUT/sheets/<arm>_s<seed>_sniff_face.jpg. argv: the JSON paths.
"""
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import PADY, H, W, decode_frames, log, stage_dir, video_path  # noqa: E402

CROP, TILE = 320, 320


def main():
    stage = stage_dir()
    for jp in sys.argv[1:]:
        J = json.loads(Path(jp).read_text())
        rows = J['frames']
        cell = 1024 // rows[0]['grid']

        def dec(q):
            loc = stage / f'{q}.mp4'
            shutil.copyfile(video_path('train', q), loc)
            want = [r['i'] for r in rows if r['video'] == q]  # offset 0 (plan/offset.json)
            fr, _ = decode_frames(loc, want, threads=2)
            loc.unlink()
            return q, fr
        with ThreadPoolExecutor(6) as ex:
            F = dict(ex.map(dec, sorted({r['video'] for r in rows})))
        sheet = Image.new('RGB', (4 * TILE, 4 * (TILE + 18)), (255, 255, 255))
        for k, r in enumerate(rows):
            im = F[r['video']][r['i']]
            pr, pc = divmod(r['peak_pos'], rows[0]['grid'])
            cy, cx = pr * cell + cell // 2 - PADY, pc * cell + cell // 2
            y0 = int(np.clip(cy - CROP // 2, 0, H - CROP))
            x0 = int(np.clip(cx - CROP // 2, 0, W - CROP))
            t = Image.fromarray(im[y0:y0 + CROP, x0:x0 + CROP])
            d = ImageDraw.Draw(t)
            bx, by = cx - x0 - cell // 2, cy - y0 - cell // 2
            d.rectangle([bx - 1, by - 1, bx + cell, by + cell], outline=(255, 230, 0), width=2)
            d.rectangle([0, 0, CROP - 1, CROP - 1], outline=(0, 200, 0) if r['label'] else (230, 0, 0), width=6)
            t = t.resize((TILE, TILE))
            x, y = (k % 4) * TILE, (k // 4) * (TILE + 18)
            sheet.paste(t, (x, y))
            ImageDraw.Draw(sheet).text((x + 4, y + TILE + 3), f"{'POS' if r['label'] else 'neg'} {r['video'][:8]} f{r['i']} "
                                       f"v{r['value']:.2f} {r['stratum']}", fill=(0, 0, 0))
        out = Path(jp).with_suffix('.jpg')
        sheet.save(out, quality=88)
        log(f"{out}: neuron {J['neuron']}, {sum(r['label'] for r in rows)} / {len(rows)} positives")


if __name__ == '__main__':
    main()
