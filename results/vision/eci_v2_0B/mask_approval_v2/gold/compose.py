import sys, glob
from pathlib import Path
from PIL import Image, ImageDraw
C = Path(__file__).parent / 'check'
pre, out, per = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3
keys = sorted({f.name.rsplit('_z', 1)[0] for f in C.glob(pre + '*_z*.png')})
T = 400
for p in range(0, len(keys), per):
    ks = keys[p:p + per]
    rows = []
    for k in ks:
        zs = sorted(C.glob(k + '_z*.png'))
        r = Image.new('RGB', (T * max(3, len(zs)), T + 14))
        for j, z in enumerate(zs):
            r.paste(Image.open(z).resize((T, T)), (j * T, 14))
        ImageDraw.Draw(r).text((3, 1), k, fill=(255, 255, 0))
        rows.append(r)
    W = max(r.width for r in rows)
    sh = Image.new('RGB', (W, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        sh.paste(r, (0, y)); y += r.height
    sh.save(f'{out}_{p // per}.png')
    print(f'{out}_{p // per}.png', ks)
