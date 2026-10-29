import sys
from pathlib import Path
from PIL import Image, ImageDraw
C = Path(__file__).parent / 'check'
pre, out, per = sys.argv[1], sys.argv[2], int(sys.argv[3])
fs = sorted(C.glob(pre + '*_full.png'))
T = 600
for p in range(0, len(fs), per):
    grp = fs[p:p + per]
    sh = Image.new('RGB', (T * 2, T * ((len(grp) + 1) // 2)))
    for i, f in enumerate(grp):
        t = Image.open(f).resize((T, T))
        ImageDraw.Draw(t).text((3, 3), f.stem[:-5], fill=(255, 255, 0))
        sh.paste(t, ((i % 2) * T, (i // 2) * T))
    sh.save(f'{out}_{p // per}.png'); print(f'{out}_{p // per}.png', [f.stem for f in grp])
