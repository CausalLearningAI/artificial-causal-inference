#!/usr/bin/env python3
"""Assemble the mice status report as a single self-contained HTML file.

Kept in the repo (not /tmp) because the build was lost once to a cluster restart. Every figure is
re-read from results/vision/mice/frame/_figures at build time, so the page cannot drift from the
runs it describes.

    python scripts/mice_behavior/build_report.py -o /tmp/mice_report.html
"""
import argparse, base64, io, json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / 'results' / 'vision' / 'mice' / 'frame'
FIG = F / '_figures'
SSL = F / 'res448_k2_frozen_d4photo_sslinit'
XF = F / 'xfit_f2'

SRC = {
    'behav': FIG / 'story_behaviours.png', 'ppi': FIG / 'story_ppi.png',
    'within': FIG / 'story_within_phase.png', 'window': FIG / 'story_window_sensitivity.png',
    'outcome': FIG / 'story_outcome_choice.png', 'lcurve': FIG / 'story_learning_curve.png',
    'tokpix': FIG / 'story_tokens_pixels.png', 'apcausal': FIG / 'story_ap_vs_causal.png',
    'conf_nt': SSL / 'confusion_examples_nt.png', 'conf_nn': SSL / 'confusion_examples_nn.png',
    'un_nn_v1': XF / 'confident_nn_v1.png', 'un_nt_v1': XF / 'confident_nt_v1.png',
    'un_nn_v2': XF / 'confident_nn_v2.png', 'un_nt_v2': XF / 'confident_nt_v2.png',
}


def enc(p: Path, maxw: int, q: int = 80) -> str:
    im = Image.open(p).convert('RGB')
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'JPEG', quality=q, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='/tmp/mice_report.html')
    a = ap.parse_args()
    img = {}
    for k, p in SRC.items():
        if not p.exists():
            print(f'  [skip] {k}: {p} missing'); continue
        img[k] = enc(p, 1400 if k.startswith('conf') else 1300)
    P = json.load(open(FIG / 'ppi_results.json'))
    head = (Path(__file__).parent / 'report_head.html').read_text()
    body = (Path(__file__).parent / 'report_body.py')
    ns = {'img': img, 'P': P}
    exec(compile(body.read_text(), str(body), 'exec'), ns)
    Path(a.out).write_text(head + ns['BODY'])
    mb = Path(a.out).stat().st_size / 1024 / 1024
    print(f'wrote {a.out}  ({mb:.2f} MB, {len(img)} figures)')


if __name__ == '__main__':
    main()
