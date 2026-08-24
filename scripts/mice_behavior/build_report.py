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

SRC = {'lcurve': FIG / 'story_learning_curve.png'}


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
        img[k] = enc(p, 1300)
    # The interactive figure is a VIEW over estimates.json -- the whole grid of estimates,
    # every one computed by build_estimates.py. Nothing is recomputed in the browser and nothing
    # is transcribed, so the figure cannot disagree with the numbers in the text.
    est_p = FIG / 'estimates.json'
    if not est_p.exists():
        raise SystemExit(f'{est_p} missing -- run scripts/mice_behavior/build_estimates.py first')
    est = json.load(open(est_p))
    chart = (Path(__file__).parent / 'report_chart.html').read_text()
    chart = chart.replace('__ESTIMATES_JSON__', json.dumps(est, separators=(',', ':')))

    # Same contract for the within-protocol decay figure: a VIEW over decay.json, every series
    # and every phase mean precomputed by build_decay.py from the human labels.
    dec_p = FIG / 'decay.json'
    if not dec_p.exists():
        raise SystemExit(f'{dec_p} missing -- run scripts/mice_behavior/build_decay.py first')
    decay = (Path(__file__).parent / 'report_decay.html').read_text()
    decay = decay.replace('__DECAY_JSON__', dec_p.read_text().strip())

    # And again for the model figure: a VIEW over models.json, one point per finished run, each
    # carrying its own full specification so a point can be read without decoding a run name.
    mod_p = FIG / 'models.json'
    if not mod_p.exists():
        raise SystemExit(f'{mod_p} missing -- run scripts/mice_behavior/build_models.py first')
    models = (Path(__file__).parent / 'report_models.html').read_text()
    models = models.replace('__MODELS_JSON__', mod_p.read_text().strip())

    # And the qualitative error figure: model x cohort x behaviour x annotated-or-not, sliced in
    # the browser from one payload of embedded thumbnails instead of six baked PNG grids.
    ex_p = FIG / 'examples.json'
    if not ex_p.exists():
        raise SystemExit(f'{ex_p} missing -- run scripts/mice_behavior/build_examples.py first')
    examples = (Path(__file__).parent / 'report_examples.html').read_text()
    examples = examples.replace('__EXAMPLES_JSON__', ex_p.read_text().strip())
    ex = json.load(open(ex_p))

    # The outcome-unit figure is a VIEW over outcome.json's `dist` block: the two distributions
    # section 02 argues from -- events per recording, and bout length as a share of bouts against
    # a share of time -- so the shape is on the page rather than a percentile quoted from it.
    out_p = FIG / 'outcome.json'
    if not out_p.exists():
        raise SystemExit(f'{out_p} missing -- run scripts/mice_behavior/build_outcome.py first')
    O_ = json.load(open(out_p))
    if 'dist' not in O_:
        raise SystemExit(f'{out_p} predates the `dist` block -- rerun build_outcome.py')
    units = (Path(__file__).parent / 'report_units.html').read_text()
    units = units.replace('__UNITS_JSON__', json.dumps(O_['dist'], separators=(',', ':')))

    head = (Path(__file__).parent / 'report_head.html').read_text()
    body = (Path(__file__).parent / 'report_body.py')
    rob_p = FIG / 'ppci_robustness.json'
    if not rob_p.exists():
        raise SystemExit(f'{rob_p} missing -- run build_ppci_robustness.py first')
    derm_p = FIG / 'derm.json'
    if not derm_p.exists():
        raise SystemExit(f'{derm_p} missing -- run scripts/mice_behavior/build_derm.py first')
    ns = {'img': img, 'CHART': chart, 'DECAY': decay, 'MODELS': models, 'EXAMPLES': examples,
          'UNITS': units,
          'E': est, 'M': json.load(open(mod_p)), 'O': O_, 'X': ex,
          'R': json.load(open(rob_p)), 'D': json.load(open(derm_p))}
    exec(compile(body.read_text(), str(body), 'exec'), ns)
    Path(a.out).write_text(head + ns['BODY'])
    mb = Path(a.out).stat().st_size / 1024 / 1024
    n_cells = len(est.get('cells', []))
    print(f'wrote {a.out}  ({mb:.2f} MB, {len(img)} figures, {n_cells} estimates)')
    if est.get('meta', {}).get('missing'):
        print(f"  NOTE incomplete estimates -- waiting on: {', '.join(est['meta']['missing'])}")


if __name__ == '__main__':
    main()
