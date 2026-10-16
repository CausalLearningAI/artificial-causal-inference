"""
Headless check of the built atlas page (results/vision/eci_atlas_v37/site): serve it locally, open it in Chromium
(playwright), walk every domain x model x method x temporal aggregation x comparison, and report JS errors, what each
view lists (discoveries, empty states) and whether a neuron detail renders. Screenshots of a few views go to --shots.
Asset-store clip URLs (/_blob/...) do not resolve locally: video load errors are expected and not counted.
    python scripts/eci/atlas_v37_pagetest.py [--site DIR] [--shots DIR]   (run inside a Slurm job: srun -p defaultp)
"""
import argparse
import functools
import http.server
import json
import threading
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--site', default=str(Path(__file__).resolve().parents[2] / 'results/vision/eci_atlas_v37/site'))
ap.add_argument('--shots', default=str(Path(__file__).resolve().parents[2] / 'results/vision/eci_atlas_v37/pagetest'))
ap.add_argument('--port', type=int, default=8737)
a = ap.parse_args()
Path(a.shots).mkdir(parents=True, exist_ok=True)
for f in Path(a.shots).glob('*.png'):
    f.unlink()

H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=a.site)
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *x):
        pass


H = functools.partial(Q, directory=a.site)
srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
a.port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

from playwright.sync_api import sync_playwright  # noqa: E402

errors, report = [], []
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1300, 'height': 1000})
    pg.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
    pg.on('console', lambda m: errors.append(f'console {m.type}: {m.text}') if m.type == 'error' and '_blob' not in m.text
          and 'Failed to load resource' not in m.text else None)
    pg.goto(f'http://127.0.0.1:{a.port}/index.html')
    pg.wait_for_timeout(1500)
    doms = pg.eval_on_selector_all('#domSel option', 'os => os.map(o => o.value)')
    shot = 0
    for d in doms:
        pg.select_option('#domSel', d)
        pg.wait_for_timeout(500)
        for model in pg.eval_on_selector_all('#ax_model option:not([disabled])', 'os => os.map(o => o.value)'):
            pg.select_option('#ax_model', model)
            pg.wait_for_timeout(400)
            for meth in pg.eval_on_selector_all('#ax_method option', 'os => os.map(o => [o.value, o.disabled, o.textContent])'):
                if meth[1]:
                    report.append(f'{d} · {model} · method {meth[2]} (disabled)')
                    continue
                pg.select_option('#ax_method', meth[0])
                pg.wait_for_timeout(400)
                for kind in pg.eval_on_selector_all('#kindSeg button', 'bs => bs.map(b => [b.dataset.kind, b.disabled])'):
                    if kind[1]:
                        report.append(f'{d} · {model} · {meth[0]} · {kind[0]}: disabled')
                        continue
                    pg.click(f'#kindSeg button[data-kind="{kind[0]}"]')
                    pg.wait_for_timeout(600)
                    if d == 'mice':
                        cbs = pg.eval_on_selector_all('.cb[data-aid]', 'bs => bs.map(b => [b.dataset.aid, b.textContent.trim()])')
                        report.append(f'{d} · {model} · {meth[0]} · {kind[0]}: ' + ' '.join(f'{x}={n}' for x, n in cbs))
                        aids = [x for x, n in cbs if n not in ('0', '·', '–')][:3]
                        for aid in aids:
                            pg.click(f'.cb[data-aid="{aid}"]')
                            pg.wait_for_timeout(900)
                            h2 = pg.eval_on_selector('#detail', 'e => (e.querySelector("h2")||{}).textContent || "(no detail)"')
                            nr = pg.eval_on_selector_all('#side .nr .id', 'es => es.map(e => e.textContent)')
                            report.append(f'    {aid}: listed {nr[:12]}; detail {h2}')
                            if a.shots and shot < 12:
                                pg.screenshot(path=f'{a.shots}/shot_{shot:02d}_{d}_{model}_{meth[0]}_{kind[0]}_{aid}.png'.replace(' ', ''), full_page=False)
                                shot += 1
                    else:
                        exps = pg.eval_on_selector_all('#apE option', 'os => os.map(o => [o.value, o.disabled, o.textContent])')
                        for e in exps:
                            if e[1]:
                                report.append(f'{d} · {model} · {meth[0]} · {kind[0]}: experiment {e[2]} (disabled)')
                                continue
                            pg.select_option('#apE', e[0])
                            pg.wait_for_timeout(900)
                            ts = pg.eval_on_selector_all('#apT option:not([disabled])', 'os => os.map(o => o.value)')
                            for t in ts:
                                pg.select_option('#apT', t)
                                pg.wait_for_timeout(900)
                                n = pg.eval_on_selector('#apN', 'e => e.textContent')
                                nr = pg.eval_on_selector_all('#side .nr .id', 'es => es.map(e => e.textContent)')
                                h2 = pg.eval_on_selector('#detail', 'e => (e.querySelector("h2")||{}).textContent || "(no detail)"')
                                emp = pg.eval_on_selector('#side', 'e => (e.querySelector(".empty")||{}).textContent || ""')
                                report.append(f'{d} · {model} · {meth[0]} · {kind[0]} · {e[0]} c={pg.eval_on_selector("#apC", "e=>e.value")} t={t}: '
                                              f'{n}; listed {nr[:10]} {emp}; detail {h2}')
                                if not nr:
                                    pass
                                elif pg.locator('#side .nr:visible').count() == 0:
                                    pg.click('#side summary')
                                    pg.wait_for_timeout(300)
                                if nr:
                                    pg.click('#side .nr:visible >> nth=0')
                                    pg.wait_for_timeout(900)
                                    h2 = pg.eval_on_selector('#detail', 'e => (e.querySelector("h2")||{}).textContent || "(no detail)"')
                                    report.append(f'      clicked first listed: detail {h2}')
                                if a.shots and shot < 24 and nr:
                                    pg.wait_for_timeout(900)
                                    pg.screenshot(path=f'{a.shots}/shot_{shot:02d}_{d}_{model}_{meth[0]}_{kind[0]}_{e[0]}_{t}.png'.replace(' ', ''))
                                    shot += 1
    note = pg.eval_on_selector('#mNote', 'e => e.textContent')
    report.append(f'last model note: {note}')
    b.close()
out = '\n'.join(report) + f'\nJS errors: {len(errors)}\n' + '\n'.join('   ' + e for e in errors[:30])
print(out)
Path(a.shots, 'report.txt').write_text(out)
srv.shutdown()
