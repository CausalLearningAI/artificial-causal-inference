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
ap.add_argument('--disc-only', action='store_true', help='only the discoverability / alignment shots')
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
    doms = [] if a.disc_only else pg.eval_on_selector_all('#domSel option', 'os => os.map(o => o.value)')
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
                    if pg.eval_on_selector(f'#kindSeg button[data-kind="{kind[0]}"]', 'b => b.disabled'):
                        # an earlier pick switched to an entry (another series) without this aggregation: back to the
                        # model / method of this loop on the domain's first entry of it
                        pg.select_option('#domSel', d)
                        pg.wait_for_timeout(500)
                        pg.select_option('#ax_model', model)
                        pg.wait_for_timeout(400)
                        pg.select_option('#ax_method', meth[0])
                        pg.wait_for_timeout(400)
                        if pg.eval_on_selector(f'#kindSeg button[data-kind="{kind[0]}"]', 'b => b.disabled'):
                            report.append(f'{d} · {model} · {meth[0]} · {kind[0]}: disabled on this entry')
                            continue
                    pg.click(f'#kindSeg button[data-kind="{kind[0]}"]')
                    pg.wait_for_timeout(600)
                    if d == 'mice':
                        # mice picker: Contrast (#mC), then Control (#mK) and Strata (#mX); every enabled choice
                        seen = []
                        for fam in pg.eval_on_selector_all('#mC option', 'os => os.map(o => [o.value, o.disabled])'):
                            if fam[1]:
                                report.append(f'{d} · {model} · {meth[0]} · {kind[0]} · contrast {fam[0]} (disabled)')
                                continue
                            pg.select_option('#mC', fam[0])
                            pg.wait_for_timeout(400)
                            ks = pg.eval_on_selector_all('#mK option:not([disabled])', 'os => os.map(o => o.value)')
                            for kv in (ks if fam[0] != 'B' else [None]):
                                if kv is not None and pg.eval_on_selector('#mK', 'e => e.value') != kv:
                                    pg.select_option('#mK', kv)
                                    pg.wait_for_timeout(400)
                                for sv in pg.eval_on_selector_all('#mX option:not([disabled])', 'os => os.map(o => o.value)'):
                                    if pg.eval_on_selector('#mX', 'e => e.disabled') is False:
                                        pg.select_option('#mX', sv)
                                    pg.wait_for_timeout(500)
                                    aid = pg.eval_on_selector('#mX', 'e => e.value')
                                    if aid in seen:
                                        continue
                                    seen.append(aid)
                                    n = pg.eval_on_selector('#mN', 'e => e.textContent')
                                    nr = pg.eval_on_selector_all('#side .nr .id', 'es => es.map(e => e.textContent)')
                                    emp = pg.eval_on_selector('#side', 'e => (e.querySelector(".empty")||{}).textContent || ""')
                                    topd = pg.locator('#side details.topd').count()
                                    h2 = '(no detail)'
                                    if nr:
                                        if pg.locator('#side .nr:visible').count() == 0:
                                            pg.click('#side summary')
                                            pg.wait_for_timeout(300)
                                        pg.click('#side .nr:visible >> nth=0')
                                        pg.wait_for_timeout(600)
                                        h2 = pg.eval_on_selector('#detail', 'e => (e.querySelector("h2")||{}).textContent || "(no detail)"')
                                    ann = pg.eval_on_selector_all('#side .warnbox, #side .note', 'es => es.map(e => e.textContent.trim()).filter(t => /annotat/.test(t))')
                                    if ann:
                                        report.append(f'      annotation check {model} {meth[0]} {kind[0]} {aid}: {ann}')
                                    report.append(f'{d} · {model} · {meth[0]} · {kind[0]} · {aid}: {n}; listed {nr[:10]} '
                                                  f'{emp}{" (candidates)" if topd else ""}; detail {h2}')
                                    if a.shots and shot < 24 and nr and (model == 'v0' or shot < 8):
                                        pg.screenshot(path=f'{a.shots}/shot_{shot:02d}_{d}_{model}_{meth[0]}_{kind[0]}_{aid}.png'.replace(' ', ''), full_page=True)
                                        shot += 1
                    else:
                        exps = pg.eval_on_selector_all('#apE option', 'os => os.map(o => [o.value, o.disabled, o.textContent])')
                        for e in exps:
                            if e[1]:
                                report.append(f'{d} · {model} · {meth[0]} · {kind[0]}: experiment {e[2]} (disabled)')
                                continue
                            pg.select_option('#apE', e[0])
                            pg.wait_for_timeout(900)
                            # ants Exp4 (v4): contrast group x colony / treatment instead of control x treatment
                            if pg.locator('#apGo').is_visible():
                                picks = []
                                for g in pg.eval_on_selector_all('#apG option', 'os => os.map(o => [o.value, o.disabled])'):
                                    if g[1]:
                                        report.append(f'{d} · {model} · {meth[0]} · {kind[0]} · v4 group {g[0]} (disabled)')
                                        continue
                                    pg.select_option('#apG', g[0])
                                    pg.wait_for_timeout(700)
                                    if pg.locator('#apSo').is_visible():
                                        for sv in pg.eval_on_selector_all('#apS option:not([disabled])', 'os => os.map(o => o.value)'):
                                            picks.append(('#apS', sv, f'group {g[0]} · {sv}', g[0]))
                                    else:
                                        picks.append((None, None, f'group {g[0]}', g[0]))
                            else:
                                picks = [('#apT', t, f'c={pg.eval_on_selector("#apC", "e=>e.value")} t={t}', None)
                                         for t in pg.eval_on_selector_all('#apT option:not([disabled])', 'os => os.map(o => o.value)')]
                            for sel, val, lab, grp in picks:
                                if grp is not None and pg.eval_on_selector('#apG', 'e => e.value') != grp:
                                    pg.select_option('#apG', grp)
                                    pg.wait_for_timeout(700)
                                if sel:
                                    pg.select_option(sel, val)
                                pg.wait_for_timeout(900)
                                if sel == '#apT':
                                    lab = f'c={pg.eval_on_selector("#apC", "e=>e.value")} t={val}'
                                n = pg.eval_on_selector('#apN', 'e => e.textContent')
                                nr = pg.eval_on_selector_all('#side .nr .id', 'es => es.map(e => e.textContent)')
                                h2 = pg.eval_on_selector('#detail', 'e => (e.querySelector("h2")||{}).textContent || "(no detail)"')
                                emp = pg.eval_on_selector('#side', 'e => (e.querySelector(".empty")||{}).textContent || ""')
                                topd = pg.locator('#side details.topd').count()
                                ann = pg.eval_on_selector_all('#side .warnbox, #side .note', 'es => es.map(e => e.textContent.trim()).filter(t => /annotat/.test(t))')
                                if ann:
                                    report.append(f'      annotation check {model} {meth[0]} {kind[0]} {e[0]} {lab}: {ann}')
                                report.append(f'{d} · {model} · {meth[0]} · {kind[0]} · {e[0]} {lab}: '
                                              f'{n}; listed {nr[:10]} {emp}{" (candidates)" if topd else ""}; detail {h2}')
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
                                    pg.screenshot(path=f'{a.shots}/shot_{shot:02d}_{d}_{model}_{meth[0]}_{kind[0]}_{e[0]}_{lab}.png'.replace(' ', '').replace('·', '_').replace('=', ''))
                                    shot += 1
                # spatial aggregation selector (model bar): max pooling on, the others off
                ag = pg.eval_on_selector_all('#aggSeg button', 'bs => bs.map(b => [b.dataset.agg, b.disabled, b.getAttribute("aria-pressed")])')
                report.append(f'{d} · {model} · spatial aggregation buttons [agg, disabled, pressed]: {ag}')
    H2 = 'e => (e.querySelector("h2")||{}).textContent || "(none)"'
    # a fresh page (no saved state): the views a first-time visitor gets
    pg = b.new_context(viewport={'width': 1300, 'height': 1000}).new_page()
    pg.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
    pg.goto(f'http://127.0.0.1:{a.port}/index.html')
    pg.wait_for_timeout(1500)
    # discoverability shots: frogs default view, frogs -> tadpoles phase (from the default model), a tadpole neuron;
    # the behaviour alignment panel of mice and ants
    pg.select_option('#domSel', 'frogs')
    pg.wait_for_timeout(1200)
    pg.screenshot(path=f'{a.shots}/disc_frogs_default.png', full_page=True)
    tad = [o for o in pg.eval_on_selector_all('#apE option', 'os => os.map(o => [o.value, o.disabled, o.textContent])')
           if o[0].startswith('tadpoles')]
    report.append(f'frogs default: phase options {pg.eval_on_selector_all("#apE option", "os => os.map(o => o.textContent)")}; '
                  f'model {pg.eval_on_selector("#ax_model", "e => e.value")}')
    if tad and not tad[0][1]:
        pg.select_option('#apE', tad[0][0])
        pg.wait_for_timeout(1500)
        if pg.locator('#side .nr:visible').count() == 0 and pg.locator('#side summary').count():
            pg.click('#side summary')
            pg.wait_for_timeout(300)
        if pg.locator('#side .nr:visible').count():
            pg.click('#side .nr:visible >> nth=0')
            pg.wait_for_timeout(1500)
        report.append(f'frogs -> tadpoles: model {pg.eval_on_selector("#ax_model", "e => e.value")}, '
                      f'{pg.eval_on_selector("#apN", "e => e.textContent")}, detail '
                      f'{pg.eval_on_selector("#detail", H2)}, '
                      f'strip tiles {pg.locator("#detail .v0t").count()}')
        pg.screenshot(path=f'{a.shots}/disc_frogs_tadpoles.png', full_page=True)
    else:
        report.append(f'frogs -> tadpoles: option {tad}')
    for d, shot_name in (('mice', 'align_mice'), ('ants', 'align_ants')):
        pg.select_option('#domSel', d)
        pg.wait_for_timeout(1200)
        pg.eval_on_selector('#alDet', 'e => e.open = true')
        pg.wait_for_timeout(500)
        report.append(f'{d} alignment panel: {pg.eval_on_selector("#alBox", "e => e.innerText.slice(0, 600)")!r}')
        pg.screenshot(path=f'{a.shots}/{shot_name}.png', full_page=True)
        links = pg.locator('#alBox .alj')
        if links.count():
            links.nth(0).click()
            pg.wait_for_timeout(1500)
            report.append(f'{d} alignment link -> detail '
                          f'{pg.eval_on_selector("#detail", H2)}')
    ban = pg.query_selector('#v0Ban')
    report.append(f'v0 banner (last state): {ban.text_content() if ban else None}')
    b.close()
bad = [r for r in report if '(candidates)' in r and "listed []" not in r and ' 0 found' not in r]
report.append(f'candidate lists shown next to discoveries (must be 0): {len(bad)}')
out = '\n'.join(report) + f'\nJS errors: {len(errors)}\n' + '\n'.join('   ' + e for e in errors[:30])
print(out)
Path(a.shots, 'report.txt').write_text(out)
srv.shutdown()
