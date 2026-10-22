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
VSTAT = {'checked': 0, 'webm': 0, 'loaded': 0, 'in_page': 0, 'looping': 0, 'bad': []}
VJS = '''() => { const v = document.querySelector('#vM'); if(!v) return null;
  return {src: v.currentSrc || v.src || '', ds: v.dataset.src || '', ready: v.readyState, t: v.currentTime,
          dur: v.duration, err: v.error ? v.error.code : null}; }'''


def set_chips(pg, root, label, values):
    """Make exactly `values` pressed in the chip group `label` (None = the only group) under root."""
    js = """([root, label]) => { const R = document.querySelector(root); if(!R) return [];
      const g = label ? [...R.querySelectorAll('.opt')].find(o => (o.querySelector('.tip')||{}).textContent===label) : R;
      return g ? [...g.querySelectorAll('.tchip')].map(b => [b.dataset.v, b.getAttribute('aria-pressed')==='true']) : []; }"""
    click = """([root, label, v]) => { const R = document.querySelector(root);
      const g = label ? [...R.querySelectorAll('.opt')].find(o => (o.querySelector('.tip')||{}).textContent===label) : R;
      g.querySelector(`.tchip[data-v="${v}"]`).click(); }"""
    for want in (True, False):
        for _ in range(8):
            st = pg.evaluate(js, [root, label])
            todo = [v for v, on in st if (v in values) == want and on != want]
            if not todo:
                break
            pg.evaluate(click, [root, label, todo[0]])
            pg.wait_for_timeout(450)


def vcheck(pg, lab):
    """The clip pack of the shown neuron: loaded, playing inside its page (seek), and looping (1 s clips)."""
    pg.wait_for_timeout(2500)
    a_ = pg.evaluate(VJS)
    for _ in range(12):  # a pack read from NFS can take a few seconds more
        if not a_ or a_['ready'] >= 2 or a_['err'] is not None:
            break
        pg.wait_for_timeout(500)
        a_ = pg.evaluate(VJS)
    if not a_ or not a_['ds']:
        return
    VSTAT['checked'] += 1
    VSTAT['webm'] += a_['ds'].endswith('.webm')
    if a_['ready'] >= 2 and a_['err'] is None:
        VSTAT['loaded'] += 1
    else:
        VSTAT['bad'].append((lab, a_))
        return
    ts = [a_['t']]
    for _ in range(4):
        pg.wait_for_timeout(400)
        ts.append(pg.evaluate(VJS)['t'])
    page = pg.evaluate('() => { const C = document.querySelector("#vM"); return C ? C.dataset.src : null; }')
    VSTAT['in_page'] += ts[0] > 0 or '_frame_' in a_['ds']
    VSTAT['looping'] += len(set(round(t, 2) for t in ts)) > 1 or '_frame_' in a_['ds']
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
        pairs = []
        for m_ in pg.eval_on_selector_all('#ax_model option:not([disabled])', 'os => os.map(o => o.value)'):
            pg.select_option('#ax_model', m_)
            pg.wait_for_timeout(400)
            ros = pg.eval_on_selector_all('#ax_readout option', 'os => os.map(o => [o.value, o.disabled, o.textContent])')
            report.append(f'{d} · {m_}: outcome options {ros}')
            pairs += [(m_, r[0]) for r in ros if not r[1]] or [(m_, None)]
        for model, ro in pairs:
            pg.select_option('#domSel', d)
            pg.wait_for_timeout(300)
            pg.select_option('#ax_model', model)
            pg.wait_for_timeout(400)
            if ro and pg.eval_on_selector('#ax_readout', 'e => e.value') != ro:
                pg.select_option('#ax_readout', ro)
                pg.wait_for_timeout(400)
            report.append(f'{d} · {model} · outcome {ro}: entry {pg.evaluate("() => DATA.sae")}')
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
                        # mice picker: Contrast (#mC), Control (#mK), strata chips (Phase / Genotype); then NOT RUN YET
                        # combinations (pooled phases, both genotypes, two lines, the sex contrast)
                        seen = []
                        for fam in pg.eval_on_selector_all('#mC option', 'os => os.map(o => o.value)'):
                            pg.select_option('#mC', fam)
                            pg.wait_for_timeout(500)
                            combos = []
                            if fam == 'B':
                                combos = [(None, ('Phase', [str(x)])) for x in range(1, 7)] + [(None, ('Phase', ['1', '2']))]
                            elif fam in ('A', 'C'):
                                for kv in pg.eval_on_selector_all('#mK option', 'os => os.map(o => o.value)'):
                                    combos += ([(kv, ('Genotype', [g])) for g in ('het', 'wt')] + [(kv, ('Genotype', ['het', 'wt']))]
                                               if fam == 'A' else [(kv, None)])
                            else:
                                combos = [(None, None)]
                            for kv, ch in combos:
                                if kv is not None and pg.eval_on_selector('#mK', 'e => e.value') != kv:
                                    pg.select_option('#mK', kv)
                                    pg.wait_for_timeout(400)
                                if ch:
                                    set_chips(pg, '#mStrata', ch[0], ch[1])
                                lab = f'{fam} ctrl={kv} {ch[0] + "=" + "+".join(ch[1]) if ch else ""}'
                                n = pg.eval_on_selector('#mN', 'e => e.textContent')
                                if 'NOT RUN' in n:
                                    nrun = pg.eval_on_selector('#side', 'e => (e.querySelector(".notrun")||{}).textContent || "(no NOT RUN box)"')
                                    report.append(f'{d} · {model} · {meth[0]} · {kind[0]} · {lab}: {n} | {nrun}')
                                    continue
                                if lab in seen:
                                    continue
                                seen.append(lab)
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
                                    if model.startswith('new') and kind[0] == 'time':
                                        vcheck(pg, f'{d} {model} {meth[0]} {kind[0]} {lab}')
                                ann = pg.eval_on_selector_all('#side .badge', 'es => es.map(e => e.textContent.trim())')
                                if ann:
                                    report.append(f'      annotation check {model} {meth[0]} {kind[0]} {lab}: {ann}')
                                report.append(f'{d} · {model} · {meth[0]} · {kind[0]} · {lab}: {n}; listed {nr[:10]} '
                                              f'{emp}{" (candidates)" if topd else ""}; detail {h2}')
                        # line strata: one line, two lines (NOT RUN YET), back to all
                        for lines in (['ash1l'], ['ash1l', 'kdm6b'], ['ash1l', 'kdm6b', 'kmt5b']):
                            set_chips(pg, '#mStrata', 'Line', lines)
                            report.append(f'{d} · {model} · {meth[0]} · {kind[0]} · lines {lines}: {pg.eval_on_selector("#mN", "e => e.textContent")}')
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
                                        vals = pg.eval_on_selector_all('#apSc .tchip', 'bs => bs.map(b => b.dataset.v)')
                                        for sv in [[v] for v in vals] + [vals]:
                                            picks.append(('chips', sv, f'group {g[0]} · {"+".join(sv)}', g[0]))
                                    else:
                                        picks.append((None, None, f'group {g[0]}', g[0]))
                            else:
                                picks = [('#apT', t, f'c={pg.eval_on_selector("#apC", "e=>e.value")} t={t}', None)
                                         for t in pg.eval_on_selector_all('#apT option:not([disabled])', 'os => os.map(o => o.value)')]
                            for sel, val, lab, grp in picks:
                                if grp is not None and pg.eval_on_selector('#apG', 'e => e.value') != grp:
                                    pg.select_option('#apG', grp)
                                    pg.wait_for_timeout(700)
                                if sel == 'chips':
                                    set_chips(pg, '#apSo', None, val)
                                elif sel:
                                    pg.select_option(sel, val)
                                pg.wait_for_timeout(900)
                                if sel == '#apT':
                                    lab = f'c={pg.eval_on_selector("#apC", "e=>e.value")} t={val}'
                                n = pg.eval_on_selector('#apN', 'e => e.textContent')
                                nr = pg.eval_on_selector_all('#side .nr .id', 'es => es.map(e => e.textContent)')
                                h2 = pg.eval_on_selector('#detail', 'e => (e.querySelector("h2")||{}).textContent || "(no detail)"')
                                emp = pg.eval_on_selector('#side', 'e => (e.querySelector(".empty")||{}).textContent || ""')
                                topd = pg.locator('#side details.topd').count()
                                ann = pg.eval_on_selector_all('#side .badge', 'es => es.map(e => e.textContent.trim())')
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
                                    if model.startswith('new'):
                                        vcheck(pg, f'{d} {model} {meth[0]} {kind[0]} {e[0]} {lab}')
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
    pg.on('console', lambda m: errors.append(f'console {m.type}: {m.text}') if m.type == 'error' and '_blob' not in m.text
          and 'Failed to load resource' not in m.text else None)
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
        vcheck(pg, 'tadpole first listed')
        report.append(f'tadpole clip: {pg.evaluate(VJS)}')
        pg.screenshot(path=f'{a.shots}/disc_frogs_tadpoles.png', full_page=True)
        pg.locator('#modelBar').screenshot(path=f'{a.shots}/ui_representation_bar.png')
        pg.locator('nav.bar').screenshot(path=f'{a.shots}/ui_tab_row.png')
    else:
        report.append(f'frogs -> tadpoles: option {tad}')
    # Alignment panel x Neurons prefix x Outcome (user, 2026-10-22): the listed neurons lie inside the selected prefix,
    # the panel follows the Outcome input, every model's best neuron opens an interpretation box with its clips / frames
    IDS_JS = '''() => [...document.querySelectorAll('#alBox table.alg td .pn, #alBox table.al td button, #alBox table.al td span')]
      .map(e => +(e.textContent.match(/#(\\d+)/)||[0,-1])[1]).filter(x => x >= 0)'''
    PX_JS = '''() => [...document.querySelectorAll('#prefixSeg button')].map(b => [b.textContent, b.disabled, b.getAttribute('aria-pressed'), b.title])'''
    BOX_JS = '''() => { const B = document.querySelector('#alIBox'); if(!B) return null;
      return {h2: (B.querySelector('h2')||{}).textContent || null, canvases: B.querySelectorAll('canvas').length,
              tiles: B.querySelectorAll('.v0t').length, video: !!B.querySelector('#vM'), charts: B.querySelectorAll('.chart').length,
              best: (B.querySelector('.pa')||{}).textContent || null, note: (B.querySelector('.note')||{}).textContent || null}; }'''

    def open_al(q):
        if q.eval_on_selector('#alBox', 'e => e.hidden'):
            q.click('#alBtn')
            q.wait_for_timeout(700)

    def set_px(q, p_):
        q.click(f'#prefixSeg button[data-p="{p_}"]')
        q.wait_for_timeout(800)

    pg.select_option('#domSel', 'mice')
    pg.wait_for_timeout(1000)
    for m_ in ('v0', 'online', 'new'):
        pg.select_option('#ax_model', m_)
        pg.wait_for_timeout(700)
        if a.shots:
            pg.locator('#prefixOpt').screenshot(path=f'{a.shots}/neurons_bar_{m_}.png')
        report.append(f'prefix buttons mice {m_}: {pg.evaluate(PX_JS)}; v0 note shown '
                      f'{not pg.eval_on_selector("#pxNote", "e => e.hidden")}')
    for d in ('mice', 'ants'):
        pg.select_option('#domSel', d)
        pg.wait_for_timeout(1000)
        pg.select_option('#ax_model', 'new')
        pg.wait_for_timeout(700)
        ros = [r[0] for r in pg.eval_on_selector_all('#ax_readout option', 'os => os.map(o => [o.value, o.disabled])') if not r[1]]
        open_al(pg)
        for ro in ros:
            pg.select_option('#ax_readout', ro)
            pg.wait_for_timeout(900)
            open_al(pg)
            for p_ in (128, 256):
                if pg.eval_on_selector(f'#prefixSeg button[data-p="{p_}"]', 'b => b.disabled'):
                    report.append(f'align {d} v2 {ro} {p_}: prefix button disabled')
                    continue
                set_px(pg, p_)
                ids = pg.evaluate(IDS_JS)
                v12 = [x for x in ids if x < 4608]  # v0 ids are its own (1..4608)
                ctx_ = pg.eval_on_selector('#alBox', 'e => (e.querySelector(".pa")||{}).textContent')
                cols = pg.eval_on_selector_all('#alBox table.alg tbody tr', 'rs => rs.map(r => [...r.querySelectorAll("td")].slice(1).map(td => (td.querySelector(".pn")||{}).textContent || td.textContent))')
                report.append(f'align {d} v2 {ro} prefix {p_}: {ctx_!r}; best per behaviour [v0, v1, v2] {cols}; '
                              f'v1/v2 top-5 ids {sorted(set(ids))[:40]}')
                over = [x for x in pg.evaluate('''() => [...document.querySelectorAll('#alBox table.alg tr')].slice(1).flatMap(r => [...r.querySelectorAll('td')].slice(2).map(td => +((td.textContent.match(/#(\\d+)/)||[0,-1])[1])))''') if x >= p_]
                report.append(f'   v1 / v2 best neurons outside the first {p_}: {over} (must be [])')
                if d == 'mice' and ro == 'continuous':
                    pg.locator('#alBox').screenshot(path=f'{a.shots}/align_panel_mice_v2_continuous_{p_}.png')
    # the interpretation box: click every best neuron of every model (mice, ants v2/v3/v4 via the series of the shown
    # comparison), record what the box shows
    for d in ('mice', 'ants'):
        pg.select_option('#domSel', d)
        pg.wait_for_timeout(1000)
        pg.select_option('#ax_model', 'new')
        pg.wait_for_timeout(700)
        if pg.eval_on_selector('#ax_readout', 'e => e.value') != 'continuous':
            pg.select_option('#ax_readout', 'continuous')
            pg.wait_for_timeout(700)
        sers = ['-']
        if d == 'ants':
            sers = [o[0] for o in pg.eval_on_selector_all('#apE option', 'os => os.map(o => [o.value, o.disabled])') if not o[1]]
        for se in sers:
            if se != '-':
                pg.click('#tabCmp')
                pg.wait_for_timeout(400)
                pg.select_option('#domSel', d)
                pg.wait_for_timeout(700)
                pg.select_option('#ax_model', 'new')
                pg.wait_for_timeout(600)
                pg.select_option('#apE', se)
                pg.wait_for_timeout(900)
            open_al(pg)
            if d == 'ants':
                lid = pg.eval_on_selector_all('#alBox table.alg tbody tr', 'rs => rs.filter(r => /lid/.test(r.textContent)).map(r => [...r.querySelectorAll("td")].map(td => [td.textContent.trim().slice(0, 60), td.title]))')
                report.append(f'lid rows ants {se}: {lid}')
                if a.shots:
                    pg.locator('#alBox table.alg').screenshot(path=f'{a.shots}/align_table_ants_{se}.png')
            n_btn = pg.locator('#alBox table.alg .alj').count()
            n_cell = pg.locator('#alBox table.alg td .pn').count()
            report.append(f'interp {d} {se}: {n_btn} of {n_cell} best-neuron cells clickable')
            for i in range(n_btn):
                btn = pg.locator('#alBox table.alg .alj').nth(i)
                lab = f'{btn.get_attribute("data-m")} #{btn.get_attribute("data-n")} ({btn.get_attribute("data-b")})'
                btn.click()
                pg.wait_for_timeout(1800)
                bx = pg.evaluate(BOX_JS)
                report.append(f'   box {d} {se} {lab}: {bx}')
                if d == 'mice' and i == 2 and a.shots:   # a v2 box
                    pg.wait_for_timeout(1500)
                    pg.locator('#alBox').screenshot(path=f'{a.shots}/interp_box_mice_{btn.get_attribute("data-m")}_{btn.get_attribute("data-n")}.png')
                if bx and bx['video']:
                    vcheck(pg, f'interp box {d} {se} {lab}')
                pg.select_option('#domSel', d) if False else None
                if d == 'ants' and se != '-' and pg.eval_on_selector('#apE', 'e => e.value') != se:
                    pg.select_option('#apE', se)
                    pg.wait_for_timeout(700)
                open_al(pg)
            if pg.locator('#ibClose').count():
                pg.click('#ibClose')
                pg.wait_for_timeout(500)
    # #73 (mice, v2 continuous, event rate, sex contrast at stage 2): its activity label
    pg.select_option('#domSel', 'mice')
    pg.wait_for_timeout(900)
    pg.select_option('#ax_model', 'new')
    pg.wait_for_timeout(600)
    if pg.eval_on_selector('#ax_readout', 'e => e.value') != 'continuous':
        pg.select_option('#ax_readout', 'continuous')
        pg.wait_for_timeout(600)
    set_px(pg, 128)
    pg.click('#tabCmp')
    pg.wait_for_timeout(400)
    pg.click('#kindSeg button[data-kind="rate"]')
    pg.wait_for_timeout(700)
    pg.select_option('#mC', 'S')
    pg.wait_for_timeout(700)
    set_chips(pg, '#mStrata', 'Phase', ['2'])
    set_chips(pg, '#mStrata', 'Genotype', ['het', 'wt'])
    pg.wait_for_timeout(900)
    ids = pg.eval_on_selector_all('#side .nr .id', 'es => es.map(e => e.textContent)')
    k73 = [i for i, x in enumerate(ids) if x.strip().lstrip('#') == '73']
    if k73:
        pg.click(f'#side .nr >> nth={k73[0]}')
        pg.wait_for_timeout(1200)
    facts = pg.eval_on_selector('#detail', 'e => (e.querySelector(".facts")||{}).textContent || null')
    ftip = pg.eval_on_selector('#detail', 'e => ((e.querySelector(".facts .tip")||{}).title) || null')
    side = pg.eval_on_selector('#side', 'e => [(e.querySelector("h2")||{}).textContent, (e.querySelector(".nmeta")||{}).textContent]')
    il = pg.eval_on_selector('#detail', 'e => (e.querySelector(".interp")||{}).textContent || null')
    report.append(f'#73 discovery panel head: {side}; interpretation line {il!r}')
    report.append(f'#73 check: listed {ids}; detail {pg.eval_on_selector("#detail", H2)}; facts {facts!r}; '
                  f'activity tooltip {ftip!r}')
    if a.shots:
        pg.screenshot(path=f'{a.shots}/n73_mice_v2_continuous_rate_S2.png', full_page=False)
    for d_, m_ in (('ants', 'new'), ('ants', 'v0'), ('frogs', 'new'), ('mice', 'online')):
        pg.select_option('#domSel', d_)
        pg.wait_for_timeout(900)
        pg.select_option('#ax_model', m_)
        pg.wait_for_timeout(900)
        pg.click('#tabCmp')
        pg.wait_for_timeout(500)
        hd = pg.eval_on_selector('#side', 'e => [(e.querySelector("h2")||{}).textContent, (e.querySelector(".nmeta")||{}).textContent]')
        fc = pg.eval_on_selector('#detail', 'e => (e.querySelector(".facts")||{}).textContent || null')
        report.append(f'discovery head {d_} {m_}: {hd}; first detail facts {fc!r}')
    if not pg.eval_on_selector('#alBox', 'e => e.hidden'):
        pg.click('#alBtn')
    pg.click('#specBtn')
    pg.wait_for_timeout(400)
    st = pg.eval_on_selector('#specBox', 'e => e.innerText')
    if a.shots:
        pg.locator('header').screenshot(path=f'{a.shots}/details_panel_mice.png')
    report.append(f'Details panel prose left (must be all False): Priors {"Priors:" in st}, Assumptions {"Assumptions:" in st}, '
                  f'v2 final decisions {"v2 final decisions" in st}, v0 on this comparison {"v0 on this comparison" in st}, '
                  f'v0 SAE {"v0 SAE:" in st}')
    pg.click('#specBtn')
    NOTRUN_JS = 'e => (e.querySelector(".notrun")||{}).textContent || ""'
    # strata chips: ants v4 colony contrast (Pathogen exposure), mice (Line / Sex / Genotype); annotation badges
    pg.select_option('#domSel', 'ants')
    pg.wait_for_timeout(1000)
    pg.select_option('#ax_model', 'new')
    pg.wait_for_timeout(800)
    pg.click('#tabCmp')
    pg.wait_for_timeout(500)
    pg.select_option('#apE', 'v4')
    pg.wait_for_timeout(900)
    pg.select_option('#apG', 'colony')
    pg.wait_for_timeout(900)
    for sv in (['t2'], ['t6'], ['t2', 't6']):
        set_chips(pg, '#apSo', None, sv)
        pg.wait_for_timeout(600)
        report.append(f'ants v4 colony strata {sv}: {pg.eval_on_selector("#apN", "e => e.textContent")}')
    pg.screenshot(path=f'{a.shots}/strata_ants_v4.png', full_page=False)
    pg.select_option('#apG', 'exposure')
    pg.wait_for_timeout(900)
    set_chips(pg, '#apSo', None, ['healthy', 'infested'])
    report.append(f'ants v4 exposure strata healthy+infested: {pg.eval_on_selector("#apN", "e => e.textContent")} | '
                  f'{pg.eval_on_selector("#side", NOTRUN_JS)}')
    pg.screenshot(path=f'{a.shots}/strata_ants_v4_notrun.png', full_page=False)
    pg.select_option('#domSel', 'mice')
    pg.wait_for_timeout(1000)
    pg.select_option('#ax_model', 'new')
    pg.wait_for_timeout(800)
    pg.click('#tabCmp')
    pg.wait_for_timeout(500)
    pg.select_option('#mC', 'A')
    pg.wait_for_timeout(600)
    pg.select_option('#mK', '4')
    pg.wait_for_timeout(600)
    set_chips(pg, '#mStrata', 'Genotype', ['het'])
    pg.wait_for_timeout(1200)
    report.append(f'mice A het 4->5 badges: {pg.eval_on_selector_all("#side .badge", "es => es.map(e => e.textContent)")}')
    pg.screenshot(path=f'{a.shots}/strata_mice_badge.png', full_page=False)
    set_chips(pg, '#mStrata', 'Line', ['ash1l'])
    report.append(f'mice line ash1l: {pg.eval_on_selector("#mN", "e => e.textContent")}')
    pg.screenshot(path=f'{a.shots}/strata_mice_line.png', full_page=False)
    set_chips(pg, '#mStrata', 'Line', ['ash1l', 'kdm6b', 'kmt5b'])
    pg.select_option('#mC', 'S')
    pg.wait_for_timeout(800)
    report.append(f'mice sex contrast: {pg.eval_on_selector("#mN", "e => e.textContent")} | '
                  f'{pg.eval_on_selector("#side", NOTRUN_JS)}')
    pg.screenshot(path=f'{a.shots}/strata_mice_sex_notrun.png', full_page=False)
    pg.select_option('#mC', 'B')
    pg.wait_for_timeout(800)
    report.append(f'mice B badges: {pg.eval_on_selector_all("#side .badge", "es => es.map(e => e.textContent)")}')
    pg.screenshot(path=f'{a.shots}/badge_neutral.png', full_page=False)
    ban = pg.query_selector('#v0Ban')
    report.append(f'v0 banner (last state): {ban.text_content() if ban else None}')
    # fresh-page shots of the Behavioural representation box (+ tab row), its Alignment and Details panels:
    # mice, ants v4 (colony contrast), frogs tadpoles; each on a new page with no saved state
    for d, tag in (('mice', 'mice'), ('ants', 'ants_v4'), ('frogs', 'frogs_tadpoles')):
        ctx = b.new_context(viewport={'width': 1300, 'height': 1000})
        q = ctx.new_page()
        q.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
        q.goto(f'http://127.0.0.1:{a.port}/index.html')
        q.wait_for_timeout(1500)
        q.select_option('#domSel', d)
        q.wait_for_timeout(1200)
        if tag == 'ants_v4':
            q.select_option('#apE', 'v4')
            q.wait_for_timeout(900)
            if q.eval_on_selector_all('#apG option', 'os => os.map(o => o.value)').count('colony'):
                q.select_option('#apG', 'colony')
                q.wait_for_timeout(900)
        if tag == 'frogs_tadpoles':
            tv = [o[0] for o in q.eval_on_selector_all('#apE option', 'os => os.map(o => [o.value, o.disabled])')
                  if o[0].startswith('tadpoles') and not o[1]]
            if tv:
                q.select_option('#apE', tv[0])
                q.wait_for_timeout(1500)
        q.locator('#modelBar').screenshot(path=f'{a.shots}/rep_box_{tag}.png')
        q.locator('header').screenshot(path=f'{a.shots}/rep_header_{tag}.png')
        q.locator('nav.bar').screenshot(path=f'{a.shots}/tab_row_{tag}.png')
        q.screenshot(path=f'{a.shots}/fresh_{tag}.png', full_page=False)
        if q.eval_on_selector('#alBtn', 'e => e.disabled'):  # no behaviour annotations (frogs): the button says so
            report.append(f'fresh {tag}: Alignment disabled ({q.eval_on_selector("#alBtn", "e => e.title")})')
        else:
            q.click('#alBtn')
            q.wait_for_timeout(700)
        q.locator('header').screenshot(path=f'{a.shots}/alignment_{tag}.png')
        report.append(f'fresh {tag}: model {q.eval_on_selector("#ax_model", "e => e.value")}, readout '
                      f'{q.eval_on_selector_all("#ax_readout option", "os => os.map(o => o.textContent)")}, prefix '
                      f'{q.eval_on_selector_all("#prefixSeg button", "bs => bs.map(b => b.textContent)")}, alignment '
                      f'{q.eval_on_selector("#alBox", "e => e.innerText.slice(0, 160)")!r}')
        q.click('#specBtn')
        q.wait_for_timeout(500)
        q.locator('header').screenshot(path=f'{a.shots}/details_{tag}.png')
        q.locator('#expDet summary').click()
        q.wait_for_timeout(300)
        report.append(f'fresh {tag} experiment details: {q.eval_on_selector("#expBox", "e => e.innerText")!r}')
        q.locator('#expDet').screenshot(path=f'{a.shots}/expdetails_{tag}.png')
        ctx.close()
    b.close()
report.append(f'clip packs (v2): {json.dumps({k: v for k, v in VSTAT.items() if k != "bad"})}; not loaded: {VSTAT["bad"][:5]}')
bad = [r for r in report if '(candidates)' in r and "listed []" not in r and ' 0 found' not in r]
report.append(f'candidate lists shown next to discoveries (must be 0): {len(bad)}')
out = '\n'.join(report) + f'\nJS errors: {len(errors)}\n' + '\n'.join('   ' + e for e in errors[:30])
print(out)
Path(a.shots, 'report.txt').write_text(out)
srv.shutdown()
