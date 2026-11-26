"""
ECI v2-temporary atlas DRAFT: the published do-pilot page (atlas/index.html = live version 64, 1792705526-4069) plus ONE
new, clearly temporary section "v2-temporary" (button next to Details / Alignment), v1 content untouched.
The section: v1 vs v2 NES per contrast (results/vision/eci_v2_nes/compare.json, scripts/eci/v2/v2nes_tables.py), the
behaviour neurons' first-round effects, and the top-frame sheet of every listed v2 neuron (scripts/eci/v2/v28_sheets.py
PNGs -> webp). Writes atlas/draft_v2_temporary/{index.html, v2t/*.webp, manifest.json}. Not published.
  python scripts/eci/v2/v2nes_page.py
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

from PIL import Image

REPO = Path('/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference')
SRC = REPO / 'atlas/index.html'
OUT = REPO / 'atlas/draft_v2_temporary'
NES = REPO / 'results/vision/eci_v2_nes'
# v28_sheets.py --name v2nes_<key> output folders (--ver 29 -> eci_v2_9/sheets, --ver 27 -> eci_v2_8/sheets), then the
# existing behaviour-neuron sheets
SHEETS = {'mice': [REPO / 'results/vision/eci_v2_9/sheets/mice/v2nes_PREDk16_s2901', REPO / 'results/vision/eci_v2_9/sheets/mice/PREDk16_s2901'],
          'ants': [REPO / 'results/vision/eci_v2_8/sheets/ants/v2nes_PREDk_s2703', REPO / 'results/vision/eci_v2_8/sheets/ants/PREDk_s2703']}

CSS = """
/* v2-temporary section (draft) */
.v2t{max-width:none;gap:10px}
.v2t .tmp{display:inline-block;font:600 11px var(--mono);letter-spacing:.06em;background:var(--warn-bg);color:var(--warn);border-radius:4px;padding:1px 6px;margin-left:6px;vertical-align:middle}
.v2t .ctl{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;font-size:12.5px;color:var(--muted)}
.v2t .tw{overflow-x:auto}
.v2t table.spec3 td{white-space:normal}
.v2t .nb{border:1px solid var(--line);background:var(--bg);border-radius:4px;padding:0 5px;margin:1px 3px 1px 0;font:500 12px var(--mono);color:var(--accent);cursor:pointer}
.v2t .nb.beh{border-color:var(--accent);background:var(--accent-soft)}
.v2t .nb[aria-pressed="true"]{background:var(--accent);color:var(--accent-ink)}
.v2t .nn{font:500 12px var(--mono);color:var(--ink);margin-right:3px}
.v2t .ef{color:var(--muted);font:12px var(--mono)}
.v2t .no{color:var(--faint)}
.v2t .bh{display:block;font:12px var(--mono);color:var(--muted)}
.v2t .bh b{color:var(--ink);font-weight:500}
.v2t .bh .sel{all:unset;color:var(--pass);font-weight:600}
.v2t figure{margin:0;display:grid;gap:4px}
.v2t figure img{width:100%;max-width:900px;height:auto;border:1px solid var(--line);border-radius:6px;background:var(--bg)}
.v2t figcaption{font-size:12.5px;color:var(--muted)}
.v2t ul{margin:0;padding-left:18px}
"""

BTN = ('<button type="button" class="dbtn" id="v2tBtn" aria-expanded="false" aria-controls="v2tBox" '
       'title="TEMPORARY: NES re-run on the v2 representation (mice PREDk16_s2901, ants PREDk_s2703), v1 settings, side by '
       'side with v1">v2-temporary</button>')
BOX = '<div class="xbox spec v2t" id="v2tBox" hidden></div>'

JS = r"""
(function(){
const V = window.V2T;
const $q = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const g = (x, d=3) => (x===null||x===undefined) ? '–' : (Math.abs(x)>=1000||(Math.abs(x)<1e-3&&x!==0) ? x.toExponential(1) : (+x).toPrecision(d));
const pf = p => p<1e-3 ? p.toExponential(1) : p.toFixed(4);
const st = {dom:'mice', oc:'mean', px:'full', fig:null};
const btn = $q('#v2tBtn'), box = $q('#v2tBox');
btn.addEventListener('click', () => { box.hidden = !box.hidden; btn.setAttribute('aria-expanded', String(!box.hidden)); if(!box.hidden) draw(); });
function seg(id, opts, cur){ return `<span class="seg" id="${id}">` + opts.map(([v,l]) => `<button type="button" data-v="${v}" aria-pressed="${v===cur}">${esc(l)}</button>`).join('') + '</span>'; }
function nbtn(dom, ver, j){
  const beh = (V.beh[dom][ver]||{})[j];
  if(ver!=='v2' || !(V.sheets[dom]||{})[j]) return `<span class="nn" title="${beh?esc(beh):''}">#${j}${beh?' ('+esc(beh)+')':''}</span>`;
  return `<button type="button" class="nb${beh?' beh':''}" data-dom="${dom}" data-j="${j}" aria-pressed="${st.fig===dom+'|'+j}" title="show the top-frame sheet">#${j}${beh?' '+esc(beh):''}</button>`;
}
function selCell(dom, ver, x){
  if(!x) return '<span class="no">not run</span>';
  if(!x.selected.length) return `<span class="no">none</span>` + (x.stopped && /no overlap/.test(x.stopped) ? '' : '');
  return x.selected.map(s => `<div>${s.round}. ${nbtn(dom, ver, s.neuron)} <span class="ef">${g(s.tau)} [${g(s.ci[0])}, ${g(s.ci[1])}], p ${pf(s.p)}</span></div>`).join('');
}
function behCell(dom, ver, x){
  if(!x) return '<span class="no">–</span>';
  const b = Object.entries(x.behaviour||{});
  if(!b.length) return '<span class="no">not in this block</span>';
  return b.map(([name, v]) => v.tested ? `<span class="bh">${nbtn(dom, ver, v.neuron)} <b>${g(v.tau)}</b> [${g(v.ci[0])}, ${g(v.ci[1])}], p ${pf(v.p)} · rank ${v.rank_by_abs_tau_round1}/${x.n_tested}${v.selected?' · <span class="sel">selected</span>':''}</span>`
                                     : `<span class="bh">#${v.neuron} ${esc(name)}: not tested</span>`).join('');
}
function draw(){
  const D = V.data[st.dom], ocs = V.outcomes.filter(o => Object.values(D).some(a => a[o[0]]));
  if(!ocs.some(o => o[0]===st.oc)) st.oc = 'mean';
  const pv1 = st.px==='full' ? V.full.v1 : '256', pv2 = st.px==='full' ? V.full.v2 : '256';
  let h = `<h2>v2-temporary <span class="tmp">TEMPORARY · DRAFT</span></h2>`;
  h += `<p class="warnbox one">${V.banner}</p>`;
  h += `<div class="ctl"><span class="opt">Domain ${seg('v2tDom', [['mice','mice v1'],['ants','ants (30-min layout)']], st.dom)}</span>`
     + `<span class="opt">Temporal aggregation ${seg('v2tOc', ocs.map(o => [o[0], o[1]]), st.oc)}</span>`
     + `<span class="opt">Neurons ${seg('v2tPx', [['256','first 256'],['full','full dictionary']], st.px)}</span></div>`;
  h += `<p class="sub">${esc(V.dict[st.dom])}</p>`;
  h += `<div class="tw"><table class="spec3"><thead><tr><th>Contrast</th><th>n</th><th>v1 selected (${pv1} neurons)</th><th>v2 selected (${pv2} neurons)</th><th>v2 behaviour neurons, round-1 effect</th><th>v1 behaviour neurons, round-1 effect</th></tr></thead><tbody>`;
  for(const aid of V.order[st.dom]){
    const a = (D[aid]||{})[st.oc]; if(!a) continue;
    const x1 = (a.v1||{})[pv1], x2 = (a.v2||{})[pv2];
    const n = (x2||x1||{}).n_units ?? '';
    h += `<tr><td class="rl">${esc(V.label[aid]||aid)}<br><span class="ef">${esc(aid)}</span></td><td>${n}</td><td>${selCell(st.dom,'v1',x1)}</td><td>${selCell(st.dom,'v2',x2)}</td><td>${behCell(st.dom,'v2',x2)}</td><td>${behCell(st.dom,'v1',x1)}</td></tr>`;
  }
  h += '</tbody></table></div>';
  h += `<p class="sub">${V.legend}</p>`;
  if(st.fig){
    const [d, j] = st.fig.split('|'), s = (V.sheets[d]||{})[j];
    if(s) h += `<figure><img src="${s.src}" alt="Top-frame sheet of v2 ${d} neuron ${j}" loading="lazy"><figcaption><b>${d} v2 neuron #${j}</b>${s.label?': '+esc(s.label):''}. 12 top held-out annotated frames by raw frame code (≤ 2 per video, ≥ 2 s apart); heat = per-token code, cyan box = max token.</figcaption></figure>`;
  }
  if(V.notes[st.dom] && V.notes[st.dom].length) h += '<h3>Reading</h3><ul>' + V.notes[st.dom].map(t => `<li>${t}</li>`).join('') + '</ul>';
  box.innerHTML = h;
  box.querySelectorAll('#v2tDom button').forEach(b => b.onclick = () => { st.dom = b.dataset.v; st.fig = null; draw(); });
  box.querySelectorAll('#v2tOc button').forEach(b => b.onclick = () => { st.oc = b.dataset.v; draw(); });
  box.querySelectorAll('#v2tPx button').forEach(b => b.onclick = () => { st.px = b.dataset.v; draw(); });
  box.querySelectorAll('button.nb').forEach(b => b.onclick = () => { const k = b.dataset.dom+'|'+b.dataset.j; st.fig = st.fig===k ? null : k; draw(); });
}
})();
"""


def webp(png, dst):
    im = Image.open(png).convert('RGB')
    if im.width > 1000:
        im = im.resize((1000, round(im.height * 1000 / im.width)), Image.LANCZOS)
    im.save(dst, 'WEBP', quality=80, method=6)


def main():
    C = json.loads((NES / 'compare.json').read_text())
    names = json.loads((NES / 'neuron_names.json').read_text()) if (NES / 'neuron_names.json').exists() else {}
    notes = json.loads((NES / 'page_notes.json').read_text()) if (NES / 'page_notes.json').exists() else {}
    import importlib.util
    spec = importlib.util.spec_from_file_location('t', REPO / 'scripts/eci/v2/v2nes_tables.py')
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'v2t').mkdir(exist_ok=True)
    sheets = {}
    for dom in ('mice', 'ants'):
        sheets[dom] = {}
        want = set()
        for aid, d in C.get(dom, {}).items():
            for oc, vv in d.items():
                for P, x in vv.get('v2', {}).items():
                    want |= {s['neuron'] for s in x['selected']}
                    want |= {b['neuron'] for b in x['behaviour'].values()}
        for j in sorted(want):
            src = next((s / f'n{j}.png' for s in SHEETS[dom] if (s / f'n{j}.png').exists()), None)
            if src is None:
                continue
            rel = f'v2t/{dom}_n{j}.webp'
            webp(src, OUT / rel)
            sheets[dom][str(j)] = {'src': rel, 'label': names.get(dom, {}).get(str(j), ''), 'png': str(src.relative_to(REPO))}
    data = {
        'data': {d: C.get(d, {}) for d in ('mice', 'ants')},
        'order': T.ANALYSES, 'label': T.LABEL,
        'outcomes': [[k, {'mean': 'average time (mean activation)', 'bouts': 'event rate (bouts/min)', 'latency': 'latency (s)'}[k]]
                     for k in T.OUTCOMES],
        'full': {'v1': '1024', 'v2': '4096'},
        'beh': {d: {v: {str(j): n for j, n in m.items()} for v, m in T.BEH[d].items()} for d in T.BEH},
        'sheets': sheets, 'notes': notes,
        'dict': {'mice': 'v1 = matryoshka_btk_1024_k16_fg448al_s0 (DINOv2-base 448 px, 1024 latents). v2 = PREDk16_s2901 '
                         '(ECI v2.9, PRED-contact SAE, k = 16, 4096 latents; nose-nose #3461, nose-tail #1780), frame code = max '
                         'over the frame\'s mask tokens; all 432 videos, 2,592,000 frames.',
                 'ants': 'v1 = matryoshka_btk_1024_k16_antsfg_s0 (1024 latents). v2 = PREDk_s2703 (ECI v2.7, PRED-contact SAE, '
                         'k = 16, 4096 latents; grooming-to-focal #33), frame code = max over the frame\'s mask tokens; all 232 '
                         'released videos (v2 10 min, v3 / v4 / v4_infested 30 min), 1,824,000 frames.'},
        'banner': ('TEMPORARY section for review (built 2026-11-26, not yet approved). It re-runs the v1 NES search, with the '
                   'v1 settings (frame max, per-video summary, t-test, Bonferroni 0.05, no nuisance covariate, selection by '
                   '|effect|), on the v2 representation; only the representation changes. No clips: each v2 neuron links to '
                   'its top-frame sheet. Effects are in each dictionary\'s own code units, so v1 and v2 effect sizes are not '
                   'comparable; compare which neurons are found and their p-values.'),
        'legend': ('Selected: neurons in NES round order, effect (treatment minus control, or later minus earlier stage) '
                   '[95% CI], p. Behaviour neurons: their round-1 effect, p and rank by |effect| among all tested neurons, '
                   'whether or not NES selected them; the round-1 Bonferroni threshold is 0.05 / number tested. Mice paired '
                   'transitions usually stop after round 1 because the selected neuron changes in nearly every pool (no '
                   'overlap to condition on).'),
    }
    html = SRC.read_text()
    assert html.count('aria-controls="alBox"') == 1 and html.count('<div class="xbox spec" id="alBox" hidden></div>') == 1
    i = html.find('Alignment</button>') + len('Alignment</button>')
    html = html[:i] + BTN + html[i:]
    html = html.replace('<div class="xbox spec" id="alBox" hidden></div>', '<div class="xbox spec" id="alBox" hidden></div>\n    ' + BOX, 1)
    html = html.replace('</style>', CSS + '</style>', 1)
    html = html.rstrip() + '\n<script>\nwindow.V2T = ' + json.dumps(data, default=float, separators=(',', ':')) + ';\n' + JS + '</script>\n'
    (OUT / 'index.html').write_text(html)
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    man = {'status': 'DRAFT, not published', 'base_page': 'atlas/index.html = live version 1792705526-4069 (version 64)',
           'base_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(), 'artifact': 'https://claude.ai/artifact/JfJnP4CqjYWoprEq9dM8Qa',
           'build_command': 'python scripts/eci/v2/v2nes_page.py', 'code_commit_at_build': head,
           'inputs': ['results/vision/eci_v2_nes/compare.json', 'results/vision/eci_v2_nes/neuron_names.json'],
           'files': {'index.html': (OUT / 'index.html').stat().st_size,
                     **{v['src']: (OUT / v['src']).stat().st_size for d in sheets.values() for v in d.values()}},
           'publish_plan': 'Artifact publish url=JfJnP4CqjYWoprEq9dM8Qa file_path=atlas/draft_v2_temporary/index.html with '
                           'files {v2t/*.webp}; then copy into atlas/ and update atlas/manifest.json',
           'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
    (OUT / 'manifest.json').write_text(json.dumps(man, indent=1))
    print('wrote', OUT, 'sheets', {d: len(v) for d, v in sheets.items()}, 'page bytes', man['files']['index.html'])


if __name__ == '__main__':
    main()
