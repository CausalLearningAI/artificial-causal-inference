"""
Contact sheets for interpreting the neurons of the NES explorer page (one PNG per model x neuron).

For every neuron that can be selected on the page (scripts/eci/build_explorer.py: any outcome, aggregation, prefix
shown (PAGE_PREFIXES), full cohort or subgroup, top-by-p lists; i.e. the neurons that get clips, plan_clips), one
sheet <res>/interp/sheets/<tag>_n<j>.png:
  header   model, neuron, firing rate (share of the kept frames with codes_max > 0), the comparisons where the neuron
           is found (direction of tau, outcome short name and prefix; subgroup hits marked with the subgroup)
  row 1    top 8 frames by codes_max, at most one per video (each video's highest frame, videos ranked by it; the
           page's 'frame' rule, src/eci/viz.py scan_windows / pick_top_windows, kept videos only), raw, with the
           video label under each (mice: stage, genotype, pool; ants: experiment, treatment, recording day, video)
  row 2    the same frames with the neuron's per-patch SAE codes as heat (recomputed through DINOv2 + SAE,
           src/eci/viz.py make_patch_encoder; scale = 99th percentile of the positive patch codes of these 8 frames)
  row 3    8 least-activated frames (pick_least_windows: a random frame where codes_max = 0 in 8 random videos, seed 0;
           'lowest' when fewer videos have one), raw, labelled
  row 4    arena map of the neuron (mean SAE code per patch position over the SAE's training frames, the explorer's
           arena.npz) on the arena background, and per occupancy (divided by how often the patch is foreground) for
           mask SAEs
Aligned SAEs (fg448al, ff448al): frames are turned so the odor corner is top right, as the page does.
Index: <res>/interp/sheets/index.json [{model, sae, neuron, sheet, firing_rate, found_in, top, least}].

Needs the explorer caches of each result set (<res>/_cache/explorer/arena.npz), the build's arena backgrounds
(--build) and a GPU.
Usage: python scripts/eci/interp_sheets.py [--res <res> ...] [--overwrite]   (sbatch: scripts/eci/interp_sheets.sh)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.eci.build_explorer as B  # noqa: E402

N_TILES = 8
TILE = {'mice': 256, 'ants': 320, 'frogs': 320}
TRANS = {'1to2': 'Social H→O', '2to3': 'Social O→P', '4to5': 'Fear H→O', '5to6': 'Fear O→P'}


def comparison_label(cfg, aid):
    an = cfg.dom.analysis(aid)
    if cfg.view.get('arm'):  # frogs: mutant group minus WT
        return f'{an.meta["treatment"]}−{an.meta["control"]}'
    if cfg.dom is not B.MICE:
        return f'{an.meta["experiment"]} t={an.meta["treatment"]} vs t={an.meta["control"]}'
    if an.family == 'A':
        return f'A {an.genotype} {TRANS[aid.split("_")[2]]}'
    return f'B stage {an.where["stage"]} het−wt'


def found_in(cfg):
    """{neuron: [(comparison label, ▲/▼, 'outcome p<prefix>[ subgroup]'), ...]} over the page's primary searches."""
    out = {}
    sets = [(None, cfg.outcomes + cfg.extras)] + [(name, outs) for name, outs in cfg.subsets.items()] + \
           [(name, outs) for name, outs in cfg.xsubsets.items()]
    for name, outs in sets:
        for o in outs:
            p = B.primary_rows(o).dropna(subset=['neuron'])
            for aid, j, tau, pre in zip(p['analysis_id'], p['neuron'].astype(int), p['tau'], p['prefix']):
                out.setdefault(int(j), []).append((comparison_label(cfg, aid), '▲' if tau > 0 else '▼',
                                                   f'{o["short"]} p{int(pre)}' + (f' {name}' if name else '')))
    return out


def found_text(hits):
    """'B stage 3 het−wt ▼ [bouts p128, max-pool p256]; ...' grouped by comparison and direction."""
    g = {}
    for lab, d, what in hits:
        g.setdefault((lab, d), []).append(what)
    return [f'{lab} {d} [{", ".join(dict.fromkeys(w))}]' for (lab, d), w in g.items()]


def video_label(cfg, meta, r):
    m = meta.iloc[int(r)]
    if cfg.dom is B.MICE:
        return f'S{int(m["stage"])} {B.STAGE_LABEL[int(m["stage"])]} · {m["genotype"]} · {m["pool"]}'
    if cfg.view.get('arm'):  # frogs: group, recording session, video
        return f'{m[cfg.view["arm"]]} · session {m[cfg.view["day"]]} · {m["observation_id"]}'
    return f'{m["experiment"]} t={int(m["T"])} · day {B.day_label(str(m["recording_date"]))} · {m["observation_id"]}'


def frame_img(cfg, fp, rot, r):
    im = Image.open(B.DATASET / fp[int(r)]).convert('RGB')
    if rot is not None:
        from src.eci.foreground import rotate_image
        im = rotate_image(im, int(rot[int(r)]))
    return im


def labelled(im, text, T, lab_h):
    t = Image.new('RGB', (T, T + lab_h), (255, 255, 255))
    t.paste(im.resize((T, T), Image.LANCZOS), (0, 0))
    ImageDraw.Draw(t).text((3, T + 2), text[:48], font=B.font(max(11, T // 22)), fill=(20, 20, 20))
    return t


def arena_tile(act, bg, T, title):
    import matplotlib
    matplotlib.use('Agg')
    cmap = matplotlib.colormaps['turbo']
    g = int(round(len(act) ** 0.5))
    m = np.asarray(act, np.float64).reshape(g, g)
    m = m / m.max() if m.max() > 0 else m
    h = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((T, T), Image.NEAREST)) / 255.0
    base = np.asarray(bg.resize((T, T)).convert('RGB'), np.float64) if bg is not None else np.full((T, T, 3), 128.0)
    im = Image.fromarray((0.45 * base + 0.55 * cmap(h)[..., :3] * 255).astype(np.uint8))
    return labelled(im, title, T, 22)


def sheet(cfg, j, fp, rot, meta, top, least, heat, hvmax, fr, hits, arena, T):
    lab_h = 22
    from src.eci.viz import _overlay_rgb, frame_box
    import matplotlib
    cmap = matplotlib.colormaps['turbo']
    raw = [labelled(frame_img(cfg, fp, rot, r), video_label(cfg, meta, r), T, lab_h) for r in top]
    hot = []
    for r, m in zip(top, heat):
        im = frame_img(cfg, fp, rot, r)
        a = np.asarray(im)
        hot.append(labelled(Image.fromarray(_overlay_rgb(a, m, frame_box('fg448', a.shape[1], a.shape[0]), hvmax, cmap)),
                            'heat: where it fires', T, lab_h))
    lst = [labelled(frame_img(cfg, fp, rot, r), video_label(cfg, meta, r), T, lab_h) for r in least]
    act, occ, bg = arena
    ar = [arena_tile(act, bg, T, 'arena: mean code per patch')]
    if occ is not None and len(occ) and np.ptp(occ) > 1e-6:
        ar.append(arena_tile(np.where(occ > 0, act / np.maximum(occ, 1e-12), 0), bg, T, 'arena: per occupancy'))
    lines = [f'{cfg.tag} ({B.model_of(cfg)["input"]}) · neuron {j} · fires on {fr * 100:.1f}% of frames (codes_max > 0)']
    found = found_text(hits)
    font = B.font(15)
    width = N_TILES * T
    wrapped = []
    cur = 'found in: '
    for f in found:
        if len(cur) + len(f) + 2 > width // 8:
            wrapped.append(cur)
            cur = '   '
        cur += f + ';  '
    wrapped.append(cur)
    lines += wrapped
    head_h = 12 + 22 * len(lines)
    row_h = T + lab_h
    title_h = 20
    H = head_h + 4 * (row_h + title_h)
    S = Image.new('RGB', (width, H), (255, 255, 255))
    d = ImageDraw.Draw(S)
    for k, ln in enumerate(lines):
        d.text((8, 6 + 22 * k), ln, font=B.font(16, bold=k == 0) if k == 0 else font, fill=(10, 10, 10))
    y = head_h
    for title, row in [('most activated: highest frame of the 8 most active videos (codes_max), raw', raw),
                       ('same frames, heat = per-patch SAE codes of this neuron', hot),
                       ('least activated: frames where the neuron is 0 (8 random videos)', lst),
                       ('arena map (training frames)', ar)]:
        d.text((8, y + 2), title, font=B.font(14, bold=True), fill=(60, 60, 60))
        y += title_h
        for k, t in enumerate(row):
            S.paste(t, (k * T, y))
        y += row_h
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', action='append')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--chunk', type=int, default=8, help='neurons per patch-encoder batch')
    ap.add_argument('--build', default=str(ROOT / B.NES / 'matryoshka_btk_1024_k16_fg448al_s0' / 'explorer_v34'),
                    help='explorer build folder (arena backgrounds)')
    a = ap.parse_args()
    from types import SimpleNamespace
    from src.eci.viz import (CodeSource, load_full_codes, make_patch_encoder, pick_least_windows, pick_top_windows,
                             scan_windows, subset_windows)
    res = a.res or B.discover_res(False)
    cfgs = [B.Cfg(r, SimpleNamespace(outcome=None, crf=None), ROOT / 'results/_tmp_interp') for r in res]
    tags = [c.tag for c in cfgs]
    for c in cfgs:
        if tags.count(c.tag) > 1:
            c.tag = f'{c.dom.name}{c.tag}'
    B.plan_clips(cfgs, 0)
    for cfg in cfgs:
        out = cfg.res / 'interp' / 'sheets'
        out.mkdir(parents=True, exist_ok=True)
        neurons = sorted({int(j) for w in cfg.clipplan.values() for j in w})
        todo = [j for j in neurons if a.overwrite or not (out / f'{cfg.tag}_n{j}.png').exists()]
        hits = found_in(cfg)
        print(f'[{cfg.tag}] {len(neurons)} selectable neurons, {len(todo)} sheets to draw -> {out}', flush=True)
        T = TILE[cfg.dom.name]
        src = load_full_codes(cfg.sae, B.DATASET, ROOT / 'data', domain=cfg.vdom)
        meta = src.meta
        fp = meta['frame_path'].values
        rot = cfg.rot
        ar = np.load(cfg.work / 'arena.npz')
        bgf = Path(a.build) / 'assets' / B.bg_name(cfg)  # the explorer's arena background of the SAE
        bg = Image.open(bgf) if bgf.exists() else None
        fkeep = B.keep_mask(cfg.view, meta) if cfg.view['keep'] is not None else np.ones(len(meta), bool)
        pe = make_patch_encoder(cfg.sae, dataset_dir=B.DATASET, domain=cfg.vdom) if todo else None
        index = json.loads((out / 'index.json').read_text()) if (out / 'index.json').exists() and not a.overwrite else []
        index = [e for e in index if e['neuron'] in neurons and e['neuron'] not in todo]
        for c0 in range(0, len(todo), a.chunk):
            js = todo[c0:c0 + a.chunk]
            X = np.empty((len(meta), len(js)), np.float32)
            Z = src.codes['max']
            for s0 in range(0, len(meta), 200_000):
                X[s0:s0 + 200_000] = Z[s0:s0 + 200_000][:, js]
            loc = CodeSource('sel', {'max': X}, meta, B.DATASET, dict(src.info))
            ws = scan_windows(loc, np.arange(len(js)), 'max', (1,), seed=0)[1]
            ws.neurons = np.array(js)
            vk = B.keep_mask(cfg.view, ws.videos) if cfg.view['keep'] is not None else np.ones(len(ws.videos), bool)
            wk = ws if vk.all() else subset_windows(ws, vk)
            picks = []
            for i, j in enumerate(js):
                ts, _ = pick_top_windows(wk, i, N_TILES)
                ls, rule = pick_least_windows(wk, i, N_TILES, seed=0)
                picks.append((list(map(int, ts)), list(map(int, ls)), rule))
            rows = sorted({r for t, _, _ in picks for r in t})
            pc = pe.patch_codes([str(B.DATASET / fp[r]) for r in rows], np.array(js), np.array(rows)) if rows else None
            pos = {r: k for k, r in enumerate(rows)}
            for i, j in enumerate(js):
                ts, ls, rule = picks[i]
                heat = [pc[pos[r], :, :, i] for r in ts]
                hv = np.concatenate([h[h > 0] for h in heat]) if heat else np.zeros(0)
                hvmax = float(np.quantile(hv, 0.99)) if len(hv) else 1.0
                fr = float((X[fkeep, i] > 0).mean())
                S = sheet(cfg, j, fp, rot, meta, ts, ls, heat, hvmax, fr, hits.get(j, []),
                          (ar['act'][:, j], ar['occ'] if len(ar['occ']) else None, bg), T)
                f = out / f'{cfg.tag}_n{j}.png'
                S.save(f, optimize=True)
                index.append({'model': cfg.tag, 'sae': cfg.sae, 'input': B.model_of(cfg)['input'], 'neuron': j,
                              'sheet': str(f), 'firing_rate': round(fr, 5), 'found_in': found_text(hits.get(j, [])),
                              'top': [video_label(cfg, meta, r) for r in ts],
                              'least': [video_label(cfg, meta, r) for r in ls], 'least_rule': rule,
                              'heat_vmax': round(hvmax, 4)})
            print(f'  [{cfg.tag}] {c0 + len(js)}/{len(todo)}', flush=True)
        index.sort(key=lambda e: e['neuron'])
        (out / 'index.json').write_text(json.dumps(index, indent=1, ensure_ascii=False))
        print(f'[{cfg.tag}] sheets: {len(index)} in {out}', flush=True)


if __name__ == '__main__':
    main()
