"""
Build the NES explorer page ("Mouse Concept Atlas"): <out>/index.html + <out>/assets/.

Data-driven: everything comes from one or more NES result sets (each a directory with summary.csv,
as written by the NES runs). The first --outcome is the page default; more outcomes add an
'Outcome' selector. Neurons discovered in ANY outcome get clips, ranked by the codes the outcome is
built from (codes_mean for mean-pool outcomes, codes_max for max-pool outcomes).

Steps (each cached under <res>/_cache/explorer/, incremental: only missing neurons are recomputed):
    data     one pass over the full per-frame codes (src/eci/viz.py scan_codes) for every neuron
             discovered by any outcome's primary setting (prefix 128 / 1024, window full / trim30).
             Per neuron: top 6 frames (gallery rule pick_top, 1 per video), 6 of the gallery's 12
             least-activated frames, the stage x genotype table; per family-A hit of a mean-pool
             mean outcome: the 3 pools that drive the effect. -> picks.json
    patch    per-patch codes of the neuron for every frame of its 6 top clips, recomputed through
             DINOv2 + SAE (src/eci/viz.py PatchEncoder, GPU), checked against the stored pooled
             codes. -> patch/nXXXX.npy (90, 16, 16) float16
    render   3 s clips (15 frames at 5 fps) tiled into H.264 mp4 montages: top raw, top with the
             heatmap overlaid (turbo, on the 224 center crop = pixels 32-480 of a 512 frame),
             least raw, family-A driving pools (stage a | stage b)
    page     data.json inlined into scripts/eci/explorer_template.html -> <out>/index.html; per rendered neuron
             also its AUROC for the human labels Y_nn / Y_np / Y_nt on the held-out annotated frames
             (--eval-codes rows, as scripts/eci/eval_sae_fg.py; evaluation only, never used by the search)
    check    every referenced asset exists; unreferenced files in assets/ are deleted; counts, MB
             (fails above --max-files files / --max-mb MB)

Representations (src/eci/viz.py representation): 'crop224' SAEs are patch SAEs on the 224 center
crop (16 x 16 patches, heat on pixels 32-480); 'fg448' SAEs (foreground, src/eci/foreground.py) see
the whole frame at 448 (32 x 32 patches of 16 px) and fire only on foreground patches (others = 0).
Chips: one per single-setting change from the outcome's primary (summary.csv columns; bout_rule 1 =
min-2-frame hysteresis bouts 'min2+hyst'), plus 'size' when <outcome dir>/size_adjusted.csv exists
(round-1 neuron re-tested with the per-video mean foreground size as a covariate).
Artefact flags (neurons 50, 64, 113) belong to the ep20 SAE only.

Usage:
    python scripts/eci/build_explorer.py                         # defaults below, all steps
    python scripts/eci/build_explorer.py --res results/vision/mice/eci/nes/<sae> \
        --outcome 'Bout rate (max-pool)=maxpool_bouts:pooling=max,outcome_type=bout_rate,threshold_q=0.95,merge_gap=0,unit=bouts/min,short=bouts' \
        --outcome 'Mean activation (mean-pool)=.:short=mean-pool'
    --outcome LABEL=SUBDIR[:key=value,...]   SUBDIR relative to --res. Keys: pooling, outcome_type,
    test (default t), correction (default bonferroni), any extra summary.csv setting column
    (threshold_q, merge_gap, ...; required when it has several values), codes (mean|max, default
    = pooling), unit (tau unit), short (chip label). pooling / outcome_type default to mean/mean,
    else the only combination tested with t + bonferroni.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TEMPLATE = ROOT / 'scripts/eci/explorer_template.html'
DATASET = ROOT / 'dataset'
ARTEFACT_EP20 = {64, 50, 113}  # ep20 SAE neuron ids; other SAEs get no flags
ARTEFACT = set()
REP = 'crop224'  # set from the SAE in Cfg (src/eci/viz.py representation)
EVENTS = {'Y_nn': 'nose–nose (mutual)', 'Y_np': 'nose–nose (one-sided)', 'Y_nt': 'nose–tail'}
K, HALF, TILE = 6, 7, 200           # clips per montage, frames each side of the center, tile px
STAGE_LABEL = {1: 'H,S', 2: 'O,S', 3: 'P,S', 4: 'H,F', 5: 'O,F', 6: 'P,F'}
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FIELDS = ('pooling', 'outcome_type', 'test', 'correction')
KNOWN = set(FIELDS) | {'analysis_id', 'family', 'genotype', 'stage', 'transition', 'prefix', 'window', 'n_units',
                        'setting', 'n_tested_total', 'n_dropped', 'round', 'neuron', 'tau', 'se', 't', 'df', 'p',
                        'threshold', 'n_tested', 'direction'}
# robustness chips: (field, alternative value) -> short label, tooltip
CHIP = {('test', 'signflip'): ('flip', 'sign-flip permutation test instead of the t-test'),
        ('correction', 'bh'): ('BH', 'Benjamini-Hochberg instead of Bonferroni'),
        ('pooling', 'max'): ('max', 'max-pooling over patches instead of mean'),
        ('pooling', 'mean'): ('mean-pool', 'mean-pooling over patches instead of max'),
        ('outcome_type', 'rate'): ('rate', 'firing rate (fraction of frames > 0) instead of mean activation'),
        ('outcome_type', 'mean'): ('mean', 'per-video mean activation as the outcome'),
        ('window', 'matched'): ('match', 'time-matched windows across stages'),
        ('window', 'trim30'): ('−30s', 'first 30 s of each video dropped'),
        ('window', 'full'): ('full', 'full videos'),
        ('prefix', 128): ('128', 'searching only the first 128 neurons'),
        ('prefix', 1024): ('1024', 'searching all 1024 neurons')}
DEFAULT_OUTCOMES = ['Bout rate (max-pool)=maxpool_bouts:pooling=max,outcome_type=bout_rate,threshold_q=0.95,merge_gap=0,unit=bouts/min,short=bouts',
                    'Mean activation (mean-pool)=.:short=mean-pool']


class Cfg:
    def __init__(self, a):
        self.res = (ROOT / a.res).resolve() if not Path(a.res).is_absolute() else Path(a.res)
        self.sae = a.sae or self.res.name
        self.out = Path(a.out) if a.out else self.res / 'explorer'
        self.assets = self.out / 'assets'
        self.work = self.res / '_cache' / 'explorer'
        self.outcomes = []
        specs = a.outcome or [x for x in DEFAULT_OUTCOMES if (self.res / x.split('=', 1)[1].split(':')[0]
                                                              / 'summary.csv').exists()]
        for spec in specs:
            label, rest = spec.split('=', 1)
            sub, _, opts = rest.partition(':')
            d = (self.res / sub).resolve()
            if not (d / 'summary.csv').exists():
                raise SystemExit(f'{d}/summary.csv missing (outcome {label!r})')
            tidy = pd.read_csv(d / 'summary.csv')
            kv = dict(x.split('=', 1) for x in opts.split(',') if x)
            meta = {k: kv.pop(k) for k in ('codes', 'unit', 'short') if k in kv}
            prim = default_primary(tidy, kv)
            codes = meta.get('codes', prim['pooling'])
            self.outcomes.append({'label': label, 'dir': d, 'tidy': tidy, 'primary': prim, 'codes': codes,
                                  'unit': meta.get('unit', 'activation' if prim['outcome_type'] == 'mean'
                                                   else prim['outcome_type']),
                                  'short': meta.get('short', label.split()[0].lower()),
                                  'id': re.sub(r'\W+', '', label.lower())[:16] or 'o'})
            print(f'outcome {label!r}: {d} primary {prim} codes_{codes}')
        self.keys = sorted({o['codes'] for o in self.outcomes})
        global ARTEFACT, REP
        from src.eci.viz import representation
        REP = representation(self.sae, DATASET)
        ARTEFACT = ARTEFACT_EP20 if self.sae == 'matryoshka_btk_1024_k16_ep20_s0' else set()
        self.eval_codes = Path(a.eval_codes)
        self.max_files, self.max_mb = a.max_files, a.max_mb
        print(f'SAE {self.sae}: representation {REP}, artefact flags {sorted(ARTEFACT)}')


def extra_cols(tidy):
    return [c for c in tidy.columns if c not in KNOWN]


def default_primary(tidy, kv):
    prim = {'test': kv.pop('test', 't'), 'correction': kv.pop('correction', 'bonferroni')}
    if 'pooling' not in kv or 'outcome_type' not in kv:
        g = tidy[(tidy['test'] == prim['test']) & (tidy['correction'] == prim['correction'])]
        combos = sorted(set(map(tuple, g[['pooling', 'outcome_type']].astype(str).values)))
        if ('mean', 'mean') in combos:
            kv.setdefault('pooling', 'mean'), kv.setdefault('outcome_type', 'mean')
        elif len(combos) == 1:
            kv.setdefault('pooling', combos[0][0]), kv.setdefault('outcome_type', combos[0][1])
        else:
            raise SystemExit(f'ambiguous primary: pass pooling=..,outcome_type=..; options {combos}')
    prim.update(pooling=kv.pop('pooling'), outcome_type=kv.pop('outcome_type'))
    sub = tidy[match(tidy, prim)]
    for c in extra_cols(tidy):
        vals = sorted(sub[c].dropna().unique())
        if c in kv:
            prim[c] = float(kv.pop(c))
        elif len(vals) > 1 and c == 'bout_rule':
            prim[c] = 0.0  # plain threshold runs; 1 = the min2+hyst sensitivity
        elif len(vals) > 1:
            raise SystemExit(f'setting column {c!r} has values {vals}: pass {c}=<primary value>')
        elif vals:
            prim[c] = float(vals[0])
    if kv:
        raise SystemExit(f'unknown outcome options {kv}')
    return prim


def match(t, spec):
    """Rows matching spec; NaN in an extra setting column means 'not applicable' and matches."""
    m = np.ones(len(t), bool)
    for k, v in spec.items():
        if k == 'prefix':
            m &= (t[k] == int(v)).values
        elif k in KNOWN:
            m &= (t[k].astype(str) == str(v)).values
        else:
            col = t[k].astype(float)
            m &= (np.isclose(col, float(v)) | col.isna()).values
    return m


def primary_rows(o):
    t = o['tidy']
    return t[match(t, o['primary']) & t['window'].isin(['full', 'trim30']).values]


def mean_outcome(o):
    return (o['primary']['pooling'], o['primary']['outcome_type']) == ('mean', 'mean')


def is_mean(o):
    """per-video mean activation outcome (either pooling): driving pools from the scan's video means."""
    return o['primary']['outcome_type'] == 'mean'


def is_bout(o):
    return o['primary']['outcome_type'] == 'bout_rate'


def ckey(o, aid, j):
    return f'{aid}__{j}' if mean_outcome(o) else f'{o["id"]}_{aid}__{j}'


_rates = {}


def bout_rates(cfg, o, window, fps=5.0):
    """Per-video bouts/min of every neuron (src/eci/contrasts.py bout_outcomes, same cache as
    scripts/eci/run_nes_bouts.py) -> (Index of observation_id, (n_obs, m) array)."""
    q, g = o['primary']['threshold_q'], int(o['primary']['merge_gap'])
    w = {'full': 'full', 'trim30': 'trim', 'matched': 'last'}[window]
    k = (w, q, g)
    if k not in _rates:
        f = np.load(cfg.res / '_cache' / 'bout_summaries_max.npz')
        c, nf = f[f'{w}__{q:g}__{g}__count'], f[f'{w}__n_frames']
        _rates[k] = (pd.Index(f['observation_id'].astype(str)), c / (nf[:, None] / fps / 60.0))
    return _rates[k]


def rate_table(cfg, o, window, j):
    """stage x genotype: mean over pools of the per-video bouts/min, 95% t-CI (as viz.stage_genotype_table)."""
    from scipy import stats
    from src.eci.viz import video_meta
    if 'vm' not in _rates:
        _rates['vm'] = video_meta(ROOT / 'data')
    vm = _rates['vm'].copy()
    ids, rate = bout_rates(cfg, o, window)
    vm['y'] = rate[ids.get_indexer(vm['observation_id'].astype(str)), j]
    rows = []
    for (st, g), grp in vm.dropna(subset=['y']).groupby(['stage', 'genotype']):
        y = grp.groupby('pool')['y'].mean().values
        h = stats.t.ppf(0.975, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0.0
        rows.append({'stage': int(st), 'genotype': g, 'mean': round(float(y.mean()), 4),
                     'lo': round(float(y.mean() - h), 4), 'hi': round(float(y.mean() + h), 4), 'n_pools': int(len(y))})
    return rows


def pre(key):
    """asset / cache file prefix per ranking codes (mean keeps the historical 'n')."""
    return 'n' if key == 'mean' else 'x'


def short_obs(obs):
    """2024-07-26_14-29-30_BHVScreen_rd11_2_SocialOdor_Test -> ('rd11_2 Test', '07-26 14:29')."""
    m = re.match(r'\d{4}-(\d\d-\d\d)_(\d\d)-(\d\d)-\d\d_BHVScreen_(rd[\w]+?)_(\w+?)Odor_(\w+)$', obs)
    if not m:
        return obs[:24], ''
    return f'{m[4]} {m[6]}', f'{m[1]} {m[2]}:{m[3]}'


def stages_of(aid):
    return [int(x) for x in re.match(r'A_\w+_(\d)to(\d)', aid).groups()]


# ---------------------------------------------------------------------- step: data
def step_data(cfg, overwrite):
    out = cfg.work / 'picks.json'
    res = json.loads(out.read_text()) if out.exists() and not overwrite else {}
    if 'neurons' in res and 'by_key' not in res:  # cache from the codes_mean-only builder
        res['by_key'] = {'mean': res.pop('neurons')}
    res.setdefault('by_key', {})
    res.setdefault('contrast', {})
    res.setdefault('frame_path', {})
    want = {k: set() for k in cfg.keys}
    a_hits = {}  # family-A contrast key -> (analysis, neuron, tau, genotype, codes, outcome)
    for o in cfg.outcomes:
        p = primary_rows(o).dropna(subset=['neuron'])
        want[o['codes']] |= {int(j) for j in p['neuron']}
        if is_mean(o) or is_bout(o):
            for (aid, j), g in p[p['family'] == 'A'].groupby(['analysis_id', 'neuron']):
                a_hits.setdefault(ckey(o, aid, int(j)), (aid, int(j), float(g.sort_values(['window', 'prefix'])['tau'].iloc[0]),
                                                         g['genotype'].iloc[0], o['codes'], o))
    missing = {k: sorted(j for j in v if str(j) not in res['by_key'].get(k, {})) for k, v in want.items()}
    missing_c = [k for k in a_hits if k not in res['contrast']]
    for k in missing_c:
        missing.setdefault(a_hits[k][4], [])
    if not any(missing.values()) and not missing_c:
        print('data: cached', {k: len(v) for k, v in res['by_key'].items()}, 'neurons,', len(res['contrast']),
              'contrast montages')
        return
    from src.eci.viz import load_full_codes, pick_least, pick_top, scan_codes, stage_genotype_table
    src = load_full_codes(cfg.sae, DATASET, ROOT / 'data')
    meta = src.meta
    fp = meta['frame_path'].values
    for key, miss in missing.items():
        need_j = sorted(set(miss) | {a_hits[k][1] for k in missing_c if a_hits[k][4] == key})
        if not need_j:
            continue
        print(f'data: scanning codes_{key} for neurons', need_j, flush=True)
        neurons = np.array(need_j, dtype=np.int64)
        scan = scan_codes(src, neurons, key, 1, 2.0, seed=0)
        vids, Z = scan.videos, src.codes[key]
        row2vid = np.searchsorted(vids['lo'].values, np.arange(len(meta)), side='right') - 1

        def clip(r, j):
            v = row2vid[r]
            lo, hi = int(vids['lo'].iloc[v]), int(vids['hi'].iloc[v])
            a = min(max(lo, r - HALF), hi - (2 * HALF + 1))
            rows = list(range(a, a + 2 * HALF + 1))
            m = meta.iloc[int(r)]
            return {'center': int(r), 'rows': rows, 'act': float(Z[r, j]),
                    'trace': [float(x) for x in np.asarray(Z[rows[0]:rows[-1] + 1, j], dtype=np.float32)],
                    'obs': m['observation_id'], 'frame': int(m['frame_idx']), 'stage': int(m['stage']),
                    'genotype': m['genotype'], 'pool': str(m['pool'])}

        store = res['by_key'].setdefault(key, {})
        for j in miss:
            i = int(np.flatnonzero(neurons == j)[0])
            tr, _ = pick_top(scan, i, 12)
            lr, rule = pick_least(scan, i, 12, 0)
            tab = stage_genotype_table(scan, i)
            store[str(j)] = {
                'top': [clip(int(r), j) for r in tr[:K]],
                'least': [clip(int(r), j) for r in lr[::2][:K]],
                'least_rule': rule,
                'firing_rate': float(scan.n_active[i] / scan.n_frames),
                'table': [{k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if k in ('stage', 'n_pools')
                               else v) for k, v in rec.items()} for rec in tab.to_dict('records')]}
        # family A: the 3 pools with the largest paired change of the tested outcome (full window) in the
        # direction of tau; clips = the most activated moment (codes_<key>) of each of those videos
        for ck in [k for k in missing_c if a_hits[k][4] == key]:
            aid, j, tau, geno, _, o = a_hits[ck]
            i = int(np.flatnonzero(neurons == j)[0])
            sa, sb = stages_of(aid)
            df = vids[['observation_id', 'pool', 'stage', 'genotype']].copy()
            if is_bout(o):
                ids, rate = bout_rates(cfg, o, 'full')
                df['y'] = rate[ids.get_indexer(df['observation_id'].astype(str)), j]
            else:
                df['y'] = scan.video_mean[:, i]
            df['top_row'] = scan.top_row[:, 0, i]
            df = df[df['genotype'] == geno]
            pa = df[df['stage'] == sa].set_index('pool')
            pb = df[df['stage'] == sb].set_index('pool')
            d = (pb['y'] - pa['y']).dropna()
            pools = (d * np.sign(tau)).sort_values(ascending=False).index[:3]
            rows = [{'pool': str(p), 'delta': float(d[p]),
                     'a': dict(clip(int(pa.loc[p, 'top_row']), j), vmean=float(pa.loc[p, 'y'])),
                     'b': dict(clip(int(pb.loc[p, 'top_row']), j), vmean=float(pb.loc[p, 'y']))} for p in pools]
            res['contrast'][ck] = {'analysis_id': aid, 'neuron': j, 'stages': [sa, sb], 'tau': tau, 'pools': rows}
    need = {r for st in res['by_key'].values() for n in st.values() for c in n['top'] + n['least'] for r in c['rows']}
    need |= {r for c in res['contrast'].values() for p in c['pools'] for s in 'ab' for r in p[s]['rows']}
    res['frame_path'].update({str(r): fp[r] for r in sorted(need) if str(r) not in res['frame_path']})
    cfg.work.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print('data: wrote', out, {k: len(v) for k, v in res['by_key'].items()}, 'neurons,', len(res['contrast']),
          'contrast montages')


# ---------------------------------------------------------------------- step: patch
def step_patch(cfg, overwrite):
    picks = json.loads((cfg.work / 'picks.json').read_text())
    pdir = cfg.work / 'patch'
    pdir.mkdir(parents=True, exist_ok=True)
    todo = [(k, int(j)) for k in cfg.keys for j in picks['by_key'].get(k, {})
            if overwrite or not (pdir / f'{pre(k)}{int(j):04d}.npy').exists()]
    if not todo:
        print('patch: cached')
        return
    from src.eci.viz import load_full_codes, make_patch_encoder
    rows = {(k, j): [r for c in picks['by_key'][k][str(j)]['top'] for r in c['rows']] for k, j in todo}
    allr = sorted({r for v in rows.values() for r in v})
    js = sorted({j for _, j in todo})
    print(f'patch: {len(todo)} (codes, neuron) pairs, {len(allr)} frames through DINOv2 + SAE ({REP})', flush=True)
    pe = make_patch_encoder(cfg.sae, dataset_dir=DATASET)
    paths = [str(DATASET / picks['frame_path'][str(r)]) for r in allr]
    if pe.rep == 'fg448':
        pc, fg = pe.patch_codes(paths, np.array(js), np.array(allr), return_mask=True)
        n_fg = fg.reshape(len(allr), -1).sum(1)
    else:
        pc, n_fg = pe.patch_codes(paths, np.array(js)), None
    pos = {r: i for i, r in enumerate(allr)}
    codes = load_full_codes(cfg.sae, DATASET, ROOT / 'data').codes
    worst, mism = {}, {}
    for k, j in todo:
        ii = [pos[r] for r in rows[(k, j)]]
        m = pc[ii, :, :, js.index(j)]
        stored = np.asarray(codes[k][np.array(rows[(k, j)])][:, j], dtype=np.float32)
        if k == 'max':
            pooled = m.max((1, 2))
        elif n_fg is None:
            pooled = m.mean((1, 2))
        else:  # fg448: mean over the foreground patches only
            pooled = m.sum((1, 2)) / np.maximum(n_fg[ii], 1)
        worst[k] = max(worst.get(k, 0.0), float(np.abs(pooled - stored).max()))
        mism[k] = mism.get(k, 0) + int(((pooled > 0) != (stored > 0)).sum())
        np.save(pdir / f'{pre(k)}{j:04d}.npy', m.astype(np.float16))
    print(f'patch: recomputed patch maps pooled vs stored codes, max abs diff {worst}, '
          f'active/inactive mismatches {mism} (of {sum(len(v) for v in rows.values())} frame-neuron pairs)')


# ---------------------------------------------------------------------- step: render
_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT, size)
    return _fonts[k]


def tile_frames(c, frame_path, vmax, color, maps=None, hvmax=None):
    """15 labelled TILE x TILE frames of one clip; a bar at the bottom shows the activation of the
    current frame relative to vmax. maps: (15, 16, 16) patch codes -> heatmap overlaid (scale hvmax)."""
    from src.eci.viz import _overlay_rgb, frame_box
    import matplotlib
    matplotlib.use('Agg')
    cmap = matplotlib.colormaps['turbo']
    name, when = short_obs(c['obs'])
    frames = []
    for t, r in enumerate(c['rows']):
        im = Image.open(DATASET / frame_path[str(r)]).convert('RGB')
        if maps is not None:
            a = np.asarray(im)
            im = Image.fromarray(_overlay_rgb(a, maps[t].astype(np.float32), frame_box(REP, a.shape[1], a.shape[0]),
                                              hvmax, cmap))
        im = im.resize((TILE, TILE), Image.BILINEAR)
        d = ImageDraw.Draw(im, 'RGBA')
        d.rectangle([0, 0, TILE, 17], fill=(0, 0, 0, 140))
        d.text((5, 2), f'S{c["stage"]} {STAGE_LABEL[c["stage"]]} · {c["genotype"]} · {name}', font=font(11),
               fill=(255, 255, 255))
        d.rectangle([0, TILE - 6, TILE, TILE], fill=(0, 0, 0, 140))
        w = int(round(TILE * min(max(c['trace'][t], 0) / vmax, 1))) if vmax > 0 else 0
        if w:
            d.rectangle([0, TILE - 5, w, TILE], fill=color)
        if r == c['center']:
            d.rectangle([0, 0, TILE - 1, TILE - 1], outline=color, width=2)
        frames.append(np.asarray(im))
    return frames


def encode(grid_frames, path, crf=30):
    """grid_frames: list of rows, each a list of clips (each a list of 15 HxWx3 arrays)."""
    n = len(grid_frames[0][0])
    frames = [np.concatenate([np.concatenate([clip[t] for clip in row], 1) for row in grid_frames], 0)
              for t in range(n)]
    h, w = frames[0].shape[:2]
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
           '-r', '5', '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', '-an', str(path)]
    subprocess.run(cmd, input=b''.join(f.tobytes() for f in frames), check=True)


def step_render(cfg, overwrite):
    picks = json.loads((cfg.work / 'picks.json').read_text())
    fp = picks['frame_path']
    cfg.assets.mkdir(parents=True, exist_ok=True)
    top_c, low_c = (255, 170, 40), (120, 190, 255)
    for key in cfg.keys:
        for j, n in picks['by_key'].get(key, {}).items():
            j = int(j)
            vmax = n['top'][0]['act'] if n['top'] else 1.0
            for kind, clips_, color, heat in (('top', n['top'], top_c, False), ('least', n['least'], low_c, False),
                                              ('heat', n['top'], top_c, True)):
                out = cfg.assets / f'{pre(key)}{j:04d}_{kind}.mp4'
                if out.exists() and not overwrite:
                    continue
                maps = hv = None
                if heat:
                    maps = np.load(cfg.work / 'patch' / f'{pre(key)}{j:04d}.npy').astype(np.float32)
                    hv = float(maps.max())
                clips = [tile_frames(c, fp, vmax, color, None if maps is None else maps[k * 15:(k + 1) * 15], hv)
                         for k, c in enumerate(clips_)]
                while len(clips) < K:
                    clips.append([np.zeros((TILE, TILE, 3), np.uint8)] * (2 * HALF + 1))
                encode([clips[:3], clips[3:6]], out)
                print(f'render: codes_{key} neuron {j} {kind}', flush=True)
    for ck, c in picks['contrast'].items():
        out = cfg.assets / f'c_{ck}.mp4'
        if out.exists() and not overwrite:
            continue
        vmax = max(max(p['a']['act'], p['b']['act']) for p in c['pools'])
        grid = [[tile_frames(p['a'], fp, vmax, (200, 200, 200)), tile_frames(p['b'], fp, vmax, top_c)]
                for p in c['pools']]
        encode(grid, out)
        print('render:', ck, flush=True)


# ---------------------------------------------------------------------- step: page
def chip_name(f, v):
    if f == 'threshold_q':
        return f'q{v:.2f}', f'activity threshold at the {v:.2f} quantile'
    if f == 'merge_gap':
        return f'gap{int(v)}', f'bouts separated by <= {int(v)} frames merged'
    if f == 'bout_rule':
        return (('min2+hyst', 'hysteresis bouts: start above the q0.95 threshold, end at or below the q0.90 '
                 'threshold, at least 2 frames long') if int(v) == 1 else ('plain', 'plain threshold runs'))
    return CHIP.get((f, v), (str(v), f'{f} = {v}'))


def robustness(o, others, aid, prefix, window, neuron):
    """Chips for one discovered neuron: is it still selected (any round) when ONE setting changes
    (or, cross-outcome, by another outcome's primary search)? 'Y' / 'N', or '-' when not run."""
    t = o['tidy']
    t = t[t['analysis_id'] == aid]
    base = dict(o['primary'], window=window, prefix=prefix)
    alts = []
    for f in list(FIELDS) + extra_cols(t) + ['window']:
        if f in KNOWN:
            alts += [(f, v) for v in sorted(t[f].astype(str).unique()) if v != str(base.get(f))]
        else:
            alts += [(f, float(v)) for v in sorted(t[f].dropna().unique()) if f in base and not np.isclose(v, base[f])]
    alts += [('prefix', int(p)) for p in sorted(t['prefix'].unique()) if p != prefix]
    chips = []
    for f, v in alts:
        g = t[match(t, dict(base, **{f: v}))]
        if f == 'prefix' and v == 128 and neuron >= 128:
            val = '-'
        else:
            val = ('Y' if neuron in set(g['neuron'].dropna().astype(int)) else 'N') if len(g) else '-'
        chips.append([*chip_name(f, v), val])
    sz = size_table(o)
    if sz is not None:
        g = sz[(sz['analysis_id'] == aid) & (sz['prefix'] == prefix) & (sz['window'] == window) & (sz['neuron'] == neuron)]
        val = ('Y' if bool(g['survives'].iloc[0]) else 'N') if len(g) else '-'
        chips.append(['size-adj', 'round-1 test repeated with the per-video mean foreground size (number of mouse '
                      'patches, a proxy for how spread out or huddled the mice are) as a covariate; "not run" = '
                      'not a round-1 neuron', val])
    for o2 in others:
        g = primary_rows(o2)
        g = g[(g['analysis_id'] == aid) & (g['prefix'] == prefix) & (g['window'] == window)]
        val = ('Y' if neuron in set(g['neuron'].dropna().astype(int)) else 'N') if len(g) else '-'
        chips.append([o2['short'], f'{o2["label"]} search', val])
    return chips


_size = {}


def size_table(o):
    """<outcome dir>/size_adjusted.csv (run_nes*.py size check) or None."""
    k = str(o['dir'])
    if k not in _size:
        f = o['dir'] / 'size_adjusted.csv'
        _size[k] = pd.read_csv(f) if f.exists() else None
    return _size[k]


def label_auroc(cfg, keys_neurons):
    """{key: {neuron: {event: AUROC}}} of the per-frame codes_<key> for the human labels on the
    held-out annotated frames (rows of cfg.eval_codes/task_*.npz, as scripts/eci/eval_sae_fg.py).
    Evaluation only. Empty when the eval rows are missing."""
    from scipy.stats import rankdata
    files = sorted(f for f in cfg.eval_codes.glob('task_*.npz') if '.tmp' not in f.name)
    if not files:
        return {}, 0
    rows = np.sort(np.concatenate([np.load(f)['rows'] for f in files]))
    lab = pd.read_csv(DATASET / 'mice/v1/annotations.csv', usecols=list(EVENTS)).iloc[rows]
    from src.eci.viz import load_full_codes
    codes = load_full_codes(cfg.sae, DATASET, ROOT / 'data').codes
    out = {}
    for key, js in keys_neurons.items():
        js = sorted(js)
        X = np.asarray(codes[key][rows][:, js], dtype=np.float32)
        r = rankdata(X, axis=0)
        out[key] = {}
        for ev in EVENTS:
            y = lab[ev].values > 0
            n1, n0 = int(y.sum()), int((~y).sum())
            au = (r[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
            for j, a in zip(js, au):
                out[key].setdefault(str(j), {})[ev] = round(float(a), 3)
    return out, len(rows)


ORDER = [f'A_{g}_{t}' for g in ('het', 'wt') for t in ('1to2', '2to3', '4to5', '5to6')] + \
        [f'B_stage{s}' for s in range(1, 7)]


def step_page(cfg, overwrite):
    picks = json.loads((cfg.work / 'picks.json').read_text())
    analyses, results, outcomes = {}, {}, []
    for o in cfg.outcomes:
        prim = primary_rows(o)
        others = [x for x in cfg.outcomes if x is not o]
        outcomes.append({'id': o['id'], 'label': o['label'], 'codes': o['codes'], 'unit': o['unit'],
                         'setting': ', '.join(f'{k} {v:g}' if isinstance(v, float) else f'{k} {v}'
                                              for k, v in o['primary'].items())})
        for aid in ORDER:
            g = prim[prim['analysis_id'] == aid]
            if not len(g):
                continue
            r0 = g.iloc[0]
            if aid not in analyses:
                analyses[aid] = ({'id': aid, 'family': 'A', 'genotype': r0['genotype'], 'stages': stages_of(aid),
                                  'n_units': int(r0['n_units'])} if aid.startswith('A') else
                                 {'id': aid, 'family': 'B', 'stage': int(r0['stage']), 'n_units': int(r0['n_units'])})
            for (prefix, window), h in g.groupby(['prefix', 'window']):
                rows = []
                for _, r in h.dropna(subset=['neuron']).sort_values('round').iterrows():
                    j = int(r['neuron'])
                    ck = ckey(o, aid, j)
                    rows.append({'round': int(r['round']), 'neuron': j, 'tau': float(r['tau']), 'p': float(r['p']),
                                 'threshold': float(r['threshold']), 'n_tested': int(r['n_tested']),
                                 'rob': robustness(o, others, aid, int(prefix), window, j),
                                 'contrast': f'assets/c_{ck}.mp4' if ck in picks['contrast'] else None, 'ckey': ck})
                results[f'{o["id"]}|{aid}|{int(prefix)}|{window}'] = {
                    'n_tested_total': int(h['n_tested_total'].iloc[0]), 'rows': rows}
    # consistency check of the chips against the stored table (mean-pool result set, full window)
    det_p = cfg.res / 'galleries/stats_detail.csv'
    o0 = next((o for o in cfg.outcomes if mean_outcome(o) and o['dir'] == cfg.res), None)
    if det_p.exists() and o0 is not None:
        det = pd.read_csv(det_p, dtype=str)
        names = {'flip': 'rob_signflip', 'BH': 'rob_BH', 'max': 'rob_max-pool', 'rate': 'rob_rate',
                 'match': 'rob_matched', '−30s': 'rob_trim30', '128': 'rob_other_prefix', '1024': 'rob_other_prefix'}
        bad = 0
        for _, d in det.iterrows():
            rr = [x for x in results[f'{o0["id"]}|{d["analysis_id"]}|{d["prefix"]}|full']['rows']
                  if x['neuron'] == int(d['neuron'])][0]
            for lab, _, v in rr['rob']:
                if lab in names and v != d[names[lab]]:
                    bad += 1
                    print('  robustness mismatch', d['analysis_id'], d['prefix'], d['neuron'], lab, v)
        print(f'page: robustness chips vs stats_detail.csv: {bad} mismatches over {len(det)} rows')

    slim = lambda c: {k: c[k] for k in ('obs', 'frame', 'stage', 'genotype', 'pool', 'act')}
    aur, n_eval = label_auroc(cfg, {k: [int(j) for j in picks['by_key'][k]] for k in cfg.keys})
    neurons = {}
    for key in cfg.keys:
        neurons[key] = {}
        for j, n in picks['by_key'][key].items():
            f = f'assets/{pre(key)}{int(j):04d}'
            neurons[key][j] = {'least_rule': n['least_rule'], 'firing_rate': n['firing_rate'],
                               'table': [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in t.items()}
                                         for t in n['table']],
                               'artefact': int(j) in ARTEFACT, 'auroc': aur.get(key, {}).get(str(j)),
                               'top_mp4': f'{f}_top.mp4', 'heat_mp4': f'{f}_heat.mp4', 'least_mp4': f'{f}_least.mp4'}
    contrast = {k: {'stages': c['stages'], 'pools': [{'pool': p['pool'], 'delta': p['delta'],
                                                      'va': p['a']['vmean'], 'vb': p['b']['vmean']}
                                                     for p in c['pools']]}
                for k, c in picks['contrast'].items()}
    art = {}
    for o in cfg.outcomes:
        sel = o['dir'] / 'selected_neurons.json'
        for j, v in (json.loads(sel.read_text())['neurons'] if sel.exists() else {}).items():
            if v.get('artefact_flag'):
                art.setdefault(j, v['artefact_flag'])
    for j in ARTEFACT:
        art.setdefault(str(j), 'flagged as a likely recording artefact')
    otables = {}  # outcome -> window -> neuron -> stage x genotype table of the tested outcome
    for o in cfg.outcomes:
        if is_bout(o):
            otables[o['id']] = {w: {j: rate_table(cfg, o, w, int(j)) for j in picks['by_key'][o['codes']]}
                                for w in ('full', 'trim30')}
    rep_note = ('foreground (mouse) patches of the whole frame, DINOv2 at 448' if REP == 'fg448'
                else 'all patches of the 224 center crop')
    data = {'sae': cfg.sae, 'rep': REP, 'rep_note': rep_note, 'events': EVENTS, 'n_eval_frames': n_eval,
            'outcomes': outcomes, 'otables': otables, 'analyses': [analyses[a] for a in ORDER if a in analyses],
            'results': results, 'neurons': neurons, 'contrast': contrast, 'artefact_text': art}
    page = TEMPLATE.read_text().replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / 'index.html').write_text(page)
    print('page: wrote', cfg.out / 'index.html', f'{len(page) / 1e3:.0f} KB')


# ---------------------------------------------------------------------- step: check
def step_check(cfg, overwrite):
    page = (cfg.out / 'index.html').read_text()
    assert page.startswith('<title>'), 'page must begin with <title>'
    for tag in ('html', 'head', 'body', '!doctype'):
        assert not re.search(rf'<{tag}[\s>]', page, re.I), f'page must not contain <{tag}>'
    refs = set(re.findall(r'assets/[\w.\-]+\.(?:mp4|webp|png|jpg)', page))
    missing = sorted(r for r in refs if not (cfg.out / r).exists())
    files = sorted(p for p in cfg.assets.iterdir() if p.is_file())
    unused = [p for p in files if f'assets/{p.name}' not in refs]
    for p in unused:
        p.unlink()
    files = [p for p in files if p not in unused]
    total = sum(p.stat().st_size for p in files) + (cfg.out / 'index.html').stat().st_size
    big = max(files, key=lambda p: p.stat().st_size)
    print(f'check: {len(refs)} referenced assets, {len(missing)} missing {missing[:5]}, '
          f'{len(unused)} unused deleted {[p.name for p in unused][:6]}')
    print(f'check: {len(files)} asset files + index.html, total {total / 1e6:.1f} MB, '
          f'largest {big.name} {big.stat().st_size / 1e3:.0f} KB, page {len(page) / 1e3:.0f} KB')
    if missing or len(files) + 1 > cfg.max_files or total > cfg.max_mb * 1e6:
        raise SystemExit('check failed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='results/vision/mice/eci/nes/matryoshka_btk_1024_k16_ep20_s0',
                    help='NES result set (summary.csv); cache in <res>/_cache/explorer')
    ap.add_argument('--outcome', action='append', help='LABEL=SUBDIR[:POOL_OUTCOME_TEST_CORR], repeatable')
    ap.add_argument('--sae', default=None, help='SAE name (default: basename of --res)')
    ap.add_argument('--out', default=None, help='output dir (default <res>/explorer)')
    ap.add_argument('--steps', default='data,patch,render,page,check')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--eval-codes', default=str(DATASET / 'mice/v1/eci/fg448/eval_codes'),
                    help='task_*.npz with the held-out annotated rows (label AUROC line)')
    ap.add_argument('--max-files', type=int, default=255)
    ap.add_argument('--max-mb', type=float, default=60)
    a = ap.parse_args()
    cfg = Cfg(a)
    for s in a.steps.split(','):
        globals()[f'step_{s}'](cfg, a.overwrite)
