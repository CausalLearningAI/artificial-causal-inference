"""
Build the NES explorer page "Exploratory Causal Inference x Mice": <out>/index.html + <out>/assets/.

The page design is scripts/eci/explorer_template.html (the user's hand-edited version of the
published page); this script fills its `const ALL = /*__DATA__*/null` with the data of one or more
NES result sets (one per SAE / representation, chosen with the page's SAE selector; the first --res
is the page default) and writes the assets they reference.

One command, all steps, incremental (GPU needed for the patch and arena steps):
    sbatch scripts/eci/build_explorer.sh
or directly (on a GPU node):
    python scripts/eci/build_explorer.py            # default: mouse-only (fg448) SAE + full-frame (ep20) SAE
    python scripts/eci/build_explorer.py --res results/vision/mice/eci/nes/<sae> [--res ...]
The SAE is the basename of each --res; its full per-frame codes (dataset/mice/v1/eci/codes/<sae>/), its
training tokens (tokens_dir in the SAE's metrics.json) and, for fg448, the per-video backgrounds (codes
config.json 'backgrounds') must exist. Output: <first res>/explorer/ (publish the folder). The last step
checks the limits (files, MB, 0 missing / 0 unused assets).

Data-driven: everything comes from the result sets (summary.csv files written by the NES runs); only
the full-video window is shown. The first --outcome is the page default. Neurons discovered in ANY
outcome's primary search (prefix 128 / 1024, full window) get clips, ranked by the codes the outcome is
built from (codes_mean for mean-pool outcomes, codes_max for max-pool outcomes).

Steps (each cached under <res>/_cache/explorer/, incremental):
    data     reads the neuron's codes on every frame (2.59 M) and, per clip length (src/eci/viz.py
             scan_windows / pick_top_windows / pick_least_windows, K = 16, at most one per video):
               frame  top = the K videos' single highest frames (ranked by that frame's activation);
               1 s    windows of 5 consecutive frames (5 fps), 3 s = 15 frames: in each video the window
                      with the highest MEAN activation over its frames, the K best videos (sustained events);
               least  (every length) windows (or single frames) where the neuron is exactly 0 on EVERY
                      frame: one random such window per video, K random videos (fixed seed); when fewer
                      than K videos have one, the K videos' lowest-mean windows (rule 'lowest', labelled
                      "lowest (not silent)" on the page).
             Plus the activation histogram over all frames (40 linear bins, all frames and per
             genotype|stage), the firing rate and the stage x genotype table of per-video mean codes.
             -> picks.json
    patch    per-patch codes of the neuron on every frame of every clip (top and least, all lengths),
             recomputed through DINOv2 + SAE (src/eci/viz.py make_patch_encoder, GPU; crop224 or fg448
             geometry from the SAE's codes config), checked against the stored pooled codes. The heat
             colour scale of a neuron = the 99th percentile of its positive patch codes over all those
             frames (shared by every row and length). -> patch/<p>XXXX.npz
    arena    arena maps of ALL neurons: per patch position, the mean SAE code over the SAE's training
             frames (fg448: 1 frame per second of every video, codes 0 off the foreground, plus how
             often the patch is foreground); the arena background image. -> arena.npz, assets/<tag>_arena_bg.webp
    render   one montage per neuron x length, 200 px tiles, 8 per row, four blocks of K tiles each
             (row-major): top raw, top with heat, least raw, least with heat (turbo, the neuron's shared
             scale). frame -> one webp still; 1 s / 3 s -> one H.264 mp4 (5 / 15 frames at 5 fps).
    page     data inlined into scripts/eci/explorer_template.html -> <out>/index.html; also the per-video
             outcome values of every shown neuron (from the NES caches: <res>/_cache/video_summaries_*.npz,
             bout_summaries_max.npz) for the per-pool panel, checked against the tau of summary.csv.
    check    every referenced asset exists; unreferenced files in assets/ are deleted; counts, MB
             (fails above --max-files files / --max-mb MB / a file > 15 MB / page > 16 MB)

Representations (src/eci/viz.py representation): 'crop224' SAEs are patch SAEs on the 224 center crop
(16 x 16 patches, heat on pixels 32-480); 'fg448' SAEs (src/eci/foreground.py) see the whole frame at
448 (32 x 32 patches of 16 px) and fire only on foreground patches (others = 0).
Chips: one per single-setting change from the outcome's primary (summary.csv columns; bout_rule 1 =
min-2-frame hysteresis bouts 'min2+hyst'), plus 'size' when <outcome dir>/size_adjusted.csv exists.
Artefact flags (neurons 50, 64, 113) belong to the ep20 SAE only.

Outcome spec (default: bout rate (maxpool_bouts/) + mean activation (.) pooled as the run's primary
(selected_neurons.json 'settings'), each when its summary.csv exists; applies to every --res):
    --outcome LABEL=SUBDIR[:key=value,...]   SUBDIR relative to --res. Keys: pooling, outcome_type,
    test (default t), correction (default bonferroni), any extra summary.csv setting column
    (threshold_q, merge_gap, ...; required when it has several values), codes (mean|max, default
    = pooling), unit (tau unit), short (chip label).
"""

import argparse
import hashlib
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
NES = 'results/vision/mice/eci/nes'
DEFAULT_RES = [f'{NES}/matryoshka_btk_1024_k16_fg448_s0', f'{NES}/matryoshka_btk_1024_k16_ep20_s0']
ARTEFACT_EP20 = {64, 50, 113}  # ep20 SAE neuron ids; other SAEs get no flags
# SAE -> (label in the selector, one-line description)
SAE_LABEL = {
    'matryoshka_btk_1024_k16_fg448_s0': ('mouse only', 'Mouse only: DINOv2 sees the whole frame at 448 px; the SAE is '
                                         'trained from scratch on the mouse (foreground) patches only'),
    'matryoshka_btk_1024_k16_ep20_s0': ('full frame', 'Full frame: DINOv2 sees the 224 px center crop; the SAE is '
                                        'trained on all patches (mice and bedding)')}
K, TILE, COLS = 16, 200, 8            # clips per row block, tile px, tiles per montage row
LENGTHS = {'frame': 1, '1s': 5, '3s': 15}
BLOCKS = (('top', False), ('top', True), ('least', False), ('least', True))  # montage block order
HIST_BINS = 40
PICKS_V = 3                          # picks.json entry version (3 = per-length window selections)
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
                    'Mean activation ({p}-pool)=.:pooling={p},outcome_type=mean,short={p}-pool']


def default_outcomes(res):
    """DEFAULT_OUTCOMES whose summary.csv exists; the mean-activation outcome uses the run's primary
    pooling as recorded in <res>/selected_neurons.json 'settings' ('primary (codes_max, ...)'), else mean."""
    sel = res / 'selected_neurons.json'
    m = re.search(r'codes_(mean|max)', json.loads(sel.read_text()).get('settings', '')) if sel.exists() else None
    specs = [x.replace('{p}', m[1] if m else 'mean') for x in DEFAULT_OUTCOMES]
    return [x for x in specs if (res / x.split('=', 1)[1].split(':')[0] / 'summary.csv').exists()]


class Cfg:
    """One NES result set (one SAE)."""

    def __init__(self, res, a, out):
        self.res = (ROOT / res).resolve() if not Path(res).is_absolute() else Path(res)
        self.sae = self.res.name
        self.tag = self.sae.split('_')[-2]  # 'fg448', 'ep20': asset file prefix
        self.out = out
        self.assets = out / 'assets'
        self.work = self.res / '_cache' / 'explorer'
        self.outcomes = []
        for spec in a.outcome or default_outcomes(self.res):
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
            print(f'[{self.tag}] outcome {label!r}: {d} primary {prim} codes_{codes}')
        self.keys = sorted({o['codes'] for o in self.outcomes})
        from src.eci.viz import representation
        self.rep = representation(self.sae, DATASET)
        self.artefact = ARTEFACT_EP20 if self.sae == 'matryoshka_btk_1024_k16_ep20_s0' else set()
        self.crf = a.crf
        print(f'[{self.tag}] SAE {self.sae}: representation {self.rep}, artefact flags {sorted(self.artefact)}')


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
    """The outcome's primary setting, full-video window only (the page shows full videos only)."""
    t = o['tidy']
    return t[match(t, o['primary']) & (t['window'] == 'full').values]


def mean_outcome(o):
    return (o['primary']['pooling'], o['primary']['outcome_type']) == ('mean', 'mean')


def is_bout(o):
    return o['primary']['outcome_type'] == 'bout_rate'


_rates = {}


def bout_rates(cfg, o, window='full', fps=5.0):
    """Per-video bouts/min of every neuron (src/eci/contrasts.py bout_outcomes, same cache as
    scripts/eci/run_nes_bouts.py) -> (Index of observation_id, (n_obs, m) array)."""
    q, g = o['primary']['threshold_q'], int(o['primary']['merge_gap'])
    w = {'full': 'full', 'trim30': 'trim', 'matched': 'last'}[window]
    k = (str(cfg.res), w, q, g)
    if k not in _rates:
        f = np.load(cfg.res / '_cache' / 'bout_summaries_max.npz')
        c, nf = f[f'{w}__{q:g}__{g}__count'], f[f'{w}__n_frames']
        _rates[k] = (pd.Index(f['observation_id'].astype(str)), c / (nf[:, None] / fps / 60.0))
    return _rates[k]


def video_outcome(cfg, o):
    """(Index of observation_id, (n_obs, m)) per-video value of the tested outcome, full videos, as the
    NES runs computed it: bouts/min, or the per-video mean of codes_<pooling>."""
    if is_bout(o):
        return bout_rates(cfg, o, 'full')
    if o['primary']['outcome_type'] != 'mean':
        raise SystemExit(f'per-video values: outcome_type {o["primary"]["outcome_type"]!r} not supported')
    f = np.load(cfg.res / '_cache' / f'video_summaries_{o["primary"]["pooling"]}.npz')
    return pd.Index(f['observation_id'].astype(str)), f['full__mean']


def vmeta():
    from src.eci.viz import video_meta
    if 'vm' not in _rates:
        _rates['vm'] = video_meta(ROOT / 'data')
    return _rates['vm'].copy()


def rate_table(cfg, o, j):
    """stage x genotype: mean over pools of the per-video bouts/min, 95% t-CI (as viz.stage_genotype_table)."""
    from scipy import stats
    vm = vmeta()
    ids, rate = bout_rates(cfg, o)
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


def wanted(cfg):
    want = {k: set() for k in cfg.keys}
    for o in cfg.outcomes:
        want[o['codes']] |= {int(j) for j in primary_rows(o).dropna(subset=['neuron'])['neuron']}
    return want


# ---------------------------------------------------------------------- step: data
def activation_hist(x, group, n_groups):
    """40 linear bins from 0 to max(x): counts over all frames and per group id."""
    hi = float(x.max()) if x.max() > 0 else 1.0
    b = np.minimum((np.maximum(x, 0) / hi * HIST_BINS).astype(np.int64), HIST_BINS - 1)
    c = np.bincount(group * HIST_BINS + b, minlength=n_groups * HIST_BINS).reshape(n_groups, HIST_BINS)
    return np.linspace(0, hi, HIST_BINS + 1), c


def step_data(cfg, overwrite):
    out = cfg.work / 'picks.json'
    res = json.loads(out.read_text()) if out.exists() and not overwrite else {}
    res.pop('contrast', None)
    want = wanted(cfg)
    # keep only what is wanted now (drops stale entries, e.g. an older outcome's codes key)
    res['by_key'] = {k: {j: v for j, v in res.get('by_key', {}).get(k, {}).items() if int(j) in want[k]}
                     for k in cfg.keys}
    missing = {k: sorted(j for j in v if res['by_key'][k].get(str(j), {}).get('v') != PICKS_V)
               for k, v in want.items()}
    if not any(missing.values()):
        print(f'[{cfg.tag}] data: cached', {k: len(v) for k, v in res['by_key'].items()}, 'neurons')
        out.write_text(json.dumps(res))
        return
    from types import SimpleNamespace
    from src.eci.viz import (CodeSource, load_full_codes, pick_least_windows, pick_top_windows, scan_windows,
                             stage_genotype_table)
    src = load_full_codes(cfg.sae, DATASET, ROOT / 'data')
    meta = src.meta
    fp = meta['frame_path'].values
    gkeys = sorted({f'{g}|{int(s)}' for g, s in zip(meta['genotype'], meta['stage'])})
    gid = pd.Index(gkeys).get_indexer(meta['genotype'].astype(str) + '|' + meta['stage'].astype(int).astype(str))
    obs, fidx = meta['observation_id'].values, meta['frame_idx'].values
    stg, gen, pool = meta['stage'].values, meta['genotype'].values, meta['pool'].astype(str).values
    for key, miss in missing.items():
        if not miss:
            continue
        print(f'[{cfg.tag}] data: reading codes_{key} of neurons', miss, flush=True)
        neurons = np.array(miss, dtype=np.int64)
        Z = src.codes[key]
        X = np.empty((len(meta), len(neurons)), np.float32)  # every frame of the selected neurons
        for a in range(0, len(meta), 200_000):
            X[a:a + 200_000] = Z[a:a + 200_000][:, neurons]
        loc = CodeSource('sel', {key: X}, meta, DATASET)
        wss = scan_windows(loc, np.arange(len(neurons)), key, tuple(LENGTHS.values()), seed=0)
        for ws in wss.values():
            ws.neurons = neurons  # column i of X is neuron neurons[i] (seeds use the neuron id)
        vids = next(iter(wss.values())).videos
        vmean = np.stack([X[lo:hi].mean(0) for lo, hi in zip(vids['lo'], vids['hi'])])
        store = res['by_key'].setdefault(key, {})
        for i, j in enumerate(miss):
            def clip(start, w):
                tr = X[start:start + w, i]
                return {'start': int(start), 'rows': list(range(int(start), int(start) + w)),
                        'act': float(tr.mean()), 'trace': [float(x) for x in tr], 'obs': str(obs[start]),
                        'frame': int(fidx[start]), 'stage': int(stg[start]), 'genotype': str(gen[start]),
                        'pool': str(pool[start])}
            clips = {}
            for L, w in LENGTHS.items():
                ts, _ = pick_top_windows(wss[w], i, K)
                ls, rule = pick_least_windows(wss[w], i, K, seed=0)
                clips[L] = {'top': [clip(s, w) for s in ts], 'least': [clip(s, w) for s in ls], 'least_rule': rule}
                assert all(len({c['obs'] for c in clips[L][k]}) == len(clips[L][k]) for k in ('top', 'least'))
                if rule == 'silent':
                    assert all(max(c['trace']) == 0 for c in clips[L]['least'])
            edges, counts = activation_hist(X[:, i], gid, len(gkeys))
            tab = stage_genotype_table(SimpleNamespace(videos=vids, video_mean=vmean), i)
            store[str(j)] = {
                'v': PICKS_V, 'clips': clips,
                'firing_rate': float((X[:, i] > 0).mean()),
                'max_frame': float(X[:, i].max()),
                'hist': {'edges': [float(f'{e:.5g}') for e in edges],
                         'counts': dict({'all': [int(c) for c in counts.sum(0)]},
                                        **{g: [int(c) for c in counts[k]] for k, g in enumerate(gkeys)})},
                'table': [{k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if k in ('stage', 'n_pools')
                               else v) for k, v in rec.items()} for rec in tab.to_dict('records')]}
            print(f'  [{cfg.tag}] codes_{key} neuron {j}: ' + ', '.join(
                f'{L} top {len(c["top"])} (mean {c["top"][0]["act"]:.3g}..{c["top"][-1]["act"]:.3g}) '
                f'least {len(c["least"])} {c["least_rule"]} (max {max(max(x["trace"]) for x in c["least"]):.3g})'
                for L, c in clips.items() if c['top']), flush=True)
        del X
    res['frame_path'] = {str(r): fp[r] for r in sorted({r for st in res['by_key'].values() for n in st.values()
                                                        for c in n['clips'].values() for kind in ('top', 'least')
                                                        for x in c[kind] for r in x['rows']})}
    cfg.work.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(f'[{cfg.tag}] data: wrote', out, {k: len(v) for k, v in res['by_key'].items()}, 'neurons')


# ---------------------------------------------------------------------- step: patch
def patch_file(cfg, key, j):
    return cfg.work / 'patch' / f'{pre(key)}{int(j):04d}.npz'


def clip_rows(n):
    """{'<L>_<kind>': rows of every clip, in clip order} of one neuron."""
    return {f'{L}_{kind}': [r for x in c[kind] for r in x['rows']] for L, c in n['clips'].items()
            for kind in ('top', 'least')}


def step_patch(cfg, overwrite, chunk=6):
    picks = json.loads((cfg.work / 'picks.json').read_text())
    (cfg.work / 'patch').mkdir(parents=True, exist_ok=True)
    todo = []
    for k in cfg.keys:
        for j, n in picks['by_key'].get(k, {}).items():
            f, rr = patch_file(cfg, k, j), clip_rows(n)
            sig = hashlib.md5(json.dumps(rr).encode()).hexdigest()
            if overwrite or not f.exists() or str(np.load(f)['sig']) != sig:
                todo.append((k, int(j), rr, sig))
    if not todo:
        print(f'[{cfg.tag}] patch: cached')
        return
    from src.eci.viz import load_full_codes, make_patch_encoder
    print(f'[{cfg.tag}] patch: {len(todo)} neurons, '
          f'{sum(len(set(r for v in rr.values() for r in v)) for _, _, rr, _ in todo)} frame-neuron pairs '
          f'through DINOv2 + SAE ({cfg.rep})', flush=True)
    pe = make_patch_encoder(cfg.sae, dataset_dir=DATASET)
    codes = load_full_codes(cfg.sae, DATASET, ROOT / 'data').codes
    worst, mism, npair = {}, {}, 0
    for c0 in range(0, len(todo), chunk):
        part = todo[c0:c0 + chunk]
        js = [j for _, j, _, _ in part]
        allr = sorted({r for _, _, rr, _ in part for v in rr.values() for r in v})
        paths = [str(DATASET / picks['frame_path'][str(r)]) for r in allr]
        if pe.rep == 'fg448':
            pc, fg = pe.patch_codes(paths, np.array(js), np.array(allr), return_mask=True)
            n_fg = fg.reshape(len(allr), -1).sum(1)
        else:
            pc, n_fg = pe.patch_codes(paths, np.array(js)), None
        pos = {r: i for i, r in enumerate(allr)}
        for ci, (k, j, rr, sig) in enumerate(part):
            ur = sorted({r for v in rr.values() for r in v})
            ii = np.array([pos[r] for r in ur])
            m = pc[ii, :, :, ci]
            stored = np.asarray(codes[k][np.array(ur)][:, j], dtype=np.float32)
            if k == 'max':
                pooled = m.max((1, 2))
            elif n_fg is None:
                pooled = m.mean((1, 2))
            else:  # fg448: mean over the foreground patches only
                pooled = m.sum((1, 2)) / np.maximum(n_fg[ii], 1)
            worst[k] = max(worst.get(k, 0.0), float(np.abs(pooled - stored).max()))
            mism[k] = mism.get(k, 0) + int(((pooled > 0) != (stored > 0)).sum())
            npair += len(ur)
            posv = m[m > 0]
            vmax = float(np.quantile(posv, 0.99)) if len(posv) else 1.0
            lp = {r: t for t, r in enumerate(ur)}
            np.savez(patch_file(cfg, k, j), sig=sig, vmax=vmax,
                     **{name: m[[lp[r] for r in v]].astype(np.float16) for name, v in rr.items()})
        print(f'  [{cfg.tag}] patch: {c0 + len(part)}/{len(todo)} neurons', flush=True)
    print(f'[{cfg.tag}] patch: recomputed patch maps pooled vs stored codes, max abs diff {worst}, '
          f'active/inactive mismatches {mism} (of {npair} frame-neuron pairs)')


# ---------------------------------------------------------------------- step: arena
def bg_name(cfg):
    return f'{cfg.tag}_arena_bg.webp'


def step_arena(cfg, overwrite):
    """Per patch position: summed SAE codes of every neuron and foreground counts over the SAE's
    training frames (fg448: FgTokenStore shards; crop224: patch_tokens.npy, every patch counts)."""
    import torch
    from src.eci.sae import load_sae
    out = cfg.work / 'arena.npz'
    bg_out = cfg.assets / bg_name(cfg)
    sae_dir = DATASET / 'mice/v1/eci/sae' / cfg.sae
    tokens_dir = Path(json.loads((sae_dir / 'metrics.json').read_text())['args']['tokens_dir'])
    if not out.exists() or overwrite:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sae, norm, _ = load_sae(sae_dir / 'sae.pt', dev)
        chk = []
        if cfg.rep == 'fg448':
            codes = __import__('src.eci.viz', fromlist=['load_full_codes']).load_full_codes(
                cfg.sae, DATASET, ROOT / 'data').codes['max']
            from src.eci.foreground import GRID, FgTokenStore
            st = FgTokenStore(tokens_dir)
            P, n_frames = GRID * GRID, int(sum(i['n_frames'] for i in st.info))
            S = torch.zeros(P, sae.n_latents, dtype=torch.float64, device=dev)
            occ = torch.zeros(P, dtype=torch.float64, device=dev)
            for s in range(len(st.dirs)):
                tok, pos, row = st.tokens(s), torch.from_numpy(st.pos(s).astype(np.int64)), st.row(s)
                for a in range(0, len(pos), 250_000):
                    with torch.no_grad():
                        x = torch.from_numpy(np.array(tok[a:a + 250_000])).to(dev)
                        z = sae.encode(norm(x), mode='threshold')
                        p = pos[a:a + 250_000].to(dev)
                        S.index_add_(0, p, z.double())
                        occ.index_add_(0, p, torch.ones_like(p, dtype=torch.float64))
                        if s == 0 and a == 0:  # check: per-frame max over positions = stored codes_max
                            r = torch.from_numpy(row[:len(x)].astype(np.int64)).to(dev)
                            u, inv = torch.unique(r, return_inverse=True)
                            mx = torch.zeros(len(u), z.shape[1], device=dev).scatter_reduce(
                                0, inv[:, None].expand_as(z), z, 'amax', include_self=True)
                            keep = slice(0, len(u) - 1)  # the last frame may continue in the next chunk
                            ref = np.asarray(codes[u[keep].cpu().numpy()], dtype=np.float32)
                            chk = [float(np.abs(mx[keep].cpu().numpy() - ref).max()), len(u) - 1]
                print(f'[{cfg.tag}] arena: shard {s + 1}/{len(st.dirs)}', flush=True)
            n_vid = len({o for i in st.info for o in i['observations']})
            stride = st.info[0].get('stride')
            note = (f'the {n_frames:,} frames of the SAE training sample (every {stride}th frame, i.e. 1 per '
                    f'second, of all {n_vid} videos)' if stride else f'the {n_frames:,} SAE training frames')
            grid = GRID
        else:
            tok = np.load(tokens_dir / 'patch_tokens.npy', mmap_mode='r')
            n_frames, P = tok.shape[0], tok.shape[1]
            grid = int(round(P ** 0.5))
            S = torch.zeros(P, sae.n_latents, dtype=torch.float64, device=dev)
            for a in range(0, n_frames, 1024):
                with torch.no_grad():
                    x = torch.from_numpy(np.array(tok[a:a + 1024])).to(dev)
                    z = sae.encode(norm(x.reshape(-1, x.shape[-1])), mode='threshold').view(len(x), P, -1)
                    S += z.double().sum(0)
                if a % 51200 == 0:
                    print(f'[{cfg.tag}] arena: {a}/{n_frames} frames', flush=True)
            occ = None
            note = f'the {n_frames:,} frames of the SAE training sample'
        np.savez(out, act=(S / n_frames).float().cpu().numpy(), grid=grid, n_frames=n_frames, note=note,
                 occ=(occ / n_frames).float().cpu().numpy() if occ is not None else np.zeros(0))
        print(f'[{cfg.tag}] arena: {n_frames} frames, grid {grid}; check per-frame max vs stored codes_max: '
              f'max abs diff {chk[0] if chk else "n/a"} over {chk[1] if chk else 0} frames', flush=True)
    else:
        print(f'[{cfg.tag}] arena: cached')
    if not bg_out.exists() or overwrite:  # median over videos of the empty-bedding pixel background
        cfg.assets.mkdir(parents=True, exist_ok=True)
        cc = DATASET / 'mice/v1/eci/codes' / cfg.sae / 'config.json'
        bgd = Path(json.loads(cc.read_text()).get('backgrounds', '')) if cc.exists() else None
        if bgd is not None and bgd.is_dir() and str(bgd) != '.':
            ims = np.stack([np.load(f)['pix_bg'] for f in sorted(bgd.glob('*.npz'))])
            im = np.median(ims, 0).astype(np.uint8)
            src = f'median of {len(ims)} per-video empty-bedding backgrounds'
        else:  # no stored backgrounds (crop224): median of one frame per video
            meta = pd.read_csv(DATASET / 'mice/v1/annotations.csv', usecols=['observation_id', 'frame_path'])
            fr = meta.groupby('observation_id')['frame_path'].nth(0).values
            im = np.median(np.stack([np.asarray(Image.open(DATASET / f).convert('L')) for f in fr]), 0).astype(np.uint8)
            src = f'median of the first frame of {len(fr)} videos'
        from src.eci.viz import frame_box
        x0, y0, c = frame_box(cfg.rep, im.shape[1], im.shape[0])
        im = Image.fromarray(im).crop((int(round(x0)), int(round(y0)), int(round(x0 + c)), int(round(y0 + c))))
        im.resize((384, 384), Image.BILINEAR).save(bg_out, 'WEBP', quality=80)
        print(f'[{cfg.tag}] arena: wrote', bg_out, src)


# ---------------------------------------------------------------------- step: render
_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT, size)
    return _fonts[k]


def tile_frames(c, frame_path, vmax, color, rep, maps=None, hvmax=None):
    """len(c['rows']) labelled TILE x TILE frames of one clip; a bar at the bottom shows the activation
    of the current frame relative to vmax. maps: (w, grid, grid) patch codes -> heatmap overlaid (scale hvmax)."""
    from src.eci.viz import _overlay_rgb, frame_box
    import matplotlib
    matplotlib.use('Agg')
    cmap = matplotlib.colormaps['turbo']
    name, _ = short_obs(c['obs'])
    frames = []
    for t, r in enumerate(c['rows']):
        im = Image.open(DATASET / frame_path[str(r)]).convert('RGB')
        if maps is not None:
            a = np.asarray(im)
            im = Image.fromarray(_overlay_rgb(a, maps[t].astype(np.float32), frame_box(rep, a.shape[1], a.shape[0]),
                                              hvmax, cmap))
        im = im.resize((TILE, TILE), Image.BILINEAR)
        d = ImageDraw.Draw(im, 'RGBA')
        d.rectangle([0, 0, TILE, 17], fill=(0, 0, 0, 140))
        d.text((5, 2), f'S{c["stage"]} {STAGE_LABEL[c["stage"]]} · {c["genotype"]} · {name}', font=font(11),
               fill=(255, 255, 255))
        d.rectangle([0, TILE - 6, TILE, TILE], fill=(0, 0, 0, 255))  # opaque: the page re-reads this bar (row 196)
        w = int(round(TILE * min(max(c['trace'][t], 0) / vmax, 1))) if vmax > 0 else 0
        if w:
            d.rectangle([0, TILE - 5, w, TILE], fill=color)
        frames.append(np.asarray(im))
    return frames


def encode(frames, path, crf=30):
    """frames: list of HxWx3 uint8 arrays -> H.264 mp4 at 5 fps."""
    h, w = frames[0].shape[:2]
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
           '-r', '5', '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', '-an', str(path)]
    subprocess.run(cmd, input=b''.join(f.tobytes() for f in frames), check=True)


def asset_name(cfg, key, j, L):
    return f'{cfg.tag}_{pre(key)}{int(j):04d}_{L}.{"webp" if L == "frame" else "mp4"}'


def _render_one(task):
    out, n_clips, fp, vmax, rep, maps_file, L, crf = task
    w = LENGTHS[L]
    pz = np.load(maps_file)
    hv = float(pz['vmax'])
    blank = [np.zeros((TILE, TILE, 3), np.uint8)] * w
    tiles = []
    for kind, heat in BLOCKS:
        maps = pz[f'{L}_{kind}'].astype(np.float32) if heat else None
        color = (255, 170, 40) if kind == 'top' else (120, 190, 255)
        cl = [tile_frames(c, fp, vmax, color, rep, None if maps is None else maps[k * w:(k + 1) * w], hv)
              for k, c in enumerate(n_clips[kind])]
        tiles += cl + [blank] * (K - len(cl))
    rows = [tiles[r:r + COLS] for r in range(0, len(tiles), COLS)]
    frames = [np.concatenate([np.concatenate([clip[t] for clip in row], 1) for row in rows], 0) for t in range(w)]
    if L == 'frame':
        Image.fromarray(frames[0]).save(out, 'WEBP', quality=72, method=6)
    else:
        encode(frames, out, crf)
    return out


def step_render(cfg, overwrite):
    import multiprocessing as mp
    import os
    picks = json.loads((cfg.work / 'picks.json').read_text())
    fp = picks['frame_path']
    cfg.assets.mkdir(parents=True, exist_ok=True)
    man_p = cfg.work / 'render.json'
    man = json.loads(man_p.read_text()) if man_p.exists() and not overwrite else {}
    tasks, sigs = [], {}
    for key in cfg.keys:
        for j, n in picks['by_key'].get(key, {}).items():
            vmax = n['max_frame'] or 1.0  # bar scale: the neuron's highest frame, every row and length
            pf = patch_file(cfg, key, j)
            hv = float(np.load(pf)['vmax'])
            for L in LENGTHS:
                c = n['clips'][L]
                out = cfg.assets / asset_name(cfg, key, j, L)
                sig = json.dumps([cfg.rep, TILE, COLS, K, cfg.crf, vmax, hv, BLOCKS,
                                  [x['start'] for x in c['top']], [x['start'] for x in c['least']]])
                if out.exists() and man.get(out.name) == sig:
                    continue
                sigs[out.name] = sig
                tasks.append((out, c, fp, vmax, cfg.rep, str(pf), L, cfg.crf))
    if not tasks:
        print(f'[{cfg.tag}] render: cached')
        return
    nproc = max(1, min(len(tasks), len(os.sched_getaffinity(0))))
    print(f'[{cfg.tag}] render: {len(tasks)} montages on {nproc} processes', flush=True)
    with mp.get_context('fork').Pool(nproc) as pool:
        for i, out in enumerate(pool.imap_unordered(_render_one, tasks)):
            man[out.name] = sigs[out.name]
            man_p.write_text(json.dumps(man))
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f'[{cfg.tag}] render: {i + 1}/{len(tasks)} {out.name}', flush=True)


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


ORDER = [f'A_{g}_{t}' for g in ('het', 'wt') for t in ('1to2', '2to3', '4to5', '5to6')] + \
        [f'B_stage{s}' for s in range(1, 7)]


def sig4(v):
    return float(f'{float(v):.4g}')


def bout_thresholds(cfg):
    """{codes key: (m,) threshold of that key's bout outcome} (the q of its primary), from the
    run's _cache/bout_thresholds_*.npz; empty when no bout outcome."""
    out = {}
    for o in cfg.outcomes:
        if is_bout(o) and o['codes'] not in out:
            fs = sorted((cfg.res / '_cache').glob('bout_thresholds_*.npz'))
            if not fs:
                raise SystemExit(f'{cfg.res}/_cache/bout_thresholds_*.npz missing (bout outcome {o["label"]!r})')
            f = np.load(fs[0])
            k = np.flatnonzero(np.isclose(f['qs'], o['primary']['threshold_q']))
            out[o['codes']] = f['thr'][int(k[0])]
    return out


def tau_check(vm, vals, results, oid, round_=1):
    """max relative |tau(summary.csv) - raw contrast of the per-video values| over the page's round-`round_`
    rows: A = mean over pools of (stage b - stage a), B = mean het - mean wt. Only round 1 of an
    unconditioned search should match exactly: later rounds (and every round of a search conditioned on a
    nuisance such as n_fg) report the effect adjusted for the already-selected neurons (src/eci/nes.py)."""
    worst = 0.0
    for rk, R in results.items():
        o_, aid, _, _ = rk.split('|')
        if o_ != oid:
            continue
        for r in R['rows']:
            if r['round'] != round_:
                continue
            y = np.asarray(vals[str(r['neuron'])], dtype=np.float64)
            if aid.startswith('A'):
                g, (a, b) = aid.split('_')[1], stages_of(aid)
                d = vm[vm['genotype'] == g].assign(y=y[vm.index[vm['genotype'] == g]])
                p = d.pivot_table(index='pool', columns='stage', values='y')
                t = float((p[b] - p[a]).mean())
            else:
                s = int(aid[len('B_stage'):])
                d = vm.assign(y=y)[vm['stage'] == s]
                t = float(d[d['genotype'] == 'het']['y'].mean() - d[d['genotype'] == 'wt']['y'].mean())
            worst = max(worst, abs(t - r['tau']) / max(abs(r['tau']), 1e-9))
    return worst


def page_data(cfg):
    picks = json.loads((cfg.work / 'picks.json').read_text())
    analyses, results, outcomes = {}, {}, []
    for o in cfg.outcomes:
        prim = primary_rows(o)
        others = [x for x in cfg.outcomes if x is not o]
        outcomes.append({'id': o['id'], 'label': o['label'], 'codes': o['codes'], 'unit': o['unit'],
                         'bout': is_bout(o),
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
                    rows.append({'round': int(r['round']), 'neuron': j, 'tau': float(r['tau']), 'p': float(r['p']),
                                 'threshold': float(r['threshold']), 'n_tested': int(r['n_tested']),
                                 'rob': robustness(o, others, aid, int(prefix), window, j)})
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
        print(f'[{cfg.tag}] page: robustness chips vs stats_detail.csv: {bad} mismatches over {len(det)} rows')

    thr = bout_thresholds(cfg)
    ar = np.load(cfg.work / 'arena.npz')
    grid, act_all, occ = int(ar['grid']), ar['act'], ar['occ']
    bg = f'assets/{bg_name(cfg)}' if (cfg.assets / bg_name(cfg)).exists() else None
    neurons = {}
    for key in cfg.keys:
        neurons[key] = {}
        for j, n in picks['by_key'][key].items():
            vmax = n['max_frame'] or 1.0
            t = float(thr[key][int(j)]) if key in thr else None
            e = {'firing_rate': n['firing_rate'],
                 'table': [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in t_.items()} for t_ in n['table']],
                 'artefact': int(j) in cfg.artefact,
                 'heat_vmax': sig4(np.load(patch_file(cfg, key, j))['vmax']),
                 'clips': {L: {'src': f'assets/{asset_name(cfg, key, j, L)}', 'n_top': len(c['top']),
                               'n_least': len(c['least']), 'least_rule': c['least_rule'],
                               'top_mean': [sig4(x['act']) for x in c['top']]}
                           for L, c in n['clips'].items()},
                 'hist': dict(n['hist'], thr=t),
                 'bout_thr_bar': (round(min(t / vmax, 1.0), 4) if t is not None and vmax > 0 else None),
                 'arena': {'rows': grid, 'cols': grid, 'act': [sig4(v) for v in act_all[:, int(j)]]}}
            if len(occ):
                e['arena']['occ'] = [sig4(v) for v in occ]
            if bg:
                e['arena']['bg'] = bg
            neurons[key][j] = e
    art = {}
    for o in cfg.outcomes:
        sel = o['dir'] / 'selected_neurons.json'
        for j, v in (json.loads(sel.read_text())['neurons'] if sel.exists() else {}).items():
            if v.get('artefact_flag'):
                art.setdefault(j, v['artefact_flag'])
    for j in cfg.artefact:
        art.setdefault(str(j), 'flagged as a likely recording artefact')
    # per-video values of the tested outcome, for the per-pool panel (one list per outcome and neuron,
    # in the order of 'videos'); tables of bout outcomes by stage x genotype
    vm = vmeta().reset_index(drop=True)
    videos = [[str(p), int(s), g] for p, s, g in zip(vm['pool'], vm['stage'], vm['genotype'])]
    vals, otables = {}, {}
    for o in cfg.outcomes:
        ids, Y = video_outcome(cfg, o)
        ix = ids.get_indexer(vm['observation_id'].astype(str))
        assert (ix >= 0).all(), 'videos missing from the NES cache'
        js = sorted({str(r['neuron']) for k, R in results.items() if k.startswith(o['id'] + '|') for r in R['rows']},
                    key=int)
        vals[o['id']] = {j: [sig4(v) for v in Y[ix, int(j)]] for j in js}
        print(f'[{cfg.tag}] page: {o["label"]}: tau recomputed from per-video values vs summary.csv, '
              f'max relative diff {tau_check(vm, vals[o["id"]], results, o["id"]):.2e}')
        if is_bout(o):
            otables[o['id']] = {'full': {j: rate_table(cfg, o, int(j)) for j in picks['by_key'][o['codes']]}}
    label, desc = SAE_LABEL.get(cfg.sae, (cfg.sae, cfg.sae))
    return {'sae': cfg.sae, 'label': label, 'desc': desc, 'rep': cfg.rep, 'outcomes': outcomes, 'otables': otables,
            'analyses': [analyses[a] for a in ORDER if a in analyses], 'results': results, 'neurons': neurons,
            'artefact_text': art, 'arena_note': str(ar['note']), 'videos': videos, 'vals': vals,
            'n_frames': int(sum(next(iter(picks['by_key'][cfg.keys[0]].values()))['hist']['counts']['all']))}


def step_page(cfgs, overwrite):
    from scipy import stats
    data = {'default': cfgs[0].sae, 'saes': [page_data(c) for c in cfgs],
            'montage': {'K': K, 'cols': COLS, 'tile': TILE, 'blocks': [f'{k}{"_heat" if h else ""}' for k, h in BLOCKS],
                        'lengths': {L: w for L, w in LENGTHS.items()}},
            'tcrit': {str(n): round(float(stats.t.ppf(0.975, n - 1)), 4) for n in range(2, 121)}}
    tpl = TEMPLATE.read_text()
    assert tpl.count('/*__DATA__*/null') == 1, 'template placeholder missing'
    page = tpl.replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
    out = cfgs[0].out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'index.html').write_text(page)
    print('page: wrote', out / 'index.html', f'{len(page) / 1e3:.0f} KB')


# ---------------------------------------------------------------------- step: check
def step_check(cfgs, a):
    out = cfgs[0].out
    assets = out / 'assets'
    page = (out / 'index.html').read_text()
    assert page.startswith('<title>'), 'page must begin with <title>'
    for tag in ('html', 'head', 'body', '!doctype'):
        assert not re.search(rf'<{tag}[\s>]', page, re.I), f'page must not contain <{tag}>'
    refs = set(re.findall(r'assets/[\w.\-]+\.(?:mp4|webp|png|jpg)', page))
    missing = sorted(r for r in refs if not (out / r).exists())
    files = sorted(p for p in assets.iterdir() if p.is_file())
    unused = [p for p in files if f'assets/{p.name}' not in refs]
    for p in unused:
        p.unlink()
    files = [p for p in files if p not in unused]
    total = sum(p.stat().st_size for p in files) + (out / 'index.html').stat().st_size
    big = max(files, key=lambda p: p.stat().st_size)
    print(f'check: {len(refs)} referenced assets, {len(missing)} missing {missing[:5]}, '
          f'{len(unused)} unused deleted {[p.name for p in unused][:6]}')
    print(f'check: {len(files)} asset files + index.html, total {total / 1e6:.1f} MB, '
          f'largest {big.name} {big.stat().st_size / 1e3:.0f} KB, page {len(page) / 1e3:.0f} KB')
    for c in cfgs:
        n = [p for p in files if p.name.startswith(c.tag + '_')]
        print(f'check: [{c.tag}] {len(n)} files, {sum(p.stat().st_size for p in n) / 1e6:.1f} MB')
    if (missing or len(files) + 1 > a.max_files or total > a.max_mb * 1e6 or len(page.encode()) > 16e6
            or big.stat().st_size > 15e6):
        raise SystemExit('check failed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', action='append', help=f'NES result set, repeatable; first = page default '
                    f'(default {DEFAULT_RES}); cache in <res>/_cache/explorer')
    ap.add_argument('--outcome', action='append', help='LABEL=SUBDIR[:key=value,...], repeatable')
    ap.add_argument('--out', default=None, help='output dir (default <first res>/explorer)')
    ap.add_argument('--steps', default='data,patch,arena,render,page,check')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--max-files', type=int, default=480)
    ap.add_argument('--max-mb', type=float, default=150)
    ap.add_argument('--crf', type=int, default=30, help='H.264 quality of the montages (higher = smaller)')
    a = ap.parse_args()
    res = a.res or DEFAULT_RES
    r0 = (ROOT / res[0]) if not Path(res[0]).is_absolute() else Path(res[0])
    out = Path(a.out) if a.out else r0 / 'explorer'
    cfgs = [Cfg(r, a, out) for r in res]
    for s in a.steps.split(','):
        if s == 'page':
            step_page(cfgs, a.overwrite)
        elif s == 'check':
            step_check(cfgs, a)
        else:
            for c in cfgs:
                globals()[f'step_{s}'](c, a.overwrite)
