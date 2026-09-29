"""
Build the NES explorer page "Exploratory Causal Inference x Mice": <out>/index.html + <out>/assets/.

The page design is scripts/eci/explorer_template.html (the user's hand-edited version of the
published page); this script only fills its `const DATA0 = /*__DATA__*/null` with the data of ONE
NES result set (one SAE / representation) and writes the assets it references.

Rerun for a new representation (one command, all steps, incremental; GPU needed for patch / arena):
    sbatch scripts/eci/build_explorer.sh                     # default --res below
    EXTRA_ARGS="--res results/vision/mice/eci/nes/<sae>" sbatch scripts/eci/build_explorer.sh
or directly: python scripts/eci/build_explorer.py --res results/vision/mice/eci/nes/<sae>
The SAE is the basename of --res (override with --sae); its full per-frame codes
(dataset/mice/v1/eci/codes/<sae>/), its training tokens (tokens_dir in the SAE's metrics.json) and,
for fg448, the per-video backgrounds (codes config.json 'backgrounds') must exist.
Output: <res>/explorer/ (publish the folder: index.html + assets/). The last step checks the limits.

Data-driven: everything comes from the result set (summary.csv files as written by the NES runs). The
first --outcome is the page default; more outcomes add an 'Outcome' selector. Neurons discovered in
ANY outcome get clips, ranked by the codes the outcome is built from (codes_mean for mean-pool
outcomes, codes_max for max-pool outcomes).

Steps (each cached under <res>/_cache/explorer/, incremental: only missing neurons are recomputed):
    data     one pass over the full per-frame codes (src/eci/viz.py scan_codes) for every neuron
             discovered by any outcome's primary setting (prefix 128 / 1024, window full / trim30).
             Per neuron (K = 16 clips each, one per video): top frames (pick_top), least activated
             (every other of pick_least's 2K), median-activation frames (a random frame per video
             with activation between the 45th and 55th percentile of the neuron's active frames,
             K random videos), the activation histogram over ALL frames (40 linear bins from 0 to
             the max; all frames and per genotype|stage), the stage x genotype table. -> picks.json
    patch    per-patch codes of the neuron for every frame of its top and median clips, recomputed
             through DINOv2 + SAE (src/eci/viz.py make_patch_encoder, GPU), checked against the stored
             pooled codes. -> patch/<p>XXXX.npy, patch/<p>XXXX_mid.npy (n_clips * 15, grid, grid)
    arena    arena maps of ALL neurons: per patch position, the mean SAE code over the SAE's training
             frames (fg448: 1 frame per second of every video, codes 0 off the foreground) and how
             often the patch is foreground; plus the arena background image (per-pixel median of the
             videos' stored empty-bedding backgrounds). -> arena.npz, assets/arena_bg.webp
    render   3 s clips (15 frames at 5 fps) tiled 4 per row (200 px tiles, row-major = rank) into
             H.264 mp4 montages: top raw, top with the heatmap overlaid (turbo), median raw, median
             with heatmap, least raw. Re-rendered when the clips change (render.json signatures).
    page     data inlined into scripts/eci/explorer_template.html -> <out>/index.html
    check    every referenced asset exists; unreferenced files in assets/ are deleted; counts, MB
             (fails above --max-files files / --max-mb MB)

Representations (src/eci/viz.py representation): 'crop224' SAEs are patch SAEs on the 224 center
crop (16 x 16 patches, heat on pixels 32-480); 'fg448' SAEs (foreground, src/eci/foreground.py) see
the whole frame at 448 (32 x 32 patches of 16 px) and fire only on foreground patches (others = 0).
Chips: one per single-setting change from the outcome's primary (summary.csv columns; bout_rule 1 =
min-2-frame hysteresis bouts 'min2+hyst'), plus 'size' when <outcome dir>/size_adjusted.csv exists
(round-1 neuron re-tested with the per-video mean foreground size as a covariate).
Artefact flags (neurons 50, 64, 113) belong to the ep20 SAE only.

Outcome spec (default: bout rate (maxpool_bouts/) + mean activation (.) pooled as the run's primary
(selected_neurons.json 'settings'), each when its summary.csv exists):
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
K, HALF, TILE, COLS = 16, 7, 200, 4  # clips per montage, frames each side of the center, tile px, tiles per row
HIST_BINS = 40
PICKS_V = 2                          # picks.json entry version (2 = 16 clips + median clips + histogram)
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
    def __init__(self, a):
        self.res = (ROOT / a.res).resolve() if not Path(a.res).is_absolute() else Path(a.res)
        self.sae = a.sae or self.res.name
        self.out = Path(a.out) if a.out else self.res / 'explorer'
        self.assets = self.out / 'assets'
        self.work = self.res / '_cache' / 'explorer'
        self.outcomes = []
        specs = a.outcome or default_outcomes(self.res)
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
        self.max_files, self.max_mb, self.crf = a.max_files, a.max_mb, a.crf
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
def mid_rows(x, row2vid, n, seed):
    """Up to n rows, at most one per video, whose activation lies between the 45th and 55th percentile
    of the neuron's active (> 0) frames; random frame within each video, random videos, ordered by
    closeness to the median. -> (rows, [q45, q55])"""
    act = x[x > 0]
    if not len(act):
        return np.array([], np.int64), None
    q45, q50, q55 = (float(v) for v in np.quantile(act, [0.45, 0.5, 0.55]))
    cand = np.flatnonzero((x >= q45) & (x <= q55))
    rng = np.random.default_rng([seed, 45])
    cand = cand[rng.permutation(len(cand))]
    _, first = np.unique(row2vid[cand], return_index=True)
    pick = cand[first][rng.permutation(len(first))[:n]]
    pick = pick[np.argsort(np.abs(x[pick] - q50), kind='stable')]
    return pick, [q45, q55]


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
    res.setdefault('by_key', {})
    res.setdefault('frame_path', {})
    want = {k: set() for k in cfg.keys}
    for o in cfg.outcomes:
        want[o['codes']] |= {int(j) for j in primary_rows(o).dropna(subset=['neuron'])['neuron']}
    missing = {k: sorted(j for j in v if res['by_key'].get(k, {}).get(str(j), {}).get('v') != PICKS_V)
               for k, v in want.items()}
    if not any(missing.values()):
        print('data: cached', {k: len(v) for k, v in res['by_key'].items()}, 'neurons')
        return
    from src.eci.viz import load_full_codes, pick_least, pick_top, scan_codes, stage_genotype_table
    src = load_full_codes(cfg.sae, DATASET, ROOT / 'data')
    meta = src.meta
    fp = meta['frame_path'].values
    gkeys = sorted({f'{g}|{int(s)}' for g, s in zip(meta['genotype'], meta['stage'])})
    gid = pd.Index(gkeys).get_indexer(meta['genotype'].astype(str) + '|' + meta['stage'].astype(int).astype(str))
    for key, miss in missing.items():
        if not miss:
            continue
        print(f'data: scanning codes_{key} for neurons', miss, flush=True)
        neurons = np.array(miss, dtype=np.int64)
        scan = scan_codes(src, neurons, key, 1, 2.0, seed=0)
        vids, Z = scan.videos, src.codes[key]
        row2vid = np.searchsorted(vids['lo'].values, np.arange(len(meta)), side='right') - 1
        X = np.empty((len(meta), len(neurons)), np.float32)  # every frame of the selected neurons
        for a in range(0, len(meta), 200_000):
            X[a:a + 200_000] = Z[a:a + 200_000][:, neurons]

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
        for i, j in enumerate(miss):
            tr, _ = pick_top(scan, i, K)
            lr, rule = pick_least(scan, i, 2 * K, 0)
            mr, band = mid_rows(X[:, i], row2vid, K, j)
            edges, counts = activation_hist(X[:, i], gid, len(gkeys))
            tab = stage_genotype_table(scan, i)
            store[str(j)] = {
                'v': PICKS_V,
                'top': [clip(int(r), j) for r in tr[:K]],
                'least': [clip(int(r), j) for r in lr[::2][:K]],
                'mid': [clip(int(r), j) for r in mr], 'mid_band': band,
                'least_rule': rule,
                'firing_rate': float(scan.n_active[i] / scan.n_frames),
                'hist': {'edges': [float(f'{e:.5g}') for e in edges],
                         'counts': dict({'all': [int(c) for c in counts.sum(0)]},
                                        **{g: [int(c) for c in counts[k]] for k, g in enumerate(gkeys)})},
                'table': [{k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if k in ('stage', 'n_pools')
                               else v) for k, v in rec.items()} for rec in tab.to_dict('records')]}
            print(f'  codes_{key} neuron {j}: {len(store[str(j)]["top"])} top, {len(mr)} median '
                  f'(band {band}), {len(store[str(j)]["least"])} least ({rule})', flush=True)
        del X
    need = {r for st in res['by_key'].values() for n in st.values()
            for c in n['top'] + n['least'] + n.get('mid', []) for r in c['rows']}
    res['frame_path'].update({str(r): fp[r] for r in sorted(need) if str(r) not in res['frame_path']})
    cfg.work.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print('data: wrote', out, {k: len(v) for k, v in res['by_key'].items()}, 'neurons')


# ---------------------------------------------------------------------- step: patch
def patch_file(cfg, key, j, kind):
    return cfg.work / 'patch' / f'{pre(key)}{int(j):04d}{"" if kind == "top" else "_" + kind}.npy'


def step_patch(cfg, overwrite):
    picks = json.loads((cfg.work / 'picks.json').read_text())
    (cfg.work / 'patch').mkdir(parents=True, exist_ok=True)
    rows, todo = {}, []
    for k in cfg.keys:
        for j, n in picks['by_key'].get(k, {}).items():
            for kind in ('top', 'mid'):
                rr = [r for c in n.get(kind, []) for r in c['rows']]
                f = patch_file(cfg, k, j, kind)
                if rr and (overwrite or not f.exists() or np.load(f, mmap_mode='r').shape[0] != len(rr)):
                    rows[(k, int(j), kind)] = rr
                    todo.append((k, int(j), kind))
    if not todo:
        print('patch: cached')
        return
    from src.eci.viz import load_full_codes, make_patch_encoder
    allr = sorted({r for v in rows.values() for r in v})
    js = sorted({j for _, j, _ in todo})
    print(f'patch: {len(todo)} (codes, neuron, clip set) maps, {len(allr)} frames through DINOv2 + SAE ({REP})',
          flush=True)
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
    for k, j, kind in todo:
        ii = [pos[r] for r in rows[(k, j, kind)]]
        m = pc[ii, :, :, js.index(j)]
        stored = np.asarray(codes[k][np.array(rows[(k, j, kind)])][:, j], dtype=np.float32)
        if k == 'max':
            pooled = m.max((1, 2))
        elif n_fg is None:
            pooled = m.mean((1, 2))
        else:  # fg448: mean over the foreground patches only
            pooled = m.sum((1, 2)) / np.maximum(n_fg[ii], 1)
        worst[k] = max(worst.get(k, 0.0), float(np.abs(pooled - stored).max()))
        mism[k] = mism.get(k, 0) + int(((pooled > 0) != (stored > 0)).sum())
        np.save(patch_file(cfg, k, j, kind), m.astype(np.float16))
    print(f'patch: recomputed patch maps pooled vs stored codes, max abs diff {worst}, '
          f'active/inactive mismatches {mism} (of {sum(len(v) for v in rows.values())} frame-neuron pairs)')


# ---------------------------------------------------------------------- step: arena
def step_arena(cfg, overwrite):
    """Per patch position: summed SAE codes of every neuron and foreground counts over the SAE's
    training frames (fg448: FgTokenStore shards; crop224: patch_tokens.npy, every patch counts)."""
    import torch
    from src.eci.sae import load_sae
    out = cfg.work / 'arena.npz'
    bg_out = cfg.assets / 'arena_bg.webp'
    sae_dir = DATASET / 'mice/v1/eci/sae' / cfg.sae
    tokens_dir = Path(json.loads((sae_dir / 'metrics.json').read_text())['args']['tokens_dir'])
    if not out.exists() or overwrite:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sae, norm, _ = load_sae(sae_dir / 'sae.pt', dev)
        codes = __import__('src.eci.viz', fromlist=['load_full_codes']).load_full_codes(
            cfg.sae, DATASET, ROOT / 'data').codes['max']
        chk = []
        if REP == 'fg448':
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
                print(f'arena: shard {s + 1}/{len(st.dirs)}', flush=True)
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
            for a in range(0, n_frames, 2048):
                with torch.no_grad():
                    x = torch.from_numpy(np.array(tok[a:a + 2048])).to(dev)
                    z = sae.encode(norm(x.reshape(-1, x.shape[-1])), mode='threshold').view(len(x), P, -1)
                    S += z.double().sum(0)
            occ = None
            note = f'the {n_frames:,} frames of the SAE training sample'
        np.savez(out, act=(S / n_frames).float().cpu().numpy(), grid=grid, n_frames=n_frames, note=note,
                 occ=(occ / n_frames).float().cpu().numpy() if occ is not None else np.zeros(0))
        print(f'arena: {n_frames} frames, grid {grid}; check per-frame max vs stored codes_max: '
              f'max abs diff {chk[0] if chk else "n/a"} over {chk[1] if chk else 0} frames', flush=True)
    else:
        print('arena: cached')
    if not bg_out.exists() or overwrite:  # median over videos of the empty-bedding pixel background
        cfg.assets.mkdir(parents=True, exist_ok=True)
        cc = DATASET / 'mice/v1/eci/codes' / cfg.sae / 'config.json'
        bgd = Path(json.loads(cc.read_text()).get('backgrounds', '')) if cc.exists() else None
        if bgd is not None and bgd.is_dir():
            ims = np.stack([np.load(f)['pix_bg'] for f in sorted(bgd.glob('*.npz'))])
            im = np.median(ims, 0).astype(np.uint8)
            src = f'median of {len(ims)} per-video empty-bedding backgrounds'
        else:  # no stored backgrounds (crop224): median of one frame per video
            meta = pd.read_csv(DATASET / 'mice/v1/annotations.csv', usecols=['observation_id', 'frame_path'])
            fr = meta.groupby('observation_id')['frame_path'].nth(0).values
            im = np.median(np.stack([np.asarray(Image.open(DATASET / f).convert('L')) for f in fr]), 0).astype(np.uint8)
            src = f'median of the first frame of {len(fr)} videos'
        from src.eci.viz import frame_box
        x0, y0, c = frame_box(REP, im.shape[1], im.shape[0])
        im = Image.fromarray(im).crop((int(round(x0)), int(round(y0)), int(round(x0 + c)), int(round(y0 + c))))
        im.resize((384, 384), Image.BILINEAR).save(bg_out, 'WEBP', quality=80)
        print('arena: wrote', bg_out, src)


_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT, size)
    return _fonts[k]


def tile_frames(c, frame_path, vmax, color, maps=None, hvmax=None):
    """15 labelled TILE x TILE frames of one clip; a bar at the bottom shows the activation of the
    current frame relative to vmax. maps: (15, grid, grid) patch codes -> heatmap overlaid (scale hvmax)."""
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
        d.rectangle([0, TILE - 6, TILE, TILE], fill=(0, 0, 0, 255))  # opaque: the page re-reads this bar (row 196)
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


MONTAGES = (('top', 'top', False), ('heat', 'top', True), ('mid', 'mid', False), ('mid_heat', 'mid', True),
            ('least', 'least', False))


def _render_one(task):
    out, clips_, fp, vmax, color, maps_file, crf = task
    maps = hv = None
    if maps_file:  # heat scale: the montage's own max patch code
        maps = np.load(maps_file).astype(np.float32)
        hv = float(maps.max())
    clips = [tile_frames(c, fp, vmax, color, None if maps is None else maps[k * 15:(k + 1) * 15], hv)
             for k, c in enumerate(clips_)]
    while len(clips) % COLS:
        clips.append([np.zeros((TILE, TILE, 3), np.uint8)] * (2 * HALF + 1))
    encode([clips[r:r + COLS] for r in range(0, len(clips), COLS)], out, crf)
    return out


def step_render(cfg, overwrite):
    crf = cfg.crf
    import multiprocessing as mp
    import os
    picks = json.loads((cfg.work / 'picks.json').read_text())
    fp = picks['frame_path']
    cfg.assets.mkdir(parents=True, exist_ok=True)
    man_p = cfg.work / 'render.json'
    man = json.loads(man_p.read_text()) if man_p.exists() and not overwrite else {}
    colors = {'top': (255, 170, 40), 'mid': (255, 170, 40), 'least': (120, 190, 255)}
    tasks, sigs = [], {}
    for key in cfg.keys:
        for j, n in picks['by_key'].get(key, {}).items():
            vmax = n['top'][0]['act'] if n['top'] else 1.0  # bar scale: the neuron's top activation, all rows
            for kind, src, heat in MONTAGES:
                clips_ = n.get(src, [])
                out = cfg.assets / f'{pre(key)}{int(j):04d}_{kind}.mp4'
                sig = json.dumps([REP, TILE, COLS, crf, heat, vmax, [c['center'] for c in clips_]])
                if not clips_ or (out.exists() and man.get(out.name) == sig):
                    continue
                sigs[out.name] = sig
                tasks.append((out, clips_, fp, vmax, colors[src],
                              str(patch_file(cfg, key, j, src)) if heat else None, crf))
    if not tasks:
        print('render: cached')
        return
    nproc = max(1, min(len(tasks), len(os.sched_getaffinity(0))))
    print(f'render: {len(tasks)} montages on {nproc} processes', flush=True)
    with mp.get_context('fork').Pool(nproc) as pool:
        for i, out in enumerate(pool.imap_unordered(_render_one, tasks)):
            man[out.name] = sigs[out.name]
            man_p.write_text(json.dumps(man))
            print(f'render: {i + 1}/{len(tasks)} {out.name}', flush=True)


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
        print(f'page: robustness chips vs stats_detail.csv: {bad} mismatches over {len(det)} rows')

    thr = bout_thresholds(cfg)
    ar = np.load(cfg.work / 'arena.npz')
    grid, act_all, occ = int(ar['grid']), ar['act'], ar['occ']
    bg = 'assets/arena_bg.webp' if (cfg.assets / 'arena_bg.webp').exists() else None
    neurons = {}
    for key in cfg.keys:
        neurons[key] = {}
        for j, n in picks['by_key'][key].items():
            f = f'assets/{pre(key)}{int(j):04d}'
            vmax = n['top'][0]['act'] if n['top'] else 1.0
            t = float(thr[key][int(j)]) if key in thr else None
            e = {'least_rule': n['least_rule'], 'firing_rate': n['firing_rate'],
                 'table': [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in t_.items()} for t_ in n['table']],
                 'artefact': int(j) in ARTEFACT,
                 'top_mp4': f'{f}_top.mp4', 'heat_mp4': f'{f}_heat.mp4', 'least_mp4': f'{f}_least.mp4',
                 'n_top': len(n['top']), 'n_least': len(n['least']), 'n_mid': len(n.get('mid', [])),
                 'hist': dict(n['hist'], thr=t),
                 'bout_thr_bar': (round(min(t / vmax, 1.0), 4) if t is not None and vmax > 0 else None),
                 'arena': {'rows': grid, 'cols': grid, 'act': [sig4(v) for v in act_all[:, int(j)]]}}
            if len(occ):
                e['arena']['occ'] = [sig4(v) for v in occ]
            if bg:
                e['arena']['bg'] = bg
            if n.get('mid'):
                e['mid_mp4'], e['mid_heat_mp4'] = f'{f}_mid.mp4', f'{f}_mid_heat.mp4'
            neurons[key][j] = e
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
    data = {'sae': cfg.sae, 'rep': REP, 'outcomes': outcomes, 'otables': otables,
            'analyses': [analyses[a] for a in ORDER if a in analyses], 'results': results, 'neurons': neurons,
            'artefact_text': art, 'arena_note': str(ar['note'])}
    tpl = TEMPLATE.read_text()
    assert tpl.count('/*__DATA__*/null') == 1, 'template placeholder missing'
    page = tpl.replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
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
    if missing or len(files) + 1 > cfg.max_files or total > cfg.max_mb * 1e6 or len(page.encode()) > 16e6:
        raise SystemExit('check failed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='results/vision/mice/eci/nes/matryoshka_btk_1024_k16_fg448_s0',
                    help='NES result set (summary.csv); cache in <res>/_cache/explorer')
    ap.add_argument('--outcome', action='append', help='LABEL=SUBDIR[:key=value,...], repeatable')
    ap.add_argument('--sae', default=None, help='SAE name (default: basename of --res)')
    ap.add_argument('--out', default=None, help='output dir (default <res>/explorer)')
    ap.add_argument('--steps', default='data,patch,arena,render,page,check')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--max-files', type=int, default=250)
    ap.add_argument('--max-mb', type=float, default=40)
    ap.add_argument('--crf', type=int, default=30, help='H.264 quality of the montages (higher = smaller)')
    a = ap.parse_args()
    cfg = Cfg(a)
    for s in a.steps.split(','):
        globals()[f'step_{s}'](cfg, a.overwrite)
