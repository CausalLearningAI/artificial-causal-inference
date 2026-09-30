"""
Build the NES explorer page "Exploratory Causal Inference x Mice / Ants": <out>/index.html + <out>/assets/.

The page design is scripts/eci/explorer_template.html (the user's hand-edited version of the
published page); this script fills its `const ALL = /*__DATA__*/null` with the core data of one or more
NES result sets (one per SAE / representation, chosen with the page's Model bar: MODELS gives each SAE
its encoder / SAE type / input; the first --res is the page default), writes the rest as data files the
page fetches on demand (assets/data/) and writes the media they reference.

One command, all steps, incremental (GPU needed for the patch and arena steps):
    sbatch scripts/eci/build_explorer.sh
or directly (on a GPU node):
    python scripts/eci/build_explorer.py            # default: mice fg448 + mice ep20 + ants antsfg SAEs
    python scripts/eci/build_explorer.py --res results/vision/mice/eci/nes/<sae> [--res ...]
The SAE is the basename of each --res; its domain (src/eci/domain.py: mice, ants) is the NES root the
result set lies under. Its full per-frame codes (<domain eci dir>/codes/<sae>/), its training tokens
(tokens_dir in the SAE's metrics.json) and, for fg448 representations, the per-video backgrounds (codes
config.json 'backgrounds') must exist. Output: <first res>/explorer/ (publish the folder). The last step
checks the limits (files, MB, 0 missing / 0 unused assets).

Domains: the page's Domain switch shows one domain at a time (its SAEs in the Model bar, its comparisons,
videos and charts). Mice: stage x genotype design, families A (paired stage change) and B (het vs wt),
gene line / sex subgroups. Ants (VIEW): one family, treated vs control videos of one experiment (labelled
with the raw treatment numbers, "t=2 vs t=8"), a confound note for v3_2_vs_8 (recording day), the recording
day on every clip, the top-by-p neurons of every search (VIEW 'top', so comparisons without NES selections
can be browsed) and packed montages (pack_layout): the mice page fills the artifact file limit, so the ants
clips share files (--max-files-new) and neurons beyond that budget are listed without clips.

Outcome controls: Outcome (event rate = bouts/min, average time = per-video mean) x Spatial aggregation
(max pooling, average pooling, SOMP). The core outcomes (--outcome / DEFAULT_OUTCOMES) keep their data
files; extra_outcomes adds the other poolings found in their summary.csv and the sibling result sets
<res>_<aggregation> (<sae>_mean, <sae>_somp) as <tag>_x.json bundles (full cohort only, no clips).

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
             The same rules again among the videos of each contrast the neuron is found in (primary search,
             any prefix, full cohort; 'Clips from: this comparison' on the page), top and least each among
             all the videos of the contrast together: A (e.g. A_wt_1to2) = that genotype's videos of stages
             a and b; B (B_stage3) = the het and wt videos of stage 3. Neurons found only in a subgroup
             search (<res>/subsets/, gene line / sex) get 'all' clips only; if the projected file count
             exceeds --max-files, only their round-1 neurons, and the rest are listed without clips.
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
    render   per neuron x length, montage files of 144 px tiles, 16 per row, one block of K = 16 clips per
             row: per clip source ('all' videos: top, least; each contrast: top, least) each row kind
             raw then with heat (turbo, the neuron's shared scale). Sources are packed whole into files of
             at most 16 blocks (montage_layout; all + 3 contrasts = 16). frame -> one webp still;
             1 s / 3 s -> one H.264 mp4 (5 / 15 frames at 5 fps). -> assets/<tag>_<p>XXXX_<L>_<file>.<ext>
    page     core data (domains, model bar, outcomes, analyses, video list, subgroup sizes and the searches of the
             default SAE's first outcome) inlined into scripts/eci/explorer_template.html -> <out>/index.html
             (< --max-page-kb); the rest fetched by the page when needed (split_data): assets/data/
             <tag>_r_<outcome>.json = searches of one SAE x outcome (full cohort + every subgroup),
             <tag>_<codes>_<k>.json = neuron chunks of ~CHUNK_BYTES (clip lists, histogram, arena map,
             per-video outcome values from the NES caches <res>/_cache/video_summaries_*.npz,
             bout_summaries_max.npz, checked against the tau of summary.csv). The page shows 8 of the K
             clips per row by default (8 / 16 control).
    check    every referenced media / data file exists; unreferenced files in assets/ and assets/data/ are
             deleted; counts, MB (fails above --max-version-files files / --max-mb MB / a file > 15 MB /
             index.html > --max-page-kb KB). --max-files is the media budget of plan_budget.

Page only (CPU, all caches present): python scripts/eci/build_explorer.py --steps page,check

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

from src.eci.domain import get_domain  # noqa: E402

MICE = get_domain('mice')  # stage labels, analysis ids / order, gene lines and sexes, experiment.csv
ANTS = get_domain('ants')  # experiments, treatments, recording days, analysis ids / order
TEMPLATE = ROOT / 'scripts/eci/explorer_template.html'
DATASET = ROOT / 'dataset'
NES = 'results/vision/mice/eci/nes'
DEFAULT_RES = [f'{NES}/matryoshka_btk_1024_k16_fg448_s0', f'{NES}/matryoshka_btk_1024_k16_ep20_s0',
               'results/vision/ants/eci/nes/matryoshka_btk_1024_k16_antsfg_s0']
ARTEFACT_EP20 = {64, 50, 113}  # ep20 SAE neuron ids; other SAEs get no flags
# Model bar of the page: SAE result set -> its value on each axis (encoder, SAE type, input), plus a
# tooltip per value. Adding an encoder / SAE type = one entry here (and its --res); an axis with a single
# value is shown as a fixed selector. The Domain switch shows the SAEs of one domain at a time.
MODEL_AXES = [('encoder', 'Encoder'), ('sae', 'SAE'), ('input', 'Input')]
MODELS = {
    'matryoshka_btk_1024_k16_fg448_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'mice only'},
    'matryoshka_btk_1024_k16_ep20_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'full frame'},
    'matryoshka_btk_1024_k16_antsfg_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'ants only'}}
MODEL_TIPS = {
    ('encoder', 'DINOv2'): 'DINOv2 patch features',
    ('sae', 'Matryoshka'): 'Matryoshka BatchTopK sparse autoencoder, 1024 neurons, k = 16',
    ('input', 'mice only'): 'Whole frame at 448 px; SAE trained on the mouse (foreground) patches only',
    ('input', 'full frame'): '224 px center crop; SAE trained on all patches (mice and bedding)',
    ('input', 'ants only'): 'Whole frame at 448 px; SAE trained on the ant (foreground) patches only'}
# Per domain: page title word, subject words of the tooltips, clip layout and top-by-p list length.
# packed = the SAE's clips of all neurons share montage files (pack_layout, budget --max-files-new):
# the mice SAEs fill the artifact's file limit with one montage set per neuron, so a later domain packs.
# top = per search, the N neurons with the smallest first-round p that NES did not select (listed on the
# page, so comparisons where nothing is selected can still be browsed); 0 = none.
VIEW = {'mice': {'title': 'Mice', 'subject': 'mouse', 'subjects': 'mice', 'packed': False, 'top': 0},
        'ants': {'title': 'Ants', 'subject': 'ant', 'subjects': 'ants', 'packed': True, 'top': 10}}
# The page's two outcome controls. Outcome kind: event rate (bouts per minute above the threshold,
# scripts/eci/run_nes_bouts.py) or average time (per-video mean of the frame code, scripts/eci/run_nes.py).
# Spatial aggregation (how the patch codes of one frame become one value per neuron): the pooling of the
# codes. An (outcome, aggregation) pair is offered when its NES results exist (core outcomes, then
# extra_outcomes), otherwise its button is disabled.
OUTCOME_KINDS = [('rate', 'event rate', 'Bouts per minute: runs of frames above the neuron\'s activity threshold '
                  '(scripts/eci/run_nes_bouts.py)'),
                 ('time', 'average time', 'Per-video mean over time of the neuron\'s frame value '
                  '(scripts/eci/run_nes.py)')]
AGGS = [('max', 'max pooling', 'Frame value = max over the frame\'s patch codes (codes_max)'),
        ('mean', 'average pooling', 'Frame value = mean over the frame\'s patch codes (codes_mean)'),
        ('somp', 'SOMP', 'Simultaneous Orthogonal Matching Pursuit over the frame\'s patch tokens')]
CHUNK_BYTES = 200_000                 # target size of one neuron data file (assets/data/)
K, TILE, COLS = 16, 144, 16           # clips per row block, tile px, tiles per montage row (1 block = 1 row)
MAX_BLOCKS = 16                       # blocks per montage file (16 x 144 px = 2304 px high at most)
LENGTHS = {'frame': 1, '1s': 5, '3s': 15}
# clip sources of a neuron: 'all' (every video) and one per contrast the neuron is found in (aid)
KINDS = {'all': ('top', 'least'), 'cmp': ('top', 'least')}
HIST_BINS = 40
PICKS_V = 5                          # picks.json entry version (5 = one top row per contrast; 4 = + per-contrast selections)
STAGE_LABEL = MICE.stage_label  # 1 -> 'H,S'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FIELDS = ('pooling', 'outcome_type', 'test', 'correction')
KNOWN = set(FIELDS) | {'analysis_id', 'family', 'genotype', 'stage', 'transition', 'prefix', 'window', 'n_units',
                        'setting', 'n_tested_total', 'n_dropped', 'round', 'neuron', 'tau', 'se', 't', 'df', 'p',
                        'threshold', 'n_tested', 'direction', 'experiment', 'control', 'treatment', 'confound'}
# robustness chips: (field, alternative value) -> short label, tooltip
CHIP = {('test', 'signflip'): ('flip', 'sign-flip permutation test instead of the t-test'),
        ('correction', 'bh'): ('BH', 'Benjamini-Hochberg instead of Bonferroni'),
        ('pooling', 'max'): ('max', 'max-pooling over patches instead of mean'),
        ('pooling', 'mean'): ('mean-pool', 'mean-pooling over patches instead of max'),
        ('outcome_type', 'rate'): ('rate', 'firing rate (fraction of frames > 0) instead of mean activation'),
        ('outcome_type', 'mean'): ('mean', 'per-video mean activation as the outcome'),
        ('window', 'matched'): ('match', 'time-matched windows across stages'),
        ('window', 'full'): ('full', 'full videos'),
        ('prefix', 128): ('128', 'searching only the first 128 neurons'),
        ('prefix', 1024): ('1024', 'searching all 1024 neurons')}
LINES, SEXES = MICE.subgroups['line'], MICE.subgroups['sex']
SUBSETS = [f'{l}_{x}' for l in ('all',) + LINES for x in ('all',) + SEXES if (l, x) != ('all', 'all')]
DEFAULT_OUTCOMES = ['Bout rate (max-pool)=maxpool_bouts:pooling=max,outcome_type=bout_rate,threshold_q=0.95,merge_gap=0,unit=bouts/min,short=bouts',
                    'Mean activation ({p}-pool)=.:pooling={p},outcome_type=mean,short={p}-pool']


def default_outcomes(res):
    """DEFAULT_OUTCOMES whose summary.csv exists; the mean-activation outcome uses the run's primary
    pooling as recorded in <res>/selected_neurons.json 'settings' ('primary (codes_max, ...)'), else mean."""
    sel = res / 'selected_neurons.json'
    m = re.search(r'codes_(mean|max)', json.loads(sel.read_text()).get('settings', '')) if sel.exists() else None
    specs = [x.replace('{p}', m[1] if m else 'mean') for x in DEFAULT_OUTCOMES]
    return [x for x in specs if (res / x.split('=', 1)[1].split(':')[0] / 'summary.csv').exists()]


def domain_of(res):
    """The domain whose NES root (src/eci/domain.py nes_root) contains the result set."""
    for d in (MICE, ANTS):
        if res.resolve().is_relative_to(d.nes_root.resolve()):
            return d
    raise SystemExit(f'{res}: not under a domain NES root ({MICE.nes_root}, {ANTS.nes_root})')


def outcome_kind(o):
    return 'rate' if o['primary']['outcome_type'] == 'bout_rate' else 'time'


def extra_outcomes(cfg):
    """The (outcome kind, spatial aggregation) pairs the core outcomes do not cover, when their NES results
    exist: (1) the other poolings of a core outcome in its own summary.csv (the NES robustness grid runs
    full searches for them); (2) sibling result sets <res>_<aggregation> (e.g. <sae>_mean, <sae>_somp, written
    by scripts/eci/run_nes*.py on those codes): their summary.csv (average time) and <subdir>/summary.csv
    (event rate: outcome_type bout_rate, threshold q 0.95 / gap 0 when several), caches in their _cache with
    the run_nes*.py names (video_summaries_<pooling>.npz, bout_summaries_max_<aggregation>.npz). Their neurons
    are listed without clips (no clip budget left); per-video values are skipped with a warning when the
    cache file is missing."""
    have = {(o['kind'], o['agg']) for o in cfg.outcomes}
    agg_ids = [g for g, _, _ in AGGS]
    out = []
    for o in cfg.outcomes:
        t = o['tidy']
        for p in agg_ids:
            if (o['kind'], p) in have or p not in set(t['pooling'].astype(str)):
                continue
            prim = dict(o['primary'], pooling=p)
            if not match(t, prim).any():
                continue
            label = re.sub(r'\((\w+)-pool\)', f'({p}-pool)', o['label']) if '-pool)' in o['label'] else f'{o["label"]} ({p}-pool)'
            out.append({**o, 'label': label, 'primary': prim, 'codes': p, 'agg': p, 'pool': p, 'extra': True,
                        'short': f'{p}-pool', 'id': f'x{o["kind"]}{p}'})
            have.add((o['kind'], p))
    for agg in agg_ids:
        d = cfg.res.parent / f'{cfg.res.name}_{agg}'
        if not d.is_dir():
            continue
        for s in sorted(d.glob('summary.csv')) + sorted(d.glob('*/summary.csv')):
            t = pd.read_csv(s)
            for kind, ot in (('rate', 'bout_rate'), ('time', 'mean')):
                g = t[t['outcome_type'] == ot]
                if (kind, agg) in have or not len(g) or (kind == 'time' and s.parent != d):
                    continue
                kv = {'pooling': str(g['pooling'].iloc[0]), 'outcome_type': ot}
                if kind == 'rate':
                    kv.update({c: v for c, v in (('threshold_q', '0.95'), ('merge_gap', '0')) if c in t.columns})
                try:
                    prim = default_primary(t, kv)
                except SystemExit as e:
                    print(f'[{cfg.tag}] WARNING: {s} not used: {e}')
                    continue
                name = 'SOMP' if agg == 'somp' else f'{agg}-pool'
                out.append({'label': f'{"Bout rate" if kind == "rate" else "Mean activation"} ({name})', 'dir': s.parent,
                            'tidy': t, 'primary': prim, 'codes': agg, 'pool': prim['pooling'], 'kind': kind,
                            'agg': agg, 'extra': True, 'cache': d / '_cache', 'bfile': f'bout_summaries_max_{agg}.npz',
                            'unit': 'bouts/min' if kind == 'rate' else 'activation', 'short': name,
                            'id': f'x{kind}{agg}'})
                have.add((kind, agg))
    for o in out:
        print(f'[{cfg.tag}] extra outcome {o["label"]!r} ({o["kind"]} x {o["agg"]}): {o["dir"]} primary {o["primary"]}')
    return out


class Cfg:
    """One NES result set (one SAE)."""

    def __init__(self, res, a, out):
        self.res = (ROOT / res).resolve() if not Path(res).is_absolute() else Path(res)
        self.sae = self.res.name
        self.tag = self.sae.split('_')[-2]  # 'fg448', 'ep20', 'antsfg': asset file prefix
        self.dom = domain_of(self.res)
        self.view = VIEW[self.dom.name]
        self.vdom = None if self.dom is MICE else self.dom  # src/eci/viz.py domain argument (None = mice v1)
        self.order = self.dom.analysis_ids()
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
            self.outcomes[-1].update(kind=outcome_kind(self.outcomes[-1]), agg=prim['pooling'], pool=prim['pooling'],
                                     extra=False, cache=self.res / '_cache')
            print(f'[{self.tag}] outcome {label!r}: {d} primary {prim} codes_{codes}')
        self.extras = extra_outcomes(self)
        # subgroup result sets (scripts/eci/run_nes*.py --line/--sex --primary-only): the same outcomes
        # read from <res>/subsets/<line>_<sex>/<SUBDIR>/, when their summary.csv exists
        self.subsets = {}
        for name in SUBSETS:
            outs = []
            for o in self.outcomes:
                d = self.res / 'subsets' / name / o['dir'].relative_to(self.res)
                if (d / 'summary.csv').exists():
                    outs.append({**o, 'dir': d, 'tidy': pd.read_csv(d / 'summary.csv')})
            if len(outs) == len(self.outcomes):
                self.subsets[name] = outs
            elif outs:
                raise SystemExit(f'subset {name}: only {len(outs)} of {len(self.outcomes)} outcomes present')
        print(f'[{self.tag}] subgroups: {sorted(self.subsets)}')
        self.noclip = set()  # (codes key, neuron) found only in subgroups and left without clips (file budget)
        self.clipplan = None  # packed SAEs: {codes key: {neuron: contrasts}} given clips (plan_packed)
        self.keys = sorted({o['codes'] for o in self.outcomes})
        from src.eci.viz import representation
        self.rep = representation(self.sae, DATASET, domain=self.vdom)
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
    k = (str(o['cache']), o['pool'], w, q, g)
    if k not in _rates:
        f = np.load(o['cache'] / o.get('bfile', f'bout_summaries_{o["pool"]}.npz'))
        c, nf = f[f'{w}__{q:g}__{g}__count'], f[f'{w}__n_frames']
        _rates[k] = (pd.Index(f['observation_id'].astype(str)), c / (nf[:, None] / fps / 60.0))
    return _rates[k]


def video_outcome(cfg, o):
    """(Index of observation_id, (n_obs, m)) per-video value of the tested outcome, full videos, as the
    NES runs computed it: bouts/min, or the per-video mean of codes_<pooling> (from the outcome's result
    set _cache). Extra outcomes whose cache file is missing -> None (page: no per-video values)."""
    f = o['cache'] / (o.get('bfile', f'bout_summaries_{o["pool"]}.npz') if is_bout(o) else f'video_summaries_{o["pool"]}.npz')
    if o['extra'] and not f.exists():
        print(f'[{cfg.tag}] WARNING: {f} missing: no per-video values for {o["label"]!r}')
        return None
    if is_bout(o):
        return bout_rates(cfg, o, 'full')
    if o['primary']['outcome_type'] != 'mean':
        raise SystemExit(f'per-video values: outcome_type {o["primary"]["outcome_type"]!r} not supported')
    f = np.load(f)
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
    return [int(x) for x in MICE.analysis(aid).stages]


def wanted(cfg, subsets=True):
    """{codes key: {neuron: sorted contrasts (analysis ids) it is found in}}, over the primary searches
    (any prefix, full window) of the outcomes ranked by that key. Neurons found only in a subgroup search
    get 'all videos' clips only (no contrasts: 'this comparison' clips stay full-cohort), unless they are
    in cfg.noclip (left without clips to respect the file budget). Packed SAEs: cfg.clipplan."""
    if cfg.clipplan is not None:
        return {k: {j: sorted(v, key=cfg.order.index) for j, v in w.items()} for k, w in cfg.clipplan.items()}
    want = {k: {} for k in cfg.keys}
    for o in cfg.outcomes:
        p = primary_rows(o).dropna(subset=['neuron'])
        for aid, j in zip(p['analysis_id'], p['neuron'].astype(int)):
            want[o['codes']].setdefault(int(j), set()).add(aid)
    if subsets:
        for k, j in subgroup_neurons(cfg):
            if (k, j) not in cfg.noclip:
                want[k].setdefault(j, set())
    return {k: {j: sorted(v) for j, v in w.items()} for k, w in want.items()}


def subgroup_neurons(cfg, round1_only=False):
    """[(codes key, neuron)] found in any subgroup's primary search (full window, any prefix) but not in
    the full cohort, sorted."""
    full = {(o['codes'], int(j)) for o in cfg.outcomes for j in primary_rows(o)['neuron'].dropna()}
    out = set()
    for outs in cfg.subsets.values():
        for o in outs:
            p = primary_rows(o).dropna(subset=['neuron'])
            if round1_only:
                p = p[p['round'] == 1]
            out |= {(o['codes'], int(j)) for j in p['neuron']} - full
    return sorted(out)


def n_files(n_contrasts):
    """Montage files per length of a neuron with clips from all videos + n_contrasts contrasts
    (montage_layout packing: 4 blocks for 'all', 6 per contrast, at most MAX_BLOCKS per file)."""
    files, cur = 0, MAX_BLOCKS + 1
    for b in [2 * len(KINDS['all'])] + [2 * len(KINDS['cmp'])] * n_contrasts:
        if cur + b > MAX_BLOCKS:
            files, cur = files + 1, 0
        cur += b
    return files


def subgroup_hits(cfg):
    """{(codes key, neuron): (number of round-1 hits over the subgroup searches, best p)} of the
    subgroup-only neurons (primary, full window, any prefix / outcome / analysis)."""
    full = {(o['codes'], int(j)) for o in cfg.outcomes for j in primary_rows(o)['neuron'].dropna()}
    out = {}
    for outs in cfg.subsets.values():
        for o in outs:
            p = primary_rows(o).dropna(subset=['neuron'])
            for j, pv in zip(p[p['round'] == 1]['neuron'].astype(int), p[p['round'] == 1]['p']):
                k = (o['codes'], int(j))
                if k not in full:
                    n, b = out.get(k, (0, 1.0))
                    out[k] = (n + 1, min(b, float(pv)))
    return out


def plan_budget(cfgs, max_files):
    """Projected asset files (+ index.html). If the subgroup-only neurons do not all fit, only their
    round-1 neurons are candidates, ranked over both SAEs by the number of round-1 subgroup hits, then
    best p; they get clips in that order while the total stays <= max_files. The others are listed
    (cfg.noclip) and shown on the page without clips."""
    def total():
        t = 1 + len(cfgs)  # index.html + one arena background per SAE
        for c in cfgs:
            for k, w in wanted(c).items():
                t += sum(len(LENGTHS) * n_files(len(a)) for a in w.values())
        return t
    for c in cfgs:
        c.noclip = set()
    t = total()
    print(f'budget: {t} files projected with every subgroup-only neuron '
          f'({sum(len(subgroup_neurons(c)) for c in cfgs)}) given clips; limit {max_files}')
    if t > max_files:
        for c in cfgs:
            c.noclip = set(subgroup_neurons(c))
        t = total()
        cand = sorted(((h, c, k) for c in cfgs for k, h in subgroup_hits(c).items()),
                      key=lambda x: (-x[0][0], x[0][1], x[1].tag, x[2]))
        for h, c, k in cand:
            if t + len(LENGTHS) * n_files(0) > max_files:
                break
            c.noclip.discard(k)
            t += len(LENGTHS) * n_files(0)
        print(f'budget: {t} files; subgroup-only neurons with clips: '
              f'{sum(len(subgroup_neurons(c)) - len(c.noclip) for c in cfgs)} of '
              f'{sum(len(subgroup_neurons(c)) for c in cfgs)} (round-1 ones ranked by hit count, then p)')
    for c in cfgs:
        if c.noclip:
            print(f'budget: [{c.tag}] left without clips ({len(c.noclip)}): {sorted(c.noclip)}')
    return t


def plan_packed(cfgs, max_files):
    """Clips of the packed SAEs (VIEW 'packed'): neurons in priority order get clips while the SAE's montage
    files (pack_layout, len(LENGTHS) per packed file) stay <= max_files: first the neurons selected by a
    core outcome's primary search (full window, any prefix; best p first), with the contrasts they are
    selected in; then the top-by-p lists (VIEW 'top') of the core outcomes, rank 1 of every list, then rank
    2, ..., each adding its contrast. Stops at the first candidate that does not fit. The rest are listed
    on the page without clips."""
    for c in cfgs:
        cand = {}  # (codes key, neuron) -> (best p, contrasts), selected neurons
        for o in c.outcomes:
            p = primary_rows(o).dropna(subset=['neuron'])
            for aid, j, pv in zip(p['analysis_id'], p['neuron'].astype(int), p['p']):
                b, s = cand.get((o['codes'], j), (1.0, set()))
                cand[(o['codes'], j)] = (min(b, float(pv)), s | {aid})
        seq = [(k, j, s) for (k, j), (b, s) in sorted(cand.items(), key=lambda x: (x[1][0], x[0]))]
        lists = [[(o['codes'], j, {aid}) for j in L] for o in c.outcomes for (aid, _, _), L in
                 sorted(top_lists(c, o).items(), key=lambda x: (c.order.index(x[0][0]), -x[0][1]))]
        seq += [L[r] for r in range(max(map(len, lists), default=0)) for L in lists if r < len(L)]
        plan = {k: {} for k in c.keys}
        for k, j, s in seq:
            trial = {kk: {jj: set(v) for jj, v in w.items()} for kk, w in plan.items()}
            trial[k].setdefault(j, set()).update(s)
            if trial == plan:
                continue
            if len(LENGTHS) * sum(len(pack_layout(c, kk, w)) for kk, w in trial.items()) > max_files:
                break
            plan = trial
        c.clipplan = plan
        n = sum(len(w) for w in plan.values())
        print(f'[{c.tag}] packed clips: {n} neurons, '
              f'{len(LENGTHS) * sum(len(pack_layout(c, k, w)) for k, w in plan.items())} media files '
              f'(limit {max_files}): ' + ', '.join(f'codes_{k} {sorted(w)}' for k, w in plan.items()))


def pack_layout(cfg, key, clips):
    """Packed montage files of one SAE and codes key: [[(neuron, source, block name, kind, heat), ...] per
    file]. clips = {neuron: contrasts} in priority order (plan_packed / picks order); per neuron the sources
    'all' then its contrasts (cfg.order), each packed whole (all its blocks in one file) into files of at
    most MAX_BLOCKS blocks, in that order."""
    files = []
    for j, aids in clips.items():
        for src in ['all'] + sorted(aids, key=cfg.order.index):
            bl = [(int(j), src, f'{k}{"_heat" if h else ""}', k, h) for k in kinds(src) for h in (False, True)]
            if not files or len(files[-1]) + len(bl) > MAX_BLOCKS:
                files.append([])
            files[-1] += bl
    return files


def read_json(f):
    """json file content, cached (result.json files are read once per build)."""
    k = f'json_{f}'
    if k not in _rates:
        _rates[k] = json.loads(Path(f).read_text())
    return _rates[k]


def top_lists(cfg, o):
    """{(analysis id, prefix, window): [neuron, ...]} the VIEW 'top' neurons with the smallest first-round
    p of each primary search (full window) of outcome o that NES did not select, from
    <outcome dir>/<analysis>/result.json [setting] 'first_round'; {} when VIEW 'top' is 0."""
    if not cfg.view['top']:
        return {}
    out = {}
    prim = primary_rows(o)
    for (aid, prefix, window), h in prim.groupby(['analysis_id', 'prefix', 'window']):
        f = o['dir'] / aid / 'result.json'
        fr = read_json(f).get(h['setting'].iloc[0], {}).get('first_round') if f.exists() else None
        if fr is None:
            print(f'WARNING: {f} [{h["setting"].iloc[0]}] missing: no top-by-p list for {o["label"]!r} {aid} p{prefix}')
            continue
        sel = set(h['neuron'].dropna().astype(int))
        order = np.argsort(np.asarray(fr['p'], dtype=np.float64), kind='stable')
        out[(aid, int(prefix), window)] = [int(fr['neuron'][i]) for i in order
                                           if int(fr['neuron'][i]) not in sel][:cfg.view['top']]
    return out


def top_rows(cfg, o, aid, prefix, window):
    """[[neuron, tau, p], ...] of top_lists (first-round tau / p, as in result.json)."""
    L = top_lists(cfg, o).get((aid, int(prefix), window), [])
    if not L:
        return []
    prim = primary_rows(o)
    h = prim[(prim['analysis_id'] == aid) & (prim['prefix'] == prefix) & (prim['window'] == window)]
    fr = read_json(o['dir'] / aid / 'result.json')[h['setting'].iloc[0]]['first_round']
    at = {int(j): i for i, j in enumerate(fr['neuron'])}
    return [[j, sig4(fr['tau'][at[j]]), float(f'{fr["p"][at[j]]:.4g}')] for j in L]


def kinds(src):
    return KINDS['all' if src == 'all' else 'cmp']


def contrast_groups(cfg, aid, vids):
    """The videos of one contrast, as (top0 mask, top1 mask) over the rows of vids: A (e.g. A_wt_1to2) =
    that genotype's videos in stage a / stage b; B (B_stage3) = het / wt videos of that stage; ants
    (v3_2_vs_8) = the control / treatment videos of that experiment. 'least' uses the union."""
    an = cfg.dom.analysis(aid)
    if cfg.dom is not MICE:
        m = np.ones(len(vids), bool)
        for k, v in an.where.items():
            m &= (vids[k] == v).values
        return m & vids[an.arm].isin(an.control).values, m & vids[an.arm].isin(an.treatment).values
    st, gen = vids['stage'].astype(int).values, vids['genotype'].astype(str).values
    if an.family == 'A':
        g, (a, b) = an.genotype, an.stages
        return (gen == g) & (st == a), (gen == g) & (st == b)
    s = an.where['stage']
    return (gen == 'het') & (st == s), (gen == 'wt') & (st == s)


# ---------------------------------------------------------------------- step: data
GROUP_COLS = {'mice': ('genotype', 'stage'), 'ants': ('experiment', 'T')}  # histogram / table groups


def day_label(d):
    """Recording day: v2 dates '18.04.2024' -> '18.04', v3 days 'A' / 'B' / 'C' unchanged."""
    return d[:5] if re.match(r'\d\d\.\d\d\.\d{4}$', d) else d


def clip_group(x):
    """Group of a clip in the logs: 'het|S2' (mice), 'v3 t=8 day C' (ants)."""
    return f'{x["genotype"]}|S{x["stage"]}' if 'stage' in x else f'{x["experiment"]} t={x["T"]} day {x["day"]}'


def group_table(vids, y):
    """Ants: per (experiment, T) the mean over videos of the per-video values y, 95% t-CI, n videos."""
    from scipy import stats
    rows = []
    for (e, t), g in pd.DataFrame({'e': vids['experiment'].astype(str), 't': vids['T'].astype(int),
                                   'y': y}).groupby(['e', 't']):
        v = g['y'].values.astype(np.float64)
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        rows.append({'experiment': e, 'T': int(t), 'mean': float(v.mean()), 'lo': float(v.mean() - h),
                     'hi': float(v.mean() + h), 'n': int(len(v))})
    return rows


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
    missing = {k: sorted(j for j, aids in v.items() if res['by_key'][k].get(str(j), {}).get('v') != PICKS_V
                         or sorted(res['by_key'][k][str(j)]['clips']) != sorted(['all'] + aids))
               for k, v in want.items()}
    if not any(missing.values()):
        print(f'[{cfg.tag}] data: cached', {k: len(v) for k, v in res['by_key'].items()}, 'neurons')
        out.write_text(json.dumps(res))
        return
    from types import SimpleNamespace
    from src.eci.viz import (CodeSource, load_full_codes, pick_least_windows, pick_top_windows, scan_windows,
                             stage_genotype_table, subset_windows)
    src = load_full_codes(cfg.sae, DATASET, ROOT / 'data', domain=cfg.vdom)
    meta = src.meta
    fp = meta['frame_path'].values
    g0, g1 = GROUP_COLS[cfg.dom.name]  # histogram groups: 'genotype|stage' (mice), 'experiment|T' (ants)
    gkeys = sorted({f'{g}|{int(s)}' for g, s in zip(meta[g0], meta[g1])})
    gid = pd.Index(gkeys).get_indexer(meta[g0].astype(str) + '|' + meta[g1].astype(int).astype(str))
    obs, fidx = meta['observation_id'].values, meta['frame_idx'].values
    if cfg.dom is MICE:
        stg, gen, pool = meta['stage'].values, meta['genotype'].values, meta['pool'].astype(str).values
    else:
        exp_, trt, day = meta['experiment'].astype(str).values, meta['T'].values, meta['recording_date'].astype(str).values
    for key, miss in missing.items():
        if not miss:
            continue
        print(f'[{cfg.tag}] data: reading codes_{key} of neurons', miss, flush=True)
        neurons = np.array(miss, dtype=np.int64)
        Z = src.codes[key]
        X = np.empty((len(meta), len(neurons)), np.float32)  # every frame of the selected neurons
        for a in range(0, len(meta), 200_000):
            X[a:a + 200_000] = Z[a:a + 200_000][:, neurons]
        loc = CodeSource('sel', {key: X}, meta, DATASET, dict(src.info))
        wss = scan_windows(loc, np.arange(len(neurons)), key, tuple(LENGTHS.values()), seed=0)
        for ws in wss.values():
            ws.neurons = neurons  # column i of X is neuron neurons[i] (seeds use the neuron id)
        vids = next(iter(wss.values())).videos
        vmean = np.stack([X[lo:hi].mean(0) for lo, hi in zip(vids['lo'], vids['hi'])])
        store = res['by_key'].setdefault(key, {})
        for i, j in enumerate(miss):
            def clip(start, w):
                tr = X[start:start + w, i]
                c = {'start': int(start), 'rows': list(range(int(start), int(start) + w)),
                     'act': float(tr.mean()), 'trace': [float(x) for x in tr], 'obs': str(obs[start]),
                     'frame': int(fidx[start])}
                if cfg.dom is MICE:
                    c.update(stage=int(stg[start]), genotype=str(gen[start]), pool=str(pool[start]))
                else:
                    c.update(experiment=exp_[start], T=int(trt[start]), day=day_label(day[start]))
                return c
            clips = {'all': {}}
            for aid in want[key][j]:
                clips[aid] = {}
            for L, w in LENGTHS.items():
                ts, _ = pick_top_windows(wss[w], i, K)
                ls, rule = pick_least_windows(wss[w], i, K, seed=0)
                clips['all'][L] = {'top': [clip(s, w) for s in ts], 'least': [clip(s, w) for s in ls],
                                   'least_rule': rule}
                for aid in want[key][j]:  # the same rules, among the videos of that contrast only
                    m0, m1 = contrast_groups(cfg, aid, vids)
                    ts, _ = pick_top_windows(subset_windows(wss[w], m0 | m1), i, K)  # one row: both groups
                    c = {'top': [clip(s, w) for s in ts]}
                    assert all(x['obs'] in set(vids['observation_id'][m0 | m1]) for x in c['top'])
                    ls, rule = pick_least_windows(subset_windows(wss[w], m0 | m1), i, K, seed=0)
                    c['least'], c['least_rule'] = [clip(s, w) for s in ls], rule
                    assert all(x['obs'] in set(vids['observation_id'][m0 | m1]) for x in c['least'])
                    clips[aid][L] = c
                for sname, cs in clips.items():
                    c = cs[L]
                    assert all(len({x['obs'] for x in c[k]}) == len(c[k]) for k in kinds(sname))
                    if c['least_rule'] == 'silent':
                        assert all(max(x['trace']) == 0 for x in c['least'])
            edges, counts = activation_hist(X[:, i], gid, len(gkeys))
            if cfg.dom is MICE:
                tab = stage_genotype_table(SimpleNamespace(videos=vids, video_mean=vmean), i).to_dict('records')
            else:
                tab = group_table(vids, vmean[:, i])
            store[str(j)] = {
                'v': PICKS_V, 'clips': clips,
                'firing_rate': float((X[:, i] > 0).mean()),
                'max_frame': float(X[:, i].max()),
                'hist': {'edges': [float(f'{e:.5g}') for e in edges],
                         'counts': dict({'all': [int(c) for c in counts.sum(0)]},
                                        **{g: [int(c) for c in counts[k]] for k, g in enumerate(gkeys)})},
                'table': [{k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if k in ('stage', 'n_pools')
                               else v) for k, v in rec.items()} for rec in tab]}
            print(f'  [{cfg.tag}] codes_{key} neuron {j}: ' + ', '.join(
                f'{L} top {len(c["top"])} (mean {c["top"][0]["act"]:.3g}..{c["top"][-1]["act"]:.3g}) '
                f'least {len(c["least"])} {c["least_rule"]} (max {max(max(x["trace"]) for x in c["least"]):.3g})'
                for L, c in clips['all'].items() if c['top']), flush=True)
            for aid in want[key][j]:
                print(f'      {aid}: ' + ', '.join(
                    f'{L} top {len(c["top"])} (' + ', '.join(f'{g} {n}' for g, n in sorted(__import__('collections').Counter(clip_group(x) for x in c['top']).items())) + f') least {len(c["least"])} {c["least_rule"]}'
                    for L, c in clips[aid].items()), flush=True)
        del X
    res['frame_path'] = {str(r): fp[r] for r in sorted({r for st in res['by_key'].values() for n in st.values()
                                                        for rr in clip_rows(n).values() for r in rr})}
    cfg.work.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(f'[{cfg.tag}] data: wrote', out, {k: len(v) for k, v in res['by_key'].items()}, 'neurons')


# ---------------------------------------------------------------------- step: patch
def patch_file(cfg, key, j):
    return cfg.work / 'patch' / f'{pre(key)}{int(j):04d}.npz'


def clip_rows(n):
    """{'<source>__<L>__<kind>': rows of every clip, in clip order} of one neuron (source = 'all' or a
    contrast id)."""
    return {f'{src}__{L}__{kind}': [r for x in c[kind] for r in x['rows']] for src, cs in n['clips'].items()
            for L, c in cs.items() for kind in kinds(src)}


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
    pe = make_patch_encoder(cfg.sae, dataset_dir=DATASET, domain=cfg.vdom)
    codes = load_full_codes(cfg.sae, DATASET, ROOT / 'data', domain=cfg.vdom).codes
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
    sae_dir = cfg.dom.eci_dir / 'sae' / cfg.sae
    tokens_dir = Path(json.loads((sae_dir / 'metrics.json').read_text())['args']['tokens_dir'])
    if not out.exists() or overwrite:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sae, norm, _ = load_sae(sae_dir / 'sae.pt', dev)
        chk = []
        if cfg.rep == 'fg448':
            codes = __import__('src.eci.viz', fromlist=['load_full_codes']).load_full_codes(
                cfg.sae, DATASET, ROOT / 'data', domain=cfg.vdom).codes['max']
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
        cc = cfg.dom.eci_dir / 'codes' / cfg.sae / 'config.json'
        bgd = Path(json.loads(cc.read_text()).get('backgrounds', '')) if cc.exists() else None
        if bgd is not None and bgd.is_dir() and str(bgd) != '.':
            ims = np.stack([np.load(f)['pix_bg'] for f in sorted(bgd.glob('*.npz'))])
            im = np.median(ims, 0).astype(np.uint8)
            src = f'median of {len(ims)} per-video empty-bedding backgrounds'
        else:  # no stored backgrounds (crop224): median of one frame per video
            meta = pd.read_csv(cfg.dom.ann_path, usecols=['observation_id', 'frame_path'])
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


def tile_text(c):
    """Label burnt into a clip's tiles: 'S2 O,S · het · rd11_2 Test' (mice), 'v3 t=8 · day C · 3_21_4' (ants)."""
    if 'stage' in c:
        return f'S{c["stage"]} {STAGE_LABEL[c["stage"]]} · {c["genotype"]} · {short_obs(c["obs"])[0]}'
    return f'{c["experiment"]} t={c["T"]} · day {c["day"]} · {c["obs"]}'


def tile_frames(c, frame_path, vmax, color, rep, maps=None, hvmax=None):
    """len(c['rows']) labelled TILE x TILE frames of one clip; a bar at the bottom shows the activation
    of the current frame relative to vmax. maps: (w, grid, grid) patch codes -> heatmap overlaid (scale hvmax)."""
    from src.eci.viz import _overlay_rgb, frame_box
    import matplotlib
    matplotlib.use('Agg')
    cmap = matplotlib.colormaps['turbo']
    frames = []
    for t, r in enumerate(c['rows']):
        im = Image.open(DATASET / frame_path[str(r)]).convert('RGB')
        if maps is not None:
            a = np.asarray(im)
            im = Image.fromarray(_overlay_rgb(a, maps[t].astype(np.float32), frame_box(rep, a.shape[1], a.shape[0]),
                                              hvmax, cmap))
        im = im.resize((TILE, TILE), Image.BILINEAR)
        d = ImageDraw.Draw(im, 'RGBA')
        d.rectangle([0, 0, TILE, 12], fill=(0, 0, 0, 140))
        d.text((3, 1), tile_text(c), font=font(8), fill=(255, 255, 255))
        d.rectangle([0, TILE - 6, TILE, TILE], fill=(0, 0, 0, 255))  # opaque: the page re-reads this bar (row T-4)
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


def montage_layout(cfg, n):
    """The montage files of one neuron and length: [[(source, block name, kind, heat), ...] per file].
    Blocks of a source: each of its kinds raw then with heat (top, least). Sources ('all' first, then the contrasts in cfg.order) are packed whole into files of at
    most MAX_BLOCKS blocks, so a view (one source) always reads one file."""
    srcs = ['all'] + sorted((x for x in n['clips'] if x != 'all'), key=cfg.order.index)
    files = []
    for src in srcs:
        bl = [(src, f'{k}{"_heat" if h else ""}', k, h) for k in kinds(src) for h in (False, True)]
        if not files or len(files[-1]) + len(bl) > MAX_BLOCKS:
            files.append([])
        files[-1] += bl
    return files


def asset_name(cfg, key, j, L, f):
    return f'{cfg.tag}_{pre(key)}{int(j):04d}_{L}_{f}.{"webp" if L == "frame" else "mp4"}'


def montages(cfg, key, picks):
    """Every montage file of one SAE and codes key: [(asset name of length L = fn(L), [(neuron, source, block
    name, kind, heat), ...])]. One set per neuron (montage_layout, mice), or the neurons packed together
    (pack_layout, VIEW 'packed', in plan order)."""
    by = picks['by_key'].get(key, {})
    if cfg.view['packed']:
        want = wanted(cfg).get(key, {})
        assert set(map(int, by)) == set(want), f'[{cfg.tag}] picks {sorted(map(int, by))} != plan {sorted(want)}'
        return [(lambda L, f=f: f'{cfg.tag}_{pre(key)}pack_{L}_{f}.{"webp" if L == "frame" else "mp4"}', bl)
                for f, bl in enumerate(pack_layout(cfg, key, want))]
    return [(lambda L, j=j, f=f: asset_name(cfg, key, j, L, f), [(int(j), *b) for b in bl])
            for j, n in by.items() for f, bl in enumerate(montage_layout(cfg, n))]


def _render_one(task):
    """task: (out, blocks [(source, name, kind, heat, clips, bar vmax, patch file)], frame paths, rep, L, crf);
    each block uses its neuron's bar scale and patch maps (heat scale = that neuron's patch-file vmax)."""
    out, blocks, fp, rep, L, crf = task
    w = LENGTHS[L]
    pzs = {}
    blank = [np.zeros((TILE, TILE, 3), np.uint8)] * w
    tiles = []
    for src, _, kind, heat, clips, vmax, maps_file in blocks:
        if maps_file not in pzs:
            pzs[maps_file] = np.load(maps_file)
        pz = pzs[maps_file]
        hv = float(pz['vmax'])
        maps = pz[f'{src}__{L}__{kind}'].astype(np.float32) if heat else None
        color = (120, 190, 255) if kind == 'least' else (255, 170, 40)
        cl = [tile_frames(c, fp, vmax, color, rep, None if maps is None else maps[k * w:(k + 1) * w], hv)
              for k, c in enumerate(clips)]
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
        by = picks['by_key'].get(key, {})
        # bar scale: the neuron's highest frame, every row, length and source; heat scale: its patch file
        vm = {int(j): n['max_frame'] or 1.0 for j, n in by.items()}
        pf = {int(j): patch_file(cfg, key, j) for j in by}
        hv = {j: float(np.load(f)['vmax']) for j, f in pf.items()}
        for name, bl in montages(cfg, key, picks):
            for L in LENGTHS:
                out = cfg.assets / name(L)
                blocks = [(src, bn, k, h, by[str(j)]['clips'][src][L][k], vm[j], str(pf[j])) for j, src, bn, k, h in bl]
                if cfg.view['packed']:
                    sig = json.dumps([cfg.rep, TILE, COLS, K, cfg.crf, [(j, src, bn, vm[j], hv[j], [x['start'] for x in c])
                                                                       for (j, src, bn, _, _), (_, _, _, _, c, _, _) in zip(bl, blocks)]])
                else:  # one neuron per file (the historical signature)
                    j = bl[0][0]
                    sig = json.dumps([cfg.rep, TILE, COLS, K, cfg.crf, vm[j], hv[j],
                                      [(src, bn, [x['start'] for x in c]) for src, bn, _, _, c, _, _ in blocks]])
                if out.exists() and man.get(out.name) == sig:
                    continue
                sigs[out.name] = sig
                tasks.append((out, blocks, fp, cfg.rep, L, cfg.crf))
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


def robustness(o, others, aid, prefix, window, neuron, words=('mouse', 'mice')):
    """Chips for one discovered neuron: is it still selected (any round) when ONE setting changes
    (or, cross-outcome, by another outcome's primary search)? 'Y' / 'N', or '-' when not run. The size
    check (size_adjusted.csv) belongs to the core outcome's primary pooling: no chip for extra outcomes."""
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
    sz = None if o['extra'] else size_table(o)
    if sz is not None:
        g = sz[(sz['analysis_id'] == aid) & (sz['prefix'] == prefix) & (sz['window'] == window) & (sz['neuron'] == neuron)]
        val = ('Y' if bool(g['survives'].iloc[0]) else 'N') if len(g) else '-'
        chips.append(['size-adj', f'round-1 test repeated with the per-video mean foreground size (number of {words[0]} '
                      f'patches, a proxy for how spread out or huddled the {words[1]} are) as a covariate; "not run" = '
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


ORDER = MICE.analysis_ids()  # A_het_1to2 .. A_wt_5to6, B_stage1 .. B_stage6


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


def tau_check(vm, vals, results, oid, round_=1, dom=MICE):
    """max relative |tau(summary.csv) - raw contrast of the per-video values| over the page's round-`round_`
    rows: A = mean over pools of (stage b - stage a), B = mean het - mean wt; ants = mean treatment - mean
    control videos of the experiment. Only round 1 of an
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
            if dom is not MICE:
                an = dom.analysis(aid)
                d = vm.assign(y=y[vm.index])
                d = d[(d[list(an.where)] == pd.Series(an.where)).all(1)]
                t = float(d[d[an.arm].isin(an.treatment)]['y'].mean() - d[d[an.arm].isin(an.control)]['y'].mean())
            elif aid.startswith('A'):
                g, (a, b) = aid.split('_')[1], stages_of(aid)
                d = vm[vm['genotype'] == g].assign(y=y[vm.index[vm['genotype'] == g]])
                p = d.pivot_table(index='pool', columns='stage', values='y')
                t = float((p[b] - p[a]).mean())
            else:
                s = int(aid[len('B_stage'):])
                d = vm.assign(y=y[vm.index])[vm['stage'] == s]
                t = float(d[d['genotype'] == 'het']['y'].mean() - d[d['genotype'] == 'wt']['y'].mean())
            worst = max(worst, abs(t - r['tau']) / max(abs(r['tau']), 1e-9))
    return worst


def video_table():
    """The page's video list (video_meta order): [[pool, stage, genotype, line, sex] ...], [[name, time] ...]
    (short_obs), and {observation_id: index}."""
    vm = vmeta().reset_index(drop=True)
    ex = pd.read_csv(MICE.experiment_csv).drop_duplicates('pool').set_index('pool')
    vm['line'], vm['sex'] = ex.loc[vm['pool'], 'line'].values, ex.loc[vm['pool'], 'sex'].values
    videos = [[str(p), int(s), g, l, x] for p, s, g, l, x in zip(vm['pool'], vm['stage'], vm['genotype'], vm['line'],
                                                                  vm['sex'])]
    return vm, videos, [list(short_obs(o)) for o in vm['observation_id'].astype(str)], \
        {o: i for i, o in enumerate(vm['observation_id'].astype(str))}


def video_table_ants():
    """Ants video list (experiment.csv order): [[experiment, T, recording day, batch, position] ...],
    [[observation id, 'day <d>'] ...], {observation_id: index}."""
    vm = ANTS.video_meta().reset_index(drop=True)
    vm['day'] = [day_label(str(d)) for d in vm['recording_date']]
    videos = [[str(e), int(t), d, str(b), int(p)] for e, t, d, b, p in zip(vm['experiment'], vm['T'], vm['day'],
                                                                          vm['batch'], vm['position'])]
    return vm, videos, [[str(o), f'day {d}'] for o, d in zip(vm['observation_id'], vm['day'])], \
        {o: i for i, o in enumerate(vm['observation_id'].astype(str))}


def page_data_clip(x):
    """[video index (page video list of the clip's domain), mean activation] of one clip; the page builds
    the tooltip label 'S<stage> <label> · <genotype> · <name> · <time> · pool <pool>' (mice) or
    '<experiment> · t=<T> · day <d> · <video> · batch <b> · pos <p>' (ants) from the video list."""
    if 'stage' not in x:
        _, videos, _, vidx = _vt(ANTS)
        i = vidx[x['obs']]
        assert (videos[i][0], videos[i][1], videos[i][2]) == (x['experiment'], int(x['T']), x['day']), x['obs']
        return [i, sig4(x['act'])]
    vm, videos, _, vidx = _vt()
    i = vidx[x['obs']]
    assert (videos[i][0], videos[i][1], videos[i][2]) == (str(x['pool']), int(x['stage']), str(x['genotype'])), x['obs']
    return [i, sig4(x['act'])]


def _vt(dom=MICE):
    k = f'vt_{dom.name}'
    if k not in _rates:
        _rates[k] = video_table() if dom is MICE else video_table_ants()
    return _rates[k]


def page_clips(cfg, key, j, n, mont):
    """{source: {L: {src, blocks: {block name: index in the file}, n: {kind: count}, least_rule,
    info: {kind: [[label, mean], ...]}}}} of one neuron (montages order; mont = montages(cfg, key, picks))."""
    out = {}
    for name, bl in mont:
        for b, (jj, src, bn, k, h) in enumerate(bl):
            if jj != int(j):
                continue
            for L in LENGTHS:
                c = n['clips'][src][L]
                e = out.setdefault(src, {}).setdefault(L, {'src': f'assets/{name(L)}',
                                                           'blocks': {}, 'least_rule': c['least_rule'],
                                                           'n': {x: len(c[x]) for x in kinds(src)},
                                                           'info': {x: [page_data_clip(y) for y in c[x]]
                                                                    for x in kinds(src)}})
                e['blocks'][bn] = b
    return out


def analysis_meta(cfg, aid, r0):
    """Page metadata of one analysis (r0 = a summary.csv row of it)."""
    if cfg.dom is not MICE:
        m = cfg.dom.analysis(aid).meta
        return {'id': aid, 'family': 'B', 'experiment': m['experiment'], 'control': int(m['control']),
                'treatment': int(m['treatment']), 'confound': m['confound'], 'n_units': int(r0['n_units'])}
    return ({'id': aid, 'family': 'A', 'genotype': r0['genotype'], 'stages': stages_of(aid),
             'n_units': int(r0['n_units'])} if aid.startswith('A') else
            {'id': aid, 'family': 'B', 'stage': int(r0['stage']), 'n_units': int(r0['n_units'])})


def search_results(cfg, outs, analyses=None, others=None, top=False):
    """{'<outcome id>|<aid>|<prefix>|<window>': {n_tested_total, rows[, top]}} of the primary searches of
    one result set (full cohort or one subgroup); analyses (dict, filled) gets the per-analysis metadata.
    Cross-outcome chips: the other outcomes of outs, or others(o) when given. top = top_rows (VIEW 'top',
    full cohort only)."""
    analyses = {} if analyses is None else analyses
    results = {}
    words = (cfg.view['subject'], cfg.view['subjects'])
    for o in outs:
        prim = primary_rows(o)
        oth = [x for x in outs if x is not o] if others is None else others(o)
        for aid in cfg.order:
            g = prim[prim['analysis_id'] == aid]
            if not len(g):
                continue
            r0 = g.iloc[0]
            if aid not in analyses:
                analyses[aid] = analysis_meta(cfg, aid, r0)
            for (prefix, window), h in g.groupby(['prefix', 'window']):
                rows = []
                for _, r in h.dropna(subset=['neuron']).sort_values('round').iterrows():
                    j = int(r['neuron'])
                    rows.append({'round': int(r['round']), 'neuron': j, 'tau': float(r['tau']), 'p': float(r['p']),
                                 'threshold': float(r['threshold']), 'n_tested': int(r['n_tested']),
                                 'rob': robustness(o, oth, aid, int(prefix), window, j, words)})
                results[f'{o["id"]}|{aid}|{int(prefix)}|{window}'] = {
                    'n_tested_total': int(h['n_tested_total'].iloc[0]), 'rows': rows}
                if top and cfg.view['top']:
                    results[f'{o["id"]}|{aid}|{int(prefix)}|{window}']['top'] = top_rows(cfg, o, aid, prefix, window)
    return results


def subset_meta(outs):
    """Units per analysis, pools and skipped analyses of one subgroup (from its bout / mean sanity.json)."""
    o = outs[0]
    prim = primary_rows(o)
    n_units = {aid: int(g['n_units'].iloc[0]) for aid, g in prim.groupby('analysis_id')}
    sj = json.loads((o['dir'] / 'sanity.json').read_text())
    sg = sj.get('subgroup', {})
    return {'n_units': n_units, 'n_pools': sg.get('n_pools'), 'n_het_pools': sg.get('n_het_pools'),
            'skipped': [x['analysis_id'] for x in sj.get('skipped', [])]}


def page_data(cfg):
    picks = json.loads((cfg.work / 'picks.json').read_text())
    analyses, outcomes = {}, []
    allo = cfg.outcomes + cfg.extras
    for o in allo:
        outcomes.append({'id': o['id'], 'label': o['label'], 'codes': o['codes'], 'unit': o['unit'],
                         'bout': is_bout(o),
                         'setting': ', '.join(f'{k} {v:g}' if isinstance(v, float) else f'{k} {v}'
                                              for k, v in o['primary'].items()),
                         'kind': o['kind'], 'agg': o['agg']})
    results = search_results(cfg, cfg.outcomes, analyses, top=True)
    # extra outcomes: full cohort only (the subgroup runs are primary-only); cross-outcome chips = the core
    # outcomes of the other kind
    xres = search_results(cfg, cfg.extras, analyses, others=lambda o: [x for x in cfg.outcomes if x['kind'] != o['kind']],
                          top=True)
    subsets = {}
    for name, outs in cfg.subsets.items():
        subsets[name] = {'results': search_results(cfg, outs), **subset_meta(outs)}
        print(f'[{cfg.tag}] page: subgroup {name}: ' + ', '.join(
            f'{o["id"]} {sum(len(R["rows"]) for k, R in subsets[name]["results"].items() if k.startswith(o["id"] + "|"))} hits'
            for o in outs))
    # consistency check of the chips against the stored table (mean-pool result set, full window)
    det_p = cfg.res / 'galleries/stats_detail.csv'
    o0 = next((o for o in cfg.outcomes if mean_outcome(o) and o['dir'] == cfg.res), None)
    if det_p.exists() and o0 is not None:
        det = pd.read_csv(det_p, dtype=str)
        names = {'flip': 'rob_signflip', 'BH': 'rob_BH', 'max': 'rob_max-pool', 'rate': 'rob_rate',
                 'match': 'rob_matched', '128': 'rob_other_prefix', '1024': 'rob_other_prefix'}
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

    def arena_of(j):
        e = {'rows': grid, 'cols': grid, 'act': [sig4(v) for v in act_all[:, int(j)]]}
        if len(occ):
            e['occ'] = [sig4(v) for v in occ]
        if bg:
            e['bg'] = bg
        return e
    neurons = {}
    for key in cfg.keys:
        neurons[key] = {}
        mont = montages(cfg, key, picks)
        for j, n in picks['by_key'][key].items():
            vmax = n['max_frame'] or 1.0
            t = float(thr[key][int(j)]) if key in thr else None
            e = {'firing_rate': n['firing_rate'],
                 'table': [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in t_.items()} for t_ in n['table']],
                 'artefact': int(j) in cfg.artefact,
                 'heat_vmax': sig4(np.load(patch_file(cfg, key, j))['vmax']),
                 'clips': page_clips(cfg, key, j, n, mont),
                 'hist': dict(n['hist'], thr=t),
                 'bout_thr_bar': (round(min(t / vmax, 1.0), 4) if t is not None and vmax > 0 else None),
                 'arena': arena_of(j)}
            neurons[key][j] = e
        for k2, j in sorted(cfg.noclip):
            if k2 == key:
                neurons[key][str(j)] = {'noclip': True, 'artefact': int(j) in cfg.artefact, 'arena': arena_of(j)}
    # every other listed neuron (extra outcomes, top-by-p lists, packed SAEs beyond the clip budget): arena
    # map, chart and per-video panel only; 'x' = not in the historical neuron files (split_data)
    listed = {o['id']: set() for o in allo}
    for RR in [results, xres] + [v['results'] for v in subsets.values()]:
        for k, R in RR.items():
            listed[k.split('|')[0]] |= {str(r['neuron']) for r in R['rows']} | {str(t[0]) for t in R.get('top', [])}
    for o in allo:
        for j in sorted(listed[o['id']], key=int):
            if j not in neurons.setdefault(o['codes'], {}):
                neurons[o['codes']][j] = {'noclip': True, 'artefact': int(j) in cfg.artefact, 'arena': arena_of(j),
                                          'x': True}
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
    vm, videos, _, _ = _vt(cfg.dom)
    vm = vm.copy()
    vals, otables = {}, {}
    for o in allo:
        ov = video_outcome(cfg, o)
        if ov is None:
            continue
        ids, Y = ov
        ix = ids.get_indexer(vm['observation_id'].astype(str))
        assert (ix >= 0).all(), 'videos missing from the NES cache'
        js = sorted({str(r['neuron']) for RR in [results, xres] + [v['results'] for v in subsets.values()]
                     for k, R in RR.items() if k.startswith(o['id'] + '|') for r in R['rows']}, key=int)
        if cfg.view['packed']:  # every neuron of the page gets every outcome's values (chart, per-video panel)
            js = sorted(set(js) | set(neurons.get(o['codes'], {})) | listed[o['id']], key=int)
        vals[o['id']] = {j: [sig4(v) for v in Y[ix, int(j)]] for j in js}
        print(f'[{cfg.tag}] page: {o["label"]}: tau recomputed from per-video values vs summary.csv, '
              f'max relative diff {tau_check(vm, vals[o["id"]], xres if o["extra"] else results, o["id"], dom=cfg.dom):.2e}')
        for name, v in subsets.items():
            if o['extra']:
                continue
            l, x = name.split('_')
            sm = vm[((vm['line'] == l) | (l == 'all')) & ((vm['sex'] == x) | (x == 'all'))]
            print(f'[{cfg.tag}] page: {o["label"]} subgroup {name}: tau check max relative diff '
                  f'{tau_check(sm, vals[o["id"]], v["results"], o["id"]):.2e}')
        if is_bout(o) and not o['extra'] and cfg.dom is MICE:
            otables[o['id']] = {'full': {j: rate_table(cfg, o, int(j)) for j in picks['by_key'][o['codes']]}}
    for o in cfg.outcomes:  # every neuron shown in a subgroup needs its per-video values (chart, per-pool panel)
        miss = {str(j) for (k, j) in cfg.noclip if k == o['codes']} - set(vals[o['id']])
        if miss:
            ids, Y = video_outcome(cfg, o)
            ix = ids.get_indexer(vm['observation_id'].astype(str))
            vals[o['id']].update({j: [sig4(v) for v in Y[ix, int(j)]] for j in miss})
    if cfg.sae not in MODELS:
        raise SystemExit(f'{cfg.sae}: add it to MODELS (encoder / sae / input of the model bar)')
    return {'sae': cfg.sae, 'tag': cfg.tag, 'model': MODELS[cfg.sae], 'rep': cfg.rep, 'outcomes': outcomes,
            'otables': otables, 'artefact': sorted(int(j) for j in cfg.artefact),
            'subsets': subsets,
            'analyses': [analyses[a] for a in cfg.order if a in analyses], 'results': results, 'neurons': neurons,
            'artefact_text': art, 'arena_note': str(ar['note']), 'videos': videos, 'vals': vals,
            'n_frames': int(sum(next(iter(picks['by_key'][cfg.keys[0]].values()))['hist']['counts']['all'])),
            'xresults': xres, 'xids': [o['id'] for o in cfg.extras], 'packed': cfg.view['packed'],
            'domain': cfg.dom.name, 'obs': _vt(cfg.dom)[2]}


def q16(v):
    """Non-negative values -> {'max', 'q': base64 of little-endian uint16 round(v / max * 65535)} (arena maps:
    display only)."""
    import base64
    v = np.maximum(np.asarray(v, dtype=np.float64), 0)
    mx = float(v.max()) if v.max() > 0 else 1.0
    return {'max': sig4(mx), 'q': base64.b64encode(np.round(v / mx * 65535).astype('<u2').tobytes()).decode()}


def jdump(x):
    return json.dumps(x, separators=(',', ':'))


def split_data(d, data_dir, inline):
    """One SAE's page data -> (the part kept in index.html, {data file name: content}). Files (fetched by
    the page on demand, assets/data/): <tag>_r_<outcome id>.json = that outcome's searches (full cohort and
    every subgroup); <tag>_<codes key>_<k>.json = chunk k of the neurons ranked by that codes key (clip
    lists, histogram, arena map, per-video values and stage x genotype tables of each outcome built from
    that key), ~CHUNK_BYTES each. inline: the results file whose content goes into index.html instead.
    <tag>_x.json = one bundle {results, subsets, vals: {outcome: {neuron: values}}, by: {codes key: {neuron:
    entry}}} with the extra outcomes' searches and values and the neurons not in the files above ('x'
    entries); a packed SAE (VIEW 'packed') puts everything in its bundle (one data file)."""
    tag, files, packed, xids = d['tag'], {}, d['packed'], set(d['xids'])
    core = {k: d[k] for k in ('sae', 'tag', 'model', 'rep', 'outcomes', 'analyses', 'artefact', 'artefact_text',
                              'arena_note', 'n_frames')}
    core['subsets'] = {name: {k: v for k, v in S.items() if k != 'results'} for name, S in d['subsets'].items()}
    core['rfiles'], core['pre'] = {}, {}
    xf, X = f'{tag}_x.json', {'results': {}, 'subsets': {}, 'vals': {}, 'by': {}}
    for o in d['outcomes']:
        pre_ = o['id'] + '|'
        R = {'results': {k: v for k, v in (d['xresults'] if o['id'] in xids else d['results']).items()
                         if k.startswith(pre_)},
             'subsets': {} if o['id'] in xids else {name: {k: v for k, v in S['results'].items() if k.startswith(pre_)}
                                                    for name, S in d['subsets'].items()}}
        f = f'{tag}_r_{o["id"]}.json'
        if packed or o['id'] in xids:
            X['results'].update(R['results'])
            for name, S in R['subsets'].items():
                X['subsets'].setdefault(name, {}).update(S)
            X['vals'][o['id']] = d['vals'].get(o['id'], {})
            core['rfiles'][o['id']] = f'{data_dir}/{xf}'
        elif (tag, o['id']) == inline:  # key 'pre:<file>': not a file, the page reads it from ALL
            core['rfiles'][o['id']] = f'pre:{f}'
            core['pre'][f'pre:{f}'] = R
        else:
            core['rfiles'][o['id']] = f'{data_dir}/{f}'
            files[f] = R
    arena0 = None
    core['nfile'] = {}
    for key, ns in d['neurons'].items():
        chunks, cur, size, xs = [], {}, 0, {}
        for j in sorted(ns, key=int):
            e = dict(ns[j])
            A = e.pop('arena')
            isx = e.pop('x', False)
            if arena0 is None:
                arena0 = {'rows': A['rows'], 'cols': A['cols'], 'bg': A.get('bg'),
                          'occ': q16(A['occ']) if 'occ' in A else None}
            e['arena'] = q16(A['act'])
            if packed or isx:
                xs[j] = e
                continue
            e['vals'] = {o['id']: d['vals'][o['id']][j] for o in d['outcomes']
                         if o['codes'] == key and o['id'] not in xids and j in d['vals'].get(o['id'], {})}
            e['otable'] = {o['id']: d['otables'][o['id']]['full'][j] for o in d['outcomes']
                           if o['codes'] == key and o['id'] in d['otables'] and j in d['otables'][o['id']]['full']}
            n = len(jdump(e))
            if cur and size + n > CHUNK_BYTES:
                chunks.append(cur)
                cur, size = {}, 0
            cur[j], size = e, size + n
        if cur:
            chunks.append(cur)
        core['nfile'][key] = {}
        for k, c in enumerate(chunks):
            f = f'{tag}_{key}_{k}.json'
            files[f] = {'neurons': c}
            core['nfile'][key].update({j: k for j in c})
        if xs:
            X['by'][key] = xs
            core['nfile'][key].update({j: len(chunks) for j in xs})
        core['nfile'][key] = {'files': [f'{data_dir}/{tag}_{key}_{k}.json' for k in range(len(chunks))]
                                       + ([f'{data_dir}/{xf}'] if xs else []),
                              'of': core['nfile'][key]}
    core['arena'] = arena0
    if packed or xids or X['by']:
        files[xf] = X
    # every per-video value / table of the data went to some neuron file
    for o in d['outcomes']:
        lost = set(d['vals'].get(o['id'], {})) - set(d['neurons'][o['codes']])
        assert not lost, f'{tag} {o["id"]}: per-video values of neurons without an entry {sorted(lost)[:5]}'
    # page settings of the SAE's domain; a non-mice SAE carries its own video list
    v = VIEW[d['domain']]
    core.update(domain=d['domain'], subject=v['subject'], subjects=v['subjects'], sub_ui=d['domain'] == 'mice',
                top=v['top'])
    if d['domain'] != 'mice':
        core.update(videos=d['videos'], obs=d['obs'])
    return core, files


def step_page(cfgs, overwrite):
    from scipy import stats
    saes = [page_data(c) for c in cfgs]
    has_mice = any(c.dom is MICE for c in cfgs)
    _, videos, obs, _ = _vt() if has_mice else (None, [], [], None)
    doms = list(dict.fromkeys(c.dom.name for c in cfgs))
    data = {'default': cfgs[0].sae, 'saes': [],
            'axes': [{'key': k, 'label': lab, 'tips': {v: MODEL_TIPS.get((k, v), v) for v in
                                                       dict.fromkeys(MODELS[d['sae']][k] for d in saes)}}
                     for k, lab in MODEL_AXES],
            'videos': videos, 'obs': obs,
            'montage': {'K': K, 'cols': COLS, 'tile': TILE, 'lengths': {L: w for L, w in LENGTHS.items()}},
            'tcrit': {str(n): round(float(stats.t.ppf(0.975, n - 1)), 4) for n in range(2, 121)},
            'domains': [{'id': x, 'title': VIEW[x]['title'], 'default': next(c.sae for c in cfgs if c.dom.name == x)}
                        for x in doms],
            'kinds': [{'id': k, 'label': lab, 'tip': tip} for k, lab, tip in OUTCOME_KINDS],
            'aggs': [{'id': k, 'label': lab, 'tip': tip} for k, lab, tip in AGGS]}
    out = cfgs[0].out
    ddir = out / 'assets' / 'data'
    ddir.mkdir(parents=True, exist_ok=True)
    inline = (saes[0]['tag'], saes[0]['outcomes'][0]['id'])  # the first view's searches
    written = set()
    for d in saes:
        core, files = split_data(d, 'assets/data', inline)
        data['saes'].append(core)
        for f, c in files.items():
            (ddir / f).write_text(jdump(c))
            written.add(f)
        print(f'[{d["tag"]}] page: {len(files)} data files, {sum(len(jdump(c)) for c in files.values()) / 1e3:.0f} KB, '
              f'largest {max(len(jdump(c)) for c in files.values()) / 1e3:.0f} KB')
    for f in ddir.glob('*.json'):
        if f.name not in written:
            f.unlink()
    tpl = TEMPLATE.read_text()
    assert tpl.count('/*__DATA__*/null') == 1, 'template placeholder missing'
    page = tpl.replace('/*__DATA__*/null', jdump(data))
    (out / 'index.html').write_text(page)
    print('page: wrote', out / 'index.html', f'{len(page.encode()) / 1e3:.0f} KB')


# ---------------------------------------------------------------------- step: check
def step_check(cfgs, a):
    """index.html references the data files (assets/data/*.json); the page and the data files reference the
    media. Every reference must exist; unreferenced files are deleted; limits: files per version, MB, a
    file > 15 MB, index.html > --max-page-kb."""
    out = cfgs[0].out
    assets = out / 'assets'
    page = (out / 'index.html').read_text()
    assert page.startswith('<title>'), 'page must begin with <title>'
    for tag in ('html', 'head', 'body', '!doctype'):
        assert not re.search(rf'<{tag}[\s>]', page, re.I), f'page must not contain <{tag}>'
    drefs = set(re.findall(r'assets/data/[\w.\-]+\.json', page))
    dmiss = sorted(r for r in drefs if not (out / r).exists())
    texts = [page] + [(out / r).read_text() for r in sorted(drefs) if (out / r).exists()]
    refs = {r for t in texts for r in re.findall(r'assets/[\w.\-]+\.(?:mp4|webp|png|jpg)', t)}
    missing = sorted(r for r in refs if not (out / r).exists()) + dmiss
    files = sorted(p for p in assets.iterdir() if p.is_file())
    dfiles = sorted((assets / 'data').glob('*')) if (assets / 'data').is_dir() else []
    unused = [p for p in files if f'assets/{p.name}' not in refs] + \
             [p for p in dfiles if f'assets/data/{p.name}' not in drefs]
    for p in unused:
        p.unlink()
    files = [p for p in files + dfiles if p not in unused]
    total = sum(p.stat().st_size for p in files) + (out / 'index.html').stat().st_size
    big = max(files, key=lambda p: p.stat().st_size)
    n_data = sum(1 for p in files if p.parent.name == 'data')
    print(f'check: {len(refs)} referenced media + {len(drefs)} data files, {len(missing)} missing {missing[:5]}, '
          f'{len(unused)} unused deleted {[p.name for p in unused][:6]}')
    print(f'check: {len(files) + 1} files to publish (index.html + {len(files) - n_data} media + {n_data} data), '
          f'total {total / 1e6:.1f} MB, largest {big.name} {big.stat().st_size / 1e3:.0f} KB, '
          f'page {len(page.encode()) / 1e3:.0f} KB')
    for c in cfgs:
        n = [p for p in files if p.name.startswith(c.tag + '_')]
        print(f'check: [{c.tag}] {len(n)} files, {sum(p.stat().st_size for p in n) / 1e6:.1f} MB')
    if (missing or len(files) + 1 > a.max_version_files or total > a.max_mb * 1e6
            or len(page.encode()) > a.max_page_kb * 1e3 or big.stat().st_size > 15e6):
        raise SystemExit('check failed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', action='append', help=f'NES result set, repeatable; first = page default '
                    f'(default {DEFAULT_RES}); cache in <res>/_cache/explorer')
    ap.add_argument('--outcome', action='append', help='LABEL=SUBDIR[:key=value,...], repeatable')
    ap.add_argument('--out', default=None, help='output dir (default <first res>/explorer)')
    ap.add_argument('--steps', default='data,patch,arena,render,page,check')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--max-files', type=int, default=480, help='budget of media files + index.html (decides '
                    'which subgroup-only neurons get clips); the data files come on top')
    ap.add_argument('--max-files-new', type=int, default=10, help='media budget of each packed SAE (VIEW packed: '
                    'ants), what the mice pages leave of --max-version-files: 511 - 480 (mice media + index.html) '
                    '- 19 (mice data files) - 2 (the ants data file and arena background)')
    ap.add_argument('--max-version-files', type=int, default=511, help='all published files (artifact version limit)')
    ap.add_argument('--max-page-kb', type=float, default=400, help='index.html size limit')
    ap.add_argument('--max-mb', type=float, default=250)
    ap.add_argument('--crf', type=int, default=30, help='H.264 quality of the montages (higher = smaller)')
    a = ap.parse_args()
    res = a.res or DEFAULT_RES
    r0 = (ROOT / res[0]) if not Path(res[0]).is_absolute() else Path(res[0])
    out = Path(a.out) if a.out else r0 / 'explorer'
    cfgs = [Cfg(r, a, out) for r in res]
    plan_budget([c for c in cfgs if not c.view['packed']], a.max_files)
    plan_packed([c for c in cfgs if c.view['packed']], a.max_files_new)
    for s in a.steps.split(','):
        if s == 'page':
            step_page(cfgs, a.overwrite)
        elif s == 'check':
            step_check(cfgs, a)
        else:
            for c in cfgs:
                globals()[f'step_{s}'](c, a.overwrite)
