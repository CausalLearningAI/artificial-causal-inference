"""
ECI concept atlas v37: two models per domain ("online" = the published v36 page, "new" = R4), two methods (NES, DINES),
mice family C (genotype x phase transition). Output: results/vision/eci_atlas_v37/ (site/ = the page to publish).

The page design is scripts/eci/explorer_template_v37.html (the v36 template of scripts/eci/explorer_template.html plus a
method selector, family C, pending / not-computed states). This script reuses scripts/eci/build_explorer.py (imported,
never edited) for everything it already does (clip picks, clip rendering, page data, data files), with these changes:

  new model (R4, scripts/eci/rollout_r4.py: DINOv3-L/16 block 20, input sized to the animal, lab masks, d2 = [z_t,
  z_t - z_(t-2 s)], Matryoshka BatchTopK 512 / k 16, primary firing rule fnorm): one build per R4 domain (--dom mice,
  ants_v2, ants_v3, ants_v4, frogs; each ants series has its own SAE). Its "virtual result set"
  results/vision/eci_atlas_v37/res/<dom>_<variant>/<sae>/ holds
      nes/    the R4 NES result set <nes root>/<sae> (the R4 agent's, read only; symlinked) + family C NES rows (mice)
      dines/  DINES (scripts/eci/atlas_v37_methods.py) on the same per-video outcomes, + family C DINES rows (mice)
      _cache/ symlinks to the NES caches (per-video summaries) + explorer/ (this build's clip cache)
  One build reads both methods, so every neuron of either gets its clips once; the page entry is then split into a
  NES and a DINES entry that share the neuron files.
  Heat maps: R4 stores no per-patch codes, only per frame and neuron the position of the strongest token (argmax,
  the parent codes' argmax.npy). The "heat" blocks of the new model show that peak patch (its code at the argmax cell,
  Gaussian-blurred, sigma 1 patch), not a full per-patch map; the arena map is the same peak-patch quantity summed
  over every frame (code at the argmax cell / number of frames). No GPU is needed.
  Clip frames: the 512 px dataset JPGs of the 5 fps grid (the same frames as the R4 rows), staged to local disk.

  online model: the published v36 data files are reused as they are (models fg448al, antsfg, frogsfg; max pooling
  only: the mean / SOMP aggregations and the full-frame models are dropped from the selector). New for it: DINES
  searches and mice family C searches (NES + DINES), read through the same build_explorer functions; their neurons
  that have no v36 entry get a no-clip entry (arena map, charts, per-video values). The v36 mice page analysed
  habituation on its whole 30 min (published 2026-10-04, before the 2026-10-07 change to its last 15 min); its DINES
  and family C searches use that same window (build_explorer video_windows -> 'hwhole').

Steps (--steps, comma list):
  prep      virtual result set (new) / online virtual set
  data      clip picks (build_explorer step_data)                          [new]
  patch     peak-patch heat files from argmax                              [new]
  arena     peak-patch arena map + arena background                        [new]
  render    clip pages and packs (frames staged to $LOCAL_DIR)             [new]
  dump      page data of the build -> results/vision/eci_atlas_v37/build/<key>/{core.json, data/, assets/}
  assemble  (no --dom) the page: v36 entries + every dump -> results/vision/eci_atlas_v37/site/ + manifest.json

Usage (CPU jobs: scripts/eci/build_atlas_v37.sh):
  python scripts/eci/build_atlas_v37.py --dom ants_v2 --model new --steps prep,data,patch,arena,render,dump
  python scripts/eci/build_atlas_v37.py --dom mice --model online --steps prep,dump
  python scripts/eci/build_atlas_v37.py --steps assemble [--asset-map map.json]
"""
import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/vision/eci_atlas_v37'
R4 = ROOT / 'results/vision/eci_rollout_r4'
V36 = ROOT / 'results/vision/mice/eci/nes/matryoshka_btk_1024_k16_fg448al_s0/explorer_v36'
# the online model's input page: live version 39 (git show ea6a534:atlas/index.html). atlas/index.html itself is the
# output of this builder since version 42, so reading it would feed the build its own online entries back (duplicates)
V36_PAGE = OUT / 'live_v39' / 'index.html'
# versions 38 / 39 (another session, commits ca7ef24 / ea6a534) only drop the mean-pool and SOMP outcomes and mark those
# aggregations off in ALL.aggs; the max-pool data of every entry is unchanged since version 37
# version 37 (published 2026-10-16) = v36 with the ants entries replaced by the 30-min ants build (explorer_v39, ANTS_ECI
# eci30); mice and frogs data byte-identical to v36. The data files of each online entry are in its build's folder.
V39 = ROOT / 'results/vision/mice/eci/nes/matryoshka_btk_1024_k16_fg448al_s0/explorer_v39'
PUB_DIR = {'antsfg': V39, 'antsfull': V39}  # tag -> build folder of the published data (default V36)
TEMPLATE = ROOT / 'scripts/eci/explorer_template_v37.html'
# R4 build keys: (domain, ANTS_ECI layout, NES root relative to ROOT, R4 rollout domain)
NEW = {'mice': ('mice', None, 'results/vision/mice/eci/nes', 'mice'),
       'ants_v2': ('ants', 'eci_r4v2', 'results/vision/ants/eci_r4v2/nes', 'ants_v2'),
       'ants_v3': ('ants', 'eci_r4v3', 'results/vision/ants/eci_r4v3/nes', 'ants_v3'),
       'ants_v4': ('ants', 'eci_r4v4', 'results/vision/ants/eci_r4v4/nes', 'ants_v4'),
       'frogs': ('frogs', None, 'results/vision/frogs/eci/nes', 'frogs'),
       # R4 pooled ants (rollout_r4.py, the user's design correction of 2026-10-16): ONE SAE per variant for every ants
       # series (v2, v3, v4 + v4_infested), each series in its own layout eci_r4<s>p; it replaces the per-series SAEs
       # above on the page whenever a pooled build exists (step_assemble)
       'ants_v2p': ('ants', 'eci_r4v2p', 'results/vision/ants/eci_r4v2p/nes', 'ants_pooled'),
       'ants_v3p': ('ants', 'eci_r4v3p', 'results/vision/ants/eci_r4v3p/nes', 'ants_pooled'),
       'ants_v4p': ('ants', 'eci_r4v4p', 'results/vision/ants/eci_r4v4p/nes', 'ants_pooled'),
       # frog tadpoles (stage 44-48) on the new model (scripts/eci/tadpole_r4.py, its own SAE, crops of the raw video):
       # step 'tad' only (step_dump_tadpoles: NES + NESS entries without clips)
       'frogs_tad': ('tadpoles', None, 'results/vision/frogs/eci_tadpole_r4/nes', 'frogs_tadpole'),
       # the FROZEN v2 (scripts/eci/v2/, results/vision/eci_v2/: DINOv3-L/16 block 20, native-capped resolution,
       # canonical rotation, tight mask, Matryoshka 2048 k16, 0.6 s readout smoothing), one SAE per domain; variants
       # 'smooth' (<sae>_smooth, no firing threshold: the page's primary v2) and 'thr' (<sae>, fnorm-thresholded)
       'v2_mice': ('mice', None, 'results/vision/mice/eci/nes', 'mice'),
       'v2_ants_v2p': ('ants', 'eci_r4v2p', 'results/vision/ants/eci_r4v2p/nes', 'ants'),
       'v2_ants_v3p': ('ants', 'eci_r4v3p', 'results/vision/ants/eci_r4v3p/nes', 'ants'),
       'v2_ants_v4p': ('ants', 'eci_r4v4p', 'results/vision/ants/eci_r4v4p/nes', 'ants'),
       'v2_frogs': ('frogs', None, 'results/vision/frogs/eci/nes', 'frogs'),
       'v2_frogs_tad': ('tadpoles', None, 'results/vision/frogs/eci_tadpole_r4/nes', 'frogs')}
V2 = ROOT / 'results/vision/eci_v2'
V2_SERIES = {'v2_mice': 'v1', 'v2_ants_v2p': 'v2', 'v2_ants_v3p': 'v3', 'v2_ants_v4p': 'v4', 'v2_frogs': 'juv',
             'v2_frogs_tad': 'tad'}
V2_NESS = {'v2_mice': 'mice', 'v2_ants_v2p': 'ants_eci_r4v2p', 'v2_ants_v3p': 'ants_eci_r4v3p',
           'v2_ants_v4p': 'ants_eci_r4v4p', 'v2_frogs': 'frogs', 'v2_frogs_tad': 'frogs_tadpole'}
V2_MODEL = {'smooth': 'new', 'thr': 'new (thr)', 'prob': 'new'}  # page model values of the v2 input sets (readout_axis)
V2_SUF = {'smooth': '_smooth', 'thr': '', 'prob': '_prob'}
# online builds: (domain, ANTS_ECI, v36 result set, v36 tag)
ONLINE = {'mice': ('mice', None, 'results/vision/mice/eci/nes/matryoshka_btk_1024_k16_fg448al_s0', 'fg448al'),
          'ants': ('ants', 'eci30', 'results/vision/ants/eci30/nes/matryoshka_btk_1024_k16_antsfg_s0', 'antsfg'),
          'frogs': ('frogs', None, 'results/vision/frogs/eci/nes/matryoshka_btk_1024_k16_frogsfg_s0', 'frogsfg'),
          # ants v4 + v4_infested through the online model (scripts/eci/online_v4.py, another session, 2026-10-16): the
          # v36 SAE on the v4-only layout; the run without recording-day conditioning, as the v36 ants searches
          'ants_v4': ('ants', 'eci_c1v4_noday',
                      'results/vision/ants/eci_c1v4_noday/nes/matryoshka_btk_1024_k16_antsfg_s0', 'antsfg')}
# methods output (scripts/eci/atlas_v37_methods.py): <METHODS>/<domain key>/<sae>/{dines, familyC_nes, familyC_dines}
METHODS = OUT / 'methods'
METHOD_KEY = {'mice': 'mice', 'ants_v2': 'ants_eci_r4v2', 'ants_v3': 'ants_eci_r4v3', 'ants_v4': 'ants_eci_r4v4',
              'ants_v2p': 'ants_eci_r4v2p', 'ants_v3p': 'ants_eci_r4v3p', 'ants_v4p': 'ants_eci_r4v4p',
              'frogs': 'frogs', 'frogs_tad': 'frogs_tadpole', 'online_mice': 'mice', 'online_ants': 'ants_eci30', 'online_frogs': 'frogs'}
# the three temporal aggregations: (kind, outcome sub dir, NES outcome id as build_explorer names it, label, unit, short)
OUTS = [('rate', 'maxpool_bouts', 'boutratemaxpool', 'Bout rate (max-pool)', 'bouts/min', 'bouts'),
        ('time', '.', 'meanactivationma', 'Mean activation (max-pool)', 'activation', 'max-pool'),
        ('latency', 'maxpool_latency', 'xlatencymax', 'Latency to first bout (max-pool)', 's', 'latency')]
TRANS = {'1to2': (1, 2), '2to3': (2, 3), '4to5': (4, 5), '5to6': (5, 6)}

ap = argparse.ArgumentParser()
ap.add_argument('--dom', default=None, help='build key: ' + ', '.join(NEW) + ' (new) / mice, ants, frogs (online)')
ap.add_argument('--model', default='new', choices=('new', 'online'))
ap.add_argument('--variant', default='d2', choices=('d2', 'd0', 'smooth', 'thr', 'prob'), help='r4: d2 / d0; v2: smooth / thr / prob')
ap.add_argument('--steps', default='prep,data,patch,arena,render,dump', help='frogs_tad: tad')
ap.add_argument('--max-pages', type=int, default=0, help='clip pages per build (build_explorer --max-pages)')
ap.add_argument('--kr', type=int, default=None, help='clips rendered per row (default: build_explorer VIEW kr; mice 8)')
ap.add_argument('--pack-mb', type=float, default=12.0)
ap.add_argument('--asset-map', action='append', default=[], help='assemble: JSON {pack name: asset URL}, repeatable')
ap.add_argument('--overwrite', action='store_true')
A = ap.parse_args()

os.environ.setdefault('TADPOLE_TAG', '_r4')  # TadpolesDomain = the r4 tadpole layout (dataset/frogs/eci_tadpole_r4)
if A.dom is not None:  # the ants layout is fixed when src/eci/domain.py is imported (AntsDomain.ECI)
    lay = (NEW if A.model == 'new' else ONLINE)[A.dom][1]
    if lay:
        os.environ['ANTS_ECI'] = lay
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/eci'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import src.eci.domain as Dm  # noqa: E402

# the eci_r4v3* layouts: the core v3 contrasts t=2 vs t=6 and t=2 vs t=8 (the pooled t=2 vs t=6+8 patch of the R4 NES
# wrapper was dropped on 2026-10-16; its old rows are left out by the analysis list)

# mice family C (genotype x phase transition): unit = pool, outcome = stage b - stage a per pool, het vs wt
_mice_an = Dm.MiceDomain.analyses.fget


def _mice_analyses(self):
    out = _mice_an(self)
    for tr, (a, b) in TRANS.items():
        out.append(Dm.Analysis(f'C_{tr}', 'C', 'pool', {'genotype': 'het_vs_wt', 'stage': '', 'transition': tr},
                               ('het>wt', 'het<wt'), stages=(a, b), arm='genotype', control=('wt',),
                               treatment=('het',)))
    return out


def _mice_analyses_s(self):
    """+ family S (sex, the frozen v2's runs, scripts/eci/v2/nes_mice_dispatch.py mode S): female vs male pools within
    a stage, also within one genotype; unit = pool, tau > 0 = higher in females."""
    out = _mice_analyses(self)
    for s_ in range(1, 7):
        out.append(Dm.Analysis(f'S_stage{s_}', 'B', 'pool', {'genotype': 'all', 'stage': s_, 'transition': ''},
                               ('f>m', 'f<m'), where={'stage': s_}, arm='sex', control=('m',), treatment=('f',)))
    for g in ('het', 'wt'):
        for s_ in range(1, 7):
            out.append(Dm.Analysis(f'S_{g}_stage{s_}', 'B', 'pool', {'genotype': g, 'stage': s_, 'transition': ''},
                                   ('f>m', 'f<m'), where={'stage': s_, 'genotype': g}, arm='sex', control=('m',),
                                   treatment=('f',)))
    return out


Dm.MiceDomain.analyses = property(_mice_analyses_s)

import src.eci.viz as VZ  # noqa: E402

_lfc = VZ.load_full_codes


def _load_full_codes(sae_name, dataset_dir='dataset', data_dir='data', subject='mice', version='v1', codes_dir=None,
                     domain=None):
    """The R4 thresholded codes sets (<sae>_fnorm) hold codes_max only: codes_mean is then an alias of codes_max (only
    the 'max' key is used by this build)."""
    d = Path(codes_dir) if codes_dir else VZ._eci_dir(Path(dataset_dir), subject, version, domain) / 'codes' / sae_name
    if (d / 'codes_mean.npy').exists():
        return _lfc(sae_name, dataset_dir, data_dir, subject, version, codes_dir, domain)
    tmp = Path(os.environ.get('LOCAL_DIR') or OUT / 'tmp') / 'codes' / sae_name
    tmp.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        link(f.resolve(), tmp / f.name)
    link((d / 'codes_max.npy').resolve(), tmp / 'codes_mean.npy')
    return _lfc(sae_name, dataset_dir, data_dir, subject, version, tmp, domain)


VZ.load_full_codes = _load_full_codes

import build_explorer as BE  # noqa: E402

DOMOBJ = {'mice': BE.MICE, 'ants': BE.ANTS, 'frogs': BE.FROGS}
_cg_s = BE.contrast_groups


def _contrast_groups(cfg, aid, vids):
    """+ mice family S: (female, male) videos of the stage (within the genotype for S_<g>_stage<s>); the pool's sex from
    the mice experiment.csv (the video blocks carry pool / stage / genotype only)."""
    if cfg.dom.name == 'tadpoles' and 'experiment' in vids:  # contrast clips only from the QC-kept videos
        c0, t0 = _cg_s(cfg, aid, vids)
        k = vids['experiment'].astype(str).values == TAD_EXP
        return c0 & k, t0 & k
    if cfg.dom is not BE.MICE or not str(aid).startswith('S_'):
        return _cg_s(cfg, aid, vids)
    an = cfg.dom.analysis(aid)
    ex = pd.read_csv(BE.MICE.experiment_csv).drop_duplicates('pool').set_index('pool')['sex']
    sx = ex.loc[vids['pool'].astype(str)].astype(str).values
    m = vids['stage'].astype(int).values == an.where['stage']
    if 'genotype' in an.where:
        m &= vids['genotype'].astype(str).values == an.where['genotype']
    return m & (sx == 'f'), m & (sx == 'm')


BE.contrast_groups = _contrast_groups
# the DINES tables' own columns (scripts/eci/atlas_v37_methods.py) are not setting columns
BE.KNOWN |= {'status', 'p_kind', 'rho', 'set_aside', 'terminal_complete', 'method', 'hab_window', 'stages',
             'nes_W_frames'}
# DINES family A (paired, within one genotype) is not shown: after within-pool centring a pool's two videos are mirror
# images, so any neuron that changes consistently predicts the stage and DINES's overlap rule sets it aside; on the
# online mice data 132 of 133 reported bout-rate neurons were 'undetermined' and 3 of 24 searches hit the 30-neuron cap
DINES_A_BLOCK = ('Not shown for DINES: in the paired design (within-pool centred) every neuron that changes '
                 'consistently predicts the stage, so DINES cannot separate the neurons (almost all come out '
                 '"undetermined"). Use NES for family A.')


def annotate_status(results, outs):
    """DINES rows: r['st'] = 'kept' / 'undetermined' (status column of the DINES tables)."""
    for o in outs:
        if 'status' not in o['tidy'].columns:
            continue
        t = BE.primary_rows(o).dropna(subset=['neuron'])
        st = {(a, int(p), int(j)): x for a, p, j, x in zip(t['analysis_id'], t['prefix'], t['neuron'], t['status'])}
        for k, R in results.items():
            oid, aid, p, _ = k.split('|')
            if oid != o['id']:
                continue
            for r in R['rows']:
                x = st.get((aid, int(p), int(r['neuron'])))
                if isinstance(x, str):
                    r['st'] = x
    return results


def block_dines_A(results, oids, aids):
    """DINES searches + a 'blocked' (empty) search for every family A analysis of each DINES outcome."""
    out = dict(results)
    for o in oids:
        for a in aids:
            if a.startswith('A_'):
                for p in BE.PAGE_PREFIXES:
                    out[f'{o}|{a}|{p}|full'] = {'rows': [], 'n_tested_total': 0, 'blocked': DINES_A_BLOCK}
    return out


# ---------------------------------------------------------------------- family C in build_explorer
def _stages_of(aid):
    return [int(x) for x in BE.MICE.analysis(aid).stages]


BE.stages_of = _stages_of
_am = BE.analysis_meta


def analysis_meta(cfg, aid, r0):
    if aid.startswith('C_'):
        return {'id': aid, 'family': 'C', 'stages': _stages_of(aid), 'n_units': int(r0['n_units'])}
    return _am(cfg, aid, r0)


BE.analysis_meta = analysis_meta
_cg = BE.contrast_groups


def contrast_groups(cfg, aid, vids):
    """Family C: (het videos, wt videos) of the transition's two stages (both genotypes = all its pools)."""
    if aid.startswith('C_'):
        a, b = _stages_of(aid)
        st, gen = vids['stage'].astype(int).values, vids['genotype'].astype(str).values
        s = np.isin(st, (a, b))
        return s & (gen == 'het'), s & (gen == 'wt')
    return _cg(cfg, aid, vids)


BE.contrast_groups = contrast_groups
_tc = BE.tau_check


def tau_check(vm, vals, results, oid, round_=1, dom=BE.MICE):
    """+ family C: tau = mean over het pools of (stage b - stage a) - the same over wt pools."""
    # family S (sex) is left out of build_explorer's own check (it reads every non-A id as B_stage<s>)
    worst = _tc(vm, vals, {k: R for k, R in results.items() if '|C_' not in k and '|S_' not in k}, oid, round_, dom)
    for rk, R in results.items():
        o_, aid, _, _ = rk.split('|')
        if o_ != oid or not aid.startswith('C_'):
            continue
        a, b = _stages_of(aid)
        for r in R['rows']:
            if r['round'] != round_:
                continue
            y = np.asarray(vals[str(r['neuron'])], dtype=np.float64)
            d = vm.assign(y=y[vm.index])
            p = d.pivot_table(index=['pool', 'genotype'], columns='stage', values='y').reset_index()
            p['D'] = p[b] - p[a]
            t = float(p[p['genotype'] == 'het']['D'].mean() - p[p['genotype'] == 'wt']['D'].mean())
            worst = max(worst, abs(t - r['tau']) / max(abs(r['tau']), 1e-9))
    return worst


BE.tau_check = tau_check


# ---------------------------------------------------------------------- helpers
def log(*x):
    print('[v37]', *x, flush=True)


def best_sae(rdom, variant):
    return json.loads((R4 / rdom / 'sae' / 'best.json').read_text())[variant]['name']


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(src, dst)


def read_tidy(files):
    """Concatenate summary.csv-shaped files (columns unioned)."""
    ts = [pd.read_csv(f) for f in files if f.exists()]
    return pd.concat(ts, ignore_index=True, sort=False) if ts else None


def mirror_outcome(dst, srcs):
    """dst/summary.csv = the union of the srcs' summary.csv (+ C_summary.csv); dst/<aid> -> the src analysis dirs
    (result.json: first-round tables, latency W); size_adjusted.csv of the first src that has one. A src may be
    (path, analysis ids): only those analyses of it are taken."""
    dst.mkdir(parents=True, exist_ok=True)
    parts = []
    for s in srcs:
        s, only = s if isinstance(s, tuple) else (s, None)
        for d in sorted(p for p in s.iterdir() if p.is_dir() and (p / 'result.json').exists()):
            if only is None or d.name in only:
                link(d.resolve(), dst / d.name)
        t = read_tidy([s / 'summary.csv', s / 'C_summary.csv'])
        if t is not None and only is not None:
            t = t[t['analysis_id'].astype(str).isin(only)]
        if t is not None:
            parts.append(t)
    if not parts:
        return False
    pd.concat(parts, ignore_index=True, sort=False).to_csv(dst / 'summary.csv', index=False)
    for s in srcs:
        if isinstance(s, tuple):
            continue
        if (s / 'size_adjusted.csv').exists() and not (dst / 'size_adjusted.csv').exists():
            shutil.copy(s / 'size_adjusted.csv', dst / 'size_adjusted.csv')
    return True


def make_cfg(res, sae, dom, tag, out):
    """A build_explorer Cfg without its __init__ (whose outcome discovery assumes one NES result set): the outcomes
    are set by add_outcomes."""
    c = object.__new__(BE.Cfg)
    c.res, c.sae, c.tag, c.dom = res, sae, tag, dom
    c.view = BE.VIEW[dom.name]
    c.vdom = None if dom is BE.MICE else dom
    c.order = BE.kept_analyses(dom, c.view)
    c.out, c.assets, c.work = out, out / 'assets', res / '_cache' / 'explorer'
    c.rdir, c.ameta = res, {}
    c.outcomes, c.extras, c.subsets, c.xsubsets = [], [], {}, {}
    nf = dom.eci_dir / 'codes' / sae / 'n_fg.npy'
    c.size_ok = not (nf.exists() and np.ptp(np.load(nf, mmap_mode='r')[::97]) == 0)
    c.noclip, c.clipplan = set(), None
    c.keys = ['max']
    c.rep, c.align, c.rot = 'fg448', 'none', None  # whole-frame square grid (frame_box), no alignment
    c.artefact, c.interp = set(), {}
    c.crf = c.view['crf']
    c.tile, c.cols, c.kr, c.crop = (c.view[k] for k in ('tile', 'cols', 'kr', 'crop'))
    return c


def add_outcomes(c, base, prefix, cache, kinds=('rate', 'time', 'latency'), dines=False):
    """Outcomes of one method from base/<sub>/summary.csv: ids prefix + the NES id (build_explorer names); rate and
    time are core outcomes (own data file), latency an extra one (the bundle), as build_explorer does."""
    added = []
    for kind, sub, oid, label, unit, short in OUTS:
        d = (base / sub).resolve() if sub != '.' else base
        if kind not in kinds or not (d / 'summary.csv').exists():
            continue
        t = c.trim(pd.read_csv(d / 'summary.csv'))
        if dines:  # family A is not shown for DINES (DINES_A_BLOCK)
            t = t[~t['analysis_id'].astype(str).str.startswith('A_')].reset_index(drop=True)
        if not len(t):
            log(f'{c.tag}: {d}/summary.csv has no rows of the shown analyses / prefixes')
            continue
        kv = {'pooling': 'max', 'outcome_type': {'rate': 'bout_rate', 'time': 'mean', 'latency': 'latency'}[kind]}
        if kind in ('rate', 'latency'):
            kv['threshold_q'] = '0.95'
        if kind == 'rate' and 'merge_gap' in t.columns:
            kv['merge_gap'] = '0'
        kv = {k: v for k, v in kv.items() if k in ('pooling', 'outcome_type') or k in t.columns}
        prim = BE.default_primary(t, kv)
        o = {'label': label, 'dir': d, 'tidy': t, 'primary': prim, 'codes': 'max', 'pool': 'max', 'kind': kind,
             'agg': 'max', 'extra': kind == 'latency', 'cache': cache, 'unit': unit, 'short': short,
             'id': prefix + oid, 'size_ok': c.size_ok}
        if kind == 'latency':
            o.update(lfile='latency.npz', pwin='common')
        (c.extras if o['extra'] else c.outcomes).append(o)
        added.append(o)
        n = len(BE.primary_rows(o).dropna(subset=['neuron']))
        log(f'{c.tag}: outcome {o["id"]} ({kind}) from {d}: primary {prim}, {n} selection rows')
    return added


# ---------------------------------------------------------------------- new model (R4)
# searches added to a build's virtual result set: key -> (NES subdir of the R4 result set, methods out-root, analyses).
# v3 t=2 vs t=8 shares no recording day, so the --day runs skip it; it is searched without --day (n_fg only,
# scripts/eci/r4_v3_2v8.sh), its NES rows from <R4 result set>/v3_2_vs_8_noday, its DINES rows from methods_v3_2v8
EXTRA_SEARCHES = {'ants_v3p_d2': ('v3_2_vs_8_noday', OUT / 'methods_v3_2v8', ('v3_2_vs_8',)),
                  'v2_ants_v3p_thr': ('v3_2_vs_8_noday', V2 / 'ness_v3_2v8', ('v3_2_vs_8',)),
                  'v2_ants_v3p_smooth': ('v3_2_vs_8_noday', V2 / 'ness_v3_2v8', ('v3_2_vs_8',)),
                  'v2_ants_v3p_prob': ('v3_2_vs_8_noday', V2 / 'ness_v3_2v8', ('v3_2_vs_8',))}


def v2_name(rdom):
    return json.loads((V2 / rdom / 'best.json').read_text())['name']


def new_paths():
    dom_name, lay, nes_root, rdom = NEW[A.dom]
    key = f'{A.dom}_{A.variant}'
    if A.dom.startswith('v2_'):
        assert A.variant in V2_MODEL, A.variant
        sae = v2_name(rdom) + V2_SUF[A.variant]
        return dom_name, sae, ROOT / nes_root / sae, OUT / 'res' / key / sae, OUT / 'build' / key, \
            V2 / 'ness' / V2_NESS[A.dom] / sae, key
    sae = best_sae(rdom, A.variant) + '_fnorm'
    return dom_name, sae, ROOT / nes_root / sae, OUT / 'res' / key / sae, OUT / 'build' / key, \
        METHODS / METHOD_KEY[A.dom] / sae, key


def step_prep_new():
    dom_name, sae, src, V, bout, M, key = new_paths()
    for s in ('.', 'maxpool_bouts', 'maxpool_latency'):
        if not (src / s / 'summary.csv').exists():
            raise SystemExit(f'{src}/{s}/summary.csv missing: the R4 NES run of {key} is not finished')
    (V / '_cache' / 'explorer').mkdir(parents=True, exist_ok=True)
    for f in ('video_summaries_max.npz', 'bout_summaries_max.npz', 'latency.npz'):
        link((src / '_cache' / f).resolve(), V / '_cache' / f)
    for f in (src / '_cache').glob('bout_thresholds*.npz'):
        link(f.resolve(), V / '_cache' / f.name)
    have = {}
    xs = EXTRA_SEARCHES.get(key)
    xn = [(src / xs[0], xs[2])] if xs else []
    xk = V2_NESS[A.dom] if A.dom.startswith('v2_') else METHOD_KEY.get(A.dom)
    xd = [(xs[1] / xk / sae / 'dines', xs[2])] if xs else []
    xn = [x for x in xn if x[0].is_dir()]
    xd = [x for x in xd if x[0].is_dir()]
    if A.dom == 'v2_mice':  # family S (sex) of the frozen v2: its own NES / NESS result sets
        xn += [src / 'family_S'] if (src / 'family_S').is_dir() else []
        xd += [V2 / 'ness_S' / 'mice' / sae / 'dines'] if (V2 / 'ness_S' / 'mice' / sae / 'dines').is_dir() else []
    for meth, srcs in (('nes', [src, M / 'familyC_nes'] + xn), ('dines', [M / 'dines', M / 'familyC_dines'] + xd)):
        have[meth] = []
        for kind, sub, *_ in OUTS:
            ss = [((s[0] / sub if sub != '.' else s[0]), s[1]) if isinstance(s, tuple) else (s / sub if sub != '.' else s)
                  for s in srcs if ((s[0] if isinstance(s, tuple) else s) / sub).is_dir()]
            if ss and mirror_outcome(V / meth / sub if sub != '.' else V / meth, ss):
                have[meth].append(kind)
        if (srcs[0] / 'selected_neurons.json').exists():
            shutil.copy(srcs[0] / 'selected_neurons.json', V / meth / 'selected_neurons.json')
    (V / 'prep.json').write_text(json.dumps({'sae': sae, 'nes_src': str(src), 'methods_src': str(M), 'have': have},
                                            indent=1))
    log(f'prep {key}: {V}; outcomes {have}')


def step_checks_new():
    """The confound checks of the published page on the new model's NES picks, run read-only on this build's NES mirror
    (res/<key>/<sae>/nes, the R4 NES result set + family C rows): mice camera-period checks (scripts/eci/period_check.py,
    max pooling), frogs recording-session checks (scripts/eci/session_check.py, level session). Written to
    <V>/nes/period_check|session_check and linked as <V>/<name> (build_explorer period_checks / session_checks)."""
    dom_name, sae, src, V, bout, M, key = new_paths()
    if dom_name not in ('mice', 'frogs'):
        log(f'checks {key}: none for {dom_name}')
        return
    link((V / '_cache').resolve(), V / 'nes' / '_cache')
    root = OUT / 'res' / f'{key}_checks'
    link((V / 'nes').resolve(), root / sae)
    if dom_name == 'mice':
        cmd = [sys.executable, '-u', str(ROOT / 'scripts/eci/period_check.py'), '--sae', sae, '--pooling', 'max',
               '--out-root', str(root)]
        name = 'period_check'
    else:
        cmd = [sys.executable, '-u', str(ROOT / 'scripts/eci/session_check.py'), '--domain', 'frogs', '--sae', sae,
               '--pooling', 'max', '--level', 'session', '--out-root', str(root)]
        name = 'session_check'
    log('checks:', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    link((V / 'nes' / name).resolve(), V / name)


def new_cfg():
    dom_name, sae, src, V, bout, M, key = new_paths()
    dom = DOMOBJ[dom_name]
    tag = ('v2' + {'v2_mice': 'mice', 'v2_ants_v2p': 'v2p', 'v2_ants_v3p': 'v3p', 'v2_ants_v4p': 'v4p',
                   'v2_frogs': 'frogs', 'v2_frogs_tad': 'tad'}[A.dom] + {'smooth': 's', 'thr': 't', 'prob': 'p'}[A.variant]) if A.dom.startswith('v2_') else \
        'r4' + {'mice': 'mice', 'ants_v2': 'v2', 'ants_v3': 'v3', 'ants_v4': 'v4', 'frogs': 'frogs', 'ants_v2p': 'v2p',
                'ants_v3p': 'v3p', 'ants_v4p': 'v4p'}[A.dom] + A.variant
    c = make_cfg(V, sae, dom, tag, bout)
    if A.kr:
        c.kr = A.kr
    elif dom is BE.MICE:
        c.kr = 8  # the page shows 8 per row by default; 16 doubles the packs (asset store budget)
    add_outcomes(c, V / 'nes', '', V / '_cache')
    if (V / 'dines' / 'summary.csv').exists() or (V / 'dines' / 'maxpool_bouts' / 'summary.csv').exists():
        add_outcomes(c, V / 'dines', 'd', V / '_cache', dines=True)
    if A.dom == 'v2_mice':
        v2_subsets(c, src, sae)
    BE.plan_clips([c], A.max_pages)
    return c


def v2_subsets(c, src, sae):
    """Line / sex strata of the frozen v2 mice runs (families A and B): NES <res>/subsets/<line>_<sex>/<sub>, NESS
    results/vision/eci_v2/ness_strata/<line>_<sex>/mice/<sae>/dines/<sub>; a stratum enters only when every outcome
    of the build has it (core outcomes -> c.subsets, latency -> c.xsubsets)."""
    sub_of = {o['id']: sub for o in c.outcomes + c.extras for kind, sub, oid, *_ in OUTS
              if o['id'] in (oid, 'd' + oid)}
    for name in BE.SUBSETS:
        core, extra = [], []
        for o in c.outcomes + c.extras:
            base = (V2 / 'ness_strata' / name / 'mice' / sae / 'dines') if o['id'].startswith('d') else (src / 'subsets' / name)
            d = base / sub_of[o['id']] if sub_of[o['id']] != '.' else base
            if (d / 'summary.csv').exists():
                t = c.trim(pd.read_csv(d / 'summary.csv'))
                if o['id'].startswith('d'):
                    t = t[~t['analysis_id'].astype(str).str.startswith('A_')].reset_index(drop=True)
                (extra if o['extra'] else core).append({**o, 'dir': d, 'tidy': t})
        if len(core) == len(c.outcomes) and core:
            core.sort(key=lambda o: (o['id'].startswith('d'), sub_of[o['id']] != '.'))  # subset_meta reads core[0]
            c.subsets[name] = core
            if len(extra) == len(c.extras):
                c.xsubsets[name] = extra
    log(f'{c.tag} strata: {sorted(c.subsets)} (latency: {sorted(c.xsubsets)})')


def codes_cfg(c):
    cc = json.loads((c.dom.eci_dir / 'codes' / c.sae / 'config.json').read_text())
    for _ in range(3):  # the frozen v2's prob set points at the smoothed set, whose parent (raw) holds argmax.npy
        pc = Path(cc['parent_codes'])
        if (pc / 'argmax.npy').exists() or not (pc / 'config.json').exists():
            break
        nxt = json.loads((pc / 'config.json').read_text())
        if 'parent_codes' not in nxt:
            break
        cc = dict(cc, parent_codes=nxt['parent_codes'])
    codes = np.load(c.dom.eci_dir / 'codes' / c.sae / 'codes_max.npy', mmap_mode='r')
    arg = np.load(Path(cc['parent_codes']) / 'argmax.npy', mmap_mode='r')
    return cc, codes, arg, int(cc['G'])


def step_patch_new(c):
    """Peak-patch heat per clip frame: the frame's code (fnorm codes_max) at the neuron's argmax token, Gaussian
    sigma 1 patch, rescaled so its peak = the code (build_explorer patch file format)."""
    from scipy.ndimage import gaussian_filter
    picks = json.loads((c.work / 'picks.json').read_text())
    (c.work / 'patch').mkdir(parents=True, exist_ok=True)
    cc, codes, arg, G = codes_cfg(c)
    n_done, bad = 0, 0
    for j, n in picks['by_key'].get('max', {}).items():
        f = BE.patch_file(c, 'max', j)
        rr = BE.clip_rows(n, c.kr, BE.SHOW_LENGTHS)
        sig = 'peak-' + json.dumps(rr)
        if f.exists() and not A.overwrite and str(np.load(f)['sig']) in (sig, 'r4heat-' + json.dumps(rr)):
            continue  # r4heat-: real per-patch codes (scripts/eci/atlas_v37_r4heat.py, GPU)
        jj = int(j)
        arrs, pos = {}, []
        for name, rows in rr.items():
            m = np.zeros((len(rows), G, G), np.float32)
            for t, r in enumerate(rows):
                v, g = float(codes[r, jj]), int(arg[r, jj])
                if v > 0 and g >= 0:
                    one = np.zeros((G, G), np.float32)
                    one.flat[g] = 1.0
                    one = gaussian_filter(one, 1.0)
                    m[t] = one / one.max() * v
                    pos.append(v)
                elif v > 0:
                    bad += 1
            arrs[name] = m.astype(np.float16)
        vmax = float(np.quantile(pos, 0.99)) if pos else 1.0
        np.savez(f, sig=sig, vmax=vmax, **arrs)
        n_done += 1
    log(f'{c.tag} patch: {n_done} peak-patch files written (grid {G}); firing frames without an argmax: {bad}')


def step_arena_new(c):
    """Per patch position and neuron: sum over every frame of the code at the neuron's argmax cell / frames; the
    arena background: median of the per-video backgrounds (as build_explorer step_arena)."""
    from PIL import Image
    out = c.work / 'arena.npz'
    cc, codes, arg, G = codes_cfg(c)
    P, m = G * G, codes.shape[1]
    turned = c.dom is BE.MICE  # mice: every frame's cell turned with v1's frame turns (see V1_MICE_SAE)
    if turned:
        rot = mice_turns()
        assert len(rot) == len(codes), (len(rot), len(codes))
        I = np.arange(P).reshape(G, G)
        cellmap = np.empty((4, P), np.int64)  # cellmap[k, g] = index of cell g after k counter-clockwise turns
        for k in range(4):
            cellmap[k, np.rot90(I, k).ravel()] = np.arange(P)
        if out.exists() and not (c.work / 'arena_unturned.npz').exists():
            shutil.copy(out, c.work / 'arena_unturned.npz')
    if not out.exists() or A.overwrite or (turned and 'turned' not in np.load(out).files):
        S = np.zeros(P * m, np.float64)
        cols = np.arange(m, dtype=np.int64)[None]
        for a in range(0, len(codes), 100_000):
            z = np.asarray(codes[a:a + 100_000], np.float32)
            g = np.asarray(arg[a:a + 100_000], np.int64)
            k = (z > 0) & (g >= 0)
            if turned and not A.dom.startswith('v2_'):  # the frozen v2's argmax is already in turned coordinates
                g = np.where(g >= 0, cellmap[np.asarray(rot[a:a + 100_000], np.int64)[:, None], np.maximum(g, 0)], g)
            S += np.bincount((g * m + cols)[k], weights=z[k], minlength=P * m)
        n = len(codes)
        np.savez(out, act=(S / n).reshape(P, m).astype(np.float32), grid=G, n_frames=n, occ=np.zeros(0),
                 **({'turned': np.array(1)} if turned else {}),
                 note=f'every frame ({n:,}): per patch, the neuron\'s code summed over the frames where that patch is '
                      f'its strongest token, divided by the number of frames (peak-patch map from the stored argmax '
                      f'token)' + ('; mice: each frame turned so the odour corner is at the top right, as v1' if turned
                                   else ''))
        log(f'{c.tag} arena: {n} frames, grid {G}')
    bg_out = c.assets / BE.bg_name(c)
    if c.dom.name == 'tadpoles':  # the crop follows the animal: no arena background
        log(f'{c.tag} arena: crop-grid map of the argmax token (animal-centred, no background)')
        return
    if not bg_out.exists() or A.overwrite or turned:
        c.assets.mkdir(parents=True, exist_ok=True)
        if 'backgrounds' in cc:
            bgd = Path(cc['backgrounds'])
        else:  # frozen v2: the mask background of its domain config (configs/eci_v2/<domain>.json)
            vc = json.loads((ROOT / f'configs/eci_v2/{ {"mice": "mice", "ants": "ants", "frogs": "frogs"}[c.dom.name] }.json')
                            .read_text())
            sc = next(x for x in vc['series'] if x['name'] == V2_SERIES[A.dom])
            bgd = ROOT / sc['mask']['bg']
        zs = [np.load(f) for f in sorted(bgd.glob('*.npz'))]
        vids = set(pd.read_csv(c.dom.ann_path, usecols=['observation_id'])['observation_id'].astype(str))
        ims = [z['pix_bg'] for f, z in zip(sorted(bgd.glob('*.npz')), zs) if f.stem.split('__')[0] in vids] or \
              [z['pix_bg'] for z in zs]
        al = [str(z['align']) if 'align' in z.files else 'none' for z in zs[:1]]
        if turned and al == ['none']:  # each video's background turned like its frames (unless already aligned)
            ao = pd.read_csv(c.dom.ann_path, usecols=['observation_id'])['observation_id'].astype(str).to_numpy()
            first = {o: i for i, o in reversed(list(enumerate(ao)))}
            ims = [np.rot90(z['pix_bg'], int(rot[first[f.stem.split('__')[0]]]))
                   for f, z in zip(sorted(bgd.glob('*.npz')), zs) if f.stem.split('__')[0] in first]
        im = np.median(np.stack(ims), 0).astype(np.uint8)
        Image.fromarray(im).resize((384, 384), Image.BILINEAR).save(bg_out, 'WEBP', quality=80)
        log(f'{c.tag} arena: background = median of {len(ims)} per-video backgrounds -> {bg_out}')


# mice: v1 (fg448al) draws every clip on the frame turned so the odour corner is at the top right (its codes are of
# that turned frame); the R4 codes are of the unturned frame. So the two models are drawn identically, the new
# model's clips are turned the same way (v1's per-row turns, src/eci/viz.py frame_rot90 of the v1 SAE) and its real
# per-token maps are turned with them (np.rot90, the same counter-clockwise quarter turns as PIL ROTATE_90):
# step 'rotheat' writes <work>/patch_rot/ from the real files in <work>/patch/, and render / dump read patch_rot.
V1_MICE_SAE = 'matryoshka_btk_1024_k16_fg448al_s0'


_TURNS = {}


def mice_turns():
    if 'm' in _TURNS:
        return _TURNS['m']
    rot = _TURNS['m'] = VZ.frame_rot90(V1_MICE_SAE, ROOT / 'dataset')
    assert rot is not None, 'v1 mice SAE has no frame alignment'
    return rot


def step_rotheat(c):
    """Real per-token heat files of a mice build turned with v1's frame turns -> <work>/patch_rot/ (sig kept)."""
    from PIL import Image
    from src.eci.foreground import rotate_image
    rot = mice_turns()
    picks = json.loads((c.work / 'picks.json').read_text())
    # the frame rows of the picks are annotations.csv rows (the v1 turns index): check a few paths against it
    ann = pd.read_csv(ROOT / 'dataset/mice/v1/annotations.csv', usecols=['frame_path'])
    fp = picks['frame_path']
    for r in list(fp)[:: max(1, len(fp) // 200)]:
        assert ann['frame_path'].iloc[int(r)] == fp[r], (r, fp[r], ann['frame_path'].iloc[int(r)])
    assert len(rot) == len(ann), (len(rot), len(ann))
    t = np.arange(12, dtype=np.float32).reshape(3, 4)  # np.rot90 == PIL ROTATE_90 (counter-clockwise)
    for k in range(4):
        im = np.asarray(rotate_image(Image.fromarray(t), k))
        assert np.array_equal(im, np.rot90(t, k)), k
    out = c.work / 'patch_rot'
    out.mkdir(exist_ok=True)
    n, turns = 0, {}
    for j in picks['by_key'].get('max', {}):
        f = c.work / 'patch' / f'{BE.pre("max")}{int(j):04d}.npz'
        z = np.load(f)
        sig = str(z['sig'])
        if not (sig.startswith('r4heat-') or sig.startswith('v2heat-')):
            raise SystemExit(f'{f}: not a real per-token heat file ({sig[:10]}): run scripts/eci/atlas_v37_r4heat.py')
        rr = json.loads(sig[len('r4heat-'):])
        arrs = {}
        for k, rows in rr.items():
            a = z[k]
            assert len(a) == len(rows), (f, k)
            arrs[k] = np.stack([np.rot90(a[i], int(rot[int(r)])) for i, r in enumerate(rows)]) if len(rows) else a
            for r in rows:
                turns[int(rot[int(r)])] = turns.get(int(rot[int(r)]), 0) + 1
        np.savez(out / f.name, sig=sig, vmax=z['vmax'], rot=np.array(1), **arrs)
        n += 1
    log(f'{c.tag} rotheat: {n} real heat files turned with the v1 frame turns -> {out}; turns of the clip frames {turns}')


def use_rotated(c):
    """mice render / dump: the v1 frame turns; r4: the turned heat files (patch_rot). The frozen v2 encodes the
    canonical (turned) frame itself, so its heat files are already in turned coordinates (patch/)."""
    c.rot, c.align = mice_turns(), 'odor'
    if not A.dom.startswith('v2_'):
        BE.patch_file = lambda cfg, key, j: cfg.work / 'patch_rot' / f'{BE.pre(key)}{int(j):04d}.npz'


def step_v2heat(c):
    """Real per-token heat files of the build's clips from the frozen v2's stored token codes (scripts/eci/v2/run.py
    heat --picks: exact, its max over tokens equals the raw frame code) -> <work>/patch/ (sig 'v2heat-'). CPU."""
    dom_name, *_ = new_paths()
    dom = {'mice': 'mice', 'ants': 'ants', 'frogs': 'frogs', 'tadpoles': 'frogs'}[dom_name]
    (c.work / 'patch').mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/eci/v2/run.py'), 'heat', '--domain', dom, '--series',
           V2_SERIES[A.dom], '--picks', str(c.work / 'picks.json'), '--out-dir', str(c.work / 'patch')]
    log('v2heat:', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def step_render_new(c):
    """Clip frames staged to $LOCAL_DIR (one sequential tar of the needed JPGs), then build_explorer step_render."""
    picks = json.loads((c.work / 'picks.json').read_text())
    loc = Path(os.environ.get('LOCAL_DIR', '')) / 'ds'
    if os.environ.get('LOCAL_DIR'):
        lst = loc.parent / 'frames.txt'
        loc.mkdir(parents=True, exist_ok=True)
        rows = {r for n in picks['by_key'].get('max', {}).values()
                for rr in BE.clip_rows(n, c.kr, BE.SHOW_LENGTHS).values() for r in rr}  # the rendered clips only
        need = sorted({picks['frame_path'][str(r)] for r in rows})
        if c.dom.name == 'tadpoles':
            tad_crops(picks, rows, loc)
        else:
            lst.write_text('\n'.join(need) + '\n')
            subprocess.run(f'tar -C {BE.DATASET.resolve()} -cf - -T {lst} | tar -C {loc} -xf -', shell=True, check=True)
        log(f'{c.tag} render: staged {len(need)} frames (of {len(set(picks["frame_path"].values()))} picked) to {loc}')
        BE.DATASET = loc
    BE.step_render(c, A.overwrite, A.pack_mb)


def dump(c, entries, out):
    """page_data + split_data of the build -> out/core.json ({entries: [core, ...]}) and out/data/*.json."""
    d = BE.page_data(c)
    annotate_status(d['results'], c.outcomes)
    annotate_status(d['xresults'], c.extras)
    if c.dom is BE.MICE:
        d['results'] = block_dines_A(d['results'], [o['id'] for o in c.outcomes if o['id'].startswith('d')], c.order)
        d['xresults'] = block_dines_A(d['xresults'], [o['id'] for o in c.extras if o['id'].startswith('d')], c.order)
    core, files = BE.split_data(d, 'assets/data', ('', ''))
    (out / 'data').mkdir(parents=True, exist_ok=True)
    for f in (out / 'data').glob('*.json'):
        f.unlink()
    for f, x in files.items():
        (out / 'data' / f).write_text(BE.jdump(x))
    picks = json.loads((c.work / 'picks.json').read_text())
    sigs = [str(np.load(BE.patch_file(c, 'max', j))['sig']) for j in picks['by_key'].get('max', {})]
    real = lambda x: x.startswith('r4heat-') or x.startswith('v2heat-')  # noqa: E731
    heat = 'real' if sigs and all(real(x) for x in sigs) else 'peak'
    log(f'dump {c.tag}: heat maps {heat} ({sum(real(x) for x in sigs)}/{len(sigs)} real)')
    if heat != 'real' and A.variant != 'd0':  # the page never shows the blurred peak-patch stand-in
        raise SystemExit(f'dump {c.tag}: {sum(not x.startswith("r4heat-") for x in sigs)} neurons without real heat '
                         f'maps: run scripts/eci/atlas_v37_r4heat_key.sh {A.dom} (MOVE=1) first')
    core['heat_kind'] = heat
    if A.dom.startswith('v2_'):
        cc = json.loads((c.dom.eci_dir / 'codes' / c.sae / 'config.json').read_text())
        core['v2spec'] = [cc['S'], cc['G']]
    for a in core.get('analyses', []):  # mice family S: its own page family (BE labels it as a family B contrast)
        if str(a['id']).startswith('S_'):
            g = a['id'].split('_')[1] if a['id'].count('_') == 2 else 'all'
            a.update(family='S', genotype=g, stage=int(a['id'].rsplit('stage', 1)[1]))
    if c.dom.name == 'tadpoles':  # tadpole comparisons get the prefix tad_ (the juvenile ones share the names)
        core['tadpole'] = True
        core['tadpole_clips'] = True
        core['domain'] = 'frogs'
        core['arena_note'] = 'every frame: the crop-grid cell of the strongest token (the crop follows the tadpole, so this is a body map, not an arena map)'
        ren = lambda t: t.replace('"FoxP1_vs_WT"', '"tad_FoxP1_vs_WT"').replace('"En1_vs_WT"', '"tad_En1_vs_WT"') \
            .replace('|FoxP1_vs_WT|', '|tad_FoxP1_vs_WT|').replace('|En1_vs_WT|', '|tad_En1_vs_WT|')  # noqa: E731
        core = json.loads(ren(json.dumps(core)))
        for f in (out / 'data').glob('*.json'):
            f.write_text(ren(f.read_text()))
    cores = entries(core)
    (out / 'core.json').write_text(json.dumps({'entries': cores, 'videos_mice': BE._vt()[1] if c.dom is BE.MICE else None},
                                              separators=(',', ':')))
    log(f'dump {out}: {len(files)} data files, {sum(len(BE.jdump(x)) for x in files.values()) / 1e6:.2f} MB; entries '
        + ', '.join(f'{e["sae"]} ({len(e["outcomes"])} outcomes)' for e in cores))


def split_methods(core, key, model_value):
    """One build entry (NES + DINES outcomes) -> [NES entry, DINES entry] sharing every data file."""
    out = []
    for meth, pick in (('NES', lambda i: not i.startswith('d')), ('DINES', lambda i: i.startswith('d'))):
        e = copy.deepcopy(core)
        e['outcomes'] = [o for o in core['outcomes'] if pick(o['id'])]
        if not e['outcomes']:
            continue
        e['rfiles'] = {k: v for k, v in core['rfiles'].items() if pick(k)}
        e['lwin'] = {k: v for k, v in (core.get('lwin') or {}).items() if pick(k)}
        e['sae'] = f'{core["sae"]}@{meth.lower()}'
        e['model'] = {'model': model_value, 'method': meth}
        e['build'] = key
        out.append(e)
    return out


# ---------------------------------------------------------------------- online model: DINES + family C
def online_paths():
    dom_name, lay, res, tag = ONLINE[A.dom]
    res = ROOT / res
    if A.dom == 'ants_v4':  # its DINES run wrote to <layout>/methods/ants_<layout>/<sae>
        M = ROOT / 'results/vision/ants' / lay / 'methods' / f'ants_{lay}' / res.name
    else:
        M = METHODS / METHOD_KEY[f'online_{A.dom}'] / res.name
    return dom_name, res.name, res, OUT / 'res' / f'online_{A.dom}' / res.name, OUT / 'build' / f'online_{A.dom}', \
        M, tag


def step_prep_online():
    dom_name, sae, src, V, bout, M, tag = online_paths()
    (V / '_cache').mkdir(parents=True, exist_ok=True)
    for f in ('video_summaries_max.npz', 'bout_summaries_max.npz', 'latency.npz'):
        link((src / '_cache' / f).resolve(), V / '_cache' / f)
    for f in (src / '_cache').glob('bout_thresholds*.npz'):
        link(f.resolve(), V / '_cache' / f.name)
    have = {}
    nes_src = [src / 'pairs'] if (src / 'pairs' / 'summary.csv').exists() else [src]  # build_explorer rdir_of
    for meth, srcs in (('dines', [M / 'dines', M / 'familyC_dines']), ('nesC', [M / 'familyC_nes']),
                       ('nes', nes_src if A.dom == 'ants_v4' else [])):
        have[meth] = []
        for kind, sub, *_ in OUTS:
            ss = [s / sub if sub != '.' else s for s in srcs if (s / sub).is_dir()]
            if ss and mirror_outcome(V / meth / sub if sub != '.' else V / meth, ss):
                have[meth].append(kind)
    (V / 'prep.json').write_text(json.dumps({'sae': sae, 'methods_src': str(M), 'have': have}, indent=1))
    log(f'prep online {A.dom}: {V}; outcomes {have}')


def step_dump_online():
    """The online model's new searches (DINES; mice: family C NES and DINES) with the v36 outcome ids, + no-clip
    entries (arena map, per-video values) for their neurons that have no v36 neuron entry."""
    dom_name, sae, src, V, bout, M, tag = online_paths()
    dom = DOMOBJ[dom_name]
    if dom is BE.MICE:  # v36 analysed habituation on its whole 30 min
        _vw = BE.video_windows
        BE.video_windows = lambda cfg, ids, window='full': _vw(cfg, ids, 'hwhole' if window == 'full' else window)
    v36 = v36_all()
    e36 = next(e for e in v36['saes'] if e['tag'] == tag)
    have36 = set(e36['nfile']['max']['of'])
    bout.mkdir(parents=True, exist_ok=True)
    (bout / 'data').mkdir(exist_ok=True)
    c = make_cfg(V, sae, dom, tag, bout)
    ar = np.load(src / '_cache' / 'explorer' / 'arena.npz')
    vm = BE._vt(dom)[0]
    res_all, analyses = {}, {}
    for meth in ('dines', 'nesC'):
        if not (V / meth).is_dir():
            continue
        c.outcomes, c.extras = [], []
        outs = add_outcomes(c, V / meth, '', V / '_cache', dines=meth == 'dines')
        if not outs:
            continue
        R = annotate_status(BE.search_results(c, outs, analyses, top=True), outs)
        vals, extra = {}, {}
        for o in outs:
            ids, Y = BE.video_outcome(c, o)
            ix = ids.get_indexer(vm['observation_id'].astype(str))
            js = sorted({r['neuron'] for k, x in R.items() if k.startswith(o['id'] + '|') for r in x['rows']}
                        | {t[0] for k, x in R.items() if k.startswith(o['id'] + '|') for t in x.get('top', [])})
            vals[o['id']] = {str(j): [BE.sig4(v) for v in Y[ix, int(j)]] for j in js}
            if o['kind'] != 'latency':
                print(f'[{tag}] {meth} {o["id"]}: tau check vs per-video values, max relative diff '
                      f'{BE.tau_check(vm, vals[o["id"]], R, o["id"], dom=dom):.2e}')
            for j in js:
                if str(j) not in have36:
                    extra[str(j)] = {'noclip': True, 'artefact': False, 'arena': BE.q16(ar['act'][:, int(j)]),
                                     'vals': {}}
        for j, e in extra.items():
            e['vals'] = {oid: v[j] for oid, v in vals.items() if j in v}
        res_all[meth] = {'results': R, 'vals': vals, 'extra': extra,
                         'lwin': {o['id']: {aid: BE.sig4(BE.latency_W(o, aid) / dom.fps) for aid in c.order
                                            if (BE.primary_rows(o)['analysis_id'] == aid).any()}
                                  for o in outs if o['kind'] == 'latency'},
                         'outcomes': [o['id'] for o in outs],
                         'analyses_meta': [analyses[a] for a in c.order if a in analyses]}
        log(f'online {A.dom} {meth}: ' + ', '.join(
            f'{o["id"]} {sum(len(x["rows"]) for k, x in R.items() if k.startswith(o["id"] + "|"))} rows' for o in outs)
            + f'; {len(extra)} neurons without a v36 entry')
    (bout / 'online.json').write_text(json.dumps({'tag': tag, 'sae': sae, 'methods': res_all,
                                                  'analyses': analyses}, separators=(',', ':')))
    log(f'dump online {A.dom} -> {bout}/online.json')


def step_dump_online_v4():
    """Online model on ants v4 / v4_infested: NES and DINES entries of the v36 SAE on the v4-only layout. No clips (the
    old model's clip packs are not extended): every listed neuron gets a no-clip entry (the v36 arena map, per-video
    values); the same neuron number has its clips on the online v2 / v3 comparisons (same SAE)."""
    dom_name, sae, src, V, bout, M, tag = online_paths()
    dom = DOMOBJ[dom_name]
    v36 = v36_all()
    e36 = next(e for e in v36['saes'] if e['tag'] == tag)
    ar = np.load(ROOT / ONLINE['ants'][2] / '_cache' / 'explorer' / 'arena.npz')
    vm, videos, obs, _ = BE._vt(dom)
    (bout / 'data').mkdir(parents=True, exist_ok=True)
    for f in (bout / 'data').glob('*.json'):
        f.unlink()
    c = make_cfg(V, sae, dom, tag + 'v4', bout)
    entries = []
    for meth, base in (('NES', V / 'nes'), ('DINES', V / 'dines')):
        if not base.is_dir():
            log(f'online v4: no {meth} results in {base}')
            continue
        c.outcomes, c.extras = [], []
        outs = add_outcomes(c, base, '', V / '_cache', dines=meth == 'DINES')
        analyses = {}
        R = annotate_status(BE.search_results(c, outs, analyses, top=True), outs)
        x = f'{tag}v4{meth.lower()}'
        rfiles, nb, lw = {}, {}, {}
        for o in outs:
            ids, Y = BE.video_outcome(c, o)
            ix = ids.get_indexer(vm['observation_id'].astype(str))
            assert (ix >= 0).all()
            Ro = {k: v for k, v in R.items() if k.startswith(o['id'] + '|')}
            js = sorted({str(r['neuron']) for v in Ro.values() for r in v['rows']}
                        | {str(t[0]) for v in Ro.values() for t in v.get('top', [])}, key=int)
            vals = {j: [BE.sig4(v) for v in Y[ix, int(j)]] for j in js}
            if o['kind'] != 'latency':
                print(f'[{x}] {o["id"]}: tau check {BE.tau_check(vm, vals, Ro, o["id"], dom=dom):.2e}')
            else:
                lw[o['id']] = {aid: BE.sig4(BE.latency_W(o, aid) / dom.fps) for aid in c.order
                               if (BE.primary_rows(o)['analysis_id'] == aid).any()}
            for j in js:
                nb.setdefault(j, {'noclip': True, 'artefact': False, 'arena': BE.q16(ar['act'][:, int(j)]),
                                  'vals': {}})['vals'][o['id']] = vals[j]
            f = f'{x}_r_{o["id"]}.json'
            (bout / 'data' / f).write_text(json.dumps({'results': Ro, 'vals': {o['id']: vals}}, separators=(',', ':')))
            rfiles[o['id']] = f'assets/data/{f}'
        nf = f'{x}_n.json'
        (bout / 'data' / nf).write_text(json.dumps({'by': {'max': nb}}, separators=(',', ':')))
        e = {k: copy.deepcopy(e36[k]) for k in ('rep', 'artefact', 'artefact_text', 'arena_note', 'montage', 'arena',
                                                 'subject', 'subjects', 'top') if k in e36}
        e.update(sae=f'{e36["sae"]}@v4{meth.lower()}', tag=x, model={'model': 'online', 'method': meth},
                 outcomes=[copy.deepcopy(next(q for q in e36['outcomes'] if q['id'] == o['id'])) for o in outs],
                 analyses=[analyses[a] for a in c.order if a in analyses], subsets={}, rfiles=rfiles, pre={},
                 lwin=lw, nfile={'max': {'files': [f'assets/data/{nf}'], 'of': {j: 0 for j in nb}}},
                 domain='ants', sub_ui=False, videos=videos, obs=obs, vnote='v4 and v4_infested t=2, 6',
                 n_frames=int(len(np.load(dom.eci_dir / 'codes' / sae / 'n_fg.npy', mmap_mode='r'))),
                 noclip_note='Online model on v4: no clips rendered (the old model\'s clip packs are reused, not '
                             'extended); the same neuron number has clips on the online v2 / v3 comparisons',
                 note='Online model (the v36 SAE, unchanged) applied to the Exp4 series v4 / v4_infested (30 min; '
                      'codes of the 30-min ants layout, re-checked by scripts/eci/online_v4.py); no recording-day '
                      'conditioning, as the v36 ants searches. No clips for these comparisons.')
        entries.append(e)
        log(f'online v4 {meth}: ' + ', '.join(
            f'{o["id"]} {sum(len(v["rows"]) for k, v in R.items() if k.startswith(o["id"] + "|"))} rows' for o in outs)
            + f'; {len(nb)} neurons (no clips)')
    (bout / 'core.json').write_text(json.dumps({'entries': entries}, separators=(',', ':')))
    log(f'dump online v4 -> {bout}/core.json')


# ---------------------------------------------------------------------- new model: frog tadpoles
TAD_EXP = 'tadpoles (stage 44–48)'  # the frogs phase selector value (the juveniles are experiment 'hour 1')
TAD_QC = R4 / 'frogs_tadpole' / 'qc.json'
TAD_NESS = R4 / 'ness' / 'frogs_tadpole'


def tad_setup(clips=False):
    """The tadpole domain standing in for the juvenile frogs in build_explorer (contrasts WT vs FoxP1 / En1, the QC
    exclusion). clips: also the standard clip steps (data / v2heat / arena / render / dump) on it: the experiment label
    of the frogs view = TAD_EXP, and annotations get a synthetic frame_path tadcrop/<obs>/frame_<k>.jpg (the crops
    are cut from the raw video at render time, step_render_new)."""
    Dm.TadpolesDomain.CONTRASTS = (('WT', 'FoxP1'), ('WT', 'En1'))
    excl = set(json.loads(TAD_QC.read_text())['excluded'])
    _ld, _vm = Dm.TadpolesDomain.load_design, Dm.TadpolesDomain.video_meta

    def load_qc(self):
        d = _ld(self)
        d = d[~d['observation_id'].isin(excl)].reset_index(drop=True)
        d['obs_row'] = np.arange(len(d))
        return d

    def vmeta_qc(self):
        v = _vm(self)
        v = v[~v['observation_id'].isin(excl) & v['group'].isin(['WT', 'FoxP1', 'En1'])].reset_index(drop=True)
        v = v.assign(mutant_side='')  # bilateral crispants: no edited side
        return v.assign(experiment=TAD_EXP) if clips else v
    Dm.TadpolesDomain.load_design, Dm.TadpolesDomain.video_meta = load_qc, vmeta_qc
    TAD = Dm.get_domain('tadpoles')
    assert str(TAD.eci_dir).endswith('eci_tadpole_r4'), TAD.eci_dir
    assert TAD.analysis_ids() == ['FoxP1_vs_WT', 'En1_vs_WT'], TAD.analysis_ids()
    BE.VIEW['tadpoles'] = dict(BE.VIEW['frogs'], title='Frogs', exp=TAD_EXP, subject='tadpole', subjects='tadpoles')
    BE.FROGS = TAD  # build_explorer's frogs code paths (video table, arms, labels) on the tadpole domain
    DOMOBJ['tadpoles'] = TAD
    if clips:
        # clips only from the QC-kept videos of the three groups (VIEW 'keep' on the experiment column)
        BE.VIEW['tadpoles']['keep'] = {TAD_EXP: None}
        BE.VIEW['frogs'] = BE.VIEW['tadpoles']
        BE.GROUP_COLS['tadpoles'] = BE.GROUP_COLS['frogs']
        _prev = VZ.load_full_codes

        def load_tad(sae_name, dataset_dir='dataset', data_dir='data', subject='mice', version='v1', codes_dir=None,
                     domain=None):
            d = Path(codes_dir) if codes_dir else TAD.eci_dir / 'codes' / sae_name
            ann = pd.read_csv(TAD.ann_path, usecols=['observation_id', 'frame_idx', 'fps'])
            ann['frame_path'] = [f'tadcrop/{o}/frame_{k:06d}.jpg' for o, k in zip(ann.observation_id, ann.frame_idx)]
            vm = _vm(TAD).assign(mutant_side='')  # every recorded video (the QC-excluded ones are never shown)
            meta = ann.merge(vm, on='observation_id', how='left', validate='m:1')
            keep = ~meta['observation_id'].isin(excl) & meta['group'].isin(['WT', 'FoxP1', 'En1'])
            meta['experiment'] = np.where(keep, TAD_EXP, 'not shown')
            codes = {k: np.load(d / 'codes_max.npy', mmap_mode='r') for k in ('mean', 'max')}
            assert codes['max'].shape[0] == len(meta), (codes['max'].shape, len(meta))
            return VZ.CodeSource('full', codes, meta, Path(dataset_dir), {'codes_dir': str(d), 'n_rows': len(meta),
                                                                          'meta_cols': tuple(TAD.meta_cols) + ('experiment', 'T')})
        VZ.load_full_codes = load_tad
    return TAD


def tad_crops(picks, rows, loc):
    """Cut the tadpole crops of the given picks rows from the raw videos (tadpole_r4: the 5 fps frame k = source frame
    12 k + 5, the locate_v2 dead-band box of frame k), resized to 384 px like the encoder input -> loc/tadcrop/..."""
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    sys.path.insert(0, str(ROOT / 'scripts/eci'))
    import tadpole_r4 as T
    P = json.loads((R4 / 'frogs_tadpole' / 'plan.json').read_text())
    vb = {v['obs']: v for v in P['videos']}
    need = {}
    for r in rows:
        f = picks['frame_path'][str(r)]
        o, k = f.split('/')[1], int(f.split('frame_')[1][:6])
        need.setdefault(o, set()).add(k)

    def one(o):
        v, L = vb[o], T.load_locate(o)
        ks = sorted(need[o])
        runs, a = [], ks[0]
        for i in range(1, len(ks) + 1):
            if i == len(ks) or ks[i] != ks[i - 1] + 1:
                runs.append((a, ks[i - 1]))
                if i < len(ks):
                    a = ks[i]
        (loc / 'tadcrop' / o).mkdir(parents=True, exist_ok=True)
        for a0, a1 in runs:
            for j, fr in enumerate(T.decode_iter(v['path'], v['w'], v['h'], a0, a1 - a0 + 1, threads=2)):
                k = a0 + j
                im = Image.fromarray(np.ascontiguousarray(T.crop(fr, int(L['x0'][k]), int(L['y0'][k]))))
                im.resize((384, 384), Image.BICUBIC).save(loc / 'tadcrop' / o / f'frame_{k:06d}.jpg', quality=92)
        return len(ks)
    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(one, sorted(need)))
    log(f'tadpole crops: {n} frames of {len(need)} videos cut from the raw videos -> {loc}/tadcrop')


def step_dump_tadpoles():
    """New model on the frog tadpoles: NES (results/vision/frogs/eci_tadpole_r4/nes/<sae>) and NESS
    (results/vision/eci_rollout_r4/ness/frogs_tadpole/<sae>/dines) entries of the d2 fnorm SAE, read through the
    build_explorer search functions with the tadpole domain standing in for the juvenile frogs (contrasts WT vs FoxP1,
    WT vs En1; the QC exclusion of qc.json, as scripts/eci/tadpole_r4_nes.py ran them). No clips and no arena map:
    the codes are of a moving body crop of the raw video (no dataset frames, no fixed arena). Analysis ids get the
    prefix 'tad_' (the juvenile comparisons share the names). -> OUT/build/frogs_tad_<variant>/{core.json, data/}."""
    TAD = tad_setup()
    excl = set(json.loads(TAD_QC.read_text())['excluded'])
    v2m = A.dom == 'v2_frogs_tad'  # the frozen v2 (its frogs SAE) on the tadpoles
    if v2m:
        sae = v2_name('frogs') + V2_SUF[A.variant]
        ness = V2 / 'ness' / 'frogs_tadpole' / sae
    else:
        sae = best_sae('frogs_tadpole', A.variant) + '_fnorm'
        ness = TAD_NESS / sae
    src = ROOT / NEW[A.dom][2] / sae
    key = f'{A.dom}_{A.variant}'
    tg = ('v2tad' + {'smooth': 's', 'thr': 't', 'prob': 'p'}[A.variant]) if v2m else 'r4tad' + A.variant
    V, bout = OUT / 'res' / key / sae, OUT / 'build' / key
    (V / '_cache').mkdir(parents=True, exist_ok=True)
    for f in (src / '_cache').glob('*.npz'):
        link(f.resolve(), V / '_cache' / f.name)
    for meth, base in (('nes', src), ('dines', ness / 'dines')):
        for kind, sub, *_ in OUTS:
            if (base / sub / 'summary.csv').exists():
                mirror_outcome(V / meth / sub if sub != '.' else V / meth, [base / sub if sub != '.' else base])
    vm, videos, obs, _ = BE._vt(TAD)
    videos = [[TAD_EXP] + v[1:] for v in videos]  # video_table_frogs names the experiment by VIEW['frogs'] ('hour 1')
    assert len(vm) == json.loads(TAD_QC.read_text())['n_videos_kept'], len(vm)
    (bout / 'data').mkdir(parents=True, exist_ok=True)
    for f in (bout / 'data').glob('*.json'):
        f.unlink()
    c = make_cfg(V, sae, TAD, tg, bout)
    ren = lambda a: f'tad_{a}'  # noqa: E731
    entries = []
    for meth, base in (('NES', V / 'nes'), ('DINES', V / 'dines')):
        if not base.is_dir():
            log(f'tadpoles: no {meth} results in {base}')
            continue
        c.outcomes, c.extras = [], []
        outs = add_outcomes(c, base, '' if meth == 'NES' else 'd', V / '_cache', dines=meth == 'DINES')
        analyses = {}
        R = annotate_status(BE.search_results(c, outs, analyses, top=True), outs)
        R = {'|'.join([k.split('|')[0], ren(k.split('|')[1])] + k.split('|')[2:]): v for k, v in R.items()}
        x = f'{tg}{meth.lower()}'
        rfiles, nb, lw = {}, {}, {}
        for o in outs:
            ids, Y = BE.video_outcome(c, o)
            ix = ids.get_indexer(vm['observation_id'].astype(str))
            assert (ix >= 0).all()
            Ro = {k: v for k, v in R.items() if k.startswith(o['id'] + '|')}
            js = sorted({str(r['neuron']) for v in Ro.values() for r in v['rows']}
                        | {str(t[0]) for v in Ro.values() for t in v.get('top', [])}, key=int)
            vals = {j: [BE.sig4(v) for v in Y[ix, int(j)]] for j in js}
            if o['kind'] != 'latency':
                Rt = {'|'.join([k.split('|')[0], k.split('|')[1][4:]] + k.split('|')[2:]): v for k, v in Ro.items()}
                print(f'[{x}] {o["id"]}: tau check {BE.tau_check(vm, vals, Rt, o["id"], dom=TAD):.2e}')
            else:
                lw[o['id']] = {ren(aid): BE.sig4(BE.latency_W(o, aid) / TAD.fps) for aid in c.order
                               if (BE.primary_rows(o)['analysis_id'] == aid).any()}
            for j in js:
                nb.setdefault(j, {'noclip': True, 'artefact': False, 'vals': {}})['vals'][o['id']] = vals[j]
            f = f'{x}_r_{o["id"]}.json'
            (bout / 'data' / f).write_text(json.dumps({'results': Ro}, separators=(',', ':')))
            rfiles[o['id']] = f'assets/data/{f}'
        nf = f'{x}_n.json'
        (bout / 'data' / nf).write_text(json.dumps({'by': {'max': nb}}, separators=(',', ':')))
        ans = []
        for a in c.order:
            if a in analyses:
                m = dict(analyses[a], id=ren(a), experiment=TAD_EXP)
                m['arms'] = [[n, [[TAD_EXP, v] for _, v in cells]] for n, cells in m['arms']]
                ans.append(m)
        po = []
        for o in outs:
            q = {k: o[k] for k in ('id', 'label', 'codes', 'unit', 'kind', 'agg')}
            q.update(bout=o['kind'] == 'rate', prefixes=sorted(int(p) for p in BE.primary_rows(o)['prefix'].unique()
                                                               if int(p) in (128, 256)),
                     setting=str(BE.primary_rows(o)['setting'].iloc[0]))
            po.append(q)
        e = {'sae': f'{sae}@{meth.lower()}', 'tag': x, 'model': {'model': V2_MODEL[A.variant] if v2m else 'new', 'method': meth},
             'domain': 'frogs',
             'outcomes': po, 'analyses': ans, 'subsets': {}, 'rfiles': rfiles, 'pre': {}, 'lwin': lw,
             'nfile': {'max': {'files': [f'assets/data/{nf}'], 'of': {j: 0 for j in nb}}},
             'artefact': [], 'artefact_text': {}, 'sub_ui': False, 'videos': videos, 'obs': obs,
             'arms': ['WT', 'FoxP1', 'En1'], 'subject': 'tadpole', 'subjects': 'tadpoles', 'top': 3,
             'vnote': 'tadpoles WT, FoxP1, En1 (hour 1)', 'tadpole': True, 'heat_kind': 'none',
             **({'v2spec': [384, 24]} if v2m else {}),
             'n_frames': int(len(np.load(TAD.eci_dir / 'codes' / sae / 'n_fg.npy', mmap_mode='r'))),
             'noclip_note': 'Tadpoles: no clips and no arena map: the model sees a crop that follows the tadpole in '
                            'the raw video (no dataset frames, no fixed arena)'}
        entries.append(e)
        log(f'tadpoles {meth}: ' + ', '.join(
            f'{o["id"]} {sum(len(v["rows"]) for k, v in R.items() if k.startswith(o["id"] + "|"))} rows' for o in outs)
            + f'; {len(nb)} neurons (no clips); {len(vm)} videos')
    (bout / 'core.json').write_text(json.dumps({'entries': entries}, separators=(',', ':')))
    log(f'dump tadpoles -> {bout}/core.json')


# ---------------------------------------------------------------------- assemble
def v36_all():
    s = V36_PAGE.read_text()
    i = s.find('const ALL = ')
    j = s.find('\n', i)
    return json.loads(s[i + len('const ALL = '):j].rstrip(';'))


KEEP_IDS = ('boutratemaxpool', 'meanactivationma', 'xlatencymax')  # the v36 max-pooling outcomes
MODEL_TIPS = {'online': 'v1 (deployed): DINOv2-base at 448 px, the animal mask, token only, Matryoshka BatchTopK SAE '
                        'with 1024 neurons (k 16), max pooling',
              'new': 'v2 (frozen 2026-10-20): DINOv3-L/16 block 20 at native-capped resolution, canonical rotation '
                     '(mice odour corner top right, frogs mutant side right), tight animal mask, no temporal channel, '
                     'one Matryoshka BatchTopK SAE per domain (2048 neurons, k 16), max over kept tokens, 0.6 s '
                     'smoothing; Outcome selects continuous (default) or binary codes',
              'new (thr)': 'v2 with the firing threshold: the same frozen v2 SAE, its 0.6 s smoothed codes kept only '
                           'above each neuron\'s fnorm firing threshold (the pre-specified NES input)',
              'new_r4': 'v2 (in progress): DINOv3-L/16 at an input sized to the animal, the same lab masks, token + its '
                     'change over 2 s, Matryoshka BatchTopK SAE with 512 neurons (k 16), max pooling',
              'new (no Δ)': 'v2 without the 2 s change half (variant d0)'}
METHOD_TIPS = {'NES': 'Neural Effect Search (Mencattini, Cadei, Locatello 2025): add the most significant neuron, '
                      're-test the others given the ones found, repeat until nothing is significant (Bonferroni)',
               'DINES': 'Direct-Indirect Neural Effect Search (successor of NES, src/eci/dines.py): a forward phase '
                        'ordered by p-value, then a backward check of every found neuron given every subset of the '
                        'others (generalized covariance measure tests, cross-fitted); reports the neurons changed '
                        'directly by the treatment',
               'NESS': 'NESS: minimal neuron sets that carry the treatment information (design draft, '
                       'docs of commit e04f371); not implemented yet'}
METHOD_NOTES = {'DINES': 'DINES: reported neurons = kept (significant given every tested subset of the other found '
                         'neurons) plus undetermined (significant except given a set of neurons that nearly reveals '
                         'the treatment). τ = the plain difference of the per-video means (paired: mean change), p = '
                         'the largest p over the tested sets. No robustness chips: the sensitivity grid was run for '
                         'NES only. Family A (paired) is not shown for DINES (see the comparison). Covariate: '
                         'every DINES search conditions on the per-video foreground size (n_fg; new ants model: and '
                         'the recording day), as the new model\'s NES does; the online NES searches did not.'}


def rw_json(src, dst, amap=None):
    """Copy a data file, mapping local pack references to their asset-store URLs."""
    t = Path(src).read_text()
    for name, url in (amap or {}).items():
        t = t.replace(f'assets/{name}', url)
    Path(dst).write_text(t)


# ---- design spec per entry (the model bar line + its Details panel). Sources: online = the codes config.json of each
# online SAE (dataset/<dom>/.../codes/<sae>/config.json: facebook/dinov2-base, last_hidden_state after the final
# LayerNorm, 512 px frame resized to 448 = 32 x 32 patches, motion_delta 0, 'global threshold', max over foreground
# patches); new = scripts/eci/rollout_r4.py (docstring + DOMS: DINOv3-L/16 blocks 1-20 + final LayerNorm, S per domain,
# LAG 10 frames = 2 s, WIDTH 512, K 16, PREFIXES 128/256/512, 3 seeds, best held-out FVE) and scripts/eci/rollout_r4_nes.sh
# (--nuisance nfg, --day for ants); NESS covariates: METHOD_NOTES above; outcome settings: the entries' own 'setting'.
R4_SIZE = {'frogs_tad': (384, 24), 'mice': (1024, 64), 'ants_v2p': (768, 48), 'ants_v3p': (832, 52), 'ants_v4p': (816, 51), 'frogs': (800, 50)}
MASK = {'mice': 'mouse mask (rule fg448, per-video backgrounds)', 'ants': 'ant mask (rule ants, per-video backgrounds)',
        'frogs': 'frog mask (rule frogs, per-video backgrounds)'}


def spec_of(e):
    """The compact Details cells of an entry (page table v0 | v1 | v2; v0 = the ECI paper recipe, not filled yet)."""
    dom, online, meth = e['domain'], e['model']['model'] == 'online', e['model']['method']
    key = e['sae'].split('#')[1].rsplit('_', 1)[0] if '#' in e['sae'] else None
    if e['model']['model'] in ('new', 'new (thr)') and (e.get('v2spec') or (key or '').startswith('v2_')):
        S, G = e.get('v2spec', (None, None))
        cells = {'fm': f'DINOv3-L/16 block 20 · {S} px native-capped ({G}×{G} patches)' if S else 'DINOv3-L/16 block 20 · native-capped',
                 'input': 'token', 'mask': {'mice': 'tight mouse mask · odour corner rotated to top right',
                                            'ants': 'tight ant mask', 'frogs': 'tight frog mask · mutant side right'}[dom]
                 + (' · tadpole crop (96 px → 384)' if e.get('tadpole') else ''),
                 'sae': 'Matryoshka BatchTopK · 2048 · k=16 · one SAE per domain',
                 'pool': 'max over the mask · 0.6 s smoothing' + (' · no firing threshold' if e['model']['model'] == 'new'
                                                                   else ' · fnorm firing threshold'),
                 'cov': 'foreground size (n_fg)' + (' + recording day (not for t=8 vs t=2, infested vs healthy)'
                                                    if dom == 'ants' else '')}
        if dom == 'mice':
            cells['hab'] = 'last 15 min'
        return {'cells': cells}
    if online:
        fm = 'DINOv2-base · 448 px (32×32 patches)'
        sae = 'Matryoshka BatchTopK · 1024 · k=16'
        mask = {'mice': 'mouse mask · odor corner rotated to top right', 'ants': 'ant mask',
                'frogs': 'frog mask · mutant side mirrored right'}[dom]
        cov = 'none' if meth == 'NES' else 'foreground size (n_fg)'
    else:
        S, G = R4_SIZE[key]
        fm = f'DINOv3-L/16 · {S} px ({G}×{G} patches)'
        sae = 'Matryoshka BatchTopK · 512 · k=16' + (' · one SAE for all series' if dom == 'ants' else '')
        mask = {'mice': 'mouse mask · not rotated', 'ants': 'ant mask', 'frogs': 'frog mask · mutant side mirrored right'}[dom]
        if key == 'frogs_tad':  # tadpoles: a crop that follows the tadpole (scripts/eci/tadpole_r4.py)
            fm, mask = f'DINOv3-L/16 · {S} px crop around the tadpole ({G}×{G} patches)', 'frog pixel mask on the crop'
            sae = 'Matryoshka BatchTopK · 512 · k=16 · own tadpole SAE'
        cov = 'foreground size (n_fg)' + (' + recording day (not for t=8 vs t=2, infested vs healthy)' if dom == 'ants' else '')
    cells = {'fm': fm, 'input': 'token' if online else 'token + 2 s change', 'mask': mask, 'sae': sae,
             'pool': 'max over the mask', 'cov': cov}
    if dom == 'mice':
        cells['hab'] = 'whole 30 min' if online else 'last 15 min'
    return {'cells': cells}


# ---- v0 = the original ECI paper recipe (scripts/eci/v0/: DINOv2-S/14 224 px tokens, TopK SAE 4608 k=20 applied to
# the mean-pooled token, video mean, single-pass t-test per code, no covariates). Its searches are shown as numbers in
# the Details panel only (no clips, no neuron pages). Sources: ants v2 = results/vision/ants/eci_v0/ (+ checks.json),
# the other series / domains = results/vision/<subject>/eci_v0_<domain>/results.json (scripts/eci/v0/v0_domains.py).
V0_RUNS = (('ants', 'ants_v3', 'results/vision/ants/eci_v0_ants_v3'), ('ants', 'ants_v4', 'results/vision/ants/eci_v0_ants_v4'),
           ('mice', 'mice_v1', 'results/vision/mice/eci_v0_mice_v1'), ('frogs', 'frogs_v1', 'results/vision/frogs/eci_v0_frogs_v1'))
V0_WIN = {'ants_v3': '30 min', 'ants_v4': '30 min', 'mice_v1': '15 min per stage, habituation its last 15', 'frogs_v1': 'hour 1'}
V0_LAB = {'ants': 'grooming', 'mice': 'nose contact (nn / np / nt, union)'}


def v0_table():
    """{domain: {analysis id: v0 summary}}: n per arm, testable codes, raw-p hits vs expected false positives,
    Bonferroni / FDR survivors, SAE quality on its real input, behaviour alignment, nuisance structure of the hits."""
    out = {}
    v2 = ROOT / 'results/vision/ants/eci_v0'
    if (v2 / 'results.json').exists():
        r = json.loads((v2 / 'results.json').read_text())
        chk = json.loads((v2 / 'checks.json').read_text()) if (v2 / 'checks.json').exists() else {}
        sp, q, g = r['single_pass'], r['quality'], r['all_codes_grooming']
        found = r['found']
        nuis = []
        for f in found:
            b = chk.get('found_background', {}).get(f['neuron'], {})
            if b:
                nuis.append(f"{f['neuron']}: position explains {100 * b['eta2_position']:.0f}% of the video means "
                            f"(permutation p {b['eta2_position_perm_p']:.3f})")
        bf = max(found, key=lambda f: f['f1_groom_any']) if found else None
        out.setdefault('ants', {})['v2_1_vs_2'] = {
            'n': f"t=1 {r['n_control']} · t=2 {r['n_treated']} videos (10 min, 2 fps)",
            'tested': sp['n_testable_codes'], 'hits': sp['n_found_raw_p05'], 'expected': sp['expected_false_positives'],
            'bonf': sp['n_found_bonferroni'], 'fdr': None, 'names': [f['neuron'] for f in found],
            'fve_tok': q['fve_tokens'], 'fve_pool': q['fve_meanpooled'], 'alive_pool': q['alive_codes_meanpooled_frames'],
            'align': (f"grooming: best found neuron F1 {bf['f1_groom_any']:.3f} ({bf['neuron']}) vs always-on F1 "
                      f"{chk.get('always_on_f1_groom_any', float('nan')):.3f}; best of all alive codes F1 "
                      f"{g['best_f1']['f1']:.3f}, AUROC {g['best_auroc']['auroc']:.3f}") if bf else '',
            'nuis': '; '.join(nuis), 'confound': ''}
    for dom, key, rel in V0_RUNS:
        f = ROOT / rel / 'results.json'
        if not f.exists():
            log(f'v0: {f} missing, {key} left out')
            continue
        r = json.loads(f.read_text())
        assert not r.get('smoke'), f'{f} is a smoke run'
        q, al = r['quality'], r.get('alignment', {})
        for a in r['analyses']:
            if a['kind'] == 'paired':
                n = f"{a['label']}: {a['n_pools']} pools, paired"
            elif a['kind'] == 'did':
                n = f"{a['label']}: {a['n_control']} wt · {a['n_treatment']} het pools"
            else:
                n = f"{a['label']}: {a['n_control']} control · {a['n_treatment']} treated videos"
            n += f" ({V0_WIN[key]}, 2 fps)"
            ali = ''
            if 'any' in al:
                x = al['any']
                bh = a.get('best_aligned_hit_any')
                ali = (f"{V0_LAB[dom]}: always-on F1 {x['always_on_f1']:.3f} (prevalence {x['prevalence']:.3f}); best of "
                       f"all alive codes F1 {x['best_f1']['f1']:.3f} ({x['best_f1']['neuron']}); AUROC farthest from 0.5 "
                       f"{x['best_auroc']['auroc']:.3f} ({x['best_auroc']['neuron']}{', fires less during it' if x['best_auroc']['auroc'] < 0.5 else ''})")
                if bh:
                    ali += f"; best found neuron F1 {bh['best_f1_any']:.3f} ({bh['best_f1_neuron']}), AUROC farthest from 0.5 {bh['auroc_any']:.3f} ({bh['neuron']})"
            ns = a.get('nuisance_summary', {})
            nuis = ''
            if ns:
                test = next(iter(ns.values()))['test']
                nuis = ('found neurons structured by a recording field (' + test + ' permutation p < 0.05; 5% expected by '
                        'chance): ' + ', '.join(f"{c.replace('_', ' ')} {v['n_perm_p05']}/{v['n_checked']}"
                                                for c, v in ns.items()))
            out.setdefault(dom, {})[a['id']] = {
                'n': n, 'tested': a['n_testable_codes'], 'hits': a['n_found_raw_p05'], 'expected': a['expected_false_positives'],
                'bonf': a['n_found_bonferroni'], 'fdr': a['n_found_fdr_bh'], 'names': [],
                'fve_tok': q.get('fve_tokens_every5th_batch'), 'fve_pool': q['fve_meanpooled'],
                'alive_pool': q['alive_codes_meanpooled_frames'], 'align': ali, 'nuis': nuis, 'confound': a['confound']}
        hits = pd.read_csv(ROOT / rel / 'found_neurons.csv') if (ROOT / rel / 'found_neurons.csv').stat().st_size > 2 else None
        if hits is not None and len(hits):
            for aid, g in hits.groupby('analysis'):
                out[dom][aid]['names'] = g.sort_values('p')['neuron'].head(8).tolist()
    return out


# ---- v0 as a selectable model (model value 'v0'): one entry per v0 run, the baseline next to v1 / v2. Its search =
# the original recipe's single-pass t-test per code (Welch; mice A: paired t-test over pools, C: Welch on the per-pool
# changes; scripts/eci/v0/v0_domains.py run_analysis): listed = raw p < 0.05, no correction (the paper's rule), each
# row chipped Bonferroni / raw p only; no hit -> the 3 smallest p as candidates (the page's candidates rule). Outcome =
# the per-video mean of the code (one temporal aggregation; no bouts or latency in v0). Neuron number = v0's N_<j+1>.
# No arena maps or heat maps (v0 codes one mean token per frame, no patches); frames: the strips of
# scripts/eci/v0/v0_atlas_frames.py (results/vision/eci_atlas_v37/v0/<key>/) when they exist.
V0_OID = 'v0meantok'
# key -> (domain, results dir, the page entry whose video list / comparison metadata v0 reuses (sae suffix or tag))
V0_ENTRIES = {'mice_v1': ('mice', 'results/vision/mice/eci_v0_mice_v1', 'fg448al'),
              'ants_v2': ('ants', 'results/vision/ants/eci_v0', '#ants_v2p_d2'),
              'ants_v3': ('ants', 'results/vision/ants/eci_v0_ants_v3', '#ants_v3p_d2'),
              'ants_v4': ('ants', 'results/vision/ants/eci_v0_ants_v4', '#ants_v4p_d2'),
              'frogs_v1': ('frogs', 'results/vision/frogs/eci_v0_frogs_v1', 'frogsfg')}
V0_FRAMES = OUT / 'v0'
V0_NUIS = {'odor_corner': 'odor corner (camera period)', 'line': 'gene line', 'sex': 'sex', 'date': 'recording date',
           'batch': 'batch', 'position': 'arena position', 'recording_date': 'recording day', 'nestbox': 'nest box',
           'session': 'recording session', 'slot': 'dish slot', 'day': 'recording day'}
MODEL_TIPS['v0'] = ('v0 (baseline): the original ECI paper recipe: DINOv2-S/14 at 224 px on the full frame, the mean of '
                    'its 256 patch tokens, TopK SAE with 4608 neurons (k 20) trained on single patch tokens, '
                    'single-pass t-test per neuron, raw p < 0.05, no correction, no covariates')


def v0_entries(site, saes, data):
    """The v0 page entries + their data files (site/assets/data/v0_<key>_{r,n}.json) and frame strips."""
    from scipy.stats import ttest_1samp, ttest_ind
    out = []
    for key, (dom, rel, ref_key) in V0_ENTRIES.items():
        res = ROOT / rel
        if not (res / 'results.json').exists():
            log(f'v0: {res} missing, {key} left out')
            continue
        R0 = json.loads((res / 'results.json').read_text())
        ref = next(e for e in saes if e['model'] == {'model': 'new' if ref_key.startswith('#') else 'online',
                                                       'method': 'NES'}
                   and ((e['sae'].endswith(ref_key) or e['sae'].endswith(ref_key.replace('#ants_', '#v2_ants_')
                                                                          .replace('_d2', '_smooth')))
                        if ref_key.startswith('#') else e['tag'] == ref_key))
        vids = pd.read_csv(res / 'videos.csv', dtype={'observation_id': str})
        X = np.load(res / 'video_codes.npy').astype(np.float64)
        obs = ref.get('obs') or data['obs']
        V = ref.get('videos') or data['videos']
        page_ix = {o[0]: i for i, o in enumerate(obs)}
        assert len(obs) == len(vids) and set(page_ix) == set(vids.observation_id), (key, len(obs), len(vids))
        Y = X[[vids.index[vids.observation_id == o[0]][0] for o in obs]]  # page video order
        P = pd.read_csv(res / 'pvalues_all_codes.csv')
        hits = pd.read_csv(res / 'found_neurons.csv')
        if key == 'ants_v2':  # single analysis; nuisance checks of its hits in checks.json
            sp = R0['single_pass']
            A0 = [{'id': 'v2_1_vs_2', 'kind': 'welch', 'n_testable_codes': sp['n_testable_codes'],
                   'n_found_raw_p05': sp['n_found_raw_p05'], 'n_found_bonferroni': sp['n_found_bonferroni'],
                   'n_control': R0['n_control'], 'n_treatment': R0['n_treated']}]
            P = P.rename(columns={'p': 'p_v2_1_vs_2'})
            hits['analysis'] = 'v2_1_vs_2'
            chk = json.loads((res / 'checks.json').read_text()).get('found_background', {})
            for c in ('position', 'batch', 'day'):
                hits[f'{c}_eta2'] = [chk.get(n, {}).get(f'eta2_{c}', np.nan) for n in hits.neuron]
                hits[f'{c}_perm_p'] = [chk.get(n, {}).get(f'eta2_{c}_perm_p', np.nan) for n in hits.neuron]
            hits = hits.rename(columns={'auroc_groom_any': 'auroc_any', 'f1_groom_any': 'f1_any'})
            align = {'lab': 'grooming', 'always_on_f1': json.loads((res / 'checks.json').read_text()).get(
                'always_on_f1_groom_any')}
        else:
            A0 = R0['analyses']
            al = R0.get('alignment', {}).get('any')
            align = {'lab': V0_LAB[dom], 'always_on_f1': al['always_on_f1']} if al else None
        ameta = {a['id']: a for a in ref['analyses']}
        FR = json.loads((V0_FRAMES / key / 'frames.json').read_text()) if (V0_FRAMES / key / 'frames.json').exists() \
            else None
        results, vals, an_out, shown = {}, {}, [], set()
        for a in A0:
            aid = a['id']
            m = copy.deepcopy(ameta[aid])
            # the test, recomputed on the page's per-video values (arms / pools as the page reads them)
            with np.errstate(invalid='ignore', divide='ignore'):
                if dom == 'mice':
                    pool = np.array([v[0] for v in V])
                    st, gt = np.array([v[1] for v in V]), np.array([v[2] for v in V])
                    if m['family'] == 'B':
                        c, t = (st == m['stage']) & (gt == 'wt'), (st == m['stage']) & (gt == 'het')
                        tt, p = ttest_ind(Y[t], Y[c], axis=0, equal_var=False)
                        tau, n_u = Y[t].mean(0) - Y[c].mean(0), int(c.sum() + t.sum())
                    else:
                        s0, s1 = m['stages']
                        pools = sorted({pool[i] for i in range(len(V)) if m['family'] == 'C' or gt[i] == m['genotype']})
                        ia = [int(np.where((pool == q) & (st == s0))[0][0]) for q in pools]
                        ib = [int(np.where((pool == q) & (st == s1))[0][0]) for q in pools]
                        D = Y[ib] - Y[ia]
                        if m['family'] == 'A':
                            tt, p = ttest_1samp(D, 0.0, axis=0)
                            tau = D.mean(0)
                        else:
                            h = gt[ia] == 'het'
                            tt, p = ttest_ind(D[h], D[~h], axis=0, equal_var=False)
                            tau = D[h].mean(0) - D[~h].mean(0)
                        n_u = len(pools)
                else:
                    m.setdefault('arms', [[str(m['control']), [[m['experiment'], m['control']]]],
                                          [str(m['treatment']), [[m['experiment'], m['treatment']]]]])
                    arm = lambda i: np.array([any(v[0] == e and v[1] == tv for e, tv in m['arms'][i][1])  # noqa: E731
                                              for v in V])
                    c, t = arm(0), arm(1)
                    assert (int(c.sum()), int(t.sum())) == (a['n_control'], a['n_treatment']), (key, aid)
                    tt, p = ttest_ind(Y[t], Y[c], axis=0, equal_var=False)
                    tau, n_u = Y[t].mean(0) - Y[c].mean(0), int(c.sum() + t.sum())
            p0 = P[f'p_{aid}'].to_numpy(float)
            ok = np.isfinite(p0)
            assert (np.isfinite(p) == ok).all() and np.allclose(p[ok], p0[ok], rtol=1e-4, atol=1e-6), \
                (key, aid, np.nanmax(np.abs(p - p0)))
            nt = int(ok.sum())
            assert nt == a['n_testable_codes'], (key, aid)
            sig = np.where(ok & (p < 0.05))[0]
            sig = sig[np.argsort(p[sig], kind='stable')]
            assert len(sig) == a['n_found_raw_p05'], (key, aid)
            if 'index0' in hits:  # the same hits as the run's own list
                assert set(sig.tolist()) == set(hits.loc[hits.analysis == aid, 'index0']), (key, aid)
            hh = hits[hits.analysis == aid].set_index('index0')
            rows = []
            for j in sig:
                h = hh.loc[j] if j in hh.index else None
                warn = []
                for col, lab in V0_NUIS.items():
                    for k in (f'{col}_perm_p_within_arm', f'{col}_perm_p'):
                        if h is not None and k in h.index and np.isfinite(h[k]):
                            if h[k] < 0.05:
                                e2 = h[k.replace('perm_p', 'eta2')]
                                warn.append([lab, f'{lab} explains {100 * e2:.0f}% of the per-unit values'
                                                  f'{" within arm" if "within" in k else ""} (permutation p '
                                                  f'{h[k]:.3f})'])
                            break
                ali = None
                if h is not None and 'auroc_any' in h.index and np.isfinite(h['auroc_any']) and align:
                    ali = [BE.sig4(float(h['auroc_any'])), BE.sig4(float(h['f1_any']))]
                rows.append({'neuron': int(j) + 1, 'tau': BE.sig4(float(tau[j])), 'p': float(p[j]), 'round': 1,
                             'threshold': 0.05, 'n_tested': nt, 'rob': [],
                             'st': 'bonf' if p[j] < 0.05 / nt else 'raw', 'nuis': warn, 'align': ali})
            idx = np.where(ok)[0]
            top = idx[np.argsort(p[idx], kind='stable')[:3]]
            results[f'{V0_OID}|{aid}|4608|full'] = {
                'rows': rows, 'top': [[int(j) + 1, BE.sig4(float(tau[j])), float(p[j]), None] for j in top],
                'n_tested_total': nt, 'v0': {'hits': len(sig), 'expected': BE.sig4(0.05 * nt),
                                             'bonf': int((ok & (p < 0.05 / nt)).sum())}}
            assert results[f'{V0_OID}|{aid}|4608|full']['v0']['bonf'] == a['n_found_bonferroni'], (key, aid)
            shown |= {int(j) for j in sig} | {int(j) for j in top}
            m['n_units'] = n_u
            an_out.append(m)
        vals = {str(j + 1): [BE.sig4(float(v)) for v in Y[:, j]] for j in sorted(shown)}
        nb = {}
        for j in sorted(shown):
            n = {'noclip': True, 'artefact': False, 'vals': {}}
            f = (FR or {}).get('neurons', {}).get(str(j + 1))
            if f:
                n['firing_rate'] = f['firing_rate']
                n['frames'] = {'sprite': f'assets/{f["sprite"]}', 'tile': FR['tile'],
                               'top': [[page_ix[o], BE.sig4(v), k] for o, v, k in f['top']],
                               'least': [[page_ix[o], BE.sig4(v), k] for o, v, k in f['least']]}
                shutil.copy(V0_FRAMES / key / f['sprite'], site / 'assets' / f['sprite'])
            nb[str(j + 1)] = n
        rf = f'assets/data/v0_{key}_r.json'
        nf = f'assets/data/v0_{key}_n.json'
        (site / rf).write_text(json.dumps({'results': results, 'vals': {V0_OID: vals}}, separators=(',', ':')))
        (site / nf).write_text(json.dumps({'by': {'v0': nb}}, separators=(',', ':')))
        q = R0['quality']
        e = {'sae': f'v0_topk4608_k20#{key}', 'tag': f'v0{key}', 'domain': dom,
             'model': {'model': 'v0', 'method': 'NES'},
             'outcomes': [{'id': V0_OID, 'label': 'Mean activation (mean token)', 'codes': 'v0', 'unit': 'activation',
                           'bout': False, 'kind': 'time', 'agg': 'mean token', 'prefixes': [4608],
                           'setting': 'v0: per-video mean of the code on the mean-pooled token, single-pass t-test, '
                                      'raw p < 0.05'}],
             'analyses': an_out, 'rfiles': {V0_OID: rf}, 'nfile': {'v0': {'files': [nf], 'of': {j: 0 for j in nb}}},
             'artefact': [], 'artefact_text': {}, 'lwin': {}, 'pre': {}, 'subsets': {}, 'sub_ui': False,
             'subject': ref.get('subject'), 'top': 3, 'n_frames': int(q['n_frames']) if 'n_frames' in q else None,
             'v0': {'fve_pool': BE.sig4(q['fve_meanpooled']), 'alive': q['alive_codes_meanpooled_frames'],
                    'frames': FR is not None, 'align': align},
             'spec': {'cells': {}}}
        if 'subjects' in ref:
            e['subjects'] = ref['subjects']
        for k in ('videos', 'obs', 'arms', 'vnote'):
            if k in ref:
                e[k] = copy.deepcopy(ref[k])
        if dom == 'frogs':
            e['pending'] = {TAD_EXP: 'not run: no tadpole run for this model'}
        if e['n_frames'] is None:  # ants v2: 2 fps frames of its 10 min
            e['n_frames'] = int(sum(R0['frames_per_video'].values()))
        out.append(e)
        log(f'v0 {key}: {len(an_out)} comparisons, ' + ', '.join(
            f'{k.split("|")[1]} {len(r["rows"])}' for k, r in results.items()) + f'; {len(nb)} neurons, frames '
            f'{"yes" if FR else "no"}')
    return out


def source_table():
    """{domain: {observation id: [source file name, source offset, source frames per 5 fps step, source fps]}} from the
    R4 plans (results/vision/eci_rollout_r4/<dom>/plan.json; source frame = offset + step k, k = frame_idx on the 5 fps
    grid; rollout_r4.py DOMS: mice offset 2, frogs 5, ants per video). The page shows a clip's time in its raw file."""
    off0 = {'mice': 2, 'frogs': 5}
    out = {}
    for rd, dom in (('mice', 'mice'), ('ants_v2p', 'ants'), ('ants_v3p', 'ants'), ('ants_v4p', 'ants'), ('frogs', 'frogs')):
        P = json.loads((R4 / rd / 'plan.json').read_text())
        T = out.setdefault(dom, {})
        for v in P['videos']:
            num, den = (int(x) for x in v['probe']['rate'].split('/'))
            fps = num / den
            step = fps / 5
            assert step == int(step), (rd, v['obs'], fps)
            row = [Path(v['path']).name, int(v.get('offset', off0.get(dom, -1))), int(step), fps]
            assert row[1] >= 0, (rd, v['obs'])
            assert T.get(v['obs'], row) == row, (rd, v['obs'], T[v['obs']], row)
            T[v['obs']] = row
    P = json.loads((R4 / 'frogs_tadpole' / 'plan.json').read_text())  # tadpoles: 60 fps raw, source frame 12 k + 5
    for v in P['videos']:
        out['frogs'].setdefault(v['obs'], [Path(v['path']).name, int(v['offset']), 12, 60.0])
    log('source table: ' + ', '.join(f'{d} {len(t)} videos' for d, t in out.items()))
    return out


# ---- behaviour alignment panel + annotation check (scripts/eci/atlas_align.py -> OUT/align/<series>.json)
ALIGN = OUT / 'align'
ALIGN_CUT = 0.2  # a neuron counts as aligned with a behaviour when its cross-fitted R~ >= this (pre-stated)
BEH_LAB = {'Y2F': 'grooming by yellow (Y2F)', 'B2F': 'grooming by blue (B2F)', 'groom_any': 'grooming, either (Y2F or B2F)',
           'YOL': 'yellow on the lid (YOL)', 'BOL': 'blue on the lid (BOL)', 'onlid_any': 'on the lid, either (YOL or BOL)',
           'nn': 'nose-nose (nn)', 'np': 'nose-body (np)', 'nn_or_np': 'nn or np', 'nt': 'nose-tail (nt)'}


def align_table():
    """{series: {behaviours, models: {v0|v1|v2: {beh: {base, n, top: [[neuron, R~cf, AUROCcf, directions]]} | None}}}}"""
    out = {}
    for f in sorted(ALIGN.glob('*.json')):
        d = json.loads(f.read_text())
        out[d['series']] = {
            'n_frames': d['n_frames'], 'n_units': d['n_units'], 'units': d['units'], 'frames': d['frames'],
            'beh': [[b, BEH_LAB[b]] for b in d['behaviours']], 'cut': ALIGN_CUT, 'top_k': d['top_k'],
            'models': {m: {'source': x['source'], 'n_alive': x['n_alive'], 'n_neurons': x['n_neurons'],
                           'beh': {b: (None if v is None else {'base': round(v['base_rate'], 4), 'n': v['n_frames'],
                                                               'cf1': round(v['cf_top1_rprec'], 4), 'top': v['top']})
                                   for b, v in x['beh'].items()}}
                       for m, x in d['models'].items()}}
    log('alignment: ' + ', '.join(out))
    return out


def ann_check(saes, data):
    """{series: {analysis id: {beh: [tau, p, n_control, n_treated, test]}}} of the annotated behaviours (per-video time
    fraction over the labelled frames, scripts/eci/atlas_align.py per_video_time), with the page's contrast: ants
    within-series = treated vs control videos with recording-day fixed effects (OLS, only days holding both arms), ants
    infested vs healthy = Welch (no common day); mice B = Welch het vs wt, A = paired over pools, C = Welch on the
    per-pool changes (annotated pools only)."""
    import statsmodels.formula.api as smf
    from scipy.stats import ttest_1samp, ttest_ind
    out = {}
    for f in sorted(ALIGN.glob('*.json')):
        d = json.loads(f.read_text())
        ser, PV = d['series'], d['per_video_time']
        res = out.setdefault(ser, {})
        if ser == 'mice':
            ent = next(e for e in saes if e['domain'] == 'mice' and e['model'] == {'model': 'online', 'method': 'NES'})
            V, O = data['videos'], data['obs']
        else:
            ent = next(e for e in saes if e['domain'] == 'ants' and e['model'] == {'model': 'new', 'method': 'NES'}
                       and (e['sae'].endswith(f'#ants_{ser[5:]}p_d2') or e['sae'].endswith(f'#v2_ants_{ser[5:]}p_smooth')))
            V, O = ent['videos'], ent['obs']
        obs = [o[0] for o in O]
        for a in ent['analyses']:
            r = res.setdefault(a['id'], {})
            for b in d['behaviours']:
                y = np.array([PV[b].get(o, [np.nan])[0] for o in obs], float)
                if np.isfinite(y).sum() < 6:
                    continue
                with np.errstate(invalid='ignore', divide='ignore'):
                    if ser != 'mice':
                        c = np.array([any(v[0] == e and v[1] == t for e, t in a['arms'][0][1]) for v in V]) & np.isfinite(y)
                        t = np.array([any(v[0] == e and v[1] == t for e, t in a['arms'][1][1]) for v in V]) & np.isfinite(y)
                        if c.sum() < 3 or t.sum() < 3:
                            continue
                        day = np.array([v[2] for v in V])
                        both = set(day[c]) & set(day[t])
                        if a.get('cross') or not both:
                            tt, p = ttest_ind(y[t], y[c], equal_var=False)
                            r[b] = [BE.sig4(float(y[t].mean() - y[c].mean())), float(p), int(c.sum()), int(t.sum()), 'Welch']
                        else:
                            m = (c | t) & np.isin(day, list(both))
                            df = pd.DataFrame({'y': y[m], 'T': t[m].astype(int), 'day': day[m]})
                            fit = smf.ols('y ~ T + C(day)' if len(both) > 1 else 'y ~ T', df).fit()
                            r[b] = [BE.sig4(float(fit.params['T'])), float(fit.pvalues['T']), int((c & m).sum()),
                                    int((t & m).sum()), 'OLS with recording-day fixed effects' if len(both) > 1 else 'OLS']
                    else:
                        pool = np.array([v[0] for v in V])
                        st, gt = np.array([v[1] for v in V]), np.array([v[2] for v in V])
                        if a['family'] == 'B':
                            c = (st == a['stage']) & (gt == 'wt') & np.isfinite(y)
                            t = (st == a['stage']) & (gt == 'het') & np.isfinite(y)
                            if c.sum() < 3 or t.sum() < 3:
                                continue
                            tt, p = ttest_ind(y[t], y[c], equal_var=False)
                            r[b] = [BE.sig4(float(y[t].mean() - y[c].mean())), float(p), int(c.sum()), int(t.sum()), 'Welch']
                        else:
                            s0, s1 = a['stages']
                            ps = sorted({pool[i] for i in range(len(V)) if np.isfinite(y[i]) and
                                         (a['family'] == 'C' or gt[i] == a['genotype'])})
                            D, G_ = [], []
                            for q in ps:
                                i0 = np.flatnonzero((pool == q) & (st == s0))
                                i1 = np.flatnonzero((pool == q) & (st == s1))
                                if len(i0) and len(i1) and np.isfinite(y[i0[0]]) and np.isfinite(y[i1[0]]):
                                    D.append(y[i1[0]] - y[i0[0]])
                                    G_.append(gt[i0[0]])
                            D, G_ = np.array(D), np.array(G_)
                            if a['family'] == 'A':
                                if len(D) < 3:
                                    continue
                                tt, p = ttest_1samp(D, 0.0)
                                r[b] = [BE.sig4(float(D.mean())), float(p), int(len(D)), int(len(D)), 'paired t-test']
                            else:
                                h = G_ == 'het'
                                if h.sum() < 3 or (~h).sum() < 3:
                                    continue
                                tt, p = ttest_ind(D[h], D[~h], equal_var=False)
                                r[b] = [BE.sig4(float(D[h].mean() - D[~h].mean())), float(p), int((~h).sum()), int(h.sum()),
                                        'Welch on the per-pool changes']
            if not r:
                res.pop(a['id'])
    log('annotation check: ' + ', '.join(f'{k} {len(v)} comparisons' for k, v in out.items()))
    return out


TAD_STRIPS = OUT / 'tadpole_strips'


def tadpole_strips(site, saes):
    """Frame strips with real heat maps of the tadpole neurons (scripts/eci/tadpole_atlas_strips.py) into the
    tadpole entries' neuron files."""
    ents = [e for e in saes if e.get('tadpole') and not e.get('tadpole_clips')]
    done = set()
    for e in ents:
        sd = TAD_STRIPS.with_name('tadpole_strips_v2smooth' if e['tag'].startswith('v2tads') else
                                  'tadpole_strips_v2thr' if e['tag'].startswith('v2tadt') else 'tadpole_strips')
        if not (sd / 'strips.json').exists():
            log(f'tadpole strips: none for {e["sae"]} ({sd})')
            continue
        S = json.loads((sd / 'strips.json').read_text())
        ix = {o[0]: i for i, o in enumerate(e['obs'])}
        for nf in e['nfile']['max']['files']:
            if nf in done:
                continue
            done.add(nf)
            D = json.loads((site / nf).read_text())
            if 'by' not in D:
                continue
            for j, n in D['by']['max'].items():
                x = S['neurons'].get(str(j))
                if not x:
                    continue
                shutil.copy(sd / x['sprite'], site / 'assets' / x['sprite'])
                it = lambda L: [[ix[o], v, k] for o, v, k in L]  # noqa: E731
                n['frames'] = {'sprite': f'assets/{x["sprite"]}', 'tile': S['tile'], 'nrow': 3, 'heat_vmax': x['heat_vmax'],
                               'rows': [['Most activated frames', 'highest frame, 1 per video', 0, it(x['top'])],
                                        ['Where it fires (real per-token heat map)', f'full colour = {x["heat_vmax"]}', 1,
                                         it(x['top'])],
                                        ['Least activated frames', '0, random, 1 per video', 2, it(x['least'])]]}
            (site / nf).write_text(json.dumps(D, separators=(',', ':')))
        e['noclip_note'] = ('Tadpoles: frame strips instead of clips, cut from the raw video with the crop boxes used '
                            'for encoding; real per-token heat maps (' + (
                                'the stored token codes of the full pass' if S['check'].get('exact') else
                                f're-encoded, argmax agreement with the stored codes '
                                f'{100 * S["check"]["arg_agree_frac"]:.1f}%') +
                            '); no arena map (the crop follows the tadpole)')
        e['heat_kind'] = 'real'
        log(f'tadpole strips {e["sae"]}: {len(S["neurons"])} neurons, check {json.dumps(S["check"])}')


PICKS = {'r4miced2': OUT / 'res/mice_d2', 'r4v2pd2': OUT / 'res/ants_v2p_d2', 'r4v3pd2': OUT / 'res/ants_v3p_d2',
         'r4v4pd2': OUT / 'res/ants_v4p_d2', 'r4frogsd2': OUT / 'res/frogs_d2',
         'fg448al': ROOT / ONLINE['mice'][2], 'antsfg': ROOT / ONLINE['ants'][2], 'frogsfg': ROOT / ONLINE['frogs'][2],
         **{f'v2{t}{v[0]}': OUT / f'res/{k}_{v}' for k, t in (('v2_mice', 'mice'), ('v2_ants_v2p', 'v2p'),
                                                               ('v2_ants_v3p', 'v3p'), ('v2_ants_v4p', 'v4p'),
                                                               ('v2_frogs', 'frogs'), ('v2_frogs_tad', 'tad')) for v in ('smooth', 'thr', 'prob')}}


def clip_times(site, saes, data):
    """Every clip of the page's neuron files gets its frame_idx (5 fps grid, first frame of the clip) as a third info
    element, [video, mean, k], read from the build's picks.json: the clip at the same position (same source, length,
    row) when its video and mean match, else the unique picks clip of that neuron and length with the same video and
    mean (the online model's published clips vs its later picks cache); null when there is none or several frames
    (the page then says 'time unknown')."""
    stats = {}
    for tag, base in PICKS.items():
        es = [e for e in saes if e['tag'] == tag]
        if not es:
            continue
        pj = [base / '_cache/explorer/picks.json'] if (base / '_cache/explorer/picks.json').exists() else \
            sorted(base.glob('*/_cache/explorer/picks.json'))
        assert len(pj) == 1, (tag, pj)
        P = json.loads(pj[0].read_text())['by_key']['max']
        obs = es[0].get('obs') or data['obs']
        src = data['src'][es[0]['domain']]
        files = sorted({f for e in es for f in e['nfile']['max']['files']})
        st = stats.setdefault(tag, {'direct': 0, 'search': 0, 'exact': 0, 'unknown': 0, 'nosrc': 0})
        # online mice / frogs: the picks cache was re-run after the publish, so the frames come from the codes
        # themselves (scripts/eci/atlas_clip_frames.py: the unique window of that video with that mean, else null)
        xf = OUT / 'clip_frames' / f'{tag}.json'
        XF = json.loads(xf.read_text()) if tag in ('fg448al', 'frogsfg') and xf.exists() else None
        if tag in ('fg448al', 'frogsfg') and XF is None:
            raise SystemExit(f'{xf} missing: run scripts/eci/atlas_clip_frames.sh first')
        for f in files:
            D = json.loads((site / f).read_text())
            N = D['by']['max'] if 'by' in D else D['neurons']
            for j, n in N.items():
                pn = P.get(str(j))
                for sname, Ls in (n.get('clips') or {}).items():
                    for L, c in Ls.items():
                        for kind, lst in c['info'].items():
                            for i, it in enumerate(lst):
                                assert len(it) == 2, (f, j)
                                o, k = obs[it[0]][0], None
                                if o not in src:
                                    st['nosrc'] += 1
                                if XF is not None:
                                    k = XF[f'{j}|{L}|{it[0]}|{it[1]:g}']
                                    st['exact' if k is not None else 'unknown'] += 1
                                elif pn is not None:
                                    d = pn['clips'].get(sname, {}).get(L, {}).get(kind, [])
                                    if i < len(d) and d[i]['obs'] == o and BE.sig4(d[i]['act']) == it[1]:
                                        k = int(d[i]['frame'])
                                        st['direct'] += 1
                                    else:
                                        ks = {int(x['frame']) for s2 in pn['clips'].values() for kk in ('top', 'least')
                                              for x in s2.get(L, {}).get(kk, [])
                                              if x['obs'] == o and BE.sig4(x['act']) == it[1]}
                                        k = ks.pop() if len(ks) == 1 else None
                                        st['search' if k is not None else 'unknown'] += 1
                                else:
                                    st['unknown'] += 1
                                it.append(k)
            (site / f).write_text(json.dumps(D, separators=(',', ':')))
    log('clip times: ' + json.dumps(stats))
    return stats


# the ants new-model 1 s pack r4v4pd2_1s_6dbcf5fe22.mp4 was refused by the artifact upload filter; the live page holds it
# split in three version files (results/vision/eci_atlas_v37/remux/split/: pages 0-10, 11-28, 29-45; the middle one as
# WebM), and its data files point at those (remux/split/patch_json.py; the WebM variant is the live one)
SPLIT_NAME, SPLIT = 'r4v4pd2_1s_6dbcf5fe22', [(0, 0, 10, 'mp4'), (1, 11, 28, 'webm'), (2, 29, 45, 'mp4')]


def split_v4_pack(site):
    pat = re.compile(r'"src":"assets/' + SPLIT_NAME + r'\.mp4","page":(\d+),')

    def rep(m):
        p = int(m.group(1))
        k, a, _, ext = next(c for c in SPLIT if c[1] <= p <= c[2])
        return f'"src":"assets/{SPLIT_NAME}_p{k}.{ext}","page":{p - a},'
    n = 0
    for f in sorted((site / 'assets' / 'data').glob('*.json')):
        t = f.read_text()
        if SPLIT_NAME + '.mp4' not in t:
            continue
        t, m = pat.subn(rep, t)
        assert SPLIT_NAME + '.mp4' not in t, f
        f.write_text(t)
        n += m
    for k, _, _, ext in SPLIT:
        shutil.copy(OUT / 'remux' / 'split' / f'{SPLIT_NAME}_p{k}.{ext}', site / 'assets' / f'{SPLIT_NAME}_p{k}.{ext}')
    log(f'split pack: {n} references -> {SPLIT_NAME}_p0/1/2')


# ---- clip packs as WebM (VP9): the artifact upload filter misdetects some H.264 MP4 packs as "cpio archive"; a VP9
# WebM re-encode passes. Every page starts with a keyframe (g = page length in frames: 'frame' packs all-intra, '1s'
# packs every 5th frame), so the page's seek to a page start decodes from that page alone. Cache: OUT/webm/.
WEBM = OUT / 'webm'
WEBM_CRF = 40  # VP9 at crf 30 was 1.3-1.9x the H.264 packs (largest 19.9 MB > the 15 MB file limit)


def pack_w(name):
    return 1 if '_frame_' in name else 5 if '_1s_' in name else None


def to_webm(src):
    """<pack>.mp4 -> OUT/webm/<pack>.webm (cached by source size + mtime)."""
    w = pack_w(src.name)
    assert w, src.name
    out = WEBM / (src.stem + '.webm')
    tag = WEBM / (src.stem + '.src')
    key = f'{src.stat().st_size}|{int(src.stat().st_mtime)}|crf{WEBM_CRF}|g{w}'
    if out.exists() and tag.exists() and tag.read_text() == key:
        return out
    tmp = out.with_suffix('.tmp.webm')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src), '-an', '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuv420p',
           '-crf', str(WEBM_CRF), '-b:v', '0', '-g', str(w), '-keyint_min', str(w),
           '-force_key_frames', f'expr:eq(mod(n,{w}),0)', '-auto-alt-ref', '0', '-lag-in-frames', '0',
           '-deadline', 'good', '-cpu-used', '2', '-row-mt', '1', '-threads', '4', str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.rename(out)
    tag.write_text(key)
    return out


def frames_at(path, idx, w, h):
    """rgb frames (n, h, w, 3) uint8 of the given frame indices of a video (one sequential decode)."""
    sel = '+'.join(f'eq(n\\,{i})' for i in idx)
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', f"select='{sel}'", '-vsync', '0',
                        '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, h, w, 3)


def webm_check(pairs):
    """Pixel MAE of 3 sample pages per pack type between the MP4 render and the WebM, at the page-start frames, and
    the WebM's keyframe positions (every page start must be one)."""
    rep_ = {}
    by = {}
    for m, wv in pairs:
        by.setdefault(pack_w(m.name), []).append((m, wv))
    for w, L in by.items():
        out = []
        for m, wv in L[:3]:
            pr = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                                            '-show_entries', 'stream=width,height,nb_read_frames', '-of', 'json',
                                            str(m)], capture_output=True, check=True).stdout)['streams'][0]
            W, H, n = int(pr['width']), int(pr['height']), int(pr['nb_read_frames'])
            pages = n // w
            idx = sorted({0, (pages // 2) * w, (pages - 1) * w})
            a, b = frames_at(m, idx, W, H), frames_at(wv, idx, W, H)
            assert a.shape == b.shape, (m, a.shape, b.shape)
            kf = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                 'frame=key_frame', '-of', 'csv=p=0', str(wv)], capture_output=True, check=True
                                ).stdout.decode().split()
            kfi = [i for i, k in enumerate(kf) if k.strip().startswith('1')]
            starts = list(range(0, n, w))
            out.append({'pack': m.name, 'frames': n, 'pages': pages, 'checked_frames': idx,
                        'mae': [round(float(np.abs(a[i].astype(np.int16) - b[i].astype(np.int16)).mean()), 3)
                                for i in range(len(idx))],
                        'keyframe_every_page_start': set(starts) <= set(kfi), 'n_keyframes': len(kfi),
                        'mp4_mb': round(m.stat().st_size / 1e6, 2), 'webm_mb': round(wv.stat().st_size / 1e6, 2)})
        rep_[{1: 'frame', 5: '1s'}[w]] = out
    (WEBM / 'check.json').write_text(json.dumps(rep_, indent=1))
    log('webm check: ' + json.dumps({k: [(x['pack'], x['mae'], x['keyframe_every_page_start']) for x in v]
                                     for k, v in rep_.items()}))
    bad = [x for v in rep_.values() for x in v if not x['keyframe_every_page_start']]
    if bad:
        raise SystemExit(f'webm: page starts without a keyframe in {[x["pack"] for x in bad]}')


# ---- one model "v2" with an Outcome selector (user, 2026-10-21): readout = continuous (the smoothed codes, <sae>_smooth,
# default) | binary (the fnorm-thresholded smoothed codes, <sae>) | prob (per-neuron P(on), 'coming' until the v2
# manifest lists it). v0 and v1 have continuous codes only.
READOUT_TIPS = {'continuous': 'Continuous: the neuron\'s smoothed code (0.6 s mean of the frame max over the animal mask), '
                              'no firing threshold. The default (user decision 2026-10-21)',
                'binary': 'Binary: the same smoothed code kept only above the neuron\'s firing threshold (fnorm), else 0 '
                          '(v2 only)',
                'prob': 'Probability: per-neuron P(on) per frame (v2 only). Average time = the expected fraction of time '
                        'active; event rate and latency threshold P(on) at its pooled 0.95 quantile'}


V2_CHECKS = {'window_hwhole': 'with habituation taken whole (30 min) instead of its last 15 min',
             'window_trim30': 'with the first 30 s of every video dropped',
             'test_signflip': 'with a sign-flip permutation test instead of the t-test',
             'test_rank': 'with a rank-based t-test',
             'input_prob': 'with the outcome input P(on) instead of the continuous code',
             'input_binary': 'with the thresholded (binary) input instead of the continuous code',
             'q0.90': 'with the bout threshold at quantile 0.90 instead of 0.95',
             'q0.99': 'with the bout threshold at quantile 0.99 instead of 0.95',
             'gap2': 'with bouts merged across gaps of 2 frames'}


def _nfiles(e):
    nf = e['nfile'].get('max') or next(iter(e['nfile'].values()))
    return nf['files']


def _neurons(D):
    return D['neurons'] if 'neurons' in D else D.get('by', {}).get('max', {})


def share_v2_clips(site, saes):
    """Version budget (256 MB): the v2 binary / prob outcomes show the clips of the continuous outcome (same SAE, same
    neuron) instead of their own packs; a neuron with no continuous clips is listed without clips (its arena map,
    histogram and per-video values are its own)."""
    cont = {}
    for e in saes:
        if e['model']['model'] == 'new' and e['model'].get('readout') == 'continuous' and '#' in e['sae']:
            k = e['sae'].split('#')[1].rsplit('_', 1)[0]
            for f in _nfiles(e):
                if (site / f).exists():
                    cont.setdefault(k, {}).update(_neurons(json.loads((site / f).read_text())))
    done, n_sh, n_no = set(), 0, 0
    for e in saes:
        if e['model']['model'] != 'new' or e['model'].get('readout') == 'continuous' or '#' not in e['sae']:
            continue
        k = e['sae'].split('#')[1].rsplit('_', 1)[0]
        C = cont.get(k, {})
        for f in _nfiles(e):
            if f in done or not (site / f).exists():
                continue
            done.add(f)
            D = json.loads((site / f).read_text())
            for j, n in _neurons(D).items():
                if not n.get('clips'):
                    continue
                cn = C.get(str(j))
                if cn and cn.get('clips'):
                    n['clips'] = cn['clips']
                    for x in ('heat_vmax', 'bout_thr_bar'):
                        if x in cn:
                            n[x] = cn[x]
                        else:
                            n.pop(x, None)
                    n['clips_from'] = 'continuous'
                    n_sh += 1
                else:
                    n.pop('clips', None)
                    n['noclip'] = True
                    n_no += 1
            (site / f).write_text(json.dumps(D, separators=(',', ':')))
        e['noclip_note'] = ('No clips for this neuron with this outcome: clips are rendered for the continuous outcome '
                            '(the version size budget); this neuron was not listed there')
    log(f'v2 clips shared: {n_sh} neurons of the binary / prob outcomes show the continuous clips, {n_no} without clips')


def v2_sensitivity(site, saes):
    """'stable in k of n checks' of the frozen v2 from its fixed sensitivity grid (results/vision/eci_v2/
    sensitivity.json; continuous input, NES only): every v2 row gets exactly the checks run for it (the grid's
    {aggregation|prefix: {neuron: {k, n, checks}}}); a v2 row with no grid entry (NESS, binary / prob input, a
    selection outside the index) gets none, so the page shows no chip for it."""
    f = V2 / 'sensitivity.json'
    if not f.exists():
        log('v2 sensitivity: none')
        return
    IX = json.loads(f.read_text())['index']
    agg = {'rate': 'bouts', 'time': 'mean', 'latency': 'latency'}
    n_set = n_rows = 0
    for e in saes:
        if e['model']['model'] != 'new':
            continue
        key = e['sae'].split('#')[1] if '#' in e['sae'] else ''
        prim = e['model'].get('readout') == 'continuous' and e['model']['method'] == 'NES'
        rs = ('mice' if key.startswith('v2_mice') else 'frogs_tad' if key.startswith('v2_frogs_tad') else
              'frogs_juv' if key.startswith('v2_frogs') else f'ants_{key.split("_")[2][:2]}' if key.startswith('v2_ants') else None)
        kinds = {o['id']: o['kind'] for o in e['outcomes']}
        for oid, rel in e['rfiles'].items():
            p_ = site / rel
            if not p_.exists():
                continue
            D = json.loads(p_.read_text())

            def fix(R, sub):
                nonlocal n_set, n_rows
                for k, v in R.items():
                    o_, aid, px = k.split('|')[:3]
                    if o_ not in kinds:  # a bundle shared with the other method's entry: its own rows only
                        continue
                    a0 = aid[4:] if aid.startswith('tad_') else aid
                    r_ = rs if not sub else f'{rs}/subsets/{sub}'
                    if aid.startswith('S_'):
                        r_ = f'{rs}/family_S'
                    if rs == 'ants_v3' and a0 == 'v3_2_vs_8':
                        r_ = 'ants_v3/v3_2_vs_8_noday'
                    cell = IX.get(r_, {}).get(a0, {}).get(f'{agg[kinds.get(o_, "time")]}|p{px}', {}) if prim else {}
                    for r in v.get('rows', []):
                        n_rows += 1
                        x = cell.get(str(r['neuron']))
                        r['rob'] = [[ck, V2_CHECKS.get(ck, ck), 'Y' if ok else 'N', ''] for ck, ok in x['checks'].items()] if x else []
                        n_set += bool(x)
            fix(D.get('results', {}), None)
            for sub, R in (D.get('subsets') or {}).items():
                fix(R, sub)
            p_.write_text(json.dumps(D, separators=(',', ':')))
    log(f'v2 sensitivity: {n_set} of {n_rows} v2 rows carry grid checks (the others: no chip)')


def readout_axis(saes, data):
    man = V2 / 'manifest.json'
    has_prob = man.exists() and any('prob' in S.get('sets', {}) for D in json.loads(man.read_text())['domains'].values()
                                    for S in D.get('series', {}).values())
    for e in saes:
        m = e['model']
        binary = m['model'] == 'new (thr)' or e['sae'].endswith('_thr')
        prob = '_prob@' in e['sae'] or e['sae'].endswith('_prob')
        m['readout'] = 'binary' if binary else 'prob' if prob else 'continuous'
        if m['model'] == 'new (thr)':
            m['model'] = 'new'
    ax0 = data['axes'][0]
    ax0['values'] = [v for v in ax0['values'] if v != 'new (thr)']
    data['axes'].insert(1, {'key': 'readout', 'label': 'Outcome', 'tips': READOUT_TIPS,
                            'values': ['continuous', 'binary', 'prob'],
                            'missing': {'prob': 'listed in the v2 manifest, page build pending' if has_prob else
                                        'coming'}})
    log(f'readout axis: prob in manifest {has_prob}; ' + ', '.join(
        f'{r} {sum(e["model"]["readout"] == r for e in saes)}' for r in ('continuous', 'binary', 'prob')))


def step_assemble():
    site = OUT / 'site'
    if site.exists():
        shutil.rmtree(site)
    (site / 'assets' / 'data').mkdir(parents=True)
    amap = {}
    for f in A.asset_map:
        amap.update(json.loads(Path(f).read_text()))
    v36 = v36_all()
    ddir = lambda e: PUB_DIR.get(e['tag'], V36) / 'assets' / 'data'  # noqa: E731
    saes, notes, copied = [], {}, set()

    def put36(rel, e):  # a published data file, as published
        if rel.startswith('pre:'):
            return rel
        name = Path(rel).name
        if name not in copied:
            shutil.copy(ddir(e) / name, site / 'assets' / 'data' / name)
            copied.add(name)
        return rel

    def write(name, obj):
        (site / 'assets' / 'data' / name).write_text(json.dumps(obj, separators=(',', ':')))
        return f'assets/data/{name}'

    def load36(rel, e):
        return e['pre'][rel] if rel.startswith('pre:') else json.loads((ddir(e) / Path(rel).name).read_text())

    # ---- new model entries (one build per R4 domain; NES + DINES entries share the build's data files)
    new_entries, builds = {}, {}
    keys_all = [f'{k}_{v}' for k in NEW for v in ('d2', 'd0', 'smooth', 'thr', 'prob')]
    WEBM.mkdir(parents=True, exist_ok=True)
    webm_pairs = []
    v2_built = any(k.startswith('v2_') and (OUT / 'build' / k / 'core.json').exists() for k in keys_all)
    for key in keys_all:
        b = OUT / 'build' / key
        if not (b / 'core.json').exists():
            continue
        if v2_built and not key.startswith('v2_'):  # the frozen v2 replaces the R4 builds
            log(f'assemble: {key} left out (the frozen v2 replaces the R4 builds)')
            continue
        cj = json.loads((b / 'core.json').read_text())
        # clip packs of this build as WebM (to_webm); references renamed in its data files
        mp4s = sorted({m for f in (b / 'data').glob('*.json') for m in re.findall(r'assets/([\w.\-]+\.mp4)', f.read_text())
                       if m not in amap})
        if key.startswith('v2_') and not key.endswith('_smooth'):
            mp4s = []  # binary / prob outcomes: clips shared from the continuous build (share_v2_clips)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(6) as ex:
            wm = dict(zip(mp4s, ex.map(lambda m: to_webm(b / 'assets' / m), mp4s)))
        webm_pairs.extend((b / 'assets' / m, wv) for m, wv in wm.items())
        for f in sorted((b / 'data').glob('*.json')):
            rw_json(f, site / 'assets' / 'data' / f.name, amap)
            t = (site / 'assets' / 'data' / f.name).read_text()
            for m in wm:
                t = t.replace(f'assets/{m}', f'assets/{Path(m).stem}.webm')
            (site / 'assets' / 'data' / f.name).write_text(t)
        for m, wv in wm.items():
            shutil.copy(wv, site / 'assets' / wv.name)
        for f in sorted((b / 'assets').glob('*.webp')):
            shutil.copy(f, site / 'assets' / f.name)
        ents = []
        for e in cj['entries']:
            if amap:
                e = json.loads(rw_text(json.dumps(e), amap))
            if e.get('domain') == 'tadpoles':  # tadpole clip builds (dumped before the domain fix)
                e['domain'] = 'frogs'
            if e.get('heat_kind') not in ('real', 'none'):
                raise SystemExit(f'assemble: {key} has no real heat maps (heat_kind {e.get("heat_kind")})')
            dom = e['domain']
            k0 = key.rsplit('_', 1)[0]
            series = {'ants_v2': 'v2', 'ants_v3': 'v3', 'ants_v4': 'v4'}.get(k0)
            e['pooled'] = k0.endswith('p') and k0.startswith('ants')
            e['sae'] = f'{e["sae"]}#{key}'  # unique per build (the pooled ants series share one SAE name)
            e['note'] = ('v2. ' if key.startswith('v2_') else 'New model (R4, ' + ('2 s change, d2' if key.endswith('d2') else 'no change half, d0') +
                         '). ' + (f'Ants: one SAE per series; this is the {series} SAE, so its neuron numbers are '
                                  f'its own. ' if series else '') +
                         ('Ants: one SAE for all series (v2, v3, v4), so a neuron number means the same neuron in '
                          'every series. ' if e['pooled'] else '') +
                         ('Mice: habituation on its last 15 min. ' if dom == 'mice' else '') +
                         ('Camera-period checks not run for this model. ' if dom == 'mice' and not e.get('period')
                          else '') +
                         ('Recording-session checks not run for this model. ' if dom == 'frogs' and not e.get('session')
                          else ''))
            ents.append(e)
        builds[key] = cj
        new_entries[key] = ents
    # the new model on the page (coordinator, 2026-10-16 07:55): ants = the pooled SAE only (the per-series SAEs never
    # appear), variant d2 only (d0 was built for frogs and the per-series ants only: left out rather than partial)
    v2_on = any(k.startswith('v2_') for k in new_entries)
    for k in list(new_entries):
        if v2_on and not k.startswith('v2_'):  # the frozen v2 replaces the R4 builds (coordinator, 2026-10-21)
            log(f'assemble: {k} left out (the frozen v2 replaces the R4 builds)')
            new_entries.pop(k)
        elif k.endswith('_d0') or (k.startswith('ants_') and not k.endswith('p_d2')):
            log(f'assemble: {k} left out (new model on the page = d2, pooled ants SAE)')
            new_entries.pop(k)
    have_v4 = any(k.startswith('ants_v4') for k in new_entries)
    order_new = [k for k in ('mice_d2', 'ants_v2p_d2', 'ants_v3p_d2', 'ants_v4p_d2', 'ants_v2_d2', 'ants_v3_d2',
                             'ants_v4_d2', 'frogs_d2', 'frogs_tad_d2', 'mice_d0', 'ants_v2p_d0', 'ants_v3p_d0', 'ants_v4p_d0',
                             'ants_v2_d0', 'ants_v3_d0', 'ants_v4_d0', 'frogs_d0') if k in new_entries] + \
        [f'{k}_{v}' for v in ('smooth', 'thr', 'prob') for k in ('v2_mice', 'v2_ants_v2p', 'v2_ants_v3p', 'v2_ants_v4p', 'v2_frogs',
                                                         'v2_frogs_tad') if f'{k}_{v}' in new_entries]
    for k in order_new:
        for e in new_entries[k]:
            if e['domain'] == 'ants' and not have_v4:
                e['pending'] = {'v4': 'pending: the new model\'s v4 run (Exp4 colonies, t=2 vs 6 within colony, '
                                      'infested vs healthy) has not finished'}
            saes.append(e)

    # ---- online model entries (v36 data as published) + DINES / family C searches
    for e in v36['saes']:
        if e['tag'] not in ('fg448al', 'antsfg', 'frogsfg'):
            continue
        dkey = {'fg448al': 'mice', 'antsfg': 'ants', 'frogsfg': 'frogs'}[e['tag']]
        bg = PUB_DIR.get(e['tag'], V36) / 'assets' / f'{e["tag"]}_arena_bg.webp'
        if bg.exists():
            shutil.copy(bg, site / 'assets' / bg.name)
        on = OUT / 'build' / f'online_{dkey}' / 'online.json'
        on = json.loads(on.read_text())['methods'] if on.exists() else {}
        base = copy.deepcopy(e)
        base['outcomes'] = [o for o in e['outcomes'] if o['id'] in KEEP_IDS]
        base['rfiles'] = {k: v for k, v in e['rfiles'].items() if k in KEEP_IDS}
        base['nfile'] = {'max': copy.deepcopy(e['nfile']['max'])}
        for f in base['nfile']['max']['files']:
            put36(f, e)
        base['pre'] = {}
        for k in ('lwin', 'period', 'session'):
            if k in base:
                base[k] = {i: v for i, v in base[k].items() if i in KEEP_IDS}
        base['model'] = {'model': 'online', 'method': 'NES'}
        base['note'] = ('Online model: the published page (version 39 of 2026-10-16, max pooling data unchanged since '
                        'version 37; mice and frogs unchanged since v36 of 2026-10-04), data reused as published'
                        + ('; habituation analysed on its whole 30 min (before the 2026-10-07 choice of its last '
                           '15 min). Family C is new: computed on the same per-video values.' if dkey == 'mice' else
                           '; ants: the 30-min results published in version 37 (v2: its whole 10 min).'
                           if dkey == 'ants' else '.'))

        def add_method(ent, M, tagx, replace):
            """Searches of one new method (online.json) into entry ent: its rfiles become new files (the v36 searches
            plus these when not replace), its analyses get the new ones, neurons without a v36 entry get a no-clip
            entry (arena, per-video values) in an extra neuron file."""
            for o in ent['outcomes']:
                oid = o['id']
                R = {k: v for k, v in M['results'].items() if k.startswith(oid + '|')}
                if replace and dkey == 'mice':  # DINES
                    R = block_dines_A(R, [oid], [a['id'] for a in ent['analyses']])
                old = load36(e['rfiles'][oid], e) if not replace else {'results': {}, 'subsets': {}}
                F = {'results': {**{k: v for k, v in old['results'].items() if k.startswith(oid + '|')}, **R},
                     'subsets': {n: {k: v for k, v in S.items() if k.startswith(oid + '|')}
                                 for n, S in old.get('subsets', {}).items()}}
                V = dict((old.get('vals') or {}).get(oid, {}))
                V.update(M['vals'].get(oid, {}))
                if V:
                    F['vals'] = {oid: V}
                ent['rfiles'][oid] = write(f'{e["tag"]}{tagx}_r_{oid}.json', F)
            for x in M.get('analyses_meta', []):
                if not any(a['id'] == x['id'] for a in ent['analyses']):
                    ent['analyses'].append(x)
            if M['extra']:
                nf = ent['nfile']['max']
                f = write(f'{e["tag"]}{tagx}_xn.json', {'by': {'max': M['extra']}})
                nf['files'].append(f)
                for j in M['extra']:
                    nf['of'][j] = len(nf['files']) - 1
            ent['lwin'] = {**ent.get('lwin', {}), **M.get('lwin', {})}

        nes = copy.deepcopy(base)
        for k, v in base['rfiles'].items():
            put36(v, e) if not v.startswith('pre:') else None
        if 'nesC' in on:
            add_method(nes, on['nesC'], 'v37', replace=False)
        else:
            for oid, rel in base['rfiles'].items():  # materialise an inlined v36 search file
                if rel.startswith('pre:'):
                    nes['rfiles'][oid] = write(rel[4:], e['pre'][rel])
        entries = [nes]
        if 'dines' in on:
            di = copy.deepcopy(base)
            di['sae'] = e['sae'] + '@dines'
            di['model'] = {'model': 'online', 'method': 'DINES'}
            for k in ('period', 'session'):
                di.pop(k, None)
            add_method(di, on['dines'], 'dines', replace=True)
            entries.append(di)
        saes += entries
    ov4 = OUT / 'build' / 'online_ants_v4' / 'core.json'
    if ov4.exists():  # online model on ants v4 (step_dump_online_v4)
        for f in sorted((ov4.parent / 'data').glob('*.json')):
            shutil.copy(f, site / 'assets' / 'data' / f.name)
        v4e = json.loads(ov4.read_text())['entries']
        for x in saes:
            if x['domain'] == 'ants' and x['model']['model'] == 'online' and any(y['model'] == x['model'] for y in v4e):
                x.pop('pending', None)
        saes += v4e

    tad_models = {(e['model']['model'], e['model']['method']) for e in saes if e.get('tadpole')}
    for e in saes:  # frogs: the tadpole phase (stage 44-48) exists for the new model only (step_dump_tadpoles)
        if e['domain'] == 'frogs' and (e['model']['model'], e['model']['method']) not in tad_models:
            e.setdefault('pending', {})[TAD_EXP] = 'not run: no tadpole run for this model'
    # ---- top level
    mice_new = next((e for e in saes if e['domain'] == 'mice' and e['model']['model'] == 'new'), None)
    if mice_new is not None:  # the shared mice video list
        vb = builds[mice_new['build']]['videos_mice']
        assert vb == v36['videos'], 'mice video list differs between the new build and v36'
    doms = []
    for d, title in (('mice', 'Mice'), ('ants', 'Ants'), ('frogs', 'Frogs')):
        es = [x for x in saes if x['domain'] == d]
        if es:
            # default = v1 (the deployed online model, NES); v2 (new) is still being iterated (user, 2026-10-16)
            doms.append({'id': d, 'title': title, 'default': next((x['sae'] for x in es if x['model'] ==
                                                                   {'model': 'online', 'method': 'NES'}), es[0]['sae'])})
    saes.sort(key=lambda x: ({'mice': 0, 'ants': 1, 'frogs': 2}[x['domain']], x['model']['model'] != 'new',
                             x['model']['method'] != 'NES'))
    data = {'default': doms[0]['default'], 'saes': saes,
            'axes': [{'key': 'model', 'label': 'Model', 'tips': MODEL_TIPS, 'values': ['new', 'online']},
                     {'key': 'method', 'label': 'Method', 'tips': METHOD_TIPS, 'values': ['NES', 'DINES', 'NESS'],
                      'missing': {'DINES': 'coming', 'NESS': 'to be implemented'}}],
            'videos': v36['videos'], 'obs': v36['obs'], 'montage': v36['montage'], 'tcrit': v36['tcrit'],
            'domains': doms, 'kinds': v36['kinds'],
            # every aggregation of the live page is listed (live version 39), only max pooling is on
            'aggs': [g if g['id'] == 'max' else {**g, 'off': g.get('off', 'not available in this version')}
                     for g in v36['aggs']],
            'method_notes': METHOD_NOTES}
    for e in saes:  # the page names the method NESS (the user's name for DINES, src/eci/dines.py)
        if e['model']['method'] == 'DINES':
            e['model']['method'] = 'NESS'
            e['sae'] = e['sae'].replace('@dines', '@ness')
    for f in (site / 'assets' / 'data').glob('*.json'):  # the blocked family-A text of the method's searches
        t = f.read_text()
        if 'Not shown for DINES' in t:
            f.write_text(t.replace('Not shown for DINES', 'Not shown for NESS').replace('so DINES cannot', 'so NESS cannot')
                         .replace('Use NES for family A.', 'Use NES for phase transitions.'))
    data['axes'][1] = {'key': 'method', 'label': 'Method', 'values': ['NES', 'NESS'],
                       'missing': {'NESS': 'coming'},
                       'tips': {'NES': METHOD_TIPS['NES'], 'NESS': METHOD_TIPS['DINES'].replace(
                           'Direct-Indirect Neural Effect Search (successor of NES', 'NESS (successor of NES; '
                           'implemented as DINES')}}
    data['method_notes'] = {'NESS': METHOD_NOTES['DINES'].replace('DINES', 'NESS')}
    # local clip packs of the online entries (the live page's ants packs that are version files, not asset-store
    # uploads): copied as published
    for e in saes:
        if e['model']['model'] != 'online':
            continue
        txt = json.dumps(e) + ''.join((site / f).read_text() for f in set(e['rfiles'].values()) |
                                      set(e['nfile']['max']['files']) if not f.startswith('pre:') and (site / f).exists())
        for name in set(re.findall(r'assets/([\w.\-]+\.mp4)', txt)):
            src = PUB_DIR.get(e['tag'], V36) / 'assets' / name
            if not (site / 'assets' / name).exists():
                shutil.copy(src, site / 'assets' / name)
                log(f'assemble: {name} copied from {src.parent} (a version file of the live page)')
    # ---- page cleanup (user, 2026-10-16): no model / method prose (the Details panel holds the design), the pooled
    # v3 t=2 vs t=6+8 is not shown, clip times in the raw source video, the split v4 pack
    for e in saes:
        e.pop('note', None)
        e['analyses'] = [a for a in e['analyses'] if a['id'] != 'v3_2_vs_6_8']
        e['spec'] = spec_of(e)
    data.pop('method_notes', None)
    data['src'] = source_table()
    data['v0'] = v0_table()
    v0e = v0_entries(site, saes, data)  # v0 = the baseline (original ECI paper recipe), a selectable model
    saes += v0e
    webm_check(webm_pairs)
    data['align'] = align_table()
    data['anncheck'] = ann_check(saes, data)
    tadpole_strips(site, saes)
    data['axes'][0]['values'] = data['axes'][0]['values'] + (['v0'] if v0e else [])
    readout_axis(saes, data)
    share_v2_clips(site, saes)
    v2_sensitivity(site, saes)
    clip_times(site, saes, data)
    split_v4_pack(site)
    tpl = TEMPLATE.read_text()
    assert tpl.count('/*__DATA__*/null') == 1
    page = tpl.replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
    (site / 'index.html').write_text(page)
    check_site(site, amap, saes)


def rw_text(t, amap):
    for name, url in amap.items():
        t = t.replace(f'assets/{name}', url)
    return t


def check_site(site, amap, saes):
    """Every local reference exists; sizes and counts vs the artifact limits; manifest.json + publish.json."""
    page = (site / 'index.html').read_text()
    drefs = set(re.findall(r'assets/data/[\w.\-]+\.json', page))
    texts = [page] + [(site / r).read_text() for r in sorted(drefs) if (site / r).exists()]
    refs = {r for t in texts for r in re.findall(r'assets/[\w.\-]+\.(?:mp4|webm|webp|png|jpg)', t)}
    urls = {u for t in texts for u in re.findall(r'/_blob/[0-9a-f]+', t)}
    missing = sorted(r for r in drefs | refs if not (site / r).exists())
    files = [p for p in (site / 'assets').rglob('*') if p.is_file()]
    unused = [p for p in files if str(p.relative_to(site)) not in drefs | refs]
    for p in unused:
        p.unlink()
    files = [p for p in files if p not in unused]
    packs = [p for p in files if p.suffix in ('.mp4', '.webm')]
    big = max((p.stat().st_size for p in files), default=0)
    total = sum(p.stat().st_size for p in files) + len(page.encode())
    log(f'check: page {len(page.encode()) / 1e3:.0f} KB, {len(drefs)} data files, {len(refs)} local media refs '
        f'({len(packs)} local packs, {sum(p.stat().st_size for p in packs) / 1e6:.1f} MB), {len(urls)} distinct '
        f'asset-store URLs, {len(missing)} missing {missing[:5]}, {len(unused)} unused removed; largest file '
        f'{big / 1e6:.1f} MB; total version {total / 1e6:.1f} MB in {len(files) + 1} files')
    man = {'artifact': 'https://claude.ai/artifact/JfJnP4CqjYWoprEq9dM8Qa', 'artifact_version': 'build only: the published version id is set in atlas/manifest.json after publishing',
           'build_command': 'ARGS="--steps assemble [--asset-map <map.json>]" sbatch scripts/eci/build_atlas_v37.sh',
           'build_dir': str(site.relative_to(ROOT)),
           'models': [f'{e["domain"]}: {e["model"]["model"]} · {e["model"]["method"]} ({e["sae"]})' for e in saes],
           'options': {'prefixes': [128, 256], 'default_prefix': 128, 'spatial_aggregation': ['max'],
                       'clip_lengths': ['frame', '1s'], 'methods': ['NES', 'NESS'], 'models': ['new', 'online', 'v0']},
           'version_files': {str(p.relative_to(site)): p.stat().st_size for p in sorted(files) if p.suffix not in ('.mp4', '.webm')},
           'packs_to_upload': {p.name: p.stat().st_size for p in sorted(packs) if p.name.startswith(('r4', 'v2'))},
           'live_version_file_packs': {p.name: p.stat().st_size for p in sorted(packs) if not p.name.startswith('r4')},
           'asset_store_urls_referenced': len(urls)}
    man['version_files']['index.html'] = len(page.encode())
    (site / 'manifest.json').write_text(json.dumps(man, indent=1))
    if missing or big > 15e6 or len(page.encode()) > 16e6:
        raise SystemExit('check failed')


if __name__ == '__main__':
    steps = A.steps.split(',')
    if steps == ['assemble']:
        step_assemble()
        sys.exit(0)
    if A.model == 'new' and A.dom in ('frogs_tad', 'v2_frogs_tad') and steps == ['tad']:
        step_dump_tadpoles()
    elif A.model == 'new' and A.dom == 'v2_frogs_tad':  # tadpole clips: the standard steps on the tadpole domain
        tad_setup(clips=True)
        step_prep_new() if 'prep' in steps else None
        c = new_cfg()
        for s in [x for x in steps if x != 'prep']:
            {'data': lambda: BE.step_data(c, A.overwrite), 'v2heat': lambda: step_v2heat(c),
             'arena': lambda: step_arena_new(c), 'render': lambda: step_render_new(c),
             'dump': lambda: dump(c, lambda core: split_methods(core, new_paths()[-1], V2_MODEL[A.variant]), c.out)}[s]()
    elif A.model == 'new':
        if 'prep' in steps:
            step_prep_new()
        if 'checks' in steps:
            step_checks_new()
        rest = [s for s in steps if s not in ('prep', 'checks')]
        if rest:
            c = new_cfg()
            for s in rest:
                if s in ('render', 'dump') and c.dom is BE.MICE:
                    use_rotated(c)
                if s == 'data':
                    BE.step_data(c, A.overwrite)
                elif s == 'rotheat':
                    step_rotheat(c)
                elif s == 'v2heat':
                    step_v2heat(c)
                elif s == 'patch':
                    step_patch_new(c)
                elif s == 'arena':
                    step_arena_new(c)
                elif s == 'render':
                    step_render_new(c)
                elif s == 'dump':
                    key = new_paths()[-1]
                    mv = V2_MODEL[A.variant] if A.dom.startswith('v2_') else \
                        'new' if A.variant == 'd2' else 'new (no Δ)'
                    dump(c, lambda core: split_methods(core, key, mv), c.out)
                else:
                    raise SystemExit(f'unknown step {s}')
    else:
        if 'prep' in steps:
            step_prep_online()
        if 'dump' in steps:
            step_dump_online_v4() if A.dom == 'ants_v4' else step_dump_online()
