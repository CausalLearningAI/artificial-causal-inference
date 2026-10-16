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
       'ants_v4p': ('ants', 'eci_r4v4p', 'results/vision/ants/eci_r4v4p/nes', 'ants_pooled')}
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
              'frogs': 'frogs', 'online_mice': 'mice', 'online_ants': 'ants_eci30', 'online_frogs': 'frogs'}
# the three temporal aggregations: (kind, outcome sub dir, NES outcome id as build_explorer names it, label, unit, short)
OUTS = [('rate', 'maxpool_bouts', 'boutratemaxpool', 'Bout rate (max-pool)', 'bouts/min', 'bouts'),
        ('time', '.', 'meanactivationma', 'Mean activation (max-pool)', 'activation', 'max-pool'),
        ('latency', 'maxpool_latency', 'xlatencymax', 'Latency to first bout (max-pool)', 's', 'latency')]
TRANS = {'1to2': (1, 2), '2to3': (2, 3), '4to5': (4, 5), '5to6': (5, 6)}

ap = argparse.ArgumentParser()
ap.add_argument('--dom', default=None, help='build key: ' + ', '.join(NEW) + ' (new) / mice, ants, frogs (online)')
ap.add_argument('--model', default='new', choices=('new', 'online'))
ap.add_argument('--variant', default='d2', choices=('d2', 'd0'))
ap.add_argument('--steps', default='prep,data,patch,arena,render,dump')
ap.add_argument('--max-pages', type=int, default=0, help='clip pages per build (build_explorer --max-pages)')
ap.add_argument('--kr', type=int, default=None, help='clips rendered per row (default: build_explorer VIEW kr; mice 8)')
ap.add_argument('--pack-mb', type=float, default=12.0)
ap.add_argument('--asset-map', action='append', default=[], help='assemble: JSON {pack name: asset URL}, repeatable')
ap.add_argument('--overwrite', action='store_true')
A = ap.parse_args()

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


Dm.MiceDomain.analyses = property(_mice_analyses)

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
    worst = _tc(vm, vals, {k: R for k, R in results.items() if '|C_' not in k}, oid, round_, dom)
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
EXTRA_SEARCHES = {'ants_v3p_d2': ('v3_2_vs_8_noday', OUT / 'methods_v3_2v8', ('v3_2_vs_8',))}


def new_paths():
    dom_name, lay, nes_root, rdom = NEW[A.dom]
    sae = best_sae(rdom, A.variant) + '_fnorm'
    key = f'{A.dom}_{A.variant}'
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
    xd = [(xs[1] / METHOD_KEY[A.dom] / sae / 'dines', xs[2])] if xs else []
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
    tag = 'r4' + {'mice': 'mice', 'ants_v2': 'v2', 'ants_v3': 'v3', 'ants_v4': 'v4', 'frogs': 'frogs', 'ants_v2p': 'v2p',
                  'ants_v3p': 'v3p', 'ants_v4p': 'v4p'}[A.dom] + A.variant
    c = make_cfg(V, sae, dom, tag, bout)
    if A.kr:
        c.kr = A.kr
    elif dom is BE.MICE:
        c.kr = 8  # the page shows 8 per row by default; 16 doubles the packs (asset store budget)
    add_outcomes(c, V / 'nes', '', V / '_cache')
    if (V / 'dines' / 'summary.csv').exists() or (V / 'dines' / 'maxpool_bouts' / 'summary.csv').exists():
        add_outcomes(c, V / 'dines', 'd', V / '_cache', dines=True)
    BE.plan_clips([c], A.max_pages)
    return c


def codes_cfg(c):
    cc = json.loads((c.dom.eci_dir / 'codes' / c.sae / 'config.json').read_text())
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
    if not out.exists() or A.overwrite:
        S = np.zeros(P * m, np.float64)
        cols = np.arange(m, dtype=np.int64)[None]
        for a in range(0, len(codes), 100_000):
            z = np.asarray(codes[a:a + 100_000], np.float32)
            g = np.asarray(arg[a:a + 100_000], np.int64)
            k = (z > 0) & (g >= 0)
            S += np.bincount((g * m + cols)[k], weights=z[k], minlength=P * m)
        n = len(codes)
        np.savez(out, act=(S / n).reshape(P, m).astype(np.float32), grid=G, n_frames=n, occ=np.zeros(0),
                 note=f'every frame ({n:,}): per patch, the neuron\'s code summed over the frames where that patch is '
                      f'its strongest token, divided by the number of frames (peak-patch map; R4 stores no per-patch '
                      f'codes)')
        log(f'{c.tag} arena: {n} frames, grid {G}')
    bg_out = c.assets / BE.bg_name(c)
    if not bg_out.exists() or A.overwrite:
        c.assets.mkdir(parents=True, exist_ok=True)
        bgd = Path(cc['backgrounds'])
        zs = [np.load(f) for f in sorted(bgd.glob('*.npz'))]
        vids = set(pd.read_csv(c.dom.ann_path, usecols=['observation_id'])['observation_id'].astype(str))
        ims = [z['pix_bg'] for f, z in zip(sorted(bgd.glob('*.npz')), zs) if f.stem.split('__')[0] in vids] or \
              [z['pix_bg'] for z in zs]
        im = np.median(np.stack(ims), 0).astype(np.uint8)
        Image.fromarray(im).resize((384, 384), Image.BILINEAR).save(bg_out, 'WEBP', quality=80)
        log(f'{c.tag} arena: background = median of {len(ims)} per-video backgrounds -> {bg_out}')


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
    heat = 'real' if sigs and all(x.startswith('r4heat-') for x in sigs) else 'peak'
    log(f'dump {c.tag}: heat maps {heat} ({sum(x.startswith("r4heat-") for x in sigs)}/{len(sigs)} real)')
    core['heat_kind'] = heat
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


# ---------------------------------------------------------------------- assemble
def v36_all():
    s = V36_PAGE.read_text()
    i = s.find('const ALL = ')
    j = s.find('\n', i)
    return json.loads(s[i + len('const ALL = '):j].rstrip(';'))


KEEP_IDS = ('boutratemaxpool', 'meanactivationma', 'xlatencymax')  # the v36 max-pooling outcomes
MODEL_TIPS = {'online': 'The published configuration (atlas version 39): DINOv2 patch features at 448 px, the '
                        'animal mask, Matryoshka BatchTopK SAE with 1024 neurons (k 16), max pooling',
              'new': 'R4: DINOv3-L/16 (block 20) at an input sized to the animal (mice 1024 px, ants 768-832 px, '
                     'frogs 800 px), the same lab animal masks, each patch described by its token and its change over '
                     'the last 2 s ([z_t, z_t - z_(t-2 s)]), Matryoshka BatchTopK SAE with 512 neurons (k 16), max '
                     'pooling; a frame fires when its code, divided by the neuron\'s typical size, passes one '
                     'domain-wide threshold (fnorm)',
              'new (no Δ)': 'R4 without the 2 s change half: DINOv3-L/16 token only (variant d0)'}
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
R4_SIZE = {'mice': (1024, 64), 'ants_v2p': (768, 48), 'ants_v3p': (832, 52), 'ants_v4p': (816, 51), 'frogs': (800, 50)}
MASK = {'mice': 'mouse mask (rule fg448, per-video backgrounds)', 'ants': 'ant mask (rule ants, per-video backgrounds)',
        'frogs': 'frog mask (rule frogs, per-video backgrounds)'}


def spec_of(e):
    dom, online, meth = e['domain'], e['model']['model'] == 'online', e['model']['method']
    key = e['sae'].split('#')[1].rsplit('_', 1)[0] if '#' in e['sae'] else None
    seed = re.search(r'_s(\d)', e['sae']).group(1)
    if online:
        fm, layer = 'DINOv2-base', 'last layer (after the final LayerNorm)'
        res = '448 px (32×32 patches): the 512 px frame resized'
        sae = 'Matryoshka BatchTopK 1024, k=16'
        inp = 'token'
        fire = 'global threshold (the SAE\'s inference threshold)'
        seedtxt = f'seed {seed} (single run)'
        rot = ('arena rotated so the odor corner is top right' if dom == 'mice' else
               'videos mirrored so the mutant side is on the right' if dom == 'frogs' else 'none')
        cov = 'none' if meth == 'NES' else 'foreground size (n_fg)'
    else:
        S, G = R4_SIZE[key]
        fm, layer = 'DINOv3-L/16', 'block 20 of 24 (+ the final LayerNorm)'
        res = f'{S} px ({G}×{G} patches), decoded from the native video'
        sae = 'Matryoshka BatchTopK 512, k=16'
        inp = 'token + 2 s change'
        fire = 'fnorm: a frame fires when its code ÷ the neuron\'s typical size passes one domain-wide threshold'
        seedtxt = f'seed {seed}: best held-out reconstruction (FVE) of 3 seeds'
        rot = ('none (videos not rotated)' if dom == 'mice' else
               'videos mirrored so the mutant side is on the right' if dom == 'frogs' else 'none')
        cov = 'foreground size (n_fg)' + (' + recording day' if dom == 'ants' else '')
    line = [['FM', fm], ['SAE', sae.replace(' BatchTopK', '')], ['Input', inp], ['Resolution', res.split(' (')[0]],
            ['Pooling', 'max']]
    win = {'mice': ('habituation: whole 30 min; odor, post: whole 15 min' if online else
                    'habituation: last 15 min (minutes 15–30); odor, post: whole 15 min'),
           'ants': 'whole video (v2: 10 min; v3, v4, v4_infested: 30 min)',
           'frogs': 'hour 1 of each frog (18,000 frames)'}[dom]
    test = ('t-test, Bonferroni; forward: add the most significant neuron, re-test the others given those found'
            if meth == 'NES' else
            'generalized covariance measure tests (cross-fitted); forward by p-value, then a backward check of each '
            'found neuron given every subset of the others')
    sets = {o['kind']: o['setting'] for o in e['outcomes']}
    th = re.search(r'threshold_q ([\d.]+)', sets.get('rate', '') or sets.get('latency', ''))
    details = [
        ['Foundation model', f'{fm}, {layer}'],
        ['Frames', '5 per second'],
        ['Resolution', res],
        ['Mask', MASK[dom] + ('' if online else ', projected from the 32×32 grid')],
        ['Rotation / mirroring', rot],
        ['Patch input', 'token z_t' if online else '[z_t, z_t − z_(t − 2 s)] (2048-d)'],
        ['SAE', sae + (', prefixes 128 / 256 / 512' if not online else ', Matryoshka prefixes')
         + (', one SAE for v2, v3, v4, v4_infested' if dom == 'ants' and not online else '')],
        ['Seed', seedtxt],
        ['Frame firing rule', fire],
        ['Spatial pooling', 'max over the mask patches'],
        ['Temporal aggregation', 'bout rate (bouts/min), mean activation, latency to first bout'
         + (f'; bout threshold q {th.group(1)}' if th else '')],
        ['Neurons searched', 'first 128 (main) or first 256'],
        ['Search', f'{meth}: {test}'],
        ['Search covariates', cov],
        ['Time window', win],
        ['Priors (built in)', 'the effect is on the animals (mask); fixed camera, so a patch is a fixed arena spot; '
                              'a concept anywhere on the animal counts (max pooling)'
         + ('' if online else '; ~10 patches per body length; behaviour includes motion (2 s change)')],
        ['Hypotheses (assumed)', 'an effect shows as a shift of a per-video neuron outcome; units independent (video, '
                                 'or pool when paired); known confounds are only those listed in the experiment details'],
    ]
    return {'line': line, 'details': details}


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
    log('source table: ' + ', '.join(f'{d} {len(t)} videos' for d, t in out.items()))
    return out


PICKS = {'r4miced2': OUT / 'res/mice_d2', 'r4v2pd2': OUT / 'res/ants_v2p_d2', 'r4v3pd2': OUT / 'res/ants_v3p_d2',
         'r4v4pd2': OUT / 'res/ants_v4p_d2', 'r4frogsd2': OUT / 'res/frogs_d2',
         'fg448al': ROOT / ONLINE['mice'][2], 'antsfg': ROOT / ONLINE['ants'][2], 'frogsfg': ROOT / ONLINE['frogs'][2]}


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
    for key in [f'{k}_{v}' for k in NEW for v in ('d2', 'd0')]:
        b = OUT / 'build' / key
        if not (b / 'core.json').exists():
            continue
        cj = json.loads((b / 'core.json').read_text())
        for f in sorted((b / 'data').glob('*.json')):
            rw_json(f, site / 'assets' / 'data' / f.name, amap)
        for f in sorted((b / 'assets').glob('*')):
            if f.suffix == '.webp' or (f.suffix == '.mp4' and f.name not in amap):
                shutil.copy(f, site / 'assets' / f.name)
        ents = []
        for e in cj['entries']:
            if amap:
                e = json.loads(rw_text(json.dumps(e), amap))
            if e.get('heat_kind') != 'real':
                e['heat_note'] = ('New model: R4 stores, per frame and neuron, only the position of its strongest '
                                  'patch, so the heat shows that peak patch (blurred), not a full map of every patch.')
            dom = e['domain']
            k0 = key.rsplit('_', 1)[0]
            series = {'ants_v2': 'v2', 'ants_v3': 'v3', 'ants_v4': 'v4'}.get(k0)
            e['pooled'] = k0.endswith('p') and k0.startswith('ants')
            e['sae'] = f'{e["sae"]}#{key}'  # unique per build (the pooled ants series share one SAE name)
            e['note'] = ('New model (R4, ' + ('2 s change, d2' if key.endswith('d2') else 'no change half, d0') +
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
    for k in list(new_entries):
        if k.endswith('_d0') or (k.startswith('ants_') and not k.endswith('p_d2')):
            log(f'assemble: {k} left out (new model on the page = d2, pooled ants SAE)')
            new_entries.pop(k)
    have_v4 = any(k.startswith('ants_v4') for k in new_entries)
    order_new = [k for k in ('mice_d2', 'ants_v2p_d2', 'ants_v3p_d2', 'ants_v4p_d2', 'ants_v2_d2', 'ants_v3_d2',
                             'ants_v4_d2', 'frogs_d2', 'mice_d0', 'ants_v2p_d0', 'ants_v3p_d0', 'ants_v4p_d0',
                             'ants_v2_d0', 'ants_v3_d0', 'ants_v4_d0', 'frogs_d0') if k in new_entries]
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

    for e in saes:  # frogs: the tadpole phase (stage 44-48) has no codes in either model yet
        if e['domain'] == 'frogs':
            e.setdefault('pending', {})['tadpoles (stage 44–48)'] = ('pending: no tadpole codes exist yet for this '
                                                                     'model (WT, FoxP1, En1; 118 videos of 60 min)')
    # ---- top level
    mice_new = next((e for e in saes if e['domain'] == 'mice' and e['model']['model'] == 'new'), None)
    if mice_new is not None:  # the shared mice video list
        vb = builds[mice_new['build']]['videos_mice']
        assert vb == v36['videos'], 'mice video list differs between the new build and v36'
    doms = []
    for d, title in (('mice', 'Mice'), ('ants', 'Ants'), ('frogs', 'Frogs')):
        es = [x for x in saes if x['domain'] == d]
        if es:
            doms.append({'id': d, 'title': title, 'default': next((x['sae'] for x in es if x['model'] ==
                                                                   {'model': 'new', 'method': 'NES'}), es[0]['sae'])})
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
                       'clip_lengths': ['frame', '1s'], 'methods': ['NES', 'NESS'], 'models': ['new', 'online']},
           'version_files': {str(p.relative_to(site)): p.stat().st_size for p in sorted(files) if p.suffix not in ('.mp4', '.webm')},
           'packs_to_upload': {p.name: p.stat().st_size for p in sorted(packs) if p.name.startswith('r4')},
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
    if A.model == 'new':
        if 'prep' in steps:
            step_prep_new()
        if 'checks' in steps:
            step_checks_new()
        rest = [s for s in steps if s not in ('prep', 'checks')]
        if rest:
            c = new_cfg()
            for s in rest:
                if s == 'data':
                    BE.step_data(c, A.overwrite)
                elif s == 'patch':
                    step_patch_new(c)
                elif s == 'arena':
                    step_arena_new(c)
                elif s == 'render':
                    step_render_new(c)
                elif s == 'dump':
                    key = new_paths()[-1]
                    dump(c, lambda core: split_methods(core, key, 'new' if A.variant == 'd2' else 'new (no Δ)'),
                         c.out)
                else:
                    raise SystemExit(f'unknown step {s}')
    else:
        if 'prep' in steps:
            step_prep_online()
        if 'dump' in steps:
            step_dump_online_v4() if A.dom == 'ants_v4' else step_dump_online()
