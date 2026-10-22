"""
ECI v2 rollout tools (called through scripts/eci/v2/run.py): align, selftest, manifest.

align (GPU): behaviour alignment of the deployed v2 codes vs reference models on the SAME frames and splits.
    Frame value per neuron: v2 'raw' (frame max, theta*) and v2 'nes' (the NES input: 0.6 s smoothed, fnorm
    thresholded); refs: the v1 online model (deployed atlas SAE) and r4 where their codes exist in the same row space.
    Cross-fitted R~ (neuron picked on one half of the units, scored on the other, 10 random splits) and cf AUROC
    (src/eci/v2/score.py). Units: mice pools, ants / frogs videos.
    mice: the 144 annotated videos (every 5 fps frame), labels nose_nose (Y_nn | Y_np), nose_tail, nn_mutual,
          np_directional; frame sets 'all' and 'consistent' (annotators SG / SD / MR / CP).
    ants: per labelled series (v2, v3): every labelled 5 fps frame, groom_any / yellow / blue, v3 on_lid_*.
selftest (GPU): per series one video (first 64 frames): decode -> canonical -> mask -> tokens -> codes through the
    library, checked against (a) the stored full-pass codes of that video when they exist (max |diff|, fire
    agreement), (b) for mice the round-2 diagnosis tokens of the same frames (encoder + mask equality).
manifest (CPU): results/vision/eci_v2/manifest.json listing every output path of the rollout.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))
import run as RUN  # noqa: E402
from src.eci.v2.score import crossfit, splits_of  # noqa: E402

log = RUN.log
OUT, BEE = RUN.OUT, RUN.BEE


def mice_labels(rows, ann='dataset/mice/v1/annotations.csv'):
    a = pd.read_csv(REPO / ann,
                    usecols=['observation_id', 'Y_nn', 'Y_np', 'Y_nt', 'W_annotator'], dtype={'W_annotator': str})
    y = a.iloc[rows]
    Y = np.stack([(y.Y_nn.values > 0) | (y.Y_np.values > 0), y.Y_nt.values > 0, y.Y_nn.values > 0, y.Y_np.values > 0], 1)
    unit = pd.Series(y.observation_id.values).str.rsplit('_', n=2).str[0].values
    return Y, unit, y.W_annotator.values, ('nose_nose', 'nose_tail', 'nn_mutual', 'np_directional')


def ants_labels(layout, rows):
    a = pd.read_csv(REPO / layout / 'annotations.csv', usecols=['observation_id', 'Y_Y2F', 'Y_B2F', 'Y_YOL', 'Y_BOL', 'Y_FOL'])
    y = a.iloc[rows]
    g = lambda c_: y[c_].fillna(0).values > 0  # noqa: E731
    Y = np.stack([g('Y_Y2F') | g('Y_B2F'), g('Y_Y2F'), g('Y_B2F'), g('Y_YOL') | g('Y_BOL') | g('Y_FOL'), g('Y_YOL'),
                  g('Y_BOL'), g('Y_FOL')], 1)
    labs = ('groom_any', 'groom_yellow', 'groom_blue', 'on_lid_any', 'on_lid_yellow', 'on_lid_blue', 'on_lid_focal')
    lid = y[['Y_YOL', 'Y_BOL', 'Y_FOL']].notna().all(1).all()
    if not lid:
        Y, labs = Y[:, :3], labs[:3]
    return Y, y.observation_id.values, labs


def ref_rows(ref_ann, layout_ann, rows):
    """Rows of a reference model's annotations.csv matching layout rows (observation_id, frame_idx); -1 = absent."""
    A = pd.read_csv(ref_ann, usecols=['observation_id', 'frame_idx'])
    L = pd.read_csv(layout_ann, usecols=['observation_id', 'frame_idx']).iloc[rows]
    key = pd.MultiIndex.from_frame(A)
    idx = pd.Series(np.arange(len(A)), index=key)
    m = idx.reindex(pd.MultiIndex.from_frame(L.reset_index(drop=True))).values
    return np.where(np.isnan(m), -1, m).astype(np.int64)


def cmd_align(args):
    import torch
    dev = torch.device('cuda')
    c = RUN.dcfg(args.domain)
    name = RUN.best_name(c)
    out = {'sae': name, 'series': {}}
    for sc in c['series']:
        al = sc.get('align')
        if not al:
            continue
        lay_ann = RUN.layout_ann(sc)
        cols = pd.read_csv(lay_ann, nrows=1).columns
        if 'Y_nn' in cols:
            a = pd.read_csv(lay_ann, usecols=['Y_nn'])
            rows = np.flatnonzero(a.Y_nn.notna().values)
            Y, unit, ann, labs = mice_labels(rows, lay_ann)
            sets = {'all': np.arange(len(rows)),
                    'consistent': np.flatnonzero(pd.Series(ann).isin(('SG', 'SD', 'MR', 'CP')).values)}
        elif 'Y_Y2F' in cols:
            a = pd.read_csv(lay_ann, usecols=['Y_Y2F', 'Y_B2F'])
            rows = np.flatnonzero(a[['Y_Y2F', 'Y_B2F']].notna().all(1).values)
            if not len(rows):
                continue
            Y, unit, labs = ants_labels(sc['layout'], rows)
            sets = {'all': np.arange(len(rows))}
        else:
            continue
        units, SPL = splits_of(set(unit))
        uidx = pd.Series(range(len(units)), index=units).loc[unit].values
        models = {'v2_raw': (RUN.codes_dir(c, sc, f'{name}_raw') / 'codes_max.npy', None),
                  'v2_nes': (RUN.codes_dir(c, sc, name) / 'codes_max.npy', None),
                  'v2_smooth': (RUN.codes_dir(c, sc, f'{name}_smooth') / 'codes_max.npy', None),
                  'v2_prob': (RUN.codes_dir(c, sc, f'{name}_prob') / 'codes_max.npy', None),
                  'v2_map': (RUN.codes_dir(c, sc, f'{name}_map') / 'codes_max.npy', None),
                  'v2_mapgamma': (RUN.codes_dir(c, sc, f'{name}_mapgamma') / 'codes_max.npy', None)}
        for nm, (path, ref_ann) in al.get('refs', {}).items():
            models[nm] = (REPO / path / 'codes_max.npy', REPO / ref_ann if ref_ann else None)
        res = {'n_frames': int(len(rows)), 'n_units': len(units),
               'base_rate': {lb: float(Y[:, i].mean()) for i, lb in enumerate(labs)}, 'models': {}}
        for nm, (path, ref_ann) in models.items():
            if not Path(path).exists():
                log(f'{sc["name"]} {nm}: {path} missing, skipped')
                continue
            mx = np.load(path, mmap_mode='r')
            rr = rows if ref_ann is None else ref_rows(ref_ann, lay_ann, rows)
            ok = rr >= 0
            X = np.zeros((len(rows), mx.shape[1]), np.float16)
            o = np.argsort(rr[ok])
            X[np.flatnonzero(ok)[o]] = np.asarray(mx[rr[ok][o]])
            res['models'][nm] = {'width': int(mx.shape[1]), 'rows_present': float(ok.mean()), 'sets': {}}
            for sn, ix in sets.items():
                ix = ix[ok[ix]]
                Xg = torch.from_numpy(X[ix]).to(dev)
                Yg = torch.from_numpy(Y[ix]).to(dev)
                o2 = crossfit(Xg, Yg, {k: h[uidx[ix]] for k, h in SPL.items()}, labs)
                res['models'][nm]['sets'][sn] = {'mean': o2['mean'],
                                                 'rnd0': {lb: o2['splits']['rnd0'][lb]['neurons'] for lb in labs}}
                log(f'{sc["name"]} {nm} {sn}: ' + ', '.join(f'{lb} R~ {o2["mean"][lb]["R"]:.3f} AUROC '
                                                           f'{o2["mean"][lb]["A"]:.3f}' for lb in labs))
                del Xg, Yg
                torch.cuda.empty_cache()
        mp = RUN.codes_dir(c, sc, f'{name}_map') / 'codes_max.npy'
        if mp.exists() and 'v2_smooth' in res['models']:
            M_ = np.load(mp, mmap_mode='r')
            res['map_active_fraction_of_selected'] = {}
            for lb in labs:
                js = sorted(set(res['models']['v2_smooth']['sets']['all']['rnd0'][lb]) |
                            set(res['models'].get('v2_map', {}).get('sets', {}).get('all', {}).get('rnd0', {}).get(lb, [])))
                res['map_active_fraction_of_selected'][lb] = {
                    int(j): {'all_frames': float(np.asarray(M_[rows[::7], j], np.float32).mean()),
                             'label_positive_frames': float(np.asarray(M_[rows[Y[:, labs.index(lb)]][::3], j],
                                                                       np.float32).mean())} for j in js}
        out['series'][sc['name']] = res
    (OUT / args.domain / 'align.json').write_text(json.dumps(out, indent=1, default=float))
    L = [f'# {args.domain}: behaviour alignment of {name}', '',
         'R~ = cross-fitted normalised R-precision (best neuron on one half of the units, scored on the other, 10 random '
         'splits); AUROC = cf AUROC. v2_raw = frame max codes; v2_nes = the NES input (0.6 s smoothed, fnorm '
         'thresholded).', '']
    for ser, r in out['series'].items():
        labs = list(r['base_rate'])
        L += [f'## {ser} ({r["n_frames"]:,} frames, {r["n_units"]} units)', '',
              '| model | set | ' + ' | '.join(f'{lb} R~ / AUROC' for lb in labs) + ' |', '|---|---|' + '---|' * len(labs)]
        for nm, m in r['models'].items():
            for sn, s_ in m['sets'].items():
                L.append(f'| {nm} (W {m["width"]}) | {sn} | ' + ' | '.join(
                    f'{s_["mean"][lb]["R"]:.3f} / {s_["mean"][lb]["A"]:.3f}' for lb in labs) + ' |')
        L.append('')
    (OUT / args.domain / 'align.md').write_text('\n'.join(L) + '\n')


def cmd_selftest(args):
    """One video per series: library pipeline vs the stored full pass (and, mice, vs the round-2 diagnosis tokens)."""
    import torch
    from src.eci.v2 import sae as SA
    from src.eci.v2.context import sae_input
    from src.eci.v2.pipeline import Series
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    c = RUN.dcfg(args.domain)
    w = c['spec']['context_window']
    b = json.loads((OUT / args.domain / 'best.json').read_text())
    sae, norm = SA.load(Path(b['dir']) / 'sae_theta.pt', dev)
    out = {}
    for sc in c['series']:
        Se = Series(sc, dev)
        vi = args.shard if args.shard < len(Se.videos) else 0
        v = Se.videos[vi]
        n = 64
        fr = Se.run_job(**Se.job(v, 0, n))
        t0 = time.time()
        cms = []
        for b0 in range(0, n, 16):
            kk = np.arange(b0, b0 + 16)
            x, fd = Se.frames(v, fr[b0:b0 + 16], kk)
            z, keep = Se.tokens(x, fd)
            bi, pi = keep.nonzero(as_tuple=True)
            cm, _, _ = SA.frame_codes(sae, norm, sae_input(z[bi, pi], pi, bi, 16, Se.G, w), bi, 16, pos=pi)
            cms.append(cm.cpu().numpy())
        cm = np.concatenate(cms)
        r = {'obs': v['obs'], 'frames': n, 'fps': n / (time.time() - t0), 'kept_per_frame': None}
        f = BEE / args.domain / 'videos' / sc['name'] / f'{v["obs"]}.npz'
        if f.exists():
            st = np.load(f)['codes_max'][:n].astype(np.float32)
            r['vs_fullpass'] = {'max_abs_diff': float(np.abs(st - cm).max()),
                                'fire_agreement': float(((st > 0) == (cm > 0)).mean())}
        if args.domain == 'mice':
            r['vs_round2_tokens'] = mice_token_check(Se, dev)
        out[sc['name']] = r
        log(f'{sc["name"]}: ' + json.dumps(r))
    (OUT / args.domain / 'selftest.json').write_text(json.dumps(out, indent=1))


def mice_token_check(Se, dev, n=24):
    """Library encoder + tight mask vs the round-2 diagnosis store (scripts/eci/nnnt/r2.py, nnnt.Pipe) on n frames of
    one annotated video: mask agreement and token cosine on the common kept tokens."""
    import torch
    root = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/nnnt/r2/diag')
    F = pd.read_parquet(root / 'frames.parquet')
    tf = np.load(root / 'tok_frame.npy', mmap_mode='r')
    pos = np.load(root / 'tok_pos.npy', mmap_mode='r')
    tok = np.memmap(root / 'tok.f16', np.float16, 'r', shape=(len(tf), 1024))
    starts = np.r_[0, np.cumsum(np.bincount(np.asarray(tf), minlength=len(F)))]
    o = F.obs.values[0]
    fr_ = np.flatnonzero(F.obs.values == o)[:n]
    v = Se.V[o]
    agree, cos = [], []
    for f in fr_:
        k = int(F.k.values[f])
        x0 = Se.run_job(**Se.job(v, k, 1))
        x, fd = Se.frames(v, x0, np.array([k]))
        z, keep = Se.tokens(x, fd)
        p_lib = set(torch.nonzero(keep[0]).flatten().cpu().numpy().tolist())
        p_r2 = np.asarray(pos[starts[f]:starts[f + 1]]).astype(np.int64)
        agree.append(len(p_lib & set(p_r2.tolist())) / max(1, len(p_lib | set(p_r2.tolist()))))
        zr = torch.from_numpy(np.asarray(tok[starts[f]:starts[f + 1]]).astype(np.float32)).to(dev)
        zl = z[0, torch.from_numpy(p_r2).to(dev)].float()
        cos.append(float(torch.nn.functional.cosine_similarity(zl, zr, dim=1).min()))
    return {'video': o, 'frames': int(len(fr_)), 'mask_iou_mean': float(np.mean(agree)), 'mask_iou_min': float(np.min(agree)),
            'token_cos_min': float(np.min(cos))}


def strata_index(name_of):
    """Which extra contrasts / strata exist (status RUN / NOT RUN YET) per outcome variant, with their result dirs."""
    V = REPO / 'results/vision'
    out = {}
    nm = name_of['mice']
    rows = [('mice', 'family_S_sex', 'family_S', 'eci_v2/ness_S/mice')]
    for st in ('ash1l_all', 'kdm6b_all', 'kmt5b_all', 'all_f', 'all_m'):
        rows.append(('mice', f'families_AB_{st}', f'subsets/{st}', f'eci_v2/ness_strata/{st}/mice'))
    for dom, sid, sub, nroot in rows:
        for var, suf in (('continuous', '_smooth'), ('binary', ''), ('prob', '_prob')):
            s_ = nm + suf
            nd, sd = V / 'mice/eci/nes' / s_ / sub, V / nroot / s_
            ok = (nd / 'summary.csv').exists() and (nd / 'maxpool_latency' / 'summary.csv').exists()
            okn = (sd / 'dines' / 'summary.csv').exists()
            out.setdefault(dom, {}).setdefault(sid, {})[var] = {'NES': 'RUN' if ok else 'NOT RUN YET', 'NES_dir': str(nd),
                                                                'NESS': 'RUN' if okn else 'NOT RUN YET', 'NESS_dir': str(sd)}
    for st in ('ash1l_all', 'kdm6b_all', 'kmt5b_all', 'all_f', 'all_m'):
        out['mice'][f'family_C_{st}'] = {v: {'NES': 'NOT RUN YET', 'NESS': 'NOT RUN YET',
                                             'note': 'family C is computed inside atlas_v37_methods.py on the full cohort'}
                                         for v in ('continuous', 'binary', 'prob')}
    na = name_of['ants']
    for aid in ('infested_vs_healthy_t2', 'infested_vs_healthy_t6', 'infested_vs_healthy_t2_6'):
        for var, suf in (('continuous', '_smooth'), ('binary', ''), ('prob', '_prob')):
            nd = V / 'ants/eci_r4v4p/nes' / (na + suf)
            ok = (nd / aid).exists()
            out.setdefault('ants', {}).setdefault(f'v4_colony_{aid}', {})[var] = {
                'NES': 'RUN' if ok else 'NOT RUN YET', 'NES_dir': str(nd / aid),
                'NESS': 'RUN' if (V / 'eci_v2/ness/ants_eci_r4v4p' / (na + suf) / 'dines' / aid).exists() else 'NOT RUN YET',
                'NESS_dir': str(V / 'eci_v2/ness/ants_eci_r4v4p' / (na + suf) / 'dines' / aid)}
    return out


def cmd_manifest(args):
    """results/vision/eci_v2/manifest.json: every output path of the v2 rollout, per domain and series."""
    import glob
    import subprocess
    V = REPO / 'results/vision'
    g = lambda pat: sorted(glob.glob(str(pat), recursive=True))  # noqa: E731
    try:
        head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:  # CPU nodes without git: read the ref file
        ref = (REPO / '.git/HEAD').read_text().split()[-1]
        head = (REPO / '.git' / ref).read_text().strip() if (REPO / '.git' / ref).exists() else ref
    man = {'created': time.strftime('%Y-%m-%d %H:%M'), 'code_head': head,
           'nes_code_records': g(OUT / 'nes_code' / '*.json'),
           'spec': str(REPO / 'configs/eci_v2/representation.json'), 'library': str(REPO / 'src/eci/v2'),
           'entry_point': str(REPO / 'scripts/eci/v2/run.py'), 'nes_wrapper': str(REPO / 'scripts/eci/v2/nes.sh'),
           'freeze': {'decision': str(OUT / 'freeze/decision.json'), 'table': str(OUT / 'freeze/table.md'),
                      'sheets': g(OUT / 'freeze/*/sheets')},
           'discoveries': {'json': str(OUT / 'discoveries.json'), 'md': str(OUT / 'discoveries.md')},
           'nes_input_sets': {
               'primary': 'USER DECISION 2026-10-21: <sae>_smooth = centred 0.6 s mean of the frame max codes, no firing threshold (keeps '
                                   'the behaviour alignment; align.json v2_smooth)',
               'secondary': '<sae>: the same smoothed codes zeroed below the fnorm firing threshold (pre-specified, but it '
                            'destroys alignment: mice nose_nose R~ 0.174 raw -> 0.093, ants v2 groom_any 0.629 -> 0.075)',
               'prob': '<sae>_prob: per-neuron P(on | smoothed code), label-free (src/eci/v2/prob.py): a 2-component '
                       'Gaussian mixture on log of the positive smoothed codes of the calibration videos; P(on) = '
                       'posterior of the higher component (a probability under the mixture model), monotone, 0 on silent '
                       'frames. NES mean on it = expected fraction of time active per video; the runners\' bout rate and '
                       'latency threshold P(on) at its pooled q 0.95 (not an expected bout count). Single-neuron rank '
                       'alignment (R~ / AUROC) is invariant to this monotone per-neuron map, so align.json v2_prob is a '
                       'sanity check only.',
               'raw': '<sae>_raw: frame max codes, no smoothing (alignment reference)',
               'binary_map': '<sae>_map: 1[P(on) > 0.5], per-neuron MAP of the lognormal mixture (won the held-out '
                             'family selection vs gamma); bouts / latency on it use threshold 0 (above = active)',
               'outcome_variants': {'continuous': '<sae>_smooth', 'binary': '<sae>', 'prob': '<sae>_prob',
                                    'binary_map': '<sae>_map'}},
           'domains': {}}
    roots = {'mice': {'v1': ('mice/eci/nes', 'mice')},
             'ants': {s_: (f'ants/{e}/nes', f'ants_{e}') for s_, e in (('v2', 'eci_r4v2p'), ('v3', 'eci_r4v3p'),
                                                                     ('v4', 'eci_r4v4p'))},
             'frogs': {'juv': ('frogs/eci/nes', 'frogs'), 'tad': ('frogs/eci_tadpole_r4/nes', 'frogs_tadpole')}}
    for dom in ('mice', 'ants', 'frogs'):
        c = RUN.dcfg(dom)
        name = RUN.best_name(c)
        b = json.loads((OUT / dom / 'best.json').read_text())
        d = {'config': str(REPO / f'configs/eci_v2/{dom}.json'), 'sae': b, 'sae_checkpoint': b['dir'] + '/sae_theta.pt',
             'train': str(OUT / dom / 'train.json'), 'health': str(OUT / dom / 'health.json'),
             'fire': str(OUT / dom / 'fire.json'), 'align_json': str(OUT / dom / 'align.json'),
             'align_md': str(OUT / dom / 'align.md'), 'selftest': str(OUT / dom / 'selftest.json'), 'series': {}}
        for sc in c['series']:
            nr, key = roots[dom][sc['name']]
            s_ = {'layout': str(REPO / sc['layout']), 'layout_annotations': str(RUN.layout_ann(sc)),
                  'pool': str(RUN.pool_dir(c, sc)), 'per_video_full_pass': str(BEE / dom / 'videos' / sc['name']),
                  'summaries': str(OUT / dom / f'summaries_{sc["name"]}.npz'),
                  'heat_dir': str(OUT / dom / 'heat' / sc['name']), 'heat_index': str(OUT / dom / 'heat' / sc['name'] / 'index.json'),
                  'heat_files': len(g(OUT / dom / 'heat' / sc['name'] / 'n*.npz')), 'sets': {}}
            for tag, nm in (('primary', f'{name}_smooth'), ('secondary', name), ('prob', f'{name}_prob'),
                            ('binary_map', f'{name}_map'), ('raw', f'{name}_raw')):
                e = {'name': nm, 'codes': str(RUN.codes_dir(c, sc, nm)), 'codes_link': str(REPO / sc['layout'] / 'codes' / nm)}
                if tag != 'raw':
                    e['nes_dir'] = str(V / nr / nm)
                    e['nes_result_json'] = g(V / nr / nm / '**' / 'result.json')
                    e['nes_summary_csv'] = g(V / nr / nm / '**' / 'summary.csv')
                    e['ness_dir'] = str(OUT / 'ness' / key / nm)
                    e['ness_result_json'] = g(OUT / 'ness' / key / nm / '**' / 'result.json') + \
                        g(OUT / 'ness_v3_2v8' / key / nm / '**' / 'result.json')
                    e['ness_summary_csv'] = g(OUT / 'ness' / key / nm / '**' / '*summary.csv') + \
                        g(OUT / 'ness_v3_2v8' / key / nm / '**' / '*summary.csv')
                s_['sets'][tag] = e
            d['series'][sc['name']] = s_
        d['prob'] = str(OUT / dom / 'prob.json')
        d['binarisation'] = {'map': str(OUT / dom / 'map.json'), 'bimodality_smoothed': str(OUT / dom / 'bimodality.npz'),
                             'bimodality_raw': str(OUT / dom / 'bimodality_raw.json'),
                             'mixture_family': str(OUT / dom / 'mixsel.json'),
                             'decision': 'MAP binary NOT proposed as default (pre-registered tolerance failed on every '
                                         'label; prob.py docstring); continuous stays primary'}
        man['domains'][dom] = d
    man['sensitivity'] = {'index': str(OUT / 'sensitivity.json'), 'summary': str(OUT / 'sensitivity_summary.json'),
                          'code': str(REPO / 'scripts/eci/v2/sensitivity.py'),
                          'note': 'stable in k of n checks per primary-selected neuron; per-setting result.json paths in '
                                  'sensitivity.json files'}
    man['strata'] = strata_index(name_of={dm: RUN.best_name(RUN.dcfg(dm)) for dm in man['domains']})
    (OUT / 'manifest.json').write_text(json.dumps(man, indent=1))
    log(f'manifest: {OUT / "manifest.json"}')

# ---------------------------------------------------------------------------------------------- heat maps
class TokenCodes:
    """Sparse per-token codes of one video (full pass npz): heat(k, j) -> (G*G,) float32 activation map."""

    def __init__(self, f):
        z = np.load(f)
        self.G = int(z['G'])
        self.nk = z['n_kept'].astype(np.int64)
        self.tok0 = np.r_[0, np.cumsum(self.nk)]
        self.pos = z['tok_pos'].astype(np.int64)
        ac = z['act_count'].astype(np.int64)
        self.act0 = np.r_[0, np.cumsum(ac)]
        self.idx, self.val = z['act_idx'], z['act_val']
        self.cm = z['codes_max']

    def heat(self, k, j):
        h = np.zeros(self.G * self.G, np.float32)
        t0, t1 = self.tok0[k], self.tok0[k + 1]
        for t in range(t0, t1):
            a0, a1 = self.act0[t], self.act0[t + 1]
            m = self.idx[a0:a1] == j
            if m.any():
                h[self.pos[t]] = float(self.val[a0:a1][m][0])
        return h


def heat_rows(c, sc, rows_js):
    """{layout row: set(neurons)} of one series -> {(row, j): (G, G) fp16 activation map}. Exact: the stored token codes
    of the full pass (their max over tokens equals codes_max_raw)."""
    P = json.loads((REPO / sc['plan']).read_text())
    lay = np.array([v['lay_lo'] for v in P['videos']])
    out = {}
    by_v = {}
    for r, js in rows_js.items():
        vi = int(np.searchsorted(lay, r, side='right') - 1)
        by_v.setdefault(vi, []).append((r, js))
    for vi, items in by_v.items():
        v = P['videos'][vi]
        T = TokenCodes(BEE / c['domain'] / 'videos' / sc['name'] / f'{v["obs"]}.npz')
        for r, js in items:
            k = r - v['lay_lo']
            for j in js:
                h = T.heat(k, j)
                assert abs(h.max() - float(T.cm[k, j])) < 1e-3, (v['obs'], k, j, h.max(), float(T.cm[k, j]))
                out[(r, j)] = h.reshape(T.G, T.G).astype(np.float16)
    return out


def cmd_heat(args):
    """Per-neuron real heat maps. --picks <build_explorer picks.json>: the atlas' own clips (BE.clip_rows, kr 8), written
    as <out>/x<j>.npz in the r4heat format (sig 'v2heat-' + json(clip rows), vmax = p99 of the positive values, one
    (rows, G, G) fp16 array per clip-row key). Without --picks: every neuron reported by NES / NESS in this series' result
    sets, its 8 top videos by peak NES-code value, a 1 s clip (5 frames) centred on that peak -> <out>/n<j>.npz (rows,
    heat (rows, G, G), video, k, vmax)."""
    c = RUN.dcfg(args.domain)
    name = RUN.best_name(c)
    sc = RUN.series_of(c, args.series)
    outd = Path(args.out_dir) if args.out_dir else OUT / args.domain / 'heat' / sc['name']
    outd.mkdir(parents=True, exist_ok=True)
    if args.picks:
        sys.path.insert(0, str(REPO / 'scripts/eci'))
        import build_explorer as BE
        picks = json.loads(Path(args.picks).read_text())
        neurons = picks['by_key'].get('max', {})
        RR = {int(j): BE.clip_rows(neurons[j], 8, BE.SHOW_LENGTHS) for j in neurons}
        need = {}
        for j, rr in RR.items():
            for rows in rr.values():
                for r in rows:
                    need.setdefault(int(r), set()).add(j)
        H = heat_rows(c, sc, need)
        for j, rr in RR.items():
            arrs = {k: np.stack([H[(int(r), j)] for r in rows]) for k, rows in rr.items()}
            pv = np.concatenate([a[a > 0].astype(np.float32) for a in arrs.values()] + [np.zeros(0, np.float32)])
            np.savez(outd / f'{BE.pre("max")}{j:04d}.npz', sig='v2heat-' + json.dumps({k: [int(x) for x in v]
                                                                                         for k, v in rr.items()}),
                     vmax=float(np.quantile(pv, 0.99)) if len(pv) else 1.0, **arrs)
        log(f'{len(RR)} neurons -> {outd}')
        return
    js = set()
    import glob
    nes_root = {'mice': 'mice/eci/nes', 'frogs': 'frogs/eci/nes' if sc['name'] == 'juv' else 'frogs/eci_tadpole_r4/nes',
                'ants': f'ants/{Path(sc["layout"]).name}/nes'}[args.domain]
    key = {'mice': 'mice', 'frogs': 'frogs' if sc['name'] == 'juv' else 'frogs_tadpole',
           'ants': f'ants_{Path(sc["layout"]).name}'}[args.domain]
    files = []
    for nm in (name, f'{name}_smooth'):
        files += glob.glob(str(REPO / 'results/vision' / nes_root / nm / '**' / 'result.json'), recursive=True)
        files += glob.glob(str(OUT / 'ness' / key / nm / '**' / 'result.json'), recursive=True)
        files += glob.glob(str(OUT / 'ness_v3_2v8' / key / nm / '**' / 'result.json'), recursive=True)
    why = {}
    for f in files:
        d = json.loads(Path(f).read_text())
        sets = [k_ for k_, r in d.items() if isinstance(r, dict) and 'bonferroni' in k_ and
                (k_.endswith('_full') or k_.endswith('_common'))]
        for k_ in sets:
            r = d[k_]
            for x in r.get('selected', []) or []:
                j = int(x['neuron'] if isinstance(x, dict) else x)
                js.add(j)
                why.setdefault(j, set()).add('selected')
        # candidates: the 5 lowest first-round p of the widest prefix of every setting (NES result.json only)
        wide = {}
        for k_ in sets:
            fr = d[k_].get('first_round')
            if isinstance(fr, dict) and isinstance(fr.get('p'), list) and fr['p']:
                pre = int(k_.split('_')[0][1:])
                base = k_.split('_', 1)[1]
                if pre >= wide.get(base, (0, None))[0]:
                    wide[base] = (pre, fr)
        for pre, fr in wide.values():
            pv = np.array([x if x is not None else np.nan for x in fr['p']], float)
            for i in np.argsort(np.where(np.isfinite(pv), pv, np.inf))[:5]:
                j = int(fr['neuron'][i])
                js.add(j)
                why.setdefault(j, set()).add('candidate')
    al = OUT / args.domain / 'align.json'
    if al.exists():
        A = json.loads(al.read_text())['series'].get(sc['name'], {}).get('models', {})
        for m in ('v2_raw', 'v2_smooth'):
            for lb, ns in A.get(m, {}).get('sets', {}).get('all', {}).get('rnd0', {}).items():
                for j in ns:
                    js.add(int(j))
                    why.setdefault(int(j), set()).add(f'aligned:{lb}')
    P = json.loads((REPO / sc['plan']).read_text())
    mx = np.load(RUN.codes_dir(c, sc, f'{name}_smooth') / 'codes_max.npy', mmap_mode='r')
    js = sorted(js)
    log(f'{len(js)} reported neurons')
    peaks = np.zeros((len(P['videos']), len(js)), np.float32)
    arg = np.zeros((len(P['videos']), len(js)), np.int64)
    for i, v in enumerate(P['videos']):
        m = np.asarray(mx[v['lay_lo']:v['lay_lo'] + v['n']][:, js]).astype(np.float32)
        peaks[i], arg[i] = m.max(0), m.argmax(0)
    col = {j: q for q, j in enumerate(js)}
    if args.nshards > 1:
        js = [j for i, j in enumerate(js) if i % args.nshards == args.shard]
    js = [j for j in js if args.force or not (outd / f'n{j:04d}.npz').exists()]
    need, meta = {}, {}
    for j in js:
        q = col[j]
        top = [i for i in np.argsort(-peaks[:, q])[:8] if peaks[i, q] > 0]
        rows = []
        for i in top:
            v = P['videos'][i]
            k0 = int(np.clip(arg[i, q] - 2, 0, v['n'] - 5))
            rows += [v['lay_lo'] + k for k in range(k0, k0 + 5)]
        meta[j] = rows
        for r in rows:
            need.setdefault(r, set()).add(j)
    H = heat_rows(c, sc, need)
    import io
    import torch
    from PIL import Image
    from src.eci.v2.pipeline import Series
    Se = Series(sc, torch.device('cpu'), load_encoder=False)
    lay = np.array([v['lay_lo'] for v in P['videos']])
    thumb_cache = {}

    def thumb(r):
        if r not in thumb_cache:
            vi = int(np.searchsorted(lay, r, side='right') - 1)
            v = P['videos'][vi]
            k = r - v['lay_lo']
            fr = Se.run_job(**Se.job(v, k, 1))
            x, _ = Se.frames(v, fr, np.array([k]))
            im = Image.fromarray(x[0].permute(1, 2, 0).numpy()).resize((256, 256), Image.BILINEAR)
            b = io.BytesIO()
            im.save(b, format='JPEG', quality=85)
            thumb_cache[r] = np.frombuffer(b.getvalue(), np.uint8)
        return thumb_cache[r]
    index = {}
    for j, rows in meta.items():
        if not rows:
            continue
        a = np.stack([H[(r, j)] for r in rows])
        pv = a[a > 0].astype(np.float32)
        peaks = rows[2::5]
        th = [thumb(r) for r in peaks]
        P_ = json.loads((REPO / sc['plan']).read_text())['videos']
        obs = [P_[int(np.searchsorted(lay, r, side='right') - 1)]['obs'] for r in rows]
        np.savez(outd / f'n{j:04d}.npz', rows=np.array(rows), obs=np.array(obs), heat=a,
                 vmax=float(np.quantile(pv, 0.99)) if len(pv) else 1.0, sae=name, why=np.array(sorted(why.get(j, []))),
                 grid=a.shape[-1], frame=('canonical encoder input: ' + (f'rot90 CCW k = per-video odour-corner turn '
                                          '(src/eci/v2/pipeline.py)' if sc.get('rot') else
                                          'hflip per experiment.csv' if sc['name'] == 'juv' else
                                          f'{sc["crop"]["side"]} px animal-following crop -> {sc["S"]} px'
                                          if sc.get('geometry') == 'crop' else 'native frame')),
                 rot90=np.array([Se.rot.get(o, 0) for o in obs]), peak_rows=np.array(peaks),
                 thumbs=np.concatenate(th), thumb_len=np.array([len(t) for t in th]),
                 note='per-token activation (theta*) of the full pass on the canonical frame; 8 top videos x 1 s clip '
                      '(5 frames) centred on the peak of the smoothed codes; thumbs = 256 px JPEG of the canonical '
                      'frame at each clip peak (concatenated, lengths thumb_len)')
        index[j] = sorted(why.get(j, []))
    tag = f'_{args.shard}of{args.nshards}' if args.nshards > 1 else ''
    (outd / f'index{tag}.json').write_text(json.dumps({'sae': name, 'series': sc['name'], 'neurons': index,
                                                       'why_all': {int(k): sorted(v) for k, v in why.items()}}, indent=1))
    log(f'{len(meta)} neurons -> {outd}')


def cmd_health(args):
    """Per series of a domain: held-out (eval-video) FVE and token L0 of the deployed SAE on that series' pool tokens,
    dead / alive neurons, and the overlap of the neurons each series uses (fires on >= 0.1% of its eval tokens)."""
    import torch
    from src.eci.v2 import sae as SA
    c = RUN.dcfg(args.domain)
    b = json.loads((OUT / args.domain / 'best.json').read_text())
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sae, norm = SA.load(Path(b['dir']) / 'sae_theta.pt', dev)
    out, used = {}, {}
    for sc in c['series']:
        St = RUN.Store(RUN.pool_dir(c, sc), sc['S'] // 16)
        sel = St.F.role.values == 'eval'
        X = St.inputs(c['spec']['context_window'], sel, dev).float()
        with torch.no_grad():
            xn = norm(X)
            z = sae.encode(xn, mode='threshold')
            rec = sae.decode(z)
            fve = 1 - ((xn - rec) ** 2).sum() / ((xn - xn.mean(0)) ** 2).sum()
        rate = (z > 0).float().mean(0).cpu().numpy()
        used[sc['name']] = set(np.flatnonzero(rate >= 0.001).tolist())
        out[sc['name']] = {'eval_tokens': int(len(X)), 'fve': float(fve), 'token_l0': float((z > 0).sum(1).float().mean()),
                           'neurons_used': len(used[sc['name']])}
    names = list(used)
    if len(names) > 1:
        inter = set.intersection(*used.values())
        out['shared_neurons'] = len(inter)
        out['only'] = {n: len(used[n] - set.union(*[used[m] for m in names if m != n])) for n in names}
    (OUT / args.domain / 'health.json').write_text(json.dumps(out, indent=1))
    log(json.dumps(out))
