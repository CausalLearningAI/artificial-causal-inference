"""
ECI v0 / v1 / v2 at a glance: an AUDIT of the labelled evaluation with the ALREADY-CHOSEN metrics. Nothing new is
pre-registered here: the primary metric (cross-fitted AUROC of the best single neuron), the secondaries (nAP, best-
threshold F1) and the legacy R~ come from the v2_prob metric audit (scripts/eci/v2_prob/analysis.py, decision.json);
the neuron eligibility (s > 0 on >= 0.2% of the cell's frames), the splits (atlas + rnd0-4, both directions) and the
metric kernels are imported from that file unchanged.

FRAME SETS (one per domain; every model is scored on exactly the same frames of a row)
  mice  the v2_prob Tier B probe windows: 144 annotated videos x 600 contiguous 5 fps frames (2 min, centred in the
        atlas window). Codes of v1 s0 (deployed) + v1 retrained s1-10 and V2F_N2 s900-919 already exist
        (eci_codes/v2_prob/mice/codes). v0 has codes only on its 2 fps grid (k % 5 in {0, 2}): its column uses that
        subset of the same windows; v1 / v2 are also reported on that subset (matched-frame check).
  ants  every LABELLED frame on the V2-full 1 fps grid (k % 5 == 2; the 'full1' token store): v2 44 videos, v3 107,
        v4 40, v4_infested 21, 600 frames each (= the whole labelled 10 min at 1 fps). Reason: the Tier B probe windows
        (v2 2 min, v3 1 min, no v4) hold too few lid bouts (v3 on-lid 6 bouts) and no v4 (coverage.md). V2F_N2 codes
        are computed here from the full1 N2 tokens; v1 codes from v1's exact token pipeline on the same frames
        (scripts/eci/v2_diag/matched.py v1tok; identity check vs the stored deployed codes_max); v0 from its stored
        2 fps frame codes (all these frames are on v0's grid).
READOUT per model: v1 = its deployed readout, the raw frame max of the threshold-mode code (codes_max); v2 (V2F_N2) =
  the 0.6 s centred mean of the frame max (src/eci/v2/readout.smooth, h = 1) where 5 fps frames are contiguous (mice);
  at 1 fps (ants) no smoothing is possible, so v2 is the raw frame max there (as release_ab); v0 = its mean-pooled
  token code (2 fps, raw). Mice also reports v1 smoothed and v2 raw (sensitivity).
ELIGIBILITY of a behaviour row (stated by the coordinator; it did not exist in code before, see coverage.md): >= 20
  bouts AND >= 3 videos with a positive frame in EACH half of the atlas split, counted on the row's scored frames.
  Rows below it are scored and listed separately, flagged.
NULL: label vectors permuted across videos of the same series (all windows have 600 frames), 3 permutations, for
  v1 s0, V2F_N2 deployment seed and v0.

Steps (scripts/eci/v2_eval/job.sh, SCRIPT=eval.py, EXTRA = arguments):
  frames                (CPU) ants frame table -> BEE/ants/frames.parquet; mice -> BEE/mice/frames.parquet
  codes_ants            (GPU) V2F_N2 s900-932 (full1 N2 tokens) + v1 s0-10 (v1tok) -> BEE/ants/codes/<dict>.npy
  score --dom D --dicts a,b,...   (CPU) -> OUT/<D>/<dict>__<variant>.json
  report                (CPU) -> OUT/table.md, table.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci/v2_prob'))
import analysis as AN  # noqa: E402  (metric kernels, splits, smoothing, eligibility of the metric audit)

OUT = REPO / 'results/vision/eci_v2_eval'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_eval')
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
V0 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_v0')
SPLITS = ['atlas'] + [f'rnd{s}' for s in range(5)]
MET = ('Rt', 'nAP', 'AUROC', 'F1max')  # column index in AN.metrics_cols output: 0, 1, 2, 3
BEH = {'ants': {'Y2F': ['Y_Y2F'], 'B2F': ['Y_B2F'], 'groom_any': ['Y_Y2F', 'Y_B2F'], 'FOL': ['Y_FOL'],
                'YOL': ['Y_YOL'], 'BOL': ['Y_BOL'], 'onlid_any': ['Y_YOL', 'Y_BOL'],
                'lid_any3': ['Y_FOL', 'Y_YOL', 'Y_BOL']},
       'mice': {'nn': ['Y_nn'], 'np': ['Y_np'], 'nose_nose': ['Y_nn', 'Y_np'], 'nt': ['Y_nt']}}
SER = {'ants': ('v2', 'v3', 'v4', 'v4_infested'), 'mice': ('v1',)}
ANN = {'eci30': 'dataset/ants/eci30/annotations.csv', 'c1v4': 'dataset/ants/eci_c1v4_noday/annotations.csv'}
V1CODES = {'eci30': 'dataset/ants/eci30/codes/matryoshka_btk_1024_k16_antsfg_s0/codes_max.npy',
           'c1v4': 'dataset/ants/eci_c1v4_noday/codes/matryoshka_btk_1024_k16_antsfg_s0/codes_max.npy'}
V1SAE = {'ants': 'dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0/sae.pt',
         'mice': 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448al_s0/sae.pt'}
V2SEEDS = {'mice': tuple(range(900, 920)), 'ants': tuple(range(900, 933))}
DEPLOY = 900  # V2F decide step: highest held-out mice FVE among 900-902 (diag.confirm_arm); one seed for both domains
NULL_DICTS = ('v1_s0', f'V2F_N2_s{DEPLOY}', 'v0')
N_PERM = 3


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def dicts_of(dom):
    return ['v0'] + [f'v1_s{s}' for s in range(11)] + [f'V2F_N2_s{s}' for s in V2SEEDS[dom]]


# ---------------------------------------------------------------------------------------------- frames
def cmd_frames(args):
    BEE.mkdir(parents=True, exist_ok=True)
    cols = sorted({c for v in BEH['ants'].values() for c in v})
    # ants: full1 1 fps frames x labelled rows
    fs = [pd.read_parquet(p / 'frames.parquet', columns=['obs', 'k', 'series'])
          for p in sorted((EC / 'v2_diag/ants/full1').glob('s*'), key=lambda p: int(p.name[1:]))]
    F1 = pd.concat(fs, ignore_index=True)
    F1['f1_idx'] = np.arange(len(F1))
    parts = []
    for src, f in ANN.items():
        a = pd.read_csv(REPO / f, dtype={'observation_id': str},
                        usecols=['observation_id', 'frame_idx', 'experiment', 'frame_path'] + cols)
        a['src'], a['row'] = src, np.arange(len(a))
        a = a[a.Y_Y2F.notna() & a.experiment.isin(SER['ants'])]
        if src == 'eci30':
            a = a[a.experiment.isin(['v2', 'v3'])]
        else:
            a = a[a.experiment.isin(['v4', 'v4_infested'])]
        parts.append(a)
    A = pd.concat(parts, ignore_index=True)
    assert not A.duplicated(['observation_id', 'frame_idx']).any(), 'obs ids collide across experiments'
    F = A.merge(F1.rename(columns={'obs': 'observation_id', 'k': 'frame_idx'}), on=['observation_id', 'frame_idx'],
                how='inner')
    F = F.rename(columns={'observation_id': 'obs', 'frame_idx': 'k'}).sort_values(['experiment', 'obs', 'k'])
    F = F.reset_index(drop=True)
    n_per = F.groupby('obs').size()
    log(f'ants frames: {len(F):,}; per experiment {F.experiment.value_counts().to_dict()}; frames per video '
        f'{n_per.value_counts().to_dict()}; k % 5 {(F.k % 5).value_counts().to_dict()}; full1 series of v4_infested '
        f'{F[F.experiment == "v4_infested"].series.unique().tolist()}')
    (BEE / 'ants').mkdir(parents=True, exist_ok=True)
    F.to_parquet(BEE / 'ants' / 'frames.parquet')
    # mice: the Tier B probe windows
    P = pd.read_parquet(REPO / 'results/vision/eci_v2_prob/mice/probe.parquet')
    a = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv', usecols=['observation_id', 'frame_idx', 'Y_nn', 'Y_np',
                                                                        'Y_nt'])
    idx = pd.Series(np.arange(len(a)), index=pd.MultiIndex.from_arrays([a.observation_id, a.frame_idx]))
    r = idx.reindex(pd.MultiIndex.from_arrays([P.obs.astype(str), P.k.astype(int)])).values
    assert not np.isnan(r).any()
    M = a.iloc[r.astype(int)].rename(columns={'observation_id': 'obs', 'frame_idx': 'k'}).reset_index(drop=True)
    M['experiment'] = 'v1'
    M['row'] = r.astype(np.int64)
    M['code_row'] = np.arange(len(M))  # rows of eci_codes/v2_prob/mice/codes/<dict>.npy
    (BEE / 'mice').mkdir(parents=True, exist_ok=True)
    M.to_parquet(BEE / 'mice' / 'frames.parquet')
    log(f'mice frames: {len(M):,}, videos {M.obs.nunique()}')


# ---------------------------------------------------------------------------------------------- codes (ants, GPU)
def cmd_codes_ants(args):
    import torch
    from src.eci.sae import load_sae
    sys.path.insert(0, str(REPO / 'scripts/eci/v2_diag'))
    import diag as D
    T2, N, R4 = D.T2, D.N, D.R4
    dev = torch.device('cuda')
    F = pd.read_parquet(BEE / 'ants' / 'frames.parquet')
    od = BEE / 'ants' / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    meta = {}
    # ---- V2F_N2 on the full1 N2 tokens of these frames
    if not args.skip_v2:
        old = T2.BEE
        T2.BEE = D.BEE
        T2.KINDS['ants'].setdefault('full1', {'series': D.F1_SERIES['ants'][0]})
        try:
            t0 = time.time()
            U = T2.Union('ants', 'full1')
        finally:
            T2.BEE = old
        log(f'full1 staged: {len(U.F):,} frames, {len(U.tf):,} tokens, {time.time() - t0:.0f}s')
        fsel = np.full(len(U.F), -1, np.int64)
        fsel[F.f1_idx.values] = np.arange(len(F))
        chk = U.F.iloc[F.f1_idx.values]
        assert (chk.obs.values.astype(str) == F.obs.values).all() and (chk.k.values == F.k.values).all()
        m = U.arm('N2') & (fsel[U.tf] >= 0)
        tok = U.tokens(m)
        tfr = fsel[U.tf[m]]
        o = np.argsort(tfr, kind='stable')
        tok, tfr = tok[o], tfr[o]
        starts = np.r_[0, np.cumsum(np.bincount(tfr, minlength=len(F)))]
        meta['v2_tokens_per_frame'] = float(len(tfr) / len(F))
        meta['v2_frames_without_token'] = int((np.diff(starts) == 0).sum())
        log(f'N2 tokens of the scored frames: {len(tfr):,} ({meta["v2_tokens_per_frame"]:.1f} per frame)')
        T = torch.from_numpy(tok).to(dev)
        tf_g = torch.from_numpy(tfr).to(dev)
        for s in V2SEEDS['ants']:
            if (od / f'V2F_N2_s{s}.npy').exists():
                continue
            sae, norm, _ = load_sae(EC / f'v2_diag/ants/sae/V2F_N2_s{s}/sae_theta.pt', dev)
            X = torch.zeros((len(F), sae.n_latents), device=dev)
            with torch.no_grad():
                for a in range(0, len(T), 400_000):
                    c = sae.encode(norm(T[a:a + 400_000].float()), mode='threshold').float()
                    X.index_reduce_(0, tf_g[a:a + 400_000], c, 'amax', include_self=True)
            np.save(od / f'V2F_N2_s{s}.npy', X.half().cpu().numpy())
            log(f'V2F_N2_s{s}: active share {(X > 0).float().mean().item():.4f}; {D.gpu_line()}')
        del T, tf_g, U, tok
        torch.cuda.empty_cache()
    # ---- v1 (deployed s0 + retrained s1-10) on v1's own tokens of these frames (matched.py v1tok pipeline)
    if not args.skip_v1:
        from src.eci.foreground import RULES, FgBackgrounds, FgEncoder, encode_batch, obs_rows
        enc = FgEncoder('dinov2_base', dev)
        saes = {'v1_s0': load_sae(REPO / V1SAE['ants'], dev)[:2]}
        for s in range(1, 11):
            saes[f'v1_s{s}'] = load_sae(EC / f'v2_diag/v1retrain/ants/V1R_full_s{s}/sae.pt', dev)[:2]
        X = {n: np.zeros((len(F), 1024), np.float16) for n in saes}
        ntok = 0
        for src, g in F.groupby('src'):
            ann = REPO / ANN[src]
            fp = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
            fb = FgBackgrounds(REPO / 'dataset/ants/eci30/fg448/background', obs_rows(ann), RULES['ants'], dev,
                               align='none', cache=8)
            rows = g.row.values.astype(np.int64)
            ds = enc.dataset([str(REPO / 'dataset' / p) for p in fp[rows]], rows, None)
            dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, pin_memory=True, prefetch_factor=4,
                                             num_workers=max(2, int(os.environ.get('SLURM_CPUS_PER_TASK', 8)) - 2))
            gi, o, t0 = g.index.values, 0, time.time()
            for pix, grey, rr in dl:
                with torch.no_grad():
                    tok = encode_batch(enc.model, pix, dev)
                    mk, _ = fb.mask(tok, grey.to(dev), rr.numpy())
                    fi, pi = torch.nonzero(mk, as_tuple=True)
                    x = tok[fi, pi]
                    ntok += len(x)
                    for n_, (sae, norm) in saes.items():
                        z = sae.encode(norm(x), mode='threshold')
                        mx = torch.zeros(len(tok), z.shape[1], device=dev).index_reduce_(0, fi, z.float(), 'amax',
                                                                                         include_self=True)
                        X[n_][gi[o:o + len(tok)]] = mx.half().cpu().numpy()
                o += len(tok)
            assert o == len(g)
            ref = np.load(REPO / V1CODES[src], mmap_mode='r')
            rc = np.asarray(ref[np.sort(rows)])[np.argsort(np.argsort(rows))].astype(np.float32)
            mine = X['v1_s0'][gi].astype(np.float32)
            meta[f'identity_v1_s0_{src}'] = {'frames': int(len(g)), 'fire_agree': float(((mine > 0) == (rc > 0)).mean()),
                                             'mean_abs_diff': float(np.abs(mine - rc).mean()),
                                             'max_abs_diff': float(np.abs(mine - rc).max()), 'wall_s': time.time() - t0}
            log(f'v1 {src}: {json.dumps(meta[f"identity_v1_s0_{src}"])}; {D.gpu_line()}')
        meta['v1_tokens_per_frame'] = float(ntok / len(F))
        for n_, x in X.items():
            np.save(od / f'{n_}.npy', x)
        # the stored deployed codes themselves (the atlas input) as a separate dictionary 'v1_dep_stored'
        Y = np.zeros((len(F), 1024), np.float16)
        for src, g in F.groupby('src'):
            ref = np.load(REPO / V1CODES[src], mmap_mode='r')
            rows = g.row.values.astype(np.int64)
            Y[g.index.values] = np.asarray(ref[np.sort(rows)])[np.argsort(np.argsort(rows))]
        np.save(od / 'v1_dep_stored.npy', Y)
    (OUT / 'ants_codes_meta.json').write_text(json.dumps(meta, indent=1))
    log(json.dumps(meta))


# ---------------------------------------------------------------------------------------------- v0 codes
def v0_codes(dom, F):
    """dense (len(F), 4608) float32 v0 frame codes and a bool mask of the frames v0 has."""
    out = np.zeros((len(F), 4608), np.float32)
    have = np.zeros(len(F), bool)
    runs = {'mice': [('mice_v1', None)], 'ants': [('ants_v2', 'v2'), ('ants_v3', 'v3'), ('ants_v4', 'v4'),
                                                   ('ants_v4', 'v4_infested')]}[dom]
    for bd, ex in runs:
        fr = pd.read_csv(V0 / bd / 'frames.csv', dtype={'observation_id': str})
        if bd == 'ants_v2':
            k = np.round((fr['source_frame'].to_numpy() - 2) / 6).astype(int)
        else:
            k = fr['frame_idx'].to_numpy()
        idx = pd.Series(np.arange(len(fr)), index=pd.MultiIndex.from_arrays([fr.observation_id.to_numpy(), k]))
        idx = idx[~idx.index.duplicated()]
        sel = np.ones(len(F), bool) if ex is None else (F.experiment.values == ex)
        r = idx.reindex(pd.MultiIndex.from_arrays([F.obs.values[sel], F.k.values[sel]])).to_numpy()
        ok = np.isfinite(r)
        rows_F = np.flatnonzero(sel)[ok]
        r = r[ok].astype(np.int64)
        o = np.argsort(r)
        if bd == 'ants_v2':
            Xm = np.load(V0 / bd / 'frame_codes_meanpool.npy', mmap_mode='r')
            out[rows_F[o]] = np.asarray(Xm[r[o]], np.float32)
        else:
            ti = np.load(V0 / bd / 'frame_codes_top20_idx.npy', mmap_mode='r')
            tv = np.load(V0 / bd / 'frame_codes_top20_val.npy', mmap_mode='r')
            ii, vv = np.asarray(ti[r[o]], np.int64), np.asarray(tv[r[o]], np.float32)
            blk = np.zeros((len(o), 4608), np.float32)
            np.put_along_axis(blk, ii, np.maximum(vv, 0), 1)
            out[rows_F[o]] = blk
        have[rows_F] = True
    return out, have


# ---------------------------------------------------------------------------------------------- scoring
def frame_table(dom):
    F = pd.read_parquet(BEE / dom / 'frames.parquet')
    F['series'] = F.experiment
    F['unit'] = F.obs.str.rsplit('_', n=2).str[0] if dom == 'mice' else F.obs
    for b, cols in BEH[dom].items():
        F[f'y_{b}'] = (F[cols].fillna(0).to_numpy() > 0).any(1)
        F[f'v_{b}'] = F[cols].notna().all(1).to_numpy()
    return F


def load_dict(dom, dname, F):
    if dname == 'v0':
        return v0_codes(dom, F)
    if dom == 'mice':
        X = np.load(EC / 'v2_prob/mice/codes' / f'{dname}.npy', mmap_mode='r')
        X = np.asarray(X[F.code_row.values], np.float32)
    else:
        X = np.load(BEE / 'ants/codes' / f'{dname}.npy').astype(np.float32)
    return X, np.ones(len(F), bool)


def perm_labels(F, cols, rng):
    """permute whole label vectors across the videos of each series (every video has the same number of frames and
    its frames are contiguous in F)."""
    G = F.copy()
    for ser, g in F.groupby('series', sort=False):
        ix = g.index.values
        obs = g.obs.values
        nv = len(pd.unique(obs))
        assert len(ix) % nv == 0 and (obs.reshape(nv, -1) == obs.reshape(nv, -1)[:, :1]).all(), ('layout', ser)
        p = rng.permutation(nv)
        for c in cols:
            G.loc[ix, c] = g[c].values.reshape(nv, -1)[p].ravel()
    return G


def score_cells(F, X, use, units_split):
    """cross-fitted metrics of every (series, behaviour) cell. X (n, W) float32 (already read out), use = frames this
    variant scores. -> {cell: {...}}"""
    out = {}
    vcode = pd.factorize(F.obs.values)[0].astype(np.int64)
    nvid = int(vcode.max()) + 1
    starts = AN.seq_starts(F)
    for ser in F.series.unique():
        for b in [c[2:] for c in F.columns if c.startswith('y_')]:
            vmask = F[f'v_{b}'].values & (F.series.values == ser) & use
            if vmask.sum() == 0 or F[f'y_{b}'].values[vmask].sum() == 0:
                continue
            idx_c, st_c = AN.sub_starts(vmask, starts)
            yc = F[f'y_{b}'].values[idx_c]
            Xc0 = X[idx_c]
            rate = (Xc0 > 0).mean(0)
            elig = np.flatnonzero(rate >= AN.ELIG)
            if len(elig) == 0:
                continue
            Xc = np.asfortranarray(Xc0[:, elig])
            vc = vcode[idx_c]
            per = {}
            halves = {nm: np.array([units_split[nm][u] for u in F.unit.values[idx_c]]) for nm in SPLITS}
            for nm in SPLITS:
                for h in (0, 1):
                    sel = np.flatnonzero(halves[nm] == h)
                    if len(sel) == 0 or yc[sel].sum() == 0 or yc[sel].all():
                        per[(nm, h)] = None
                        continue
                    _, st_h = AN.sub_starts(halves[nm] == h, st_c)
                    per[(nm, h)] = AN.metrics_cols(Xc, sel.astype(np.int64), yc[sel], st_h, vc[sel], nvid)[0]
            cf = {M: [] for M in MET}
            pick = []
            for nm in SPLITS:
                for h in (0, 1):
                    A_, B_ = per[(nm, h)], per[(nm, 1 - h)]
                    if A_ is None or B_ is None:
                        continue
                    for mi, M in enumerate(MET):
                        col = A_[:, mi]
                        if not np.isfinite(col).any():
                            continue
                        q = int(np.nanargmax(col))
                        cf[M].append(float(B_[q, mi]))
                        if M == 'AUROC':
                            pick.append([nm, h, int(elig[q])])
            out[f'{ser}|{b}'] = {'n_frames': int(len(idx_c)), 'base_rate': float(yc.mean()), 'n_elig': int(len(elig)),
                                 'n_dirs': len(cf['AUROC']),
                                 **{M: (float(np.mean(v)) if v else None) for M, v in cf.items()},
                                 'AUROC_dirs': cf['AUROC'], 'AUROC_neurons': pick}
    return out


def cmd_score(args):
    dom = args.dom
    F = frame_table(dom)
    SP = AN.splits(F.unit.values, SPLITS)
    od = OUT / dom
    od.mkdir(parents=True, exist_ok=True)
    starts = AN.seq_starts(F)
    v0have = None
    for dname in args.dicts.split(','):
        t0 = time.time()
        X, have = load_dict(dom, dname, F)
        if dname == 'v0':
            v0have = have
        variants = {}
        if dom == 'mice' and dname != 'v0':
            S = AN.smooth_seq(X, starts)
            if v0have is None:
                v0have = v0_codes(dom, F)[1]
            variants = {'raw': (X, have), 'smooth': (S, have), 'raw_v0frames': (X, v0have),
                        'smooth_v0frames': (S, v0have)}
        else:
            variants = {'raw': (X, have)}
        for vn, (Xv, use) in variants.items():
            f = od / f'{dname}__{vn}.json'
            if f.exists() and not args.force:
                continue
            res = {'dict': dname, 'variant': vn, 'dom': dom, 'n_frames_used': int(use.sum()),
                   'cells': score_cells(F, Xv, use, SP)}
            f.write_text(json.dumps(res))
            log(f'{dom} {dname} {vn}: ' + ', '.join(f'{c} AUROC {v["AUROC"]:.3f}' for c, v in res['cells'].items()
                                                    if v['AUROC'] is not None) + f' ({time.time() - t0:.0f}s)')
        # label-shuffle null
        if dname in NULL_DICTS:
            vn = 'smooth' if (dom == 'mice' and dname.startswith('V2F')) else 'raw'
            Xv, use = variants[vn]
            f = od / f'{dname}__{vn}__null.json'
            if f.exists() and not args.force:
                continue
            cols = [c for c in F.columns if c.startswith('y_') or c.startswith('v_')]
            nulls = []
            for p in range(N_PERM):
                G = perm_labels(F, cols, np.random.default_rng(7000 + p))
                nulls.append(score_cells(G, Xv, use, SP))
            f.write_text(json.dumps({'dict': dname, 'variant': vn, 'perms': nulls}))
            log(f'{dom} {dname} null: ' + ', '.join(
                f'{c} AUROC {np.mean([n[c]["AUROC"] for n in nulls if c in n and n[c]["AUROC"] is not None]):.3f}'
                for c in nulls[0]))


# ---------------------------------------------------------------------------------------------- selftest of the scorer
def cmd_selftest(args):
    """the scorer must reproduce the v2_prob A3 cross-fitted values on the v2_prob Tier B frames (same codes, same
    smoothing for every dictionary), for the dictionaries given."""
    dom = args.dom
    od = OUT / 'selftest'
    od.mkdir(parents=True, exist_ok=True)
    F = AN.frame_table(dom, 'B')
    starts = AN.seq_starts(F)
    SP = AN.splits(F.unit.values, SPLITS)
    G = F[['obs', 'k', 'series', 'unit']].copy()
    for ser, lb in AN.CELLS[dom]:
        G[f'y_{ser}:{lb}'] = F[f'y_{lb}'].values
        G[f'v_{ser}:{lb}'] = F[f'v_{ser}_{lb}'].values
    res = {}
    for dname in args.dicts.split(','):
        X = np.asarray(np.load(EC / f'v2_prob/{dom}/codes/{dname}.npy', mmap_mode='r'), np.float32)
        S = AN.smooth_seq(X, starts)
        mine = score_cells(G, S, np.ones(len(G), bool), SP)
        z = json.loads((REPO / f'results/vision/eci_v2_prob/{dom}/tierB/{dname}.json').read_text())['ds']
        cmp = {}
        for ser, lb in AN.CELLS[dom]:
            cell = f'{ser}_{lb}'
            for M in MET:
                ref = float(np.mean([d['cf_value'] for d in z[cell][M]]))
                m = mine.get(f'{ser}|{ser}:{lb}', {}).get(M)
                cmp[f'{cell}|{M}'] = [ref, m]
        res[dname] = cmp
        log(f'selftest {dom} {dname}: ' + json.dumps({k: [round(a, 4), None if b is None else round(b, 4)]
                                                      for k, (a, b) in cmp.items() if k.endswith('AUROC')}))
    (od / f'{dom}.json').write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------------- report
def welch(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d = a.mean() - b.mean()
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    return [float(d), float(d - h), float(d + h)]


def row_counts(dom, F, SP):
    """bouts / positive videos per atlas half on the scored frames of every (series, behaviour)."""
    out = {}
    for ser in F.series.unique():
        for b in BEH[dom]:
            m = F[f'v_{b}'].values & (F.series.values == ser)
            g = F[m]
            y = g[f'y_{b}'].values
            o, k = g.obs.values, g.k.values
            step = 1 if dom == 'mice' else 5  # ants frames are 1 fps (k step 5): a gap of one grid step continues a bout
            new = y & np.r_[True, (o[1:] != o[:-1]) | (k[1:] != k[:-1] + step) | ~y[:-1]]
            h = np.array([SP['atlas'][u] for u in g.unit.values])
            r = {'frames': int(len(g)), 'pos': int(y.sum()), 'bouts': int(new.sum()),
                 'videos_pos': int(len(np.unique(o[y])))}
            for hh in (0, 1):
                r[f'h{hh}_bouts'] = int((new & (h == hh)).sum())
                r[f'h{hh}_videos_pos'] = int(len(np.unique(o[y & (h == hh)])))
            r['eligible'] = bool(min(r['h0_bouts'], r['h1_bouts']) >= 20 and min(r['h0_videos_pos'], r['h1_videos_pos']) >= 3)
            out[f'{ser}|{b}'] = r
    return out


def fmt(x, nd=3):
    return '-' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def md_table(J, M, eligible):
    L = [f'| domain | series | behaviour | scored frames (positive) | bouts half A / B | v0 {M} | v1 deployed {M} | '
         f'v1 {M} mean ± sd (n) | v2 {M} mean ± sd (n) | v2 deploy s{DEPLOY} {M} | v2 - v1 mean {M} [Welch 95% CI] |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for key, row in J['rows'].items():
        if row['counts']['eligible'] != eligible:
            continue
        dom, ser, b = key.split('|')
        c, r = row['counts'], row[M]
        d = r['v2_minus_v1mean']
        L.append(f'| {dom} | {ser} | {b} | {c["frames"]:,} ({c["pos"]:,}) | {c["h0_bouts"]} / {c["h1_bouts"]} | '
                 f'{fmt(r["v0"])} | {fmt(r["v1_dep"])} | {fmt(r["v1_mean"])} ± {fmt(r["v1_sd"])} ({r["v1_n"]}) | '
                 f'{fmt(r["v2_mean"])} ± {fmt(r["v2_sd"])} ({r["v2_n"]}) | {fmt(r["v2_dep"])} | '
                 + (f'{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]' if d else '-') + ' |')
    if eligible:
        for dom, dl in J['domain'].items():
            r = dl[M]
            d = r['v2_minus_v1mean']
            L.append(f'| **{dom} (mean of {len(dl["rows"])} rows)** | | | | | {fmt(r["v0"])} | {fmt(r["v1_dep"])} | '
                     f'{fmt(r["v1_mean"])} ± {fmt(r["v1_sd"])} ({r["v1_n"]}) | {fmt(r["v2_mean"])} ± {fmt(r["v2_sd"])} '
                     f'({r["v2_n"]}) | {fmt(r["v2_dep"])} | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] |')
    return L


def cmd_report(args):
    PRIMARY_V = {'mice': {'v0': 'raw', 'v1': 'raw', 'v2': 'smooth'}, 'ants': {'v0': 'raw', 'v1': 'raw', 'v2': 'raw'}}
    J = {'note': 'AUDIT with already-chosen metrics (scripts/eci/v2_prob decision.json: primary AUROC, secondary nAP, '
                 'F1max; R~ legacy). Nothing new is pre-registered.', 'rows': {}, 'domain': {}, 'checks': {}}
    for dom in ('mice', 'ants'):
        F = frame_table(dom)
        SP = AN.splits(F.unit.values, SPLITS)
        cnt = row_counts(dom, F, SP)
        R = {}
        for f in sorted((OUT / dom).glob('*.json')):
            if f.name.endswith('__null.json'):
                continue
            d = json.loads(f.read_text())
            R[(d['dict'], d['variant'])] = d['cells']
        v1n = [f'v1_s{s}' for s in range(11)]
        v2n = [f'V2F_N2_s{s}' for s in V2SEEDS[dom] if (f'V2F_N2_s{s}', PRIMARY_V[dom]['v2']) in R]
        pv = PRIMARY_V[dom]
        per_dict = {}  # (arm, dict) -> {cell: {metric: value}} for the domain line
        for cell, c in cnt.items():
            if c['pos'] == 0:
                continue
            row = {'counts': c, 'frames': 'mice Tier B probe windows (5 fps)' if dom == 'mice' else 'ants labelled 1 fps (full1)'}
            for M in MET:
                g = lambda dn, vn: R.get((dn, vn), {}).get(cell, {}).get(M)  # noqa: E731
                v0 = g('v0', 'raw')
                v1 = [g(n, pv['v1']) for n in v1n]
                v2 = [g(n, pv['v2']) for n in v2n]
                v1 = [x for x in v1 if x is not None]
                v2 = [x for x in v2 if x is not None]
                r = {'v0': v0, 'v1_dep': g('v1_s0', pv['v1']),
                     'v1_mean': float(np.mean(v1)) if v1 else None, 'v1_sd': float(np.std(v1, ddof=1)) if len(v1) > 1 else None,
                     'v1_n': len(v1), 'v2_mean': float(np.mean(v2)) if v2 else None,
                     'v2_sd': float(np.std(v2, ddof=1)) if len(v2) > 1 else None, 'v2_n': len(v2),
                     'v2_dep': g(f'V2F_N2_s{DEPLOY}', pv['v2']),
                     'v2_minus_v1mean': welch(v2, v1) if len(v1) > 1 and len(v2) > 1 else None}
                if dom == 'mice':
                    r['sens_v1_smooth_mean'] = float(np.nanmean([g(n, 'smooth') for n in v1n if g(n, 'smooth') is not None]))
                    r['sens_v2_raw_mean'] = float(np.nanmean([g(n, 'raw') for n in v2n if g(n, 'raw') is not None]))
                    r['v0frames_v1_mean'] = float(np.nanmean([g(n, 'raw_v0frames') for n in v1n if g(n, 'raw_v0frames') is not None]))
                    r['v0frames_v2_mean'] = float(np.nanmean([g(n, 'smooth_v0frames') for n in v2n if g(n, 'smooth_v0frames') is not None]))
                if dom == 'ants':
                    r['v1_dep_stored_codes'] = g('v1_dep_stored', 'raw')
                row[M] = r
                if c['eligible']:
                    for arm, names, vn in (('v0', ['v0'], 'raw'), ('v1', v1n, pv['v1']), ('v2', v2n, pv['v2'])):
                        for n in names:
                            x = g(n, vn)
                            if x is not None:
                                per_dict.setdefault((arm, n), {}).setdefault(M, {})[cell] = x
            J['rows'][f'{dom}|{cell}'] = row
        # domain line: per dictionary, mean over the eligible rows; then mean / sd over dictionaries, Welch v2 - v1
        el = [c for c, v in cnt.items() if v['eligible']]
        dl = {'rows': el}
        for M in MET:
            vals = {arm: [np.mean([d[M][c] for c in el]) for (a, n), d in per_dict.items() if a == arm
                          and M in d and all(c in d[M] for c in el)] for arm in ('v0', 'v1', 'v2')}
            dep = {arm: [np.mean([d[M][c] for c in el]) for (a, n), d in per_dict.items() if a == arm and n == nm
                         and all(c in d[M] for c in el)] for arm, nm in (('v1', 'v1_s0'), ('v2', f'V2F_N2_s{DEPLOY}'))}
            dl[M] = {'v0': vals['v0'][0] if vals['v0'] else None,
                     'v1_dep': dep['v1'][0] if dep['v1'] else None,
                     'v1_mean': float(np.mean(vals['v1'])), 'v1_sd': float(np.std(vals['v1'], ddof=1)), 'v1_n': len(vals['v1']),
                     'v2_mean': float(np.mean(vals['v2'])), 'v2_sd': float(np.std(vals['v2'], ddof=1)), 'v2_n': len(vals['v2']),
                     'v2_dep': dep['v2'][0] if dep['v2'] else None, 'v2_minus_v1mean': welch(vals['v2'], vals['v1'])}
        J['domain'][dom] = dl
        # null
        nl = {}
        for f in sorted((OUT / dom).glob('*__null.json')):
            d = json.loads(f.read_text())
            for cell in d['perms'][0]:
                for M in ('AUROC', 'Rt', 'nAP'):
                    xs = [p[cell][M] for p in d['perms'] if cell in p and p[cell][M] is not None]
                    nl.setdefault(f'{d["dict"]}|{cell}', {})[M] = [float(np.mean(xs)), float(np.min(xs)), float(np.max(xs))] if xs else None
        J['checks'][f'null_{dom}'] = nl
    for f in ('ants_codes_meta.json',):
        if (OUT / f).exists():
            J['checks'][f] = json.loads((OUT / f).read_text())
    for dom in ('mice', 'ants'):
        f = OUT / 'selftest' / f'{dom}.json'
        if f.exists():
            st = json.loads(f.read_text())
            J['checks'][f'selftest_{dom}_max_abs_diff_vs_v2prob'] = float(max(abs(a - b) for v in st.values() for a, b in v.values()))
    (OUT / 'table.json').write_text(json.dumps(J, indent=1))
    log('table.json written')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step')
    p.add_argument('--dom', default='')
    p.add_argument('--dicts', default='')
    p.add_argument('--force', action='store_true')
    p.add_argument('--skip_v1', action='store_true')
    p.add_argument('--skip_v2', action='store_true')
    a = p.parse_args()
    {'frames': cmd_frames, 'codes_ants': cmd_codes_ants, 'score': cmd_score, 'selftest': cmd_selftest, 'report': cmd_report}[a.step](a)


if __name__ == '__main__':
    main()
