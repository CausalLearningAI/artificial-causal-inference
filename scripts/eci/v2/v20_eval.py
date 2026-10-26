"""
ECI v2.0: labelled alignment of the deliverable and the PRE-REGISTERED 5 fps vs 1 fps dictionary comparison.

==================================================================================================================
PRE-REGISTRATION (written 2026-10-26 BEFORE any 5 fps SAE was scored on any label or by the capacity suite; the user's
rule of 2026-10-25, relayed by the coordinator. Known when writing: the PASS A deliverable = V2F_N2 s900 codes; the
V2F_N2 s900-919 scores of the evaluation audit (results/vision/eci_v2_eval/) exist and were not re-read for this.)
QUESTION: does training the v2.0 dictionary on 5 fps tokens of every video (V20F5; mice v1 + v2, ants v2 + v3 + v4,
    frogs juveniles + tadpoles; PASS A training store = a fixed random 20% of each frame's N2 tokens; v20.py train5:
    the same recipe, 2048 (128/256/512/2048), k16, 30k x 4096, tokens per role drawn uniformly up to 24M / 2M / 2M)
    lose alignment vs the 1 fps dictionary V2F_N2 (V2-full 'full1' store)?
ARMS: V2F_N2 seeds 900-904 vs V20F5 seeds 950-954 (5 vs 5). Frogs: v20 1 fps frogs SAE seeds 900-902 vs V20F5 950-954.
FRAMES / SCORER / READOUT: exactly those of the evaluation audit (scripts/eci/v2_eval/eval.py: frame tables, splits
    atlas + rnd0-4 both directions, neuron eligibility, score_cells kernels): mice = the 144 x 600 contiguous 5 fps
    probe windows, readout = 0.6 s centred mean (eval 'smooth'); ants = every labelled frame on the 1 fps grid, raw frame
    max (eval 'raw'). Codes of every SAE from the same N2 tokens (mice v2_prob dense store keep('N2'); ants full1 N2 arm).
METRICS: PRIMARY cross-fitted AUROC of the best single neuron; secondary nAP and F1max (same kernels); every labelled
    behaviour row (all (series, behaviour) cells that eval.py scores); the domain mean = mean over the cells that pass
    eval.py's row eligibility (>= 20 bouts and >= 3 positive videos in each atlas half), per seed.
CAPACITY (label-free, secondary, not decisive for mice / ants): distinct behaviour neurons @256 of the behaviour suite
    (scripts/eci/v2_behav/behav.py, amendment-2 checks C1-C8, its probe set), V2F_N2 900-904 vs V20F5 950-954.
RULE (per domain): d = mean_seeds(V20F5 domain-mean AUROC) - mean_seeds(V2F_N2 ...), Welch 95% CI. 5 fps is ADOPTED
    unless it is worse, i.e. unless the whole CI lies below -0.01 (upper bound < -0.01). Reported too: the
    non-inferiority reading (lower bound >= -0.01). Frogs (no labels): adopted unless capacity@256 is worse (Welch CI
    upper bound < 0). PASS B (re-encode with the deployment seed of V20F5 = highest held-out FVE) runs only for adopted
    domains and only if the measured cost keeps the rollout <= 240 GPU-h.
==================================================================================================================

Steps (scripts/eci/v2/v20.sh with SCRIPT=v20_eval.py is NOT used; run: python -u scripts/eci/v2/v20_eval.py <step>):
    codes  (GPU) --dom mice|ants: V20F5 seeds -> BEE/eval/<dom>/V20F5_s<seed>.npy (eval frame order)
    score  (CPU) --dom mice|ants: compare arms + the deliverable readouts (PASS A s900: raw / smooth / HMM P(on) read
           from the full 5 fps codes sets at the eval frames) -> OUT/eval/<dom>/*.json, OUT/eval/<dom>_summary.json
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2_eval', 'scripts/eci/v2_prob', 'scripts/eci/v2_behav', 'scripts/eci/v2'):
    sys.path.insert(0, str(REPO / p))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
BEE = EC / 'v2_0' / 'eval'
OUT = REPO / 'results/vision/eci_v2_0/eval'
V2F_SEEDS, F5_SEEDS = (900, 901, 902, 903, 904), (950, 951, 952, 953, 954)


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def f5_path(dom, s):
    return EC / f'v2_0/{dom}/sae5/v20_{dom}_N2f5_s{s}/sae_theta.pt'


def cmd_codes(a):
    import torch
    from src.eci.sae import load_sae
    import eval as EV
    dev = torch.device('cuda')
    dom = a.dom
    od = BEE / dom
    od.mkdir(parents=True, exist_ok=True)
    if dom == 'mice':
        import study as ST
        BH = ST.BH
        Pr = pd.read_parquet(ST.OUT / dom / 'probe.parquet')
        key = lambda F_: pd.MultiIndex.from_arrays([F_.vid.values.astype(int), F_.k.values.astype(int)])  # noqa
        pidx = pd.Series(np.arange(len(Pr)), index=key(Pr))
        St = BH.Store(dom, 'v2', 'dense', 1024)
        fidx = pidx.reindex(key(St.F)).values
        assert not np.isnan(fidx).any() and len(np.unique(fidx)) == len(Pr)
        fidx = fidx.astype(np.int64)
        keep = St.keep('N2')
        saes = {f'V20F5_s{s}': load_sae(f5_path(dom, s), dev)[:2] for s in F5_SEEDS}
        saes['V2F_N2_s900_check'] = load_sae(EC / 'v2_diag/mice/sae/V2F_N2_s900/sae_theta.pt', dev)[:2]
        X = {n: torch.zeros((len(Pr), 2048), dtype=torch.float16, device=dev) for n in saes}
        for f0 in range(0, St.nf, 400):
            f1 = min(St.nf, f0 + 400)
            a_, b_ = int(St.starts[f0]), int(St.starts[f1])
            if a_ == b_:
                continue
            z = torch.from_numpy(St.tok(a_, b_)).to(dev)
            fl = torch.from_numpy(St.tf[a_:b_] - f0).to(dev)
            rows = torch.from_numpy(fidx[f0:f1]).to(dev)
            sel = torch.from_numpy(keep[a_:b_]).to(dev)
            for n, (sae, norm) in saes.items():
                with torch.no_grad():
                    c = sae.encode(norm(z[sel].float()), mode='threshold').float()
                    fm = torch.zeros((f1 - f0, 2048), device=dev).index_reduce_(0, fl[sel], c, 'amax', include_self=True)
                    X[n][rows] = fm.half()
        ref = np.load(EC / 'v2_prob/mice/codes/V2F_N2_s900.npy')
        chk = X.pop('V2F_N2_s900_check').cpu().numpy()
        ident = {'V2F_N2_s900_vs_stored_max_abs': float(np.abs(chk.astype(np.float32) - ref.astype(np.float32)).max())}
        for n, x in X.items():
            np.save(od / f'{n}.npy', x.cpu().numpy())
    else:
        sys.path.insert(0, str(REPO / 'scripts/eci/v2_diag'))
        import diag as D
        T2 = D.T2
        F = pd.read_parquet(EV.BEE / 'ants' / 'frames.parquet')
        old = T2.BEE
        T2.BEE = D.BEE
        T2.KINDS['ants'].setdefault('full1', {'series': D.F1_SERIES['ants'][0]})
        try:
            U = T2.Union('ants', 'full1')
        finally:
            T2.BEE = old
        fsel = np.full(len(U.F), -1, np.int64)
        fsel[F.f1_idx.values] = np.arange(len(F))
        m = U.arm('N2') & (fsel[U.tf] >= 0)
        tok = U.tokens(m)
        tfr = fsel[U.tf[m]]
        T = torch.from_numpy(tok).to(dev)
        tg = torch.from_numpy(tfr).to(dev)
        names = {f'V20F5_s{s}': f5_path(dom, s) for s in F5_SEEDS}
        names['V2F_N2_s900_check'] = EC / 'v2_diag/ants/sae/V2F_N2_s900/sae_theta.pt'
        ident = {}
        for n, p in names.items():
            sae, norm, _ = load_sae(p, dev)
            X = torch.zeros((len(F), 2048), device=dev)
            with torch.no_grad():
                for c0 in range(0, len(T), 400_000):
                    c = sae.encode(norm(T[c0:c0 + 400_000].float()), mode='threshold').float()
                    X.index_reduce_(0, tg[c0:c0 + 400_000], c, 'amax', include_self=True)
            x = X.half().cpu().numpy()
            if n.endswith('_check'):
                ref = np.load(EV.BEE / 'ants/codes/V2F_N2_s900.npy')
                ident['V2F_N2_s900_vs_stored_max_abs'] = float(np.abs(x.astype(np.float32) - ref.astype(np.float32)).max())
            else:
                np.save(od / f'{n}.npy', x)
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / f'{dom}_codes_identity.json').write_text(json.dumps(ident, indent=1))
    log(f'{dom}: codes saved; identity {ident}')


def deliverable_codes(dom, F, suffix):
    """PASS A codes set <sae><suffix> (full 5 fps video sequences) read at the eval frames (obs, k)."""
    sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
    import v20 as V
    c = V.dcfg(dom)
    name = V.R.best_name(c)
    X = np.zeros((len(F), 2048), np.float32)
    have = np.zeros(len(F), bool)
    for sc in c['series']:
        P = json.loads((REPO / sc['plan']).read_text())
        lay = {v['obs']: v['lay_lo'] for v in P['videos']}
        m = F.obs.map(lay).notna().values
        if dom == 'mice':
            m &= F.experiment.values == 'v1' if sc['name'] == 'v1' else F.experiment.values == sc['name']
        if not m.any():
            continue
        rows = (F.obs[m].map(lay).values + F.k[m].values).astype(np.int64)
        mx = np.load(V.R.codes_dir(c, sc, name + suffix) / 'codes_max.npy', mmap_mode='r')
        o = np.argsort(rows)
        Y = np.empty((m.sum(), 2048), np.float32)
        Y[o] = np.asarray(mx[rows[o]], np.float32)
        X[m] = Y
        have[m] = True
    return X, have


def domain_mean(res, elig):
    cells = [c for c in res['cells'] if elig.get(c, {}).get('eligible')]
    out = {}
    for M in ('AUROC', 'nAP', 'F1max'):
        v = [res['cells'][c][M] for c in cells if res['cells'][c].get(M) is not None]
        out[M] = float(np.mean(v)) if v else None
    out['n_cells'] = len(cells)
    return out


def cmd_score(a):
    import analysis as AN
    import eval as EV
    dom = a.dom
    F = EV.frame_table(dom)
    SP = AN.splits(F.unit.values, EV.SPLITS)
    elig = EV.row_counts(dom, F, SP)
    starts = AN.seq_starts(F)
    od = OUT / dom
    od.mkdir(parents=True, exist_ok=True)
    dicts = {}
    for s in V2F_SEEDS:
        p = (EC / f'v2_prob/mice/codes/V2F_N2_s{s}.npy') if dom == 'mice' else (EV.BEE / f'ants/codes/V2F_N2_s{s}.npy')
        dicts[f'V2F_N2_s{s}'] = ('cmp', p)
    for s in F5_SEEDS:
        dicts[f'V20F5_s{s}'] = ('cmp', BEE / dom / f'V20F5_s{s}.npy')
    for suf in ('_raw', '_smooth', '_hmm'):
        dicts[f'deliv{suf}'] = ('deliv', suf)
    summ = {'eligibility': elig, 'dicts': {}}
    for n, (kind, p) in dicts.items():
        f = od / f'{n}.json'
        if f.exists() and not a.force:
            res = json.loads(f.read_text())
        else:
            if kind == 'cmp':
                if not Path(p).exists():
                    log(f'{n}: missing {p}')
                    continue
                X = np.load(p, mmap_mode='r')
                X = np.asarray(X[F.code_row.values] if (dom == 'mice' and n.startswith('V2F')) else X, np.float32)
                if dom == 'mice' and n.startswith('V20F5'):
                    X = X[F.code_row.values]
                use = np.ones(len(F), bool)
                if dom == 'mice':
                    X = AN.smooth_seq(X, starts)
            else:
                X, use = deliverable_codes(dom, F, p)
            res = {'dict': n, 'cells': EV.score_cells(F, X, use, SP), 'n_frames_used': int(use.sum())}
            f.write_text(json.dumps(res))
        summ['dicts'][n] = {'domain_mean': domain_mean(res, elig),
                            'cells': {c: {M: v.get(M) for M in ('AUROC', 'nAP', 'F1max', 'base_rate')}
                                      for c, v in res['cells'].items()}}
        log(f'{dom} {n}: {summ["dicts"][n]["domain_mean"]}')
    cmp = {}
    for M in ('AUROC', 'nAP', 'F1max'):
        A_ = [summ['dicts'][f'V20F5_s{s}']['domain_mean'][M] for s in F5_SEEDS if f'V20F5_s{s}' in summ['dicts']]
        B_ = [summ['dicts'][f'V2F_N2_s{s}']['domain_mean'][M] for s in V2F_SEEDS if f'V2F_N2_s{s}' in summ['dicts']]
        if len(A_) >= 2 and len(B_) >= 2:
            d, lo, hi = EV.welch(np.array(A_), np.array(B_))
            cmp[M] = {'V20F5': A_, 'V2F_N2': B_, 'diff': d, 'ci': [lo, hi]}
    if 'AUROC' in cmp:
        cmp['adopt_5fps'] = bool(cmp['AUROC']['ci'][1] >= -0.01)
        cmp['noninferior_reading'] = bool(cmp['AUROC']['ci'][0] >= -0.01)
    summ['compare'] = cmp
    (OUT / f'{dom}_summary.json').write_text(json.dumps(summ, indent=1, default=float))
    log(f'{dom} compare: {json.dumps(cmp, default=float)}')


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('codes', 'score'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    {'codes': cmd_codes, 'score': cmd_score}[a.step](a)


if __name__ == '__main__':
    main()
