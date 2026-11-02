"""
ECI v2.1 SMALL-SAE test: does v1's smaller SAE recipe close the ants top-1% precision gap to v1 without hurting mice?
Pre-registration: scripts/eci/v2/v21_small_prereg.md (committed before any score of this file existed).

Arms (same tokens, mask v5 = M_t, inputs, TokenNorm, data sample, calibration and readouts as S / V of v21_train.py; ONLY
the SAE size / schedule changes, to v1's recipe = rung D3r1 of v21_v1gap.py):
    Ssm  = S with SAE 1024 latents, prefixes 128/256/512/1024, k 16, 8,380 steps x 4096, lr 5e-4
    Vsm  = V with the same SAE
Seeds 1004-1006 (the fresh seeds of S / V), both domains.

Steps
  train --dom D --arm Ssm|Vsm   (GPU) 3 seeds -> BEE/<dom>/sae/<key>, OUT/train/<dom>_<key>.json (+ node / GPU)
  codes --dom D                 (GPU) frame max and top10 codes of the eval frames for the 6 small SAEs, plus V_s1004
                                recomputed through this code path and compared to its stored codes (identity of codes)
  cols --dom D --model KEY [--agg top10]   (CPU) the ladder scorer (v21_ladder.cmd_cols, P(on) HMM readout) into OUT/ladder
                                KEY = V_s1004 re-scores the stored V model: identity vs the stored score json (fail loud
                                inside cmd_cols) AND vs the ladder's own cols json
  boot --dom D                  (CPU) the ladder bootstrap with the SAME replicates as boot_<dom>.npz for the new keys (+
                                V_s1004 as a replicate identity check), merged with the stored ladder rows
  report                        (CPU) tables + pre-registered verdict -> OUT/report.{json,md}
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
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
import v21_ladder as L  # noqa: E402

T = L.T
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_1/small')
OUT = T.OUT / 'small'
LB0, LO0 = L.LB, L.LO  # the ladder's own rows (v1, S, V)
L.LB, L.LO = BEE / 'ladder', OUT / 'ladder'
RECIPE = dict(W=1024, prefixes=[128, 256, 512, 1024], k=16, steps=8380, bs=4096, lr=5e-4)
BASE = {'Ssm': 'S', 'Vsm': 'V'}
SEEDS = (1004, 1005, 1006)
log = L.log


def devinfo():
    import socket
    import torch
    return {'node': socket.gethostname(), 'gpu': torch.cuda.get_device_name(0),
            'gpu_mem_gb': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}


# ---------------------------------------------------------------------------------------------- train
def cmd_train(a):
    import torch
    import v20 as V
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    od = T.BEE / a.dom / 'data'
    X = {}
    for r in T.CAP:  # exactly v21_train.cmd_train's inputs for arm S / V
        B_ = np.load(od / f'{r}_B.npy', mmap_mode='r')
        if BASE[a.arm] == 'V':
            X[r] = torch.from_numpy(np.asarray(B_[:, :1024], np.float32) + np.load(od / f'{r}_C.npy').astype(np.float32)).half()
        else:
            X[r] = torch.from_numpy(np.ascontiguousarray(B_[:, :1024]))
    cfg = {**V.R.spec()['sae'], **RECIPE, 'context_window': 0, 'arm': a.arm, 'base_arm': BASE[a.arm], 'store': T.VER}
    (OUT / 'train').mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        key = f'{a.arm}_s{s}'
        rec = SA.train(cfg, s, V.HostTok(X['train'], dev), X['calib'], X['eval'], BEE / a.dom / 'sae' / key, dev)
        rec.update(devinfo())
        (OUT / 'train' / f'{a.dom}_{key}.json').write_text(json.dumps(rec, indent=1, default=float))
        log(f'{a.dom} {key}: eval FVE {rec["eval_fve"]:.5f} L0 {rec["eval_l0"]:.2f} dead {rec["dead_at_end_train"]} '
            f'({rec["train_time_s"]:.0f}s, {rec["gpu"]})')


# ---------------------------------------------------------------------------------------------- codes
def cmd_codes(a):
    """v21_train.cmd_codes (eval set) for SAEs of any width: frame max (index_reduce amax, as there) and top10 (mean of
    the 10 largest token codes, as there) of the M_t (bit0) tokens of every 5 fps frame of the eval spans."""
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    dom = a.dom
    P = torch.load(T.BEE / dom / 'data' / 'pca.pt', map_location=dev)
    saes = {f'{arm}_s{s}': (SA.load(BEE / dom / 'sae' / f'{arm}_s{s}' / 'sae_theta.pt', dev), BASE[arm])
            for arm in BASE for s in SEEDS}
    saes['V_s1004'] = (SA.load(T.BEE / dom / 'sae' / 'V_s1004' / 'sae_theta.pt', dev), 'V')  # identity of the code path
    od = BEE / dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Fs = T.frame_sets(dom)['eval']
    spans = [(ser, obs, int(g.k.min()), int(g.k.max()) + 1) for (ser, obs), g in Fs.groupby(['series', 'obs'], sort=False)]
    R = pd.DataFrame([(ser, obs, k) for ser, obs, k0, k1 in spans for k in range(k0, k1)], columns=['series', 'obs', 'k'])
    Wd = {k: m[0][0].n_latents for k, m in saes.items()}
    Xc = {(k, g_): np.zeros((len(R), Wd[k]), np.float16) for k in saes for g_ in ('max', 'top10')}
    base = 0
    for si, (ser, obs, k0, k1) in enumerate(spans):
        V_ = T.Video(dom, ser, obs)
        kpos = {int(k): i for i, k in enumerate(V_.k)}
        i0, i1 = kpos[k0], kpos[k1 - 1] + 1
        for b0 in range(i0, i1, 600):
            W = V_.window(b0, min(i1, b0 + 600))
            c, nb = T.centre_rows(W, 1)
            if not len(c):
                continue
            x, cd, rs, cnt = T.raw_feats(W['tok'], c, nb, dev)
            f = {'S': x, 'V': x + cd}  # = v21_train.arm_inputs 'S' / 'V'
            fl = torch.from_numpy(W['fi'][c] - i0).to(dev)
            nfr = min(i1, b0 + 600) - b0
            lf = fl - (b0 - i0)
            o_ = torch.argsort(lf, stable=True)
            lfs = lf[o_]
            cnt_ = torch.bincount(lfs, minlength=nfr)
            st_ = torch.cumsum(cnt_, 0) - cnt_
            rk = torch.arange(len(lfs), device=dev) - st_[lfs]
            cf = cnt_.clamp(min=1).float()[:, None]
            for k, ((sae, norm), arm) in saes.items():
                with torch.no_grad():
                    cc = sae.encode(norm(f[arm]), mode='threshold').float()
                m = torch.zeros((i1 - i0, Wd[k]), device=dev).index_reduce_(0, fl, cc, 'amax', include_self=True)
                Xc[(k, 'max')][base + b0 - i0: base + b0 - i0 + nfr] = m[b0 - i0: b0 - i0 + nfr].half().cpu().numpy()
                D = torch.zeros((nfr, int(cnt_.max()), Wd[k]), device=dev)
                D[lfs, rk] = cc[o_]
                v_ = D.topk(min(10, D.shape[1]), dim=1).values.sum(1) / cf.clamp(max=10)
                Xc[(k, 'top10')][base + b0 - i0: base + b0 - i0 + nfr] = v_.half().cpu().numpy()
                del D
        base += i1 - i0
        if si % 50 == 0:
            log(f'{dom} codes span {si}/{len(spans)} ({time.time() - t0:.0f}s)')
    R.to_parquet(od / 'eval_rows.parquet')
    # identity of the code path: recomputed V_s1004 vs its stored codes (same rows required)
    R0 = pd.read_parquet(T.BEE / dom / 'codes' / 'eval_rows.parquet')
    same = bool(len(R0) == len(R) and (R0.values == R.values).all())
    idc = {'rows_identical_to_stored': same}
    for g_, fn in ((('max', 'V_s1004_eval.npy'), ('top10', 'V_s1004_eval_top10.npy')) if same else ()):
        ref = np.load(T.BEE / dom / 'codes' / fn, mmap_mode='r')
        mx, nd = 0.0, 0
        for b in range(0, len(R), 50_000):
            d = np.abs(Xc[('V_s1004', g_)][b:b + 50_000].astype(np.float32) - np.asarray(ref[b:b + 50_000], np.float32))
            mx, nd = max(mx, float(d.max())), nd + int((d > 0).sum())
        idc[g_] = {'max_abs_diff': mx, 'n_diff_entries': nd, 'n_entries': int(len(R) * ref.shape[1])}
    for (k, g_), x in Xc.items():
        if k != 'V_s1004':
            np.save(od / (f'{k}_eval.npy' if g_ == 'max' else f'{k}_eval_{g_}.npy'), x)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{dom}_codes_identity.json').write_text(json.dumps({'V_s1004_recomputed_vs_stored': idc, **devinfo()}, indent=1))
    log(f'{dom}: {len(R):,} frames, codes saved; V_s1004 code identity {idc} ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- cols
_readout0 = L.readout


def readout(dom, key, agg, F):
    if key.split('_s')[0] not in BASE:
        return _readout0(dom, key, agg, F)  # stored model (identity check vs the stored score json inside cmd_cols)
    b0, o0 = T.BEE, T.OUT
    T.BEE, T.OUT = BEE, OUT / 'no_score_json'  # codes of this file; no stored score json -> fresh model
    try:
        return _readout0(dom, key, agg, F)
    finally:
        T.BEE, T.OUT = b0, o0


def cmd_cols(a):
    L.readout = readout
    L.cmd_cols(argparse.Namespace(dom=a.dom, model=a.model, agg=a.agg or None))
    tag = f'{a.model}_{a.agg}' if a.agg else a.model
    if a.model.split('_s')[0] in BASE:
        return
    # second identity check: this cols json vs the ladder's own cols json of the same model (every cell, every metric)
    A = json.loads((L.LO / a.dom / f'{tag}.json').read_text())
    B = json.loads((LO0 / a.dom / f'{tag}.json').read_text())
    mx, n = 0.0, 0
    for part in [p for p in ('gate', 'focal') if p in B]:
        assert set(A[part]) == set(B[part]), part
        for c, r in B[part].items():
            for M in ('AUROC', 'nAP', 'top1_prec', 'F1max'):
                if r.get(M) is not None:
                    mx, n = max(mx, abs(A[part][c][M] - r[M])), n + 1
            assert A[part][c]['AUROC_neurons'] == r['AUROC_neurons'], (part, c)
    res = {'tag': tag, 'vs_stored_score_json': A['identity'], 'vs_ladder_cols_json': {'n': n, 'max_abs_diff': mx}}
    (OUT / f'{a.dom}_identity_{tag}.json').write_text(json.dumps(res, indent=1))
    log(f'IDENTITY {a.dom} {tag}: {res}')
    if mx > 1e-6:
        raise SystemExit(f'IDENTITY vs ladder cols FAILED {tag}: {mx}')


# ---------------------------------------------------------------------------------------------- boot
def new_keys(dom):
    return [f'{arm}_s{s}{g}' for arm in BASE for s in SEEDS for g in ('', '_top10')]


def cmd_boot(a):
    from multiprocessing import get_context
    dom = a.dom
    F, SP = L.frames(dom)
    rng = np.random.default_rng(L.SEED)  # identical replicates to the ladder's boot_<dom>.npz (v21_ladder.cmd_boot)
    Wser = {}
    for ser in sorted(F.series.unique()):
        units = np.array(sorted(set(F.unit.values[F.series.values == ser])))
        W = np.stack([np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)) for _ in range(L.NB)])
        Wser[ser] = (units, W.astype(np.float64))
    keys = [k for k in new_keys(dom) + ['V_s1004'] if (L.LO / dom / f'{k}.json').exists()]
    missing = [k for k in new_keys(dom) if k not in keys]
    if missing:
        raise SystemExit(f'missing cols for {missing}')
    parts = ['gate'] + (['focal'] if dom == 'ants' else [])
    res = {}
    with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        for tag, part, out in pool.imap_unordered(L.model_boot, [(dom, k, p, Wser) for k in keys for p in parts]):
            for c, m in out.items():
                if m is not None:
                    res[f'{tag}|{part}|{c}'] = m
            log(f'boot {dom} {tag} {part} done')
    mx = 0.0
    for k in keys:
        J = json.loads((L.LO / dom / f'{k}.json').read_text())
        for part in parts:
            for c, r in J[part].items():
                if f'{k}|{part}|{c}' in res:
                    for i, M in enumerate(L.MN):
                        if r[M] is not None:
                            mx = max(mx, abs(res[f'{k}|{part}|{c}'][i, 0] - r[M]))
    z0 = np.load(LB0 / f'boot_{dom}.npz')
    rep = max(float(np.nanmax(np.abs(res[x] - z0[x]))) for x in res if x.startswith('V_s1004|'))
    idt = {'weights1_vs_scorer_max_abs_diff': mx, 'V_s1004_replicates_vs_ladder_boot_max_abs_diff': rep}
    log(f'boot identity {dom}: {idt}')
    if mx > 1e-6 or rep > 1e-9:
        raise SystemExit(f'BOOT IDENTITY CHECK FAILED {idt}')
    np.savez_compressed(BEE / f'boot_new_{dom}.npz', **{x: m for x, m in res.items() if not x.startswith('V_s1004|')})
    (OUT / f'boot_{dom}_identity.json').write_text(json.dumps(idt, indent=1))


# ---------------------------------------------------------------------------------------------- report
RUNGS = {'v1': [f'v1_s{i}' for i in range(11)], 'S': [f'S_s{s}' for s in SEEDS], 'S_small': [f'Ssm_s{s}' for s in SEEDS],
         'V': [f'V_s{s}' for s in SEEDS], 'V_small': [f'Vsm_s{s}' for s in SEEDS]}
CONTRASTS = [('V_small', 'V'), ('S_small', 'S'), ('V_small', 'v1'), ('S_small', 'v1'), ('V', 'v1'), ('S', 'v1'),
             ('V_small', 'S_small')]


def cmd_report(a):
    import v20_gate as G
    lad = json.loads((LO0 / 'ladder.json').read_text())
    res, md = {}, ['# ECI v2.1 small-SAE test (pre-registration scripts/eci/v2/v21_small_prereg.md)', '']
    for dom in ('ants', 'mice'):
        gcells = lad[dom]['gate_cells']
        fc = lad[dom]['focal_cells'] if dom == 'ants' else {}
        groups = {'all': ('gate', gcells), **{f'focal_{f}': ('gate' if f == 'or' else 'focal', cs) for f, cs in fc.items() if cs}}
        z0 = np.load(LB0 / f'boot_{dom}.npz')
        z1 = np.load(BEE / f'boot_new_{dom}.npz')
        Z = {**{k: z0[k] for k in z0.files}, **{k: z1[k] for k in z1.files}}
        rng = np.random.default_rng(L.SEED + 11)
        R = res[dom] = {'gate_cells': gcells, 'focal_cells': fc, 'readouts': {}}
        for agg in ('', 'top10'):
            tg = lambda k: k if (not agg or k.startswith('v1_')) else f'{k}_{agg}'  # noqa: E731  v1: stored frame max only
            dm, lev, percell = {}, {}, {}
            for rung, keys in RUNGS.items():
                for g, (part, cs) in groups.items():
                    ks = [k for k in keys if all(f'{tg(k)}|{part}|{c}' in Z for c in cs)]
                    assert len(ks) == len(keys), (dom, agg, rung, g, len(ks))
                    A = np.stack([np.mean([Z[f'{tg(k)}|{part}|{c}'] for c in cs], 0) for k in ks])  # (seeds, 3, NB+1)
                    dm[(rung, g)] = A
                    lev.setdefault(rung, {})[g] = {M: {'mean': float(A[:, i, 0].mean()), 'seeds': A[:, i, 0].tolist()}
                                                   for i, M in enumerate(L.MN)}
                for part, cs in (('gate', gcells), ('focal', [c for f, x in fc.items() if f != 'or' for c in x])):
                    for c in cs:
                        v = np.stack([Z[f'{tg(k)}|{part}|{c}'][:, 0] for k in keys])
                        percell.setdefault(rung, {})[f'{part}:{c}'] = {M: float(v[:, i].mean()) for i, M in enumerate(L.MN)}
            con = {}
            for x, y in CONTRASTS:
                for g in groups:
                    ax, ay = dm[(x, g)], dm[(y, g)]
                    r = {}
                    for i, M in enumerate(L.MN):
                        bx, by = ax[:, i, 1:], ay[:, i, 1:]
                        ix = rng.integers(0, len(bx), (L.NB, len(bx)))
                        iy = rng.integers(0, len(by), (L.NB, len(by)))
                        rr = np.arange(L.NB)[:, None]
                        d = np.nanmean(bx[ix, rr], 1) - np.nanmean(by[iy, rr], 1)
                        r[M] = {'diff': float(ax[:, i, 0].mean() - ay[:, i, 0].mean()),
                                'ci': np.nanpercentile(d, [2.5, 97.5]).tolist(),
                                'ci_units_only': np.nanpercentile(bx.mean(0) - by.mean(0), [2.5, 97.5]).tolist()}
                    con.setdefault(f'{x} - {y}', {})[g] = r
            R['readouts'][agg or 'max'] = {'levels': lev, 'contrasts': con, 'per_cell': percell}
    # training records
    tr = {}
    for dom in ('ants', 'mice'):
        for arm in ('S', 'V', 'Ssm', 'Vsm'):
            for s in SEEDS:
                f = (OUT / 'train' / f'{dom}_{arm}_s{s}.json') if arm in BASE else (T.OUT / f'{dom}_train_{arm}_s{s}.json')
                J = json.loads(f.read_text())
                tr[f'{dom}|{arm}_s{s}'] = {k: J.get(k) for k in ('eval_fve', 'eval_l0', 'dead_at_end_train', 'train_time_s', 'gpu', 'node')}
    res['train'] = tr
    # pre-registered verdict (primary readout: P(on) of the frame max)
    cA = res['ants']['readouts']['max']['contrasts']['V_small - V']['all']
    cM = res['mice']['readouts']['max']['contrasts']['V_small - V']['all']
    c1 = cA['top1_prec']['ci'][0] > 0
    c2 = cA['AUROC']['ci'][0] >= -0.01
    c3 = cM['AUROC']['ci'][0] >= -0.01
    res['verdict'] = {'C1 ants top-1% prec V_small - V CI > 0': c1, 'C2 ants mean AUROC V_small - V lower >= -0.01': c2,
                      'C3 mice mean AUROC V_small - V lower >= -0.01': c3, 'ADOPT small recipe': bool(c1 and c2 and c3)}
    (OUT / 'report.json').write_text(json.dumps(res, indent=1, default=float))
    f3 = lambda v: f'{v:+.3f}'  # noqa: E731
    md += ['Verdict (P(on), frame max): ' + '; '.join(f'{k}: {"YES" if v else "NO"}' for k, v in res['verdict'].items()), '']
    for dom in ('ants', 'mice'):
        for rd, Rr in res[dom]['readouts'].items():
            md += [f'## {dom}, readout {rd} -> P(on) (v1 always raw frame max)', '',
                   '| rung | n | mean AUROC | top-1% prec | nAP |' + (' focal single AUROC / top-1% | focal double AUROC / top-1% | focal or AUROC / top-1% |' if dom == 'ants' else ''),
                   '|---|---|---|---|---|' + ('---|---|---|' if dom == 'ants' else '')]
            for rung, lv in Rr['levels'].items():
                tail = ''.join(f" {lv[g]['AUROC']['mean']:.3f} / {lv[g]['top1_prec']['mean']:.3f} |"
                               for g in ('focal_single', 'focal_double', 'focal_or') if g in lv)
                md.append(f"| {rung} | {len(lv['all']['AUROC']['seeds'])} | {lv['all']['AUROC']['mean']:.3f} | "
                          f"{lv['all']['top1_prec']['mean']:.3f} | {lv['all']['nAP']['mean']:.3f} |" + tail)
            md += ['', '| contrast | group | d mean AUROC [95% CI videos+seeds] | d top-1% prec [CI] | d nAP [CI] |', '|---|---|---|---|---|']
            for name, gs in Rr['contrasts'].items():
                for g, r in gs.items():
                    md.append(f'| {name} | {g} | ' + ' | '.join(f"{f3(r[M]['diff'])} [{f3(r[M]['ci'][0])}, {f3(r[M]['ci'][1])}]"
                                                                 for M in ('AUROC', 'top1_prec', 'nAP')) + ' |')
            md += ['', '| cell | ' + ' | '.join(f'{r} AUROC / top-1%' for r in RUNGS) + ' |', '|---|' + '---|' * len(RUNGS)]
            for c in Rr['per_cell']['v1']:
                md.append(f'| {c} | ' + ' | '.join(f"{Rr['per_cell'][r][c]['AUROC']:.3f} / {Rr['per_cell'][r][c]['top1_prec']:.3f}"
                                                   for r in RUNGS) + ' |')
            md.append('')
    md += ['## SAE training (eval FVE, L0, dead, seconds, GPU)', '', '| model | FVE | L0 | dead | s | GPU |', '|---|---|---|---|---|---|']
    md += [f"| {k} | {v['eval_fve']:.4f} | {v['eval_l0']:.2f} | {v['dead_at_end_train']} | {v['train_time_s']:.0f} | {v.get('gpu') or '-'} |"
           for k, v in tr.items()]
    (OUT / 'report.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    _ = G


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('train', 'codes', 'cols', 'boot', 'report'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--arm', choices=tuple(BASE))
    p.add_argument('--model', default='')
    p.add_argument('--agg', default='')
    a = p.parse_args()
    {'train': cmd_train, 'codes': cmd_codes, 'cols': cmd_cols, 'boot': cmd_boot, 'report': cmd_report}[a.step](a)
