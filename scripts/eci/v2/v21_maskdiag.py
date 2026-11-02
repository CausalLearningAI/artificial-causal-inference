"""
ECI v2.1 mask diagnosis: why does static x on mask v5 (arm S) align worse with mice behaviours than static x on the old
N2 mask restricted to U_t (arm SN2)? Pre-registration: scripts/eci/v2/v21_maskdiag_prereg.md.

Token categories inside the stored set U_t of a frame (flag bit0 = M_t mask v5, bit1 = N2):
  a = M and N2, b = M only, c = N2 only, d = neither (U_t only: temporal dilation of v5).
  S = a+b (mode M), SN2 = a+c (mode N), SA = a (mode A), SMN = a+b+c (mode MN), SU = a+b+c+d (mode U).

Steps
  stats --dom D        (CPU) per eval frame: token counts per category, distance of c / d tokens to the nearest M_t token,
                       probe probability of c tokens, on positive vs negative frames -> OUT/maskdiag/stats_<dom>.json
  prep --dom D         (CPU) training / calib / eval samples of the static arms SA, SMN, SU (the S window plan, rng 0;
                       16M / 1.5M / 1.5M tokens, rng 1) -> T.BEE/<dom>/data/<role>_<arm>.npy
  train                via v21_train.py train (ARMS2 extended here)
  codes --dom D --spec NAME=SAEKEY@MODE,...  (GPU) static frame-max codes over the MODE tokens of every eval / probe frame
                       -> T.BEE/<dom>/codes/<NAME>_s<seed>_{eval,probe}.npy; --peaks: argmax-token category of the
                       best neurons (nn / nt / np / nose_nose of the SAE's own ladder pick) on every eval frame
  score / cols         via v21_train.py score and v21_ladder.py cols (ARMS2 / BIT extended here)
  boot --dom D / report  bootstrap of AMENDMENT 4 B on the diagnostic arms (own ladder dir) and the contrasts
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
import v21_train as T  # noqa: E402

MODES = {'M': lambda f: (f & 1) > 0, 'N': lambda f: (f & 2) > 0, 'A': lambda f: (f & 3) == 3,
         'MN': lambda f: (f & 3) > 0, 'U': lambda f: np.ones(len(f), bool)}
NEW = {'SA': 'A', 'SMN': 'MN', 'SU': 'U'}
T.ARMS2.update({k: 1024 for k in NEW})
T.ARMS2.update({'SxN2': 1024, 'SN2xM': 1024})
MD = T.OUT / 'maskdiag'
LBD = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_1/ladder_maskdiag')
log = T.log


def eval_index(dom):
    """eval frames in eval.py order with the store series name."""
    import eval as EV
    F = EV.frame_table(dom)
    E = T.frame_sets(dom)['eval']
    return F, E


# ---------------------------------------------------------------------------------------------- stats
def cmd_stats(a):
    from scipy import ndimage as nd
    dom = a.dom
    F, E = eval_index(dom)
    n = len(E)
    cnt = np.zeros((n, 4), np.int32)  # a, b, c, d
    dist_c = np.zeros((n, 6), np.int32)  # c tokens at chessboard distance 1, 2, 3, 4, 5+, no M token in the frame
    dist_d = np.zeros((n, 6), np.int32)
    bridge_c = np.zeros(n, np.int32)  # c tokens within 4 patches of two distinct 8-connected M_t components
    ncomp = np.zeros(n, np.int32)
    prob_c, prob_b = [], []
    t0 = time.time()
    for (ser, obs), g in E.groupby(['series', 'obs'], sort=False):
        z = np.load(T.STORE / dom / T.VER / 'videos' / ser / obs / 'index.npz')
        G = int(z['G'])
        kpos = {int(k): i for i, k in enumerate(z['k'])}
        ptr, idx, flag, prob = z['ptr'], z['idx'].astype(np.int64), z['flag'], z['prob']
        for row, k in zip(g.index.values, g.k.values):
            i = kpos[int(k)]
            r0, r1 = int(ptr[i]), int(ptr[i + 1])
            f, p, pr = flag[r0:r1] & 3, idx[r0:r1], prob[r0:r1]
            cat = np.select([f == 3, f == 1, f == 2], [0, 1, 2], 3)
            cnt[row] = np.bincount(cat, minlength=4)
            m = np.zeros((G, G), bool)
            m.ravel()[p[(f & 1) > 0]] = True
            for cc, out in ((2, dist_c), (3, dist_d)):
                q = p[cat == cc]
                if not len(q):
                    continue
                if not m.any():
                    out[row, 5] = len(q)
                    continue
                D = nd.distance_transform_cdt(~m, metric='chessboard')
                out[row, :5] = np.bincount(np.clip(D.ravel()[q], 1, 5) - 1, minlength=5)
            lab, nc = nd.label(m, structure=np.ones((3, 3)))
            big = [j for j, sz in enumerate(np.bincount(lab.ravel())[1:], 1) if sz >= 4][:6]  # components >= 4 tokens
            ncomp[row] = len(big)
            q = p[cat == 2]
            if len(big) >= 2 and len(q):
                near = []
                for j in big:
                    near.append(nd.distance_transform_cdt(lab != j, metric='chessboard').ravel()[q] <= 4)
                bridge_c[row] = int((np.sum(near, 0) >= 2).sum())
            if row % 50 == 0:
                prob_c.append(pr[cat == 2])
                prob_b.append(pr[cat == 1])
    log(f'{dom} stats loop done ({time.time() - t0:.0f}s)')
    labs = [c[2:] for c in F.columns if c.startswith('y_')]

    def summ(sel):
        s = {'frames': int(sel.sum())}
        c4 = cnt[sel]
        s['tokens_per_frame_mean'] = dict(zip(['a_M_and_N2', 'b_M_only', 'c_N2_only', 'd_U_only'], c4.mean(0).round(2).tolist()))
        s['tokens_per_frame_median'] = dict(zip(['a', 'b', 'c', 'd'], np.median(c4, 0).tolist()))
        s['S_tokens_mean'] = float((c4[:, 0] + c4[:, 1]).mean())
        s['SN2_tokens_mean'] = float((c4[:, 0] + c4[:, 2]).mean())
        s['U_tokens_mean'] = float(c4.sum(1).mean())
        s['frames_with_0_M_tokens'] = float(((c4[:, 0] + c4[:, 1]) == 0).mean())
        s['frames_with_0_SN2_tokens'] = float(((c4[:, 0] + c4[:, 2]) == 0).mean())
        for nm, D in (('c', dist_c), ('d', dist_d)):
            tot = D[sel].sum()
            s[f'{nm}_dist_share_1_2_3_4_5plus_noM'] = (D[sel].sum(0) / max(tot, 1)).round(4).tolist()
        s['c_bridge_tokens_per_frame'] = float(bridge_c[sel].mean())
        s['c_bridge_share_of_c'] = float(bridge_c[sel].sum() / max(cnt[sel, 2].sum(), 1))
        s['frames_with_ge1_bridge_c'] = float((bridge_c[sel] > 0).mean())
        s['M_components_mean'] = float(ncomp[sel].mean())
        s['share_frames_M_ge2_components'] = float((ncomp[sel] >= 2).mean())
        return s
    out = {'all': summ(np.ones(n, bool))}
    for b in labs:
        v, y = F[f'v_{b}'].values, F[f'y_{b}'].values.astype(bool)
        out[f'{b}+'] = summ(v & y)
        out[f'{b}-'] = summ(v & ~y)
    pc, pb = np.concatenate(prob_c), np.concatenate(prob_b)
    out['probe_prob_uint8_quartiles'] = {'c_N2_only': np.percentile(pc, [25, 50, 75]).tolist() if len(pc) else None,
                                         'b_M_only': np.percentile(pb, [25, 50, 75]).tolist() if len(pb) else None}
    MD.mkdir(parents=True, exist_ok=True)
    (MD / f'stats_{dom}.json').write_text(json.dumps(out, indent=1))
    log(json.dumps(out['all']))


# ---------------------------------------------------------------------------------------------- prep
def plan_S(dom):
    """the exact window plan of v21_train.cmd_prep (rng 0, M_t tokens per frame from 6 videos)."""
    rng = np.random.default_rng(0)
    vids = T.videos_of(dom)
    mpf = np.mean([((T.Video(dom, v[0], v[1]).flag & 1) > 0).sum() / len(T.Video(dom, v[0], v[1]).k)
                   for v in vids[::max(1, len(vids) // 6)][:6]])
    plan = []
    for r in T.CAP:
        vs = [v for v in vids if v[2] == r]
        n_win = max(1, int(np.ceil(1.1 * T.CAP[r] / mpf / (T.WIN * max(1, len(vs))))))
        for ser, obs, role in vs:
            nf = len(T.Video(dom, ser, obs).k)
            starts = np.sort(rng.choice(max(1, nf - T.WIN), min(n_win, max(1, nf - T.WIN)), replace=False))
            plan += [(ser, obs, role, int(s_)) for s_ in starts]
    return plan


def _rows(z, s_, mode):
    i0, i1 = s_, min(len(z['k']), s_ + T.WIN)
    r0, r1 = int(z['ptr'][i0]), int(z['ptr'][i1])
    return r0, np.flatnonzero(MODES[mode](z['flag'][r0:r1]))


def _prep_video(args):
    dom, ser, obs, wins, arms, keep_off, paths = args
    d = T.STORE / dom / T.VER / 'videos' / ser / obs
    z = dict(np.load(d / 'index.npz'))
    tok = np.memmap(d / 'tok.f16', np.float16, 'r', shape=(int(z['ptr'][-1]), 1024))
    outs = {k: np.load(p, mmap_mode='r+') for k, p in paths.items()}
    for wi, (role, s_) in wins:
        i0, i1 = s_, min(len(z['k']), s_ + T.WIN)
        r0, r1 = int(z['ptr'][i0]), int(z['ptr'][i1])
        X = np.asarray(tok[r0:r1])
        fl = z['flag'][r0:r1]
        for arm in arms:
            sel = np.flatnonzero(MODES[NEW.get(arm, 'M')](fl))
            kk, off = keep_off[(arm, wi)]  # local positions to keep (into sel) and output row offset
            if len(kk):
                outs[(arm, role)][off:off + len(kk)] = X[sel[kk]]
    for o in outs.values():
        o.flush()
    return ser, obs


def cmd_prep(a):
    from multiprocessing import get_context
    dom = a.dom
    arms = a.arms.split(',') if a.arms else list(NEW)
    t0 = time.time()
    plan = plan_S(dom)
    # pass 1: counts per window and arm (flags only)
    cnts = {arm: np.zeros(len(plan), np.int64) for arm in arms + ['S']}
    zc = {}
    for wi, (ser, obs, role, s_) in enumerate(plan):
        if (ser, obs) not in zc:
            zc.clear()
            zc[(ser, obs)] = np.load(T.STORE / dom / T.VER / 'videos' / ser / obs / 'index.npz')
        z = zc[(ser, obs)]
        i0, i1 = s_, min(len(z['k']), s_ + T.WIN)
        fl = z['flag'][int(z['ptr'][i0]):int(z['ptr'][i1])]
        for arm in cnts:
            cnts[arm][wi] = MODES[NEW.get(arm, 'M')](fl).sum()
    roles = np.array([p[2] for p in plan])
    info = {'windows': len(plan), 'S_available_check': {r: int(cnts['S'][roles == r].sum()) for r in T.CAP}}
    log(f'{dom} pass 1 ({time.time() - t0:.0f}s): {json.dumps(info)}')
    keep_off, paths = {}, {}
    od = T.BEE / dom / 'data'
    for arm in arms:
        rng2 = np.random.default_rng(1)
        info[arm] = {}
        for r in T.CAP:
            w = np.flatnonzero(roles == r)
            c = cnts[arm][w]
            tot = int(c.sum())
            keep = np.sort(rng2.choice(tot, min(tot, T.CAP[r]), replace=False))
            info[arm][r] = {'available': tot, 'used': len(keep), 'tokens_per_frame': float(tot / (len(w) * T.WIN))}
            start = np.r_[0, np.cumsum(c)]
            # per window: keep indices inside it and output offset
            lo = np.searchsorted(keep, start[:-1])
            hi = np.searchsorted(keep, start[1:])
            for j, wi in enumerate(w):
                keep_off[(arm, int(wi))] = (keep[lo[j]:hi[j]] - start[j], int(lo[j]))
            p = od / f'{r}_{arm}.npy'
            np.lib.format.open_memmap(p, 'w+', np.float16, (len(keep), 1024)).flush()
            paths[(arm, r)] = p
    log(f'{dom} keep plan done: {json.dumps({k: v for k, v in info.items() if k in arms})}')
    byv = {}
    for wi, (ser, obs, role, s_) in enumerate(plan):
        byv.setdefault((ser, obs), []).append((wi, (role, s_)))
    jobs = [(dom, ser, obs, [(wi, rs) for wi, rs in L], arms, {k: v for k, v in keep_off.items() if k[1] in {x[0] for x in L}},
             {k: v for k, v in paths.items()}) for (ser, obs), L in byv.items()]
    with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        for j, _ in enumerate(pool.imap_unordered(_prep_video, jobs)):
            if j % 20 == 0:
                log(f'{dom} prep {j}/{len(jobs)} videos ({time.time() - t0:.0f}s)')
    MD.mkdir(parents=True, exist_ok=True)
    (MD / f'{dom}_prep.json').write_text(json.dumps(info, indent=1))
    log(f'{dom} prep done ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- codes
def cmd_codes(a):
    """--spec NAME=SAEARM@MODE,... for --seeds; frame max over the MODE tokens (static x, as S / SN2)."""
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    dom = a.dom
    seeds = [int(s) for s in a.seeds.split(',')]
    spec = [(x.split('=')[0], *x.split('=')[1].split('@')) for x in a.spec.split(',')]
    saes = {}
    for nm, arm, mode in spec:
        for s in seeds:
            saes[f'{nm}_s{s}'] = (SA.load(T.BEE / dom / 'sae' / f'{arm}_s{s}' / 'sae_theta.pt', dev), mode, arm, s)
    # peak tracking: neuron list per SAE key (ladder pick of the SAE's own native arm), mice only
    peaks = {}
    if a.peaks:
        for k, (_, mode, arm, s) in saes.items():
            J = json.loads((T.OUT / 'ladder' / dom / f'{arm}_s{s}.json').read_text())['gate']
            peaks[k] = sorted({J[c]['best_neuron'] for c in J if J[c]['best_neuron'] is not None})
    od = T.BEE / dom / 'codes'
    t0 = time.time()
    sets = ['eval'] if a.eval_only else ['eval', 'probe']
    for nm in sets:
        Fs = T.frame_sets(dom)[nm]
        spans = [(ser, obs, int(g.k.min()), int(g.k.max()) + 1) for (ser, obs), g in Fs.groupby(['series', 'obs'], sort=False)]
        rows = [(ser, obs, k) for ser, obs, k0, k1 in spans for k in range(k0, k1)]
        R = pd.DataFrame(rows, columns=['series', 'obs', 'k'])
        R0 = pd.read_parquet(od / f'{nm}_rows.parquet')
        assert R.equals(R0), f'{nm} rows differ from the stored S rows'
        Xc = {k: np.zeros((len(R), 2048), np.float16) for k in saes}
        PK = {k: np.full((len(R), len(peaks[k]), 2), -1, np.int16) for k in peaks} if nm == 'eval' else {}
        PV = {k: np.zeros((len(R), len(peaks[k])), np.float16) for k in peaks} if nm == 'eval' else {}
        base = 0
        for ser, obs, k0, k1 in spans:
            V_ = T.Video(dom, ser, obs)
            kpos = {int(k): i for i, k in enumerate(V_.k)}
            i0, i1 = kpos[k0], kpos[k1 - 1] + 1
            for b0 in range(i0, i1, 600):
                b1 = min(i1, b0 + 600)
                r0, r1 = int(V_.ptr[b0]), int(V_.ptr[b1])
                tok = torch.from_numpy(np.asarray(V_.tok[r0:r1])).to(dev).float()
                fl = V_.flag[r0:r1]
                fi = np.repeat(np.arange(b0, b1), np.diff(V_.ptr[b0:b1 + 1])) - b0
                cat = np.select([(fl & 3) == 3, (fl & 3) == 1, (fl & 3) == 2], [0, 1, 2], 3)
                for k, ((sae, norm), mode, arm, s) in saes.items():
                    c = np.flatnonzero(MODES[mode](fl))
                    if not len(c):
                        continue
                    ct = torch.from_numpy(c).to(dev)
                    with torch.no_grad():
                        cc = sae.encode(norm(tok[ct]), mode='threshold').float()
                    flt = torch.from_numpy(fi[c]).to(dev)
                    m = torch.zeros((b1 - b0, 2048), device=dev).index_reduce_(0, flt, cc, 'amax', include_self=True)
                    o = base + b0 - i0
                    Xc[k][o:o + b1 - b0] = m.half().cpu().numpy()
                    if k in PK:
                        for jj, j in enumerate(peaks[k]):
                            v = cc[:, j]
                            # argmax token per frame: sort by (frame, value)
                            order = np.lexsort((-v.cpu().numpy(), fi[c]))
                            first = order[np.r_[True, fi[c][order][1:] != fi[c][order][:-1]]]
                            ff = fi[c][first]
                            PK[k][o + ff, jj, 0] = cat[c[first]]
                            PK[k][o + ff, jj, 1] = 1
                            PV[k][o + ff, jj] = v.cpu().numpy()[first]
            base += i1 - i0
        for k, x in Xc.items():
            p = od / f'{k}_{nm}.npy'
            if a.check_only_existing and p.exists():
                old = np.load(p)
                d = float(np.abs(old.astype(np.float32) - x.astype(np.float32)).max())
                log(f'IDENTITY {k}_{nm}: max abs diff vs stored codes {d:.3e}')
                if d > 1e-2:
                    raise SystemExit(f'IDENTITY CHECK FAILED {k}_{nm}: {d}')
                continue
            np.save(p, x)
        for k in PK:
            (LBD.parent / 'maskdiag_peaks').mkdir(parents=True, exist_ok=True)
            np.savez(LBD.parent / 'maskdiag_peaks' / f'{dom}_{k}.npz', cat=PK[k], val=PV[k], neurons=np.array(peaks[k]))
        log(f'{dom} {nm}: {len(R):,} frames, codes of {len(saes)} SAE readouts ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- bootstrap / report
def ladder_mod():
    import v21_ladder as L
    L.LO = MD / 'ladder'
    L.LB = LBD
    return L


def cmd_link(a):
    """copy the existing ladder jsons / npz of the reference models into the diagnostic ladder dir."""
    import shutil
    dom = a.dom
    (MD / 'ladder' / dom).mkdir(parents=True, exist_ok=True)
    (LBD / dom).mkdir(parents=True, exist_ok=True)
    for t in a.keys.split(','):
        shutil.copyfile(T.OUT / 'ladder' / dom / f'{t}.json', MD / 'ladder' / dom / f'{t}.json')
        src = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_1/ladder') / dom / f'{t}.npz'
        if not (LBD / dom / f'{t}.npz').exists():
            os.symlink(src, LBD / dom / f'{t}.npz')


def cmd_cols(a):
    L = ladder_mod()
    L.cmd_cols(a)


def cmd_boot(a):
    L = ladder_mod()
    L.cmd_boot(a)


RUNGS = {'S': 'S', 'SN2': 'SN2', 'SA': 'SA', 'SMN': 'SMN', 'SU': 'SU', 'SxN2': 'SxN2', 'SN2xM': 'SN2xM', 'PB': 'PB', 'V': 'V'}
CON = [('SN2', 'S'), ('SxN2', 'S'), ('SN2', 'SN2xM'), ('SN2xM', 'S'), ('SN2', 'SxN2'), ('SMN', 'S'), ('SMN', 'SN2'),
       ('SU', 'S'), ('SU', 'SMN'), ('SU', 'SN2'), ('SA', 'S'), ('SA', 'SN2'), ('V', 'S'), ('V', 'SU')]


def cmd_report(a):
    import v20_gate as G
    L = ladder_mod()
    out, md = {}, ['# v2.1 mask diagnosis: contrasts (readout P(on), frame max; 95% CI = pools/videos + seeds resampled)', '']
    for dom in ('mice', 'ants'):
        bp = LBD / f'boot_{dom}.npz'
        if not bp.exists():
            continue
        S0 = json.loads((G.OUT / f'{dom}_scores.json').read_text())
        gcells = [c for c, r in S0['eligibility'].items() if r['eligible'] and c.split('|')[1] not in G.ONLID]
        z = np.load(bp)
        dm, lv = {}, {}
        for rung in RUNGS:
            ks = ['PB'] if rung == 'PB' else [f'{rung}_s{s}' for s in (1001, 1002, 1003)]
            ks = [k for k in ks if all(f'{k}|gate|{c}' in z for c in gcells)]
            if not ks:
                continue
            dm[rung] = np.stack([np.mean([z[f'{k}|gate|{c}'] for c in gcells], 0) for k in ks])
            lv[rung] = {'n': len(ks), **{M: dm[rung][:, i, 0].tolist() for i, M in enumerate(L.MN)},
                        'cells_AUROC': {c: [float(z[f'{k}|gate|{c}'][0, 0]) for k in ks] for c in gcells}}
        rng = np.random.default_rng(L.SEED + 7)
        con = {}
        NB = L.NB
        for x, y in CON:
            if x not in dm or y not in dm:
                continue
            r = {}
            for mi, M in enumerate(L.MN):
                ax, ay = dm[x][:, mi, 1:], dm[y][:, mi, 1:]
                ix = rng.integers(0, len(ax), (NB, len(ax)))
                iy = rng.integers(0, len(ay), (NB, len(ay)))
                rr = np.arange(NB)[:, None]
                d = np.nanmean(ax[ix, rr], 1) - np.nanmean(ay[iy, rr], 1)
                r[M] = {'diff': float(dm[x][:, mi, 0].mean() - dm[y][:, mi, 0].mean()),
                        'ci_units_seeds': np.nanpercentile(d, [2.5, 97.5]).tolist(),
                        'ci_units_only': np.nanpercentile(ax.mean(0) - ay.mean(0), [2.5, 97.5]).tolist()}
            con[f'{x} - {y}'] = r
        out[dom] = {'gate_cells': gcells, 'levels': lv, 'contrasts': con}
        md += [f'## {dom} ({len(gcells)} gate cells)', '', '| arm | n | mean AUROC P(on) (median [range] of seeds) | top-1% prec P(on) | nAP P(on) |', '|---|---|---|---|---|']
        f = lambda v: f'{np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]' if len(v) > 1 else f'{v[0]:.3f}'  # noqa: E731
        for rung, R in lv.items():
            md.append(f'| {rung} | {R["n"]} | {f(R["AUROC"])} | {f(R["top1_prec"])} | {f(R["nAP"])} |')
        md += ['', '| contrast | d mean AUROC [95% CI pools+seeds] (pools only) | d top-1% prec | d nAP |', '|---|---|---|---|']
        for k, r in con.items():
            cell = lambda m: f'{r[m]["diff"]:+.3f} [{r[m]["ci_units_seeds"][0]:+.3f}, {r[m]["ci_units_seeds"][1]:+.3f}] ([{r[m]["ci_units_only"][0]:+.3f}, {r[m]["ci_units_only"][1]:+.3f}])'  # noqa
            md.append(f'| {k} | {cell("AUROC")} | {cell("top1_prec")} | {cell("nAP")} |')
        md.append('')
    (MD / 'report.json').write_text(json.dumps(out, indent=1))
    (MD / 'report.md').write_text('\n'.join(md))
    print('\n'.join(md))


def cmd_train(a):
    T.cmd_train(a)


def cmd_score(a):
    """v21_train.score with the diagnostic arms; a cross-readout key reuses its SAE's train record (FVE)."""
    import shutil
    src = {'SxN2': 'S', 'SN2xM': 'SN2'}
    for arm in a.arms.split(','):
        for s in T.seeds_of(a):
            f = T.OUT / f'{a.dom}_train_{arm}_s{s}.json'
            if arm in src and not f.exists():
                shutil.copyfile(T.OUT / f'{a.dom}_train_{src[arm]}_s{s}.json', f)
    T.cmd_score(a)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('stats', 'prep', 'train', 'codes', 'score', 'link', 'cols', 'boot', 'report'))
    p.add_argument('--arm', default='S')
    p.add_argument('--seed', type=int, default=1001)
    p.add_argument('--only', default='')
    p.add_argument('--outdir', default='')
    p.add_argument('--ref', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--no_rows', action='store_true')
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--arms', default='')
    p.add_argument('--spec', default='')
    p.add_argument('--seeds', default='1001,1002,1003')
    p.add_argument('--peaks', action='store_true')
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--check_only_existing', action='store_true', help='codes: existing outputs are compared, not rewritten')
    p.add_argument('--keys', default='')
    p.add_argument('--model', default='')
    p.add_argument('--agg', default='')
    a = p.parse_args()
    {'stats': cmd_stats, 'prep': cmd_prep, 'train': cmd_train, 'score': cmd_score, 'codes': cmd_codes, 'link': cmd_link, 'cols': cmd_cols, 'boot': cmd_boot,
     'report': cmd_report}[a.step](a)


if __name__ == '__main__':
    main()
