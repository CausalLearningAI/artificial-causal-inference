"""
Train/test-MATCHED v1 vs v2 comparison (pre-registration: scripts/eci/v2_diag/diag.py docstring block 'MATCHED').
Overlap measured first (scripts/eci/v2_diag/overlap.py -> results/vision/eci_v2_diag/overlap.json).

Steps (job.sh SCRIPT=matched.py):
    v1tok   --dom D [--selftest]   (GPU) v1 tokens of the scored frames, exactly v1's pipeline: dataset 512 px JPG ->
            (odour rotation, mice) -> DINOv2-B 448 (src/eci/foreground FgEncoder) -> v1 mask (FgBackgrounds, rule
            fg448 / ants, v1 backgrounds) -> BEE/v1tok/<D>/<kind>/ (tok.f16 768-d, tok_frame, tok_pos, frames.parquet
            in the step-2 frame order). Identity check: the DEPLOYED v1 SAE on these tokens vs the stored v1 codes_max.
    excl    --dom D                (CPU) annotations.csv rows of every scored frame in v1's own row space -> npy
    v1train --dom D --arm A --seeds a-b  (GPU) v1's recipe (scripts/eci/train_sae_fg.py, deployed args) on v1's store,
            A = V1R_full (all) | V1R_ex (scored frames excluded); the store is staged to /localhome once per job.
    score   --dom D --arm A --seeds a-b  (GPU) release_ab scores of v1-recipe SAEs on the v1 tokens -> OUT/<D>/ab/
    decide                         (CPU) matched pairs -> OUT/matched.json, OUT/matched_table.md
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2_diag'))
import diag as D  # noqa: E402

FZ, MT, T2, N, R4, RA = D.FZ, D.MT, D.T2, D.N, D.R4, D.RA
log = N.log
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_diag')
CFG = {
    'mice': dict(ann='dataset/mice/v1/annotations.csv', bg='dataset/mice/v1/eci/fg448al/background', rule='fg448',
                 align='odor', kinds=('diag',),
                 store='dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448al_fps1',
                 sae='dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448al_s0',
                 codes='dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0/codes_max.npy'),
    'ants': dict(ann='dataset/ants/eci30/annotations.csv', bg='dataset/ants/eci30/fg448/background', rule='ants',
                 align='none', kinds=('scored_v2', 'scored_v3'),
                 store='dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfg_fps1',
                 sae='dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0',
                 codes='dataset/ants/eci30/codes/matryoshka_btk_1024_k16_antsfg_s0/codes_max.npy'),
}
V1_SEEDS = tuple(range(1, 11))
PLUS_SEEDS = tuple(range(700, 710))


def step2_frames(dom, kind):
    root = T2.BEE / dom / kind
    return pd.concat([pd.read_parquet(p / 'frames.parquet') for p in sorted(root.glob('s*'), key=lambda p: int(p.name[1:]))],
                     ignore_index=True)


# ---------------------------------------------------------------------------------------------- v1 tokens
def cmd_v1tok(args):
    import torch
    from src.eci.foreground import RULES, FgBackgrounds, FgEncoder, align_rot90, encode_batch, obs_rows
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    c = CFG[args.dom]
    ann_path = REPO / c['ann']
    fp = pd.read_csv(ann_path, usecols=['frame_path'])['frame_path'].values
    rot = align_rot90(c['align'], ann_path)
    enc = FgEncoder('dinov2_base', dev)
    fb = FgBackgrounds(REPO / c['bg'], obs_rows(ann_path), RULES[c['rule']], dev, align=c['align'], cache=8)
    sae, norm, _ = load_sae(REPO / c['sae'] / 'sae.pt', dev)
    ref = np.load(REPO / c['codes'], mmap_mode='r')
    out = {}
    for kind in c['kinds']:
        F = step2_frames(args.dom, kind)
        rows = F.row.values.astype(np.int64)
        if args.dom == 'ants':  # step-2 ants 'row' is NOT an eci30 row: map (obs, frame_idx) -> eci30 row (release_ab)
            a30 = pd.read_csv(ann_path, usecols=['observation_id', 'frame_idx'], dtype={'observation_id': str})
            idx = pd.Series(np.arange(len(a30)), index=pd.MultiIndex.from_arrays([a30.observation_id, a30.frame_idx]))
            r_ = idx.reindex(pd.MultiIndex.from_arrays([F.obs.values.astype(str), F.k.values.astype(int)])).values
            assert not np.isnan(r_).any(), 'ants frames without an eci30 row'
            rows = r_.astype(np.int64)
        if args.selftest:
            F, rows = F.iloc[:256], rows[:256]
        ds = enc.dataset([str(REPO / 'dataset' / p) for p in fp[rows]], rows,
                         None if rot is None else rot[rows])
        dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=max(2, int(os.environ.get('SLURM_CPUS_PER_TASK', 8)) - 2),
                                         shuffle=False, pin_memory=True, prefetch_factor=4)
        loc = N.local_dir() / f'v1tok_{args.dom}_{kind}'
        loc.mkdir(parents=True, exist_ok=True)
        ftok = open(loc / 'tok.f16', 'wb')
        tfs, poss, mx_all, t0, o = [], [], [], time.time(), 0
        for pix, grey, rr in dl:
            with torch.no_grad():
                tok = encode_batch(enc.model, pix, dev)
                m, _ = fb.mask(tok, grey.to(dev), rr.numpy())
                fi, pi = torch.nonzero(m, as_tuple=True)
                x = tok[fi, pi]
                z = sae.encode(norm(x), mode='threshold')
                mx = torch.zeros(len(tok), z.shape[1], device=dev).index_reduce_(0, fi, z, 'amax', include_self=True)
            ftok.write(x.cpu().numpy().tobytes())
            tfs.append((fi.cpu().numpy() + o).astype(np.int32))
            poss.append(pi.cpu().numpy().astype(np.int16))
            mx_all.append(mx.cpu().numpy())
            o += len(tok)
        ftok.close()
        np.save(loc / 'tok_frame.npy', np.concatenate(tfs))
        np.save(loc / 'tok_pos.npy', np.concatenate(poss))
        F.to_parquet(loc / 'frames.parquet')
        mx = np.concatenate(mx_all)
        if args.dom == 'mice':
            r_ref = rows
        else:  # ants v1 codes are on eci30 rows already (release_ab reads them by (obs, frame_idx) on eci30)
            r_ref = rows
        rc = np.asarray(ref[np.sort(r_ref)])[np.argsort(np.argsort(r_ref))].astype(np.float32)
        chk = {'frames': int(len(F)), 'tokens_per_frame': float(len(np.concatenate(poss)) / len(F)),
               'max_abs_diff_codes_max': float(np.abs(mx - rc).max()), 'mean_abs_diff': float(np.abs(mx - rc).mean()),
               'fire_agree': float(((mx > 0) == (rc > 0)).mean()), 'wall_s': time.time() - t0}
        log(f'v1tok {args.dom} {kind}: {json.dumps(chk)}; {MT.gpu_line()}')
        out[kind] = chk
        if not args.selftest:
            R4.atomic_copy(loc, BEE / 'v1tok' / args.dom / kind)
        shutil.rmtree(loc)
    od = D.OUT / 'matched'
    od.mkdir(parents=True, exist_ok=True)
    (od / f'v1tok_{args.dom}{"_selftest" if args.selftest else ""}.json').write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------- exclusion rows
def cmd_excl(args):
    """Scored frames in v1's TRAINING row space (mice: v1 annotations rows = the step-2 'row'; ants: v1's store indexes
    dataset/ants/eci/annotations.csv rows; scored frames (eci30 rows) are matched by frame_path)."""
    od = D.OUT / 'matched'
    od.mkdir(parents=True, exist_ok=True)
    if args.dom == 'mice':
        r = step2_frames('mice', 'diag').row.values.astype(np.int64)
    else:
        a10 = pd.read_csv(REPO / 'dataset/ants/eci/annotations.csv', usecols=['frame_path'])['frame_path']
        a30 = pd.read_csv(REPO / 'dataset/ants/eci30/annotations.csv', usecols=['frame_path'])['frame_path'].values
        idx = pd.Series(np.arange(len(a10)), index=a10.values)
        paths = np.concatenate([a30[step2_frames('ants', k).row.values.astype(np.int64)] for k in ('scored_v2', 'scored_v3')])
        r = idx.reindex(paths).dropna().values.astype(np.int64)
    np.save(od / f'excl_rows_{args.dom}.npy', np.unique(r))
    log(f'excl {args.dom}: {len(np.unique(r)):,} rows')


# ---------------------------------------------------------------------------------------------- v1 retrain
def v1_dir(dom, arm, s):
    return BEE / 'v1retrain' / dom / f'{arm}_s{s}'


def cmd_v1train(args):
    c = CFG[args.dom]
    loc = N.local_dir() / 'v1store'
    if not loc.exists():
        t0 = time.time()
        shutil.copytree(REPO / c['store'], loc)
        log(f'staged {c["store"]} -> {loc} in {time.time() - t0:.0f}s')
    dep = json.loads((REPO / c['sae'] / 'metrics.json').read_text())['args']
    for s in FZ.parse_seeds(args.seeds, V1_SEEDS):
        od = v1_dir(args.dom, args.arm, s)
        if (od / 'metrics.json').exists():
            continue
        cmd = [sys.executable, '-u', str(REPO / 'scripts/eci/train_sae_fg.py'), '--domain', args.dom,
               '--tokens-dir', str(loc), '--tag', dep['tag'], '--seed', str(s), '--n-latents', str(dep['n_latents']),
               '--prefixes', dep['prefixes'], '--k', str(dep['k']), '--epochs', str(dep['epochs']),
               '--batch-size', str(dep['batch_size']), '--lr', str(dep['lr']), '--out-dir', str(od)]
        if dep.get('val_from'):
            cmd += ['--val-from', dep['val_from']]
        if args.arm == 'V1R_ex':
            cmd += ['--exclude-rows', str(D.OUT / 'matched' / f'excl_rows_{args.dom}.npy')]
        log(' '.join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO)


# ---------------------------------------------------------------------------------------------- score
class V1Store:
    def __init__(self, dom, kind, dev):
        import torch
        root = BEE / 'v1tok' / dom / kind
        self.F = pd.read_parquet(root / 'frames.parquet')
        self.tf = np.load(root / 'tok_frame.npy')
        self.pos = np.load(root / 'tok_pos.npy').astype(np.int64)
        self.tok = np.asarray(np.memmap(root / 'tok.f16', np.float16, 'r', shape=(len(self.tf), 768)))
        self.nf, self.G, self.dev, self.d = len(self.F), 32, dev, 768
        self.starts = np.r_[0, np.cumsum(np.bincount(self.tf, minlength=self.nf))]
        self.T = D.R2.tokens_to(torch, self.tok, dev)

    chunks = FZ.TokStore.chunks


def cmd_score(args):
    import torch
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    dom = args.dom
    od = D.OUT / dom / 'ab'
    if dom == 'mice':
        St = V1Store('mice', 'diag', dev)
        Y, sets, half, _, _ = MT.mice_sets(St)
        Yg = torch.from_numpy(Y).to(dev)
    else:
        sts = {ser: V1Store('ants', f'scored_{ser}', dev) for ser in ('v2', 'v3')}
        lab = {ser: MT.ants_set(S_, ser) for ser, S_ in sts.items()}
    seeds = [0] if args.arm == 'V1_dep' else FZ.parse_seeds(args.seeds, V1_SEEDS)
    for s in seeds:
        f = od / f'{args.arm}_s{s}.json'
        p = REPO / CFG[dom]['sae'] / 'sae.pt' if args.arm == 'V1_dep' else v1_dir(dom, args.arm, s) / 'sae.pt'
        if f.exists() or not p.exists():
            continue
        sae, norm, _ = load_sae(p, dev)
        if dom == 'mice':
            ab = RA.ab_mice(D.codes(St, sae, norm), Yg, half['all'], St, FZ.MLAB)
            log(f'mice {args.arm}_s{s}: nn {ab["full"]["nose_nose"]["R_atlas"]:.4f} nt {ab["full"]["nose_tail"]["R_atlas"]:.4f}')
        else:
            ab = {}
            for ser, S_ in sts.items():
                Y, hh, labs = lab[ser]
                ab[ser] = RA.ab_ants(D.codes(S_, sae, norm), torch.from_numpy(Y).to(dev), hh, labs)
            log(f'ants {args.arm}_s{s}: v2 groom {ab["v2"]["full"]["groom_any"]["R_atlas"]:.4f} v3 groom '
                f'{ab["v3"]["full"]["groom_any"]["R_atlas"]:.4f} lid {ab["v3"]["full"]["on_lid_any"]["R_atlas"]:.4f}')
        f.write_text(json.dumps(ab, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- decide
PAIRS = (('neither_seen', 'M0', 'V1R_ex'), ('both_seen', 'M0_plus', 'V1R_full'))


def vals(dom, arm, ser, lb, p):
    if arm == 'M0':
        return np.array(list(D.cell_vals(dom, 'M0', ser, lb, p).values()))
    seeds = PLUS_SEEDS if arm == 'M0_plus' else ([0] if arm == 'V1_dep' else V1_SEEDS)
    return np.array(list(D.cell_vals(dom, arm, ser, lb, p, seeds=seeds).values()))


def cmd_decide(args):
    res, L = {}, ['| pair | v2 arm (n) | v1 arm (n) | cell | prefix | v2 mean | v1 mean | v2 - v1 (Welch 95% CI) | B pass |',
                  '|---|---|---|---|---|---|---|---|---|']
    for pair, a2, a1 in PAIRS:
        ok = True
        for dom, ser, lb in D.DCELLS:
            for p in ('full', '256'):
                x, y = vals(dom, a2, ser, lb, p), vals(dom, a1, ser, lb, p)
                if len(x) < 2 or len(y) < 2:
                    continue
                d = D.welch_ci(x, y)
                ps = bool(d[1] >= -0.01)
                ok &= ps
                nm = f'{dom}{" " + ser if ser else ""} {lb}'
                res.setdefault(pair, {})[f'{nm} | {p}'] = {'v2': float(x.mean()), 'v1': float(y.mean()), 'n2': len(x),
                                                          'n1': len(y), 'diff': list(d), 'pass': ps}
                L.append(f'| {pair} | {a2} ({len(x)}) | {a1} ({len(y)}) | {nm} | {p} | {x.mean():.4f} | {y.mean():.4f} | '
                         f'{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] | {ps} |')
        res.setdefault(pair, {})['B_PASS'] = bool(ok)
    # A for the v2 arms; seen-frame effects; deployed v1 vs its retrained distribution
    for dom, ser, lb in D.DCELLS:
        for p in ('full', '256'):
            nm = f'{dom}{" " + ser if ser else ""} {lb} | {p}'
            for eff, a, b in (('v1_seen_effect', 'V1R_full', 'V1R_ex'), ('v2_seen_effect', 'M0_plus', 'M0')):
                x, y = vals(dom, a, ser, lb, p), vals(dom, b, ser, lb, p)
                if len(x) > 1 and len(y) > 1:
                    res.setdefault(eff, {})[nm] = list(D.welch_ci(x, y))
            dv = vals(dom, 'V1_dep', ser, lb, p)
            fr = vals(dom, 'V1R_full', ser, lb, p)
            if len(dv) and len(fr):
                res.setdefault('deployed_v1_vs_retrained', {})[nm] = {'deployed': float(dv[0]), 'retrained_mean': float(fr.mean()),
                                                                     'retrained_sd': float(fr.std(ddof=1))}
    A = {}
    for arm, seeds in (('M0_plus', PLUS_SEEDS),):
        Rm = {s: json.loads((D.OUT / 'mice' / 'ab' / f'{arm}_s{s}.json').read_text()) for s in seeds
              if (D.OUT / 'mice' / 'ab' / f'{arm}_s{s}.json').exists()}
        A[arm] = {p: float(np.mean([RA.distinct(r[p]) for r in Rm.values()])) for p in ('full', '256')} if Rm else None
    res['A_share_distinct'] = A
    L += ['', 'B PASS: ' + json.dumps({k: res[k]['B_PASS'] for k, _, _ in PAIRS}), 'A share distinct (M0_plus): ' + json.dumps(A),
          'seen-frame effects (Welch): ' + json.dumps({k: {c: [round(v, 4) for v in d] for c, d in res.get(k, {}).items()}
                                                       for k in ('v1_seen_effect', 'v2_seen_effect')}),
          'deployed v1 vs retrained v1 (full data): ' + json.dumps(res.get('deployed_v1_vs_retrained', {}))]
    (D.OUT / 'matched.json').write_text(json.dumps(res, indent=1, default=float))
    (D.OUT / 'matched_table.md').write_text('\n'.join(L))
    print('\n'.join(L))


def cmd_seeddecide(args):
    """v1 seed distribution (deployed s0 + V1R_full 1-10, all on the v1 tokens) and V2F arms mean-vs-mean."""
    res, L = {}, ['## v1 seed distribution (R~ atlas split)', '',
                  '| cell | prefix | deployed s0 | retrained mean ± sd (n) | min / max | deployed rank (of n+1, 1 = best) | z |',
                  '|---|---|---|---|---|---|---|']
    for dom, ser, lb in D.DCELLS:
        for p in ('full', '256'):
            nm = f'{dom}{" " + ser if ser else ""} {lb}'
            dv = vals(dom, 'V1_dep', ser, lb, p)
            fr = vals(dom, 'V1R_full', ser, lb, p)
            if not len(dv) or len(fr) < 2:
                continue
            d0 = float(dv[0])
            rank = int((fr > d0).sum()) + 1
            z = float((d0 - fr.mean()) / fr.std(ddof=1))
            res.setdefault('v1_seeds', {})[f'{nm} | {p}'] = {'deployed': d0, 'mean': float(fr.mean()),
                                                           'sd': float(fr.std(ddof=1)), 'n': len(fr), 'rank': rank, 'z': z,
                                                           'percentile': float((fr < d0).mean())}
            L.append(f'| {nm} | {p} | {d0:.4f} | {fr.mean():.4f} ± {fr.std(ddof=1):.4f} ({len(fr)}) | {fr.min():.4f} / '
                     f'{fr.max():.4f} | {rank} / {len(fr) + 1} | {z:+.2f} |')
    L += ['', '## V2F arms vs the v1 seed distribution (deployed s0 + 10 retrained; Welch 95% CI; pass = lower bound >= -0.01)',
          '', '| arm | cell | prefix | arm mean (n) | v1 mean (n) | arm - v1 (Welch) | pass | arm - deployed v1 (t-CI, original) |',
          '|---|---|---|---|---|---|---|---|']
    for arm in ('V2F_M0', 'V2F_N2'):
        ok = True
        for dom, ser, lb in D.DCELLS:
            for p in ('full', '256'):
                nm = f'{dom}{" " + ser if ser else ""} {lb}'
                x = np.array(list(D.cell_vals(dom, arm, ser, lb, p, seeds=D.F1_SEEDS[dom]).values()))
                y = np.concatenate([vals(dom, 'V1_dep', ser, lb, p), vals(dom, 'V1R_full', ser, lb, p)])
                if len(x) < 2 or len(y) < 2:
                    continue
                d = D.welch_ci(x, y)
                ps = bool(d[1] >= -0.01)
                ok &= ps
                o = RA.tci(x - D.v1_val(dom, ser, lb, p))
                res.setdefault(arm, {})[f'{nm} | {p}'] = {'arm': float(x.mean()), 'v1': float(y.mean()), 'diff': list(d),
                                                         'pass': ps, 'vs_deployed_original': list(o)}
                L.append(f'| {arm} | {nm} | {p} | {x.mean():.4f} ({len(x)}) | {y.mean():.4f} ({len(y)}) | {d[0]:+.4f} '
                         f'[{d[1]:+.4f}, {d[2]:+.4f}] | {ps} | {o[0]:+.4f} [{o[1]:+.4f}, {o[2]:+.4f}] |')
        res.setdefault(arm, {})['B_PASS_mean_vs_mean'] = bool(ok)
    L += ['', 'B (mean vs mean) PASS: ' + json.dumps({a: res.get(a, {}).get('B_PASS_mean_vs_mean') for a in ('V2F_M0', 'V2F_N2')})]
    (D.OUT / 'v1seeds.json').write_text(json.dumps(res, indent=1, default=float))
    (D.OUT / 'v1seeds_table.md').write_text('\n'.join(L))
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('v1tok', 'excl', 'v1train', 'score', 'decide', 'seeddecide'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--arm')
    p.add_argument('--seeds')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    {'v1tok': cmd_v1tok, 'excl': cmd_excl, 'v1train': cmd_v1train, 'score': cmd_score, 'decide': cmd_decide,
     'seeddecide': cmd_seeddecide}[a.step](a)


if __name__ == '__main__':
    main()
