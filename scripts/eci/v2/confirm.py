"""
ECI v2.1 CONFIRMATION (fresh seeds): does F2-8192 (3 x 3 spatial context tokens, Matryoshka BatchTopK 8192, k 16)
beat the shipped v2 = F0-2048 (no context, Matryoshka 2048)?

==================================================================================================================
PRE-REGISTRATION (written 2026-10-21 ~11:40 CEST, before any SAE of this round was trained; user request relayed by the
orchestrator). Motivation (post hoc, therefore tested fresh): in the freeze round (scripts/eci/v2/freeze.py, seeds
6-11) F2-8192 had mice primary R~ 0.205 +- 0.048 vs F0-2048 seeds 6-25 0.119 +- 0.006.

ARMS (everything else = the frozen v2 recipe and the freeze-round data, trainer and scorer: freeze.py)
    F2-8192   x_t = [z_t, mean of the other kept tokens in the 3 x 3 window], Matryoshka 8192 (128/512/2048/8192)
    F0-2048   x_t = z_t, Matryoshka 2048 (128/512/2048)  = shipped v2 (reference)
    F0-8192   x_t = z_t, Matryoshka 8192                  = width control
SEEDS: 30-35 for every arm, in mice and in ants (never used before: rounds 1-4 used 0-25, freeze 6-25 / 6-11 / 6-17).
DATA / METRICS: freeze.py exactly. Mice v1: round-2 pool, 86,400 diagnosis frames, primary R~ = mean(R~ nose_nose,
    R~ nose_tail) (cross-fitted normalised R-precision, 10 random 12 / 12 pool splits). Ants v2 + v3: freeze pools and
    scored frames, primary R~ = mean over v2 / v3 of R~(groom_any); v3 on_lid_any reported.
INTERPRETABILITY: per arm the deployment seed (highest held-out FVE among seeds 30-32, label-free); nose_nose and
    nose_tail neurons picked on pools half 0 of split rnd0; their 24 top held-out frames (<= 1 per video): share labelled
    positive, tokens at >= half max of the CONTRIBUTION map (src/eci/v2/heatmap.py) and of the activation map.
RULE: adopt F2-8192 as v2.1 iff ALL of
    (a) mice primary R~: mean(F2-8192) - mean(F0-2048) >= +0.03 AND its Welch 95% CI excludes 0 (6 vs 6 seeds);
    (b) ants primary R~: the Welch 95% CI lower bound of mean(F2-8192) - mean(F0-2048) >= -0.01 (not worse by > 0.01);
    (c) top-24 labelled share of the deployment-seed neuron of F2-8192 >= that of F0-2048, for nose_nose AND nose_tail.
    F0-8192 is reported (F2-8192 vs F0-8192 isolates context from width) and never decides. If adopted, every domain is
    re-rolled with configs/eci_v2/representation_v2.1.json (context_window 3, W 8192, prefixes 128/512/2048/8192) through
    the same scripts/eci/v2/run.py pipeline and a manifest_v2.1.json.
==================================================================================================================

Steps: training / sheets use freeze.py (mtrain, atrain, msheets, asheets with --seeds 30-35 / --names); here:
    mscore --arms A --width W   (GPU) -> OUT/mice/score_confirm_<A>w<W>.json
    ascore --arms A --width W   (GPU) -> OUT/ants/score_confirm_<A>w<W>.json
    decide                       (CPU) -> results/vision/eci_v2/confirm/decision.json, table.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import freeze as F  # noqa: E402

SEEDS = tuple(range(30, 36))
ARMS = (('F2', 8192), ('F0', 2048), ('F0', 8192))
OUTC = F.OUT.parent / 'confirm'


def cmd_mscore(args):
    import torch
    dev = F.dev_of()
    St = F.mice_diag(dev)
    Y, unit, _ = F.mice_labels(St.F)
    sc = json.loads((F.N.OUT / 'score.json').read_text())
    units, SPL = sc['units'], {k_: np.array(v_) for k_, v_ in sc['splits'].items()}
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    half = {s_: h[uidx] for s_, h in SPL.items()}
    Yg = torch.from_numpy(Y).to(dev)
    arm, W = args.arms, args.width
    f = F.OUT / 'mice' / f'score_confirm_{arm}w{W}.json'
    res = json.loads(f.read_text()) if f.exists() else {}
    for s in SEEDS:
        name = f'{arm}_w{W}_s{s}'
        if name in res:
            continue
        sae, norm = F.load_sae('mice', arm, W, s, dev)
        X = St.codes(sae, norm, F.ARMS[arm]['w'], 'all')
        o = F.crossfit(X, Yg, half, F.MLAB)
        o['primary'] = float(np.mean([o['mean']['nose_nose']['R'], o['mean']['nose_tail']['R']]))
        res[name] = {'all': F.compact(o) | {'primary': o['primary']}, 'eval_fve': F.fve_of('mice', arm, W, s)}
        F.log(f'{name}: primary R~ {o["primary"]:.4f} | nn R~ {o["mean"]["nose_nose"]["R"]:.4f} | nt R~ '
              f'{o["mean"]["nose_tail"]["R"]:.4f}')
        f.write_text(json.dumps(res, indent=1, default=float))
        del X, sae
        torch.cuda.empty_cache()


def cmd_ascore(args):
    import torch
    dev = F.dev_of()
    St = F.ants_stores('scored', dev, on_gpu=True)
    lab = {}
    for ser, s_ in St.items():
        ok, lid_ok, Y, _ = F.ants_labels(ser)
        rows = s_.F.row.values
        obs = s_.F.obs.values
        units, SPL = F.splits_of(set(obs))
        uidx = pd.Series(range(len(units)), index=units).loc[obs].values
        labs = list(F.ALAB) if lid_ok[rows].all() else list(F.ALAB[:3])
        lab[ser] = (torch.from_numpy(Y[rows][:, :len(labs)]).to(dev), {k: h[uidx] for k, h in SPL.items()}, labs)
    arm, W = args.arms, args.width
    f = F.OUT / 'ants' / f'score_confirm_{arm}w{W}.json'
    res = json.loads(f.read_text()) if f.exists() else {}
    for s in SEEDS:
        name = f'{arm}_w{W}_s{s}'
        if name in res:
            continue
        sae, norm = F.load_sae('ants', arm, W, s, dev)
        out = {}
        for ser, St_ in St.items():
            Yg, h, labs = lab[ser]
            out[ser] = F.compact(F.crossfit(St_.codes(sae, norm, F.ARMS[arm]['w'], 'all'), Yg, h, labs))
        out['primary'] = float(np.mean([out[x]['mean']['groom_any']['R'] for x in F.ASER]))
        out['lid_any'] = out['v3']['mean']['on_lid_any']['R']
        out['eval_fve'] = F.fve_of('ants', arm, W, s)
        res[name] = out
        F.log(f'ants {name}: primary R~ {out["primary"]:.4f} | lid R~ {out["lid_any"]:.4f}')
        f.write_text(json.dumps(res, indent=1, default=float))


def cmd_decide(args):
    M, A = {}, {}
    for a, W in ARMS:
        for d, D_ in (('mice', M), ('ants', A)):
            f = F.OUT / d / f'score_confirm_{a}w{W}.json'
            if f.exists():
                D_.update(json.loads(f.read_text()))
    mp = lambda a, W, q='primary': [M[f'{a}_w{W}_s{s}']['all'][q] if q == 'primary' else  # noqa: E731
                                    M[f'{a}_w{W}_s{s}']['all']['mean'][q]['R'] for s in SEEDS if f'{a}_w{W}_s{s}' in M]
    ap = lambda a, W, q='primary': [A[f'{a}_w{W}_s{s}'][q] for s in SEEDS if f'{a}_w{W}_s{s}' in A]  # noqa: E731
    dec = {'rule': 'scripts/eci/v2/confirm.py docstring (pre-registered 2026-10-21 ~11:40)', 'seeds': list(SEEDS),
           'mice': {f'{a}_w{W}': {q: F.mean_sd(mp(a, W, q)) for q in ('primary', 'nose_nose', 'nose_tail')}
                    for a, W in ARMS},
           'ants': {f'{a}_w{W}': {q: F.mean_sd(ap(a, W, q)) for q in ('primary', 'lid_any')} for a, W in ARMS}}
    ma = F.welch(mp('F2', 8192), mp('F0', 2048))
    aa = F.welch(ap('F2', 8192), ap('F0', 2048))
    dec['a_mice_F2w8192_minus_F0w2048'] = ma
    dec['b_ants_F2w8192_minus_F0w2048'] = aa
    dec['ants_lid_F2w8192_minus_F0w2048'] = F.welch(ap('F2', 8192, 'lid_any'), ap('F0', 2048, 'lid_any'))
    dec['control_mice_F2w8192_minus_F0w8192'] = F.welch(mp('F2', 8192), mp('F0', 8192))
    dec['control_ants_F2w8192_minus_F0w8192'] = F.welch(ap('F2', 8192), ap('F0', 8192))
    L = json.loads((F.OUT / 'mice' / 'locality.json').read_text())
    dep = {}
    for a, W in (('F2', 8192), ('F0', 2048), ('F0', 8192)):
        fv = {s: F.fve_of('mice', a, W, s) for s in (30, 31, 32)}
        s = max(fv, key=fv.get)
        dep[f'{a}_w{W}'] = s
        for lb in ('nose_nose', 'nose_tail'):
            k = f'{a}_w{W}_s{s}|{lb}'
            if k in L:
                dec.setdefault('sheets', {})[k] = {q: L[k]['summary'][q] for q in ('share_label_pos', 'contrib_n_half_max',
                                                                                   'act_n_half_max', 'contrib_peak_fdark')}
    dec['deployment_seeds'] = dep
    sh = dec.get('sheets', {})
    c_ok = all(sh.get(f'F2_w8192_s{dep["F2_w8192"]}|{lb}', {}).get('share_label_pos', -1) >=
               sh.get(f'F0_w2048_s{dep["F0_w2048"]}|{lb}', {}).get('share_label_pos', 9) for lb in ('nose_nose', 'nose_tail'))
    dec['criteria'] = {'a': bool(ma[0] >= 0.03 and ma[1] > 0), 'b': bool(aa[1] >= -0.01), 'c': bool(c_ok)}
    dec['ADOPT_v2.1'] = all(dec['criteria'].values())
    OUTC.mkdir(parents=True, exist_ok=True)
    (OUTC / 'decision.json').write_text(json.dumps(dec, indent=1, default=float))
    print(json.dumps(dec, indent=1, default=float))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('mscore', 'ascore', 'decide'))
    p.add_argument('--arms', default='F2')
    p.add_argument('--width', type=int, default=8192)
    args = p.parse_args()
    {'mscore': cmd_mscore, 'ascore': cmd_ascore, 'decide': cmd_decide}[args.step](args)


if __name__ == '__main__':
    main()
