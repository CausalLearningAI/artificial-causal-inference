"""
ECI v2.0 per-label DEPLOYMENT GATE (Part A) and the ants-gap variants (Part B).

==================================================================================================================
PRE-REGISTRATION (written 2026-10-29 BEFORE any top-1% precision or variant score existed; user's gate relayed by the
coordinator. Known: AUROC / nAP per cell of v1 (11 seeds) and v2.0 (v20_vs_v1 align.md).)
FRAMES / SCORER: the evaluation audit (scripts/eci/v2_eval/eval.py): frame tables, splits atlas + rnd0-4 (both
    directions), neuron eligibility, score_cells (cross-fitted best single neuron by AUROC). Cells = (series, label):
    mice nn (mutual), np (directional), nt (nose-to-tail), nose_nose (nn | np); ants Y2F (yellow grooming), B2F (blue),
    groom_any, YOL / BOL / FOL (on lid; focal) and onlid_any, for every series eval.py scores (v2 / v3 / v4 / v4_infested).
MODELS: v1 = s0-10 (raw frame max); v2.0 = final PASS B model, readouts P(on) (PRIMARY) and continuous (0.6 s).
TOP-1% PRECISION: for every cross-fit direction, the neuron picked by AUROC on one half; on the other half the frames
    with the top ceil(1%) codes of that neuron (rank order, ties broken by a fixed random order, rng 0) -> share carrying
    the label (precision) and precision / base rate of that half (lift); averaged over the directions.
PASS (per cell, v2.0 P(on); continuous reported too): (1) AUROC not worse than v1: v2 - mean(v1 seeds), t CI (df 10),
    upper bound >= -0.01; (2) top-1% precision >= 3 x base rate; (3) top-1% precision >= mean_v1 - 0.05.
    Cells that fail eval.py's bout eligibility (>= 20 bouts and >= 3 positive videos per atlas half; on-lid labels) are
    scored DESCRIPTIVELY with their bout counts and are not part of the decision.
MICE DISTINCT: nn and nt best neurons of v2.0 = the neuron picked most often over the AUROC directions; distinct iff they
    differ (reported for P(on) and continuous).
PART B variants (SAE level, 3 seeds, the v2.0 recipe otherwise): context mask = NOT re-run (the diag ablation V1M_d2 and
    B_d3l20dil already measured it; reported); DYN = token input [z_t, z_t - z_(t - 1 s)] at the same grid position
    (difference 0 when that position is not kept 1 s earlier), ants trained on the full1 1 fps N2 store, mice on the
    behaviour-suite 5 fps probe dense N2 store (lag 5 frames); scored with this gate on the eval frames (ants full1 N2
    tokens, 1 fps lag 1 frame; mice v2_prob dense N2 tokens, lag 5 frames; mice readout 0.6 s smooth, ants raw) and the
    dynamic share @256 (v20_vs_v1 'dyn' rule). ADOPT a variant only if it passes MORE ants cells than v2.0 AND loses no
    mice cell that v2.0 passes; otherwise v2.0 stays frozen.
AMENDMENT 1 (2026-10-29, user additions relayed by the coordinator; written BEFORE any variant was trained and before
    any proximity number existed; the Part A gate scores of v1 / v2.0 were running, none read):
  (a) DYN is REPLACED by DYN3: SAE input = [z_t, z_t - z_(t-1), std_(t-2..t+2)(z)] at the same grid position (5 fps,
      consecutive frames; difference 0 if the position is not kept at t-1; std over the window frames where the position
      is kept, 0 if only t), each 1024 block normalised (per-block TokenNorm). Trained on the suite's contiguous
      all-token 5 fps probe stores (mice and ants), seeds 970-972, train tokens capped at 12M. No 30 fps.
  (b) ANTS variant scoring needs 5 fps context, so the ants comparison of v1 / v2.0 / DYN3 uses the v2_prob Tier B
      probe frames (contiguous 5 fps, labelled; ants v2 44 x 2 min, v3 107 x 1 min) with the same scorer / splits /
      eligibility and this gate (v1 s0-10 codes on those frames exist); mice uses the eval frames (already 5 fps).
      ADOPT DYN3 iff it passes more ants cells than v2.0 on that table AND loses no mice cell v2.0 passes.
  (c) MASK domain invariance: a mask variant is adopted only if it passes in ants AND mice. The context (dilated) mask is
      NOT re-run: the diag ablation measured it twice (B_d3l20dil one-cell dilation; V1M_d2 = v1's exact dilated mask on
      the v2 encoder: ants v2 groom R~ -0.082 vs tight, 50-65% background neurons); reported instead.
  (d) PROXIMITY-MATCHED TEST (no training): proxy per frame = minimum Euclidean distance (in patches) between distinct
      8-connected components of the tight-body token grid (tokens with >= 25% frozen-cue animal pixels, fr_raw >= 0.25),
      0 when the grid has exactly one component (all visible animals touching), NaN when it has none. Cells: ants groom_any / Y2F / B2F, mice nn / np / nt / nose_nose. For the AUROC-picked best
      neuron of each cross-fit direction, on the held-out half: strata = deciles of the proxy over the cell's frames
      (unique quantile edges; ties form one stratum), AUROC inside every stratum with >= 10 positive and >= 10 negative
      frames, mean weighted by positives = STRATIFIED AUROC; averaged over directions. PROXY AUROC = AUROC of -proxy
      (closer = higher) on the cell's frames. Reading: a neuron that only encodes 'bodies close' drops to ~0.5 within
      strata. Reported for v1 (11 seeds), v2.0 (P(on), continuous) and DYN3; descriptive (no pass rule).
AMENDMENT 2 (2026-10-29, user; before any variant score): GPU cap 120 GPU-h. On-lid labels (YOL / BOL / FOL, onlid_any)
    are REMOVED from the gate and the tables; ants gate labels = Y2F, B2F, groom_any. DYN3 is ON HOLD (not trained;
    jobs cancelled before any code / score). The CONTEXT-MASK variant CTX is run after all (store missing -> encode):
    CTX = the v2.0 N2 token set dilated by one token (3 x 3 neighbourhood on the DINOv3 grid = v1's one-patch ring),
    same encoder / resolution / canonical frame, frame max over the kept tokens. Tokens encoded for: the suite's 5 fps
    probe frames (training, label-free) and the eval.py frames (scoring); SAE = the v2.0 recipe (2048, 128/256/512/2048,
    k16, 30k), seeds 980-982, train tokens capped at 16M (uniform, rng 0). Ants first, then mice (domain-invariant rule:
    adopted only if it passes in both, AMENDMENT 1 (c)); scored with this gate on the eval.py frames, the proximity test
    and the dynamic share @256 (suite probe frames). ADOPT CTX iff it passes more ants cells than v2.0 AND loses no mice
    cell v2.0 passes.
AMENDMENT 3 (2026-10-29, user-approved design; written BEFORE any of its features / SAEs / scores existed):
  DYNAMIC-INPUT arms (scripts/eci/v2/v20_dyn2.py). x_t = a patch's DINOv3 token at frame t (5 fps, fixed position, N2
  tokens). Block 1: c_t = (x_(t+1) - x_(t-1)) / 2 (one-sided x_(t+1) - x_t or x_t - x_(t-1) when one neighbour is
  missing (video edge / position not kept), 0 when both are), U1 = top 128 PCA directions of c (per domain, ~1M random
  training patches), block1 = U1 c_t. Block 2: window t-2..t+2 (available frames), m_t = window mean, r = x_(t+k) - m_t,
  U2 = top 128 PCA directions of r, block2 = sqrt(mean_k (U2 r_(t+k))^2) (>= 0; 0 with a single frame). Each block
  rescaled to the mean squared norm of x (on the PCA sample), then one global TokenNorm. ARMS: X0 = [x] (control, same
  frames), A = [x, block1] (1152), B = [x, block1, block2] (1280); Matryoshka BatchTopK 2048, k16, 128/256/512/2048,
  30k; 3 seeds each (990-992); trained on the suite's contiguous 5 fps probe stores (v2_behav, ants and mice), 12M
  train tokens (uniform, rng 0). SCORED on the v2_prob Tier B probe frames (mice = the eval.py frames; ants = the 5 fps
  probe table of v20_prox.py; readout mice 0.6 s smooth, ants raw) with this gate (on-lid removed), the proximity test,
  the dynamic share @256 (suite probe frames) and the number of neurons whose decoder squared-norm share on blocks 1 + 2
  exceeds 0.25. Adoption of an arm is NOT decided here (no re-encoding; the user decides on these numbers).
POST-HOC AMENDMENT 4 (2026-10-29, user, AFTER the Part A gate results were seen): criterion (2) 'top-1% precision >= 3 x
  base rate' is unreachable when the base rate > 1/3; it is REPLACED by 'top-1% precision >= base + 0.5 (1 - base)'.
  Both the original (PASS) and the amended (PASS_am4) decisions are reported; criteria (1) and (3) unchanged.
POST-HOC AMENDMENT 5 (2026-10-29, user's FINAL rule, after AMENDMENT 4 results were seen): criterion (2) =
  'top-1% precision >= min(3 x base rate, base + 0.5 (1 - base))'; (1) and (3) unchanged. Reported side by side:
  PASS (original 3x), PASS_am4 (base + 0.5 (1 - base)), PASS_final (this rule).
AMENDMENT 6 (2026-10-29, user-approved motion retry; written BEFORE any of these SAEs existed): arms A and B of
  AMENDMENT 3 with the motion blocks DOWN-WEIGHTED: with x keeping its scale, the motion blocks (A: block1; B: block1 and
  block2, equal shares) are rescaled so that together they make w = 10% (arms A10 / B10) or 20% (A20 / B20) of the mean
  squared norm of the input; same probe stores, same token sample (rng 0), same PCA (U1, U2), same recipe, seeds
  990-992; X0 = the existing control. Scored like AMENDMENT 3 (gate final rule, mean AUROC, dynamic share @256,
  neurons with decoder squared-norm share > 0.25 on the motion blocks). No adoption rule here: reported to the user.
==================================================================================================================
Steps: gate (CPU) ; dyn_train --dom D (GPU) ; dyn_codes --dom D (GPU) ; dyn_gate --dom D (CPU)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2', 'scripts/eci/v2_eval', 'scripts/eci/v2_prob', 'scripts/eci/v2_behav'):
    sys.path.insert(0, str(REPO / p))
V = REPO / 'results/vision'
OUT = V / 'eci_v2_0B/gate'
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
BEE = EC / 'v2_0B/gate'
ONLID = ('YOL', 'BOL', 'FOL', 'onlid_any', 'lid_any3')  # removed from the gate (AMENDMENT 2)
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def one_vs(x, b):
    from scipy import stats
    b = np.asarray(b, float)
    h = stats.t.ppf(0.975, len(b) - 1) * b.std(ddof=1) / np.sqrt(len(b))
    return [float(x - b.mean()), float(x - b.mean() - h), float(x - b.mean() + h)]


def top1(F, X, use, SP, cells_res):
    """{cell: {'prec', 'lift', 'base'}} for the AUROC-picked neurons of score_cells output cells_res."""
    import eval as EV
    out = {}
    rng = np.random.default_rng(0)
    for cell, r in cells_res.items():
        ser, b = cell.split('|')
        vm = F[f'v_{b}'].values & (F.series.values == ser) & use
        idx = np.flatnonzero(vm)
        y = F[f'y_{b}'].values[idx]
        units = F.unit.values[idx]
        tb = rng.random(len(idx))
        P, L, B = [], [], []
        for nm, h, j in r['AUROC_neurons']:
            half = np.array([SP[nm][u] for u in units])
            o = np.flatnonzero(half == 1 - h)
            if len(o) == 0 or y[o].sum() == 0:
                continue
            c = np.asarray(X[idx[o], j], np.float64)
            n = max(1, int(np.ceil(0.01 * len(o))))
            top = np.lexsort((tb[o], -c))[:n]
            p_ = float(y[o][top].mean())
            P.append(p_)
            B.append(float(y[o].mean()))
            L.append(p_ / max(B[-1], 1e-12))
        out[cell] = {'prec': float(np.mean(P)) if P else None, 'lift': float(np.mean(L)) if L else None,
                     'base': float(np.mean(B)) if B else None}
    return out


def score(dom, X, use, F, SP):
    import eval as EV
    cr = EV.score_cells(F, X, use, SP)
    t = top1(F, X, use, SP, cr)
    for c in cr:
        cr[c].update({f'top1_{k}': v for k, v in t[c].items()})
        js = pd.Series([x[2] for x in cr[c]['AUROC_neurons']]).value_counts()
        cr[c]['best_neuron'] = int(js.index[0]) if len(js) else None
    return cr


def v1_X(dom, s, F):
    import eval as EV
    return EV.load_dict(dom, f'v1_s{s}', F)


def cmd_gate(a):
    import analysis as AN
    import eval as EV
    os.environ.setdefault('V2_OUT', str(V / 'eci_v2_0B'))
    os.environ.setdefault('V2_BEE', str(EC / 'v2_0B'))
    import v20_eval as VE
    OUT.mkdir(parents=True, exist_ok=True)
    for dom in a.doms.split(','):
        F = EV.frame_table(dom)
        SP = AN.splits(F.unit.values, EV.SPLITS)
        elig = EV.row_counts(dom, F, SP)
        res = {'eligibility': elig, 'v1': {}, 'v2': {}}
        for s in range(11):
            f = OUT / f'{dom}_v1_s{s}.json'
            if not f.exists():
                X, use = v1_X(dom, s, F)
                f.write_text(json.dumps(score(dom, X, use, F, SP)))
            res['v1'][s] = json.loads(f.read_text())
            log(f'{dom} v1 s{s} done')
        for k, suf in (('hmm', '_hmm'), ('smooth', '_smooth')):
            f = OUT / f'{dom}_v2_{k}.json'
            if not f.exists():
                X, use = VE.deliverable_codes(dom, F, suf)
                f.write_text(json.dumps(score(dom, X, use, F, SP)))
            res['v2'][k] = json.loads(f.read_text())
        (OUT / f'{dom}_scores.json').write_text(json.dumps(res))
        log(f'{dom}: gate scores written')
    table(a)


def gate_rows(dom, v1, v2, elig):
    rows = {}
    for cell in v2:
        v1c = [v1[s][cell] for s in v1 if cell in v1[s] and v1[s][cell].get('AUROC') is not None]
        if not v1c or v2[cell].get('AUROC') is None:
            continue
        r = {'eligible': bool(elig.get(cell, {}).get('eligible')), 'bouts': [elig.get(cell, {}).get('h0_bouts'), elig.get(cell, {}).get('h1_bouts')],
             'base': v2[cell].get('top1_base'), 'best_neuron': v2[cell].get('best_neuron')}
        for M in ('AUROC', 'nAP', 'top1_prec', 'top1_lift'):
            r[f'v1_{M}'] = float(np.mean([c[M] for c in v1c if c.get(M) is not None]))
            r[f'v2_{M}'] = v2[cell].get(M)
        r['dAUROC'] = one_vs(v2[cell]['AUROC'], [c['AUROC'] for c in v1c])
        c1 = r['dAUROC'][2] >= -0.01
        c2 = (r['v2_top1_lift'] or 0) >= 3
        c2b = (r['v2_top1_prec'] or 0) >= r['base'] + 0.5 * (1 - r['base'])
        c2f = (r['v2_top1_prec'] or 0) >= min(3 * r['base'], r['base'] + 0.5 * (1 - r['base']))
        c3 = (r['v2_top1_prec'] or 0) >= r['v1_top1_prec'] - 0.05
        r.update(c1_auroc=bool(c1), c2_lift3=bool(c2), c2b_half=bool(c2b), c3_prec=bool(c3), PASS=bool(c1 and c2 and c3),
                 PASS_am4=bool(c1 and c2b and c3), c2f_final=bool(c2f), PASS_final=bool(c1 and c2f and c3))
        rows[cell] = r
    return rows


def table(a):
    L = ['| domain | cell | elig (bouts h0/h1) | base | v1 AUROC | v2 AUROC (P(on)) | dAUROC [CI] | v1 nAP | v2 nAP | v1 top1% prec | v2 top1% prec (lift) | v2 cont top1% prec | PASS P(on) orig/am4/final | PASS cont orig/am4/final |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    summ = {}
    for dom in a.doms.split(','):
        S = json.loads((OUT / f'{dom}_scores.json').read_text())
        v1 = S['v1']
        R = {k: gate_rows(dom, v1, S['v2'][k], S['eligibility']) for k in ('hmm', 'smooth')}
        summ[dom] = R
        for cell, r in R['hmm'].items():
            if cell.split('|')[1] in ONLID:
                continue
            rc = R['smooth'].get(cell, {})
            d = r['dAUROC']
            L.append(f'| {dom} | {cell} | {"Y" if r["eligible"] else "N (descr.)"} ({r["bouts"][0]}/{r["bouts"][1]}) | {r["base"]:.4f} | '
                     f'{r["v1_AUROC"]:.3f} | {r["v2_AUROC"]:.3f} | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | {r["v1_nAP"]:.3f} | '
                     f'{r["v2_nAP"]:.3f} | {r["v1_top1_prec"]:.3f} | {r["v2_top1_prec"]:.3f} ({r["v2_top1_lift"]:.1f}x) | '
                     f'{rc.get("v2_top1_prec", float("nan")):.3f} | {"P" if r["PASS"] else "F"}/{"P" if r["PASS_am4"] else "F"}/{"P" if r["PASS_final"] else "F"} | {"P" if rc.get("PASS") else "F"}/{"P" if rc.get("PASS_am4") else "F"}/{"P" if rc.get("PASS_final") else "F"} |')
        if dom == 'mice':
            for k in ('hmm', 'smooth'):
                nn, nt = R[k]['v1|nn']['best_neuron'], R[k]['v1|nt']['best_neuron']
                summ[dom][f'distinct_nn_nt_{k}'] = {'nn': nn, 'nt': nt, 'distinct': nn != nt}
    (OUT / 'gate.json').write_text(json.dumps(summ, indent=1, default=float))
    (OUT / 'gate.md').write_text('\n'.join(L))
    print('\n'.join(L))
    for dom, R in summ.items():
        for k in ('hmm', 'smooth'):
            el = {c: r for c, r in R[k].items() if r['eligible'] and c.split('|')[1] not in ONLID}
            log(f'{dom} {k}: PASS {sum(r["PASS"] for r in el.values())} (am4 {sum(r["PASS_am4"] for r in el.values())}, final {sum(r["PASS_final"] for r in el.values())}) / {len(el)} eligible cells; '
                + (json.dumps(R.get(f'distinct_nn_nt_{k}')) if dom == 'mice' else ''))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('gate', 'table'))
    p.add_argument('--doms', default='mice,ants')
    a = p.parse_args()
    {'gate': cmd_gate, 'table': table}[a.step](a)


if __name__ == '__main__':
    main()
