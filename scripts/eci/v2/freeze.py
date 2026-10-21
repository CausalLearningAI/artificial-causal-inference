"""
ECI v2 FREEZE ROUND (step 1). Which spatial-context design and which dictionary go into the frozen v2 pipeline?
Labels are used ONLY to score / select among the pre-registered arms, never in training.

==================================================================================================================
PRE-REGISTRATION (written 2026-10-20 ~14:30 CEST, BEFORE any score of this round existed).
Numbers known beforehand (rounds 3-4, results/vision/eci_nnnt/r3/): single-SAE scores of seeds 0-5 of R5 (= F1 here),
R7 (= F3), S2 (= the F0 recipe): mice primary R~ 0.117 / 0.141 / 0.107; the 10- and 20-seed UNION scores of R7 and
R0 (seeds 6-25): 20-seed primary R~ 0.204 (R7) / 0.130 (R0). NOT known: any single-seed score of seeds 6-25, any
3 x 3 context score, any 8192-wide context dictionary, any ants or frogs score with the tight mask.

FIXED PARTS (accepted priors): DINOv3-L/16 block 20 (bf16, SDPA, revision pinned in src/eci/v2/encode.py), native
resolution (mice 1024 px = 64 x 64 grid; ants v2 768 px = 48 x 48, v3 832 px = 52 x 52; frogs 800 px = 50 x 50),
canonical rotation (mice: odour corner top right), TIGHT mask (token kept iff >= 25% of its 16 x 16 patch are animal
pixels by the domain's pixel cue, src/eci/v2/mask.py), no temporal channel, frame value = max over the readout tokens,
Matryoshka BatchTopK SAE k = 16, theta* calibrated on held-out CALIBRATION videos (mean L0 = k), FVE on EVALUATION
videos, recipe 'B' = 2048 latents (prefixes 128 / 512 / 2048), 30,000 steps x 4096 tokens, Adam 5e-4 (r3.train_arm).
Wide variant = 8192 latents (prefixes 128 / 512 / 2048 / 8192), else identical.

ARMS (token input x_t; readout tokens)
    F0  x_t = z_t (no context); readout over all kept tokens.            [= r3 'R0'; mice checkpoints reused]
    F1  x_t = [z_t, mean of the other kept tokens in the 5 x 5 window]; readout over all kept tokens.  [= r3 'R5']
    F2  as F1 with a 3 x 3 window.
    F3  as F1, trained and read out on EDGE tokens only (mask boundary, chessboard distance 1) [= r3 'R7'; mice
        checkpoints reused]. Interaction REFERENCE only: the edge readout is interaction specific and would hide solo
        behaviours (frogs, rearing, self-grooming), so F3 is NOT eligible for v2.
SEEDS: mice 6-25 (20 per arm, identical for every arm; F0 / F3 = the existing r3 round-4 checkpoints, trained by the same
    code on the same pool); ants 6-17 (12 per arm). Wide 8192: seeds 6-11 for F1 and F2 (mice and ants).

DATA (identical frames / splits for every arm)
  MICE v1: train pool = round-2 pool (bursts, all 432 videos, 62,208 frames; 'train' / 'calib' / 'eval' videos); scored
    frames = round-2 diagnosis set (86,400 = every 10th 5 fps frame of the 144 annotated videos, disjoint from the pool).
    Labels nose_nose = Y_nn | Y_np, nose_tail = Y_nt (+ nn_mutual, np_directional). Frame sets 'all' (primary) and
    'consistent' = videos of annotators SG / SD / MR / CP (secondary). Units = 24 pools, 10 random 12 / 12 splits.
  ANTS v2 + v3 (one SAE on both series; v4 / v4_infested have no labels and join only in the rollout): train pool =
    bursts of 10 consecutive 5 fps frames at random starts (rng 7) in every video of the r4 pooled plans (ants_v2p 44
    videos, ants_v3p 107), frames with k = 0 mod 10 dropped, ~50,000 frames per series (v2 124 bursts / video, v3 52),
    plan roles. Scored frames = every labelled frame with k = 0 mod 10 (as the resolution sweep: v2 13,200, v3 32,100).
    Labels groom_any = Y_Y2F | Y_B2F (primary), groom_yellow, groom_blue; on_lid_any = Y_YOL | Y_BOL | Y_FOL (v3 only),
    on_lid_yellow / blue / focal. Units = videos of each series, 10 random halves.
  FROGS juveniles (no labels): health and locality only (step 'fhealth').

METRICS (src/eci/v2/score.py = the rounds 1-4 scorer): R~ = cross-fitted normalised R-precision (neuron with the best
R~ on one half of the units, scored on the other half, both directions, mean over the 10 random splits); cf AUROC.
    mice primary = mean(R~ nose_nose, R~ nose_tail) per SAE; ants primary = mean over v2 and v3 of R~(groom_any);
    ants secondary = R~(on_lid_any) on v3.
INTERPRETABILITY (descriptive): neuron selected on pools / videos half 0 of split rnd0, its 24 highest held-out (half 1)
    frames (<= 1 per video): share labelled positive, tokens at >= half max of the CONTRIBUTION map (src/eci/v2/
    heatmap.py: each firing readout token's activation shared exactly among the source tokens of its input, own token
    and context neighbours, by their positive linear terms) and of the activation map, peak cell animal-pixel
    fraction, activation mass on animal pixels.

DECISION RULE
  1. Context (single 2048 dictionaries, 20 mice seeds): best = the arm with the highest mean mice primary R~ among
     F0-F3. Eligible arms in order of cleanliness: F0 (no context) < F2 (3 x 3) < F1 (5 x 5). v2 context = the FIRST
     eligible arm whose mean mice primary R~ >= best - 0.02 (tolerance 0.02 R~) and that does NOT LOSE on ants; if none
     is within tolerance, the eligible arm with the highest mean. LOSES on ants = (arm - F0) mean ants primary R~ <= -0.02
     with the Welch 95% CI excluding 0, or the same for v3 on_lid_any R~ (F0 cannot lose against itself).
  2. Dictionary, for the better of F1 / F2 by mice primary (and for the chosen arm if it is F0, with the F0 seeds):
     single 2048 (seeds 6-25) vs single 8192 (seeds 6-11) vs union of N seeds' 2048 codes side by side (N = 5: the
     four disjoint unions 6-10, 11-15, 16-20, 21-25; N = 20: 6-25; the neuron, and so the seed, picked on one half).
     Width: 8192 replaces 2048 iff its 6-seed mean mice primary R~ beats 2048 seeds 6-11 by >= +0.02 with the Welch
     95% CI excluding 0. Union (clean-pipeline principle: prefer ONE dictionary) is adopted only iff the mean of the
     four 5-seed unions >= the chosen single width's seed mean + 0.03 AND the 20-seed union >= that mean + 0.05
     (mice primary R~), AND the union does not lose on ants (same ants rule, unions of ants seeds 6-10 / 11-15 vs the
     single seed mean).
  3. Deployment seed of a single dictionary (label-free, as every rollout): the highest held-out FVE among its first
     three seeds (6, 7, 8).
  Reported with the decision, never deciding: nose_nose and nose_tail separately, AUROC, the consistent-annotator set,
  ants per series and per label, interpretability numbers, frogs health / locality.
==================================================================================================================

Steps (scripts/eci/v2/freeze.sh runs each in Slurm; heavy inputs staged to $LOCAL_DIR)
    mtrain   (GPU) mice SAEs of F1 / F2 (--arms, --seeds, --width) on the round-2 pool -> BEE/mice/sae/<arm>_w<W>_s<seed>
    mscore   (GPU) one arm: codes of every seed on the diagnosis frames, single scores, unions -> OUT/mice/score_<arm>.json
    msheets  (GPU) heat maps of the named SAEs / unions -> OUT/mice/sheets/, OUT/mice/locality.json
    aprep    (GPU) ants token store of one series (--series v2|v3, --kind pool|scored) -> BEE/ants/<kind>_<series>/
    atrain   (GPU) ants SAEs (--arms, --seeds, --width) -> BEE/ants/sae/
    ascore   (GPU) one arm -> OUT/ants/score_<arm>.json
    asheets  (GPU) -> OUT/ants/sheets/, OUT/ants/locality.json
    fprep / fhealth (GPU) frogs juveniles pool, 3 seeds per eligible arm, health + locality -> OUT/frogs/health.json
    decide   (CPU) -> OUT/decision.json, OUT/table.md
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
sys.path.insert(0, str(REPO / 'scripts/eci'))
sys.path.insert(0, str(REPO / 'scripts/eci/nnnt'))
import nnnt as N  # noqa: E402  (read-only: plan, decode helpers, contact sheet)
import r2 as R2  # noqa: E402  (read-only: label reader, token loader)
import r3 as R3  # noqa: E402  (read-only: Store, train_arm, recipe B)

from src.eci.v2.context import edge_distance, sae_input  # noqa: E402
from src.eci.v2.heatmap import locality, neuron_maps  # noqa: E402
from src.eci.v2.score import block_of, crossfit, splits_of, welch  # noqa: E402

R4 = N.R4
OUT = Path(os.environ.get('V2F_OUT') or REPO / 'results/vision/eci_v2/freeze')
BEE = Path(os.environ.get('V2F_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2/freeze')
log = N.log
D = 1024
ARMS = {'F0': dict(w=0, sel='all'), 'F1': dict(w=5, sel='all'), 'F2': dict(w=3, sel='all'),
        'F3': dict(w=5, sel='edge')}
ELIGIBLE = ('F0', 'F2', 'F1')  # order of cleanliness
MICE_REUSE = {'F0': 'R0', 'F3': 'R7'}  # r3 round-4 checkpoints, seeds 6-25
SEEDS_M = tuple(range(6, 26))
SEEDS_A = tuple(range(6, 18))
SEEDS_W = tuple(range(6, 12))
WIDTHS = {2048: (128, 512, 2048), 8192: (128, 512, 2048, 8192)}
U5 = ((6, 10), (11, 15), (16, 20), (21, 25))
U5_ANTS = ((6, 10), (11, 15))
CONSISTENT = ('SG', 'SD', 'MR', 'CP')
MLAB = N.LABELS  # nose_nose, nose_tail, nn_mutual, np_directional
ALAB = ('groom_any', 'groom_yellow', 'groom_blue', 'on_lid_any', 'on_lid_yellow', 'on_lid_blue', 'on_lid_focal')


def cfg_of(arm, W):
    return {**R3.BASE, 'W': W, 'prefixes': WIDTHS[W], 'prep': f'ctx{ARMS[arm]["w"]}' if ARMS[arm]['w'] else 'id',
            'sel': ARMS[arm]['sel'], 'sampler': 'uniform', 'arm_v2': arm}


def sae_dir(dom, arm, W, s):
    if dom == 'mice' and arm in MICE_REUSE and W == 2048 and s in SEEDS_M:
        return R3.BEE3 / 'sae' / f'{MICE_REUSE[arm]}_s{s}'
    return BEE / dom / 'sae' / f'{arm}_w{W}_s{s}'


def load_sae(dom, arm, W, s, dev):
    from src.eci.sae import load_sae as ls
    sae, norm, ck = ls(sae_dir(dom, arm, W, s) / 'sae_theta.pt', dev)
    return sae, norm


def gpu_line():
    import torch
    return f'GPU {torch.cuda.get_device_name(0)}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB'


# ---------------------------------------------------------------------------------------------- token stores
class TokStore:
    """A token store: frames.parquet, tok_frame.npy, tok_pos.npy, tok.f16 [, tok_fdark.npy, tok_dist.npy]; G fixed per
    store. Iterates frame-aligned chunks of kept tokens on the GPU."""

    def __init__(self, root, G, dev, on_gpu=True, dist=None):
        import torch
        self.F = pd.read_parquet(root / 'frames.parquet')
        self.tf = np.load(root / 'tok_frame.npy')
        self.pos = np.load(root / 'tok_pos.npy').astype(np.int64)
        self.tok = np.memmap(root / 'tok.f16', np.float16, 'r', shape=(len(self.tf), D))
        self.fd = np.load(root / 'tok_fdark.npy').astype(np.float32) if (root / 'tok_fdark.npy').exists() else None
        self.nf, self.G, self.dev = len(self.F), G, dev
        self.starts = np.r_[0, np.cumsum(np.bincount(self.tf, minlength=self.nf))]
        if dist is not None:
            self.dist = dist
        elif (root / 'tok_dist.npy').exists():
            self.dist = np.load(root / 'tok_dist.npy')
        else:
            self.dist = None
        self.T = R2.tokens_to(torch, self.tok, dev) if on_gpu else None

    def chunks(self, frames_per=48):
        import torch
        for f0 in range(0, self.nf, frames_per):
            f1 = min(self.nf, f0 + frames_per)
            a, b = int(self.starts[f0]), int(self.starts[f1])
            if a == b:
                continue
            z = R2.chunk_of(self.T, a, b, self.dev) if self.T is not None else \
                torch.from_numpy(np.ascontiguousarray(self.tok[a:b])).to(self.dev)
            yield f0, f1, a, b, z

    def inputs(self, w, tok_sel, frames_per=48):
        """SAE input rows (fp16 cuda) of the tokens where tok_sel (numpy bool per token) holds."""
        import torch
        out = torch.empty((int(tok_sel.sum()), D * (2 if w else 1)), dtype=torch.float16, device=self.dev)
        o = 0
        for f0, f1, a, b, z in self.chunks(frames_per):
            m = tok_sel[a:b]
            if not m.any():
                continue
            mt = torch.from_numpy(m).to(self.dev)
            fl = torch.from_numpy(self.tf[a:b] - f0).to(self.dev)
            ps = torch.from_numpy(self.pos[a:b]).to(self.dev)
            x = sae_input(z, ps, fl, f1 - f0, self.G, w)[mt].half()
            out[o:o + len(x)] = x
            o += len(x)
        assert o == len(out)
        return out

    def codes(self, sae, norm, w, sel):
        """Frame codes (nf, W) fp16 cuda: max over the readout tokens (sel 'all' | 'edge')."""
        import torch
        W = sae.n_latents
        X = torch.zeros((self.nf, W), dtype=torch.float16, device=self.dev)
        edge = torch.from_numpy(self.dist == 1).to(self.dev) if sel == 'edge' else None
        fp = max(4, min(48, int(2**27 // W // 160)))
        with torch.no_grad():
            for f0, f1, a, b, z in self.chunks(fp):
                fl = torch.from_numpy(self.tf[a:b] - f0).to(self.dev)
                ps = torch.from_numpy(self.pos[a:b]).to(self.dev)
                x = sae_input(z, ps, fl, f1 - f0, self.G, w)
                if edge is not None:
                    m = edge[a:b]
                    x, fl = x[m], fl[m]
                c = sae.encode(norm(x), mode='threshold')
                X[f0:f1] = torch.zeros((f1 - f0, W), device=self.dev).index_reduce_(0, fl, c, 'amax',
                                                                                     include_self=True).half()
        return X


# ---------------------------------------------------------------------------------------------- mice
def mice_pool(dev):
    loc = N.local_dir()
    R4.atomic_copy(R2.BEE2 / 'pool', loc / 'pool')
    g = np.load(R3.BEE3 / 'geom_pool.npz')
    return TokStore(loc / 'pool', 64, dev, on_gpu=False, dist=g['dist'])


def mice_diag(dev):
    loc = N.local_dir()
    R4.atomic_copy(R2.BEE2 / 'diag', loc / 'diag')
    g = np.load(R3.BEE3 / 'geom_diag.npz')
    return TokStore(loc / 'diag', 64, dev, on_gpu=True, dist=g['dist'])


def train_set(St, arms, seeds, W, dom, args, roles_of_token):
    """Train every (arm, seed) not yet on disk; one input stream per arm."""
    import torch
    res_f = OUT / dom / f'train_{args.tag or "_".join(arms)}_w{W}.json'
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    for arm in arms:
        todo = [s for s in seeds if args.force or not (sae_dir(dom, arm, W, s) / 'sae_theta.pt').exists()]
        if not todo:
            log(f'{arm} w{W}: all seeds on disk')
            continue
        a = ARMS[arm]
        sel = np.ones(len(St.tf), bool) if a['sel'] == 'all' else (St.dist == 1)
        t0 = time.time()
        X = {r: St.inputs(a['w'], sel & (roles_of_token == r)) for r in ('train', 'calib', 'eval')}
        X['calib'], X['eval'] = X['calib'].cpu(), X['eval'].cpu()
        log(f'{dom} {arm} w{W}: ' + ', '.join(f'{r} {len(x):,}' for r, x in X.items()) +
            f' tokens x {X["train"].shape[1]} built in {time.time() - t0:.0f}s; {gpu_line()}')
        for s in todo:
            od = sae_dir(dom, arm, W, s)
            cfg = cfg_of(arm, W)
            if args.steps:
                cfg['steps'] = args.steps
            ti = R3.train_arm(torch, cfg, s, X['train'], X['calib'], X['eval'], od, dev_of())
            res[f'{arm}_w{W}_s{s}'] = {'cfg': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                                       **{k: ti[k] for k in ('train_time_s', 'steps_per_s', 'n_train_tokens', 'theta_star',
                                                              'eval_fve', 'eval_l0', 'dead_at_end_train')}}
            log(f'{dom} {arm}_w{W}_s{s}: {ti["train_time_s"]:.0f}s, theta* {ti["theta_star"]:.4f}, eval FVE '
                f'{ti["eval_fve"]:.4f}, L0 {ti["eval_l0"]:.2f}, dead {ti["dead_at_end_train"]}')
            res_f.parent.mkdir(parents=True, exist_ok=True)
            res_f.write_text(json.dumps(res, indent=1, default=float))
        del X
        torch.cuda.empty_cache()


def dev_of():
    import torch
    return torch.device('cuda')


def cmd_mtrain(args):
    St = mice_pool(dev_of())
    role = St.F.role.values[St.tf]
    train_set(St, args.arms.split(','), parse_seeds(args.seeds, SEEDS_M), args.width, 'mice', args, role)


def mice_labels(F):
    Y, unit = R2.labels_of(F.row.values)
    a = pd.read_csv(REPO / N.C['ann'], usecols=['W_annotator'], dtype={'W_annotator': str})
    ann = a.W_annotator.values[F.row.values]
    return Y, unit, ann


def fve_of(dom, arm, W, s):
    """Held-out FVE of a trained SAE (train json of this round, or of r3 for the reused checkpoints)."""
    if dom == 'mice' and arm in MICE_REUSE and W == 2048 and s in SEEDS_M:
        for f in ('train_r4_R0.json', 'train_r4_R7.json'):
            d = json.loads((R3.OUT3 / f).read_text())
            if f'{MICE_REUSE[arm]}_s{s}' in d:
                return d[f'{MICE_REUSE[arm]}_s{s}']['eval_fve']
        return np.nan
    for f in (OUT / dom).glob('train_*.json'):
        d = json.loads(f.read_text())
        if f'{arm}_w{W}_s{s}' in d:
            return d[f'{arm}_w{W}_s{s}']['eval_fve']
    return np.nan


def compact(o):
    """Keep the 'mean' block and, per split, R / A and selected neurons."""
    return {'mean': o['mean'], 'splits': {sn: {lb: {k: v[k] for k in ('R', 'A', 'neurons')} for lb, v in r.items()}
                                          for sn, r in o['splits'].items()}}


def cmd_mscore(args):
    """Codes of every seed of one arm on the diagnosis frames (kept in host RAM), single scores, unions."""
    import torch
    dev = dev_of()
    St = mice_diag(dev)
    Y, unit, ann = mice_labels(St.F)
    sc = json.loads((N.OUT / 'score.json').read_text())
    units, SPL = sc['units'], {k_: np.array(v_) for k_, v_ in sc['splits'].items()}
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    sets = {'all': np.arange(St.nf), 'consistent': np.flatnonzero(pd.Series(ann).isin(CONSISTENT).values)}
    half = {k: {s_: h[uidx[ix]] for s_, h in SPL.items()} for k, ix in sets.items()}
    Yg = {k: torch.from_numpy(Y[ix]).to(dev) for k, ix in sets.items()}
    ixg = {k: torch.from_numpy(ix).to(dev) for k, ix in sets.items()}
    cf, _, _ = R3.contact_frames(St.F, dict(np.load(R3.BEE3 / 'geom_diag.npz')))
    arm = args.arms
    res_f = OUT / 'mice' / f'score_{arm}{args.tag}.json'
    res = json.loads(res_f.read_text()) if res_f.exists() and not args.force else {}
    res['_frames'] = {k: int(len(ix)) for k, ix in sets.items()}
    res['_base'] = {k: {lb: float(Y[ix, i].mean()) for i, lb in enumerate(MLAB)} for k, ix in sets.items()}
    a = ARMS[arm]
    codes = {}

    def score(X, name, extra=None):
        out = {}
        for k in sets:
            Xk = X if k == 'all' else X.index_select(0, ixg[k])
            o = crossfit(Xk, Yg[k], half[k], MLAB)
            o['primary'] = float(np.mean([o['mean']['nose_nose']['R'], o['mean']['nose_tail']['R']]))
            if k == 'all':
                o['spec'] = R3.spec_metrics(X, o, Y, half['all'], cf)
            out[k] = compact(o) | {'primary': o['primary'], **({'spec': o['spec']} if k == 'all' else {})}
        out.update(extra or {})
        res[name] = out
        m = out['all']['mean']
        log(f'{name}: primary R~ {out["all"]["primary"]:.4f} | nn R~ {m["nose_nose"]["R"]:.4f} AUROC '
            f'{m["nose_nose"]["A"]:.4f} | nt R~ {m["nose_tail"]["R"]:.4f} AUROC {m["nose_tail"]["A"]:.4f} | '
            f'consistent primary R~ {out["consistent"]["primary"]:.4f}; {gpu_line()}')
        res_f.parent.mkdir(parents=True, exist_ok=True)
        res_f.write_text(json.dumps(res, indent=1, default=float))

    todo = [(W, s) for W in (2048, 8192) for s in (SEEDS_M if W == 2048 else SEEDS_W)
            if (sae_dir('mice', arm, W, s) / 'sae_theta.pt').exists()]
    if args.seeds:
        todo = [(W, s) for W, s in todo if s in parse_seeds(args.seeds, SEEDS_M)]
    for W, s in todo:
        name = f'{arm}_w{W}_s{s}'
        sae, norm = load_sae('mice', arm, W, s, dev)
        t0 = time.time()
        X = St.codes(sae, norm, a['w'], a['sel'])
        te = time.time() - t0
        if W == 2048:
            codes[s] = X.cpu()
        if name in res and not args.force:
            continue
        score(X, name, {'eval_fve': fve_of('mice', arm, W, s), 'encode_s': te,
                        'frame_fire_rate': float((X > 0).float().mean())})
        del X, sae
        torch.cuda.empty_cache()
    groups = [(f'U5_{arm}_{a_}-{b_}', list(range(a_, b_ + 1))) for a_, b_ in U5] + [(f'U20_{arm}_6-25', list(SEEDS_M))]
    for name, ss in groups:
        if name in res and not args.force:
            continue
        if not all(s in codes for s in ss):
            log(f'{name}: missing seeds {[s for s in ss if s not in codes]}, skipped')
            continue
        X = torch.cat([codes[s] for s in ss], 1).to(dev)
        score(X, name, {'blocks': [[s, 2048] for s in ss]})
        del X
        torch.cuda.empty_cache()


def cmd_mcheck(args):
    """Pipeline check: this code re-scores r3 SAEs (R5_s0, R7_s0, S2_s0) on the same frames; numbers must equal r3's."""
    import torch
    dev = dev_of()
    St = mice_diag(dev)
    Y, unit, _ = mice_labels(St.F)
    sc = json.loads((N.OUT / 'score.json').read_text())
    units, SPL = sc['units'], {k_: np.array(v_) for k_, v_ in sc['splits'].items()}
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    half = {s_: h[uidx] for s_, h in SPL.items()}
    Yg = torch.from_numpy(Y).to(dev)
    ref = {}
    for f in ('score_a.json', 'score_b.json'):
        ref.update(json.loads((R3.OUT3 / f).read_text()))
    out = {}
    for name, w, sel, root in (('R5_s0', 5, 'all', R3.BEE3), ('R7_s0', 5, 'edge', R3.BEE3), ('S2_s0', 0, 'all', R2.BEE2)):
        from src.eci.sae import load_sae as ls
        sae, norm, _ = ls(root / 'sae' / name / 'sae_theta.pt', dev)
        X = St.codes(sae, norm, w, sel)
        o = crossfit(X, Yg, half, MLAB)
        out[name] = {lb: {'v2': o['mean'][lb]['R'], 'r3': ref[name]['mean'][lb]['R'],
                          'v2_A': o['mean'][lb]['A'], 'r3_A': ref[name]['mean'][lb]['A']} for lb in MLAB}
        log(f'{name}: ' + json.dumps(out[name]))
    d = max(abs(v['v2'] - v['r3']) for r in out.values() for v in r.values())
    out['max_abs_diff_R'] = d
    (OUT / 'mice').mkdir(parents=True, exist_ok=True)
    (OUT / 'mice' / 'check_vs_r3.json').write_text(json.dumps(out, indent=1))
    log(f'max |R~ v2 - r3| = {d:.2e}')
    assert d < 1e-4, 'v2 scorer / codes differ from round 3'


# ---------------------------------------------------------------------------------------------- sheets (generic)
def frame_sheet(St, frames, sae, norm, w, sel, j, labels_of_frame, want, decode_frame, fn, title):
    """Top frames -> maps, locality numbers, two contact sheets (contribution map, activation map)."""
    import torch
    tiles_h, tiles_a, recs = [], [], []
    G = St.G
    for f in frames:
        a0, b0 = int(St.starts[f]), int(St.starts[f + 1])
        z = torch.from_numpy(np.ascontiguousarray(St.tok[a0:b0])).to(St.dev)
        ps = torch.from_numpy(St.pos[a0:b0]).to(St.dev)
        rd = torch.from_numpy(St.dist[a0:b0] == 1).to(St.dev) if sel == 'edge' else None
        A, H = neuron_maps(sae, norm, z, ps, G, w, j, rd)
        fdk = np.zeros(G * G, np.float32)
        fdk[St.pos[a0:b0]] = St.fd[a0:b0]
        lab = labels_of_frame(f)
        la, lh = locality(A, fdk), locality(H, fdk)
        val = float(A[St.pos[a0:b0]][St.dist[a0:b0] == 1].max()) if sel == 'edge' and (St.dist[a0:b0] == 1).any() \
            else float(A.max())
        r = {'frame': int(f), 'obs': St.F.obs.values[f], 'k': int(St.F.k.values[f]), 'value': val, 'label': lab,
             'positive': bool(any(x in lab for x in want)), **{f'act_{k}': v for k, v in la.items()},
             **{f'contrib_{k}': v for k, v in lh.items()}, 'n_kept': int(b0 - a0)}
        recs.append(r)
        x = decode_frame(f)
        tiles_h.append((x, H.reshape(G, G), {'obs': r['obs'], 'k': r['k'], 'label': lab, 'stored': val,
                                             'n_half_max': lh['n_half_max']}))
        tiles_a.append((x, A.reshape(G, G), {'obs': r['obs'], 'k': r['k'], 'label': lab, 'stored': val,
                                             'n_half_max': la['n_half_max']}))
    R = pd.DataFrame(recs)
    summ = {'share_label_pos': float(R.positive.mean()), 'label_counts': R.label.value_counts().to_dict()}
    for c in ('act_n_half_max', 'contrib_n_half_max', 'act_peak_fdark', 'contrib_peak_fdark', 'act_mass_animal_px',
              'contrib_mass_animal_px', 'n_kept'):
        summ[c] = float(R[c].mean())
    summ['contrib_single_token_frames'] = int((R.contrib_n_half_max == 1).sum())
    N.contact_sheet(tiles_h, Path(str(fn) + '_contrib.jpg'), title + ' | CONTRIBUTION map', G)
    N.contact_sheet(tiles_a, Path(str(fn) + '_act.jpg'), title + ' | ACTIVATION map', G)
    return summ, recs


def top_heldout(X_j, hold, obs, n=24):
    """Frame indices of the n highest values among held-out frames, at most one per video."""
    c = pd.DataFrame({'f': np.arange(len(X_j)), 'v': X_j, 'obs': obs})[hold]
    c = c.sort_values('v', ascending=False).drop_duplicates('obs').head(n)
    return c.f.values


def resolve(name, dom):
    """SAE / union name of this round -> (arm, [(seed, W)])."""
    if name.startswith('U'):
        _, arm, rng = name.split('_')
        a_, b_ = (int(x) for x in rng.split('-'))
        return arm, [(s, 2048) for s in range(a_, b_ + 1)]
    arm, w, s = name.split('_')
    return arm, [(int(s[1:]), int(w[1:]))]


def cmd_msheets(args):
    import torch
    dev = dev_of()
    St = mice_diag(dev)
    St.fd = np.load(N.local_dir() / 'diag' / 'tok_fdark.npy').astype(np.float32)
    sc = json.loads((N.OUT / 'score.json').read_text())
    hmap = dict(zip(sc['units'], sc['splits']['rnd0']))
    hold = np.array([hmap[o.rsplit('_', 2)[0]] == 1 for o in St.F.obs.values])
    ann = pd.read_csv(REPO / N.C['ann'], usecols=['Y_nn', 'Y_np', 'Y_nt'])
    V = {v['obs']: v for v in N.load_plan()['videos']}
    C = N.C

    def lab(f):
        y = ann.iloc[int(St.F.row.values[f])]
        return ''.join(c for c, col in (('nn', 'Y_nn'), ('np', 'Y_np'), ('nt', 'Y_nt')) if y[col] > 0) or '-'

    def dec(f):
        o, k = St.F.obs.values[f], int(St.F.k.values[f])
        v = V[o]
        fr = R4.decode(v['path'], v['offset'] + C['step'] * k, 1, C['step'], 1024, v['hflip'], C['fps'], threads=4)[0]
        return np.rot90(fr, v['rot'], axes=(0, 1)).transpose(2, 0, 1).copy()
    scores = {}
    for f in (OUT / 'mice').glob('score_*.json'):
        scores.update(json.loads(f.read_text()))
    out_f = OUT / 'mice' / 'locality.json'
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    (OUT / 'mice' / 'sheets').mkdir(parents=True, exist_ok=True)
    for name in args.names.split(','):
        arm, parts = resolve(name, 'mice')
        a = ARMS[arm]
        for lb, want in (('nose_nose', ('nn', 'np')), ('nose_tail', ('nt',))):
            key = f'{name}|{lb}'
            if key in out and not args.force:
                continue
            jg = scores[name]['all']['splits']['rnd0'][lb]['neurons'][0]
            s, j = block_of(jg, [(s_, W_) for s_, W_ in parts]) if len(parts) > 1 else (parts[0][0], jg)
            W = dict(parts)[s]
            sae, norm = load_sae('mice', arm, W, s, dev)
            X = St.codes(sae, norm, a['w'], a['sel'])[:, j].float().cpu().numpy()
            frames = top_heldout(X, hold, St.F.obs.values)
            fn = OUT / 'mice' / 'sheets' / f'{name}_{lb}_s{s}n{j}'
            summ, recs = frame_sheet(St, frames, sae, norm, a['w'], a['sel'], j, lab, want, dec, fn,
                                     f'v2 freeze {name} {lb} neuron s{s} n{j} (picked on pools half 0 of rnd0; top 24 '
                                     f'held-out frames, <= 1 per video)')
            out[key] = {'seed': s, 'neuron': j, 'W': W, 'summary': summ, 'frames': recs, 'sheet': str(fn)}
            log(f'{key} s{s} n{j}: ' + json.dumps(summ))
            out_f.write_text(json.dumps(out, indent=1, default=float))
            del sae
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- ants
# ants pixel cue AMENDMENT (2026-10-20 ~15:00, label-free, before any ants score): darker than the background by > 40
# grey levels WITHOUT the absolute limit grey < 60 (dark_abs 256 = off). Mask sheets (freeze/ants/mask_sheet_scored_*_
# selftest_abs60 / _abs256.jpg): the absolute limit drops the painted abdomens and the reddish heads / bodies (16 vs 40
# tokens per v3 frame, animal-pixel recall 0.86 vs 0.92), the relative cue keeps whole ants and no background.
ASER = {'v2': dict(dom='ants_v2p', S=768, bursts=124, dark_abs=256),
        'v3': dict(dom='ants_v3p', S=832, bursts=52, dark_abs=256)}
# frogs juveniles (no labels): health / locality only; pixel cue 'frog' (dark_rel 30, dish ROI, no SLEAP restriction)
FSER = {'fj': dict(dom='frogs', S=800, bursts=127, cue='frog', dark_rel=30)}


def ants_plan(ser):
    return R4.load_plan({**ASER, **FSER}[ser]['dom'])


def ants_labels(ser):
    c = R4.DOMS[ASER[ser]['dom']]
    a = pd.read_csv(REPO / c['eci'] / 'annotations.csv',
                    usecols=['observation_id', 'Y_Y2F', 'Y_B2F', 'Y_YOL', 'Y_BOL', 'Y_FOL'])
    ok = a[['Y_Y2F', 'Y_B2F']].notna().all(1).values
    g = lambda c_: a[c_].fillna(0).values > 0  # noqa: E731
    Y = np.stack([g('Y_Y2F') | g('Y_B2F'), g('Y_Y2F'), g('Y_B2F'), g('Y_YOL') | g('Y_BOL') | g('Y_FOL'), g('Y_YOL'),
                  g('Y_BOL'), g('Y_FOL')], 1)
    lid_ok = a[['Y_YOL', 'Y_BOL', 'Y_FOL']].notna().all(1).values
    return ok, lid_ok, Y, a.observation_id.values


def cmd_aprep(args):
    """Tight-mask token store of one ants series: pool (bursts) or scored frames (labelled, k = 0 mod 10)."""
    import torch
    from src.eci.v2.encode import Encoder
    from src.eci.v2.mask import THR, TightMask, mask_sheet
    from src.eci.v2.video import Prefetch
    dev = dev_of()
    ser, kind = args.series, args.kind
    cfg = {**ASER, **FSER}[ser]
    c = R4.DOMS[cfg['dom']]
    S, G = cfg['S'], cfg['S'] // 16
    P = ants_plan(ser)
    _, Rr = R4.obs_ranges_of(REPO / c['ann'])
    ok, aobs = (ants_labels(ser)[0], ants_labels(ser)[3]) if ser in ASER else (None, None)
    vids = P['videos'][:args.max_videos or None]
    items = []  # (video, [k...]) decode runs of consecutive 5 fps frames (step) and the kept k
    if kind == 'pool':
        g = np.random.default_rng(7)
        for v in vids:
            starts = np.sort(g.choice(v['n'] - 10, min(cfg['bursts'], v['n'] - 10), replace=False))
            for a_ in starts:
                kk0 = np.arange(a_, a_ + 10)
                items.append((v, kk0, kk0[kk0 % 10 != 0], 1))
    else:
        for v in vids:
            lab = ok[v['lay_lo']:v['lay_lo'] + v['n']]
            assert (aobs[v['lay_lo']:v['lay_lo'] + v['n']] == v['obs']).all()
            ks = np.flatnonzero(lab)
            ks = ks[ks % 10 == 0]
            for c0 in range(0, len(ks), 60):
                kk = ks[c0:c0 + 60]
                assert (np.diff(kk) == 10).all()
                items.append((v, kk, kk, 10))
    if args.max_frames:
        items = items[:args.max_frames]
    jobs = [((i, ), dict(path=v['path'], s0=R4.voff(c, v) + c['step'] * int(kk0[0]), n=len(kk0),
                         step=c['step'] * st, S=S, hflip=v['hflip'], fps=c['fps'], threads=2))
            for i, (v, kk0, kk, st) in enumerate(items)]
    enc = Encoder(dev)
    dark_abs = args.dark_abs if args.dark_abs else cfg.get('dark_abs', 60)
    tm = TightMask(REPO / c['bg'], Rr, cfg.get('cue', 'dark'), dev, dark_rel=cfg.get('dark_rel', 40), dark_abs=dark_abs)
    loc = N.local_dir() / f'{kind}_{ser}'
    loc.mkdir(parents=True, exist_ok=True)
    ftok = open(loc / 'tok.f16', 'wb')
    meta = {'tok_frame': [], 'tok_pos': [], 'tok_fdark': []}
    frames, sheet, sheet_obs = [], [], set()
    t0 = time.time()
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4))
    bs = 32
    for (i,), fr in pf:
        v, kk0, kk, st = items[i]
        sel = np.isin(kk0, kk)
        x_all = torch.from_numpy(np.ascontiguousarray(fr[sel])).to(dev).permute(0, 3, 1, 2).contiguous()
        rows = v['src_lo'] + kk
        for b in range(0, len(kk), bs):
            x = x_all[b:b + bs]
            with torch.no_grad():
                fd = tm.fraction(x, rows[b:b + bs])
                m = fd >= THR
                z = enc(x)
                px = tm.pixels(x, rows[b:b + bs])
            for ii in range(len(x)):
                pos = torch.nonzero(m[ii]).flatten()
                fid = len(frames)
                ftok.write(z[ii, pos].cpu().numpy().tobytes())
                meta['tok_frame'].append(np.full(len(pos), fid, np.int32))
                meta['tok_pos'].append(pos.cpu().numpy().astype(np.int16))
                meta['tok_fdark'].append(fd[ii, pos].cpu().numpy().astype(np.float16))
                npx = float(px[ii].sum())
                kept_px = float((fd[ii, pos] * 256).sum())
                k_ = int(kk[b + ii])
                frames.append({'series': ser, 'obs': v['obs'], 'k': k_, 'row': int(v['lay_lo'] + k_),
                               'src_row': int(v['src_lo'] + k_), 'role': v['role'], 'n_kept': int(len(pos)),
                               'animal_px': npx, 'px_recall': kept_px / npx if npx > 0 else np.nan,
                               'n_any_px': int((fd[ii] > 0).sum())})
                if len(sheet) < 8 and (v['role'] == 'eval' or args.selftest) and v['obs'] not in sheet_obs:
                    sheet_obs.add(v['obs'])
                    sheet.append((x[ii].cpu().numpy(), m[ii].cpu().numpy(), f'{v["obs"]} k{k_} kept {len(pos)}'))
        if i % 500 == 0:
            el = time.time() - t0
            log(f'{kind} {ser}: {i}/{len(items)} runs, {len(frames):,} frames, {el:.0f}s ({len(frames) / max(el, 1):.1f} '
                f'fps; GPU busy {enc.gpu_s:.0f}s; decode thread-s {pf.dec_s:.0f})')
    ftok.close()
    for k_, x in meta.items():
        np.save(loc / f'{k_}.npy', np.concatenate(x))
    Fm = pd.DataFrame(frames)
    Fm.to_parquet(loc / 'frames.parquet')
    tf = np.concatenate(meta['tok_frame'])
    pos = np.concatenate(meta['tok_pos']).astype(np.int64)
    np.save(loc / 'tok_dist.npy', edge_distance(tf, pos, len(Fm), G))
    el = time.time() - t0
    info = {'series': ser, 'kind': kind, 'S': S, 'G': G, 'n_frames': len(Fm), 'n_tokens': int(len(tf)),
            'tokens_per_frame': float(Fm.n_kept.mean()), 'frames_0_kept': int((Fm.n_kept == 0).sum()),
            'animal_px_recall_mean': float(Fm.px_recall.mean()), 'tokens_any_animal_px_per_frame':
            float(Fm.n_any_px.mean()), 'roles': Fm.role.value_counts().to_dict(), 'wall_s': el, 'gpu_s': enc.gpu_s,
            'frames_per_s_wall': len(Fm) / el, 'gpu': torch.cuda.get_device_name(0)}
    log(json.dumps(info))
    dom_ = 'ants' if ser in ASER else 'frogs'
    od = OUT / dom_
    od.mkdir(parents=True, exist_ok=True)
    tag = '_selftest' if args.selftest else ''
    tag = tag + (f'_abs{dark_abs}' if args.dark_abs else '')
    (od / f'prep_{kind}_{ser}{tag}.json').write_text(json.dumps(info, indent=1))
    if sheet:
        mask_sheet(sheet, od / f'mask_sheet_{kind}_{ser}{tag}.jpg', G)
    if not args.selftest:
        R4.atomic_copy(loc, BEE / dom_ / f'{kind}_{ser}')


def ants_stores(kind, dev, on_gpu):
    out = {}
    for ser, cfg in ASER.items():
        src = BEE / 'ants' / f'{kind}_{ser}'
        dst = N.local_dir() / f'{kind}_{ser}'
        if not dst.exists():
            R4.atomic_copy(src, dst)
        out[ser] = TokStore(dst, cfg['S'] // 16, dev, on_gpu=on_gpu)
        out[ser].fd = np.load(dst / 'tok_fdark.npy').astype(np.float32)
    return out


class Concat:
    """Two series stores as one training stream (inputs built per series with its own grid)."""

    def __init__(self, stores):
        self.stores = stores
        self.tf = np.concatenate([s.tf for s in stores.values()])
        self.dist = np.concatenate([s.dist for s in stores.values()])
        self.role = np.concatenate([s.F.role.values[s.tf] for s in stores.values()])

    def inputs(self, w, sel):
        import torch
        o, parts = 0, []
        for s in self.stores.values():
            n = len(s.tf)
            parts.append(s.inputs(w, sel[o:o + n]))
            o += n
        return torch.cat(parts)


def cmd_atrain(args):
    import torch
    St = Concat(ants_stores('pool', dev_of(), on_gpu=False))
    St.F = None
    W = args.width
    arms, seeds = args.arms.split(','), parse_seeds(args.seeds, SEEDS_A)
    res_f = OUT / 'ants' / f'train_{args.tag or "_".join(arms)}_w{W}.json'
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    for arm in arms:
        todo = [s for s in seeds if args.force or not (sae_dir('ants', arm, W, s) / 'sae_theta.pt').exists()]
        if not todo:
            continue
        a = ARMS[arm]
        sel = np.ones(len(St.tf), bool) if a['sel'] == 'all' else (St.dist == 1)
        t0 = time.time()
        X = {r: St.inputs(a['w'], sel & (St.role == r)) for r in ('train', 'calib', 'eval')}
        X['calib'], X['eval'] = X['calib'].cpu(), X['eval'].cpu()
        log(f'ants {arm} w{W}: ' + ', '.join(f'{r} {len(x):,}' for r, x in X.items()) + f' tokens built in '
            f'{time.time() - t0:.0f}s')
        for s in todo:
            cfg = cfg_of(arm, W)
            if args.steps:
                cfg['steps'] = args.steps
            ti = R3.train_arm(torch, cfg, s, X['train'], X['calib'], X['eval'], sae_dir('ants', arm, W, s), dev_of())
            res[f'{arm}_w{W}_s{s}'] = {'cfg': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                                       **{k: ti[k] for k in ('train_time_s', 'n_train_tokens', 'theta_star', 'eval_fve',
                                                              'eval_l0', 'dead_at_end_train')}}
            log(f'ants {arm}_w{W}_s{s}: {ti["train_time_s"]:.0f}s, eval FVE {ti["eval_fve"]:.4f}, L0 '
                f'{ti["eval_l0"]:.2f}, dead {ti["dead_at_end_train"]}')
            res_f.parent.mkdir(parents=True, exist_ok=True)
            res_f.write_text(json.dumps(res, indent=1, default=float))
        del X
        torch.cuda.empty_cache()


def cmd_ascore(args):
    import torch
    dev = dev_of()
    St = ants_stores('scored', dev, on_gpu=True)
    arm = args.arms
    a = ARMS[arm]
    lab = {}
    for ser, s_ in St.items():
        ok, lid_ok, Y, _ = ants_labels(ser)
        rows = s_.F.row.values
        assert ok[rows].all()
        obs = s_.F.obs.values
        units, SPL = splits_of(set(obs))
        uidx = pd.Series(range(len(units)), index=units).loc[obs].values
        labs = list(ALAB) if lid_ok[rows].all() else list(ALAB[:3])
        lab[ser] = (torch.from_numpy(Y[rows][:, :len(labs)]).to(dev), {k: h[uidx] for k, h in SPL.items()}, labs)
    res_f = OUT / 'ants' / f'score_{arm}{args.tag}.json'
    res = json.loads(res_f.read_text()) if res_f.exists() and not args.force else {}
    res['_base'] = {ser: {lb: float(l_[0][:, i].float().mean()) for i, lb in enumerate(l_[2])} for ser, l_ in lab.items()}
    codes = {}

    def score(Xs, name, extra=None):
        out = {}
        for ser, X in Xs.items():
            Yg, half, labs = lab[ser]
            out[ser] = compact(crossfit(X, Yg, half, labs))
        out['primary'] = float(np.mean([out[s]['mean']['groom_any']['R'] for s in ASER]))
        out['lid_any'] = out['v3']['mean']['on_lid_any']['R']
        out.update(extra or {})
        res[name] = out
        log(f'ants {name}: primary R~ (groom_any, mean v2 v3) {out["primary"]:.4f} | v2 groom_any R~ '
            f'{out["v2"]["mean"]["groom_any"]["R"]:.4f} AUROC {out["v2"]["mean"]["groom_any"]["A"]:.4f} | v3 '
            f'{out["v3"]["mean"]["groom_any"]["R"]:.4f} AUROC {out["v3"]["mean"]["groom_any"]["A"]:.4f} | v3 on_lid_any '
            f'R~ {out["lid_any"]:.4f} AUROC {out["v3"]["mean"]["on_lid_any"]["A"]:.4f}')
        res_f.parent.mkdir(parents=True, exist_ok=True)
        res_f.write_text(json.dumps(res, indent=1, default=float))
    todo = [(W, s) for W in (2048, 8192) for s in (SEEDS_A if W == 2048 else SEEDS_W)
            if (sae_dir('ants', arm, W, s) / 'sae_theta.pt').exists()]
    for W, s in todo:
        name = f'{arm}_w{W}_s{s}'
        sae, norm = load_sae('ants', arm, W, s, dev)
        Xs = {ser: s_.codes(sae, norm, a['w'], a['sel']) for ser, s_ in St.items()}
        if W == 2048:
            codes[s] = {ser: X.cpu() for ser, X in Xs.items()}
        if name not in res or args.force:
            score(Xs, name, {'eval_fve': fve_of('ants', arm, W, s)})
        del Xs, sae
        torch.cuda.empty_cache()
    for a_, b_ in U5_ANTS:
        name = f'U5_{arm}_{a_}-{b_}'
        ss = list(range(a_, b_ + 1))
        if (name in res and not args.force) or not all(s in codes for s in ss):
            continue
        score({ser: torch.cat([codes[s][ser] for s in ss], 1).to(dev) for ser in ASER}, name,
              {'blocks': [[s, 2048] for s in ss]})


def cmd_asheets(args):
    import torch
    dev = dev_of()
    St = ants_stores('scored', dev, on_gpu=True)
    scores = {}
    for f in (OUT / 'ants').glob('score_*.json'):
        scores.update(json.loads(f.read_text()))
    out_f = OUT / 'ants' / 'locality.json'
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    (OUT / 'ants' / 'sheets').mkdir(parents=True, exist_ok=True)
    for name in args.names.split(','):
        arm, parts = resolve(name, 'ants')
        a = ARMS[arm]
        for ser, lb, want in (('v2', 'groom_any', ('Y2F', 'B2F')), ('v3', 'groom_any', ('Y2F', 'B2F')),
                              ('v3', 'on_lid_any', ('YOL', 'BOL', 'FOL'))):
            key = f'{name}|{ser}|{lb}'
            if key in out and not args.force:
                continue
            s_ = St[ser]
            ok, _, Y, _ = ants_labels(ser)
            P = ants_plan(ser)
            V = {v['obs']: v for v in P['videos']}
            c = R4.DOMS[ASER[ser]['dom']]
            A_ = pd.read_csv(REPO / c['eci'] / 'annotations.csv', usecols=['Y_Y2F', 'Y_B2F', 'Y_YOL', 'Y_BOL', 'Y_FOL'])
            units, SPL = splits_of(set(s_.F.obs.values))
            hmap = dict(zip(units, SPL['rnd0']))
            hold = np.array([hmap[o] == 1 for o in s_.F.obs.values])
            jg = scores[name][ser]['splits']['rnd0'][lb]['neurons'][0]
            sd, j = block_of(jg, parts) if len(parts) > 1 else (parts[0][0], jg)
            W = dict(parts)[sd]
            sae, norm = load_sae('ants', arm, W, sd, dev)
            X = s_.codes(sae, norm, a['w'], a['sel'])[:, j].float().cpu().numpy()
            frames = top_heldout(X, hold, s_.F.obs.values)

            def labf(f, s_=s_, A_=A_):
                y = A_.iloc[int(s_.F.row.values[f])]
                return '+'.join(c_[2:] for c_ in ('Y_Y2F', 'Y_B2F', 'Y_YOL', 'Y_BOL', 'Y_FOL') if y[c_] > 0) or '-'

            def dec(f, s_=s_, V=V, c=c, ser=ser):
                v = V[s_.F.obs.values[f]]
                k = int(s_.F.k.values[f])
                return R4.decode(v['path'], R4.voff(c, v) + c['step'] * k, 1, c['step'], ASER[ser]['S'], v['hflip'],
                                 c['fps'], threads=4)[0].transpose(2, 0, 1).copy()
            fn = OUT / 'ants' / 'sheets' / f'{name}_{ser}_{lb}_s{sd}n{j}'
            summ, recs = frame_sheet(s_, frames, sae, norm, a['w'], a['sel'], j, labf, want, dec, fn,
                                     f'v2 freeze ants {name} {ser} {lb} neuron s{sd} n{j} (picked on videos half 0 of '
                                     f'rnd0; top 24 held-out frames, <= 1 per video)')
            out[key] = {'seed': sd, 'neuron': j, 'W': W, 'summary': summ, 'frames': recs, 'sheet': str(fn)}
            log(f'{key}: ' + json.dumps(summ))
            out_f.write_text(json.dumps(out, indent=1, default=float))


def cmd_fhealth(args):
    """Frogs juveniles (no labels): F0 / F1 / F2 seeds 6-8 on the frogs pool; per SAE held-out FVE / L0 / dead (train),
    frame firing on the held-out (calib + eval video) frames, and locality of 16 neurons drawn at random (rng 0) among
    those firing on >= 0.5% of the held-out frames: their 12 highest held-out frames (one per burst), contribution and
    activation maps -> OUT/frogs/health.json, 4 contact sheets per arm."""
    import torch
    dev = dev_of()
    src, dst = BEE / 'frogs' / 'pool_fj', N.local_dir() / 'pool_fj'
    if not dst.exists():
        R4.atomic_copy(src, dst)
    St = TokStore(dst, FSER['fj']['S'] // 16, dev, on_gpu=True)
    role = St.F.role.values[St.tf]
    arms, seeds = args.arms.split(','), parse_seeds(args.seeds, (6, 7, 8))
    train_set(St, arms, seeds, args.width, 'frogs', args, role)
    P = ants_plan('fj')
    V = {v['obs']: v for v in P['videos']}
    c = R4.DOMS['frogs']
    held = St.F.role.isin(['calib', 'eval']).values
    burst = (St.F.obs.astype(str) + '_' + (St.F.k // 10).astype(str)).values
    out_f = OUT / 'frogs' / 'health.json'
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    (OUT / 'frogs' / 'sheets').mkdir(parents=True, exist_ok=True)

    def dec(f):
        v = V[St.F.obs.values[f]]
        k = int(St.F.k.values[f])
        return R4.decode(v['path'], R4.voff(c, v) + c['step'] * k, 1, c['step'], FSER['fj']['S'], v['hflip'], c['fps'],
                         threads=4)[0].transpose(2, 0, 1).copy()
    for arm in arms:
        a = ARMS[arm]
        for s in seeds:
            name = f'{arm}_w{args.width}_s{s}'
            if name in out and not args.force:
                continue
            sae, norm = load_sae('frogs', arm, args.width, s, dev)
            X = St.codes(sae, norm, a['w'], a['sel'])
            Xh = X[torch.from_numpy(np.flatnonzero(held)).to(dev)].float()
            rate = (Xh > 0).float().mean(0).cpu().numpy()
            rec = {'eval_fve': fve_of('frogs', arm, args.width, s), 'frames_heldout': int(held.sum()),
                   'neurons_firing_heldout': int((rate > 0).sum()), 'neurons_ge_0.5pct': int((rate >= 0.005).sum()),
                   'fires_per_frame': float((Xh > 0).sum(1).float().mean()), 'max_neuron_rate': float(rate.max()),
                   'neurons': {}}
            cand = np.flatnonzero(rate >= 0.005)
            pick = np.sort(np.random.default_rng(0).choice(cand, min(16, len(cand)), replace=False))
            Xn = X.float().cpu().numpy()
            sums = []
            for i, j in enumerate(pick):
                d_ = pd.DataFrame({'f': np.arange(St.nf), 'v': Xn[:, j], 'b': burst})[held]
                frames = d_[d_.v > 0].sort_values('v', ascending=False).drop_duplicates('b').head(12).f.values
                fn = OUT / 'frogs' / 'sheets' / f'{name}_n{j}'
                if i < 4:
                    summ, _ = frame_sheet(St, frames, sae, norm, a['w'], a['sel'], int(j), lambda f: '-', (), dec, fn,
                                          f'v2 freeze frogs {name} neuron {j} (random alive neuron; top held-out frames)')
                else:
                    summ = locality_only(St, frames, sae, norm, a['w'], a['sel'], int(j))
                rec['neurons'][int(j)] = {'rate': float(rate[j]), **summ}
                sums.append(summ)
            for q in ('contrib_n_half_max', 'act_n_half_max', 'contrib_peak_fdark', 'act_mass_animal_px', 'n_kept'):
                rec[f'mean_{q}'] = float(np.nanmean([x[q] for x in sums]))
            out[name] = rec
            log(f'frogs {name}: ' + json.dumps({k: v for k, v in rec.items() if k != 'neurons'}))
            out_f.write_text(json.dumps(out, indent=1, default=float))
            del X, Xh, sae
            torch.cuda.empty_cache()


def locality_only(St, frames, sae, norm, w, sel, j):
    import torch
    rs = []
    for f in frames:
        a0, b0 = int(St.starts[f]), int(St.starts[f + 1])
        z = torch.from_numpy(np.ascontiguousarray(St.tok[a0:b0])).to(St.dev)
        ps = torch.from_numpy(St.pos[a0:b0]).to(St.dev)
        A, H = neuron_maps(sae, norm, z, ps, St.G, w, j)
        fdk = np.zeros(St.G * St.G, np.float32)
        fdk[St.pos[a0:b0]] = St.fd[a0:b0]
        la, lh = locality(A, fdk), locality(H, fdk)
        rs.append({**{f'act_{k}': v for k, v in la.items()}, **{f'contrib_{k}': v for k, v in lh.items()},
                   'n_kept': b0 - a0})
    R = pd.DataFrame(rs)
    return {c_: float(R[c_].mean()) for c_ in R.columns}


# ---------------------------------------------------------------------------------------------- decide
def mean_sd(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, len(x))


def cmd_decide(args):
    M, A = {}, {}
    for f in (OUT / 'mice').glob('score_*.json'):
        M.update(json.loads(f.read_text()))
    for f in (OUT / 'ants').glob('score_*.json'):
        A.update(json.loads(f.read_text()))

    def mp(arm, W, seeds, q=('all', 'primary')):
        out = []
        for s in seeds:
            r = M.get(f'{arm}_w{W}_s{s}')
            if r:
                out.append(r[q[0]][q[1]] if len(q) == 2 else r[q[0]]['mean'][q[1]][q[2]])
        return out

    def ap(arm, W, seeds, key='primary'):
        return [A[f'{arm}_w{W}_s{s}'][key] for s in seeds if f'{arm}_w{W}_s{s}' in A]
    dec = {'rule': 'scripts/eci/v2/freeze.py docstring (pre-registered 2026-10-20 ~14:30)', 'step1': {}}
    means = {arm: mean_sd(mp(arm, 2048, SEEDS_M)) for arm in ARMS}
    dec['step1']['mice_primary_R'] = means
    best = max(means, key=lambda k: means[k][0])
    dec['step1']['best'] = best
    ants_eff = {}
    for arm in ARMS:
        if arm == 'F0':
            continue
        ants_eff[arm] = {'primary': welch(ap(arm, 2048, SEEDS_A), ap('F0', 2048, SEEDS_A)) if len(ap(arm, 2048, SEEDS_A)) > 1
                         and len(ap('F0', 2048, SEEDS_A)) > 1 else None,
                         'lid_any': welch(ap(arm, 2048, SEEDS_A, 'lid_any'), ap('F0', 2048, SEEDS_A, 'lid_any'))
                         if len(ap(arm, 2048, SEEDS_A)) > 1 and len(ap('F0', 2048, SEEDS_A)) > 1 else None}
    dec['step1']['ants_vs_F0'] = ants_eff
    loses = lambda arm: arm != 'F0' and any(e is not None and e[0] <= -0.02 and e[2] < 0  # noqa: E731
                                            for e in ants_eff.get(arm, {}).values())
    dec['step1']['ants_loses'] = {arm: bool(loses(arm)) for arm in ARMS}
    chosen = None
    for arm in ELIGIBLE:
        if means[arm][2] and means[arm][0] >= means[best][0] - 0.02 and not loses(arm):
            chosen = arm
            break
    if chosen is None:
        cand = [a_ for a_ in ELIGIBLE if means[a_][2] and not loses(a_)]
        chosen = max(cand, key=lambda k: means[k][0]) if cand else None
        dec['step1']['note'] = 'no eligible arm within 0.02 of the best: highest eligible arm'
    dec['step1']['chosen_context'] = chosen
    dec['step1']['mice_effects_vs_best'] = {arm: welch(mp(arm, 2048, SEEDS_M), mp(best, 2048, SEEDS_M))
                                            for arm in ARMS if arm != best and len(mp(arm, 2048, SEEDS_M)) > 1}
    # step 2
    f12 = max(('F1', 'F2'), key=lambda k: means[k][0])
    s2 = {'arm': f12 if chosen in ('F1', 'F2') else chosen, 'better_of_F1_F2': f12}
    arm2 = s2['arm']
    w8, w2 = mp(arm2, 8192, SEEDS_W), mp(arm2, 2048, SEEDS_W)
    s2['w8192_vs_w2048_seeds6-11'] = welch(w8, w2) if len(w8) > 1 and len(w2) > 1 else None
    use8 = bool(s2['w8192_vs_w2048_seeds6-11'] and s2['w8192_vs_w2048_seeds6-11'][0] >= 0.02 and
                s2['w8192_vs_w2048_seeds6-11'][1] > 0)
    W = 8192 if use8 else 2048
    single = mp(arm2, W, SEEDS_W if W == 8192 else SEEDS_M)
    s2['width'] = W
    s2['single_mean'] = mean_sd(single)
    u5 = [M[f'U5_{arm2}_{a_}-{b_}']['all']['primary'] for a_, b_ in U5 if f'U5_{arm2}_{a_}-{b_}' in M]
    u20 = M.get(f'U20_{arm2}_6-25', {}).get('all', {}).get('primary')
    s2['U5'] = u5
    s2['U20'] = u20
    ua = [A[f'U5_{arm2}_{a_}-{b_}']['primary'] for a_, b_ in U5_ANTS if f'U5_{arm2}_{a_}-{b_}' in A]
    s2['ants_U5'] = ua
    s2['ants_single_mean'] = mean_sd(ap(arm2, 2048, SEEDS_A))
    ants_union_loses = bool(ua and np.mean(ua) - s2['ants_single_mean'][0] <= -0.02)
    adopt = bool(len(u5) == 4 and u20 is not None and np.mean(u5) >= s2['single_mean'][0] + 0.03 and
                 u20 >= s2['single_mean'][0] + 0.05 and not ants_union_loses)
    s2['union_adopted'] = adopt
    s2['union_rule'] = {'mean_U5_minus_single': float(np.mean(u5) - s2['single_mean'][0]) if u5 else None,
                        'U20_minus_single': float(u20 - s2['single_mean'][0]) if u20 is not None else None,
                        'ants_union_loses': ants_union_loses}
    fv = {s: fve_of('mice', arm2, W, s) for s in (6, 7, 8)}
    s2['deployment_seed_mice'] = int(max(fv, key=lambda s: fv[s] if np.isfinite(fv[s]) else -1))
    s2['deployment_fve_mice'] = fv
    fva = {s: fve_of('ants', arm2, W, s) for s in (6, 7, 8)}
    s2['deployment_seed_ants'] = int(max(fva, key=lambda s: fva[s] if np.isfinite(fva[s]) else -1))
    dec['step2'] = s2
    dec['v2'] = {'context_arm': arm2, 'context_window': ARMS[arm2]['w'], 'readout': ARMS[arm2]['sel'],
                 'dictionary': f'union of 20 x 2048' if adopt else f'single Matryoshka {W}'}
    (OUT / 'decision.json').write_text(json.dumps(dec, indent=1, default=float))
    table(M, A, dec)
    log(json.dumps(dec['v2']))


def table(M, A, dec):
    L = ['# ECI v2 freeze round — results', '',
         'Rule: scripts/eci/v2/freeze.py docstring (pre-registered). R~ = cross-fitted normalised R-precision (10 random '
         'unit splits); AUROC = cf AUROC; mean ± sd over SAE seeds. mice primary R~ = mean(R~ nose_nose, R~ nose_tail); '
         'ants primary R~ = mean over v2 / v3 of R~ groom_any.', '', '## Mice v1 (86,400 diagnosis frames)', '',
         '| arm | W | seeds | primary R~ | nose_nose R~ | nose_nose AUROC | nose_tail R~ | nose_tail AUROC | '
         'consistent primary R~ | held-out FVE |', '|---|---|---|---|---|---|---|---|---|---|']
    f = lambda x: f'{x[0]:.3f} ± {x[1]:.3f}' if x[2] > 1 else (f'{x[0]:.3f}' if x[2] else '-')  # noqa: E731
    for arm in ARMS:
        for W, seeds in ((2048, SEEDS_M), (8192, SEEDS_W)):
            rs = [M[f'{arm}_w{W}_s{s}'] for s in seeds if f'{arm}_w{W}_s{s}' in M]
            if not rs:
                continue
            c = lambda g: mean_sd([g(r) for r in rs])  # noqa: E731
            L.append(f'| {arm} | {W} | {len(rs)} | {f(c(lambda r: r["all"]["primary"]))} | '
                     f'{f(c(lambda r: r["all"]["mean"]["nose_nose"]["R"]))} | {f(c(lambda r: r["all"]["mean"]["nose_nose"]["A"]))} | '
                     f'{f(c(lambda r: r["all"]["mean"]["nose_tail"]["R"]))} | {f(c(lambda r: r["all"]["mean"]["nose_tail"]["A"]))} | '
                     f'{f(c(lambda r: r["consistent"]["primary"]))} | {f(c(lambda r: r.get("eval_fve")))} |')
        for nm in [k for k in M if k.startswith(('U5_', 'U20_')) and k.split('_')[1] == arm]:
            r = M[nm]
            L.append(f'| {nm} | union | {len(r["blocks"])} | {r["all"]["primary"]:.3f} | '
                     f'{r["all"]["mean"]["nose_nose"]["R"]:.3f} | {r["all"]["mean"]["nose_nose"]["A"]:.3f} | '
                     f'{r["all"]["mean"]["nose_tail"]["R"]:.3f} | {r["all"]["mean"]["nose_tail"]["A"]:.3f} | '
                     f'{r["consistent"]["primary"]:.3f} | - |')
    L += ['', '## Ants v2 + v3 (scored frames: every labelled frame with k = 0 mod 10)', '',
          '| arm | W | seeds | primary R~ | v2 groom_any R~ | v2 AUROC | v3 groom_any R~ | v3 AUROC | v2 yellow / blue R~ | '
          'v3 yellow / blue R~ | v3 on_lid_any R~ | v3 on_lid_any AUROC | held-out FVE |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in ARMS:
        for W, seeds in ((2048, SEEDS_A), (8192, SEEDS_W)):
            rs = [A[f'{arm}_w{W}_s{s}'] for s in seeds if f'{arm}_w{W}_s{s}' in A]
            if not rs:
                continue
            c = lambda g: f(mean_sd([g(r) for r in rs]))  # noqa: E731
            L.append(f'| {arm} | {W} | {len(rs)} | {c(lambda r: r["primary"])} | {c(lambda r: r["v2"]["mean"]["groom_any"]["R"])} | '
                     f'{c(lambda r: r["v2"]["mean"]["groom_any"]["A"])} | {c(lambda r: r["v3"]["mean"]["groom_any"]["R"])} | '
                     f'{c(lambda r: r["v3"]["mean"]["groom_any"]["A"])} | {c(lambda r: r["v2"]["mean"]["groom_yellow"]["R"])} / '
                     f'{c(lambda r: r["v2"]["mean"]["groom_blue"]["R"])} | {c(lambda r: r["v3"]["mean"]["groom_yellow"]["R"])} / '
                     f'{c(lambda r: r["v3"]["mean"]["groom_blue"]["R"])} | {c(lambda r: r["lid_any"])} | '
                     f'{c(lambda r: r["v3"]["mean"]["on_lid_any"]["A"])} | {c(lambda r: r.get("eval_fve"))} |')
        for nm in [k for k in A if k.startswith('U5_') and k.split('_')[1] == arm]:
            r = A[nm]
            L.append(f'| {nm} | union | {len(r["blocks"])} | {r["primary"]:.3f} | {r["v2"]["mean"]["groom_any"]["R"]:.3f} | '
                     f'{r["v2"]["mean"]["groom_any"]["A"]:.3f} | {r["v3"]["mean"]["groom_any"]["R"]:.3f} | '
                     f'{r["v3"]["mean"]["groom_any"]["A"]:.3f} | - | - | {r["lid_any"]:.3f} | - | - |')
    for dom in ('mice', 'ants'):
        lf = OUT / dom / 'locality.json'
        if lf.exists():
            Lc = json.loads(lf.read_text())
            L += ['', f'## Interpretability ({dom}; 24 top held-out frames of the neuron picked on half 0 of rnd0)', '',
                  '| set | neuron | share labelled | contribution-map tokens at half max | activation-map tokens at half max | '
                  'peak animal-pixel fraction (contribution) | activation mass on animal pixels | kept tokens / frame |',
                  '|---|---|---|---|---|---|---|---|']
            for k, v in Lc.items():
                s_ = v['summary']
                L.append(f'| {k} | s{v["seed"]} n{v["neuron"]} | {s_["share_label_pos"]:.2f} | {s_["contrib_n_half_max"]:.1f} | '
                         f'{s_["act_n_half_max"]:.1f} | {s_["contrib_peak_fdark"]:.2f} | {s_["act_mass_animal_px"]:.2f} | '
                         f'{s_["n_kept"]:.0f} |')
    L += ['', '## Decision', '', '```', json.dumps(dec, indent=1, default=float), '```']
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')


def parse_seeds(s, default):
    if not s:
        return default
    out = []
    for p in s.split(','):
        if '-' in p:
            a_, b_ = p.split('-')
            out += list(range(int(a_), int(b_) + 1))
        else:
            out.append(int(p))
    return tuple(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('mtrain', 'mscore', 'mcheck', 'msheets', 'aprep', 'atrain', 'ascore', 'asheets',
                                    'fhealth', 'decide'))
    p.add_argument('--arms', default='F1')
    p.add_argument('--seeds', default='')
    p.add_argument('--width', type=int, default=2048)
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--series', default='v2')
    p.add_argument('--kind', default='pool')
    p.add_argument('--names', default='')
    p.add_argument('--tag', default='')
    p.add_argument('--max-videos', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--dark-abs', type=int, default=0, help='aprep: override the absolute darkness limit (256 = off)')
    p.add_argument('--force', action='store_true')
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()
    {'mtrain': cmd_mtrain, 'mscore': cmd_mscore, 'mcheck': cmd_mcheck, 'msheets': cmd_msheets, 'aprep': cmd_aprep,
     'atrain': cmd_atrain, 'ascore': cmd_ascore, 'asheets': cmd_asheets, 'fhealth': cmd_fhealth,
     'decide': cmd_decide}[args.step](args)


if __name__ == '__main__':
    main()
