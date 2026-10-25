"""
LABEL-FREE BEHAVIOUR-NEURON SUITE for the ECI sparse dictionaries, and a v1 vs v2 comparison with it in every domain.

==================================================================================================================
PRE-REGISTRATION (written 2026-10-24 BEFORE any code of any dictionary was computed on the probe set below and BEFORE
any score of this study existed; the user chose these priors on 2026-10-24. Known before writing: the release-scorer
alignment tables results/vision/eci_v2_diag/{v1seeds,v2f_V2F_M0,v2f_V2F_N2}_table.md and the background-neuron counts
of the earlier tests (v2 tight / thin masks: 0 background neurons; v1's dilated mask on v2 tokens: 50-65%).)
AMENDMENT 1 (user guidance relayed 2026-10-24, folded in BEFORE this block was first committed): the PRIMARY metric is
CAPACITY (distinct behaviour neurons), not the share; the suite is also reported as a treatment-blind NES prefilter.

QUESTION. A neuron is a candidate BEHAVIOUR neuron when it behaves like a behaviour in time and space, judged without
any label: it sits on the animals, changes within a video, switches on and off in bouts, is not explained by the
recording conditions, re-appears in other seeds of the same recipe and does not change when the image is rotated,
flipped or slightly re-lit. Labels only VALIDATE the suite; they never define a check or a threshold.

REPRESENTATIONS (existing dictionaries only; no new SAE is trained; at most 5 seeds per arm)
    v1      deployed matryoshka_btk_1024_k16_{fg448al (mice) | antsfg (ants)}_s0 + v1's exact recipe retrained,
            seeds 1-4 (eci_codes/v2_diag/v1retrain/<dom>/V1R_full_s<s>/sae.pt): 5 seeds; W 1024 (128/256/512/1024),
            its stored EMA threshold. Tokens = v1's exact pipeline (dataset 512 px JPG -> odour rotation (mice) ->
            DINOv2-B/14 at 448 -> v1 mask (fg448 / ants rule, v1 backgrounds); scripts/eci/v2_diag/matched.py v1tok).
    v2dep   the deployed v2 recipe v2_<dom>_c0_w2048_s{6,7,8} (deployed: mice s7, ants s6, frogs s6); 3 seeds (all that
            exist); W 2048 (128/512/2048).
    V2F_M0  V2-full candidate, tight mask, seeds 900-904; W 2048 (128/256/512/2048).
    V2F_N2  V2-full candidate, thin-part mask, seeds 900-904.
    v2 tokens = the frozen v2 pipeline (src/eci/v2: native decode -> canonical -> DINOv3-L/16 block 20), mask M0 (tight,
    >= 25% frozen-cue animal pixels) for v2dep / V2F_M0, N2 (mask2 cue, r25t, existing per-window backgrounds) for
    V2F_N2. Readout everywhere = frame max of the threshold-mode codes over the kept tokens, UNSMOOTHED.
DOMAINS mice v1 (all four arms); ants v2 + v3 + v4 (all four arms; v1 and v2dep never trained on v4 = unlabelled
    out-of-training data); frogs juveniles + tadpoles (v2dep only: no V2F or v1 dictionary exists for frogs).

PROBE SET D (label-free, fixed here; one per domain): 72 videos x one window of 1,500 consecutive 5 fps frames (5 min)
    centred in the video (k0 = (n - 1500) // 2). Videos: mice = one per pool (72 pools sorted by pool id; pool i
    contributes its stage-((i mod 6) + 1) video, so every stage appears 12 times); ants = 24 per series v2 / v3 / v4
    (v4 = v4 + v4_infested), evenly spaced in sorted observation order (round(linspace(0, n - 1, 24))); frogs = all 35
    juveniles + 37 tadpoles evenly spaced among 118. 108,000 frames per domain.
    I = every 10th frame of each D window (2 s apart; 10,800 per domain) = the invariance sample.
    Identity checks (reported): the deployed v1 SAE on the D tokens vs its stored codes_max rows, the deployed v2 SAE
    vs its stored full-pass raw codes (fire agreement, max |diff|).

ELIGIBLE neuron: fires (frame code > 0) on >= 20 frames of I (original images). Only eligible neurons can be behaviour
    neurons; per-check medians are over eligible neurons. (This is the treatment-blind activity prefilter the user
    asked for on 2026-10-22: a rate floor of about 0.2% of frames.)

CHECKS (each is one rule; the cut is the same 'at most half / at least half' rule wherever a share is judged)
  C1 NOT BACKGROUND  off-animal share = among the D frames where the neuron fires, the share whose argmax token has < 5%
        animal pixels (LEAK 0.05 on the any-animal cue of mask2; scripts/eci/v2_mask2/mask2_test.py token_stats, the
        existing criterion: background neuron iff share > 0.5). PASS iff share <= 0.5. For v1 tokens (32-grid cells
        of the same canonical frame) the animal fraction = the any-cue pixel fraction of the cell (adaptive average
        pool of the decoded frame's any-cue pixel mask to 32 x 32). Frogs: the frozen frog cue. Prior: a behaviour
        is something an animal does, so it is read on the animal.
  C2 NOT STATIC  eta2 = between-video sum of squares / total sum of squares of the neuron's frame codes over D (videos =
        the 72 windows). PASS iff eta2 <= 0.5. Prior: behaviour changes within a session; a neuron that mostly
        separates videos (lighting, bedding, session, animal identity) is a nuisance.
  C3 BOUT-LIKE TIMING  rho(l) = within-window autocorrelation of the frame codes pooled over the 72 windows (each window
        demeaned by its own mean; sum of lagged products / sum of pair counts, over the lag-0 value). tau = the first
        lag where rho < 1/e (linear interpolation, seconds at 5 fps; infinite if rho >= 1/e up to 150 s). PASS iff
        0.2 s < tau <= 120 s, i.e. the code 0.2 s later is still correlated (rho(0.2 s) >= 1/e: no single-frame
        flicker) and it forgets within 2 minutes. Prior: behaviours come in bouts of about 0.2 s to a few minutes.
        Caveat stated now: per-window demeaning shortens tau of slow drifts (biased toward passing the upper bound).
  C4 NOT TIED TO RECORDING CONDITIONS  y = per-video mean frame code over the D window (72 values); X = one-hot recording
        conditions; ridge regression (alpha 1, y standardised) 5-fold cross-validated, 10 repeats (KFold shuffle,
        random_state 0-9); R2cv = 1 - sum (y - yhat_oof)^2 / sum (y - mean y)^2, mean over repeats. PASS iff R2cv <= 0.5.
        Conditions (never treatment, genotype, stage, odour or labels):
          mice   camera period (odor_corner TR / BL / BR; the camera zoom differs), recording month, recording hour
                 (data/mice/v1/experiment.csv; the recording DATE is not used: 52 dates for 72 pools, so a date is
                 essentially one pool = one genotype group, i.e. the animals, not a condition);
          ants   recording_date, batch, position (dataset/ants/eci30/experiment.csv);
          frogs  date, slot, hour (dataset/frogs/eci/experiment.csv, dataset/frogs/eci_tadpole_r4/experiment.csv).
        Prior: a behaviour neuron's video-level amount should not be predictable from day / session / camera.
        Caveat stated now: some conditions are partly aliased with treatment (mice camera period vs genotype; ants v3
        t = 8 only on day C), so C4 can remove a neuron whose treatment effect is aliased with a condition.
  C5 REPRODUCIBLE  for neuron j of seed s: m(s') = max over neurons i of seed s' (same arm) of the Pearson correlation of
        their D frame codes; score = median over the arm's other seeds s'. PASS iff score >= 0.5. Prior: a behaviour
        is a property of the data, not of the initialisation.
  C6 INVARIANT  on I, the frame codes of four transformed copies: rot90 (counter-clockwise), horizontal flip, brightness
        x 0.9 and x 1.1 (pixel values, rounded and clipped); the token mask is computed once on the original image and
        moved with the pixels, so only encoder + dictionary are tested. r_t = Pearson over I of the original vs the
        transformed frame codes; score = min over the 4 transforms. PASS iff score >= 0.5. Prior: a behaviour seen from
        above does not depend on the arena's orientation, mirror image or a small change of light.
  BEHAVIOUR NEURON = eligible AND C1-C6 pass.

PRIMARY METRIC = CAPACITY (user 2026-10-24): the number of DISTINCT behaviour neurons. Greedy in dictionary-index order
    (Matryoshka order, label-free): a behaviour neuron joins the first existing group whose founder has Pearson r >= 0.5
    with it, or firing-set Jaccard >= 0.5, on the D frame codes; otherwise it founds a new group. Capacity = number of
    groups. Reported at the FULL dictionary and @256 (neurons with index < 256, de-duplicated among themselves).
    Diversity (reported): N_eff = (sum lambda)^2 / sum lambda^2 of the correlation matrix of the behaviour neurons' D
    codes. Share of behaviour neurons = n_behaviour / W: SECONDARY descriptor only.
    COMPARISON: per domain, each v2 arm vs v1: mean capacity over seeds, Welch 95% CI of the difference (seed level;
    n = 5 / 5, v2dep 3). 'More behaviour neurons' iff the CI lower bound > 0, at full AND @256 reported separately.
    'Better behaviour neurons' is read from the validation (below), per arm: the best label alignment AMONG the
    behaviour neurons.
NES PREFILTER (reported, NES not rerun): candidates = behaviour neurons inside prefix p, p in {128, 256, 512, W};
    Bonferroni per-test threshold 0.05 / m (m = candidates) vs 0.05 / p today. The filter reads only the codes on D / I
    and the recording-condition variables of C4; it never reads treatment or labels.
DESCRIPTORS (never filters): extent = median over firing D frames of (# kept tokens with code >= 1/2 of the frame max)
    x token area / A1 area, A1 = the median single-animal blob size; social level = median over firing D frames of the
    animal count of the blob holding the argmax token (blobs = 8-connected components of native-grid tokens with
    any-cue fraction >= 0.25; animals = round(blob tokens / A1 tokens); A1 = median blob size over D frames with
    exactly N blobs of >= 2 tokens, N = 4 mice, 3 ants; frogs: one animal per video, level not computed); classes
    0 off-animal, 1 single, 2 pair, >= 3 group. v1 tokens are located on the native grid by their cell centre.

VALIDATION (labels; computed only after this block is committed). Frames S = the release-check scored frames (mice diag
    set 43,200; ants scored v2 13,200 and v3 32,100); labels mice nose_nose, nose_tail; ants v2 groom_any, v3
    groom_any and on_lid_any. Codes of every dictionary on S (v1: v1tok tokens; v2 arms: step-2 M0 / N2 stores).
    R~(j, l) = R~ of neuron j for label l on all S frames of that series (src/eci/v2/score.half_stats).
    Labelled behaviour neuron of (dictionary, label) = argmax_j R~(j, l) (full dictionary; also @256).
    V1 pass rate: share of labelled behaviour neurons (all arms, seeds, labels; full) passing all six checks; per check.
    V2 enrichment: per dictionary, AUROC of a_j = max_l R~(j, l) for behaviour vs non-behaviour ELIGIBLE neurons;
        Spearman of (number of checks passed) vs a_j; mean over seeds per arm.
    V3 (user point 3): across the arms of a labelled domain, the capacity ranking vs the labelled-alignment ranking
        (mean R~ of the labelled behaviour neurons per arm); Spearman, descriptive (n = 4 arms).
    THE SUITE IS VALIDATED iff (i) >= half of the labelled behaviour neurons pass all checks AND (ii) the enrichment
        AUROC is > 0.5 with the seed-level 95% t-CI (all dictionaries of the domain) excluding 0.5, in BOTH labelled
        domains. Otherwise the comparison is still reported, flagged 'suite not validated'.
TOP NEURONS: per domain, the deployment seed of the arm with the highest mean full capacity (deployment seed: v1 s0,
    v2dep the deployed seed, V2F the highest held-out FVE among 900-902): group founders ranked by C5 score; the top 8
    listed with every score and descriptor (+ best label R~, validation only); contact sheets of the top 6 (16 D frames
    with the highest code, <= 2 per video, per-token activation overlay) -> results/vision/eci_v2_behav/sheets/.
BUDGET <= 15 GPU-h (estimate: v2 encode ~4, v1 encode ~1, codes / suite / validation ~1.5, sheets ~0.2).
==================================================================================================================

Steps (scripts/eci/v2_behav/job.sh, EXTRA = arguments):
    plan                                  (CPU)  probe set D + conditions -> OUT/<dom>/probe.parquet, videos.parquet
    enc2  --dom D --shard i --nshard n    (GPU)  v2 tokens of D (+ I transforms) -> BEE/<dom>/v2/{dense,inv}/s<i>/
    enc1  --dom D --shard i --nshard n    (GPU)  v1 tokens of D (+ I transforms) -> BEE/<dom>/v1/{dense,inv}/s<i>/
    codes --dom D [--arms]                (GPU)  frame codes + token accumulators per dictionary -> BEE/<dom>/codes/
    suite --dom D                         (GPU)  checks C1-C6, capacity, prefilter -> OUT/<dom>/suite/
    vcodes / validate / sheets / table    see below
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
OUT = Path(os.environ.get('BEHAV_OUT') or REPO / 'results/vision/eci_v2_behav')
BEE = Path(os.environ.get('BEHAV_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_behav')
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
WIN, NVID, INV_EVERY = 1500, 72, 10
SERIES = {'mice': ('v1',), 'ants': ('v2', 'v3', 'v4'), 'frogs': ('juv', 'tad')}
N_ANIMALS = {'mice': 4, 'ants': 3, 'frogs': 1}
TRANSFORMS = ('rot90', 'hflip', 'bright0.9', 'bright1.1')
LEAK, MIN_FIRE_I, HALF = 0.05, 20, 0.5
TAU_LO, TAU_HI, TAU_MAX = 0.2, 120.0, 150.0
FPS = 5.0
V1 = {'mice': dict(ann='dataset/mice/v1/annotations.csv', bg='dataset/mice/v1/eci/fg448al/background', rule='fg448',
                   align='odor', sae='dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448al_s0',
                   codes='dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0/codes_max.npy'),
      'ants': dict(ann='dataset/ants/eci30/annotations.csv', bg='dataset/ants/eci30/fg448/background', rule='ants',
                   align='none', sae='dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0',
                   codes='dataset/ants/eci30/codes/matryoshka_btk_1024_k16_antsfg_s0/codes_max.npy')}
DEPLOYED_V2 = {'mice': 7, 'ants': 6, 'frogs': 6}


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def local_dir():
    d = Path(os.environ.get('LOCAL_DIR') or (OUT / 'tmp_local'))
    d.mkdir(parents=True, exist_ok=True)
    return d


def atomic_copy(src, dst):
    import shutil
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(tmp, dst)


def series_cfg(dom, name):
    c = json.loads((REPO / f'configs/eci_v2/{dom}.json').read_text())
    return next(s for s in c['series'] if s['name'] == name)


def gpu_line():
    import torch
    return f'GPU {torch.cuda.get_device_name(0)}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB'


# ---------------------------------------------------------------------------------------------- arms
def arms_of(dom):
    """{arm: [(seed, sae path, W, mask, family)]} (family 'v1' | 'v2'; mask 'M0' | 'N2' for v2)."""
    out = {}
    if dom in V1:
        out['v1'] = [(0, REPO / V1[dom]['sae'] / 'sae.pt', 1024, 'v1', 'v1')] + \
            [(s, EC / f'v2_diag/v1retrain/{dom}/V1R_full_s{s}/sae.pt', 1024, 'v1', 'v1') for s in (1, 2, 3, 4)]
    out['v2dep'] = [(s, EC / f'v2/{dom}/sae/v2_{dom}_c0_w2048_s{s}/sae_theta.pt', 2048, 'M0', 'v2') for s in (6, 7, 8)]
    if dom != 'frogs':
        for a, m in (('V2F_M0', 'M0'), ('V2F_N2', 'N2')):
            out[a] = [(s, EC / f'v2_diag/{dom}/sae/{a}_s{s}/sae_theta.pt', 2048, m, 'v2') for s in range(900, 905)]
    return out


def deploy_seed(dom, arm):
    if arm == 'v1':
        return 0
    if arm == 'v2dep':
        return DEPLOYED_V2[dom]
    fv = {}
    for s in (900, 901, 902):
        f = REPO / f'results/vision/eci_v2_diag/{dom}/train/{arm}_s{s}.json'
        fv[s] = json.loads(f.read_text()).get('eval_fve', -1) if f.exists() else -1
    return max(fv, key=fv.get)


# ---------------------------------------------------------------------------------------------- plan
def cmd_plan(args):
    for dom in (args.dom.split(',') if args.dom else SERIES):
        vids = []
        for ser in SERIES[dom]:
            cfg = series_cfg(dom, ser)
            P = json.loads((REPO / cfg['plan']).read_text())['videos']
            V = {v['obs']: v for v in P}
            if dom == 'mice':
                oc = pd.read_csv(REPO / 'dataset/mice/v1/eci/odor_corner.csv')
                oc = oc[oc.observation_id.isin(V)]
                pick = []
                for i, (pool, g) in enumerate(sorted(oc.groupby('pool'), key=lambda t: t[0])):
                    g = g.sort_values('stage')
                    want = (i % 6) + 1
                    pick.append(g[g.stage == want].observation_id.values[0])
            elif dom == 'ants':
                obs = sorted(V)
                pick = [obs[i] for i in np.round(np.linspace(0, len(obs) - 1, 24)).astype(int)]
            else:
                obs = sorted(V)
                pick = obs if ser == 'juv' else [obs[i] for i in np.round(np.linspace(0, len(obs) - 1, 37)).astype(int)]
            for o in pick:
                n = int(V[o]['n'])
                k0 = max(0, (n - WIN) // 2)
                vids.append(dict(series=ser, obs=o, n=n, k0=k0, nwin=min(WIN, n)))
        Vd = pd.DataFrame(vids)
        assert len(Vd) == NVID and Vd.obs.is_unique, (dom, len(Vd))
        Vd['vid'] = np.arange(len(Vd))
        Vd = Vd.merge(conditions(dom), on='obs', how='left')
        cc = [c for c in Vd.columns if c.startswith('c_')]
        assert not Vd[cc].isna().any().any(), Vd[cc].isna().sum()
        rows = [dict(vid=r.vid, series=r.series, obs=r.obs, k=int(r.k0 + i), inv=bool(i % INV_EVERY == 0))
                for r in Vd.itertuples() for i in range(r.nwin)]
        Fr = pd.DataFrame(rows)
        od = OUT / dom
        od.mkdir(parents=True, exist_ok=True)
        Vd.to_parquet(od / 'videos.parquet')
        Fr.to_parquet(od / 'probe.parquet')
        log(f'plan {dom}: {len(Vd)} videos ({Vd.series.value_counts().to_dict()}), {len(Fr):,} D frames, '
            f'{int(Fr.inv.sum()):,} I frames; conditions {cc} levels ' +
            str({c: int(Vd[c].nunique()) for c in cc}))


def conditions(dom):
    """obs -> recording-condition columns c_* (never treatment / genotype / stage / odour / labels)."""
    if dom == 'mice':
        e = pd.read_csv(REPO / 'data/mice/v1/experiment.csv')
        oc = pd.read_csv(REPO / 'dataset/mice/v1/eci/odor_corner.csv').set_index('observation_id')['odor_corner']
        # the recording time from the observation file name (experiment.csv date / time is corrupt on 20 rows)
        ts = e.observation_file.str.extract(r'^(\d{4}-\d{2})-\d{2}_(\d{2})-')
        return pd.DataFrame({'obs': e.observation_id, 'c_period': oc.reindex(e.observation_id).values,
                             'c_month': ts[0].values, 'c_hour': ts[1].values})
    if dom == 'ants':
        e = pd.read_csv(REPO / 'dataset/ants/eci30/experiment.csv', dtype={'observation_id': str})
        return pd.DataFrame({'obs': e.observation_id, 'c_date': e.recording_date.astype(str),
                             'c_batch': e.batch.astype(str), 'c_position': e.position.astype(str)})
    out = []
    for f in ('dataset/frogs/eci/experiment.csv', 'dataset/frogs/eci_tadpole_r4/experiment.csv'):
        e = pd.read_csv(REPO / f, dtype={'observation_id': str})
        out.append(pd.DataFrame({'obs': e.observation_id, 'c_date': e.date.astype(str), 'c_slot': e.slot.astype(str),
                                 'c_hour': e.hour.astype(str)}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------- transforms
def tf_img(x, t):
    """x (B, 3, S, S) uint8 cuda -> transformed copy."""
    import torch
    if t == 'rot90':
        return torch.rot90(x, 1, dims=(2, 3)).contiguous()
    if t == 'hflip':
        return x.flip(3).contiguous()
    f = float(t[6:])
    return (x.float() * f).round().clamp(0, 255).to(torch.uint8)


def tf_grid(g, t, G):
    """per-token values (B, G*G) moved with the pixels."""
    import torch
    if t.startswith('bright'):
        return g
    h = g.view(g.shape[0], G, G)
    h = torch.rot90(h, 1, dims=(1, 2)) if t == 'rot90' else h.flip(2)
    return h.reshape(g.shape[0], G * G).contiguous()


class Writer:
    """Union-layout token shard: tok.f16, tok_frame, tok_pos, fr_raw / fr_new / fr_app / fr_any (fp16), n_app,
    frames.parquet (+ optional per-frame arrays)."""

    def __init__(self, root, d):
        root.mkdir(parents=True, exist_ok=True)
        self.root, self.d = root, d
        self.f = open(root / 'tok.f16', 'wb')
        self.m = {k: [] for k in ('tok_frame', 'tok_pos', 'fr_raw', 'fr_new', 'fr_app', 'fr_any', 'n_app')}
        self.extra, self.recs = {}, []

    def add(self, z, pos, rec, fr=None, extra=None):
        """z (n, d) fp16 numpy of the kept tokens, pos (n,) int, fr {name: (n,) values}."""
        fid = len(self.recs)
        self.f.write(np.ascontiguousarray(z).tobytes())
        self.m['tok_frame'].append(np.full(len(pos), fid, np.int32))
        self.m['tok_pos'].append(pos.astype(np.int16))
        for k in ('fr_raw', 'fr_new', 'fr_app', 'fr_any'):
            self.m[k].append((fr[k] if fr is not None else np.ones(len(pos))).astype(np.float16))
        self.m['n_app'].append((fr['n_app'] if fr is not None else np.zeros(len(pos))).astype(np.int16))
        for k, v in (extra or {}).items():
            self.extra.setdefault(k, []).append(v)
        rec['n_tok'] = int(len(pos))
        self.recs.append(rec)

    def close(self):
        self.f.close()
        for k, v in self.m.items():
            np.save(self.root / f'{k}.npy', np.concatenate(v) if v else np.zeros(0))
        for k, v in self.extra.items():
            np.save(self.root / f'{k}.npy', np.stack(v))
        pd.DataFrame(self.recs).to_parquet(self.root / 'frames.parquet')
        return len(self.recs)


# ---------------------------------------------------------------------------------------------- v2 encode
def cmd_enc2(args):
    """v2 tokens of the probe set: frozen v2 encoder, M0 | N2 union (mask2 cue) for mice / ants, the frozen cue for
    frogs; per frame the native-grid any-cue fraction (anyG, padded to the domain's largest grid) and its 32 x 32
    adaptive pool (any32, v1's grid); I frames also encoded under the four transforms."""
    import torch
    import torch.nn.functional as F
    from src.eci.v2.encode import Encoder
    from src.eci.v2.pipeline import Series
    from src.eci.v2.video import Prefetch
    dev = torch.device('cuda')
    dom = args.dom
    m2 = dom in ('mice', 'ants')
    if m2:
        sys.path.insert(0, str(REPO / 'scripts/eci/v2_mask2'))
        import mask2_test as T2
        from src.eci.v2 import mask2 as M2
    Vd = pd.read_parquet(OUT / dom / 'videos.parquet')
    Vd = Vd[Vd.vid % args.nshard == args.shard]
    if args.selftest:
        Vd = Vd.iloc[:1]
    enc = Encoder(dev)
    SE = {ser: Series(series_cfg(dom, ser), dev, encoder=enc) for ser in Vd.series.unique()}
    Gmax = max(series_cfg(dom, s)['S'] // 16 for s in SERIES[dom])
    jobs, meta = [], []
    CH = 300
    for r in Vd.itertuples():
        se = SE[r.series]
        nwin = 60 if args.selftest else r.nwin
        for c0 in range(0, nwin, CH):
            n = min(CH, nwin - c0)
            jobs.append(((len(meta),), se.job(se.V[r.obs], int(r.k0 + c0), n, 1, threads=2)))
            meta.append((r.series, r.obs, r.vid, r.k0, np.arange(r.k0 + c0, r.k0 + c0 + n)))
    tag = '_selftest' if args.selftest else ''
    loc = local_dir() / f'enc2_{dom}_s{args.shard}{tag}'
    Wd, Wi = Writer(loc / 'dense', 1024), Writer(loc / 'inv', 1024)
    t0, n_fr = time.time(), 0
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) // 2 - 2), fn=Series.run_job)
    pack_key, pack = None, None
    for (j,), frm in pf:
        ser, o, vid, k0, kk = meta[j]
        se = SE[ser]
        v = se.V[o]
        G = se.G
        bs = 16 if se.S >= 1000 else 32
        if m2:
            h = T2.h_of(dom, ser if ser != 'v4' else 'v3')
            w = T2.window_of(dom, int(kk[0]))
            assert all(T2.window_of(dom, int(k)) == w for k in kk)
            if pack_key != (ser, o, w):
                pack_key, pack = (ser, o, w), T2.load_pack(dom, ser, o, w, dev)
        for b in range(0, len(kk), bs):
            kb = kk[b:b + bs]
            with torch.no_grad():
                x, fd = se.frames(v, frm[b:b + bs], kb)
                pool = lambda m: F.avg_pool2d(m.float()[:, None], 16).flatten(1)  # noqa: E731
                if m2:
                    core0 = se.mask.pixels(x, v['src_lo'] + kb)
                    rr = M2.cue(x, pack, core0, h)
                    _, app = M2.appendage(rr['pix'], h)
                    anyp = core0 | (rr['d'] > M2.T_LO)
                    raw, frn, fapp, fany = pool(core0), pool(rr['pix']), pool(app), pool(anyp)
                    ng = fapp * 256
                    keep = (raw >= 0.25) | M2.keep(frn, ng, 'r25t')
                    any32 = F.adaptive_avg_pool2d(anyp.float()[:, None], 32).flatten(1)
                else:
                    raw = frn = fany = fd.float()
                    fapp = ng = torch.zeros_like(raw)
                    keep = raw >= 0.25
                    any32 = None
                z = enc(x)
                grids = dict(fr_raw=raw, fr_new=frn, fr_app=fapp, fr_any=fany, n_app=ng)
                put(Wd, z, keep, grids, kb, dict(vid=vid, series=ser, obs=o, tr=0), G, Gmax, fany, any32)
                isel = np.flatnonzero((kb - k0) % INV_EVERY == 0)
                if len(isel):
                    it = torch.from_numpy(isel).to(dev)
                    for ti, t in enumerate(TRANSFORMS, 1):
                        xt = tf_img(x.index_select(0, it), t)
                        gt = {k_: tf_grid(g_.index_select(0, it), t, G) for k_, g_ in grids.items()}
                        kt = tf_grid(keep.index_select(0, it).float(), t, G) > 0.5
                        put(Wi, enc(xt), kt, gt, kb[isel], dict(vid=vid, series=ser, obs=o, tr=ti), G, Gmax)
        n_fr += len(kk)
        if j % 10 == 0:
            el = time.time() - t0
            log(f'enc2 {dom} s{args.shard}: chunk {j + 1}/{len(jobs)}, {n_fr:,} frames, {el:.0f}s '
                f'({n_fr / max(el, 1):.1f} fps; GPU busy {enc.gpu_s:.0f}s; decode thread-s {pf.dec_s:.0f}); {gpu_line()}')
    nd, ni = Wd.close(), Wi.close()
    el = time.time() - t0
    info = dict(dom=dom, shard=args.shard, nshard=args.nshard, n_dense=nd, n_inv=ni, wall_s=el, gpu_s=enc.gpu_s,
                fps_wall=nd / max(el, 1), gpu=torch.cuda.get_device_name(0))
    log(json.dumps(info, default=float))
    (OUT / dom / 'enc').mkdir(parents=True, exist_ok=True)
    (OUT / dom / 'enc' / f'enc2_s{args.shard}{tag}.json').write_text(json.dumps(info, indent=1, default=float))
    if not args.selftest:
        for kind in ('dense', 'inv'):
            atomic_copy(loc / kind, BEE / dom / 'v2' / kind / f's{args.shard}')
    else:
        selftest_sheet(dom, loc, SE)


def put(Wr, z, keep, grids, kb, base, G, Gmax, fany=None, any32=None):
    for ii in range(len(kb)):
        pos = keep[ii].nonzero().flatten()
        fr = {k_: g_[ii, pos].float().cpu().numpy() for k_, g_ in grids.items()}
        ex = None
        if fany is not None:
            ag = np.zeros(Gmax * Gmax, np.float16)
            ag[:G * G] = fany[ii].cpu().numpy()
            ex = {'anyG': ag}
            if any32 is not None:
                ex['any32'] = any32[ii].cpu().numpy().astype(np.float16)
        Wr.add(z[ii, pos].cpu().numpy(), pos.cpu().numpy(), dict(base, k=int(kb[ii]), G=int(G)), fr, ex)


def selftest_sheet(dom, loc, SE):
    """selftest: a few decoded frames with their kept tokens (eyeball check of geometry / mask)."""
    import torch
    from src.eci.v2.mask import mask_sheet
    Fm = pd.read_parquet(loc / 'dense' / 'frames.parquet')
    tf = np.load(loc / 'dense' / 'tok_frame.npy')
    pos = np.load(loc / 'dense' / 'tok_pos.npy').astype(int)
    items = []
    for i in (0, len(Fm) // 2, len(Fm) - 1):
        r = Fm.iloc[i]
        se = SE[r.series]
        v = se.V[r.obs]
        fr = se.run_job(**se.job(v, int(r.k), 1, threads=2))
        x, _ = se.frames(v, fr, [int(r.k)])
        kept = np.zeros(r.G * r.G, bool)
        kept[pos[tf == i]] = True
        items.append((x[0].cpu().numpy(), kept, f'{r.obs} k{r.k}'))
    p = OUT / dom / 'enc' / 'selftest_v2.jpg'
    mask_sheet(items, p, int(Fm.G.iloc[0]))
    log(f'selftest sheet -> {p}')


# ---------------------------------------------------------------------------------------------- v1 encode
def row_index(ann_path):
    a = pd.read_csv(ann_path, usecols=['observation_id', 'frame_idx'], dtype={'observation_id': str})
    return pd.Series(np.arange(len(a)), index=pd.MultiIndex.from_arrays([a.observation_id.values, a.frame_idx.values]))


def cmd_enc1(args):
    """v1 tokens of the probe set: v1's exact pipeline (matched.py v1tok); I frames also under the four transforms
    (applied to the processor output; brightness on the de-normalised pixels), v1 mask computed on the original."""
    import torch
    from src.eci.foreground import RULES, FgBackgrounds, FgEncoder, align_rot90, encode_batch, obs_rows
    dev = torch.device('cuda')
    dom = args.dom
    c = V1[dom]
    ann = REPO / c['ann']
    Fr = pd.read_parquet(OUT / dom / 'probe.parquet')
    Fr = Fr[Fr.vid % args.nshard == args.shard].reset_index(drop=True)
    if args.selftest:
        Fr = Fr[Fr.vid == Fr.vid.iloc[0]].iloc[:64].reset_index(drop=True)
    ri = row_index(ann)
    rows = ri.reindex(pd.MultiIndex.from_arrays([Fr.obs.values.astype(str), Fr.k.values.astype(int)])).values
    assert not np.isnan(rows).any(), 'probe frames without an annotations row'
    rows = rows.astype(np.int64)
    fp = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    rot = align_rot90(c['align'], ann)
    enc = FgEncoder('dinov2_base', dev)
    fb = FgBackgrounds(REPO / c['bg'], obs_rows(ann), RULES[c['rule']], dev, align=c['align'], cache=8)
    ds = enc.dataset([str(REPO / 'dataset' / p) for p in fp[rows]], rows, None if rot is None else rot[rows])
    dl = torch.utils.data.DataLoader(ds, batch_size=50, num_workers=max(2, int(os.environ.get('SLURM_CPUS_PER_TASK', 8)) - 4),
                                     shuffle=False, pin_memory=True, prefetch_factor=4)
    mean = torch.tensor(enc.processor.image_mean, device=dev)[None, :, None, None]
    std = torch.tensor(enc.processor.image_std, device=dev)[None, :, None, None]
    tag = '_selftest' if args.selftest else ''
    loc = local_dir() / f'enc1_{dom}_s{args.shard}{tag}'
    Wd, Wi = Writer(loc / 'dense', 768), Writer(loc / 'inv', 768)
    t0, o, gpu_s = time.time(), 0, 0.0
    for pix, grey, rr in dl:
        g0 = time.time()
        with torch.no_grad():
            pix = pix.to(dev, non_blocking=True)
            tok = encode_batch(enc.model, pix, dev)
            m, _ = fb.mask(tok, grey.to(dev), rr.numpy())
            B = len(tok)
            fr_b = Fr.iloc[o:o + B]
            for ii in range(B):
                pos = m[ii].nonzero().flatten()
                r = fr_b.iloc[ii]
                Wd.add(tok[ii, pos].cpu().numpy(), pos.cpu().numpy(), dict(vid=int(r.vid), series=r.series, obs=r.obs,
                                                                          k=int(r.k), tr=0, row=int(rr[ii]), G=32))
            isel = np.flatnonzero(fr_b.inv.values)
            if len(isel):
                it = torch.from_numpy(isel).to(dev)
                p0, m0 = pix.index_select(0, it), m.index_select(0, it).float()
                for ti, t in enumerate(TRANSFORMS, 1):
                    if t == 'rot90':
                        pt = torch.rot90(p0, 1, dims=(2, 3)).contiguous()
                    elif t == 'hflip':
                        pt = p0.flip(3).contiguous()
                    else:
                        pt = (((p0 * std + mean) * float(t[6:])).clamp(0, 1) - mean) / std
                    tt = encode_batch(enc.model, pt, dev)
                    mt = tf_grid(m0, t, 32) > 0.5
                    for jj, ii in enumerate(isel):
                        pos = mt[jj].nonzero().flatten()
                        r = fr_b.iloc[ii]
                        Wi.add(tt[jj, pos].cpu().numpy(), pos.cpu().numpy(),
                               dict(vid=int(r.vid), series=r.series, obs=r.obs, k=int(r.k), tr=ti, row=int(rr[ii]), G=32))
        torch.cuda.synchronize()
        gpu_s += time.time() - g0
        o += B
        if (o // B) % 100 == 0:
            el = time.time() - t0
            log(f'enc1 {dom} s{args.shard}: {o:,}/{len(Fr):,} frames, {el:.0f}s ({o / max(el, 1):.1f} fps, '
                f'GPU-side {gpu_s:.0f}s); {gpu_line()}')
    nd, ni = Wd.close(), Wi.close()
    el = time.time() - t0
    info = dict(dom=dom, shard=args.shard, nshard=args.nshard, n_dense=nd, n_inv=ni, wall_s=el, gpu_side_s=gpu_s,
                fps_wall=nd / max(el, 1), gpu=torch.cuda.get_device_name(0))
    log(json.dumps(info, default=float))
    (OUT / dom / 'enc').mkdir(parents=True, exist_ok=True)
    (OUT / dom / 'enc' / f'enc1_s{args.shard}{tag}.json').write_text(json.dumps(info, indent=1, default=float))
    if not args.selftest:
        for kind in ('dense', 'inv'):
            atomic_copy(loc / kind, BEE / dom / 'v1' / kind / f's{args.shard}')


# ---------------------------------------------------------------------------------------------- stores (read)
class Store:
    """All shards of BEE/<dom>/<fam>/<kind>: frames F (global order of the shards), per-token arrays, tok memmaps."""

    def __init__(self, dom, fam, kind, d):
        root = BEE / dom / fam / kind
        sh = sorted([p for p in root.glob('s*') if not p.name.endswith('.tmp')], key=lambda p: int(p.name[1:]))
        assert sh, root
        Fs, tfs, o, self.mm, self.bounds = [], [], 0, [], []
        cat = {k: [] for k in ('tok_pos', 'fr_raw', 'fr_new', 'fr_any', 'n_app')}
        self.extra = {}
        nt = 0
        for p in sh:
            F_ = pd.read_parquet(p / 'frames.parquet')
            tf = np.load(p / 'tok_frame.npy')
            tfs.append(tf + o)
            for k in cat:
                cat[k].append(np.load(p / f'{k}.npy'))
            for k in ('anyG', 'any32'):
                if (p / f'{k}.npy').exists():
                    self.extra.setdefault(k, []).append(np.load(p / f'{k}.npy'))
            self.mm.append(np.memmap(p / 'tok.f16', np.float16, 'r', shape=(len(tf), d)))
            self.bounds.append((nt, nt + len(tf)))
            nt += len(tf)
            Fs.append(F_)
            o += len(F_)
        self.F = pd.concat(Fs, ignore_index=True)
        self.tf = np.concatenate(tfs)
        self.pos = np.concatenate(cat['tok_pos']).astype(np.int64)
        self.raw, self.new, self.any = (np.concatenate(cat[k]).astype(np.float32) for k in ('fr_raw', 'fr_new', 'fr_any'))
        self.ng = np.concatenate(cat['n_app']).astype(np.float32)
        self.extra = {k: np.concatenate(v) for k, v in self.extra.items()}
        self.nf, self.d = len(self.F), d
        self.starts = np.r_[0, np.cumsum(np.bincount(self.tf, minlength=self.nf))]
        assert np.all(np.diff(self.tf) >= 0)

    def tok(self, a, b):
        out = []
        for (lo, hi), mm in zip(self.bounds, self.mm):
            x0, x1 = max(a, lo), min(b, hi)
            if x0 < x1:
                out.append(np.asarray(mm[x0 - lo:x1 - lo]))
        return np.concatenate(out)

    def keep(self, mask):
        if mask == 'N2':
            return (self.new >= 0.25) | (self.ng >= 16)
        if mask == 'M0':
            return self.raw >= 0.25
        return np.ones(len(self.tf), bool)


def blob_maps(dom, Sd):
    """Per D frame (v2 dense store order): animals of the any-cue blob holding each native-grid cell (uint8, Gmax^2),
    and A1 = median single-animal blob size (tokens)."""
    from scipy import ndimage
    A = Sd.extra['anyG'].astype(np.float32)
    Gs = Sd.F.G.values
    Gmax = int(np.sqrt(A.shape[1]))
    st = np.ones((3, 3), bool)
    labs, sizes = [], []
    for i in range(Sd.nf):
        G = int(Gs[i])
        m = A[i, :G * G].reshape(G, G) >= 0.25
        lb, n = ndimage.label(m, structure=st)
        sz = np.bincount(lb.ravel(), minlength=n + 1)
        labs.append((lb, sz))
        sizes.append(sz[1:][sz[1:] >= 2])
    N = N_ANIMALS[dom]
    one = np.concatenate([s for s in sizes if len(s) == N]) if N > 1 else np.concatenate([s for s in sizes if len(s) == 1])
    A1 = float(np.median(one)) if len(one) else float('nan')
    out = np.zeros((Sd.nf, Gmax * Gmax), np.uint8)
    for i, (lb, sz) in enumerate(labs):
        G = lb.shape[0]
        an = np.where(sz >= 1, np.clip(np.round(sz / A1), 1, 255), 0).astype(np.uint8)
        an[0] = 0
        out[i, :G * G] = an[lb.ravel()]
    return out, A1, int(sum(len(s) == N for s in sizes))


def cmd_codes(args):
    """One pass over each token store, every dictionary of the domain per chunk: D frame codes (fp16, frames in
    (vid, k) order), I codes per transform, C1 accumulators and the sparse descriptor records."""
    import torch
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    dom = args.dom
    A = arms_of(dom)
    if args.arms:
        A = {k: v for k, v in A.items() if k in args.arms.split(',')}
    od = BEE / dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    Pr = pd.read_parquet(OUT / dom / 'probe.parquet')
    key = lambda F_: pd.MultiIndex.from_arrays([F_.vid.values.astype(int), F_.k.values.astype(int)])  # noqa: E731
    pidx = pd.Series(np.arange(len(Pr)), index=key(Pr))
    iidx = pd.Series(np.arange(int(Pr.inv.sum())), index=key(Pr[Pr.inv]))
    S2 = Store(dom, 'v2', 'dense', 1024)
    d_of = pidx.reindex(key(S2.F)).values
    assert not np.isnan(d_of).any() and len(np.unique(d_of)) == len(Pr), 'v2 dense store != probe set'
    bl, A1, n1 = blob_maps(dom, S2)
    bl_D = np.zeros_like(bl)
    bl_D[d_of.astype(int)] = bl
    G_D = np.zeros(len(Pr), np.int64)
    G_D[d_of.astype(int)] = S2.F.G.values
    any32_D = None
    if 'any32' in S2.extra:
        any32_D = np.zeros((len(Pr), 1024), np.float16)
        any32_D[d_of.astype(int)] = S2.extra['any32']
    S_px = {s: series_cfg(dom, s)['S'] for s in SERIES[dom]}
    meta = {'A1_tokens': A1, 'A1_frames': n1, 'n_D': len(Pr), 'n_I': len(iidx)}
    log(f'{dom}: A1 = {A1} tokens ({n1} frames with {N_ANIMALS[dom]} blobs)')
    fams = sorted({x[4] for v in A.values() for x in v})
    for fam in fams:
        dicts = [(arm, s, p, W, m) for arm, v in A.items() for (s, p, W, m, f) in v if f == fam and p.exists()]
        missing = [(arm, s) for arm, v in A.items() for (s, p, W, m, f) in v if f == fam and not p.exists()]
        assert not missing, missing
        saes = {}
        for arm, s, p, W, m in dicts:
            sae, norm, _ = load_sae(p, dev)
            saes[(arm, s)] = (sae, norm, W, m)
        for kind in ('dense', 'inv'):
            St = S2 if (fam, kind) == ('v2', 'dense') else Store(dom, fam, kind, 1024 if fam == 'v2' else 768)
            if kind == 'dense':
                fidx = pidx.reindex(key(St.F)).values
                assert not np.isnan(fidx).any() and len(np.unique(fidx)) == len(Pr), (fam, 'dense store != probe set')
                nrow = len(Pr)
            else:
                fidx = iidx.reindex(key(St.F)).values
                assert not np.isnan(fidx).any(), (fam, 'inv frames outside I')
                fidx = fidx + (St.F.tr.values.astype(int) - 1) * len(iidx)  # (transform, I frame) row
                nrow = len(TRANSFORMS) * len(iidx)
            fidx = fidx.astype(np.int64)
            keeps = {m: St.keep(m) for m in {x[3] for x in saes.values()}}
            # per-token C1 / level inputs (dense only)
            if kind == 'dense':
                if fam == 'v2':
                    t_off = St.any < LEAK
                    t_lev = bl_D[fidx[St.tf], St.pos]
                    t_area = np.full(len(St.tf), 256.0, np.float32)
                else:
                    fr_ = fidx[St.tf]
                    t_off = (any32_D[fr_, St.pos].astype(np.float32) < LEAK) if any32_D is not None else np.zeros(len(St.tf), bool)
                    Gt = G_D[fr_]
                    rr, cc = St.pos // 32, St.pos % 32
                    gr = np.minimum(((rr + 0.5) * Gt / 32).astype(np.int64), Gt - 1)
                    gc = np.minimum(((cc + 0.5) * Gt / 32).astype(np.int64), Gt - 1)
                    t_lev = bl_D[fr_, gr * Gt + gc]
                    t_area = np.array([(S_px[s] / 32) ** 2 for s in St.F.series.values], np.float32)[St.tf]
            X = {k_: torch.zeros((nrow, v[2]), dtype=torch.float16, device=dev) for k_, v in saes.items()}
            acc = {k_: {n: torch.zeros(v[2], dtype=torch.float64, device=dev) for n in ('fire', 'hit', 'off')}
                   for k_, v in saes.items()}
            sparse = {k_: [] for k_ in saes}
            fp = 400 if fam == 'v2' else 800
            t0 = time.time()
            for f0 in range(0, St.nf, fp):
                f1 = min(St.nf, f0 + fp)
                a, b = int(St.starts[f0]), int(St.starts[f1])
                if a == b:
                    continue
                z = torch.from_numpy(St.tok(a, b)).to(dev)
                fl_all = torch.from_numpy(St.tf[a:b] - f0).to(dev)
                rows_t = torch.from_numpy(fidx[f0:f1]).to(dev)
                for k_, (sae, norm, W, m) in saes.items():
                    sel = torch.from_numpy(keeps[m][a:b]).to(dev)
                    zz, fl = z[sel], fl_all[sel]
                    with torch.no_grad():
                        c = sae.encode(norm(zz.float()), mode='threshold').float()
                        fm = torch.zeros((f1 - f0, W), device=dev).index_reduce_(0, fl, c, 'amax', include_self=True)
                        X[k_][rows_t] = fm.half()
                        if kind == 'dense':
                            sl = slice(a, b)
                            offt = torch.from_numpy(t_off[sl][keeps[m][sl]]).to(dev)
                            levt = torch.from_numpy(t_lev[sl][keeps[m][sl]].astype(np.float32)).to(dev)
                            art = torch.from_numpy(t_area[sl][keeps[m][sl]]).to(dev)
                            fmt = fm[fl]
                            hit = (c > 0) & (c == fmt)
                            acc[k_]['fire'] += (fm > 0).sum(0).double()
                            acc[k_]['hit'] += hit.sum(0).double()
                            acc[k_]['off'] += hit[offt].sum(0).double()
                            hm = ((c >= HALF * fmt) & (c > 0)).float() * art[:, None]
                            ext = torch.zeros((f1 - f0, W), device=dev).index_add_(0, fl, hm)
                            lev = torch.zeros((f1 - f0, W), device=dev).index_reduce_(
                                0, fl, hit.float() * (levt[:, None] + 1), 'amax', include_self=True) - 1
                            nz = torch.nonzero(fm > 0)
                            sparse[k_].append(np.stack([
                                rows_t[nz[:, 0]].cpu().numpy(), nz[:, 1].cpu().numpy(),
                                ext[nz[:, 0], nz[:, 1]].round().cpu().numpy(),
                                lev[nz[:, 0], nz[:, 1]].cpu().numpy()]).astype(np.int32))
                        del c
                if (f0 // fp) % 25 == 0:
                    log(f'codes {dom} {fam} {kind}: {f1:,}/{St.nf:,} frames, {time.time() - t0:.0f}s; {gpu_line()}')
            for (arm, s), x in X.items():
                np.save(od / f'{arm}_s{s}_{kind[0].upper()}.npy', x.cpu().numpy())
                if kind == 'dense':
                    sp = np.concatenate(sparse[(arm, s)], axis=1)
                    np.savez(od / f'{arm}_s{s}_desc.npz', frame=sp[0], neuron=sp[1].astype(np.int16),
                             ext_px=sp[2], lev=sp[3].astype(np.int16),
                             **{n: v.cpu().numpy() for n, v in acc[(arm, s)].items()})
            log(f'codes {dom} {fam} {kind}: saved {len(X)} dictionaries; {gpu_line()}')
            if kind == 'dense':
                meta[f'tokens_per_frame_{fam}'] = {m: float(keeps[m].sum() / St.nf) for m in keeps}
            del X
            torch.cuda.empty_cache()
    meta['identity'] = identity(dom, A, od)
    (OUT / dom / 'codes_meta.json').write_text(json.dumps(meta, indent=1, default=float))
    log(json.dumps(meta, default=float))


def identity(dom, A, od):
    """Deployed v1 SAE on the D v1 tokens vs its stored codes_max; deployed v2 SAE vs the full-pass per-video codes."""
    Pr = pd.read_parquet(OUT / dom / 'probe.parquet')
    out = {}
    if 'v1' in A and dom in V1:
        Fv = Store(dom, 'v1', 'dense', 768).F
        key = pd.MultiIndex.from_arrays([Pr.vid.values, Pr.k.values])
        row = pd.Series(Fv.row.values, index=pd.MultiIndex.from_arrays([Fv.vid.values, Fv.k.values])).reindex(key).values
        X = np.load(od / 'v1_s0_D.npy').astype(np.float32)
        ref = np.load(REPO / V1[dom]['codes'], mmap_mode='r')
        o = np.argsort(row)
        R = np.empty_like(X)
        R[o] = np.asarray(ref[row[o]]).astype(np.float32)
        out['v1_s0'] = {'fire_agree': float(((X > 0) == (R > 0)).mean()), 'max_abs_diff': float(np.abs(X - R).max()),
                        'mean_abs_diff': float(np.abs(X - R).mean())}
    if 'v2dep' in A:
        s = DEPLOYED_V2[dom]
        X = np.load(od / f'v2dep_s{s}_D.npy').astype(np.float32)
        ag, md, n = [], 0.0, 0
        for (ser, obs), g in Pr.groupby(['series', 'obs'], sort=False):
            f = EC / f'v2/{dom}/videos/{ser}/{obs}.npz'
            if not f.exists():
                continue
            ref = np.load(f)['codes_max'][g.k.values].astype(np.float32)
            x = X[g.index.values]
            ag.append(((x > 0) == (ref > 0)).mean())
            md = max(md, float(np.abs(x - ref).max()))
            n += 1
        out[f'v2dep_s{s}'] = {'videos': n, 'fire_agree': float(np.mean(ag)) if ag else None, 'max_abs_diff': md}
    return out


# ---------------------------------------------------------------------------------------------- suite
def acf_tau(torch, X, starts, max_lag):
    """X (n, W) float cuda frames in window order, starts = window boundaries -> rho (max_lag + 1, W) pooled within-
    window autocorrelation (each window demeaned), tau seconds (W,) (first lag with rho < 1/e; inf if none)."""
    W = X.shape[1]
    num = torch.zeros((max_lag + 1, W), dtype=torch.float64, device=X.device)
    cnt = torch.zeros(max_lag + 1, dtype=torch.float64, device=X.device)
    for a, b in zip(starts[:-1], starts[1:]):
        x = X[a:b] - X[a:b].mean(0, keepdim=True)
        n = b - a
        L = min(max_lag, n - 1)
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        f = torch.fft.rfft(x.double(), n=nfft, dim=0)
        ac = torch.fft.irfft(f * f.conj(), n=nfft, dim=0)[:L + 1]
        num[:L + 1] += ac
        cnt[:L + 1] += torch.arange(n, n - L - 1, -1, device=X.device, dtype=torch.float64)
    cov = num / cnt[:, None]
    rho = cov / cov[0:1].clamp_min(1e-30)
    below = rho < np.exp(-1)
    first = torch.where(below.any(0), below.float().argmax(0), torch.full((W,), -1, device=X.device)).long()
    tau = torch.full((W,), float('inf'), dtype=torch.float64, device=X.device)
    ok = first > 0
    l1 = first[ok]
    r0, r1 = rho[l1 - 1, ok.nonzero().flatten()], rho[l1, ok.nonzero().flatten()]
    frac = (r0 - np.exp(-1)) / (r0 - r1).clamp_min(1e-12)
    tau[ok] = (l1 - 1 + frac) / FPS
    tau[first == 0] = 0.0  # zero-variance neuron
    return rho, tau


def cond_r2(Y, C, seeds=range(10), folds=5, alpha=1.0):
    """Y (n_videos, W) per-video means, C one-hot (n_videos, p) -> cross-validated R^2 per column (mean over repeats)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    sd = Y.std(0)
    Z = (Y - Y.mean(0)) / np.where(sd > 0, sd, 1)
    out = []
    for s in seeds:
        P = np.zeros_like(Z)
        for tr, te in KFold(folds, shuffle=True, random_state=s).split(Z):
            P[te] = Ridge(alpha=alpha).fit(C[tr], Z[tr]).predict(C[te])
        ss = ((Z - Z.mean(0)) ** 2).sum(0)
        out.append(1 - ((Z - P) ** 2).sum(0) / np.where(ss > 0, ss, 1))
    r2 = np.mean(out, 0)
    r2[sd == 0] = np.nan
    return r2


def zcols(torch, X):
    Z = X - X.mean(0, keepdim=True)
    sd = Z.norm(dim=0)
    return Z / sd.clamp_min(1e-12), sd > 0


def cmd_suite(args):
    import torch
    dev = torch.device('cuda')
    dom = args.dom
    A = arms_of(dom)
    cd = BEE / dom / 'codes'
    Pr = pd.read_parquet(OUT / dom / 'probe.parquet')
    Vd = pd.read_parquet(OUT / dom / 'videos.parquet').set_index('vid')
    meta = json.loads((OUT / dom / 'codes_meta.json').read_text())
    A1 = meta['A1_tokens']
    vid = Pr.vid.values
    starts = np.r_[0, np.flatnonzero(np.diff(vid) != 0) + 1, len(vid)]
    assert np.all(np.diff(Pr.k.values)[np.diff(vid) == 0] == 1), 'D frames not consecutive within a window'
    inv_rows = np.flatnonzero(Pr.inv.values)
    cc = [c for c in Vd.columns if c.startswith('c_')]
    C = pd.get_dummies(Vd[cc].astype(str)).values.astype(np.float64)
    vcount = np.bincount(vid)
    od = OUT / dom / 'suite'
    od.mkdir(parents=True, exist_ok=True)
    Zc = {}
    for arm, lst in A.items():
        for (s, p, W, m, fam) in lst:
            Zc[(arm, s)] = None
    # reproducibility needs the other seeds' standardised codes: one arm at a time on the GPU
    for arm, lst in A.items():
        seeds = [x[0] for x in lst]
        XS = {s: torch.from_numpy(np.load(cd / f'{arm}_s{s}_D.npy')).to(dev) for s in seeds}
        ZS = {s: zcols(torch, XS[s].float()) for s in seeds}
        for (s, p, W, m, fam) in lst:
            X = XS[s].float()
            # eligibility (I originals)
            XI0 = X[torch.from_numpy(inv_rows).to(dev)]
            fireI = (XI0 > 0).sum(0).cpu().numpy()
            elig = fireI >= MIN_FIRE_I
            # C1
            dz = np.load(cd / f'{arm}_s{s}_desc.npz')
            off = np.where(dz['hit'] > 0, dz['off'] / np.maximum(dz['hit'], 1), np.nan)
            # C2
            vt = torch.from_numpy(vid).to(dev)
            mu_v = torch.zeros((len(vcount), W), device=dev).index_add_(0, vt, X) / torch.from_numpy(vcount).to(dev)[:, None].float()
            mu = X.mean(0)
            sst = ((X - mu) ** 2).sum(0)
            ssb = (torch.from_numpy(vcount).to(dev)[:, None].float() * (mu_v - mu) ** 2).sum(0)
            eta2 = (ssb / sst.clamp_min(1e-30)).cpu().numpy()
            eta2[sst.cpu().numpy() <= 0] = np.nan
            # C3
            _, tau = acf_tau(torch, X, starts, int(TAU_MAX * FPS))
            tau = tau.cpu().numpy()
            # C4
            r2c = cond_r2(mu_v.cpu().numpy().astype(np.float64), C)
            # C5
            Zs, oks = ZS[s]
            rep_o = []
            for s2 in seeds:
                if s2 == s:
                    continue
                Z2, ok2 = ZS[s2]
                M = Zs.T @ Z2[:, ok2]
                rep_o.append(M.max(1).values.cpu().numpy())
            rep = np.median(np.stack(rep_o), 0)
            rep[~oks.cpu().numpy()] = np.nan
            # C6
            XI = torch.from_numpy(np.load(cd / f'{arm}_s{s}_I.npy')).to(dev).float()
            nI = len(inv_rows)
            Za, oka = zcols(torch, XI0)
            rts = []
            for ti in range(len(TRANSFORMS)):
                Zt, okt = zcols(torch, XI[ti * nI:(ti + 1) * nI])
                r = (Za * Zt).sum(0)
                r[~(oka & okt)] = float('nan')
                rts.append(r.cpu().numpy())
            rts = np.stack(rts)
            inv = rts.min(0)
            # descriptors
            sp = pd.DataFrame({'neuron': dz['neuron'].astype(int), 'ext': dz['ext_px'] / (A1 * 256.0),
                               'lev': dz['lev'].astype(int)})
            gm = sp.groupby('neuron').median()
            ext = gm['ext'].reindex(range(W)).values
            lev = gm['lev'].reindex(range(W)).values
            T = pd.DataFrame({'neuron': np.arange(W), 'fire_I': fireI, 'eligible': elig, 'rate_D': (X > 0).float().mean(0).cpu().numpy(),
                              'off_share': off, 'eta2': eta2, 'tau_s': tau, 'r2_cond': r2c, 'repro': rep, 'inv_min': inv,
                              **{f'inv_{t}': rts[i] for i, t in enumerate(TRANSFORMS)}, 'extent_bodies': ext, 'level': lev})
            T['c1'] = T.off_share <= 0.5
            T['c2'] = T.eta2 <= 0.5
            T['c3'] = (T.tau_s > TAU_LO) & (T.tau_s <= TAU_HI)
            T['c4'] = T.r2_cond <= 0.5
            T['c5'] = T.repro >= 0.5
            T['c6'] = T.inv_min >= 0.5
            chk = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
            T['n_checks'] = T[chk].sum(1)
            T['behaviour'] = T.eligible & T[chk].all(1)
            # capacity: greedy groups in index order
            B = np.flatnonzero(T.behaviour.values)
            group = np.full(W, -1)
            neff = float('nan')
            if len(B):
                XB = X[:, torch.from_numpy(B).to(dev)]
                ZB, _ = zcols(torch, XB)
                R = (ZB.T @ ZB).cpu().numpy()
                Fb = (XB > 0).float()
                inter = (Fb.T @ Fb).cpu().numpy()
                nb = np.diag(inter)
                Jac = inter / np.maximum(nb[:, None] + nb[None, :] - inter, 1)
                founders = []
                for ii, j in enumerate(B):
                    g = next((f for f in founders if R[ii, f[1]] >= 0.5 or Jac[ii, f[1]] >= 0.5), None)
                    if g is None:
                        founders.append((j, ii))
                        group[j] = j
                    else:
                        group[j] = g[0]
                ev = np.clip(np.linalg.eigvalsh(np.nan_to_num(R)), 0, None)
                neff = float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30))
            T['group'] = group
            T['founder'] = T.group.values == np.arange(W)
            T.to_parquet(od / f'{arm}_s{s}.parquet')
            el = T[T.eligible]
            pre = {str(pp): int(T.behaviour.values[:pp].sum()) for pp in (128, 256, 512, W)}
            summ = {'arm': arm, 'seed': s, 'W': W, 'deployed': s == deploy_seed(dom, arm), 'n_eligible': int(elig.sum()),
                    'n_behaviour': int(T.behaviour.sum()), 'capacity_full': int(T.founder.sum()),
                    'capacity_256': int(T.founder.values[:256].sum()), 'n_behaviour_256': int(T.behaviour.values[:256].sum()),
                    'n_eff': neff, 'share': float(T.behaviour.sum() / W),
                    'median': {k: float(np.nanmedian(el[k])) if len(el) else None
                               for k in ('off_share', 'eta2', 'tau_s', 'r2_cond', 'repro', 'inv_min')},
                    'pass_rate_eligible': {c: float(el[c].mean()) if len(el) else None for c in chk},
                    'prefilter_candidates': pre,
                    'bonferroni': {str(pp): {'today': 0.05 / pp, 'filtered': 0.05 / max(pre[str(pp)], 1)}
                                   for pp in (128, 256, 512, W)}}
            (od / f'{arm}_s{s}.json').write_text(json.dumps(summ, indent=1, default=float))
            log(f'{dom} {arm}_s{s}: eligible {summ["n_eligible"]}, behaviour {summ["n_behaviour"]}, capacity full '
                f'{summ["capacity_full"]} / @256 {summ["capacity_256"]}, N_eff {neff:.1f}; pass rates '
                f'{json.dumps({k: round(v, 3) for k, v in summ["pass_rate_eligible"].items() if v is not None})}')
        del XS, ZS
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- validation
LABELS = {'mice': {'v1': ('nose_nose', 'nose_tail')},
          'ants': {'v2': ('groom_any',), 'v3': ('groom_any', 'on_lid_any')}}


def cmd_vcodes(args):
    """Codes of every dictionary on the labelled scored frames S; per neuron R~ for each label (all S frames)."""
    import torch
    from src.eci.sae import load_sae
    from src.eci.v2.score import half_stats
    sys.path.insert(0, str(REPO / 'scripts/eci/v2_diag'))
    import diag as D
    import matched as M
    dev = torch.device('cuda')
    dom = args.dom
    A = arms_of(dom)
    od = OUT / dom / 'valid'
    od.mkdir(parents=True, exist_ok=True)
    kinds = {'mice': {'v1': 'diag'}, 'ants': {'v2': 'scored_v2', 'v3': 'scored_v3'}}[dom]
    for ser, kind in kinds.items():
        stores = {}
        for fam_mask in ('v1', 'M0', 'N2'):
            if fam_mask == 'v1':
                St = M.V1Store(dom, kind, dev)
            else:
                St = D.T2.arm_store(D.T2.Union(dom, kind), fam_mask, dev, on_gpu=True)
                St.d = 1024
            if dom == 'mice':
                Y = D.MT.mice_sets(St)[0]
                labs = list(D.FZ.MLAB)
            else:
                Y, _, labs = D.MT.ants_set(St, ser)
            want = [labs.index(l_) for l_ in LABELS[dom][ser]]
            stores[fam_mask] = (St, torch.from_numpy(Y[:, want]).to(dev))
        for arm, lst in A.items():
            for (s, p, W, m, fam) in lst:
                f = od / f'{arm}_s{s}_{ser}.npz'
                if f.exists():
                    continue
                St, Yg = stores['v1' if fam == 'v1' else m]
                sae, norm, _ = load_sae(p, dev)
                X = D.codes(St, sae, norm)
                hs = half_stats(X, Yg, torch.arange(St.nf, device=dev))
                np.savez(f, R=hs['R'], A=hs['A'], AP=hs['AP'], rate=hs['rate'], base=hs['base'],
                         labels=np.array(LABELS[dom][ser]))
                log(f'vcodes {dom} {ser} {arm}_s{s}: best R~ ' +
                    ', '.join(f'{l_} {np.nanmax(hs["R"][:, i]):.3f} (n{int(np.nanargmax(hs["R"][:, i]))})'
                              for i, l_ in enumerate(LABELS[dom][ser])))
                del X
        del stores
        torch.cuda.empty_cache()


def auroc(pos, neg):
    from scipy.stats import mannwhitneyu
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def cmd_validate(args):
    from scipy import stats
    res = {}
    rows = []
    for dom in ('mice', 'ants'):
        A = arms_of(dom)
        res[dom] = {}
        for arm, lst in A.items():
            for (s, p, W, m, fam) in lst:
                T = pd.read_parquet(OUT / dom / 'suite' / f'{arm}_s{s}.parquet')
                Rs = []
                for ser in LABELS[dom]:
                    z = np.load(OUT / dom / 'valid' / f'{arm}_s{s}_{ser}.npz')
                    for i, lb in enumerate(z['labels']):
                        r = z['R'][:, i]
                        Rs.append(r)
                        for pref, lim in (('full', W), ('256', 256)):
                            j = int(np.nanargmax(np.where(np.isfinite(r[:lim]), r[:lim], -np.inf)))
                            t = T.iloc[j]
                            rows.append(dict(dom=dom, arm=arm, seed=s, series=ser, label=str(lb), prefix=pref, neuron=j,
                                             R=float(r[j]), eligible=bool(t.eligible), behaviour=bool(t.behaviour),
                                             group=int(t.group), **{c: bool(t[c]) for c in ('c1', 'c2', 'c3', 'c4', 'c5', 'c6')},
                                             n_checks=int(t.n_checks), off_share=t.off_share, eta2=t.eta2, tau_s=t.tau_s,
                                             r2_cond=t.r2_cond, repro=t.repro, inv_min=t.inv_min,
                                             extent=t.extent_bodies, level=t.level))
                a = np.nanmax(np.stack(Rs), 0)
                el = T.eligible.values
                b = T.behaviour.values
                au = auroc(a[el & b & np.isfinite(a)], a[el & ~b & np.isfinite(a)])
                ok = el & np.isfinite(a)
                rho = float(stats.spearmanr(T.n_checks.values[ok], a[ok]).statistic) if ok.sum() > 2 else float('nan')
                bestB = float(np.nanmax(np.where(b, a, np.nan))) if b.any() else float('nan')
                res[dom].setdefault(arm, {})[s] = {'enrich_auroc': au, 'spearman_checks_vs_alignment': rho,
                                                   'best_alignment_among_behaviour': bestB,
                                                   'best_alignment_all': float(np.nanmax(a))}
    L = pd.DataFrame(rows)
    L.to_csv(OUT / 'validation_neurons.csv', index=False)
    out = {'per_dictionary': res}
    for dom in ('mice', 'ants'):
        f = L[(L.dom == dom) & (L.prefix == 'full')]
        e = np.array([v['enrich_auroc'] for a_ in res[dom].values() for v in a_.values()], float)
        e = e[np.isfinite(e)]
        ci = stats.t.interval(0.95, len(e) - 1, loc=e.mean(), scale=e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else (np.nan, np.nan)
        out[dom] = {'labelled_pass_all_share_full': float(f.behaviour.mean()),
                    'labelled_pass_all_share_256': float(L[(L.dom == dom) & (L.prefix == '256')].behaviour.mean()),
                    'labelled_pass_rate_per_check_full': {c: float(f[c].mean()) for c in ('c1', 'c2', 'c3', 'c4', 'c5', 'c6')},
                    'labelled_eligible_share_full': float(f.eligible.mean()),
                    'enrich_auroc_mean': float(e.mean()) if len(e) else None, 'enrich_auroc_ci95': list(ci),
                    'n_dictionaries': int(len(e))}
        out[dom]['validated'] = bool(out[dom]['labelled_pass_all_share_full'] >= 0.5 and ci[0] > 0.5)
    out['SUITE_VALIDATED'] = bool(out['mice']['validated'] and out['ants']['validated'])
    (OUT / 'validation.json').write_text(json.dumps(out, indent=1, default=float))
    log(json.dumps({k: v for k, v in out.items() if k != 'per_dictionary'}, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- top neurons / sheets
def cmd_sheets(args):
    """Top behaviour neurons of the deployment seed of the arm with the highest mean full capacity (or --arms) and
    contact sheets: 16 D frames with the highest code (<= 2 per video), per-token activation overlay (red, alpha ~ code /
    the neuron's max over the sheet), kept-token outline not drawn."""
    import torch
    from PIL import Image, ImageDraw
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    dom = args.dom
    A = arms_of(dom)
    caps = {}
    for arm, lst in A.items():
        v = [json.loads((OUT / dom / 'suite' / f'{arm}_s{s}.json').read_text())['capacity_full'] for (s, *_r) in lst]
        caps[arm] = float(np.mean(v))
    arms = args.arms.split(',') if args.arms else [max(caps, key=caps.get)]
    Pr = pd.read_parquet(OUT / dom / 'probe.parquet')
    out = {}
    for arm in arms:
        s = deploy_seed(dom, arm)
        s_, p, W, m, fam = next(x for x in A[arm] if x[0] == s)
        T = pd.read_parquet(OUT / dom / 'suite' / f'{arm}_s{s}.parquet')
        vf = {ser: OUT / dom / 'valid' / f'{arm}_s{s}_{ser}.npz' for ser in LABELS.get(dom, {})}
        al = {}
        for ser, f in vf.items():
            if f.exists():
                z = np.load(f)
                for i, lb in enumerate(z['labels']):
                    al[f'{ser}_{lb}'] = z['R'][:, i]
        top = T[T.founder].sort_values('repro', ascending=False).head(8)
        X = np.load(BEE / dom / 'codes' / f'{arm}_s{s}_D.npy')
        St = Store(dom, fam, 'dense', 1024 if fam == 'v2' else 768)
        key = pd.MultiIndex.from_arrays([St.F.vid.values, St.F.k.values])
        srow = pd.Series(np.arange(St.nf), index=key)
        sae, norm, _ = load_sae(p, dev)
        keep = St.keep(m)
        recs = []
        se_cache = {}
        for rank, t in enumerate(top.itertuples()):
            j = int(t.neuron)
            rec = {k: (float(getattr(t, k)) if isinstance(getattr(t, k), (float, np.floating)) else getattr(t, k))
                   for k in ('neuron', 'rate_D', 'off_share', 'eta2', 'tau_s', 'r2_cond', 'repro', 'inv_min',
                             'extent_bodies', 'level')}
            rec['group_size'] = int((T.group == j).sum())
            rec['label_R~ (validation only)'] = {k: float(v[j]) for k, v in al.items()}
            if rank < 6:
                o = np.argsort(-X[:, j].astype(np.float32))
                pick, per = [], {}
                for i in o:
                    v_ = int(Pr.vid.values[i])
                    if per.get(v_, 0) < 2 and X[i, j] > 0:
                        pick.append(i)
                        per[v_] = per.get(v_, 0) + 1
                    if len(pick) == 16:
                        break
                tiles = []
                for i in pick:
                    r = Pr.iloc[i]
                    fi = int(srow[(int(r.vid), int(r.k))])
                    a, b = int(St.starts[fi]), int(St.starts[fi + 1])
                    kk = keep[a:b]
                    z = torch.from_numpy(St.tok(a, b)[kk]).to(dev)
                    with torch.no_grad():
                        c = sae.encode(norm(z.float()), mode='threshold')[:, j].float().cpu().numpy()
                    pos = St.pos[a:b][kk]
                    G = 32 if fam == 'v1' else int(St.F.G.values[fi])
                    hm = np.zeros(G * G, np.float32)
                    hm[pos] = c
                    img = frame_image(dom, fam, r, se_cache, dev)
                    tiles.append((img, hm.reshape(G, G), f'{r.obs} k{r.k} {X[i, j]:.2f}'))
                mx = max(float(h.max()) for _, h, _ in tiles) or 1.0
                tile = 256
                sh = Image.new('RGB', (4 * tile, 4 * tile))
                for q, (img, h, title) in enumerate(tiles):
                    im = Image.fromarray(img).resize((tile, tile), Image.BILINEAR)
                    hh = np.asarray(Image.fromarray((h / mx * 255).clip(0, 255).astype(np.uint8)).resize((tile, tile), Image.BILINEAR)) / 255.0
                    base = np.asarray(im).astype(np.float32)
                    al_ = (0.65 * hh ** 0.7)[..., None]
                    comp = base * (1 - al_) + np.array([255, 30, 30]) * al_
                    t_ = Image.fromarray(comp.clip(0, 255).astype(np.uint8))
                    ImageDraw.Draw(t_).text((3, 3), title, fill=(255, 255, 0))
                    sh.paste(t_, ((q % 4) * tile, (q // 4) * tile))
                pth = OUT / 'sheets' / f'{dom}_{arm}_s{s}_n{j}.jpg'
                pth.parent.mkdir(parents=True, exist_ok=True)
                sh.save(pth, quality=85)
                rec['sheet'] = str(pth)
            recs.append(rec)
            log(f'{dom} {arm}_s{s} #{rank + 1}: {json.dumps(rec, default=float)}')
        out[f'{arm}_s{s}'] = {'capacity_mean': caps, 'top': recs}
    (OUT / dom / 'top_neurons.json').write_text(json.dumps(out, indent=1, default=float))


def frame_image(dom, fam, r, cache, dev):
    """(S, S, 3) uint8 canonical image of a probe frame as the family's encoder saw it."""
    if fam == 'v1':
        from PIL import Image
        from src.eci.foreground import align_rot90, rotate_image
        c = V1[dom]
        if 'ann' not in cache:
            ann = REPO / c['ann']
            cache['ann'] = (pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values, align_rot90(c['align'], ann),
                            row_index(ann))
        fp, rot, ri = cache['ann']
        row = int(ri[(str(r.obs), int(r.k))])
        im = Image.open(REPO / 'dataset' / fp[row]).convert('RGB')
        if rot is not None:
            im = rotate_image(im, rot[row])
        return np.asarray(im)
    from src.eci.v2.pipeline import Series
    se = cache.setdefault(r.series, Series(series_cfg(dom, r.series), dev, load_encoder=False))
    v = se.V[r.obs]
    fr = se.run_job(**se.job(v, int(r.k), 1, threads=2))
    x, _ = se.frames(v, fr, [int(r.k)])
    return x[0].permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------------------------------------- table
def ms(x):
    x = np.asarray(x, float)
    return f'{x.mean():.1f} ± {x.std(ddof=1):.1f}' if len(x) > 1 else f'{x.mean():.1f}'


def cmd_table(args):
    sys.path.insert(0, str(REPO))
    from src.eci.v2.score import welch
    L = ['# Behaviour-neuron suite (label-free) - per representation and domain', '',
         'Capacity = number of DISTINCT behaviour neurons (pass all six checks, de-duplicated at Pearson r >= 0.5 or '
         'firing Jaccard >= 0.5). Mean ± sd over seeds (v1 5, v2dep 3, V2F 5). Welch 95% CI vs v1.', '',
         '| domain | arm (n seeds) | W | eligible | behaviour neurons | capacity full | capacity @256 | N_eff | share | '
         'capacity full - v1 (Welch) | @256 - v1 (Welch) |', '|---|---|---|---|---|---|---|---|---|---|---|']
    res = {}
    for dom in SERIES:
        if not (OUT / dom / 'suite').exists():
            continue
        A = arms_of(dom)
        S = {arm: [json.loads((OUT / dom / 'suite' / f'{arm}_s{s}.json').read_text()) for (s, *_r) in lst
                   if (OUT / dom / 'suite' / f'{arm}_s{s}.json').exists()] for arm, lst in A.items()}
        res[dom] = S
        for arm, ss in S.items():
            if not ss:
                continue
            g = lambda k: [x[k] for x in ss]  # noqa: E731
            d_f = d_2 = ''
            if arm != 'v1' and 'v1' in S and len(ss) > 1:
                a, lo, hi = welch(g('capacity_full'), [x['capacity_full'] for x in S['v1']])
                d_f = f'{a:+.1f} [{lo:+.1f}, {hi:+.1f}]'
                a, lo, hi = welch(g('capacity_256'), [x['capacity_256'] for x in S['v1']])
                d_2 = f'{a:+.1f} [{lo:+.1f}, {hi:+.1f}]'
            L.append(f'| {dom} | {arm} ({len(ss)}) | {ss[0]["W"]} | {ms(g("n_eligible"))} | {ms(g("n_behaviour"))} | '
                     f'{ms(g("capacity_full"))} | {ms(g("capacity_256"))} | {ms(g("n_eff"))} | '
                     f'{np.mean(g("share")):.3f} | {d_f} | {d_2} |')
    L += ['', '## Median score per check over eligible neurons (mean over seeds) and pass rate among eligible', '',
          '| domain | arm | off-animal share (C1, <= 0.5) | eta2 between videos (C2, <= 0.5) | tau s (C3, 0.2-120) | '
          'conditions R2cv (C4, <= 0.5) | reproducibility r (C5, >= 0.5) | invariance min r (C6, >= 0.5) | pass rates C1-C6 |',
          '|---|---|---|---|---|---|---|---|---|']
    for dom, S in res.items():
        for arm, ss in S.items():
            if not ss:
                continue
            md = {k: np.nanmean([x['median'][k] if x['median'][k] is not None else np.nan for x in ss])
                  for k in ss[0]['median']}
            pr = {k: np.mean([x['pass_rate_eligible'][k] for x in ss]) for k in ss[0]['pass_rate_eligible']}
            L.append(f'| {dom} | {arm} | {md["off_share"]:.3f} | {md["eta2"]:.3f} | {md["tau_s"]:.2f} | '
                     f'{md["r2_cond"]:.3f} | {md["repro"]:.3f} | {md["inv_min"]:.3f} | '
                     + ' / '.join(f'{v:.2f}' for v in pr.values()) + ' |')
    L += ['', '## NES prefilter: candidates (behaviour neurons inside the prefix), mean over seeds; deployment seed in '
          'brackets; Bonferroni per-test alpha = 0.05 / m', '',
          '| domain | arm | 128 | 256 | 512 | W | alpha @256 today -> filtered (deployment seed) |', '|---|---|---|---|---|---|---|']
    for dom, S in res.items():
        for arm, ss in S.items():
            if not ss:
                continue
            dep = next((x for x in ss if x['deployed']), ss[0])
            W = str(ss[0]['W'])
            c = lambda p: f'{np.mean([x["prefilter_candidates"][p] for x in ss]):.1f} [{dep["prefilter_candidates"][p]}]'  # noqa
            L.append(f'| {dom} | {arm} | {c("128")} | {c("256")} | {c("512")} | {c(W)} | '
                     f'{0.05 / 256:.2e} -> {dep["bonferroni"]["256"]["filtered"]:.2e} |')
    vf = OUT / 'validation.json'
    if vf.exists():
        V = json.loads(vf.read_text())
        L += ['', '## Validation (labels used only here)', '']
        for dom in ('mice', 'ants'):
            v = V[dom]
            L.append(f'- {dom}: labelled behaviour neurons passing all checks {v["labelled_pass_all_share_full"]:.2f} (full) / '
                     f'{v["labelled_pass_all_share_256"]:.2f} (@256); eligible {v["labelled_eligible_share_full"]:.2f}; per check '
                     + ', '.join(f'{k} {x:.2f}' for k, x in v['labelled_pass_rate_per_check_full'].items())
                     + f'; enrichment AUROC (alignment of behaviour vs other eligible neurons) {v["enrich_auroc_mean"]:.3f} '
                       f'[{v["enrich_auroc_ci95"][0]:.3f}, {v["enrich_auroc_ci95"][1]:.3f}]; validated: {v["validated"]}')
        L.append(f'- SUITE VALIDATED (pre-registered rule): {V["SUITE_VALIDATED"]}')
        L += ['', '| domain | arm | enrichment AUROC | Spearman(#checks, alignment) | best R~ among behaviour neurons | '
              'best R~ overall |', '|---|---|---|---|---|---|']
        for dom in ('mice', 'ants'):
            for arm, d_ in V['per_dictionary'][dom].items():
                vals = list(d_.values())
                f = lambda k: np.nanmean([x[k] for x in vals])  # noqa: E731
                L.append(f'| {dom} | {arm} | {f("enrich_auroc"):.3f} | {f("spearman_checks_vs_alignment"):.3f} | '
                         f'{f("best_alignment_among_behaviour"):.3f} | {f("best_alignment_all"):.3f} |')
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('step')
    p.add_argument('--dom', default='')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--arms', default='')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    STEPS[a.step](a)


STEPS = {'plan': cmd_plan, 'enc2': cmd_enc2, 'enc1': cmd_enc1, 'codes': cmd_codes, 'suite': cmd_suite,
         'vcodes': cmd_vcodes, 'validate': cmd_validate, 'table': cmd_table, 'sheets': cmd_sheets}

if __name__ == '__main__':
    main()
