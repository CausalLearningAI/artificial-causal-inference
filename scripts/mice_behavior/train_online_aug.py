"""Online per-batch DINOv2 encoding from a JPEG-bytes RAM cache, with D4 augmentation.

Why this exists
---------------
train_patchgrid_online.py caches DINOv2 *tokens*, which is both the largest possible
representation and the one thing that makes pixel-space augmentation impossible: tokens are
computed from one fixed rendering of each frame, so a rotated image needs different tokens.
That restricted us to embedding-space augmentation (patch dropout / noise / frame dropout),
and a 4-way ablation of those found no effect.

Caching the JPEG BYTES instead and encoding per batch inverts every part of that trade:

    cache contents          size (444k frames)     augmentation?
    DINOv2 tokens @448px    650 GiB                no
    DINOv2 tokens @224px    163 GiB                no
    raw JPEG bytes           17 GiB (measured)     YES

Costs ~2-10x more encoder forward passes (each frame is re-encoded every epoch it is drawn
rather than once), but the encoder was never the bottleneck: frame reads are NFS-latency-bound
at ~100 ms/frame, and this cache pays that cost EXACTLY ONCE, after which every epoch reads
from RAM. It also removes the 300-700 GiB allocations that caused repeated OOM aborts.

Why D4 and not arbitrary rotation
---------------------------------
The cage is filmed top-down, so the dihedral group D4 (4 rotations x optional flip = 8
variants) is a genuine symmetry of the data: "nose-to-tail sniffing" does not depend on the
camera's orientation. Restricting to multiples of 90 degrees keeps the transform EXACT --
no interpolation blur, no black corners, no resampling artefacts, unlike arbitrary angles.
With only ~1,186 nt and ~3,696 nn independent bouts in train, this 8x label-preserving
expansion targets the actual constraint, which is effective sample size.

The SAME transform is applied to every frame in a sample's context window. Applying
independent transforms per frame would destroy the temporal relationship, which the
context_k sweep showed is what nose-tail detection depends on (nt full-val AP 0.0449 at
context_k=0 vs 0.1411 at context_k=2, a 3.1x gap; nn is nearly context-free by comparison).

Usage:
    python scripts/mice_behavior/train_online_aug.py --context-k 2 --augment d4
"""
import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.checkpoint          # not pulled in by `import torch` alone
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import get_fixed_val_pools
from src.mice_behavior.metrics import ap_report, format_ap_report, rate_report, format_rate_report
from src.mice_behavior.viz import plot_confusion_examples, plot_error_strips
from src.mice_behavior.head_cfg import get_head_cfg
from train_patchgrid_online import dummy_loader

MODEL_ID = 'facebook/dinov2-base'
EMB_DIM, PATCH_SIZE = 768, 14
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DATASET_ROOT = Path('dataset')


class _BytesReader(Dataset):
    """Reads raw JPEG bytes off NFS. Used once, with many workers: reads are latency-bound
    (~100 ms/frame; the decode itself is only ~2-3 ms), so workers sit blocked on I/O and
    should far exceed the CPU count -- 48 workers measured 4.8x over 16."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with open(DATASET_ROOT / self.paths[i], 'rb') as f:
            return i, np.frombuffer(f.read(), dtype=np.uint8)


def d4_transform(img: Image.Image, op: int) -> Image.Image:
    """One of the 8 dihedral-group symmetries. Exact: 90-degree rotations and flips are
    lossless pixel permutations, so no interpolation or padding is introduced."""
    if op & 4:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    r = op & 3
    if r == 1:
        img = img.transpose(Image.ROTATE_90)
    elif r == 2:
        img = img.transpose(Image.ROTATE_180)
    elif r == 3:
        img = img.transpose(Image.ROTATE_270)
    return img


class _SampleDataset(Dataset):
    """Decodes + augments + normalises one whole batch of samples per __getitem__.

    jpeg_cache is inherited by workers through fork copy-on-write, so the ~17 GiB is shared,
    not duplicated per worker.
    """

    # Set by main() from --pixel-source. 0 = off. When >0 each frame is first resized DOWN to
    # this edge length and then back up to input_size, so the encoder still sees exactly
    # (input_size/14)^2 tokens while the pixel information is capped at pixel_source^2. This is
    # the only ablation in the repo that separates TOKEN COUNT from PIXEL RESOLUTION: every
    # previous resolution experiment moved both together, because DINOv2's patch size is fixed
    # at 14 so input_size alone sets both. A mouse is ~35 px in the 512 px stored frame -> 2.19
    # patches at 448 and 2.46 at 504, which is why 504 bought nothing; whether the 224->448
    # jump (+41% macro AP) was extra tokens or extra pixels has never been tested.
    pixel_source = 0

    def __init__(self, meta, batches, jpeg_cache, input_size, augment, seed,
                 photo_brightness=(0.80, 1.25), photo_contrast=(0.80, 1.25),
                 photo_gamma=(0.83, 1.20), fixed_op=None, envs=None):
        # envs: (n_samples,) int env code per sample, or None -> zeros. Carried through the
        # loader so the training loop can group a batch's per-sample losses by environment
        # without re-deriving the observation of every anchor on the hot path.
        self.envs = envs
        # fixed_op pins the D4 transform instead of drawing it, which is what test-time
        # augmentation needs: the same deterministic rendering applied to every sample, then
        # averaged over ops. None keeps the training behaviour (draw per sample).
        self.fixed_op = fixed_op
        self.photo_brightness = photo_brightness
        self.photo_contrast = photo_contrast
        self.photo_gamma = photo_gamma
        # jpeg_cache is keyed by GLOBAL frame index and is populated lazily, epoch by epoch.
        # Workers fork per-epoch (a new DataLoader is built each epoch for the resampled
        # negatives), so each fork inherits whatever the main process has cached so far.
        self.meta, self.batches, self.cache = meta, batches, jpeg_cache
        self.input_size = input_size
        self.augment, self.seed = augment, seed

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, bi):
        sample_idx = self.batches[bi]
        offsets = self.meta.offsets_grid
        abs_idx = self.meta.gi[sample_idx][:, None] + offsets[None, :]
        mask = self.meta.pad_mask[sample_idx]
        B, T = abs_idx.shape
        S = self.input_size
        out = torch.zeros((B, T, 3, S, S), dtype=torch.float32)
        rng = np.random.default_rng((self.seed, bi))
        use_d4 = self.augment in ('d4', 'd4_photo')
        use_photo = self.augment == 'd4_photo'
        for b in range(B):
            # one transform per SAMPLE, shared by all T context frames -- per-frame transforms
            # would break the temporal relationship nose-tail detection relies on.
            op = (self.fixed_op if self.fixed_op is not None
                  else (int(rng.integers(0, 8)) if use_d4 else 0))
            # Photometric jitter, drawn once per sample for the same reason: independent
            # per-frame jitter would inject spurious frame-to-frame change into a window the
            # model reads temporally. D4 gives only 8 exact renderings of each positive, and
            # with every positive drawn every epoch that is the binding limit on effective
            # sample size -- this makes the rendering continuous instead. It also attacks
            # video-level appearance nuisance (lighting/exposure differs per recording), the
            # plausible reason frame ranking is much stronger than between-video rate ranking.
            # Frames are pure grayscale (R=G=B exactly, verified), so hue/saturation jitter
            # would be a no-op; brightness/contrast/gamma are the only meaningful axes.
            if use_photo:
                bright = float(rng.uniform(*self.photo_brightness))
                contrast = float(rng.uniform(*self.photo_contrast))
                gamma = float(rng.uniform(*self.photo_gamma))
            for t in range(T):
                if mask[b, t]:
                    continue
                buf = self.cache[int(abs_idx[b, t])]
                with Image.open(io.BytesIO(buf.tobytes())) as im:
                    im = im.convert('RGB')
                    if op:
                        im = d4_transform(im, op)
                    if self.pixel_source:
                        # down then up: destroys pixel detail, leaves the token grid untouched.
                        im = im.resize((self.pixel_source, self.pixel_source), Image.BILINEAR)
                    im = im.resize((S, S), Image.BILINEAR)
                    arr = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())
                x = arr.permute(2, 0, 1).float() / 255.0
                if use_photo:
                    x = x.pow(gamma).mul(bright)
                    m = x.mean()
                    x = ((x - m) * contrast + m).clamp_(0.0, 1.0)
                out[b, t] = (x - IMAGENET_MEAN) / IMAGENET_STD
        env = (torch.from_numpy(self.envs[sample_idx]) if self.envs is not None
               else torch.zeros(B, dtype=torch.int64))
        return (out,
                torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()),
                torch.from_numpy(self.meta.labels[sample_idx]),
                torch.from_numpy(mask),
                env)


def population_var(ann_csv, train_obs, o2e, env_names, cols=('Y_nt', 'Y_nn')):
    """Var(Y | E=e) per environment and label over EVERY annotated frame of the training
    observations -- the POPULATION, not the bounded anchor set.

    This has to be read off the annotations rather than off `tm`. FrameBatchData is bounded by
    --max-train-frames and keeps positives preferentially: a real run holds 79,546 anchors out of
    2,574,000 annotated frames, 18% of them positive against a true rate of 0.4-1.4%. Estimating
    the confound on that is estimating it on a set that has already had most of the confound
    removed, unevenly.
    """
    a = pd.read_csv(ann_csv, usecols=['observation_id', *cols], low_memory=False)
    a = a[a.observation_id.isin(set(train_obs))].dropna(subset=[cols[0]])
    a = a.assign(_e=a.observation_id.map(o2e))
    idx = {n: i for i, n in enumerate(env_names)}
    out = np.full((len(env_names), len(cols)), np.nan)
    for name, g in a.groupby('_e', sort=False):
        i = idx.get(str(name))
        if i is None:
            continue
        for l, c in enumerate(cols):
            q = float((g[c] > 0.5).mean())
            out[i, l] = q * (1.0 - q)
    return out


def derm_table(labels: np.ndarray, envs: np.ndarray, idx: np.ndarray, n_env: int,
               floor: float = 0.02, var_pop=None):
    """DERM per-sample weights as a lookup table [env, label, y], normalised to mean 1 over `idx`.

    WHAT DERM ASKS FOR, stated as a target on the reweighted training distribution:

        mass(y=1, e) == mass(y=0, e)   for every e     the environment carries no information
                                                       about how PREVALENT the behaviour is
        mass(  .  , e) proportional to Var(Y | E=e)    an environment where nothing varies stops
                                                       dominating by sheer size

    A sample's weight is therefore its cell's target mass divided by how often the sampler draws
    that cell:

        w(y, e) = Var(Y | E=e) / P_sampled(Y=y, E=e)

    and that single line is the whole method. Note what it does NOT contain: no environment prior
    and no keep-rate. Both cancel, because the target is a mass and the denominator is whatever
    distribution the loss happens to be averaged over. Feed it the SAMPLED variance and you get
    the old behaviour exactly; feed it the POPULATION variance and you get DERM as derived.

    WHY THAT DISTINCTION IS THE WHOLE BUG (Riccardo, 2026-08-25). Training subsamples twice --
    --max-train-frames keeps positives preferentially, then neg_ratio balances 1:1 -- which drives
    p_e from 0.4-1.4% up to 16-39%. Var(Y|E) = p(1-p) SATURATES there, so the ratio across
    environments collapses: measured on the fear session, 3.56x -> 1.58x on nose-to-tail and
    2.43x -> 1.32x on nose-to-nose. 77-78% of the confound is gone before DERM sees it. The
    confound is a property of the POPULATION and the subsampling is a compute convenience, so the
    numerator must be the population variance.

    IT ALSO FIXES THE DURATION PROBLEM for free. The old form divided by P(E), the environment's
    share of anchors, which for --env-key phase IS duration -- H runs 30 minutes against O and P's
    15, so P_e came out 0.50/0.25/0.25 and on the social session the positive weights varied 2.5x
    across environments on phase length alone, carrying no prevalence signal at all. Here the
    target mass depends only on a RATE, Var(Y|E=e), so a longer phase gets no extra mass for being
    longer. That is the "Y mean, not total" point.

    Compare arms on the MASS, printed by the caller, never on the per-sample weight: a cell drawn
    less often must carry more weight per sample to reach the same mass, so per-sample spreads are
    not comparable across arms.

    `floor` clips the variance away from 0 so an environment holding no positives for one label is
    strongly downweighted rather than deleted (Var = 0 would zero it out and silently drop data).
    """
    y = labels[idx]
    e = envs[idx].astype(np.int64)
    n, L = y.shape
    yi = (y > 0.5).astype(np.int64)
    v_floor = floor * (1.0 - floor)

    tab = np.zeros((n_env, L, 2), dtype=np.float64)
    for l in range(L):
        # P_sampled(Y=y, E=e): what the loss is actually averaged over, this epoch.
        joint = np.zeros((n_env, 2), dtype=np.float64)
        for k in (0, 1):
            joint[:, k] = np.bincount(e, weights=(yi[:, l] == k).astype(np.float64),
                                      minlength=n_env)
        joint /= max(n, 1)
        # Var(Y|E=e): the population if we were given it, else the sampled set (old behaviour).
        if var_pop is not None and np.isfinite(var_pop[:, l]).all():
            var = var_pop[:, l].copy()
        else:
            tot = joint.sum(1)
            p = np.divide(joint[:, 1], tot, out=np.full(n_env, 0.5), where=tot > 0)
            var = p * (1.0 - p)
        if np.any(var < v_floor):
            # This has bitten twice. --derm-floor 0.02 is safe against the SAMPLED variance and
            # clips EVERY environment against the population one (Var = p(1-p) with p ~ 0.4-1.4%
            # is 0.004-0.013, under 0.0196), which makes the target mass uniform and the whole
            # correction an exact no-op. Never let that pass silently.
            print(f'  !! DERM variance floor {v_floor:.2e} BINDS on '
                  f'{int(np.sum(var < v_floor))}/{n_env} environments for label {l} '
                  f'(Var {np.array2string(var, precision=5)}) -- the target mass is being '
                  f'flattened toward a no-op. Lower --derm-floor.', flush=True)
        var = np.maximum(var, v_floor)
        for k in (0, 1):
            tab[:, l, k] = np.divide(var, joint[:, k], out=np.zeros(n_env),
                                     where=joint[:, k] > 0)

    w = tab[e[:, None], np.arange(L)[None, :], yi]
    m = float(w.mean())
    if m > 0:
        tab /= m
    return tab.astype(np.float32), m


def derm_mass(tab, labels, envs, idx, n_env):
    """Reweighted mass per (env, label, y) -- the quantity DERM actually specifies.

    Returns an array [env, label, y] summing to 1 over each label. Equal mass in y=0 and y=1
    within an environment means the prevalence channel is closed; mass across environments
    proportional to Var(Y|E) means the size channel is closed.
    """
    y = (labels[idx] > 0.5).astype(np.int64)
    e = envs[idx].astype(np.int64)
    n, L = y.shape
    out = np.zeros((n_env, L, 2), dtype=np.float64)
    for l in range(L):
        for k in (0, 1):
            sel = y[:, l] == k
            out[:, l, k] = np.bincount(e[sel], weights=tab[e[sel], l, k], minlength=n_env)
    return out / max(n, 1)


def checkpoint_blocks(blocks):
    """Recompute each block's internal activations during backward instead of storing them.

    WHY
    ---
    BitFit-6 at 448 px, batch 64, context_k=2 encodes 64 x 5 = 320 images of 1025 tokens per
    step, and backprop through six unfrozen blocks has to retain every intermediate tensor in
    each of them. Measured on an L40S that peaks at 42.53 GiB of a 44.42 GiB card and dies in
    the encoder forward, 1.88 GiB short (the DINOv2 MLP hidden, 320 x 1025 x 3072, fp16) --
    before a single optimiser step. Checkpointing keeps only each block's INPUT and re-runs the
    block in backward, trading roughly one extra encoder forward (~30% more step time) for the
    activation memory of six blocks. The A100 does not need this; the L40S fleet does, and
    there are nineteen of those against one A100 node.

    WHY use_reentrant=False IS MANDATORY HERE, and must not be "simplified" back
    ---------------------------------------------------------------------------
    With --unfreeze-blocks 6, blocks 0-5 are frozen, so the hidden state ARRIVING at the first
    checkpointed block does NOT require grad. The legacy reentrant implementation decides
    whether to build a backward graph by looking only at the checkpointed call's INPUTS: if no
    input requires grad it produces NO GRADIENTS AT ALL for anything inside the block --
    including the block's own trainable biases -- and it does not raise. The run trains to
    completion, the loss falls plausibly because the head still learns, and every BitFit tensor
    stays at its initial value. A silently null encoder fine-tune, seven hours long.

    The non-reentrant implementation tracks the block's own PARAMETERS as well as its inputs,
    so gradients reach the biases whether or not the incoming activations require grad. The
    tripwire in the training loop below (`encoder grad: N/M tensors, norm ...`) exists to catch
    a regression here, and it should stay.

    preserve_rng_state is left at its default True so any stochastic op inside a block
    (DINOv2-base ships drop_path=0.0 and attention dropout 0.0, but that is a config value, not
    a guarantee) draws the same numbers on the recomputation as it did on the forward.
    """
    for blk in blocks:
        orig = blk.forward

        def fwd(*a, _orig=orig, **kw):
            # Under no_grad -- evaluate(), and the full-val pass at the end -- there is no
            # backward to recompute for, so call straight through and keep eval speed exactly
            # as it was.
            if not torch.is_grad_enabled():
                return _orig(*a, **kw)
            return torch.utils.checkpoint.checkpoint(_orig, *a, use_reentrant=False, **kw)

        blk.forward = fwd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--context-k', type=int, default=2)
    p.add_argument('--stride', type=int, default=1,
                    help='context positions sit at +-context_k*stride in steps of stride, so the '
                         'window widens WITHOUT adding positions: still 2k+1 frames encoded, same '
                         'token and attention cost. At 5 fps, k=2 spans only +-0.4 s at stride 1 '
                         'vs +-0.8 s at stride 2. nt is the context-dependent label (AP monotone '
                         'in k: 0.105/0.154/0.197 at 504px) and the one furthest from target, so '
                         'reach is worth probing separately from density.')
    p.add_argument('--neg-ratio', type=int, default=1)
    p.add_argument('--max-train-frames', type=int, default=300_000,
                    help='NOT a cap on training data in the way the name suggests, and easy to '
                         'misread: FrameBatchData keeps EVERY positive frame (+ its +-reach '
                         'context) unconditionally, BEFORE this budget is consulted, then tops up '
                         'with random NEGATIVE frames until the budget is reached. So all 23,280 '
                         'positive anchors (~4.9k bouts) are already in at the 300k default, and '
                         'raising this buys negative DIVERSITY only -- never more positives, never '
                         'more bouts. Epoch cost is unaffected too: with neg_ratio=1 an epoch is '
                         'n_pos positives + n_pos sampled negatives regardless of pool size. '
                         'The original rationale (embedding RAM: patch-grid tokens are 16x CLS) is '
                         'void on this online path, which passes a dummy 1-dim loader and deletes '
                         '.flat immediately -- no embeddings are ever stored. What still binds is '
                         'the JPEG-bytes cache at ~45 KB/frame: 300k -> ~19 GiB, 900k -> ~45 GiB, '
                         'unbounded (2.59M) -> ~114 GiB, plus the one-time NFS read of each frame.')
    p.add_argument('--input-size', type=int, default=224)
    p.add_argument('--train-odour', default='', choices=['', 'F', 'S'],
                   help="Restrict TRAINING to one exposure session and evaluate on the other. "
                        "Each pool is filmed twice -- fear and social -- through the same three "
                        "phases, so this holds the cage, the animals, the annotator and the "
                        "lighting fixed and varies only the exposure. It is a MECHANISM split, "
                        "not a deployment split: the model has seen every test pool, so its bias "
                        "there is smaller than on a genuinely unseen pool, and PPI++ cannot use it "
                        "(the rectifier would sit on trained-on pools). What it buys is that the "
                        "two exposures carry OPPOSITE true effects on nose-to-tail, so a model "
                        "importing its training session's phase prior biases the test session in a "
                        "direction that flips when the split direction flips -- which a plain "
                        "generalisation gap cannot do. The monitor set stays inside the TRAINING "
                        "exposure (the held-out pools' same-odour recordings) so early stopping "
                        "never touches the test session.")
    p.add_argument('--val-pools', default=None,
                    help='comma-separated pool ids to hold out, overriding the standing split. '
                         'Used for cross-fitting: several runs whose val sets tile all 24 '
                         'annotated pools give the out-of-fold predictions PPI requires.')
    p.add_argument('--pool-grid', type=int, default=0,
                    help='pool each frame into a GxG grid of REGION vectors instead of collapsing '
                         'it to one, and run the temporal attention over T*G^2 region tokens with '
                         'separable temporal+spatial position codes. 0 = off (current behaviour). '
                         'Fixes the structural defect that all spatial layout is destroyed before '
                         'the temporal stage -- the reason --use-motion could not work. G=4 at '
                         '448px gives 16 regions of 8x8 patches each.')
    p.add_argument('--n-train-pools', type=int, default=0,
                    help='train on only this many of the 20 labelled training pools (0 = all). '
                         'Nested across sizes at fixed --seed, so smaller points are subsets of '
                         'larger ones. This is the learning curve that matters: it varies the '
                         'number of ANNOTATED CAGES, which is what a labelling budget buys.')
    p.add_argument('--env-key',
                    choices=['none', 'condition', 'phase', 'annotator', 'pool', 'line'],
                    default='none',
                    help="grouping variable defining vREx environments. 'none' = plain ERM. "
                         "'condition' = phase x odor, the 6 EXPERIMENTAL CELLS -- the motivated "
                         "default for the phase-ATE estimand: the odor delivery visibly changes "
                         "the scene, so a classifier can acquire a condition-dependent error, and "
                         "a bias that moves WITH the treatment is exactly what biases an ATE "
                         "estimated without a rectifier (measured: predicted/true rate ratio "
                         "2.26 (H) / 1.80 (O) / 3.51 (P) on the best model). Equalising RISK "
                         "across cells removes that bias without touching the effect itself. "
                         "'annotator' targets a different problem -- label-convention shift across "
                         "the 6 people who labelled v1 -- which is a training-time nuisance, not a "
                         "test-time environment, since v2 has no annotator at all.")
    p.add_argument('--derm', action='store_true',
                    help='DECONFOUNDED ERM (ours): reweight every sample by '
                         'Var(Y|E)/P(Y,E) over the --env-key environments, instead of adding '
                         "vREx's risk-variance penalty. For binary labels this equalises the "
                         'positive/negative mass inside each environment, which removes the '
                         '"phase predicts prevalence" shortcut without constraining the '
                         'predictions. Requires --env-key. Composes with --vrex-beta (the '
                         'weights are applied inside each per-environment risk), but the '
                         'motivated arm is DERM alone: --env-key phase --derm --vrex-beta 0.')
    p.add_argument('--derm-prevalence', choices=['sampled', 'population'], default='sampled',
                   help="which Var(Y|E) sets DERM's target mass. 'sampled' (default, and what "
                        'every arm before 2026-08-25 used) reads it off the subsampled epoch, where '
                        'p_e is pushed from 0.4-1.4% to 16-39%, p(1-p) saturates and the ratio '
                        'across environments collapses 3.56x -> 1.58x -- 77% of the confound gone '
                        "before DERM sees it. 'population' reads it off every annotated frame of "
                        'the training observations, which is where the confound actually lives. It '
                        'also removes the phase-duration artefact for free, since a rate carries no '
                        'information about how long the phase ran.')
    p.add_argument('--derm-floor', type=float, default=None,
                    help='clip P(Y=1|E) into [floor, 1-floor] so an environment with no '
                         'positives for one label is downweighted, not deleted. DEFAULT DEPENDS '
                         'ON --derm-prevalence, and it has to: 0.02 is safe against the SAMPLED '
                         'rate (1:1 subsampling puts every p_e at 16-39%, so it never binds) and '
                         'is CATASTROPHIC against the population rate (0.4-1.4%), where it clips '
                         'every environment to the floor, makes p_e identical everywhere and turns '
                         'the whole correction into an exact no-op -- verified, weight spread '
                         '1.00x. So: 0.02 for sampled, 1e-4 for population. A binding clip warns.')
    p.add_argument('--vrex-beta', type=float, default=0.0,
                    help='weight on the across-environment RISK VARIANCE. 0 = ERM even when '
                         '--env-key is set (useful as the exact-same-code control).')
    p.add_argument('--vrex-warmup-epochs', type=int, default=5,
                    help='epochs of pure ERM before the penalty switches on. Applying it from '
                         'step 0 admits the degenerate invariant solution (equally bad everywhere).')
    p.add_argument('--vrex-min-env', type=int, default=4,
                    help='minimum samples an environment needs IN A BATCH to contribute a risk '
                         'term; smaller groups give a mean too noisy to take a variance over.')
    p.add_argument('--pixel-source', type=int, default=0,
                    help='cap PIXEL detail at this edge length while keeping the token grid at '
                         '(input_size/14)^2: each frame is resized down to pixel_source then back '
                         'up to input_size. 0 = off. Separates token count from pixel resolution, '
                         'which every prior resolution experiment confounded (DINOv2 patch size is '
                         'fixed at 14, so input_size sets both at once). Run at input_size=448 '
                         'with --pixel-source 224 against the plain 448 control: if macro AP holds '
                         'up, the 224->448 gain (+41%) was TOKENS and the next lever is more tokens '
                         'per mouse (tiling); if it collapses to the 224 level, it was PIXELS and '
                         'only going back to the 2060x2062 source (per-animal crops) can help.')
    p.add_argument('--augment', choices=['none', 'd4', 'd4_photo'], default='d4',
                    help="'d4_photo' adds per-sample brightness/contrast/gamma jitter on top of "
                         'the 8 exact D4 renderings. Every positive is drawn every epoch, so with '
                         'd4 alone each one is seen 40 times across only 8 distinct renderings; '
                         'the photometric draw is continuous, so no two are identical.')
    p.add_argument('--photo-strength', type=float, default=1.0,
                    help='scales the d4_photo jitter ranges about 1.0 (0.5 = half as strong, '
                         '2.0 = double). 1.0 = brightness/contrast in [0.80, 1.25], gamma in '
                         '[0.83, 1.20].')
    p.add_argument('--n-epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--select', choices=['monitor_ap', 'last'], default='monitor_ap',
                   help="which epoch to keep. 'monitor_ap' (default) keeps the highest UNWEIGHTED "
                        'validation AP -- which for a DERM or vREx arm rewards exactly the '
                        'prior-exploitation the objective exists to remove, so the selection rule '
                        'and the objective pull against each other and the comparison against ERM '
                        "is confounded by it. 'last' keeps the final epoch and disables early "
                        'stopping: a fixed epoch budget, identical for both arms, so a DERM-vs-ERM '
                        'difference is attributable to the objective and to nothing else. Use it '
                        'for any arm whose point is a comparison across objectives.')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--read-workers', type=int, default=32, help='NFS reads: latency-bound, use many')
    p.add_argument('--decode-workers', type=int, default=16, help='decode+augment: CPU-bound, ~= n_cpus')
    p.add_argument('--val-monitor-size', type=int, default=12_500,
                    help='per-epoch monitor size. Was 25k, but the monitor pass ran 25k val '
                         'samples against a 46,560-sample training set -- a third of each epoch '
                         'spent monitoring rather than learning (visible as a 99%%->70%% GPU '
                         'utilisation dip). 12.5k still yields ~122 nt / ~216 nn positives, '
                         'about 9%% relative noise on the AP estimate, which is ample for RANKING '
                         'checkpoints; the reported number always comes from full val anyway.')
    p.add_argument('--cross-attn-dim', type=int, default=64)
    p.add_argument('--patch-pool-dim', type=int, default=256)
    p.add_argument('--patch-selfattn-dim', type=int, default=0,
                    help='0 = off (the head every prior run used). Non-zero inserts one '
                         'bottlenecked self-attention layer over the P patch tokens before '
                         'pooling -- the ONLY operation in this head that is a joint function of '
                         'two patch features, since attention scores are query-vs-key and both '
                         'existing attention ops pool a single query. Relational predicates like '
                         '"nose of one mouse at the tail of another" are therefore computable '
                         'only inside DINOv2\'s blocks under the default head, which is one '
                         'explanation for why unfreezing them helped; this supplies the missing '
                         'term to a FROZEN encoder so the two explanations separate. Costs '
                         'O(P^2) attention (P=1024 at 448px), hence the bottleneck. Injected as a '
                         'ZERO-INITIALISED residual, so at step 0 the model is exactly the '
                         'baseline head.')
    p.add_argument('--pool-queries', type=int, default=1,
                    help='number of PatchAttnPool queries, concatenated. 1 = every prior run. A '
                         'single query must compress a frame holding four mice into one weighted '
                         'average, while the label is a predicate over ONE pair; K queries let '
                         'different queries land on different regions. Cheapest capacity test '
                         'available (~K x patch_pool_dim new params plus a wider temporal_proj), '
                         'and MouseOPairClassifier has always used 4 -- the frame classifier '
                         'simply never inherited that design.')
    p.add_argument('--use-motion', action='store_true',
                    help='pool the per-patch delta to the previous context position through a '
                         'second PatchAttnPool and concatenate it with the content pooling (see '
                         'MouseFrameClassifier.use_motion). Motivated by a zero-learning probe: '
                         'the mean patch-delta L2 norm alone reaches ROC-AUC 0.61 (nt) / 0.54 (nn) '
                         'vs 0.53/0.46 for a same-shape raw-content-magnitude baseline, so the '
                         'frame-to-frame change carries signal the content-only path has to infer '
                         'indirectly. Costs one extra pooling pass (~5%% of step time; the frozen '
                         'encoder dominates).')
    p.add_argument('--lr-decay-epochs', type=int, default=6)
    p.add_argument('--warmup-epochs', type=int, default=0,
                    help='linear LR ramp before the cosine decay starts. The trainable stack has '
                         'no normalization anywhere by default (raw DINO patch tokens -> '
                         'unnormalized MHA -> unnormalized MHA -> MLP), and every run to date '
                         'started at the full LR on step 1; the res448 run scored monitor AP 0.107 '
                         'in epoch 1 vs 0.242 in epoch 2, consistent with a wasted/unstable first '
                         'epoch. 0 reproduces the old schedule exactly.')
    p.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam',
                    help="'adam' applies weight_decay as L2 coupled to Adam's adaptive scaling, "
                         'which makes it a near-no-op as a regularizer -- it was inherited '
                         'unexamined from a config tuned on a 4x4 coarse patch grid. Now that the '
                         'model demonstrably overfits (train loss still falling at epoch 40 while '
                         'val AP is flat from ~24), decoupled AdamW decay is the knob that '
                         'actually bites.')
    p.add_argument('--lr', type=float, default=None, help='override inherited cfg lr')
    p.add_argument('--weight-decay', type=float, default=None, help='override inherited cfg')
    p.add_argument('--dropout', type=float, default=None, help='override inherited cfg')
    p.add_argument('--unfreeze-blocks', type=int, default=0,
                    help='SUPERVISED fine-tuning: unfreeze the last N of DINOv2-base\'s 12 '
                         'transformer blocks and train them with the same BCE loss. Every result '
                         'to date comes from a FROZEN encoder feeding a ~0.5M-param head, so the '
                         'representation itself has never been adapted to top-down grayscale '
                         'mouse video -- far from DINOv2\'s pretraining distribution. This is the '
                         'largest remaining capacity lever, and also the riskiest: the model '
                         'already overfits ~4.9k bouts, so the encoder gets its own much lower LR '
                         '(--encoder-lr) with layer-wise decay rather than the head\'s.')
    p.add_argument('--ft-mode', choices=['full', 'bitfit'], default='full',
                    help="which parameters inside the unfrozen blocks actually train. 'full' is "
                         'every weight (14,180,352 at --unfreeze-blocks 2). \'bitfit\' trains only '
                         'the LayerNorm affine parameters, the LayerScale gains and every bias '
                         '(24,576, a 577x cut) -- the parameters that can RESCALE and RESHIFT an '
                         'existing feature but can never form a new one, because none of them '
                         'touches a weight matrix. That makes the two modes a test of WHY '
                         'fine-tuning helped: if the gain was really new relational computation, '
                         "bitfit cannot reproduce it; if it was recalibrating DINOv2's statistics "
                         'for top-down grayscale rodent video, bitfit is sufficient and the '
                         'remaining 14.18M params were never the point.')
    p.add_argument('--grad-checkpoint', action='store_true',
                    help='recompute the unfrozen blocks\' activations in backward instead of '
                         'storing them (see checkpoint_blocks). Costs roughly one extra encoder '
                         'forward per step; buys back the ~2 GiB that makes BitFit-6 at batch 64 '
                         'and 448 px die on a 44 GiB L40S but fit on an 80 GiB A100. Default OFF, '
                         'so every run predating this flag is bit-for-bit unaffected. No effect '
                         'when --unfreeze-blocks is 0: a frozen encoder runs under no_grad and '
                         'stores no activations to begin with.')
    p.add_argument('--encoder-lr', type=float, default=1e-5,
                    help='LR for the outermost unfrozen block; deeper blocks are scaled down by '
                         '--layerwise-decay per block. NOTE for --ft-mode bitfit: 1e-5 was chosen '
                         'for 14.2M densely-coupled weights, and bias-only tuning is normally run '
                         'orders of magnitude higher, so a bitfit null at 1e-5 alone would be '
                         'uninterpretable -- it must be probed at a larger value too.')
    p.add_argument('--layerwise-decay', type=float, default=0.65,
                    help='held at 0.65 for every fine-tuning run in this repo to date, i.e. '
                         'inherited rather than tuned -- the decay schedule itself is untested.')
    p.add_argument('--seed', type=int, default=None,
                    help='seeds model init, dropout, negative resampling and the augmentation '
                         'draw. Defaults to gsf.SEED, reproducing every prior run. Until now '
                         'torch was never seeded on this path at all, so head init and dropout '
                         'varied run to run while the data order did not -- meaning "same config, '
                         'different result" was possible but a deliberate seed replicate was not. '
                         'Vary this (not the config) to separate a real effect from run-to-run '
                         'noise, which matters because the metric that ranks these models '
                         '(per-observation Pearson r at n=24) has SE ~0.2.')
    p.add_argument('--init-encoder', type=str, default=None,
                    help='overlay a saved encoder checkpoint onto the pretrained weights before '
                         'training, applied with strict=False in the same way every scoring '
                         'script already applies best_encoder.pt. Added for the ssl_dapt arm, '
                         'whose Stage A adapts the encoder on UNLABELLED frames and whose Stage B '
                         'must then train the head on top of it with --unfreeze-blocks 0 -- i.e. '
                         'this changes where the encoder STARTS, not whether it trains. Combining '
                         'it with --unfreeze-blocks > 0 is legal but means fine-tuning an already '
                         'adapted encoder, which is a different experiment.')
    p.add_argument('--wandb', action='store_true', help='log the run to Weights & Biases')
    p.add_argument('--wandb-project', type=str, default='mice-behavior-frame')
    p.add_argument('--tag', type=str, default='online_aug')
    p.add_argument('--jpeg-cache-file', type=str, default=None,
                    help='persist/reuse the JPEG-bytes cache at this path (no extension). The read '
                         'phase is 444k RANDOM small-file NFS opens taking ~33 min at ~226 f/s, '
                         'during which the GPU sits at 0%% utilisation (measured). Written once as '
                         'ONE ~17 GiB file, it reloads as a single SEQUENTIAL read (or is memory-'
                         'mapped, so not read at all up front) in well under 2 min. Only the JPEG '
                         'representation is small enough for this -- the token cache is 163 GiB at '
                         '224px and 650 GiB at 448px. NOTE: shared storage was at 95%% capacity '
                         'when this was added, so this is opt-in rather than default.')
    p.add_argument('--smoke', action='store_true',
                    help='tiny end-to-end check: caps train AND val so the (NFS-bound) read phase '
                         'takes ~1 min instead of ~30, surfacing pipeline bugs before a real run')
    args = p.parse_args()
    if args.seed is None:
        args.seed = gsf.SEED
    torch.manual_seed(args.seed)
    if args.smoke:
        args.max_train_frames, args.n_epochs, args.batch_size = 4_000, 2, 32
        args.val_monitor_size, args.tag = 600, 'online_aug_smoke'

    if args.pixel_source and args.pixel_source >= args.input_size:
        p.error(f'--pixel-source {args.pixel_source} >= --input-size {args.input_size} is a no-op '
                '(or an upsample); it only makes sense strictly below input_size.')
    _SampleDataset.pixel_source = args.pixel_source

    n_patches = (args.input_size // PATCH_SIZE) ** 2
    # The old 'patchgrid256_dinov2_' prefix is gone: every run here is a 256-dim patch grid over
    # DINOv2, so it distinguished nothing and just pushed the informative part of the name out of
    # sight. Pass a --tag matching the scheme in rename_runs.py -- res<input>_k<context>_<encoder>
    # _<augment> -- or run that script afterwards to fold this run into it.
    OUT = gsf.FRAME_DIR / args.tag
    OUT.mkdir(parents=True, exist_ok=True)
    best_cfg = get_head_cfg()
    if args.derm_floor is None:
        args.derm_floor = 1e-4 if args.derm_prevalence == 'population' else 0.02
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'context_k={args.context_k} (T={2*args.context_k+1}, stride={args.stride}, '
          f'reach +-{args.context_k*args.stride} frames)  input_size={args.input_size} '
          f'({n_patches} patches)  augment={args.augment}  neg_ratio={args.neg_ratio}'
          f'  motion={args.use_motion}', flush=True)

    pair_labels = gsf.build_pair_labels(gsf.DATA_DIR, gsf.DATASET_DIR, overwrite=False)
    ann_csv = gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    o2p = gsf.load_obs_to_pool_map(gsf.DATA_DIR)
    all_obs = pd.read_parquet(pair_labels)['observation_id'].unique().tolist()
    pools = sorted({o2p[o] for o in all_obs})
    if args.val_pools:
        # Explicit fold for CROSS-FITTING. PPI's rectifier needs f on the labelled pools to be
        # out-of-fold: an in-sample f is shrunk toward the labels, which understates the
        # correction and produces an interval that is too narrow -- the exact failure PPI exists
        # to prevent. Three folds of 8 tile all 24 annotated pools, each held out exactly once.
        val_pools = set(args.val_pools.split(','))
        missing = val_pools - set(pools)
        if missing:
            raise SystemExit(f'--val-pools not in the annotated set: {sorted(missing)}')
    else:
        val_pools = get_fixed_val_pools(pools)
    print(f'  val pools ({len(val_pools)}): {sorted(val_pools)}', flush=True)
    train_obs = [o for o in all_obs if o2p[o] not in val_pools]
    val_obs = [o for o in all_obs if o2p[o] in val_pools]
    if args.n_train_pools:
        # Learning curve over LABELLED POOLS -- the axis that actually costs money, since a pool
        # is 6 observations one annotator had to sit through. Subsample pools, never observations:
        # dropping observations would keep all 20 cages in the training set and so would measure
        # something else entirely (less data per animal, not fewer animals). Sorted before the
        # seeded draw so the subset is reproducible, and nested across sizes at a fixed seed so
        # the n=5 pools are a subset of the n=10 pools -- otherwise each point on the curve moves
        # for two reasons at once.
        tp = sorted({o2p[o] for o in train_obs})
        if args.n_train_pools > len(tp):
            raise SystemExit(f'--n-train-pools {args.n_train_pools} > {len(tp)} available')
        keep = set(np.random.default_rng(args.seed).permutation(tp)[:args.n_train_pools])
        train_obs = [o for o in train_obs if o2p[o] in keep]
        print(f'  learning-curve point: {args.n_train_pools}/{len(tp)} train pools '
              f'({len(train_obs)} obs) -> {sorted(keep)}', flush=True)
    if args.train_odour:
        # Both sides keep the SAME exposure. train = that odour on the training pools; monitor =
        # that odour on the held-out pools, so early stopping is honest and never sees the test
        # session. The test session itself is not scored here -- run predict_dense.py afterwards,
        # which dumps every v1 observation including the other odour's.
        o2o = dict(zip(*[pd.read_csv(gsf.DATA_DIR / 'mice' / 'v1' / 'experiment.csv')[c]
                         for c in ('observation_id', 'odor')]))
        n0, v0 = len(train_obs), len(val_obs)
        train_obs = [o for o in train_obs if o2o.get(o) == args.train_odour]
        val_obs = [o for o in val_obs if o2o.get(o) == args.train_odour]
        if not train_obs or not val_obs:
            raise SystemExit(f'--train-odour {args.train_odour} left '
                             f'{len(train_obs)} train / {len(val_obs)} monitor observations')
        print(f'  --train-odour {args.train_odour}: train {n0} -> {len(train_obs)} obs, '
              f'monitor {v0} -> {len(val_obs)} obs. The other exposure is the TEST session and is '
              f'not scored here; use predict_dense.py.', flush=True)
    if args.smoke:
        # val is 144k frames and dominates the read phase, so cap it too or "smoke" isn't smoke
        train_obs, val_obs = train_obs[:2], val_obs[:1]
    print(f'train {len(train_obs)} obs / val {len(val_obs)} obs (pools {sorted(val_pools)})', flush=True)

    tm = FrameBatchData(str(ann_csv), str(pair_labels), train_obs, args.context_k, 1,
                        dummy_loader(1, 1), n_patches=1, stride=args.stride,
                        max_frames=args.max_train_frames, seed=args.seed)
    vm = FrameBatchData(str(ann_csv), str(pair_labels), val_obs, args.context_k, 1,
                        dummy_loader(1, 1), n_patches=1, stride=args.stride)
    del tm.flat, vm.flat

    # ---- environments for vREx -------------------------------------------------------------
    # Environment = a group across which the LABELLING CONVENTION, not the biology, is expected
    # to shift. `annotator` is the motivated default: 6 people annotated v1, none of the 432
    # observations was double-annotated, and within a single line x genotype cell their rates
    # differ by up to 3.7x. ERM fits a blend of those conventions; vREx keeps only what predicts
    # equally well for all of them -- which is exactly what has to survive the move to v2, where
    # there is no annotator at all. `pool` is the alternative (cage/cohort/date) and `line` the
    # coarsest. None = plain ERM.
    train_envs, env_names = None, None
    if args.env_key != 'none':
        a3 = pd.read_csv(ann_csv, usecols=['observation_id', 'frame_idx'])
        a3 = a3[a3.observation_id.isin(set(train_obs))].reset_index()
        st_ = {o: int(g['index'].values[0]) for o, g in a3.groupby('observation_id', sort=False)}
        starts = np.array(sorted(st_.values()))
        names = np.array([k for k, _ in sorted(st_.items(), key=lambda x: x[1])])
        obs_of_sample = names[np.searchsorted(starts, tm.gi, side='right') - 1]
        exp = pd.read_csv(gsf.DATA_DIR / 'mice' / 'v1' / 'experiment.csv')
        if args.env_key == 'condition':
            exp['condition'] = exp['phase'].astype(str) + '_' + exp['odor'].astype(str)
        key = {'condition': 'condition', 'phase': 'phase', 'annotator': 'annotator',
               'pool': 'pool', 'line': 'line'}[args.env_key]
        o2e = dict(zip(exp['observation_id'], exp[key].astype(str)))
        raw = np.array([o2e.get(o, 'NA') for o in obs_of_sample])
        env_names, train_envs = np.unique(raw, return_inverse=True)
        train_envs = train_envs.astype(np.int64)
        cnt = np.bincount(train_envs, minlength=len(env_names))
        print(f'  env_key={args.env_key}: {len(env_names)} environments over {len(train_envs):,} '
              f'train anchors -> ' + ', '.join(f'{n}:{c:,}' for n, c in zip(env_names, cnt)),
              flush=True)
        if len(env_names) < 2:
            raise SystemExit(f'--env-key {args.env_key} yields <2 environments; nothing to be '
                             'invariant across.')
    if args.derm and train_envs is None:
        raise SystemExit('--derm needs environments to deconfound against; pass --env-key '
                         '(phase is the motivated default for the phase-transition estimand).')

    # Var(Y|E) for DERM's target mass, off the ANNOTATIONS rather than off `tm`: tm is bounded by
    # --max-train-frames and keeps positives preferentially, so its prevalence is ~18% against a
    # true 0.4-1.4% and the confound is mostly gone from it already.
    derm_var_pop = None
    if args.derm and args.derm_prevalence == 'population':
        if train_envs is None:
            raise SystemExit('--derm-prevalence population needs --env-key')
        derm_var_pop = population_var(ann_csv, train_obs, o2e, env_names)
        if not np.isfinite(derm_var_pop).all():
            raise SystemExit(f'population Var(Y|E) has holes: {derm_var_pop}')
        print('  population Var(Y|E): ' + '  '.join(
            f'{n} nt {derm_var_pop[i, 0]:.5f} nn {derm_var_pop[i, 1]:.5f}'
            for i, n in enumerate(env_names)), flush=True)
        # REFUSE, do not warn. A floor that clips every environment makes the target mass uniform
        # and the whole objective an exact no-op, and a warning after submission is a wasted run --
        # this had already cost two launches. Checked here, before anything is trained.
        _vf = args.derm_floor * (1.0 - args.derm_floor)
        if (derm_var_pop < _vf).all():
            raise SystemExit(
                f'--derm-floor {args.derm_floor:g} gives a variance floor of {_vf:.2e}, which is '
                f'above EVERY population Var(Y|E) here (max {derm_var_pop.max():.5f}). That clips '
                'all environments to one value, makes the target mass uniform and turns DERM into '
                'an exact no-op. Pass a smaller --derm-floor.')

    pos_idx = np.where(tm.labels.sum(1) > 0)[0]
    neg_idx = np.where(tm.labels.sum(1) == 0)[0]
    n_neg = min(len(neg_idx), args.neg_ratio * max(len(pos_idx), 1))
    saturated = n_neg >= len(neg_idx)
    print(f'{args.neg_ratio}:1 sampling: {len(pos_idx):,} pos, {len(neg_idx):,} neg available -> '
          f'{"SATURATED" if saturated else f"fresh {n_neg:,} each epoch"}', flush=True)

    # deliberately gsf.SEED, not args.seed: the monitor subset is evaluation infrastructure, and
    # early stopping has to score every arm of a sweep against the identical val frames or the
    # arms are not comparable. --seed varies the training run, never what it is measured on.
    v_rng = np.random.default_rng(gsf.SEED)
    val_keep = np.sort(v_rng.choice(len(vm), size=min(len(vm), args.val_monitor_size), replace=False))
    vp = vm.labels[val_keep]
    print(f'val monitor {len(val_keep):,} (prevalence-preserving: nt {100*vp[:,0].mean():.2f}% '
          f'nn {100*vp[:,1].mean():.2f}%)', flush=True)

    def needed(meta, si):
        a = meta.gi[si][:, None] + meta.offsets_grid[None, :]
        return np.unique(a[~meta.pad_mask[si]])

    # Frames the run COULD touch. No longer pre-read -- used only to size the save manifest
    # and to report what fraction the lazy path actually avoided.
    all_needed = np.unique(np.concatenate([
        needed(tm, np.concatenate([pos_idx, neg_idx])),
        needed(vm, np.arange(len(vm)))]))
    val_full_frames = needed(vm, np.arange(len(vm)))

    ann = pd.read_csv(ann_csv, usecols=['frame_path', 'frame_idx'])
    frame_paths = ann.frame_path.values
    frame_idx_all = ann.frame_idx.values      # global frame index -> position within its video
    cache_bin = Path(f'{args.jpeg_cache_file}.bin') if args.jpeg_cache_file else None
    cache_idx = Path(f'{args.jpeg_cache_file}.npz') if args.jpeg_cache_file else None

    jpeg_cache = {}
    if cache_bin and cache_bin.exists() and cache_idx.exists():
        m = np.load(cache_idx)
        t0 = time.time()
        blob = np.memmap(cache_bin, dtype=np.uint8, mode='r')
        offs, keys = m['offsets'], m['all_needed']
        jpeg_cache = {int(k): blob[offs[i]:offs[i+1]] for i, k in enumerate(keys)}
        print(f'JPEG cache REUSED from {cache_bin} ({int(offs[-1])/1024**3:.1f} GiB, memory-mapped, '
              f'{len(jpeg_cache):,} frames) in {time.time()-t0:.1f}s', flush=True)

    def ensure_cached(frame_idx, what):
        """Read only the frames we do not already hold. Lazy by design: the upfront read of
        every candidate frame blocked training for ~33 min at 0% GPU, and ~32% of it was
        full-val frames that are not touched until the final evaluation."""
        missing = np.array(sorted(set(int(g) for g in frame_idx) - set(jpeg_cache)), dtype=np.int64)
        if not len(missing):
            return 0.0
        t0 = time.time()
        dl = DataLoader(_BytesReader(frame_paths[missing]), batch_size=None,
                        num_workers=args.read_workers, prefetch_factor=6, collate_fn=lambda x: x)
        for i, buf in dl:
            jpeg_cache[int(missing[i])] = buf
        dt = time.time() - t0
        print(f'  [{what}] read {len(missing):,} new frames in {dt/60:.1f} min '
              f'({len(missing)/max(dt,1e-9):.0f} f/s); cache now {len(jpeg_cache):,}', flush=True)
        return dt

    encoder = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    encoder.requires_grad_(False)
    if args.init_encoder:
        # strict=False: these checkpoints hold only the tensors that trained, so the rest of the
        # encoder stays at the hub weights. An empty intersection means the wrong file was passed
        # and every metric below would silently be a stock-encoder number, so it is fatal.
        sd = torch.load(args.init_encoder, map_location=dev, weights_only=True)
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        if unexpected:
            raise SystemExit(f'--init-encoder has keys the encoder does not: {unexpected[:5]}')
        overlaid = len(sd) - len(unexpected)
        if not overlaid:
            raise SystemExit(f'--init-encoder {args.init_encoder} shares no keys with the encoder')
        print(f'init encoder: overlaid {overlaid}/{len(sd)} tensors from {args.init_encoder}',
              flush=True)
    # best_cfg comes from a search run on a 4x4 (16-token) coarse patch grid with CACHED tokens,
    # the old val split and neg_ratio=15 -- every one of those conditions has since changed, so
    # each field is overridable rather than inherited on faith.
    lr = args.lr if args.lr is not None else best_cfg['lr']
    wd = args.weight_decay if args.weight_decay is not None else best_cfg['weight_decay']
    dropout = args.dropout if args.dropout is not None else best_cfg['dropout']
    model = MouseFrameClassifier(
        emb_dim=EMB_DIM, n_heads=best_cfg['n_heads'], hidden_dim=best_cfg['hidden_dim'],
        use_patch_grid=True, dropout=dropout, use_motion=args.use_motion,
        cross_attn_dim=args.cross_attn_dim or None, patch_pool_dim=args.patch_pool_dim or None,
        patch_selfattn_dim=args.patch_selfattn_dim or None,
        n_pool_queries=args.pool_queries, pool_grid=args.pool_grid).to(dev)
    print(f'classifier params: {sum(p.numel() for p in model.parameters()):,} '
          f'(DINOv2 frozen, {sum(p.numel() for p in encoder.parameters())/1e6:.0f}M)', flush=True)
    print(f'head: patch_selfattn={args.patch_selfattn_dim or "off"}  '
          f'pool_queries={args.pool_queries}  patch_pool_dim={args.patch_pool_dim}  '
          f'cross_attn_dim={args.cross_attn_dim}  pool_grid={args.pool_grid or "off"}', flush=True)
    print(f'optimizer={args.optimizer} lr={lr:g} weight_decay={wd:g} dropout={dropout:g} '
          f'warmup={args.warmup_epochs} decay_epochs={args.lr_decay_epochs}', flush=True)
    opt_cls = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    groups = [{'params': list(model.parameters()), 'lr': lr}]
    def is_bitfit(name: str) -> bool:
        """Parameters that can only rescale or reshift a feature the frozen weights already
        compute: every bias, both LayerNorm gains, and the LayerScale residual gains. No weight
        MATRIX qualifies, so nothing here can build a new function of two inputs."""
        return (name.endswith('.bias') or name.endswith('lambda1')
                or ('norm' in name and name.endswith('.weight')))

    if args.unfreeze_blocks > 0:
        blocks = encoder.encoder.layer            # DINOv2-base: 12 blocks
        n = min(args.unfreeze_blocks, len(blocks))
        for depth, blk in enumerate(reversed(blocks[-n:])):
            if args.ft_mode == 'bitfit':
                ps = [p for nm, p in blk.named_parameters() if is_bitfit(nm)]
                for prm in ps:
                    prm.requires_grad_(True)
            else:
                blk.requires_grad_(True)
                ps = list(blk.parameters())
            # depth 0 = outermost (closest to the head) and most task-specific, so it moves
            # fastest; deeper blocks hold the general features we do NOT want to disturb.
            blk_lr = args.encoder_lr * (args.layerwise_decay ** depth)
            groups.append({'params': ps, 'lr': blk_lr})
        # LayerNorm after the last block feeds the tokens we actually consume (and is bitfit-legal
        # in its entirety, so this line is identical in both modes)
        encoder.layernorm.requires_grad_(True)
        groups.append({'params': list(encoder.layernorm.parameters()), 'lr': args.encoder_lr})
        if args.grad_checkpoint:
            # ONLY the unfrozen blocks. The frozen ones run inside torch.set_grad_enabled(False)
            # for the head's sake anyway, so they already store nothing -- checkpointing them
            # would buy no memory and pay a second forward for it.
            checkpoint_blocks(blocks[-n:])
            print(f'grad checkpointing ON for the {n} unfrozen blocks (use_reentrant=False)',
                  flush=True)
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f'fine-tuning last {n}/{len(blocks)} encoder blocks [{args.ft_mode}]: '
              f'{trainable/1e6:.3f}M params '
              f'({100*trainable/sum(p.numel() for p in encoder.parameters()):.2f}% of the encoder), '
              f'lr {args.encoder_lr:g} decaying x{args.layerwise_decay} per block inward',
              flush=True)
    elif args.grad_checkpoint:
        raise SystemExit('--grad-checkpoint with --unfreeze-blocks 0 is a no-op: a frozen '
                         'encoder already runs under no_grad. Refusing, so a launcher that '
                         'meant to fine-tune does not quietly train a frozen encoder instead.')
    opt = opt_cls(groups, lr=lr, weight_decay=wd)
    finetune = args.unfreeze_blocks > 0
    eta = 0.01

    def lr_factor(e):
        # warmup_epochs=0 reduces this EXACTLY to the previous cosine-to-floor schedule.
        if e < args.warmup_epochs:
            return (e + 1) / (args.warmup_epochs + 1)
        prog = (e - args.warmup_epochs) / max(args.lr_decay_epochs - args.warmup_epochs, 1)
        return eta if prog >= 1 else eta + 0.5 * (1 - eta) * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    run = None
    if args.wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=args.tag, config=vars(args) | {
                'n_patches': n_patches, 'lr': lr, 'weight_decay': wd, 'dropout': dropout,
                'n_heads': best_cfg['n_heads'], 'hidden_dim': best_cfg['hidden_dim'],
                'n_params': sum(p.numel() for p in model.parameters())})
            print(f'wandb: {run.url}', flush=True)
        except Exception as e:   # never let telemetry kill a multi-hour training run
            print(f'wandb disabled ({e.__class__.__name__}: {e})', flush=True)
    crit = nn.BCEWithLogitsLoss()
    crit_ns = nn.BCEWithLogitsLoss(reduction='none')   # per-sample, for the vREx grouping
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')

    def scaled(lo, hi):
        return (1 - (1 - lo) * args.photo_strength, 1 + (hi - 1) * args.photo_strength)

    def make_loader(meta, order, augment, seed, envs=None):
        batches = [order[i:i+args.batch_size] for i in range(0, len(order), args.batch_size)]
        return DataLoader(_SampleDataset(meta, batches, jpeg_cache,
                                         args.input_size, augment, seed,
                                         scaled(0.80, 1.25), scaled(0.80, 1.25), scaled(0.83, 1.20),
                                         envs=envs),
                          batch_size=None, num_workers=args.decode_workers,
                          pin_memory=(dev.type == 'cuda'), prefetch_factor=4)

    @torch.no_grad()
    def evaluate(order):
        model.eval()
        P, L = [], []
        for imgs, offs, lbl, mask, _env in make_loader(vm, order, 'none', 0):
            imgs = imgs.to(dev, non_blocking=True)
            B, T = imgs.shape[:2]
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                tok = encoder(pixel_values=imgs.view(B*T, *imgs.shape[2:])).last_hidden_state[:, 1:]
                logits = model(tok.view(B, T, n_patches, EMB_DIM),
                               offsets=offs.to(dev), key_padding_mask=mask.to(dev))
            P.append(torch.sigmoid(logits).float().cpu()); L.append(lbl)
        return torch.cat(P).numpy(), torch.cat(L).numpy()

    rng = np.random.default_rng(args.seed)
    best, since, hist = -1.0, 0, []
    grad_checked = False              # the encoder-gradient tripwire fires once, at step 1
    for ep in range(1, args.n_epochs + 1):
        model.train()
        order = (np.concatenate([pos_idx, neg_idx]) if saturated else
                 np.concatenate([pos_idx, rng.choice(neg_idx, size=n_neg, replace=False)]))
        rng.shuffle(order)
        read_s = ensure_cached(needed(tm, order), f'epoch {ep} train')
        if ep == 1:
            read_s += ensure_cached(needed(vm, val_keep), 'val monitor')
        tot, seen, t0 = 0.0, 0, time.time()
        step_i, t_step = 0, None
        # vREx penalty is annealed in: applying it from step 0 pins the model at a degenerate
        # solution where every environment is equally badly predicted, which is trivially
        # invariant. Standard practice for IRM/vREx and the reason the ERM phase exists.
        beta_ep = args.vrex_beta if (train_envs is not None and ep > args.vrex_warmup_epochs) else 0.0
        ep_pen, ep_pen_n = 0.0, 0
        # Rebuilt every epoch because the negatives are RESAMPLED every epoch: the weights
        # must describe the distribution the loss is actually averaged over this epoch, not a
        # different one computed once at startup.
        derm_tab, derm_raw_mean = (None, None)
        if args.derm:
            _tab, derm_raw_mean = derm_table(tm.labels, train_envs, order, len(env_names),
                                             args.derm_floor, var_pop=derm_var_pop)
            derm_tab = torch.from_numpy(_tab).to(dev)
            if ep == 1:
                # Print the MASS, not the per-sample weight. A cell the sampler draws less often
                # must carry more weight per sample to reach the same mass, so per-sample spreads
                # are not comparable across arms -- while the mass is exactly what DERM specifies.
                # A DERM arm that is silently a no-op is the failure worth logging against.
                mass = derm_mass(_tab, tm.labels, train_envs, order, len(env_names))
                for li, ln in enumerate(('nt', 'nn')):
                    # `emass`, NOT `tot`: `tot` is the epoch's loss accumulator, initialised just
                    # above this block, and shadowing it with a per-environment vector made
                    # `tot += loss.item() * B` accumulate into a 3-array. Training was unaffected
                    # (the gradient uses `loss`) but the logged loss was nonsense and the print
                    # raised. Cost two launches; the [types] line below is what found it.
                    emass = mass[:, li, :].sum(1)
                    tgt = derm_var_pop[:, li] if derm_var_pop is not None else None
                    print(f'  DERM[{ln}] Var(Y|E) from {args.derm_prevalence}'
                          + ('' if tgt is None else '  target ratio '
                             f'{tgt.max() / max(tgt.min(), 1e-12):.2f}x'), flush=True)
                    print(f'      mass  ' + '  '.join(
                        f'{n}: y1 {mass[k, li, 1]:.4f} y0 {mass[k, li, 0]:.4f}'
                        for k, n in enumerate(env_names)), flush=True)
                    print(f'      env mass share ' + ' '.join(
                        f'{n}:{v:.3f}' for n, v in zip(env_names, emass / max(emass.sum(), 1e-12)))
                        + f'   ratio {emass.max() / max(emass.min(), 1e-12):.2f}x', flush=True)
        for imgs, offs, lbl, mask, env in make_loader(tm, order, args.augment,
                                                      args.seed * 1000 + ep, envs=train_envs):
            imgs, lbl = imgs.to(dev, non_blocking=True), lbl.to(dev, non_blocking=True)
            env = env.to(dev, non_blocking=True)
            B, T = imgs.shape[:2]
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                # frozen encoder: no graph, no backward through DINOv2. When fine-tuning we
                # must keep the graph, which is what makes the step ~2x more expensive.
                with torch.set_grad_enabled(finetune):
                    tok = encoder(pixel_values=imgs.view(B*T, *imgs.shape[2:])).last_hidden_state[:, 1:]
                tok = tok.view(B, T, n_patches, EMB_DIM)
                logits = model(tok if finetune else tok.detach(),
                               offsets=offs.to(dev), key_padding_mask=mask.to(dev))
                # One per-label loss matrix feeds every variant, so ERM stays bit-identical
                # to `crit(logits, lbl)` (both are the mean over B x L) and the only thing an
                # arm changes is the weighting or the grouping applied on top of it.
                per_lbl = crit_ns(logits, lbl)                        # (B, L)
                if derm_tab is not None:
                    # w[b, l] = table[env[b], l, int(lbl[b, l])] -- DERM reweights the SAMPLING
                    # DISTRIBUTION, so it multiplies the loss rather than adding a penalty.
                    yi = (lbl > 0.5).long()
                    per_lbl = per_lbl * derm_tab[env].gather(2, yi.unsqueeze(-1)).squeeze(-1)
                per_sample = per_lbl.mean(dim=1)
                if beta_ep:
                    # vREx (Krueger et al. 2021): minimise mean risk + beta * VARIANCE of risk
                    # across environments. Penalising the spread -- not the predictions -- is what
                    # keeps a genuine treatment effect intact: forcing the PREDICTIONS to be
                    # invariant across phase would suppress the very phase effect we estimate.
                    risks = [per_sample[env == e].mean()
                             for e in torch.unique(env)
                             if int((env == e).sum()) >= args.vrex_min_env]
                    if len(risks) >= 2:
                        rs = torch.stack(risks)
                        pen = rs.var(unbiased=False)
                        loss = rs.mean() + beta_ep * pen
                        ep_pen += float(pen.detach()); ep_pen_n += 1
                    else:
                        loss = per_sample.mean()
                else:
                    loss = per_sample.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if finetune and not grad_checked:
                # THE tripwire for gradient checkpointing. torch.utils.checkpoint with the
                # legacy use_reentrant=True produces NO gradients at all for a block whose
                # inputs do not require grad -- which is every checkpointed block here, since
                # the frozen blocks below them emit a non-grad hidden state. It fails silently:
                # the head still learns, the loss still falls, and the encoder never moves. So
                # read the encoder's gradients once, out loud, before trusting seven hours of
                # compute. Read-only: no clipping, no scaling, nothing written back.
                grad_checked = True
                enc_p = [q for q in encoder.parameters() if q.requires_grad]
                withg = [q for q in enc_p if q.grad is not None]
                gn = float(torch.sqrt(sum((q.grad.float() ** 2).sum() for q in withg))
                           ) if withg else 0.0
                print(f'  encoder grad: {len(withg)}/{len(enc_p)} tensors have a gradient, '
                      f'norm {gn:.6g}', flush=True)
                if not withg:
                    raise SystemExit(
                        'no encoder gradient at the first optimiser step. The encoder is '
                        'unfrozen, so this cannot be right -- the usual cause is a checkpoint '
                        'wrapper built with use_reentrant=True. Aborting now rather than '
                        'training a null fine-tune to completion.')
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g['params'] if p.grad is not None], 0.5)
            scaler.step(opt); scaler.update()
            tot += loss.item() * B; seen += B
            # Peak GPU memory is what decides which cards a config can run on at all, and until
            # now the only way to learn it was to read an OOM traceback: BitFit-6 at batch 64 and
            # 448 px peaks at 42.5 GiB and dies on a 44 GiB L40S, while fitting an 80 GiB A100
            # with room to spare, and nobody could see that coming. Timed over steps 5..30 so
            # the first steps' cuDNN autotuning does not distort the rate.
            step_i += 1
            if ep == 1 and dev.type == 'cuda' and step_i in (5, 30):
                torch.cuda.synchronize()
                if step_i == 5:
                    t_step = time.time()
                else:
                    print(f'  [mem] train {(time.time()-t_step)/25:.2f} s/step   peak allocated '
                          f'{torch.cuda.max_memory_allocated()/1024**3:.2f} GiB   peak reserved '
                          f'{torch.cuda.max_memory_reserved()/1024**3:.2f} GiB   of '
                          f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GiB',
                          flush=True)
        sched.step()
        probs, labs = evaluate(val_keep)
        if ep == 1 and dev.type == 'cuda':
            # again AFTER the monitor pass: training is not necessarily where the peak lands, and
            # a run that survives epoch 1 only to OOM in validation has wasted the whole epoch.
            print(f'  [mem] after epoch 1 + val: peak allocated '
                  f'{torch.cuda.max_memory_allocated()/1024**3:.2f} GiB   peak reserved '
                  f'{torch.cuda.max_memory_reserved()/1024**3:.2f} GiB', flush=True)
        per_ap = [float(average_precision_score(labs[:, i], probs[:, i])) for i in (0, 1)]
        per_auc = [float(roc_auc_score(labs[:, i], probs[:, i])) for i in (0, 1)]
        ap = float(np.mean(per_ap))
        # val BCE on the PREVALENCE-PRESERVING monitor set, whereas train loss is on balanced
        # 1:1 data -- the two are not comparable in absolute terms, but the val trend is the
        # overfitting signal that a plateauing AP alone does not distinguish from convergence.
        p = np.clip(probs, 1e-7, 1 - 1e-7)
        val_loss = float(-(labs * np.log(p) + (1 - labs) * np.log(1 - p)).mean())
        row = {'epoch': ep, 'train_loss': tot/seen, 'monitor_ap': ap, 'val_loss': val_loss,
               'ap_nt': per_ap[0], 'ap_nn': per_ap[1],
               'auc_nt': per_auc[0], 'auc_nn': per_auc[1],
               'lr': float(opt.param_groups[0]['lr'])}
        hist.append(row)
        if run is not None:
            try:
                run.log(row, step=ep)
            except Exception:
                pass
        # Every number is coerced through float() before formatting. A LENGTH-1 numpy array
        # raises TypeError from __format__ (numpy scalars and 0-d arrays do not), and one of these
        # was arriving that way and killing the run AFTER a full epoch of compute -- an epoch of
        # GPU thrown away by a log line. On epoch 1 anything that is not already a plain float is
        # named, so the next occurrence is diagnosed from the log instead of by bisection.
        _f = {'loss': tot / seen, 'val_loss': val_loss, 'monitor_ap': ap,
              'ap_nt': per_ap[0], 'ap_nn': per_ap[1], 'auc_nt': per_auc[0], 'auc_nn': per_auc[1],
              'lr': opt.param_groups[0]['lr'], 'read_s': read_s,
              'derm_w_raw': derm_raw_mean if derm_tab is not None else 0.0}
        if ep == 1:
            odd = {k: type(v).__name__ + (f'{getattr(v, "shape", "")}' or '')
                   for k, v in _f.items() if not isinstance(v, float)}
            if odd:
                print(f'  [types] not plain float: {odd}', flush=True)
        _f = {k: float(v) for k, v in _f.items()}
        _dw, _rs = _f['derm_w_raw'], _f['read_s']
        print(f'epoch {ep:3d}/{args.n_epochs}  loss={_f["loss"]:.4f}  '
              f'val_loss={_f["val_loss"]:.4f}  monitor_ap={_f["monitor_ap"]:.4f}  '
              f'nt={_f["ap_nt"]:.4f}  nn={_f["ap_nn"]:.4f}  '
              f'auc_nt={_f["auc_nt"]:.4f}  auc_nn={_f["auc_nn"]:.4f}  '
              f'{f"vrex_pen={float(ep_pen)/max(ep_pen_n,1):.3e} (b={float(beta_ep):g})  " if beta_ep else ""}'
              f'{f"derm_w_raw={_dw:.2f}  " if derm_tab is not None else ""}'
              f'lr={_f["lr"]:.2e}  ({time.time()-t0:.1f}s compute'
              f'{f", {_rs:.0f}s new-frame read" if read_s > 1 else ""})', flush=True)
        # --select last: a FIXED EPOCH BUDGET. Keep every epoch (so the last one survives) and
        # never early-stop, which takes the selection rule out of the ERM-vs-DERM contrast
        # entirely. Motivated rather than merely different: selecting on unweighted AP asks for
        # the epoch that best uses the phase prior, which is the thing DERM removes.
        if args.select == 'last' or ap > best:
            best, since = max(ap, best), 0
            torch.save(model.state_dict(), OUT / 'best_model.pt')
            if finetune:
                # The head alone is not a usable checkpoint once the encoder has moved -- but
                # only the unfrozen blocks moved. Saving the whole encoder wrote 346 MB per run
                # of which the frozen blocks are a byte-identical copy of facebook/dinov2-base:
                # at --unfreeze-blocks 2 that is 10 of 12 blocks, ~287 MB of pure duplication,
                # per saved epoch, per run. Store only the tensors that actually train; every
                # loader pairs this with a freshly pretrained encoder and applies it with
                # strict=False, so the frozen remainder comes from the hub as it always did.
                train_keys = {n for n, p in encoder.named_parameters() if p.requires_grad}
                torch.save({k: v for k, v in encoder.state_dict().items() if k in train_keys},
                           OUT / 'best_encoder.pt')
        elif args.select != 'last':
            since += 1
            if since >= args.patience:
                print(f'early stopping (no improvement for {args.patience})', flush=True)
                break

    model.load_state_dict(torch.load(OUT / 'best_model.pt', map_location=dev, weights_only=True))
    if finetune:
        # strict=False: best_encoder.pt holds only the unfrozen tensors (see the save above).
        # `encoder` already carries the pretrained weights, so this overlays the trained ones.
        # Also accepts the pre-2026-08-14 full-encoder files, where every key simply matches.
        encoder.load_state_dict(torch.load(OUT / 'best_encoder.pt', map_location=dev,
                                           weights_only=True), strict=False)
        encoder.eval()
    ensure_cached(val_full_frames, 'full-val (deferred to the end)')
    probs, labs = evaluate(np.arange(len(vm)))
    sample_obs = np.zeros(len(vm), dtype=object)   # observation id per val sample
    a2 = pd.read_csv(ann_csv, usecols=['observation_id', 'frame_idx'])
    a2 = a2[a2.observation_id.isin(set(val_obs))].reset_index()
    b = {o: int(g['index'].values[0]) for o, g in a2.groupby('observation_id', sort=False)}
    st = np.array(sorted(b.values())); nm = np.array([k for k, _ in sorted(b.items(), key=lambda x: x[1])])
    sample_obs = nm[np.searchsorted(st, vm.gi, side='right') - 1]

    print(f'\n{"="*70}\nFULL-VAL\n{"="*70}')
    apr = ap_report(probs, labs, sample_obs, tolerances=(0, 1, 2))
    print(format_ap_report(apr, tolerances=(0, 1, 2)))
    rr = rate_report(probs, labs, sample_obs)
    auc = {n: float(roc_auc_score(labs[:, i], probs[:, i])) for i, n in enumerate(('nt', 'nn'))}
    print(f'\nROC-AUC  nt {auc["nt"]:.4f}  nn {auc["nn"]:.4f}')
    print()
    print(format_rate_report(rr))

    # Persist the full-val predictions BEFORE drawing anything. Every figure below is a view
    # of this array, and until 2026-08-14 the array itself was thrown away with the process --
    # so any later change to the figure code stranded every existing figure, redrawable only
    # by repeating a 6-hour training run. 2.3 MB per run buys unlimited redraws
    # (regen_confusion_figs.py --from-cache) and makes the numbers behind a figure auditable.
    try:
        np.savez_compressed(OUT / 'val_probs.npz', probs=probs, labels=labs, gi=vm.gi,
                            obs=sample_obs.astype(str))
    except Exception as e:
        print(f'  [viz] val_probs.npz not saved ({e.__class__.__name__}: {e})', flush=True)

    # qualitative TP/FP/FN/TN grids for both behaviours. The val frames are already in the
    # JPEG cache from the full-val pass, so this costs decoding only -- no NFS re-read.
    viz_files = {}
    try:
        viz_files = plot_confusion_examples(
            probs, labs, sample_obs, vm.gi, frame_paths, OUT, jpeg_cache=jpeg_cache,
            title_prefix=f'{args.tag}  ')
        # ...and the temporal view of the two error buckets: a single still cannot distinguish
        # contact from approach at 5 fps, and the neighbouring frames carry the annotations
        # that say whether an error is a boundary disagreement or a real miss.
        viz_files |= plot_error_strips(
            probs, labs, sample_obs, vm.gi, frame_paths, OUT, jpeg_cache=jpeg_cache,
            title_prefix=f'{args.tag}  ', frame_idx=frame_idx_all[vm.gi])
    except Exception as e:                # never let a figure lose a finished training run
        print(f'  [viz] skipped ({e.__class__.__name__}: {e})', flush=True)

    if run is not None:
        try:
            import wandb
            run.log({f'confusion/{k}': wandb.Image(str(v)) for k, v in viz_files.items()})
        except Exception:
            pass
        try:
            # the whole suite, not just AP: plain + tolerant AP (bouts are 1-2 frames, so
            # tol1/tol2 separate genuine misses from boundary jitter), ROC-AUC, and the
            # per-observation rate agreement that metrics.py flags as the select-on quantity.
            run.summary.update(
                {f'fullval/ap_{k.replace("/", "_")}': v['ap'] for k, v in apr.items()}
                | {f'fullval/enrich_{k.replace("/", "_")}': v['enrichment']
                   for k, v in apr.items() if 'enrichment' in v}
                | {f'fullval/auc_{k}': v for k, v in auc.items()}
                | {f'fullval/rate_{n}_{k}': v for n, d in rr.items() for k, v in d.items()
                   if isinstance(v, (int, float))})
            run.finish()
        except Exception:
            pass
    json.dump({'cfg': best_cfg, 'context_k': args.context_k, 'input_size': args.input_size,
               'pixel_source': args.pixel_source, 'init_encoder': args.init_encoder,
               'n_train_pools': args.n_train_pools, 'pool_grid': args.pool_grid,
               'val_pools': sorted(val_pools),
               'env_key': args.env_key, 'vrex_beta': args.vrex_beta,
               'derm': bool(args.derm), 'derm_floor': args.derm_floor,
               'derm_prevalence': args.derm_prevalence,
               'vrex_warmup_epochs': args.vrex_warmup_epochs,
               'n_environments': (len(env_names) if env_names is not None else 0),
               'n_patches': n_patches, 'augment': args.augment, 'neg_ratio': args.neg_ratio,
               'use_motion': args.use_motion, 'lr_decay_epochs': args.lr_decay_epochs, 'stride': args.stride,
               'cross_attn_dim': args.cross_attn_dim, 'patch_pool_dim': args.patch_pool_dim,
               'patch_selfattn_dim': args.patch_selfattn_dim, 'pool_queries': args.pool_queries,
               'unfreeze_blocks': args.unfreeze_blocks, 'ft_mode': args.ft_mode,
               'grad_checkpoint': bool(args.grad_checkpoint),
               'encoder_lr': args.encoder_lr, 'layerwise_decay': args.layerwise_decay,
               'n_head_params': sum(p.numel() for p in model.parameters()),
               'n_encoder_trainable': sum(p.numel() for p in encoder.parameters() if p.requires_grad),
               'photo_strength': args.photo_strength, 'optimizer': args.optimizer, 'seed': args.seed,
               'warmup_epochs': args.warmup_epochs, 'lr': lr, 'weight_decay': wd, 'dropout': dropout,
               'n_epochs': args.n_epochs, 'batch_size': args.batch_size,
               'max_train_frames': args.max_train_frames, 'val_pools': sorted(val_pools),
               'train_odour': args.train_odour or None, 'select': args.select,
               'jpeg_cache_gib': sum(len(b) for b in jpeg_cache.values())/1024**3,
               'jpeg_cache_frames': len(jpeg_cache), 'ap_report': apr, 'rate_report': rr,
               'best_ap': apr['macro/tol0']['ap'], 'history': hist},
              open(OUT / 'config.json', 'w'), indent=2)
    if cache_bin and not cache_bin.exists():
        t0 = time.time()
        cache_bin.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(sorted(jpeg_cache), dtype=np.int64)
        offs = np.zeros(len(keys) + 1, dtype=np.int64)
        with open(cache_bin, 'wb') as fh:
            for i, k in enumerate(keys):
                b = jpeg_cache[int(k)]
                fh.write(b.tobytes() if hasattr(b, 'tobytes') else bytes(b))
                offs[i+1] = offs[i] + len(b)
        np.savez(cache_idx, all_needed=keys, offsets=offs)
        print(f'Saved reusable JPEG cache -> {cache_bin} ({offs[-1]/1024**3:.1f} GiB, '
              f'{len(keys):,} frames, {time.time()-t0:.0f}s)', flush=True)
    print(f'\nSaved {OUT}/', flush=True)


if __name__ == '__main__':
    main()
