"""
Build the NES explorer page "Exploratory Causal Inference x Mice / Ants / Frogs": <out>/index.html + <out>/assets/.

The page design is scripts/eci/explorer_template.html (the user's hand-edited version of the
published page); this script fills its `const ALL = /*__DATA__*/null` with the core data of one or more
NES result sets (one per SAE / representation, chosen with the page's Model bar: MODELS gives each SAE
its encoder / SAE type / input; the first --res is the page default), writes the rest as data files the
page fetches on demand (assets/data/) and writes the media they reference.

One command, all steps, incremental (GPU needed for the patch and arena steps):
    sbatch scripts/eci/build_explorer.sh
or directly (on a GPU node):
    python scripts/eci/build_explorer.py            # default: DEFAULT_RES (mice fg448al, ff448al; ants antsfg, antsfull;
                                                    # frogs frogsfg, frogsfull)
    python scripts/eci/build_explorer.py --res results/vision/mice/eci/nes/<sae> [--res ...]
Without --res: DEFAULT_RES (the representations that earned their place); --discover adds every other finished
SAE result set under the domains' NES roots (discover_res; MODELS names it on the model bar, else model_of
derives its values). The SAE is the basename of each --res; its domain (src/eci/domain.py: mice,
ants, frogs) is the NES root the result set lies under, with the analysis set covering its summary.csv analyses
(ants 'pairs': every treatment pair of an experiment). Its full per-frame codes (<domain eci dir>/codes/<sae>/), its training tokens
(tokens_dir in the SAE's metrics.json) and, for fg448 representations, the per-video backgrounds (codes
config.json 'backgrounds') must exist. Output: <first res>/explorer/ (publish the folder). The last step
checks the limits (files, MB, 0 missing / 0 unused assets).

Domains: the domain word of the page title is a selector showing one domain at a time (its SAEs in the
Model bar, its comparisons, videos and charts). VIEW 'keep' restricts a domain's videos (ants: v2 and v3 t=2, 6, 8):
only the analyses whose two arms are kept are shown, and every clip ('all videos' included), histogram, firing
rate, table and video list uses the kept videos only. Mice: stage x genotype design, families A (paired stage
change) and B (het vs wt), gene line / sex subgroups. Ants (VIEW): one family, treated vs control videos of
one experiment, picked with Experiment / Control / Treatment selectors over the analyses on disk (labelled
with the raw treatment numbers, "t=2 vs t=8"), a confound note from each analysis's metadata 'confound'
(CONFOUND_FALLBACK when missing), the recording day on every clip and the top-by-p neurons of every search
(VIEW 'top', so comparisons without NES selections can be browsed). Frogs (VIEW 'arm'): the same treated vs control
page, the arms being the groups (WT control, FoxP1 / En1 mutant; one video per frog, hour 1), the session in place of
the recording day, a domain note under the title (VIEW 'note') and the recording-session checks
(scripts/eci/session_check.py, session_checks) in place of the mice camera-period checks.

Outcome controls: Temporal Aggregation (event rate = bouts/min, average time = per-video mean, latency = seconds to
the first bout within the comparison's common window W, right-censored at W) x Spatial aggregation (max pooling, average pooling, SOMP). The
core outcomes (--outcome / DEFAULT_OUTCOMES) keep their data files; extra_outcomes adds the other poolings found in
their summary.csv, the sibling result sets <res>_<aggregation> (<sae>_mean, <sae>_somp) and the latency runs
(<res>/<P>_latency/, scripts/eci/run_nes_latency.py; per-video values and censoring from _cache/latency*.npz) as
<tag>_x.json bundles (full cohort only; latency: primary window 'common', the per-video values capped by the page
at each analysis's W from result.json, LATENCY_AGGS = max / mean). Latency rows carry the censored videos per arm (side panel, per-video
panel). Mask + motion SAEs carry each neuron's change share (decoder weight on the change half of the input).

Data-driven: everything comes from the result sets (summary.csv files written by the NES runs); only
the full-video window is shown. The first --outcome is the page default. Every neuron of a search (CLIP_KINDS: event
rate, average time, latency) gets clips (clip_candidates / plan_clips; --max-pages caps them, 0 = no cap): found in
a primary search (core or extra, PAGE_PREFIXES 128 / 256, full window), in its top-by-p list, or only in a subgroup
search; ranked by the codes the outcome is built from (codes_max, codes_mean, codes_somp from the <sae>_somp
codes).

Steps (each cached under <res>/_cache/explorer/, incremental):
    data     reads the neuron's codes on every frame (2.59 M) and, per clip length (src/eci/viz.py
             scan_windows / pick_top_windows / pick_least_windows, K = 16, at most one per video):
               frame  top = the K videos' single highest frames (ranked by that frame's activation);
               1 s    windows of 5 consecutive frames (5 fps), 3 s = 15 frames: in each video the window
                      with the highest MEAN activation over its frames, the K best videos (sustained events);
               least  (every length) windows (or single frames) where the neuron is exactly 0 on EVERY
                      frame: one random such window per video, K random videos (fixed seed); when fewer
                      than K videos have one, the K videos' lowest-mean windows (rule 'lowest', labelled
                      "lowest (not silent)" on the page).
             The same rules again among the videos of each contrast the neuron is found in (primary search,
             any prefix, full cohort, or listed top-by-p there; 'Clips from: this comparison' on the page),
             top and least each among all the videos of the contrast together: A (e.g. A_wt_1to2) = that
             genotype's videos of stages a and b; B (B_stage3) = the het and wt videos of stage 3; ants =
             the control and treated videos of the experiment.
             Neurons found only in a subgroup search get 'all' clips only. Plus the activation histogram
             over all frames (40 linear bins, all frames and per genotype|stage), the firing rate and the
             stage x genotype table of per-video mean codes.
             -> picks.json
    patch    per-patch codes of the neuron on every frame of every clip (top and least, SHOW_LENGTHS),
             recomputed through DINOv2 + SAE (src/eci/viz.py make_patch_encoder, GPU; crop224 or fg448
             geometry from the SAE's codes config), checked against the stored pooled codes (mean / max;
             SOMP clips show the neuron's SAE patch codes as heat, unchecked). The heat
             colour scale of a neuron = the 99th percentile of its positive patch codes over all those
             frames (shared by every row and length). -> patch/<p>XXXX.npz
    arena    arena maps of ALL neurons: per patch position, the mean SAE code over the SAE's training
             frames (fg448: 1 frame per second of every video, codes 0 off the foreground, plus how
             often the patch is foreground); the arena background image. -> arena.npz, assets/<tag>_arena_bg.webp
    render   clip pages: one per codes key x neuron x clip source ('all' videos, each contrast) and length
             (SHOW_LENGTHS: frame = 1 frame, 1 s = 5 at 5 fps; the 3 s picks are not rendered) = blocks top,
             top heat, least, least heat (turbo, the neuron's shared scale) of VIEW kr tiles, cols per row (mice: 224 px
             whole frames, 8 per row; ants: the whole frame at 320 px (--ants-tile), 4 per row, or with --ants-clips
             crop 256 px full-resolution cuts around each top clip's activation peak). Each
             page is one cached H.264 segment (<work>/seg/), joined without re-encoding into packs of at
             most --pack-mb MB per SAE and length: page p of a pack = frames [p w, (p + 1) w). A pack of the
             previous build whose pages are all unchanged is kept as it is (same name = same uploaded asset);
             only the other pages are packed anew.
             -> assets/<tag>_<L>_<md5>.mp4, <work>/packs.json
    page     core data (domains, model bar, outcomes, analyses, video list, subgroup sizes and the searches of the
             default SAE's first outcome) inlined into scripts/eci/explorer_template.html -> <out>/index.html
             (< --max-page-kb); the rest fetched by the page when needed (split_data): assets/data/
             <tag>_r_<outcome>.json = searches of one SAE x outcome (full cohort + every subgroup),
             <tag>_<codes>_<k>.json = neuron chunks of ~CHUNK_BYTES (clip lists, histogram, arena map,
             per-video outcome values from the NES caches <res>/_cache/video_summaries_*.npz,
             bout_summaries_max.npz, checked against the tau of summary.csv). The page shows 8 clips per
             row by default (8 / 16 control where 16 are rendered). A clip names its pack (src: assets/<pack>,
             or its asset-store URL from --asset-map) and page; the page seeks a hidden <video> to it.
    check    every referenced media / data file exists; unreferenced files in assets/ and assets/data/ are
             deleted; counts, MB; writes <out>/publish.json (version files, packs with sizes and asset URLs).
             Fails on a missing file, a pack > 15 MB, index.html > --max-page-kb KB (and, with --require-fit,
             version files above --max-version-files / --max-mb).

Publishing (the artifact version holds at most 511 files / 256 MB): the packs can be version files or be
uploaded to the artifact's asset store (the page must declare the 'assets' capability); then rebuild with
--steps page,check --asset-map <json {pack name: asset url}> so the page reads them from their URLs.

Page only (CPU, all caches present): python scripts/eci/build_explorer.py --steps page,check

Representations (src/eci/viz.py representation): 'crop224' SAEs are patch SAEs on the 224 center crop
(16 x 16 patches, heat on pixels 32-480); 'fg448' SAEs (src/eci/foreground.py) see the whole frame at
448 (32 x 32 patches of 16 px) and fire only on foreground patches (others = 0).
Chips: one per single-setting change from the outcome's primary (summary.csv columns; bout_rule 1 =
min-2-frame hysteresis bouts 'min2+hyst'), plus 'size' when <outcome dir>/size_adjusted.csv exists.
Artefact flags (neurons 50, 64, 113) belong to the ep20 SAE only.
Odor-aligned SAEs (codes config 'align' = 'odor', e.g. fg448al): their frames are turned so the odor corner is at
the top right (src/eci/foreground.py align_rot90) before DINOv2 (patch step, src/eci/viz.py PatchEncoderFG) and in
the rendered clips (raw and heat), so clip positions agree with the arena map (aligned backgrounds).
Mice family B (het vs wt): the camera period (odor-corner group, dataset/mice/v1/eci/odor_corner.csv) is unbalanced
by genotype. No video is dropped and the primary analysis is unchanged; each selected family-B neuron carries the
read-only checks of scripts/eci/period_check.py (period_checks; <result set>/period_check/ of the outcome's pooling:
the SAE's own for its primary pooling, <sae>_mean for mean pooling): a robustness chip 'day' (the round test with the
recording day as a covariate still below the round's threshold, same sign) and a warning chip 'camera period' when
the discovery looks due to the period: within-genotype period AUC score >= the family-wise threshold, or >= the
one-latent threshold and the day-adjusted re-test fails.

Outcome spec (default: bout rate (maxpool_bouts/) + mean activation (.) pooled as the run's primary
(selected_neurons.json 'settings'), each when its summary.csv exists; applies to every --res):
    --outcome LABEL=SUBDIR[:key=value,...]   SUBDIR relative to --res. Keys: pooling, outcome_type,
    test (default t), correction (default bonferroni), any extra summary.csv setting column
    (threshold_q, merge_gap, ...; required when it has several values), codes (mean|max, default
    = pooling), unit (tau unit), short (chip label).
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eci.domain import get_domain  # noqa: E402

MICE = get_domain('mice')  # stage labels, analysis ids / order, gene lines and sexes, experiment.csv
ANTS = get_domain('ants')  # experiments, treatments, recording days, analysis ids / order
FROGS = get_domain('frogs')  # groups (WT, FoxP1, En1), sessions, analysis ids / order
DOMS = (MICE, ANTS, FROGS)
TEMPLATE = ROOT / 'scripts/eci/explorer_template.html'
DATASET = ROOT / 'dataset'
NES = 'results/vision/mice/eci/nes'
ANES = 'results/vision/ants/eci/nes'
FNES = 'results/vision/frogs/eci/nes'
# The representations on the page (the ones that earned their place): per domain the foreground-mask SAE and the
# full-frame SAE. DINOv3, background-subtracted and mask + motion SAEs (fg448mot, antsfgmot) are left out on purpose
# (they did not beat these), as is the old mice full-frame SAE ep20 (224 center crop, unaligned): their result sets
# stay on disk (--res brings one back). The mice mask SAE is the odor-aligned fg448al (every video turned so the odor
# corner is top right; it replaced fg448). The mice full-frame SAE is the odor-aligned ff448al (whole frame at 448,
# rule 'all' = every patch, scripts/eci/ff448al_chain.sh; it replaced ep20). Frogs: the frog mask SAE frogsfg
# (foreground rule 'frogs', src/eci/foreground.py) and the full-frame SAE frogsfull (scripts/eci/frogs_chain.sh).
DEFAULT_RES = [f'{NES}/matryoshka_btk_1024_k16_fg448al_s0', f'{NES}/matryoshka_btk_1024_k16_ff448al_s0',
               f'{ANES}/matryoshka_btk_1024_k16_antsfg_s0', f'{ANES}/matryoshka_btk_1024_k16_antsfull_s0',
               f'{FNES}/matryoshka_btk_1024_k16_frogsfg_s0', f'{FNES}/matryoshka_btk_1024_k16_frogsfull_s0']
# without --res: DEFAULT_RES (first = page default); with --discover also every other finished SAE result set
# found under the domains' NES roots (discover_res: <root>/<sae>/summary.csv + SUMMARY.md, full codes merged,
# not an aggregation sibling <sae>_mean / <sae>_somp)
ARTEFACT_EP20 = {64, 50, 113}  # ep20 SAE neuron ids; other SAEs get no flags
# Model bar of the page: SAE result set -> its value on each axis (encoder, SAE type, input), plus a
# tooltip per value. An SAE missing here gets its values from its name and representation (model_of);
# an axis with a single value is shown as a fixed selector. The title's domain selector shows the SAEs of
# one domain at a time.
MODEL_AXES = [('encoder', 'Encoder'), ('sae', 'SAE'), ('input', 'Input')]
MODELS = {
    'matryoshka_btk_1024_k16_fg448_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'mouse mask'},
    'matryoshka_btk_1024_k16_fg448al_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'mouse mask'},
    'matryoshka_btk_1024_k16_ep20_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'full frame'},
    'matryoshka_btk_1024_k16_ff448al_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'full frame'},
    'matryoshka_btk_1024_k16_antsfg_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'ants mask'},
    'matryoshka_btk_1024_k16_antsfull_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'full frame'},
    'matryoshka_btk_1024_k16_fg448mot_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'mask + motion'},
    'matryoshka_btk_1024_k16_antsfgmot_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'mask + motion'},
    'matryoshka_btk_1024_k16_frogsfg_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'frog mask'},
    'matryoshka_btk_1024_k16_frogsfull_s0': {'encoder': 'DINOv2', 'sae': 'Matryoshka', 'input': 'full frame'}}
MODEL_TIPS = {
    ('encoder', 'DINOv2'): 'DINOv2 patch features',
    ('sae', 'Matryoshka'): 'Matryoshka BatchTopK sparse autoencoder, 1024 neurons, k = 16',
    ('input', 'mouse mask'): 'Whole frame at 448 px; SAE trained on the mouse (foreground) patches only',
    ('input', 'full frame'): 'SAE trained on all patches of the frame (animals and background)',
    ('input', 'ants mask'): 'Whole frame at 448 px; SAE trained on the ant (foreground) patches only',
    ('input', 'frog mask'): 'Whole frame at 448 px; SAE trained on the frog (foreground) patches only',
    ('input', 'mask + motion'): 'Foreground patches only, as the mask SAE, but each patch is described by its token '
                                'and its change over the last second ([token_t, token_t - token_(t-5)], 5 frames '
                                'at 5 fps, both halves scaled equally)'}
# per-SAE tooltip of an axis value (replaces MODEL_TIPS for the SAE's own value when that SAE is on the page)
MODEL_SAE_TIPS = {
    ('matryoshka_btk_1024_k16_fg448al_s0', 'input'): 'Whole frame at 448 px; SAE trained on the mouse (foreground) '
                                                     'patches only. Odor-aligned: every video rotated so the odor '
                                                     'corner is top right',
    ('matryoshka_btk_1024_k16_ff448al_s0', 'input'): 'Whole frame at 448 px; SAE trained on all 1024 patches of the '
                                                     'frame (mice and background). Odor-aligned: every video rotated '
                                                     'so the odor corner is top right. Its picks can be setup features '
                                                     '(odor bag, bedding): see the note under the model bar'}
# per-SAE note shown under the model bar while that SAE is selected
MODEL_NOTES = {
    'matryoshka_btk_1024_k16_ff448al_s0': 'Full frame: the strongest stage-change picks include setup features, e.g. '
                                          'neurons 45 and 26 light up the odor bag outside the arena in the odor '
                                          'stages, others the bedding texture. Read them as setup effects, not '
                                          'behaviour; the mouse mask avoids this.'}
# Mice genotype (family B): the camera period (odor-corner group) is unbalanced by genotype. Read-only checks of the
# primary family-B picks (scripts/eci/period_check.py; period_checks): per outcome, <set>/period_check/
# period_flags.json (per latent: within-genotype period AUC score, its pair and raw AUC; family-wise 'threshold',
# one-latent 'pointwise_threshold') and day_adjusted.csv (round test with the recording day as a covariate).
# PERIOD_OUTCOME: page outcome kind -> its outcome name in those files.
PERIOD_OUTCOME = {'time': 'mean', 'rate': 'bout_rate'}
# Frogs: group and recording session are fully confounded. Read-only checks of the primary picks
# (scripts/eci/session_check.py; session_checks): per outcome, <set>/session_check/session_flags.json (per latent:
# within-group session score, the group it is attained in; family-wise 'threshold', one-latent 'pointwise_threshold')
# and loso.csv (leave-one-session-out re-test of each pick). Same outcome names as PERIOD_OUTCOME.
# Frogs: what the page says under the title (data, contrasts, the session confound and what to trust), and its hover
# text (the details).
FROGS_NOTE = ('Xenopus juvenile froglets (Sweeney group, ISTA), filmed top-down, one frog per dish; hour 1, one video per '
              'frog: 13 WT, 14 FoxP1 gRNA1 half, 8 En1 gRNA4 half. Half = one side of the body CRISPR-edited; videos are '
              'mirrored so the mutant side is always the frog\'s right. Caveat: each group was filmed in its own '
              'recording sessions, so group is fully confounded with session and date. The strongest evidence of a real '
              'phenotype is a neuron that becomes one-sided in FoxP1 frogs only (a session cannot create a body-side '
              'asymmetry). En1 vs WT: nothing in the primary search (max pooling); its few picks under other '
              'aggregations look like session or degenerate effects.',
              'Comparisons: FoxP1 vs WT and En1 vs WT (one video = one frog, tau = mutant minus WT). Two chains: frog '
              'mask (frog patches only) and full frame; DINOv2 -> Matryoshka SAE -> NES at 5 fps. All frogs are '
              'genotyped. 5 of the 13 WT videos are mirrored at random too, to balance the mirrored dish labels. No '
              'recording session holds WT and mutant frogs, so the session check (each discovery\'s chips) can only '
              'measure how much a neuron varies between sessions of one group and whether a pick survives dropping any '
              'one session; it cannot separate group from session. With 21-27 frogs per comparison, late NES rounds '
              'can be degenerate fits (huge tau, tiny p), and latency picks with p far below 1e-20 come from censoring '
              'ties: read both with caution. En1: the full-frame average-pooling pick 97 is a dish-wide background '
              'neuron (likely a session artefact).')
# Per domain: page title word, subject words of the tooltips, top-by-p list length and clip rendering.
# top = per search, the N neurons with the smallest first-round p that NES did not select (listed on the
# page, so comparisons where nothing is selected can still be browsed); 0 = none.
# Clips: one page per (codes key, neuron, clip source) = its 4 blocks (top, top heat, least, least heat) of
# kr tiles (the first kr of the K picked clips), cols tiles per row, tile px; crop = side (source px) of the
# square cut around the activation peak of each 'top' clip (least rows and crop None: the whole frame);
# crf = H.264 quality. Tiles are rendered above their display size (the page shows 4 per row) so that the page's
# zoom (2x around each top clip's activation peak, clip_peaks) shows real pixels: mice whole frames at 224 px (was
# 144), ants whole frames at 320 px (--ants-tile; was 256; at 144 px ants were ~11 px long). Pack size per clip page vs
# the earlier tiles, measured on 8 pages: mice 224 / 144 = x2.40, ants 320 / 256 = x1.38. --ants-clips crop gives the
# earlier 256 px full-resolution cuts around each top clip's activation peak (legs and antennae, no context).
# cmp_top = top-by-p neurons also get 'this comparison' clips (ants: no; with 16 treatment pairs x 2 SAEs x 6
# outcomes that would be ~3000 extra clip pages; every listed neuron still gets its 'all videos' clips).
# keep = {experiment: treatment values kept, None = all} restricts the domain's videos and comparisons (only the
# analyses whose two arms are both kept; every clip, histogram, table and count uses the kept videos only);
# None = every video. vnote = how the page names the kept video set.
VIEW = {'mice': {'title': 'Mice', 'subject': 'mouse', 'subjects': 'mice', 'top': 0, 'cmp_top': True,
                 'tile': 224, 'cols': 8, 'kr': 16, 'crop': None, 'crf': 30, 'keep': None, 'vnote': ''},
        'ants': {'title': 'Ants', 'subject': 'ant', 'subjects': 'ants', 'top': 10, 'cmp_top': False,
                 'tile': 320, 'cols': 4, 'kr': 8, 'crop': None, 'crf': 27, 'keep': {'v2': None, 'v3': (2, 6, 8)},
                 'drop': ('v3_6_vs_8',), 'vnote': 'v2 (t=1, 2) and v3 t=2, 6, 8'},
        # frogs: one experiment (exp, hour 1 of every frog); the arms are the design column 'arm' (group, its values in
        # 'arms' order, control first), the recording session ('day' column) stands where the ants' recording day does;
        # clips: whole frames at 320 px as the ants (the frog, ~80 px of the 512 px frame, is ~50 px long; the page's 2x
        # zoom around each top clip's activation peak shows it at ~100 px); note = the domain note under the title;
        # top 3 = what the page lists (it shows at most 3 top-by-p neurons per search)
        'frogs': {'title': 'Frogs', 'subject': 'frog', 'subjects': 'frogs', 'top': 3, 'cmp_top': False,
                  'tile': 320, 'cols': 4, 'kr': 8, 'crop': None, 'crf': 28, 'keep': None, 'vnote': '',
                  'exp': 'hour 1', 'arm': 'group', 'arms': ('WT', 'FoxP1', 'En1'), 'day': 'session',
                  'size_text': 'with frog size (foreground patch count: how much of the frame the frog covers, '
                               'i.e. its posture and stretch) as a covariate (round-1 test)',
                  'note': FROGS_NOTE}}
# analysis metadata 'confound' missing (older result sets): these analyses keep their known confound note
CONFOUND_FALLBACK = {'v3_2_vs_8': 'recording day: t=8 only on day C (brighter arena), t=2 only on days A/B'}
# The page's two outcome controls. Outcome kind: event rate (bouts per minute above the threshold,
# scripts/eci/run_nes_bouts.py) or average time (per-video mean of the frame code, scripts/eci/run_nes.py).
# Spatial aggregation (how the patch codes of one frame become one value per neuron): the pooling of the
# codes. An (outcome, aggregation) pair is offered when its NES results exist (core outcomes, then
# extra_outcomes), otherwise its button is disabled.
OUTCOME_KINDS = [('rate', 'event rate', 'Bouts per minute: runs of frames above the neuron\'s activity threshold '
                  '(scripts/eci/run_nes_bouts.py)'),
                 ('time', 'average time', 'Per-video mean over time of the neuron\'s frame value '
                  '(scripts/eci/run_nes.py)'),
                 ('latency', 'latency', 'Seconds from the start of the video to the neuron\'s first bout (same '
                  'threshold as the event rate), within the first W seconds of every video of the comparison; a video '
                  'without a bout there gets W (right-censored) '
                  '(scripts/eci/run_nes_latency.py)')]
# Data-driven: an aggregation <id> is offered for an SAE when its NES results exist (the pooling column of the
# SAE's own summary.csv files, or a sibling result set <sae>_<id>). A pairwise / zone aggregation is added by
# appending its entry here once its NES result sets (<sae>_pairs, <sae>_zones) are final.
AGGS = [('max', 'max pooling', 'Frame value = max over the frame\'s patch codes (codes_max)'),
        ('mean', 'average pooling', 'Frame value = mean over the frame\'s patch codes (codes_mean)'),
        ('somp', 'SOMP', 'Simultaneous Orthogonal Matching Pursuit over the frame\'s patch tokens')]
CHUNK_BYTES = 200_000                 # target size of one neuron data file (assets/data/)
# Outcome kinds whose neurons (selections and top-by-p lists) get clips: every kind, so every neuron that can be
# selected on the page shows its clips (page_data audits it).
CLIP_KINDS = ('rate', 'time', 'latency')
# Feature prefixes shown on the page (Cfg.trim drops the rows of other prefixes, e.g. 1024, from every summary.csv:
# neurons found only there get no data and no clips)
PAGE_PREFIXES = (128, 256)
# Spatial aggregations whose latency runs are shown (SOMP latency is left out: ~40 % censored, degenerate)
LATENCY_AGGS = ('max', 'mean')
K = 16                                # clips picked per row (picks.json); VIEW kr of them are rendered
BLOCKS = ('top', 'top_heat', 'least', 'least_heat')  # the blocks of one clip page, in page order
LENGTHS = {'frame': 1, '1s': 5, '3s': 15}
# the lengths rendered and shown (render, page): the 3 s clips are picked (picks.json and the patch files keep them, so
# their caches stay valid) but not rendered, which keeps the clip packs inside the artifact's 1 GiB asset store
SHOW_LENGTHS = ('frame', '1s')
# clip sources of a neuron: 'all' (every video) and one per contrast the neuron is found in (aid)
KINDS = {'all': ('top', 'least'), 'cmp': ('top', 'least')}
HIST_BINS = 40
PICKS_V = 5                          # picks.json entry version (5 = one top row per contrast; 4 = + per-contrast selections)
STAGE_LABEL = MICE.stage_label  # 1 -> 'H,S'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FIELDS = ('pooling', 'outcome_type', 'test', 'correction', 'transform')  # string settings ('transform': latency runs)
KNOWN = set(FIELDS) | {'analysis_id', 'family', 'genotype', 'stage', 'transition', 'prefix', 'window', 'n_units',
                        'setting', 'n_tested_total', 'n_dropped', 'round', 'neuron', 'tau', 'se', 't', 'df', 'p',
                        'threshold', 'n_tested', 'direction', 'experiment', 'control', 'treatment', 'confound',
                        'stopped'}
# robustness chips: (field, alternative value) -> short label, tooltip
CHIP = {('test', 'signflip'): ('flip', 'sign-flip permutation test instead of the t-test'),
        ('correction', 'bh'): ('BH', 'Benjamini-Hochberg instead of Bonferroni'),
        ('pooling', 'max'): ('max', 'max-pooling over patches instead of mean'),
        ('pooling', 'mean'): ('mean-pool', 'mean-pooling over patches instead of max'),
        ('outcome_type', 'rate'): ('rate', 'firing rate (fraction of frames > 0) instead of mean activation'),
        ('outcome_type', 'mean'): ('mean', 'per-video mean activation as the outcome'),
        ('window', 'matched'): ('match', 'time-matched windows across stages'),
        ('window', 'full'): ('full', 'full videos'),
        ('window', 'common'): ('common', 'common window: the first W seconds of every video of the analysis'),
        ('transform', 'rank'): ('rank', 'latencies replaced by their ranks (censored videos tie), a Mann-Whitney-like '
                                'test, instead of raw seconds'),
        ('prefix', 128): ('128', 'searching only the first 128 neurons'),
        ('prefix', 256): ('256', 'searching only the first 256 neurons'),
        ('prefix', 1024): ('1024', 'searching all 1024 neurons')}
LINES, SEXES = MICE.subgroups['line'], MICE.subgroups['sex']
SUBSETS = [f'{l}_{x}' for l in ('all',) + LINES for x in ('all',) + SEXES if (l, x) != ('all', 'all')]
DEFAULT_OUTCOMES = ['Bout rate (max-pool)=maxpool_bouts:pooling=max,outcome_type=bout_rate,threshold_q=0.95,merge_gap=0,unit=bouts/min,short=bouts',
                    'Mean activation ({p}-pool)=.:pooling={p},outcome_type=mean,short={p}-pool']


def rdir_of(res):
    """Where a result set's searches are: <res>/pairs/ when that analysis-set run exists (ants: every treatment
    pair, scripts/eci/run_nes*.py --analysis-set pairs; its first analyses repeat the core ones), else <res>.
    The per-video caches stay in <res>/_cache."""
    return res / 'pairs' if (res / 'pairs' / 'summary.csv').exists() else res


def default_outcomes(res):
    """DEFAULT_OUTCOMES whose summary.csv exists; the mean-activation outcome uses the run's primary
    pooling as recorded in <res>/selected_neurons.json 'settings' ('primary (codes_max, ...)'), else mean."""
    sel = res / 'selected_neurons.json'
    m = re.search(r'codes_(mean|max)', json.loads(sel.read_text()).get('settings', '')) if sel.exists() else None
    specs = [x.replace('{p}', m[1] if m else 'mean') for x in DEFAULT_OUTCOMES]
    return [x for x in specs if (res / x.split('=', 1)[1].split(':')[0] / 'summary.csv').exists()]


def domain_of(res):
    """The domain whose NES root (src/eci/domain.py nes_root) contains the result set, with the analysis set
    (get_domain(name, analysis_set), when the domain has several) that covers every analysis id of the
    result set's summary.csv files (e.g. ants 'pairs' = every treatment pair of an experiment)."""
    for d in DOMS:
        if res.resolve().is_relative_to(d.nes_root.resolve()):
            sets = getattr(type(d), 'analysis_sets', ('core',))
            if len(sets) < 2:
                return d
            ids = set()
            for f in [res / 'summary.csv'] + sorted(res.glob('*/summary.csv')):
                if f.exists():
                    ids |= set(pd.read_csv(f, usecols=['analysis_id'])['analysis_id'].astype(str))
            for name in sets:
                dd = get_domain(d.name, name)
                if ids <= set(dd.analysis_ids()):
                    return dd
            raise SystemExit(f'{res}: analyses {sorted(ids)} not covered by any analysis set {sets} of {d.name}')
    raise SystemExit(f'{res}: not under a domain NES root ({", ".join(str(d.nes_root) for d in DOMS)})')


def discover_res(discover=False):
    """DEFAULT_RES (those with a summary.csv); discover: then every other finished SAE result set under the
    domains' NES roots (summary.csv and SUMMARY.md written, full codes merged; not an aggregation sibling
    <sae>_<agg>), sorted per domain."""
    out = [str((ROOT / r).resolve()) for r in DEFAULT_RES if (rdir_of(ROOT / r) / 'summary.csv').exists()]
    for r in DEFAULT_RES:
        if not (rdir_of(ROOT / r) / 'summary.csv').exists():
            print(f'WARNING: default result set {r} has no summary.csv: left out')
    if not discover:
        return out
    aggs = tuple(f'_{g}' for g, _, _ in AGGS)
    for d in DOMS:
        for r in sorted(p for p in d.nes_root.glob('*') if p.is_dir()):
            if (str(r.resolve()) in out or r.name.endswith(aggs) or not (r / 'summary.csv').exists()
                    or not (r / 'SUMMARY.md').exists() or not (d.eci_dir / 'codes' / r.name / 'DONE').exists()):
                continue
            out.append(str(r.resolve()))
            print(f'discovered result set {r}')
    return out


def keep_mask(view, df):
    """(len(df),) bool: rows (videos or frames, columns experiment and T) of the kept videos (VIEW 'keep')."""
    k = view['keep']
    if k is None:
        return np.ones(len(df), bool)
    e, t = df['experiment'].astype(str).values, df['T'].astype(int).values
    return np.array([x in k and (k[x] is None or int(v) in k[x]) for x, v in zip(e, t)], bool)


def kept_analyses(dom, view):
    """The domain's analysis ids whose control and treatment videos are both kept (VIEW 'keep') and that are not in
    VIEW 'drop', in order."""
    ids = [a for a in dom.analysis_ids() if a not in view.get('drop', ())]
    if view['keep'] is None:
        return ids
    out = []
    for aid in ids:
        m = dom.analysis(aid).meta
        arms = pd.DataFrame({'experiment': [m['experiment']] * 2, 'T': [m['control'], m['treatment']]})
        if keep_mask(view, arms).all():
            out.append(aid)
    return out


def model_of(cfg):
    """Model bar values of an SAE: MODELS, else from its name (encoder DINOv2, SAE type = the name's first
    word) and representation (fg448 = '<subject> mask', else 'full frame')."""
    if cfg.sae in MODELS:
        return MODELS[cfg.sae]
    word = {'mice': 'mouse', 'ants': 'ants', 'frogs': 'frog'}.get(cfg.dom.name, cfg.dom.name)
    return {'encoder': 'DINOv2', 'sae': cfg.sae.split('_')[0].capitalize(),
            'input': f'{word} mask' if cfg.rep == 'fg448' else 'full frame'}


def outcome_kind(o):
    return 'rate' if o['primary']['outcome_type'] == 'bout_rate' else 'time'


def extra_outcomes(cfg):
    """The (outcome kind, spatial aggregation) pairs the core outcomes do not cover, when their NES results
    exist: (1) the other poolings of a core outcome in its own summary.csv (the NES robustness grid runs
    full searches for them); (2) sibling result sets <res>_<aggregation> (e.g. <sae>_mean, <sae>_somp, written
    by scripts/eci/run_nes*.py on those codes): their summary.csv (average time) and <subdir>/summary.csv
    (event rate: outcome_type bout_rate, threshold q 0.95 / gap 0 when several), caches in their _cache with
    the run_nes*.py names (video_summaries_<pooling>.npz, bout_summaries_max_<aggregation>.npz). Their neurons
    are listed without clips (no clip budget left); per-video values are skipped with a warning when the
    cache file is missing."""
    have = {(o['kind'], o['agg']) for o in cfg.outcomes}
    agg_ids = [g for g, _, _ in AGGS]
    out = []
    for o in cfg.outcomes:
        t = o['tidy']
        for p in agg_ids:
            if (o['kind'], p) in have or p not in set(t['pooling'].astype(str)):
                continue
            prim = dict(o['primary'], pooling=p)
            if not match(t, prim).any():
                continue
            label = re.sub(r'\((\w+)-pool\)', f'({p}-pool)', o['label']) if '-pool)' in o['label'] else f'{o["label"]} ({p}-pool)'
            out.append({**o, 'label': label, 'primary': prim, 'codes': p, 'agg': p, 'pool': p, 'extra': True,
                        'short': f'{p}-pool', 'id': f'x{o["kind"]}{p}'})
            have.add((o['kind'], p))
    for agg in agg_ids:
        d = cfg.res.parent / f'{cfg.res.name}_{agg}'
        if not d.is_dir():
            continue
        rd = rdir_of(d)
        for s in sorted(rd.glob('summary.csv')) + sorted(p for p in rd.glob('*/summary.csv') if p.parent.name != 'pairs'
                                                         and not p.parent.name.endswith('_latency')):
            t = cfg.trim(pd.read_csv(s))
            for kind, ot in (('rate', 'bout_rate'), ('time', 'mean')):
                g = t[t['outcome_type'] == ot]
                if (kind, agg) in have or not len(g) or (kind == 'time' and s.parent != rd):
                    continue
                kv = {'pooling': str(g['pooling'].iloc[0]), 'outcome_type': ot}
                if kind == 'rate':
                    kv.update({c: v for c, v in (('threshold_q', '0.95'), ('merge_gap', '0')) if c in t.columns})
                try:
                    prim = default_primary(t, kv)
                except SystemExit as e:
                    print(f'[{cfg.tag}] WARNING: {s} not used: {e}')
                    continue
                name = 'SOMP' if agg == 'somp' else f'{agg}-pool'
                out.append({'label': f'{"Bout rate" if kind == "rate" else "Mean activation"} ({name})', 'dir': s.parent,
                            'tidy': t, 'primary': prim, 'codes': agg, 'pool': prim['pooling'], 'kind': kind,
                            'agg': agg, 'extra': True, 'cache': d / '_cache', 'bfile': f'bout_summaries_max_{agg}.npz',
                            'unit': 'bouts/min' if kind == 'rate' else 'activation', 'short': name,
                            'id': f'x{kind}{agg}'})
                have.add((kind, agg))
    # latency to the first bout (scripts/eci/run_nes_latency.py): <rdir>/<P>_latency/ of the result set itself
    # (P = its frame pooling: maxpool_latency = max, meanpool_latency = mean) and of the aggregation siblings
    # <sae>_<agg> (agg = that sibling's aggregation); per-video values from <_cache>/latency[_<P>].npz
    for agg0, d in [(None, cfg.res)] + [(g, cfg.res.parent / f'{cfg.res.name}_{g}') for g in agg_ids]:
        if not d.is_dir():
            continue
        for s in sorted(rdir_of(d).glob('*_latency/summary.csv')):
            P = {'maxpool': 'max', 'meanpool': 'mean'}.get(s.parent.name[:-len('_latency')], s.parent.name[:-len('_latency')])
            agg = agg0 or P
            if ('latency', agg) in have or agg not in LATENCY_AGGS:
                continue
            t = cfg.trim(pd.read_csv(s))
            try:
                prim = default_primary(t, {'pooling': P, 'outcome_type': 'latency', 'threshold_q': '0.95'})
            except SystemExit as e:
                print(f'[{cfg.tag}] WARNING: {s} not used: {e}')
                continue
            lf = d / '_cache' / ('latency.npz' if P == 'max' else f'latency_{P}.npz')
            name = 'SOMP' if agg == 'somp' else f'{agg}-pool'
            out.append({'label': f'Latency to first bout ({name})', 'dir': s.parent, 'tidy': t, 'primary': prim,
                        'codes': agg, 'pool': P, 'kind': 'latency', 'agg': agg, 'extra': True, 'cache': d / '_cache',
                        'lfile': lf.name, 'unit': 's', 'short': 'latency', 'id': f'xlatency{agg}', 'pwin': 'common'})
            have.add(('latency', agg))
    for o in out:
        print(f'[{cfg.tag}] extra outcome {o["label"]!r} ({o["kind"]} x {o["agg"]}): {o["dir"]} primary {o["primary"]}')
    return out


def extra_subsets(cfg):
    """Subgroup searches of the extra outcomes (event rate / average time, not latency): {subgroup: [outcome, ...]}
    from the aggregation sibling <sae>_<agg> of each outcome (its subgroup runs, scripts/eci/run_nes_subsets.sh SPECS
    <sae>_<agg>:<agg>): <sibling>/subsets/<line>_<sex>/summary.csv (average time) or its */summary.csv with that
    outcome type (event rate). The outcome keeps its primary setting (columns the subgroup table lacks dropped)."""
    out = {}
    for o in cfg.extras:
        if o['kind'] not in ('rate', 'time'):  # latency runs have no subgroup searches
            continue
        sib = cfg.res.parent / f'{cfg.res.name}_{o["agg"]}'
        for name in SUBSETS:
            d = sib / 'subsets' / name
            fs = [d / 'summary.csv'] if o['kind'] == 'time' else sorted(d.glob('*/summary.csv'))
            for f in fs:
                if not f.exists():
                    continue
                t = cfg.trim(pd.read_csv(f))
                prim = {k: v for k, v in o['primary'].items() if k in KNOWN or k in t.columns}
                if not match(t, prim).any():
                    continue
                out.setdefault(name, []).append({**o, 'dir': f.parent, 'tidy': t, 'primary': prim})
                break
    n = {o['id']: sum(o['id'] in {x['id'] for x in v} for v in out.values()) for o in cfg.extras}
    print(f'[{cfg.tag}] extra-outcome subgroups: ' + ', '.join(f'{k} {v}' for k, v in n.items() if v))
    return out


class Cfg:
    """One NES result set (one SAE)."""

    def __init__(self, res, a, out):
        self.res = (ROOT / res).resolve() if not Path(res).is_absolute() else Path(res)
        self.sae = self.res.name
        self.tag = self.sae.split('_')[-2]  # 'fg448', 'ep20', 'antsfg': asset file prefix
        self.dom = domain_of(self.res)
        self.view = VIEW[self.dom.name]
        self.vdom = None if self.dom is MICE else self.dom  # src/eci/viz.py domain argument (None = mice v1)
        self.order = kept_analyses(self.dom, self.view)  # the analyses shown (VIEW 'keep'); others are dropped
        dropped = [x for x in self.dom.analysis_ids() if x not in self.order]
        if dropped:
            print(f'[{self.sae}] analyses left out (VIEW keep {self.view["keep"]}): {dropped}')
        self.out = out
        self.assets = out / 'assets'
        self.work = self.res / '_cache' / 'explorer'
        self.outcomes = []
        self.rdir = rdir_of(self.res)
        af = self.rdir / 'analyses.json'  # per analysis: experiment, control, treatment, confounded, confound
        self.ameta = {r['analysis_id']: r for r in json.loads(af.read_text())} if af.exists() else {}
        for spec in a.outcome or default_outcomes(self.rdir):
            label, rest = spec.split('=', 1)
            sub, _, opts = rest.partition(':')
            d = (self.rdir / sub).resolve()
            if not (d / 'summary.csv').exists():
                raise SystemExit(f'{d}/summary.csv missing (outcome {label!r})')
            tidy = self.trim(pd.read_csv(d / 'summary.csv'))
            kv = dict(x.split('=', 1) for x in opts.split(',') if x)
            meta = {k: kv.pop(k) for k in ('codes', 'unit', 'short') if k in kv}
            prim = default_primary(tidy, kv)
            codes = meta.get('codes', prim['pooling'])
            self.outcomes.append({'label': label, 'dir': d, 'tidy': tidy, 'primary': prim, 'codes': codes,
                                  'unit': meta.get('unit', 'activation' if prim['outcome_type'] == 'mean'
                                                   else prim['outcome_type']),
                                  'short': meta.get('short', label.split()[0].lower()),
                                  'id': re.sub(r'\W+', '', label.lower())[:16] or 'o'})
            self.outcomes[-1].update(kind=outcome_kind(self.outcomes[-1]), agg=prim['pooling'], pool=prim['pooling'],
                                     extra=False, cache=self.res / '_cache')
            print(f'[{self.tag}] outcome {label!r}: {d} primary {prim} codes_{codes}')
        self.extras = extra_outcomes(self)
        # size check (n_fg covariate) is meaningless when every frame has the same foreground size (full frame)
        nf = self.dom.eci_dir / 'codes' / self.sae / 'n_fg.npy'
        self.size_ok = not (nf.exists() and np.ptp(np.load(nf, mmap_mode='r')[::97]) == 0)
        for o in self.outcomes + self.extras:
            o['size_ok'] = self.size_ok
        # subgroup result sets (scripts/eci/run_nes*.py --line/--sex --primary-only): the same outcomes
        # read from <res>/subsets/<line>_<sex>/<SUBDIR>/, when their summary.csv exists
        self.subsets = {}
        for name in SUBSETS:
            outs = []
            for o in self.outcomes:
                d = self.res / 'subsets' / name / o['dir'].relative_to(self.res)
                if (d / 'summary.csv').exists():
                    outs.append({**o, 'dir': d, 'tidy': self.trim(pd.read_csv(d / 'summary.csv'))})
            if len(outs) == len(self.outcomes):
                self.subsets[name] = outs
            elif outs:
                raise SystemExit(f'subset {name}: only {len(outs)} of {len(self.outcomes)} outcomes present')
        print(f'[{self.tag}] subgroups: {sorted(self.subsets)}')
        self.xsubsets = extra_subsets(self)
        self.noclip = set()  # (codes key, neuron) listed without clips (beyond --max-pages, plan_clips)
        self.clipplan = None  # {codes key: {neuron: contrasts}} given clips (plan_clips)
        self.keys = sorted({o['codes'] for o in self.outcomes + self.extras})  # clip rankings (codes_<key>)
        from src.eci.viz import frame_rot90, representation
        self.rep = representation(self.sae, DATASET, domain=self.vdom)
        cc = self.dom.eci_dir / 'codes' / self.sae / 'config.json'
        self.align = json.loads(cc.read_text()).get('align', 'none') if cc.exists() else 'none'
        self.rot = frame_rot90(self.sae, DATASET, domain=self.vdom)  # None or per-row 90-degree turns (clips)
        self.artefact = ARTEFACT_EP20 if self.sae == 'matryoshka_btk_1024_k16_ep20_s0' else set()
        # Claude's per-neuron interpretations (<res>/interp/interpretations.json {neuron: {text, conf, tags}}), if any
        fi = self.res / 'interp' / 'interpretations.json'
        self.interp = {str(k): v for k, v in json.loads(fi.read_text()).items()} if fi.exists() else {}
        self.crf = a.crf if a.crf is not None else self.view['crf']
        self.tile, self.cols, self.kr, self.crop = (self.view[k] for k in ('tile', 'cols', 'kr', 'crop'))
        print(f'[{self.tag}] SAE {self.sae}: representation {self.rep}, align {self.align}, '
              f'artefact flags {sorted(self.artefact)}')

    def trim(self, t):
        """summary.csv rows of the analyses shown (self.order) and the prefixes shown (PAGE_PREFIXES). A run where no
        search selected anything writes no per-neuron columns: they are added empty."""
        for c in ('neuron', 'tau', 'se', 't', 'df', 'p', 'threshold', 'n_tested', 'direction'):
            if c not in t.columns:
                t = t.assign(**{c: np.nan})
        return t[t['analysis_id'].astype(str).isin(self.order) & t['prefix'].isin(PAGE_PREFIXES)].reset_index(drop=True)


def extra_cols(tidy):
    return [c for c in tidy.columns if c not in KNOWN]


def default_primary(tidy, kv):
    prim = {'test': kv.pop('test', 't'), 'correction': kv.pop('correction', 'bonferroni')}
    if 'pooling' not in kv or 'outcome_type' not in kv:
        g = tidy[(tidy['test'] == prim['test']) & (tidy['correction'] == prim['correction'])]
        combos = sorted(set(map(tuple, g[['pooling', 'outcome_type']].astype(str).values)))
        if ('mean', 'mean') in combos:
            kv.setdefault('pooling', 'mean'), kv.setdefault('outcome_type', 'mean')
        elif len(combos) == 1:
            kv.setdefault('pooling', combos[0][0]), kv.setdefault('outcome_type', combos[0][1])
        else:
            raise SystemExit(f'ambiguous primary: pass pooling=..,outcome_type=..; options {combos}')
    prim.update(pooling=kv.pop('pooling'), outcome_type=kv.pop('outcome_type'))
    if 'transform' in tidy.columns:  # latency runs: raw seconds are the primary, ranks a sensitivity
        prim['transform'] = kv.pop('transform', 'none')
    sub = tidy[match(tidy, prim)]
    for c in extra_cols(tidy):
        vals = sorted(sub[c].dropna().unique())
        if c in kv:
            prim[c] = float(kv.pop(c))
        elif len(vals) > 1 and c == 'bout_rule':
            prim[c] = 0.0  # plain threshold runs; 1 = the min2+hyst sensitivity
        elif len(vals) > 1:
            raise SystemExit(f'setting column {c!r} has values {vals}: pass {c}=<primary value>')
        elif vals:
            prim[c] = float(vals[0])
    if kv:
        raise SystemExit(f'unknown outcome options {kv}')
    return prim


def match(t, spec):
    """Rows matching spec; NaN in an extra setting column means 'not applicable' and matches."""
    m = np.ones(len(t), bool)
    for k, v in spec.items():
        if k == 'prefix':
            m &= (t[k] == int(v)).values
        elif k in KNOWN:
            m &= (t[k].astype(str) == str(v)).values
        else:
            col = t[k].astype(float)
            m &= (np.isclose(col, float(v)) | col.isna()).values
    return m


def primary_rows(o):
    """The outcome's primary setting, primary window only (full videos; latency runs: 'common', the first W frames
    of every video of the analysis). The page keys these searches by window 'full' (search_results)."""
    t = o['tidy']
    return t[match(t, o['primary']) & (t['window'] == o.get('pwin', 'full')).values]


def mean_outcome(o):
    return (o['primary']['pooling'], o['primary']['outcome_type']) == ('mean', 'mean')


def is_bout(o):
    return o['primary']['outcome_type'] == 'bout_rate'


_rates = {}


def bout_rates(cfg, o, window='full', fps=5.0):
    """Per-video bouts/min of every neuron (src/eci/contrasts.py bout_outcomes, same cache as
    scripts/eci/run_nes_bouts.py) -> (Index of observation_id, (n_obs, m) array)."""
    q, g = o['primary']['threshold_q'], int(o['primary']['merge_gap'])
    w = {'full': 'full', 'trim30': 'trim', 'matched': 'last'}[window]
    k = (str(o['cache']), o['pool'], w, q, g)
    if k not in _rates:
        f = np.load(o['cache'] / o.get('bfile', f'bout_summaries_{o["pool"]}.npz'))
        c, nf = f[f'{w}__{q:g}__{g}__count'], f[f'{w}__n_frames']
        _rates[k] = (pd.Index(f['observation_id'].astype(str)), c / (nf[:, None] / fps / 60.0))
    return _rates[k]


def is_latency(o):
    return o['primary']['outcome_type'] == 'latency'


def latency_W(o, aid):
    """W (frames) of latency outcome o in analysis aid: the common window (first W frames of every video of the
    analysis, scripts/eci/run_nes_latency.py), from <outcome dir>/<aid>/result.json [primary setting] 'W_frames'."""
    prim = primary_rows(o)
    h = prim[prim['analysis_id'] == aid]
    r = read_json(o['dir'] / aid / 'result.json')[h['setting'].iloc[0]]
    W = {int(read_json(o['dir'] / aid / 'result.json')[x]['W_frames']) for x in h['setting'].unique()}
    if len(W) != 1:
        raise SystemExit(f'{o["dir"]}/{aid}: several W over the primary settings {W}')
    assert int(r['window_frames']) == int(r['W_frames']), r
    return W.pop()


def latency_frames(o):
    """(Index of observation_id, (n_obs, m) first-bout frame index in the full video (= its length when no bout),
    (n_obs,) video length in frames) from the latency run's cache (scripts/eci/run_nes_latency.py). Uncapped: the
    latency of an analysis = min(index, W) (latency_W), censored where index >= W."""
    k = ('lat', str(o['cache'] / o['lfile']), o['primary']['threshold_q'])
    if k not in _rates:
        f = np.load(o['cache'] / o['lfile'])
        _rates[k] = (pd.Index(f['observation_id'].astype(str)), f[f'full__{o["primary"]["threshold_q"]:.2f}'],
                     f['nf__full'])
    return _rates[k]


def video_outcome(cfg, o):
    """(Index of observation_id, (n_obs, m)) per-video value of the tested outcome, full videos, as the
    NES runs computed it: bouts/min, the per-video mean of codes_<pooling>, or the latency to the first bout in
    seconds (from the outcome's result set _cache). Extra outcomes whose cache file is missing -> None (page: no
    per-video values)."""
    f = o['cache'] / (o.get('bfile', f'bout_summaries_{o["pool"]}.npz') if is_bout(o) else o['lfile'] if is_latency(o)
                      else f'video_summaries_{o["pool"]}.npz')
    if o['extra'] and not f.exists():
        print(f'[{cfg.tag}] WARNING: {f} missing: no per-video values for {o["label"]!r}')
        return None
    if is_latency(o):  # uncapped first-bout time (s); the page caps it at the analysis's W (DATA.lwin)
        ids, lat, _ = latency_frames(o)
        return ids, lat / cfg.dom.fps
    if is_bout(o):
        return bout_rates(cfg, o, 'full')
    if o['primary']['outcome_type'] != 'mean':
        raise SystemExit(f'per-video values: outcome_type {o["primary"]["outcome_type"]!r} not supported')
    f = np.load(f)
    return pd.Index(f['observation_id'].astype(str)), f['full__mean']


def vmeta():
    from src.eci.viz import video_meta
    if 'vm' not in _rates:
        _rates['vm'] = video_meta(ROOT / 'data')
    return _rates['vm'].copy()


def rate_table(cfg, o, j):
    """stage x genotype: mean over pools of the per-video bouts/min, 95% t-CI (as viz.stage_genotype_table)."""
    from scipy import stats
    vm = vmeta()
    ids, rate = bout_rates(cfg, o)
    vm['y'] = rate[ids.get_indexer(vm['observation_id'].astype(str)), j]
    rows = []
    for (st, g), grp in vm.dropna(subset=['y']).groupby(['stage', 'genotype']):
        y = grp.groupby('pool')['y'].mean().values
        h = stats.t.ppf(0.975, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0.0
        rows.append({'stage': int(st), 'genotype': g, 'mean': round(float(y.mean()), 4),
                     'lo': round(float(y.mean() - h), 4), 'hi': round(float(y.mean() + h), 4), 'n_pools': int(len(y))})
    return rows


def pre(key):
    """cache file prefix per ranking codes (historical: mean 'n', max 'x'; others their key, e.g. 'somp')."""
    return {'mean': 'n', 'max': 'x'}.get(key, key)


def short_obs(obs):
    """2024-07-26_14-29-30_BHVScreen_rd11_2_SocialOdor_Test -> ('rd11_2 Test', '07-26 14:29')."""
    m = re.match(r'\d{4}-(\d\d-\d\d)_(\d\d)-(\d\d)-\d\d_BHVScreen_(rd[\w]+?)_(\w+?)Odor_(\w+)$', obs)
    if not m:
        return obs[:24], ''
    return f'{m[4]} {m[6]}', f'{m[1]} {m[2]}:{m[3]}'


def stages_of(aid):
    return [int(x) for x in MICE.analysis(aid).stages]


def wanted(cfg):
    """{codes key: {neuron: sorted contrasts (analysis ids) it is shown in}} of the neurons given clips
    (cfg.clipplan, plan_clips), else every neuron the page lists: the primary searches (any prefix, full
    window) of every outcome (core and extra) ranked by that key, the top-by-p lists (VIEW 'top') with their
    contrast, and the neurons found only in a subgroup search ('all videos' clips only: 'this comparison'
    clips stay full-cohort)."""
    if cfg.clipplan is not None:
        return {k: {j: sorted(v, key=cfg.order.index) for j, v in w.items()} for k, w in cfg.clipplan.items()}
    return {k: {j: sorted(v, key=cfg.order.index) for j, v in w.items()} for k, w in clip_candidates(cfg)[0].items()}


def clip_candidates(cfg):
    """({codes key: {neuron: contrasts}} of every listed neuron, [(codes key, neuron, contrasts)] in clip
    priority order): neurons selected by a primary search (best p first, with the contrasts they are selected
    in), then the top-by-p lists rank 1 of every list, rank 2, ..., each with its contrast, then the
    subgroup-only neurons (most round-1 subgroup hits, then best p)."""
    sel = {}  # (codes key, neuron) -> (best p, contrasts)
    clip_outs = [o for o in cfg.outcomes + cfg.extras if o['kind'] in CLIP_KINDS]
    for o in clip_outs:
        p = primary_rows(o).dropna(subset=['neuron'])
        for aid, j, pv in zip(p['analysis_id'], p['neuron'].astype(int), p['p']):
            b, st = sel.get((o['codes'], int(j)), (1.0, set()))
            sel[(o['codes'], int(j))] = (min(b, float(pv)), st | {aid})
    seq = [(k, j, st) for (k, j), (b, st) in sorted(sel.items(), key=lambda x: (x[1][0], x[0]))]
    lists = [[(o['codes'], j, {aid} if cfg.view['cmp_top'] else set()) for j in L]
             for o in clip_outs for (aid, _, _), L in
             sorted(top_lists(cfg, o).items(), key=lambda x: (cfg.order.index(x[0][0]), -x[0][1]))]
    seq += [L[r] for r in range(max(map(len, lists), default=0)) for L in lists if r < len(L)]
    hits = subgroup_hits(cfg)
    seq += [(k, j, set()) for k, j in sorted(subgroup_neurons(cfg), key=lambda x: (-hits.get(x, (0, 1.0))[0],
                                                                                   hits.get(x, (0, 1.0))[1], x))]
    want = {k: {} for k in cfg.keys}
    for k, j, st in seq:
        want.setdefault(k, {}).setdefault(int(j), set()).update(st)
    return want, seq


def full_neurons(cfg):
    """{(codes key, neuron)} of the full-cohort primary searches (full window, any prefix) of the core outcomes and
    the extra outcomes that get clips (CLIP_KINDS)."""
    return {(o['codes'], int(j)) for o in cfg.outcomes + [x for x in cfg.extras if x['kind'] in CLIP_KINDS]
            for j in primary_rows(o)['neuron'].dropna()}


def all_subsets(cfg):
    """[[subgroup outcome, ...], ...]: the subgroup searches of the core outcomes and of the extra outcomes."""
    return list(cfg.subsets.values()) + list(cfg.xsubsets.values())


def subgroup_neurons(cfg, round1_only=False):
    """[(codes key, neuron)] found in any subgroup's primary search (full window, any prefix; core and extra
    outcomes) but not in the full cohort, sorted."""
    full = full_neurons(cfg)
    out = set()
    for outs in all_subsets(cfg):
        for o in outs:
            p = primary_rows(o).dropna(subset=['neuron'])
            if round1_only:
                p = p[p['round'] == 1]
            out |= {(o['codes'], int(j)) for j in p['neuron']} - full
    return sorted(out)


def subgroup_hits(cfg):
    """{(codes key, neuron): (number of round-1 hits over the subgroup searches, best p)} of the
    subgroup-only neurons (primary, full window, any prefix / outcome / analysis)."""
    full = full_neurons(cfg)
    out = {}
    for outs in all_subsets(cfg):
        for o in outs:
            p = primary_rows(o).dropna(subset=['neuron'])
            for j, pv in zip(p[p['round'] == 1]['neuron'].astype(int), p[p['round'] == 1]['p']):
                k = (o['codes'], int(j))
                if k not in full:
                    n, b = out.get(k, (0, 1.0))
                    out[k] = (n + 1, min(b, float(pv)))
    return out


def plan_clips(cfgs, max_pages):
    """Clip pages (one per codes key x neuron x clip source, each rendered at every length) of each SAE:
    every listed neuron gets its 'all videos' page and one page per contrast it is shown in, in
    clip_candidates priority order while the SAE's pages stay <= max_pages (0 = no limit). A neuron that does
    not fit is listed without clips (cfg.noclip)."""
    for c in cfgs:
        want, seq = clip_candidates(c)
        plan, n = {k: {} for k in c.keys}, 0
        for k, j, st in seq:
            add = (0 if j in plan[k] else 1) + len(set(st) - plan[k].get(j, set()))
            if max_pages and n + add > max_pages:
                continue
            plan[k].setdefault(j, set()).update(st)
            n += add
        c.clipplan = plan
        c.noclip = {(k, j) for k, w in want.items() for j in w if j not in plan[k]}
        print(f'[{c.tag}] clips: {sum(len(w) for w in plan.values())} neurons, {n} pages per length '
              f'(limit {max_pages or "none"}): ' + ', '.join(f'codes_{k} {len(w)}' for k, w in plan.items())
              + (f'; listed without clips (page limit): {sorted(c.noclip)}' if c.noclip else ''))


def read_json(f):
    """json file content, cached (result.json files are read once per build)."""
    k = f'json_{f}'
    if k not in _rates:
        _rates[k] = json.loads(Path(f).read_text())
    return _rates[k]


def top_lists(cfg, o):
    """{(analysis id, prefix, window): [neuron, ...]} the VIEW 'top' neurons with the smallest first-round
    p of each primary search (full window) of outcome o that NES did not select, from
    <outcome dir>/<analysis>/result.json [setting] 'first_round'; {} when VIEW 'top' is 0."""
    if not cfg.view['top']:
        return {}
    out = {}
    prim = primary_rows(o)
    for (aid, prefix, window), h in prim.groupby(['analysis_id', 'prefix', 'window']):
        f = o['dir'] / aid / 'result.json'
        fr = read_json(f).get(h['setting'].iloc[0], {}).get('first_round') if f.exists() else None
        if fr is None:
            print(f'WARNING: {f} [{h["setting"].iloc[0]}] missing: no top-by-p list for {o["label"]!r} {aid} p{prefix}')
            continue
        sel = set(h['neuron'].dropna().astype(int))
        order = np.argsort(np.asarray(fr['p'], dtype=np.float64), kind='stable')
        out[(aid, int(prefix), window)] = [int(fr['neuron'][i]) for i in order
                                           if int(fr['neuron'][i]) not in sel][:cfg.view['top']]
    return out


def top_rows(cfg, o, aid, prefix, window):
    """[[neuron, tau, p], ...] of top_lists (first-round tau / p, as in result.json)."""
    L = top_lists(cfg, o).get((aid, int(prefix), window), [])
    if not L:
        return []
    prim = primary_rows(o)
    h = prim[(prim['analysis_id'] == aid) & (prim['prefix'] == prefix) & (prim['window'] == window)]
    fr = read_json(o['dir'] / aid / 'result.json')[h['setting'].iloc[0]]['first_round']
    at = {int(j): i for i, j in enumerate(fr['neuron'])}
    return [[j, sig4(fr['tau'][at[j]]), float(f'{fr["p"][at[j]]:.4g}')] for j in L]


def kinds(src):
    return KINDS['all' if src == 'all' else 'cmp']


def contrast_groups(cfg, aid, vids):
    """The videos of one contrast, as (top0 mask, top1 mask) over the rows of vids: A (e.g. A_wt_1to2) =
    that genotype's videos in stage a / stage b; B (B_stage3) = het / wt videos of that stage; ants
    (v3_2_vs_8) = the control / treatment videos of that experiment. 'least' uses the union."""
    an = cfg.dom.analysis(aid)
    if cfg.dom is not MICE:
        m = np.ones(len(vids), bool)
        for k, v in an.where.items():
            m &= (vids[k] == v).values
        return m & vids[an.arm].isin(an.control).values, m & vids[an.arm].isin(an.treatment).values
    st, gen = vids['stage'].astype(int).values, vids['genotype'].astype(str).values
    if an.family == 'A':
        g, (a, b) = an.genotype, an.stages
        return (gen == g) & (st == a), (gen == g) & (st == b)
    s = an.where['stage']
    return (gen == 'het') & (st == s), (gen == 'wt') & (st == s)


# ---------------------------------------------------------------------- step: data
GROUP_COLS = {'mice': ('genotype', 'stage'), 'ants': ('experiment', 'T'),
              'frogs': ('experiment', 'group')}  # histogram / table groups (frogs: experiment = VIEW 'exp')


def arm_of(view, v):
    """An arm value as the page holds it: the treatment number (ants, int) or the group name (frogs, VIEW 'arm')."""
    return str(v) if view.get('arm') else int(v)


def with_exp(view, df):
    """df with the column 'experiment' = VIEW 'exp' added when the domain has a single unnamed experiment (frogs)."""
    if view.get('exp') and 'experiment' not in df.columns:
        df['experiment'] = view['exp']
    return df


def day_label(d):
    """Recording day: v2 dates '18.04.2024' -> '18.04', v3 days 'A' / 'B' / 'C' unchanged."""
    return d[:5] if re.match(r'\d\d\.\d\d\.\d{4}$', d) else d


def clip_group(x):
    """Group of a clip in the logs: 'het|S2' (mice), 'v3 t=8 day C' (ants), 'hour 1 t=FoxP1 day 153' (frogs)."""
    return f'{x["genotype"]}|S{x["stage"]}' if 'stage' in x else f'{x["experiment"]} t={x["T"]} day {x["day"]}'


def group_table(vids, y, view=None):
    """Ants: per (experiment, T) the mean over videos of the per-video values y, 95% t-CI, n videos (frogs: T = the
    group, VIEW 'arm')."""
    from scipy import stats
    view = view or {}
    arm = view.get('arm', 'T')
    rows = []
    for (e, t), g in pd.DataFrame({'e': with_exp(view, vids.copy())['experiment'].astype(str),
                                   't': vids[arm].astype(str if view.get('arm') else int), 'y': y}).groupby(['e', 't']):
        v = g['y'].values.astype(np.float64)
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        rows.append({'experiment': e, 'T': arm_of(view, t), 'mean': float(v.mean()), 'lo': float(v.mean() - h),
                     'hi': float(v.mean() + h), 'n': int(len(v))})
    return rows


def activation_hist(x, group, n_groups):
    """40 linear bins from 0 to max(x): counts over all frames and per group id."""
    hi = float(x.max()) if x.max() > 0 else 1.0
    b = np.minimum((np.maximum(x, 0) / hi * HIST_BINS).astype(np.int64), HIST_BINS - 1)
    c = np.bincount(group * HIST_BINS + b, minlength=n_groups * HIST_BINS).reshape(n_groups, HIST_BINS)
    return np.linspace(0, hi, HIST_BINS + 1), c


def codes_array(cfg, src, key):
    """(n_frames, m) codes_<key> of the SAE, full codes row order: src.codes (mean, max), else the
    aggregation sibling <codes dir>/<sae>_<key>/codes_<key>.npy (e.g. SOMP), same row order (checked)."""
    if key in src.codes:
        return src.codes[key]
    f = cfg.dom.eci_dir / 'codes' / f'{cfg.sae}_{key}' / f'codes_{key}.npy'
    if not f.exists():
        raise SystemExit(f'{f} missing (clips ranked by codes_{key})')
    Z = np.load(f, mmap_mode='r')
    if Z.shape[0] != len(src.meta):
        raise SystemExit(f'{f}: {Z.shape[0]} rows, the full codes have {len(src.meta)}')
    return Z


def step_data(cfg, overwrite):
    out = cfg.work / 'picks.json'
    res = json.loads(out.read_text()) if out.exists() and not overwrite else {}
    res.pop('contrast', None)
    want = wanted(cfg)
    # keep only what is wanted now (drops stale entries, e.g. an older outcome's codes key)
    res['by_key'] = {k: {j: v for j, v in res.get('by_key', {}).get(k, {}).items() if int(j) in want[k]}
                     for k in cfg.keys}
    # vset: the kept video set (VIEW 'keep') the entry was picked among (None = every video, the mice entries)
    vset = None if cfg.view['keep'] is None else json.dumps({e: (sorted(t) if t else None)
                                                             for e, t in sorted(cfg.view['keep'].items())})
    missing = {k: sorted(j for j, aids in v.items() if res['by_key'][k].get(str(j), {}).get('v') != PICKS_V
                         or res['by_key'][k][str(j)].get('vset') != vset
                         or sorted(res['by_key'][k][str(j)]['clips']) != sorted(['all'] + aids))
               for k, v in want.items()}
    if not any(missing.values()):
        print(f'[{cfg.tag}] data: cached', {k: len(v) for k, v in res['by_key'].items()}, 'neurons')
        out.write_text(json.dumps(res))
        return
    from types import SimpleNamespace
    from src.eci.viz import (CodeSource, load_full_codes, pick_least_windows, pick_top_windows, scan_windows,
                             stage_genotype_table, subset_windows)
    src = load_full_codes(cfg.sae, DATASET, ROOT / 'data', domain=cfg.vdom)
    meta = src.meta
    fp = meta['frame_path'].values
    with_exp(cfg.view, meta)  # frogs: experiment = VIEW 'exp'
    # histogram groups: 'genotype|stage' (mice), 'experiment|T' (ants), 'experiment|group' (frogs)
    g0, g1 = GROUP_COLS[cfg.dom.name]
    gv = (lambda c: c.astype(str)) if cfg.view.get('arm') else (lambda c: c.astype(int).astype(str))
    # frames of the kept videos (VIEW 'keep'): histogram, firing rate, bar scale and tables use only these
    fkeep = keep_mask(cfg.view, meta) if cfg.view['keep'] is not None else slice(None)
    gkeys = sorted({f'{g}|{s}' for g, s in zip(meta[g0][fkeep], gv(meta[g1][fkeep]))})
    gid = pd.Index(gkeys).get_indexer(meta[g0].astype(str) + '|' + gv(meta[g1]))
    obs, fidx = meta['observation_id'].values, meta['frame_idx'].values
    if cfg.dom is MICE:
        stg, gen, pool = meta['stage'].values, meta['genotype'].values, meta['pool'].astype(str).values
    else:  # ants: treatment T and recording day; frogs: group and recording session (VIEW 'arm', 'day')
        exp_, trt, day = (meta['experiment'].astype(str).values, meta[cfg.view.get('arm', 'T')].values,
                          meta[cfg.view.get('day', 'recording_date')].astype(str).values)
    for key, miss in missing.items():
        if not miss:
            continue
        print(f'[{cfg.tag}] data: reading codes_{key} of neurons', miss, flush=True)
        neurons = np.array(miss, dtype=np.int64)
        Z = codes_array(cfg, src, key)
        X = np.empty((len(meta), len(neurons)), np.float32)  # every frame of the selected neurons
        for a in range(0, len(meta), 200_000):
            X[a:a + 200_000] = Z[a:a + 200_000][:, neurons]
        loc = CodeSource('sel', {key: X}, meta, DATASET, dict(src.info))
        wss = scan_windows(loc, np.arange(len(neurons)), key, tuple(LENGTHS.values()), seed=0)
        for ws in wss.values():
            ws.neurons = neurons  # column i of X is neuron neurons[i] (seeds use the neuron id)
        vids = next(iter(wss.values())).videos
        vmean = np.stack([X[lo:hi].mean(0) for lo, hi in zip(vids['lo'], vids['hi'])])
        # 'all videos' clips: among the kept videos only (every video for mice)
        vkeep = keep_mask(cfg.view, vids) if cfg.view['keep'] is not None else np.ones(len(vids), bool)
        wall = {w: (ws if vkeep.all() else subset_windows(ws, vkeep)) for w, ws in wss.items()}
        kobs = set(vids['observation_id'][vkeep])
        store = res['by_key'].setdefault(key, {})
        for i, j in enumerate(miss):
            def clip(start, w):
                tr = X[start:start + w, i]
                c = {'start': int(start), 'rows': list(range(int(start), int(start) + w)),
                     'act': float(tr.mean()), 'trace': [float(x) for x in tr], 'obs': str(obs[start]),
                     'frame': int(fidx[start])}
                if cfg.dom is MICE:
                    c.update(stage=int(stg[start]), genotype=str(gen[start]), pool=str(pool[start]))
                else:
                    c.update(experiment=exp_[start], T=arm_of(cfg.view, trt[start]), day=day_label(day[start]))
                return c
            clips = {'all': {}}
            for aid in want[key][j]:
                clips[aid] = {}
            for L, w in LENGTHS.items():
                ts, _ = pick_top_windows(wall[w], i, K)
                ls, rule = pick_least_windows(wall[w], i, K, seed=0)
                clips['all'][L] = {'top': [clip(s, w) for s in ts], 'least': [clip(s, w) for s in ls],
                                   'least_rule': rule}
                assert all(x['obs'] in kobs for k in ('top', 'least') for x in clips['all'][L][k])
                for aid in want[key][j]:  # the same rules, among the videos of that contrast only
                    m0, m1 = contrast_groups(cfg, aid, vids)
                    ts, _ = pick_top_windows(subset_windows(wss[w], m0 | m1), i, K)  # one row: both groups
                    c = {'top': [clip(s, w) for s in ts]}
                    assert all(x['obs'] in set(vids['observation_id'][m0 | m1]) for x in c['top'])
                    ls, rule = pick_least_windows(subset_windows(wss[w], m0 | m1), i, K, seed=0)
                    c['least'], c['least_rule'] = [clip(s, w) for s in ls], rule
                    assert all(x['obs'] in set(vids['observation_id'][m0 | m1]) for x in c['least'])
                    clips[aid][L] = c
                for sname, cs in clips.items():
                    c = cs[L]
                    assert all(len({x['obs'] for x in c[k]}) == len(c[k]) for k in kinds(sname))
                    if c['least_rule'] == 'silent':
                        assert all(max(x['trace']) == 0 for x in c['least'])
            xi = X[fkeep, i]
            edges, counts = activation_hist(xi, gid[fkeep], len(gkeys))
            if cfg.dom is MICE:
                tab = stage_genotype_table(SimpleNamespace(videos=vids, video_mean=vmean), i).to_dict('records')
            else:
                tab = group_table(vids[vkeep], vmean[vkeep, i], cfg.view)
            store[str(j)] = {
                'v': PICKS_V, 'vset': vset, 'clips': clips,
                'firing_rate': float((xi > 0).mean()),
                'max_frame': float(xi.max()),
                'hist': {'edges': [float(f'{e:.5g}') for e in edges],
                         'counts': dict({'all': [int(c) for c in counts.sum(0)]},
                                        **{g: [int(c) for c in counts[k]] for k, g in enumerate(gkeys)})},
                'table': [{k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if k in ('stage', 'n_pools')
                               else v) for k, v in rec.items()} for rec in tab]}
            print(f'  [{cfg.tag}] codes_{key} neuron {j}: ' + ', '.join(
                f'{L} top {len(c["top"])} (mean {c["top"][0]["act"]:.3g}..{c["top"][-1]["act"]:.3g}) '
                f'least {len(c["least"])} {c["least_rule"]} (max {max(max(x["trace"]) for x in c["least"]):.3g})'
                for L, c in clips['all'].items() if c['top']), flush=True)
            for aid in want[key][j]:
                print(f'      {aid}: ' + ', '.join(
                    f'{L} top {len(c["top"])} (' + ', '.join(f'{g} {n}' for g, n in sorted(__import__('collections').Counter(clip_group(x) for x in c['top']).items())) + f') least {len(c["least"])} {c["least_rule"]}'
                    for L, c in clips[aid].items()), flush=True)
        del X
    res['frame_path'] = {str(r): fp[r] for r in sorted({r for st in res['by_key'].values() for n in st.values()
                                                        for rr in clip_rows(n).values() for r in rr})}
    cfg.work.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(f'[{cfg.tag}] data: wrote', out, {k: len(v) for k, v in res['by_key'].items()}, 'neurons')


# ---------------------------------------------------------------------- step: patch
def patch_file(cfg, key, j):
    return cfg.work / 'patch' / f'{pre(key)}{int(j):04d}.npz'


def clip_rows(n, kr=K, lengths=None):
    """{'<source>__<L>__<kind>': rows of the first kr clips (the rendered ones), in clip order} of one neuron
    (source = 'all' or a contrast id), every picked length or only those in lengths."""
    return {f'{src}__{L}__{kind}': [r for x in c[kind][:kr] for r in x['rows']] for src, cs in n['clips'].items()
            for L, c in cs.items() if lengths is None or L in lengths for kind in kinds(src)}


def motion_delta(cfg):
    """D of a mask + motion SAE (SAE input [token_t, token_t - token_(t-D)], checkpoint 'motion_delta'), else 0."""
    k = f'md_{cfg.sae}'
    if k not in _rates:
        import torch
        ck = torch.load(cfg.dom.eci_dir / 'sae' / cfg.sae / 'sae.pt', map_location='cpu', weights_only=False)
        _rates[k] = int(ck.get('motion_delta', 0) or 0)
    return _rates[k]


def change_share(cfg):
    """Mask + motion SAE: per neuron, the share of its decoder row's squared norm on the 'change' half of the input
    ([token_t, token_t - token_(t-D)], both halves scaled equally by the SAE's block norm), in [0, 1]; None for
    other SAEs. 0 = the neuron reconstructs only the static token, 1 = only the change."""
    if not motion_delta(cfg):
        return None
    from src.eci.sae import load_sae
    sae, _, _ = load_sae(cfg.dom.eci_dir / 'sae' / cfg.sae / 'sae.pt', 'cpu')
    W = sae.W_dec.detach().double().numpy()
    d = W.shape[1] // 2
    return (W[:, d:] ** 2).sum(1) / (W ** 2).sum(1)


class PatchEncoderMotion:
    """Per-patch codes of a mask + motion SAE (fg448 geometry): as src/eci/viz.py PatchEncoderFG, but the SAE input
    of a foreground patch is [token_t, token_t - token_(t-D)] (the same patch D rows earlier in the same video,
    clipped to its first frame), exactly as src/eci/fg_encode.py encodes the full codes (its _Runner)."""

    rep = 'fg448'

    def __init__(self, cfg):
        from src.eci.fg_encode import _Runner
        from src.eci.foreground import GRID
        cc = json.loads((cfg.dom.eci_dir / 'codes' / cfg.sae / 'config.json').read_text())
        fp = pd.read_csv(cfg.dom.ann_path, usecols=['frame_path'])['frame_path'].values
        self.run = _Runner([cfg.dom.eci_dir / 'sae' / cfg.sae / 'sae.pt'], cc['backgrounds'], cfg.dom.ann_path,
                           'cuda', fp, DATASET)
        self.D, self.grid = self.run.deltas[0], GRID
        assert self.D > 0, f'{cfg.sae}: not a motion SAE'

    def patch_codes(self, paths, neurons, rows, batch_size=32, num_workers=8, return_mask=False):
        """paths / rows: the frames (annotations.csv rows). Each frame is loaded next to its frame D rows earlier
        (same video, clipped to its first row; src/eci/fg_encode.py _Runner._prev_tokens), both through the
        DataLoader workers."""
        import torch
        from src.eci.foreground import encode_batch
        run, (sae, norm) = self.run, self.run.saes[0]
        rows = np.asarray(rows)
        prow = np.maximum(rows - self.D, run.bgs.starts[run.bgs.obs_index(rows)])
        pp = [str(DATASET / run.frame_paths[int(x)]) for x in prow]
        two = [x for pair in zip(list(paths), pp) for x in pair]  # frame, its earlier frame, frame, ...
        two_rows = np.stack([rows, prow], 1).reshape(-1)
        nsel = torch.as_tensor(np.asarray(neurons), device=run.device)
        out, masks = [], []
        with torch.no_grad():
            for pix, grey, r, *pix2 in run.loader(two, two_rows, 2 * batch_size, num_workers):
                assert not pix2, 'motion SAEs take DINOv2 tokens (the mask encoder) only'
                r = r.numpy()[0::2]
                tok2 = encode_batch(run.model, pix, run.device)
                tok, prev = tok2[0::2], tok2[1::2]
                mask, _ = run.bgs.mask(tok, grey[0::2].to(run.device), r)
                B, P, _ = tok.shape
                fi, pi = torch.nonzero(mask, as_tuple=True)
                x = tok[fi, pi]
                x = torch.cat([x, (x.float() - prev[fi, pi].float()).half()], 1)
                z = sae.encode(norm(x), mode='threshold')[:, nsel]
                full = torch.zeros(B, P, len(nsel), device=run.device)
                full[fi, pi] = z
                out.append(full.view(B, self.grid, self.grid, -1).cpu().numpy())
                masks.append(mask.view(B, self.grid, self.grid).cpu().numpy())
        pc = np.concatenate(out)
        return (pc, np.concatenate(masks)) if return_mask else pc


PATCH_SHARD = (0, 1)  # --patch-shard


def step_patch(cfg, overwrite, chunk=6):
    picks = json.loads((cfg.work / 'picks.json').read_text())
    (cfg.work / 'patch').mkdir(parents=True, exist_ok=True)
    todo = []
    for k in cfg.keys:
        for j, n in picks['by_key'].get(k, {}).items():
            f, rr = patch_file(cfg, k, j), clip_rows(n, cfg.kr, SHOW_LENGTHS)
            sig = hashlib.md5(json.dumps(rr).encode()).hexdigest()
            # a file made for all K clips or for every picked length (3 s included) also serves (its first kr
            # clips of the shown lengths are the rendered ones)
            ok = {sig} | {hashlib.md5(json.dumps(clip_rows(n, kk, ls)).encode()).hexdigest()
                          for kk in (cfg.kr, K) for ls in (None, SHOW_LENGTHS)}
            if overwrite or not f.exists() or str(np.load(f)['sig']) not in ok:
                todo.append((k, int(j), rr, sig))
    if PATCH_SHARD[1] > 1:  # --patch-shard i/n: every n-th neuron to do from the i-th (parallel GPU jobs)
        todo = todo[PATCH_SHARD[0]::PATCH_SHARD[1]]
    if not todo:
        print(f'[{cfg.tag}] patch: cached')
        return
    from src.eci.viz import load_full_codes, make_patch_encoder
    print(f'[{cfg.tag}] patch: {len(todo)} neurons, '
          f'{sum(len(set(r for v in rr.values() for r in v)) for _, _, rr, _ in todo)} frame-neuron pairs '
          f'through DINOv2 + SAE ({cfg.rep})', flush=True)
    pe = PatchEncoderMotion(cfg) if motion_delta(cfg) else make_patch_encoder(cfg.sae, dataset_dir=DATASET, domain=cfg.vdom)
    codes = load_full_codes(cfg.sae, DATASET, ROOT / 'data', domain=cfg.vdom).codes
    worst, mism, npair = {}, {}, 0
    for c0 in range(0, len(todo), chunk):
        part = todo[c0:c0 + chunk]
        js = [j for _, j, _, _ in part]
        allr = sorted({r for _, _, rr, _ in part for v in rr.values() for r in v})
        paths = [str(DATASET / picks['frame_path'][str(r)]) for r in allr]
        if pe.rep == 'fg448':
            pc, fg = pe.patch_codes(paths, np.array(js), np.array(allr), return_mask=True)
            n_fg = fg.reshape(len(allr), -1).sum(1)
        else:
            pc, n_fg = pe.patch_codes(paths, np.array(js)), None
        pos = {r: i for i, r in enumerate(allr)}
        for ci, (k, j, rr, sig) in enumerate(part):
            ur = sorted({r for v in rr.values() for r in v})
            ii = np.array([pos[r] for r in ur])
            m = pc[ii, :, :, ci]
            if k in ('max', 'mean'):  # SOMP (and other non-pooled codes): heat = the neuron's patch codes, no check
                stored = np.asarray(codes[k][np.array(ur)][:, j], dtype=np.float32)
                if k == 'max':
                    pooled = m.max((1, 2))
                elif n_fg is None:
                    pooled = m.mean((1, 2))
                else:  # fg448: mean over the foreground patches only
                    pooled = m.sum((1, 2)) / np.maximum(n_fg[ii], 1)
                worst[k] = max(worst.get(k, 0.0), float(np.abs(pooled - stored).max()))
                mism[k] = mism.get(k, 0) + int(((pooled > 0) != (stored > 0)).sum())
                npair += len(ur)
            posv = m[m > 0]
            vmax = float(np.quantile(posv, 0.99)) if len(posv) else 1.0
            lp = {r: t for t, r in enumerate(ur)}
            np.savez(patch_file(cfg, k, j), sig=sig, vmax=vmax,
                     **{name: m[[lp[r] for r in v]].astype(np.float16) for name, v in rr.items()})
        print(f'  [{cfg.tag}] patch: {c0 + len(part)}/{len(todo)} neurons', flush=True)
    print(f'[{cfg.tag}] patch: recomputed patch maps pooled vs stored codes, max abs diff {worst}, '
          f'active/inactive mismatches {mism} (of {npair} frame-neuron pairs)')


# ---------------------------------------------------------------------- step: arena
def bg_name(cfg):
    return f'{cfg.tag}_arena_bg.webp'


def step_arena(cfg, overwrite):
    """Per patch position: summed SAE codes of every neuron and foreground counts over the SAE's
    training frames (fg448: FgTokenStore shards; crop224: patch_tokens.npy, every patch counts)."""
    import torch
    from src.eci.sae import load_sae
    out = cfg.work / 'arena.npz'
    bg_out = cfg.assets / bg_name(cfg)
    sae_dir = cfg.dom.eci_dir / 'sae' / cfg.sae
    tokens_dir = Path(json.loads((sae_dir / 'metrics.json').read_text())['args']['tokens_dir'])
    if not out.exists() or overwrite:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sae, norm, _ = load_sae(sae_dir / 'sae.pt', dev)
        chk = []
        if cfg.rep == 'fg448':
            codes = __import__('src.eci.viz', fromlist=['load_full_codes']).load_full_codes(
                cfg.sae, DATASET, ROOT / 'data', domain=cfg.vdom).codes['max']
            from src.eci.foreground import GRID, FgTokenStore
            st = FgTokenStore(tokens_dir)
            D = motion_delta(cfg)  # mask + motion SAE: input [token, token - the same patch D frames earlier]
            if D and int(st.info[0].get('motion_delta', 0) or 0) != D:
                raise SystemExit(f'{tokens_dir}: token store motion_delta {st.info[0].get("motion_delta")} != SAE {D}')
            P, n_frames = GRID * GRID, int(sum(i['n_frames'] for i in st.info))
            S = torch.zeros(P, sae.n_latents, dtype=torch.float64, device=dev)
            occ = torch.zeros(P, dtype=torch.float64, device=dev)
            for s in range(len(st.dirs)):
                tok, pos, row = st.tokens(s), torch.from_numpy(st.pos(s).astype(np.int64)), st.row(s)
                prv = st.prev(s) if D else None
                for a in range(0, len(pos), 250_000):
                    with torch.no_grad():
                        x = torch.from_numpy(np.array(tok[a:a + 250_000])).to(dev)
                        if D:  # as scripts/eci/train_sae_fg.py load_split (float32 difference rounded to fp16)
                            x = torch.cat([x, (x.float() - torch.from_numpy(np.array(prv[a:a + 250_000])).to(dev).float()).half()], 1)
                        z = sae.encode(norm(x), mode='threshold')
                        p = pos[a:a + 250_000].to(dev)
                        S.index_add_(0, p, z.double())
                        occ.index_add_(0, p, torch.ones_like(p, dtype=torch.float64))
                        if s == 0 and a == 0:  # check: per-frame max over positions = stored codes_max
                            r = torch.from_numpy(row[:len(x)].astype(np.int64)).to(dev)
                            u, inv = torch.unique(r, return_inverse=True)
                            mx = torch.zeros(len(u), z.shape[1], device=dev).scatter_reduce(
                                0, inv[:, None].expand_as(z), z, 'amax', include_self=True)
                            keep = slice(0, len(u) - 1)  # the last frame may continue in the next chunk
                            ref = np.asarray(codes[u[keep].cpu().numpy()], dtype=np.float32)
                            chk = [float(np.abs(mx[keep].cpu().numpy() - ref).max()), len(u) - 1]
                print(f'[{cfg.tag}] arena: shard {s + 1}/{len(st.dirs)}', flush=True)
            n_vid = len({o for i in st.info for o in i['observations']})
            stride = st.info[0].get('stride')
            note = (f'the {n_frames:,} frames of the SAE training sample (every {stride}th frame, i.e. 1 per '
                    f'second, of all {n_vid} videos)' if stride else f'the {n_frames:,} SAE training frames')
            grid = GRID
        else:
            tok = np.load(tokens_dir / 'patch_tokens.npy', mmap_mode='r')
            n_frames, P = tok.shape[0], tok.shape[1]
            grid = int(round(P ** 0.5))
            S = torch.zeros(P, sae.n_latents, dtype=torch.float64, device=dev)
            for a in range(0, n_frames, 1024):
                with torch.no_grad():
                    x = torch.from_numpy(np.array(tok[a:a + 1024])).to(dev)
                    z = sae.encode(norm(x.reshape(-1, x.shape[-1])), mode='threshold').view(len(x), P, -1)
                    S += z.double().sum(0)
                if a % 51200 == 0:
                    print(f'[{cfg.tag}] arena: {a}/{n_frames} frames', flush=True)
            occ = None
            note = f'the {n_frames:,} frames of the SAE training sample'
        np.savez(out, act=(S / n_frames).float().cpu().numpy(), grid=grid, n_frames=n_frames, note=note,
                 occ=(occ / n_frames).float().cpu().numpy() if occ is not None else np.zeros(0))
        print(f'[{cfg.tag}] arena: {n_frames} frames, grid {grid}; check per-frame max vs stored codes_max: '
              f'max abs diff {chk[0] if chk else "n/a"} over {chk[1] if chk else 0} frames', flush=True)
    else:
        print(f'[{cfg.tag}] arena: cached')
    if not bg_out.exists() or overwrite:  # median over videos of the empty-bedding pixel background
        cfg.assets.mkdir(parents=True, exist_ok=True)
        cc = cfg.dom.eci_dir / 'codes' / cfg.sae / 'config.json'
        bgd = Path(json.loads(cc.read_text()).get('backgrounds', '')) if cc.exists() else None
        if bgd is not None and bgd.is_dir() and str(bgd) != '.':
            zs = [np.load(f) for f in sorted(bgd.glob('*.npz'))]
            bad = [i for i, z in enumerate(zs) if (str(z['align']) if 'align' in z.files else 'none') != cfg.align]
            if bad:  # aligned SAE: the arena image must be the aligned (turned) backgrounds, and vice versa
                raise SystemExit(f'{bgd}: {len(bad)} backgrounds whose align differs from the SAE ({cfg.align})')
            ims = np.stack([z['pix_bg'] for z in zs])
            im = np.median(ims, 0).astype(np.uint8)
            src = f'median of {len(ims)} per-video empty-bedding backgrounds'
        else:  # no stored backgrounds (crop224): median of one frame per video
            meta = pd.read_csv(cfg.dom.ann_path, usecols=['observation_id', 'frame_path'])
            fr = meta.groupby('observation_id')['frame_path'].nth(0).values
            im = np.median(np.stack([np.asarray(Image.open(DATASET / f).convert('L')) for f in fr]), 0).astype(np.uint8)
            src = f'median of the first frame of {len(fr)} videos'
        from src.eci.viz import frame_box
        x0, y0, c = frame_box(cfg.rep, im.shape[1], im.shape[0])
        im = Image.fromarray(im).crop((int(round(x0)), int(round(y0)), int(round(x0 + c)), int(round(y0 + c))))
        im.resize((384, 384), Image.BILINEAR).save(bg_out, 'WEBP', quality=80)
        print(f'[{cfg.tag}] arena: wrote', bg_out, src)


# ---------------------------------------------------------------------- step: render
_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT, size)
    return _fonts[k]


def tile_text(c):
    """Label burnt into a clip's tiles: 'S2 O,S · het · rd11_2 Test' (mice), 'v3 t=8 · day C · 3_21_4' (ants),
    'FoxP1_153_2' (frogs: the video name = group, session, dish slot)."""
    if 'stage' in c:
        return f'S{c["stage"]} {STAGE_LABEL[c["stage"]]} · {c["genotype"]} · {short_obs(c["obs"])[0]}'
    if isinstance(c['T'], str):
        return c['obs']
    return f'{c["experiment"]} t={c["T"]} · day {c["day"]} · {c["obs"]}'


def crop_origin(maps, rep, W, H, crop):
    """(x0, y0) of the crop x crop square (source px) centred on the activation peak of one clip: the argmax
    of its patch codes summed over the clip's frames (Gaussian-smoothed, sigma 1 patch), mapped through the
    patch grid's frame box and clamped inside the frame. None when the clip has no activation."""
    from scipy.ndimage import gaussian_filter
    from src.eci.viz import frame_box
    acc = gaussian_filter(np.maximum(maps.astype(np.float32), 0).sum(0), 1.0)
    if not acc.max() > 0:
        return None
    iy, ix = np.unravel_index(int(np.argmax(acc)), acc.shape)
    bx, by, bc = frame_box(rep, W, H)
    cx, cy = bx + (ix + 0.5) * bc / acc.shape[1], by + (iy + 0.5) * bc / acc.shape[0]
    return (int(round(min(max(cx - crop / 2, 0), W - crop))), int(round(min(max(cy - crop / 2, 0), H - crop))))


def tile_frames(c, frame_path, vmax, color, rep, tile, maps=None, hvmax=None, origin=None, crop=None, rot=None):
    """len(c['rows']) labelled tile x tile frames of one clip; a bar at the bottom shows the activation
    of the current frame relative to vmax. maps: (w, grid, grid) patch codes -> heatmap overlaid (scale hvmax).
    origin / crop: the clip is cut to the crop x crop square at origin (source px, crop_origin), else the
    whole frame is shown. rot: {row: 90-degree CCW turns} of an aligned SAE: each frame is turned first (its patch
    codes are aligned-frame positions)."""
    from src.eci.foreground import rotate_image
    from src.eci.viz import _overlay_rgb, frame_box
    import matplotlib
    matplotlib.use('Agg')
    cmap = matplotlib.colormaps['turbo']
    frames = []
    fs, hh = (8, 12) if tile <= 144 else (int(round(8 * tile / 144)), int(round(12 * tile / 144)))
    for t, r in enumerate(c['rows']):
        im = Image.open(DATASET / frame_path[str(r)]).convert('RGB')
        if rot is not None:
            im = rotate_image(im, rot[str(r)])
        if maps is not None:
            a = np.asarray(im)
            im = Image.fromarray(_overlay_rgb(a, maps[t].astype(np.float32), frame_box(rep, a.shape[1], a.shape[0]),
                                              hvmax, cmap))
        if origin is not None:
            im = im.crop((origin[0], origin[1], origin[0] + crop, origin[1] + crop))
            if crop != tile:
                im = im.resize((tile, tile), Image.LANCZOS)
        elif im.size != (tile, tile):
            im = im.resize((tile, tile), Image.BILINEAR if tile <= 144 else Image.LANCZOS)
        d = ImageDraw.Draw(im, 'RGBA')
        d.rectangle([0, 0, tile, hh], fill=(0, 0, 0, 140))
        d.text((3, 1) if tile <= 144 else (5, 2), tile_text(c), font=font(fs), fill=(255, 255, 255))
        d.rectangle([0, tile - 6, tile, tile], fill=(0, 0, 0, 255))  # opaque: the page re-reads this bar (row T-4)
        w = int(round(tile * min(max(c['trace'][t], 0) / vmax, 1))) if vmax > 0 else 0
        if w:
            d.rectangle([0, tile - 5, w, tile], fill=color)
        frames.append(np.asarray(im))
    return frames


def encode(frames, path, crf=30):
    """frames: list of HxWx3 uint8 arrays -> one H.264 segment at 5 fps, one closed GOP (the first frame is
    an IDR frame), so segments can be joined without re-encoding (packs) and each starts decodable."""
    h, w = frames[0].shape[:2]
    n = len(frames)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
           '-r', '5', '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
           '-x264-params', f'keyint={n}:min-keyint={n}:scenecut=0', '-an', str(path)]
    subprocess.run(cmd, input=b''.join(f.tobytes() for f in frames), check=True)


RENDER_V = 2  # page rendering version (part of the segment signature)


def page_list(cfg, picks):
    """The clip pages of one SAE, in pack order: [(page id '<key>|<neuron>|<source>', key, neuron, source)];
    per codes key and neuron (ascending) its sources 'all' then its contrasts (cfg.order)."""
    out = []
    for key in cfg.keys:
        for j in sorted(picks['by_key'].get(key, {}), key=int):
            n = picks['by_key'][key][j]
            for src in ['all'] + sorted((x for x in n['clips'] if x != 'all'), key=cfg.order.index):
                out.append((f'{key}|{int(j)}|{src}', key, int(j), src))
    return out


def _render_page(task):
    """task: (segment path, blocks [(source, block name, kind, heat, clips, bar vmax, patch file)], frame paths,
    rep, L, crf, (tile, cols, kr, crop)[, rot]): one page = BLOCKS of kr tiles, cols per row, encoded as one segment.
    Crop (top rows only): each clip cut around its own activation peak (the same square raw and with heat).
    rot (aligned SAEs): {row: 90-degree CCW turns}, every frame is turned before drawing."""
    out, blocks, fp, rep, L, crf, (tile, cols, kr, crop), *rest = task
    rot = rest[0] if rest else None
    w = LENGTHS[L]
    pzs = {}
    blank = [np.zeros((tile, tile, 3), np.uint8)] * w
    tiles = []
    for src, _, kind, heat, clips, vmax, maps_file in blocks:
        if heat is None:  # least heat of silent clips: identical to the raw block, which the page shows instead
            tiles += [blank] * kr
            continue
        if maps_file not in pzs:
            pzs[maps_file] = np.load(maps_file)
        pz = pzs[maps_file]
        hv = float(pz['vmax'])
        allm = pz[f'{src}__{L}__{kind}'].astype(np.float32)
        color = (120, 190, 255) if kind == 'least' else (255, 170, 40)
        cl = []
        for k, c in enumerate(clips[:kr]):
            m = allm[k * w:(k + 1) * w]
            org = None
            if crop and kind == 'top':
                im0 = Image.open(DATASET / fp[str(c['rows'][0])])
                org = crop_origin(m, rep, im0.size[0], im0.size[1], crop)
            cl.append(tile_frames(c, fp, vmax, color, rep, tile, m if heat else None, hv, org, crop, rot))
        tiles += cl + [blank] * (kr - len(cl))
    rows = [tiles[r:r + cols] for r in range(0, len(tiles), cols)]
    frames = [np.concatenate([np.concatenate([clip[t] for clip in row], 1) for row in rows], 0) for t in range(w)]
    tmp = out.with_suffix('.tmp.mp4')
    encode(frames, tmp, crf)
    tmp.rename(out)
    return out


def pack_name(cfg, L, segs):
    return f'{cfg.tag}_{L}_{hashlib.md5("|".join(segs).encode()).hexdigest()[:10]}.mp4'


def step_render(cfg, overwrite, pack_mb=8.0):
    """Every clip page of the SAE (page_list) at every length, as one cached segment each
    (<work>/seg/<L>/<signature md5>.mp4, reused while its content is unchanged), then joined without
    re-encoding into packs of at most pack_mb MB (assets/<tag>_<L>_<md5 of the members>.mp4): page p of a pack
    is frames [p w, (p + 1) w) (w = frames per clip). -> <work>/packs.json {L: [{name, pages: [page id]}]}."""
    import multiprocessing as mp
    import os
    picks = json.loads((cfg.work / 'picks.json').read_text())
    fp = picks['frame_path']
    rot = None if cfg.rot is None else {r: int(cfg.rot[int(r)]) for r in fp}  # aligned SAE: turns per clip row
    cfg.assets.mkdir(parents=True, exist_ok=True)
    pages = page_list(cfg, picks)
    geo = (cfg.tile, cfg.cols, cfg.kr, cfg.crop)
    vm, hv, pf = {}, {}, {}
    for key in cfg.keys:  # bar scale: the neuron's highest frame; heat scale: its patch file
        for j, n in picks['by_key'].get(key, {}).items():
            vm[(key, int(j))] = n['max_frame'] or 1.0
            pf[(key, int(j))] = patch_file(cfg, key, j)
            hv[(key, int(j))] = float(np.load(pf[(key, int(j))])['vmax'])
    tasks, segs = [], {L: [] for L in SHOW_LENGTHS}
    for pid, key, j, src in pages:
        n = picks['by_key'][key][str(j)]
        for L in SHOW_LENGTHS:
            # silent least clips of pooled codes have all patch codes 0 (heat = raw): the block is left blank
            silent = n['clips'][src][L]['least_rule'] == 'silent' and key in ('max', 'mean')
            blocks = [(src, bn, bn.split('_')[0], None if bn == 'least_heat' and silent else bn.endswith('_heat'),
                       n['clips'][src][L][bn.split('_')[0]], vm[(key, j)], str(pf[(key, j)])) for bn in BLOCKS]
            sig = json.dumps([RENDER_V, cfg.rep, geo, cfg.crf, vm[(key, j)], hv[(key, j)],
                              [(bn, ht, [x['start'] for x in c[:cfg.kr]]) for _, bn, _, ht, c, _, _ in blocks]]
                             + ([f'align={cfg.align}'] if cfg.rot is not None else []))
            h = hashlib.md5(sig.encode()).hexdigest()
            out = cfg.work / 'seg' / L / f'{h}.mp4'
            segs[L].append((pid, h, out))
            if overwrite or not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                tasks.append((out, blocks, fp, cfg.rep, L, cfg.crf, geo) + ((rot,) if rot is not None else ()))
    tasks = list({str(t[0]): t for t in tasks}.values())
    if tasks:
        nproc = max(1, min(len(tasks), len(os.sched_getaffinity(0))))
        print(f'[{cfg.tag}] render: {len(tasks)} clip pages x length on {nproc} processes', flush=True)
        with mp.get_context('fork').Pool(nproc) as pool:
            for i, out in enumerate(pool.imap_unordered(_render_page, tasks)):
                if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                    print(f'[{cfg.tag}] render: {i + 1}/{len(tasks)}', flush=True)
    else:
        print(f'[{cfg.tag}] render: segments cached ({sum(len(v) for v in segs.values())})')
    # Sticky packing: a pack of the previous build (<work>/packs.json) whose pages all still exist with the same
    # content (pack name = md5 of its member segments, recomputed from the current segments) is kept as it is, so
    # its uploaded asset stays valid; only the other pages are packed anew (page_list order).
    pf_ = cfg.work / 'packs.json'
    old = json.loads(pf_.read_text()) if pf_.exists() and not overwrite else {}
    packs, kept = {}, 0
    for L, ss in segs.items():
        byp = {pid: (pid, h, out) for pid, h, out in ss}
        reuse = []
        for pk in old.get(L, []):
            g = [byp.get(pid) for pid in pk['pages']]
            if all(g) and pack_name(cfg, L, [h for _, h, _ in g]) == pk['name']:
                reuse.append(g)
        used = {pid for g in reuse for pid, _, _ in g}
        kept += len(reuse)
        groups, cur, size = list(reuse), [], 0
        for x in (x for x in ss if x[0] not in used):
            b = x[2].stat().st_size
            if cur and size + b > pack_mb * 1e6:
                groups.append(cur)
                cur, size = [], 0
            cur.append(x)
            size += b
        if cur:
            groups.append(cur)
        packs[L] = []
        for g in groups:
            name = pack_name(cfg, L, [h for _, h, _ in g])
            out = cfg.assets / name
            if overwrite or not out.exists():
                lst = cfg.work / 'seg' / f'{name}.txt'
                lst.write_text(''.join(f"file '{p}'\n" for _, _, p in g))
                tmp = out.with_suffix('.tmp.mp4')
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(lst),
                                '-c', 'copy', '-movflags', '+faststart', str(tmp)], check=True)
                n = int(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                                        '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', str(tmp)],
                                       capture_output=True, text=True, check=True).stdout.strip())
                if n != len(g) * LENGTHS[L]:
                    raise SystemExit(f'{tmp}: {n} frames, expected {len(g)} pages x {LENGTHS[L]}')
                tmp.rename(out)
                lst.unlink()
            packs[L].append({'name': name, 'pages': [pid for pid, _, _ in g]})
        print(f'[{cfg.tag}] render: {L}: {len(ss)} pages in {len(groups)} packs ({len(reuse)} kept from the previous '
              f'build), {sum((cfg.assets / p["name"]).stat().st_size for p in packs[L]) / 1e6:.1f} MB')
    (cfg.work / 'packs.json').write_text(json.dumps(packs))


# ---------------------------------------------------------------------- step: page
def chip_name(f, v):
    if f == 'threshold_q':
        return f'q{v:.2f}', f'activity threshold at the {v:.2f} quantile'
    if f == 'merge_gap':
        return f'gap{int(v)}', f'bouts separated by <= {int(v)} frames merged'
    if f == 'bout_rule':
        return (('min2+hyst', 'hysteresis bouts: start above the q0.95 threshold, end at or below the q0.90 '
                 'threshold, at least 2 frames long') if int(v) == 1 else ('plain', 'plain threshold runs'))
    return CHIP.get((f, v), (str(v), f'{f} = {v}'))


# plain-words robustness checks (robustness): setting change -> (key, text after "still found ..." / "not found ...")
CHECK_TEXT = {('test', 'signflip'): ('flip', 'with a permutation (sign-flip) test instead of the t-test'),
              ('correction', 'bh'): ('BH', 'with Benjamini-Hochberg instead of Bonferroni'),
              ('window', 'trim30'): ('trim30', 'when the first 30 s of every video are dropped'),
              ('window', 'matched'): ('match', 'with time-matched windows'),
              ('window', 'full'): ('full', 'on the full videos instead of the common window'),
              ('window', 'common'): ('common', 'on the common window (first W seconds of every video)'),
              ('outcome_type', 'rate'): ('rate', 'when the outcome is the firing rate (share of active frames)'),
              ('outcome_type', 'mean'): ('mean', 'when the outcome is the per-video mean activation'),
              ('transform', 'rank'): ('rank', 'with ranks (a Mann-Whitney-like test) instead of raw seconds')}


def robustness(o, others, aid, prefix, window, neuron, words=('mouse', 'mice'), size_text=None):
    """Robustness checks of one discovered neuron, in plain words: [[key, text, 'Y' / 'N' / '-', note], ...] where
    Y = still selected (any round) when ONE setting changes (or by another outcome's primary search), N = not, '-' =
    not run. The bout threshold quantiles are one check (Y only when every quantile passes; note = the partial result),
    likewise the bout definition (merge gap, hysteresis bouts). Prefix and pooling changes are not checks here (the
    page has its own selectors for them). The size check (size_adjusted.csv) belongs to the core outcome's primary
    pooling: none for extra outcomes."""
    t = o['tidy']
    t = t[t['analysis_id'] == aid]
    base = dict(o['primary'], window=window, prefix=prefix)
    alts = []
    for f in list(FIELDS) + extra_cols(t) + ['window']:
        if f not in t.columns or f == 'pooling':
            continue
        if f in KNOWN:
            alts += [(f, v) for v in sorted(t[f].astype(str).unique()) if v != str(base.get(f))]
        else:
            alts += [(f, float(v)) for v in sorted(t[f].dropna().unique()) if f in base and not np.isclose(v, base[f])]
    val_of = {}
    for f, v in alts:
        g = t[match(t, dict(base, **{f: v}))]
        val_of[(f, v)] = ('Y' if neuron in set(g['neuron'].dropna().astype(int)) else 'N') if len(g) else '-'
    out = []
    for (f, v), val in val_of.items():
        if f == 'outcome_type' and any(o2['primary']['outcome_type'] == v for o2 in others):
            continue  # the same question as the cross-outcome check below
        if (f, v) in CHECK_TEXT:
            out.append([*CHECK_TEXT[(f, v)], val, ''])
        elif f in KNOWN:
            out.append([f'{f}={v}', f'with {f} = {v}', val, ''])
    for key, fields, text in (('thr', ('threshold_q',), 'with the bout threshold at the other quantiles ({})'),
                              ('bout', ('merge_gap', 'bout_rule'), 'with the other bout definitions ({})')):
        parts = [(chip_name(f, v)[0], val) for (f, v), val in val_of.items() if f in fields]
        run = [(n, x) for n, x in parts if x != '-']
        if not parts:
            continue
        val = '-' if not run else 'Y' if all(x == 'Y' for _, x in run) else 'N'
        note = ', '.join(f'{n} {"✓" if x == "Y" else "✗" if x == "N" else "not run"}' for n, x in parts)
        out.append([key, text.format(', '.join(n for n, _ in parts)), val, note if len(parts) > 1 else ''])
    sz = None if o['extra'] or not o.get('size_ok', True) else size_table(o)
    if sz is not None:
        g = sz[(sz['analysis_id'] == aid) & (sz['prefix'] == prefix) & (sz['window'] == window) & (sz['neuron'] == neuron)]
        val = ('Y' if bool(g['survives'].iloc[0]) else 'N') if len(g) else '-'
        out.append(['size', size_text or f'with {words[0]} size (foreground patch count: how spread out or huddled the '
                    f'{words[1]} are) as a covariate (round-1 test)', val, ''])
    for o2 in others:
        g = primary_rows(o2)
        g = g[(g['analysis_id'] == aid) & (g['prefix'] == prefix) & (g['window'] == window)]
        val = ('Y' if neuron in set(g['neuron'].dropna().astype(int)) else 'N') if len(g) else '-'
        out.append([o2['short'], f'when the outcome is {o2["label"].lower()} instead of {o["label"].lower()}', val, ''])
    return out


_size = {}


def size_table(o):
    """<outcome dir>/size_adjusted.csv (run_nes*.py size check) or None."""
    k = str(o['dir'])
    if k not in _size:
        f = o['dir'] / 'size_adjusted.csv'
        _size[k] = pd.read_csv(f) if f.exists() else None
    return _size[k]


ORDER = MICE.analysis_ids()  # A_het_1to2 .. A_wt_5to6, B_stage1 .. B_stage6


def sig4(v):
    return float(f'{float(v):.4g}')


def bout_thresholds(cfg):
    """{codes key: (m,) threshold of that key's bout outcome} (the q of its primary), from the
    run's _cache/bout_thresholds_*.npz; empty when no bout outcome."""
    out = {}
    for o in cfg.outcomes:
        if is_bout(o) and o['codes'] not in out:
            fs = sorted((cfg.res / '_cache').glob('bout_thresholds_*.npz'))
            if not fs:
                raise SystemExit(f'{cfg.res}/_cache/bout_thresholds_*.npz missing (bout outcome {o["label"]!r})')
            f = np.load(fs[0])
            k = np.flatnonzero(np.isclose(f['qs'], o['primary']['threshold_q']))
            out[o['codes']] = f['thr'][int(k[0])]
    return out


def tau_check(vm, vals, results, oid, round_=1, dom=MICE):
    """max relative |tau(summary.csv) - raw contrast of the per-video values| over the page's round-`round_`
    rows: A = mean over pools of (stage b - stage a), B = mean het - mean wt; ants = mean treatment - mean
    control videos of the experiment. Only round 1 of an
    unconditioned search should match exactly: later rounds (and every round of a search conditioned on a
    nuisance such as n_fg) report the effect adjusted for the already-selected neurons (src/eci/nes.py)."""
    worst = 0.0
    for rk, R in results.items():
        o_, aid, _, _ = rk.split('|')
        if o_ != oid:
            continue
        for r in R['rows']:
            if r['round'] != round_:
                continue
            y = np.asarray(vals[str(r['neuron'])], dtype=np.float64)
            if dom is not MICE:
                an = dom.analysis(aid)
                d = vm.assign(y=y[vm.index])
                d = d[(d[list(an.where)] == pd.Series(an.where)).all(1)]
                t = float(d[d[an.arm].isin(an.treatment)]['y'].mean() - d[d[an.arm].isin(an.control)]['y'].mean())
            elif aid.startswith('A'):
                g, (a, b) = aid.split('_')[1], stages_of(aid)
                d = vm[vm['genotype'] == g].assign(y=y[vm.index[vm['genotype'] == g]])
                p = d.pivot_table(index='pool', columns='stage', values='y')
                t = float((p[b] - p[a]).mean())
            else:
                s = int(aid[len('B_stage'):])
                d = vm.assign(y=y[vm.index])[vm['stage'] == s]
                t = float(d[d['genotype'] == 'het']['y'].mean() - d[d['genotype'] == 'wt']['y'].mean())
            worst = max(worst, abs(t - r['tau']) / max(abs(r['tau']), 1e-9))
    return worst


def video_table():
    """The page's video list (video_meta order): [[pool, stage, genotype, line, sex] ...], [[name, time] ...]
    (short_obs), and {observation_id: index}."""
    vm = vmeta().reset_index(drop=True)
    ex = pd.read_csv(MICE.experiment_csv).drop_duplicates('pool').set_index('pool')
    vm['line'], vm['sex'] = ex.loc[vm['pool'], 'line'].values, ex.loc[vm['pool'], 'sex'].values
    videos = [[str(p), int(s), g, l, x] for p, s, g, l, x in zip(vm['pool'], vm['stage'], vm['genotype'], vm['line'],
                                                                  vm['sex'])]
    return vm, videos, [list(short_obs(o)) for o in vm['observation_id'].astype(str)], \
        {o: i for i, o in enumerate(vm['observation_id'].astype(str))}


def video_table_ants():
    """Ants video list (experiment.csv order): [[experiment, T, recording day, batch, position] ...],
    [[observation id, 'day <d>'] ...], {observation_id: index}."""
    vm = ANTS.video_meta()
    vm = vm[keep_mask(VIEW['ants'], vm)].reset_index(drop=True)  # the kept videos only (VIEW 'keep')
    vm['day'] = [day_label(str(d)) for d in vm['recording_date']]
    videos = [[str(e), int(t), d, str(b), int(p)] for e, t, d, b, p in zip(vm['experiment'], vm['T'], vm['day'],
                                                                          vm['batch'], vm['position'])]
    return vm, videos, [[str(o), f'day {d}'] for o, d in zip(vm['observation_id'], vm['day'])], \
        {o: i for i, o in enumerate(vm['observation_id'].astype(str))}


def page_data_clip(x):
    """[video index (page video list of the clip's domain), mean activation] of one clip; the page builds
    the tooltip label 'S<stage> <label> · <genotype> · <name> · <time> · pool <pool>' (mice),
    '<experiment> · t=<T> · day <d> · <video> · batch <b> · pos <p>' (ants) or '<group> · session <s> · <video> ·
    slot <k> · mutant side <side>' (frogs: T = the group name) from the video list."""
    if 'stage' not in x:
        frogs = isinstance(x['T'], str)
        _, videos, _, vidx = _vt(FROGS if frogs else ANTS)
        i = vidx[x['obs']]
        assert (videos[i][0], videos[i][1], videos[i][2]) == (x['experiment'], x['T'] if frogs else int(x['T']),
                                                              x['day']), x['obs']
        return [i, sig4(x['act'])]
    vm, videos, _, vidx = _vt()
    i = vidx[x['obs']]
    assert (videos[i][0], videos[i][1], videos[i][2]) == (str(x['pool']), int(x['stage']), str(x['genotype'])), x['obs']
    return [i, sig4(x['act'])]


def video_table_frogs():
    """Frogs video list (experiment.csv order): [[experiment (VIEW 'exp'), group, session, dish slot, mutant side] ...],
    [[observation id, 'session <s>'] ...], {observation_id: index}. Mutant side: the side of the body edited (half
    crispants; the video is mirrored so that it is the right), '' for WT."""
    vm = with_exp(VIEW['frogs'], FROGS.video_meta().reset_index(drop=True))
    side = vm['mutant_side'].fillna('').astype(str)
    videos = [[str(e), str(g), str(s), int(k), d] for e, g, s, k, d in zip(vm['experiment'], vm['group'], vm['session'],
                                                                          vm['slot'], side)]
    return vm, videos, [[str(o), f'session {s}'] for o, s in zip(vm['observation_id'], vm['session'])], \
        {o: i for i, o in enumerate(vm['observation_id'].astype(str))}


def _vt(dom=MICE):
    k = f'vt_{dom.name}'
    if k not in _rates:
        _rates[k] = video_table() if dom is MICE else video_table_frogs() if dom is FROGS else video_table_ants()
    return _rates[k]


AMAP = {}  # --asset-map: pack name -> URL in the artifact's asset store (packs not in it stay version files)


def media_url(name):
    return AMAP.get(name, f'assets/{name}')


def clip_peaks(pz, src, L, n_clips):
    """[[x, y], ...] activation peak of each of the first n_clips top clips of one source and length, as fractions
    of the tile (patch codes summed over the clip's frames, Gaussian-smoothed, sigma 1 patch; argmax at the patch
    centre; the patch grid covers the whole tile for fg448 SAEs, turned with the frame for aligned SAEs); None when
    the clip has no activation. The page's zoom centres on it."""
    from scipy.ndimage import gaussian_filter
    w = LENGTHS[L]
    m = pz[f'{src}__{L}__top'].astype(np.float32)
    out = []
    for k in range(n_clips):
        acc = gaussian_filter(np.maximum(m[k * w:(k + 1) * w], 0).sum(0), 1.0)
        if not acc.max() > 0:
            out.append(None)
            continue
        iy, ix = np.unravel_index(int(np.argmax(acc)), acc.shape)
        out.append([round((ix + 0.5) / acc.shape[1], 3), round((iy + 0.5) / acc.shape[0], 3)])
    return out


def page_clips(cfg, key, j, n, where):
    """{source: {L: {src, page, blocks: {block name: index in the page}, n: {kind: count}, least_rule,
    info: {kind: [[video, mean], ...]}, pk: top clip peaks (clip_peaks)}}} of one neuron; where = {L: {page id:
    (pack name, page index)}} (packs.json). Page p of a pack = frames [p w, (p + 1) w); the page holds BLOCKS of
    VIEW kr tiles."""
    pz = np.load(patch_file(cfg, key, j)) if cfg.rep == 'fg448' and not cfg.crop else None  # zoom: whole-frame tiles
    out = {}
    for src in n['clips']:
        for L in SHOW_LENGTHS:
            name, p = where[L][f'{key}|{int(j)}|{src}']
            c = n['clips'][src][L]
            blocks = {bn: i for i, bn in enumerate(BLOCKS)}
            if c['least_rule'] == 'silent' and key in ('max', 'mean'):  # heat block blank in the page: heat = raw
                blocks['least_heat'] = blocks['least']
            out.setdefault(src, {})[L] = {'src': media_url(name), 'page': p, 'blocks': blocks,
                                          'least_rule': c['least_rule'],
                                          'n': {x: min(len(c[x]), cfg.kr) for x in kinds(src)},
                                          'info': {x: [page_data_clip(y) for y in c[x][:cfg.kr]] for x in kinds(src)}}
            if pz is not None:
                out[src][L]['pk'] = clip_peaks(pz, src, L, min(len(c['top']), cfg.kr))
    return out


def analysis_meta(cfg, aid, r0):
    """Page metadata of one analysis (r0 = a summary.csv row of it)."""
    if cfg.dom is not MICE:
        m = dict(cfg.dom.analysis(aid).meta, **cfg.ameta.get(aid, {}))  # analyses.json of the run first
        if 'confounded' in m:
            conf = (m.get('confound') or 'recording day') if m['confounded'] else ''
        else:
            conf = m.get('confound', CONFOUND_FALLBACK.get(aid, ''))
        conf = conf if isinstance(conf, str) else ''
        return {'id': aid, 'family': 'B', 'experiment': m.get('experiment', cfg.view.get('exp')),
                'control': arm_of(cfg.view, m['control']), 'treatment': arm_of(cfg.view, m['treatment']),
                'confound': conf, 'n_units': int(r0['n_units'])}
    return ({'id': aid, 'family': 'A', 'genotype': r0['genotype'], 'stages': stages_of(aid),
             'n_units': int(r0['n_units'])} if aid.startswith('A') else
            {'id': aid, 'family': 'B', 'stage': int(r0['stage']), 'n_units': int(r0['n_units'])})


def censored_arms(cfg, o, aid):
    """Latency outcome o, analysis aid: ((n_videos, m) bool censored (no bout in the first W frames: latency = W)
    over the page's video list, [(arm label, (n_videos,) mask)] of the two arms (ants: control / treated videos;
    mice A: that genotype's videos of stage a / stage b; B: het / wt videos of that stage)."""
    vm = _vt(cfg.dom)[0]
    ids, lat, nf = latency_frames(o)
    ix = ids.get_indexer(vm['observation_id'].astype(str))
    assert (ix >= 0).all(), 'videos missing from the latency cache'
    cz = lat[ix] >= latency_W(o, aid)  # no bout in the analysis's common window (first W frames)
    m0, m1 = contrast_groups(cfg, aid, vm)
    an = cfg.dom.analysis(aid)
    if cfg.dom is FROGS:
        labs = (an.meta['control'], an.meta['treatment'])
    elif cfg.dom is not MICE:
        labs = (f't={an.meta["control"]}', f't={an.meta["treatment"]}')
    elif an.family == 'A':
        labs = tuple(f'stage {x}' for x in an.stages)
    else:
        labs = ('het', 'wt')
    return cz, [(labs[0], m0), (labs[1], m1)]


def cens_of(cz, arms, j):
    """[[arm label, censored videos, videos], ...] of neuron j."""
    return [[lab, int(cz[m, int(j)].sum()), int(m.sum())] for lab, m in arms]


def search_results(cfg, outs, analyses=None, others=None, top=False):
    """{'<outcome id>|<aid>|<prefix>|<window>': {n_tested_total, rows[, top]}} of the primary searches of
    one result set (full cohort or one subgroup); analyses (dict, filled) gets the per-analysis metadata.
    Cross-outcome chips: the other outcomes of outs, or others(o) when given. top = top_rows (VIEW 'top',
    full cohort only)."""
    analyses = {} if analyses is None else analyses
    results = {}
    words = (cfg.view['subject'], cfg.view['subjects'])
    for o in outs:
        prim = primary_rows(o)
        oth = [x for x in outs if x is not o] if others is None else others(o)
        for aid in cfg.order:
            g = prim[prim['analysis_id'] == aid]
            if not len(g):
                continue
            r0 = g.iloc[0]
            if aid not in analyses:
                analyses[aid] = analysis_meta(cfg, aid, r0)
            for (prefix, window), h in g.groupby(['prefix', 'window']):
                rows = []
                cz = censored_arms(cfg, o, aid) if is_latency(o) else None
                for _, r in h.dropna(subset=['neuron']).sort_values('round').iterrows():
                    j = int(r['neuron'])
                    rows.append({'round': int(r['round']), 'neuron': j, 'tau': float(r['tau']), 'p': float(r['p']),
                                 'threshold': float(r['threshold']), 'n_tested': int(r['n_tested']),
                                 'rob': robustness(o, oth, aid, int(prefix), window, j, words,
                                                   cfg.view.get('size_text'))})
                    if cz is not None:  # latency: censored videos per arm (side panel)
                        rows[-1]['cens'] = cens_of(*cz, j)
                wk = 'full'  # the page's key for the primary window (latency: the common window)
                results[f'{o["id"]}|{aid}|{int(prefix)}|{wk}'] = {
                    'n_tested_total': int(h['n_tested_total'].iloc[0]), 'rows': rows}
                rf = o['dir'] / aid / 'result.json'
                stop = read_json(rf).get(h['setting'].iloc[0], {}).get('stopped') if rf.exists() else None
                if stop:  # NES stopped early (no residual degrees of freedom left)
                    results[f'{o["id"]}|{aid}|{int(prefix)}|{wk}']['stopped'] = str(stop)
                if top and cfg.view['top']:
                    tr = top_rows(cfg, o, aid, prefix, window)
                    if cz is not None:  # [neuron, tau, p, censored per arm]
                        tr = [x + [cens_of(*cz, x[0])] for x in tr]
                    results[f'{o["id"]}|{aid}|{int(prefix)}|{wk}']['top'] = tr
    return results


def subset_meta(outs):
    """Units per analysis, pools and skipped analyses of one subgroup (from its bout / mean sanity.json)."""
    o = outs[0]
    prim = primary_rows(o)
    n_units = {aid: int(g['n_units'].iloc[0]) for aid, g in prim.groupby('analysis_id')}
    sj = json.loads((o['dir'] / 'sanity.json').read_text())
    sg = sj.get('subgroup', {})
    return {'n_units': n_units, 'n_pools': sg.get('n_pools'), 'n_het_pools': sg.get('n_het_pools'),
            'skipped': [x['analysis_id'] for x in sj.get('skipped', [])]}


def period_checks(cfg, results, xres, subsets):
    """Mice family-B camera-period checks (scripts/eci/period_check.py) of the SAE's mean-activation and event-rate
    outcomes (core or extra): {outcome id: {'thr', 'pthr', 'pair_n', 'lat': {neuron: [score, pair, auc]},
    'day': {'<analysis>|<prefix>': {neuron: [p_day, p, survives]}}}}. The files are those of the result set whose
    period_flags.json 'pooling' is the outcome's spatial aggregation (the SAE's own, else <sae>_<agg>). 'lat' covers
    the neurons of the outcome's family-B searches on the page (full cohort and subgroups); 'day' rows (full cohort) must match a
    selection on the page (same analysis, prefix, round, neuron and tau), else they are left out with a warning.
    {} for other domains or when no outcome has the files."""
    if cfg.dom is not MICE:
        return {}
    out = {}
    for o in cfg.outcomes + cfg.extras:
        if o['kind'] not in PERIOD_OUTCOME:
            continue
        d = next((x / 'period_check' for x in (cfg.res, cfg.res.parent / f'{cfg.res.name}_{o["agg"]}')
                  if (x / 'period_check' / 'period_flags.json').exists()
                  and json.loads((x / 'period_check' / 'period_flags.json').read_text())['pooling'] == o['agg']), None)
        if d is None:
            print(f'[{cfg.tag}] page: no period_check for {o["label"]!r} ({o["kind"]} x {o["agg"]})')
            continue
        PF = json.loads((d / 'period_flags.json').read_text())
        F = PF['outcomes'].get(PERIOD_OUTCOME[o['kind']])
        if F is None:
            continue
        pairs = PF['pairs']  # latents 'pair' = index into this list
        lat = F['latents']
        R = xres if o['extra'] else results
        fam_b = {aid for aid in cfg.order if cfg.dom.analysis(aid).family == 'B'}
        js = {r['neuron'] for RR in [R] + [S['results'] for S in subsets.values()] for k, v in RR.items()
              if k.startswith(o['id'] + '|') and k.split('|')[1] in fam_b for r in v['rows']}
        e = {'thr': F['threshold'], 'pthr': F['pointwise_threshold'],
             'lat': {str(j): [round(float(lat['score'][j]), 4), pairs[int(lat['pair'][j])], round(float(lat['auc'][j]), 4)]
                     for j in sorted(js)}}
        da = pd.read_csv(d / 'day_adjusted.csv')
        da = da[(da['outcome'] == PERIOD_OUTCOME[o['kind']]) & da['prefix'].isin(PAGE_PREFIXES)]
        n_bad = 0
        for _, r in da.iterrows():
            rows = R.get(f'{o["id"]}|{r["analysis_id"]}|{int(r["prefix"])}|full', {}).get('rows', [])
            hit = [x for x in rows if x['neuron'] == int(r['neuron']) and x['round'] == int(r['round'])
                   and abs(x['tau'] - r['tau']) <= 1e-3 * max(1.0, abs(r['tau']))]
            if not hit:
                n_bad += 1
                print(f'[{cfg.tag}] WARNING day-adjusted {o["label"]!r} {r["analysis_id"]} p{int(r["prefix"])} round '
                      f'{int(r["round"])} neuron {int(r["neuron"])}: no matching selection on the page, left out')
                continue
            e.setdefault('day', {}).setdefault(f'{r["analysis_id"]}|{int(r["prefix"])}', {})[str(int(r['neuron']))] = \
                [sig4(r['p_day']), sig4(r['p']), int(bool(r['survives']))]
        n_rows = sum(len(v['rows']) for k, v in R.items() if k.startswith(o['id'] + '|') and k.split('|')[1] in fam_b)
        n_day = sum(len(v) for v in e.get('day', {}).values())
        print(f'[{cfg.tag}] page: period checks {o["label"]!r} from {d}: {len(js)} family-B neurons, '
              f'{sum(v[0] >= e["thr"] for v in e["lat"].values())} period-dependent (>= {e["thr"]}), '
              f'{sum(e["pthr"] <= v[0] < e["thr"] for v in e["lat"].values())} possibly (>= {e["pthr"]}); '
              f'day-adjusted {n_day} of {n_rows} family-B selections ({n_bad} file rows without a page selection)')
        out[o['id']] = e
    return out


def session_checks(cfg, results, xres):
    """Frogs recording-session checks (scripts/eci/session_check.py) of the SAE's mean-activation and event-rate
    outcomes (core or extra): {outcome id: {'thr', 'pthr', 'lat': {neuron: [score, group, p_perm]}, 'loso':
    {'<analysis>|<prefix>': {neuron: [p_max, worst session, survives, sessions, same sign]}}}}. score = how much the
    neuron's per-video outcome varies between the recording sessions of one group (eta^2 of the within-group ranks on
    session, max over the groups; group = where it is attained), thr = family-wise threshold, pthr = one-latent
    threshold; loso = the leave-one-session-out re-test of each pick (survives = p_max below the round's threshold and
    tau keeps its sign in every drop). Files: <set>/session_check/ of the result set whose session_flags.json 'pooling'
    is the outcome's spatial aggregation (the SAE's own, else <sae>_<agg>). 'lat' covers the neurons of the outcome's
    searches on the page; 'loso' rows must match a selection on the page (analysis, prefix, round, neuron, tau), else
    they are left out with a warning. {} for other domains or when no outcome has the files."""
    if cfg.dom is not FROGS:
        return {}
    out = {}
    for o in cfg.outcomes + cfg.extras:
        if o['kind'] not in PERIOD_OUTCOME:
            continue
        d = next((x / 'session_check' for x in (cfg.res, cfg.res.parent / f'{cfg.res.name}_{o["agg"]}')
                  if (x / 'session_check' / 'session_flags.json').exists()
                  and read_json(x / 'session_check' / 'session_flags.json')['pooling'] == o['agg']), None)
        if d is None:
            print(f'[{cfg.tag}] page: no session_check for {o["label"]!r} ({o["kind"]} x {o["agg"]})')
            continue
        SF = read_json(d / 'session_flags.json')
        F = SF['outcomes'].get(PERIOD_OUTCOME[o['kind']])
        if F is None:
            continue
        lat, groups = F['latents'], F['groups_scored']
        R = xres if o['extra'] else results
        js = {r['neuron'] for k, v in R.items() if k.startswith(o['id'] + '|') for r in v['rows']}
        e = {'thr': F['threshold'], 'pthr': F['pointwise_threshold'],
             'lat': {str(j): [round(float(lat['score'][j]), 4), groups[int(lat['group'][j])], sig4(lat['p_perm'][j])]
                     for j in sorted(js)}}
        lo = pd.read_csv(d / 'loso.csv')
        lo = lo[(lo['outcome'] == PERIOD_OUTCOME[o['kind']]) & lo['prefix'].isin(PAGE_PREFIXES)]
        n_bad = 0
        for _, r in lo.iterrows():
            rows = R.get(f'{o["id"]}|{r["analysis_id"]}|{int(r["prefix"])}|full', {}).get('rows', [])
            hit = [x for x in rows if x['neuron'] == int(r['neuron']) and x['round'] == int(r['round'])
                   and abs(x['tau'] - r['tau']) <= 1e-3 * max(1.0, abs(r['tau']))]
            if not hit:
                n_bad += 1
                print(f'[{cfg.tag}] WARNING leave-one-session-out {o["label"]!r} {r["analysis_id"]} p{int(r["prefix"])} '
                      f'round {int(r["round"])} neuron {int(r["neuron"])}: no matching selection on the page, left out')
                continue
            ok = bool(r['p_max_below']) and bool(r['same_sign'])
            e.setdefault('loso', {}).setdefault(f'{r["analysis_id"]}|{int(r["prefix"])}', {})[str(int(r['neuron']))] = \
                [sig4(r['p_max']), str(r['worst_session']), int(ok), int(r['n_sessions']), int(bool(r['same_sign']))]
        n_rows = sum(len(v['rows']) for k, v in R.items() if k.startswith(o['id'] + '|'))
        n_lo = sum(len(v) for v in e.get('loso', {}).values())
        print(f'[{cfg.tag}] page: session checks {o["label"]!r} from {d}: {len(js)} neurons, '
              f'{sum(v[0] >= e["thr"] for v in e["lat"].values())} session-dependent (>= {e["thr"]}), '
              f'{sum(e["pthr"] <= v[0] < e["thr"] for v in e["lat"].values())} possibly (>= {e["pthr"]}); '
              f'leave-one-session-out {n_lo} of {n_rows} selections, {sum(x[2] for v in e.get("loso", {}).values() for x in v.values())} '
              f'survive ({n_bad} file rows without a page selection)')
        out[o['id']] = e
    return out


def page_data(cfg):
    picks = json.loads((cfg.work / 'picks.json').read_text())
    analyses, outcomes = {}, []
    allo = cfg.outcomes + cfg.extras
    for o in allo:
        outcomes.append({'id': o['id'], 'label': o['label'], 'codes': o['codes'], 'unit': o['unit'],
                         'bout': is_bout(o),
                         'setting': ', '.join(f'{k} {v:g}' if isinstance(v, float) else f'{k} {v}'
                                              for k, v in o['primary'].items()),
                         'kind': o['kind'], 'agg': o['agg'],
                         'prefixes': sorted(int(x) for x in primary_rows(o)['prefix'].unique())})
    results = search_results(cfg, cfg.outcomes, analyses, top=True)
    # extra outcomes: full cohort only (the subgroup runs are primary-only); cross-outcome chips = the core
    # outcomes of the other kind
    xres = search_results(cfg, cfg.extras, analyses, others=lambda o: [x for x in cfg.outcomes if x['kind'] != o['kind']],
                          top=True)
    subsets = {}
    for name, outs in cfg.subsets.items():
        xouts = cfg.xsubsets.get(name, [])  # extra outcomes' subgroup searches (aggregation siblings)
        subsets[name] = {'results': {**search_results(cfg, outs),
                                     **search_results(cfg, xouts, others=lambda o: [x for x in outs if x['kind'] != o['kind']])},
                         **subset_meta(outs)}
        print(f'[{cfg.tag}] page: subgroup {name}: ' + ', '.join(
            f'{o["id"]} {sum(len(R["rows"]) for k, R in subsets[name]["results"].items() if k.startswith(o["id"] + "|"))} hits'
            for o in outs + xouts))
    # consistency check of the chips against the stored table (mean-pool result set, full window)
    det_p = cfg.res / 'galleries/stats_detail.csv'
    o0 = next((o for o in cfg.outcomes if mean_outcome(o) and o['dir'] == cfg.res), None)
    if det_p.exists() and o0 is not None:
        det = pd.read_csv(det_p, dtype=str)
        names = {'flip': 'rob_signflip', 'BH': 'rob_BH', 'rate': 'rob_rate', 'match': 'rob_matched'}
        bad = 0
        det = det[det['prefix'].astype(int).isin(PAGE_PREFIXES)]
        for _, d in det.iterrows():
            rr = [x for x in results[f'{o0["id"]}|{d["analysis_id"]}|{d["prefix"]}|full']['rows']
                  if x['neuron'] == int(d['neuron'])][0]
            for lab, _, v, _ in rr['rob']:
                if lab in names and v != d[names[lab]]:
                    bad += 1
                    print('  robustness mismatch', d['analysis_id'], d['prefix'], d['neuron'], lab, v)
        print(f'[{cfg.tag}] page: robustness chips vs stats_detail.csv: {bad} mismatches over {len(det)} rows')

    thr = bout_thresholds(cfg)
    ar = np.load(cfg.work / 'arena.npz')
    grid, act_all, occ = int(ar['grid']), ar['act'], ar['occ']
    bg = f'assets/{bg_name(cfg)}' if (cfg.assets / bg_name(cfg)).exists() else None

    def arena_of(j):
        e = {'rows': grid, 'cols': grid, 'act': [sig4(v) for v in act_all[:, int(j)]]}
        if len(occ):
            e['occ'] = [sig4(v) for v in occ]
        if bg:
            e['bg'] = bg
        return e
    packs = json.loads((cfg.work / 'packs.json').read_text())
    where = {L: {pid: (pk['name'], i) for pk in P for i, pid in enumerate(pk['pages'])} for L, P in packs.items()}
    neurons = {}
    for key in cfg.keys:
        neurons[key] = {}
        for j, n in picks['by_key'][key].items():
            vmax = n['max_frame'] or 1.0
            t = float(thr[key][int(j)]) if key in thr else None
            e = {'firing_rate': n['firing_rate'],
                 'table': [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in t_.items()} for t_ in n['table']],
                 'artefact': int(j) in cfg.artefact,
                 'heat_vmax': sig4(np.load(patch_file(cfg, key, j))['vmax']),
                 'clips': page_clips(cfg, key, j, n, where),
                 'hist': dict(n['hist'], thr=t),
                 'bout_thr_bar': (round(min(t / vmax, 1.0), 4) if t is not None and vmax > 0 else None),
                 'arena': arena_of(j)}
            if str(j) in cfg.interp:
                e['interp'] = {k: cfg.interp[str(j)][k] for k in ('text', 'conf') if k in cfg.interp[str(j)]}
            neurons[key][j] = e
        for k2, j in sorted(cfg.noclip):
            if k2 == key:
                neurons[key][str(j)] = {'noclip': True, 'artefact': int(j) in cfg.artefact, 'arena': arena_of(j)}
    # every other listed neuron (beyond the --max-pages clip budget): arena
    # map, chart and per-video panel only; 'x' = not in the historical neuron files (split_data)
    listed = {o['id']: set() for o in allo}
    for RR in [results, xres] + [v['results'] for v in subsets.values()]:
        for k, R in RR.items():
            listed[k.split('|')[0]] |= {str(r['neuron']) for r in R['rows']} | {str(t[0]) for t in R.get('top', [])}
    for o in allo:
        for j in sorted(listed[o['id']], key=int):
            if j not in neurons.setdefault(o['codes'], {}):
                neurons[o['codes']][j] = {'noclip': True, 'artefact': int(j) in cfg.artefact, 'arena': arena_of(j),
                                          'x': True}
    # audit: every neuron the page lets one select (any outcome, search, subgroup, top-by-p list) has its clips
    sel = {(o['codes'], j) for o in allo for j in listed[o['id']]}
    bad = sorted((k, int(j)) for k, j in sel if neurons[k][j].get('noclip') or 'clips' not in neurons[k][j])
    print(f'[{cfg.tag}] page: audit: {len(sel)} selectable (codes key, neuron) pairs, {len(bad)} without clips {bad[:10]}')
    if bad:
        raise SystemExit(f'[{cfg.tag}] {len(bad)} selectable neurons without clips')
    art = {}
    for o in cfg.outcomes:
        sel = o['dir'] / 'selected_neurons.json'
        for j, v in (json.loads(sel.read_text())['neurons'] if sel.exists() else {}).items():
            if v.get('artefact_flag'):
                art.setdefault(j, v['artefact_flag'])
    for j in cfg.artefact:
        art.setdefault(str(j), 'flagged as a likely recording artefact')
    # per-video values of the tested outcome, for the per-pool panel (one list per outcome and neuron,
    # in the order of 'videos'); tables of bout outcomes by stage x genotype
    vm, videos, _, _ = _vt(cfg.dom)
    vm = vm.copy()
    vals, otables = {}, {}
    for o in allo:
        ov = video_outcome(cfg, o)
        if ov is None:
            continue
        ids, Y = ov
        ix = ids.get_indexer(vm['observation_id'].astype(str))
        assert (ix >= 0).all(), 'videos missing from the NES cache'
        js = sorted({str(r['neuron']) for RR in [results, xres] + [v['results'] for v in subsets.values()]
                     for k, R in RR.items() if k.startswith(o['id'] + '|') for r in R['rows']}, key=int)
        if cfg.view['top']:  # every neuron of the page gets every outcome's values (chart, per-video panel)
            js = sorted(set(js) | set(neurons.get(o['codes'], {})) | listed[o['id']], key=int)
        vals[o['id']] = {j: [sig4(v) for v in Y[ix, int(j)]] for j in js}
        if is_latency(o):  # per analysis: values capped at its W (as the page shows them)
            wr = 0.0
            for aid in cfg.order:
                if not (primary_rows(o)['analysis_id'] == aid).any():
                    continue
                Ws = latency_W(o, aid) / cfg.dom.fps
                cv = {j: [min(v, Ws) for v in V] for j, V in vals[o['id']].items()}
                wr = max(wr, tau_check(vm, cv, {k: R for k, R in xres.items() if k.split('|')[1] == aid}, o['id'], dom=cfg.dom))
            print(f'[{cfg.tag}] page: {o["label"]}: tau recomputed from per-video values capped at W vs summary.csv, '
                  f'max relative diff {wr:.2e}')
        else:
            print(f'[{cfg.tag}] page: {o["label"]}: tau recomputed from per-video values vs summary.csv, '
                  f'max relative diff {tau_check(vm, vals[o["id"]], xres if o["extra"] else results, o["id"], dom=cfg.dom):.2e}')
        for name, v in subsets.items():
            if is_latency(o):
                continue
            l, x = name.split('_')
            sm = vm[((vm['line'] == l) | (l == 'all')) & ((vm['sex'] == x) | (x == 'all'))]
            print(f'[{cfg.tag}] page: {o["label"]} subgroup {name}: tau check max relative diff '
                  f'{tau_check(sm, vals[o["id"]], v["results"], o["id"]):.2e}')
        if is_bout(o) and not o['extra'] and cfg.dom is MICE:
            otables[o['id']] = {'full': {j: rate_table(cfg, o, int(j)) for j in picks['by_key'][o['codes']]}}
    for o in cfg.outcomes:  # every neuron shown in a subgroup needs its per-video values (chart, per-pool panel)
        miss = {str(j) for (k, j) in cfg.noclip if k == o['codes']} - set(vals[o['id']])
        if miss:
            ids, Y = video_outcome(cfg, o)
            ix = ids.get_indexer(vm['observation_id'].astype(str))
            vals[o['id']].update({j: [sig4(v) for v in Y[ix, int(j)]] for j in miss})
    # latency outcomes: {analysis: W (s)}, the common window; the page caps the per-video values at W and marks a
    # value at W censored
    lwin = {o['id']: {aid: sig4(latency_W(o, aid) / cfg.dom.fps) for aid in analyses if (primary_rows(o)['analysis_id'] == aid).any()}
            for o in allo if is_latency(o) and o['id'] in vals}
    cs = change_share(cfg)
    if cs is not None:
        print(f'[{cfg.tag}] page: change share of the decoder rows (motion SAE): median {np.median(cs):.3f}, '
              f'quartiles {np.quantile(cs, 0.25):.3f} / {np.quantile(cs, 0.75):.3f}')
    return {'sae': cfg.sae, 'tag': cfg.tag, 'model': model_of(cfg), 'rep': cfg.rep, 'outcomes': outcomes,
            'period': period_checks(cfg, results, xres, subsets), 'session': session_checks(cfg, results, xres),
            'lwin': lwin, 'change_share': None if cs is None else [round(float(x), 3) for x in cs],
            'otables': otables, 'artefact': sorted(int(j) for j in cfg.artefact),
            'subsets': subsets,
            'analyses': [analyses[a] for a in cfg.order if a in analyses], 'results': results, 'neurons': neurons,
            'artefact_text': art, 'arena_note': str(ar['note']), 'videos': videos, 'vals': vals,
            'n_frames': int(sum(next(iter(picks['by_key'][cfg.keys[0]].values()))['hist']['counts']['all'])),
            'xresults': xres, 'xids': [o['id'] for o in cfg.extras],
            'montage': {'K': K, 'kr': cfg.kr, 'cols': cfg.cols, 'tile': cfg.tile, 'crop': cfg.crop,
                        'lengths': {L: LENGTHS[L] for L in SHOW_LENGTHS}},
            'domain': cfg.dom.name, 'obs': _vt(cfg.dom)[2]}


def q16(v):
    """Non-negative values -> {'max', 'q': base64 of little-endian uint16 round(v / max * 65535)} (arena maps:
    display only)."""
    import base64
    v = np.maximum(np.asarray(v, dtype=np.float64), 0)
    mx = float(v.max()) if v.max() > 0 else 1.0
    return {'max': sig4(mx), 'q': base64.b64encode(np.round(v / mx * 65535).astype('<u2').tobytes()).decode()}


def jdump(x):
    return json.dumps(x, separators=(',', ':'))


def split_data(d, data_dir, inline):
    """One SAE's page data -> (the part kept in index.html, {data file name: content}). Files (fetched by
    the page on demand, assets/data/): <tag>_r_<outcome id>.json = that outcome's searches (full cohort and
    every subgroup); <tag>_<codes key>_<k>.json = chunk k of the neurons ranked by that codes key (clip
    lists, histogram, arena map, per-video values and stage x genotype tables of each outcome built from
    that key), ~CHUNK_BYTES each. inline: the results file whose content goes into index.html instead.
    <tag>_x.json = one bundle {results, subsets, vals: {outcome: {neuron: values}}, by: {codes key: {neuron:
    entry}}} with the extra outcomes' searches and values and the neurons not in the files above ('x'
    entries)."""
    tag, files, xids = d['tag'], {}, set(d['xids'])
    core = {k: d[k] for k in ('sae', 'tag', 'model', 'rep', 'outcomes', 'analyses', 'artefact', 'artefact_text',
                              'arena_note', 'n_frames', 'montage')}
    core['subsets'] = {name: {k: v for k, v in S.items() if k != 'results'} for name, S in d['subsets'].items()}
    for k in ('lwin', 'change_share', 'period', 'session'):  # only SAEs that have them (latency, motion SAEs,
        # mice period checks, frogs session checks)
        if d.get(k):
            core[k] = d[k]
    core['rfiles'], core['pre'] = {}, {}
    xf, X = f'{tag}_x.json', {'results': {}, 'subsets': {}, 'vals': {}, 'by': {}}
    for o in d['outcomes']:
        pre_ = o['id'] + '|'
        R = {'results': {k: v for k, v in (d['xresults'] if o['id'] in xids else d['results']).items()
                         if k.startswith(pre_)},
             'subsets': {name: {k: v for k, v in S['results'].items() if k.startswith(pre_)}
                         for name, S in d['subsets'].items()}}
        f = f'{tag}_r_{o["id"]}.json'
        if o['id'] in xids:
            X['results'].update(R['results'])
            for name, S in R['subsets'].items():
                if S:
                    X['subsets'].setdefault(name, {}).update(S)
            X['vals'][o['id']] = d['vals'].get(o['id'], {})
            core['rfiles'][o['id']] = f'{data_dir}/{xf}'
        elif (tag, o['id']) == inline:  # key 'pre:<file>': not a file, the page reads it from ALL
            core['rfiles'][o['id']] = f'pre:{f}'
            core['pre'][f'pre:{f}'] = R
        else:
            core['rfiles'][o['id']] = f'{data_dir}/{f}'
            files[f] = R
    arena0 = None
    core['nfile'] = {}
    for key, ns in d['neurons'].items():
        chunks, cur, size, xs = [], {}, 0, {}
        for j in sorted(ns, key=int):
            e = dict(ns[j])
            A = e.pop('arena')
            isx = e.pop('x', False)
            if arena0 is None:
                arena0 = {'rows': A['rows'], 'cols': A['cols'], 'bg': A.get('bg'),
                          'occ': q16(A['occ']) if 'occ' in A else None}
            e['arena'] = q16(A['act'])
            if isx:
                xs[j] = e
                continue
            e['vals'] = {o['id']: d['vals'][o['id']][j] for o in d['outcomes']
                         if o['codes'] == key and o['id'] not in xids and j in d['vals'].get(o['id'], {})}
            e['otable'] = {o['id']: d['otables'][o['id']]['full'][j] for o in d['outcomes']
                           if o['codes'] == key and o['id'] in d['otables'] and j in d['otables'][o['id']]['full']}
            n = len(jdump(e))
            if cur and size + n > CHUNK_BYTES:
                chunks.append(cur)
                cur, size = {}, 0
            cur[j], size = e, size + n
        if cur:
            chunks.append(cur)
        core['nfile'][key] = {}
        for k, c in enumerate(chunks):
            f = f'{tag}_{key}_{k}.json'
            files[f] = {'neurons': c}
            core['nfile'][key].update({j: k for j in c})
        if xs:
            X['by'][key] = xs
            core['nfile'][key].update({j: len(chunks) for j in xs})
        core['nfile'][key] = {'files': [f'{data_dir}/{tag}_{key}_{k}.json' for k in range(len(chunks))]
                                       + ([f'{data_dir}/{xf}'] if xs else []),
                              'of': core['nfile'][key]}
    core['arena'] = arena0
    if xids or X['by']:
        files[xf] = X
    # every per-video value / table of the data went to some neuron file
    for o in d['outcomes']:
        lost = set(d['vals'].get(o['id'], {})) - set(d['neurons'][o['codes']])
        assert not lost, f'{tag} {o["id"]}: per-video values of neurons without an entry {sorted(lost)[:5]}'
    # page settings of the SAE's domain; a non-mice SAE carries its own video list
    v = VIEW[d['domain']]
    core.update(domain=d['domain'], subject=v['subject'], subjects=v['subjects'], sub_ui=d['domain'] == 'mice',
                top=v['top'])
    if d['domain'] != 'mice':
        core.update(videos=d['videos'], obs=d['obs'])
    if v['vnote']:
        core['vnote'] = v['vnote']
    if v.get('arm'):  # frogs: arm values in display order (control first), the domain note and its hover text
        core.update(arms=list(v['arms']), dnote=v['note'][0], dnote_tip=v['note'][1])
    if d['sae'] in MODEL_NOTES:
        core['note'] = MODEL_NOTES[d['sae']]
    return core, files


def step_page(cfgs, overwrite):
    from scipy import stats
    saes = [page_data(c) for c in cfgs]
    has_mice = any(c.dom is MICE for c in cfgs)
    _, videos, obs, _ = _vt() if has_mice else (None, [], [], None)
    doms = list(dict.fromkeys(c.dom.name for c in cfgs))
    data = {'default': cfgs[0].sae, 'saes': [],
            'axes': [{'key': k, 'label': lab, 'tips': {**{v: MODEL_TIPS.get((k, v), v) for v in
                                                          dict.fromkeys(d['model'][k] for d in saes)},
                                                       **{d['model'][k]: MODEL_SAE_TIPS[(d['sae'], k)] for d in saes
                                                          if (d['sae'], k) in MODEL_SAE_TIPS}}}
                     for k, lab in MODEL_AXES],
            'videos': videos, 'obs': obs,
            'montage': saes[0]['montage'],  # each SAE carries its own (core 'montage'); this = the default's
            'tcrit': {str(n): round(float(stats.t.ppf(0.975, n - 1)), 4) for n in range(2, 121)},
            'domains': [{'id': x, 'title': VIEW[x]['title'], 'default': next(c.sae for c in cfgs if c.dom.name == x)}
                        for x in doms],
            'kinds': [{'id': k, 'label': lab, 'tip': tip} for k, lab, tip in OUTCOME_KINDS],
            'aggs': [{'id': k, 'label': lab, 'tip': tip} for k, lab, tip in AGGS]}
    out = cfgs[0].out
    ddir = out / 'assets' / 'data'
    ddir.mkdir(parents=True, exist_ok=True)
    inline = (saes[0]['tag'], saes[0]['outcomes'][0]['id'])  # the first view's searches
    written = set()
    for d in saes:
        core, files = split_data(d, 'assets/data', inline)
        data['saes'].append(core)
        for f, c in files.items():
            (ddir / f).write_text(jdump(c))
            written.add(f)
        print(f'[{d["tag"]}] page: {len(files)} data files, {sum(len(jdump(c)) for c in files.values()) / 1e3:.0f} KB, '
              f'largest {max(len(jdump(c)) for c in files.values()) / 1e3:.0f} KB')
    for f in ddir.glob('*.json'):
        if f.name not in written:
            f.unlink()
    tpl = TEMPLATE.read_text()
    assert tpl.count('/*__DATA__*/null') == 1, 'template placeholder missing'
    page = tpl.replace('/*__DATA__*/null', jdump(data))
    (out / 'index.html').write_text(page)
    print('page: wrote', out / 'index.html', f'{len(page.encode()) / 1e3:.0f} KB')


# ---------------------------------------------------------------------- step: check
def step_check(cfgs, a):
    """index.html references the data files (assets/data/*.json); the page and the data files reference the
    media (clip packs: assets/<name>.mp4, or their asset-store URL when --asset-map maps the name). Every
    local reference must exist; unreferenced local files are deleted. Writes <out>/publish.json: the version
    files (index.html, data, arena backgrounds, packs not in the asset map) and the packs with their sizes and
    asset URL (null = not uploaded). Fails on a missing file, a pack > 15 MB (version file limit; the asset
    store takes 20 MiB), index.html > --max-page-kb, or version files over --max-version-files / --max-mb."""
    out = cfgs[0].out
    assets = out / 'assets'
    page = (out / 'index.html').read_text()
    assert page.startswith('<title>'), 'page must begin with <title>'
    for tag in ('html', 'head', 'body', '!doctype'):
        assert not re.search(rf'<{tag}[\s>]', page, re.I), f'page must not contain <{tag}>'
    drefs = set(re.findall(r'assets/data/[\w.\-]+\.json', page))
    dmiss = sorted(r for r in drefs if not (out / r).exists())
    texts = [page] + [(out / r).read_text() for r in sorted(drefs) if (out / r).exists()]
    refs = {r for t in texts for r in re.findall(r'assets/[\w.\-]+\.(?:mp4|webp|png|jpg)', t)}
    inv = {u: n for n, u in AMAP.items()}
    mapped = {inv[u] for t in texts for u in inv if u in t}  # packs the page reads from the asset store
    missing = sorted(r for r in refs if not (out / r).exists()) + dmiss
    missing += sorted(f'assets/{n} (mapped, local copy)' for n in mapped if not (assets / n).exists())
    files = sorted(p for p in assets.iterdir() if p.is_file())
    dfiles = sorted((assets / 'data').glob('*')) if (assets / 'data').is_dir() else []
    unused = [p for p in files if f'assets/{p.name}' not in refs and p.name not in mapped] + \
             [p for p in dfiles if f'assets/data/{p.name}' not in drefs]
    for p in unused:
        p.unlink()
    files = [p for p in files + dfiles if p not in unused]
    vfiles = [p for p in files if p.name not in mapped]
    packs = [p for p in files if p.suffix == '.mp4']
    vbytes = sum(p.stat().st_size for p in vfiles) + (out / 'index.html').stat().st_size
    big = max(packs, key=lambda p: p.stat().st_size) if packs else None
    n_data = sum(1 for p in files if p.parent.name == 'data')
    print(f'check: {len(refs)} local media refs + {len(mapped)} asset-store packs + {len(drefs)} data files, '
          f'{len(missing)} missing {missing[:5]}, {len(unused)} unused deleted {[p.name for p in unused][:6]}')
    print(f'check: clip packs {len(packs)}, {sum(p.stat().st_size for p in packs) / 1e6:.1f} MB '
          f'(largest {big.name if big else "-"} {big.stat().st_size / 1e6 if big else 0:.1f} MB); '
          f'{len(mapped)} served from the asset store ({sum((assets / n).stat().st_size for n in mapped if (assets / n).exists()) / 1e6:.1f} MB)')
    print(f'check: version files {len(vfiles) + 1} (index.html + {n_data} data + {len(vfiles) - n_data} media), '
          f'{vbytes / 1e6:.1f} MB (limits {a.max_version_files} files, {a.max_mb:g} MB); page {len(page.encode()) / 1e3:.0f} KB')
    for c in cfgs:
        n = [p for p in files if p.name.startswith(c.tag + '_')]
        print(f'check: [{c.tag}] {len(n)} files, {sum(p.stat().st_size for p in n) / 1e6:.1f} MB')
    man = {'page': 'index.html', 'page_bytes': len(page.encode()),
           'version_files': {str(p.relative_to(out)): p.stat().st_size for p in vfiles},
           'packs': {p.name: {'bytes': p.stat().st_size, 'asset_url': AMAP.get(p.name)} for p in packs}}
    (out / 'publish.json').write_text(json.dumps(man, indent=1))
    fit = len(vfiles) + 1 <= a.max_version_files and vbytes <= a.max_mb * 1e6
    if not fit:
        print('check: WARNING the version files do not fit one artifact version: upload the packs to the asset '
              'store and rebuild the page with --asset-map (publish.json lists them)')
    if missing or len(page.encode()) > a.max_page_kb * 1e3 or (big is not None and big.stat().st_size > 15e6) \
            or (a.require_fit and not fit):
        raise SystemExit('check failed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', action='append', help=f'NES result set, repeatable; first = page default '
                    f'(default: {DEFAULT_RES}, + discovered ones with --discover); cache in <res>/_cache/explorer')
    ap.add_argument('--discover', action='store_true', help='without --res: add every other finished SAE result '
                    'set under the NES roots to DEFAULT_RES')
    ap.add_argument('--outcome', action='append', help='LABEL=SUBDIR[:key=value,...], repeatable')
    ap.add_argument('--out', default=None, help='output dir (default <first res>/explorer)')
    ap.add_argument('--steps', default='data,patch,arena,render,page,check')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--max-pages', type=int, default=0, help='clip pages per SAE (codes key x neuron x source, '
                    'each at every length), in clip_candidates priority order; 0 = every listed neuron')
    ap.add_argument('--pack-mb', type=float, default=8.0, help='clip pack size (MB); packs are version files '
                    '(<= 15 MB) or asset-store uploads (<= 20 MiB)')
    ap.add_argument('--asset-map', action='append', default=[], help='JSON {pack name: asset URL} of packs uploaded '
                    'to the artifact asset store, repeatable (merged in order); the page reads those from their URL (the '
                    'rest stay version files)')
    ap.add_argument('--max-version-files', type=int, default=511, help='all published files (artifact version limit)')
    ap.add_argument('--max-page-kb', type=float, default=400, help='index.html size limit')
    ap.add_argument('--max-mb', type=float, default=250, help='version size limit (the artifact takes 256 MB)')
    ap.add_argument('--require-fit', action='store_true', help='check fails when the version files do not fit')
    ap.add_argument('--crf', type=int, default=None, help='H.264 quality of the clips (default per domain, VIEW crf)')
    ap.add_argument('--ants-clips', default='whole', choices=('crop', 'whole'),
                    help="ant clip tiles: 'whole' (default: the whole frame, resized to --ants-tile px) or 'crop' (256 px "
                         "full-resolution cuts around each top clip's activation peak)")
    ap.add_argument('--ants-tile', type=int, default=320, help="tile px of the ant clips")
    ap.add_argument('--patch-shard', default='0/1', help='i/n: the patch step does every n-th missing neuron from the '
                    'i-th (n parallel GPU jobs with --steps patch; then one job runs the remaining steps)')
    a = ap.parse_args()
    PATCH_SHARD = tuple(int(x) for x in a.patch_shard.split('/'))
    # ant clip geometry (part of the segment signatures: a change re-renders only the ant SAEs' packs)
    VIEW['ants'].update(crop=256 if a.ants_clips == 'crop' else None, tile=256 if a.ants_clips == 'crop' else a.ants_tile)
    res = a.res or discover_res(a.discover)
    r0 = (ROOT / res[0]) if not Path(res[0]).is_absolute() else Path(res[0])
    out = Path(a.out) if a.out else r0 / 'explorer'
    for f in a.asset_map:
        AMAP.update(json.loads(Path(f).read_text()))
    cfgs = [Cfg(r, a, out) for r in res]
    tags = [c.tag for c in cfgs]
    for c in cfgs:  # file prefixes must be unique across domains (e.g. two 'ep20' SAEs)
        if tags.count(c.tag) > 1:
            c.tag = f'{c.dom.name}{c.tag}'
    plan_clips(cfgs, a.max_pages)
    for s in a.steps.split(','):
        if s == 'page':
            step_page(cfgs, a.overwrite)
        elif s == 'check':
            step_check(cfgs, a)
        elif s == 'render':
            for c in cfgs:
                step_render(c, a.overwrite, a.pack_mb)
        else:
            for c in cfgs:
                globals()[f'step_{s}'](c, a.overwrite)
