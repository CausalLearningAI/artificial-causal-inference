#!/bin/bash
#
# Why did encoder fine-tuning help? -- an ablation that can actually discriminate.
# ==============================================================================
# We have ONE data point and a story attached to it:
#
#     res448_k2_ft2_d4photo              macro AP 0.4889   r 0.542   (2 blocks unfrozen, 14.2M)
#     res448_k2_frozen_d4photo_decay40   macro AP 0.4381   r 0.273   (frozen)
#
# RESOLVED 2026-08-15 -- every arm has landed. See the RESULT block at the bottom of this file.
# Short version: ft_bitfit WINS, both head arms are FLAT, so the answer is recalibration and
# ssl_dapt is licensed (scripts/mice_behavior/ablate_ssl_dapt.sh).
#
# The story was "the head cannot do relational computation, so the encoder had to". That story
# has never been tested, and the architectural half of it was overstated: softmax couples the
# patch weights competitively and n_heads=8 gives eight pooling patterns, so a Deep-Sets
# argument says a single-query head CAN represent permutation-invariant relational functions
# given expressive enough per-patch features. The precise claim that survives is narrower --
# NO operation in the head computes a function of two patch features JOINTLY (attention scores
# are query-vs-key, never key-vs-key) -- and whether that matters here is empirical.
#
# Three competing explanations, one arm each:
#
#   (1) the HEAD was the bottleneck; the frozen representation was already sufficient
#         head_selfattn    frozen encoder + one bottlenecked self-attention layer over the 1024
#                          patch tokens. Adds exactly the missing pairwise term, nothing else;
#                          it is a zero-initialised residual, so at step 0 it is bit-for-bit the
#                          baseline head and any difference is attributable to that term.
#         head_multiquery  frozen encoder + K=4 pooling queries instead of 1. The cheapest
#                          possible capacity test, and the design MouseOPairClassifier has
#                          always used -- the frame classifier just never inherited it.
#
#   (2) DINOv2 needed RECALIBRATION, not new computation
#         ft_bitfit        same 2 blocks unfrozen, but only their biases, LayerNorm gains and
#                          LayerScale gains train: 24,576 params vs 14,180,352, a 577x cut.
#                          None of them touches a weight matrix, so they can rescale/reshift a
#                          feature the frozen weights already compute but can never form a new
#                          one. Run at TWO encoder LRs -- 1e-5 was chosen for 14.2M densely
#                          coupled weights, and a bias-only null at that LR alone would be
#                          uninterpretable rather than informative.
#
#   (3) domain-adaptive SSL would do better than supervised adaptation
#         ssl_dapt         DELIBERATELY NOT LAUNCHED HERE. It is the expensive arm and it only
#                          addresses hypothesis (2); run it if and only if ft_bitfit shows that
#                          cheap recalibration is what mattered.
#
# CONTROLS (not optional, and not in the original arm list):
#   The frozen reference above ran lr_decay_epochs=40/n_epochs=40 while the fine-tuned one ran
#   30/30, so "frozen vs fine-tuned" is confounded with schedule length. The frozen head arms
#   here run at 30 to match the fine-tuned config, which means they need their own frozen
#   control at 30 -- otherwise a head-arm win could just be the schedule. That control is also
#   run at a second seed, because with n=24 val observations nothing else in this experiment
#   tells us how large a difference has to be before it means anything.
#
#         frozen_ctrl_s42  frozen, decay 30 -- the reference the two head arms are measured against
#         frozen_ctrl_s1   identical but --seed 1 -- the run-to-run noise floor
#
# READING THE RESULT (macro AP first, then the full suite -- never AP alone):
#
#     ft_bitfit   head_selfattn   conclusion
#     ---------   -------------   ----------------------------------------------------------
#     wins        flat            domain shift; cheap recalibration suffices -> run ssl_dapt
#     flat        wins            the head was the bottleneck; full fine-tuning is overkill
#     both win    both win        complementary: redesign the head AND adapt the encoder
#     flat        flat            genuinely needs full-block adaptation; the architectural
#                                 explanation is incomplete and should be revised
#
#   "wins"/"flat" are judged against frozen_ctrl_s42 vs frozen_ctrl_s1, not against zero.
#   Per-observation Pearson r has SE ~0.2 at n=24 and CANNOT rank close arms; report it with
#   that caveat, never as a sort key.
#
# Usage:
#     bash scripts/mice_behavior/ablate_head_vs_encoder.sh          # submit every arm
#     ARMS="ft_bitfit ft_bitfit_hi" bash .../ablate_head_vs_encoder.sh   # a subset
#     DRY=1 bash scripts/mice_behavior/ablate_head_vs_encoder.sh    # print, submit nothing
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference

# Everything held fixed at the winning config. Only the per-arm block below varies.
export INPUT_SIZE=448 CONTEXT_K=2 STRIDE=1 AUGMENT=d4_photo PHOTO_STRENGTH=1.0
export OPTIMIZER=adamw LR=3e-4 WEIGHT_DECAY=0.05 DROPOUT=0.4
export WARMUP_EPOCHS=3 LR_DECAY_EPOCHS=30 N_EPOCHS=30 PATIENCE=10
export BATCH_SIZE=64 NEG_RATIO=1 MAX_TRAIN_FRAMES=300000
export LAYERWISE_DECAY=0.65     # inherited, never tuned -- true of every FT run in this repo
export JPEG_CACHE_FILE=dataset/mice/v1/jpegcache_k2   # complete (444k frames); skips the ~33 min NFS read
export WANDB=1

# --mem=180G: the JPEG cache fills lazily and grows every epoch, so peak RSS lands near the END
# of training. 120G OOM-killed four arms of the previous sweep mid-run.
#
# MEM is overridable because those OOMs came from runs that BUILD the cache from scratch: they
# hold ~18.6 GiB of JPEG bytes as anonymous numpy arrays and drag the page cache of 444k
# individual NFS files in behind them, and the cgroup limit counts both (the killed jobs showed
# only ~29 GiB of process RSS). Every arm here instead passes JPEG_CACHE_FILE and memory-maps
# one pre-built 18.6 GiB file, so the same bytes are CLEAN FILE-BACKED pages the kernel reclaims
# under pressure rather than OOM-killing on, and the small-file churn disappears. Measured peak
# RSS of the runs that do build the cache is 30.3-34.9 GiB, so ~80G is ample for a mapping run
# -- which matters because the node has idle A100s it cannot fill in 180G chunks.
SB=(--partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:A100:1}" --time="${TIME:-14:00:00}"
    --mem="${MEM:-180G}" --cpus-per-task=32)

# arm -> "TAG | per-arm env overrides". Tags already match rename_runs.py's scheme, so the
# post-hoc rename is a confirmation rather than a reshuffle.
declare -A ARM=(
  [head_selfattn]="res448_k2_frozen_sa128_d4photo   | UNFREEZE_BLOCKS=0 PATCH_SELFATTN_DIM=128 SEED=42"
  [head_multiquery]="res448_k2_frozen_q4_d4photo    | UNFREEZE_BLOCKS=0 POOL_QUERIES=4 SEED=42"
  [ft_bitfit]="res448_k2_bit2_d4photo_elr1e05       | UNFREEZE_BLOCKS=2 FT_MODE=bitfit ENCODER_LR=1e-5 SEED=42"
  [ft_bitfit_hi]="res448_k2_bit2_d4photo_elr0.001   | UNFREEZE_BLOCKS=2 FT_MODE=bitfit ENCODER_LR=1e-3 SEED=42"
  [frozen_ctrl_s42]="res448_k2_frozen_d4photo_decay30_seed42 | UNFREEZE_BLOCKS=0 SEED=42"
  [frozen_ctrl_s1]="res448_k2_frozen_d4photo_decay30_seed1   | UNFREEZE_BLOCKS=0 SEED=1"
  # ---- follow-ups added 2026-08-15, after the six arms above resolved ----------------------
  # bit6 is the test the RESULT block names as the open question. bitfit matching full FT is
  # established only AT 2 BLOCKS; ft6 (0.5243) still beats bit2 (0.4902) by ~3.8x the noise
  # floor, so "recalibration explains fine-tuning" is NOT yet established -- it is established
  # at matched depth. If bit6 reaches ft6, recalibration explains all of it at 1/577th the
  # params. If it does not, depth of WEIGHT adaptation buys something bias-only cannot.
  # AUGMENT=d4 (not the d4_photo every other arm here uses) because ft6, the run this is
  # measured against, was trained at d4 -- matching it is the whole point.
  # ENCODER_LR=1e-3 is bitfit's own good value, not ft6's 1e-5. That asymmetry is deliberate
  # and defensible: full FT gets WORSE with a bigger step (ft2 0.4866 at 1e-5 -> 0.4449 at
  # 3e-5) while bitfit gets better (0.4509 -> 0.4902), so each method is run where it works.
  [ft_bit6]="res448_k2_bit6_d4                      | UNFREEZE_BLOCKS=6 FT_MODE=bitfit ENCODER_LR=1e-3 SEED=42 AUGMENT=d4"
  # Every arm above except the two controls is a SINGLE seed, so the headline (+0.061) rests on
  # one draw measured against a noise floor estimated from a different config. This replicates
  # the winner at seed 1.
  [ft_bitfit_hi_s1]="res448_k2_bit2_d4photo_elr0.001_seed1 | UNFREEZE_BLOCKS=2 FT_MODE=bitfit ENCODER_LR=1e-3 SEED=1"
)
ORDER="head_selfattn head_multiquery ft_bitfit ft_bitfit_hi frozen_ctrl_s42 frozen_ctrl_s1"

for arm in ${ARMS:-$ORDER}; do
    spec="${ARM[$arm]:-}"
    [ -z "$spec" ] && { echo "unknown arm: $arm"; exit 1; }
    tag=$(echo "${spec%%|*}" | xargs)
    envs=$(echo "${spec##*|}" | xargs)
    if [ -e "results/vision/mice/frame/$tag/config.json" ]; then
        echo "SKIP  $arm -> $tag already has results"; continue
    fi
    # A config.json only appears when a run FINISHES, so that check alone happily submits a
    # second copy of an arm that is queued or mid-training -- and both would write the same
    # results/.../$tag directory, interleaving checkpoints into a run that is neither.
    if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$arm"; then
        echo "SKIP  $arm -> already queued or running"; continue
    fi
    if [ -n "${DRY:-}" ]; then
        echo "[dry] $arm: TAG=$tag $envs"; continue
    fi
    jid=$(env $envs TAG="$tag" sbatch "${SB[@]}" --job-name="$arm" \
              --output="logs/${arm}_%j.out" --error="logs/${arm}_%j.err" \
              --parsable scripts/mice_behavior/train_online_aug.sh)
    echo "submitted $jid  $arm -> $tag"
done

# RESULT -- 2026-08-15, all six arms landed
# ==============================================================================
# Full val, macro AP at tol0. The two controls define the noise floor FIRST; every other
# number is read as a distance from them, which is the whole reason they were run.
#
#     frozen_ctrl_s42                          0.4289      seed 42
#     frozen_ctrl_s1                           0.4200      seed 1
#     --> noise floor: spread 0.0089 over a seed change, nothing else varied
#
#     head_selfattn   (sa128)                  0.4431      +0.014   ~1.6x the spread
#     head_multiquery (q4)                     0.4262      -0.003   below the control
#     ft_bitfit       (elr 1e-5)               0.4509      +0.022   ~2.5x
#     ft_bitfit_hi    (elr 1e-3)               0.4902      +0.061   ~6.9x   <-- the result
#
#     res448_k2_ft2_d4photo (full FT, 14.18M)  0.4889      the arm bitfit is measured against
#
# READING IT, against the decision table above: ft_bitfit wins, head_selfattn flat ->
# "domain shift; cheap recalibration suffices -> run ssl_dapt".
#
# The sharpest fact here is not the win, it is the price. bitfit_hi reaches 0.4902 with 24,576
# params against full fine-tuning's 0.4889 with 14,180,352 -- a 577x cut for, if anything, a
# hair MORE. Nothing in bitfit can form a new function of two patch features; it can only
# rescale and reshift features the frozen weights already compute. Matching full fine-tuning
# under that constraint says the 14.2M densely-coupled weights were never the point, and the
# "the head cannot do relational computation, so the encoder had to" story is wrong as stated.
#
# ENCODER LR WAS THE WHOLE DIFFERENCE between an informative arm and a null: bitfit at 1e-5
# scores 0.4509 and at 1e-3 scores 0.4902. Had this been run at the inherited 1e-5 alone -- the
# value tuned for 14.2M weights -- it would have read as a weak +0.022 and the conclusion would
# have been the opposite one. That is exactly why --encoder-lr's help text insists a bias-only
# null at 1e-5 is uninterpretable, and it is the single most transferable lesson here.
#
# The head arms are flat and one is BELOW its control, so the head was not the bottleneck.
# Note that head_selfattn's +0.014 is only ~1.6x a seed change: at n=24 val observations that
# is not a win, and calling it one would be reading noise.
#
# CAVEAT, unchanged: per-observation Pearson r has SE ~0.2 at n=24 and cannot rank close arms.
# Two seeds give a spread, not a standard error -- 0.0089 bounds the noise loosely, it does not
# estimate it. bitfit_hi at ~6.9x that spread is safe; head_selfattn at ~1.6x is not.
#
# NOT AN ARM OF THIS ABLATION, but it landed alongside and is the best single model to date:
#     res448_k2_ft6_d4 (6 blocks unfrozen)     0.5243
# It says depth of adaptation still buys something on top of everything above (ft2 0.4889 ->
# ft4 0.5212 -> ft6 0.5243, flattening). That does not undercut the bitfit result: bitfit was
# held at 2 blocks precisely to be comparable to ft2, and the open question it raises is
# whether bitfit at 6 blocks would also close the gap to ft6 at 1/577th the parameters.
