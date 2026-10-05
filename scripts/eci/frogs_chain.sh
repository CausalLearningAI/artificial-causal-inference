#!/bin/bash
#
# Frogs v1 ECI chain (src/eci/domain.py FrogsDomain): the mice / ants foreground pipeline, unchanged, on two SAEs.
# Not an sbatch script: run it with plain bash on a login node; it submits every step with dependencies.
#
#   frogsfull  FULL FRAME: the whole 512 px frame at 448 (32 x 32 = 1024 patches), rule 'all'; training tokens = a
#              seeded 25% of the patches of every 1 fps frame (stride 5, as mice ff448al / ants antsfull)
#   frogsfg    FROG MASK: rule 'frogs' (src/eci/foreground.py FG_RULE_FROGS: dark frog pixels against a frog-free
#              pixel background, inside the dish, near the frame's SLEAP nodes; ~15-25 of the 1024 patches);
#              training tokens = every mask patch of EVERY 5 fps frame (stride 1: at 1 fps the mask would give only
#              ~2.3M tokens, a third of the ants mask store; stride 1 gives ~9M)
# Both: Matryoshka BatchTopK SAE, 1024 latents, k=16, prefixes 128/256/512/1024, the train_sae_fg.py defaults; held
# out = ~10% of the videos of every group (FrogsDomain.val_split: 1 WT, 1 FoxP1, 1 En1).
#
#   0. scripts/eci/frogs_prepare.py --step source (experiment.csv, hflip, symlinks; run once by hand)
#   1. frogs_frames.sh          35 tasks, one video each: standardize (5 fps, 512 px, mirrored if hflip) + JPEG frames
#   2. tables                   get_metadata, frogs_prepare.py --step tables, frogs_background.py (needs the SLEAP
#                               npz of /usr/bin/python3 scripts/eci/frogs_sleap.py, run once by hand)
#   3. fg_extract_train.sh      both chains in parallel, 35 tasks each (one video per task)
#   4. train_sae_fg.sh          both chains in parallel (gpu partition)
#   5. fused_encode_all.sh      24 shards per chain: pooled codes (max / mean over the chain's patches) + SOMP
#   6. fused_encode_merge.sh    merge + verify + <sae>_mean link
#   7. NES per chain: run_nes.sh (primary codes_max), run_nes_bouts.sh, run_nes_somp.sh (mean + SOMP),
#      run_nes_latency.sh, session_check.sh — all CPU, all in parallel
#
# Usage: bash scripts/eci/frogs_chain.sh                 (all steps)
#        FROM=3 bash scripts/eci/frogs_chain.sh          (from step 3; steps before must be finished)
#        DRY=1 bash scripts/eci/frogs_chain.sh           (prints the sbatch commands only)
# Finished outputs are skipped, so the chain can be resubmitted after a failure.
#
# STAGE=44-48 (tadpoles, domain 'tadpoles', src/eci/domain.py TadpolesDomain; default STAGE=Juv = everything above,
# unchanged): ONE chain, the tadpole BODY-CROP mask model (no full-frame model):
#   0. by hand, once: scripts/eci/frogs_prepare.py --stage 44-48 --step source; /usr/bin/python3
#      scripts/eci/frogs_sleap.py --stage 44-48; EXTRA_ARGS=--quality-only sbatch --array=0-172
#      scripts/eci/tadpoles_crops.sh; frogs_prepare.py --stage 44-48 --step exclude (sets experiment.csv valid)
#   1. tadpoles_crops.sh      one task per valid video: 5 fps 96 px body crops at 512 px + background (~3 min, 340 MB)
#   2. tadpoles_tables.sh     frogs_prepare.py --stage 44-48 --step tables, tadpoles_background.py
#   3. fg_extract_train.sh    rule 'tadpoles', 1 fps (stride 5), a seeded PATCH_FRAC of the mask patches per frame
#   4. train_sae_fg.sh        Matryoshka BatchTopK 1024, k 16 (the frogsfg settings), tag tadpolefg
#   5-6. fused_encode_all.sh + fused_encode_merge.sh (max / mean / SOMP codes)
#   7. NES at prefixes 128 / 256 for FoxP1_vs_WT, En1_vs_WT, FoxP1_vs_Scrambled, En1_vs_Scrambled: run_nes.sh (max),
#      run_nes_bouts.sh, run_nes_somp.sh, run_nes_latency.sh; session_check.sh with LEVELS="session substage"
# Test run on a few videos: TADPOLE_TAG=_test OBS=WT_100_1,... NES_EXTRA='--min-units 2' TEST_MAX_FRAMES=3600 TOKEN_STRIDE=1
# STAGE=44-48 bash scripts/eci/frogs_chain.sh (TEST_MAX_FRAMES: only the first 12 min of each video; TOKEN_STRIDE=1:
# every frame's tokens, as the SAE initialisation needs >= 500k training tokens; separate
# dirs dataset/frogs/eci_tadpole_test, results/vision/frogs/eci_tadpole_test; tables with --subset).
#   STAGE=44-48 DRY=1 bash scripts/eci/frogs_chain.sh
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
if [ "${STAGE:-Juv}" = "44-48" ]; then
    export DOMAIN=tadpoles
    unset EXTRA_ARGS SAE
    FROM=${FROM:-1}
    TAG=${TADPOLE_TAG:-}
    export TADPOLE_TAG=$TAG
    PATCH_FRAC=${PATCH_FRAC:-0.5}
    T_TP=dataset/frogs/eci_tadpole$TAG/train_tokens/dinov2_base_l-1_tadpolefg_fps1_pf$(python3 -c "print(round(100 * $PATCH_FRAC))")
    S=matryoshka_btk_1024_k16_tadpolefg_s0
    PFX0="--prefixes 128,256"
    PFX="$PFX0 ${NES_EXTRA:-}"  # NES_EXTRA: e.g. '--min-units 2' for a test on a few videos
    run() {  # submit (or print with DRY=1) and echo the job id
        if [ -n "${DRY:-}" ]; then echo "TADPOLE_TAG=$TAG OBS=${OBS:-} SAE=${SAE:-} LEVELS='${LEVELS:-}' SPECS='${SPECS:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' $*" >&2; echo "DRY"
        else "$@"; fi
    }
    dep() { [ -z "$1" ] || [[ "$1" == *DRY* ]] && echo "" || echo "--dependency=afterok:$1"; }
    if [ -n "${OBS:-}" ]; then
        [ -n "$TAG" ] || { echo "OBS (a subset) needs TADPOLE_TAG" >&2; exit 1; }
        N=$(echo "$OBS" | tr ',' '\n' | wc -l); SUB="--subset ${TEST_MAX_FRAMES:+--max-frames $TEST_MAX_FRAMES}"; NT=$N
    else
        N=$(python3 -c "import csv; print(sum(r['valid'] == '1' for r in csv.DictReader(open('data/frogs/v1_tadpole/experiment.csv'))))")
        SUB=""; NT=${NT:-35}
        [ -z "$TAG" ] || { echo "TADPOLE_TAG without OBS: the test dirs would hold the full set" >&2; exit 1; }
    fi
    export OBS=${OBS:-}
    j1=""; j2=""; jt=""; js=""; e=""; m=""
    [ "$FROM" -le 1 ] && j1=$(run sbatch --parsable --export=ALL --array=0-$((N - 1)) scripts/eci/tadpoles_crops.sh)
    [ "$FROM" -le 2 ] && j2=$(SUBSET=$SUB run sbatch --parsable --export=ALL $(dep "$j1") scripts/eci/tadpoles_tables.sh)
    [ "$FROM" -le 3 ] && jt=$(EXTRA_ARGS="--rule tadpoles --stride ${TOKEN_STRIDE:-5} --patch-frac $PATCH_FRAC --n-tasks $NT --out-dir $T_TP" \
        run sbatch --parsable --export=ALL -p gpu --array=0-$((NT - 1)) $(dep "$j2") scripts/eci/fg_extract_train.sh)
    [ "$FROM" -le 4 ] && js=$(EXTRA_ARGS="--tokens-dir $T_TP --tag tadpolefg" \
        run sbatch --parsable --export=ALL -p gpu --gres=gpu:1 --array=0 --mem=${SAE_MEM:-120G} $(dep "$jt") scripts/eci/train_sae_fg.sh)
    if [ "$FROM" -le 5 ]; then
        e=$(SAE=$S run sbatch --parsable --export=ALL -p gpu --mem=96G $(dep "$js") scripts/eci/fused_encode_all.sh)
        m=$(SAE=$S run sbatch --parsable --export=ALL -p gpu $(dep "$e") scripts/eci/fused_encode_merge.sh)
    fi
    n1=$(SAE=$S EXTRA_ARGS="--primary-pooling max $PFX" run sbatch --parsable --export=ALL $(dep "$m") scripts/eci/run_nes.sh)
    n2=$(SAE=$S EXTRA_ARGS="--compare-pooling max $PFX" run sbatch --parsable --export=ALL $(dep "$m") scripts/eci/run_nes_bouts.sh)
    n3=$(SAE=$S EXTRA_ARGS="$PFX" run sbatch --parsable --export=ALL $(dep "$m") scripts/eci/run_nes_somp.sh)
    n4=$(SAE=$S EXTRA_ARGS="$PFX0" run sbatch --parsable --export=ALL $(dep "$m") scripts/eci/run_nes_latency.sh)
    n5=$(SPECS="$S:max ${S}_mean:mean" LEVELS="session substage" run sbatch --parsable --export=ALL \
        $(dep "${n1}:${n2}:${n3}") scripts/eci/session_check.sh)
    echo "crops $j1 ($N videos) -> tables $j2 -> tokens $jt -> SAE $js -> encode $e -> merge $m -> NES $n1, bouts $n2," \
        "mean+SOMP $n3, latency $n4, session + sub-stage checks $n5"
    exit 0
fi
export DOMAIN=frogs
unset EXTRA_ARGS SAE
FROM=${FROM:-1}

TT=dataset/frogs/eci/train_tokens
T_FULL=$TT/dinov2_base_l-1_frogsfull448_fps1_pf25
T_FG=$TT/dinov2_base_l-1_frogsfg_fps5
SAES="matryoshka_btk_1024_k16_frogsfull_s0 matryoshka_btk_1024_k16_frogsfg_s0"

run() {  # submit (or print with DRY=1) and echo the job id
    if [ -n "${DRY:-}" ]; then echo "SAE=${SAE:-} EXTRA_ARGS='${EXTRA_ARGS:-}' $*" >&2; echo "DRY"
    else "$@"; fi
}
dep() { [ -z "$1" ] || [[ "$1" == *DRY* ]] && echo "" || echo "--dependency=afterok:$1"; }

j1=""; j2=""
[ "$FROM" -le 1 ] && j1=$(run sbatch --parsable scripts/eci/frogs_frames.sh)
[ "$FROM" -le 2 ] && j2=$(run sbatch --parsable $(dep "$j1") --job-name=frogs_tables -p defaultp -c 8 --mem=32G \
    -t 02:00:00 -o logs/frogs_tables_%j.out -e logs/frogs_tables_%j.err --wrap "source ~/.bashrc >/dev/null 2>&1; module load conda; conda activate crl; \
    set -euo pipefail; python -u src/data/get_metadata.py experiment=frogs/v1 && \
    python -u scripts/eci/frogs_prepare.py --step tables && python -u scripts/eci/frogs_background.py")
TOK="-p gpu --array=0-34 $(dep "$j2")"
jf=""; jg=""
if [ "$FROM" -le 3 ]; then
    jf=$(EXTRA_ARGS="--rule all --patch-frac 0.25 --n-tasks 35 --out-dir $T_FULL" \
        run sbatch --parsable --export=ALL $TOK scripts/eci/fg_extract_train.sh)
    jg=$(EXTRA_ARGS="--rule frogs --stride 1 --n-tasks 35 --out-dir $T_FG" \
        run sbatch --parsable --export=ALL $TOK scripts/eci/fg_extract_train.sh)
fi
SAE_ARGS="-p gpu --gres=gpu:1 --array=0"
sf=""; sg=""
if [ "$FROM" -le 4 ]; then
    sf=$(EXTRA_ARGS="--tokens-dir $T_FULL --tag frogsfull" \
        run sbatch --parsable --export=ALL $SAE_ARGS --mem=200G $(dep "$jf") scripts/eci/train_sae_fg.sh)
    sg=$(EXTRA_ARGS="--tokens-dir $T_FG --tag frogsfg" \
        run sbatch --parsable --export=ALL $SAE_ARGS --mem=120G $(dep "$jg") scripts/eci/train_sae_fg.sh)
fi
i=0
for S in $SAES; do
    d=$([ $i -eq 0 ] && echo "$sf" || echo "$sg"); i=$((i + 1))
    e=""; m=""
    if [ "$FROM" -le 5 ]; then
        e=$(SAE=$S run sbatch --parsable --export=ALL -p gpu --mem=96G $(dep "$d") scripts/eci/fused_encode_all.sh)
        m=$(SAE=$S run sbatch --parsable --export=ALL -p gpu $(dep "$e") scripts/eci/fused_encode_merge.sh)
    fi
    n1=$(SAE=$S EXTRA_ARGS="--primary-pooling max" run sbatch --parsable --export=ALL $(dep "$m") scripts/eci/run_nes.sh)
    n2=$(SAE=$S EXTRA_ARGS="--compare-pooling max" run sbatch --parsable --export=ALL $(dep "$m") scripts/eci/run_nes_bouts.sh)
    n3=$(SAE=$S run sbatch --parsable --export=ALL $(dep "$m") scripts/eci/run_nes_somp.sh)
    n4=$(SAE=$S run sbatch --parsable --export=ALL $(dep "$m") scripts/eci/run_nes_latency.sh)
    n5=$(SPECS="$S:max ${S}_mean:mean" run sbatch --parsable --export=ALL $(dep "${n1}:${n2}:${n3}") \
        scripts/eci/session_check.sh)
    echo "$S: encode $e -> merge $m -> NES $n1, bouts $n2, mean+SOMP $n3, latency $n4, session check $n5"
done
echo "frames $j1 -> tables $j2 -> tokens full $jf / mask $jg -> SAE full $sf / mask $sg"
