#!/bin/bash
#
# Ants 30-min ECI layout (ANTS_ECI=eci30), steps 1-8, submitted as one Slurm dependency chain (run on the login node:
# it only calls sbatch). Inputs: data/ants/{v2,v3,v4,v4_infested}/experiment.csv, the PPCI frames (v3 / v4 frames
# 3000-8999, all v4_infested frames) and dataset/ants/{v2,v3,v4}/annotations.csv. Outputs: dataset/ants/eci30/,
# results/vision/ants/eci30/nes/. Logs: results/vision/eci_ants30/logs/.
#   1 prepare   (CPU) every needed frame dir holds frames 0..8999, then scripts/eci/ants_prepare.py
#   2 setup     (CPU) scripts/eci/fg_encode_extend.py setup (row map, SAE links, first-window background links)
#   3 bg        (GPU x 8) scripts/eci/fg_background.py: the other 10-min windows
#   4 encode    (GPU x 24) scripts/eci/fg_encode_extend.py encode, both SAEs in one pass
#   5 merge     (GPU) merge + verify (reused rows byte-equal, re-encoded rows), <sae>_mean siblings
#   6 recon     (GPU) scripts/eci/ants_recon_check.py
#   7 nes       (CPU, one job per SAE) the atlas NES settings: core and pairs sets, activation (max), bouts (max),
#               mean-pooling sibling (activation + bouts), latency (max, mean; pairs); prefixes 128/256/1024,
#               selection by |tau|, no nuisance, no --day
#   8 interp    (CPU) the interpretations of the 10-min result sets linked into the new ones (same neuron ids)
#
#   bash scripts/eci/ants30_chain.sh            (FROM=3 bash ... to restart at a step)
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export ANTS_ECI=eci30
L=results/vision/eci_ants30/logs
mkdir -p $L logs
S="matryoshka_btk_1024_k16_antsfg_s0 matryoshka_btk_1024_k16_antsfull_s0"
FROM=${FROM:-1}
ENV="module load conda; conda activate crl; export PYTHONUNBUFFERED=1 ANTS_ECI=eci30; cd $PWD"
cpu() { sbatch --parsable -p defaultp --cpus-per-task=${C:-4} --mem=${M:-32G} --time=${T:-02:00:00} --output=$L/%x_%j.out "$@"; }
dep() { [ -n "${1:-}" ] && echo "--dependency=afterok:$1" || true; }
j1= ; j2= ; j3= ; j4= ; j5=
if [ "$FROM" -le 1 ]; then
  j1=$(cpu --job-name=a30_prepare --wrap "bash -lc '$ENV; python -u scripts/eci/ants30_checks.py frames && python -u scripts/eci/ants_prepare.py'")
fi
if [ "$FROM" -le 2 ]; then
  SA=""; for s in $S; do SA="$SA --sae $s"; done
  j2=$(cpu --job-name=a30_setup $(dep $j1) --wrap "bash -lc '$ENV; python -u scripts/eci/fg_encode_extend.py setup $SA'")
fi
if [ "$FROM" -le 3 ]; then
  j3=$(DOMAIN=ants sbatch --parsable --export=ALL --exclude=gpu150,gpu242,gpu230 $(dep $j2) scripts/eci/fg_background.sh)
fi
if [ "$FROM" -le 4 ]; then
  j4=$(STEP=encode SAES="$S" sbatch --parsable --array=0-23 --time=04:00:00 --export=ALL $(dep $j3) scripts/eci/fg_encode_extend.sh)
fi
if [ "$FROM" -le 5 ]; then
  j5=$(STEP=merge SAES="$S" sbatch --parsable --time=04:00:00 --export=ALL $(dep $j4) scripts/eci/fg_encode_extend.sh)
fi
j6=$(sbatch --parsable -p gpu --gres=gpu:1 --exclude=gpu150,gpu242,gpu230 --cpus-per-task=12 --mem=48G --time=01:00:00 \
  --job-name=a30_recon --output=$L/%x_%j.out $(dep $j5) \
  --wrap "bash -lc '$ENV; python -u scripts/eci/ants_recon_check.py $(for s in $S; do echo --sae $s; done)'")
J7=""
for s in $S; do
  N="python -u scripts/eci/run_nes.py --domain ants"; B="python -u scripts/eci/run_nes_bouts.py --domain ants"
  LAT="python -u scripts/eci/run_nes_latency.py --domain ants --analysis-set pairs"
  CMD="$N --sae $s --primary-pooling max && $N --sae $s --primary-pooling max --analysis-set pairs"
  CMD="$CMD && $B --sae $s --compare-pooling max && $B --sae $s --compare-pooling max --analysis-set pairs"
  for set in core pairs; do
    CMD="$CMD && $N --sae ${s}_mean --primary-pooling mean --poolings mean --analysis-set $set"
    CMD="$CMD && $B --sae ${s}_mean --frame-pooling mean --compare-pooling mean --analysis-set $set"
  done
  CMD="$CMD && $LAT --sae $s --frame-pooling max && $LAT --sae ${s}_mean --frame-pooling mean"
  j=$(C=8 M=64G T=12:00:00 cpu --job-name=a30_nes_${s##*k16_} $(dep $j5) --wrap "bash -lc '$ENV; $CMD'")
  J7="$J7:$j"
done
j8=$(C=1 M=2G T=00:10:00 cpu --job-name=a30_interp --dependency=afterok${J7} --wrap "bash -c 'cd $PWD; for s in $S; do \
  ln -sfn \$PWD/results/vision/ants/eci/nes/\$s/interp results/vision/ants/eci30/nes/\$s/interp; done; ls -la results/vision/ants/eci30/nes/*/interp'")
echo "prepare $j1 setup $j2 bg $j3 encode $j4 merge $j5 recon $j6 nes${J7} interp $j8"
